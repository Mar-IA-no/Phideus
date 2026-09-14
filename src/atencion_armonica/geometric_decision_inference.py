"""Observable-only signed readout and coordinate-transport diagnostic.

No checkpoint loading, producer, fitter, sidecar, or execution authority.
The caller authenticates states/inputs, admits resources and seals outputs.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from .generative_evidence import MAX_CANDIDATES, MAX_GROUPS, partitions_checked
from .geometric_decision_core import INPUT_KEYS, route_inputs, choose_energy
from .geometric_decision_model import GeometricDecisionHead, collate


def validate_rows(rows):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 512:
        raise ValueError("readout requires one to 512 explicitly identified observable rows")
    counts = []
    for row in rows:
        if (not isinstance(row, dict) or set(row) != INPUT_KEYS
                or any(not isinstance(v, np.ndarray) or v.dtype != np.float32
                       or not np.isfinite(v).all() for v in row.values())):
            raise ValueError("readout contains supervision or invalid observable arrays")
        groups, candidates = row["groups"], row["globals"]
        if (groups.ndim != 2 or groups.shape[1] != 9 or len(groups) > MAX_GROUPS
                or candidates.ndim != 2 or candidates.shape[1] != 17 or len(candidates) > MAX_CANDIDATES
                or row["incidence"].shape != (len(candidates), len(groups))
                or row["evidence"].shape != (len(candidates), 8)
                or bool(len(groups)) != bool(len(candidates))):
            raise ValueError("readout candidate/group extent differs")
        counts.append(len(candidates))
    return counts


def _model_device(model, device):
    if (not isinstance(model, GeometricDecisionHead) or device not in ("cpu", "cuda:0")
            or any(p.device != torch.device(device) or p.dtype != torch.float32 for p in model.parameters())):
        raise ValueError("readout requires the authenticated signed head on its admitted device")


def predict(model, rows, *, device, check):
    _model_device(model, device)
    counts = validate_rows(rows)
    offsets = np.r_[np.int64(0), np.cumsum(counts, dtype=np.int64)]
    components = np.empty((int(offsets[-1]), 2), np.float64)
    eligible = [i for i, count in enumerate(counts) if count]
    check()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(eligible), 32):
            check()
            ids = eligible[start:start+32]
            batch = {k: v.to(device) for k, v in collate([route_inputs(rows[i], model.route) for i in ids]).items()}
            values = model(batch).cpu().numpy()
            for j, i in enumerate(ids):
                components[offsets[i]:offsets[i+1]] = values[j, :counts[i]]
    check()
    if not np.isfinite(components).all():
        raise ValueError("readout produced nonfinite signed components")
    return {"components": components, "energy": components.sum(-1, dtype=np.float64), "offsets": offsets}


def transport_prediction(model, row, partitions, *, device, check):
    """Reverse candidate/group/channel coordinates and transport model columns.

    The changed channel coordinates reach the linear operation: this is not
    undone preprocessing. Fixed bypass indices move with their scalar channel.
    Candidate coordinates are undone only after the forward for comparison.
    Reordering float32 reductions need not preserve bits or exact decisions.
    """
    _model_device(model, device)
    validate_rows([row])
    count = len(row["globals"])
    if not count or len(partitions) != count:
        raise ValueError("transport requires an eligible candidate universe")
    n = sum(map(len, partitions[0]))
    ps = partitions_checked(partitions, n)
    groups = sorted({g for p in ps for g in p})
    expected = np.array([[len(g)/n if g in p else 0 for g in groups] for p in ps], np.float32)
    if not np.array_equal(row["incidence"], expected):
        raise ValueError("transport incidence does not match canonical partitions/groups")
    baseline = predict(model, [row], device=device, check=check)
    candidate_order = np.arange(count-1, -1, -1, dtype=np.int64)
    group_order = np.arange(len(groups)-1, -1, -1, dtype=np.int64)
    channel_order = np.arange(7, -1, -1, dtype=np.int64)
    routed = route_inputs(row, model.route)
    changed = {"groups": routed["groups"][group_order].copy(),
        "globals": routed["globals"][candidate_order].copy(),
        "incidence": routed["incidence"][np.ix_(candidate_order, group_order)].copy(),
        "evidence": routed["evidence"][np.ix_(candidate_order, channel_order)].copy()}
    check()
    batch = {k: v.to(device) for k, v in collate([changed]).items()}
    column_order = torch.as_tensor(np.r_[np.arange(33), 33+channel_order], dtype=torch.long, device=device)
    bypass = 6 if model.route == "geometric" else 7 if model.route == "decoupled" else None
    moved_bypass = None if bypass is None else int(np.flatnonzero(channel_order == bypass)[0])
    with torch.no_grad():
        hidden = torch.relu(model.group2(torch.relu(model.group1(batch["groups"]))))
        aggregate = torch.bmm(batch["incidence"], hidden)
        values = torch.cat((aggregate, batch["globals"], batch["evidence"]), dim=-1)
        hidden = torch.relu(F.linear(values, model.partition1.weight.index_select(1, column_order), model.partition1.bias))
        residual = model.partition2(hidden).double()
        scalar = torch.zeros_like(residual[..., 0]) if moved_bypass is None else batch["evidence"][..., moved_bypass].double()
        moved = (residual+scalar[..., None]/2)[0].cpu().numpy()
    check()
    if moved.dtype != np.float64 or not np.isfinite(moved).all():
        raise ValueError("transport produced invalid signed output")
    restored = moved[np.argsort(candidate_order)].copy()
    moved_energy = moved.sum(-1, dtype=np.float64)
    restored_energy = restored.sum(-1, dtype=np.float64)
    moved_ps = [ps[i] for i in candidate_order]
    before = choose_energy(baseline["energy"], ps)
    after = choose_energy(moved_energy, moved_ps)
    def margin(energy):
        ordered = np.sort(energy)
        return None if len(ordered) < 2 else float(ordered[1]-ordered[0])
    return {"baseline": baseline, "transported_components": moved, "restored_components": restored,
        "transported_energy": moved_energy, "restored_energy": restored_energy,
        "candidate_order": candidate_order, "group_order": group_order, "channel_order": channel_order,
        "weight_column_order": column_order.cpu().numpy(), "bypass_channel": moved_bypass,
        "transported_inputs": changed,
        "diagnostic": {"atol": 1e-6, "rtol": 1e-5,
            "within_numeric_tolerance": bool(np.allclose(restored, baseline["components"], atol=1e-6, rtol=1e-5)),
            "max_component_error": float(np.max(np.abs(restored-baseline["components"]))),
            "max_energy_error": float(np.max(np.abs(restored_energy-baseline["energy"]))),
            "same_exact_choice": ps[before] == moved_ps[after],
            "original_margin": margin(baseline["energy"]), "transported_margin": margin(moved_energy),
            "original_choice": ps[before], "transported_choice": moved_ps[after],
            "scope": "coordinate transport and float32 reduction stability, not learned physical invariance"}}
