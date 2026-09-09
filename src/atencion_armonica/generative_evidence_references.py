"""Pure observable system choices and separate post-seal privileged metrics.

No file access, fitting, network forward, or test authority. The caller seals
observable choices and all 45 learned/intervention outputs before supplying
truth to evaluate_references. System references are not capacity-matched arms.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from . import generative_evidence as ge
from .generative_evidence_supervision import candidate_supervision
from .learned_partition_core import partition_errors
from .learned_partition_metrics import METRICS, _augment
from .partial_compatibility_evaluation import read_partition
from .structured_source_metrics import partition_metrics

THRESHOLDS = dict(zip(ge.CHECKPOINTS, (.55, .65, .60)))


def observable_choices(partitions, fits, logits, canonical_to_observed):
    """Base/Extended once and Historical once per checkpoint; no labels."""
    order = np.asarray(canonical_to_observed)
    n = len(order)
    if (order.dtype.kind not in "iu" or order.ndim != 1 or not 8 <= n <= 32
            or not np.array_equal(np.sort(order), np.arange(n)) or set(logits) != set(ge.CHECKPOINTS)):
        raise ValueError("reference event mapping or checkpoint roster differs")
    ps = ge.partitions_checked(partitions, n)
    ge.candidate_channel(ps, fits, n)  # Exact candidate/branch/bound identities.
    result = {"status": "OBSERVABLE_SYSTEM_CHOICES_NOT_TEST_AUTHORIZED", "n": n,
              "candidate_count": len(ps), "base": None, "extended": None, "historical": {}}
    for family, allowed in ge.law.FAMILIES.items():
        rows = []
        for i, (p, fit) in enumerate(zip(ps, fits)):
            branches = [b for b in allowed if b in fit["branches"]]
            branch = min(branches, key=lambda b: (fit["branches"][b]["UB"], b))
            rows.append({"candidate_index": i, "partition": p,
                         "UB": float(fit["branches"][branch]["UB"]), "branch": branch})
        result[family] = min(rows, key=lambda r: (r["UB"], r["partition"])) if rows else None
    inverse = np.argsort(order)
    for cp in ge.CHECKPOINTS:
        z = logits[cp]
        if (not isinstance(z, np.ndarray) or z.dtype != np.float32 or z.shape != (n, n)
                or not np.isfinite(z).all() or not np.array_equal(z, z.T)):
            raise ValueError("Historical requires finite symmetric preserved float32 logits")
        observed = read_partition(expit(z.astype(np.float64)), THRESHOLDS[cp])
        canonical = ge.law.signature([inverse[list(g)].tolist() for g in observed])
        result["historical"][str(cp)] = {"threshold": THRESHOLDS[cp], "partition": canonical}
    return result


def _metrics(partition, labels):
    p = ge.law.validate_partition(partition, len(labels))
    result = _augment(partition_metrics(p, labels), partition_errors(p, labels)["raw"],
                      len(labels), len(np.unique(labels)))
    if set(result) != set(METRICS) or not np.isfinite([result[k] for k in METRICS]).all():
        raise ValueError("reference metrics differ from the inherited thirteen")
    return result


def evaluate_references(partitions, inventory, choices, canonical_labels):
    """Post-seal truth port: all candidate metrics, oracle and planted support.

    Labels are already in canonical frequency rank order. The caller must
    authenticate the choices; this helper does not reopen privileged files.
    """
    y = np.asarray(canonical_labels)
    if (y.ndim != 1 or y.dtype.kind not in "iu" or not 8 <= len(y) <= 32
            or set(choices) != {"status", "n", "candidate_count", "base", "extended", "historical"}
            or choices["status"] != "OBSERVABLE_SYSTEM_CHOICES_NOT_TEST_AUTHORIZED"
            or choices["n"] != len(y)):
        raise ValueError("reference labels or observable choice schema differs")
    ps = ge.partitions_checked(partitions, len(y))
    if choices["candidate_count"] != len(ps) or set(choices["historical"]) != {str(c) for c in ge.CHECKPOINTS}:
        raise ValueError("reference candidate/checkpoint roster differs")
    rows = inventory["candidates"]
    signatures = [ge.law.validate_partition(row["partition"], len(y)) for row in rows]
    if (len(set(signatures)) != len(signatures)
            or any(row["origin"] not in ("pool", "neighbor") for row in rows)
            or any(row["status"] != ("SUPPORTED" if ge.law.supported(p) else "OUTSIDE_GENERATIVE_CARDINALITY")
                   for row, p in zip(rows, signatures))
            or sorted(p for p in signatures if ge.law.supported(p)) != ps):
        raise ValueError("candidate inventory differs from evaluated supported universe")
    candidate = candidate_supervision(ps, y)
    references = {}
    for family in ge.law.FAMILIES:
        choice = choices[family]
        if not ps:
            if choice is not None:
                raise ValueError("empty candidate universe cannot have a system choice")
            references[family] = None
            continue
        if (not isinstance(choice, dict) or set(choice) != {"candidate_index", "partition", "UB", "branch"}
                or type(choice["candidate_index"]) is not int or not 0 <= choice["candidate_index"] < len(ps)
                or ge.law.signature(choice["partition"]) != ps[choice["candidate_index"]]
                or not np.isfinite(choice["UB"]) or choice["UB"] < 0
                or choice["branch"] not in ge.law.FAMILIES[family]):
            raise ValueError("system reference no longer identifies its candidate")
        references[family] = {"choice": choice, "metrics": candidate["metrics"][choice["candidate_index"]]}
    historical = {}
    for cp in ge.CHECKPOINTS:
        value = choices["historical"][str(cp)]
        if set(value) != {"partition", "threshold"} or value["threshold"] != THRESHOLDS[cp]:
            raise ValueError("Historical threshold identity differs")
        historical[str(cp)] = {**value, "metrics": _metrics(value["partition"], y)}
    planted = ge.law.signature([np.flatnonzero(y == sid).tolist() for sid in np.unique(y)])
    origins = [row["origin"] for row, p in zip(rows, signatures) if p == planted]
    metrics = candidate["metrics"]
    return {"status": "PRIVILEGED_EVALUATION_NOT_DEPLOYABLE", "candidate_metrics": metrics,
        "raw_entropies": candidate["raw"], "normalized_targets": candidate["targets"],
        "references": {**references, "historical": historical},
        "coverage": {"has_output": bool(ps), "candidate_count": len(ps),
                     "planted": origins[0] if origins else "absent"},
        "oracle": {"maximum_ari": max(m["ari"] for m in metrics) if metrics else None,
                   "minimum_vi": min(m["vi"] for m in metrics) if metrics else None,
                   "authority": "PRIVILEGED_EVALUATION_NOT_DEPLOYABLE"}}
