"""Post-seal evaluation adapter and OPEN-compatible measurement kernel.

Public fresh port authenticates the global COMPLETE seal before its only
sidecar parser. Numeric batch evaluation receives explicit labels; it does
not grant permission to read them. Stores and budgets belong to the caller.
"""
from __future__ import annotations

import numpy as np

from . import generative_evidence as ge
from . import geometric_decision_metrics as metrics
from .geometric_decision_core import ARMS, READER_SEEDS
from .geometric_decision_campaign import ROSTER
from .geometric_decision_release import admitted_seal
from .geometric_decision_scene_store import fixed_arrays
from .operator_objective_core import observable_strata, describe_score

STAGES = ("initial", "selected")
CLASSICAL = ("base", "extended", "z", "d")
NAMES = ("regret_tM", "regret_tD", "ari", "optimal_tM", "optimal_tD", "exact",
         "k_absolute_error", "mse_components", "tau_b")


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            return None
        raise ValueError("infinite evaluation value")
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def save_json(output, path, value, *, replay):
    if replay and not output.path(path).is_file():
        raise ValueError("replay cannot repair a missing evaluation JSON")
    return output.publish_json(path, plain(value))


def save_arrays(output, path, value, *, replay):
    if replay and not output.path(path).is_file():
        raise ValueError("replay cannot repair missing evaluation arrays")
    return fixed_arrays(output, path, value)


def vector(description):
    decision = description["decision"]
    if decision is None:
        return np.full(len(NAMES), np.nan)
    return np.asarray([*(float(decision[k]) for k in NAMES[:7]),
        description["errors"]["mse"] if description["errors"] is not None else np.nan,
        description["tau"]["value"] if description["tau"]["value"] is not None else np.nan], np.float64)


def strata_metrics(energy, target, strata, components=None):
    """Preserve per-stratum scope; no aggregate can replace the full primary."""
    legacy = {"t64": target["tM"], "t32": target["tD"],
              **{k: target[k] for k in ("ari", "k_error", "exact")}}
    result = {}
    for scheme, groups in strata.items():
        if scheme == "full":
            continue
        result[scheme] = {}
        for name, ids in groups.items():
            row = describe_score(energy, legacy, ids)
            chosen = row["choice"]["chosen"]
            decision = row["decision"]
            values = {k: None for k in NAMES}
            if decision is not None:
                values.update(regret_tM=decision["target64_regret"], regret_tD=decision["target32_regret"],
                    ari=decision["ari"], optimal_tM=decision["target64_optimal"],
                    optimal_tD=bool(target["tD"][chosen] == np.min(target["tD"][ids])),
                    exact=decision["exact"], k_absolute_error=decision["k_absolute_error"],
                    tau_b=row["tau"]["value"])
                if components is not None:
                    values["mse_components"] = float(np.square(components[ids]-target["u32"][ids].astype(np.float64)).mean(axis=1).mean())
            result[scheme][name] = {"candidate_ids": ids, "chosen": chosen,
                "metrics": values, "tau_status": row["tau"]["status"]}
    return result


def mean_supported(values, axis):
    mask = np.isfinite(values)
    count = mask.sum(axis=axis)
    total = np.where(mask, values, 0.).sum(axis=axis, dtype=np.float64)
    result = np.full(total.shape, np.nan, np.float64)
    np.divide(total, count, out=result, where=count > 0)
    return result, count


def summarize(learned, classical, eligible):
    """All scopes descriptive here; the separately invoked primary is strict."""
    def summary(values):
        mean, count = mean_supported(values, axis=0)
        return {"mean": mean, "defined_scenes": count, "total_scenes": len(values)}
    scene_means, cell_counts = mean_supported(learned, axis=(3, 4))
    gap, _ = mean_supported(learned[:, 1]-learned[:, 0], axis=(2, 3))
    loss_effect = np.stack([learned[:, 1, 2*i+1]-learned[:, 1, 2*i] for i in range(4)], axis=1)
    loss_means, _ = mean_supported(loss_effect, axis=(2, 3))
    return plain({"metric_order": NAMES, "arm_order": ARMS, "stage_order": STAGES,
        "checkpoint_order": ge.CHECKPOINTS, "reader_order": READER_SEEDS,
        "classical_order": CLASSICAL, "eligible_scenes": int(eligible.sum()),
        "total_scenes": len(eligible), "arm_scene_first": summary(scene_means),
        "cell_descriptive": summary(learned), "defined_cells_by_scene": cell_counts,
        "selected_minus_initial": summary(gap), "decision_minus_mse_by_route": summary(loss_means),
        "classical": summary(classical),
        "selected_minus_classical": {name: summary(mean_supported(
            learned[:, 1]-classical[:, j, None, None, None, :], axis=(2, 3))[0])
            for j, name in enumerate(CLASSICAL)},
        "authority": "descriptive; conditioned cells are not independent scenes"})


def evaluate_batch(store, batch, labels, output, folder, *, check, replay=False):
    """Given explicit canonical labels, evaluate144 frozen outputs once each.

`batch` selects either the original or the complete roundtrip subrecord.
This function is also used with already-open TRAIN data for cost measurement.
"""
    sources, inputs = store.json(batch["sources"]), store.json(batch["inputs"])
    ids = sources["scene_ids"]
    if (set(labels) != set(ids) or not 1 <= len(ids) <= 512 or ids != sorted(set(ids))
            or inputs["sources"] != batch["sources"] or inputs["scene_ids"] != ids
            or sources["binding"] != store.binding or inputs["binding"] != store.binding
            or len(inputs["records"]) != len(ids) or len(sources["sources"]) != len(ids)):
        raise ValueError("evaluation source/input/label roster differs")
    classical_index = store.json(batch["classical"])
    if classical_index["scene_ids"] != ids or classical_index["inputs"] != batch["inputs"]:
        raise ValueError("classical evaluation roster differs")
    scene_data, target_refs = [], []
    for i, sid in enumerate(ids):
        check()
        src = store.json(sources["sources"][i])
        source = src["scene"]
        ps = ge.partitions_checked(source["partitions"], len(source["q32"]))
        target = metrics.targets(ps, labels[sid])
        inp = store.json(inputs["records"][i])
        if inp["source"] != sources["sources"][i] or source["observation"]["scene_id"] != sid:
            raise ValueError("evaluation target is paired with another source")
        arrays = store.arrays(inp["arrays"])
        cref = classical_index["records"][i]
        classic = store.json(cref)
        if classic["source"] != sources["sources"][i] or classic["inputs"] != inputs["records"][i]:
            raise ValueError("classical source differs from evaluation universe")
        strata = observable_strata(ps, arrays[f"raw/{ge.CHECKPOINTS[0]}/available"], classic["upper_branches"]["extended"])
        planted = ge.law.signature([np.flatnonzero(labels[sid] == y).tolist() for y in np.unique(labels[sid])])
        origins = {c["origin"] for c in source["inventory"]["candidates"] if ge.law.signature(c["partition"]) == planted}
        presence = "pool" if "pool" in origins else "neighbor" if "neighbor" in origins else "absent"
        tref = save_arrays(output, f"{folder}/targets/{sid:05d}.npz",
            {**target, "labels": labels[sid].astype(np.int64)}, replay=replay)
        meta = save_json(output, f"{folder}/targets/{sid:05d}.json", {
            "source": sources["sources"][i], "scene_id": sid, "arrays": tref,
            "candidate_count": len(ps), "event_count": len(labels[sid]), "planted_presence": presence,
            "strata": strata, "diagnostics": inp["diagnostics"],
            "target_arithmetic": "tM=sum64(u64); tD=sum64(float64(u32))"}, replay=replay)
        target_refs.append(meta)
        scene_data.append((ps, target, strata, classic, cref, presence))
    counts = [len(p) for p, *_ in scene_data]
    offsets = np.r_[np.int64(0), np.cumsum(counts, dtype=np.int64)]
    learned = np.full((len(ids), 2, 8, 3, 3, len(NAMES)), np.nan)
    head_results = []
    expected = [(cp, arm, seed, stage) for cp, arm, seed in ROSTER for stage in STAGES]
    if len(batch["records"]) != len(expected):
        raise ValueError("evaluation requires every initial/selected head")
    for row, identity in zip(batch["records"], expected):
        check()
        pred = store.json(row["prediction"])
        cp, arm, seed, stage = identity
        if ((pred["checkpoint_seed"], pred["arm"], pred["reader_seed"], pred["stage"]) != identity
                or pred["inputs"] != batch["inputs"] or pred["scene_ids"] != ids
                or pred["binding"] != store.binding or pred["head"] != row["head"]):
            raise ValueError("evaluation prediction identity/order differs")
        arrays = store.arrays(pred["arrays"])
        if (set(arrays) != {"offsets", "components", "energy"} or arrays["offsets"].dtype != np.int64
                or not np.array_equal(arrays["offsets"], offsets)
                or arrays["components"].shape != (int(offsets[-1]), 2)
                or arrays["energy"].shape != (int(offsets[-1]),)):
            raise ValueError("evaluation prediction offsets/extent differ")
        rows = []
        for i, sid in enumerate(ids):
            check()
            ps, target, strata, *_ = scene_data[i]
            a, b = offsets[i:i+2]
            h, energy = arrays["components"][a:b], arrays["energy"][a:b]
            description = metrics.describe(ps, target, energy, components=h)
            if description["chosen"] != pred["choices"][i]:
                raise ValueError("evaluation choice differs from sealed prediction")
            learned[i, STAGES.index(stage), ARMS.index(arm), ge.CHECKPOINTS.index(cp), READER_SEEDS.index(seed)] = vector(description)
            description["errors"] = {"mse": description["errors"]["mse"]}
            rows.append({"scene_id": sid, "full": description,
                         "strata": strata_metrics(energy, target, strata, h)})
        head_results.append(save_json(output, f"{folder}/heads/{cp}/{arm}/{seed}/{stage}.json",
            {"prediction": row["prediction"], "identity": list(identity), "scene_ids": ids, "rows": rows}, replay=replay))
    classical_values = np.full((len(ids), 4, len(NAMES)), np.nan)
    classical_rows = []
    for i, (ps, target, strata, classic, cref, presence) in enumerate(scene_data):
        check()
        scores = store.arrays(classic["arrays"])
        if set(scores) != set(CLASSICAL):
            raise ValueError("classical score names differ")
        row = {}
        for j, name in enumerate(CLASSICAL):
            description = metrics.describe(ps, target, scores[name])
            if description["chosen"] != classic["choices"][name]:
                raise ValueError("classical evaluation choice differs from seal")
            classical_values[i, j] = vector(description)
            row[name] = {"full": description, "strata": strata_metrics(scores[name], target, strata)}
        classical_rows.append({"scene_id": ids[i], "source": cref, "values": row})
    eligible = np.asarray(counts) > 0
    arrays = save_arrays(output, folder+"/metrics.npz", {"learned": learned, "classical": classical_values,
        "eligible": eligible, "scene_ids": np.asarray(ids, np.int64)}, replay=replay)
    classical_ref = save_json(output, folder+"/classical.json", classical_rows, replay=replay)
    summary = summarize(learned, classical_values, eligible)
    summary["presence"] = {name: summarize(learned[mask], classical_values[mask], eligible[mask])
        for name in ("pool", "neighbor", "absent")
        for mask in [np.asarray([s[-1] == name for s in scene_data], bool)]}
    complete = save_json(output, folder+"/complete.json", {"schema": "geometric-decision-batch-evaluation-v1",
        "binding": output.binding, "observed_sources": batch["sources"], "scene_ids": ids,
        "targets": target_refs, "heads": head_results, "classical": classical_ref,
        "arrays": arrays, "summary": summary}, replay=replay)
    return complete, learned, eligible


def labels_by_event(original, probe, canonical_labels):
    """Rank is not event identity; tied frequencies retain delivered positions."""
    first = np.asarray(original["canonical_to_observed"], np.int64)
    second = np.asarray(probe["canonical_to_observed"], np.int64)
    n = len(canonical_labels)
    if (first.shape != (n,) or second.shape != (n,) or canonical_labels.shape != (n,)
            or canonical_labels.dtype.kind not in "iu"
            or not np.array_equal(np.sort(first), np.arange(n))
            or not np.array_equal(np.sort(second), np.arange(n))):
        raise ValueError("invalid event-to-rank correspondence")
    events = np.empty(n, np.int64)
    events[first] = canonical_labels
    return events[second]


def evaluate_fresh(control, finish_ref, freeze_ref, output, *, check, replay=False):
    """Only fresh label parser in this module, downstream of the global seal."""
    store, seal, seal_ref = admitted_seal(control, finish_ref, freeze_ref, check=check)
    if output.binding != {"test_freeze": freeze_ref, "prediction_seal": seal_ref}:
        raise ValueError("evaluation output belongs to another sealed campaign")
    from .generative_evidence_supervision import reconstruct_truth
    results = []
    for batch in seal["batches"]:
        check()
        split = batch["split"]
        observed = store.json(batch["observed"])
        draws = store.json(batch["draws"])
        labels, original_scenes = {}, {}
        source_refs = store.json(observed["sources"])["sources"]
        for sid, draw_ref in enumerate(draws["records"]):
            check()
            draw = store.json(draw_ref)
            source = store.json(source_refs[sid])["scene"]
            observation = store.json(draw["observation"])
            if observation != source["observation"] or observation["scene_id"] != sid:
                raise ValueError("sealed source/draw identity differs before label access")
            truth = store.json(draw["sidecar"])
            labels[sid] = reconstruct_truth(observation, truth, split)["labels"]
            original_scenes[sid] = source
        ref, learned, eligible = evaluate_batch(store, observed, labels, output,
            split+"/original", check=check, replay=replay)
        transformed = None
        if observed["roundtrip"] is not None:
            derived = store.json(observed["roundtrip"]["sources"])
            moved_labels = {}
            for sid, source_ref in zip(derived["scene_ids"], derived["sources"]):
                source = store.json(source_ref)["scene"]
                moved_labels[sid] = labels_by_event(original_scenes[sid], source, labels[sid])
            transformed, _, _ = evaluate_batch(store, observed["roundtrip"], moved_labels,
                output, split+"/roundtrip", check=check, replay=replay)
        primary_ref = None
        if split == "deformed_family":
            if observed["scene_ids"] != list(range(512)):
                raise ValueError("primary must cover all512deformed scenes")
            primary = metrics.primary(learned[:, 1, :, :, :, NAMES.index("regret_tM")], eligible, check=check)
            arrays = save_arrays(output, "primary.npz", primary["arrays"], replay=replay)
            primary_ref = save_json(output, "primary.json", {"summary": primary["summary"],
                "source": ref, "split": split, "stage": "selected", "metric": "regret_tM", "arrays": arrays}, replay=replay)
        results.append({"split": split, "original": ref, "roundtrip": transformed, "primary": primary_ref})
    # A changed source cannot acquire an evaluation COMPLETE receipt.
    _, verified, again = admitted_seal(control, finish_ref, freeze_ref, check=check)
    if again != seal_ref or verified != seal:
        raise ValueError("observable seal changed during evaluation")
    return save_json(output, "complete.json", {"schema": "geometric-decision-evaluation-complete-v1",
        "binding": output.binding, "original_scenes": 2048, "results": results,
        "primary": results[-1]["primary"], "replay_requires_exact_outputs": True}, replay=replay)
