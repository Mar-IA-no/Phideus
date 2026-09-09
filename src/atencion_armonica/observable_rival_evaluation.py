"""Privileged retrospective diagnostics, reachable only after the 96-scene seal."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import resource
import time
import uuid

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from . import observable_source_rivals as math
from . import observable_rival_campaign as campaign

READERS = ("pairs_structure", "local_compatibility", "shared_source", "decoupled_source",
           "fixed_pairs", "fixed_shared", "historical")
READER_SEEDS = (2026090891, 2026090892, 2026090893)


def labels(partition, n):
    result = np.empty(n, dtype=np.int64)
    for i, group in enumerate(math.validate_partition(partition, n)):
        result[list(group)] = i
    return result


def validate_truth(scene, truth):
    """Reconstruct preserved source law and exact q32; never redraw a scene."""
    n = len(scene["q32"])
    if truth["scene_id"] != scene["scene_id"] or truth["sigma_cents"] != 2.:
        raise ValueError("truth identity or noise differs")
    permutation = np.asarray(truth["permutation"])
    if permutation.dtype.kind not in "iu" or not np.array_equal(np.sort(permutation), np.arange(n)):
        raise ValueError("invalid source permutation")
    split = scene["split"]
    branch = ("base-high" if split == "ood_beta" else
              "deformed-low" if split == "deformed_family" else "base-low")
    beta_range, gamma_range, _ = math.BRANCHES[branch]
    expected_k = (4,) if split == "ood_polyphony" else (2, 3)
    if len(truth["sources"]) not in expected_k:
        raise ValueError("planted source cardinality differs")
    ideal, source_labels, partials = [], [], []
    for i, source in enumerate(truth["sources"]):
        ns = source["indices"]
        if (not 100 <= source["f0"] <= 500 or not beta_range[0] <= source["beta"] <= beta_range[1]
                or not gamma_range[0] <= source["gamma"] <= gamma_range[1]
                or not 4 <= len(ns) <= 8 or ns != sorted(set(ns))
                or any(type(x) is not int or not 1 <= x <= 8 for x in ns)):
            raise ValueError("planted parameters outside original support")
        ns = np.asarray(ns, dtype=np.float64)
        # Preserve the generator's arithmetic order, including its left-associated sums.
        ideal.extend((np.log(source["f0"])+np.log(ns)
                      +.5*np.log1p(source["beta"]*ns**2+source["gamma"]*ns**4)).tolist())
        source_labels.extend([i]*len(ns))
        partials.extend(source["indices"])
    observed_ideal = np.asarray(truth["log_f_ideal"], dtype=np.float64)
    noise = np.asarray(truth["sensor_log_noise"], dtype=np.float64)
    if (len(ideal) != n or observed_ideal.shape != (n,) or noise.shape != (n,)
            or not np.isfinite(noise).all()
            or not np.array_equal(np.asarray(ideal)[permutation], observed_ideal)
            or not np.array_equal(np.asarray(source_labels)[permutation], truth["source_ids"])
            or not np.array_equal(np.asarray(partials)[permutation], truth["partial_indices"])):
        raise ValueError("sidecar source reconstruction differs")
    measured = observed_ideal+noise
    center = float(measured[np.argsort(permutation)].mean())
    if center != truth["mean_log_f_observed"]:
        raise ValueError("sidecar centering differs")
    order = np.asarray(scene["canonical_to_observed"])
    delivered = (measured-center).astype(np.float32)[order].astype(np.float64)
    if not np.array_equal(delivered, scene["q32"]):
        raise ValueError("sidecar does not reconstruct delivered q32 exactly")
    y = np.asarray(truth["source_ids"])[order]
    p = math.signature([np.flatnonzero(y == i).tolist() for i in range(len(truth["sources"]))])
    return {"partition": p, "labels": y, "ideal_canonical": observed_ideal[order],
            "branch": branch, "sigma_cents": truth["sigma_cents"]}


def reader_scene_means(records, split):
    """Require all 512 x 3 x 3 old cells, then average within scene, never pool seeds."""
    indexed = {}
    for row in records:
        key = row["scene_id"], row["checkpoint_seed"], row["reader_seed"]
        if row["split"] != split or key in indexed or set(row["metrics"]) != set(READERS):
            raise ValueError("old metric identity, reader roster or duplication differs")
        indexed[key] = row
    expected = set((i, c, s) for i in range(512) for c in campaign.CHECKPOINTS for s in READER_SEEDS)
    if set(indexed) != expected:
        raise ValueError("old metric cell roster differs")
    output = {}
    for i in campaign.selected_ids()[split]:
        values = {name: [indexed[i, c, s]["metrics"][name]["ari"]
                        for c in campaign.CHECKPOINTS for s in READER_SEEDS] for name in READERS}
        if not all(np.isfinite(v).all() and np.all(np.abs(v) <= 1) for v in map(np.asarray, values.values())):
            raise ValueError("invalid old ARI")
        output[i] = {name: float(np.mean(v)) for name, v in values.items()}
    return output


def evaluate_scene(scene, observable, truth_fit, truth, old_means):
    q, n = scene["q32"], len(scene["q32"])
    rows = []
    for family in math.FAMILIES:
        eligible = sorted([f for f in observable["fits"] if f["status"] == "FITTED"],
                          key=lambda f: (f["families"][family]["UB"], math.signature(f["partition"])))
        choices = []
        for fit in eligible[:2]:
            b = fit["families"][family]
            branch = fit["branches"][b["upper_branch"]]
            k = len(fit["partition"])
            choices.append({"partition": fit["partition"], "LB": b["LB"], "UB": b["UB"],
                "branch": b["upper_branch"], "k": k, "sizes": [len(g) for g in fit["partition"]],
                "nominal_continuous_dimension": (3*k-1 if b["upper_branch"] == "deformed-low" else 2*k-1),
                "ari": float(adjusted_rand_score(truth["labels"], labels(fit["partition"], n))),
                "rms_cents": branch["rms_cents"],
                "quantization_objective_bound": branch["quantization_objective_bound"],
                "sample_fit_status": branch["sample_fit_status"]})
        planted = truth_fit["families"][family]
        pbranch = truth_fit["branches"][planted["upper_branch"]]
        rows.append({"split": scene["split"], "scene_id": scene["scene_id"], "family": family,
            "N": n, "q_tie_count": scene["q_tie_count"], "status": observable["status"],
            "best": choices[0] if choices else None, "second": choices[1] if len(choices) > 1 else None,
            "second_minus_best": choices[1]["UB"]-choices[0]["UB"] if len(choices) > 1 else None,
            "rival_margin": math.rival_margin(observable["fits"], truth_fit, family),
            "planted_reference": {**planted, "partition": truth_fit["partition"],
                "rms_cents": pbranch["rms_cents"], "authority": "PRIVILEGED_GRID_FIT"},
            "true_parameters_reference": {**math.witness_metrics(q, truth["ideal_canonical"]),
                "branch": truth["branch"], "authority": "PRIVILEGED_OFF_GRID_NOT_A_CANDIDATE"},
            "coverage": {"original_pool": scene["pool_count"],
                "neighbors_retained": scene["neighbor_count_retained"],
                "fitted": len(eligible), "outside_cardinality": len(observable["fits"])-len(eligible),
                "planted_in_pool": any(c["origin"] == "pool" and math.signature(c["partition"])
                    == math.signature(truth_fit["partition"]) for c in scene["candidates"]),
                "planted_in_neighbors": any(c["origin"] == "neighbor" and math.signature(c["partition"])
                    == math.signature(truth_fit["partition"]) for c in scene["candidates"])},
            "old_reader_mean_ari": old_means, "noiseless_collision_status": "NOT_TESTED"})
    return rows


def distribution(values):
    a = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not len(a):
        return {"n": 0, "median": None, "q10": None, "q90": None, "negative_fraction": None}
    return {"n": len(a), "median": float(np.median(a)), "q10": float(np.quantile(a, .1)),
            "q90": float(np.quantile(a, .9)), "negative_fraction": float(np.mean(a < 0))}


def summarize(rows):
    output = []
    for split in campaign.SPLITS:
        for family in math.FAMILIES:
            selected = [row for row in rows if row["split"] == split and row["family"] == family]
            if [r["scene_id"] for r in selected] != campaign.selected_ids()[split]:
                raise ValueError("summary denominator is not the exact 24-scene roster")
            associations = {}
            for margin_name in ("rival_delta_UB", "second_minus_best"):
                pairs = [(r["rival_margin"]["delta_UB"] if margin_name == "rival_delta_UB"
                          else r["second_minus_best"], r["old_reader_mean_ari"]) for r in selected]
                pairs = [(x, ari) for x, ari in pairs if x is not None]
                associations[margin_name] = {}
                for name in READERS:
                    x = np.asarray([p[0] for p in pairs])
                    y = np.asarray([1-p[1][name] for p in pairs])
                    value = (float(spearmanr(x, y).statistic) if len(x) >= 2
                             and np.ptp(x) > 0 and np.ptp(y) > 0 else None)
                    associations[margin_name][name] = {"n": len(x), "spearman": value}
            output.append({"split": split, "family": family, "n": len(selected),
                "statuses": dict(Counter(r["status"] for r in selected)),
                "grid_comparison_statuses": dict(Counter(r["rival_margin"]["status"] for r in selected)),
                "rival_delta_UB": distribution([r["rival_margin"]["delta_UB"] for r in selected]),
                "second_minus_best": distribution([r["second_minus_best"] for r in selected]),
                "best_ari": distribution([r["best"]["ari"] if r["best"] else None for r in selected]),
                "best_J": distribution([r["best"]["UB"] if r["best"] else None for r in selected]),
                "best_rms_cents": distribution([r["best"]["rms_cents"] if r["best"] else None for r in selected]),
                "spearman_error_associations": associations})
    return output


def run(root, replay=False):
    root = Path(root)
    # This call verifies ALL observable scenes before any sidecar/old metric parse.
    manifest, record_refs = campaign.verify_seal(root)
    spent = campaign.elapsed_budget(root)
    if spent >= 7200:
        raise TimeoutError("cumulative campaign time exhausted")
    release = campaign.checked_json(manifest["binding"]["release"])
    fitter = None if replay else math.GroupFitter(device=manifest["device"])
    attempt = root/"attempts"/(uuid.uuid4().hex+".json")
    campaign.write_once(attempt, {"kind": "evaluation_replay" if replay else "evaluation",
                                  "started_unix": time.time()})
    start = time.monotonic()
    rows, refs, source_refs = [], [], []
    try:
        for split in campaign.SPLITS:
            data_ref = release["splits"][split]["data"]
            sidepath, sideref, dm = campaign.read_bundle_member(data_ref, "sidecars.jsonl")
            sidecars = [json.loads(line) for line in sidepath.read_bytes().splitlines()]
            metricpath, metricref, _ = campaign.read_bundle_member(
                release["splits"][split]["evaluation"]["result"], "metrics.json")
            means = reader_scene_means(json.loads(metricpath.read_bytes()), split)
            source_refs.extend([sideref, metricref])
            if len(sidecars) != 512:
                raise ValueError("sidecar roster incomplete")
            for scene, obsref in zip(manifest["scenes"], record_refs):
                if scene["split"] != split:
                    continue
                if spent+time.monotonic()-start >= 7200:
                    raise TimeoutError("cumulative campaign time exhausted")
                sid = scene["scene_id"]
                if sidecars[sid]["split_seed"] != dm["binding"]["split_seed"]:
                    raise ValueError("sidecar split identity differs")
                truth = validate_truth(scene, sidecars[sid])
                refpath = root/"privileged"/split/f"{sid:05d}.json"
                if refpath.exists():
                    saved = json.loads(refpath.read_bytes())
                    if saved["observable"] != obsref or saved["sidecar"] != sideref:
                        raise ValueError("privileged reference input changed")
                    result = saved["result"]
                else:
                    if replay:
                        raise FileNotFoundError("replay cannot fit a missing privileged reference")
                    result = math.fit_candidates(scene["q32"], [truth["partition"]], fitter)
                    campaign.write_once(refpath, {"observable": obsref, "sidecar": sideref,
                                                  "result": result})
                if replay:
                    fitted = math.replay_fits(scene["q32"], result["group_factors"], [truth["partition"]])
                    if campaign.encoded(fitted) != campaign.encoded(result["fits"]):
                        raise ValueError("privileged witness replay differs")
                observable = campaign.checked_json(obsref)
                rows.extend(evaluate_scene(scene, observable, result["fits"][0], truth, means[sid]))
                refs.append(campaign.reference(refpath))
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 4*1024**2:
                    raise MemoryError("evaluation RSS exceeded four GiB")
                if sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) > 5*1024**3:
                    raise OSError("campaign artifacts exceeded five GiB")
        payload = {"schema": "observable-rivals-evaluation-v1", "rows": rows,
                   "summary": summarize(rows), "privileged_fits": refs,
                   "sources": source_refs, "observable_seal": campaign.reference(root/"observable_seal.json")}
        if replay:
            if campaign.encoded(payload) != (root/"evaluation.json").read_bytes():
                raise ValueError("evaluation replay differs")
            return {"status": "PASS", "count": 96, "seconds": time.monotonic()-start,
                    "evaluation": campaign.reference(root/"evaluation.json")}
        return campaign.write_once(root/"evaluation.json", payload)
    finally:
        campaign.write_once(attempt.with_suffix(".end.json"), {"seconds": time.monotonic()-start})
