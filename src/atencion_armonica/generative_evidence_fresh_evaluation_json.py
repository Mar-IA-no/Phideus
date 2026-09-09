"""JSON-equality successor of the frozen fresh-test evaluation module.

Static copy: the only functional change is the verifier/reader import.
Metrics, truth reconstruction, output schemas and replay are unchanged.

Post-seal metrics and read-only scientific replay for the four fresh tests.

No sampler, fitter construction, model loading or forward. The public port
requires the complete 45-output authority before parsing any test sidecar.
The caller owns the single heavy operator and cumulative four-hour budget.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import generative_evidence as ge
from . import generative_evidence_evaluation as evaluation
from . import generative_evidence_inference as inference
from . import generative_evidence_references as references
from .generative_evidence_fresh_store import CANONICAL, TESTS, FreshObservableStore
from .generative_evidence_reuse import ROOT, VerifiedBytes
from .generative_evidence_selection import SelectionArtifacts
from .generative_evidence_supervision import reconstruct_truth
from .learned_partition_metrics import METRICS
from .partial_compatibility_cache import encoded

TEMPORARY = ROOT/".agent-work/phideus-generative-fresh-evaluation-tests-20260909"
KEYS = ("arm", "checkpoint_seed", "reader_seed", "intervention")


def _key(value):
    return tuple(value[k] for k in KEYS)


def _means(array, mask):
    return array[mask].mean(axis=0).tolist() if mask.any() else None


def summarize_readouts(records, *, split, eligible, sham):
    """45 metric arrays, not 45 independent models or 9 independent scenes."""
    if split not in TESTS or eligible.dtype != bool or eligible.shape != (512,) or len(sham) != 512:
        raise ValueError("test readouts require all scenes and the common observable mask")
    expected = {_key(row) for row in inference.inference_roster()}
    values = {}
    for record in records:
        key, metric = _key(record), record["metrics"]
        if (key not in expected or key in values or metric.dtype != np.float64
                or metric.shape != (512, len(METRICS)) or not np.isfinite(metric[eligible]).all()
                or not np.isnan(metric[~eligible]).all()):
            raise ValueError("readout identity, metric extent or common support differs")
        values[key] = metric
    if set(values) != expected:
        raise ValueError("summary requires all 45 readouts")
    learned = {arm: np.stack([values[arm, cp, seed, "original"] for cp in ge.CHECKPOINTS
                            for seed in ge.READER_SEEDS], axis=1).reshape(512, 3, 3, len(METRICS))
               for arm in ge.ARMS}
    primary = evaluation.summarize_learned(learned, split=split, eligible=eligible)
    indices = primary.pop("bootstrap_indices")
    changed = []
    for row in sham:
        mask = row["changed_mask"]
        if (not isinstance(mask, list) or any(type(v) is not bool for v in mask)
                or row["status"] != ("INPUT_CHANGED" if any(mask) else "INPUT_UNCHANGED")):
            raise ValueError("sham support differs from delivered changed mask")
        changed.append(any(mask))
    changed = np.asarray(changed, bool)
    subsets = {}
    for name, mask in (("changed", eligible & changed), ("unchanged", eligible & ~changed)):
        subsets[name] = {"scene_ids": np.flatnonzero(mask).tolist(), "count": int(mask.sum()),
            "arm_means": {a: _means(v.mean(axis=(1, 2)), mask) for a, v in learned.items()},
            "cell_means": {a: _means(v, mask) for a, v in learned.items()},
            "authority": "DESCRIPTIVE_SUBSET_NOT_REDEFINED_PRIMARY"}
    cell_contrasts = {f"generative-minus-{arm}": _means(learned["generative"]-learned[arm], eligible)
                      for arm in ("decoupled", "local")}
    interventions = {}
    for mode in ("zero", "decoupled"):
        adjusted = np.stack([values["generative", cp, seed, mode] for cp in ge.CHECKPOINTS
                             for seed in ge.READER_SEEDS], axis=1).reshape(512, 3, 3, len(METRICS))
        interventions[mode] = {"cell_means": _means(adjusted, eligible),
            "mean": _means(adjusted.mean(axis=(1, 2)), eligible),
            "changed_minus_original_cell_means": _means(adjusted-learned["generative"], eligible),
            "authority": "FIXED_HEAD_INFERENCE_INTERVENTION_NOT_RETRAINING"}
    signs = {name: np.sign(np.asarray(delta)).astype(int).tolist() if delta is not None else None
             for name, delta in cell_contrasts.items()}
    return {"primary": primary, "cell_contrasts": cell_contrasts, "cell_contrast_signs": signs,
            "sham_subsets": subsets,
            "interventions": interventions, "metric_order": list(METRICS)}, indices


def summarize_references(scenes, eligible):
    if len(scenes) != 512 or eligible.dtype != bool or eligible.shape != (512,):
        raise ValueError("reference summary requires the full common scene roster")
    base, extended = [np.full((512, len(METRICS)), np.nan, np.float64) for _ in range(2)]
    historical = np.empty((512, 3, len(METRICS)), np.float64)
    planted = {name: 0 for name in ("pool", "neighbor", "absent")}
    oracle = np.full((512, 2), np.nan, np.float64)
    for i, scene in enumerate(scenes):
        if scene["coverage"]["has_output"] is not bool(eligible[i]):
            raise ValueError("system reference has different output support")
        planted[scene["coverage"]["planted"]] += 1
        for name, target in (("base", base), ("extended", extended)):
            value = scene["references"][name]
            if eligible[i]:
                target[i] = [value["metrics"][m] for m in METRICS]
            elif value is not None:
                raise ValueError("absent output has a system partition")
        for j, cp in enumerate(ge.CHECKPOINTS):
            historical[i, j] = [scene["references"]["historical"][str(cp)]["metrics"][m] for m in METRICS]
        if eligible[i]:
            oracle[i] = [scene["oracle"]["maximum_ari"], scene["oracle"]["minimum_vi"]]
    if (not np.isfinite(historical).all() or not np.isfinite(base[eligible]).all()
            or not np.isfinite(extended[eligible]).all() or not np.isfinite(oracle[eligible]).all()):
        raise ValueError("nonfinite observed system metrics")
    summary = {"count": 512, "output_count": int(eligible.sum()), "coverage": float(eligible.mean()),
        "base": {"common_support_mean": _means(base, eligible)},
        "extended": {"common_support_mean": _means(extended, eligible)},
        "historical": {str(cp): {"all_512_mean": historical[:, j].mean(axis=0).tolist(),
                                 "common_support_mean": _means(historical[:, j], eligible)}
                       for j, cp in enumerate(ge.CHECKPOINTS)},
        "planted_presence": planted, "oracle_mean": _means(oracle, eligible),
        "oracle_order": ["maximum_ari", "minimum_vi"], "metric_order": list(METRICS),
        "authority": "SYSTEM_REFERENCES_NOT_CAPACITY_MATCHED_ARMS"}
    return summary, {"base": base, "extended": extended, "historical": historical, "oracle": oracle}


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _root_ref(root, ref):
    return {**ref, "path": (Path(root)/ref["path"]).relative_to(ROOT).as_posix()}


class _Artifacts:
    path = SelectionArtifacts.path
    reference = SelectionArtifacts.reference
    json = SelectionArtifacts.json
    arrays = SelectionArtifacts.arrays

    def __init__(self, root, *, split, binding, replay):
        self.root, self.binding, self.replay = Path(root).resolve(), _plain(binding), replay
        if split not in TESTS or (self.root != CANONICAL/split/"evaluation" and not
                (self.root.is_relative_to(TEMPORARY) and self.root != TEMPORARY)):
            raise ValueError("evaluation root or split differs")
        if replay and not self.path("index.json").is_file():
            raise FileNotFoundError("replay requires a complete primary evaluation")
        self.root.mkdir(parents=True, exist_ok=True)
        self.write_json("binding.json", self.binding)

    def write_json(self, relative, value):
        if self.replay and not self.path(relative).is_file():
            raise FileNotFoundError("replay cannot repair a missing scientific JSON")
        return SelectionArtifacts.write_json(self, relative, _plain(value))

    def write_arrays(self, relative, arrays):
        if self.replay and not self.path(relative).is_file():
            raise FileNotFoundError("replay cannot repair missing scientific arrays")
        return SelectionArtifacts.write_arrays(self, relative, arrays)


def _run(split, *, freeze_ref, check, replay):
    if not callable(check) or split not in TESTS:
        raise ValueError("post-seal evaluation needs a fixed test and resource callback")
    check()
    from .generative_evidence_fresh_inference_json import verify_predictions, read_prediction
    verified = verify_predictions(split, freeze_ref=freeze_ref, check=check)
    # No sidecar parser or output store is reachable until all 45 are verified.
    from .generative_evidence_fresh_data import FreshObservations
    observations = FreshObservations(split, freeze_ref=freeze_ref, check=check)
    store = FreshObservableStore(CANONICAL, binding={"test_freeze": freeze_ref})
    binding = {"test_freeze": freeze_ref, "prediction_seal": verified["seal"], "split": split}
    artifacts = _Artifacts(CANONICAL/split/"evaluation", split=split, binding=binding, replay=replay)
    choices = verified["choices"]
    if choices["scene_ids"] != list(range(512)) or len(choices["records"]) != 512:
        raise ValueError("post-seal system choice roster differs")
    partitions, scenes, scene_refs = [], [], []
    for scene_id in range(512):
        check()
        observation = observations.observation(scene_id)
        draw = observations.files.reader.json(observations.index["records"][scene_id])
        # This is the only file-backed privileged parse in the fresh evaluator.
        truth = observations.files.reader.json(draw["sidecar"])
        labels = reconstruct_truth(observation, truth, split)["labels"]
        fitted, fit_ref = store.load_fit(split, scene_id)
        if fitted["observation"] != observation:
            raise ValueError("post-seal fit and observation differ")
        ps = ge.partitions_checked(sorted(ge.law.signature(row["partition"])
            for row in fitted["inventory"]["candidates"] if row["status"] == "SUPPORTED"), len(labels))
        if replay:
            q = np.sort(np.asarray(observation["log_f"], np.float32), kind="stable")
            replayed = ge.law.replay_fits(q, fitted["group_factors"], ps)
            if encoded(replayed) != encoded(fitted["fits"]):
                raise ValueError("CPU factor replay differs from preserved fits")
        choice = choices["records"][scene_id]
        if choice["scene_id"] != scene_id:
            raise ValueError("post-seal system choice order differs")
        result = references.evaluate_references(ps, fitted["inventory"], choice["choice"], labels)
        value = {"scene_id": scene_id, "binding": binding, "fit": _root_ref(store.root, fit_ref),
                 "draw": _root_ref(observations.files.root, observations.index["records"][scene_id]),
                 "result": result}
        scene_refs.append(artifacts.write_json(f"scenes/{scene_id:05d}.json", value))
        partitions.append(ps)
        scenes.append(result)
    eligible = np.asarray([bool(ps) for ps in partitions], bool)
    metrics, readout_refs, all_metrics = [], [], []
    candidate_metrics = [scene["candidate_metrics"] for scene in scenes]
    positions = {_key(record): i for i, record in enumerate(verified["records"])}
    for record in verified["records"]:
        check()
        index = positions[_key(record)]
        prediction = read_prediction(verified, record)
        chosen = evaluation.chosen_metrics(partitions, candidate_metrics, **prediction)
        if not np.array_equal(chosen["eligible"], eligible):
            raise ValueError("learned output support differs from common candidates")
        metrics.append({**{k: record[k] for k in KEYS}, "metrics": chosen["metrics"]})
        all_metrics.append(chosen["metrics"])
        readout = {"identity": {k: record[k] for k in KEYS}, "prediction": record["prediction"],
                   "decisions": chosen["decisions"]}
        if record["intervention"] != "original":
            original_key = ("generative", record["checkpoint_seed"], record["reader_seed"], "original")
            original = read_prediction(verified, verified["records"][positions[original_key]])
            diagnostics = []
            for i, ps in enumerate(partitions):
                a, b = prediction["offsets"][i:i+2]
                diagnostics.append(inference.dependence(original["components"][a:b],
                                                       prediction["components"][a:b], ps))
            readout["dependence"] = diagnostics
        readout_refs.append(artifacts.write_json(f"readouts/{index:02d}.json", readout))
    shams = []
    for cp in ge.CHECKPOINTS:
        check()
        delivered = VerifiedBytes(ROOT).json(verified["index"]["delivered"][str(cp)])
        shams.append(delivered["sham"])
    if any(encoded(sham) != encoded(shams[0]) for sham in shams[1:]):
        raise ValueError("checkpoint sham support is not shared")
    learned_summary, bootstrap = summarize_readouts(metrics, split=split, eligible=eligible, sham=shams[0])
    system_summary, system_arrays = summarize_references(scenes, eligible)
    check()
    arrays = artifacts.write_arrays("metrics.npz", {"readouts": np.stack(all_metrics, axis=1),
        "eligible": eligible, "bootstrap_indices": bootstrap, **system_arrays})
    summary = artifacts.write_json("summary.json", {"binding": binding, "learned": learned_summary,
        "systems": system_summary, "sham": shams[0], "status": "EVALUATED_NOT_PROMOTED"})
    result = {"schema": "generative-evidence-fresh-evaluation-v1", "binding": binding,
        "status": "EVALUATED_NOT_PROMOTED", "scene_ids": list(range(512)),
        "scenes": scene_refs, "readouts": readout_refs, "metrics": arrays, "summary": summary,
        "prediction_count": 45, "truth_reconstruction": "EQUATION_NOISE_ORDER_Q32_WITHOUT_SAMPLER"}
    ref = artifacts.write_json("index.json", result)
    check()
    return {"path": artifacts.path(ref["path"]).relative_to(ROOT).as_posix(), "sha256": ref["sha256"]}


def evaluate_test(split, *, freeze_ref, check):
    return _run(split, freeze_ref=freeze_ref, check=check, replay=False)


def replay_test(split, *, freeze_ref, check):
    return _run(split, freeze_ref=freeze_ref, check=check, replay=True)
