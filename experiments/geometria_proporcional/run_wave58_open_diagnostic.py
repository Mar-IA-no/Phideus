#!/usr/bin/env python3
"""Run the audited Wave 58 adaptive open-data diagnostic on CPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
HERE = Path(__file__).resolve().parent
for path in (SRC, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from geometria_proporcional.wave52_policy import authorized_actions  # noqa: E402
from geometria_proporcional.wave55_policy_bridge import action_metric_arrays  # noqa: E402
from geometria_proporcional.wave58_open_diagnostic import (  # noqa: E402
    LOGISTIC_KWARGS,
    Q_GUARD_8,
    Q_GUARD_9,
    Q_GUARD_WAVE57,
    Q_PROPOSER,
    TARGET_ORDER,
    apply_selection,
    array_sha256,
    canonical_candidate_ids,
    derive_targets,
    fit_hgb_state,
    fit_logistic_state,
    fit_mask,
    fit_ridge_state,
    paired_bootstrap_indices,
    paired_delta_ci,
    score_grid,
    select_candidate,
    select_sequential,
    sha256_file,
    summarize_actions,
    token_support,
    validate_runtime_contract,
)
from run_wave56_retrospective import load_utilities  # noqa: E402


CONFIG_PATH = HERE / "configs/wave58_open_model_class_diagnostic.json"
SCIENTIFIC_FILES = (
    "config.json",
    "fit/model_states.json",
    "fit/fit_freeze.json",
    "fit/model_scores.npz",
    "select/selection_grids.json",
    "select/selection_freeze.json",
    "select/selection_arrays.npz",
    "analysis.json",
    "REPORT.md",
    "scores_and_masks.npz",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, default=json_default, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{key: np.asarray(value) for key, value in sorted(arrays.items())})


def worker_config(config: dict[str, Any], phase: str) -> dict[str, Any]:
    allowed = (
        "schema_version",
        "status",
        "claim_scope",
        "device",
        "cpu_threads",
        "penalty",
        "epsilon",
        "proposer_quantiles",
        "guard_quantiles_8",
        "guard_quantiles_9",
        "legacy_guard_quantiles",
        "legacy_fixed_proposer_threshold",
        "minimums",
        "selection",
        "bootstrap",
        "canonical_factors",
        "historical_probe_ids",
    )
    return {"phase": phase, **{key: config[key] for key in allowed}}


def validate_fit_freeze(directory: Path) -> dict[str, Any]:
    freeze = load_json(directory / "fit_freeze.json")
    if freeze.get("phase") != "fit-complete-before-validation-access":
        raise RuntimeError("Wave 58 FIT freeze phase drifted")
    expected = {
        "model_states_sha256": directory / "model_states.json",
        "model_scores_sha256": directory / "model_scores.npz",
    }
    for key, path in expected.items():
        if freeze.get(key) != sha256_file(path):
            raise RuntimeError(f"Wave 58 FIT freeze hash drifted: {key}")
    return freeze


def validate_selection_freeze(directory: Path, fit_directory: Path) -> dict[str, Any]:
    validate_fit_freeze(fit_directory)
    freeze = load_json(directory / "selection_freeze.json")
    if freeze.get("phase") != "selection-complete-before-open-monitor-access":
        raise RuntimeError("Wave 58 SELECT freeze phase drifted")
    expected = {
        "selection_grids_sha256": directory / "selection_grids.json",
        "selection_arrays_sha256": directory / "selection_arrays.npz",
    }
    for key, path in expected.items():
        if freeze.get(key) != sha256_file(path):
            raise RuntimeError(f"Wave 58 SELECT freeze hash drifted: {key}")
    if freeze.get("fit_freeze_sha256") != sha256_file(fit_directory / "fit_freeze.json"):
        raise RuntimeError("Wave 58 SELECT does not bind the FIT freeze")
    return freeze


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def file_record(path: Path, relative_to: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(relative_to)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_config(config: dict[str, Any], *, require_frozen_sources: bool) -> None:
    validate_runtime_contract()
    if config.get("schema_version") != "wave58-open-model-class-diagnostic-v1":
        raise RuntimeError("Wave 58 config schema drifted")
    if config.get("device") != "cpu" or config.get("cpu_threads") != 4:
        raise RuntimeError("Wave 58 CPU contract drifted")
    if config.get("historical_probe_ids") is None or len(config["historical_probe_ids"]) != 24:
        raise RuntimeError("Wave 58 historical ledger drifted")
    if len(canonical_candidate_ids()) != 36:
        raise RuntimeError("Wave 58 canonical roster drifted")
    if tuple(config["proposer_quantiles"]) != Q_PROPOSER:
        raise RuntimeError("Wave 58 proposer grid drifted")
    if tuple(config["guard_quantiles_8"]) != Q_GUARD_8 or tuple(config["guard_quantiles_9"]) != Q_GUARD_9:
        raise RuntimeError("Wave 58 guard grids drifted")
    if tuple(config["legacy_guard_quantiles"]) != Q_GUARD_WAVE57:
        raise RuntimeError("Wave 58 legacy Wave 57 guard grid drifted")
    if tuple(config["replay_exact_files"]) != SCIENTIFIC_FILES:
        raise RuntimeError("Wave 58 replay inventory drifted")
    for record in (config["plan"], config["accepted_audit"]):
        path = REPO / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"bound authority drifted: {record['path']}")
    for name, (relative, expected) in config["inputs"].items():
        actual = sha256_file(REPO / relative)
        if actual != expected:
            raise RuntimeError(f"input hash drifted: {name}")
    if require_frozen_sources:
        implementation_commit = config.get("implementation_commit")
        if implementation_commit == "TO_BE_FROZEN":
            raise RuntimeError("implementation commit is not frozen")
        commit_exists = subprocess.run(
            ["git", "cat-file", "-e", f"{implementation_commit}^{{commit}}"],
            cwd=REPO,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0
        is_ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", implementation_commit, "HEAD"],
            cwd=REPO,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0
        if not commit_exists or not is_ancestor:
            raise RuntimeError("implementation commit is absent from the checkout ancestry")
        expected = config.get("source_sha256", {})
        if sorted(expected) != sorted(config["source_paths"]):
            raise RuntimeError("source hash inventory is incomplete")
        for relative, digest in expected.items():
            if sha256_file(REPO / relative) != digest:
                raise RuntimeError(f"source hash drifted: {relative}")


def load_bundle(path: Path) -> dict[str, np.ndarray]:
    result = load_npz(path)
    required = {
        "pair_token", "target", "design", "hard_actions", "posterior_actions",
        "gain", "weights", "primary", "disagreement",
    }
    if not required <= set(result):
        raise RuntimeError(f"bundle missing {sorted(required - set(result))}")
    return result


def fit_models(data: dict[str, np.ndarray], utilities: np.ndarray, penalty: float):
    targets = derive_targets(data, utilities, penalty)
    active = fit_mask(data)
    x = data["design"][active]
    weights = data["weights"][active]
    states: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}

    state, _ = fit_ridge_state(x, targets["gain"][active], weights)
    states["ridge_gain"] = state
    state, extra = fit_hgb_state(
        "hgb_gain", x, targets["gain"][active], weights, classifier=False, seed=5801
    )
    states["hgb_gain"] = state
    arrays.update(extra)

    specs = (
        ("log_harm", "harm", None),
        ("log_compatibility", "compatibility_loss", None),
        ("log_compatibility_balanced", "compatibility_loss", "balanced"),
        ("log_incompatibility", "posterior_incompatibility", None),
        ("log_incompatibility_balanced", "posterior_incompatibility", "balanced"),
        ("log_accuracy", "accuracy_loss", None),
        ("log_accuracy_balanced", "accuracy_loss", "balanced"),
        ("log_tail", "tail_breach", None),
    )
    for name, target_name, class_weight in specs:
        state, _ = fit_logistic_state(
            x, targets[target_name][active], weights, class_weight=class_weight
        )
        state["target"] = target_name
        state["objective_changed"] = class_weight is not None
        states[name] = state

    for name, target_name, seed in (
        ("hgb_harm", "harm", 5802),
        ("hgb_incompatibility", "posterior_incompatibility", 5803),
    ):
        state, extra = fit_hgb_state(
            name, x, targets[target_name][active], weights, classifier=True, seed=seed
        )
        state["target"] = target_name
        states[name] = state
        arrays.update(extra)

    for name, state in states.items():
        scores = score_grid(state, arrays, data)
        arrays[f"train__score__{name}"] = scores
        state["train_score_sha256"] = array_sha256(scores)
    for name, values in targets.items():
        arrays[f"train__target__{name}"] = values
    arrays["train__fit_mask"] = active
    return states, arrays


def run_fit_phase(stage: Path, output: Path) -> None:
    config = load_json(stage / "config.json")
    validate_runtime_contract()
    data = load_bundle(stage / "train_bundle.npz")
    utilities, _ = load_utilities(stage / "policy_manifest.json")
    states, arrays = fit_models(data, utilities, float(config["penalty"]))
    write_json(output / "model_states.json", {"status": "FIT_COMPLETE", "models": states})
    write_npz(output / "model_scores.npz", arrays)
    write_json(
        output / "fit_freeze.json",
        {
            "phase": "fit-complete-before-validation-access",
            "model_states_sha256": sha256_file(output / "model_states.json"),
            "model_scores_sha256": sha256_file(output / "model_scores.npz"),
            "model_count": len(states),
            "input_sha256": sha256_file(stage / "train_bundle.npz"),
        },
    )


def score_all(states: dict[str, Any], model_arrays: dict[str, np.ndarray], data):
    return {name: score_grid(state, model_arrays, data) for name, state in states.items()}


def canonical_selections(scores, data, utilities, penalty):
    results = {}
    for proposer in ("RIDGE", "HGB"):
        proposer_score = scores["ridge_gain" if proposer == "RIDGE" else "hgb_gain"]
        for risk in ("LOGISTIC", "HGB"):
            harm = scores["log_harm" if risk == "LOGISTIC" else "hgb_harm"]
            incompat = scores[
                "log_incompatibility" if risk == "LOGISTIC" else "hgb_incompatibility"
            ]
            for guard_set, guards in (
                ("HARM", [("harm", harm)]),
                ("INCOMPATIBILITY", [("posterior_incompatibility", incompat)]),
                ("HARM_AND_INCOMPATIBILITY", [("harm", harm), ("posterior_incompatibility", incompat)]),
            ):
                for selector in ("SEQUENTIAL", "JOINT", "JOINT_SHARD_ROBUST"):
                    candidate_id = f"C-{proposer}-{risk}-{guard_set}-{selector}"
                    if selector == "SEQUENTIAL":
                        selected = select_sequential(proposer_score, guards, data, utilities, penalty)
                    else:
                        selected = select_candidate(
                            proposer_score, guards, data, utilities, penalty, selector=selector
                        )
                    results[candidate_id] = selected
    if sorted(results) != sorted(canonical_candidate_ids()):
        raise AssertionError("canonical candidate inventory mismatch")
    return results


def historical_specs(scores):
    ridge = scores["ridge_gain"]
    hgb = scores["hgb_gain"]
    h = scores["log_harm"]
    c = scores["log_compatibility"]
    t = scores["log_tail"]
    i = scores["log_incompatibility"]
    ib = scores["log_incompatibility_balanced"]
    cb = scores["log_compatibility_balanced"]
    a = scores["log_accuracy"]
    ab = scores["log_accuracy_balanced"]
    hh = scores["hgb_harm"]
    hi = scores["hgb_incompatibility"]
    maxht = np.maximum(h, t)
    return {
        "P1-H": (ridge, [("harm", h)], "FIXED", (Q_GUARD_8,)),
        "P1-C": (ridge, [("compatibility_loss", c)], "FIXED", (Q_GUARD_8,)),
        "P1-T": (ridge, [("tail_breach", t)], "FIXED", (Q_GUARD_8,)),
        "P1-HC": (ridge, [("harm", h), ("compatibility_loss", c)], "FIXED", (Q_GUARD_8, Q_GUARD_8)),
        "P1-HT": (ridge, [("harm", h), ("tail_breach", t)], "FIXED", (Q_GUARD_8, Q_GUARD_8)),
        "P1-CT": (ridge, [("compatibility_loss", c), ("tail_breach", t)], "FIXED", (Q_GUARD_8, Q_GUARD_8)),
        "P1-HCT": (ridge, [("harm", h), ("compatibility_loss", c), ("tail_breach", t)], "FIXED", (Q_GUARD_8, Q_GUARD_8, Q_GUARD_8)),
        "P2-H-J": (ridge, [("harm", h)], "JOINT", (Q_GUARD_8,)),
        "P2-H-R": (ridge, [("harm", h)], "JOINT_SHARD_ROBUST", (Q_GUARD_8,)),
        "P2-T-J": (ridge, [("tail_breach", t)], "JOINT", (Q_GUARD_8,)),
        "P2-T-R": (ridge, [("tail_breach", t)], "JOINT_SHARD_ROBUST", (Q_GUARD_8,)),
        "P2-MAXHT-J": (ridge, [("max_harm_tail", maxht)], "JOINT", (Q_GUARD_8,)),
        "P2-MAXHT-R": (ridge, [("max_harm_tail", maxht)], "JOINT_SHARD_ROBUST", (Q_GUARD_8,)),
        "P3-C-N": (ridge, [("harm", h), ("compatibility_loss", c)], "JOINT", (Q_GUARD_8, Q_GUARD_9)),
        "P3-C-B": (ridge, [("harm", h), ("compatibility_loss", cb)], "JOINT", (Q_GUARD_8, Q_GUARD_9)),
        "P3-I-N": (ridge, [("harm", h), ("posterior_incompatibility", i)], "JOINT", (Q_GUARD_8, Q_GUARD_9)),
        "P3-I-B": (ridge, [("harm", h), ("posterior_incompatibility", ib)], "JOINT", (Q_GUARD_8, Q_GUARD_9)),
        "P3-A-N": (ridge, [("harm", h), ("accuracy_loss", a)], "JOINT", (Q_GUARD_8, Q_GUARD_9)),
        "P3-A-B": (ridge, [("harm", h), ("accuracy_loss", ab)], "JOINT", (Q_GUARD_8, Q_GUARD_9)),
        "P4-RL": (ridge, [("harm", h)], "JOINT", (Q_GUARD_9,)),
        "P4-RLI": (ridge, [("harm", h), ("posterior_incompatibility", i)], "JOINT", (Q_GUARD_9, Q_GUARD_9)),
        "P4-HH": (hgb, [("harm", hh)], "JOINT", (Q_GUARD_9,)),
        "P4-HHI": (hgb, [("harm", hh), ("posterior_incompatibility", hi)], "JOINT", (Q_GUARD_9, Q_GUARD_9)),
        "P4-HLI": (hgb, [("harm", h), ("posterior_incompatibility", i)], "JOINT", (Q_GUARD_9, Q_GUARD_9)),
    }


def canonical_components(candidate_id: str) -> tuple[str, str, str, str]:
    parts = candidate_id.split("-", 4)
    if len(parts) != 5 or parts[0] != "C":
        raise ValueError(f"invalid canonical candidate id: {candidate_id}")
    return parts[1], parts[2], parts[3], parts[4]


def historical_selections(scores, data, utilities, penalty, config):
    results = {}
    for probe_id, (proposer, guards, selector, quantile_grids) in historical_specs(scores).items():
        if selector == "FIXED":
            result = select_candidate(
                proposer,
                guards,
                data,
                utilities,
                penalty,
                selector="FIXED-W57-PROPOSER",
                guard_quantile_grids=quantile_grids,
                fixed_proposer_threshold=float(config["legacy_fixed_proposer_threshold"]),
            )
        else:
            result = select_candidate(
                proposer,
                guards,
                data,
                utilities,
                penalty,
                selector=selector,
                guard_quantile_grids=quantile_grids,
            )
        results[probe_id] = result
    if sorted(results) != sorted(config["historical_probe_ids"]):
        raise AssertionError("historical candidate inventory mismatch")
    return results


def selected_arrays(prefix, selections, scores, data, utilities, penalty):
    arrays = {}
    summaries = {}
    for candidate_id, selection in selections.items():
        proposer_name = "hgb_gain" if candidate_id.startswith("C-HGB") else None
        if candidate_id.startswith("P4-H"):
            proposer_name = "hgb_gain"
        if proposer_name is None:
            proposer_name = "ridge_gain"
        guard_map = {name: score for name, score in historical_specs(scores).get(candidate_id, (None, [], None, None))[1]}
        if candidate_id.startswith("C-"):
            _, risk, _, _ = canonical_components(candidate_id)
            guard_map = {
                "harm": scores["hgb_harm" if risk == "HGB" else "log_harm"],
                "posterior_incompatibility": scores[
                    "hgb_incompatibility" if risk == "HGB" else "log_incompatibility"
                ],
            }
        actions, proposals, authorized = apply_selection(
            selection, scores[proposer_name], guard_map, data
        )
        summary, metrics = summarize_actions(actions, data, utilities, penalty)
        safe = candidate_id.replace("-", "__")
        arrays[f"{prefix}__{safe}__proposals"] = proposals
        arrays[f"{prefix}__{safe}__authorized"] = authorized
        arrays[f"{prefix}__{safe}__actions"] = actions
        for metric in ("accuracy", "compatible", "regret", "worst_regret"):
            arrays[f"{prefix}__{safe}__metric__{metric}"] = metrics[metric]
        summaries[candidate_id] = {
            "status": selection.get("status", "PASS"),
            "reason": selection.get("reason"),
            "summary": summary,
            "proposal_support": token_support(proposals, data["primary"]),
            "authorization_support": token_support(authorized, data["primary"]),
        }
    return summaries, arrays


def run_select_phase(stage: Path, output: Path) -> None:
    config = load_json(stage / "config.json")
    fit_freeze = validate_fit_freeze(stage / "fit")
    data = load_bundle(stage / "validation_bundle.npz")
    utilities, _ = load_utilities(stage / "policy_manifest.json")
    states = load_json(stage / "fit/model_states.json")["models"]
    model_arrays = load_npz(stage / "fit/model_scores.npz")
    scores = score_all(states, model_arrays, data)
    canonical = canonical_selections(scores, data, utilities, float(config["penalty"]))
    historical = historical_selections(scores, data, utilities, float(config["penalty"]), config)
    legacy_wave57 = select_sequential(
        scores["ridge_gain"],
        [("harm", scores["log_harm"])],
        data,
        utilities,
        float(config["penalty"]),
        guard_quantiles=Q_GUARD_WAVE57,
    )
    canonical_summaries, arrays = selected_arrays(
        "validation", canonical, scores, data, utilities, float(config["penalty"])
    )
    historical_summaries, historical_arrays = selected_arrays(
        "validation", historical, scores, data, utilities, float(config["penalty"])
    )
    arrays.update(historical_arrays)
    for name, score in scores.items():
        arrays[f"validation__score__{name}"] = score
    arrays["validation__pair_token"] = data["pair_token"]
    arrays["validation__primary"] = data["primary"]
    write_json(
        output / "selection_grids.json",
        {
            "canonical": canonical,
            "historical": historical,
            "legacy_wave57": legacy_wave57,
        },
    )
    write_npz(output / "selection_arrays.npz", arrays)
    write_json(
        output / "selection_freeze.json",
        {
            "phase": "selection-complete-before-open-monitor-access",
            "canonical": {key: value["selected"] for key, value in canonical.items()},
            "historical": {key: value["selected"] for key, value in historical.items()},
            "legacy_wave57": legacy_wave57["selected"],
            "canonical_summaries": canonical_summaries,
            "historical_summaries": historical_summaries,
            "selection_grids_sha256": sha256_file(output / "selection_grids.json"),
            "selection_arrays_sha256": sha256_file(output / "selection_arrays.npz"),
            "fit_freeze_sha256": sha256_file(stage / "fit/fit_freeze.json"),
            "fit_freeze": fit_freeze,
        },
    )


def restore_selection(selected: dict[str, Any], selector: str) -> dict[str, Any]:
    return {"selector": selector, "selected": selected, "grid": []}


def restore_frozen_selection(
    selected: dict[str, Any], selector: str, summary: dict[str, Any]
) -> dict[str, Any]:
    result = restore_selection(selected, selector)
    result["status"] = summary.get("status", "PASS")
    result["reason"] = summary.get("reason")
    return result


def verify_legacy(
    stage,
    states,
    validation,
    monitor,
    utilities,
    penalty,
    val_scores,
    mon_scores,
    frozen_selected,
):
    recomputed = restore_selection(frozen_selected, "SEQUENTIAL")
    actions, proposals, authorized = apply_selection(
        recomputed,
        val_scores["ridge_gain"],
        {"harm": val_scores["log_harm"]},
        validation,
    )
    mon_actions, mon_prop, mon_auth = apply_selection(
        recomputed,
        mon_scores["ridge_gain"],
        {"harm": mon_scores["log_harm"]},
        monitor,
    )
    validation_metrics = action_metric_arrays(
        actions, validation["target"], utilities, penalty
    )
    monitor_metrics = action_metric_arrays(
        mon_actions, monitor["target"], utilities, penalty
    )
    # References remain unopened until the independent refit/reselect/frozen
    # selection has already produced validation and monitor consequences.
    legacy_selection = load_npz(stage / "legacy_selection.npz")
    legacy_results = load_npz(stage / "legacy_results.npz")
    checks = {}
    checks["ridge_validation_score"] = np.array_equal(
        val_scores["ridge_gain"], legacy_selection["score__mean_proposer"], equal_nan=True
    )
    checks["harm_validation_score"] = np.array_equal(
        val_scores["log_harm"], legacy_selection["score__harm_guard"], equal_nan=True
    )
    state_pairs = {
        "ridge_mean": (states["ridge_gain"]["mean"], "gate_fit_archive__model__mean_proposer__mean"),
        "ridge_scale": (states["ridge_gain"]["scale"], "gate_fit_archive__model__mean_proposer__scale"),
        "ridge_coef": (states["ridge_gain"]["coef"], "gate_fit_archive__model__mean_proposer__coef"),
        "ridge_intercept": (states["ridge_gain"]["intercept"], "gate_fit_archive__model__mean_proposer__intercept"),
        "harm_mean": (states["log_harm"]["mean"], "gate_fit_archive__model__harm_guard__mean"),
        "harm_scale": (states["log_harm"]["scale"], "gate_fit_archive__model__harm_guard__scale"),
        "harm_coef": (states["log_harm"]["coef"], "gate_fit_archive__model__harm_guard__coef"),
        "harm_intercept": (states["log_harm"]["intercept"], "gate_fit_archive__model__harm_guard__intercept"),
        "harm_classes": (states["log_harm"]["classes"], "gate_fit_archive__model__harm_guard__classes"),
        "harm_n_iter": (states["log_harm"]["n_iter"], "gate_fit_archive__model__harm_guard__n_iter"),
    }
    for name, (observed, legacy_key) in state_pairs.items():
        checks[f"state__{name}"] = np.array_equal(
            np.asarray(observed), legacy_results[legacy_key]
        )
    checks.update(
        {
            "proposer_threshold": float(recomputed["selected"]["proposer_threshold"])
            == float(legacy_selection["selection__proposer__threshold"]),
            "guard_threshold": float(recomputed["selected"]["guard_thresholds"][0])
            == float(legacy_selection["selection__guard__threshold"]),
            "validation_proposals": np.array_equal(proposals, legacy_selection["selection__proposer__proposals"]),
            "validation_authorized": np.array_equal(authorized, legacy_selection["selection__guard__authorized"]),
            "validation_actions": np.array_equal(actions, legacy_selection["selection__guard__actions"]),
            "ridge_monitor_score": np.array_equal(mon_scores["ridge_gain"], legacy_results["score__mean_proposer"], equal_nan=True),
            "harm_monitor_score": np.array_equal(mon_scores["log_harm"], legacy_results["score__harm_guard"], equal_nan=True),
        }
    )
    checks["monitor_proposals"] = np.array_equal(mon_prop, legacy_results["proposal_mask"])
    checks["monitor_authorized"] = np.array_equal(mon_auth, legacy_results["arm__mean_plus_harm_guard__overrides"])
    checks["monitor_actions"] = np.array_equal(mon_actions, legacy_results["arm__mean_plus_harm_guard__actions"])
    for metric in (
        "accuracy",
        "compatible",
        "regret",
        "worst_regret",
        "regret_by_policy",
        "compatible_by_policy",
    ):
        checks[f"validation_metric__{metric}"] = np.array_equal(
            validation_metrics[metric],
            legacy_selection[f"selection__guard__metric__{metric}"],
        )
        checks[f"monitor_metric__{metric}"] = np.array_equal(
            monitor_metrics[metric],
            legacy_results[f"arm__mean_plus_harm_guard__metric__{metric}"],
        )
    if not all(checks.values()):
        raise RuntimeError(f"LEGACY-W57 exact replay failed: {checks}")
    return {"all_exact": True, "checks": checks, "selection": recomputed["selected"]}


def dominates(left, right, atol=1e-12):
    directions = {"accuracy": 1, "compatible": 1, "regret": -1, "worst_regret": -1}
    no_worse = True
    strict = False
    for split in ("validation", "monitor"):
        for metric, direction in directions.items():
            delta = direction * (left[split][metric] - right[split][metric])
            no_worse &= delta >= -atol
            strict |= delta > atol
    return bool(no_worse and strict)


def nominate(results, hard):
    eligible = []
    for candidate_id, row in results.items():
        if row.get("status") != "PASS":
            continue
        if row["validation_support"] < 25 or row["monitor_support"] < 25:
            continue
        ok = True
        for split in ("validation", "monitor"):
            metrics = row[split]
            baseline = hard[split]
            ok &= (
                metrics["accuracy"] >= baseline["accuracy"] - 0.01
                and metrics["compatible"] >= baseline["compatible"]
                and metrics["worst_regret"] <= baseline["worst_regret"] + 0.01
            )
        if ok:
            eligible.append(candidate_id)
    front = [
        candidate_id for candidate_id in eligible
        if not any(dominates(results[other], results[candidate_id]) for other in eligible if other != candidate_id)
    ]
    if not front:
        return {"status": "NO_CANDIDATE", "eligible": eligible, "pareto_front": []}

    def complexity(candidate_id):
        proposer_name, risk_name, guard_set, _ = canonical_components(candidate_id)
        guards = 2 if guard_set == "HARM_AND_INCOMPATIBILITY" else 1
        proposer = 1 if proposer_name == "HGB" else 0
        risk = 1 if risk_name == "HGB" else 0
        return guards, proposer, risk

    def key(candidate_id):
        row = results[candidate_id]
        return (
            row["monitor"]["regret"], row["validation"]["regret"],
            row["monitor"]["worst_regret"], row["validation"]["worst_regret"],
            -row["monitor"]["compatible"], -row["validation"]["compatible"],
            -row["monitor"]["accuracy"], -row["validation"]["accuracy"],
            *complexity(candidate_id), candidate_id,
        )
    chosen = min(front, key=key)
    return {
        "status": "NOMINATED",
        "label": "OPEN-DATA / ADAPTIVE / SELECTED-AFTER-MONITOR-INSPECTION",
        "candidate_id": chosen,
        "eligible": eligible,
        "pareto_front": front,
    }


def report_markdown(analysis):
    lines = [
        "# Wave 58 — open model-class diagnostic",
        "",
        "> **Scope:** `OPEN-DATA / ADAPTIVE / SELECTED-AFTER-MONITOR-INSPECTION / CPU-ONLY`",
        "",
        "Este diagnóstico usa el monitor ya abierto de la Ola 57 y no puede validar una arquitectura.",
        "",
        "## Inventario y replay legacy",
        "",
        f"Candidatos canónicos: `{analysis['candidate_counts']['canonical']}`. Probes históricos: `{analysis['candidate_counts']['historical']}`.",
        "",
        f"`LEGACY-W57`: `{analysis['legacy_wave57']['all_exact']}` en {len(analysis['legacy_wave57']['checks'])} comprobaciones.",
        "",
        "| Referencia | Validation acc. | Validation compat. | Validation regret | Monitor acc. | Monitor compat. | Monitor regret |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| hard-set | {analysis['hard_set']['validation']['accuracy']:.6f} | {analysis['hard_set']['validation']['compatible']:.6f} | {analysis['hard_set']['validation']['regret']:.6f} | {analysis['hard_set']['monitor']['accuracy']:.6f} | {analysis['hard_set']['monitor']['compatible']:.6f} | {analysis['hard_set']['monitor']['regret']:.6f} |",
        f"| oracle positive-gain | {analysis['oracle_positive_gain']['validation']['accuracy']:.6f} | {analysis['oracle_positive_gain']['validation']['compatible']:.6f} | {analysis['oracle_positive_gain']['validation']['regret']:.6f} | {analysis['oracle_positive_gain']['monitor']['accuracy']:.6f} | {analysis['oracle_positive_gain']['monitor']['compatible']:.6f} | {analysis['oracle_positive_gain']['monitor']['regret']:.6f} |",
        "",
        "## Nominación diagnóstica",
        "",
        f"Estado: `{analysis['nomination']['status']}`.",
    ]
    if analysis["nomination"]["status"] == "NOMINATED":
        cid = analysis["nomination"]["candidate_id"]
        row = analysis["canonical_candidates"][cid]
        lines.extend(
            [
                f"Candidato: `{cid}` (`{analysis['nomination']['label']}`).",
                "",
                "| Split | Accuracy | Compatibilidad | Regret | Worst regret |",
                "|---|---:|---:|---:|---:|",
                f"| validation | {row['validation']['accuracy']:.6f} | {row['validation']['compatible']:.6f} | {row['validation']['regret']:.6f} | {row['validation']['worst_regret']:.6f} |",
                f"| open monitor | {row['monitor']['accuracy']:.6f} | {row['monitor']['compatible']:.6f} | {row['monitor']['regret']:.6f} | {row['monitor']['worst_regret']:.6f} |",
            ]
        )
    for title, rows in (
        ("Roster canónico completo", analysis["canonical_candidates"]),
        ("Ledger histórico completo", analysis["historical_probes"]),
    ):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| ID | Estado | Thresholds val. | Shards factibles | Val. acc./comp./regret/worst | Val. soporte | Mon. acc./comp./regret/worst | Mon. soporte | Δregret val./mon. |",
                "|---|---|---|---|---|---:|---|---:|---|",
            ]
        )
        for candidate_id in sorted(rows):
            row = rows[candidate_id]
            selected = row["validation_selected"]
            thresholds = (
                "HARD_ONLY"
                if selected.get("terminal") == "HARD_ONLY"
                else json.dumps(
                    {
                        "p": selected["proposer_threshold"],
                        "g": selected["guard_thresholds"],
                    },
                    separators=(",", ":"),
                )
            )
            shards = selected.get("shards", {})
            shard_status = (
                ",".join(
                    f"{key}:{'Y' if value['feasible'] else 'N'}"
                    for key, value in sorted(shards.items())
                )
                if shards
                else "—"
            )
            val = row["validation"]
            mon = row["monitor"]
            delta_val = row["validation_minus_hard"]["regret"]["mean_diff"]
            delta_mon = row["monitor_minus_hard"]["regret"]["mean_diff"]
            lines.append(
                f"| `{candidate_id}` | {row['status']} | `{thresholds}` | {shard_status} | "
                f"{val['accuracy']:.6f}/{val['compatible']:.6f}/{val['regret']:.6f}/{val['worst_regret']:.6f} | "
                f"{row['validation_support']} | "
                f"{mon['accuracy']:.6f}/{mon['compatible']:.6f}/{mon['regret']:.6f}/{mon['worst_regret']:.6f} | "
                f"{row['monitor_support']} | {delta_val:+.6f}/{delta_mon:+.6f} |"
            )
    lines.extend(
        [
            "",
            "Las matrices completas de propuestas, autorizaciones, acciones y overrides se preservan en `scores_and_masks.npz`; las grillas y deltas de las cuatro métricas permanecen en `analysis.json` y `select/selection_grids.json`.",
            "",
            "## Frontera de autoridad",
            "",
            "El resultado sólo genera hipótesis. No promueve una arquitectura ni declara GO/NO-GO.",
            "",
        ]
    )
    return "\n".join(lines)


def run_monitor_phase(stage: Path, output: Path) -> None:
    config = load_json(stage / "config.json")
    penalty = float(config["penalty"])
    freeze = validate_selection_freeze(stage / "select", stage / "fit")
    validation = load_bundle(stage / "validation_bundle.npz")
    utilities, _ = load_utilities(stage / "policy_manifest.json")
    states = load_json(stage / "fit/model_states.json")["models"]
    model_arrays = load_npz(stage / "fit/model_scores.npz")
    val_scores = score_all(states, model_arrays, validation)
    canonical = {
        key: restore_frozen_selection(
            value,
            key.rsplit("-", 1)[-1],
            freeze["canonical_summaries"][key],
        )
        for key, value in freeze["canonical"].items()
    }
    historical = {
        key: restore_frozen_selection(
            value, "HISTORICAL", freeze["historical_summaries"][key]
        )
        for key, value in freeze["historical"].items()
    }
    val_arrays = load_npz(stage / "select/selection_arrays.npz")
    # The open monitor is not opened until both preceding freezes validate.
    monitor = load_bundle(stage / "monitor_bundle.npz")
    mon_scores = score_all(states, model_arrays, monitor)
    canonical_results = {}
    historical_results = {}
    final_arrays = {
        key: value
        for key, value in val_arrays.items()
        if key.startswith("validation__")
    }
    hard = {}
    oracle = {}
    for split, data, scores, arrays in (
        ("validation", validation, val_scores, val_arrays),
        ("monitor", monitor, mon_scores, None),
    ):
        hard_summary, hard_metrics = summarize_actions(data["hard_actions"], data, utilities, penalty)
        hard[split] = hard_summary
        oracle_mask = data["disagreement"] & (data["gain"] > 1e-12)
        oracle_actions = np.where(
            oracle_mask, data["posterior_actions"], data["hard_actions"]
        )
        oracle_summary, oracle_metrics = summarize_actions(
            oracle_actions, data, utilities, penalty
        )
        oracle[split] = oracle_summary
        indices = np.flatnonzero(data["primary"])
        tokens = data["pair_token"][indices].astype(str)
        bootstrap = paired_bootstrap_indices(tokens)
        final_arrays[f"{split}__bootstrap_indices"] = bootstrap
        final_arrays[f"{split}__hard_actions"] = data["hard_actions"]
        final_arrays[f"{split}__oracle_positive_gain_actions"] = oracle_actions
        final_arrays[f"{split}__oracle_positive_gain_overrides"] = oracle_mask
        for metric, values in hard_metrics.items():
            final_arrays[f"{split}__hard_metric__{metric}"] = values
        for metric, values in oracle_metrics.items():
            final_arrays[f"{split}__oracle_positive_gain_metric__{metric}"] = values
        targets = derive_targets(data, utilities, penalty)
        for name, values in targets.items():
            final_arrays[f"{split}__target__{name}"] = values
        for group, selections, destination in (
            ("canonical", canonical, canonical_results),
            ("historical", historical, historical_results),
        ):
            specs = historical_specs(scores) if group == "historical" else None
            for candidate_id, selection in selections.items():
                safe = candidate_id.replace("-", "__")
                if split == "validation":
                    metrics = {
                        metric: arrays[f"validation__{safe}__metric__{metric}"]
                        for metric in ("accuracy", "compatible", "regret", "worst_regret")
                    }
                    actions = arrays[f"validation__{safe}__actions"]
                    authorized = arrays[f"validation__{safe}__authorized"]
                else:
                    proposer_name = "hgb_gain" if candidate_id.startswith("C-HGB") or candidate_id.startswith("P4-H") else "ridge_gain"
                    if group == "canonical":
                        _, risk, _, _ = canonical_components(candidate_id)
                        guard_map = {
                            "harm": scores["hgb_harm" if risk == "HGB" else "log_harm"],
                            "posterior_incompatibility": scores["hgb_incompatibility" if risk == "HGB" else "log_incompatibility"],
                        }
                    else:
                        guard_map = {name: score for name, score in specs[candidate_id][1]}
                    actions, proposals, authorized = apply_selection(
                        selection, scores[proposer_name], guard_map, data
                    )
                    _, metrics = summarize_actions(actions, data, utilities, penalty)
                    final_arrays[f"monitor__{safe}__actions"] = actions
                    final_arrays[f"monitor__{safe}__authorized"] = authorized
                    final_arrays[f"monitor__{safe}__proposals"] = proposals
                summary = {metric: float(metrics[metric][indices].mean()) for metric in metrics}
                contrasts = {
                    metric: paired_delta_ci(metrics[metric][indices], hard_metrics[metric][indices], bootstrap)
                    for metric in metrics
                }
                row = destination.setdefault(candidate_id, {})
                row["status"] = selection.get("status", "PASS")
                row["reason"] = selection.get("reason")
                row[split] = summary
                row[f"{split}_support"] = int((authorized[data["primary"]]).any(axis=1).sum())
                row[f"{split}_minus_hard"] = contrasts
                row[f"{split}_selected"] = selection["selected"]
                threshold_prefix = f"{split}__{safe}__threshold"
                chosen = selection["selected"]
                if chosen.get("terminal") == "HARD_ONLY":
                    final_arrays[f"{threshold_prefix}__proposer"] = np.asarray(np.nan)
                    final_arrays[f"{threshold_prefix}__guards"] = np.empty(0, dtype=np.float64)
                else:
                    final_arrays[f"{threshold_prefix}__proposer"] = np.asarray(
                        chosen["proposer_threshold"], dtype=np.float64
                    )
                    final_arrays[f"{threshold_prefix}__guards"] = np.asarray(
                        chosen["guard_thresholds"], dtype=np.float64
                    )
                if split == "monitor":
                    for metric, values in metrics.items():
                        final_arrays[f"monitor__{safe}__metric__{metric}"] = values
    nomination = nominate(canonical_results, hard)
    legacy = verify_legacy(
        stage,
        states,
        validation,
        monitor,
        utilities,
        penalty,
        val_scores,
        mon_scores,
        freeze["legacy_wave57"],
    )
    analysis = {
        "schema_version": config["schema_version"],
        "status": "COMPLETE",
        "scope": config["claim_scope"],
        "legacy_wave57": legacy,
        "hard_set": hard,
        "oracle_positive_gain": oracle,
        "canonical_candidates": canonical_results,
        "historical_probes": historical_results,
        "nomination": nomination,
        "candidate_counts": {"canonical": len(canonical_results), "historical": len(historical_results)},
        "bootstrap": config["bootstrap"],
        "scientific_decision": None,
        "architecture_promoted": False,
    }
    for name, score in mon_scores.items():
        final_arrays[f"monitor__score__{name}"] = score
    for name, score in val_scores.items():
        final_arrays[f"validation__score__{name}"] = score
    write_json(output / "analysis.json", analysis)
    (output / "REPORT.md").write_text(report_markdown(analysis), encoding="utf-8")
    write_npz(output / "scores_and_masks.npz", final_arrays)


def worker_main(phase: str, stage: Path, output: Path) -> None:
    if os.geteuid() != 65534 or os.getegid() != 65534:
        raise RuntimeError("Wave 58 worker must run as nobody/nogroup")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Wave 58 requires CUDA_VISIBLE_DEVICES empty")
    expected_entries = {
        "fit": {"config.json", "train_bundle.npz", "policy_manifest.json"},
        "select": {
            "config.json",
            "validation_bundle.npz",
            "policy_manifest.json",
            "fit",
        },
        "monitor": {
            "config.json",
            "validation_bundle.npz",
            "monitor_bundle.npz",
            "legacy_selection.npz",
            "legacy_results.npz",
            "policy_manifest.json",
            "fit",
            "select",
        },
    }
    observed = {path.name for path in stage.iterdir()}
    if observed != expected_entries[phase]:
        raise RuntimeError(
            f"Wave 58 {phase} staging allowlist mismatch: {sorted(observed)}"
        )
    if output.exists():
        if any(output.iterdir()):
            raise RuntimeError("Wave 58 worker output must be empty")
    else:
        output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=4):
        oversized = [
            item for item in threadpool_info() if int(item.get("num_threads", 0)) > 4
        ]
        if oversized:
            raise RuntimeError(f"Wave 58 threadpool limit ineffective: {oversized}")
        if phase == "fit":
            run_fit_phase(stage, output)
        elif phase == "select":
            run_select_phase(stage, output)
        elif phase == "monitor":
            run_monitor_phase(stage, output)
        else:
            raise ValueError(phase)


def copy_readonly(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target)
        for path in sorted(target.rglob("*"), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        target.chmod(0o555)
    else:
        shutil.copy2(source, target)
        target.chmod(0o444)


def run_worker(phase: str, files: dict[str, Path], destination: Path) -> float:
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"wave58-{phase}-") as raw:
        root = Path(raw)
        root.chmod(0o755)
        stage = root / "stage"
        stage.mkdir(mode=0o755)
        for relative, source in files.items():
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            copy_readonly(source, target)
        for directory in sorted((p for p in stage.rglob("*") if p.is_dir()), reverse=True):
            directory.chmod(0o555)
        stage.chmod(0o555)
        worker_output = root / "worker-output"
        worker_output.mkdir(mode=0o770)
        os.chown(worker_output, 65534, 65534)
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "PYTHONPATH": f"{SRC}:{HERE}",
        }
        command = [
            "setpriv", "--reuid=65534", "--regid=65534", "--clear-groups",
            sys.executable, str(Path(__file__).resolve()),
            "--worker-phase", phase, "--stage", str(stage), "--phase-output", str(worker_output),
        ]
        subprocess.run(command, check=True, env=env, cwd=stage)
        shutil.copytree(worker_output, destination)
    return time.monotonic() - started


def compare_outputs(primary: Path, replay: Path) -> dict[str, Any]:
    checks = {}
    for relative in SCIENTIFIC_FILES:
        left, right = primary / relative, replay / relative
        if relative.endswith(".npz"):
            la, ra = load_npz(left), load_npz(right)
            exact = la.keys() == ra.keys() and all(
                np.array_equal(la[key], ra[key], equal_nan=True)
                if la[key].dtype.kind in "fc"
                else np.array_equal(la[key], ra[key])
                for key in la
            )
        else:
            exact = left.read_bytes() == right.read_bytes()
        checks[relative] = bool(exact)
    return {"all_exact": all(checks.values()), "checks": checks}


def coordinator(config_path: Path, output: Path, reference: Path | None) -> None:
    config = load_json(config_path)
    validate_config(config, require_frozen_sources=True)
    if output.exists():
        raise RuntimeError("Wave 58 output already exists")
    before = {
        name: sha256_file(REPO / relative)
        for name, (relative, _) in config["inputs"].items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    try:
        shutil.copy2(config_path, work / "config.json")
        worker_configs = work / "_worker_configs"
        for phase in ("fit", "select", "monitor"):
            write_json(worker_configs / f"{phase}.json", worker_config(config, phase))
        timings = {}
        inputs = {name: REPO / value[0] for name, value in config["inputs"].items()}
        timings["fit_seconds"] = run_worker(
            "fit",
            {
                "config.json": worker_configs / "fit.json",
                "train_bundle.npz": inputs["train_bundle"],
                "policy_manifest.json": inputs["policy_manifest"],
            },
            work / "fit",
        )
        validate_fit_freeze(work / "fit")
        timings["select_seconds"] = run_worker(
            "select",
            {
                "config.json": worker_configs / "select.json",
                "validation_bundle.npz": inputs["validation_bundle"],
                "policy_manifest.json": inputs["policy_manifest"],
                "fit": work / "fit",
            },
            work / "select",
        )
        validate_selection_freeze(work / "select", work / "fit")
        timings["monitor_seconds"] = run_worker(
            "monitor",
            {
                "config.json": worker_configs / "monitor.json",
                "validation_bundle.npz": inputs["validation_bundle"],
                "monitor_bundle.npz": inputs["monitor_bundle"],
                "legacy_selection.npz": inputs["legacy_selection"],
                "legacy_results.npz": inputs["legacy_results"],
                "policy_manifest.json": inputs["policy_manifest"],
                "fit": work / "fit",
                "select": work / "select",
            },
            work / "monitor-final",
        )
        for name in ("analysis.json", "REPORT.md", "scores_and_masks.npz"):
            shutil.move(work / "monitor-final" / name, work / name)
        (work / "monitor-final").rmdir()
        shutil.rmtree(worker_configs)
        after = {
            name: sha256_file(REPO / relative)
            for name, (relative, _) in config["inputs"].items()
        }
        if before != after:
            raise RuntimeError("Wave 58 original inputs changed during execution")
        replay = compare_outputs(reference, work) if reference is not None else None
        write_json(
            work / "runtime.json",
            {
                "device": "cpu",
                "threads": 4,
                "worker_uid": 65534,
                "worker_gid": 65534,
                "phase_seconds": timings,
                "max_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                "replay": replay,
            },
        )
        records = [
            file_record(path, work)
            for path in sorted(work.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        ]
        write_json(
            work / "manifest.json",
            {
                "schema_version": config["schema_version"],
                "implementation_commit": config["implementation_commit"],
                "source_sha256": config["source_sha256"],
                "input_sha256_before": before,
                "input_sha256_after": after,
                "replay_exact_files": list(SCIENTIFIC_FILES),
                "replay_exclusions": ["runtime.json", "manifest.json"],
                "files": records,
            },
        )
        if replay is not None and not replay["all_exact"]:
            raise RuntimeError("Wave 58 replay is not exact")
        os.replace(work, output)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--worker-phase", choices=("fit", "select", "monitor"))
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--phase-output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker_phase:
        worker_main(args.worker_phase, args.stage.resolve(strict=True), args.phase_output)
        return
    if args.output is None:
        raise SystemExit("--output is required")
    coordinator(
        args.config.resolve(strict=True),
        args.output.resolve(),
        args.reference.resolve(strict=True) if args.reference else None,
    )


if __name__ == "__main__":
    main()
