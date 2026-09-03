#!/usr/bin/env python3
"""Adjudicate the fresh Wave 55 conservative posterior-to-policy bridge."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import itertools
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from scipy.special import expit

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geometria_proporcional.wave49_schema import sha256_file, write_json  # noqa: E402
from geometria_proporcional.wave52_policy import (  # noqa: E402
    authorized_actions,
    explicit_set_actions,
)
from geometria_proporcional.wave53_uncertainty import (  # noqa: E402
    independent_nonempty_mass,
    paired_bootstrap_indices,
    paired_delta_ci,
)
from geometria_proporcional.wave54_joint_set import (  # noqa: E402
    expected_regret_from_mass,
    posterior_mass,
    target_set_indices,
)
from geometria_proporcional.wave55_policy_bridge import (  # noqa: E402
    HARD_ONLY,
    action_metric_arrays,
    algebraic_sign,
    bridge_actions,
    override_diagnostics,
    select_gamma,
)

PLAN_PATH = REPO_ROOT / "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_55_CONSERVATIVE_POLICY_BRIDGE_PLAN.md"
CONFIG_DEFAULT = REPO_ROOT / "experiments/geometria_proporcional/configs/wave55_policy_bridge.json"
PRIMITIVES = REPO_ROOT / "src/geometria_proporcional/wave55_policy_bridge.py"
BRIDGE_POSTERIORS = {
    "bridge_joint_full": "joint_full",
    "bridge_joint_unary_cardinality": "joint_unary_cardinality",
    "bridge_independent_platt": "independent_platt",
    "bridge_joint_target_shuffled": "joint_full_target_shuffled",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--wave52-dir", type=Path, required=True)
    parser.add_argument("--wave53-dir", type=Path, required=True)
    parser.add_argument("--wave54-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def digest(path: Path) -> str:
    return sha256_file(path.resolve(strict=True))


def require_hash(path: Path, expected: str) -> dict[str, Any]:
    actual = digest(path)
    if actual != expected:
        raise RuntimeError(f"hash mismatch for {path}: {actual} != {expected}")
    return {"path": str(path.resolve()), "sha256": actual, "bytes": path.stat().st_size}


def require_sources_at_head(paths: list[Path]) -> str:
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT, text=True
    ).strip()
    if dirty:
        raise RuntimeError("tracked worktree must be clean before Wave 55 adjudication")
    for path in paths:
        relative = path.resolve(strict=True).relative_to(REPO_ROOT)
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(relative)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def prepare_output(path: Path, force: bool) -> Path | None:
    archived = None
    if path.exists():
        if not force:
            raise FileExistsError(f"output exists: {path}")
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        archived = path.with_name(f"{path.name}.superseded_{stamp}")
        path.rename(archived)
    path.mkdir(parents=True)
    return archived


def load_bundle(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    required = {
        "pair_token", "cluster_id", "target", "per_seed_logits", "ensemble_logits",
        "design_stratum", "cardinality", "split_role",
    }
    if required - set(result):
        raise RuntimeError(f"bundle missing keys: {sorted(required - set(result))}")
    if not np.all(np.isfinite(result["ensemble_logits"])):
        raise RuntimeError("bundle logits are non-finite")
    return result


def utilities_from_manifest(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    levels = np.asarray(payload["levels"], dtype=np.float64)
    permutations = np.asarray(payload["rank_permutations"], dtype=np.int64)
    utilities = levels[permutations]
    expected = np.asarray(list(itertools.permutations(range(4))), dtype=np.int64)
    if permutations.shape != (24, 4) or set(map(tuple, permutations)) != set(map(tuple, expected)):
        raise RuntimeError("policy manifest does not contain all 24 ordinal policies")
    return utilities


def posterior_masses(
    logits: np.ndarray,
    theta: dict[str, np.ndarray],
    platt: dict[str, Any],
) -> dict[str, np.ndarray]:
    calibrated = expit(float(platt["coefficient"]) * logits + float(platt["intercept"]))
    return {
        "joint_full": posterior_mass(logits, theta["joint_full"], "joint_full"),
        "joint_unary_cardinality": posterior_mass(
            logits, theta["joint_unary_cardinality"], "joint_unary_cardinality"
        ),
        "independent_platt": independent_nonempty_mass(calibrated)[1],
        "joint_full_target_shuffled": posterior_mass(
            logits, theta["joint_full_target_shuffled"], "joint_full"
        ),
    }


def hard_policy(logits: np.ndarray, utilities: np.ndarray, tau: float) -> np.ndarray:
    actions, _ = explicit_set_actions(expit(logits), utilities, tau)
    return actions


def summary(metrics: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    return {
        name: float(values[mask].mean())
        for name, values in metrics.items()
        if values.ndim == 1
    }


def population_mask(bundle: dict[str, np.ndarray], primary: bool) -> np.ndarray:
    if not primary:
        return np.ones(len(bundle["pair_token"]), dtype=bool)
    return (bundle["design_stratum"].astype(str) == "NEAR_RIVAL") & (bundle["cardinality"] >= 2)


def evaluate_bridge_grid(
    bundle: dict[str, np.ndarray],
    action_risk: np.ndarray,
    hard_actions: np.ndarray,
    utilities: np.ndarray,
    config: dict[str, Any],
    mask: np.ndarray,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    hard_metrics = action_metric_arrays(
        hard_actions, bundle["target"], utilities, config["incompatible_regret_penalty"]
    )
    candidates = []
    states = {}
    for gamma in config["gamma_grid"]:
        state = bridge_actions(
            action_risk,
            hard_actions,
            gamma,
            atol=float(config["gamma_advantage_atol"]),
        )
        metrics = action_metric_arrays(
            state["actions"], bundle["target"], utilities, config["incompatible_regret_penalty"]
        )
        row = {"gamma": gamma, **summary(metrics, mask)}
        candidates.append(row)
        states[str(gamma)] = {"state": state, "metrics": metrics}
    selected = select_gamma(
        candidates,
        hard_accuracy=float(hard_metrics["accuracy"][mask].mean()),
        hard_compatible=float(hard_metrics["compatible"][mask].mean()),
        accuracy_margin=float(config["selection"]["accuracy_noninferiority_margin"]),
    )
    return selected, states


def evaluate_bundle(
    bundle: dict[str, np.ndarray],
    theta: dict[str, np.ndarray],
    platt: dict[str, Any],
    utilities: np.ndarray,
    config: dict[str, Any],
    selected_gamma: dict[str, float | str],
) -> dict[str, Any]:
    masses = posterior_masses(bundle["ensemble_logits"], theta, platt)
    decisions = {
        name: expected_regret_from_mass(mass, utilities, config["incompatible_regret_penalty"])
        for name, mass in masses.items()
    }
    hard = hard_policy(bundle["ensemble_logits"], utilities, config["hard_set_tau"])
    actions = {
        "hard_set_policy": hard,
        "pure_joint_full": decisions["joint_full"]["actions"],
        "oracle_set_then_utility": authorized_actions(bundle["target"], utilities),
    }
    bridges = {}
    for arm, posterior in BRIDGE_POSTERIORS.items():
        bridge = bridge_actions(
            decisions[posterior]["action_risk"], hard, selected_gamma[arm],
            atol=float(config["gamma_advantage_atol"]),
        )
        bridges[arm] = bridge
        actions[arm] = bridge["actions"]
    metrics = {
        arm: action_metric_arrays(
            action, bundle["target"], utilities, config["incompatible_regret_penalty"]
        )
        for arm, action in actions.items()
    }
    diagnostics = {
        arm: override_diagnostics(
            bridge, hard, bundle["target"], utilities, config["incompatible_regret_penalty"]
        )
        for arm, bridge in bridges.items()
    }
    return {
        "masses": masses,
        "decisions": decisions,
        "actions": actions,
        "bridges": bridges,
        "metrics": metrics,
        "override_diagnostics": diagnostics,
    }


def diagnostic_summary(values: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in values.items()
        if not isinstance(value, np.ndarray)
    }


def contrast_cis(
    evaluation: dict[str, Any], mask: np.ndarray, bootstrap: np.ndarray
) -> dict[str, Any]:
    pairs = {
        "bridge_full_minus_hard": ("bridge_joint_full", "hard_set_policy", ("regret", "accuracy", "compatible")),
        "bridge_full_minus_pure_joint": ("bridge_joint_full", "pure_joint_full", ("regret", "accuracy")),
        "bridge_full_minus_independent": ("bridge_joint_full", "bridge_independent_platt", ("regret",)),
        "bridge_full_minus_shuffled": ("bridge_joint_full", "bridge_joint_target_shuffled", ("regret",)),
        "bridge_full_minus_unary_cardinality": ("bridge_joint_full", "bridge_joint_unary_cardinality", ("regret", "accuracy")),
    }
    output = {}
    for name, (left, right, metrics) in pairs.items():
        output[name] = {
            metric: paired_delta_ci(
                evaluation["metrics"][left][metric][mask],
                evaluation["metrics"][right][metric][mask],
                bootstrap,
            )
            for metric in metrics
        }
    return output


def selector_delta_state(evaluation: dict[str, Any], mask: np.ndarray) -> dict[str, float]:
    metrics = evaluation["metrics"]
    mean = lambda arm, key: float(metrics[arm][key][mask].mean())
    return {
        "regret_vs_hard": mean("bridge_joint_full", "regret") - mean("hard_set_policy", "regret"),
        "accuracy_vs_hard": mean("bridge_joint_full", "accuracy") - mean("hard_set_policy", "accuracy"),
        "compatible_vs_hard": mean("bridge_joint_full", "compatible") - mean("hard_set_policy", "compatible"),
        "regret_vs_pure_joint": mean("bridge_joint_full", "regret") - mean("pure_joint_full", "regret"),
        "accuracy_vs_pure_joint": mean("bridge_joint_full", "accuracy") - mean("pure_joint_full", "accuracy"),
        "regret_vs_independent": mean("bridge_joint_full", "regret") - mean("bridge_independent_platt", "regret"),
        "regret_vs_shuffled": mean("bridge_joint_full", "regret") - mean("bridge_joint_target_shuffled", "regret"),
    }


def criteria(contrasts: dict[str, Any], selector_sensitive: bool, replay_exact: bool | None, config: dict[str, Any]) -> dict[str, Any]:
    c = config["diagnostic_criteria"]
    hard_regret = contrasts["bridge_full_minus_hard"]["regret"]
    hard_accuracy = contrasts["bridge_full_minus_hard"]["accuracy"]
    hard_compatible = contrasts["bridge_full_minus_hard"]["compatible"]
    pure_regret = contrasts["bridge_full_minus_pure_joint"]["regret"]
    pure_accuracy = contrasts["bridge_full_minus_pure_joint"]["accuracy"]
    independent = contrasts["bridge_full_minus_independent"]["regret"]
    shuffled = contrasts["bridge_full_minus_shuffled"]["regret"]
    values = {
        "regret_vs_hard": hard_regret["mean_diff"] <= -c["regret_reduction_vs_hard_min"] and hard_regret["ci95_high"] < 0.0,
        "accuracy_vs_hard": hard_accuracy["ci95_low"] >= c["accuracy_noninferiority_ci_low"],
        "compatible_vs_hard": hard_compatible["ci95_low"] >= c["compatible_noninferiority_ci_low"],
        "accuracy_vs_pure_joint": pure_accuracy["ci95_low"] > c["accuracy_superiority_vs_pure_joint_ci_low"],
        "regret_vs_pure_joint": pure_regret["ci95_high"] <= c["regret_noninferiority_vs_pure_joint_ci_high"],
        "regret_vs_independent": independent["mean_diff"] <= -c["regret_reduction_vs_controls_min"] and independent["ci95_high"] < 0.0,
        "regret_vs_shuffled": shuffled["mean_diff"] <= -c["regret_reduction_vs_controls_min"] and shuffled["ci95_high"] < 0.0,
        "not_selector_sensitive": not selector_sensitive,
        "replay_exact": replay_exact,
    }
    resolved = [value for value in values.values() if value is not None]
    return {"conditions": values, "n_true": int(sum(resolved)), "n_resolved": len(resolved), "all_satisfied": all(resolved) if replay_exact is not None else None}


def compare_reference(output: Path, reference: Path) -> dict[str, Any]:
    if output.resolve() == reference.resolve():
        raise ValueError("adjudication replay cannot reference itself")
    checks = {}
    with np.load(output / "result_arrays.npz", allow_pickle=False) as left, np.load(reference / "result_arrays.npz", allow_pickle=False) as right:
        checks["result_arrays"] = set(left.files) == set(right.files) and all(
            np.array_equal(left[key], right[key]) for key in left.files
        )
    checks["analysis_core"] = (output / "analysis_core.json").read_bytes() == (reference / "analysis_core.json").read_bytes()
    if not all(checks.values()):
        raise RuntimeError(f"Wave 55 adjudication replay mismatch: {checks}")
    return {"array_exact": checks, "all_exact": True}


def verify_primary_theta(
    theta: dict[str, np.ndarray], freeze: dict[str, Any]
) -> None:
    for name, value in theta.items():
        frozen = np.asarray(freeze["selected_models"][name]["theta"], dtype=np.float64)
        if not np.array_equal(value, frozen):
            raise RuntimeError(f"selection_state theta differs from semantic freeze for {name}")
    for name in ("joint_full", "joint_unary_cardinality"):
        primary = np.asarray(freeze["selections"][name]["primary"]["theta"], dtype=np.float64)
        if not np.array_equal(theta[name], primary):
            raise RuntimeError(f"{name} theta is not the primary Wave 54 selection")


def validate_replay_provenance(
    bundle_dir: Path, output: Path, reference: Path | None
) -> None:
    manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text(encoding="utf-8"))
    if reference is None:
        if manifest.get("execution_mode") == "replay":
            raise RuntimeError("a replay bundle cannot adjudicate as a primary run")
        return
    if output.resolve() == reference.resolve():
        raise ValueError("replay output and reference output must differ")
    if manifest.get("execution_mode") != "replay":
        raise RuntimeError("adjudication replay requires an independently regenerated replay bundle")
    receipt_path = bundle_dir / "preparation_replay.json"
    if not receipt_path.is_file():
        raise RuntimeError("replay bundle lacks its preparation replay receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("all_exact") is not True:
        raise RuntimeError("preparation replay is not exact")
    primary_bundle_dir = Path(receipt["reference_dir"]).resolve(strict=True)
    reference_report = json.loads(
        (reference / "REPORT_WAVE55_POLICY_BRIDGE.json").read_text(encoding="utf-8")
    )
    if digest(primary_bundle_dir / "bundle_manifest.json") != reference_report["bundle_manifest_sha256"]:
        raise RuntimeError("replay preparation does not descend from the adjudication reference bundle")


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    archived = prepare_output(output, args.force)
    config_path = args.config.resolve(strict=True)
    sources = [Path(__file__), PRIMITIVES, PLAN_PATH, config_path]
    commit = require_sources_at_head(sources)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bundle_dir = args.bundle_dir.resolve(strict=True)
    reference = args.reference_dir.resolve(strict=True) if args.reference_dir else None
    validate_replay_provenance(bundle_dir, output, reference)
    wave52, wave53, wave54 = args.wave52_dir.resolve(), args.wave53_dir.resolve(), args.wave54_dir.resolve()
    binding = config["source_binding"]
    require_hash(wave52 / "policy_manifest.json", binding["wave52_policy_manifest_sha256"])
    require_hash(wave53 / "platt_calibrator.json", binding["wave53_platt_sha256"])
    require_hash(wave54 / "selection_freeze.json", binding["wave54_selection_freeze_sha256"])
    require_hash(wave54 / "selection_state.npz", binding["wave54_selection_state_sha256"])
    bundle_manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text(encoding="utf-8"))
    decision_path = bundle_dir / "bundles/decision_select.npz"
    monitor_path = bundle_dir / "bundles/sealed_monitor.npz"
    require_hash(decision_path, bundle_manifest["bundles"]["decision_select.npz"])
    decision = load_bundle(decision_path)
    with np.load(wave54 / "selection_state.npz", allow_pickle=False) as data:
        theta = {
            "joint_full": data["theta__joint_full"],
            "joint_unary_cardinality": data["theta__joint_unary_cardinality"],
            "joint_full_target_shuffled": data["theta__joint_full_target_shuffled"],
        }
        observed_fit = data["observed_fit"].astype(bool)
    wave54_freeze = json.loads((wave54 / "selection_freeze.json").read_text(encoding="utf-8"))
    verify_primary_theta(theta, wave54_freeze)
    platt = json.loads((wave53 / "platt_calibrator.json").read_text(encoding="utf-8"))
    utilities = utilities_from_manifest(wave52 / "policy_manifest.json")
    decision_masses = posterior_masses(decision["ensemble_logits"], theta, platt)
    decision_risks = {
        name: expected_regret_from_mass(mass, utilities, config["incompatible_regret_penalty"])["action_risk"]
        for name, mass in decision_masses.items()
    }
    decision_hard = hard_policy(decision["ensemble_logits"], utilities, config["hard_set_tau"])

    selections = {}
    global_selections = {}
    for arm, posterior in BRIDGE_POSTERIORS.items():
        primary_selection, _ = evaluate_bridge_grid(
            decision, decision_risks[posterior], decision_hard, utilities, config,
            population_mask(decision, True),
        )
        global_selection, _ = evaluate_bridge_grid(
            decision, decision_risks[posterior], decision_hard, utilities, config,
            population_mask(decision, False),
        )
        selections[arm] = primary_selection
        global_selections[arm] = global_selection
    selected_gamma = {arm: row["selected"]["gamma"] for arm, row in selections.items()}
    global_gamma = {arm: row["selected"]["gamma"] for arm, row in global_selections.items()}
    write_json(
        output / "selection_freeze.json",
        {
            "phase": "gamma-selection-frozen-before-sealed-monitor-access",
            "git_commit": commit,
            "decision_select_sha256": digest(decision_path),
            "expected_sealed_monitor": bundle_manifest["bundles"]["sealed_monitor.npz"],
            "selected_gamma": selected_gamma,
            "global_sensitivity_gamma": global_gamma,
            "selection_grid": selections,
            "global_selection_grid": global_selections,
            "sealed_monitor_accessed": False,
        },
    )

    require_hash(monitor_path, bundle_manifest["bundles"]["sealed_monitor.npz"])
    monitor = load_bundle(monitor_path)
    overlap = set(decision["pair_token"].astype(str)) & set(monitor["pair_token"].astype(str))
    if overlap:
        raise RuntimeError("decision and monitor bundles overlap")
    primary_mask = population_mask(monitor, True)
    primary_indices = np.flatnonzero(primary_mask)
    bootstrap = paired_bootstrap_indices(
        len(primary_indices), config["bootstrap"]["replicates"], config["bootstrap"]["seed"]
    )
    evaluation = evaluate_bundle(monitor, theta, platt, utilities, config, selected_gamma)
    global_evaluation = evaluate_bundle(monitor, theta, platt, utilities, config, global_gamma)
    primary_state = selector_delta_state(evaluation, primary_mask)
    global_state = selector_delta_state(global_evaluation, primary_mask)
    selector_signs = {
        key: {
            "primary": primary_state[key],
            "global": global_state[key],
            "primary_sign": algebraic_sign(primary_state[key]),
            "global_sign": algebraic_sign(global_state[key]),
            "changed": algebraic_sign(primary_state[key]) != algebraic_sign(global_state[key]),
        }
        for key in primary_state
    }
    selector_sensitive = any(row["changed"] for row in selector_signs.values())
    contrasts = contrast_cis(evaluation, primary_mask, bootstrap)

    common_gamma = selected_gamma["bridge_joint_full"]
    common_selected = {arm: common_gamma for arm in BRIDGE_POSTERIORS}
    common_evaluation = evaluate_bundle(monitor, theta, platt, utilities, config, common_selected)
    support_indices = target_set_indices(monitor["target"])
    absent = np.flatnonzero(~observed_fit)
    absent_mask = np.isin(support_indices, absent)
    support = {
        "absent_set_indices": absent.tolist(),
        "monitor_count": int(absent_mask.sum()),
        "n_min": int(config["unseen_support_n_min"]),
        "status": "EVALUABLE" if absent_mask.sum() >= config["unseen_support_n_min"] else "NOT_EVALUABLE",
    }
    support_bootstrap = None
    if support["status"] == "EVALUABLE":
        support_bootstrap = paired_bootstrap_indices(
            int(absent_mask.sum()), config["bootstrap"]["replicates"], config["bootstrap"]["seed"]
        )
        support["summaries"] = {
            arm: summary(metrics, absent_mask)
            for arm, metrics in evaluation["metrics"].items()
        }
        support["contrasts"] = contrast_cis(evaluation, absent_mask, support_bootstrap)
    else:
        support["summaries"] = None
        support["contrasts"] = None

    arrays: dict[str, np.ndarray] = {
        "pair_token": monitor["pair_token"],
        "target": monitor["target"],
        "primary_mask": primary_mask,
        "bootstrap_indices": bootstrap,
    }
    for arm, action in evaluation["actions"].items():
        arrays[f"actions__{arm}"] = action
    for arm, bridge in evaluation["bridges"].items():
        arrays[f"advantage__{arm}"] = bridge["advantage"]
        arrays[f"override__{arm}"] = bridge["override"]
        arrays[f"estimated_hard_risk__{arm}"] = bridge["hard_risk"]
    for arm, metrics in evaluation["metrics"].items():
        for name, values in metrics.items():
            arrays[f"metric__{arm}__{name}"] = values
    np.savez_compressed(output / "result_arrays.npz", **arrays)
    np.savez_compressed(output / "bootstrap_indices.npz", pair_token=monitor["pair_token"][primary_mask], indices=bootstrap)
    if support_bootstrap is not None:
        np.savez_compressed(
            output / "unseen_support_bootstrap_indices.npz",
            pair_token=monitor["pair_token"][absent_mask],
            indices=support_bootstrap,
        )

    summaries = {
        population: {
            arm: summary(metrics, population_mask(monitor, population == "primary"))
            for arm, metrics in evaluation["metrics"].items()
        }
        for population in ("primary", "all_in_catalog")
    }
    overrides = {
        arm: diagnostic_summary(
            override_diagnostics(
                evaluation["bridges"][arm],
                evaluation["actions"]["hard_set_policy"],
                monitor["target"],
                utilities,
                config["incompatible_regret_penalty"],
                token_mask=primary_mask,
            )
        )
        for arm in evaluation["bridges"]
    }
    common_sensitivity = {
        arm: summary(common_evaluation["metrics"][arm], primary_mask)
        for arm in BRIDGE_POSTERIORS
    }
    core = {
        "selected_gamma": selected_gamma,
        "global_sensitivity_gamma": global_gamma,
        "summaries": summaries,
        "contrasts": contrasts,
        "selector_sensitivity": {"sensitive": selector_sensitive, "contrasts": selector_signs},
        "common_gamma_sensitivity": {"gamma": common_gamma, "primary": common_sensitivity},
        "override_diagnostics": overrides,
        "unseen_support": support,
    }
    write_json(output / "analysis_core.json", core)
    replay = compare_reference(output, reference) if reference else None
    criterion = criteria(contrasts, selector_sensitive, replay["all_exact"] if replay else None, config)
    if replay:
        write_json(output / "replay_receipt.json", replay)
    write_json(
        output / "runtime.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "torch")
            },
            "device": "cpu",
        },
    )
    write_json(
        output / "REPORT_WAVE55_POLICY_BRIDGE.json",
        {
            "status": "FRESH_MONITOR_ADJUDICATED_NO_GO_NOGO",
            "scope": "same-generator fresh realization; posterior-to-policy interface only",
            "git_commit": commit,
            "config_sha256": digest(config_path),
            "bundle_manifest_sha256": digest(bundle_dir / "bundle_manifest.json"),
            "selection_freeze_sha256": digest(output / "selection_freeze.json"),
            "analysis_core_sha256": digest(output / "analysis_core.json"),
            "result_arrays_sha256": digest(output / "result_arrays.npz"),
            "criteria": criterion,
            "replay": replay,
            "superseded_output": str(archived) if archived else None,
            **core,
            "decision_authority": "user",
        },
    )
    artifact_paths = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    write_json(
        output / "artifact_manifest.json",
        {
            "phase": "wave55-complete-artifact-inventory",
            "files": {
                str(path.relative_to(output)): {"sha256": digest(path), "bytes": path.stat().st_size}
                for path in artifact_paths
            },
        },
    )
    print(json.dumps({"selected_gamma": selected_gamma, "criteria": criterion}, sort_keys=True))


if __name__ == "__main__":
    main()
