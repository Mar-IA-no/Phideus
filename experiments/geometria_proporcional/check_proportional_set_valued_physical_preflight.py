#!/usr/bin/env python3
"""Independent checker for the physical set-valued CPU package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import stat
import subprocess
import sys
import time
from typing import Any, Callable
import zipfile

import numpy as np
import scipy
import sklearn


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
import check_proportional_set_valued_native_preflight as independent  # noqa: E402


CONFIG_DEFAULT = REPO_ROOT / "experiments/geometria_proporcional/configs/proportional_set_valued_physical_preflight_v1.json"
FREEZE_DEFAULT = REPO_ROOT / "experiments/geometria_proporcional/configs/proportional_set_valued_physical_source_freeze_v1.json"
INPUT_DEFAULT = REPO_ROOT / "data/geometria_proporcional/proportional_set_valued_physical_input_v1"
PHASES = (
    "posterior_fit", "policy_fit", "selection_propose", "selection_evaluate",
    "selection_freeze", "evaluation_apply", "evaluation_truth",
)
STATES = (
    "POSTERIOR_FIT_COMPLETE", "POLICY_FIT_COMPLETE", "SELECTION_CANDIDATES_FROZEN",
    "SELECTION_DECISION_FROZEN", "SELECTION_POLICY_FROZEN",
    "EVALUATION_ACTIONS_FROZEN", "COMPLETE",
)
REASONS = {
    "P1_AUTHORITY": "AUTHORITY_INVALID", "P2_PREPARATION": "PREPARATION_INVALID",
    "P3_PHYSICAL_BOUNDARY": "PHYSICAL_BOUNDARY_INVALID", "P4_STATE_MACHINE": "STATE_MACHINE_INVALID",
    "P5_POSTERIOR": "POSTERIOR_INVALID", "P6_POLICY": "POLICY_INVALID",
    "P7_SELECTION_PROPOSE": "SELECTION_PROPOSE_INVALID", "P8_SELECTION_EVALUATE": "SELECTION_EVALUATE_INVALID",
    "P9_SELECTION_FREEZE": "SELECTION_FREEZE_INVALID", "P10_EVALUATION_APPLY": "EVALUATION_APPLY_INVALID",
    "P11_EVALUATION_TRUTH": "EVALUATION_TRUTH_INVALID", "P12_RESTART_REPLAY": "RESTART_OR_REPLAY_INVALID",
    "P13_INVENTORY": "INVENTORY_INVALID", "P14_SCOPE": "SCOPE_INVALID", "P15_COST": "COST_INVALID",
}
TRUTH_FORBIDDEN = ("target", "truth", "oracle", "label", "gain", "regret", "harm", "compatible", "authorized")


class CheckFailure(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--input-package", type=Path, default=INPUT_DEFAULT)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--source-freeze", type=Path, default=FREEZE_DEFAULT)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--only", choices=tuple(REASONS))
    return parser.parse_args()


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant forbidden: {value}")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_array(left: np.ndarray, right: np.ndarray, label: str) -> None:
    if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
        raise CheckFailure(f"array mismatch: {label}")


def assert_value(left: Any, right: Any, label: str) -> None:
    if left != right:
        raise CheckFailure(f"value mismatch: {label}")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


class Checker:
    def __init__(self, artifact: Path, input_package: Path, reference: Path | None, config: Path, source_freeze: Path):
        self.root = artifact.resolve(strict=True)
        self.input = input_package.resolve(strict=True)
        self.reference = None if reference is None else reference.resolve(strict=True)
        self.config_path = config.resolve(strict=True)
        self.freeze_path = source_freeze.resolve(strict=True)
        self.config = read_json(self.config_path)
        self.freeze = read_json(self.freeze_path)
        self.posterior_truth = load_npz(self.input / "prepared/truth/posterior_fit_truth.npz")
        self.policy_truth = load_npz(self.input / "prepared/truth/policy_fit_truth.npz")
        self.decision_public = load_npz(self.input / "prepared/public/decision_select_public.npz")
        self.decision_truth = load_npz(self.input / "prepared/truth/decision_select_truth.npz")
        self.evaluate_public = load_npz(self.input / "prepared/public/evaluate_public.npz")
        self.evaluate_truth = load_npz(self.input / "prepared/truth/evaluate_truth.npz")
        self.utility = np.load(self.input / "prepared/public/utilities.npy", allow_pickle=False)
        self.posterior_states = read_json(self.root / "posterior_fit/posterior_states.json")
        self.posterior_arrays = load_npz(self.root / "posterior_fit/posterior_state_arrays.npz")
        self.policy_states = read_json(self.root / "policy_fit/policy_states.json")
        self.candidate = load_npz(self.root / "selection_propose/candidate_public.npz")
        self.selection_keys = read_json(self.root / "selection_propose/selection_key_metadata.json")
        self.apply_metadata = read_json(self.root / "selection_propose/apply_metadata.json")
        self.decision = read_json(self.root / "selection_evaluate/selection_decision.json")
        self.policy = read_json(self.root / "selection_freeze/selection_policy.json")
        self.eval_metadata = load_npz(self.root / "evaluation_apply/evaluation_metadata.npz")
        self.eval_actions = load_npz(self.root / "evaluation_apply/evaluation_actions.npz")
        self.eval_masses = load_npz(self.root / "evaluation_apply/evaluation_masses.npz")
        self.eval_sensitivity = load_npz(self.root / "evaluation_apply/evaluation_sensitivities.npz")
        self.apply_status = read_json(self.root / "evaluation_apply/evaluation_apply_status.json")
        self.raw = load_npz(self.root / "evaluation_truth/diagnostic_arrays.npz")
        self.boot = load_npz(self.root / "evaluation_truth/bootstrap_indices.npz")
        self.estimands = read_json(self.root / "evaluation_truth/estimand_table.json")
        self.penalty = float(self.config["reader"]["penalty"])
        self.selection_public: dict[str, dict[str, np.ndarray]] = {}
        self.selection_scores: dict[str, dict[str, np.ndarray]] = {}
        self.evaluation_public: dict[str, dict[str, np.ndarray]] = {}

    def mass(self, name: str, logits: np.ndarray, shuffled: bool = False) -> np.ndarray:
        role = "target_shuffled" if shuffled else "real"
        if name == "marginal":
            return independent.marginal_mass(self.posterior_states["marginal"][role], logits)
        return independent.posterior_mass(logits, self.posterior_arrays[f"joint_{role}__final_theta"], "joint_full")

    def p1(self) -> None:
        if self.config["schema_version"] != "proportional-set-valued-physical-preflight-v1" or self.freeze["schema_version"] != "proportional-set-valued-physical-source-freeze-v1": raise CheckFailure("authority schema drifted")
        freeze_rel = self.freeze_path.relative_to(REPO_ROOT).as_posix(); commit = git("log", "-1", "--format=%H", "--", freeze_rel)
        if git("rev-parse", f"{commit}^") != self.freeze["implementation_commit"]: raise CheckFailure("freeze parent drifted")
        if git("diff-tree", "--no-commit-id", "--name-only", "-r", commit).splitlines() != [freeze_rel]: raise CheckFailure("freeze commit scope drifted")
        for relative, digest in self.freeze["files"].items():
            if sha256_file(REPO_ROOT / relative) != digest: raise CheckFailure(f"frozen file drifted: {relative}")
        bindings = read_json(self.root / "bindings.json")
        if bindings["source_freeze_sha256"] != sha256_file(self.freeze_path): raise CheckFailure("run source freeze binding drifted")
        if read_json(self.root / "config.snapshot.json") != self.config: raise CheckFailure("config snapshot drifted")
        source_text = Path(__file__).read_text(encoding="utf-8")
        imported = []
        for node in ast.walk(ast.parse(source_text)):
            if isinstance(node, ast.Import): imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module: imported.append(node.module)
        if any(name.endswith(("proportional_set_valued_native", "run_proportional_set_valued_physical_preflight", "_proportional_set_valued_phase_worker")) for name in imported): raise CheckFailure("physical checker imports tested implementation")

    def p2(self) -> None:
        expected = {
            "posterior": (self.posterior_truth, {"pair_token", "cluster_id", "design_stratum", "cardinality", "ensemble_logits", "per_seed_logits", "target", "split_role"}),
            "policy": (self.policy_truth, {"pair_token", "cluster_id", "design_stratum", "cardinality", "ensemble_logits", "per_seed_logits", "target", "split_role"}),
            "decision_public": (self.decision_public, {"pair_token", "design_stratum", "cardinality", "ensemble_logits", "per_seed_logits"}),
            "decision_truth": (self.decision_truth, {"pair_token", "target"}),
            "evaluate_public": (self.evaluate_public, {"pair_token", "design_stratum", "cardinality", "ensemble_logits", "per_seed_logits"}),
            "evaluate_truth": (self.evaluate_truth, {"pair_token", "target"}),
        }
        for name, (bundle, keys) in expected.items():
            if set(bundle) != keys: raise CheckFailure(f"bundle keys drifted: {name}")
        for public, truth, name in ((self.decision_public, self.decision_truth, "decision"), (self.evaluate_public, self.evaluate_truth, "evaluate")):
            assert_array(public["pair_token"], truth["pair_token"], f"{name} identity")
            if not np.array_equal(public["cardinality"], truth["target"].sum(axis=1).astype("<i8")): raise CheckFailure(f"{name} cardinality drifted")
            if any(fragment in key.lower() for key in public for fragment in TRUTH_FORBIDDEN): raise CheckFailure(f"{name} public semantic leak")
        token_sets = [set(bundle["pair_token"].astype(str)) for bundle in (self.posterior_truth, self.policy_truth, self.decision_public, self.evaluate_public)]
        if any(left & right for left, right in itertools.combinations(token_sets, 2)): raise CheckFailure("role overlap")
        rows = self.config["opened_fixture_roles"]
        if [len(item) for item in token_sets] != [rows["posterior_fit"], rows["policy_fit"], rows["decision_select"], rows["evaluate"]]: raise CheckFailure("role counts drifted")
        for public in (self.posterior_truth, self.policy_truth, self.decision_public, self.evaluate_public):
            if not np.array_equal(public["ensemble_logits"], np.mean(public["per_seed_logits"], axis=0, dtype=np.float64)): raise CheckFailure("ensemble relation drifted")
            if set(public["design_stratum"].astype(str)) != {"FAR_RIVAL", "NEAR_RIVAL"}: raise CheckFailure("stratum vocabulary drifted")
        prep = read_json(self.input / "preparation_freeze.json"); escrow = read_json(self.input / "opened_fixture_escrow.json")
        if prep["package_id"] != escrow["package_id"] or prep["checkpoint_axis"] != [17, 29, 43] or prep["prospective_evidence"] is not False or escrow["generation_escrow"] is not False: raise CheckFailure("preparation authority drifted")

    def p3(self) -> None:
        if stat.S_IMODE(self.root.stat().st_mode) != 0o700 or self.root.stat().st_uid != 0: raise CheckFailure("run root permissions drifted")
        for phase in PHASES:
            directory = self.root / phase
            if directory.is_symlink() or stat.S_IMODE(directory.stat().st_mode) != 0o500 or directory.stat().st_uid != 0: raise CheckFailure(f"phase directory boundary drifted: {phase}")
            receipt = read_json(directory / "worker_receipt.json"); runtime = receipt["runtime"]
            identity = runtime["identity"]
            if identity["Uid"].split()[0] != "65534" or identity["Gid"].split()[0] != "65534" or identity["Groups"] != "" or identity["NoNewPrivs"] != "1": raise CheckFailure(f"worker identity drifted: {phase}")
            if any(identity[key] != "0000000000000000" for key in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")): raise CheckFailure(f"capability set drifted: {phase}")
            if runtime["torch_imported"] or runtime["gpu_used_or_queried"] or runtime["environment"]["CUDA_VISIBLE_DEVICES"] != "": raise CheckFailure(f"CPU boundary drifted: {phase}")
            if set(runtime["modules"]) != {"geometria_proporcional", "geometria_proporcional.wave49_schema", "geometria_proporcional.proportional_set_valued_native", "geometria_proporcional.wave53_uncertainty", "geometria_proporcional.wave54_joint_set"}: raise CheckFailure(f"runtime module set drifted: {phase}")
            if any(row["outcome"] not in {"PermissionError", "FileNotFoundError"} for row in receipt["probes"]): raise CheckFailure(f"probe opened: {phase}")
            if any(int(pool.get("num_threads", 0)) > 1 for pool in runtime["threadpools"]): raise CheckFailure(f"threadpool drifted: {phase}")

    def p4(self) -> None:
        previous = "PREPARED"; levels = ("POSTERIOR_FIT", "POLICY_FIT", "POLICY_FIT", "DECISION_SELECT", "DECISION_SELECT", "DECISION_SELECT", "EVALUATE")
        package_id = read_json(self.input / "preparation_freeze.json")["package_id"]
        for phase, state, level in zip(PHASES, STATES, levels, strict=True):
            journal = read_json(self.root / "journals" / f"{phase}.json")
            if journal["phase"] != phase or journal["previous_state"] != previous or journal["new_state"] != state or journal["maximum_truth_materialized"] != level or journal["package_id"] != package_id: raise CheckFailure(f"journal transition drifted: {phase}")
            actual = {path.name: sha256_file(path) for path in (self.root / phase).iterdir() if path.is_file()}
            if journal["output_hashes"] != dict(sorted(actual.items())): raise CheckFailure(f"journal output hash drifted: {phase}")
            previous = state

    def p5(self) -> None:
        parity = read_json(self.root / "r564_parity_receipt.json")
        if not parity["all_exact"] or parity["passed"] != parity["total"] or parity["total"] < 20: raise CheckFailure("R564 posterior parity incomplete")
        shuffled = load_npz(self.root / "posterior_fit/target_shuffle_arrays.npz")
        folds = independent.fold_ids(self.posterior_truth["pair_token"], self.posterior_truth["design_stratum"], self.posterior_truth["cardinality"])
        donor, rows, permutable = independent.target_derangement(self.posterior_truth["pair_token"], folds, self.posterior_truth["design_stratum"], self.posterior_truth["cardinality"], 53602)
        assert_array(donor, shuffled["donor_index"], "posterior donor"); assert_array(permutable, shuffled["permutable"], "posterior permutable")
        assert_array(self.posterior_truth["target"][donor], shuffled["target_shuffled"], "posterior shuffled target")
        if rows != read_json(self.root / "posterior_fit/target_shuffle_map.json"): raise CheckFailure("posterior semantic shuffle map drifted")

    def p6(self) -> None:
        schema = read_json(self.root / "policy_fit/feature_schema.json")
        if tuple(schema["feature_names"]) != independent.FEATURE_NAMES or schema["count"] != 17: raise CheckFailure("policy feature schema drifted")
        if len(self.policy_states["marginal"]["controls"]) != 5 or len(self.policy_states["joint"]["controls"]) != 5: raise CheckFailure("policy controls drifted")
        if [row["seed"] for row in self.policy_states["marginal"]["controls"]] != self.config["matched_controls"]["seeds"]: raise CheckFailure("policy seed order drifted")
        arrays = load_npz(self.root / "policy_fit/policy_state_arrays.npz")
        if not arrays or any(value.dtype.hasobject for value in arrays.values()): raise CheckFailure("portable policy arrays invalid")

    def p7(self) -> None:
        for name in ("marginal", "joint"):
            mass = self.mass(name, self.decision_public["ensemble_logits"])
            pdata = independent.public_design(self.decision_public["ensemble_logits"], self.decision_public["per_seed_logits"], mass, self.utility, self.penalty)
            scores = independent.score_triplet(self.policy_states[name]["true"]["states"], pdata)
            self.selection_public[name] = pdata; self.selection_scores[name] = scores
            rows = self.apply_metadata["posteriors"][name]; key_rows = self.selection_keys["posteriors"][name]
            if len(rows) != 344 or len(key_rows) != 344: raise CheckFailure("candidate count drifted")
            rebuilt_actions = []; rebuilt_override = []
            for index, row in enumerate(rows):
                if row["candidate_index"] != index or set(key_rows[index]) != {"candidate_index", "kind", "proposer_quantile", "harm_quantile", "incompatibility_quantile"}: raise CheckFailure("candidate metadata schema drifted")
                if row["kind"] == "hard_only": action = pdata["hard_actions"]; override = np.zeros_like(action, dtype=bool)
                else:
                    expected_threshold = independent.thresholds(scores, pdata["disagreement"], (row["proposer_quantile"], row["harm_quantile"], row["incompatibility_quantile"]))
                    if {key: row[key] for key in expected_threshold} != expected_threshold: raise CheckFailure("candidate threshold drifted")
                    action, override = independent.apply_thresholds(scores, pdata, row)
                rebuilt_actions.append(action); rebuilt_override.append(override)
            assert_array(np.asarray(rebuilt_actions), self.candidate[f"{name}__actions"], f"{name} candidate actions")
            assert_array(np.asarray(rebuilt_override), self.candidate[f"{name}__override"], f"{name} candidate override")

    def p8(self) -> None:
        private = load_npz(self.root / "selection_evaluate/candidate_metrics_private.npz")
        target = self.decision_truth["target"]
        for name in ("marginal", "joint"):
            actions = self.candidate[f"{name}__actions"]; overrides = self.candidate[f"{name}__override"]; hard = self.candidate[f"{name}__hard_actions"]
            hard_regret = independent.regret(hard, target, self.utility, self.penalty)
            metrics = {"mean_regret": [], "incompatibility_rate": [], "harm_rate": [], "authorized_rows": []}
            for index, candidate in enumerate(actions):
                row = independent.action_metrics(candidate, target, self.utility, self.penalty)
                metrics["mean_regret"].append(float(row["regret"].mean())); metrics["incompatibility_rate"].append(float(row["incompatibility_by_policy"].mean())); metrics["harm_rate"].append(float(np.mean(row["regret_by_policy"] > hard_regret + 1e-12))); metrics["authorized_rows"].append(int(overrides[index].sum()))
            for key, values in metrics.items(): assert_array(np.asarray(values, dtype=np.int64 if key == "authorized_rows" else np.float64), private[f"{name}__{key}"], f"{name} candidate {key}")
            key_rows = self.selection_keys["posteriors"][name]
            selected = min(range(344), key=lambda index: (metrics["mean_regret"][index], metrics["incompatibility_rate"][index], metrics["harm_rate"][index], -metrics["authorized_rows"][index], key_rows[index]["proposer_quantile"], key_rows[index]["harm_quantile"], key_rows[index]["incompatibility_quantile"]))
            decision = self.decision["posteriors"][name]
            if decision["selected_index"] != selected or set(decision) != {"selected_index", "mean_regret", "incompatibility_rate", "harm_rate", "authorized_rows", "candidate_freeze_sha256", "selected_actions_sha256", "selected_override_sha256"}: raise CheckFailure("minimal selection decision drifted")

    def p9(self) -> None:
        selected = load_npz(self.root / "selection_freeze/selected_actions.npz"); matches = load_npz(self.root / "selection_freeze/selection_matches.npz")
        for name in ("marginal", "joint"):
            index = self.decision["posteriors"][name]["selected_index"]
            assert_array(self.candidate[f"{name}__actions"][index], selected[f"{name}__actions"], f"{name} selected action")
            assert_array(self.candidate[f"{name}__override"][index], selected[f"{name}__override"], f"{name} selected override")
            common = selected[f"{name}__override"].any(axis=1)
            for control, states in zip(self.policy["posteriors"][name]["controls"], self.policy_states[name]["controls"], strict=True):
                scores = independent.score_triplet(states["states"], self.selection_public[name])
                matched = independent.matched_actions(selected[f"{name}__override"], scores, self.selection_public[name], control["thresholds"])
                for key, value in matched.items(): assert_array(np.asarray(value), matches[f"{name}__control_{control['seed']}__{key}"], f"{name} control {control['seed']} {key}")
                common &= matched["match_valid"]
            assert_array(common, matches[f"{name}__u_common"], f"{name} common support")

    def p10(self) -> None:
        assert_array(self.evaluate_public["pair_token"], self.eval_metadata["pair_token"], "evaluation metadata token")
        assert_array(self.evaluate_public["design_stratum"], self.eval_metadata["design_stratum"], "evaluation metadata stratum")
        assert_array(self.evaluate_public["cardinality"], self.eval_metadata["cardinality"], "evaluation metadata cardinality")
        if set(self.eval_metadata) != {"pair_token", "design_stratum", "cardinality"}: raise CheckFailure("evaluation metadata widened")
        for name in ("marginal", "joint"):
            real = self.mass(name, self.evaluate_public["ensemble_logits"]); shuffled = self.mass(name, self.evaluate_public["ensemble_logits"], True)
            assert_array(real, self.eval_masses[f"{name}__real"], f"{name} evaluation mass"); assert_array(shuffled, self.eval_masses[f"{name}__target_shuffled"], f"{name} evaluation shuffled mass")
            pdata = independent.public_design(self.evaluate_public["ensemble_logits"], self.evaluate_public["per_seed_logits"], real, self.utility, self.penalty)
            self.evaluation_public[name] = pdata
            scores = independent.score_triplet(self.policy_states[name]["true"]["states"], pdata); selected = self.policy["posteriors"][name]["selected"]
            contextual = pdata["hard_actions"] if selected["kind"] == "hard_only" else independent.apply_thresholds(scores, pdata, selected)[0]
            assert_array(pdata["hard_actions"], self.eval_actions[f"{name}__hard_actions"], f"{name} evaluation hard"); assert_array(contextual, self.eval_actions[f"{name}__contextual_actions"], f"{name} evaluation contextual")
            for cp_index, epoch in enumerate(self.config["checkpoint_epochs"]):
                cp_mass = self.mass(name, self.evaluate_public["per_seed_logits"][cp_index]); assert_array(cp_mass, self.eval_sensitivity[f"checkpoint_{epoch}__{name}__mass"], f"{name} checkpoint mass {epoch}")
                cpdata = independent.public_design(self.evaluate_public["per_seed_logits"][cp_index], self.evaluate_public["per_seed_logits"], cp_mass, self.utility, self.penalty); cp_scores = independent.score_triplet(self.policy_states[name]["true"]["states"], cpdata)
                cp_contextual = cpdata["hard_actions"] if selected["kind"] == "hard_only" else independent.apply_thresholds(cp_scores, cpdata, selected)[0]
                assert_array(cpdata["hard_actions"], self.eval_sensitivity[f"checkpoint_{epoch}__{name}__hard_actions"], f"{name} checkpoint hard {epoch}"); assert_array(cp_contextual, self.eval_sensitivity[f"checkpoint_{epoch}__{name}__contextual_actions"], f"{name} checkpoint contextual {epoch}")

    def p11(self) -> None:
        target = self.evaluate_truth["target"]
        assert_array(self.evaluate_truth["pair_token"], self.eval_metadata["pair_token"], "evaluation truth identity")
        global_boot = np.random.Generator(np.random.PCG64(self.config["bootstrap"]["global_seed"])).integers(0, len(target), size=(5000, len(target)), dtype=np.int64)
        assert_array(global_boot, self.boot["global_pair_token_index"], "global bootstrap")
        expected_raw: dict[str, np.ndarray] = {"pair_token": self.eval_metadata["pair_token"].astype(str), "target": target}
        set_rows = {}; action_rows = {}
        for name in ("marginal", "joint"):
            set_rows[name] = {}
            for role in ("real", "target_shuffled"):
                values = independent.set_metrics(self.eval_masses[f"{name}__{role}"], target); set_rows[name][role] = values
                for key, value in values.items(): expected_raw[f"{name}__{role}__{key}"] = value
            for reader in ("hard", "contextual"):
                values = independent.action_metrics(self.eval_actions[f"{name}__{reader}_actions"], target, self.utility, self.penalty); action_rows[f"{name}_{reader}"] = values
                for key, value in values.items(): expected_raw[f"{name}__{reader}__{key}"] = value
            for seed in self.config["matched_controls"]["seeds"]:
                values = independent.action_metrics(self.eval_actions[f"{name}__control_{seed}__actions"], target, self.utility, self.penalty)
                for key, value in values.items(): expected_raw[f"{name}__control_{seed}__{key}"] = value
        for epoch in self.config["checkpoint_epochs"]:
            for name in ("marginal", "joint"):
                for reader in ("hard", "contextual"):
                    values = independent.action_metrics(self.eval_sensitivity[f"checkpoint_{epoch}__{name}__{reader}_actions"], target, self.utility, self.penalty)
                    for key, value in values.items(): expected_raw[f"checkpoint_{epoch}__{name}__{reader}__{key}"] = value
        if set(expected_raw) != set(self.raw): raise CheckFailure("evaluation raw inventory drifted")
        for key, value in expected_raw.items():
            if not np.array_equal(np.asarray(value), self.raw[key]): raise CheckFailure(f"evaluation raw drifted: {key}")
        if {row["id"] for row in self.estimands["rows"]} != set(self.config["required_decision_rows"]): raise CheckFailure("estimand coverage drifted")
        if self.estimands["scientific_decision"] is not None or self.estimands["architecture_promoted"] or self.estimands["prospective_evidence"]: raise CheckFailure("estimand authority drifted")

    def p12(self) -> None:
        receipt = read_json(self.root / "replay_receipt.json")
        if self.reference is None:
            if receipt["mode"] != "primary" or receipt["byte_exact"] is not None: raise CheckFailure("primary replay receipt drifted")
            return
        if receipt["mode"] != "replay" or receipt["byte_exact"] is not True or not receipt["semantic_exclusions_valid"]: raise CheckFailure("replay receipt drifted")
        excluded = {"worker_receipt.json", "artifact_manifest.json", "runtime.json", "replay_receipt.json", "recovery_origin.json", "REPORT.md"}
        def files(root: Path) -> dict[str, Path]: return {path.relative_to(root).as_posix(): path for path in root.rglob("*") if path.is_file() and path.name not in excluded and not path.relative_to(root).as_posix().startswith("journals/") and path.name not in {"config.snapshot.json", "bindings.json"}}
        current, previous = files(self.root), files(self.reference)
        if set(current) != set(previous) or any(sha256_file(current[key]) != sha256_file(previous[key]) for key in current): raise CheckFailure("replay scientific mismatch")

    def p13(self) -> None:
        manifest = read_json(self.root / "artifact_manifest.json")
        actual = {path.relative_to(self.root).as_posix(): path for path in self.root.rglob("*") if path.name != "artifact_manifest.json"}
        recorded = {row["path"]: row for row in manifest["files_and_directories"]}
        if set(actual) != set(recorded): raise CheckFailure("artifact inventory drifted")
        for relative, path in actual.items():
            row = recorded[relative]; info = path.lstat()
            if row["mode"] != stat.S_IMODE(info.st_mode) or row["uid"] != info.st_uid or row["gid"] != info.st_gid or row["type"] != ("directory" if path.is_dir() else "file"): raise CheckFailure(f"artifact metadata drifted: {relative}")
            if path.is_file() and (row["sha256"] != sha256_file(path) or row["bytes"] != info.st_size): raise CheckFailure(f"artifact bytes drifted: {relative}")
        for path in self.root.rglob("*.json"):
            if path.read_bytes() != json_bytes(read_json(path)): raise CheckFailure(f"noncanonical JSON: {path.relative_to(self.root)}")
        for path in self.root.rglob("*.npz"):
            with zipfile.ZipFile(path) as archive:
                infos = archive.infolist()
                if [row.filename for row in infos] != sorted(row.filename for row in infos) or any(row.date_time != (1980, 1, 1, 0, 0, 0) or row.compress_type != zipfile.ZIP_DEFLATED or row.external_attr != (0o600 << 16) for row in infos): raise CheckFailure(f"noncanonical NPZ: {path.relative_to(self.root)}")

    def p14(self) -> None:
        runtime = read_json(self.root / "runtime.json")
        expected_false = ("fresh_draw_created_or_opened", "monitor_or_lockbox_opened", "gpu_used_or_queried", "torch_imported", "architecture_promoted", "prospective_evidence")
        if runtime["status"] != "PHYSICAL_PROSPECTIVE_PACKAGE_PREFLIGHT_VALID" or any(runtime[key] for key in expected_false) or runtime["scientific_decision"] is not None or runtime["decision_authority"] != "user": raise CheckFailure("scope claim drifted")
        if "torch" in sys.modules or os.environ.get("CUDA_VISIBLE_DEVICES") != "": raise CheckFailure("checker CPU scope drifted")
        report = (self.root / "REPORT.md").read_text(encoding="utf-8")
        if "prospective_evidence=false" not in report or "no promueve una arquitectura" not in report: raise CheckFailure("report claim boundary drifted")

    def p15(self) -> None:
        runtime = read_json(self.root / "runtime.json")
        limit = int(self.config["budgets"]["per_process_rss_bytes"])
        if runtime["coordinator_peak_rss_bytes"] > limit or any(row["peak_rss_bytes"] > limit for row in runtime["phases_executed"]): raise CheckFailure("run RSS budget exceeded")
        if self.reference is not None:
            total = runtime["wall_seconds"] + read_json(self.reference / "runtime.json")["wall_seconds"]
            if total > self.config["budgets"]["primary_plus_replay_seconds"]: raise CheckFailure("primary plus replay budget exceeded")
        if int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024 > limit: raise CheckFailure("checker RSS budget exceeded")
        if sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file()) > 512 * 1024 * 1024: raise CheckFailure("output disk budget exceeded")


PREDICATES: tuple[tuple[str, str], ...] = tuple((name, f"p{index}") for index, name in enumerate(REASONS, start=1))


def check_evidence(path: Path) -> dict[str, Any]:
    root = path.resolve(strict=True); manifest = read_json(root / "evidence_manifest.json")
    required = {"unit_test_receipt.json", "primary_check_receipt.json", "replay_check_receipt.json", "mutation_receipt.json", "recovery_receipt.json"}
    actual = {item.name for item in root.iterdir() if item.is_file() and item.name != "evidence_manifest.json"}
    if not required.issubset(actual): raise CheckFailure("evidence receipt missing")
    recorded = {row["path"]: row for row in manifest["files"]}
    if set(recorded) != actual or any(sha256_file(root / name) != row["sha256"] or (root / name).stat().st_size != row["bytes"] for name, row in recorded.items()): raise CheckFailure("evidence manifest drifted")
    freeze = FREEZE_DEFAULT.resolve(strict=True); freeze_sha = sha256_file(freeze)
    config = read_json(CONFIG_DEFAULT)
    unit = read_json(root / "unit_test_receipt.json")
    primary = read_json(root / "primary_check_receipt.json")
    replay = read_json(root / "replay_check_receipt.json")
    mutations = read_json(root / "mutation_receipt.json")
    recovery = read_json(root / "recovery_receipt.json")
    receipts = (unit, primary, replay, mutations, recovery)
    if any(row.get("status") != "PASS" or row.get("exit") != 0 for row in receipts):
        raise CheckFailure("evidence receipt status/exit drifted")
    if unit.get("passed") != 10 or unit.get("total") != 10:
        raise CheckFailure("unit receipt count drifted")
    for name, row in (("primary", primary), ("replay", replay)):
        result = row.get("stdout_last_json", {})
        if result.get("status") != "PASS" or result.get("passed") != 15 or result.get("total") != 15 or len(result.get("checks", [])) != 15:
            raise CheckFailure(f"{name} checker receipt coverage drifted")
    if mutations.get("source_freeze_sha256") != freeze_sha or mutations.get("passed") != mutations.get("total") or mutations.get("total") != 63:
        raise CheckFailure("mutation receipt coverage/freeze drifted")
    if any(not row.get("passed") or row.get("reason_code_expected") != row.get("reason_code_observed") for row in mutations.get("cases", [])):
        raise CheckFailure("mutation case result drifted")
    if recovery.get("source_freeze_sha256") != freeze_sha or recovery.get("passed") != recovery.get("total") or recovery.get("total") != 7:
        raise CheckFailure("recovery receipt coverage/freeze drifted")
    if any(not row.get("scientific_byte_exact") or not row.get("journal_absent") for row in recovery.get("cases", [])):
        raise CheckFailure("recovery case result drifted")
    freeze_relative = FREEZE_DEFAULT.relative_to(REPO_ROOT).as_posix()
    for row in (unit, primary, replay):
        bound = row.get("inputs", {}).get(freeze_relative, {})
        if bound.get("sha256") != freeze_sha:
            raise CheckFailure("suite receipt source freeze drifted")
    limits = config["budgets"]
    if unit["wall_seconds"] > limits["unit_and_permissions_seconds"]:
        raise CheckFailure("unit campaign wall budget exceeded")
    if primary["wall_seconds"] + replay["wall_seconds"] > limits["checker_plus_mutations_seconds"]:
        raise CheckFailure("checker campaign wall budget exceeded")
    if mutations["wall_seconds"] > limits["checker_plus_mutations_seconds"] or recovery["wall_seconds"] > limits["recovery_seconds"]:
        raise CheckFailure("mutation/recovery wall budget exceeded")
    rss_fields = [unit["peak_rss_bytes"], primary["peak_rss_bytes"], replay["peak_rss_bytes"], mutations["peak_rss_bytes"], mutations["children_peak_rss_bytes"], recovery["peak_rss_bytes"], recovery["children_peak_rss_bytes"]]
    if any(value > limits["per_process_rss_bytes"] for value in rss_fields):
        raise CheckFailure("evidence campaign RSS budget exceeded")
    if mutations["peak_temporary_bytes"] > 1024**3 or recovery["peak_temporary_bytes"] > 1024**3:
        raise CheckFailure("evidence scratch budget exceeded")
    if sum((root / name).stat().st_size for name in actual | {"evidence_manifest.json"}) > 128 * 1024**2:
        raise CheckFailure("preserved evidence budget exceeded")
    primary_manifest = REPO_ROOT / "data/geometria_proporcional/proportional_set_valued_physical_preflight_v1/artifact_manifest.json"
    replay_manifest = REPO_ROOT / "data/geometria_proporcional/proportional_set_valued_physical_preflight_replay_v1/artifact_manifest.json"
    if manifest.get("source_freeze_sha256") != freeze_sha or manifest.get("primary_artifact_manifest_sha256") != sha256_file(primary_manifest) or manifest.get("replay_artifact_manifest_sha256") != sha256_file(replay_manifest):
        raise CheckFailure("evidence root binding drifted")
    return {"status": "PASS", "files": len(actual), "unit": "10/10", "primary": "15/15", "replay": "15/15", "mutations": "63/63", "recovery": "7/7", "manifest_sha256": sha256_file(root / "evidence_manifest.json")}


def main() -> int:
    args = parse_args(); started = time.monotonic()
    if args.evidence:
        try: payload = check_evidence(args.evidence); print(json.dumps(payload, sort_keys=True)); return 0
        except Exception as error: print(json.dumps({"status": "FAIL", "reason_code": "INVENTORY_INVALID", "error": str(error)}, sort_keys=True)); return 1
    if args.artifact is None: raise SystemExit("--artifact is required unless --evidence is used")
    results = []
    try: checker = Checker(args.artifact, args.input_package, args.reference, args.config, args.source_freeze)
    except Exception as error:
        print(json.dumps({"status": "FAIL", "reason_code": "AUTHORITY_INVALID", "error": str(error)}, sort_keys=True)); return 1
    predicates = PREDICATES if args.only is None else tuple(row for row in PREDICATES if row[0] == args.only)
    for name, method in predicates:
        try:
            getattr(checker, method)(); results.append({"id": name, "status": "PASS", "reason_code": None})
        except Exception as error:
            results.append({"id": name, "status": "FAIL", "reason_code": REASONS[name], "error": f"{type(error).__name__}: {error}"})
            print(json.dumps({"status": "FAIL", "checks": results, "failed": name, "reason_code": REASONS[name], "wall_seconds": time.monotonic() - started}, sort_keys=True)); return 1
    print(json.dumps({"status": "PASS", "passed": len(results), "total": len(predicates), "checks": results, "wall_seconds": time.monotonic() - started, "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024}, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
