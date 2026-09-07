#!/usr/bin/env python3
"""Run the CPU-only opened-data preflight for the set-valued native branch."""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any
import zipfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geometria_proporcional import proportional_set_valued_native as native  # noqa: E402


DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/geometria_proporcional/configs/proportional_set_valued_native_preflight_v1.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data/geometria_proporcional/proportional_set_valued_native_preflight_v1"
)
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
PUBLIC_SELECTION_KEYS = (
    "pair_token",
    "ensemble_logits",
    "per_seed_logits",
    "design_stratum",
    "cardinality",
)
TRAIN_KEYS = PUBLIC_SELECTION_KEYS + ("cluster_id", "split_role", "target")
REPLAY_EXCLUDED = {
    "runtime.json",
    "replay_receipt.json",
    "artifact_manifest.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            _jsonable(payload),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(arrays):
            array = np.ascontiguousarray(arrays[name])
            if array.dtype.hasobject:
                raise TypeError(f"object array is forbidden: {name}")
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            info = zipfile.ZipInfo(
                f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(
                info,
                buffer.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def validate_environment(config: dict[str, Any]) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly empty")
    threads = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    if threads != {name: "1" for name in THREAD_VARIABLES}:
        raise RuntimeError(f"thread environment must be exactly one: {threads}")
    if "torch" in sys.modules:
        raise RuntimeError("torch must not be imported by the preflight runtime")
    versions = native.validate_runtime_versions()
    if versions != config["versions"]:
        raise RuntimeError("runtime versions differ from config")
    if config.get("device") != "cpu" or config.get("threads") != 1:
        raise RuntimeError("preflight device/thread config drifted")
    return {"cuda_visible_devices": "", "threads": threads, "versions": versions}


def source_preflight(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    if config.get("schema_version") != "proportional-set-valued-native-preflight-v1":
        raise RuntimeError("preflight schema drifted")
    if config.get("status") != "OPENED_DATA_IMPLEMENTATION_DIAGNOSTIC":
        raise RuntimeError("preflight claim boundary drifted")
    forbidden = tuple(str(value).lower() for value in config["forbidden_input_path_fragments"])
    observed: list[dict[str, str]] = []
    for role, relative, expected in config["source_bindings"]:
        path = (REPO_ROOT / relative).resolve(strict=True)
        if path.relative_to(REPO_ROOT) != Path(relative):
            raise RuntimeError(f"non-canonical source path: {relative}")
        if role in {
            "POSTERIOR_FIT_SOURCE",
            "POLICY_FIT_SOURCE",
            "SELECTION_PUBLIC_REFERENCE",
            "SELECTION_TRUTH_SOURCE",
            "OPENED_DATA_RECEIPT",
        } and any(fragment in relative.lower() for fragment in forbidden):
            raise RuntimeError(f"forbidden input path: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"source hash mismatch for {role}: {actual} != {expected}")
        observed.append({"role": role, "path": relative, "sha256": actual})
    config_resolved = config_path.resolve(strict=True)
    if config_resolved.relative_to(REPO_ROOT) != Path(
        "experiments/geometria_proporcional/configs/proportional_set_valued_native_preflight_v1.json"
    ):
        raise RuntimeError("non-canonical preflight config path")
    for relative in config["execution_sources"]:
        path = (REPO_ROOT / relative).resolve(strict=True)
        if path.relative_to(REPO_ROOT) != Path(relative):
            raise RuntimeError(f"non-canonical execution source: {relative}")
        subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", relative],
            cwd=REPO_ROOT,
            check=True,
        )
        if not git_output("ls-files", "--error-unmatch", "--", relative):
            raise RuntimeError(f"execution source is not tracked: {relative}")
        observed.append(
            {"role": "EXECUTION_SOURCE", "path": relative, "sha256": sha256_file(path)}
        )
    if "torch" in sys.modules:
        raise RuntimeError("source preflight imported torch")
    return {
        "git_commit": git_output("rev-parse", "HEAD"),
        "sources": observed,
    }


def prepare_output(path: Path, force: bool) -> Path | None:
    resolved = path.resolve()
    if resolved == REPO_ROOT or REPO_ROOT not in resolved.parents:
        raise ValueError("output must be a strict child of the repository")
    archived: Path | None = None
    if resolved.exists():
        if not force:
            raise FileExistsError(f"output exists: {resolved}")
        index = 1
        while True:
            candidate = resolved.with_name(f"{resolved.name}.archived.{index:03d}")
            if not candidate.exists():
                archived = candidate
                break
            index += 1
        resolved.rename(archived)
    resolved.mkdir(parents=True)
    return archived


def _binding_paths(config: dict[str, Any]) -> dict[str, Path]:
    return {role: REPO_ROOT / relative for role, relative, _ in config["source_bindings"]}


def _select_keys(data: dict[str, np.ndarray], keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    missing = set(keys) - set(data)
    if missing:
        raise RuntimeError(f"input bundle missing keys: {sorted(missing)}")
    return {key: np.asarray(data[key]).copy() for key in keys}


def prepare_opened_fixtures(
    paths: dict[str, Path], output: Path, expected_roles: dict[str, int]
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    wave54 = load_npz(paths["POSTERIOR_FIT_SOURCE"])
    roles = np.asarray(wave54["split_role"]).astype(str)
    posterior_mask = roles == "calibration_fit"
    posterior = {}
    for key in TRAIN_KEYS:
        value = np.asarray(wave54[key])
        posterior[key] = (
            value[:, posterior_mask].copy()
            if key == "per_seed_logits"
            else value[posterior_mask].copy()
        )
    posterior["split_role"] = np.full(len(posterior["pair_token"]), "posterior_fit")

    policy_source = load_npz(paths["POLICY_FIT_SOURCE"])
    policy = _select_keys(policy_source, TRAIN_KEYS)
    if set(np.asarray(policy["split_role"]).astype(str)) != {"gate_fit"}:
        raise RuntimeError("policy-fit source role drifted")
    policy["split_role"] = np.full(len(policy["pair_token"]), "policy_fit")

    selection_source = load_npz(paths["SELECTION_TRUTH_SOURCE"])
    selection_full = _select_keys(selection_source, TRAIN_KEYS)
    if set(np.asarray(selection_full["split_role"]).astype(str)) != {"gate_select"}:
        raise RuntimeError("decision-select source role drifted")
    public = _select_keys(selection_full, PUBLIC_SELECTION_KEYS)
    truth = {
        "pair_token": np.asarray(selection_full["pair_token"]).copy(),
        "target": np.asarray(selection_full["target"], dtype=bool).copy(),
    }
    public_reference = load_npz(paths["SELECTION_PUBLIC_REFERENCE"])
    if not np.array_equal(
        np.asarray(public_reference["pair_token"]).astype(str),
        np.asarray(public["pair_token"]).astype(str),
    ):
        raise RuntimeError("selection public reference identity drifted")

    fixtures = {
        "posterior_fit": posterior,
        "policy_fit": policy,
        "decision_select_public": public,
        "decision_select_truth": truth,
    }
    expected = {
        "posterior_fit": int(expected_roles["posterior_fit"]),
        "policy_fit": int(expected_roles["policy_fit"]),
        "decision_select_public": int(expected_roles["decision_select"]),
        "decision_select_truth": int(expected_roles["decision_select"]),
    }
    token_sets: dict[str, set[str]] = {}
    for name, fixture in fixtures.items():
        tokens = np.asarray(fixture["pair_token"]).astype(str)
        if len(tokens) != expected[name] or len(np.unique(tokens)) != len(tokens):
            raise RuntimeError(f"fixture role count/identity drifted: {name}")
        token_sets[name] = set(tokens.tolist())
    physical_sets = {
        "posterior_fit": token_sets["posterior_fit"],
        "policy_fit": token_sets["policy_fit"],
        "decision_select": token_sets["decision_select_public"],
    }
    for left, right in itertools.combinations(physical_sets, 2):
        if physical_sets[left] & physical_sets[right]:
            raise RuntimeError(f"fixture phases overlap: {left}/{right}")
    if token_sets["decision_select_public"] != token_sets["decision_select_truth"]:
        raise RuntimeError("selection public/truth identity mismatch")
    if any(fragment in key.lower() for key in public for fragment in ("target", "truth", "oracle", "gain", "regret", "harm")):
        raise RuntimeError("selection public bundle contains forbidden semantic key")

    prepared = output / "prepared"
    write_npz(prepared / "posterior_fit_truth.npz", posterior)
    write_npz(prepared / "policy_fit_truth.npz", policy)
    # The target-free view is deliberately materialized before its truth companion.
    write_npz(prepared / "decision_select_public.npz", public)
    write_npz(prepared / "decision_select_truth.npz", truth)
    manifest = {
        "schema_version": "proportional-opened-fixture-manifest-v1",
        "status": "OPENED_DATA_IMPLEMENTATION_DIAGNOSTIC",
        "logical_phase_separation_only": True,
        "physical_isolation_claimed": False,
        "roles": {
            name: {
                "rows": int(len(fixture["pair_token"])),
                "keys": sorted(fixture),
                "pair_token_sha256": native.array_digest(
                    np.asarray(fixture["pair_token"]).astype("U")
                ),
            }
            for name, fixture in fixtures.items()
        },
        "pairwise_overlap": {
            f"{left}__{right}": int(len(physical_sets[left] & physical_sets[right]))
            for left, right in itertools.combinations(physical_sets, 2)
        },
    }
    write_json(prepared / "fixture_manifest.json", manifest)
    return fixtures, manifest


def _prefix_arrays(prefix: str, arrays: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> None:
    for key, value in arrays.items():
        target[f"{prefix}__{key}"] = np.asarray(value)


def fit_posteriors(
    posterior: dict[str, np.ndarray], config: dict[str, Any], output: Path
) -> dict[str, Any]:
    tokens = np.asarray(posterior["pair_token"]).astype(str)
    logits = np.asarray(posterior["ensemble_logits"], dtype=np.float64)
    target = np.asarray(posterior["target"], dtype=bool)
    folds = native.posterior_fold_ids(
        tokens, posterior["design_stratum"], posterior["cardinality"]
    )
    shuffle = native.target_derangement_v1(
        tokens,
        folds,
        posterior["design_stratum"],
        posterior["cardinality"],
        seed=int(config["posterior"]["target_shuffle"]["seed"]),
    )
    if shuffle["permutable_fraction"] < float(
        config["posterior"]["target_shuffle"]["minimum_permutable_fraction"]
    ):
        raise RuntimeError("TARGET_SHUFFLE_NOT_MATERIALIZABLE")
    shuffled_target = target[np.asarray(shuffle["donor_index"], dtype=np.int64)]
    marginal_real = native.fit_marginal_state(logits, target)
    marginal_shuffled = native.fit_marginal_state(logits, shuffled_target)
    joint_real = native.fit_joint_cv(logits, target, folds)
    joint_shuffled = native.fit_joint_cv(logits, shuffled_target, folds)

    fixture_rows = [
        ("a", 0, "FAR", 2),
        ("b", 0, "FAR", 2),
        ("c", 0, "FAR", 2),
        ("d", 0, "NEAR", 1),
        ("e", 1, "FAR", 2),
        ("f", 1, "FAR", 2),
    ]
    fixture = native.target_derangement_v1(
        [row[0] for row in fixture_rows],
        np.asarray([row[1] for row in fixture_rows]),
        [row[2] for row in fixture_rows],
        np.asarray([row[3] for row in fixture_rows]),
    )
    fixture_sha = hashlib.sha256(canonical_json_bytes(fixture["rows"])).hexdigest()
    if fixture_sha != config["posterior"]["target_shuffle"]["fixture_sha256"]:
        raise RuntimeError("target-shuffle canonical fixture drifted")

    states = {
        "schema_version": "proportional-posterior-states-v1",
        "marginal": {"real": marginal_real, "target_shuffled": marginal_shuffled},
        "joint": {
            "real": joint_real["state"],
            "target_shuffled": joint_shuffled["state"],
        },
        "target_shuffle": {
            "seed": int(config["posterior"]["target_shuffle"]["seed"]),
            "fixture_sha256": fixture_sha,
            "permutable_fraction": float(shuffle["permutable_fraction"]),
            "singletons": shuffle["singletons"],
            "same_map_for_representations": True,
        },
    }
    state_arrays: dict[str, np.ndarray] = {}
    _prefix_arrays("joint_real", joint_real["arrays"], state_arrays)
    _prefix_arrays("joint_target_shuffled", joint_shuffled["arrays"], state_arrays)
    oof_arrays = {
        key: value
        for key, value in state_arrays.items()
        if any(fragment in key for fragment in ("fold_id", "oof_", "fold_"))
    }
    final_arrays = {
        key: value
        for key, value in state_arrays.items()
        if key not in oof_arrays
    }
    write_json(output / "posterior_fit/states.json", states)
    write_npz(output / "posterior_fit/state_arrays.npz", final_arrays)
    write_npz(output / "posterior_fit/oof_arrays.npz", oof_arrays)
    write_json(output / "posterior_fit/target_shuffle_map.json", shuffle["rows"])
    write_npz(
        output / "posterior_fit/target_shuffle_arrays.npz",
        {
            "donor_index": shuffle["donor_index"],
            "permutable": shuffle["permutable"],
            "target_shuffled": shuffled_target,
        },
    )
    return {
        "states": states,
        "marginal_real": marginal_real,
        "marginal_shuffled": marginal_shuffled,
        "joint_real": joint_real,
        "joint_shuffled": joint_shuffled,
        "shuffle": shuffle,
        "shuffled_target": shuffled_target,
    }


def posterior_mass_for(
    name: str, posterior_fit: dict[str, Any], logits: np.ndarray, *, shuffled: bool = False
) -> np.ndarray:
    suffix = "shuffled" if shuffled else "real"
    if name == "marginal":
        return native.marginal_set_mass(posterior_fit[f"marginal_{suffix}"], logits)
    if name == "joint":
        fit = posterior_fit[f"joint_{suffix}"]
        return native.joint_set_mass(
            fit["state"], np.asarray(fit["arrays"]["final_theta"]), logits
        )
    raise ValueError(f"unknown posterior: {name}")


def _portable_state_arrays(states: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for posterior_name in ("marginal", "joint"):
        family = states[posterior_name]
        for role, bundle in [("true", family["true"])] + [
            (f"control_{row['seed']}", row) for row in family["controls"]
        ]:
            for model_name, state in bundle["states"].items():
                prefix = f"{posterior_name}__{role}__{model_name}"
                arrays[f"{prefix}__mean"] = np.asarray(state["mean"], dtype=np.float64)
                arrays[f"{prefix}__scale"] = np.asarray(state["scale"], dtype=np.float64)
                arrays[f"{prefix}__coef"] = np.asarray(state["coef"], dtype=np.float64)
                arrays[f"{prefix}__intercept"] = np.asarray(
                    [state["intercept"]], dtype=np.float64
                )
                if "n_iter" in state:
                    arrays[f"{prefix}__n_iter"] = np.asarray(state["n_iter"], dtype=np.int64)
    return arrays


def fit_policies(
    policy: dict[str, np.ndarray],
    posterior_fit: dict[str, Any],
    utilities: np.ndarray,
    config: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    logits = np.asarray(policy["ensemble_logits"], dtype=np.float64)
    seed_logits = np.asarray(policy["per_seed_logits"], dtype=np.float64)
    target = np.asarray(policy["target"], dtype=bool)
    tokens = np.asarray(policy["pair_token"]).astype(str)
    penalty = float(config["reader"]["penalty"])
    result: dict[str, Any] = {}
    state_document: dict[str, Any] = {
        "schema_version": "proportional-contextual-reader-states-v1",
        "feature_names": list(native.FEATURE_NAMES),
    }
    score_arrays: dict[str, np.ndarray] = {}
    control_arrays: dict[str, np.ndarray] = {}
    control_map_document: dict[str, Any] = {
        "schema_version": "proportional-matched-control-maps-v1"
    }
    for posterior_name in ("marginal", "joint"):
        mass = posterior_mass_for(posterior_name, posterior_fit, logits)
        training = native.reader_training_data(
            ensemble_logits=logits,
            per_seed_logits=seed_logits,
            target=target,
            set_mass=mass,
            utilities=utilities,
            penalty=penalty,
        )
        true_fit = native.fit_reader_states(training)
        controls = native.fit_control_states(training, tokens)["controls"]
        true_scores = native.score_reader_states(
            true_fit["states"], training["design"], training["disagreement"]
        )
        result[posterior_name] = {
            "mass": mass,
            "training": training,
            "true": true_fit,
            "controls": controls,
        }
        state_document[posterior_name] = {
            "true": true_fit,
            "controls": [
                {
                    "seed": row["seed"],
                    "states": row["states"],
                    "diagnostics": row["diagnostics"],
                }
                for row in controls
            ],
        }
        control_map_document[posterior_name] = [row["diagnostics"] for row in controls]
        prefix = f"{posterior_name}__true"
        _prefix_arrays(prefix, true_scores, score_arrays)
        for key in (
            "design",
            "weights",
            "disagreement",
            "gain",
            "harm",
            "incompatibility",
            "hard_actions",
            "posterior_actions",
        ):
            score_arrays[f"{posterior_name}__{key}"] = np.asarray(training[key])
        for row in controls:
            seed = int(row["seed"])
            control_scores = native.score_reader_states(
                row["states"], training["design"], training["disagreement"]
            )
            row["fit_scores"] = control_scores
            _prefix_arrays(f"{posterior_name}__control_{seed}", control_scores, score_arrays)
            control_arrays[f"{posterior_name}__control_{seed}__mapping"] = np.asarray(
                row["mapping"], dtype=np.int64
            )
            for target_name in ("gain", "harm", "incompatibility"):
                control_arrays[
                    f"{posterior_name}__control_{seed}__{target_name}"
                ] = np.asarray(row[target_name])

    if tuple(config["reader"]["feature_names"]) != native.FEATURE_NAMES:
        raise RuntimeError("feature schema drifted")
    write_json(
        output / "policy_fit/feature_schema.json",
        {
            "schema_version": "proportional-contextual-map-features-v1",
            "count": len(native.FEATURE_NAMES),
            "feature_names": list(native.FEATURE_NAMES),
            "hard_adapter": "HARD_MAP_SET",
            "weighting": "one_total_weight_per_active_token",
        },
    )
    write_json(output / "policy_fit/states.json", state_document)
    write_npz(output / "policy_fit/state_arrays.npz", _portable_state_arrays(state_document))
    write_npz(output / "policy_fit/fit_scores.npz", score_arrays)
    write_json(output / "policy_fit/control_maps.json", control_map_document)
    write_npz(output / "policy_fit/control_arrays.npz", control_arrays)
    return result


def select_and_apply(
    public: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    posterior_fit: dict[str, Any],
    policies: dict[str, Any],
    utilities: np.ndarray,
    config: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    logits = np.asarray(public["ensemble_logits"], dtype=np.float64)
    seed_logits = np.asarray(public["per_seed_logits"], dtype=np.float64)
    target = np.asarray(truth["target"], dtype=bool)
    penalty = float(config["reader"]["penalty"])
    scores_arrays: dict[str, np.ndarray] = {}
    candidate_arrays: dict[str, np.ndarray] = {}
    action_arrays: dict[str, np.ndarray] = {}
    matches_arrays: dict[str, np.ndarray] = {}
    freeze: dict[str, Any] = {
        "schema_version": "proportional-selection-freeze-v1",
        "status": "OPENED_DATA_IMPLEMENTATION_DIAGNOSTIC",
        "target_blind_application": True,
        "selection_truth_used_only_by_evaluator": True,
    }
    result: dict[str, Any] = {}
    for posterior_name in ("marginal", "joint"):
        mass = posterior_mass_for(posterior_name, posterior_fit, logits)
        shuffled_mass = posterior_mass_for(
            posterior_name, posterior_fit, logits, shuffled=True
        )
        public_data = native.reader_public_data(
            ensemble_logits=logits,
            per_seed_logits=seed_logits,
            set_mass=mass,
            utilities=utilities,
            penalty=penalty,
        )
        true_scores = native.score_reader_states(
            policies[posterior_name]["true"]["states"],
            public_data["design"],
            public_data["disagreement"],
        )
        grid = native.candidate_grid(
            true_scores,
            public_data["disagreement"],
            public_data["hard_actions"],
            public_data["posterior_actions"],
        )
        evaluated = native.evaluate_candidate_grid(
            grid, target, utilities, penalty, public_data["hard_actions"]
        )
        selected = evaluated["selected"]
        selected_index = int(evaluated["selected_index"])
        true_actions = np.asarray(grid["actions"][selected_index], dtype=np.int64)
        true_override = np.asarray(grid["override"][selected_index], dtype=bool)
        # Reapply only target-free state and demand bit-exact identity.
        if selected["kind"] == "hard_only":
            reapplied_actions = np.asarray(public_data["hard_actions"], dtype=np.int64)
            reapplied_override = np.zeros_like(true_override)
        else:
            reapplied = native.apply_threshold_triplet(
                true_scores,
                public_data["disagreement"],
                public_data["hard_actions"],
                public_data["posterior_actions"],
                selected,
            )
            reapplied_actions = reapplied["actions"]
            reapplied_override = reapplied["override"]
        if not np.array_equal(reapplied_actions, true_actions) or not np.array_equal(
            reapplied_override, true_override
        ):
            raise AssertionError("frozen target-blind reapplication drifted")

        controls: list[dict[str, Any]] = []
        valid_masks: list[np.ndarray] = []
        for control in policies[posterior_name]["controls"]:
            seed = int(control["seed"])
            control_scores = native.score_reader_states(
                control["states"], public_data["design"], public_data["disagreement"]
            )
            if selected["kind"] == "hard_only":
                thresholds: dict[str, float] = {
                    "proposer_quantile": 2.0,
                    "harm_quantile": 2.0,
                    "incompatibility_quantile": 2.0,
                    "proposer_threshold": float("inf"),
                    "harm_threshold": float("-inf"),
                    "incompatibility_threshold": float("-inf"),
                }
                matched = {
                    "actions": np.asarray(public_data["hard_actions"], dtype=np.int64),
                    "selected": np.zeros_like(true_override),
                    "authorized_universe": np.zeros_like(true_override),
                    "match_valid": np.ones(len(true_override), dtype=bool),
                    "requested_k": np.zeros(len(true_override), dtype=np.int64),
                }
            else:
                quantiles = (
                    float(selected["proposer_quantile"]),
                    float(selected["harm_quantile"]),
                    float(selected["incompatibility_quantile"]),
                )
                thresholds = native.threshold_triplet(
                    control_scores, public_data["disagreement"], quantiles
                )
                matched = native.matched_control_actions(
                    true_override=true_override,
                    scores=control_scores,
                    thresholds=thresholds,
                    disagreement=public_data["disagreement"],
                    hard_actions=public_data["hard_actions"],
                    candidate_actions=public_data["posterior_actions"],
                )
            valid_masks.append(np.asarray(matched["match_valid"], dtype=bool))
            controls.append(
                {
                    "seed": seed,
                    "scores": control_scores,
                    "thresholds": thresholds,
                    "matched": matched,
                }
            )
            _prefix_arrays(f"{posterior_name}__control_{seed}", control_scores, scores_arrays)
            for key, value in matched.items():
                matches_arrays[f"{posterior_name}__control_{seed}__{key}"] = np.asarray(value)
        u_true = true_override.any(axis=1)
        common = u_true.copy()
        for valid in valid_masks:
            common &= valid
        coverage = float(common.sum() / max(1, u_true.sum()))
        if u_true.any() and coverage < float(
            config["matched_controls"]["minimum_common_coverage"]
        ):
            common_status = "NOT_EVALUABLE_CONTROL_SUPPORT"
        elif not u_true.any():
            common_status = "NOT_EVALUABLE_NO_TRUE_OVERRIDES"
        else:
            common_status = "EVALUABLE"

        _prefix_arrays(f"{posterior_name}__true", true_scores, scores_arrays)
        for key in (
            "map_set_index",
            "map_set",
            "map_set_mass",
            "hard_actions",
            "posterior_actions",
            "disagreement",
            "design",
            "weights",
        ):
            scores_arrays[f"{posterior_name}__{key}"] = np.asarray(public_data[key])
        scores_arrays[f"{posterior_name}__set_mass_real"] = mass
        scores_arrays[f"{posterior_name}__set_mass_target_shuffled"] = shuffled_mass
        _prefix_arrays(posterior_name, evaluated["arrays"], candidate_arrays)
        candidate_arrays[f"{posterior_name}__actions"] = np.asarray(grid["actions"])
        candidate_arrays[f"{posterior_name}__override"] = np.asarray(grid["override"])
        action_arrays[f"{posterior_name}__hard_actions"] = np.asarray(
            public_data["hard_actions"]
        )
        action_arrays[f"{posterior_name}__posterior_actions"] = np.asarray(
            public_data["posterior_actions"]
        )
        action_arrays[f"{posterior_name}__contextual_actions"] = true_actions
        action_arrays[f"{posterior_name}__true_override"] = true_override
        matches_arrays[f"{posterior_name}__u_true"] = u_true
        matches_arrays[f"{posterior_name}__u_common"] = common
        freeze[posterior_name] = {
            "selected_index": selected_index,
            "selected": selected,
            "candidate_count": len(grid["metadata"]),
            "candidate_metadata": grid["metadata"],
            "control_thresholds": {
                str(row["seed"]): row["thresholds"] for row in controls
            },
            "u_true_count": int(u_true.sum()),
            "u_common_count": int(common.sum()),
            "common_coverage": coverage,
            "common_support_status": common_status,
        }
        result[posterior_name] = {
            "mass": mass,
            "shuffled_mass": shuffled_mass,
            "public": public_data,
            "true_scores": true_scores,
            "grid": grid,
            "evaluated": evaluated,
            "true_actions": true_actions,
            "true_override": true_override,
            "controls": controls,
            "u_true": u_true,
            "u_common": common,
            "common_support_status": common_status,
        }

    write_npz(output / "decision_select/scores.npz", scores_arrays)
    write_npz(output / "decision_select/candidate_metrics.npz", candidate_arrays)
    write_json(output / "decision_select/selection_freeze.json", freeze)
    write_npz(output / "decision_select/action_arrays.npz", action_arrays)
    write_json(
        output / "apply_fixture/action_freeze.json",
        {
            "schema_version": "proportional-target-blind-action-freeze-v1",
            "status": "OPENED_DATA_IMPLEMENTATION_DIAGNOSTIC",
            "public_bundle_keys": sorted(public),
            "truth_keys_received_by_applier": [],
            "posteriors": {
                name: {
                    "selected_index": freeze[name]["selected_index"],
                    "selected": freeze[name]["selected"],
                    "action_sha256": native.array_digest(result[name]["true_actions"]),
                    "override_sha256": native.array_digest(result[name]["true_override"]),
                }
                for name in ("marginal", "joint")
            },
        },
    )
    write_npz(output / "apply_fixture/actions_and_matches.npz", matches_arrays)
    return result


def _metric_summary(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "accuracy": float(np.mean(metrics["accuracy"])),
        "incompatibility": float(np.mean(metrics["incompatibility"])),
        "regret": float(np.mean(metrics["regret"])),
        "worst_regret": float(np.mean(metrics["worst_regret"])),
    }


def _estimand(
    *,
    row_id: str,
    instance: str,
    left_name: str,
    right_name: str,
    left: np.ndarray,
    right: np.ndarray,
    indices: np.ndarray | None,
    allow_zero_upper: bool,
    evaluable: bool = True,
    diagnostic_only: bool = False,
) -> dict[str, Any]:
    if not evaluable or indices is None or not len(left):
        return {
            "id": row_id,
            "instance": instance,
            "left": left_name,
            "right": right_name,
            "orientation": "left_minus_right",
            "status": "NOT_EVALUABLE",
            "diagnostic_only": diagnostic_only,
            "n_tokens": int(len(left)),
        }
    summary = native.paired_delta_summary(left, right, indices)
    status = (
        "DESCRIPTIVE_ONLY"
        if diagnostic_only
        else native.classify_loss_delta(
            summary, allow_zero_upper=allow_zero_upper, support_ok=True
        )
    )
    return {
        "id": row_id,
        "instance": instance,
        "left": left_name,
        "right": right_name,
        "orientation": "left_minus_right",
        "status": status,
        "diagnostic_only": diagnostic_only,
        "allow_zero_upper": bool(allow_zero_upper),
        **summary,
    }


def evaluate_diagnostics(
    public: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    selection: dict[str, Any],
    posterior_fit: dict[str, Any],
    policies: dict[str, Any],
    utilities: np.ndarray,
    config: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    tokens = np.asarray(public["pair_token"]).astype(str)
    target = np.asarray(truth["target"], dtype=bool)
    penalty = float(config["reader"]["penalty"])
    n = len(tokens)
    global_boot = native.bootstrap_indices(
        n, int(config["bootstrap"]["replicates"]), int(config["bootstrap"]["global_seed"])
    )
    boot_arrays: dict[str, np.ndarray] = {
        "global_pair_token_index": global_boot,
        "global_pair_token": tokens,
    }
    raw: dict[str, np.ndarray] = {"pair_token": tokens, "target": target}
    metrics_document: dict[str, Any] = {
        "schema_version": "proportional-opened-diagnostic-metrics-v1",
        "status": "OPENED_DATA_IMPLEMENTATION_DIAGNOSTIC",
        "posteriors": {},
    }
    cells: dict[str, dict[str, np.ndarray]] = {}
    set_metrics: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    action_metrics: dict[str, dict[str, Any]] = {}
    common_boot: dict[str, np.ndarray | None] = {}
    for posterior_index, posterior_name in enumerate(("marginal", "joint")):
        row = selection[posterior_name]
        set_real = native.exact_set_metric_arrays(row["mass"], target)
        set_shuffled = native.exact_set_metric_arrays(row["shuffled_mass"], target)
        set_metrics[posterior_name] = {"real": set_real, "target_shuffled": set_shuffled}
        hard = native.action_metric_arrays(
            row["public"]["hard_actions"], target, utilities, penalty
        )
        contextual = native.action_metric_arrays(
            row["true_actions"], target, utilities, penalty
        )
        controls_metrics: list[dict[str, np.ndarray]] = []
        for control in row["controls"]:
            cm = native.action_metric_arrays(
                control["matched"]["actions"], target, utilities, penalty
            )
            controls_metrics.append(cm)
            seed = int(control["seed"])
            for metric_name, values in cm.items():
                raw[f"{posterior_name}__control_{seed}__{metric_name}"] = values
        action_metrics[posterior_name] = {
            "hard": hard,
            "contextual": contextual,
            "controls": controls_metrics,
        }
        for state_name, state_metrics in (("hard", hard), ("contextual", contextual)):
            cells[f"{posterior_name}_{state_name}"] = state_metrics
            for metric_name, values in state_metrics.items():
                raw[f"{posterior_name}__{state_name}__{metric_name}"] = values
        for fit_name, values in (("real", set_real), ("target_shuffled", set_shuffled)):
            for metric_name, metric_values in values.items():
                raw[f"{posterior_name}__{fit_name}__{metric_name}"] = metric_values
        common = np.asarray(row["u_common"], dtype=bool)
        support_tokens = tokens[common]
        if len(support_tokens):
            seed = int(
                config["bootstrap"][
                    "marginal_common_seed" if posterior_name == "marginal" else "joint_common_seed"
                ]
            )
            boot = native.bootstrap_indices(len(support_tokens), 5000, seed)
            common_boot[posterior_name] = boot
            boot_arrays[f"{posterior_name}__common_index"] = boot
            boot_arrays[f"{posterior_name}__common_pair_token"] = support_tokens
        else:
            common_boot[posterior_name] = None
            boot_arrays[f"{posterior_name}__common_index"] = np.empty((5000, 0), dtype=np.int64)
            boot_arrays[f"{posterior_name}__common_pair_token"] = support_tokens
        metrics_document["posteriors"][posterior_name] = {
            "set_real": {key: float(np.mean(value)) for key, value in set_real.items()},
            "set_target_shuffled": {
                key: float(np.mean(value)) for key, value in set_shuffled.items()
            },
            "hard": _metric_summary(hard),
            "contextual": _metric_summary(contextual),
            "matched_controls": [
                {
                    "seed": int(row["controls"][i]["seed"]),
                    **_metric_summary(value),
                }
                for i, value in enumerate(controls_metrics)
            ],
            "u_true_count": int(row["u_true"].sum()),
            "u_common_count": int(common.sum()),
            "common_support_status": row["common_support_status"],
        }

    estimands: list[dict[str, Any]] = []
    estimands.append(
        _estimand(
            row_id="SET_JOINT_NLL",
            instance="joint_minus_marginal",
            left_name="joint_real_exact_set_nll",
            right_name="marginal_real_exact_set_nll",
            left=set_metrics["joint"]["real"]["exact_set_nll"],
            right=set_metrics["marginal"]["real"]["exact_set_nll"],
            indices=global_boot,
            allow_zero_upper=False,
        )
    )
    estimands.append(
        _estimand(
            row_id="SET_JOINT_BRIER",
            instance="joint_minus_marginal",
            left_name="joint_real_marginal_brier",
            right_name="marginal_real_marginal_brier",
            left=set_metrics["joint"]["real"]["marginal_brier"],
            right=set_metrics["marginal"]["real"]["marginal_brier"],
            indices=global_boot,
            allow_zero_upper=True,
        )
    )
    for posterior_name in ("marginal", "joint"):
        estimands.append(
            _estimand(
                row_id="SET_SHUFFLE",
                instance=posterior_name,
                left_name=f"{posterior_name}_real_exact_set_nll",
                right_name=f"{posterior_name}_target_shuffled_exact_set_nll",
                left=set_metrics[posterior_name]["real"]["exact_set_nll"],
                right=set_metrics[posterior_name]["target_shuffled"]["exact_set_nll"],
                indices=global_boot,
                allow_zero_upper=False,
            )
        )
        pairs = (
            ("READER_REGRET", "regret", False),
            ("READER_COMPAT", "incompatibility", True),
            ("READER_WORST", "worst_regret", True),
        )
        for row_id, metric_name, allow_zero in pairs:
            estimands.append(
                _estimand(
                    row_id=row_id,
                    instance=posterior_name,
                    left_name=f"{posterior_name}_contextual_{metric_name}",
                    right_name=f"{posterior_name}_hard_{metric_name}",
                    left=action_metrics[posterior_name]["contextual"][metric_name],
                    right=action_metrics[posterior_name]["hard"][metric_name],
                    indices=global_boot,
                    allow_zero_upper=allow_zero,
                )
            )
        common = np.asarray(selection[posterior_name]["u_common"], dtype=bool)
        controls_regret = np.stack(
            [value["regret"] for value in action_metrics[posterior_name]["controls"]], axis=0
        )
        estimands.append(
            _estimand(
                row_id="READER_CONTROL",
                instance=posterior_name,
                left_name=f"{posterior_name}_contextual_regret",
                right_name=f"{posterior_name}_matched_control_mean_regret",
                left=action_metrics[posterior_name]["contextual"]["regret"][common],
                right=np.mean(controls_regret[:, common], axis=0),
                indices=common_boot[posterior_name],
                allow_zero_upper=False,
                evaluable=selection[posterior_name]["common_support_status"] == "EVALUABLE",
            )
        )
    interaction_left = (
        action_metrics["joint"]["contextual"]["regret"]
        - action_metrics["joint"]["hard"]["regret"]
    )
    interaction_right = (
        action_metrics["marginal"]["contextual"]["regret"]
        - action_metrics["marginal"]["hard"]["regret"]
    )
    estimands.append(
        _estimand(
            row_id="FACTOR_INTERACTION",
            instance="joint_reader_delta_minus_marginal_reader_delta",
            left_name="joint_contextual_minus_hard_regret",
            right_name="marginal_contextual_minus_hard_regret",
            left=interaction_left,
            right=interaction_right,
            indices=global_boot,
            allow_zero_upper=True,
            diagnostic_only=True,
        )
    )
    observed_ids = {row["id"] for row in estimands}
    if observed_ids != set(config["required_decision_rows"]):
        raise AssertionError("estimand row coverage drifted")

    def _row_status(row_id: str, instance: str | None = None) -> bool:
        matching = [
            row
            for row in estimands
            if row["id"] == row_id and (instance is None or row["instance"] == instance)
        ]
        return bool(matching) and all(row["status"] == "CONDITION_SATISFIED" for row in matching)

    patterns = {
        "JOINT_PATTERN_PRESENT": all(
            [
                _row_status("SET_JOINT_NLL"),
                _row_status("SET_JOINT_BRIER"),
                _row_status("SET_SHUFFLE", "joint"),
            ]
        ),
        "CONTEXTUAL_PATTERN_PRESENT": {
            posterior_name: all(
                _row_status(row_id, posterior_name)
                for row_id in (
                    "READER_REGRET",
                    "READER_COMPAT",
                    "READER_WORST",
                    "READER_CONTROL",
                )
            )
            for posterior_name in ("marginal", "joint")
        },
        "interpretation": "OPENED_DATA_IMPLEMENTATION_DIAGNOSTIC_ONLY",
    }
    estimand_document = {
        "schema_version": "proportional-set-valued-estimands-v1",
        "status": "OPENED_DATA_IMPLEMENTATION_DIAGNOSTIC",
        "bootstrap_unit": "pair_token",
        "training_seed_population_claimed": False,
        "rows": estimands,
        "patterns": patterns,
    }

    sensitivity_arrays: dict[str, np.ndarray] = {}
    sensitivity_rows: list[dict[str, Any]] = []
    per_seed_logits = np.asarray(public["per_seed_logits"], dtype=np.float64)
    cardinality = np.asarray(public["cardinality"], dtype=np.int64)
    for checkpoint_index, checkpoint_epoch in enumerate(config["checkpoint_epochs"]):
        checkpoint_logits = per_seed_logits[checkpoint_index]
        for posterior_name in ("marginal", "joint"):
            mass = posterior_mass_for(posterior_name, posterior_fit, checkpoint_logits)
            pdata = native.reader_public_data(
                ensemble_logits=checkpoint_logits,
                per_seed_logits=per_seed_logits,
                set_mass=mass,
                utilities=utilities,
                penalty=penalty,
            )
            scores = native.score_reader_states(
                policies[posterior_name]["true"]["states"],
                pdata["design"],
                pdata["disagreement"],
            )
            selected = selection[posterior_name]["evaluated"]["selected"]
            if selected["kind"] == "hard_only":
                contextual_actions = pdata["hard_actions"]
            else:
                contextual_actions = native.apply_threshold_triplet(
                    scores,
                    pdata["disagreement"],
                    pdata["hard_actions"],
                    pdata["posterior_actions"],
                    selected,
                )["actions"]
            for reader_name, actions in (
                ("hard", pdata["hard_actions"]),
                ("contextual", contextual_actions),
            ):
                metric = native.action_metric_arrays(actions, target, utilities, penalty)
                prefix = f"checkpoint_{checkpoint_epoch}__{posterior_name}__{reader_name}"
                sensitivity_arrays[f"{prefix}__actions"] = np.asarray(actions)
                sensitivity_arrays[f"{prefix}__regret_by_policy"] = metric[
                    "regret_by_policy"
                ]
                sensitivity_arrays[f"{prefix}__incompatibility_by_policy"] = metric[
                    "incompatibility_by_policy"
                ]
                for card in sorted(np.unique(cardinality).tolist()):
                    mask = cardinality == card
                    sensitivity_rows.append(
                        {
                            "checkpoint_epoch": int(checkpoint_epoch),
                            "checkpoint_is_population_seed": False,
                            "posterior": posterior_name,
                            "reader": reader_name,
                            "cardinality": int(card),
                            "n_tokens": int(mask.sum()),
                            "mean_regret": float(metric["regret"][mask].mean()),
                            "mean_incompatibility": float(
                                metric["incompatibility"][mask].mean()
                            ),
                            "mean_accuracy": float(metric["accuracy"][mask].mean()),
                        }
                    )
    metrics_document["checkpoint_sensitivity"] = sensitivity_rows

    duplications: list[dict[str, Any]] = []
    cell_actions = {
        "marginal_hard": selection["marginal"]["public"]["hard_actions"],
        "marginal_contextual": selection["marginal"]["true_actions"],
        "joint_hard": selection["joint"]["public"]["hard_actions"],
        "joint_contextual": selection["joint"]["true_actions"],
    }
    for left, right in itertools.combinations(sorted(cells), 2):
        action_left = np.asarray(cell_actions[left])
        action_right = np.asarray(cell_actions[right])
        equal_fraction = float(np.mean(action_left == action_right))
        duplications.append(
            {
                "left": left,
                "right": right,
                "actions_exact": bool(np.array_equal(action_left, action_right)),
                "action_position_equal_fraction": equal_fraction,
                "regret_exact": bool(
                    np.array_equal(cells[left]["regret_by_policy"], cells[right]["regret_by_policy"])
                ),
            }
        )
    duplication_document = {
        "schema_version": "proportional-cell-duplications-v1",
        "cells_retained_even_if_equal": True,
        "comparisons": duplications,
    }
    boot_arrays["global_index_sha256_utf8"] = np.asarray(
        [native.array_digest(global_boot)], dtype="U64"
    )
    write_json(output / "evaluate_fixture/diagnostic_metrics.json", metrics_document)
    write_npz(output / "evaluate_fixture/diagnostic_arrays.npz", raw)
    write_npz(output / "evaluate_fixture/bootstrap_indices.npz", boot_arrays)
    write_json(output / "evaluate_fixture/estimand_table.json", estimand_document)
    write_npz(output / "evaluate_fixture/sensitivity_arrays.npz", sensitivity_arrays)
    write_json(output / "evaluate_fixture/cell_duplications.json", duplication_document)
    return {
        "metrics": metrics_document,
        "estimands": estimand_document,
        "duplications": duplication_document,
    }


def write_report(output: Path, diagnostic: dict[str, Any]) -> None:
    rows = diagnostic["estimands"]["rows"]
    lines = [
        "# Preflight CPU de la rama set-valued nativa",
        "",
        "Estado: `RUNNER_PREFLIGHT_VALID`.",
        "",
        "Este paquete valida implementación y replay sobre poblaciones históricas ya abiertas. "
        "No crea un draw prospectivo, no usa monitor o lockbox y no emite una decisión científica.",
        "",
        "## Contratos ejercitados",
        "",
        "- Posteriores: MARGINAL pooled Platt y JOINT `joint_full` con selección OOF propia.",
        "- Readers: HARD por set MAP y CONTEXTUAL con 17 features ligadas a cada posterior.",
        "- Controles: cinco transportes matched por posterior y soporte común explícito.",
        "- Incertidumbre: bootstrap pareado de 5.000 réplicas por `pair_token`.",
        "- Sensibilidad: checkpoints 17/29/43 como cortes históricos, no como población de seeds.",
        "",
        "## Tabla diagnóstica",
        "",
        "| ID | Instancia | Estado | N | Media izquierda-derecha | CI95 |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        if "mean_diff" in row:
            interval = f"[{row['ci95_low']:.8g}, {row['ci95_high']:.8g}]"
            mean = f"{row['mean_diff']:.8g}"
        else:
            interval = "n/a"
            mean = "n/a"
        lines.append(
            f"| {row['id']} | {row['instance']} | {row['status']} | "
            f"{row.get('n_tokens', 0)} | {mean} | {interval} |"
        )
    lines.extend(
        [
            "",
            "Las etiquetas anteriores son diagnósticos de implementación sobre datos abiertos. "
            "No acreditan cobertura prospectiva, generalización ni variabilidad de entrenamiento.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def artifact_class(relative: str) -> str:
    if relative in {"config.snapshot.json", "source_bindings.json", "input_receipt.json"}:
        return "source_snapshot"
    if relative.endswith("REPORT.md"):
        return "regenerable_report"
    if relative in {"runtime.json", "replay_receipt.json"}:
        return "receipt"
    if relative.endswith(".npz") or relative.endswith("states.json") or "freeze.json" in relative:
        return "raw_state"
    return "derived_diagnostic"


def build_artifact_manifest(output: Path) -> dict[str, Any]:
    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        relative = path.relative_to(output).as_posix()
        files.append(
            {
                "path": relative,
                "class": artifact_class(relative),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": "proportional-artifact-manifest-v1",
        "self_excluded": True,
        "files": files,
    }


def compare_reference(output: Path, reference: Path | None) -> dict[str, Any]:
    if reference is None:
        return {
            "schema_version": "proportional-replay-receipt-v1",
            "mode": "primary",
            "reference_supplied": False,
            "excluded_paths": sorted(REPLAY_EXCLUDED),
            "byte_exact": None,
        }
    reference = reference.resolve(strict=True)
    current_files = {
        path.relative_to(output).as_posix(): path
        for path in output.rglob("*")
        if path.is_file() and path.relative_to(output).as_posix() not in REPLAY_EXCLUDED
    }
    reference_files = {
        path.relative_to(reference).as_posix(): path
        for path in reference.rglob("*")
        if path.is_file() and path.relative_to(reference).as_posix() not in REPLAY_EXCLUDED
    }
    if set(current_files) != set(reference_files):
        raise RuntimeError("replay artifact inventory differs from reference")
    mismatches = [
        relative
        for relative in sorted(current_files)
        if sha256_file(current_files[relative]) != sha256_file(reference_files[relative])
    ]
    if mismatches:
        raise RuntimeError(f"replay byte mismatch: {mismatches}")
    reference_runtime = read_json(reference / "runtime.json")
    return {
        "schema_version": "proportional-replay-receipt-v1",
        "mode": "replay",
        "reference_supplied": True,
        "excluded_paths": sorted(REPLAY_EXCLUDED),
        "compared_files": len(current_files),
        "byte_exact": True,
        "reference_artifact_manifest_sha256": sha256_file(
            reference / "artifact_manifest.json"
        ),
        "reference_wall_seconds": float(reference_runtime["wall_seconds"]),
    }


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve(strict=True)
    config = read_json(config_path)
    environment = validate_environment(config)
    source_bindings = source_preflight(config, config_path)
    output = args.output_dir.resolve()
    archived = prepare_output(output, args.force)
    started_wall = time.monotonic()
    started_cpu = time.process_time()
    phase_seconds: dict[str, float] = {}

    def phase(name: str, function: Any, *values: Any) -> Any:
        before = time.monotonic()
        value = function(*values)
        phase_seconds[name] = time.monotonic() - before
        return value

    write_json(output / "config.snapshot.json", config)
    write_json(output / "source_bindings.json", source_bindings)
    paths = _binding_paths(config)
    fixtures, fixture_manifest = phase(
        "prepare_opened_fixtures",
        prepare_opened_fixtures,
        paths,
        output,
        config["opened_fixture_roles"],
    )
    input_receipt = {
        "schema_version": "proportional-opened-input-receipt-v1",
        "status": "OPENED_DATA_IMPLEMENTATION_DIAGNOSTIC",
        "source_commit": source_bindings["git_commit"],
        "logical_phase_separation_only": True,
        "fixture_manifest_sha256": sha256_file(output / "prepared/fixture_manifest.json"),
        "prepared_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted((output / "prepared").glob("*.npz"))
        },
    }
    write_json(output / "input_receipt.json", input_receipt)
    posterior_fit = phase(
        "fit_posteriors", fit_posteriors, fixtures["posterior_fit"], config, output
    )
    policy_manifest = read_json(paths["POLICY_MANIFEST"])
    utilities = native.utilities_from_manifest(policy_manifest)
    policies = phase(
        "fit_policies",
        fit_policies,
        fixtures["policy_fit"],
        posterior_fit,
        utilities,
        config,
        output,
    )
    selection = phase(
        "select_and_apply",
        select_and_apply,
        fixtures["decision_select_public"],
        fixtures["decision_select_truth"],
        posterior_fit,
        policies,
        utilities,
        config,
        output,
    )
    diagnostic = phase(
        "evaluate_diagnostics",
        evaluate_diagnostics,
        fixtures["decision_select_public"],
        fixtures["decision_select_truth"],
        selection,
        posterior_fit,
        policies,
        utilities,
        config,
        output,
    )
    if "torch" in sys.modules:
        raise RuntimeError("torch was imported during CPU preflight")
    write_report(output, diagnostic)
    elapsed = time.monotonic() - started_wall
    cpu_elapsed = time.process_time() - started_cpu
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    runtime = {
        "schema_version": "proportional-preflight-runtime-v1",
        "wall_seconds": elapsed,
        "cpu_seconds": cpu_elapsed,
        "peak_rss_bytes": peak_rss,
        "phase_seconds": phase_seconds,
        "environment": environment,
        "archived_previous_output": None if archived is None else str(archived),
        "torch_imported": False,
        "gpu_used_or_queried": False,
        "status": "RUNNER_PREFLIGHT_VALID",
    }
    if peak_rss > int(config["budgets"]["peak_rss_bytes"]):
        raise RuntimeError("runner peak RSS exceeded hard budget")
    if elapsed > float(config["budgets"]["runner_primary_plus_replay_seconds"]):
        raise RuntimeError("single runner exceeded combined hard wall budget")
    write_json(output / "runtime.json", runtime)
    replay_receipt = compare_reference(output, args.reference_dir)
    if args.reference_dir is not None:
        total = float(replay_receipt["reference_wall_seconds"]) + elapsed
        replay_receipt["primary_plus_replay_wall_seconds"] = total
        replay_receipt["within_hard_budget"] = total <= float(
            config["budgets"]["runner_primary_plus_replay_seconds"]
        )
        if not replay_receipt["within_hard_budget"]:
            raise RuntimeError("primary plus replay exceeded hard wall budget")
    write_json(output / "replay_receipt.json", replay_receipt)
    write_json(output / "artifact_manifest.json", build_artifact_manifest(output))
    print(
        json.dumps(
            {
                "status": "RUNNER_PREFLIGHT_VALID",
                "output": str(output),
                "wall_seconds": elapsed,
                "peak_rss_bytes": peak_rss,
                "replay": replay_receipt["byte_exact"],
                "fixture_roles": fixture_manifest["roles"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
