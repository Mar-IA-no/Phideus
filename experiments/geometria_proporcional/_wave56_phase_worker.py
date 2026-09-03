#!/usr/bin/env python3
"""Unprivileged analytical worker for the three prospective Wave 56 phases."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any

import numpy as np


RUNTIME_ROOT = Path(__file__).resolve().parent
STAGED_RUNTIME = (
    (RUNTIME_ROOT / "run_wave56_retrospective.py").is_file()
    and (RUNTIME_ROOT / "geometria_proporcional/__init__.py").is_file()
)


def _identity_can_write(path: Path, uid: int = 65534, gid: int = 65534) -> bool:
    metadata = path.stat(follow_symlinks=False)
    mode = stat.S_IMODE(metadata.st_mode)
    if metadata.st_uid == uid:
        return bool(mode & stat.S_IWUSR)
    if metadata.st_gid == gid:
        return bool(mode & stat.S_IWGRP)
    return bool(mode & stat.S_IWOTH)


def _validate_staged_runtime_permissions() -> None:
    checked = [RUNTIME_ROOT.parent, RUNTIME_ROOT, *RUNTIME_ROOT.rglob("*")]
    for path in checked:
        if path.is_symlink():
            raise RuntimeError(f"staged runtime contains a symlink: {path}")
        if not path.is_dir() and path.suffix != ".py":
            continue
        if _identity_can_write(path):
            raise PermissionError(f"staged runtime is writable by nobody: {path}")
        if os.geteuid() == 65534 and os.access(path, os.W_OK, effective_ids=True):
            raise PermissionError(f"staged runtime ACL is writable by nobody: {path}")


def _configure_import_path() -> None:
    if STAGED_RUNTIME:
        _validate_staged_runtime_permissions()
        retained: list[str] = []
        for raw in sys.path:
            if not raw:
                continue
            resolved = Path(raw).resolve()
            if resolved == RUNTIME_ROOT or resolved == Path.cwd().resolve():
                continue
            if resolved == Path("/tmp") or Path("/tmp") in resolved.parents:
                continue
            if (resolved / "run_wave56_retrospective.py").exists():
                continue
            if (resolved / "geometria_proporcional").exists():
                continue
            retained.append(str(resolved))
        sys.path[:] = [str(RUNTIME_ROOT), *dict.fromkeys(retained)]
        return

    if RUNTIME_ROOT == Path("/tmp") or Path("/tmp") in RUNTIME_ROOT.parents:
        raise RuntimeError("temporary worker copy has an incomplete staged runtime")
    repo_root = Path(__file__).resolve().parents[2]
    for source_root in (
        repo_root / "src",
        repo_root / "experiments/geometria_proporcional",
    ):
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))


_configure_import_path()

from geometria_proporcional.wave50_neural import load_labeled_records  # noqa: E402
from geometria_proporcional.wave52_policy import authorized_actions  # noqa: E402
from geometria_proporcional.wave54_joint_set import target_set_indices  # noqa: E402
from geometria_proporcional.wave55_policy_bridge import action_metric_arrays  # noqa: E402
from geometria_proporcional.wave56_contextual_gate import (  # noqa: E402
    FEATURE_NAMES,
    apply_gate,
    stratified_gain_shuffle,
)
import run_wave56_retrospective as retrospective  # noqa: E402

# The retrospective module has a legacy checkout-path bootstrap. Remove any
# path it inferred from the temporary copy before analytical work can begin.
if STAGED_RUNTIME:
    _configure_import_path()


PHASE_TO_SPLIT = {"fit": "train", "select": "val", "adjudicate": "lockbox"}
PHASE_TO_ROLE = {
    "fit": "gate_fit",
    "select": "gate_select",
    "adjudicate": "sealed_monitor",
}
SHUFFLE_SEEDS = (56031, 56032, 56033, 56034, 56035)
THRESHOLD_QUANTILES = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975)
SCALAR_GRID: tuple[float | str, ...] = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, "hard_only")
BOOTSTRAP_SEED = 5607
BOOTSTRAP_REPLICATES = 5000
HARD_ONLY = "hard_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--forbidden-probe", type=Path, action="append", default=[])
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_module_receipts(request: dict[str, Any]) -> list[dict[str, str]]:
    if not STAGED_RUNTIME:
        raise RuntimeError("analytical execution requires the isolated staged runtime")
    expected_hashes = request.get("execution_sources")
    if not isinstance(expected_hashes, dict):
        raise RuntimeError("phase request omitted execution source hashes")

    required_modules = {
        "run_wave56_retrospective",
        "geometria_proporcional",
        "geometria_proporcional.wave50_neural",
        "geometria_proporcional.wave52_policy",
        "geometria_proporcional.wave54_joint_set",
        "geometria_proporcional.wave55_policy_bridge",
        "geometria_proporcional.wave56_contextual_gate",
    }
    observed: dict[str, Any] = {}
    for name, module in sys.modules.items():
        raw_file = getattr(module, "__file__", None)
        is_worker = bool(raw_file) and Path(raw_file).resolve() == Path(__file__).resolve()
        if is_worker or name == "run_wave56_retrospective" or name.startswith(
            "geometria_proporcional"
        ):
            observed[name] = module
    worker_modules = [
        name
        for name, module in observed.items()
        if Path(getattr(module, "__file__", "")).resolve() == Path(__file__).resolve()
    ]
    if not worker_modules:
        raise RuntimeError("staged worker module is absent from sys.modules")
    missing = required_modules - set(observed)
    if missing:
        raise RuntimeError(f"required local modules were not imported: {sorted(missing)}")

    receipts = []
    for name, module in sorted(observed.items()):
        raw_file = getattr(module, "__file__", None)
        if not raw_file:
            raise RuntimeError(f"local module has no __file__: {name}")
        module_file = Path(raw_file).resolve(strict=True)
        try:
            relative = module_file.relative_to(RUNTIME_ROOT)
        except ValueError as error:
            raise RuntimeError(
                f"local module resolved outside staged runtime: {name} -> {module_file}"
            ) from error
        if relative == Path(Path(__file__).name):
            source = "experiments/geometria_proporcional/_wave56_phase_worker.py"
        elif relative == Path("run_wave56_retrospective.py"):
            source = "experiments/geometria_proporcional/run_wave56_retrospective.py"
        elif relative.parts and relative.parts[0] == "geometria_proporcional":
            source = f"src/{relative.as_posix()}"
        else:
            raise RuntimeError(f"unrecognized local module path: {module_file}")
        expected = expected_hashes.get(source)
        actual = sha256_file(module_file)
        if not isinstance(expected, str) or actual != expected:
            raise RuntimeError(
                f"staged local module hash mismatch: {name} ({actual} != {expected})"
            )
        receipts.append(
            {
                "module": name,
                "module_file": str(module_file),
                "runtime_relative_path": relative.as_posix(),
                "source": source,
                "sha256": actual,
            }
        )
    return receipts


def _validate_json(value: Any, location: str = "$") -> None:
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError(f"non-finite JSON value at {location}")
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_json(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json(item, f"{location}[{index}]")


def write_json_atomic(path: Path, payload: Any) -> None:
    _validate_json(payload)
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def inventory(root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(root)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def process_security_state() -> dict[str, Any]:
    status: dict[str, str] = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            status[key] = value.strip()
    effective_capabilities = int(status.get("CapEff", "-1"), 16)
    no_new_privileges = int(status.get("NoNewPrivs", "0"))
    supplementary_groups = os.getgroups()
    if effective_capabilities != 0 or no_new_privileges != 1 or supplementary_groups:
        raise PermissionError("analytical worker privilege drop is incomplete")
    return {
        "effective_capabilities_hex": status["CapEff"],
        "no_new_privileges": no_new_privileges,
        "supplementary_groups": supplementary_groups,
    }


def _nested(config: dict[str, Any], *paths: tuple[str, ...], default: Any = None) -> Any:
    for path in paths:
        value: Any = config
        for key in path:
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            return value
    return default


def validate_frozen_config(config: dict[str, Any]) -> None:
    contextual_alpha = float(
        _nested(config, ("primary_model", "alpha"), ("models", "ridge_contextual", "alpha"), default=1.0)
    )
    advantage_alpha = float(
        _nested(
            config,
            ("advantage_only_control", "alpha"),
            ("advantage_only_model", "alpha"),
            ("models", "ridge_advantage_only", "alpha"),
            default=100.0,
        )
    )
    if contextual_alpha != 1.0 or advantage_alpha != 100.0:
        raise RuntimeError("prospective Ridge alphas drifted from 1.0/100.0")
    quantiles = tuple(float(value) for value in config.get("threshold_quantiles", THRESHOLD_QUANTILES))
    if quantiles != THRESHOLD_QUANTILES:
        raise RuntimeError("prospective threshold quantiles drifted")
    shuffles = tuple(int(value) for value in config.get("shuffle_seeds", SHUFFLE_SEEDS))
    if shuffles != SHUFFLE_SEEDS:
        raise RuntimeError("prospective shuffle seeds drifted")
    scalar = tuple(config.get("scalar_gamma_grid", config.get("wave55_gamma_grid", SCALAR_GRID)))
    if scalar != SCALAR_GRID:
        raise RuntimeError("Wave 55 scalar grid drifted")
    bootstrap = config.get("bootstrap", {})
    if int(bootstrap.get("seed", BOOTSTRAP_SEED)) != BOOTSTRAP_SEED:
        raise RuntimeError("bootstrap seed drifted")
    if int(bootstrap.get("replicates", BOOTSTRAP_REPLICATES)) != BOOTSTRAP_REPLICATES:
        raise RuntimeError("bootstrap replicate count drifted")
    if config.get("physical_splits") != {
        "train": "gate_fit",
        "val": "gate_select",
        "lockbox": "sealed_monitor",
    }:
        raise RuntimeError("physical split roles drifted")
    if tuple(config.get("arms", ())) != (
        "hard_set_policy",
        "pure_joint_full",
        "scalar_advantage_gate",
        "contextual_value_gate",
        "advantage_only_value_gate",
        "contextual_shuffled_gain",
        "oracle_positive_gain",
    ):
        raise RuntimeError("prospective arm inventory drifted")
    absent = config.get("absent_support", {})
    if absent.get("source") != "wave54_calibration_fit_unseen_sets":
        raise RuntimeError("absent-support provenance drifted")
    indices = absent.get("set_indices")
    if indices != [0, 4, 8, 10, 12]:
        raise RuntimeError("Wave 54 absent-support indices drifted")
    if int(config.get("minimums", {}).get("absent_support_tokens", -1)) != 30:
        raise RuntimeError("absent-support minimum drifted")


def validate_stage(stage: Path, phase: str) -> list[str]:
    request = json.loads((stage / "phase_request.json").read_text(encoding="utf-8"))
    split = PHASE_TO_SPLIT[phase]
    role = PHASE_TO_ROLE[phase]
    if request.get("phase") != phase or request.get("split") != split or request.get("role") != role:
        raise RuntimeError("phase request scope mismatch")
    files = sorted(str(path.relative_to(stage)) for path in stage.rglob("*") if path.is_file())
    required = {
        "phase_request.json",
        "config.json",
        "protocol_config.json",
        f"visible/{split}.jsonl",
        f"labels/{split}.jsonl",
        "frozen/policy_manifest.json",
        "frozen/wave54_selection_freeze.json",
        "frozen/historical_pair_tokens.json",
    }
    if phase in {"select", "adjudicate"}:
        required.update({
            "previous/fit_core.json",
            "previous/fit_arrays.npz",
            "previous/fit_freeze.json",
            "previous/feature_schema.json",
        })
    if phase == "adjudicate":
        required.update({
            "previous/selection_core.json",
            "previous/selection_arrays.npz",
            "previous/selection_freeze.json",
        })
    missing = required - set(files)
    if missing:
        raise RuntimeError(f"phase stage missing files: {sorted(missing)}")
    allowed_prefixes = required | {name for name in files if name.startswith("inference/logits/")}
    unexpected = set(files) - allowed_prefixes
    if unexpected:
        raise RuntimeError(f"phase stage has unexpected files: {sorted(unexpected)}")
    expected_inputs = request.get("staged_input_hashes")
    actual_inputs = {
        name: sha256_file(stage / name)
        for name in files
        if name != "phase_request.json"
    }
    if not isinstance(expected_inputs, dict) or actual_inputs != expected_inputs:
        raise RuntimeError("phase stage differs from the coordinator-frozen input inventory")
    forbidden = ("sealed", "oracle", "optimizer", "history", "benchmark")
    offenders = [name for name in files if any(term in name.lower() for term in forbidden)]
    if offenders:
        raise RuntimeError(f"phase stage contains forbidden material: {offenders}")
    return files


def verify_forbidden_probes(paths: list[Path]) -> list[dict[str, Any]]:
    results = []
    if not paths:
        raise RuntimeError("at least one root-only sealed probe is required")
    for path in paths:
        try:
            with path.open("rb") as handle:
                handle.read(1)
        except PermissionError:
            results.append({"name": path.name, "denied": True})
        else:
            raise RuntimeError(f"sealed probe was not denied: {path.name}")
    return results


def _logit_files(stage: Path, split: str, seeds: list[int]) -> dict[int, Path]:
    files = list((stage / "inference/logits").glob(f"*{split}*.npz"))
    result: dict[int, Path] = {}
    for seed in seeds:
        matches = [path for path in files if re.search(rf"(?:seed)?{seed}(?:\D|$)", path.name)]
        if len(matches) != 1:
            raise RuntimeError(f"expected one logits file for seed {seed}, split {split}; got {matches}")
        result[seed] = matches[0]
    return result


def build_bundle(stage: Path, split: str, role: str, seeds: list[int]) -> dict[str, np.ndarray]:
    records, _ = load_labeled_records(
        stage / "visible" / f"{split}.jsonl",
        stage / "labels" / f"{split}.jsonl",
        stage / "protocol_config.json",
        split,
    )
    predictions: dict[int, dict[str, np.ndarray]] = {}
    for seed, path in _logit_files(stage, split, seeds).items():
        with np.load(path, allow_pickle=False) as data:
            fixture_ids = data["fixture_id"].astype(str)
            logits_key = "set_logits" if "set_logits" in data.files else "logits"
            logits = np.asarray(data[logits_key], dtype=np.float64)
        if len(set(fixture_ids)) != len(fixture_ids):
            raise RuntimeError(f"duplicate fixture inference in {path.name}")
        predictions[seed] = dict(zip(fixture_ids, logits, strict=True))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["pair_token"])].append(record)
    tokens = sorted(grouped)
    targets: list[np.ndarray] = []
    strata: list[str] = []
    cardinalities: list[int] = []
    per_seed = np.empty((len(seeds), len(tokens), 4), dtype=np.float64)
    for token_index, token in enumerate(tokens):
        views = grouped[token]
        target_values = {tuple(np.asarray(view["target"], dtype=bool)) for view in views}
        stratum_values = {str(view["design_stratum"]) for view in views}
        if len(target_values) != 1 or len(stratum_values) != 1:
            raise RuntimeError(f"canonical views disagree for {token}")
        target = np.asarray(next(iter(target_values)), dtype=bool)
        targets.append(target)
        strata.append(next(iter(stratum_values)))
        cardinalities.append(int(target.sum()))
        for seed_index, seed in enumerate(seeds):
            try:
                per_seed[seed_index, token_index] = np.mean(
                    [predictions[seed][str(view["fixture_id"])] for view in views], axis=0
                )
            except KeyError as error:
                raise RuntimeError(f"missing inference for fixture {error.args[0]}") from error
    return {
        "pair_token": np.asarray(tokens),
        "cluster_id": np.asarray(tokens),
        "target": np.stack(targets),
        "per_seed_logits": per_seed,
        "ensemble_logits": per_seed.mean(axis=0),
        "design_stratum": np.asarray(strata),
        "cardinality": np.asarray(cardinalities, dtype=np.int64),
        "split_role": np.asarray([role] * len(tokens)),
    }


def _pair_token_source(tokens: Any, name: str) -> tuple[set[str], dict[str, Any]]:
    values = np.asarray(tokens).astype(str)
    if values.ndim != 1:
        raise RuntimeError(f"{name} pair_tokens must be one-dimensional")
    token_list = values.tolist()
    unique = set(token_list)
    if len(unique) != len(token_list):
        raise RuntimeError(f"{name} contains duplicate pair_tokens")
    ordered = sorted(unique)
    encoded = json.dumps(ordered, separators=(",", ":"), ensure_ascii=True).encode()
    return unique, {
        "count": len(ordered),
        "pair_token_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def split_disjointness_evidence(
    stage: Path, phase: str, current_tokens: np.ndarray
) -> dict[str, Any]:
    historical_payload = load_json(stage / "frozen/historical_pair_tokens.json")
    historical, historical_receipt = _pair_token_source(
        historical_payload.get("pair_tokens", []), "wave54_55_historical"
    )
    sources: dict[str, set[str]] = {"wave54_55_historical": historical}
    receipts: dict[str, dict[str, Any]] = {
        "wave54_55_historical": historical_receipt
    }
    previous_specs = []
    if phase in {"select", "adjudicate"}:
        previous_specs.append(("gate_fit", "fit_arrays.npz", "gate_fit__pair_token"))
    if phase == "adjudicate":
        previous_specs.append(
            ("gate_select", "selection_arrays.npz", "gate_select__pair_token")
        )
    for role, filename, key in previous_specs:
        with np.load(stage / "previous" / filename, allow_pickle=False) as previous:
            if key not in previous.files:
                raise RuntimeError(f"{filename} omitted preserved pair_tokens: {key}")
            tokens, receipt = _pair_token_source(previous[key], role)
        sources[role] = tokens
        receipts[role] = receipt

    current_role = PHASE_TO_ROLE[phase]
    current, current_receipt = _pair_token_source(current_tokens, current_role)
    sources[current_role] = current
    receipts[current_role] = current_receipt
    pairwise = []
    for left, right in itertools.combinations(sources, 2):
        overlap = sources[left] & sources[right]
        if overlap:
            raise RuntimeError(
                f"pair_token overlap between {left} and {right}: {sorted(overlap)[:5]}"
            )
        pairwise.append(
            {"left": left, "right": right, "overlap_count": 0, "disjoint": True}
        )
    return {
        "status": "PASS",
        "all_disjoint": True,
        "sources": receipts,
        "pairwise_checks": pairwise,
    }


def load_inputs(stage: Path, phase: str) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
    config = json.loads((stage / "config.json").read_text(encoding="utf-8"))
    validate_frozen_config(config)
    seeds = [int(seed) for seed in config.get("seeds", (17, 29, 43))]
    utilities, _ = retrospective.load_utilities(stage / "frozen/policy_manifest.json")
    wave54_freeze = load_json(stage / "frozen/wave54_selection_freeze.json")
    theta = np.asarray(
        wave54_freeze["selected_models"]["joint_full"]["theta"], dtype=np.float64
    )
    split = PHASE_TO_SPLIT[phase]
    role = PHASE_TO_ROLE[phase]
    bundle = build_bundle(stage, split, role, seeds)
    disjointness = split_disjointness_evidence(stage, phase, bundle["pair_token"])
    analysis_config = dict(config)
    analysis_config["primary_population"] = {
        "design_stratum": "NEAR_RIVAL",
        "minimum_true_cardinality": 2,
    }
    data = retrospective.make_dataset(bundle, theta, utilities, analysis_config)
    posterior_mass = retrospective.posterior_mass(
        np.asarray(bundle["ensemble_logits"], dtype=np.float64), theta, "joint_full"
    )
    posterior_decision = retrospective.expected_regret_from_mass(
        posterior_mass, utilities, float(config["incompatible_regret_penalty"])
    )
    np.testing.assert_array_equal(posterior_decision["actions"], data["posterior_actions"])
    data["posterior_mass"] = posterior_mass
    data["action_risk"] = posterior_decision["action_risk"]
    absent_set_indices = np.asarray(
        config["absent_support"]["set_indices"], dtype=np.int64
    )
    data["absent_support"] = np.isin(
        target_set_indices(data["target"]), absent_set_indices
    )
    data["_split_disjointness"] = disjointness
    return config, utilities, data


def minimum_counts(data: dict[str, np.ndarray]) -> dict[str, int]:
    primary = np.asarray(data["primary"], dtype=bool)
    return {
        "tokens": int(primary.sum()),
        "disagreement_rows": int((primary[:, None] & data["disagreement"]).sum()),
    }


def _split_evidence(data: dict[str, Any]) -> dict[str, Any]:
    evidence = data.get("_split_disjointness")
    if isinstance(evidence, dict):
        return evidence
    return {
        "status": "NOT_ATTESTED_DIRECT_CALL",
        "all_disjoint": None,
        "sources": {},
        "pairwise_checks": [],
    }


def _minimum(config: dict[str, Any], phase: str, name: str, default: int) -> int:
    role = PHASE_TO_ROLE[phase]
    return int(
        _nested(
            config,
            ("minimums", role, name),
            ("minimums", phase, name),
            ("minimums", f"{role}_{name}"),
            ("minimums", f"{phase}_{name}"),
            default=default,
        )
    )


def phase_minimum_failure(data: dict[str, np.ndarray], config: dict[str, Any], phase: str) -> dict[str, Any] | None:
    counts = minimum_counts(data)
    defaults = {
        "fit": {"tokens": 100, "disagreement_rows": 400},
        "select": {"tokens": 80, "disagreement_rows": 300},
        "adjudicate": {"tokens": 100, "disagreement_rows": 0},
    }[phase]
    required = {name: _minimum(config, phase, name, value) for name, value in defaults.items()}
    failed = [name for name, value in required.items() if counts[name] < value]
    if failed:
        token_payload = json.dumps(
            data["pair_token"].astype(str).tolist(), separators=(",", ":"), ensure_ascii=True
        ).encode()
        return {
            "status": "NOT_EVALUABLE",
            "phase": phase,
            "counts": counts,
            "required": required,
            "failed": failed,
            "reason": "frozen prospective minimums were not met; redraw is forbidden",
            "pair_token_sha256": hashlib.sha256(token_payload).hexdigest(),
            "split_disjointness": _split_evidence(data),
        }
    return None


def _model_spec(columns: str | list[str]) -> dict[str, Any]:
    return {"kind": "ridge", "objective": "mse", "columns": columns}


def fit_ridge(data: dict[str, np.ndarray], alpha: float, columns: str | list[str], target: np.ndarray) -> dict[str, Any]:
    spec = _model_spec(columns)
    selected = retrospective.selected_columns(spec)
    rows = retrospective.fit_rows(data, data["primary"])
    state = retrospective.fit_model(
        spec,
        {"alpha": float(alpha)},
        data["design"][rows][:, selected],
        target[rows],
        data["weights"][rows],
    )
    return retrospective.model_state_json(state)


def score_model(model: dict[str, Any], data: dict[str, np.ndarray], columns: str | list[str]) -> np.ndarray:
    spec = _model_spec(columns)
    selected = retrospective.selected_columns(spec)
    scores = np.full(data["disagreement"].shape, np.nan, dtype=np.float64)
    rows = np.asarray(data["disagreement"], dtype=bool)
    x = np.asarray(data["design"][rows][:, selected], dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coef = np.asarray(model["coef"], dtype=np.float64)
    intercept = float(np.asarray(model["intercept"], dtype=np.float64))
    scores[rows] = ((x - mean) / scale) @ coef + intercept
    return scores


def data_arrays(role: str, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    keys = (
        "pair_token", "cluster_id", "target", "per_seed_logits", "ensemble_logits",
        "design_stratum", "cardinality", "split_role", "design", "gain", "disagreement",
        "weights", "primary", "hard_actions", "posterior_actions", "hard_set", "advantage",
        "posterior_mass", "action_risk", "absent_support",
    )
    return {f"{role}__{key}": np.asarray(data[key]) for key in keys if key in data}


def model_arrays(prefix: str, model: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        f"{prefix}__mean": np.asarray(model["mean"], dtype=np.float64),
        f"{prefix}__scale": np.asarray(model["scale"], dtype=np.float64),
        f"{prefix}__coef": np.asarray(model["coef"], dtype=np.float64),
        f"{prefix}__intercept": np.asarray(model["intercept"], dtype=np.float64),
    }


def _write_freeze(output: Path, name: str, phase: str, files: tuple[str, ...], extra: dict[str, Any]) -> None:
    write_json_atomic(
        output / name,
        {
            "phase": phase,
            "files": {filename: sha256_file(output / filename) for filename in files},
            **extra,
        },
    )


def run_fit(stage: Path, output: Path, config: dict[str, Any], utilities: np.ndarray, data: dict[str, np.ndarray]) -> str:
    failure = phase_minimum_failure(data, config, "fit")
    if failure:
        write_json_atomic(output / "fit_not_evaluable.json", failure)
        return "FIT_NOT_EVALUABLE"

    contextual = fit_ridge(data, 1.0, "all", data["gain"])
    advantage = fit_ridge(data, 100.0, ["advantage"], data["gain"])
    arrays = data_arrays("gate_fit", data)
    arrays.update(model_arrays("model__ridge_contextual", contextual))
    arrays.update(model_arrays("model__ridge_advantage_only", advantage))
    shuffles = []
    shuffle_mappings = []
    shuffle_targets = []
    primary = np.asarray(data["primary"], dtype=bool)
    movable_minimum = float(_nested(config, ("minimums", "shuffle_movable_fraction"), default=0.8))
    for shuffle_index, seed in enumerate(SHUFFLE_SEEDS):
        shuffled = stratified_gain_shuffle(
            data["gain"][primary], data["disagreement"][primary], seed
        )
        target = np.asarray(data["gain"], dtype=np.float64).copy()
        target[primary] = shuffled["target"]
        model = fit_ridge(data, 1.0, "all", target)
        status = "PASS" if float(shuffled["movable_fraction"] or 0.0) >= movable_minimum else "NOT_EVALUABLE"
        shuffles.append({
            "shuffle_id": shuffle_index,
            "seed": seed,
            "status": status,
            "movable_fraction": shuffled["movable_fraction"],
            "moved_fraction": shuffled["moved_fraction"],
            "model": model,
        })
        arrays[f"shuffle__mapping__{shuffle_index}"] = shuffled["mapping"]
        arrays[f"shuffle__target__{shuffle_index}"] = target
        shuffle_mappings.append(shuffled["mapping"])
        shuffle_targets.append(target)
        arrays.update(model_arrays(f"model__shuffle__{shuffle_index}", model))
    arrays["shuffle_id"] = np.arange(len(SHUFFLE_SEEDS), dtype=np.int64)
    arrays["shuffle_mapping"] = np.stack(shuffle_mappings)
    arrays["shuffle_target"] = np.stack(shuffle_targets)

    write_npz_atomic(output / "gate_fit_bundle.npz", {key: value for key, value in data.items() if isinstance(value, np.ndarray)})
    write_npz_atomic(output / "fit_arrays.npz", arrays)
    write_json_atomic(output / "feature_schema.json", {"features": retrospective.feature_schema()})
    core = {
        "status": "FIT_COMPLETE",
        "counts": minimum_counts(data),
        "models": {
            "ridge_contextual": {"kind": "ridge", "alpha": 1.0, "columns": list(FEATURE_NAMES), "model": contextual},
            "ridge_advantage_only": {"kind": "ridge", "alpha": 100.0, "columns": ["advantage"], "model": advantage},
        },
        "shuffles": shuffles,
        "fit_population": "NEAR_RIVAL and true cardinality >= 2; disagreement rows only; token weight 1/d_t",
        "split_disjointness": _split_evidence(data),
    }
    write_json_atomic(output / "fit_core.json", core)
    _write_freeze(
        output,
        "fit_freeze.json",
        "fit-complete-before-gate-select-label-access",
        ("fit_core.json", "fit_arrays.npz", "feature_schema.json", "gate_fit_bundle.npz"),
        {
            "contextual_alpha": 1.0,
            "advantage_only_alpha": 100.0,
            "shuffle_seeds": list(SHUFFLE_SEEDS),
            "split_disjointness": _split_evidence(data),
            "provenance": load_json(stage / "phase_request.json"),
        },
    )
    return "FIT_COMPLETE"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _threshold_selection(
    scores: np.ndarray,
    data: dict[str, np.ndarray],
    utilities: np.ndarray,
    config: dict[str, Any],
    token_mask: np.ndarray,
) -> dict[str, Any]:
    return retrospective.select_operating_point(
        scores, data, utilities, config, token_mask=np.asarray(token_mask, dtype=bool)
    )


def _scalar_selection(data: dict[str, np.ndarray], utilities: np.ndarray, config: dict[str, Any], token_mask: np.ndarray) -> dict[str, Any]:
    hard_metrics = retrospective.metric_arrays_for_policies(
        data["hard_actions"], data, utilities, float(config["incompatible_regret_penalty"])
    )
    hard_summary = retrospective.summarize(hard_metrics, token_mask)
    rows = []
    for gamma in SCALAR_GRID:
        gated = apply_gate(data["advantage"], data["hard_actions"], data["posterior_actions"], gamma)
        metrics = action_metric_arrays(
            gated["actions"], data["target"], utilities, float(config["incompatible_regret_penalty"])
        )
        summary = retrospective.summarize(metrics, token_mask)
        feasible = (
            summary["accuracy"] >= hard_summary["accuracy"] - float(config["selection"]["accuracy_noninferiority_margin"])
            and summary["compatible"] >= hard_summary["compatible"] - float(config["selection"]["compatible_noninferiority_margin"])
        )
        rows.append({
            "gamma": gamma,
            "threshold": gamma,
            "coverage": float(gated["override"][token_mask].mean()),
            "feasible": bool(feasible),
            **summary,
        })
    feasible_rows = [row for row in rows if row["feasible"]]
    atol = float(config["selection"]["tie_atol"])
    best = min(float(row["regret"]) for row in feasible_rows)
    tied = [row for row in feasible_rows if float(row["regret"]) <= best + atol]
    selected = min(tied, key=lambda row: float(row["coverage"]))
    return {"selected": selected, "grid": rows}


def _threshold_arrays(threshold: float | str) -> tuple[np.ndarray, np.ndarray]:
    hard_only = threshold == HARD_ONLY
    return np.asarray(np.nan if hard_only else float(threshold)), np.asarray(hard_only)


def _store_selection_arm(
    arrays: dict[str, np.ndarray],
    name: str,
    scores: np.ndarray,
    selected: dict[str, Any],
    data: dict[str, np.ndarray],
    utilities: np.ndarray,
    config: dict[str, Any],
) -> None:
    gated = apply_gate(scores, data["hard_actions"], data["posterior_actions"], selected["threshold"])
    metrics = action_metric_arrays(
        gated["actions"], data["target"], utilities, float(config["incompatible_regret_penalty"])
    )
    threshold, hard_only = _threshold_arrays(selected["threshold"])
    arrays[f"selection__{name}__score"] = scores
    arrays[f"selection__{name}__threshold"] = threshold
    arrays[f"selection__{name}__hard_only"] = hard_only
    arrays[f"selection__{name}__actions"] = gated["actions"]
    arrays[f"selection__{name}__overrides"] = gated["override"]
    for metric, values in metrics.items():
        arrays[f"selection__{name}__metric__{metric}"] = values


def shard_assignment(tokens: np.ndarray) -> np.ndarray:
    return np.asarray([
        hashlib.sha256((str(token) + "wave56-shard").encode()).digest()[-1] & 1
        for token in tokens.astype(str)
    ], dtype=np.int8)


def _selection_minimum_failure(data: dict[str, np.ndarray], config: dict[str, Any]) -> dict[str, Any] | None:
    failure = phase_minimum_failure(data, config, "select")
    assignment = shard_assignment(data["pair_token"])
    shard_counts = {}
    failed_shards = []
    for shard in (0, 1):
        local = np.asarray(data["primary"], dtype=bool) & (assignment == shard)
        counts = {
            "tokens": int(local.sum()),
            "disagreement_rows": int((local[:, None] & data["disagreement"]).sum()),
        }
        required = {
            "tokens": int(
                _nested(config, ("minimums", "gate_select_shard_tokens"), default=40)
            ),
            "disagreement_rows": int(
                _nested(
                    config,
                    ("minimums", "gate_select_shard_disagreement_rows"),
                    default=120,
                )
            ),
        }
        shard_counts[str(shard)] = {"counts": counts, "required": required}
        if any(counts[key] < required[key] for key in required):
            failed_shards.append(shard)
    if failure or failed_shards:
        return {
            "status": "NOT_EVALUABLE",
            "phase": "select",
            "global": failure,
            "shards": shard_counts,
            "failed_shards": failed_shards,
            "split_disjointness": _split_evidence(data),
        }
    return None


def run_select(stage: Path, output: Path, config: dict[str, Any], utilities: np.ndarray, data: dict[str, np.ndarray]) -> str:
    fit_core = load_json(stage / "previous/fit_core.json")
    failure = _selection_minimum_failure(data, config)
    if failure:
        write_json_atomic(output / "selection_not_evaluable.json", failure)
        return "SELECT_NOT_EVALUABLE"

    models = fit_core["models"]
    contextual_scores = score_model(models["ridge_contextual"]["model"], data, "all")
    advantage_scores = score_model(models["ridge_advantage_only"]["model"], data, ["advantage"])
    primary = np.asarray(data["primary"], dtype=bool)
    contextual = _threshold_selection(contextual_scores, data, utilities, config, primary)
    advantage = _threshold_selection(advantage_scores, data, utilities, config, primary)
    scalar = _scalar_selection(data, utilities, config, primary)
    all_tokens = np.ones(len(primary), dtype=bool)
    all_in = _threshold_selection(contextual_scores, data, utilities, config, all_tokens)
    assignment = shard_assignment(data["pair_token"])
    shards = {
        str(shard): _threshold_selection(
            contextual_scores, data, utilities, config, primary & (assignment == shard)
        )
        for shard in (0, 1)
    }

    arrays = data_arrays("gate_select", data)
    arrays["score__ridge_contextual"] = contextual_scores
    arrays["score__ridge_advantage_only"] = advantage_scores
    arrays["shard_assignment"] = assignment
    _store_selection_arm(
        arrays, "contextual", contextual_scores, contextual["selected"], data, utilities, config
    )
    _store_selection_arm(
        arrays, "advantage_only", advantage_scores, advantage["selected"], data, utilities, config
    )
    _store_selection_arm(
        arrays, "scalar_advantage", data["advantage"], scalar["selected"], data, utilities, config
    )
    shuffle_rows = []
    shuffle_scores = []
    shuffle_thresholds = []
    shuffle_hard_only = []
    shuffle_actions = []
    shuffle_overrides = []
    for row in fit_core["shuffles"]:
        shuffle_id = int(row["shuffle_id"])
        scores = score_model(row["model"], data, "all")
        selection = _threshold_selection(scores, data, utilities, config, primary)
        arrays[f"shuffle__score__{shuffle_id}"] = scores
        gated = apply_gate(
            scores, data["hard_actions"], data["posterior_actions"], selection["selected"]["threshold"]
        )
        threshold, hard_only = _threshold_arrays(selection["selected"]["threshold"])
        shuffle_scores.append(scores)
        shuffle_thresholds.append(threshold)
        shuffle_hard_only.append(hard_only)
        shuffle_actions.append(gated["actions"])
        shuffle_overrides.append(gated["override"])
        shuffle_rows.append({
            "shuffle_id": shuffle_id,
            "seed": int(row["seed"]),
            "status": row["status"],
            "selection": selection,
        })
    arrays["shuffle_id"] = np.arange(len(shuffle_rows), dtype=np.int64)
    arrays["shuffle_score"] = np.stack(shuffle_scores)
    arrays["shuffle_threshold"] = np.stack(shuffle_thresholds)
    arrays["shuffle_hard_only"] = np.stack(shuffle_hard_only)
    arrays["shuffle_actions"] = np.stack(shuffle_actions)
    arrays["shuffle_overrides"] = np.stack(shuffle_overrides)

    write_npz_atomic(output / "gate_select_bundle.npz", {key: value for key, value in data.items() if isinstance(value, np.ndarray)})
    write_npz_atomic(output / "selection_arrays.npz", arrays)
    core = {
        "status": "SELECT_COMPLETE",
        "counts": minimum_counts(data),
        "contextual": contextual,
        "advantage_only": advantage,
        "scalar_advantage": scalar,
        "shuffles": shuffle_rows,
        "shards": shards,
        "all_in_catalog": all_in,
        "quantile_method": "higher",
        "strict_override": True,
        "split_disjointness": _split_evidence(data),
    }
    write_json_atomic(output / "selection_core.json", core)
    freeze = {
        "contextual": contextual["selected"],
        "advantage_only": advantage["selected"],
        "scalar_advantage": scalar["selected"],
        "shuffles": [
            {
                "shuffle_id": row["shuffle_id"],
                "seed": row["seed"],
                "status": row["status"],
                **row["selection"]["selected"],
            }
            for row in shuffle_rows
        ],
        "shards": {key: value["selected"] for key, value in shards.items()},
        "all_in_catalog": all_in["selected"],
    }
    _write_freeze(
        output,
        "selection_freeze.json",
        "selection-complete-before-sealed-monitor-label-access",
        ("selection_core.json", "selection_arrays.npz", "gate_select_bundle.npz"),
        {
            "selected": freeze,
            "split_disjointness": _split_evidence(data),
            "provenance": load_json(stage / "phase_request.json"),
        },
    )
    return "SELECT_COMPLETE"


def _arm(
    actions: np.ndarray,
    override: np.ndarray,
    data: dict[str, np.ndarray],
    utilities: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    metrics = action_metric_arrays(
        actions, data["target"], utilities, float(config["incompatible_regret_penalty"])
    )
    metrics["accuracy_by_policy"] = actions == authorized_actions(data["target"], utilities)
    return {
        "actions": np.asarray(actions),
        "override": np.asarray(override, dtype=bool),
        "metrics": metrics,
        "summary": retrospective.summarize(metrics, data["primary"]),
        "override_diagnostics": retrospective.override_summary(actions, override, data, utilities, config),
    }


def _gated_arm(scores: np.ndarray, threshold: float | str, data: dict[str, np.ndarray], utilities: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    gated = apply_gate(scores, data["hard_actions"], data["posterior_actions"], threshold)
    return _arm(gated["actions"], gated["override"], data, utilities, config)


def _store_arm(arrays: dict[str, np.ndarray], name: str, arm: dict[str, Any]) -> None:
    arrays[f"arm__{name}__actions"] = arm["actions"]
    arrays[f"arm__{name}__overrides"] = arm["override"]
    for metric, values in arm["metrics"].items():
        arrays[f"arm__{name}__metric__{metric}"] = np.asarray(values)


def _sign(value: float, atol: float) -> int:
    return 0 if abs(value) <= atol else int(np.sign(value))


def _contrast_signs(contextual: dict[str, Any], references: dict[str, dict[str, Any]], atol: float) -> dict[str, int]:
    result: dict[str, int] = {}
    for reference_name, reference in references.items():
        metrics = ("regret", "accuracy", "compatible")
        if reference_name in {"advantage_only", "shuffle_average"}:
            metrics = ("regret",)
        elif reference_name == "pure_joint":
            metrics = ("regret", "accuracy")
        for metric in metrics:
            result[f"{metric}_vs_{reference_name}"] = _sign(
                float(contextual["summary"][metric] - reference["summary"][metric]), atol
            )
    return result


def _ci(left: dict[str, Any], right: dict[str, Any], metric: str, primary_indices: np.ndarray, bootstrap: np.ndarray) -> dict[str, float]:
    return retrospective.paired_delta_ci(
        left["metrics"][metric][primary_indices], right["metrics"][metric][primary_indices], bootstrap
    )


def _diagnostic_pattern(
    contrasts: dict[str, dict[str, dict[str, float]]],
    shuffle_evaluable: bool,
    selector_sensitive: bool,
    config: dict[str, Any],
) -> dict[str, Any]:
    hard = contrasts["contextual_minus_hard_set_policy"]
    scalar = contrasts["contextual_minus_scalar_advantage_gate"]
    advantage = contrasts["contextual_minus_advantage_only_value_gate"]
    pure = contrasts["contextual_minus_pure_joint_full"]
    criteria = config["diagnostic_criteria"]
    conditions: dict[str, Any] = {
        "diagnostic_condition_1": bool(
            hard["regret"]["mean_diff"]
            <= -float(criteria["regret_reduction_vs_hard_min"])
            and hard["regret"]["ci95_high"]
            < float(criteria["regret_vs_hard_ci95_upper_below"])
        ),
        "diagnostic_condition_2": bool(
            hard["accuracy"]["ci95_low"]
            >= float(criteria["accuracy_vs_hard_ci95_lower_at_least"])
            and hard["compatible"]["ci95_low"]
            >= float(criteria["compatibility_vs_hard_ci95_lower_at_least"])
        ),
        "diagnostic_condition_3": bool(
            scalar["regret"]["mean_diff"]
            <= -float(criteria["regret_reduction_vs_scalar_min"])
            and scalar["regret"]["ci95_high"]
            < float(criteria["regret_vs_scalar_ci95_upper_below"])
            and advantage["regret"]["mean_diff"]
            <= -float(criteria["regret_reduction_vs_advantage_only_min"])
            and advantage["regret"]["ci95_high"]
            < float(criteria["regret_vs_advantage_only_ci95_upper_below"])
        ),
        "diagnostic_condition_5": bool(
            pure["accuracy"]["ci95_low"]
            > float(criteria["accuracy_vs_pure_joint_ci95_lower_above"])
            and pure["regret"]["ci95_high"]
            <= float(criteria["regret_vs_pure_joint_ci95_upper_at_most"])
        ),
        "diagnostic_condition_6_without_replay": bool(
            selector_sensitive == bool(criteria["selector_sensitive_required"])
        ),
    }
    if shuffle_evaluable:
        shuffled = contrasts["contextual_minus_contextual_shuffled_gain"]
        conditions["diagnostic_condition_4"] = bool(
            shuffled["regret"]["mean_diff"]
            <= -float(criteria["regret_reduction_vs_shuffled_min"])
            and shuffled["regret"]["ci95_high"]
            < float(criteria["regret_vs_shuffled_ci95_upper_below"])
        )
    else:
        conditions["diagnostic_condition_4"] = "NOT_EVALUABLE"
    booleans = [value for value in conditions.values() if isinstance(value, bool)]
    return {
        "conditions": conditions,
        "all_observed_conditions_without_replay": bool(all(booleans)) if len(booleans) == 6 else None,
        "aggregate_with_replay": None,
        "aggregate_status": "PENDING_EXACT_REPLAY" if len(booleans) == 6 and all(booleans) else "NOT_SATISFIED_OR_NOT_EVALUABLE",
        "decision_authority": "user",
    }


def _diagnostic_curve(
    scores: np.ndarray,
    grid: list[dict[str, Any]],
    data: dict[str, np.ndarray],
    utilities: np.ndarray,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    curve = []
    for row in grid:
        arm = _gated_arm(scores, row["threshold"], data, utilities, config)
        curve.append({
            "threshold": row["threshold"],
            "coverage": float(arm["override"][data["primary"]].mean()),
            **arm["summary"],
        })
    return curve


def run_adjudicate(stage: Path, output: Path, config: dict[str, Any], utilities: np.ndarray, data: dict[str, np.ndarray]) -> str:
    failure = phase_minimum_failure(data, config, "adjudicate")
    if failure:
        write_json_atomic(output / "monitor_not_evaluable.json", failure)
        return "MONITOR_NOT_EVALUABLE"
    fit_core = load_json(stage / "previous/fit_core.json")
    selection = load_json(stage / "previous/selection_freeze.json")["selected"]
    selection_core = load_json(stage / "previous/selection_core.json")
    models = fit_core["models"]
    contextual_scores = score_model(models["ridge_contextual"]["model"], data, "all")
    advantage_scores = score_model(models["ridge_advantage_only"]["model"], data, ["advantage"])

    zero_override = np.zeros(data["hard_actions"].shape, dtype=bool)
    hard = _arm(data["hard_actions"], zero_override, data, utilities, config)
    pure_override = np.asarray(data["disagreement"], dtype=bool)
    pure = _arm(data["posterior_actions"], pure_override, data, utilities, config)
    scalar = _gated_arm(data["advantage"], selection["scalar_advantage"]["threshold"], data, utilities, config)
    contextual = _gated_arm(contextual_scores, selection["contextual"]["threshold"], data, utilities, config)
    advantage = _gated_arm(advantage_scores, selection["advantage_only"]["threshold"], data, utilities, config)
    oracle_override = data["disagreement"] & (data["gain"] > 1e-12)
    oracle = _arm(
        np.where(oracle_override, data["posterior_actions"], data["hard_actions"]),
        oracle_override,
        data,
        utilities,
        config,
    )
    arms = {
        "hard_set_policy": hard,
        "pure_joint_full": pure,
        "scalar_advantage_gate": scalar,
        "contextual_value_gate": contextual,
        "advantage_only_value_gate": advantage,
        "oracle_positive_gain": oracle,
    }

    arrays = data_arrays("sealed_monitor", data)
    for source_name, prefix in (
        ("fit_arrays.npz", "gate_fit_archive"),
        ("selection_arrays.npz", "gate_select_archive"),
    ):
        with np.load(stage / "previous" / source_name, allow_pickle=False) as previous:
            for key in previous.files:
                arrays[f"{prefix}__{key}"] = previous[key]
    arrays["score__ridge_contextual"] = contextual_scores
    arrays["score__ridge_advantage_only"] = advantage_scores
    for name, arm in arms.items():
        _store_arm(arrays, name, arm)

    shuffle_arms = []
    shuffle_statuses = []
    for fit_row, selected_row in zip(fit_core["shuffles"], selection["shuffles"], strict=True):
        shuffle_id = int(fit_row["shuffle_id"])
        scores = score_model(fit_row["model"], data, "all")
        arm = _gated_arm(scores, selected_row["threshold"], data, utilities, config)
        shuffle_arms.append(arm)
        shuffle_statuses.append(str(selected_row["status"]))
        arrays[f"shuffle__score__{shuffle_id}"] = scores
        threshold, hard_only = _threshold_arrays(selected_row["threshold"])
        arrays[f"shuffle__threshold__{shuffle_id}"] = threshold
        arrays[f"shuffle__hard_only__{shuffle_id}"] = hard_only
        arrays[f"shuffle__actions__{shuffle_id}"] = arm["actions"]
        arrays[f"shuffle__overrides__{shuffle_id}"] = arm["override"]
        for metric, values in arm["metrics"].items():
            arrays[f"shuffle__metric__{metric}__{shuffle_id}"] = values
    average_metrics = {
        metric: np.mean(np.stack([arm["metrics"][metric] for arm in shuffle_arms]), axis=0)
        for metric in shuffle_arms[0]["metrics"]
    }
    shuffle_average = {
        "metrics": average_metrics,
        "summary": retrospective.summarize(average_metrics, data["primary"]),
    }
    for metric, values in average_metrics.items():
        arrays[f"shuffle__average_metric__{metric}"] = values
    arrays["shuffle_id"] = np.arange(len(shuffle_arms), dtype=np.int64)
    arrays["shuffle_score"] = np.stack(
        [arrays[f"shuffle__score__{index}"] for index in range(len(shuffle_arms))]
    )
    arrays["shuffle_threshold"] = np.stack(
        [arrays[f"shuffle__threshold__{index}"] for index in range(len(shuffle_arms))]
    )
    arrays["shuffle_hard_only"] = np.stack(
        [arrays[f"shuffle__hard_only__{index}"] for index in range(len(shuffle_arms))]
    )
    arrays["shuffle_actions"] = np.stack(
        [arrays[f"shuffle__actions__{index}"] for index in range(len(shuffle_arms))]
    )
    arrays["shuffle_overrides"] = np.stack(
        [arrays[f"shuffle__overrides__{index}"] for index in range(len(shuffle_arms))]
    )
    for metric in shuffle_arms[0]["metrics"]:
        arrays[f"shuffle_metric__{metric}"] = np.stack(
            [arrays[f"shuffle__metric__{metric}__{index}"] for index in range(len(shuffle_arms))]
        )

    primary_indices = np.flatnonzero(data["primary"])
    primary_tokens = data["pair_token"][primary_indices].astype(str)
    if primary_tokens.tolist() != sorted(primary_tokens.tolist()):
        raise RuntimeError("bootstrap population is not in canonical pair_token order")
    bootstrap = retrospective.paired_bootstrap_indices(len(primary_indices), config)
    arrays["bootstrap_indices"] = bootstrap
    references = {
        "hard_set_policy": hard,
        "scalar_advantage_gate": scalar,
        "advantage_only_value_gate": advantage,
        "contextual_shuffled_gain": shuffle_average,
        "pure_joint_full": pure,
    }
    contrasts = {
        f"contextual_minus_{name}": {
            metric: _ci(contextual, reference, metric, primary_indices, bootstrap)
            for metric in ("accuracy", "compatible", "regret", "worst_regret")
        }
        for name, reference in references.items()
    }

    atol = float(config["selection"]["tie_atol"])
    full_signs = _contrast_signs(
        contextual,
        {
            "hard": hard,
            "scalar": scalar,
            "advantage_only": advantage,
            "shuffle_average": shuffle_average,
            "pure_joint": pure,
        },
        atol,
    )
    shard_results = {}
    selector_sensitive = False
    for shard, selected in sorted(selection["shards"].items()):
        shard_arm = _gated_arm(contextual_scores, selected["threshold"], data, utilities, config)
        signs = _contrast_signs(
            shard_arm,
            {
                "hard": hard,
                "scalar": scalar,
                "advantage_only": advantage,
                "shuffle_average": shuffle_average,
                "pure_joint": pure,
            },
            atol,
        )
        different = signs != full_signs
        selector_sensitive |= different
        shard_results[shard] = {"selection": selected, "monitor_signs": signs, "differs_from_full": different}
        _store_arm(arrays, f"contextual_shard_{shard}", shard_arm)

    support_minimum = int(config["minimums"]["absent_support_tokens"])
    absent_set_indices = [int(value) for value in config["absent_support"]["set_indices"]]
    support_mask = np.asarray(data["absent_support"], dtype=bool)
    support_tokens = int(support_mask.sum())
    support = {
        "status": "EVALUABLE" if support_tokens >= support_minimum else "NOT_EVALUABLE",
        "tokens": support_tokens,
        "required": support_minimum,
        "absent_set_indices": absent_set_indices,
    }
    if support["status"] == "EVALUABLE":
        support_indices = np.flatnonzero(support_mask)
        support_bootstrap = retrospective.paired_bootstrap_indices(len(support_indices), config)
        arrays["absent_support_bootstrap_indices"] = support_bootstrap
        support["summaries"] = {
            name: retrospective.summarize(arm["metrics"], support_mask)
            for name, arm in arms.items()
        }
        support["contrasts"] = {
            f"contextual_minus_{name}": {
                metric: retrospective.paired_delta_ci(
                    contextual["metrics"][metric][support_indices],
                    reference["metrics"][metric][support_indices],
                    support_bootstrap,
                )
                for metric in ("accuracy", "compatible", "regret", "worst_regret")
            }
            for name, reference in references.items()
        }
    else:
        support["summaries"] = None
        support["contrasts"] = None
    shuffle_evaluable = all(status == "PASS" for status in shuffle_statuses)
    diagnostic = _diagnostic_pattern(
        contrasts, shuffle_evaluable, selector_sensitive, config
    )
    arm_summaries = {
        name: {
            "status": "PASS",
            "summary": arm["summary"],
            "override_diagnostics": arm["override_diagnostics"],
        }
        for name, arm in arms.items()
    }
    arm_summaries["contextual_shuffled_gain"] = {
        "status": "PASS" if shuffle_evaluable else "NOT_EVALUABLE",
        "summary": shuffle_average["summary"],
        "replicate_statuses": shuffle_statuses,
    }
    core = {
        "status": "COMPLETE",
        "counts": minimum_counts(data),
        "arms": arm_summaries,
        "contextual_shuffled_gain": {
            "status": "PASS" if shuffle_evaluable else "NOT_EVALUABLE",
            "replicate_statuses": shuffle_statuses,
            "summary": shuffle_average["summary"],
        },
        "contrasts": contrasts,
        "selector_stability": {
            "full_monitor_signs": full_signs,
            "shards": shard_results,
            "selector_sensitive": selector_sensitive,
        },
        "monitor_diagnostic_curves": {
            "contextual": _diagnostic_curve(
                contextual_scores,
                selection_core["contextual"]["grid"],
                data,
                utilities,
                config,
            ),
            "scalar_advantage": _diagnostic_curve(
                data["advantage"],
                selection_core["scalar_advantage"]["grid"],
                data,
                utilities,
                config,
            ),
        },
        "contextual_correlation": retrospective.correlation_summary(contextual_scores, data),
        "contextual_sensitivities": retrospective.system_sensitivities(
            contextual["actions"], contextual["metrics"], data, utilities
        ),
        "absent_support": support,
        "diagnostic_pattern": diagnostic,
        "diagnostic_criteria": config["diagnostic_criteria"],
        "split_disjointness": _split_evidence(data),
        "claim_scope": "fresh realization of the same law; fixed 24-policy catalogue; no GO/NO-GO",
    }
    write_npz_atomic(output / "sealed_monitor_bundle.npz", {key: value for key, value in data.items() if isinstance(value, np.ndarray)})
    write_npz_atomic(output / "result_arrays.npz", arrays)
    write_json_atomic(output / "analysis_core.json", core)
    write_json_atomic(output / "REPORT_WAVE56_STAGE1.json", core)
    return "COMPLETE"


def execute_phase(stage: Path, output: Path, phase: str) -> str:
    files = validate_stage(stage, phase)
    config, utilities, data = load_inputs(stage, phase)
    if output.exists():
        if any(output.iterdir()):
            raise RuntimeError("worker output directory must be empty")
    else:
        output.mkdir(parents=True)
    if phase == "fit":
        status = run_fit(stage, output, config, utilities, data)
    elif phase == "select":
        status = run_select(stage, output, config, utilities, data)
    elif phase == "adjudicate":
        status = run_adjudicate(stage, output, config, utilities, data)
    else:
        raise ValueError(phase)
    return status


def main() -> None:
    args = parse_args()
    if not STAGED_RUNTIME:
        raise RuntimeError("analytical worker must execute from the isolated staged runtime")
    stage = args.stage.resolve(strict=True)
    if Path.cwd().resolve() != stage:
        raise RuntimeError("worker cwd must be the isolated staging directory")
    if os.geteuid() != 65534 or os.getegid() != 65534:
        raise RuntimeError("analytical worker must run as nobody/nogroup")
    request = load_json(stage / "phase_request.json")
    phase = str(request["phase"])
    security = process_security_state()
    local_module_receipts(request)
    files = validate_stage(stage, phase)
    probes = verify_forbidden_probes(args.forbidden_probe)
    output = args.output.resolve()
    if output.parent != stage.parent or output.name != "worker-output":
        raise RuntimeError("worker output must be the isolated staging sibling worker-output")
    status = execute_phase(stage, output, phase)
    modules = local_module_receipts(request)
    write_json_atomic(
        output / "access_receipt.json",
        {
            "phase": phase,
            "status": status,
            "effective_uid": os.geteuid(),
            "effective_gid": os.getegid(),
            "process_security": security,
            "stage_inventory": files,
            "stage_hashes": {name: sha256_file(stage / name) for name in files},
            "sealed_probes": probes,
            "benchmark_root_received": False,
            "local_import_runtime": str(RUNTIME_ROOT),
            "local_module_receipts": modules,
            "output_inventory": inventory(output),
        },
    )
    print(json.dumps({"phase": phase, "status": status}, sort_keys=True))


if __name__ == "__main__":
    main()
