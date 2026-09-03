#!/usr/bin/env python3
"""Prepare fresh, physically separated Wave 55 selection and monitor bundles."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geometria_proporcional.wave49_checker import (  # noqa: E402
    validate_manifest,
    validate_semantic_attestation,
    validate_visible_package,
)
from geometria_proporcional.wave49_generator import generate_benchmark  # noqa: E402
from geometria_proporcional.wave49_oracle import compute_oracle_splits  # noqa: E402
from geometria_proporcional.wave49_schema import (  # noqa: E402
    ProtocolConfig,
    default_protocol_config,
    read_jsonl,
    sha256_file,
    write_json,
)
from geometria_proporcional.wave50_model import FeatureNormalizer  # noqa: E402
from geometria_proporcional.wave50_neural import (  # noqa: E402
    load_labeled_records,
    prepare_examples,
    split_tokens,
    stratified_token_subset,
)
from geometria_proporcional.wave51_factored import (  # noqa: E402
    DualHeadDeepSet,
    predict_dual_logits,
)

PLAN_PATH = REPO_ROOT / "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_55_CONSERVATIVE_POLICY_BRIDGE_PLAN.md"
CONFIG_DEFAULT = REPO_ROOT / "experiments/geometria_proporcional/configs/wave55_policy_bridge.json"
WORKER_PATH = REPO_ROOT / "experiments/geometria_proporcional/_wave55_infer_worker.py"
PUBLIC_KEY = REPO_ROOT / "experiments/geometria_proporcional/keys/wave49_attestation_public.pem"
PRIMITIVES = REPO_ROOT / "src/geometria_proporcional/wave55_policy_bridge.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave50-dir", type=Path, required=True)
    parser.add_argument("--wave51-dir", type=Path, required=True)
    parser.add_argument("--wave52-dir", type=Path, required=True)
    parser.add_argument("--wave53-dir", type=Path, required=True)
    parser.add_argument("--wave54-dir", type=Path, required=True)
    parser.add_argument("--wave54-input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--attestation-private-key", type=Path, required=True)
    parser.add_argument("--replay-secrets-from", type=Path)
    parser.add_argument("--recovery-secrets-from", type=Path)
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


def git_state() -> tuple[str, str]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT, text=True
    ).strip()
    return commit, dirty


def require_sources_at_head(paths: list[Path]) -> None:
    for path in paths:
        relative = path.resolve(strict=True).relative_to(REPO_ROOT)
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(relative)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
        changed = subprocess.check_output(
            ["git", "status", "--porcelain", "--", str(relative)], cwd=REPO_ROOT, text=True
        ).strip()
        if changed:
            raise RuntimeError(f"execution source differs from HEAD: {relative}")


def prepare_output(path: Path, force: bool) -> Path | None:
    archived = None
    if path.exists():
        if not force:
            raise FileExistsError(f"output exists: {path}")
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        archived = path.with_name(f"{path.name}.superseded_{stamp}")
        if archived.exists():
            raise FileExistsError(archived)
        path.rename(archived)
    path.mkdir(parents=True)
    return archived


def has_generation_key(path: Path) -> bool:
    return (path / "benchmark/sealed/generation_secret.json").is_file()


def validate_invocation(
    args: argparse.Namespace,
    output: Path,
    config: dict[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
) -> str:
    """Enforce one primary draw; replay and technical recovery must reuse keys."""
    primary_name = str(config["primary_output_name"])
    replay_name = str(config["replay_output_name"])
    canonical_parent = (repo_root / config["output_parent_relative"]).resolve()
    if output.parent != canonical_parent:
        raise ValueError(f"Wave 55 outputs must live directly under {canonical_parent}")
    if output.name not in {primary_name, replay_name}:
        raise ValueError(f"output name must be {primary_name!r} or {replay_name!r}")
    if args.replay_secrets_from and args.recovery_secrets_from:
        raise ValueError("replay and recovery modes are mutually exclusive")
    if output.name == replay_name:
        if not args.replay_secrets_from or not args.reference_dir:
            raise ValueError("replay requires --replay-secrets-from and --reference-dir")
        if args.recovery_secrets_from:
            raise ValueError("replay output cannot be used for primary recovery")
        reference = args.reference_dir.resolve(strict=True)
        secrets_from = args.replay_secrets_from.resolve(strict=True)
        if reference != secrets_from or reference == output or reference.name != primary_name:
            raise ValueError("replay must use the distinct primary package as reference and key source")
        return "replay"

    if args.replay_secrets_from or args.reference_dir:
        raise ValueError("primary output cannot use replay arguments")
    prior = sorted(output.parent.glob(f"{primary_name}.failed_*")) + sorted(
        output.parent.glob(f"{primary_name}.superseded_*")
    )
    key_bearing = [path for path in prior if has_generation_key(path)]
    if args.recovery_secrets_from:
        source = args.recovery_secrets_from.resolve(strict=True)
        if source != output and source not in prior:
            raise ValueError("recovery key source must be this primary output or one of its archived attempts")
        if source != output and not has_generation_key(source):
            raise ValueError("recovery source contains no drawn generation key")
        if source == output and not output.exists():
            raise ValueError("recovery source does not exist")
        if output.exists() and not args.force:
            raise ValueError("recovering an existing primary requires --force so it is archived")
        return "recovery"
    if output.exists():
        raise FileExistsError("the unique primary already exists; fresh redraw is forbidden")
    if key_bearing:
        raise RuntimeError(
            "a prior primary attempt already drew keys; use --recovery-secrets-from"
        )
    return "primary"


def validate_recovery_contract(source: Path, config_path: Path) -> None:
    """Recovery may change audited runtime code, but never the frozen scientific contract."""
    freeze_path = source / "pre_generation_freeze.json"
    if not freeze_path.is_file():
        raise RuntimeError("recovery source lacks pre_generation_freeze.json")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("config_sha256") != digest(config_path):
        raise RuntimeError("recovery config differs from the pre-generation freeze")
    frozen_plan = freeze.get("sources", {}).get(str(PLAN_PATH.relative_to(REPO_ROOT)))
    if frozen_plan != digest(PLAN_PATH):
        raise RuntimeError("recovery plan differs from the pre-generation freeze")


def read_key(root: Path, name: str) -> bytes:
    payload = json.loads((root / "benchmark/sealed" / name).read_text(encoding="utf-8"))
    key = bytes.fromhex(payload["key_hex"])
    if len(key) != 32:
        raise RuntimeError(f"invalid replay key: {name}")
    return key


def token_logits(examples: list[dict[str, Any]], logits: np.ndarray) -> tuple[list[str], np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for example, row in zip(examples, logits, strict=True):
        grouped[str(example["pair_token"])].append(np.asarray(row, dtype=np.float64))
    tokens = sorted(grouped)
    return tokens, np.stack([np.mean(grouped[token], axis=0) for token in tokens])


def historical_preflight(
    wave50: Path,
    wave51: Path,
    wave52: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Re-forward historical val and reproduce Wave 52 arrays before fresh data exist."""
    torch.set_num_threads(int(config["cpu_threads"]))
    torch.use_deterministic_algorithms(True)
    source = config["source_binding"]
    require_hash(wave50 / "benchmark/visible/val.jsonl", source["wave50_visible_val_sha256"])
    require_hash(wave50 / "authorized_labels/val.jsonl", source["wave50_authorized_val_sha256"])
    require_hash(wave50 / "benchmark/protocol_config.json", source["wave50_protocol_sha256"])
    require_hash(wave51 / "normalizer.npz", source["wave51_normalizer_sha256"])
    require_hash(wave51 / "split_manifest.json", source["wave51_split_manifest_sha256"])
    records, _ = load_labeled_records(
        wave50 / "benchmark/visible/val.jsonl",
        wave50 / "authorized_labels/val.jsonl",
        wave50 / "benchmark/protocol_config.json",
        "val",
    )
    records = stratified_token_subset(records, 3072, 5102)
    _, monitor = split_tokens(records, 0.5, 5102)
    manifest = json.loads((wave51 / "split_manifest.json").read_text(encoding="utf-8"))
    actual_tokens = sorted({str(row["pair_token"]) for row in monitor})
    if actual_tokens != manifest["val_monitor"]:
        raise RuntimeError("historical val-monitor split does not reproduce Wave 51")
    with np.load(wave51 / "normalizer.npz", allow_pickle=False) as data:
        normalizer = FeatureNormalizer(data["mean"], data["std"])
    examples = prepare_examples(monitor, normalizer)
    checks = []
    for seed in config["seeds"]:
        checkpoint_path = wave51 / "checkpoints" / f"seed{seed}__sigmoid_only.pt"
        require_hash(checkpoint_path, source["wave51_checkpoints_sha256"][str(seed)])
        reference_path = wave52 / "raw_eval/frozen_set" / f"seed{seed}__val_monitor.npz"
        require_hash(reference_path, source["wave52_val_monitor_sha256"][str(seed)])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = DualHeadDeepSet()
        model.load_state_dict(checkpoint["model_state"])
        set_logits, choice_logits = predict_dual_logits(
            model, examples, int(config["inference_batch_size"])
        )
        tokens, token_set = token_logits(examples, set_logits)
        choice_tokens, token_choice = token_logits(examples, choice_logits)
        target_by_token = {
            str(row["pair_token"]): np.asarray(row["target"], dtype=np.float32) for row in monitor
        }
        token_target = np.stack([target_by_token[token] for token in tokens])
        with np.load(reference_path, allow_pickle=False) as reference:
            exact = {
                "pair_token": np.array_equal(np.asarray(tokens), reference["pair_token"]),
                "choice_pair_token": tokens == choice_tokens,
                "target": np.array_equal(token_target, reference["target"]),
                "set_logits": np.array_equal(token_set, reference["set_logits"]),
                "choice_logits": np.array_equal(token_choice, reference["choice_logits"]),
            }
        if not all(exact.values()):
            raise RuntimeError(f"Wave 52 historical re-forward mismatch for seed {seed}: {exact}")
        checks.append({"seed": seed, "array_exact": exact})
    return {"status": "PASS", "checks": checks, "n_tokens": len(actual_tokens)}


def strip_checkpoints(wave51: Path, destination: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=False)
    receipts = []
    for seed in config["seeds"]:
        source = wave51 / "checkpoints" / f"seed{seed}__sigmoid_only.pt"
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        payload = {key: checkpoint[key] for key in ("model_state", "seed", "output")}
        target = destination / source.name
        torch.save(payload, target)
        receipts.append({"seed": seed, "source_sha256": digest(source), "path": str(target), "sha256": digest(target)})
    return receipts


def stage_and_infer(output: Path, benchmark: Path, wave51: Path, config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    stage = output / "inference_stage"
    (stage / "visible").mkdir(parents=True)
    for split in ("train", "val"):
        shutil.copy2(benchmark / "visible" / f"{split}.jsonl", stage / "visible" / f"{split}.jsonl")
    shutil.copy2(benchmark / "protocol_config.json", stage / "protocol_config.json")
    shutil.copy2(config_path, stage / "wave55_config.json")
    (stage / "frozen").mkdir()
    shutil.copy2(wave51 / "normalizer.npz", stage / "frozen/normalizer.npz")
    checkpoint_receipts = strip_checkpoints(wave51, stage / "frozen/checkpoints", config)
    inference = output / "inference"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_ROOT)
    subprocess.run(
        [sys.executable, str(WORKER_PATH), "--stage", str(stage), "--output", str(inference)],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )
    if (benchmark / "sealed/oracle").exists():
        raise RuntimeError("oracle appeared before inference freeze")
    files = sorted(path for path in inference.rglob("*") if path.is_file())
    manifest = {
        "phase": "fresh-visible-inference-frozen-before-oracle",
        "files": {str(path.relative_to(output)): digest(path) for path in files},
        "checkpoints": checkpoint_receipts,
        "fit_operations": False,
    }
    write_json(output / "inference_manifest.json", manifest)
    return manifest


def load_fixture_logits(output: Path, split: str, seeds: list[int]) -> dict[int, dict[str, np.ndarray]]:
    result = {}
    for seed in seeds:
        path = output / "inference/logits" / f"seed{seed}__{split}.npz"
        with np.load(path, allow_pickle=False) as data:
            fixture = data["fixture_id"].astype(str)
            if len(set(fixture)) != len(fixture):
                raise RuntimeError(f"duplicate inference fixture in {path}")
            result[seed] = {name: row for name, row in zip(fixture, data["set_logits"], strict=True)}
    return result


def build_bundle(
    benchmark: Path,
    labels: Path,
    output: Path,
    split: str,
    seeds: list[int],
) -> dict[str, np.ndarray]:
    records, _ = load_labeled_records(
        benchmark / "visible" / f"{split}.jsonl",
        labels / f"{split}.jsonl",
        benchmark / "protocol_config.json",
        split,
    )
    predictions = load_fixture_logits(output, split, seeds)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["pair_token"])].append(record)
    tokens = sorted(groups)
    targets = []
    strata = []
    cardinality = []
    per_seed = np.empty((len(seeds), len(tokens), 4), dtype=np.float64)
    for token_index, token in enumerate(tokens):
        views = groups[token]
        target_values = {tuple(np.asarray(view["target"], dtype=bool)) for view in views}
        stratum_values = {str(view["design_stratum"]) for view in views}
        if len(target_values) != 1 or len(stratum_values) != 1:
            raise RuntimeError(f"canonical views disagree for {token}")
        target = np.asarray(next(iter(target_values)), dtype=bool)
        targets.append(target)
        strata.append(next(iter(stratum_values)))
        cardinality.append(int(target.sum()))
        for seed_index, seed in enumerate(seeds):
            per_seed[seed_index, token_index] = np.mean(
                [predictions[seed][view["fixture_id"]] for view in views], axis=0
            )
    bundle = {
        "pair_token": np.asarray(tokens),
        "cluster_id": np.asarray(tokens),
        "target": np.stack(targets),
        "per_seed_logits": per_seed,
        "ensemble_logits": per_seed.mean(axis=0),
        "design_stratum": np.asarray(strata),
        "cardinality": np.asarray(cardinality, dtype=np.int64),
        "split_role": np.asarray(["decision_select" if split == "train" else "sealed_monitor"] * len(tokens)),
    }
    return bundle


def historical_tokens(wave54_inputs: Path) -> set[str]:
    result: set[str] = set()
    for name in ("fit_select_bundle.npz", "sealed_monitor_bundle.npz"):
        with np.load(wave54_inputs / name, allow_pickle=False) as data:
            result.update(data["pair_token"].astype(str).tolist())
    return result


def canonical_npz_hash(path: Path) -> str:
    """Hash array meaning independently of NPZ container metadata."""
    result = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        for key in sorted(data.files):
            array = np.ascontiguousarray(data[key])
            result.update(key.encode("utf-8"))
            result.update(array.dtype.str.encode("ascii"))
            result.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            result.update(array.tobytes())
    return result.hexdigest()


def compare_reference(primary: Path, reference: Path) -> dict[str, Any]:
    if primary.resolve() == reference.resolve():
        raise ValueError("replay cannot reference itself")
    checks = {}
    for relative in ("bundles/decision_select.npz", "bundles/sealed_monitor.npz"):
        with np.load(primary / relative, allow_pickle=False) as left, np.load(reference / relative, allow_pickle=False) as right:
            exact = set(left.files) == set(right.files) and all(
                np.array_equal(left[key], right[key]) for key in left.files
            )
        checks[relative] = exact
        checks[f"{relative}:canonical_hash"] = canonical_npz_hash(primary / relative) == canonical_npz_hash(reference / relative)
    primary_manifest = json.loads((primary / "benchmark/manifest.json").read_text(encoding="utf-8"))
    reference_manifest = json.loads((reference / "benchmark/manifest.json").read_text(encoding="utf-8"))
    checks["benchmark_key_commitments"] = all(
        primary_manifest[key] == reference_manifest[key]
        for key in (
            "generation_key_commitment", "identity_key_commitment",
            "semantic_commitment_key_commitment",
        )
    )
    checks["benchmark_analytic_file_hashes"] = primary_manifest["files"] == reference_manifest["files"]
    for split in ("train", "val"):
        for seed in (17, 29, 43):
            relative = Path("inference/logits") / f"seed{seed}__{split}.npz"
            with np.load(primary / relative, allow_pickle=False) as left, np.load(reference / relative, allow_pickle=False) as right:
                checks[str(relative)] = set(left.files) == set(right.files) and all(
                    np.array_equal(left[key], right[key]) for key in left.files
                )
    if not all(checks.values()):
        raise RuntimeError(f"fresh preparation replay mismatch: {checks}")
    return checks


def execute(args: argparse.Namespace, output: Path) -> None:
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    wave50, wave51, wave52 = args.wave50_dir.resolve(), args.wave51_dir.resolve(), args.wave52_dir.resolve()
    wave53, wave54 = args.wave53_dir.resolve(), args.wave54_dir.resolve()
    wave54_inputs = args.wave54_input_dir.resolve()
    sources = [Path(__file__), WORKER_PATH, PRIMITIVES, PLAN_PATH, config_path]
    commit, dirty = git_state()
    if dirty:
        raise RuntimeError("tracked worktree must be clean before fresh generation")
    require_sources_at_head(sources)
    binding = config["source_binding"]
    upstream = [
        require_hash(wave52 / "policy_manifest.json", binding["wave52_policy_manifest_sha256"]),
        require_hash(wave53 / "platt_calibrator.json", binding["wave53_platt_sha256"]),
        require_hash(wave54 / "selection_freeze.json", binding["wave54_selection_freeze_sha256"]),
        require_hash(wave54 / "selection_state.npz", binding["wave54_selection_state_sha256"]),
    ]
    preflight = historical_preflight(wave50, wave51, wave52, config)
    write_json(
        output / "pre_generation_freeze.json",
        {
            "phase": "all-decisions-frozen-before-key-draw",
            "git_commit": commit,
            "config_sha256": digest(config_path),
            "sources": {str(path.relative_to(REPO_ROOT)): digest(path) for path in sources},
            "upstream": upstream,
            "historical_preflight": preflight,
            "key_drawn": False,
        },
    )

    key_source = args.replay_secrets_from or args.recovery_secrets_from
    if key_source:
        replay = key_source.resolve(strict=True)
        keys = (
            read_key(replay, "generation_secret.json"),
            read_key(replay, "identity_secret.json"),
            read_key(replay, "semantic_commitment_secret.json"),
        )
    else:
        keys = (secrets.token_bytes(32), secrets.token_bytes(32), secrets.token_bytes(32))
    if len(set(keys)) != 3:
        raise RuntimeError("fresh keys must be distinct")
    benchmark = output / "benchmark"
    protocol = default_protocol_config(smoke=False)
    generate_benchmark(
        benchmark,
        protocol,
        generation_key=keys[0],
        identity_key=keys[1],
        commitment_key=keys[2],
        attestation_private_key_path=args.attestation_private_key.resolve(strict=True),
        trusted_public_key_path=PUBLIC_KEY,
    )
    validate_manifest(benchmark)
    validate_visible_package(benchmark, protocol)
    validate_semantic_attestation(benchmark, PUBLIC_KEY)
    manifest = json.loads((benchmark / "manifest.json").read_text(encoding="utf-8"))
    if manifest["generation_key_commitment"] == binding["wave50_generation_key_commitment"]:
        raise RuntimeError("fresh generation commitment unexpectedly equals Wave 50")
    old_visible = digest(wave50 / "benchmark/visible/val.jsonl")
    new_visible = {split: digest(benchmark / "visible" / f"{split}.jsonl") for split in ("train", "val")}
    if old_visible in new_visible.values():
        raise RuntimeError("fresh canonical observations equal historical visible val")
    inference_manifest = stage_and_infer(output, benchmark, wave51, config_path, config)
    compute_oracle_splits(benchmark, protocol, ("train", "val"), output / "authorized_labels")
    if (benchmark / "sealed/oracle/lockbox.jsonl").exists() or (output / "authorized_labels/lockbox.jsonl").exists():
        raise RuntimeError("reserved lockbox oracle was materialized")

    decision = build_bundle(benchmark, output / "authorized_labels", output, "train", config["seeds"])
    monitor = build_bundle(benchmark, output / "authorized_labels", output, "val", config["seeds"])
    if set(decision["pair_token"].astype(str)) & set(monitor["pair_token"].astype(str)):
        raise RuntimeError("fresh selection and monitor tokens overlap")
    old_tokens = historical_tokens(wave54_inputs)
    if old_tokens & (set(decision["pair_token"].astype(str)) | set(monitor["pair_token"].astype(str))):
        raise RuntimeError("fresh Wave 55 tokens overlap Wave 54")
    expected = int(config["fresh_benchmark"]["expected_eligible_pair_tokens_per_split"])
    if len(decision["pair_token"]) != expected or len(monitor["pair_token"]) != expected:
        raise RuntimeError("unexpected eligible token count")
    (output / "bundles").mkdir()
    np.savez_compressed(output / "bundles/decision_select.npz", **decision)
    np.savez_compressed(output / "bundles/sealed_monitor.npz", **monitor)
    bundle_manifest = {
        "status": "FRESH_PHYSICALLY_SEPARATED_BUNDLES",
        "git_commit": commit,
        "freshness": {
            "generation_key_commitment": manifest["generation_key_commitment"],
            "differs_from_wave50": True,
            "visible_sha256": new_visible,
            "visible_differs_from_wave50_val": True,
            "wave54_token_overlap": 0,
        },
        "chronology": [
            "freeze_plan_config_code_bindings",
            "draw_or_recover_keys_once",
            "generate_visible_and_sealed_truth",
            "infer_train_val_without_oracle",
            "freeze_inference_manifest",
            "materialize_train_val_oracle_only",
            "build_separate_bundles",
        ],
        "inference_manifest_sha256": digest(output / "inference_manifest.json"),
        "bundles": {
            "decision_select.npz": digest(output / "bundles/decision_select.npz"),
            "sealed_monitor.npz": digest(output / "bundles/sealed_monitor.npz"),
        },
        "counts": {"decision_select": len(decision["pair_token"]), "sealed_monitor": len(monitor["pair_token"])},
        "lockbox_oracle_materialized": False,
        "inference_fit_operations": inference_manifest["fit_operations"],
    }
    write_json(output / "bundle_manifest.json", bundle_manifest)
    if args.reference_dir:
        replay_checks = compare_reference(output, args.reference_dir.resolve(strict=True))
        write_json(
            output / "preparation_replay.json",
            {
                "phase": "fresh-generator-and-inference-replay",
                "reference_dir": str(args.reference_dir.resolve(strict=True)),
                "replay_key_source": str(args.replay_secrets_from.resolve(strict=True)),
                "checks": replay_checks,
                "all_exact": all(replay_checks.values()),
            },
        )


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    config = json.loads(args.config.resolve(strict=True).read_text(encoding="utf-8"))
    mode = validate_invocation(args, output, config)
    if mode == "recovery":
        validate_recovery_contract(args.recovery_secrets_from.resolve(strict=True), args.config.resolve(strict=True))
    recovery_was_output = bool(
        args.recovery_secrets_from
        and args.recovery_secrets_from.resolve() == output
    )
    archived = prepare_output(output, args.force)
    if recovery_was_output:
        args.recovery_secrets_from = archived
    try:
        execute(args, output)
    except Exception as error:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        failed = output.with_name(f"{output.name}.failed_{stamp}")
        write_json(output / "FAILURE.json", {"error_type": type(error).__name__, "message": str(error), "recovery_requires_same_keys_if_present": True})
        output.rename(failed)
        raise
    summary = json.loads((output / "bundle_manifest.json").read_text(encoding="utf-8"))
    summary["superseded_output"] = str(archived) if archived else None
    summary["execution_mode"] = mode
    write_json(output / "bundle_manifest.json", summary)
    print(json.dumps(summary["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
