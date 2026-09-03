#!/usr/bin/env python3
"""Run the Wave 51 CPU-only factored set/choice development smoke."""

from __future__ import annotations

import argparse
import copy
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from geometria_proporcional.wave49_schema import sha256_file, write_json  # noqa: E402
from geometria_proporcional.wave50_neural import (  # noqa: E402
    FAMILY_TO_INDEX,
    TARGET_COMPATIBILITY_DISTANCE,
    TARGET_SCHEMA_VERSION,
    balanced_target_derangement,
    fit_normalizer,
    load_labeled_records,
    prepare_examples,
    select_smoke_tau,
    smoke_metrics,
    split_tokens,
    stable_hash,
    stratified_token_subset,
)
from geometria_proporcional.wave51_factored import (  # noqa: E402
    DualHeadDeepSet,
    choice_metrics,
    make_optimizer,
    parameter_count,
    predict_dual_logits,
    state_dict_digest,
    token_metric_rows,
    train_epochs,
)


PRIMARY_ARMS = (
    "softmax_only",
    "sigmoid_only",
    "joint_multitask",
    "staged_unfrozen",
    "factored_frozen",
)
ALL_OUTPUTS = (
    "softmax_only",
    "sigmoid_only_epoch50",
    "sigmoid_only",
    "joint_multitask",
    "staged_unfrozen",
    "factored_frozen",
    "factored_true_choice_control",
    "factored_shuffled_choice_control",
)
INDEX_TO_FAMILY = {index: family for family, index in FAMILY_TO_INDEX.items()}
AUTHORIZED_WAVE50_INPUTS = {
    "benchmark/visible/train.jsonl",
    "authorized_labels/train.jsonl",
    "benchmark/visible/val.jsonl",
    "authorized_labels/val.jsonl",
    "benchmark/protocol_config.json",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave50-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "experiments/geometria_proporcional/configs/wave51_factored_smoke.json",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _validate_paths(wave50_dir: Path, output_dir: Path) -> None:
    source = wave50_dir.resolve()
    output = output_dir.resolve()
    repo = REPO_ROOT.resolve()
    allowed = (repo / "data/geometria_proporcional").resolve()
    if output == repo or output in repo.parents:
        raise ValueError("refusing output path at or above repository root")
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("output path must be disjoint from Wave 50 input")
    if allowed not in output.parents:
        raise ValueError(f"output path must be a child of {allowed}")


def _assert_no_lockbox_inputs(paths: list[Path]) -> None:
    resolved = [path.resolve(strict=True) for path in paths]
    if any(
        any("lockbox" in part.lower() for part in path.parts)
        for path in resolved
    ):
        raise ValueError("Wave 51 smoke input resolves to lockbox content")


def _validate_source_binding(source_dir: Path, config: dict) -> dict:
    """Reject any Wave 50 source that differs from the predeclared canonical artifact."""
    binding = config.get("source_binding")
    if not isinstance(binding, dict):
        raise ValueError("config must contain source_binding")
    execution_path = source_dir / "execution_manifest.json"
    training_path = source_dir / "training_manifest.json"
    configured_files = binding.get("files", {})
    if set(configured_files) != AUTHORIZED_WAVE50_INPUTS:
        raise ValueError("source_binding must cover exactly the authorized Wave 50 inputs")
    bound_paths = [source_dir / relative for relative in sorted(configured_files)]
    _assert_no_lockbox_inputs([execution_path, training_path, *bound_paths])
    manifest_hashes = {
        "execution_manifest_sha256": sha256_file(execution_path),
        "training_manifest_sha256": sha256_file(training_path),
    }
    for key, actual in manifest_hashes.items():
        if actual != binding.get(key):
            raise ValueError(f"Wave 50 source binding mismatch for {key}")

    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if training.get("git_commit") != binding.get("canonical_wave50_commit"):
        raise ValueError("Wave 50 training manifest commit differs from source binding")
    if training.get("phase") != binding.get("training_manifest_phase"):
        raise ValueError("Wave 50 training manifest phase differs from source binding")

    checked_files = []
    for relative, expected in sorted(configured_files.items()):
        path = source_dir / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Wave 50 source binding mismatch for {relative}")
        execution_entry = execution.get("files", {}).get(relative)
        training_entry = training.get("files", {}).get(relative)
        declared = [
            entry.get("sha256")
            for entry in (execution_entry, training_entry)
            if isinstance(entry, dict)
        ]
        if not declared or any(value != expected for value in declared):
            raise ValueError(f"Wave 50 manifests disagree with source binding for {relative}")
        checked_files.append({"path": relative, "sha256": actual})
    return {
        "canonical_wave50_commit": binding["canonical_wave50_commit"],
        "training_manifest_phase": binding["training_manifest_phase"],
        **manifest_hashes,
        "files": checked_files,
    }


def _execution_sources(config_path: Path) -> list[dict[str, str]]:
    paths = [
        Path(__file__).resolve(),
        (REPO_ROOT / "src/geometria_proporcional/wave51_factored.py").resolve(),
        (REPO_ROOT / "src/geometria_proporcional/wave50_model.py").resolve(),
        (REPO_ROOT / "src/geometria_proporcional/wave50_neural.py").resolve(),
        (
            REPO_ROOT
            / "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
            "WAVE_51_FACTORED_SET_POLICY_SMOKE_PLAN.md"
        ).resolve(),
        config_path.resolve(),
    ]
    records = []
    for path in paths:
        relative = path.relative_to(REPO_ROOT)
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(relative)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if tracked.returncode != 0:
            raise RuntimeError(f"execution source is not tracked: {relative}")
        if subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", str(relative)], cwd=REPO_ROOT
        ).returncode != 0:
            raise RuntimeError(f"execution source differs from HEAD: {relative}")
        records.append({"path": str(relative), "sha256": sha256_file(path)})
    return records


def _fresh(initial_state: dict[str, torch.Tensor]) -> DualHeadDeepSet:
    model = DualHeadDeepSet()
    model.load_state_dict(copy.deepcopy(initial_state))
    return model


def _save_checkpoint(
    output_dir: Path,
    stem: str,
    model: DualHeadDeepSet,
    histories: dict[str, list[dict[str, float]]],
    optimizer_states: dict[str, dict],
    config: dict,
    seed: int,
) -> str:
    path = output_dir / "checkpoints" / f"{stem}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "histories": histories,
        "optimizer_states": optimizer_states,
        "seed": seed,
        "output": stem.split("__", 1)[1],
        "config": config,
    }, path)
    return str(path.relative_to(output_dir))


def _effective_logits(
    output_name: str, set_logits: np.ndarray, choice_logits: np.ndarray
) -> tuple[np.ndarray, np.ndarray, str, str]:
    if output_name == "softmax_only":
        return choice_logits, choice_logits, "softmax", "softmax_partial"
    if output_name in {"sigmoid_only", "sigmoid_only_epoch50"}:
        return set_logits, set_logits, "sigmoid", "sigmoid_set"
    return set_logits, choice_logits, "sigmoid", "sigmoid_set"


def _save_logits(
    path: Path,
    examples: list[dict],
    set_logits: np.ndarray,
    choice_logits: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        fixture_id=np.asarray([row["fixture_id"] for row in examples]),
        pair_token=np.asarray([row["pair_token"] for row in examples]),
        design_stratum=np.asarray([row["design_stratum"] for row in examples]),
        target=np.stack([row["target"] for row in examples]),
        set_logits=np.asarray(set_logits, dtype=np.float32),
        choice_logits=np.asarray(choice_logits, dtype=np.float32),
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _target_families(target: np.ndarray) -> list[str]:
    return [
        INDEX_TO_FAMILY[index]
        for index in np.flatnonzero(np.asarray(target) >= 0.5)
    ]


def _filter_slice(
    examples: list[dict],
    set_logits: np.ndarray,
    choice_logits: np.ndarray,
    stratum: str | None,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    indices = np.asarray([
        index
        for index, row in enumerate(examples)
        if stratum is None or row["design_stratum"] == stratum
    ])
    return [examples[int(index)] for index in indices], set_logits[indices], choice_logits[indices]


def _evaluate(
    output_name: str,
    examples: list[dict],
    set_logits: np.ndarray,
    choice_logits: np.ndarray,
    tau: float,
) -> dict:
    effective_set, effective_choice, activation, wave50_arm = _effective_logits(
        output_name, set_logits, choice_logits
    )
    result = {}
    for label, stratum in (("ALL", None), ("NEAR_RIVAL", "NEAR_RIVAL"), ("FAR_RIVAL", "FAR_RIVAL")):
        subset, subset_set, subset_choice = _filter_slice(
            examples, effective_set, effective_choice, stratum
        )
        metrics = smoke_metrics(subset, subset_set, wave50_arm, tau=tau)
        metrics["overall"].update(
            choice_metrics(subset, subset_set, subset_choice, activation, tau)
        )
        result[label] = metrics
    return result


def _write_report(output_dir: Path, summary: dict) -> None:
    lines = [
        "# Ola 51 — smoke CPU de conjunto + elección factorizados",
        "",
        "> `SMOKE_ONLY / OPENED_HISTORICAL_DATA`: usa train/val ya abiertos; no leyó lockbox. "
        "No adjudica arquitectura ni GO/NO-GO.",
        "",
        f"- Commit ejecutado: `{summary['git_commit']}`",
        f"- CPU wall time: `{summary['runtime']['wall_seconds']:.2f}s`",
        f"- Parámetros por brazo: `{summary['parameter_count']}`",
        f"- Máximo delta por permutar puntos: `{summary['max_permutation_delta']:.3e}`",
        f"- Fase A exacta entre sigmoid@50 y factored: `{summary['invariants']['phase_a_exact']}`",
        f"- Set-path congelado durante fase B: `{summary['invariants']['factored_set_path_frozen']}`",
        "",
        "## Ensemble en val_monitor",
        "",
        "| salida | slice | tau | set recall | top1 elección | top1 gated | ancho | incompatible | macro AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for output_name in ALL_OUTPUTS:
        for slice_name in ("ALL", "NEAR_RIVAL", "FAR_RIVAL"):
            metrics = summary["ensemble"][output_name]["metrics"][slice_name]["overall"]
            lines.append(
                f"| {output_name} | {slice_name} | {summary['ensemble'][output_name]['tau']:.2f} | "
                f"{metrics['set_recall']:.3f} | {metrics['choice_top1_compatible']:.3f} | "
                f"{metrics['choice_top1_gated_compatible']:.3f} | {metrics['width']:.3f} | "
                f"{metrics['any_incompatible']:.3f} | {metrics['membership_macro_auc']:.3f} |"
            )
    lines.extend([
        "",
        "## Criterio diagnóstico predeclarado",
        "",
    ])
    for key, value in summary["diagnostic_pattern"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend([
        "",
        "## Control de supervisión de elección",
        "",
        f"- true − shuffled top-1: `{summary['choice_control']['true_minus_shuffled_top1']:+.3f}`",
        f"- true − shuffled top-1 gated: "
        f"`{summary['choice_control']['true_minus_shuffled_gated_top1']:+.3f}`",
        "- alcance: `MATCHED_CONTROL_SUBSET_ONLY`; no comparar directamente con el brazo main.",
        "",
        "## Contrastes de mecanismo",
        "",
        f"- staged unfrozen − joint multitask, top-1 gated: "
        f"`{summary['mechanism_contrasts']['staged_unfrozen_minus_joint_gated_top1']:+.3f}`",
        f"- factored − staged unfrozen, top-1 gated: "
        f"`{summary['mechanism_contrasts']['factored_minus_staged_unfrozen_gated_top1']:+.3f}`",
        "- estos contrastes adjudican separación/congelamiento; no deciden por sí solos si dos cabezas son útiles.",
    ])
    lines.extend([
        "",
        "## Alcance",
        "",
        "La cabeza de elección sólo aprende a escoger algún miembro compatible bajo partial-label loss. "
        "No dispone de utilidad, costo ni contexto autorizado, por lo que no constituye una política "
        "semántica. Un resultado prometedor sólo habilita diseñar un paquete prospectivo fresco.",
        "",
    ])
    (output_dir / "REPORT_WAVE51_FACTORED_SMOKE.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    source_dir = args.wave50_dir.resolve()
    output_dir = args.output_dir.resolve()
    config_path = args.config.resolve()
    _validate_paths(source_dir, output_dir)
    _assert_no_lockbox_inputs([config_path])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "SMOKE_ONLY_OPENED_HISTORICAL":
        raise ValueError("config must declare SMOKE_ONLY_OPENED_HISTORICAL")
    benchmark = source_dir / "benchmark"
    input_paths = [
        benchmark / "visible/train.jsonl",
        source_dir / "authorized_labels/train.jsonl",
        benchmark / "visible/val.jsonl",
        source_dir / "authorized_labels/val.jsonl",
        benchmark / "protocol_config.json",
    ]
    _assert_no_lockbox_inputs(input_paths)
    source_binding = _validate_source_binding(source_dir, config)
    execution_sources = _execution_sources(config_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.force:
            raise SystemExit(f"non-empty output exists: {output_dir}; pass --force to replace")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_initialized():
        raise RuntimeError("Wave 51 smoke must start before any CUDA context is initialized")
    torch.set_num_threads(int(config["cpu_threads"]))
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()

    train_all, train_reads = load_labeled_records(
        input_paths[0],
        input_paths[1],
        input_paths[4],
        "train",
        expected_schema=TARGET_SCHEMA_VERSION,
        expected_compatibility_distance=TARGET_COMPATIBILITY_DISTANCE,
    )
    val_all, val_reads = load_labeled_records(
        input_paths[2],
        input_paths[3],
        input_paths[4],
        "val",
        expected_schema=TARGET_SCHEMA_VERSION,
        expected_compatibility_distance=TARGET_COMPATIBILITY_DISTANCE,
    )
    if _validate_source_binding(source_dir, config) != source_binding:
        raise RuntimeError("Wave 50 source changed while authorized train/val were loading")
    train_records = stratified_token_subset(
        train_all, int(config["max_train_fixtures"]), int(config["subset_seed"])
    )
    val_records = stratified_token_subset(
        val_all, int(config["max_val_fixtures"]), int(config["subset_seed"]) + 1
    )
    threshold_records, monitor_records = split_tokens(
        val_records, float(config["val_threshold_fraction"]), int(config["split_seed"])
    )
    token_sets = [
        {row["pair_token"] for row in records}
        for records in (train_records, threshold_records, monitor_records)
    ]
    if token_sets[0] & token_sets[1] or token_sets[0] & token_sets[2] or token_sets[1] & token_sets[2]:
        raise RuntimeError("pair_token leakage across train/threshold/monitor")

    normalizer = fit_normalizer(train_records)
    np.savez(output_dir / "normalizer.npz", mean=normalizer.mean, std=normalizer.std)
    train_examples = prepare_examples(train_records, normalizer)
    threshold_examples = prepare_examples(threshold_records, normalizer)
    monitor_examples = prepare_examples(monitor_records, normalizer)

    eligible_tokens, _, eligibility = balanced_target_derangement(
        train_all, int(config["shuffle_seed"])
    )
    eligible = [row for row in train_all if row["pair_token"] in eligible_tokens]
    control_records = stratified_token_subset(
        eligible,
        int(config["max_control_fixtures"]),
        int(config["subset_seed"]) + 2,
        include_target_hash=True,
    )
    control_tokens, shuffled_targets, shuffle_report = balanced_target_derangement(
        control_records, int(config["shuffle_seed"]) + 1
    )
    control_records = [row for row in control_records if row["pair_token"] in control_tokens]
    control_true = prepare_examples(control_records, normalizer)
    control_shuffled = prepare_examples(
        control_records, normalizer, target_by_token=shuffled_targets
    )
    true_target_by_token = {
        row["pair_token"]: np.asarray(row["target"])
        for row in control_true
    }
    write_json(output_dir / "shuffle_manifest.json", {
        "scope": "choice_head_phase_b_only",
        "full_pool_eligibility": eligibility,
        "control_report": shuffle_report,
        "pair_tokens": sorted(control_tokens),
        "mapping": [
            {
                "pair_token": token,
                "original_families": _target_families(true_target_by_token[token]),
                "replacement_families": _target_families(shuffled_targets[token]),
            }
            for token in sorted(control_tokens)
        ],
    })
    write_json(output_dir / "split_manifest.json", {
        "train": sorted(token_sets[0]),
        "val_threshold": sorted(token_sets[1]),
        "val_monitor": sorted(token_sets[2]),
    })

    predictions: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {
        output: {"val_threshold": [], "val_monitor": []} for output in ALL_OUTPUTS
    }
    run_rows = []
    parameter_counts: set[int] = set()
    phase_a_exact_checks = []
    frozen_checks = []
    max_permutation_delta = 0.0
    phase_a_epochs = int(config["phase_a_epochs"])
    phase_b_epochs = int(config["phase_b_epochs"])
    total_epochs = phase_a_epochs + phase_b_epochs
    lr = float(config["learning_rate"])
    wd = float(config["weight_decay"])
    batch_tokens = int(config["batch_tokens"])

    for seed in config["seeds"]:
        seed = int(seed)
        torch.manual_seed(seed)
        initial_state = copy.deepcopy(DualHeadDeepSet().state_dict())

        sigmoid = _fresh(initial_state)
        parameter_counts.add(parameter_count(sigmoid))
        sigmoid_optimizer = make_optimizer(sigmoid.parameters(), lr, wd)
        sigmoid_history_a = train_epochs(
            sigmoid, train_examples, "set_bce", seed, 0, phase_a_epochs, batch_tokens, sigmoid_optimizer
        )
        sigmoid_phase_a_state = copy.deepcopy(sigmoid.state_dict())
        sigmoid_phase_a = _fresh(sigmoid_phase_a_state)
        checkpoint = _save_checkpoint(
            output_dir,
            f"seed{seed}__sigmoid_only_epoch50",
            sigmoid_phase_a,
            {"phase_a": sigmoid_history_a},
            {"phase_a": sigmoid_optimizer.state_dict()},
            config,
            seed,
        )
        run_rows.append({"seed": seed, "output": "sigmoid_only_epoch50", "checkpoint": checkpoint})
        sigmoid_history_b = train_epochs(
            sigmoid,
            train_examples,
            "set_bce",
            seed,
            phase_a_epochs,
            phase_b_epochs,
            batch_tokens,
            sigmoid_optimizer,
        )

        factored = _fresh(initial_state)
        factored_optimizer_a = make_optimizer(factored.parameters(), lr, wd)
        factored_history_a = train_epochs(
            factored, train_examples, "set_bce", seed, 0, phase_a_epochs, batch_tokens, factored_optimizer_a
        )
        phase_a_exact = state_dict_digest(factored) == state_dict_digest(sigmoid_phase_a)
        phase_a_exact_checks.append(phase_a_exact)
        if not phase_a_exact:
            raise RuntimeError(f"seed {seed}: factored phase A differs from sigmoid@50")
        phase_a_state = copy.deepcopy(factored.state_dict())
        frozen_before = state_dict_digest(
            factored, prefixes=("point_mlp.", "set_mlp.", "set_head.")
        )
        for name, parameter in factored.named_parameters():
            parameter.requires_grad_(name.startswith("choice_head."))
        factored_optimizer_b = make_optimizer(factored.choice_head.parameters(), lr, wd)
        factored_history_b = train_epochs(
            factored,
            train_examples,
            "choice_partial",
            seed,
            phase_a_epochs,
            phase_b_epochs,
            batch_tokens,
            factored_optimizer_b,
        )
        frozen_after = state_dict_digest(
            factored, prefixes=("point_mlp.", "set_mlp.", "set_head.")
        )
        frozen_checks.append(frozen_before == frozen_after)
        if frozen_before != frozen_after:
            raise RuntimeError(f"seed {seed}: factored set path changed during phase B")

        softmax = _fresh(initial_state)
        softmax_optimizer = make_optimizer(softmax.parameters(), lr, wd)
        softmax_history = train_epochs(
            softmax, train_examples, "choice_partial", seed, 0, total_epochs, batch_tokens, softmax_optimizer
        )

        joint = _fresh(initial_state)
        joint_optimizer = make_optimizer(joint.parameters(), lr, wd)
        joint_history = train_epochs(
            joint, train_examples, "joint_equal", seed, 0, total_epochs, batch_tokens, joint_optimizer
        )

        staged_unfrozen = _fresh(phase_a_state)
        staged_unfrozen_optimizer = make_optimizer(staged_unfrozen.parameters(), lr, wd)
        staged_unfrozen_history_b = train_epochs(
            staged_unfrozen,
            train_examples,
            "choice_partial",
            seed,
            phase_a_epochs,
            phase_b_epochs,
            batch_tokens,
            staged_unfrozen_optimizer,
        )

        controls = {}
        for output_name, examples in (
            ("factored_true_choice_control", control_true),
            ("factored_shuffled_choice_control", control_shuffled),
        ):
            control = _fresh(phase_a_state)
            control_frozen_before = state_dict_digest(
                control, prefixes=("point_mlp.", "set_mlp.", "set_head.")
            )
            for name, parameter in control.named_parameters():
                parameter.requires_grad_(name.startswith("choice_head."))
            optimizer = make_optimizer(control.choice_head.parameters(), lr, wd)
            history = train_epochs(
                control,
                examples,
                "choice_partial",
                seed,
                phase_a_epochs,
                phase_b_epochs,
                batch_tokens,
                optimizer,
            )
            if state_dict_digest(
                control, prefixes=("point_mlp.", "set_mlp.", "set_head.")
            ) != control_frozen_before:
                raise RuntimeError(f"seed {seed}: {output_name} changed frozen set path")
            controls[output_name] = (control, history, optimizer.state_dict())

        models = {
            "sigmoid_only_epoch50": (sigmoid_phase_a, {"phase_a": sigmoid_history_a}, {}),
            "sigmoid_only": (
                sigmoid,
                {"phase_a": sigmoid_history_a, "phase_b_budget": sigmoid_history_b},
                {"all": sigmoid_optimizer.state_dict()},
            ),
            "factored_frozen": (
                factored,
                {"phase_a": factored_history_a, "phase_b": factored_history_b},
                {"phase_a": factored_optimizer_a.state_dict(), "phase_b": factored_optimizer_b.state_dict()},
            ),
            "softmax_only": (softmax, {"all": softmax_history}, {"all": softmax_optimizer.state_dict()}),
            "joint_multitask": (joint, {"all": joint_history}, {"all": joint_optimizer.state_dict()}),
            "staged_unfrozen": (
                staged_unfrozen,
                {
                    "phase_a_reused_from_factored": factored_history_a,
                    "phase_b_choice_encoder_unfrozen": staged_unfrozen_history_b,
                },
                {"phase_b": staged_unfrozen_optimizer.state_dict()},
            ),
            **{
                name: (model, {"phase_b": history}, {"phase_b": optimizer_state})
                for name, (model, history, optimizer_state) in controls.items()
            },
        }

        for output_name, (model, histories, optimizer_states) in models.items():
            parameter_counts.add(parameter_count(model))
            if output_name != "sigmoid_only_epoch50":
                checkpoint = _save_checkpoint(
                    output_dir,
                    f"seed{seed}__{output_name}",
                    model,
                    histories,
                    optimizer_states,
                    config,
                    seed,
                )
                run_rows.append({"seed": seed, "output": output_name, "checkpoint": checkpoint})
            for split_name, examples in (
                ("val_threshold", threshold_examples),
                ("val_monitor", monitor_examples),
            ):
                set_logits, choice_logits = predict_dual_logits(
                    model, examples, int(config["inference_batch_size"])
                )
                predictions[output_name][split_name].append((set_logits, choice_logits))
                _save_logits(
                    output_dir / "logits" / f"seed{seed}__{output_name}__{split_name}.npz",
                    examples,
                    set_logits,
                    choice_logits,
                )

        probe = monitor_examples[: min(16, len(monitor_examples))]
        for output_name in PRIMARY_ARMS:
            model = models[output_name][0]
            original = predict_dual_logits(model, probe, int(config["inference_batch_size"]))
            permuted = []
            for index, example in enumerate(probe):
                clone = dict(example)
                order = np.random.default_rng(seed + index + 510_000).permutation(len(example["features"]))
                clone["features"] = example["features"][order]
                permuted.append(clone)
            shuffled = predict_dual_logits(model, permuted, int(config["inference_batch_size"]))
            max_permutation_delta = max(
                max_permutation_delta,
                *(float(np.max(np.abs(left - right))) for left, right in zip(original, shuffled, strict=True)),
            )

    if parameter_counts != {13_384}:
        raise RuntimeError(f"parameter mismatch across arms: {sorted(parameter_counts)}")
    if max_permutation_delta > float(config["permutation_atol"]):
        raise RuntimeError(f"permutation invariance failed: {max_permutation_delta}")

    ensemble = {}
    for output_name in ALL_OUTPUTS:
        threshold_pairs = predictions[output_name]["val_threshold"]
        monitor_pairs = predictions[output_name]["val_monitor"]
        threshold_set = np.mean(np.stack([row[0] for row in threshold_pairs]), axis=0)
        threshold_choice = np.mean(np.stack([row[1] for row in threshold_pairs]), axis=0)
        monitor_set = np.mean(np.stack([row[0] for row in monitor_pairs]), axis=0)
        monitor_choice = np.mean(np.stack([row[1] for row in monitor_pairs]), axis=0)
        effective_threshold, _, _, wave50_arm = _effective_logits(
            output_name, threshold_set, threshold_choice
        )
        tau, tau_selection = select_smoke_tau(
            threshold_examples, effective_threshold, wave50_arm, config["diagnostic_tau_grid"]
        )
        metrics = _evaluate(
            output_name, monitor_examples, monitor_set, monitor_choice, tau
        )
        effective_monitor_set, effective_monitor_choice, activation, _ = _effective_logits(
            output_name, monitor_set, monitor_choice
        )
        per_token = token_metric_rows(
            monitor_examples,
            effective_monitor_set,
            effective_monitor_choice,
            activation,
            tau,
        )
        _write_jsonl(
            output_dir / "metrics_by_token" / f"{output_name}__val_monitor.jsonl",
            [{"output": output_name, "tau": tau, **row} for row in per_token],
        )
        _save_logits(
            output_dir / "ensemble_logits" / f"{output_name}__val_monitor.npz",
            monitor_examples,
            monitor_set,
            monitor_choice,
        )
        ensemble[output_name] = {
            "tau": tau,
            "tau_selection": tau_selection,
            "metrics": metrics,
        }

    near_factored = ensemble["factored_frozen"]["metrics"]["NEAR_RIVAL"]["overall"]
    near_sigmoid = ensemble["sigmoid_only"]["metrics"]["NEAR_RIVAL"]["overall"]
    near_softmax = ensemble["softmax_only"]["metrics"]["NEAR_RIVAL"]["overall"]
    near_joint = ensemble["joint_multitask"]["metrics"]["NEAR_RIVAL"]["overall"]
    near_staged = ensemble["staged_unfrozen"]["metrics"]["NEAR_RIVAL"]["overall"]
    near_true_control = ensemble["factored_true_choice_control"]["metrics"]["NEAR_RIVAL"]["overall"]
    near_shuffled_control = ensemble["factored_shuffled_choice_control"]["metrics"]["NEAR_RIVAL"]["overall"]
    criteria = config["diagnostic_criteria"]
    choice_control = {
        "true_minus_shuffled_top1": (
            near_true_control["choice_top1_compatible"]
            - near_shuffled_control["choice_top1_compatible"]
        ),
        "true_minus_shuffled_gated_top1": (
            near_true_control["choice_top1_gated_compatible"]
            - near_shuffled_control["choice_top1_gated_compatible"]
        ),
        "scope": "MATCHED_CONTROL_SUBSET_ONLY",
    }
    core_conditions = {
        "phase_a_exact": all(phase_a_exact_checks),
        "set_path_frozen": all(frozen_checks),
        "set_recall_noninferior_to_sigmoid60": (
            near_factored["set_recall"] - near_sigmoid["set_recall"]
            >= float(criteria["set_recall_noninferiority_margin"])
        ),
        "width_increase_within_limit": (
            near_factored["width"] - near_sigmoid["width"]
            <= float(criteria["width_increase_max"])
        ),
        "incompatible_increase_within_limit": (
            near_factored["any_incompatible"] - near_sigmoid["any_incompatible"]
            <= float(criteria["any_incompatible_increase_max"])
        ),
        "choice_top1_improves_over_sigmoid": (
            near_factored["choice_top1_gated_compatible"]
            - near_sigmoid["choice_top1_gated_compatible"]
            >= float(criteria["choice_top1_improvement_min"])
        ),
        "choice_top1_noninferior_to_softmax": (
            near_factored["choice_top1_gated_compatible"]
            - near_softmax["choice_top1_gated_compatible"]
            >= float(criteria["softmax_top1_noninferiority_margin"])
        ),
        "choice_control_has_target_signal": (
            choice_control["true_minus_shuffled_gated_top1"]
            >= float(criteria["choice_control_improvement_min"])
        ),
    }
    mechanism_contrasts = {
        "staged_unfrozen_minus_joint_gated_top1": (
            near_staged["choice_top1_gated_compatible"]
            - near_joint["choice_top1_gated_compatible"]
        ),
        "factored_minus_staged_unfrozen_gated_top1": (
            near_factored["choice_top1_gated_compatible"]
            - near_staged["choice_top1_gated_compatible"]
        ),
    }
    mechanism_specific = all(
        value >= float(criteria["mechanism_advantage_min"])
        for value in mechanism_contrasts.values()
    )
    pattern = {
        **core_conditions,
        "factored_candidate_promising": all(core_conditions.values()),
        "staging_and_freeze_pattern_specific": (
            all(core_conditions.values()) and mechanism_specific
        ),
    }

    accessed = sorted({
        *(Path(path).resolve() for path in train_reads + val_reads),
        config_path,
        (source_dir / "execution_manifest.json").resolve(),
        (source_dir / "training_manifest.json").resolve(),
    })
    _assert_no_lockbox_inputs(accessed)
    write_json(output_dir / "access_receipt.json", {
        "status": "SMOKE_ONLY_OPENED_HISTORICAL",
        "allowed_splits": ["train", "val"],
        "lockbox_accessed": False,
        "source_binding": source_binding,
        "post_load_source_revalidation": True,
        "files_read": [
            {"path": str(path), "sha256": sha256_file(path)} for path in accessed
        ],
    })

    artifact_files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.name not in {"artifact_manifest.json", "summary.json", "REPORT_WAVE51_FACTORED_SMOKE.md"}
    )
    write_json(output_dir / "artifact_manifest.json", {
        "execution_sources": execution_sources,
        "files": [
            {"path": str(path.relative_to(output_dir)), "sha256": sha256_file(path)}
            for path in artifact_files
        ],
    })
    artifact_manifest_sha256 = sha256_file(output_dir / "artifact_manifest.json")
    summary = {
        "status": "SMOKE_ONLY_OPENED_HISTORICAL",
        "scientific_claim_allowed": False,
        "automatic_go": False,
        "decision_authority": "user",
        "git_commit": _git_commit(),
        "execution_sources": execution_sources,
        "source_binding": source_binding,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "config": config,
        "parameter_count": next(iter(parameter_counts)),
        "data": {
            "train_fixtures": len(train_examples),
            "train_pair_tokens": len(token_sets[0]),
            "val_threshold_fixtures": len(threshold_examples),
            "val_threshold_pair_tokens": len(token_sets[1]),
            "val_monitor_fixtures": len(monitor_examples),
            "val_monitor_pair_tokens": len(token_sets[2]),
            "choice_control_fixtures": len(control_true),
            "choice_control_pair_tokens": len(control_tokens),
        },
        "invariants": {
            "phase_a_exact": all(phase_a_exact_checks),
            "phase_a_exact_by_seed": phase_a_exact_checks,
            "factored_set_path_frozen": all(frozen_checks),
            "factored_set_path_frozen_by_seed": frozen_checks,
        },
        "max_permutation_delta": max_permutation_delta,
        "runs": run_rows,
        "ensemble": ensemble,
        "choice_control": choice_control,
        "mechanism_contrasts": mechanism_contrasts,
        "diagnostic_pattern": pattern,
        "runtime": {
            "device": "cpu",
            "cpu_threads": int(config["cpu_threads"]),
            "wall_seconds": time.perf_counter() - started,
            "cuda_initialized": torch.cuda.is_initialized(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
        "artifact_hash": stable_hash({
            "config": config,
            "ensemble": ensemble,
            "choice_control": choice_control,
            "mechanism_contrasts": mechanism_contrasts,
            "diagnostic_pattern": pattern,
        }),
    }
    write_json(output_dir / "summary.json", summary)
    _write_report(output_dir, summary)
    final_files = (
        "artifact_manifest.json",
        "summary.json",
        "REPORT_WAVE51_FACTORED_SMOKE.md",
    )
    write_json(output_dir / "package_manifest.json", {
        "phase": "WAVE51_SMOKE_FINAL_PACKAGE_ROOT",
        "files": [
            {"path": relative, "sha256": sha256_file(output_dir / relative)}
            for relative in final_files
        ],
    })


if __name__ == "__main__":
    main()
