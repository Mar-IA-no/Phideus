from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from geometria_proporcional.wave49_schema import default_protocol_config, write_jsonl
from geometria_proporcional.wave52_policy import constrained_regret
from geometria_proporcional.wave51_factored import DualHeadDeepSet
from geometria_proporcional.wave54_joint_set import reference_parameters
from geometria_proporcional.wave56_contextual_gate import disagreement_weights


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = REPO_ROOT / "experiments/geometria_proporcional/_wave56_phase_worker.py"
RUNNER_PATH = REPO_ROOT / "experiments/geometria_proporcional/run_wave56_contextual_gate.py"
PREP_PATH = REPO_ROOT / "experiments/geometria_proporcional/prepare_wave56_fresh.py"
CONFIG_PATH = REPO_ROOT / "experiments/geometria_proporcional/configs/wave56_contextual_gate_fresh.json"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = load_module(WORKER_PATH, "wave56_phase_worker_test")
runner = load_module(RUNNER_PATH, "wave56_runner_test")
prep = load_module(PREP_PATH, "wave56_prep_test")


PHYSICAL_TEST_REQUIREMENTS = os.geteuid() == 0 and shutil.which("setpriv") is not None


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def synthetic_visible_and_truth(
    split: str,
    token_prefix: str,
    n_tokens: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    visible: list[dict[str, object]] = []
    truth: list[dict[str, object]] = []
    latent_x = np.linspace(0.5, 2.0, 8)
    covariance = np.tile(np.diag([0.04, 0.04]), (len(latent_x), 1, 1))
    for index in range(n_tokens):
        fixture_id = f"{split}-fixture-{index:04d}"
        pair_token = f"{token_prefix}-{index:04d}"
        clean_y = latent_x.copy()
        visible.append(
            {
                "schema_version": "wave49-relational-benchmark-v2",
                "fixture_id": fixture_id,
                "split": split,
                "n": len(latent_x),
                "x": latent_x.tolist(),
                "y": clean_y.tolist(),
                "covariance": covariance.tolist(),
                "coordinate_semantics": {
                    "x_scale_to_canonical": 1.0,
                    "y_scale_to_canonical": 1.0,
                    "covariance_knowledge": "full",
                    "calibration_population": "canonical_preserving",
                },
            }
        )
        truth.append(
            {
                "schema_version": "wave49-relational-benchmark-v2",
                "fixture_id": fixture_id,
                "split": split,
                "latent_x": latent_x.tolist(),
                "clean_y": clean_y.tolist(),
                "true_covariance_canonical": covariance.tolist(),
                "family_id": "PROP",
                "generator_params": {"k": 1.0},
                "is_out_of_catalog": False,
                "target_region": "DELIBERATELY_INDISTINGUISHABLE",
                "target_region_basis": "synthetic PROP/AFFINE boundary",
                "design_separation_index": 0.0,
                "pair_token": pair_token,
                "representation": "original",
                "range_mode": "wide",
                "noise_mode": "low_balanced",
                "covariance_mode": "homoscedastic",
                "covariance_knowledge": "full",
                "rho": 0.0,
                "rival_distance_mode": "near",
                "design_stratum": "NEAR_RIVAL",
                "calibration_population": "canonical_preserving",
            }
        )
    return visible, truth


def make_synthetic_checkpoints(wave51: Path, seeds: list[int]) -> None:
    wave51.mkdir(parents=True)
    np.savez(wave51 / "normalizer.npz", mean=np.zeros(6), std=np.ones(6))
    biases = torch.tensor([0.5, 0.2, -0.2, -0.5], dtype=torch.float32)
    for seed in seeds:
        model = DualHeadDeepSet()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.set_head.bias.copy_(biases)
        checkpoint = {
            "model_state": model.state_dict(),
            "seed": seed,
            "output": "sigmoid_only",
        }
        path = wave51 / "checkpoints" / f"seed{seed}__sigmoid_only.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path, _use_new_zipfile_serialization=False)


def make_synthetic_prepared_run(
    run_dir: Path,
    support_dir: Path,
    *,
    split_prefixes: dict[str, str] | None = None,
) -> SimpleNamespace:
    """Create a non-official PREPARED package while retaining the real blind-inference boundary."""
    config = prospective_config()
    config["cpu_threads"] = 1
    config["minimums"].update(
        {
            "gate_fit_tokens": 30,
            "gate_fit_disagreement_rows": 60,
            "gate_select_tokens": 30,
            "gate_select_disagreement_rows": 60,
            "gate_select_shard_tokens": 10,
            "gate_select_shard_disagreement_rows": 20,
            "sealed_monitor_tokens": 30,
            "shuffle_movable_fraction": 0.0,
        }
    )
    support_dir.mkdir(parents=True, exist_ok=True)
    policy_manifest = support_dir / "policy_manifest.json"
    permutations = np.asarray(list(itertools.permutations(range(4))), dtype=np.int64)
    write_json(
        policy_manifest,
        {
            "levels": [-0.3, 0.2, 0.8, 1.5],
            "rank_permutations": permutations.tolist(),
            "groups": [list(range(0, 8)), list(range(8, 16)), list(range(16, 24))],
        },
    )
    wave54_freeze = support_dir / "selection_freeze.json"
    write_json(
        wave54_freeze,
        {
            "selected_models": {
                "joint_full": {"theta": reference_parameters("joint_full").tolist()}
            }
        },
    )
    historical = support_dir / "posterior_state.npz"
    np.savez(historical, pair_token=np.asarray(["historical-only-token"]))
    config["source_binding"]["wave52_policy_manifest_sha256"] = runner.sha256_file(
        policy_manifest
    )
    config["source_binding"]["wave54_selection_freeze_sha256"] = runner.sha256_file(
        wave54_freeze
    )
    config_path = support_dir / "wave56_synthetic_config.json"
    write_json(config_path, config)

    run_dir.mkdir(parents=True)
    benchmark = run_dir / "benchmark"
    protocol = default_protocol_config(smoke=True).to_dict()
    write_json(benchmark / "protocol_config.json", protocol)
    write_json(
        benchmark / "manifest.json",
        {
            "schema_version": protocol["schema_version"],
            "fixture_kind": "synthetic-test-only",
            "generation_key_commitment": "synthetic-generation-commitment",
            "identity_key_commitment": "synthetic-identity-commitment",
            "semantic_commitment_key_commitment": "synthetic-semantic-commitment",
        },
    )
    prefixes = split_prefixes or {split: split for split in ("train", "val", "lockbox")}
    for split in ("train", "val", "lockbox"):
        visible, truth = synthetic_visible_and_truth(split, prefixes[split], 40)
        write_jsonl(benchmark / "visible" / f"{split}.jsonl", visible)
        write_jsonl(benchmark / "sealed" / f"{split}.jsonl", truth)
    manifest = json.loads((benchmark / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"] = {
        f"sealed/{split}.jsonl": {
            "sha256": runner.sha256_file(benchmark / "sealed" / f"{split}.jsonl"),
            "bytes": (benchmark / "sealed" / f"{split}.jsonl").stat().st_size,
        }
        for split in ("train", "val", "lockbox")
    }
    write_json(benchmark / "manifest.json", manifest)
    for path in (benchmark / "sealed").rglob("*"):
        if path.is_file():
            path.chmod(0o600)
    (benchmark / "sealed").chmod(0o700)

    wave51 = support_dir / "wave51"
    make_synthetic_checkpoints(wave51, config["seeds"])
    inference = prep.stage_and_infer(run_dir, benchmark, wave51, config_path, config)
    upstream = [
        {
            "path": str(path.resolve()),
            "sha256": runner.sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in (policy_manifest, wave54_freeze, historical)
    ]
    freeze = {
        "schema_version": config["schema_version"],
        "phase": "prepared-with-blind-inference-before-any-oracle",
        "git_commit": runner.git_head(),
        "config_sha256": runner.sha256_file(config_path),
        "prospective_config": config,
        "sources": runner.execution_source_hashes(config_path),
        "source_bindings": config["source_binding"],
        "upstream": upstream,
        "historical_preflight": {"status": "PASS", "inputs": [upstream[0]]},
        "key_commitments": {"synthetic_test_only": "no-key-material-created"},
        "benchmark_manifest_sha256": runner.sha256_file(benchmark / "manifest.json"),
        "protocol_config_sha256": runner.sha256_file(benchmark / "protocol_config.json"),
        "visible_sha256": {
            split: runner.sha256_file(benchmark / "visible" / f"{split}.jsonl")
            for split in ("train", "val", "lockbox")
        },
        "inference_hashes": inference["inference_hashes"],
        "inference_uid": inference["effective_uid"],
        "inference_gid": inference["effective_gid"],
        "negative_truth_probe": inference["negative_truth_probe"],
        "oracle_materialized": False,
        "authorized_labels_present": False,
        "bundles_present": False,
        "fit_operations": False,
        "physical_splits": config["physical_splits"],
    }
    write_json(run_dir / "preparation_freeze.json", freeze)
    write_json(
        run_dir / "generation_receipt.json",
        {
            "phase": "synthetic-test-preparation",
            "manifest_sha256": freeze["benchmark_manifest_sha256"],
            "visible_sha256": freeze["visible_sha256"],
            "key_commitments": freeze["key_commitments"],
        },
    )
    write_json(
        run_dir / "preparation_receipt.json",
        {
            "phase": "wave56-stage1-preparation-complete",
            "execution_mode": "primary",
            "preparation_freeze_sha256": runner.sha256_file(
                run_dir / "preparation_freeze.json"
            ),
            "generation_receipt_sha256": runner.sha256_file(
                run_dir / "generation_receipt.json"
            ),
            "next_state": "PREPARED",
        },
    )
    return SimpleNamespace(
        run_dir=run_dir,
        config_path=config_path,
        policy_manifest=policy_manifest,
        wave54_freeze=wave54_freeze,
    )


def run_physical_phases(package: SimpleNamespace, *, reference_dir: Path | None = None) -> None:
    for phase in ("fit", "select", "adjudicate"):
        state = runner.run_phase(
            package.run_dir,
            package.config_path,
            phase,
            policy_manifest=package.policy_manifest,
            wave54_selection_freeze=package.wave54_freeze,
            reference_dir=reference_dir if phase == "adjudicate" else None,
            enforce_sources=False,
        )
        assert state == runner.SUCCESS_STATE[phase]


def make_attestation_keys(root: Path) -> tuple[Path, Path]:
    private = root / "attestation_private.pem"
    public = root / "attestation_public.pem"
    subprocess.run(
        ["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(private)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["openssl", "pkey", "-in", str(private), "-pubout", "-out", str(public)],
        check=True,
        capture_output=True,
    )
    return private, public


def make_full_preparation_inputs(root: Path) -> SimpleNamespace:
    """Build small but real generator/inference inputs for the preparation integration test."""
    support = root / "support"
    support.mkdir()
    config = prospective_config()
    config["cpu_threads"] = 1
    config["fresh_benchmark"].update(
        {
            "expected_visible_fixtures_per_split": 520,
            "expected_eligible_pair_tokens_per_split": 120,
        }
    )
    config["minimums"].update(
        {
            "gate_fit_tokens": 10,
            "gate_fit_disagreement_rows": 10,
            "gate_select_tokens": 10,
            "gate_select_disagreement_rows": 10,
            "gate_select_shard_tokens": 1,
            "gate_select_shard_disagreement_rows": 1,
            "sealed_monitor_tokens": 10,
            "shuffle_movable_fraction": 0.0,
        }
    )
    policy_manifest = support / "policy_manifest.json"
    permutations = np.asarray(list(itertools.permutations(range(4))), dtype=np.int64)
    write_json(
        policy_manifest,
        {
            "levels": [-0.3, 0.2, 0.8, 1.5],
            "rank_permutations": permutations.tolist(),
            "groups": [list(range(0, 8)), list(range(8, 16)), list(range(16, 24))],
        },
    )
    wave54_freeze = support / "selection_freeze.json"
    write_json(
        wave54_freeze,
        {
            "selected_models": {
                "joint_full": {"theta": reference_parameters("joint_full").tolist()}
            }
        },
    )
    historical = support / "posterior_state.npz"
    np.savez(historical, pair_token=np.asarray(["historical-only-token"]))
    config["source_binding"]["wave52_policy_manifest_sha256"] = runner.sha256_file(
        policy_manifest
    )
    config["source_binding"]["wave54_selection_freeze_sha256"] = runner.sha256_file(
        wave54_freeze
    )
    config_path = support / "wave56_full_preparation_test.json"
    write_json(config_path, config)
    wave51 = support / "wave51"
    make_synthetic_checkpoints(wave51, config["seeds"])
    private_key, public_key = make_attestation_keys(support)
    upstream = [
        {
            "path": str(path.resolve()),
            "sha256": runner.sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in (policy_manifest, wave54_freeze, historical)
    ]
    contract = {
        "git_commit": runner.git_head(),
        "config_sha256": runner.sha256_file(config_path),
        "prospective_config": config,
        "sources": runner.execution_source_hashes(config_path),
        "upstream": upstream,
        "historical_preflight": {"status": "PASS", "inputs": upstream},
        "source_bindings": config["source_binding"],
    }
    protocol = replace(
        default_protocol_config(smoke=True),
        replicates_per_condition=10,
        rival_distance_modes=("near", "far"),
    )
    return SimpleNamespace(
        config=config,
        config_path=config_path,
        wave51=wave51,
        policy_manifest=policy_manifest,
        wave54_freeze=wave54_freeze,
        private_key=private_key,
        public_key=public_key,
        contract=contract,
        protocol=protocol,
    )


@pytest.fixture(scope="module")
def physical_wave56_runs() -> SimpleNamespace:
    if not PHYSICAL_TEST_REQUIREMENTS:
        pytest.skip("physical Wave 56 integration requires root and setpriv")
    raw = Path(tempfile.mkdtemp(prefix="wave56-physical-tests-", dir="/tmp"))
    try:
        config = prospective_config()
        primary = make_synthetic_prepared_run(
            raw / config["primary_output_name"], raw / "support"
        )
        template = raw / "prepared-template"
        shutil.copytree(primary.run_dir, template)
        run_physical_phases(primary)

        replay_dir = raw / config["replay_output_name"]
        shutil.copytree(template, replay_dir)
        replay_receipt = json.loads(
            (replay_dir / "preparation_receipt.json").read_text(encoding="utf-8")
        )
        replay_receipt.update(
            {
                "execution_mode": "replay",
                "reference_dir": str(primary.run_dir.resolve()),
                "replay_exact": True,
            }
        )
        write_json(replay_dir / "preparation_receipt.json", replay_receipt)
        preparation_checks = prep.compare_preparation(replay_dir, template, prospective_config())
        write_json(
            replay_dir / "preparation_replay.json",
            {"checks": preparation_checks, "all_exact": all(preparation_checks.values())},
        )
        replay = SimpleNamespace(
            run_dir=replay_dir,
            config_path=primary.config_path,
            policy_manifest=primary.policy_manifest,
            wave54_freeze=primary.wave54_freeze,
        )
        run_physical_phases(replay, reference_dir=primary.run_dir)
        yield SimpleNamespace(
            raw=raw,
            primary=primary,
            replay=replay,
            prepared_template=template,
        )
    finally:
        shutil.rmtree(raw, ignore_errors=True)


@pytest.mark.skipif(
    not PHYSICAL_TEST_REQUIREMENTS,
    reason="full preparation integration requires root and setpriv",
)
def test_full_preparation_recovers_escrow_and_replays_exactly(tmp_path: Path) -> None:
    inputs = make_full_preparation_inputs(tmp_path)
    primary = tmp_path / inputs.config["primary_output_name"]
    primary.mkdir(mode=0o700)
    args = SimpleNamespace(
        wave51_dir=inputs.wave51,
        attestation_private_key=inputs.private_key,
        reference_dir=None,
    )
    keys = (b"g" * 32, b"i" * 32, b"c" * 32)

    def crash_after_escrow(point: str, _: Path) -> None:
        if point == "after_escrow":
            raise RuntimeError("simulated crash after durable escrow")

    with pytest.raises(RuntimeError, match="simulated crash"):
        prep.execute_preparation(
            args,
            primary,
            inputs.config_path,
            inputs.config,
            "primary",
            inputs.contract,
            None,
            keys_override=keys,
            protocol_override=inputs.protocol,
            trusted_public_key_path=inputs.public_key,
            crash_hook=crash_after_escrow,
        )
    assert (primary / prep.ESCROW_NAME).is_file()
    assert not (primary / "benchmark").exists()

    failed = prep.archive_output(primary, "failed_test")
    primary.mkdir(mode=0o700)
    recovered_escrow = prep.validate_reused_escrow(failed, inputs.contract)
    prep.execute_preparation(
        args,
        primary,
        inputs.config_path,
        inputs.config,
        "recovery",
        inputs.contract,
        recovered_escrow,
        protocol_override=inputs.protocol,
        trusted_public_key_path=inputs.public_key,
    )
    primary_package = SimpleNamespace(
        run_dir=primary,
        config_path=inputs.config_path,
        policy_manifest=inputs.policy_manifest,
        wave54_freeze=inputs.wave54_freeze,
    )
    run_physical_phases(primary_package)

    replay = tmp_path / inputs.config["replay_output_name"]
    replay.mkdir(mode=0o700)
    replay_args = SimpleNamespace(
        wave51_dir=inputs.wave51,
        attestation_private_key=inputs.private_key,
        reference_dir=primary,
    )
    replay_escrow = prep.validate_reused_escrow(primary, inputs.contract)
    prep.execute_preparation(
        replay_args,
        replay,
        inputs.config_path,
        inputs.config,
        "replay",
        inputs.contract,
        replay_escrow,
        protocol_override=inputs.protocol,
        trusted_public_key_path=inputs.public_key,
    )
    replay_package = SimpleNamespace(
        run_dir=replay,
        config_path=inputs.config_path,
        policy_manifest=inputs.policy_manifest,
        wave54_freeze=inputs.wave54_freeze,
    )
    run_physical_phases(replay_package, reference_dir=primary)

    preparation_replay = json.loads(
        (replay / "preparation_replay.json").read_text(encoding="utf-8")
    )
    phase_replay = json.loads(
        (replay / "phases/adjudicate.complete/replay_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert preparation_replay["all_exact"] is True
    assert phase_replay["all_exact"] is True
    assert (replay / "artifact_manifest.json").is_file()


@pytest.mark.skipif(
    not PHYSICAL_TEST_REQUIREMENTS,
    reason="preparation crash matrix requires root and setpriv",
)
def test_preparation_transaction_recovers_every_durable_crash_boundary(
    tmp_path: Path,
) -> None:
    inputs = make_full_preparation_inputs(tmp_path)
    args = SimpleNamespace(
        wave51_dir=inputs.wave51,
        attestation_private_key=inputs.private_key,
        reference_dir=None,
    )
    keys = (b"g" * 32, b"i" * 32, b"c" * 32)
    crash_points = (
        "before_escrow",
        "after_escrow",
        "after_pre_generation_freeze",
        "after_generation",
        "after_inference",
        "after_preparation_freeze",
    )
    reference: Path | None = None

    for crash_point in crash_points:
        output = tmp_path / f"attempt-{crash_point}"

        def crash(point: str, _: Path, expected: str = crash_point) -> None:
            if point == expected:
                raise RuntimeError(f"simulated crash at {expected}")

        with pytest.raises(RuntimeError, match=f"simulated crash at {crash_point}"):
            prep.run_preparation_transaction(
                args,
                output,
                inputs.config_path,
                inputs.config,
                "primary",
                inputs.contract,
                None,
                force=False,
                keys_override=keys,
                protocol_override=inputs.protocol,
                trusted_public_key_path=inputs.public_key,
                crash_hook=crash,
            )

        failed_attempts = sorted(tmp_path.glob(f"{output.name}.failed_*"))
        assert len(failed_attempts) == 1
        failed = failed_attempts[0]
        failure = json.loads((failed / "FAILURE.json").read_text(encoding="utf-8"))
        assert failure["message"] == f"simulated crash at {crash_point}"
        durable_before_recovery = {
            str(path.relative_to(failed)): runner.sha256_file(path)
            for path in sorted(failed.rglob("*"))
            if path.is_file()
            and str(path.relative_to(failed))
            not in {"FAILURE.json", prep.ESCROW_NAME, "generation_receipt.json"}
        }

        if crash_point == "before_escrow":
            assert failure["escrow_present"] is False
            mode = "primary"
            recovered_escrow = None
            recovery_keys = keys
        else:
            assert failure["escrow_present"] is True
            mode = "recovery"
            recovered_escrow = prep.validate_reused_escrow(failed, inputs.contract)
            recovery_keys = None

        prep.run_preparation_transaction(
            args,
            output,
            inputs.config_path,
            inputs.config,
            mode,
            inputs.contract,
            recovered_escrow,
            force=False,
            keys_override=recovery_keys,
            protocol_override=inputs.protocol,
            trusted_public_key_path=inputs.public_key,
        )
        escrow = prep.read_escrow(output)
        assert escrow["contract"] == inputs.contract
        assert prep.keys_from_escrow(escrow) == keys
        assert json.loads(
            (output / "preparation_receipt.json").read_text(encoding="utf-8")
        )["next_state"] == "PREPARED"
        for relative, expected_sha256 in durable_before_recovery.items():
            recovered = output / relative
            assert recovered.is_file(), f"recovery omitted durable artifact {relative}"
            assert runner.sha256_file(recovered) == expected_sha256

        if reference is None:
            reference = output
        else:
            assert all(prep.compare_preparation(output, reference, inputs.config).values())

        package = SimpleNamespace(
            run_dir=output,
            config_path=inputs.config_path,
            policy_manifest=inputs.policy_manifest,
            wave54_freeze=inputs.wave54_freeze,
        )
        run_physical_phases(package)
        assert runner.current_state(output) == "COMPLETE"


def prospective_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def synthetic_data(n_tokens: int = 120, *, seed: int = 5600) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(seed))
    permutations = np.asarray(list(itertools.permutations(range(4))), dtype=np.int64)
    utilities = np.asarray([-0.3, 0.2, 0.8, 1.5], dtype=np.float64)[permutations]
    token_index = np.arange(n_tokens)[:, None]
    policy_index = np.arange(24)[None, :]
    hard = ((token_index + policy_index) % 4).astype(np.int64)
    disagreement = ((token_index + 2 * policy_index) % 4) != 0
    posterior = np.where(disagreement, (hard + 1 + policy_index % 2) % 4, hard)
    target = np.zeros((n_tokens, 4), dtype=bool)
    target[np.arange(n_tokens), np.arange(n_tokens) % 4] = True
    target[np.arange(n_tokens), (np.arange(n_tokens) + 1 + np.arange(n_tokens) % 2) % 4] = True
    penalty = 1.25
    hard_regret = constrained_regret(hard, target, utilities, penalty)
    posterior_regret = constrained_regret(posterior, target, utilities, penalty)
    gain = hard_regret - posterior_regret
    design = rng.normal(size=(n_tokens, 24, 17))
    design[..., 0] = np.abs(design[..., 0])
    design[..., 1] = gain + rng.normal(scale=0.05, size=gain.shape)
    ensemble = rng.normal(size=(n_tokens, 4))
    per_seed = ensemble[None, ...] + rng.normal(scale=0.1, size=(3, n_tokens, 4))
    hard_set = np.zeros((n_tokens, 4), dtype=bool)
    hard_set[np.arange(n_tokens), np.argmax(ensemble, axis=1)] = True
    tokens = np.asarray([f"pair-{index:04d}" for index in range(n_tokens)])
    data = {
        "pair_token": tokens,
        "cluster_id": tokens.copy(),
        "target": target,
        "per_seed_logits": per_seed,
        "ensemble_logits": ensemble,
        "design_stratum": np.asarray(["NEAR_RIVAL"] * n_tokens),
        "cardinality": np.full(n_tokens, 2, dtype=np.int64),
        "split_role": np.asarray(["synthetic"] * n_tokens),
        "design": design,
        "gain": gain,
        "disagreement": disagreement,
        "weights": disagreement_weights(disagreement),
        "primary": np.ones(n_tokens, dtype=bool),
        "hard_actions": hard,
        "posterior_actions": posterior,
        "hard_set": hard_set,
        "advantage": design[..., 0],
        "absent_support": np.arange(n_tokens) % 3 == 0,
    }
    return data, utilities


def run_synthetic_pipeline(root: Path) -> None:
    config = prospective_config()
    fit_data, utilities = synthetic_data(seed=5601)
    select_data, _ = synthetic_data(seed=5602)
    monitor_data, _ = synthetic_data(seed=5603)
    fit = root / "phases/fit.complete"
    select = root / "phases/select.complete"
    adjudicate = root / "phases/adjudicate.complete"
    root.mkdir(parents=True)
    (root / "phase_request.json").write_text(
        json.dumps({"phase": "fit", "synthetic": True}), encoding="utf-8"
    )
    fit.mkdir(parents=True)
    assert worker.run_fit(root, fit, config, utilities, fit_data) == "FIT_COMPLETE"
    select_stage = root / "select-stage/previous"
    select_stage.mkdir(parents=True)
    (select_stage.parent / "phase_request.json").write_text(
        json.dumps({"phase": "select", "synthetic": True}), encoding="utf-8"
    )
    shutil.copy2(fit / "fit_core.json", select_stage / "fit_core.json")
    assert worker.run_select(select_stage.parent, select, config, utilities, select_data) == "SELECT_COMPLETE"
    adjudicate_stage = root / "adjudicate-stage/previous"
    adjudicate_stage.mkdir(parents=True)
    shutil.copy2(fit / "fit_core.json", adjudicate_stage / "fit_core.json")
    shutil.copy2(fit / "fit_arrays.npz", adjudicate_stage / "fit_arrays.npz")
    shutil.copy2(select / "selection_freeze.json", adjudicate_stage / "selection_freeze.json")
    shutil.copy2(select / "selection_core.json", adjudicate_stage / "selection_core.json")
    shutil.copy2(select / "selection_arrays.npz", adjudicate_stage / "selection_arrays.npz")
    assert worker.run_adjudicate(
        adjudicate_stage.parent, adjudicate, config, utilities, monitor_data
    ) == "COMPLETE"


def test_frozen_config_rejects_model_or_rng_drift() -> None:
    config = prospective_config()
    prep.validate_prospective_config(config)
    worker.validate_frozen_config(config)
    config["primary_model"]["alpha"] = 10.0
    with pytest.raises(RuntimeError, match="alphas drifted"):
        worker.validate_frozen_config(config)


def test_frozen_config_rejects_absent_support_drift() -> None:
    config = prospective_config()
    config["absent_support"]["set_indices"] = [0, 4, 8, 10, 13]
    with pytest.raises(RuntimeError, match="absent-support indices drifted"):
        worker.validate_frozen_config(config)


def test_preparation_rejects_incomplete_diagnostic_contract() -> None:
    config = prospective_config()
    del config["diagnostic_criteria"]["replay_exact_required"]
    with pytest.raises(RuntimeError, match="diagnostic criteria are incomplete"):
        prep.validate_prospective_config(config)


def test_execution_source_manifest_covers_local_runtime_closure() -> None:
    config = prospective_config()
    required = set(config["required_execution_sources"])
    expected = {
        "experiments/geometria_proporcional/run_wave56_retrospective.py",
        "src/geometria_proporcional/wave49_attestation.py",
        "src/geometria_proporcional/wave49_checker.py",
        "src/geometria_proporcional/wave49_generator.py",
        "src/geometria_proporcional/wave49_logic_oracle.py",
        "src/geometria_proporcional/wave49_oracle.py",
        "src/geometria_proporcional/wave49_oracle_reference.py",
        "src/geometria_proporcional/wave49_schema.py",
        "src/geometria_proporcional/wave50_model.py",
        "src/geometria_proporcional/wave50_neural.py",
        "src/geometria_proporcional/wave51_factored.py",
        "src/geometria_proporcional/wave52_policy.py",
        "src/geometria_proporcional/wave53_uncertainty.py",
        "src/geometria_proporcional/wave54_joint_set.py",
        "src/geometria_proporcional/wave55_policy_bridge.py",
        "src/geometria_proporcional/wave56_contextual_gate.py",
    }
    assert expected <= required


def test_atomic_escrow_publication_preserves_mode_and_payload(tmp_path: Path) -> None:
    path = tmp_path / "generation_escrow.json"
    payload = {"phase": "test", "keys": {"not-a-real-key": "00"}}
    digest = prep.atomic_write_json(path, payload, mode=0o600)
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_existing_primary_cannot_redraw_even_with_force(tmp_path: Path) -> None:
    config = prospective_config()
    config["output_parent_relative"] = "."
    output = tmp_path / config["primary_output_name"]
    output.mkdir()
    args = SimpleNamespace(
        replay_secrets_from=None,
        recovery_secrets_from=None,
        reference_dir=None,
        force=True,
    )
    with pytest.raises(FileExistsError, match="unique primary already exists"):
        prep.validate_invocation(args, output, config, repo_root=tmp_path)


@pytest.mark.skipif(os.geteuid() != 0 or shutil.which("setpriv") is None, reason="requires root and setpriv")
def test_setpriv_contract_yields_zero_capabilities() -> None:
    raw = Path(tempfile.mkdtemp(prefix="wave56-security-", dir="/tmp"))
    raw.chmod(0o755)
    try:
        source = raw / "source"
        runner.build_phase_runtime(source)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(source)
        completed = subprocess.run(
            [
                shutil.which("setpriv") or "setpriv",
                "--reuid=65534",
                "--regid=65534",
                "--clear-groups",
                "--no-new-privs",
                sys.executable,
                "-c",
                (
                    "import json;"
                    "import _wave56_phase_worker as worker;"
                    "print(json.dumps(worker.process_security_state(),sort_keys=True))"
                ),
            ],
            cwd=source,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
    finally:
        shutil.rmtree(raw)
    state = json.loads(completed.stdout)
    assert state == {
        "effective_capabilities_hex": "0000000000000000",
        "no_new_privileges": 1,
        "supplementary_groups": [],
    }


def test_shards_use_sha256_least_significant_bit() -> None:
    tokens = np.asarray(["alpha", "beta", "gamma"])
    expected = np.asarray([
        hashlib.sha256((token + "wave56-shard").encode()).digest()[-1] & 1
        for token in tokens
    ])
    np.testing.assert_array_equal(worker.shard_assignment(tokens), expected)


def test_phase_minimums_produce_terminal_not_evaluable(tmp_path: Path) -> None:
    config = prospective_config()
    data, utilities = synthetic_data(20)
    output = tmp_path / "fit"
    output.mkdir()
    assert worker.run_fit(tmp_path, output, config, utilities, data) == "FIT_NOT_EVALUABLE"
    payload = json.loads((output / "fit_not_evaluable.json").read_text(encoding="utf-8"))
    assert set(payload["failed"]) == {"tokens", "disagreement_rows"}
    assert not (output / "fit_core.json").exists()


def test_monitor_not_evaluable_does_not_create_analysis_core(tmp_path: Path) -> None:
    config = prospective_config()
    data, utilities = synthetic_data(20)
    output = tmp_path / "monitor"
    output.mkdir()
    assert worker.run_adjudicate(
        tmp_path, output, config, utilities, data
    ) == "MONITOR_NOT_EVALUABLE"
    assert (output / "monitor_not_evaluable.json").is_file()
    assert not (output / "analysis_core.json").exists()


def test_diagnostic_outcome_has_exactly_six_conditions(tmp_path: Path) -> None:
    pending = tmp_path / "pending"
    pending.mkdir()
    conditions = {f"diagnostic_condition_{index}": True for index in range(1, 6)}
    conditions["diagnostic_condition_6_without_replay"] = True
    (pending / "analysis_core.json").write_text(
        json.dumps(
            {
                "diagnostic_pattern": {"conditions": conditions},
                "selector_stability": {"selector_sensitive": False},
            }
        ),
        encoding="utf-8",
    )
    runner.write_diagnostic_outcome(pending, True)
    outcome = json.loads((pending / "diagnostic_outcome.json").read_text(encoding="utf-8"))
    assert set(outcome["conditions"]) == {
        f"diagnostic_condition_{index}" for index in range(1, 7)
    }
    assert outcome["prospective_pattern_observed"] is True


def test_transaction_state_machine_blocks_after_not_evaluable(tmp_path: Path) -> None:
    (tmp_path / "preparation_freeze.json").write_text("{}", encoding="utf-8")
    assert runner.current_state(tmp_path) == "PREPARED"
    terminal = tmp_path / "phases/fit.not_evaluable"
    terminal.mkdir(parents=True)
    assert runner.current_state(tmp_path) == "FIT_NOT_EVALUABLE"
    with pytest.raises(RuntimeError, match="cannot run select"):
        runner.validate_transition("FIT_NOT_EVALUABLE", "select")


def test_worker_stage_rejects_future_oracle_material(tmp_path: Path) -> None:
    required = [
        "phase_request.json",
        "config.json",
        "protocol_config.json",
        "visible/train.jsonl",
        "labels/train.jsonl",
        "frozen/policy_manifest.json",
        "frozen/wave54_selection_freeze.json",
        "frozen/historical_pair_tokens.json",
        "inference/logits/seed17__train.npz",
    ]
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    (tmp_path / "phase_request.json").write_text(
        json.dumps({"phase": "fit", "split": "train", "role": "gate_fit"}), encoding="utf-8"
    )
    future = tmp_path / "labels/lockbox.jsonl"
    future.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected files"):
        worker.validate_stage(tmp_path, "fit")


def test_synthetic_pipeline_and_exact_replay(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    replay = tmp_path / "replay"
    run_synthetic_pipeline(primary)
    run_synthetic_pipeline(replay)

    comparison = runner.compare_reference(replay, primary)
    assert comparison["all_exact"] is True
    result = replay / "phases/adjudicate.complete/result_arrays.npz"
    with np.load(result, allow_pickle=False) as arrays:
        assert arrays["shuffle_id"].shape == (5,)
        assert arrays["shuffle_score"].shape == (5, 120, 24)
        assert arrays["shuffle_actions"].shape == (5, 120, 24)
        assert arrays["shuffle_metric__regret_by_policy"].shape == (5, 120, 24)
        assert arrays["bootstrap_indices"].shape == (5000, 120)
    core = json.loads(
        (replay / "phases/adjudicate.complete/analysis_core.json").read_text(encoding="utf-8")
    )
    assert set(core["arms"]) == {
        "hard_set_policy",
        "pure_joint_full",
        "scalar_advantage_gate",
        "contextual_value_gate",
        "advantage_only_value_gate",
        "contextual_shuffled_gain",
        "oracle_positive_gain",
    }
    assert core["contextual_shuffled_gain"]["replicate_statuses"] == ["PASS"] * 5


def test_coordinator_cannot_draw_fresh_keys_or_load_truth() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "import secrets" not in source
    assert "generate_benchmark" not in source
    assert "compute_oracle_splits" not in source
    assert "load_labeled_records" not in source


def clone_prepared_package(
    template: Path,
    destination: Path,
    source: SimpleNamespace,
) -> SimpleNamespace:
    shutil.copytree(template, destination)
    return SimpleNamespace(
        run_dir=destination,
        config_path=source.config_path,
        policy_manifest=source.policy_manifest,
        wave54_freeze=source.wave54_freeze,
    )


def test_physical_pipeline_crosses_inference_materializer_setpriv_and_promotions(
    physical_wave56_runs: SimpleNamespace,
) -> None:
    primary = physical_wave56_runs.primary.run_dir
    inference_receipt = json.loads(
        (primary / "inference/access_receipt.json").read_text(encoding="utf-8")
    )
    assert inference_receipt["effective_uid"] == 65534
    assert inference_receipt["process_security"] == {
        "effective_capabilities_hex": "0000000000000000",
        "no_new_privileges": 1,
        "supplementary_groups": [],
    }
    assert inference_receipt["sealed_truth_probe"]["passed"] is True

    for phase, split, role in (
        ("fit", "train", "gate_fit"),
        ("select", "val", "gate_select"),
        ("adjudicate", "lockbox", "sealed_monitor"),
    ):
        complete = primary / "phases" / f"{phase}.complete"
        assert complete.is_dir()
        assert not (primary / "phases" / f"{phase}.pending").exists()
        journal = json.loads((complete / "transaction_journal.json").read_text(encoding="utf-8"))
        assert journal["step"] == "READY_TO_PROMOTE"
        access = json.loads(
            (complete / "analytics.complete/access_receipt.json").read_text(
                encoding="utf-8"
            )
        )
        assert access["effective_uid"] == 65534
        assert access["status"] == runner.SUCCESS_STATE[phase]
        materialization = json.loads(
            (complete / "authorized_labels/materialization_receipt.json").read_text(
                encoding="utf-8"
            )
        )
        assert materialization["split"] == split
        assert materialization["role"] == role
        assert materialization["other_splits_materialized"] is False


def test_physical_replay_is_bound_to_preparation_and_primary(
    physical_wave56_runs: SimpleNamespace,
) -> None:
    replay = physical_wave56_runs.replay.run_dir
    preparation = json.loads((replay / "preparation_replay.json").read_text(encoding="utf-8"))
    phase_replay = json.loads(
        (replay / "phases/adjudicate.complete/replay_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    outcome = json.loads(
        (replay / "phases/adjudicate.complete/diagnostic_outcome.json").read_text(
            encoding="utf-8"
        )
    )
    assert preparation["all_exact"] is True
    assert phase_replay["all_exact"] is True
    assert outcome["replay_exact"] is True
    assert isinstance(outcome["conditions"]["diagnostic_condition_6"], bool)


@pytest.mark.parametrize(
    "target",
    (
        "benchmark_manifest",
        "protocol_config",
        "visible",
        "inference_logits",
        "sealed_truth",
        "explicit_policy_manifest",
        "explicit_wave54_freeze",
    ),
)
def test_frozen_input_tamper_is_rejected_before_pending_or_oracle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    physical_wave56_runs: SimpleNamespace,
    target: str,
) -> None:
    package = clone_prepared_package(
        physical_wave56_runs.prepared_template,
        tmp_path / "tampered-run",
        physical_wave56_runs.primary,
    )
    if target == "benchmark_manifest":
        path = package.run_dir / "benchmark/manifest.json"
        path.write_bytes(path.read_bytes() + b"\n")
    elif target == "protocol_config":
        path = package.run_dir / "benchmark/protocol_config.json"
        path.write_bytes(path.read_bytes() + b"\n")
    elif target == "visible":
        path = package.run_dir / "benchmark/visible/train.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
    elif target == "inference_logits":
        path = package.run_dir / "inference/logits/seed17__train.npz"
        with np.load(path, allow_pickle=False) as source:
            arrays = {name: source[name] for name in source.files}
        arrays["set_logits"] = arrays["set_logits"].copy()
        arrays["set_logits"][0, 0] += 0.125
        np.savez_compressed(path, **arrays)
    elif target == "sealed_truth":
        path = package.run_dir / "benchmark/sealed/train.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["generator_params"] = {"k": 999.0}
        write_jsonl(path, rows)
    elif target == "explicit_policy_manifest":
        path = tmp_path / "policy_manifest.json"
        shutil.copy2(package.policy_manifest, path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["levels"][0] -= 0.125
        write_json(path, payload)
        package.policy_manifest = path
    else:
        path = tmp_path / "selection_freeze.json"
        shutil.copy2(package.wave54_freeze, path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["selected_models"]["joint_full"]["theta"][0] += 0.125
        write_json(path, payload)
        package.wave54_freeze = path

    def transaction_must_not_open(*args: object, **kwargs: object) -> Path:
        pytest.fail(f"transaction opened after frozen {target} changed")

    with pytest.raises(RuntimeError, match="changed|differs|mismatch|frozen"):
        monkeypatch.setattr(runner, "begin_or_resume", transaction_must_not_open)
        if target.startswith("explicit_"):
            runner.run_phase(
                package.run_dir,
                package.config_path,
                "fit",
                policy_manifest=package.policy_manifest,
                wave54_selection_freeze=package.wave54_freeze,
                enforce_sources=False,
            )
        else:
            monkeypatch.setattr(
                runner, "require_sources_at_head", lambda config_path: runner.git_head()
            )
            runner.run_phase(
                package.run_dir,
                package.config_path,
                "fit",
                policy_manifest=package.policy_manifest,
                wave54_selection_freeze=package.wave54_freeze,
                enforce_sources=True,
            )
    assert not (package.run_dir / "phases").exists()


@pytest.mark.parametrize("tamper", ("label", "receipt"))
def test_resume_authenticates_existing_materialization_before_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    physical_wave56_runs: SimpleNamespace,
    tamper: str,
) -> None:
    package = clone_prepared_package(
        physical_wave56_runs.prepared_template,
        tmp_path / "materialized-run",
        physical_wave56_runs.primary,
    )
    pending = runner.begin_or_resume(
        package.run_dir, "fit", runner.git_head(), package.config_path
    )
    labels = runner.materialize_labels(package.run_dir, package.config_path, "fit", pending)
    if tamper == "label":
        path = labels / "train.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        path = labels / "materialization_receipt.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        receipt["authorized_labels_sha256"] = "0" * 64
        write_json(path, receipt)

    def worker_must_not_start(*args: object, **kwargs: object) -> Path:
        pytest.fail(f"worker started with tampered materialization {tamper}")

    monkeypatch.setattr(runner, "build_worker_stage", worker_must_not_start)
    with pytest.raises(RuntimeError, match="materialization|authorized labels|receipt|hash"):
        runner.run_phase(
            package.run_dir,
            package.config_path,
            "fit",
            policy_manifest=package.policy_manifest,
            wave54_selection_freeze=package.wave54_freeze,
            enforce_sources=False,
        )
    journal = json.loads((pending / "transaction_journal.json").read_text(encoding="utf-8"))
    assert journal["step"] == "PENDING_CREATED"


def test_resume_repairs_partial_worker_result_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    physical_wave56_runs: SimpleNamespace,
) -> None:
    package = clone_prepared_package(
        physical_wave56_runs.prepared_template,
        tmp_path / "partial-copy-run",
        physical_wave56_runs.primary,
    )
    real_publish = runner.publish_worker_results

    def interrupt_copy(worker_output: Path, pending: Path) -> Path:
        partial = pending / "analytics.pending"
        partial.mkdir()
        shutil.copy2(worker_output / "access_receipt.json", partial / "access_receipt.json")
        first_payload = next(
            path for path in sorted(worker_output.iterdir()) if path.name != "access_receipt.json"
        )
        shutil.copy2(first_payload, partial / first_payload.name)
        raise RuntimeError("injected crash during worker result copy")

    monkeypatch.setattr(runner, "publish_worker_results", interrupt_copy)
    with pytest.raises(RuntimeError, match="injected crash"):
        runner.run_phase(
            package.run_dir,
            package.config_path,
            "fit",
            policy_manifest=package.policy_manifest,
            wave54_selection_freeze=package.wave54_freeze,
            enforce_sources=False,
        )
    pending = package.run_dir / "phases/fit.pending"
    journal = json.loads((pending / "transaction_journal.json").read_text(encoding="utf-8"))
    assert journal["step"] == "ORACLE_MATERIALIZED"

    monkeypatch.setattr(runner, "publish_worker_results", real_publish)
    state = runner.run_phase(
        package.run_dir,
        package.config_path,
        "fit",
        policy_manifest=package.policy_manifest,
        wave54_selection_freeze=package.wave54_freeze,
        enforce_sources=False,
    )
    assert state == "FIT_COMPLETE"
    assert (
        package.run_dir / "phases/fit.complete/analytics.complete/fit_freeze.json"
    ).is_file()
    assert not pending.exists()


def test_phase_promotion_is_recoverable_before_and_idempotent_after_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    physical_wave56_runs: SimpleNamespace,
) -> None:
    before = clone_prepared_package(
        physical_wave56_runs.prepared_template,
        tmp_path / "before-promotion-run",
        physical_wave56_runs.primary,
    )
    real_promote = runner.promote_phase
    real_run_worker = runner.run_restricted_worker

    def crash_before_promote(*args: object, **kwargs: object) -> Path:
        raise RuntimeError("injected crash before phase promotion")

    monkeypatch.setattr(runner, "promote_phase", crash_before_promote)
    with pytest.raises(RuntimeError, match="before phase promotion"):
        runner.run_phase(
            before.run_dir,
            before.config_path,
            "fit",
            policy_manifest=before.policy_manifest,
            wave54_selection_freeze=before.wave54_freeze,
            enforce_sources=False,
        )
    pending = before.run_dir / "phases/fit.pending"
    journal = json.loads((pending / "transaction_journal.json").read_text(encoding="utf-8"))
    assert journal["step"] == "READY_TO_PROMOTE"
    assert (pending / "analytics.complete/fit_freeze.json").is_file()

    monkeypatch.setattr(runner, "promote_phase", real_promote)

    def worker_must_not_restart(*args: object, **kwargs: object) -> None:
        pytest.fail("worker restarted after READY_TO_PROMOTE")

    monkeypatch.setattr(runner, "run_restricted_worker", worker_must_not_restart)
    assert runner.run_phase(
        before.run_dir,
        before.config_path,
        "fit",
        policy_manifest=before.policy_manifest,
        wave54_selection_freeze=before.wave54_freeze,
        enforce_sources=False,
    ) == "FIT_COMPLETE"
    assert not pending.exists()
    assert (before.run_dir / "phases/fit.complete").is_dir()

    after = clone_prepared_package(
        physical_wave56_runs.prepared_template,
        tmp_path / "after-promotion-run",
        physical_wave56_runs.primary,
    )
    monkeypatch.setattr(runner, "run_restricted_worker", real_run_worker)

    def crash_after_promote(*args: object, **kwargs: object) -> Path:
        destination = real_promote(*args, **kwargs)
        raise RuntimeError("injected crash after phase promotion")

    monkeypatch.setattr(runner, "promote_phase", crash_after_promote)
    with pytest.raises(RuntimeError, match="after phase promotion"):
        runner.run_phase(
            after.run_dir,
            after.config_path,
            "fit",
            policy_manifest=after.policy_manifest,
            wave54_selection_freeze=after.wave54_freeze,
            enforce_sources=False,
        )
    assert not (after.run_dir / "phases/fit.pending").exists()
    assert (after.run_dir / "phases/fit.complete").is_dir()
    monkeypatch.setattr(runner, "promote_phase", real_promote)
    with pytest.raises(RuntimeError, match="cannot run fit"):
        runner.run_phase(
            after.run_dir,
            after.config_path,
            "fit",
            policy_manifest=after.policy_manifest,
            wave54_selection_freeze=after.wave54_freeze,
            enforce_sources=False,
        )


def test_adjudicate_finalization_recovers_around_manifest_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    physical_wave56_runs: SimpleNamespace,
) -> None:
    real_promote = runner.promote_phase
    real_manifest = runner.write_public_artifact_manifest
    real_worker = runner.run_restricted_worker

    def prepared_for_adjudicate(name: str) -> SimpleNamespace:
        package = clone_prepared_package(
            physical_wave56_runs.prepared_template,
            tmp_path / name,
            physical_wave56_runs.primary,
        )
        for phase in ("fit", "select"):
            assert runner.run_phase(
                package.run_dir,
                package.config_path,
                phase,
                policy_manifest=package.policy_manifest,
                wave54_selection_freeze=package.wave54_freeze,
                enforce_sources=False,
            ) == runner.SUCCESS_STATE[phase]
        return package

    before_manifest = prepared_for_adjudicate("crash-after-adjudicate-promotion")

    def crash_after_promote(*args: object, **kwargs: object) -> Path:
        destination = real_promote(*args, **kwargs)
        raise RuntimeError("crash after adjudicate promotion")

    monkeypatch.setattr(runner, "promote_phase", crash_after_promote)
    with pytest.raises(RuntimeError, match="after adjudicate promotion"):
        runner.run_phase(
            before_manifest.run_dir,
            before_manifest.config_path,
            "adjudicate",
            policy_manifest=before_manifest.policy_manifest,
            wave54_selection_freeze=before_manifest.wave54_freeze,
            enforce_sources=False,
        )
    assert runner.current_state(before_manifest.run_dir) == "FINALIZATION_PENDING"
    assert not (before_manifest.run_dir / "artifact_manifest.json").exists()

    def worker_must_not_restart(*args: object, **kwargs: object) -> None:
        pytest.fail("worker restarted while finalizing an already promoted adjudication")

    monkeypatch.setattr(runner, "promote_phase", real_promote)
    monkeypatch.setattr(runner, "run_restricted_worker", worker_must_not_restart)
    assert runner.run_phase(
        before_manifest.run_dir,
        before_manifest.config_path,
        "adjudicate",
        policy_manifest=before_manifest.policy_manifest,
        wave54_selection_freeze=before_manifest.wave54_freeze,
        enforce_sources=False,
    ) == "COMPLETE"
    manifest_path = before_manifest.run_dir / "artifact_manifest.json"
    manifest_before = manifest_path.read_bytes()
    assert runner.current_state(before_manifest.run_dir) == "COMPLETE"
    assert runner.run_phase(
        before_manifest.run_dir,
        before_manifest.config_path,
        "adjudicate",
        policy_manifest=before_manifest.policy_manifest,
        wave54_selection_freeze=before_manifest.wave54_freeze,
        enforce_sources=False,
    ) == "COMPLETE"
    assert manifest_path.read_bytes() == manifest_before

    monkeypatch.setattr(runner, "run_restricted_worker", real_worker)
    after_manifest = prepared_for_adjudicate("crash-after-manifest-publication")

    def crash_after_manifest(run_dir: Path) -> None:
        real_manifest(run_dir)
        raise RuntimeError("crash after manifest publication")

    monkeypatch.setattr(runner, "write_public_artifact_manifest", crash_after_manifest)
    with pytest.raises(RuntimeError, match="after manifest publication"):
        runner.run_phase(
            after_manifest.run_dir,
            after_manifest.config_path,
            "adjudicate",
            policy_manifest=after_manifest.policy_manifest,
            wave54_selection_freeze=after_manifest.wave54_freeze,
            enforce_sources=False,
        )
    published = (after_manifest.run_dir / "artifact_manifest.json").read_bytes()
    assert runner.current_state(after_manifest.run_dir) == "COMPLETE"

    monkeypatch.setattr(runner, "write_public_artifact_manifest", real_manifest)
    monkeypatch.setattr(runner, "run_restricted_worker", worker_must_not_restart)
    assert runner.run_phase(
        after_manifest.run_dir,
        after_manifest.config_path,
        "adjudicate",
        policy_manifest=after_manifest.policy_manifest,
        wave54_selection_freeze=after_manifest.wave54_freeze,
        enforce_sources=False,
    ) == "COMPLETE"
    assert (after_manifest.run_dir / "artifact_manifest.json").read_bytes() == published


def test_non_evaluable_adjudicate_finalization_is_recoverable_and_tamper_evident(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    physical_wave56_runs: SimpleNamespace,
) -> None:
    base = clone_prepared_package(
        physical_wave56_runs.prepared_template,
        tmp_path / "non-evaluable-base",
        physical_wave56_runs.primary,
    )
    config = json.loads(base.config_path.read_text(encoding="utf-8"))
    config["minimums"]["sealed_monitor_tokens"] = 10**9
    config_path = tmp_path / "non-evaluable-config.json"
    write_json(config_path, config)
    base.config_path = config_path
    for phase in ("fit", "select"):
        assert runner.run_phase(
            base.run_dir,
            base.config_path,
            phase,
            policy_manifest=base.policy_manifest,
            wave54_selection_freeze=base.wave54_freeze,
            enforce_sources=False,
        ) == runner.SUCCESS_STATE[phase]

    def clone_ready(name: str) -> SimpleNamespace:
        destination = tmp_path / name
        shutil.copytree(base.run_dir, destination)
        return SimpleNamespace(
            run_dir=destination,
            config_path=base.config_path,
            policy_manifest=base.policy_manifest,
            wave54_freeze=base.wave54_freeze,
        )

    real_promote = runner.promote_phase
    real_manifest = runner.write_public_artifact_manifest
    real_worker = runner.run_restricted_worker

    def crash_after_promote(*args: object, **kwargs: object) -> Path:
        destination = real_promote(*args, **kwargs)
        raise RuntimeError("crash after non-evaluable adjudicate promotion")

    def worker_must_not_restart(*args: object, **kwargs: object) -> None:
        pytest.fail("worker restarted while finalizing non-evaluable adjudication")

    before_manifest = clone_ready("non-evaluable-before-manifest")
    monkeypatch.setattr(runner, "promote_phase", crash_after_promote)
    with pytest.raises(RuntimeError, match="non-evaluable adjudicate promotion"):
        runner.run_phase(
            before_manifest.run_dir,
            before_manifest.config_path,
            "adjudicate",
            policy_manifest=before_manifest.policy_manifest,
            wave54_selection_freeze=before_manifest.wave54_freeze,
            enforce_sources=False,
        )
    assert runner.current_state(before_manifest.run_dir) == "MONITOR_FINALIZATION_PENDING"
    marker = before_manifest.run_dir / (
        "phases/adjudicate.not_evaluable/analytics.complete/monitor_not_evaluable.json"
    )
    assert marker.is_file()
    assert not (marker.parent / "analysis_core.json").exists()
    assert not (before_manifest.run_dir / "artifact_manifest.json").exists()

    monkeypatch.setattr(runner, "promote_phase", real_promote)
    monkeypatch.setattr(runner, "run_restricted_worker", worker_must_not_restart)
    assert runner.run_phase(
        before_manifest.run_dir,
        before_manifest.config_path,
        "adjudicate",
        policy_manifest=before_manifest.policy_manifest,
        wave54_selection_freeze=before_manifest.wave54_freeze,
        enforce_sources=False,
    ) == "MONITOR_NOT_EVALUABLE"
    manifest_path = before_manifest.run_dir / "artifact_manifest.json"
    manifest_before = manifest_path.read_bytes()
    assert runner.current_state(before_manifest.run_dir) == "MONITOR_NOT_EVALUABLE"
    assert runner.run_phase(
        before_manifest.run_dir,
        before_manifest.config_path,
        "adjudicate",
        policy_manifest=before_manifest.policy_manifest,
        wave54_selection_freeze=before_manifest.wave54_freeze,
        enforce_sources=False,
    ) == "MONITOR_NOT_EVALUABLE"
    assert manifest_path.read_bytes() == manifest_before

    after_manifest = clone_ready("non-evaluable-after-manifest")

    def crash_after_manifest(run_dir: Path) -> None:
        real_manifest(run_dir)
        raise RuntimeError("crash after non-evaluable manifest publication")

    monkeypatch.setattr(runner, "write_public_artifact_manifest", crash_after_manifest)
    monkeypatch.setattr(runner, "run_restricted_worker", real_worker)
    with pytest.raises(RuntimeError, match="non-evaluable manifest publication"):
        runner.run_phase(
            after_manifest.run_dir,
            after_manifest.config_path,
            "adjudicate",
            policy_manifest=after_manifest.policy_manifest,
            wave54_selection_freeze=after_manifest.wave54_freeze,
            enforce_sources=False,
        )
    published = (after_manifest.run_dir / "artifact_manifest.json").read_bytes()
    assert runner.current_state(after_manifest.run_dir) == "MONITOR_NOT_EVALUABLE"
    monkeypatch.setattr(runner, "write_public_artifact_manifest", real_manifest)
    monkeypatch.setattr(runner, "run_restricted_worker", worker_must_not_restart)
    assert runner.run_phase(
        after_manifest.run_dir,
        after_manifest.config_path,
        "adjudicate",
        policy_manifest=after_manifest.policy_manifest,
        wave54_selection_freeze=after_manifest.wave54_freeze,
        enforce_sources=False,
    ) == "MONITOR_NOT_EVALUABLE"
    assert (after_manifest.run_dir / "artifact_manifest.json").read_bytes() == published

    tampered = clone_ready("non-evaluable-tamper")
    monkeypatch.setattr(runner, "promote_phase", crash_after_promote)
    monkeypatch.setattr(runner, "run_restricted_worker", real_worker)
    with pytest.raises(RuntimeError, match="non-evaluable adjudicate promotion"):
        runner.run_phase(
            tampered.run_dir,
            tampered.config_path,
            "adjudicate",
            policy_manifest=tampered.policy_manifest,
            wave54_selection_freeze=tampered.wave54_freeze,
            enforce_sources=False,
        )
    tampered_marker = tampered.run_dir / (
        "phases/adjudicate.not_evaluable/analytics.complete/monitor_not_evaluable.json"
    )
    tampered_marker.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(runner, "promote_phase", real_promote)
    monkeypatch.setattr(runner, "run_restricted_worker", worker_must_not_restart)
    with pytest.raises(RuntimeError, match="terminal inventory"):
        runner.run_phase(
            tampered.run_dir,
            tampered.config_path,
            "adjudicate",
            policy_manifest=tampered.policy_manifest,
            wave54_selection_freeze=tampered.wave54_freeze,
            enforce_sources=False,
        )
    assert not (tampered.run_dir / "artifact_manifest.json").exists()


@pytest.mark.parametrize("execution_mode", ("primary", "replay"))
def test_adjudication_rejects_noncanonical_reference_before_opening_monitor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    physical_wave56_runs: SimpleNamespace,
    execution_mode: str,
) -> None:
    source = (
        physical_wave56_runs.primary
        if execution_mode == "primary"
        else physical_wave56_runs.replay
    )
    package = clone_prepared_package(
        source.run_dir,
        tmp_path / source.run_dir.name,
        source,
    )
    noncanonical = tmp_path / "reference-copy"
    shutil.copytree(physical_wave56_runs.primary.run_dir, noncanonical)
    monkeypatch.setattr(
        runner, "require_sources_at_head", lambda config_path: runner.git_head()
    )
    with pytest.raises(ValueError, match="(primary|replay|canonical|reference)"):
        runner.run_phase(
            package.run_dir,
            package.config_path,
            "adjudicate",
            policy_manifest=package.policy_manifest,
            wave54_selection_freeze=package.wave54_freeze,
            reference_dir=noncanonical,
            enforce_sources=True,
        )


def test_physical_splits_are_disjoint_and_overlap_is_rejected(
    tmp_path: Path,
    physical_wave56_runs: SimpleNamespace,
) -> None:
    primary = physical_wave56_runs.primary.run_dir
    phase_tokens: dict[str, set[str]] = {}
    for phase, bundle in (
        ("fit", "gate_fit_bundle.npz"),
        ("select", "gate_select_bundle.npz"),
        ("adjudicate", "sealed_monitor_bundle.npz"),
    ):
        with np.load(
            primary / "phases" / f"{phase}.complete/analytics.complete" / bundle,
            allow_pickle=False,
        ) as data:
            phase_tokens[phase] = set(data["pair_token"].astype(str))
    assert phase_tokens["fit"].isdisjoint(phase_tokens["select"])
    assert phase_tokens["fit"].isdisjoint(phase_tokens["adjudicate"])
    assert phase_tokens["select"].isdisjoint(phase_tokens["adjudicate"])

    package = clone_prepared_package(
        physical_wave56_runs.primary.run_dir,
        tmp_path / "overlapping-run",
        physical_wave56_runs.primary,
    )
    shutil.rmtree(package.run_dir / "phases/select.complete")
    shutil.rmtree(package.run_dir / "phases/adjudicate.complete")
    (package.run_dir / "artifact_manifest.json").unlink()
    truth_path = package.run_dir / "benchmark/sealed/val.jsonl"
    rows = [json.loads(line) for line in truth_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["pair_token"] = sorted(phase_tokens["fit"])[0]
    write_jsonl(truth_path, rows)
    # Keep the synthetic manifest coherent so this test reaches the independent
    # split-disjointness defense rather than stopping at the truth-integrity gate.
    manifest_path = package.run_dir / "benchmark/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["sealed/val.jsonl"] = {
        "sha256": runner.sha256_file(truth_path),
        "bytes": truth_path.stat().st_size,
    }
    write_json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="(overlap|disjoint|pair_token)"):
        runner.run_phase(
            package.run_dir,
            package.config_path,
            "select",
            policy_manifest=package.policy_manifest,
            wave54_selection_freeze=package.wave54_freeze,
            enforce_sources=False,
        )


def test_final_package_manifest_is_integral_and_reanalysis_state_is_preserved(
    physical_wave56_runs: SimpleNamespace,
) -> None:
    replay = physical_wave56_runs.replay.run_dir
    manifest_path = replay / "artifact_manifest.json"
    assert manifest_path.is_file(), "the final manifest must be rooted at the package level"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    required_prefixes = (
        "preparation_freeze.json",
        "preparation_replay.json",
        "inference/",
        "phases/fit.complete/",
        "phases/select.complete/",
        "phases/adjudicate.complete/",
    )
    assert all(any(name == prefix or name.startswith(prefix) for name in manifest) for prefix in required_prefixes)
    for relative, expected in manifest.items():
        path = replay / relative
        assert path.is_file()
        assert expected == {
            "sha256": runner.sha256_file(path),
            "bytes": path.stat().st_size,
        }

    arrays_path = replay / "phases/adjudicate.complete/analytics.complete/result_arrays.npz"
    with np.load(arrays_path, allow_pickle=False) as arrays:
        assert arrays["sealed_monitor__posterior_mass"].shape == (40, 15)
        assert arrays["sealed_monitor__action_risk"].shape == (40, 24, 4)
        assert arrays["gate_fit_archive__gate_fit__posterior_mass"].shape == (40, 15)
        assert arrays["gate_select_archive__gate_select__action_risk"].shape == (40, 24, 4)
