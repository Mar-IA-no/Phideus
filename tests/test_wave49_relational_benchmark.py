"""Integrity tests for the Wave 49 relational benchmark."""

from __future__ import annotations

import ast
import copy
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from geometria_proporcional.wave49_checker import (  # noqa: E402
    ProtocolViolation,
    validate_all,
    validate_oracle_rows,
    validate_sealed_alignment,
    validate_semantic_attestation,
    validate_visible_package,
)
from geometria_proporcional.wave49_generator import generate_benchmark  # noqa: E402
from geometria_proporcional.wave49_schema import (  # noqa: E402
    ProtocolConfig,
    default_protocol_config,
    read_jsonl,
    sha256_file,
)
from geometria_proporcional.wave49_selector import SELECTORS  # noqa: E402


@pytest.fixture(scope="module")
def attestation_keys(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("wave49-attestation")
    private = root / "private.pem"
    public = root / "public.pem"
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


@pytest.fixture(scope="module")
def benchmark(
    tmp_path_factory: pytest.TempPathFactory,
    attestation_keys: tuple[Path, Path],
) -> tuple[Path, object, Path]:
    output = tmp_path_factory.mktemp("wave49") / "benchmark"
    private, public = attestation_keys
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments" / "geometria_proporcional" / "run_wave49_classical.py"),
            "all",
            "--smoke",
            "--output-dir",
            str(output),
            "--attestation-private-key",
            str(private),
            "--attestation-public-key",
            str(public),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return output, default_protocol_config(smoke=True), public


def test_complete_package_passes(benchmark):
    output, config, public = benchmark
    result = validate_all(output, config, {spec.name for spec in SELECTORS}, public)
    assert result["visible_counts"] == {"train": 26, "val": 26, "lockbox": 26}
    assert result["prediction_counts"] == {"train": 104, "val": 104, "lockbox": 104}
    assert result["ledger_events"] == 4


def test_restricted_executor_boundary_is_real(benchmark):
    output, _, _ = benchmark
    receipt = json.loads((output / "predictions" / "access_receipt.json").read_text(encoding="utf-8"))
    assert receipt["effective_uid"] != 0
    assert receipt["sealed_probe_denied"] is True
    assert {row["split"] for row in receipt["operations"]} == {
        "calibration_null", "train", "val", "lockbox"
    }


def test_positive_rescaling_preserves_eiv_predictions(benchmark):
    output, _, _ = benchmark
    truth = read_jsonl(output / "sealed" / "lockbox.jsonl")
    by_pair = {}
    for row in truth:
        if row["representation"] in {"original", "positive_rescale"}:
            key = (row["pair_token"], row["covariance_knowledge"])
            by_pair.setdefault(key, {})[row["representation"]] = row["fixture_id"]
    predictions = {
        (row["fixture_id"], row["selector"]): row
        for row in read_jsonl(output / "predictions" / "lockbox.jsonl")
    }
    for pair in by_pair.values():
        original = predictions[(pair["original"], "catalog_eiv")]
        scaled = predictions[(pair["positive_rescale"], "catalog_eiv")]
        assert original["status"] == scaled["status"]
        assert original["structural_compatible_set"] == scaled["structural_compatible_set"]
        assert original["family_scores"] == pytest.approx(scaled["family_scores"], rel=1e-10, abs=1e-10)


def test_translation_changes_prop_truth_to_affine(benchmark):
    output, _, _ = benchmark
    oracle = read_jsonl(output / "sealed" / "oracle" / "lockbox.jsonl")
    translated = [row for row in oracle if row["representation"] == "origin_translation_break"]
    assert len(translated) == 2
    assert all(row["family_id"] == "AFFINE_OFFSET" for row in translated)
    assert all(row["logical_structural_set"] == ["AFFINE_OFFSET"] for row in translated)
    summary = json.loads((output / "evaluation" / "summary.json").read_text(encoding="utf-8"))
    assert "origin_translation_rupture" in summary["splits"]["lockbox"]["catalog_eiv"]


def test_unknown_correlation_arm_hides_only_public_cross_term(
    tmp_path: Path,
    attestation_keys: tuple[Path, Path],
):
    config = replace(default_protocol_config(smoke=True), correlations=(0.4,))
    output = tmp_path / "correlated"
    generate_benchmark(
        output,
        config,
        attestation_private_key_path=attestation_keys[0],
        trusted_public_key_path=attestation_keys[1],
    )
    validate_visible_package(output, config)
    validate_sealed_alignment(output, config)
    visible = {row["fixture_id"]: row for row in read_jsonl(output / "visible" / "train.jsonl")}
    sealed = {row["fixture_id"]: row for row in read_jsonl(output / "sealed" / "train.jsonl")}
    hidden = [row for row in sealed.values() if row["covariance_knowledge"] == "diagonal_only"]
    assert hidden
    for truth in hidden:
        assert visible[truth["fixture_id"]]["covariance"][0][0][1] == 0.0
        assert truth["true_covariance_canonical"][0][0][1] != 0.0


def test_abstention_is_selection_calibrated_and_keeps_property_channel(benchmark):
    output, _, _ = benchmark
    calibration = json.loads((output / "predictions" / "abstention_calibration.json").read_text(encoding="utf-8"))
    assert calibration["method"] == "split-conformal-null-after-family-and-latent-selection"
    assert calibration["stratification"] == [
        "n", "covariance_knowledge", "calibration_population"
    ]
    assert calibration["sampling_law"] == "iid_uniform_over_declared_factorial_factors"
    assert calibration["coverage_scope"] == "marginal_within_n_covariance_knowledge_population"
    assert calibration["coverage_guarantee"] == "marginal_not_conditional_by_nuisance"
    assert set(calibration["target_population_law"]) == {
        "canonical_preserving", "origin_translation_break"
    }
    assert "false-abstention" in calibration["coverage_statement"]
    finite = calibration["finite_sample"]["12|full|canonical_preserving"]
    assert finite["n"] == 16
    assert finite["rank_1based"] <= finite["n"]
    assert calibration["finite_sample"]["12|diagonal_only|canonical_preserving"]["n"] == 16
    assert calibration["finite_sample"]["12|full|origin_translation_break"]["n"] == 16
    predictions = read_jsonl(output / "predictions" / "lockbox.jsonl")
    abstentions = [row for row in predictions if row["status"] == "ABSTAIN_OUT_OF_CATALOG"]
    assert all(row["property_basis"] == "empirical_observation_independent_of_structural_abstention" for row in abstentions)
    summary = json.loads((output / "evaluation" / "summary.json").read_text(encoding="utf-8"))
    strata = summary["splits"]["lockbox"]["catalog_eiv_abstain"]["by_calibration_stratum"]
    assert strata
    assert sum(row["n"] for row in strata.values()) == 26
    for row in strata.values():
        assert row["n"] == row["n_in_catalog"] + row["n_out_of_catalog"]
        assert row["n_false_abstentions"] <= row["n_in_catalog"]
        assert row["n_correct_ood_abstentions"] <= row["n_ood_expected_abstentions"]


def test_independent_oracle_matches_distance_order(benchmark):
    output, _, _ = benchmark
    rows = read_jsonl(output / "sealed" / "oracle" / "lockbox.jsonl")
    assert rows
    assert all(row["distance_order_match"] for row in rows)
    assert max(row["max_distance_delta"] for row in rows) < 1e-8


def test_oracle_checker_rejects_mutated_compatible_set(benchmark):
    output, config, _ = benchmark
    row = copy.deepcopy(next(
        candidate
        for candidate in read_jsonl(output / "sealed/oracle/lockbox.jsonl")
        if not candidate["is_out_of_catalog"]
    ))
    row["oracle_compatible_set"] = []
    with pytest.raises(ProtocolViolation, match="compatible-set mismatch"):
        validate_oracle_rows(
            [row], config.oracle_compatibility_distance, config.oracle_ood_distance
        )


def test_all_preregistered_mutations_are_rejected(benchmark):
    output, _, _ = benchmark
    rows = read_jsonl(output / "mutations" / "results.jsonl")
    assert len(rows) == 13
    assert all(row["status"] == "REJECTED" for row in rows)
    assert all(row["design_region"] == "PROTOCOL_INVALID" for row in rows)
    coherent_swap = next(row for row in rows if row["mutation_id"] == "M05_family_frontier_swap")
    assert "semantic attestation content mismatch" in coherent_swap["observed_signal"]


def test_target_regions_are_sealed_before_oracle_and_audited_afterward(benchmark):
    output, _, _ = benchmark
    sealed = read_jsonl(output / "sealed" / "lockbox.jsonl")
    oracle = {
        row["fixture_id"]: row
        for row in read_jsonl(output / "sealed" / "oracle" / "lockbox.jsonl")
    }
    assert {row["target_region"] for row in sealed} == {
        "IDENTIFIABLE", "DELIBERATELY_INDISTINGUISHABLE", "OUT_OF_CATALOG"
    }
    assert all(oracle[row["fixture_id"]]["target_region"] == row["target_region"] for row in sealed)
    assert all(isinstance(row["target_region_match"], bool) for row in oracle.values())
    assert all(row["target_region_match"] for row in oracle.values())
    assert max(row["target_design_reference_delta"] for row in oracle.values()) < 1e-8


def test_public_config_does_not_reconstruct_fixtures_and_private_keys_replay_exactly(
    tmp_path: Path,
    attestation_keys: tuple[Path, Path],
):
    config = default_protocol_config(smoke=True)
    first = tmp_path / "first"
    second = tmp_path / "second"
    replay = tmp_path / "replay"
    signing = {
        "attestation_private_key_path": attestation_keys[0],
        "trusted_public_key_path": attestation_keys[1],
    }
    generate_benchmark(first, config, **signing)
    generate_benchmark(second, config, **signing)
    ids_a = {row["fixture_id"] for row in read_jsonl(first / "visible" / "train.jsonl")}
    ids_b = {row["fixture_id"] for row in read_jsonl(second / "visible" / "train.jsonl")}
    assert ids_a.isdisjoint(ids_b)
    assert sha256_file(first / "visible" / "train.jsonl") != sha256_file(second / "visible" / "train.jsonl")

    generation_key = bytes.fromhex(json.loads(
        (first / "sealed" / "generation_secret.json").read_text(encoding="utf-8")
    )["key_hex"])
    identity_key = bytes.fromhex(json.loads(
        (first / "sealed" / "identity_secret.json").read_text(encoding="utf-8")
    )["key_hex"])
    commitment_key = bytes.fromhex(json.loads(
        (first / "sealed" / "semantic_commitment_secret.json").read_text(encoding="utf-8")
    )["key_hex"])
    generate_benchmark(
        replay,
        config,
        generation_key=generation_key,
        identity_key=identity_key,
        commitment_key=commitment_key,
        **signing,
    )
    for relative in json.loads((first / "manifest.json").read_text(encoding="utf-8"))["files"]:
        assert sha256_file(first / relative) == sha256_file(replay / relative)


def test_generation_identity_and_commitment_keys_must_be_distinct(
    tmp_path: Path,
    attestation_keys: tuple[Path, Path],
):
    repeated = b"x" * 32
    with pytest.raises(ValueError, match="must be distinct"):
        generate_benchmark(
            tmp_path / "reused-key",
            default_protocol_config(smoke=True),
            generation_key=repeated,
            identity_key=repeated,
            commitment_key=repeated,
            attestation_private_key_path=attestation_keys[0],
            trusted_public_key_path=attestation_keys[1],
        )


def test_semantic_attestation_rejects_an_untrusted_public_key(
    benchmark,
    tmp_path: Path,
):
    output, _, _ = benchmark
    wrong_private = tmp_path / "wrong-private.pem"
    wrong_public = tmp_path / "wrong-public.pem"
    subprocess.run(
        ["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(wrong_private)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["openssl", "pkey", "-in", str(wrong_private), "-pubout", "-out", str(wrong_public)],
        check=True,
        capture_output=True,
    )
    with pytest.raises(ProtocolViolation, match="trust-root mismatch"):
        validate_semantic_attestation(output, wrong_public)


def test_pilot_config_uses_current_schema_and_calibration_contract():
    path = REPO_ROOT / "experiments" / "geometria_proporcional" / "configs" / "wave49_pilot.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "wave49-relational-benchmark-v2"
    assert payload["calibration_sampling_law"] == "iid_uniform_over_declared_factorial_factors"
    assert payload["calibration_coverage_scope"] == "marginal_within_n_covariance_knowledge_population"
    config = ProtocolConfig.from_dict(payload)
    assert config.schema_version == "wave49-relational-benchmark-v2"
    assert config.calibration_sampling_law == "iid_uniform_over_declared_factorial_factors"
    assert config.calibration_coverage_scope == "marginal_within_n_covariance_knowledge_population"
    missing_schema = dict(payload)
    missing_schema.pop("schema_version")
    with pytest.raises(ValueError, match="explicit schema_version"):
        ProtocolConfig.from_dict(missing_schema)


def test_lineage_sources_are_snapshotted_and_validated(benchmark):
    output, config, public = benchmark
    manifest = json.loads((output / "prediction_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sources"]
    for record in manifest["sources"].values():
        assert (output / record["artifact"]).is_file()
    validate_all(output, config, {spec.name for spec in SELECTORS}, public)


def test_selector_and_worker_do_not_import_truth_modules():
    paths = [
        REPO_ROOT / "src" / "geometria_proporcional" / "wave49_selector.py",
        REPO_ROOT / "experiments" / "geometria_proporcional" / "_wave49_executor_worker.py",
    ]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any("generator" in name or "oracle" in name for name in imports)
