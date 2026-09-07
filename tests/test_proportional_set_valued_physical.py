from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "experiments/geometria_proporcional/run_proportional_set_valued_physical_preflight.py"
CHECKER_PATH = ROOT / "experiments/geometria_proporcional/check_proportional_set_valued_physical_preflight.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_module("physical_runner_tests", RUNNER_PATH)
checker = load_module("physical_checker_tests", CHECKER_PATH)


class PhysicalPackageTests(unittest.TestCase):
    @staticmethod
    def valid_role() -> dict[str, np.ndarray]:
        seeds = np.zeros((3, 2, 4), dtype="<f8")
        return {
            "pair_token": np.asarray(["a", "b"], dtype="<U64"),
            "cluster_id": np.asarray(["a", "b"], dtype="<U64"),
            "design_stratum": np.asarray(["FAR_RIVAL", "NEAR_RIVAL"], dtype="<U16"),
            "cardinality": np.asarray([1, 1], dtype="<i8"),
            "ensemble_logits": seeds.mean(axis=0),
            "per_seed_logits": seeds,
            "target": np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=bool),
            "split_role": np.asarray(["fixture", "fixture"], dtype="<U24"),
        }

    def test_seven_phase_state_machine_is_total(self) -> None:
        self.assertEqual(len(runner.PHASES), 7)
        self.assertEqual(set(runner.PHASES), set(runner.STATE_AFTER))
        self.assertEqual(runner.STATE_AFTER[runner.PHASES[-1]], "COMPLETE")

    def test_every_phase_has_closed_output_and_handoff(self) -> None:
        self.assertEqual(set(runner.EXPECTED_OUTPUTS), set(runner.PHASES))
        self.assertEqual(set(runner.HANDOFF), set(runner.PHASES[:-1]))
        for phase, names in runner.EXPECTED_OUTPUTS.items():
            self.assertEqual(len(names), len(set(names)), phase)
            self.assertTrue(any(name.endswith("freeze.json") for name in names), phase)

    def test_runtime_allowlist_has_worker_and_five_package_blobs(self) -> None:
        self.assertEqual(len(runner.RUNTIME_SOURCES), 6)
        self.assertEqual(len(runner.RUNTIME_MODULES), 5)
        self.assertNotIn("wave52_policy.py", " ".join(runner.RUNTIME_SOURCES))

    def test_public_and_truth_views_are_narrow(self) -> None:
        full = {
            "pair_token": np.asarray(["a"], dtype="<U64"),
            "design_stratum": np.asarray(["FAR_RIVAL"], dtype="<U16"),
            "cardinality": np.asarray([1], dtype="<i8"),
            "ensemble_logits": np.zeros((1, 4), dtype="<f8"),
            "per_seed_logits": np.zeros((3, 1, 4), dtype="<f8"),
            "target": np.asarray([[True, False, False, False]]),
        }
        self.assertEqual(set(runner.public_view(full)), {"pair_token", "design_stratum", "cardinality", "ensemble_logits", "per_seed_logits"})
        self.assertEqual(set(runner.truth_view(full)), {"pair_token", "target"})

    def test_canonical_role_normalizes_all_dtypes(self) -> None:
        source = {
            "pair_token": np.asarray(["x"]), "cluster_id": np.asarray(["x"]),
            "design_stratum": np.asarray(["FAR_RIVAL"]), "cardinality": np.asarray([1]),
            "ensemble_logits": np.ones((1, 4)), "per_seed_logits": np.ones((3, 1, 4)),
            "target": np.asarray([[1, 0, 0, 0]], dtype=bool),
        }
        result = runner.canonical_role(source, np.asarray([True]), "posterior_fit")
        self.assertEqual(result["pair_token"].dtype, np.dtype("<U64"))
        self.assertEqual(result["design_stratum"].dtype, np.dtype("<U16"))
        self.assertEqual(result["cardinality"].dtype, np.dtype("<i8"))
        self.assertEqual(result["ensemble_logits"].dtype, np.dtype("<f8"))
        self.assertEqual(result["split_role"].dtype, np.dtype("<U24"))

    def test_role_validator_rejects_ensemble_mismatch(self) -> None:
        data = self.valid_role(); data["ensemble_logits"][0, 0] = 1.0
        with self.assertRaisesRegex(RuntimeError, "LOGIT_ENSEMBLE_MISMATCH"):
            runner.validate_role("fixture", data, 2)

    def test_role_validator_rejects_extended_schema_before_execution(self) -> None:
        data = self.valid_role(); data["undeclared"] = np.zeros(2)
        with self.assertRaisesRegex(RuntimeError, "schema drifted"):
            runner.validate_role("fixture", data, 2)

    def test_role_validator_rejects_split_role_before_execution(self) -> None:
        data = self.valid_role(); data["split_role"][0] = "evaluate"
        with self.assertRaisesRegex(RuntimeError, "SPLIT_ROLE_INVALID"):
            runner.validate_role("fixture", data, 2)

    def test_role_validator_rejects_nonfinite_logits_before_execution(self) -> None:
        data = self.valid_role(); data["per_seed_logits"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "nonfinite logits"):
            runner.validate_role("fixture", data, 2)

    def test_role_validator_rejects_target_range_before_execution(self) -> None:
        data = self.valid_role(); data["target"][0] = False; data["cardinality"][0] = 0
        with self.assertRaisesRegex(RuntimeError, "CARDINALITY_TARGET_MISMATCH"):
            runner.validate_role("fixture", data, 2)

    def test_fresh_mode_rejects_before_creating_paths(self) -> None:
        for extras in ([], ["--fresh-signature", "fabricated"], ["--fresh-commitment", "fabricated"], ["--fresh-signature", "fabricated", "--fresh-commitment", "fabricated"]):
            with self.subTest(extras=extras), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                input_path = root / "input"
                output_path = root / "output"
                result = subprocess.run(
                    [str(ROOT / "venv/bin/python"), str(RUNNER_PATH), "--execution-class", "FRESH_PROSPECTIVE", *extras, "--input-package", str(input_path), "--output-dir", str(output_path), "--source-freeze", str(root / "missing.json")],
                    cwd=ROOT, text=True, capture_output=True,
                )
                self.assertEqual(result.returncode, 65)
                self.assertEqual(json.loads(result.stdout)["reason_code"], "FRESH_PROSPECTIVE_NOT_AUTHORIZED_V1")
                self.assertFalse(input_path.exists())
                self.assertFalse(output_path.exists())

    def test_checker_declares_fifteen_distinct_predicates(self) -> None:
        self.assertEqual(len(checker.PREDICATES), 15)
        self.assertEqual(len({name for name, _ in checker.PREDICATES}), 15)
        self.assertEqual(set(checker.REASONS), {name for name, _ in checker.PREDICATES})

    def test_replay_exclusions_are_field_scoped(self) -> None:
        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("excluded_fields", source)
        self.assertNotIn("excluded_directories", source)
        self.assertNotIn('and not path.relative_to(root).as_posix().startswith("journals/")', source)
        self.assertIn("worker_receipt_sha256", source)
        self.assertIn("normalized_json_files", source)

    def test_claim_ceiling_is_not_a_scientific_decision(self) -> None:
        config = json.loads((ROOT / "experiments/geometria_proporcional/configs/proportional_set_valued_physical_preflight_v1.json").read_text())
        self.assertEqual(config["maximum_status"], "PHYSICAL_PROSPECTIVE_PACKAGE_PREFLIGHT_VALID")
        self.assertEqual(config["enabled_execution_class"], "OPENED_DATA_PHYSICAL_PREFLIGHT")

    def test_evidence_receipt_schema_fails_closed_under_required_mutations(self) -> None:
        freeze = ROOT / "experiments/geometria_proporcional/configs/proportional_set_valued_physical_source_freeze_v1.json"
        output = ROOT / "tests/test_proportional_set_valued_physical.py"
        config = json.loads((ROOT / "experiments/geometria_proporcional/configs/proportional_set_valued_physical_preflight_v1.json").read_text())
        row = lambda path: {"bytes": path.stat().st_size, "sha256": checker.sha256_file(path)}
        receipt = {
            "schema_version": "proportional-physical-suite-receipt-v2", "suite": "unit_test", "source_freeze_sha256": checker.sha256_file(freeze), "status": "PASS",
            "argv": ["python"], "inputs": {freeze.relative_to(ROOT).as_posix(): row(freeze)}, "outputs": {output.relative_to(ROOT).as_posix(): row(output)},
            "versions": {"python": sys.version.split()[0], **config["versions"]}, "exit": 0, "wall_seconds": 0.1, "peak_rss_bytes": 1,
            "peak_temporary_bytes": 0, "preserved_bytes_before_receipt": output.stat().st_size, "gpu_used_or_queried": False,
            "stdout_last_json": None, "stdout_sha256": hashlib.sha256(b"").hexdigest(), "stderr_sha256": hashlib.sha256(b"").hexdigest(), "passed": 1, "total": 1,
        }
        checker._validate_short_suite_receipt(receipt, "unit_test", ["python"], [freeze], [output], 1, checker.sha256_file(freeze), config)
        mutations = []
        for path, value in ((["schema_version"], "bad"), (["suite"], "bad"), (["argv"], ["other"]), (["versions", "numpy"], "bad"), (["inputs", freeze.relative_to(ROOT).as_posix(), "sha256"], "0" * 64), (["outputs", output.relative_to(ROOT).as_posix(), "sha256"], "0" * 64), (["peak_temporary_bytes"], 1)):
            mutated = copy.deepcopy(receipt); target = mutated
            for key in path[:-1]: target = target[key]
            target[path[-1]] = value; mutations.append(mutated)
        for mutated in mutations:
            with self.assertRaises(checker.CheckFailure): checker._validate_short_suite_receipt(mutated, "unit_test", ["python"], [freeze], [output], 1, checker.sha256_file(freeze), config)
        valid_stdout = {"status": "PASS", "passed": 15, "total": 15, "checks": [{"id": name, "status": "PASS", "reason_code": None} for name, _ in checker.PREDICATES], "wall_seconds": 0.1, "peak_rss_bytes": 1}
        checker._validate_checker_stdout(valid_stdout)
        for transform in (lambda p: p["checks"].__setitem__(1, copy.deepcopy(p["checks"][0])), lambda p: p["checks"].pop()):
            mutated = copy.deepcopy(valid_stdout); transform(mutated)
            with self.assertRaises(checker.CheckFailure): checker._validate_checker_stdout(mutated)


if __name__ == "__main__":
    unittest.main()
