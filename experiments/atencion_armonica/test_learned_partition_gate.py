"""Phase/roster rejection fixtures; no data generation or GPU profiling."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.atencion_armonica.test_learned_partition_data import BASE, historical_first
from experiments.atencion_armonica.test_learned_partition_resources import fixture_reports
from src.atencion_armonica import learned_partition_gate as gate
from src.atencion_armonica.structured_source_artifacts import write_json


class GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        BASE.mkdir(parents=True, exist_ok=True)

    def test_partial_kernel_audit_cannot_authorize_campaign(self):
        common = {"fixture": True}
        records = ({"status": "PARTIAL_KERNEL_PASS"},
                   {"status": "PASS", "scope": "PROTOCOL_ONLY_READY_FOR_IMPLEMENTATION"},
                   {"status": "PASS", "scope": "FULL_IMPLEMENTATION", "common": {"other": True},
                    "target": None, "reports": [{"path": "a", "sha256": "b"}]})
        for record in records:
            with patch.object(gate.p, "read_reference", return_value=record):
                with self.assertRaises(ValueError):
                    gate.verify_audit({}, common, scope="FULL_IMPLEMENTATION")

    def test_unauthorized_test_does_not_reach_selection_loader(self):
        with patch.object(gate, "common_binding", return_value={"fixture": True}), \
             patch.object(gate.p, "read_reference", return_value={"status": "TRAIN_CALIBRATION_READY"}):
            with self.assertRaises(PermissionError):
                gate.verify_authorization({}, "iid")

    def test_profiles_choose_fastest_feasible_device_and_require_all_three(self):
        reports = fixture_reports()
        common = {"fixture": True, "runtime": {"torch": "fixture_torch", "numpy": "fixture_numpy"}}
        audit = {"audit": "mechanical"}
        record = {"profiles": dict.fromkeys(reports, {}), "implementation_audit": audit,
                  "training_device": "cuda:0", "projected_disk_bytes": reports["geometry"]["projected_disk_bytes"],
                  "projected_stages": gate.resources.stage_projection(reports["gpu"]["heads"], reports["geometry"],
                      forward_shard_seconds=reports["gpu"]["projected_forward_shard_seconds"])}
        variants = [None, "missing", "wrong_device", "over_budget", "missing_observations", "wrong_shape",
                    "extra", "missing_head", "other_geometry", "runtime", "bad_grant", "missing_fixture",
                    "amended_cell", "cell_exceeded", "amended_score", "score_exceeded",
                    "amended_forward", "forward_exceeded", "amended_inference", "inference_exceeded",
                    "profile_exceeded"]
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            for i, variant in enumerate(variants):
                values, r = copy.deepcopy(reports), copy.deepcopy(record)
                if variant == "missing":
                    r["profiles"].pop("cpu")
                if variant == "wrong_device":
                    r["training_device"] = "cpu"
                if variant == "over_budget":
                    values["gpu"]["projected_cell_seconds"] = 650.
                if variant == "missing_observations":
                    values["gpu"]["observations"] = []
                if variant == "wrong_shape":
                    values["cpu"]["group_padding"] = 12
                if variant == "extra":
                    values["cpu"]["validation_seconds"] = 1000.
                if variant == "missing_head":
                    del values["gpu"]["heads"]["shared_source"]
                if variant == "runtime":
                    values["gpu"]["runtime"]["torch"] = "other_torch"
                if variant == "missing_fixture":
                    values["geometry"]["observations"].pop()
                if variant in ("amended_score", "score_exceeded"):
                    values["geometry"]["fit_seconds_by_size"] = {
                        str(size): [.008 if variant == "amended_score" else .02]*3 for size in range(3, 9)}
                    values["geometry"].update(gate.resources.geometry_projections(values["geometry"]))
                if variant in ("amended_forward", "forward_exceeded"):
                    values["gpu"]["per_checkpoint_seconds"] = [45. if variant == "amended_forward" else 50.]*3
                    values["gpu"]["projected_forward_shard_seconds"] = 2*(
                        4*sum(values["gpu"]["per_checkpoint_seconds"])+values["gpu"]["seconds"])
                if variant == "profile_exceeded":
                    values["geometry"]["seconds"] = 120.001
                if variant and (variant.startswith("amended_") or variant.endswith("_exceeded")):
                    for name in ("cpu", "gpu"):
                        for head in values[name]["heads"].values():
                            if variant in ("amended_cell", "cell_exceeded"):
                                head["update_seconds"] = [.08 if variant == "amended_cell" else .1]*20
                            if variant in ("amended_inference", "inference_exceeded"):
                                head["evaluation_batch_io_seconds"] = [.3 if variant == "amended_inference" else .4]*5
                            head["projected_cell_seconds"] = gate.resources.head_projection(head, values["geometry"])
                        values[name]["projected_cell_seconds"] = max(h["projected_cell_seconds"] for h in values[name]["heads"].values())
                        values[name]["projected_campaign"] = gate.resources.campaign_projection(values[name]["heads"], values["geometry"])
                    selected = "cpu" if values["cpu"]["projected_cell_seconds"] <= values["gpu"]["projected_cell_seconds"] else "gpu"
                    r["training_device"] = "cpu" if selected == "cpu" else "cuda:0"
                    r["projected_stages"] = gate.resources.stage_projection(values[selected]["heads"], values["geometry"],
                        forward_shard_seconds=values["gpu"]["projected_forward_shard_seconds"])
                roots = {}
                for name, report in values.items():
                    root = Path(folder)/f"case{i}_{name}"
                    root.mkdir()
                    write_json(root/"report.json", report)
                    roots[gate.PROFILE_ROLES[name]] = root
                def bundle(ref, role, expected_common):
                    self.assertEqual(expected_common, common)
                    binding = {"common": common, "implementation_audit": audit}
                    if role != gate.PROFILE_ROLES["geometry"]:
                        binding.update(geometry={"different": True} if variant == "other_geometry" else {},
                            gpu_grant={"fixture": "grant"} if role == gate.PROFILE_ROLES["gpu"] else None)
                    return roots[role], {"binding": binding}
                grant = {"status": "DENIED" if variant == "bad_grant" else "AUTHORIZED", "project": "Phideus",
                         "device": "NVIDIA GeForce RTX 3090", "user_directive": "Mechanical fixture only"}
                with patch.object(gate, "_bundle", side_effect=bundle), patch.object(gate.p, "read_reference", return_value=grant):
                    if variant is None or variant.startswith("amended_"):
                        self.assertEqual(gate._profiles(r, common), values)
                    else:
                        with self.assertRaises((ValueError, PermissionError)):
                            gate._profiles(r, common)

    def test_missing_role_and_shard_prefix_rejected_before_data_reads(self):
        with patch.object(gate, "verify_authorization", return_value={"common": {}}), \
             patch.object(gate, "split_fingerprints") as read:
            for split, shard, previous, earlier in (
                ("train", 1, {}, []), ("calibration", 0, {}, []),
                ("iid", 0, {"train": {}}, []), ("train", 0, {"unrelated": {}}, [])):
                with self.assertRaises(ValueError):
                    gate.verify_data_stage({}, split, shard, previous, earlier)
            read.assert_not_called()

    def test_aggregation_cannot_promote_missing_shards(self):
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            output = Path(folder)/"no_output"
            with patch.object(gate, "verify_data_stage") as validate:
                with self.assertRaises(ValueError):
                    gate.aggregate_split(output, "train", authorization={}, previous={}, shards=[{}]*7)
                validate.assert_not_called()
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
