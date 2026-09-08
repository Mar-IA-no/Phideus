"""Fabricated timing tables for validator tests, NOT measured profiles."""
import copy
import unittest

from src.atencion_armonica import learned_partition_resources as r
from src.atencion_armonica.shared_partial_data import mechanical_fixture


def fixture_reports():
    obs, _ = mechanical_fixture(4)
    common = {"status": "MEASURED", "namespace": "MECHANICAL_NOT_PROSPECTIVE",
              "seconds": 10., "peak_rss_bytes": 1000}
    geometry = {**common, "observations": [obs, mechanical_fixture(4, deformed=True)[0]],
        "torch_imported": False, "source_validation_seconds": [.001]*2,
        "validation_io": {
            "hash_small": {"bytes": 256, "seconds": [.000001]*3},
            "hash_large": {"bytes": 16*1024**2, "seconds": [.001]*3},
            "inventory": {"entries": 3078, "files": 3075, "seconds": [.001]*3},
            "bundle": {"entries": 3078, "files": 3075, "bytes": 400000, "seconds": [.001]*3},
            "packed": {str(dim): {"scene_count": 512, "dim": dim, "compressed_bytes": 40000,
                "uncompressed_upper_bytes": r.packed_upper_bytes(dim), "seconds": [.001]*3} for dim in (8, 9)}},
        "analysis_seconds": {"selection": .001, "one_test_summary": .001, "one_intervention_summary": .001,
                             "support_batch_seconds": .001},
        "case_statistics": [{"n": 32, "candidate_counts": [32]*3, "group_counts": [63]*3, "bytes": 1000}]*2,
        "case_timings_seconds": [{"features": .0001, "scoring_and_raw_io": .0001,
                                  "load_normalize_and_input_io": .0001}]*2,
        "fit_seconds_by_size": {str(size): [.0001]*3 for size in range(3, 9)},
        "semantic_validation": [{"n": 32, "operations_per_repeat": r.SEMANTIC_OPERATIONS,
            "observation_feature_seconds": [.00001*r.SEMANTIC_OPERATIONS]*3,
            "observation_feature_bytes": 1000, "checkpoints": [{"candidate_count": 32, "group_count": 63,
                "pool_rows_seconds": [.00001*r.SEMANTIC_OPERATIONS]*3,
                "target_metrics_seconds": [.00001*r.SEMANTIC_OPERATIONS]*3,
                "logits_seconds": [.00001*r.SEMANTIC_OPERATIONS]*3,
                "model_inputs_seconds": [.00001*r.SEMANTIC_OPERATIONS]*3,
                "pool_rows_bytes": 1000, "target_metrics_bytes": 1000, "logits_bytes": 1000} for _ in range(3)]} for _ in range(2)],
        "metadata_validation": {"bytes": 1000, "seconds": [.000001]*3},
        "validation_plan": r.validation_plan()}
    geometry["validation_units"] = r.validation_units(geometry)
    geometry.update(r.geometry_projections(geometry))
    reports = {"geometry": geometry}
    for name in ("cpu", "gpu"):
        heads = {}
        for arm, dim, width, count in (("pairs_structure", 8, 33, 1643), ("shared_source", 9, 32, 1650)):
            h = {"parameter_count": count, "setup_seconds": .001, "initial_io_seconds": .001,
                 "update_seconds": [.05 if name == "cpu" else .005]*20,
                 "evaluation_batch_io_seconds": [.0001]*5, "metric_batch_seconds": [.0001]*5,
                 "snapshot_io_seconds": .001,
                 "validation": {"snapshot_chain_seconds": [.001]*3, "snapshot_unique_counts": [11]*3,
                    "snapshot_chain_bytes": 100000, "calibration_write_seconds": .001,
                    "calibration_read_seconds": [.001]*3, "calibration_scene_count": 512,
                    "calibration_candidate_count": 64},
                 "linear_multiply_adds_per_batch": 32*(94*(dim*width+width*16)+64*(94*16+22*32+32*2))}
            h["projected_cell_seconds"] = r.head_projection(h, geometry)
            heads[arm] = h
        value = {**common, "observations": [] if name == "cpu" else [obs],
            "device": "cpu" if name == "cpu" else "NVIDIA GeForce RTX 3090",
            "runtime": {"torch": "fixture_torch", "numpy": "fixture_numpy",
                        "device": "cpu" if name == "cpu" else "NVIDIA GeForce RTX 3090"}, "availability": None,
            "batch_size": 32, "candidate_padding": 64, "group_padding": 94, "heads": heads,
            "workload": dict(r.WORKLOAD), "projected_campaign": r.campaign_projection(heads, geometry),
            "projected_cell_seconds": max(h["projected_cell_seconds"] for h in heads.values())}
        if name == "gpu":
            value["runtime"].update(cuda="12.8", cudnn=90100)
            value["availability"] = {"device": value["device"], "uuid": "GPU-mechanical",
                "used_mib_before": 100., "total_mib": 24576., "compute_processes_before": []}
            value.update(peak_reserved_bytes=1000, per_checkpoint_seconds=[.1]*3,
                         projected_forward_shard_seconds=2*(4*.3+10))
            value["projected_forward_shard_seconds"] = 2*(4*sum(value["per_checkpoint_seconds"])+value["seconds"])
        reports[name] = value
    return reports


class ResourceTests(unittest.TestCase):
    def test_semantic_block_denominator_is_closed_and_cost_uses_all_operations(self):
        geometry = fixture_reports()["geometry"]
        self.assertAlmostEqual(geometry["validation_units"]["semantic_seconds"]["pool_rows"], .00002)
        altered = copy.deepcopy(geometry)
        altered["semantic_validation"][0]["operations_per_repeat"] = 64
        altered["validation_units"] = r.validation_units(altered)
        altered.update(r.geometry_projections(altered))
        with self.assertRaisesRegex(ValueError, "semantic block operation denominator"):
            r.validate_geometry(altered)

    def test_recomputed_costs_cannot_hide_changed_semantic_denominators(self):
        geometry = fixture_reports()["geometry"]
        for field, value in (("candidate_count", 64), ("group_count", 94)):
            altered = copy.deepcopy(geometry)
            for case in altered["semantic_validation"]:
                for row in case["checkpoints"]:
                    row[field] = value
            altered["validation_units"] = r.validation_units(altered)
            altered.update(r.geometry_projections(altered))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "semantic timing denominators"):
                r.validate_geometry(altered)

    def test_ledger_terms_are_source_bound_and_not_double_margined(self):
        values = fixture_reports()
        geometry, gpu = values["geometry"], values["gpu"]
        ledger = geometry["validation_plan"]["ledger"]
        self.assertEqual(ledger["publication"]["state_loads"], 1+sum(3*j+1 for j in range(1, 11)))
        self.assertEqual(sum(v["worker_freezes"]+v["supervisor_freezes"] for v in ledger["test_stages"].values())*4+1, 61)
        self.assertEqual(sum(v["corpora"] for v in ledger["test_stages"].values())*4+2, 126)
        stages = r.stage_projection(gpu["heads"], geometry, forward_shard_seconds=1.)
        for name, stage in stages.items():
            self.assertEqual(stage["worker_seconds"], 2*sum(stage["terms"].values()))
            self.assertEqual(stage["worker_total_seconds"], stage["worker_seconds"]+stage["compute_seconds"])
        for change in ("count", "sha", "unit", "semantic"):
            altered = copy.deepcopy(geometry)
            if change == "count":
                altered["validation_plan"]["ledger"]["publication"]["state_loads"] = 11
            elif change == "sha":
                altered["validation_plan"]["source_sha256"] = "0"*64
            elif change == "unit":
                altered["validation_units"]["cell_pass"]["corpus_pair"] /= 2
            else:
                altered["semantic_validation"][0]["checkpoints"][0]["pool_rows_seconds"].pop()
            with self.assertRaises(ValueError):
                r.validate_geometry(altered)

    def test_validation_primitive_denominators_and_intervention_summary_are_not_optional(self):
        values = fixture_reports()
        geometry = values["geometry"]
        baseline = geometry["projected_analysis_seconds"]
        geometry["analysis_seconds"]["one_intervention_summary"] += 1.
        geometry.update(r.geometry_projections(geometry))
        self.assertAlmostEqual(geometry["projected_analysis_seconds"]-baseline, 16.)
        for mutate in (lambda v: v["hash_large"].update(bytes=1024),
                       lambda v: v["inventory"].update(entries=3075),
                       lambda v: v["bundle"].update(files=3078),
                       lambda v: v["packed"]["9"].update(scene_count=511),
                       lambda v: v["packed"]["8"].update(uncompressed_upper_bytes=40000),
                       lambda v: v["packed"]["9"]["seconds"].pop()):
            changed = copy.deepcopy(geometry["validation_io"])
            mutate(changed)
            with self.assertRaises(ValueError):
                r.validate_validation_io(changed)

    def test_observable_roster_and_runtime_identity_are_closed(self):
        values = fixture_reports()
        for name, mutate in (("geometry", lambda v: v["observations"].pop()),
                ("geometry", lambda v: v["observations"].reverse()),
                ("gpu", lambda v: v.update(observations=[])),
                ("cpu", lambda v: v.update(observations=values["gpu"]["observations"])),
                ("gpu", lambda v: v.update(runtime={})),
                ("gpu", lambda v: v.update(availability=None)),
                ("gpu", lambda v: v["availability"].update(compute_processes_before=[123])),
                ("cpu", lambda v: v.update(availability={})),
                ("gpu", lambda v: v["runtime"].update(device="cpu"))):
            v = copy.deepcopy(values[name])
            mutate(v)
            with self.subTest(profile=name), self.assertRaises(ValueError):
                if name == "geometry":
                    r.validate_geometry(v)
                else:
                    r.validate_training(v, values["geometry"], gpu=name == "gpu")

    def test_complete_derivations(self):
        values = fixture_reports()
        r.validate_geometry(values["geometry"])
        for key in ("cpu", "gpu"):
            r.validate_training(values[key], values["geometry"], gpu=key == "gpu")

    def test_every_required_block_and_extra_field_rejected(self):
        values = fixture_reports()
        for key in values:
            for field in (*values[key], "ambiguous_validation_seconds"):
                v = copy.deepcopy(values[key])
                if field in v:
                    del v[field]
                else:
                    v[field] = 1000
                with self.subTest(profile=key, field=field), self.assertRaises(ValueError):
                    if key == "geometry":
                        r.validate_geometry(v)
                    else:
                        r.validate_training(v, values["geometry"], gpu=key == "gpu")

    def test_metrics_change_total_and_incomplete_heads_or_denominators_reject(self):
        values = fixture_reports()
        gpu, geometry = values["gpu"], values["geometry"]
        before = gpu["projected_cell_seconds"]
        for head in gpu["heads"].values():
            head["metric_batch_seconds"] = [2.]*5
            head["validation"]["calibration_write_seconds"] = 100.
            head["projected_cell_seconds"] = r.head_projection(head, geometry)
        gpu["projected_cell_seconds"] = max(h["projected_cell_seconds"] for h in gpu["heads"].values())
        gpu["projected_campaign"] = r.campaign_projection(gpu["heads"], geometry)
        r.validate_training(gpu, geometry, gpu=True)
        self.assertGreater(gpu["projected_cell_seconds"], before)
        self.assertGreater(gpu["projected_cell_seconds"], values["cpu"]["projected_cell_seconds"])
        for change in ("head", "count", "scalar"):
            v = copy.deepcopy(values["cpu"])
            if change == "head":
                del v["heads"]["shared_source"]
            elif change == "count":
                v["heads"]["shared_source"]["update_seconds"].pop()
            else:
                v["projected_cell_seconds"] = 1.
            with self.assertRaises(ValueError):
                r.validate_training(v, geometry, gpu=False)


if __name__ == "__main__":
    unittest.main()
