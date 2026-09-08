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
        "analysis_seconds": {"selection": .001, "one_test_summary": .001},
        "case_statistics": [{"n": 32, "candidate_counts": [32]*3, "group_counts": [63]*3, "bytes": 1000}]*2,
        "case_timings_seconds": [{"features": .0001, "scoring_and_raw_io": .0001,
                                  "load_normalize_and_input_io": .0001}]*2,
        "fit_seconds_by_size": {str(size): [.0001]*3 for size in range(3, 9)}}
    geometry.update(r.geometry_projections(geometry))
    reports = {"geometry": geometry}
    for name in ("cpu", "gpu"):
        heads = {}
        for arm, dim, width, count in (("pairs_structure", 8, 33, 1643), ("shared_source", 9, 32, 1650)):
            h = {"parameter_count": count, "setup_seconds": .001, "initial_io_seconds": .001,
                 "update_seconds": [.05 if name == "cpu" else .005]*20,
                 "evaluation_batch_io_seconds": [.0001]*5, "metric_batch_seconds": [.0001]*5,
                 "snapshot_io_seconds": .001,
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
