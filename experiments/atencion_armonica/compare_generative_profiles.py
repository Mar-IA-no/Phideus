"""Read preserved profiles and estimate resource costs; no fits or forwards.

Working projections with a factor-two allowance, not worst-case certificates
or a campaign authorization. Scientific recipe and roster are unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from src.atencion_armonica import generative_evidence_cache as cache
from src.atencion_armonica import generative_evidence as ge
from src.atencion_armonica.generative_evidence_profile import ROOT
from src.atencion_armonica.generative_evidence_storage import read_arrays, read_scene, write_json


def reference(path):
    raw = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def compare(cpu, gpu):
    folders = [Path(cpu).resolve(), Path(gpu).resolve()]
    reports = [json.loads((p/"report.json").read_bytes()) for p in folders]
    if reports[0]["binding"] != reports[1]["binding"]:
        raise ValueError("CPU/GPU source/observation bindings differ")
    exits = []
    for p, report, device in zip(folders, reports, ("cpu", "cuda:0")):
        ex = json.loads(p.with_name(p.name+".exit.json").read_bytes())
        launch_path = ROOT/ex["launch"]["path"]
        if hashlib.sha256(launch_path.read_bytes()).hexdigest() != ex["launch"]["sha256"]:
            raise ValueError("supervisor launch bytes differ")
        launch = json.loads(launch_path.read_bytes())
        if (report["status"] != "MEASURED" or report["device"] != device or ex["returncode"] != 0
                or launch["binding"] != report["binding"] or report["seconds"] > 120
                or report["peak_rss_bytes"] > 6*1024**3 or report.get("peak_reserved_bytes", 0) > 6*1024**3
                or [r["scene_id"] for r in report["fit_scenes"]] != list(range(32))
                or set(report["heads"]) != {"envelope", "observed_train"}):
            raise ValueError("incomplete or incompatible supervised profile")
        exits.append(reference(p.with_name(p.name+".exit.json")))
    for k in ("python", "numpy", "torch", "cpu_threads", "deterministic", "tf32"):
        if reports[0]["runtime"][k] != reports[1]["runtime"][k]:
            raise ValueError("shared runtime differs")
    numerical = []
    for scene_id in range(32):
        values, arrays = [], []
        for p, report in zip(folders, reports):
            receipt = report["fit_scenes"][scene_id]
            values.append(read_scene(p/f"scene_{scene_id:05d}.json.gz", receipt["scene"]))
            raw = read_arrays(p/f"row_{scene_id:05d}.npz", receipt["arrays"])
            arrays.append(cache.unpack_rows(raw, binding=report["binding"], split="train", checkpoint_seed=ge.CHECKPOINTS[0]))
        if arrays[0]["identities"] != arrays[1]["identities"] or values[0]["inventory"] != values[1]["inventory"]:
            raise ValueError("CPU/GPU candidate universe differs")
        a, b = arrays[0]["rows"][0], arrays[1]["rows"][0]
        if any(not np.array_equal(a[k], b[k]) for k in ("groups", "globals", "incidence", "available")):
            raise ValueError("CPU/GPU common observable inputs differ")
        bounds = []
        for fa, fb in zip(values[0]["fits"], values[1]["fits"]):
            if fa["partition"] != fb["partition"] or set(fa["branches"]) != set(fb["branches"]):
                raise ValueError("fit signatures/branch support differ")
            bounds.extend(abs(fa["branches"][branch][key]-fb["branches"][branch][key])
                          for branch in fa["branches"] for key in ("LB", "UB"))
        numerical.append({"scene_id": scene_id, "candidates": len(a["partitions"]),
                          "max_abs_bound_difference": max(bounds, default=0.),
                          "max_abs_raw_channel_difference": float(np.max(np.abs(a["evidence"]-b["evidence"]), initial=0.)),
                          "common_inputs_bit_identical": True})
    io_seconds = []
    for _ in range(3):
        start = time.monotonic()
        for row in reports[0]["fit_scenes"]:
            raw = read_arrays(folders[0]/f"row_{row['scene_id']:05d}.npz", row["arrays"])
            cache.unpack_rows(raw, binding=reports[0]["binding"], split="train", checkpoint_seed=ge.CHECKPOINTS[0])
        io_seconds.append(time.monotonic()-start)
    costs = {}
    for report in reports:
        heads, scenes = list(report["heads"].values()), report["fit_scenes"]
        # Use the slower measured envelope/observed batch for the capacity check.
        update = max(max(h["steady_update_seconds"]) for h in heads)
        evaluation = max(max(h["evaluation_batch_io_seconds"]) for h in heads)
        snapshot = max(h["snapshot_io_seconds"] for h in heads)
        setup = max(h["setup_seconds"] for h in heads)
        # Double raw-array loading accounts conservatively for separate targets;
        # no factor JSON is loaded per batch or cell. Real loader is still pending.
        load_cell = 2*(4096+512)/32*max(io_seconds)
        cell = 2*(setup+load_cell+50*128*update+10*16*evaluation+11*snapshot)
        per_scene = [r["input_seconds"]+r["fit_seconds"]+3*r["features_supervision_io_validation_seconds"] for r in scenes]
        prep = 2*(4608*max(per_scene)+9*report["reuse_setup_seconds"])
        costs[report["device"]] = {"fit_seconds_32": sum(r["fit_seconds"] for r in scenes),
            "profile_seconds": report["seconds"], "max_update_seconds": update,
            "max_evaluation_io_seconds": evaluation, "estimated_load_cell_seconds": load_cell,
            "estimated_cell_seconds": cell, "estimated_27_train_cal_seconds": 27*cell,
            "estimated_open_preparation_seconds": prep,
            "estimated_open_storage_bytes": 4608*max(r["scene"]["bytes"]+3*r["arrays"]["bytes"] for r in scenes)}
    fit_device = min(costs, key=lambda d: (costs[d]["estimated_open_preparation_seconds"], d != "cpu"))
    head_device = min(costs, key=lambda d: (costs[d]["estimated_27_train_cal_seconds"], d != "cpu"))
    return {"status": "RESOURCE_ESTIMATE_NOT_CAMPAIGN_AUTHORIZATION", "profiles": [reference(p/"report.json") for p in folders],
            "supervisor_exits": exits, "comparison_source": reference(Path(__file__).resolve()),
            "numerical": numerical, "io_32_rows_seconds": io_seconds, "costs": costs,
            "suggested_backends": {"fitting": fit_device, "head": head_device},
            "method": "2x allowance; max measured envelope/observed head batch; 50x128 updates, 10x16 cal batches, 11 snapshots, raw+target load per cell; 27 cells. Preparation max observed scene with 3x feature/supervision/IO term and 4608 scenes.",
            "limits": ["32 OPEN scenes are a cost sample, not a worst-case fitting proof",
                       "real compact loader, full TRAIN normalization and campaign runner remain to be verified",
                       "fresh-test preparation/forward/evaluation/replay budget needs its own complete projection",
                       "no scientific comparison of trained models or GO/NO-GO"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare(args.cpu, args.gpu)
    write_json(args.output, result)
    print(json.dumps({"suggested_backends": result["suggested_backends"], "costs": result["costs"]}, indent=2))
