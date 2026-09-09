"""Separate one-per-scene and three-per-checkpoint preparation costs.

Reuse the GPU profile's immutable fits; no new sweep, draw, forward or CUDA.
This measures actual three-checkpoint raw caches plus one supervision/factor
archive per scene, rather than multiplying all scene costs by three.
"""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np

from experiments.atencion_armonica.compare_generative_profiles import compare, reference
from experiments.atencion_armonica.profile_generative_evidence import verify_worker_guard
from src.atencion_armonica import generative_evidence as ge
from src.atencion_armonica import generative_evidence_cache as cache
from src.atencion_armonica.generative_evidence_profile import ROOT, source_binding
from src.atencion_armonica.generative_evidence_reuse import OpenReuse
from src.atencion_armonica.generative_evidence_storage import read_arrays, read_scene, write_arrays, write_json, write_scene
from src.atencion_armonica.generative_evidence_supervision import open_truths, candidate_supervision


def measure(cpu, gpu, output):
    verify_worker_guard()
    started = time.monotonic()
    output, gpu = Path(output).resolve(), Path(gpu).resolve()
    base = ROOT/".agent-work/phideus-generative-evidence-20260909"
    if not output.is_relative_to(base) or output == base:
        raise ValueError("owned project temporary output required")
    output.mkdir(exist_ok=False)
    binding = source_binding()
    profile = json.loads((gpu/"report.json").read_bytes())
    if profile["binding"] != binding or profile["status"] != "MEASURED":
        raise ValueError("preserved profile source binding differs")
    reuse = OpenReuse()
    shard = reuse.shard("train", 0)
    truths = open_truths(shard)
    setup_seconds = time.monotonic()-started
    rows = []
    for scene_id in range(32):
        before = time.monotonic()
        fitted = read_scene(gpu/f"scene_{scene_id:05d}.json.gz", profile["fit_scenes"][scene_id]["scene"])
        recovery_seconds = time.monotonic()-before
        before = time.monotonic()
        scene = shard.scene(scene_id)
        input_seconds = time.monotonic()-before
        if fitted["observation"] != scene["observation"] or fitted["inventory"] != json.loads(json.dumps(scene["inventory"])):
            raise ValueError("fit reuse observation/inventory differs")
        before = time.monotonic()
        f, obs = scene["features"], scene["observation"]
        common_rows = [ge.observable_rows(np.asarray(obs["log_f"], np.float32), scene["logits"][cp],
                      f["triples"], f["residual_cents"], scene["partitions"], fitted["fits"]) for cp in ge.CHECKPOINTS]
        supervised = candidate_supervision(scene["partitions"], truths[scene_id]["labels"])
        compute_seconds = time.monotonic()-before
        before = time.monotonic()
        raw_refs, first = {}, None
        for cp, row in zip(ge.CHECKPOINTS, common_rows):
            path = output/f"row_{scene_id:05d}_{cp}.npz"
            arrays = cache.pack_rows("train", cp, [obs], [row], binding=binding)
            ref = write_arrays(path, arrays)
            decoded = cache.unpack_rows(read_arrays(path, ref), binding=binding, split="train", checkpoint_seed=cp)
            if first is None:
                first = decoded
            elif first["identities"] != decoded["identities"]:
                raise ValueError("checkpoint candidate order differs")
            raw_refs[str(cp)] = ref
        target_path = output/f"targets_{scene_id:05d}.npz"
        target_ref = write_arrays(target_path, cache.pack_targets(first["identities"], [supervised], binding=binding))
        cache.unpack_targets(read_arrays(target_path, target_ref), first, binding=binding)
        metric_path = output/f"metrics_{scene_id:05d}.json"
        metric_ref = write_json(metric_path, {"binding": binding, "identity": first["identities"][0],
                                             "metrics": supervised["metrics"]})
        if reference(metric_path)["sha256"] != metric_ref["sha256"]:
            raise ValueError("saved candidate metric bytes differ")
        array_io_seconds = time.monotonic()-before
        before = time.monotonic()
        factor_path = output/f"factors_{scene_id:05d}.json.gz"
        factor_ref = write_scene(factor_path, fitted)
        read_scene(factor_path, factor_ref)
        factor_io_seconds = time.monotonic()-before
        rows.append({"scene_id": scene_id, "input_seconds": input_seconds, "compute_three_cp_and_target_seconds": compute_seconds,
                     "array_target_metric_io_seconds": array_io_seconds, "factor_io_seconds": factor_io_seconds,
                     "profile_fit_recovery_seconds_not_projected": recovery_seconds,
                     "raw_arrays": raw_refs, "targets": target_ref, "metrics": metric_ref, "factors": factor_ref})
    comparison = compare(cpu, gpu)
    for device, estimate in comparison["costs"].items():
        folder = Path(cpu) if device == "cpu" else gpu
        fitted_profile = json.loads((folder/"report.json").read_bytes())
        total = [r["input_seconds"]+r["compute_three_cp_and_target_seconds"]+r["array_target_metric_io_seconds"]
                 +r["factor_io_seconds"]+fit["fit_seconds"] for r, fit in zip(rows, fitted_profile["fit_scenes"])]
        estimate["previous_lumped_preparation_seconds"] = estimate["estimated_open_preparation_seconds"]
        estimate["estimated_open_preparation_seconds"] = 2*(4608*max(total)+9*setup_seconds)
        estimate["estimated_open_storage_bytes"] = 4608*max(
            r["factors"]["bytes"]+r["targets"]["bytes"]+r["metrics"]["bytes"]+sum(v["bytes"] for v in r["raw_arrays"].values())
            for r in rows)
    costs = comparison["costs"]
    comparison["suggested_backends"]["fitting"] = min(costs, key=lambda d: (costs[d]["estimated_open_preparation_seconds"], d != "cpu"))
    comparison["method"] += " Preparation refinement uses measured all-three-CP rows, targets/metrics once and full factors once; recovery read of old profile factors is not a new fitter or projected campaign cost."
    comparison["io_refinement"] = {"source": reference(Path(__file__).resolve()), "binding": binding,
                                    "rows": rows, "setup_seconds": setup_seconds, "reuse": reuse.receipt()}
    if source_binding() != binding:
        raise ValueError("source snapshot changed during I/O measurement")
    comparison["seconds"] = time.monotonic()-started
    comparison["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if comparison["seconds"] > 120 or comparison["peak_rss_bytes"] > 6*1024**3:
        raise RuntimeError("bounded I/O refinement exceeded resources")
    write_json(output/"comparison.json", comparison)
    print(json.dumps({"suggested_backends": comparison["suggested_backends"], "costs": costs,
                      "seconds": comparison["seconds"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    measure(args.cpu, args.gpu, args.output)
