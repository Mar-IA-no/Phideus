#!/usr/bin/env python3
"""Measurement campaign entry; only authenticated OPEN CPU profile is wired yet.

No test/CUDA mode is exposed until its complete runner integration is audited.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("snapshot", "profile-cpu"))
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    # Both currently exposed modes are CPU-only, before any scientific import.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    from src.atencion_armonica.measurement_store import MeasurementStore
    from src.atencion_armonica import measurement_snapshot as snapshots
    from src.atencion_armonica.measurement_control import PhaseController, LIMITS
    from src.atencion_armonica.measurement_admission import MeasurementAdmission
    from src.atencion_armonica.measurement_open import cpu_profile, validate_cpu_profile
    from src.atencion_armonica.measurement_resources import ResourceGuard
    from src.atencion_armonica.learned_partition_data import _draw_scene
    store = MeasurementStore(args.root, binding={"schema": "operator-under-measurement-v1"})
    with store.exclusive():
        if args.mode == "snapshot":
            ref = store.publish_json("source-snapshot.json", snapshots.capture())
            print(ref)
            return
        source = store.reference("source-snapshot.json")
        controller = PhaseController(store, limits=LIMITS)
        admission = MeasurementAdmission(controller, verify_sources=lambda ref: snapshots.verify(store, ref))
        guard = ResourceGuard(store.root)
        def produce(check):
            result, arrays = cpu_profile(store, source, check, draw=_draw_scene)
            guard(force=True)
            return result, arrays
        receipt, _, _ = controller.run("profile_cpu", {"phase": "profile_cpu", "source_snapshot": source},
            reservation=600., gpu=False, admit=admission.admit, produce=produce,
            validate=lambda result, arrays: validate_cpu_profile(store, source, result, arrays), resources=guard)
        print({"profile": receipt, "costs": controller.costs(), "resources": guard.observed})


if __name__ == "__main__":
    main()
