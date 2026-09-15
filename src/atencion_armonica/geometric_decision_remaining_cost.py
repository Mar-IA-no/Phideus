"""Pure arithmetic for immutable closing-profile admission; no truth reader.

The original profile operator is frozen by its completed receipt. Keep its
formula byte-for-byte equivalent in result, without importing that operator.
"""
import math

UNITS = {"admission-setup": (1, "authenticated_preparation"),
    "open-truth": (16, "TRAIN_parses_from_two_complete_512row_files"),
    "known-draw-io": (16, "identical_TRAIN_reconstruction"),
    "original-metrics": (16, "scene_144heads"), "probe-metrics": (4, "probe_144heads"),
    "original-replay": (16, "scene_144heads"), "probe-replay": (4, "probe_144heads"),
    "bootstrap512": (512, "repeated_metric_fixture_not_observations"),
    "observable-inventory": (20, "original_plus_probe_file_tree")}


def projection(timings, observed, *, overhead, profile_bytes):
    if (set(timings) != set(UNITS) or any(type(t) not in (float, int) or not math.isfinite(t) or t <= 0 for t in timings.values())
            or not math.isfinite(overhead) or overhead < 0):
        raise ValueError("remaining-cost projection needs all finite phases and overhead")
    scale, margin = 2048/16, 1.25
    inventory = timings["observable-inventory"]*scale
    setup = timings["admission-setup"]+overhead
    fresh_extra = margin*(4*scale*timings["known-draw-io"]+inventory+4*setup)
    eval_extra = margin*(scale*(timings["original-metrics"]+timings["original-replay"]+timings["open-truth"])
        +4*(timings["probe-metrics"]+timings["probe-replay"])+5*inventory+2*timings["bootstrap512"]+4*setup)
    return {"schema": "geometric-decision-closing-projection-v1", "margin": margin,
        "fresh_scenes": 2048, "original_profile_scenes": 16, "probe_profile_scenes": 4,
        "fresh_seconds": observed["observed_path_with_closing_seconds"]+observed["observable_recovery_with_closing_seconds"]+fresh_extra,
        "evaluation_seconds": observed["observable_recovery_with_closing_seconds"]+eval_extra,
        "projected_new_bytes": observed["projected_profile_bytes"]+math.ceil(margin*profile_bytes*scale),
        "fresh_additional_seconds": fresh_extra, "evaluation_additional_seconds": eval_extra,
        "preseal_recovery_count": 1, "postseal_recovery_count": 1, "evaluation_inventory_count": 5,
        "units": {k: {"count": n, "kind": kind} for k, (n, kind) in UNITS.items()},
        "not_a_worst_case_bound": True, "test_authority": False,
        "closing_tail": "add this operator finish minus forecast snapshot before admission"}


def allocate_recovery(forecast, observed, timings):
    """Explicit pre-test amendment; move work, never omit or reset its cost."""
    transfer = observed["observable_recovery_with_closing_seconds"]+1.25*128*timings["observable-inventory"]
    if forecast["evaluation_seconds"] <= transfer:
        raise ValueError("recovery allocation loses metric budget")
    return {**forecast, "accounting": "observable-recovery-in-fresh-v1",
        "original_fresh_seconds": forecast["fresh_seconds"],
        "original_evaluation_seconds": forecast["evaluation_seconds"],
        "fresh_seconds": forecast["fresh_seconds"]+transfer,
        "evaluation_seconds": forecast["evaluation_seconds"]-transfer,
        "transferred_postseal_recovery_seconds": transfer,
        "operation_seconds": {"prospective-observables": forecast["fresh_seconds"],
            "observable-replay": transfer, "metrics-and-replay": forecast["evaluation_seconds"]-transfer},
        "evaluation_inventory_count": 4, "postseal_recovery_inventory_count": 1}
