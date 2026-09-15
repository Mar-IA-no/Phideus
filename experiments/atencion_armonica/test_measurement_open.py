import numpy as np
import pytest

from src.atencion_armonica.measurement_open import open_scene, observed_audio
from src.atencion_armonica.measurement_open import cpu_profile, validate_cpu_profile
from src.atencion_armonica.measurement_store import MeasurementStore


def manual_draw(scenario, scene_id, seed):
    ideal = np.log(np.array([100., 200., 300., 400., 170., 340., 510., 680.]))
    center = float(ideal.mean())
    obs = {"scene_id": scene_id, "split_seed": seed, "log_f": (ideal-center).astype(np.float32).tolist()}
    truth = {"scene_id": scene_id, "split_seed": seed, "sigma_cents": 2.,
             "mean_log_f_observed": center, "log_f_ideal": ideal.tolist(), "sensor_log_noise": [0.]*8,
             "source_ids": [0]*4+[1]*4, "partial_indices": [1, 2, 3, 4]*2,
             "permutation": list(range(8)), "sources": [{"manual_fixture": True}]}
    return obs, truth


def test_open_producer_is_closed_to_test_and_reuses_raw_draw(tmp_path):
    store = MeasurementStore(tmp_path/"store", binding={"fixture": "open"})
    with store.exclusive():
        with pytest.raises(PermissionError):
            open_scene(store, "test", "iid", 0, source_snapshot="fixture", check=lambda: None, draw=manual_draw)
        first = open_scene(store, "development", "iid", 0, source_snapshot="fixture", check=lambda: None, draw=manual_draw)
        def forbidden(*args):
            raise AssertionError("draw repeated")
        second = open_scene(store, "development", "iid", 0, source_snapshot="fixture", check=lambda: None, draw=forbidden)
        assert first["raw"][0] == second["raw"][0]
        assert first["waveform"][0] == second["waveform"][0]
        observed = observed_audio(store, first, "nominal", height=-30., prominence=6., check=lambda: None)
        replay = observed_audio(store, second, "nominal", height=-30., prominence=6., check=lambda: None)
        assert observed[0] == replay[0]
        assert len(observed[2]["frequencies"]) == 8
        assert not any("source_ids" in key or "sidecar" in key for key in observed[1])


def test_cpu_profile_full_manual_roster_replays_without_draw(tmp_path):
    from copy import deepcopy
    store = MeasurementStore(tmp_path/"store", binding={"fixture": "profile"})
    with store.exclusive():
        result, arrays = cpu_profile(store, "manual-source", lambda: None, draw=manual_draw)
        validate_cpu_profile(store, "manual-source", result, arrays)
        assert len(result["rows"]) == 16
        assert len(result["render"]) == 4
        assert all(row["n"] >= 0 for row in result["rows"])
        def forbidden(*args):
            raise AssertionError("manual fixture unnecessarily redrawn")
        replay, arrays = cpu_profile(store, "manual-source", lambda: None, draw=forbidden)
        validate_cpu_profile(store, "manual-source", replay, arrays)
        assert [r["detected"] for r in result["rows"]] == [r["detected"] for r in replay["rows"]]
        changed = deepcopy(result)
        changed["rows"][1]["n"] += 1
        with pytest.raises(ValueError, match="detector receipt"):
            validate_cpu_profile(store, "manual-source", changed, {})
        changed = deepcopy(result)
        changed["rows"].pop()
        with pytest.raises(ValueError, match="fixed OPEN roster"):
            validate_cpu_profile(store, "manual-source", changed, {})
        changed = deepcopy(result)
        changed["render"][0]["render_and_persistence_seconds"] = -1.
        with pytest.raises(ValueError, match="duration"):
            validate_cpu_profile(store, "manual-source", changed, {})


def test_calibration_cost_preserves_matching_without_reader_port(tmp_path):
    from src.atencion_armonica.measurement_calibration import cost_record
    store = MeasurementStore(tmp_path/"store", binding={"fixture": "calibration-cost"})
    with store.exclusive():
        scene = open_scene(store, "calibration", "iid", 0, source_snapshot="manual",
                           check=lambda: None, draw=manual_draw)
        ref, value = cost_record(store, scene, "nominal", height=-30., prominence=6., check=lambda: None)
        assert value["cost"]["cost"] >= 0
        assert value["unit"]["role"] == "calibration"
        assert len(value["matching"]["detected_to_emitted"]) == 8
        second, replay = cost_record(store, scene, "nominal", height=-30., prominence=6., check=lambda: None)
        assert ref == second and replay["cost"] == value["cost"]


def test_calibration_visits_exact_grid_and_roster_without_drawing_test(tmp_path, monkeypatch):
    import src.atencion_armonica.measurement_calibration as module
    from src.atencion_armonica.measurement_contract import identity, unit_roster
    calls = []
    def scene(store, role, scenario, sid, **kwargs):
        assert role == "calibration"
        return {"unit": identity(role, scenario, "canonical", sid)}
    def cost(store, scene, condition, *, height, prominence, check):
        unit = {**scene["unit"], "condition": condition}
        calls.append((height, prominence, unit))
        return {"fixture": True}, {"cost": {"cost": float(prominence)}}
    monkeypatch.setattr(module, "open_scene", scene)
    monkeypatch.setattr(module, "cost_record", cost)
    store = MeasurementStore(tmp_path/"store", binding={"fixture": "calibration-roster"})
    with store.exclusive():
        result, arrays = module.calibrate(store, "manual", lambda: None, draw=None)
        assert not arrays and len(calls) == 1728
        assert result["units"] == unit_roster("calibration", audio_only=True)
        assert (result["height"], result["prominence"]) == (-40., 3.)
        assert np.array(result["costs"]).shape == (9, 192)
