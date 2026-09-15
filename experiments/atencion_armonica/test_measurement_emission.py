from copy import deepcopy

import numpy as np
import pytest

from src.atencion_armonica.measurement_contract import identity
from src.atencion_armonica.measurement_emission import emission_from_draw, audio_from_emission


def fixture():
    unit = identity("development", "iid", "canonical", 0)
    ideal = np.log(np.array([100., 200., 300., 400., 170., 340., 510., 680.]))
    detuning = np.arange(8)*.0001
    center = float((ideal+detuning).mean())
    obs = {"scene_id": 0, "split_seed": unit["split_seed"],
           "log_f": (ideal+detuning-center).astype(np.float32).tolist()}
    sidecar = {"scene_id": 0, "split_seed": unit["split_seed"], "sigma_cents": 2.,
               "mean_log_f_observed": center, "log_f_ideal": ideal.tolist(),
               "sensor_log_noise": detuning.tolist(), "source_ids": [0]*4+[1]*4,
               "partial_indices": [1, 2, 3, 4]*2, "permutation": list(range(8)),
               "sources": [{"manual_fixture": True}]}
    return unit, obs, sidecar


def test_emission_absolute_scale_and_no_second_noise():
    unit, obs, sidecar = fixture()
    emission, truth = emission_from_draw(unit, obs, sidecar)
    expected = np.exp(np.array(sidecar["log_f_ideal"])+sidecar["sensor_log_noise"])
    np.testing.assert_array_equal(emission["arrays"]["frequencies"], expected)
    np.testing.assert_array_equal(emission["arrays"]["canonical_q32"], np.array(obs["log_f"], np.float32))
    assert "sidecar" not in emission and "source_ids" not in emission["metadata"]
    assert truth["sidecar"] == sidecar
    assert np.min(expected) > 99  # Absolute Hz, not exp(centered q32).


def test_parameter_draw_order_and_repeatable_waveform():
    unit, obs, sidecar = fixture()
    emission, _ = emission_from_draw(unit, obs, sidecar)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([2026091540, unit["split_seed"], 0])))
    for key, expected in (("amplitude_db", rng.uniform(-12., 0., 8)),
                          ("phases", rng.uniform(0., 2*np.pi, 8)), ("noise", rng.normal(size=24000))):
        np.testing.assert_array_equal(emission["arrays"][key], expected)
    first = audio_from_emission(emission)
    second = audio_from_emission(emission_from_draw(unit, obs, sidecar)[0])
    assert first[0] == second[0]
    for key in first[1]:
        np.testing.assert_array_equal(first[1][key], second[1][key])


def test_truth_relabel_does_not_change_render_input():
    unit, obs, sidecar = fixture()
    before = emission_from_draw(unit, obs, sidecar)[0]
    changed = deepcopy(sidecar)
    changed["source_ids"] = [1-x for x in changed["source_ids"]]
    after = emission_from_draw(unit, obs, changed)[0]
    assert before["metadata"] == after["metadata"]
    for key in before["arrays"]:
        np.testing.assert_array_equal(before["arrays"][key], after["arrays"][key])
    with pytest.raises(ValueError):
        audio_from_emission({**before, "sidecar": sidecar})


def test_wrong_alignment_rejected():
    unit, obs, sidecar = fixture()
    sidecar["sensor_log_noise"][0] += .01
    with pytest.raises(ValueError, match="original delivered"):
        emission_from_draw(unit, obs, sidecar)
