import numpy as np
import pytest

from src.atencion_armonica import measurement_reporting as reporting


def test_scene_first_primary_formulas_and_fixed_bootstrap():
    neural = np.zeros((4, 128, 4, 4, 3, 3), dtype=np.float64)
    extended = np.zeros((4, 128, 4), dtype=np.float64)
    # Distinct cells: averaging all nine, not picking a reader or backbone.
    neural[3, :, 0, 1] = np.arange(9).reshape(3, 3)/16
    neural[3, :, 1, 1] = .75
    neural[3, :, 0, 0] = .25
    neural[3, :, 1, 0] = .5
    extended[3, :, 0], extended[3, :, 1] = .5, .25
    contrasts = reporting.primary_contrasts(neural, extended)
    np.testing.assert_array_equal(contrasts, np.tile([-.25, .5, .25, .5], (128, 1)))
    bootstrap = reporting.bootstrap_primaries(contrasts)
    assert bootstrap["indices"].shape == (10000, 128)
    np.testing.assert_array_equal(bootstrap["indices"], reporting.bootstrap_primaries(contrasts)["indices"])
    for row, expected in zip(bootstrap["report"].values(), [-.25, .5, .25, .5]):
        assert row["mean"] == row["lower"] == row["upper"] == expected


def test_zero_abstentions_stay_in_denominator():
    neural = np.zeros((4, 128, 4, 4, 3, 3), dtype=np.float64)
    extended = np.zeros((4, 128, 4), dtype=np.float64)
    extended[3, 0, 1] = 1
    contrasts = reporting.primary_contrasts(neural, extended)
    assert contrasts[:, 0].mean() == 1/128
    with pytest.raises(ValueError):
        reporting.primary_contrasts(neural[:, :127], extended[:, :127])
