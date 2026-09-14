"""Selection arithmetic over a complete declared cell roster, no model or data IO."""
import numpy as np
import pytest

from src.atencion_armonica.geometric_decision_selection import ROSTER, select_epochs
from src.atencion_armonica.geometric_decision_core import ARMS


def fixture():
    targets = [np.array([[.0, .0], [.5, .5]], np.float32),
               np.array([[.5, .0], [.0, .0], [.2, .1]], np.float32)]
    targets += [np.empty((0, 2), np.float32) for _ in range(510)]
    energies = {k: np.array([0., 1., 0., 1., 2.], np.float64) for k in ROSTER}
    return targets, energies


def test_one_epoch_per_arm_all_nine_cells_equal_scene_mass():
    targets, energies = fixture()
    # Only one of nine cells improves in epoch10: contributes exactly 1/9,
    # not a selected seed/backbone. Epoch15 ties10 and must lose by age.
    arm = ARMS[0]
    for epoch in (10, 15):
        energies[arm, 2026090721, 2026091491, epoch] = np.array([0., 1., 1., 0., 2.], np.float64)
    result = select_epochs(targets, energies)
    assert result["arms"][arm]["selected_epoch"] == 10
    assert all(result["arms"][a]["selected_epoch"] == 5 for a in ARMS[1:])
    rows = result["arms"][arm]["epochs"]
    assert rows[0]["mean_regret_tD"] == .25
    assert rows[1]["mean_regret_tD"] == pytest.approx((.5*8/9)/2)
    assert result["eligible_scene_ids"] == [0, 1] and len(result["empty_scene_ids"]) == 510


@pytest.mark.parametrize("mutation", ["missing", "epoch0", "float32", "nan", "extent", "short_roster"])
def test_incomplete_or_wrong_calibration_cannot_select(mutation):
    targets, energies = fixture()
    key = ROSTER[0]
    if mutation == "missing":
        energies.pop(key)
    elif mutation == "epoch0":
        energies[(*key[:3], 0)] = energies.pop(key)
    elif mutation == "float32":
        energies[key] = energies[key].astype(np.float32)
    elif mutation == "nan":
        energies[key][0] = np.nan
    elif mutation == "extent":
        energies[key] = energies[key][:-1]
    else:
        targets.pop()
    with pytest.raises(ValueError):
        select_epochs(targets, energies)
