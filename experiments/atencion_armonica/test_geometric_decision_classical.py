"""Raw-precision decisions need not match the delivered scalar's exact ties."""
import numpy as np
import pytest

from src.atencion_armonica import geometric_decision_classical as module


def test_raw_upper_bounds_not_reconstructed_from_delivered_tie():
    ps = [((0, 1, 2, 3), (4, 5, 6, 7)), ((0, 1, 2, 4), (3, 5, 6, 7))]
    fits = [{"status": "FITTED", "partition": p, "branches": {
        b: {"LB": 0., "UB": 1.+(1e-9 if i == 0 else 0.)} for b in module.ge.BRANCHES}}
        for i, p in enumerate(ps)]
    delivered = np.ones((2, 8), np.float32)
    arrays, diagnostic = module.scores(ps, fits, n=8, delivered=delivered)
    assert arrays["extended"][0] > arrays["extended"][1]
    assert diagnostic["choices"] == {"base": 1, "extended": 1, "z": 0, "d": 0}
    assert not diagnostic["extended_vs_z_same_choice"]


def test_empty_classical_universe_is_preserved():
    arrays, diagnostic = module.scores([], [], n=8, delivered=np.empty((0, 8), np.float32))
    assert all(v.shape == (0,) and v.dtype == np.float64 for v in arrays.values())
    assert all(v is None for v in diagnostic["choices"].values())


def test_classical_rejects_extra_precision_at_delivered_port():
    with pytest.raises(ValueError, match="exact delivered"):
        module.scores([], [], n=8, delivered=np.empty((0, 8), np.float64))
