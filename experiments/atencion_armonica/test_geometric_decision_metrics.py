"""Pure partition/score fixtures only: no observations, sampler or test truth."""
from copy import deepcopy

import numpy as np
import pytest

from src.atencion_armonica import geometric_decision_metrics as m

PARTITIONS = [((0, 1, 2, 3), (4, 5, 6, 7)), ((0, 2, 4, 6), (1, 3, 5, 7))]
LABELS = np.array([0, 0, 0, 0, 1, 1, 1, 1], np.int64)


def test_signed_scores_target_arithmetic_and_exact_choice():
    target = m.targets(PARTITIONS, LABELS)
    components = np.array([[-1., -1.], [-1., 0.]], np.float64)
    energy = components.sum(axis=1, dtype=np.float64)
    result = m.describe(PARTITIONS, target, energy, components=components)
    assert result["chosen"] == 0 and result["decision"]["regret_tM"] == 0
    assert result["decision"]["ari"] == 1
    expected = (components-target["u32"].astype(np.float64))**2
    np.testing.assert_array_equal(result["errors"]["squared"], expected)
    assert result["errors"]["mse"] == expected.mean(axis=1).mean()
    assert target["tD"].dtype == np.float64
    assert target["tD"][1] != target["u32"].sum(axis=1, dtype=np.float32)[1] or target["tD"][1] != target["tM"][1]


def test_delivered_components_sum64_not_old_float32_sum():
    target = m.targets(PARTITIONS, LABELS)
    # Arithmetic-only synthetic target: does not claim to be a partition label.
    target["u64"] = np.array([[.5, 2**-25], [0., 0.]])
    target["u32"] = target["u64"].astype(np.float32)
    target["tM"] = target["u64"].sum(axis=1)
    target["tD"] = target["u32"].astype(np.float64).sum(axis=1)
    energy = np.array([-1., 0.])
    result = m.describe(PARTITIONS, target, energy)
    assert result["decision"]["regret_tD"] == .5+2**-25
    assert result["decision"]["regret_tD"] != float(target["u32"].sum(axis=1, dtype=np.float32)[0])
    target["tD"] = target["u32"].sum(axis=1, dtype=np.float32).astype(np.float64)
    with pytest.raises(ValueError, match="sum arithmetic"):
        m.describe(PARTITIONS, target, energy)


def test_exact_ties_and_near_ties_do_not_share_selector():
    target = m.targets(PARTITIONS, LABELS)
    assert m.describe(PARTITIONS, target, np.array([1., 1.]))["chosen"] == 0
    assert m.describe(PARTITIONS, target, np.array([1., 1.-1e-13]))["chosen"] == 1
    with pytest.raises(ValueError, match="canonical"):
        m.describe(PARTITIONS[::-1], target, np.array([1., 1.]))


@pytest.mark.parametrize("count", [0, 1])
def test_empty_and_singleton_preserve_undefined_tau(count):
    ps = PARTITIONS[:count]
    result = m.describe(ps, m.targets(ps, LABELS), np.zeros(count), components=np.zeros((count, 2)))
    assert result["tau"]["value"] is None
    if count == 0:
        assert result["chosen"] is None and result["decision"] is None and result["errors"]["mse"] is None
    else:
        assert result["decision"]["regret_tM"] == 0


def test_all_cooptima_membership_not_just_one_index():
    target = m.targets(PARTITIONS, np.zeros(8, np.int64))
    result = m.describe(PARTITIONS, target, np.array([1., 0.]))
    assert result["chosen"] == 1 and result["oracles"]["tM"]["optima"] == [0, 1]
    assert result["decision"]["optimal_tM"] and result["decision"]["optimal_tD"]


def test_corrupt_prediction_and_target_rejected():
    target = m.targets(PARTITIONS, LABELS)
    with pytest.raises(ValueError, match="preserved energy"):
        m.describe(PARTITIONS, target, np.zeros(2), components=np.ones((2, 2)))
    bad = deepcopy(target)
    bad["tM"] = bad["tM"].astype(np.float32)
    with pytest.raises(ValueError, match="dtype"):
        m.describe(PARTITIONS, bad, np.zeros(2))


def test_primary_scene_first_and_shared_bootstrap_replay():
    regret = np.zeros((3, 8, 3, 3), np.float64)
    regret[1] = np.nan
    regret[0, m.ARMS.index("geometric_mse"), 0, 0] = 9.
    regret[2, m.ARMS.index("geometric_mse")] = 3.
    regret[0, m.ARMS.index("geometric_decision")] = 4.
    regret[2, m.ARMS.index("geometric_decision")] = 8.
    regret[0, m.ARMS.index("decoupled_decision")] = 2.
    regret[2, m.ARMS.index("decoupled_decision")] = 1.
    calls = []
    result = m.primary(regret, np.array([True, False, True]), check=lambda: calls.append(1))
    arrays = result["arrays"]
    np.testing.assert_array_equal(arrays["eligible_scene_ids"], [0, 2])
    np.testing.assert_array_equal(arrays["scene_contrasts"], [[1., 4., 3., 2.], [3., 8., 5., 7.]])
    assert result["summary"]["contrasts"][m.CONTRASTS[0]]["mean"] == 2.
    assert len(calls) == 79
    indices = np.random.Generator(np.random.PCG64(2026091494)).integers(0, 2, (10000, 2), dtype=np.int64)
    np.testing.assert_array_equal(arrays["bootstrap_indices"], indices)
    direct = arrays["scene_contrasts"][indices].mean(axis=1)
    np.testing.assert_array_equal(arrays["bootstrap_distribution"], direct)
    ci = np.quantile(direct, [.00625, .99375], axis=0, method="linear")
    for i, name in enumerate(m.CONTRASTS):
        assert result["summary"]["contrasts"][name]["interval"] == ci[:, i].tolist()


def test_primary_all_empty_is_na_not_zero():
    result = m.primary(np.full((2, 8, 3, 3), np.nan), np.zeros(2, bool))
    assert result["summary"]["eligible_scenes"] == 0
    assert all(v == {"mean": None, "interval": None} for v in result["summary"]["contrasts"].values())
    assert result["arrays"]["bootstrap_indices"].shape == (10000, 0)
    assert np.isnan(result["arrays"]["bootstrap_distribution"]).all()


@pytest.mark.parametrize("kind", ["missing_cell", "fake_empty", "negative", "wrong_roster"])
def test_primary_refuses_silent_support_changes(kind):
    regret, mask = np.zeros((2, 8, 3, 3)), np.ones(2, bool)
    if kind == "missing_cell":
        regret[0, 0, 0, 0] = np.nan
    elif kind == "fake_empty":
        mask[0] = False
    elif kind == "negative":
        regret[0, 0, 0, 0] = -1.
    else:
        regret = regret[:, :7]
    with pytest.raises(ValueError, match="complete72"):
        m.primary(regret, mask)
