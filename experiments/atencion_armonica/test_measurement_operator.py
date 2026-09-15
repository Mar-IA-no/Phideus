import numpy as np
import pytest

from src.atencion_armonica import measurement_contract as contract
from src.atencion_armonica import measurement_operator as operator


@pytest.mark.parametrize("n", [0, 7, 33])
def test_outside_domain_stops_before_features_and_assembly(n, monkeypatch):
    from src.atencion_armonica import partial_compatibility_cache as features
    from src.atencion_armonica import geometric_decision_observables as kernels
    def forbidden(*args, **kwargs):
        raise AssertionError("a frozen kernel was invoked outside its domain")
    monkeypatch.setattr(features, "feature_record", forbidden)
    monkeypatch.setattr(kernels, "scene_from_sources", forbidden)
    prepared = operator.prepare_input(contract.identity("development", "iid", "nominal", 0),
                                       np.arange(n)+100.)
    assert operator.features_for_input(prepared) is None
    row = operator.assemble_observable(prepared, None, None)
    assert row["status"] == "OUTSIDE_OPERATOR_DOMAIN" and row["scene"] is None
    with pytest.raises(ValueError):
        operator.assemble_observable(prepared, {}, {})


def test_canonical_order_is_not_recentered_or_sorted():
    q = np.array([.2, -.3, .6, -.1, -.5, 1., -.9, .4], dtype=np.float32)
    unit = contract.identity("development", "ood_beta", "canonical", 0)
    prepared = operator.prepare_input(unit, q)
    np.testing.assert_array_equal(prepared["observation"]["log_f"], q.astype(np.float64))
    with pytest.raises(ValueError):
        operator.prepare_input(unit, q.astype(np.float64)+1e-10)


def test_observable_kernel_adapter_with_manual_fixture():
    frequencies = np.array([100., 200., 300., 400., 150., 450., 600., 750.])
    q = (np.log(frequencies)-np.log(frequencies).mean()).astype(np.float32)
    unit = contract.identity("development", "iid", "canonical", 1)
    prepared = operator.prepare_input(unit, q)
    features = operator.features_for_input(prepared)
    logits = np.full((8, 8), -5., dtype=np.float32)
    logits[:4, :4] = logits[4:, 4:] = 5.
    row = operator.assemble_observable(prepared, features,
                                       {cp: logits.copy() for cp in (2026090721, 2026090722, 2026090723)})
    assert row["unit"] == unit and row["status"] == "ELIGIBLE"
    assert row["scene"]["split"] == "iid"  # Role remains outside the historical kernel.
    np.testing.assert_array_equal(row["scene"]["canonical_to_observed"], np.argsort(q, kind="stable"))
    origins = operator.candidate_origins(row["scene"])
    assert len(origins) == len(row["scene"]["partitions"])
    assert any(any(r["native_membership"].values()) for r in origins)
    with pytest.raises(ValueError, match="unexpected fields"):
        operator.features_for_input({**prepared, "truth": [0]*8})


def test_failure_is_not_scientific_abstention(monkeypatch):
    from src.atencion_armonica import geometric_decision_observables as kernels
    prepared = operator.prepare_input(contract.identity("development", "iid", "nominal", 1),
                                       np.arange(8)+100.)
    def fail(*args, **kwargs):
        raise ValueError("simulated corrupt kernel data")
    monkeypatch.setattr(kernels, "scene_from_sources", fail)
    with pytest.raises(ValueError, match="corrupt kernel"):
        operator.assemble_observable(prepared, {}, {})
