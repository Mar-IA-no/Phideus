import numpy as np
import pytest

from src.atencion_armonica import measurement_contract as contract


def test_fixed_rosters_and_paired_identities():
    for role, total in (("development", 128), ("calibration", 256), ("test", 2048)):
        units = contract.unit_roster(role)
        assert len(units) == total
        assert len({contract.digest(u) for u in units}) == total
        assert all(contract.validate_identity(u) == u for u in units)
        for i in range(0, total, 4):
            assert [u["condition"] for u in units[i:i+4]] == list(contract.CONDITIONS)
            assert len({u["scene_id"] for u in units[i:i+4]}) == 1
    wrong = {**contract.unit_roster("development")[0], "split_seed": 2026091531}
    with pytest.raises(ValueError, match="seed"):
        contract.validate_identity(wrong)
    with pytest.raises(ValueError):
        contract.identity("test", "iid", "canonical", 128)


def test_envelope_preserves_conditions_with_identical_kernel_identity():
    ref = {"path": "artifacts/example.npz", "sha256": "a"*64, "bytes": 20}
    a = contract.envelope(contract.identity("development", "iid", "nominal", 0),
                          emission=ref, waveform=ref, detection=ref, kernel_identity="b"*64)
    b = contract.envelope(contract.identity("development", "iid", "noisy", 0),
                          emission=ref, waveform=ref, detection=ref, kernel_identity="b"*64)
    assert a["envelope_sha256"] != b["envelope_sha256"]
    with pytest.raises(ValueError):
        contract.envelope(contract.identity("development", "iid", "canonical", 0),
                          emission=ref, waveform=ref, detection=None, kernel_identity=None)
    for bad in (".", "/tmp/x", "../x", "a/../../x", "a/./x", "a\\x"):
        with pytest.raises(ValueError):
            contract.reference({**ref, "path": bad})


def test_uniform_calibration_and_numeric_ties():
    costs = np.ones((9, 192), dtype=np.float64)
    units = contract.unit_roster("calibration", audio_only=True)
    result = contract.calibrate_detector(costs, units)
    assert (result["height"], result["prominence"]) == (-40, 3)
    costs[8] = .5
    result = contract.calibrate_detector(costs, units)
    assert result["selected_index"] == 8 and result["means"][8] == .5
    assert result["costs"] == costs.tolist()
    with pytest.raises(ValueError, match="order"):
        contract.calibrate_detector(costs, list(reversed(units)))
    with pytest.raises(ValueError):
        contract.calibrate_detector(costs[:, :-1], units)
    from copy import deepcopy
    aliased_units = deepcopy(units)
    # Scene 1 is not the JSON boolean true despite Python equality.
    aliased_units[3]["scene_id"] = True
    with pytest.raises(ValueError, match="order"):
        contract.calibrate_detector(costs, aliased_units)


def test_rank_mapping_with_permuted_delivery():
    actual = contract.partition_to_observed_labels([[0, 2], [1, 3]], np.array([2, 0, 3, 1]))
    assert actual.tolist() == [1, 1, 0, 0]
    np.testing.assert_array_equal(actual, contract.partition_to_observed_labels(
        ((0, 2), (1, 3)), np.array([2, 0, 3, 1])))
    with pytest.raises(ValueError):
        contract.partition_to_observed_labels([[0, 1], [1, 3]], np.arange(4))
    with pytest.raises(ValueError):
        contract.partition_to_observed_labels([[0, 1], [2, 3]], np.array([0, 1, 1, 3]))
