from io import BytesIO

import numpy as np
import pytest

from src.atencion_armonica import measurement_reuse as reuse


def archive(arrays):
    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def test_numeric_head_archive_contract_without_real_weights():
    arrays = {k: np.zeros(shape, dtype=np.float32) for k, shape in reuse.SHAPES.items()}
    result = reuse.head_arrays(archive(arrays))
    assert sum(a.size for a in result.values()) == 2258
    for k in arrays:
        np.testing.assert_array_equal(arrays[k], result[k])
    arrays["partition2.bias"] = np.zeros(2, dtype=np.float64)
    with pytest.raises(ValueError):
        reuse.head_arrays(archive(arrays))


def test_explicit_roster_has_no_winner_seed_or_extra_loss():
    assert len(reuse.EPOCHS)*len(reuse.CHECKPOINTS)*len(reuse.READERS) == 36
    assert all(arm.endswith("_decision") for arm in reuse.EPOCHS)
    assert list(reuse.EPOCHS.values()) == [30, 45, 40, 45]
