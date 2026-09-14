"""Full-count OPEN fixtures with small eligible support, no dataset producers."""
import hashlib

import numpy as np
import pytest

from experiments.atencion_armonica.test_geometric_decision_core import interface
from experiments.atencion_armonica.test_geometric_decision_training import runtime
from src.atencion_armonica.generative_evidence_cell import state_digest
from src.atencion_armonica.geometric_decision_cell import CellData, run_cell, read_calibration
from src.atencion_armonica.geometric_decision_store import ArtifactStore

CELL = {"arm": "geometric_decision", "checkpoint_seed": 2026090721,
        "reader_seed": 2026091491, "device": "cpu"}


@pytest.fixture(scope="module")
def dataset():
    inputs, raw, _ = interface()
    empty = {"groups": np.empty((0, 9), np.float32), "globals": np.empty((0, 17), np.float32),
             "incidence": np.empty((0, 0), np.float32), "evidence": np.empty((0, 8), np.float32)}
    def rows(split, count, eligible):
        return [{"scene_id": i, "identity": hashlib.sha256(f"fixture-{split}-{i}".encode()).hexdigest(),
                 "inputs": inputs if i < eligible else empty,
                 "partitions": raw["partitions"] if i < eligible else [],
                 "targets": np.array([[.6, .6], [.1, .2]], np.float32) if i < eligible
                            else np.empty((0, 2), np.float32)} for i in range(count)]
    return CellData(rows("train", 4096, 33), rows("calibration", 512, 35),
                    binding={"fixture": "arithmetic-complete-roster-small-eligible-support"})


def store_for(path, data):
    return ArtifactStore(path, binding={"schema": "geometric-decision-cell-binding-v1", "data": data.binding, **CELL})


def run(data, store, check=lambda: None):
    return run_cell(data, store, **CELL, check=check, progress=lambda row: None)


def test_fifty_epochs_full_roster_and_exact_recovery(tmp_path, dataset):
    continuous = store_for(tmp_path/"continuous", dataset)
    expected_ref = run(dataset, continuous)
    expected = continuous.json(expected_ref)
    assert expected["last_epoch"] == 50 and expected["steps"] == 100
    assert len(expected["calibration"]) == 11 and len(expected["history"]) == 50
    interrupted = store_for(tmp_path/"recovered", dataset)
    calls = []
    def stop():
        calls.append(1)
        if len(calls) == 8:
            raise InterruptedError("explicit fixture interruption")
    with pytest.raises(InterruptedError):
        run(dataset, interrupted, stop)
    actual = interrupted.json(run(dataset, interrupted))
    assert state_digest(continuous.load_state(expected["last_state"])) == state_digest(interrupted.load_state(actual["last_state"]))
    for a_ref, b_ref, epoch in zip(expected["calibration"], actual["calibration"], (0, *range(5, 51, 5))):
        a_state, b_state = continuous.json(a_ref)["state"], interrupted.json(b_ref)["state"]
        a = read_calibration(dataset, continuous, a_ref, a_state, epoch)
        b = read_calibration(dataset, interrupted, b_ref, b_state, epoch)
        assert all(a[k].tobytes() == b[k].tobytes() for k in a)
        assert a["offsets"].shape == (513,) and a["offsets"][35] == a["offsets"][-1] == 70
        assert a["components"].dtype == np.float64
    # Completed replay must not train or recompute any output.
    assert run(dataset, continuous) == expected_ref


def test_initial_calibration_interruption_and_recovery(tmp_path, dataset):
    store = store_for(tmp_path/"cell", dataset)
    calls = []
    def stop():
        calls.append(1)
        if len(calls) == 3:
            raise InterruptedError("initial calibration partial")
    with pytest.raises(InterruptedError):
        run(dataset, store, stop)
    assert store.path("snapshots/step_000000.json").exists()
    assert not store.path("calibration/epoch_00/index.json").exists()
    complete = store.json(run(dataset, store))
    assert len(complete["calibration"]) == 11


def test_missing_output_and_wrong_roster_do_not_close(tmp_path, dataset):
    with pytest.raises(ValueError, match="4096/512"):
        CellData(dataset.rows["train"][:-1], dataset.rows["calibration"], binding=dataset.binding)
    with pytest.raises(ValueError, match="eligible"):
        dataset.batch("train", [4095], "geometric")
    with pytest.raises(ValueError, match="eligible"):
        dataset.batch("train", [1, 1], "geometric")
    store = store_for(tmp_path/"cell", dataset)
    complete = store.json(run(dataset, store))
    record = store.json(complete["calibration"][0])
    blob = store.path(record["predictions"]["path"])
    blob.rename(blob.with_suffix(".fixture-preserved"))
    with pytest.raises(FileNotFoundError):
        run(dataset, store)
