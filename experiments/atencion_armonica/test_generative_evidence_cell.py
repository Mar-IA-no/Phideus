"""Full-count mechanical cells: no draws, backbone/fitter, CUDA or real training."""
from copy import deepcopy
import hashlib

import numpy as np
import pytest
import torch

from experiments.atencion_armonica.test_generative_evidence import fixture, norms
from src.atencion_armonica import generative_evidence as ge
from src.atencion_armonica.generative_evidence_cell import CellData, CellArtifacts, run_cell, read_calibration, state_digest
from src.atencion_armonica.generative_evidence_training import TrainingKernel

DATA_BINDING = {"fixture": "complete-roster-arithmetic-not-campaign"}
BINDING = {"data": DATA_BINDING, "purpose": "cell-recovery-mechanics"}
CELL = {"arm": "generative", "checkpoint_seed": ge.CHECKPOINTS[0], "reader_seed": ge.READER_SEEDS[0], "device": "cpu"}


@pytest.fixture(scope="module")
def dataset():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    row = ge.observable_rows(*fixture())
    cn, en = norms(row)
    x = ge.model_inputs(row, cn, en, "generative", split_seed=10, scene_id=0)
    empty = {"groups": np.empty((0, 9), np.float32), "globals": np.empty((0, 17), np.float32),
             "evidence": np.empty((0, 6), np.float32), "incidence": np.empty((0, 0), np.float32)}
    def rows(split, count, eligible):
        return [{"scene_id": i, "identity": hashlib.sha256(f"{split}-{i}".encode()).hexdigest(),
                 "partitions": row["partitions"] if i < eligible else [],
                 "inputs": x if i < eligible else empty,
                 "targets": np.full((len(row["partitions"]) if i < eligible else 0, 2), (i % 7)/8, np.float32)}
                for i in range(count)]
    return CellData(rows("train", 4096, 33), rows("calibration", 512, 35), binding=DATA_BINDING)


def run(data, store, check=lambda: None):
    return run_cell(data, store, **CELL, check=check, progress=lambda row: None)


def test_full_50_epoch_cell_and_exact_mid_epoch_recovery(tmp_path, dataset):
    continuous = CellArtifacts(tmp_path/"continuous", binding=BINDING)
    finished = continuous.json(run(dataset, continuous))
    assert finished["epochs"] == 50 and finished["steps"] == 100
    assert len(finished["snapshots"]) == 51 and len(finished["calibrations"]) == 10
    expected = continuous.load_state(finished["last_epoch"])
    interrupted = CellArtifacts(tmp_path/"recovered", binding=BINDING)
    calls = []
    def stop():
        calls.append(1)
        if len(calls) == 4:
            raise InterruptedError("mechanical mid-epoch stop")
    with pytest.raises(InterruptedError):
        run(dataset, interrupted, stop)
    k = TrainingKernel(**CELL, binding=BINDING, scene_ids=list(range(33)))
    interrupted.latest(k)
    assert (k.epoch, k.next_batch, k.steps) == (1, 1, 3)
    actual_record = interrupted.json(run(dataset, interrupted))
    actual = interrupted.load_state(actual_record["last_epoch"])
    assert state_digest(actual) == state_digest(expected)
    for j, epoch in enumerate(range(5, 51, 5)):
        a, off_a = read_calibration(dataset, continuous, finished["calibrations"][j], finished["snapshots"][epoch], epoch)
        b, off_b = read_calibration(dataset, interrupted, actual_record["calibrations"][j], actual_record["snapshots"][epoch], epoch)
        assert a.tobytes() == b.tobytes() and np.array_equal(off_a, off_b)
        assert off_a.shape == (513,) and off_a[35] == off_a[-1] == 70
    # A completed cell can be checked again without another update/forward.
    assert run(dataset, interrupted) == interrupted.reference(interrupted.path("complete.json"))


def test_interruption_during_calibration_keeps_epoch_state_and_completes(tmp_path, dataset):
    store = CellArtifacts(tmp_path/"cell", binding=BINDING)
    def stop():
        if store.path("snapshots/step_000010.json").exists():
            raise InterruptedError("pause before first calibration")
    with pytest.raises(InterruptedError):
        run(dataset, store, stop)
    assert store.path("snapshots/step_000010.json").exists()
    assert not store.path("calibration/epoch_05/index.json").exists()
    complete = store.json(run(dataset, store))
    assert len(complete["calibrations"]) == 10


def test_incomplete_rosters_and_wrong_bindings_are_rejected(tmp_path, dataset):
    with pytest.raises(ValueError, match="4096/512"):
        CellData(dataset.rows["train"][:-1], dataset.rows["calibration"], binding=DATA_BINDING)
    wrong = deepcopy(dataset.rows["calibration"])
    wrong[0]["identity"] = wrong[1]["identity"]
    with pytest.raises(ValueError, match="identity"):
        CellData(dataset.rows["train"], wrong, binding=DATA_BINDING)
    store = CellArtifacts(tmp_path/"wrong", binding={"data": {"fixture": "other"}})
    with pytest.raises(ValueError, match="data references"):
        run(dataset, store)


def test_missing_required_epoch_or_calibration_payload_cannot_close(tmp_path, dataset):
    store = CellArtifacts(tmp_path/"cell", binding=BINDING)
    complete = store.json(run(dataset, store))
    prediction = store.json(complete["calibrations"][0])["predictions"]
    path = store.path(prediction["path"])
    path.rename(path.with_suffix(".preserved-away"))  # No deletion or symlink fixture.
    with pytest.raises(FileNotFoundError):
        run(dataset, store)
