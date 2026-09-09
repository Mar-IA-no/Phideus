"""Post-seal evaluation mechanics, using only q8 already in the open catalog."""
from copy import deepcopy
import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from experiments.atencion_armonica.test_generative_evidence import fixture
from src.atencion_armonica import generative_evidence_fresh_evaluation as e
from src.atencion_armonica.generative_evidence_storage import write_json

FREEZE = {"path": "fixture/not-test-authority", "sha256": "a"*64}


def root_ref(path):
    return {"path": path.relative_to(e.ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def shams(changed=()):
    return [{"changed_mask": [i in changed, i in changed] if i < 3 else [],
             "status": "INPUT_CHANGED" if i in changed else "INPUT_UNCHANGED"} for i in range(512)]


def metric_records(eligible):
    records = []
    for row in e.inference.inference_roster():
        values = np.full((512, len(e.METRICS)), np.nan, np.float64)
        values[eligible] = .5 if row["arm"] == "generative" else .25
        records.append({**row, "metrics": values})
    return records


@pytest.mark.parametrize("count", [0, 1, 3])
def test_complete_roster_conditional_support_and_no_cell_pseudoreplication(count):
    eligible = np.arange(512) < count
    summary, indices = e.summarize_readouts(metric_records(eligible), split="deformed_family",
                                            eligible=eligible, sham=shams([0] if count else []))
    assert indices.shape == (2000, count)
    primary = summary["primary"]
    assert primary["count"] == 512 and primary["output_count"] == count
    delta = primary["contrasts"]["generative-minus-local"]["ari"]
    assert delta["nominal_percent"] == 97.5
    if count:
        assert delta["mean"] == .25 and delta["interval"] == [.25, .25]
        assert np.asarray(summary["cell_contrast_signs"]["generative-minus-local"]).shape == (3, 3, 13)
        assert summary["sham_subsets"]["changed"]["count"] == 1
    else:
        assert delta["mean"] is None and delta["interval"] is None
        assert summary["cell_contrasts"]["generative-minus-local"] is None
        assert primary["interval_status"] == "UNDEFINED_NO_OUTPUT"
    assert primary["eligible_scene_ids"] == list(range(count))


def test_readout_summary_rejects_missing_or_noncommon_output():
    eligible = np.arange(512) < 1
    rows = metric_records(eligible)
    with pytest.raises(ValueError, match="all 45"):
        e.summarize_readouts(rows[:-1], split="iid", eligible=eligible, sham=shams())
    rows[-1]["metrics"][1] = 0.
    with pytest.raises(ValueError, match="common support"):
        e.summarize_readouts(rows, split="iid", eligible=eligible, sham=shams())


def test_readonly_replay_never_repairs_scientific_files(tmp_path):
    folder = tmp_path/"evaluation"
    store = e._Artifacts(folder, split="iid", binding={"fixture": True}, replay=False)
    store.write_json("index.json", {"fixture": "complete"})
    store.write_arrays("values.npz", {"metric": np.array([1., np.nan])})
    replay = e._Artifacts(folder, split="iid", binding={"fixture": True}, replay=True)
    replay.write_arrays("values.npz", {"metric": np.array([1., np.nan])})
    with pytest.raises(ValueError, match="replace"):
        replay.write_arrays("values.npz", {"metric": np.array([2., np.nan])})
    with pytest.raises(FileNotFoundError, match="repair"):
        replay.write_arrays("missing.npz", {"metric": np.array([1.])})
    assert not (folder/"missing.npz").exists()


def test_missing_seal_fails_before_output_or_privileged_port(tmp_path, monkeypatch):
    fake = ModuleType("src.atencion_armonica.generative_evidence_fresh_inference")
    def blocked(*a, **kw):
        raise PermissionError("missing 45-output seal")
    fake.verify_predictions = blocked
    fake.read_prediction = lambda *args: pytest.fail("no predictions after missing seal")
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(e, "CANONICAL", tmp_path/"never-created")
    with pytest.raises(PermissionError, match="45-output"):
        e.evaluate_test("iid", freeze_ref=FREEZE, check=lambda: None)
    assert not e.CANONICAL.exists()


def test_full_mechanical_evaluation_and_exact_readonly_replay(tmp_path, monkeypatch):
    # Source generation, state loading, truth reconstruction and seal authority
    # are deliberately mocked here. Their independent tests retain that scope.
    # All 512 observation values are aliases of the already opened arithmetic q8.
    q, z, _, _, ps, fits = fixture()
    labels = np.repeat(np.arange(2), 4)
    inventory = {"candidates": [{"partition": p, "origin": "pool", "status": "SUPPORTED"} for p in ps]}
    events, replayed = [], []
    canonical = tmp_path/"fresh"
    canonical.mkdir()
    monkeypatch.setattr(e, "CANONICAL", canonical)
    from src.atencion_armonica import generative_evidence_fresh_data as data
    class Observations:
        def __init__(self, split, **kwargs):
            self.index = {"records": [{"path": f"{i}/draw.json", "sha256": "b"*64} for i in range(512)]}
            self.files = SimpleNamespace(root=canonical/"draws", reader=SimpleNamespace(json=self.read))
        def read(self, ref):
            if ref["path"].endswith("sidecar.json"):
                assert events and events[0] == "seal"
                events.append("truth")
                return {"fixture": "truth"}
            return {"sidecar": {"path": ref["path"].replace("draw", "sidecar"), "sha256": "c"*64}}
        def observation(self, i):
            return {"scene_id": i, "split_seed": 2026090982, "log_f": q.tolist()}
    monkeypatch.setattr(data, "FreshObservations", Observations)
    monkeypatch.setattr(e, "reconstruct_truth", lambda *args: {"labels": labels})
    class Store:
        root = canonical
        def load_fit(self, split, i):
            obs = {"scene_id": i, "split_seed": 2026090982, "log_f": q.tolist()}
            return {"observation": obs, "inventory": inventory if i == 0 else {"candidates": []},
                    "fits": fits if i == 0 else [], "group_factors": []}, {
                        "path": f"{split}/{i}/fit.json", "sha256": "d"*64, "bytes": 1}
    monkeypatch.setattr(e, "FreshObservableStore", lambda *a, **kw: Store())
    def replay(q, factors, partitions):
        replayed.append(True)
        return fits if partitions else []
    monkeypatch.setattr(e.ge.law, "replay_fits", replay)
    monkeypatch.setattr(e.ge.law.GroupFitter, "fit", lambda *args: pytest.fail("no new fits"))
    choices = {"scene_ids": list(range(512)), "records": []}
    for i in range(512):
        choice = e.references.observable_choices(ps if i == 0 else [], fits if i == 0 else [],
            {cp: z for cp in e.ge.CHECKPOINTS}, np.argsort(q))
        choices["records"].append({"scene_id": i, "identity": "fixture", "choice": choice})
    records = [{**r, "prediction": {"path": f"fixture/{i}.npz", "sha256": "e"*64, "bytes": 1}}
               for i, r in enumerate(e.inference.inference_roster())]
    delivered = {}
    for cp in e.ge.CHECKPOINTS:
        path = canonical/f"{cp}.json"
        write_json(path, {"sham": shams([0])})
        delivered[str(cp)] = root_ref(path)
    verified = {"seal": FREEZE, "choices": choices, "records": records, "index": {"delivered": delivered}}
    fake = ModuleType("src.atencion_armonica.generative_evidence_fresh_inference")
    def verify(*a, **kw):
        events.append("seal")
        return verified
    fake.verify_predictions = verify
    fake.read_prediction = lambda *args: {"components": np.array([[.1, .2], [.3, .4]], np.float32),
        "offsets": np.r_[np.int64(0), np.full(512, 2, np.int64)]}
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    ref = e.evaluate_test("iid", freeze_ref=FREEZE, check=lambda: None)
    folder = canonical/"iid/evaluation"
    hashes = {p.relative_to(folder): hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob("*") if p.is_file()}
    assert e.replay_test("iid", freeze_ref=FREEZE, check=lambda: None) == ref
    assert hashes == {p.relative_to(folder): hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob("*") if p.is_file()}
    assert len(replayed) == 512 and events.count("truth") == 1024
    summary = json.loads((folder/"summary.json").read_bytes())
    assert summary["learned"]["primary"]["output_count"] == 1
    assert summary["systems"]["count"] == 512
    assert len(summary["systems"]["historical"]) == 3
    assert summary["systems"]["planted_presence"] == {"pool": 1, "neighbor": 0, "absent": 511}
    with np.load(folder/"metrics.npz", allow_pickle=False) as arrays:
        assert arrays["readouts"].shape == (512, 45, 13)
        assert np.isnan(arrays["readouts"][1:]).all()
        assert arrays["historical"].shape == (512, 3, 13)
