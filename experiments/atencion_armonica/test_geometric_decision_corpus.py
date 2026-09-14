"""Complete adaptation fixtures with arithmetic source ports; no producer or data IO."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.atencion_armonica.test_geometric_decision_core import interface
from src.atencion_armonica.generative_evidence import CHECKPOINTS
from src.atencion_armonica.geometric_decision_corpus import OpenPreparation, source_identity
from src.atencion_armonica.geometric_decision_store import ArtifactStore


class ArithmeticSource:
    def __init__(self, root):
        self.inputs, self.raw, self.sham = interface()
        self.store = SimpleNamespace(root=root, binding={"fixture": "no-real-source"})
        self.corpus = SimpleNamespace(prepared_ref={"fixture": "prepared"}, normalizer_ref={"fixture": "normalizers"})
        self.delivered_ref = {"fixture": "delivered"}
        self.scale_calls = self.target_calls = 0
        self.observation_calls = []

    def geometric_scale(self, *, check):
        check()
        self.scale_calls += 1
        return {"fixture": "scale-authority-is-the-audited-source-port", "scale": 2.}

    def observable_shard(self, split, shard, *, check):
        check()
        self.observation_calls.append((split, shard))
        ids = list(range(shard*512, (shard+1)*512))
        return {"split": split, "shard": shard, "scene_ids": ids,
                "identities": [hashlib.sha256(f"{split}-{i}".encode()).hexdigest() for i in ids],
                "source_refs": {str(cp): {"fixture": f"{split}-{cp}-{shard}"} for cp in CHECKPOINTS},
                "targets_ref": {"fixture": f"targets-{split}-{shard}"},
                "raw": {cp: [self.raw]*512 for cp in CHECKPOINTS}}

    def interface_shard(self, obs, scale):
        assert scale["scale"] == 2.
        row = {"inputs": self.inputs, "sham": self.sham,
               "diagnostics": {"decoupled_six": self.inputs["evidence"][:, :6][[1, 0]]}}
        return {cp: [{**row, "scene_id": i} for i in obs["scene_ids"]] for cp in CHECKPOINTS}

    def supervision_shard(self, obs):
        self.target_calls += 1
        return [{"targets": np.array([[.7, .6], [.1, .2]], np.float32)} for _ in range(512)]


def preparation(tmp_path):
    source = ArithmeticSource(tmp_path/"source-identity-only")
    store = ArtifactStore(tmp_path/"prepared", binding={"source": source_identity(source), "fixture": "arithmetic"})
    return OpenPreparation(source, store)


def test_full_adaptation_is_separate_from_supervision_and_reusable(tmp_path):
    prep = preparation(tmp_path)
    ref = prep.prepare(check=lambda: None, progress=lambda row: None)
    complete = prep.completion(ref)
    assert len(complete["entries"]) == 27
    assert prep.source.scale_calls == 1 and prep.source.target_calls == 0
    assert prep.prepare(check=lambda: None, progress=lambda row: None) == ref
    assert prep.source.scale_calls == 1 and prep.source.target_calls == 0
    data = prep.load_checkpoint(ref, CHECKPOINTS[0], check=lambda: None)
    assert prep.source.target_calls == 9
    assert len(data.rows["train"]) == len(data.eligible["train"]) == 4096
    assert len(data.rows["calibration"]) == 512
    np.testing.assert_array_equal(data.rows["train"][0]["inputs"]["evidence"], prep.source.inputs["evidence"])
    assert data.rows["train"][0]["inputs"]["evidence"].shape == (2, 8)


def test_interrupted_preparation_reuses_scale_and_completed_shards(tmp_path):
    prep = preparation(tmp_path)
    def stop(row):
        if (row["split"], row["checkpoint_seed"], row["shard"]) == ("train", CHECKPOINTS[1], 0):
            raise InterruptedError("explicit arithmetic fixture interruption")
    with pytest.raises(InterruptedError):
        prep.prepare(check=lambda: None, progress=stop)
    assert not prep.store.path("complete.json").exists()
    first = prep.store.reference(prep.store.path(f"delivered/train/cp_{CHECKPOINTS[0]}/shard_0/index.json"))
    ref = prep.prepare(check=lambda: None, progress=lambda row: None)
    assert prep.completion(ref)["entries"][0]["index"] == first
    assert prep.source.scale_calls == 1 and prep.source.target_calls == 0


def test_completion_requires_every_entry_and_exact_evidence(tmp_path):
    prep = preparation(tmp_path)
    ref = prep.prepare(check=lambda: None, progress=lambda row: None)
    incomplete = deepcopy(prep.completion(ref))
    incomplete["entries"].pop()
    bad_ref = prep.store.publish_json("incomplete-fixture.json", incomplete)
    with pytest.raises(ValueError, match="27"):
        prep.load_checkpoint(bad_ref, CHECKPOINTS[0], check=lambda: None)
    # A changed caller source is detected even though the saved NPZ is intact.
    prep.source.inputs["evidence"] = prep.source.inputs["evidence"].copy()
    prep.source.inputs["evidence"][0, 6] += np.float32(1.)
    with pytest.raises(ValueError, match="recomputed"):
        prep.load_checkpoint(ref, CHECKPOINTS[0], check=lambda: None)
    assert prep.source.target_calls == 0


def test_source_binding_cannot_be_replaced(tmp_path):
    prep = preparation(tmp_path)
    prep.source.delivered_ref = {"fixture": "other-source"}
    with pytest.raises(ValueError, match="exact authenticated"):
        OpenPreparation(prep.source, prep.store)
