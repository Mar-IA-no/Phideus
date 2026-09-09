"""One declared archived OPEN observation proves lossless fit replay after gzip.

No new observation, grid sweep, forward, model training or metric selection.
"""
import hashlib

from src.atencion_armonica import observable_source_rivals as law
from src.atencion_armonica.generative_evidence_reuse import ROOT, VerifiedBytes
from src.atencion_armonica.generative_evidence_storage import write_scene, read_scene
from src.atencion_armonica.partial_compatibility_cache import encoded


def test_archived_complete_factors_replay_without_grid_after_compression(tmp_path, monkeypatch):
    reader = VerifiedBytes(ROOT)
    campaign = reader.json({"path": "data/atencion_armonica/observable_source_rivals_v1/campaign.json",
        "sha256": "a60213c6fb6fa3911bdc6c32733ab779b4b3895db5aaba519fe01ddae00c485c"})
    original = reader.json({"path": "data/atencion_armonica/observable_source_rivals_v1/observable/iid/00013.json",
        "sha256": "d615cf5067b9de8eefe52d32c917509e7546670ba98a03ab598cfb7aea60f340"})
    scene = next(s for s in campaign["scenes"] if s["split"] == "iid" and s["scene_id"] == 13)
    assert original["input_sha256"] == hashlib.sha256(encoded(scene)).hexdigest()
    record = {"observation": scene, "result": original}
    receipt = write_scene(tmp_path/"complete.json.gz", record)
    saved = read_scene(tmp_path/"complete.json.gz", receipt)
    assert encoded(saved) == encoded(record)
    def no_sweep(*args, **kwargs):
        raise AssertionError("replay must use preserved factors, never another fit")
    monkeypatch.setattr(law.GroupFitter, "fit", no_sweep)
    replay = law.replay_fits(saved["observation"]["q32"], saved["result"]["group_factors"],
                            [r["partition"] for r in saved["observation"]["candidates"]])
    assert encoded(replay) == encoded(original["fits"])
