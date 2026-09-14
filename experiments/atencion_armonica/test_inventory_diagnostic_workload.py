"""Observed-workload fixtures and lifecycle, without reading campaign arrays."""
from copy import deepcopy
import hashlib
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from experiments.atencion_armonica import inventory_diagnostic_workload as op


def fixture(split="iid"):
    partitions = [[[0, 1, 2, 3], [4, 5, 6, 7]], [[0, 1, 4, 5], [2, 3, 6, 7]]]
    scenes, counts = [], []
    for i in range(512):
        count = i % 3
        observation = {"scene_id": i, "split_seed": op.base.TESTS[split], "log_f": [j/8 for j in range(8)]}
        scene = {"observation": observation, "partitions": deepcopy(partitions[:count])}
        scene["identity_sha256"] = hashlib.sha256(op.base.encoded({"split": split, **scene})).hexdigest()
        scenes.append(scene)
        counts.append(count)
    offsets = op.base.np.r_[0, op.base.np.cumsum(counts)].astype(op.base.np.int64)
    return {"split": split, "inputs": {cp: {"arrays": {"candidate_offsets": offsets.copy()},
             "shard": {"scenes": deepcopy(scenes)}, "meta": {"inputs": {"path": f"fixture-{cp}.npz",
             "sha256": "0"*64}}} for cp in op.CHECKPOINTS}}


class WorkloadTests(unittest.TestCase):
    def test_complete_fixture_counts_zero_scenes_and_no_mutation(self):
        loaded = fixture()
        before = deepcopy(loaded)
        report = op.split_workload(loaded, lambda: None)
        self.assertEqual(report["scene_count"], 512)
        self.assertEqual(report["candidate_histogram"], {"0": 171, "1": 171, "2": 170})
        self.assertEqual(report["candidate_count"], 511)
        self.assertEqual(report["candidate_pairs"], 170)
        self.assertEqual(report["pairs_with_floor31"], 512*465)
        self.assertEqual(report["group_count"], 1022)
        self.assertEqual([r["scene_id"] for r in report["rows"]], list(range(512)))
        self.assertEqual(report["rows"][0]["candidate_count"], 0)
        for cp in op.CHECKPOINTS:
            op.base.np.testing.assert_array_equal(loaded["inputs"][cp]["arrays"]["candidate_offsets"],
                                                  before["inputs"][cp]["arrays"]["candidate_offsets"])
            self.assertEqual(loaded["inputs"][cp]["shard"], before["inputs"][cp]["shard"])

    def test_checkpoint_offsets_and_scene_identity_disagreement(self):
        for change in (lambda x: x["arrays"]["candidate_offsets"].__setitem__(2, 0),
                       lambda x: x["shard"]["scenes"][0].__setitem__("identity_sha256", "0"*64),
                       lambda x: x["shard"]["scenes"].pop()):
            loaded = fixture()
            change(loaded["inputs"][op.CHECKPOINTS[-1]])
            with self.assertRaises(ValueError):
                op.split_workload(loaded, lambda: None)

    def test_invalid_partitions_rejected_even_with_matching_identity(self):
        original = fixture()["inputs"][op.CHECKPOINTS[0]]["shard"]["scenes"][2]
        mutations = (lambda p: p.reverse(), lambda p: p.append(deepcopy(p[0])),
                     lambda p: p[0][0].__setitem__(0, True),
                     lambda p: p[0][0].__setitem__(0, 4),
                     lambda p: p[0][0].pop(), lambda p: p[0].reverse())
        for mutate in mutations:
            scene = deepcopy(original)
            mutate(scene["partitions"])
            scene["identity_sha256"] = hashlib.sha256(op.base.encoded({"split": "iid",
                "observation": scene["observation"], "partitions": scene["partitions"]})).hexdigest()
            with self.assertRaises(ValueError):
                op.scene_workload("iid", 2, scene, len(scene["partitions"]))
        for count in (-1, 83, True, 1):
            with self.assertRaises(ValueError):
                op.scene_workload("iid", 2, original, count)

    def test_observation_schema_identity_and_precision_not_absorbed_by_hash(self):
        mutations = (lambda o: o.__setitem__("scene_id", 999),
                     lambda o: o.__setitem__("split_seed", -1),
                     lambda o: o.__setitem__("extra", 1),
                     lambda o: o.__setitem__("scene_id", False),
                     lambda o: o["log_f"].__setitem__(0, .1),
                     lambda o: o.__setitem__("log_f", [[0.]]*8))
        for mutate in mutations:
            loaded = fixture()
            for cp in op.CHECKPOINTS:
                scene = loaded["inputs"][cp]["shard"]["scenes"][0]
                mutate(scene["observation"])
                scene["identity_sha256"] = hashlib.sha256(op.base.encoded({"split": "iid",
                    "observation": scene["observation"], "partitions": scene["partitions"]})).hexdigest()
            with self.assertRaises(ValueError):
                op.split_workload(loaded, lambda: None)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        parent = op.ROOT/op.base.WORK/"workload-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="case-", dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.store = op.base.DiagnosticStore(self.project/op.base.OUTPUT, project_root=self.project)
        self.store.initialize({"status": "PREPARED", "limits": op.base.LIMITS})
        (self.project/"audit.md").write_bytes(b"Independent fixture audit.\n")
        self.review = {"status": "WORKLOAD_REVIEW_COMPLETE", "snapshot": {"fixture": "workload"},
                       "reports": [op.prior.reference(self.project, "audit.md")]}
        (self.project/"review.json").write_bytes(op.base.encoded(self.review))
        self.ref = op.prior.reference(self.project, "review.json")

    def test_success_seal_and_first_success_even_with_alias_review(self):
        workload = {"scene_count": 2048, "splits": {s: op.split_workload(fixture(s), lambda: None)
                                                  for s in op.base.TESTS}}
        with patch.object(op, "accept_review", return_value=self.review), \
                patch.object(op, "collect", return_value=workload), patch("builtins.print"):
            report = op.execute(self.ref, project=self.project)
            finish = self.store.json(op.prior.reference(self.store.root, "attempts/0000/finish.json"))
            self.assertEqual(finish["status"], "COMPLETE")
            self.assertEqual(self.store.json(finish["completion"]["workload"]), report)
            self.assertFalse(self.store.path("complete.json").exists())
            self.assertFalse(self.store.path("replayed.json").exists())
            (self.project/"alias.json").write_bytes(op.base.encoded(self.review))
            with patch.object(op.base, "AttemptBudget", side_effect=AssertionError("no fresh timing")):
                reused = op.execute(op.prior.reference(self.project, "alias.json"), project=self.project)
            self.assertEqual(reused, report)
            self.assertFalse(self.store.path("attempts/0001").exists())

    def test_failure_and_pause_are_charged_without_completion(self):
        for index, (error, state) in enumerate(((ValueError("fixture"), "FAILED"),
                                               (op.base.Paused("fixture"), "PAUSED"),
                                               (op.base.BudgetExceeded("fixture"), "BUDGET_EXHAUSTED"))):
            handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)}
            with patch.object(op, "accept_review", return_value=self.review), \
                    patch.object(op, "collect", side_effect=error), patch("builtins.print"):
                with self.assertRaises(type(error)):
                    op.execute(self.ref, project=self.project)
            finish = self.store.json(op.prior.reference(self.store.root, f"attempts/{index:04d}/finish.json"))
            self.assertEqual(finish["status"], state)
            self.assertGreater(finish["charged_total"], 0)
            self.assertNotIn("completion", finish)
            self.assertEqual(handlers, {s: signal.getsignal(s) for s in handlers})

    def test_review_requires_current_snapshot_and_authentic_report(self):
        with patch.object(op, "snapshot", return_value=self.review["snapshot"]):
            self.assertEqual(op.accept_review(self.store, self.ref), self.review)
            (self.project/"audit.md").write_bytes(b"wrong")
            with self.assertRaises(ValueError):
                op.accept_review(self.store, self.ref)
        with patch.object(op, "snapshot", return_value={"changed": True}):
            with self.assertRaises(ValueError):
                op.accept_review(self.store, self.ref)


if __name__ == "__main__":
    unittest.main()
