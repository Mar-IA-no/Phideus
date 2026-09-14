"""Closed-campaign adapter: authenticate first, then expose one observed scene.

No creation or execution authority. Receipt inventory never decompresses fits;
`extract_scene` is a separate operation reserved for the audited profile/run.
"""
from __future__ import annotations

import hashlib

import numpy as np

from . import operator_objective_core as core
from . import operator_objective_sources as sources
from .operator_objective_scene import ARMS, CELLS, CHECKPOINTS, READERS
from .generative_evidence_supervision import reconstruct_truth

BASE = "data/atencion_armonica/generative_evidence_reader_v1"
FRESH = BASE+"/fresh"
TESTS = {"iid": 2026090982, "ood_beta": 2026090983,
         "ood_polyphony": 2026090984, "deformed_family": 2026090985}
IDS = list(range(512))
COMPLETION = {"path": FRESH+"/test_completion.json",
              "sha256": "160473a7c88ab9e9a296a9e37a3145546a6ccc7019dbec473394ccd966cae3f9"}
NORMALIZER_SHA = "46acc508952f72187c7f26dfe325c816f22bd0df1e5a8c88cb1e81e3b8f50e93"


def equal(actual, expected, message):
    if sources.encoded(actual) != sources.encoded(expected):
        raise ValueError(message)


def identity_ref(ref):
    return {k: ref[k] for k in ("path", "sha256")}


def offsets(value, *, maximum=82):
    if (not isinstance(value, np.ndarray) or value.dtype != np.int64 or value.shape != (513,)
            or value[0] != 0 or np.any(np.diff(value) < 0) or np.any(np.diff(value) > maximum)):
        raise ValueError("offsets differ from the complete fixed scene roster")
    return value


def prediction_array(arrays):
    if set(arrays) != {"components", "offsets"}:
        raise ValueError("prediction array schema differs")
    off, h = offsets(arrays["offsets"]), arrays["components"]
    if (h.dtype != np.float32 or h.shape != (int(off[-1]), 2)
            or not np.isfinite(h).all() or np.any(h < 0)):
        raise ValueError("predictions must retain finite nonnegative float32 components")
    return arrays


class ClosedCorpus:
    """Owns only verified source metadata, not files, models, or runtime leases."""

    def __init__(self, reader):
        self.reader = reader
        self.completion = reader.json(COMPLETION)
        closed = self.completion
        if (closed["schema"] != "generative-evidence-fresh-tests-complete-v1"
                or closed["status"] != "FRESH_TESTS_EVALUATED_REPLAYED_NOT_PROMOTED"
                or closed["stage_count"] != 13 or len(closed["stages"]) != 13
                or set(closed["tests"]) != set(TESTS)):
            raise ValueError("the pinned campaign is not a complete four-test closure")
        reader.json(closed["manifest"])
        self.freeze_ref = closed["stages"][0]["output"]
        self.freeze = reader.json(self.freeze_ref)
        self.binding = {"test_freeze": self.freeze_ref}
        for i, entry in enumerate(closed["stages"]):
            kind = "freeze" if i == 0 else ("predict", "evaluate", "replay")[(i-1) % 3]
            split = None if i == 0 else tuple(TESTS)[(i-1)//3]
            expected = {"index": i, "kind": kind, "split": split,
                        "device": "cuda:0" if kind == "predict" else "cpu"}
            equal(entry["stage"], expected, "closed stage sequence differs")
            launch, exit_record, worker = (reader.json(entry[k]) for k in ("launch", "exit", "worker"))
            equal(launch["stage"], expected, "launch differs from closed stage")
            equal(worker["stage"], expected, "worker differs from closed stage")
            equal(worker["output"], entry["output"], "worker output differs")
            equal(exit_record["launch"], entry["launch"], "exit receipt belongs to another launch")
            if exit_record["process_returncode"] != 0 or exit_record["terminal"] is not True:
                raise ValueError("closed stage is not terminal success")
            reader.json(entry["output"])
            if split is not None:
                role = "prediction_seal" if kind == "predict" else "evaluation"
                equal(entry["output"], closed["tests"][split][role], "evaluation/replay output lineage differs")
                if kind == "replay":
                    equal(entry["worker"], closed["tests"][split]["replay_worker"], "replay worker lineage differs")
                    if worker["status"] != "REPLAY_VERIFIED":
                        raise ValueError("closed replay was not verified")
        if (self.freeze["prediction_count_per_test"] != 45 or self.freeze["original_prediction_count_per_test"] != 27
                or self.freeze["intervention_prediction_count_per_test"] != 18):
            raise ValueError("freeze prediction extent differs")
        equal(self.freeze["tests"], [{"split": s, "split_seed": seed, "scene_count": 512, "scene_ids": IDS}
                                      for s, seed in TESTS.items()], "freeze test roster differs")
        self.normalizer_ref = self.freeze["normalizers"]
        equal(identity_ref(self.normalizer_ref), {"path": BASE+"/normalizers.json", "sha256": NORMALIZER_SHA},
              "TRAIN normalizer differs from the diagnostic contract")
        self.normalizers = reader.json(self.normalizer_ref)
        train = reader.json(self.normalizers["prepared_train"], base=BASE)
        if train["split"] != "train":
            raise ValueError("normalizer provenance is not TRAIN")
        self.selected = {}
        for row in self.freeze["selected_states"]:
            cell = row["cell"]
            key = (cell["arm"], cell["checkpoint_seed"], cell["reader_seed"])
            if key in self.selected:
                raise ValueError("duplicate selected cell")
            reader.json(row["state"])
            self.selected[key] = row["state"]
        if set(self.selected) != {(a, cp, seed) for a in ARMS for cp in CHECKPOINTS for seed in READERS}:
            raise ValueError("selected state roster differs")
        self.headers, self.receipts = {}, None

    def header(self, split):
        if split not in TESTS:
            raise ValueError("only the four already-open tests are in scope")
        if split in self.headers:
            return self.headers[split]
        reader = self.reader
        refs = self.completion["tests"][split]
        seal = reader.json(refs["prediction_seal"])
        equal(seal["binding"], self.binding, "seal freeze differs")
        if (seal["split"] != split or seal["status"] != "FRESH_PREDICTIONS_SEALED_NO_TRUTH_ACCESS"
                or seal["prediction_count"] != 45 or seal["truth_access"] is not False):
            raise ValueError("prediction seal not complete")
        index = reader.json(seal["prediction_index"])
        equal(index["binding"], self.binding, "prediction index freeze differs")
        equal(index["scene_ids"], IDS, "prediction scene roster differs")
        if (index["split"] != split or index["split_seed"] != TESTS[split]
                or index["prediction_count"] != 45 or len(index["roster"]) != 45
                or len(index["predictions"]) != 45 or index["truth_access"] is not False):
            raise ValueError("prediction roster extent/identity differs")
        evaluation = reader.json(refs["evaluation"])
        eval_binding = {**self.binding, "split": split, "prediction_seal": refs["prediction_seal"]}
        equal(evaluation["binding"], eval_binding, "evaluation binding differs")
        equal(evaluation["scene_ids"], IDS, "evaluation scene roster differs")
        if (evaluation["status"] != "EVALUATED_NOT_PROMOTED" or evaluation["prediction_count"] != 45
                or len(evaluation["scenes"]) != 512 or len(evaluation["readouts"]) != 45):
            raise ValueError("evaluation roster extent differs")
        readouts, originals, seen = [], {}, set()
        freeze_indices = [r["freeze_index"] for r in index["roster"]]
        if any(type(i) is not int for i in freeze_indices) or sorted(freeze_indices) != list(range(45)):
            raise ValueError("inference roster does not bijectively cover the frozen roster")
        for i, (roster, metadata_ref, readout_ref) in enumerate(zip(index["roster"], index["predictions"], evaluation["readouts"])):
            metadata = reader.json(metadata_ref)
            readout = reader.json(readout_ref, base=f"{FRESH}/{split}/evaluation")
            if type(roster["inference_index"]) is not int or roster["inference_index"] != i:
                raise ValueError("inference index no longer matches prediction order")
            frozen = self.freeze["prediction_roster"][roster["freeze_index"]]
            expected_identity = {k: frozen[k] for k in ("arm", "checkpoint_seed", "reader_seed")}
            expected_identity["intervention"] = "original" if frozen["kind"] == "original" else frozen["intervention"]
            key = tuple(expected_identity[k] for k in ("arm", "checkpoint_seed", "reader_seed", "intervention"))
            if key in seen:
                raise ValueError("duplicate prediction identity")
            seen.add(key)
            equal({k: roster[k] for k in expected_identity}, expected_identity, "prediction roster differs from freeze")
            equal({k: metadata[k] for k in roster}, roster, "prediction metadata differs from roster")
            equal(metadata["state"], self.selected[key[:3]], "prediction selected state differs")
            equal(metadata["binding"], self.binding, "prediction metadata freeze differs")
            equal(metadata["scene_ids"], IDS, "prediction metadata scene roster differs")
            equal(metadata["delivered"], index["delivered"][str(key[1])], "prediction delivered source differs")
            if metadata["split"] != split or metadata["split_seed"] != TESTS[split] or metadata["truth_access"] is not False:
                raise ValueError("prediction role differs")
            equal(readout["identity"], expected_identity, "evaluation readout identity differs")
            equal(readout["prediction"], metadata["prediction"], "readout no longer binds exact prediction")
            if len(readout["decisions"]) != 512:
                raise ValueError("readout decision extent differs")
            reader.bytes(metadata["prediction"])  # Authenticate all 45 before exposing original 27.
            readouts.append(readout)
            if key[3] == "original":
                originals[(key[0], f"{key[1]}:{key[2]}")] = {"metadata": metadata, "readout": readout}
        if set(originals) != {(arm, cell) for arm in ARMS for cell in CELLS}:
            raise ValueError("original 27-cell roster differs")
        choices = reader.json(index["choices"])
        equal(choices["binding"], self.binding, "observable choices freeze differs")
        equal(choices["scene_ids"], IDS, "observable choice scene roster differs")
        if (choices["split"] != split or choices["split_seed"] != TESTS[split]
                or len(choices["records"]) != 512 or choices["truth_access"] is not False):
            raise ValueError("observable choice role/extent differs")
        self.headers[split] = {"seal": seal, "index": index, "evaluation": evaluation,
                               "eval_binding": eval_binding, "originals": originals, "choices": choices}
        return self.headers[split]

    def inventory(self):
        """Authenticate all 2048 JSON fit receipts; do not read their gzip blobs."""
        records = {}
        for split in TESTS:
            header = self.header(split)
            records[split] = []
            for i, ref in enumerate(header["evaluation"]["scenes"]):
                scene = self.reader.json(ref, base=f"{FRESH}/{split}/evaluation")
                equal(scene["binding"], header["eval_binding"], "evaluated scene binding differs")
                if scene["scene_id"] != i:
                    raise ValueError("evaluated scene order differs")
                fit = self.reader.json(scene["fit"])
                equal(fit["binding"], self.binding, "fit receipt freeze differs")
                if fit["scene_id"] != i or fit["split"] != split or fit["schema"] != "generative-evidence-fit-v1":
                    raise ValueError("fit receipt identity differs")
                origin = fit["origin"]
                equal(origin["test_freeze"], self.freeze_ref, "fit origin freeze differs")
                equal(origin["grid"], {"beta_count": 257, "gamma_count": 65, "stride": 4}, "fit grid differs")
                if origin["kind"] != "NEW_FRESH_FIT" or origin["device"] != "cuda:0":
                    raise ValueError("fit origin differs")
                a = fit["artifact"]
                if a["codec"] != "canonical-json-gzip3-mtime0":
                    raise ValueError("fit codec differs")
                sources.AuthenticatedReader.parts(a["path"])
                for field in ("sha256", "decoded_sha256"):
                    sources._digest(a[field])
                sources._size(a["bytes"], 16*1024**2)
                sources._size(a["decoded_bytes"], 16*1024**2)
                records[split].append({"scene_id": i, "evaluation": ref, "fit": scene["fit"],
                                       "draw": scene["draw"], "artifact": a})
        flat = [r["artifact"] for rows in records.values() for r in rows]
        self.receipts = records
        return {"completion": COMPLETION, "records": records, "scene_count": len(flat),
                "compressed_bytes": sum(r["bytes"] for r in flat),
                "decoded_bytes": sum(r["decoded_bytes"] for r in flat),
                "maximum_decoded_bytes": max(r["decoded_bytes"] for r in flat)}

    def load_split(self, split):
        """Load delivered/prediction arrays once; no factors or truth yet."""
        if self.receipts is None:
            raise ValueError("complete receipt inventory must precede scene access")
        header = self.header(split)
        inputs = {}
        for cp in CHECKPOINTS:
            ref = header["index"]["delivered"][str(cp)]
            meta = self.reader.json(ref)
            equal(meta["binding"], self.binding, "delivered freeze differs")
            equal(meta["normalizers"], self.normalizer_ref, "delivered TRAIN normalizer differs")
            equal(meta["scene_ids"], IDS, "delivered scene roster differs")
            if meta["split"] != split or meta["checkpoint_seed"] != cp or len(meta["identities"]) != 512 or len(meta["sham"]) != 512:
                raise ValueError("delivered metadata extent differs")
            arrays = self.reader.arrays(meta["inputs"])
            required = {"groups", "globals", "local", "generative", "decoupled", "group_offsets",
                        "candidate_offsets", "incidence_offsets", "incidence", "metadata"}
            if set(arrays) != required or arrays["metadata"].dtype != np.uint8 or arrays["metadata"].ndim != 1:
                raise ValueError("delivered array schema differs")
            shard = sources._json(arrays["metadata"].tobytes())
            for key, expected in (("binding", self.binding), ("split", split), ("checkpoint_seed", cp),
                                  ("raw", meta["raw"]), ("normalizers", self.normalizer_ref), ("sham", meta["sham"])):
                equal(shard[key], expected, "delivered shard identity differs")
            if len(shard["scenes"]) != 512 or [s["identity_sha256"] for s in shard["scenes"]] != meta["identities"]:
                raise ValueError("delivered shard scene identity order differs")
            co = offsets(arrays["candidate_offsets"])
            for arm in ARMS:
                a = arrays[arm]
                if a.dtype != np.float32 or a.shape != (co[-1], 6) or not np.isfinite(a).all():
                    raise ValueError("delivered evidence array differs")
            inputs[cp] = {"meta": meta, "arrays": arrays, "shard": shard}
        predictions = {arm: {} for arm in ARMS}
        for (arm, cell), record in header["originals"].items():
            prediction = prediction_array(self.reader.arrays(record["metadata"]["prediction"]))
            cp = record["metadata"]["checkpoint_seed"]
            if not np.array_equal(prediction["offsets"], inputs[cp]["arrays"]["candidate_offsets"]):
                raise ValueError("prediction/delivered candidate offsets differ")
            predictions[arm][cell] = prediction
        return {"split": split, "inputs": inputs, "predictions": predictions}

    def extract_scene(self, loaded, scene_id):
        """Authenticated compact input; call only under profile/run budget."""
        split = loaded["split"]
        if self.receipts is None or type(scene_id) is not int or not 0 <= scene_id < 512:
            raise ValueError("scene access requires complete inventory and a fixed scene ID")
        row = self.receipts[split][scene_id]
        header = self.header(split)
        evaluation = self.reader.json(row["evaluation"], base=f"{FRESH}/{split}/evaluation")
        draw = self.reader.json(row["draw"])
        equal(draw["test_freeze"], self.freeze_ref, "draw freeze differs")
        if (draw["scene_id"] != scene_id or draw["split"] != split or draw["split_seed"] != TESTS[split]
                or draw["status"] != "DRAW_PRESERVED"):
            raise ValueError("draw identity differs")
        observation = self.reader.json(draw["observation"], base=FRESH+"/draws")
        if observation["scene_id"] != scene_id or observation["split_seed"] != TESTS[split]:
            raise ValueError("observation identity differs")
        fitted = self.reader.fit_json(row["artifact"], base=FRESH)
        compact = sources.compact_fit(fitted, observation)
        sidecar = self.reader.json(draw["sidecar"], base=FRESH+"/draws")
        labels = reconstruct_truth(observation, sidecar, split)["labels"]
        choice = header["choices"]["records"][scene_id]
        if choice["scene_id"] != scene_id:
            raise ValueError("choice scene order differs")
        equal(identity_ref(choice["fit"]), identity_ref(row["fit"]), "choice fit differs")
        origin = self.reader.json(row["fit"])["origin"]
        equal(choice["source"], origin["source"], "choice source differs from fit origin")
        source = self.reader.json(choice["source"])
        equal(source["binding"], self.binding, "observable source freeze differs")
        equal(source["partitions"], compact["partitions"], "observable candidate roster differs from fits")
        equal(source["inventory"], compact["inventory"], "source inventory differs from compact fit")
        order = np.argsort(np.asarray(observation["log_f"]), kind="stable")
        if not np.array_equal(source["canonical_to_observed"], order) or not np.array_equal(source["q32"], np.asarray(observation["log_f"])[order]):
            raise ValueError("observable canonical frequency mapping differs")
        identity = hashlib.sha256(sources.encoded({"split": split, "observation": observation,
                                                   "partitions": compact["partitions"]})).hexdigest()
        if choice["identity"] != identity:
            raise ValueError("observable choice candidate identity differs")
        channels = {}
        for cp in CHECKPOINTS:
            item = loaded["inputs"][cp]
            scene = item["shard"]["scenes"][scene_id]
            equal(scene, {"observation": observation, "partitions": compact["partitions"],
                          "identity_sha256": identity}, "delivered scene differs from fit candidates")
            a = item["arrays"]
            start, end = a["candidate_offsets"][scene_id:scene_id+2]
            delivered = {arm: a[arm][start:end] for arm in ARMS}
            sources.verify_delivered_channel(compact, self.normalizers["normalizers"]["evidence"],
                                               delivered, item["meta"]["sham"][scene_id])
            channels[str(cp)] = {**delivered, "sham": item["meta"]["sham"][scene_id]}
        predictions = {arm: {} for arm in ARMS}
        for arm in ARMS:
            for cell in CELLS:
                p = loaded["predictions"][arm][cell]
                start, end = p["offsets"][scene_id:scene_id+2]
                predictions[arm][cell] = p["components"][start:end].copy()
        return {"split": split, "scene_id": scene_id, "observation": observation,
                "canonical_labels": np.asarray(labels, np.int64), "compact": compact, "channels": channels,
                "predictions": predictions, "normalizer": self.normalizers["normalizers"]["evidence"],
                "provenance": {**row, "observation": draw["observation"], "sidecar": draw["sidecar"],
                               "normalizers": self.normalizer_ref, "candidate_identity": identity},
                "archived_result": evaluation["result"], "archived_choices": choice["choice"],
                "archived_decisions": {arm: {cell: header["originals"][(arm, cell)]["readout"]["decisions"][scene_id]
                                             for cell in CELLS} for arm in ARMS}}


def verify_scene_result(extracted, result):
    """Cotejo against preserved truth metrics and exact observable choices."""
    old = extracted["archived_result"]
    targets = result["arrays"]["targets"]
    ps = extracted["compact"]["partitions"]
    y = extracted["canonical_labels"]
    n = len(ps)
    if result["candidate_count"] != n or len(old["candidate_metrics"]) != n:
        raise ValueError("diagnostic target extent differs from archived evaluation")
    raw = np.asarray(old["raw_entropies"], np.float64).reshape(-1, 2)
    stored = np.asarray(old["normalized_targets"], np.float64).reshape(-1, 2)
    if (raw.shape != targets["raw"].shape or not np.allclose(raw, targets["raw"], rtol=0, atol=1e-12)
            or not np.array_equal(stored, targets["u32"].astype(np.float64))):
        raise ValueError("reconstructed entropy/float32 targets differ from archived truth")
    for i, row in enumerate(old["candidate_metrics"]):
        expected = {"ari": targets["ari"][i], "k_error": targets["k_error"][i],
                    "k_absolute_error": abs(targets["k_error"][i]), "exact_partition": targets["exact"][i],
                    "split_entropy": targets["raw"][i, 0], "merge_entropy": targets["raw"][i, 1],
                    "split_normalized": targets["u64"][i, 0], "merge_normalized": targets["u64"][i, 1],
                    "vi_normalized": targets["t64"][i]}
        for key, value in expected.items():
            if not np.isclose(row[key], value, rtol=0, atol=1e-12):
                raise ValueError(f"candidate metric differs: {key}")
    planted = sources.law.signature([np.flatnonzero(y == label).tolist() for label in np.unique(y)])
    origin = [r["origin"] for r in extracted["compact"]["inventory"]["candidates"]
              if sources.law.signature(r["partition"]) == planted]
    coverage = {"has_output": bool(n), "candidate_count": n, "planted": origin[0] if origin else "absent"}
    equal(old["coverage"], coverage, "planted coverage differs from original evaluation")
    for family in ("base", "extended"):
        record = result["classical"][family+"_ub"]["schemes"]["full"]["strata"]["all"]["choice"]
        i = record["chosen"]
        choice = None if i is None else {"candidate_index": i, "partition": ps[i], "UB": record["minimum"],
                                       "branch": extracted["compact"]["winning_branches"][family+"_ub"][i]}
        equal(extracted["archived_choices"][family], choice, "classical UB choice differs from preserved observable choice")
        expected = None if old["references"][family] is None else old["references"][family]["choice"]
        equal(expected, choice, "classical evaluated choice differs")
    for arm in ARMS:
        for cell in CELLS:
            record = result["learned"][arm][cell]["schemes"]["full"]["strata"]["all"]["choice"]
            i = record["chosen"]
            if i is None:
                expected = None
            else:
                expected = {"candidate_index": i, "signature": ps[i], "cost": record["minimum"],
                            "co_minimum_count": len(record["optima"]), "next_level_gap": record["next_gap"],
                            "predicted_components": extracted["predictions"][arm][cell][i].tolist()}
            equal(extracted["archived_decisions"][arm][cell], expected, "learned float32 decision differs from archive")
    return coverage
