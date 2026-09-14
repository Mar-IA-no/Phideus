"""Extend reviewed exclusions through an already-closed observable chain.

No sampler, supervision, fit, model or new test access. Historical metrics
are not parsed. A pinned inventory preserves its prior provenance; newly
consumed observations are reached through the pinned campaign closure.
"""
from __future__ import annotations

from .generative_evidence_exclusions import Inventory, SCHEMA, fingerprint

BASE = "data/atencion_armonica/generative_evidence_reader_v1"
PRIOR = {"path": ".agent-work/phideus-exclusions-20260909/review-06/inventory.json",
         "sha256": "d3b56a7446be7e0bd27b5d8be237cf22f3f50b9fa907631df4cc06eec5847de3"}
COMPLETION = {"path": BASE+"/fresh/test_completion.json",
              "sha256": "160473a7c88ab9e9a296a9e37a3145546a6ccc7019dbec473394ccd966cae3f9"}
OLD_FREEZE = {"path": BASE+"/test_freeze.json",
              "sha256": "8b1030ca466ef289eeef738149b878886a640fbc598774cb6fabd01147ee7937"}
OLD_TESTS = {"iid": 2026090982, "ood_beta": 2026090983,
             "ood_polyphony": 2026090984, "deformed_family": 2026090985}
RETIRED = 2026091482
INCIDENT = {"path": ".agent-work/phideus-geometric-decision-20260914/audit754/incident-reconstruction/receipt.json",
            "sha256": "70a9d7ab902e8076dfd87a5fca390f9a262903001cbaf8895956d9e12daef61d"}


def checked_prior(prior):
    if (prior.get("schema") != SCHEMA or prior.get("status") != "EXCLUSIONS_ONLY_NOT_TEST_AUTHORIZED"
            or prior.get("test_access") is not False or not isinstance(prior.get("groups"), list)):
        raise ValueError("prior exclusion inventory is not the reviewed observable roster")
    seen, names = set(), set()
    for row in prior["groups"]:
        hashes = row["fingerprints"]
        if (row["name"] in names or row["count"] != len(hashes) or row["unique_count"] != len(set(hashes))
                or row["additional_unique"] != len(set(hashes)-seen)
                or type(row["declared_alias"]) is not bool or (row["declared_alias"] and not set(hashes) <= seen)
                or prior["consumed_sha256"].get(row["source"]["path"]) != row["source"]["sha256"]
                or any(type(h) is not str or len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in hashes)):
            raise ValueError("prior exclusion groups or source ledger differ")
        names.add(row["name"])
        seen.update(hashes)
    if prior["fingerprints"] != sorted(seen) or prior["unique_count"] != len(seen):
        raise ValueError("prior exclusion union differs")
    return seen


def extend_exclusions(reader, mechanical_ref, *, check):
    """Use a reader with authenticated read/json and consumed ledger.

    The mechanical catalog is explicitly reviewed and pinned by the caller;
    it must include the retired-scene reconstruction and all observed q32
    fixtures. Completeness of that catalog remains an independent review
    obligation. This function does not discover fixtures by running tests.
    """
    check()
    prior = reader.json(PRIOR)
    seen = checked_prior(prior)
    inventory = Inventory(reader)
    inventory.groups = list(prior["groups"])
    inventory.seen = seen.copy()
    closed = reader.json(COMPLETION)
    if (closed.get("schema") != "generative-evidence-fresh-tests-complete-v1"
            or closed.get("status") != "FRESH_TESTS_EVALUATED_REPLAYED_NOT_PROMOTED"
            or set(closed["tests"]) != set(OLD_TESTS) or closed["stage_count"] != 13
            or len(closed["stages"]) != 13 or closed["stages"][0]["output"] != OLD_FREEZE):
        raise ValueError("historical campaign is not the pinned full closure")
    freeze = reader.json(OLD_FREEZE)
    if freeze["exclusions"] != PRIOR or freeze["tests"] != [
            {"split": s, "split_seed": seed, "scene_count": 512, "scene_ids": list(range(512))}
            for s, seed in OLD_TESTS.items()]:
        raise ValueError("historical freeze roster or prior inventory differs")
    consumed_sources = []
    for split, seed in OLD_TESTS.items():
        check()
        seal = reader.json(closed["tests"][split]["prediction_seal"])
        if (seal["binding"] != {"test_freeze": OLD_FREEZE} or seal["split"] != split
                or seal["status"] != "FRESH_PREDICTIONS_SEALED_NO_TRUTH_ACCESS"
                or seal["truth_access"] is not False or seal["prediction_count"] != 45):
            raise ValueError("historical prediction seal differs")
        index = reader.json(seal["prediction_index"])
        choices = reader.json(index["choices"])
        for value in (index, choices):
            if (value["binding"] != {"test_freeze": OLD_FREEZE} or value["split"] != split
                    or value["split_seed"] != seed or value["scene_ids"] != list(range(512))
                    or value["truth_access"] is not False):
                raise ValueError("historical observable index identity differs")
        if len(choices["records"]) != 512:
            raise ValueError("historical observable roster is incomplete")
        for sid, row in enumerate(choices["records"]):
            check()
            ref = row["source"]
            if row["scene_id"] != sid or ref["path"] != f"{BASE}/fresh/{split}/source/scenes/{sid:05d}.json":
                raise ValueError("historical source order or path differs")
            source = reader.json(ref)
            if (source["binding"] != {"test_freeze": OLD_FREEZE} or source["split"] != split
                    or source["scene_id"] != sid or source["schema"] != "generative-evidence-fresh-source-scene-v1"):
                raise ValueError("historical source identity differs")
            inventory.add(f"closed_generative_{split}_{sid:05d}", [source["q32"]], source=ref)
            consumed_sources.append(ref)
    catalog = reader.json(mechanical_ref)
    if (set(catalog) != {"schema", "scope", "records", "retired_seed", "incident_receipt"}
            or catalog["schema"] != "geometric-decision-mechanical-exclusions-v1"
            or catalog["retired_seed"] != RETIRED or catalog["incident_receipt"] != INCIDENT or not catalog["records"]):
        raise ValueError("explicit complete mechanical catalog and retired seed required")
    incident = reader.json(catalog["incident_receipt"])
    if (incident["schema"] != "r754-contaminated-fixture-reconstruction-v1"
            or incident["status"] != "DETERMINISTIC_RECONSTRUCTION_NOT_ORIGINAL_BYTE_PRESERVATION"
            or incident["original_bytes_recoverable"] is not False
            or incident["scope"]["seed"] != RETIRED or incident["scope"]["scene_id"] != 0):
        raise ValueError("incident must retain reconstruction-only authority")
    reconstruction_hash = incident["reconstructed"]["observation_q32_sorted_sha256"]
    observed = incident["reconstructed"]["observation"]
    incident_source = {"path": INCIDENT["path"].rsplit("/", 1)[0]+"/"+observed["path"], "sha256": observed["sha256"]}
    observation = reader.json(incident_source)
    if (set(observation) != {"scene_id", "split_seed", "log_f"} or observation["scene_id"] != 0
            or observation["split_seed"] != RETIRED or fingerprint(observation["log_f"]) != reconstruction_hash):
        raise ValueError("reconstructed observation differs from its pinned receipt")
    additional = set()
    for entry in catalog["records"]:
        check()
        if set(entry) != {"name", "source", "q32", "alias"} or type(entry["alias"]) is not bool:
            raise ValueError("mechanical catalog record schema differs")
        inventory.add("geometric_fixture_"+entry["name"], entry["q32"], source=entry["source"], alias=entry["alias"])
        additional.update(map(fingerprint, entry["q32"]))
    if reconstruction_hash not in additional:
        raise ValueError("retired scene reconstruction absent from mechanical catalog")
    result = inventory.result()
    new_consumed = result["consumed_sha256"].copy()
    inherited = prior["consumed_sha256"]
    if any(path in inherited and inherited[path] != digest for path, digest in new_consumed.items()):
        raise ValueError("new consumption conflicts with inherited source bytes")
    result["consumed_sha256"] = dict(sorted({**inherited, **new_consumed}.items()))
    # These are inherited receipts, not a claim that their original payloads
    # were reread in this operation. The reviewed parent bytes are pinned.
    result["extension"] = {"parent": PRIOR, "closed_campaign": COMPLETION,
        "newly_read_historical_observations": len(consumed_sources),
        "mechanical_catalog": mechanical_ref, "retired_seed": RETIRED,
        "inherited_consumed_sha256": inherited, "newly_consumed_sha256": new_consumed,
        "authority": "byte exclusions only; no semantic independence or test execution authority"}
    return result
