"""Explicit observed-fixture recipes, not test draws or semantic exclusions.

Source links are reviewed recipes: parsing Python cannot prove their full
semantic correspondence. The operator pins every source before construction;
independent review covers that correspondence and the finite fixture roster.
"""
from __future__ import annotations

import numpy as np

from .geometric_decision_core import recentered_roundtrip
from .geometric_decision_exclusions import PRIOR, INCIDENT, RETIRED, checked_prior
from .generative_evidence_exclusions import fingerprint

PREFIX = "experiments/atencion_armonica/"
SOURCE_PATHS = tuple(PREFIX+name for name in (
    "test_generative_evidence.py", "test_geometric_decision_core.py",
    "test_geometric_decision_profile.py", "test_geometric_decision_observables.py",
    "test_geometric_decision_pipeline.py", "test_geometric_decision_scene_store.py",
    "test_operator_objective_sources.py"))


def build_catalog(reader, source_refs, *, check):
    if [r["path"] for r in source_refs] != list(SOURCE_PATHS):
        raise ValueError("mechanical catalog source roster differs")
    for ref in source_refs:
        check()
        reader.read(ref)
    refs = {r["path"].rsplit("/", 1)[1]: r for r in source_refs}
    seen = checked_prior(reader.json(PRIOR)).copy()
    records = []
    def add(name, values, source):
        vectors = [v.astype(np.float64).tolist() for v in values]
        hashes = set(map(fingerprint, vectors))
        records.append({"name": name, "source": source, "q32": vectors, "alias": hashes <= seen})
        seen.update(hashes)
    q8 = np.arange(8, dtype=np.float32)/8
    # Existing q8 is a dependency of the new fixtures, not a fresh example.
    add("shared_arithmetic8", [q8], refs["test_generative_evidence.py"])
    add("arithmetic12", [np.arange(12, dtype=np.float32)/8], refs["test_geometric_decision_core.py"])
    add("profile_linspace8", [np.linspace(0, 1, 8, dtype=np.float32)], refs["test_geometric_decision_profile.py"])
    add("operator_integer8_16", [np.arange(n, dtype=np.float32) for n in (8, 16)], refs["test_operator_objective_sources.py"])
    add("canonical_integer8", [np.arange(8, dtype=np.float32)], refs["test_geometric_decision_observables.py"])
    core = np.array([-.51, -.44, -.23, -.12, .09, .31, .40, .50], np.float32)
    tied = q8.copy()
    tied[1] = tied[0]
    for name, q, source in (("core_roundtrip", core, "test_geometric_decision_core.py"),
                             ("tied_roundtrip", tied, "test_geometric_decision_observables.py"),
                             ("pipeline_roundtrip", q8, "test_geometric_decision_pipeline.py")):
        result = recentered_roundtrip(q)
        add(name, [result[k] for k in ("original", "shifted32", "q_center", "q_probe")], refs[source])
    add("store_roundtrip_alias", [recentered_roundtrip(q8)["q_probe"]], refs["test_geometric_decision_scene_store.py"])
    incident = reader.json(INCIDENT)
    ref = incident["reconstructed"]["observation"]
    observation_ref = {"path": INCIDENT["path"].rsplit("/", 1)[0]+"/"+ref["path"], "sha256": ref["sha256"]}
    observation = reader.json(observation_ref)
    if (incident["status"] != "DETERMINISTIC_RECONSTRUCTION_NOT_ORIGINAL_BYTE_PRESERVATION"
            or observation["split_seed"] != RETIRED or observation["scene_id"] != 0
            or fingerprint(observation["log_f"]) != incident["reconstructed"]["observation_q32_sorted_sha256"]):
        raise ValueError("retired reconstruction provenance differs")
    add("retired_iid0_reconstruction", [np.asarray(observation["log_f"], np.float32)], observation_ref)
    return {"schema": "geometric-decision-mechanical-exclusions-v1", "retired_seed": RETIRED,
        "incident_receipt": INCIDENT, "records": records,
        "scope": "Reviewed explicit q32 observations since prior review06, plus declared derived-coordinate diagnostics. "
        "Prior observations/aliases remain inherited; historical generative interventions changed features, not observations. "
        "Training/profile real inputs alias OPEN. Float64 fitter-group primitives, logits, scores and targets are not "
        "full q32 scenes and are not reinterpreted as such. Byte exclusions do not establish semantic independence."}
