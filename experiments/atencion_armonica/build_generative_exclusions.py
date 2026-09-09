"""Recover declared mechanical q32 fixtures and inventory prior observations.

This prepares review material, not a test freeze. It never calls a campaign
sampler, fitter, model, or sidecar reader. Seed713 below reconstructs only the
already specified arithmetic quantization test, not prospective scenes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.atencion_armonica import observable_source_rivals as law
from src.atencion_armonica.generative_evidence_exclusions import build_exclusions, fingerprint
from src.atencion_armonica.generative_evidence_reuse import ROOT, OpenReuse
from src.atencion_armonica.generative_evidence_storage import read_scene, write_json

TESTS = ROOT/"experiments/atencion_armonica"
PROFILES = (
    ("cpu", {"path": ".agent-work/phideus-generative-evidence-20260909/profile-cpu-01/report.json",
             "sha256": "f27b7bb1391057f55f220189ae136f96701cae5c31ceaab4283ced308b1e6402"}),
    ("gpu", {"path": ".agent-work/phideus-generative-evidence-20260909/profile-gpu-01/report.json",
             "sha256": "c3144164274e154f2dee4ca93eea81410d68558da623545006d751ee473051ea"}),
)


def reference(path):
    path = Path(path)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def mechanical_catalog():
    reuse = OpenReuse()
    reader, records = reuse.reader, []

    def add(name, vectors, source, alias=False):
        source = source if isinstance(source, dict) else reference(source)
        reader.read(source)
        qs = [np.asarray(q, dtype=np.float32).astype(np.float64).tolist() for q in vectors]
        for q in qs:
            fingerprint(q)
        records.append({"name": name, "source": source, "q32": qs, "alias": alias})

    # The common arithmetic fixture is reused by cache/corpus/model/inference
    # tests. Permutations and repeated scene IDs do not create new q32 vectors.
    q8, q16 = (np.arange(n, dtype=np.float32)/8 for n in (8, 16))
    add("arithmetic8_16_shift_ties", [q8, q16, q8+np.float32(2),
        np.array([.5, 0, .5, .25, .125, .75, .625, .875], np.float32)],
        TESTS/"test_generative_evidence.py")

    # Explicit equations from the already run mechanical rival tests. No fit.
    raw = np.concatenate([np.log(f0)+law.template(range(1, 5), beta, 0.)
                          for f0, beta in ((120., 1e-4), (410., 1e-3))])
    q = law.centered(raw).astype(np.float32)[np.argsort(raw, kind="stable")]
    beta, _ = law.Grid(5, 5).values("base-low")
    raw16 = np.concatenate([np.log(f0)+law.template(range(1, 9), b, 0.)
                            for f0, b in ((100., beta[1]), (500., beta[3]))])
    qgrid = law.centered(raw16).astype(np.float32)[np.argsort(raw16, kind="stable")]
    quantized = []
    rng = np.random.default_rng(713)
    for _ in range(50):
        quantized.append(law.centered(rng.normal(size=13)).astype(np.float32))
        rng.normal(size=13)  # The test's prediction draw, not an observation.
    add("rival_geometry_and_quantization", [q, qgrid, *quantized, [-1., 1.],
        np.arange(8, dtype=np.float32), np.arange(16, dtype=np.float32)],
        TESTS/"test_observable_source_rivals.py")
    # Conservative projections of rejected finite vectors are separately
    # declared; they are exclusions, never accepted physical measurements.
    add("rival_rejected_finite_q32_projections", [[.1, .2], np.linspace(-1, 1, 8)],
        TESTS/"test_observable_source_rivals.py")
    add("exclusion_rejected_scalar_vector_projection", [[.1]],
        TESTS/"test_generative_evidence_exclusions.py")

    # Equation/noise fixture from test_observable_rival_campaign.truth_fixture.
    ideal = np.concatenate([np.log(f)+np.log(np.arange(1, 5, dtype=np.float64))
        +.5*np.log1p(1e-4*np.arange(1, 5, dtype=np.float64)**2) for f in (110., 420.)])
    noise = np.linspace(-2, 2, len(ideal))*law.CENTS_TO_LOG
    qtruth = (ideal+noise-float((ideal+noise).mean())).astype(np.float32)
    add("rival_truth_and_port_fixtures", [qtruth, np.arange(8, dtype=np.float32)],
        TESTS/"test_observable_rival_campaign.py")

    # Read OPEN observations only, not OpenShard (which also loads logits).
    bundle = reuse.bundle(reuse.shards["train"][0]["data"], "learned_observation_shard")
    open_rows = [json.loads(line) for line in bundle.read("observations.jsonl").splitlines()]
    base = np.asarray(open_rows[0]["log_f"], np.float64)
    changed = base.copy()
    changed[0] = float(np.float32(changed[0]+.25))
    add("rejected_open_truth_q32_mutation", [changed], TESTS/"test_generative_evidence_supervision.py")
    changed = base.copy()
    changed[0] += 1.
    add("open_copy_isolation_q32_projection", [changed], TESTS/"test_generative_evidence_reuse.py")

    # The historical reader's validation-only .1 mutation is not an accepted
    # observation. Include its q32 projection conservatively as above.
    auth = reader.json(reuse.prepared["authorization"])
    old_ref = next(r for r in auth["prior_corpus"]["observation_references"]
                   if r["path"].endswith("structured_source_reader_v1/iid_data/observations.jsonl"))
    old = json.loads(reader.read(old_ref).splitlines()[0])
    changed = np.asarray(old["log_f"], np.float64)
    changed[0] = .1
    add("historical_validation_q32_projection", [changed], TESTS/"test_learned_partition_data.py")

    # Actual CPU/GPU profiles reused TRAIN0..31. Verify saved observable bytes,
    # not just their declared IDs, and retain each alias's precise source.
    for device, report_ref in PROFILES:
        report = reader.json(report_ref)
        if (report["status"] != "MEASURED" or report["binding"]["scene_ids"] != list(range(32))
                or report["binding"]["split"] != "train" or report["binding"]["split_seed"] != 2026090880
                or len(report["fit_scenes"]) != 32):
            raise ValueError("profile alias roster differs")
        add(f"profile_{device}_authority", [], report_ref, alias=True)
        for i, receipt in enumerate(report["fit_scenes"]):
            if receipt["scene_id"] != i:
                raise ValueError("profile alias scene ID differs")
            path = ROOT/Path(report_ref["path"]).parent/f"scene_{i:05d}.json.gz"
            ref = {"path": path.relative_to(ROOT).as_posix(), "sha256": receipt["scene"]["sha256"]}
            reader.read(ref)
            saved = read_scene(path, receipt["scene"])
            if saved["observation"] != open_rows[i]:
                raise ValueError("saved profile observable is not its OPEN alias")
            add(f"profile_{device}_train_{i:02d}", [saved["observation"]["log_f"]], ref, alias=True)
    return {"schema": "generative-evidence-mechanical-exclusions-v1", "records": records,
        "scope": {"status": "REVIEW_MATERIAL_NOT_TEST_FREEZE",
            "included": "generative arithmetic fixtures, rival geometry/noise fixtures, rejected finite projections, OPEN profile aliases",
            "excluded_nonobservations": "logits, embeddings, incidence, targets, parameter vectors, nonnumeric, nonfinite and nonvector rejection fixtures",
            "no_campaign_draws": True, "no_fits_or_forwards": True,
            "builder": reference(Path(__file__)), "consumed_sha256": dict(sorted(reader.consumed.items()))}}


def run(output):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT/".agent-work/phideus-exclusions-20260909"):
        raise ValueError("review output must stay in the owned exclusion workspace")
    if output.exists():
        raise FileExistsError("use a fresh review directory; do not overwrite evidence")
    output.mkdir(parents=True)
    catalog = output/"mechanical_catalog.json"
    write_json(catalog, mechanical_catalog())
    result = build_exclusions(reference(catalog))
    path = output/"inventory.json"
    write_json(path, result)
    return {"catalog": reference(catalog), "inventory": reference(path),
            "unique_count": result["unique_count"], "groups": len(result["groups"]),
            "test_access": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output)))
