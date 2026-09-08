"""Hash-bound prospective stage authorization; no Torch or new scene generation.

Audit receipts record a coordinator's integration of independent reports.
Hashes provide provenance and fail-closed wiring, not authentication against
an operator deliberately forging every receipt and source file.
"""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import platform

from .partial_compatibility_cache import sha_file
from .source_artifacts import ANCHORS, load_ordered_logits
from .structured_source_artifacts import safe_member, verify_bundle, write_json
from .structured_source_metrics import GAMMAS, SEEDS, SPLIT_SEEDS, select_gammas
from .shared_partial_data import SPLITS as HISTORICAL_SPLITS

ROOT = Path(__file__).resolve().parents[2]
DATA = "data/atencion_armonica"
PLAN = "experiments/atencion_armonica/PLAN_SOURCE_STRUCTURED_READER.md"
PLAN_SHA = "27fcdaff4b065d03f73d2c252fa925aef214dd429ff562b75667c740b4933397"
NEW_SOURCES = (
    PLAN, "src/atencion_armonica/source_coherence.py", "src/atencion_armonica/source_artifacts.py",
    *[f"src/atencion_armonica/structured_source_{name}.py" for name in
      ("reader", "metrics", "artifacts", "gate", "data", "runner", "profile")],
    "experiments/atencion_armonica/run_structured_source.py")


def reference(path):
    path = Path(path).resolve()
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha_file(path)}


def verify_reference(ref):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("expected exact path/hash reference")
    path = safe_member(ROOT, ref["path"])
    if sha_file(path) != ref["sha256"]:
        raise ValueError("referenced artifact changed")
    # Any failure marker in the containing stage revokes its receipts.
    for marker in ("FAILURE.json", "INCOMPLETE.json"):
        if (path.parent/marker).exists():
            raise ValueError("referenced stage is incomplete")
    return path


def read_reference(ref):
    return json.loads(verify_reference(ref).read_bytes())


def legacy_freeze():
    name = "shared_partial_evaluation_freeze_v1.json"
    return read_reference({"path": f"{DATA}/{name}", "sha256": ANCHORS[name]})


def current_sources():
    legacy = legacy_freeze()["source_sha256"]
    if len(legacy) != 25:
        raise ValueError("historical source roster changed")
    for name, digest in legacy.items():
        verify_reference({"path": name, "sha256": digest})
    sources = {name: sha_file(ROOT/name) for name in sorted(set(legacy) | set(NEW_SOURCES))}
    if sources[PLAN] != PLAN_SHA:
        raise ValueError("prospective plan changed")
    return sources


def checkpoints():
    readers = [r for r in legacy_freeze()["readers"] if r["arm"] == "pairs_descriptors"]
    readers.sort(key=lambda r: r["seed"])
    if [r["seed"] for r in readers] != list(SEEDS) or [r["threshold"] for r in readers] != [.55, .65, .6]:
        raise ValueError("historical reader selection changed")
    result = []
    for row in readers:
        ref = {"path": f"{DATA}/shared_partial_training_v1/cells/pairs_descriptors__seed_{row['seed']}/last_epoch.pt",
               "sha256": row["checkpoint_sha256"]}
        verify_reference(ref)
        result.append({**row, "checkpoint": ref})
    return result


def historical_observations():
    """Exact eight-split corpus, transitively pinned to the closed campaign."""
    frozen = legacy_freeze()
    training = frozen["bindings"]["training"]
    training_ref = {"path": Path(training["path"]).relative_to(ROOT).as_posix(), "sha256": training["sha256"]}
    roots = ((f"{DATA}/shared_partial_training_v1", read_reference(training_ref),
              "cache_manifests", "shared_partial_cache_v1", ("development", "train", "validation")),
             (f"{DATA}/shared_partial_test_logits_v1",
              read_reference({"path": f"{DATA}/shared_partial_test_logits_v1/manifest.json",
                              "sha256": ANCHORS["shared_partial_test_logits_v1/manifest.json"]}),
              "test_manifest_sha256", "shared_partial_test_cache_v1",
              ("iid", "ood_beta", "ood_polyphony", "ood_noise", "deformed_family")))
    result = []
    for root, manifest, field, cache, splits in roots:
        request = read_reference({"path": f"{root}/request.json", "sha256": manifest["request_sha256"]})
        if set(request[field]) != set(splits):
            raise ValueError("historical corpus split roster changed")
        for split in splits:
            folder = f"{DATA}/{cache}/{split}"
            m = read_reference({"path": f"{folder}/manifest.json", "sha256": request[field][split]})
            if (m["status"] != "COMPLETE" or m["split"] != split
                    or (m["count"], m["split_seed"]) != HISTORICAL_SPLITS[split]):
                raise ValueError("historical observation identity changed")
            ref = {"path": f"{folder}/observations.jsonl", "sha256": m["observations_sha256"]}
            verify_reference(ref)
            result.append(ref)
    return result


def common_binding():
    return {"plan_sha256": PLAN_SHA, "source_sha256": current_sources(), "checkpoints": checkpoints(),
            "runtime": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                        for name in ("numpy", "scipy", "scikit-learn", "torch")}},
            "protocol": {"split_seeds": SPLIT_SEEDS, "count_per_split": 256, "checkpoint_seeds": list(SEEDS),
                         "gamma_grid": list(GAMMAS), "beta_grid": [1e-5, .02, 1025, 257],
                         "cost_scale_cents": 2., "max_group_size": 8, "pair_normalization": "centered_sum_per_event",
                         "sham_seed": 2026090785, "bootstrap_seed": 2026090786, "bootstrap_resamples": 2000,
                         "cpu_phase_seconds": 1200, "cpu_phase_rss_bytes": 2*1024**3,
                         "preflight_seconds": 120, "preflight_rss_bytes": 1024**3,
                         "forward_seconds": 600, "forward_reserved_bytes": 2*1024**3}}


def verify_audit(ref, common, *, freeze_sha=None):
    audit = read_reference(ref)
    if (audit.get("status") != "PASS" or audit.get("common") != common
            or not isinstance(audit.get("reports"), list) or not audit["reports"]
            or audit.get("freeze_sha256") != freeze_sha):
        raise ValueError("missing or mismatched integrated independent audit")
    for report in audit["reports"]:
        if not verify_reference(report).read_text().strip():
            raise ValueError("empty independent report")


def bundle_reference(ref, role, common):
    path = verify_reference(ref)
    if path.name != "manifest.json":
        raise ValueError("bundle reference must identify its manifest")
    m = verify_bundle(path.parent, ref["sha256"], role=role)
    if m["binding"].get("common") != common:
        raise ValueError("bundle source/protocol/checkpoint binding differs")
    return path.parent, m


def ordered_forward(root, manifest, observations, common):
    """Exact checkpoint roster and full float32 logit identities, NumPy only."""
    rows = json.loads((root/"forward.json").read_bytes())["rows"]
    if len(rows) != len(SEEDS):
        raise ValueError("forward checkpoint roster is incomplete")
    expected_files = {"forward.json"} | {f"seed_{seed}.npz" for seed in SEEDS}
    if set(manifest["artifacts_sha256"]) != expected_files:
        raise ValueError("forward scientific inventory differs")
    result = {}
    for row, checkpoint in zip(rows, common["checkpoints"]):
        seed = checkpoint["seed"]
        if row != {"seed": seed, "checkpoint": checkpoint["checkpoint"],
                   "path": f"seed_{seed}.npz", "count": 256}:
            raise ValueError("forward checkpoint or order differs")
        result[seed] = load_ordered_logits(root/row["path"], observations)
    return result


def _verify_calibration(record, common):
    if record.get("status") != "CALIBRATION_READY" or record.get("common") != common:
        raise ValueError("calibration authorization differs from current implementation")
    verify_audit(record["implementation_audit"], common)
    for field, role in (("cpu_preflight", "cpu_preflight"), ("gpu_profile", "forward_profile")):
        root, _ = bundle_reference(record[field], role, common)
        report = json.loads((root/"report.json").read_bytes())
        if report.get("status") != "READY":
            raise ValueError("resource profile not ready")
        if field == "cpu_preflight":
            values = (report["seconds"], report["peak_rss_bytes"], report["projected_cpu_phase_seconds"], report["projected_peak_rss_bytes"])
            bounds = (120., 1024**3, 1200., 2*1024**3)
        else:
            if report.get("device") != "NVIDIA GeForce RTX 3090":
                raise ValueError("profile is not the authorized device")
            values = (report["projected_forward_seconds"], report["peak_reserved_bytes"])
            bounds = (600., 2*1024**3)
        if any(not isinstance(v, (float, int)) or not 0 <= v < bound for v, bound in zip(values, bounds)):
            raise ValueError("resource projection outside declared envelope")
    if record["historical_observations"] != historical_observations():
        raise ValueError("historical deduplication corpus is incomplete or changed")


def create_calibration_authorization(output, *, audit, cpu_preflight, gpu_profile):
    common = common_binding()
    record = {"status": "CALIBRATION_READY", "common": common, "implementation_audit": audit,
              "cpu_preflight": cpu_preflight, "gpu_profile": gpu_profile,
              "historical_observations": historical_observations()}
    _verify_calibration(record, common)
    write_json(output, record)


def verify_calibration_chain(record, common):
    """Validate all calibration bundles and independently reselect their gamma."""
    auth = record["calibration_authorization"]
    _verify_calibration(read_reference(auth), common)
    roots, manifests = {}, {}
    for key, role in (("data", "calibration_data"), ("logits", "calibration_logits"), ("analysis", "calibration_analysis")):
        roots[key], manifests[key] = bundle_reference(record[f"calibration_{key}"], role, common)
        b = manifests[key]["binding"]
        if (b.get("authorization") != auth or b.get("split") != "calibration"
                or b.get("split_seed") != SPLIT_SEEDS["calibration"] or b.get("count") != 256):
            raise ValueError("calibration stage identity or authorization differs")
    for key in ("logits", "analysis"):
        if manifests[key]["binding"].get("data") != record["calibration_data"]:
            raise ValueError("calibration stages use different observations")
    if manifests["analysis"]["binding"].get("logits") != record["calibration_logits"]:
        raise ValueError("selection uses different forward logits")
    from .structured_source_data import StructuredObservations, load_supervision, prior_corpus
    cache = StructuredObservations(record["calibration_data"], "calibration", common)
    prior = prior_corpus(read_reference(auth), "calibration", {})
    if manifests["data"]["binding"].get("previous") != {} or prior & cache.fingerprints:
        raise ValueError("calibration duplicates an earlier observation")
    load_supervision(cache)
    ordered_forward(roots["logits"], manifests["logits"], cache.observations, common)
    grids = [json.loads(line) for line in (roots["analysis"]/"gamma_grid.jsonl").read_bytes().splitlines()]
    selected = select_gammas(grids, split="calibration")
    if json.loads((roots["analysis"]/"selection.json").read_bytes()) != selected or record["selection"] != selected:
        raise ValueError("gamma was not selected from the bound complete calibration grid")


def create_freeze(output, *, authorization, data, logits, analysis):
    common = common_binding()
    analysis_root, _ = bundle_reference(analysis, "calibration_analysis", common)
    selection = json.loads((analysis_root/"selection.json").read_bytes())
    record = {"status": "FROZEN_BEFORE_TEST", "common": common, "calibration_authorization": authorization,
              "calibration_data": data, "calibration_logits": logits, "calibration_analysis": analysis,
              "selection": selection}
    verify_calibration_chain(record, common)
    write_json(output, record)


def create_test_authorization(output, *, freeze, audit):
    common = common_binding()
    record = read_reference(freeze)
    if record.get("status") != "FROZEN_BEFORE_TEST" or record.get("common") != common:
        raise ValueError("prospective freeze changed")
    verify_calibration_chain(record, common)
    verify_audit(audit, common, freeze_sha=freeze["sha256"])
    write_json(output, {"status": "TEST_READY", "common": common, "freeze": freeze, "freeze_audit": audit})


def verify_authorization(ref, split):
    if split not in SPLIT_SEEDS:
        raise ValueError("unknown prospective split")
    common = common_binding()
    record = read_reference(ref)
    if split == "calibration":
        _verify_calibration(record, common)
        return record
    if record.get("status") != "TEST_READY" or record.get("common") != common:
        raise PermissionError("test requires an independently audited prospective freeze")
    freeze = read_reference(record["freeze"])
    if freeze.get("status") != "FROZEN_BEFORE_TEST" or freeze.get("common") != common:
        raise ValueError("prospective freeze changed")
    verify_audit(record["freeze_audit"], common, freeze_sha=record["freeze"]["sha256"])
    verify_calibration_chain(freeze, common)
    return record
