"""CPU-only archive and exclusions after COMPLETE training and selection.

This preparation is not a prospective freeze and cannot draw any scene.
"""
from __future__ import annotations

import time
LAUNCH_STARTED = time.monotonic()

import fcntl
import json
import os
from pathlib import Path
import platform

import numpy as np
import torch

from experiments.atencion_armonica.prepare_geometric_decision_open import ROOT, PROTOCOL, PROTOCOL_SHA, reference, code_snapshot
from experiments.atencion_armonica.profile_geometric_decision import admitted_open
from experiments.atencion_armonica.select_geometric_decision import admitted_training, bounded_operation
from src.atencion_armonica.geometric_decision_archive_admission import admitted_selection, verify_heads
from src.atencion_armonica.geometric_decision_head_archive import preserve_heads
from src.atencion_armonica.geometric_decision_mechanical_catalog import build_catalog, SOURCE_PATHS
from src.atencion_armonica.geometric_decision_exclusions import extend_exclusions
from src.atencion_armonica.geometric_decision_store import ArtifactStore, BASES
from src.atencion_armonica.generative_evidence_reuse import VerifiedBytes

OPERATOR = "experiments/atencion_armonica/prepare_geometric_decision_archive.py"
PLAN = "experiments/atencion_armonica/PLAN_GEOMETRIC_DECISION_FRESH.md"


def sources():
    return sorted([*code_snapshot(), *[reference(p) for p in (OPERATOR, PLAN,
        "experiments/atencion_armonica/select_geometric_decision.py",
        "experiments/atencion_armonica/profile_geometric_decision.py")]], key=lambda r: r["path"])


def execute(control, output, selection, selection_ref, campaign, campaign_ref, reader, manifest_ref, *,
            source_refs, started_at, verify, reservation):
    if control.json(manifest_ref) != {"operation": "archive-and-exclusions", "binding": output.binding, "root": str(output.root)}:
        raise ValueError("archive operation manifest differs")
    def operation(check):
        archive = preserve_heads(selection, selection_ref, campaign, campaign_ref, output, check=check)
        verify_heads(output, archive, selection, selection_ref, check=check)
        catalog = build_catalog(reader, source_refs, check=check)
        catalog_ref = output.publish_json("mechanical-catalog.json", catalog)
        project_ref = {"path": output.path(catalog_ref["path"]).relative_to(reader.root).as_posix(), "sha256": catalog_ref["sha256"]}
        inventory = extend_exclusions(reader, project_ref, check=check)
        inventory_ref = output.publish_json("exclusions.json", inventory)
        result = output.publish_json("complete.json", {"schema": "geometric-decision-archive-complete-v1",
            "binding": output.binding, "heads": archive, "exclusions": inventory_ref,
            "catalog": catalog_ref, "fresh_tests": "not opened", "test_authority": False})
        return control.publish_json("outputs/archive.json", {"manifest": manifest_ref,
            "root": str(output.root), "binding": output.reference(output.path("binding.json")), "complete": result})
    return bounded_operation(control, manifest_ref, stage="fresh", started_at=started_at,
        verify=verify, reservation=reservation, operation=operation)


def main():
    if (Path.cwd().resolve() != ROOT or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))):
        raise RuntimeError("archive requires project cwd, one CPU thread and hidden CUDA")
    torch.set_num_threads(1)
    control, _, _ = admitted_open()
    with control.path("operator.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol, code = reference(PROTOCOL), sources()
        if protocol["sha256"] != PROTOCOL_SHA:
            raise ValueError("frozen protocol differs")
        campaign, campaign_ref, training = admitted_training(control, root=BASES[0]/"training")
        selection, selection_ref, selected = admitted_selection(control, root=BASES[0]/"selection",
            campaign=campaign, campaign_ref=campaign_ref, training=training)
        source_refs = [{k: reference(p)[k] for k in ("path", "sha256")} for p in SOURCE_PATHS]
        binding = {"schema": "geometric-decision-archive-binding-v1", "selection": selection_ref,
            "selection_binding": selection.binding, "admitted_selection": selected,
            "protocol": protocol, "code": code, "catalog_sources": source_refs,
            "runtime": {"device": "cpu", "threads": 1, "python": platform.python_version(),
                        "numpy": np.__version__, "torch": str(torch.__version__)},
            "scope": "archive/exclusions only, not fresh draw authority"}
        output = ArtifactStore(BASES[0]/"archive", binding=binding)
        manifest = control.publish_json("manifests/archive.json", {"operation": "archive-and-exclusions",
            "binding": binding, "root": str(output.root)})
        def verify():
            if reference(PROTOCOL) != protocol or sources() != code:
                raise ValueError("archive code/protocol changed during operation")
            for ref in source_refs:
                if reference(ref["path"])["sha256"] != ref["sha256"]:
                    raise ValueError("mechanical source changed during archive")
        result = execute(control, output, selection, selection_ref, campaign, campaign_ref,
            VerifiedBytes(ROOT), manifest, source_refs=source_refs, started_at=LAUNCH_STARTED,
            verify=verify, reservation=120.)
        print(json.dumps({"status": "ARCHIVE_COMPLETE_FRESH_NOT_OPENED", **result}), flush=True)


if __name__ == "__main__":
    main()
