"""Bounded real OPEN preparation. No training, fresh data, fitting or CUDA port."""
from __future__ import annotations

import time

LAUNCH_STARTED = time.monotonic()

import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import sys

import numpy as np
import torch

from src.atencion_armonica.geometric_decision_budget import LIMITS, StageBudget, BudgetExceeded, admit_projection
from src.atencion_armonica.geometric_decision_corpus import OpenPreparation, source_identity
from src.atencion_armonica.geometric_decision_open import OpenSource
from src.atencion_armonica.geometric_decision_store import ArtifactStore, BASES

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "experiments/atencion_armonica/PROTOCOL_GEOMETRIC_DECISION_ENERGY.md"
PROTOCOL_SHA = "72ef0c7b0e6e5ae1edc8f78cb8701f7bd5c31c2299c1ba756cd41ccdd5b62fc6"
SMOKE = ".agent-work/phideus-geometric-decision-20260914/open-source-smoke-01.json"
SMOKE_SHA = "ccc598cd86b0580fa327eeed217415c7242cb483955bc704af25b07c1eb6a1c7"
SOURCE_REFS = {
    "binding_ref": {"path": "binding.json", "bytes": 189,
        "sha256": "547d00a8f36b1d1c392cc7eb0203ee1a851cfd278de7f298a8914daf0466ee83"},
    "prepared_ref": {"path": "open_prepared.json", "bytes": 3334675,
        "sha256": "e4fe5a8bfef27fe19211598f3597998e4059bc3ab03901e8daf584b3632f8e9a"},
    "delivered_ref": {"path": "delivered/index.json", "bytes": 6342,
        "sha256": "601698dffe32f73a8069a86f4f33202283400084116d5015409d3086da799c3c"},
}


def reference(relative):
    path = ROOT/relative
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.resolve().is_relative_to(ROOT):
        raise ValueError("operator source must remain inside its project without symlinks")
    raw = path.read_bytes()
    return {"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def code_snapshot():
    paths = {Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = Path(name).resolve()
            if path.suffix == ".py" and path.is_relative_to(ROOT/"src"):
                paths.add(path)
    return [reference(p.relative_to(ROOT).as_posix()) for p in sorted(paths)]


def execute(preparation, control, manifest_ref, *, reservation_seconds, started_at,
            resource_overrides=None, verify=lambda: None, progress=print):
    """Exclusive stage operation; fixture tests can replace resources, not science."""
    lock_path = control.path("operator.lock")
    with lock_path.open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = control.json(manifest_ref)
        if (manifest["operation"] != "open" or manifest["preparation_binding"] != preparation.store.binding
                or manifest["source"] != source_identity(preparation.source)):
            raise ValueError("operation does not bind this OPEN preparation/source")
        verify()
        budget = StageBudget(control, "open", manifest_ref=manifest_ref, reservation_seconds=reservation_seconds,
            started_at=started_at, prior_charges=control.binding["prior_charges"],
            output_roots=[Path(p) for p in control.binding["output_roots"]], **(resource_overrides or {}))
        def pause(signum, frame):
            raise InterruptedError("OPEN operator interrupted; completed shards remain reusable")
        old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
        try:
            for sig in old:
                signal.signal(sig, pause)
            signal.setitimer(signal.ITIMER_REAL, max(.001, reservation_seconds-(time.monotonic()-started_at)))
            ref = preparation.prepare(check=budget.check, progress=progress)
            budget.check(force_resources=True)
            verify()
            preparation.completion(ref)
            result = control.publish_json("outputs/open.json",
                {"manifest": manifest_ref, "root": str(preparation.store.root), "complete": ref})
            finish = budget.finish("COMPLETE", completion=result)
            return {"output": result, "finish": finish}
        except BaseException as exc:
            if not budget.closed:
                status = "LIMIT_REACHED" if isinstance(exc, BudgetExceeded) else "PAUSED" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "FAILED"
                budget.finish(status)
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.)
            for sig, handler in old.items():
                signal.signal(sig, handler)


def main():
    if (Path.cwd().resolve() != ROOT or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))):
        raise RuntimeError("requires project cwd, disabled CUDA and one-thread environment")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol, smoke_ref = reference(PROTOCOL), reference(SMOKE)
    if protocol["sha256"] != PROTOCOL_SHA or smoke_ref["sha256"] != SMOKE_SHA:
        raise ValueError("fixed protocol or charged historical smoke changed")
    smoke_raw = (ROOT/SMOKE).read_bytes()
    if hashlib.sha256(smoke_raw).hexdigest() != smoke_ref["sha256"]:
        raise ValueError("consumed smoke bytes differ from the pinned receipt")
    smoke = json.loads(smoke_raw)
    prior = [{"stage": "open", "seconds": smoke["seconds"], "source": smoke_ref}]
    control = ArtifactStore(BASES[0]/"control", binding={"schema": "geometric-decision-control-v1",
        "protocol": protocol, "limits": LIMITS, "prior_charges": prior, "output_roots": [str(p) for p in BASES]})
    source = OpenSource(ROOT/"data/atencion_armonica/generative_evidence_reader_v1", **SOURCE_REFS)
    code = code_snapshot()
    runtime = {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
               "platform": platform.platform(), "device": "cpu", "threads": 1}
    binding = {"schema": "geometric-decision-open-binding-v1", "source": source_identity(source),
               "protocol": protocol, "code": code, "runtime": runtime}
    # Seventeen observable-shard passes: eight for scale, nine for adaptation.
    # The historical smoke includes setup; this is an OPEN timing forecast,
    # not a profile of the head or fitter and not a guaranteed upper bound.
    projection = admit_projection("open", measured_seconds=smoke["seconds"], units_measured=1,
                                  remaining_units=17, charged_seconds=smoke["seconds"])
    manifest_ref = control.publish_json("manifests/open.json", {"operation": "open", "source": source_identity(source),
        "preparation_binding": binding, "source_refs": SOURCE_REFS, "projection": projection})
    prepared = OpenPreparation(source, ArtifactStore(BASES[0]/"open", binding=binding))
    def verify():
        if code_snapshot() != code or reference(PROTOCOL) != protocol or reference(SMOKE) != smoke_ref:
            raise ValueError("operator code/protocol/source evidence changed during OPEN preparation")
    result = execute(prepared, control, manifest_ref, reservation_seconds=300., started_at=LAUNCH_STARTED,
                     verify=verify, progress=lambda row: print(json.dumps(row), flush=True))
    print(json.dumps({"status": "OPEN_PREPARED_NOT_TRAINED", **result}), flush=True)


if __name__ == "__main__":
    main()
