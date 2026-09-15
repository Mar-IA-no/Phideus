"""One recoverable operation under caller-authenticated admission and budget.

No implicit restart after an interrupted producer. Missing-cost recovery must
be explicitly accounted by the campaign controller before this stage resumes.
The controller owns source hashes, GPU ownership, deadlines and resource guards.
"""
from __future__ import annotations

from copy import deepcopy
import math
import time

from .partial_compatibility_cache import encoded


def run_stage(store, folder, identity, *, authorize, produce, check):
    """Under store.exclusive(), run once or recover a completed payload.

authorize(identity) MUST authenticate the actual stage/freeze and reserve cost;
check() MUST enforce its live resource/deadline budget. Both run before producer.
Cost receipts are append-only. A controller may reconcile an interrupted stage
only with a separately validated recovery receipt; this function does not invent
time or label unaccounted work successful.
"""
    if (not isinstance(identity, dict) or not identity
            or not all(callable(f) for f in (authorize, produce, check))):
        raise ValueError("stage requires identity and admission/producer/resource callbacks")
    store._require_lock()
    identity = deepcopy(identity)
    authorize(deepcopy(identity))
    check()
    start_name, finish_name = folder+"/attempt-start.json", folder+"/attempt-finish.json"
    if store.path(start_name).exists():
        start_ref = store.reference(start_name)
        start = store.json(start_ref)
        if (set(start) != {"schema", "binding", "identity"}
                or start["schema"] != "measurement-stage-attempt-v1"
                or encoded(start["binding"]) != encoded(store.binding)
                or encoded(start["identity"]) != encoded(identity)):
            raise ValueError("attempt has another binding or identity")
        if not store.path(finish_name).exists():
            # Deliberately leave any recoverable payload intact. The controller
            # records cost recovery before a subsequent call may return success.
            raise RuntimeError("interrupted stage needs explicit cost reconciliation; no producer rerun")
        finish = store.json(store.reference(finish_name))
        if (set(finish) != {"schema", "binding", "identity", "start", "status", "seconds", "error"}
                or finish["schema"] != "measurement-stage-finish-v1"
                or encoded(finish["binding"]) != encoded(store.binding)
                or encoded(finish["identity"]) != encoded(identity)
                or encoded(finish["start"]) != encoded(start_ref)
                or type(finish["seconds"]) not in (int, float)
                or not math.isfinite(finish["seconds"]) or finish["seconds"] < 0
                or finish["status"] not in ("COMPLETE", "FAILED")):
            raise ValueError("attempt cost receipt differs")
        if finish["status"] != "COMPLETE":
            raise RuntimeError("failed stage requires explicitly accounted recovery")
        restored = store.completed(folder, identity)
        if restored is None:
            raise ValueError("completed attempt has no complete payload")
        check()
        return restored
    if store.path(finish_name).exists() or store.path(folder+"/payload.npz").exists() or store.path(folder+"/complete.json").exists():
        raise ValueError("stage artifacts have no preceding attempt receipt")
    start_ref = store.publish_json(start_name, {"schema": "measurement-stage-attempt-v1",
                                               "binding": store.binding, "identity": identity})
    started = time.monotonic()
    status, error = "FAILED", None
    try:
        result, arrays = produce()
        check()
        store.publish_stage(folder, identity, result, arrays)
        check()
        restored = store.completed(folder, identity)
        status = "COMPLETE"
        return restored
    except BaseException as exception:
        error = f"{type(exception).__name__}: {exception}"
        raise
    finally:
        store.publish_json(finish_name, {"schema": "measurement-stage-finish-v1",
            "binding": store.binding, "identity": identity, "start": start_ref,
            "status": status, "seconds": time.monotonic()-started, "error": error})
