"""Admit the completed TRAIN profile and derive its exclusion extension.

Read-only computation behind the enclosing operator's budget/check. The
result is not a prospective freeze, a draw permission or a complete forecast.
"""
from __future__ import annotations

import math

from .geometric_decision_open import ReadOnlyStore
from .geometric_decision_profile_exclusions import extend_profile_exclusions

PROFILE_FINISH = {"path": "attempts/0010/finish.json", "bytes": 544,
    "sha256": "e05314b43451bb3993b9e01124c0d970684c974152dfe3537366c171cffd30d3"}


def admit_profile(control, archive, archive_complete, archive_evidence, reader, *, check):
    """Caller already admits archive COMPLETE from calibrated selection.

Archive exclusions are pinned by that complete record, never by an arbitrary
path supplied separately. Every profile source hash is revalidated before use.
"""
    check()
    finish = control.json(PROFILE_FINISH)
    if finish["status"] != "COMPLETE" or finish["completion"] is None:
        raise ValueError("observed profile is not COMPLETE")
    start = control.json(finish["start"])
    manifest = control.json(start["manifest"])
    output = control.json(finish["completion"])
    root = archive.root.parent/"profiles/observed-cuda-0"
    if (start["stage"] != "profile" or start["binding"] != control.binding
            or manifest["operation"] != "profile-observed" or manifest["root"] != str(root)
            or output["root"] != str(root) or output["manifest"] != start["manifest"]):
        raise ValueError("profile completion provenance differs")
    profile = ReadOnlyStore(root, binding_ref=output["binding"])
    binding = profile.binding
    if (binding != manifest["binding"] or binding["schema"] != "geometric-decision-observed-profile-binding-v1"
            or binding["archive"] != archive_evidence or binding["protocol"] != control.binding["protocol"]
            or binding["test_authority"] is not False or binding["scene_ids"] != list(range(16))
            or archive_complete["binding"] != archive.binding
            or archive_complete["test_authority"] is not False):
        raise ValueError("profile belongs to another archive/protocol or scope")
    if not binding["code"] or len({r["path"] for r in binding["code"]}) != len(binding["code"]):
        raise ValueError("profile source roster is empty or duplicated")
    for ref in binding["code"]:
        check()
        raw = reader.read({k: ref[k] for k in ("path", "sha256")})
        if len(raw) != ref["bytes"]:
            raise ValueError("profile source size differs")
    report = profile.json(output["result"])
    if (report["binding"] != binding or report["schema"] != "geometric-decision-observed-profile-v1"
            or report["result"] != report["recovery"] or report["new_test_observations"] != 0
            or report["forecast"]["test_authority"] is not False):
        raise ValueError("profile is not recovered OPEN-only evidence")
    seconds, elapsed = finish["seconds"], report["elapsed_to_forecast_seconds"]
    if (not all(type(v) in (int, float) and math.isfinite(v) for v in (seconds, elapsed))
            or not 0 < elapsed <= seconds <= start["reservation_seconds"]):
        raise ValueError("profile closing time is invalid")
    tail = seconds-elapsed
    prior = archive_complete["exclusions"]
    archive.read(prior)
    prior_ref = {"path": archive.path(prior["path"]).relative_to(reader.root).as_posix(),
                 "sha256": prior["sha256"]}
    exclusion = extend_profile_exclusions(reader, prior_ref, profile, output["result"], check=check)
    # Four scenario setup/closing allowances, the same convention as the profile.
    extra = 4*report["forecast"]["margin"]*tail
    forecast = {"schema": "geometric-decision-observed-admission-v1", "test_authority": False,
        "profile_finish": PROFILE_FINISH, "profile_output": finish["completion"],
        "profile_report": output["result"], "closing_tail_seconds": tail,
        "observed_path_with_closing_seconds": report["forecast"]["observed_path_seconds"]+extra,
        "observable_recovery_with_closing_seconds": report["forecast"]["observable_recovery_seconds"]+extra,
        "projected_profile_bytes": report["forecast"]["projected_bytes"],
        "still_required": ["fresh sampler/draw IO", "global seal", "privileged metrics/bootstrap",
                           "exclusion extension and final supervisor admission"]}
    check()
    return {"profile": profile, "report": report, "exclusions": exclusion, "forecast": forecast}
