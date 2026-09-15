"""Admit completed calibration selection and validate the frozen-head roster."""
from __future__ import annotations

import hashlib

from .geometric_decision_campaign import ROSTER
from .geometric_decision_head_archive import read_head_arrays, selected_epochs
from .geometric_decision_open import ReadOnlyStore
from .partial_compatibility_cache import encoded


def admitted_selection(control, *, root, campaign, campaign_ref, training):
    candidates = []
    for path in sorted(control.path("attempts").glob("*/finish.json")):
        ref = control.reference(path)
        finish = control.json(ref)
        start = control.json(finish["start"])
        manifest = control.json(start["manifest"])
        if manifest.get("operation") != "calibration-selection" or finish["status"] != "COMPLETE":
            continue
        output = control.json(finish["completion"])
        if (start["stage"] != "evaluation" or start["binding"] != control.binding
                or manifest["root"] != str(root) or output["root"] != str(root)
                or output["manifest"] != start["manifest"]):
            raise ValueError("selection completion provenance differs")
        raw = encoded(manifest["binding"])
        view = ReadOnlyStore(root, binding_ref={"path": "binding.json", "bytes": len(raw),
                                               "sha256": hashlib.sha256(raw).hexdigest()})
        selected = view.json(output["selection"])
        if (view.binding["schema"] != "geometric-decision-selection-binding-v1"
                or view.binding["campaign"] != campaign_ref or view.binding["campaign_binding"] != campaign.binding
                or view.binding["training"] != training or selected["binding"] != view.binding
                or selected["campaign"] != campaign_ref or selected["fresh_tests"] != "not opened"):
            raise ValueError("selection belongs to another admitted training campaign")
        selected_epochs(selected)
        expected = {(cp, arm, seed, epoch) for cp, arm, seed in ROSTER for epoch in range(5, 51, 5)}
        if len(selected["calibrations"]) != 720 or {(r["checkpoint_seed"], r["arm"], r["reader_seed"], r["epoch"])
                for r in selected["calibrations"]} != expected or len(selected["calibration_identities"]) != 512:
            raise ValueError("selection provenance roster is incomplete")
        # The target receipt is already bound by selection.json. Exporting
        # states does not require opening calibration target payloads again.
        candidates.append((view, output["selection"], {"finish": ref, "output": output}))
    if not candidates:
        raise ValueError("no COMPLETE selection operator; archive must wait")
    if any(row[1] != candidates[0][1] or row[0].binding != candidates[0][0].binding for row in candidates):
        raise ValueError("selection attempts authorize different epochs or provenance")
    return candidates[0]


def verify_heads(store, archive_ref, selection_store, selection_ref, *, check):
    archive = store.json(archive_ref)
    selection = selection_store.json(selection_ref)
    epochs = selected_epochs(selection)
    if (set(archive) != {"schema", "binding", "selection", "selected_epochs", "count", "records", "new_initializations", "forward"}
            or archive["schema"] != "geometric-decision-head-archive-v1" or archive["binding"] != store.binding
            or store.binding["selection"] != selection_ref or store.binding["selection_binding"] != selection_store.binding
            or archive["selection"] != selection_ref or archive["selected_epochs"] != epochs
            or archive["count"] != 144 or len(archive["records"]) != 144
            or archive["new_initializations"] is not False or archive["forward"] is not False):
        raise ValueError("head archive is not bound to the complete calibrated selection")
    provenance = {(r["checkpoint_seed"], r["arm"], r["reader_seed"], r["epoch"]): r for r in selection["calibrations"]}
    members = []
    for ref, (cp, arm, seed, stage) in zip(archive["records"],
            [(cp, arm, seed, stage) for cp, arm, seed in ROSTER for stage in ("initial", "selected")]):
        check()
        record, _ = read_head_arrays(store, ref)
        epoch = 0 if stage == "initial" else epochs[arm]
        if (ref["path"] != f"heads/cp_{cp}/{arm}/seed_{seed}/{stage}.json"
                or (record["checkpoint_seed"], record["arm"], record["reader_seed"], record["stage"], record["epoch"])
                != (cp, arm, seed, stage, epoch) or record["source"]["campaign"] != selection["campaign"]
                or record["source"]["root"] != f"cells/cp_{cp}/{arm}/seed_{seed}"):
            raise ValueError("head archive member identity/order differs")
        if stage == "selected":
            source = provenance[cp, arm, seed, epoch]
            if record["source"]["calibration"] != source["calibration"] or record["source"]["state"] != source["state"]:
                raise ValueError("archived head is not the calibration-selected state")
        members.append(record)
    return members
