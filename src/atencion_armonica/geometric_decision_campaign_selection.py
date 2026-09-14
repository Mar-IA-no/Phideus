"""Read all completed campaign calibrations and select one epoch per arm.

The external operator must authenticate a COMPLETE training finish first and
admit this OPEN-only work. There is no training, forward, fitting or test port.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .geometric_decision_admission import ProfileReadOnlyStore
from .geometric_decision_campaign import ROSTER, cell_identity, validate_complete
from .geometric_decision_cell import EPOCHS, read_calibration
from .geometric_decision_store import ArtifactStore
from .geometric_decision_selection import select_epochs
from .geometric_decision_profile import assert_same_arrays
from .generative_evidence import CHECKPOINTS
from .partial_compatibility_cache import encoded


class ReadOnlyCellStore(ProfileReadOnlyStore):
    _state_record = ArtifactStore._state_record
    load_state = ArtifactStore.load_state  # Authenticated trusted-local pickle, always map_location CPU.


def select_campaign(preparation, open_ref, campaign, campaign_ref, output, *, check):
    complete = campaign.json(campaign_ref)
    if (complete["schema"] != "geometric-decision-campaign-complete-v1"
            or complete["binding"] != campaign.binding
            or campaign.binding["preparation"] != preparation.store.binding
            or campaign.binding["open_complete"] != open_ref
            or [(c["checkpoint_seed"], c["arm"], c["reader_seed"]) for c in complete["cells"]] != list(ROSTER)
            or output.binding.get("campaign") != campaign_ref
            or output.binding.get("campaign_binding") != campaign.binding):
        raise ValueError("selection requires the complete fixed campaign and source binding")
    energies, provenance, targets, common_identity = {}, [], None, None
    for cp in CHECKPOINTS:
        check()
        data = preparation.load_checkpoint(open_ref, cp, check=check)
        rows = data.rows["calibration"]
        current_targets = [row["targets"] for row in rows]
        identity = [(row["scene_id"], row["identity"], row["partitions"]) for row in rows]
        if targets is None:
            targets, common_identity = current_targets, identity
        else:
            if identity != common_identity:
                raise ValueError("calibration identities/candidate order differ across backbones")
            for a, b in zip(targets, current_targets):
                assert_same_arrays({"targets": a}, {"targets": b})
        for entry in (c for c in complete["cells"] if c["checkpoint_seed"] == cp):
            check()
            arm, seed = entry["arm"], entry["reader_seed"]
            relative = f"cells/cp_{cp}/{arm}/seed_{seed}"
            if entry["root"] != relative:
                raise ValueError("cell root differs from canonical roster")
            expected = cell_identity(data, cp, arm, seed, campaign.binding["device"])
            raw = encoded(expected)
            cell = ReadOnlyCellStore(campaign.root/relative, binding_ref={"path": "binding.json",
                "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
            record = validate_complete(data, cell, entry["complete"])
            for epoch, ref in zip(EPOCHS, record["calibration"]):
                check()
                calibration = cell.json(ref)
                state = cell.json(calibration["state"])
                cell.read(state["state"])
                if (state["steps"] != epoch*((len(data.eligible["train"])+31)//32)
                        or (epoch == 50 and calibration["state"] != record["last_state"])):
                    raise ValueError("calibration snapshot differs from completed campaign state")
                arrays = read_calibration(data, cell, ref, calibration["state"], epoch)
                if epoch:
                    energies[arm, cp, seed, epoch] = arrays["energy"]
                    provenance.append({"checkpoint_seed": cp, "arm": arm, "reader_seed": seed,
                                       "epoch": epoch, "calibration": ref, "state": calibration["state"]})
        del data
    check()
    result = select_epochs(targets, energies)
    offsets = np.r_[np.int64(0), np.cumsum([len(t) for t in targets], dtype=np.int64)]
    target_arrays = {"targets": np.concatenate(targets), "offsets": offsets}
    target_path = output.path("calibration-targets.npz")
    if target_path.exists():
        target_ref = output.reference(target_path)
        assert_same_arrays(output.arrays(target_ref), target_arrays)
    else:
        target_ref = output.publish_arrays("calibration-targets.npz", target_arrays)
    check()
    return output.publish_json("selection.json", {"binding": output.binding, "campaign": campaign_ref,
        "result": result, "targets": target_ref, "calibrations": provenance,
        "calibration_identities": [r[1] for r in common_identity], "fresh_tests": "not opened"})
