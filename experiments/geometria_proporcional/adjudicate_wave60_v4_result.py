#!/usr/bin/env python3
"""Derive the audited Wave 60 v4 replay-normalization correction candidate.

The sealed attempt is read-only.  This checker authenticates its historical
execution context, physical identity, replay evidence, and Git authority chain
before it can create one external candidate JSON.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Iterable, Mapping
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
EXPERIMENT_ROOT = REPO_ROOT / "experiments/geometria_proporcional"
for import_root in (SRC_ROOT, EXPERIMENT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import run_wave60_frozen_policy_transport as wave60_runner  # noqa: E402
from geometria_proporcional.wave60_frozen_policy_transport import (  # noqa: E402
    config_self_binding_sha256,
    file_sha256,
    finalize_patterns,
)


class AdjudicationError(RuntimeError):
    """The candidate cannot be derived from the declared closed world."""


ATTEMPT_RELATIVE = (
    "data/geometria_proporcional/wave60_frozen_policy_transport_attempt_v4"
)
OUTPUT_RELATIVE = (
    "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
    "WAVE_60_V4_REPLAY_NORMALIZATION_CORRECTION.json"
)
CONFIG_RELATIVE = (
    "experiments/geometria_proporcional/configs/"
    "wave60_frozen_policy_transport.json"
)
ATTEMPT = REPO_ROOT / ATTEMPT_RELATIVE
OUTPUT = REPO_ROOT / OUTPUT_RELATIVE
CONFIG = REPO_ROOT / CONFIG_RELATIVE
EXECUTION_COMMIT = "789c4ea2298fcaba97c9bdecdd1db4360186012c"

TARGET_HASHES = {
    "pair/artifact_manifest.json": (
        "4a51993c420a96f9f8283a3ced5e9dee279097ce37453fbd2b7480580805e686"
    ),
    "pair/pair_status.json": (
        "98bcfcadf9476127cff6bcdb18dcf7f8ec36dd4897721f42c441bc51874ccffd"
    ),
    "pair/final_analysis.json": (
        "9292c64e9d03a45fd59fdab39c58355abb08fbb97d7402e696cb50e0c580b21c"
    ),
    "pair/replay_comparison.json": (
        "d5ac8ff07170d30f54f1ce440ae9f60aede6d8e315cd4d08674d393c655a2f29"
    ),
    "primary/artifact_manifest.json": (
        "a96497d5a06e8ec23b7844aa13a2ef7455ef0a8bf6b410980a96ef8ebf7ed982"
    ),
    "replay/artifact_manifest.json": (
        "4ffabc2bd54de623820cd373b84ad4f95e49ff45b7ddc48da8eaa894f7dc7eb0"
    ),
}

SCIENTIFIC_HASHES = {
    "evaluation/analysis.json": (
        "f1378cb22f45e10580c18d3f0d5d12d8b6a4ed19d39cea6fd612bf0121bdb5e0"
    ),
    "evaluation/analysis_arrays.npz": (
        "94f8e5266f9dae7ffb3f8cfda44a5d3fe629308c8df5ec550233b9319c906016"
    ),
    "evaluation/bootstrap_indices.npz": (
        "5b89285e827cfc396c9283ce604681a823fdb660bcaff67409b99729b40aa510"
    ),
    "evaluation/evaluation_freeze.json": (
        "555c0b3435141ee3f20f933b14cb1fa734c7bfaf0ab441372a4dd29fa91b93ee"
    ),
    "score/evaluation_index.npz": (
        "c8c1b77bb90550bb13780c30cb233158af0ee4b67bbc615b0b93652a03f146c7"
    ),
    "score/monitor_action_freeze.json": (
        "7251350df74678eda39acbbbdae1b367a1cd307279355a501f5a53fa042ac8b0"
    ),
    "score/monitor_policy_arrays.npz": (
        "7b90c53ec401888cf1f922a27064b89406dc860a0e97aa82474e541a05e631cd"
    ),
    "score/monitor_scores.npz": (
        "7b1f4383944bc36fbfe7e2300c867d9f9335b5ea1737b3fcd5a94abfd704838f"
    ),
}

SELF_MANIFEST_METADATA = {
    "primary/artifact_manifest.json": {
        "bytes": 17792,
        "uid": 0,
        "gid": 0,
        "mode": "0444",
        "sha256": TARGET_HASHES["primary/artifact_manifest.json"],
    },
    "replay/artifact_manifest.json": {
        "bytes": 18047,
        "uid": 0,
        "gid": 0,
        "mode": "0444",
        "sha256": TARGET_HASHES["replay/artifact_manifest.json"],
    },
    "pair/artifact_manifest.json": {
        "bytes": 2448,
        "uid": 0,
        "gid": 0,
        "mode": "0444",
        "sha256": TARGET_HASHES["pair/artifact_manifest.json"],
    },
}

SOURCE_BINDINGS = {
    "src/geometria_proporcional/wave60_frozen_policy_transport.py": (
        "46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65"
    ),
    "experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py": (
        "b35cd563f715bdff9b6e7489ac04712c728673563898d4a6aebf0144d4a50261"
    ),
    "experiments/geometria_proporcional/_wave60_phase_worker.py": (
        "c6c5c832633d9ed8a99079dd4fc4f5677073f7e1883a204fa6bb41f6d646bac7"
    ),
    "experiments/geometria_proporcional/prepare_wave56_fresh.py": (
        "0dd0f3389b2db1011ce95c916a37faf4c3898460c2d30fba8f8339c5075b92c8"
    ),
    "tests/test_wave60_frozen_policy_transport.py": (
        "d8ca7d06848eb17091e743aaf025abcefa9cc333489b2c6641cd6bab9cab6960"
    ),
    (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "475_wave60_source_law_recovery_implementation_reaudit.md"
    ): "e5c49ca16506469ac5099c2a3f1992819c1a54fed790f1638a82e93b9b1d9996",
    (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "476_wave60_source_law_recovery_authority_audit.md"
    ): "497bb87f7e3677c4dae11a0281e0362fa4d1ef4d6ba60252f01cdc1f2c0a8a30",
    CONFIG_RELATIVE: (
        "eab40e2d34cfcd532437c5e7567ac94b7988a46afb865bab84728fa90735e810"
    ),
}

CONFIG_BINDING = {
    "path": CONFIG_RELATIVE,
    "commit": "b156f6857eaa36edc8bda9e7687de7b8e1ea9721",
    "physical_sha256": (
        "191483d2909c3a95a1e82488e1834b55c849763f55c72549f07f2c4cf81d6416"
    ),
    "self_binding_sha256": SOURCE_BINDINGS[CONFIG_RELATIVE],
    "audit": {
        "audit_id": "R508",
        "commit": EXECUTION_COMMIT,
        "path": (
            "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
            "508_wave60_frozen_policy_transport_v4_config_reaudit.md"
        ),
        "sha256": (
            "bf63f5d67285fabd32a7d0adae44a47dbaaa202c33930663f8cd62ed13619f6d"
        ),
    },
}

IMPLEMENTATION_AUDIT_PATH = (
    "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
    "521_wave60_r509_replay_normalization_implementation_audit.md"
)
ARTIFACT_AUDIT_PATH = (
    "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
    "523_wave60_v4_replay_normalization_correction_audit.md"
)
IMPLEMENTATION_PATHS = (
    "experiments/geometria_proporcional/adjudicate_wave60_v4_result.py",
    "tests/test_wave60_v4_result_adjudication.py",
)

HISTORICAL_DOCUMENTS = {
    "r510_base_plan": {
        "commit": "fa6ee25b359e06c9bef2ce4ec08768d8f3a46ff8",
        "parent": "92305f4e54e72ee78924ca4b51ae5889369d805b",
        "path": (
            "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
            "WAVE_60_R509_REPLAY_NORMALIZATION_RESOLUTION_PLAN.md"
        ),
        "sha256": (
            "31dc9a49e5b11df9d369537878651da788c0607e80c9d0e8fffd8093a584acee"
        ),
    },
    "r512_first_resolution_plan": {
        "commit": "bb300df7bfb47021a30072a209054fe4c5be4efb",
        "parent": "86405020425f7c2310a68d66a215e3a35a00e982",
        "path": (
            "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
            "WAVE_60_R511_REPLAY_NORMALIZATION_RESOLUTION_PLAN.md"
        ),
        "sha256": (
            "faa752fa22fead8609feef293d9c791262fe59bf6a6c51b8fcd6ff173755b57e"
        ),
    },
    "r514_final_resolution_plan": {
        "commit": "574f79810e5283dec0478c8c1c53e3496c808e97",
        "parent": "b522debe93ee55b7abba9ec04e5de67669c6231b",
        "path": (
            "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
            "WAVE_60_R513_REPLAY_NORMALIZATION_FINAL_RESOLUTION_PLAN.md"
        ),
        "sha256": (
            "9f96084160627a4c9102949ff413d98b6c28a22a20bf2b6c4e115e44cc888b78"
        ),
    },
    "r516_false_source_hash_plan": {
        "commit": "d564d1c078248cc0083ecda206db81b0c80752da",
        "parent": "8b5fb79cb992034021698e7f36da8779d50824d2",
        "path": (
            "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
            "WAVE_60_R515_SOURCE_HASH_AUTHORITY_CORRECTION_PLAN.md"
        ),
        "sha256": (
            "c2b8fec2e6426c1456403bb2ed714d91ff6fe2611530c656f0ea13b976aade6b"
        ),
    },
    "r518_false_attribution_resolution_plan": {
        "commit": "bbd5616b40923fdccc44ec0db97cd180b10925f3",
        "parent": "dfb1b86938685899194f11449de481b9e14e45ee",
        "path": (
            "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
            "WAVE_60_R517_FALSE_ATTRIBUTION_RESOLUTION_PLAN.md"
        ),
        "sha256": (
            "6771b6c1f9f791b88286dba6b2fb1ec76eb0685bc0c1f38463a358066ffd5563"
        ),
    },
}


def _audit_payload(
    audit_id: str,
    scope: str,
    target: Mapping[str, Any],
    verdict: str,
    findings: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": "wave60-audit-authority-v1",
        "audit_id": audit_id,
        "scope": scope,
        "target": dict(target),
        "technical_verdict": verdict,
        "findings": dict(findings),
        "files_modified": False,
        "gpu_used_or_queried": False,
    }


HISTORICAL_AUDITS = {
    "r509_result_audit": {
        "commit": "92305f4e54e72ee78924ca4b51ae5889369d805b",
        "parent": "789c4ea2298fcaba97c9bdecdd1db4360186012c",
        "path": (
            "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
            "509_wave60_frozen_policy_transport_v4_result_or_terminal_audit.md"
        ),
        "sha256": (
            "006a43e9257a340b162583bcf1190cf34a27085186643a1ea37d987b4fa45e28"
        ),
        "authority_json": _audit_payload(
            "R509",
            "RESULT_OR_TERMINAL",
            {
                "attempt_path": ATTEMPT_RELATIVE,
                "pair_manifest_sha256": TARGET_HASHES[
                    "pair/artifact_manifest.json"
                ],
            },
            "REVISE",
            {"high": 0, "medium": 1, "low": 1},
        ),
    },
    "r511_base_plan_audit": {
        "commit": "86405020425f7c2310a68d66a215e3a35a00e982",
        "parent": "fa6ee25b359e06c9bef2ce4ec08768d8f3a46ff8",
        "path": (
            "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
            "511_wave60_r509_replay_normalization_resolution_plan_audit.md"
        ),
        "sha256": (
            "bb80e4efd3f8dc896ac20b83611ed7c46d25e2b3a229a2c358a1c694b70b789c"
        ),
        "authority_json": _audit_payload(
            "R511",
            "R509_REPLAY_NORMALIZATION_RESOLUTION_PLAN",
            {
                "plan_commit": HISTORICAL_DOCUMENTS["r510_base_plan"]["commit"],
                "plan_sha256": HISTORICAL_DOCUMENTS["r510_base_plan"]["sha256"],
            },
            "REVISE",
            {"high": 0, "medium": 2, "low": 1},
        ),
    },
    "r513_first_resolution_plan_audit": {
        "commit": "b522debe93ee55b7abba9ec04e5de67669c6231b",
        "parent": "bb300df7bfb47021a30072a209054fe4c5be4efb",
        "path": (
            "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
            "513_wave60_r511_replay_normalization_resolution_plan_audit.md"
        ),
        "sha256": (
            "dbe7b8cda6a29fd9c50dbe6ee12682b5d9245522828f0bd84b8b7fe0d8c7a828"
        ),
        "authority_json": _audit_payload(
            "R513",
            "R511_REPLAY_NORMALIZATION_RESOLUTION_PLAN",
            {
                "plan_commit": HISTORICAL_DOCUMENTS[
                    "r512_first_resolution_plan"
                ]["commit"],
                "plan_sha256": HISTORICAL_DOCUMENTS[
                    "r512_first_resolution_plan"
                ]["sha256"],
            },
            "REVISE",
            {"high": 0, "medium": 2, "low": 1},
        ),
    },
    "r515_final_resolution_plan_audit": {
        "commit": "8b5fb79cb992034021698e7f36da8779d50824d2",
        "parent": "574f79810e5283dec0478c8c1c53e3496c808e97",
        "path": (
            "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
            "515_wave60_r513_replay_normalization_resolution_plan_audit.md"
        ),
        "sha256": (
            "e81fbdd8b5067d2a9bcb2c0aedbac792bd36beadd8813860e0da32c215a6fd3e"
        ),
        "authority_json": _audit_payload(
            "R515",
            "R513_REPLAY_NORMALIZATION_RESOLUTION_PLAN",
            {
                "plan_commit": HISTORICAL_DOCUMENTS[
                    "r514_final_resolution_plan"
                ]["commit"],
                "plan_sha256": HISTORICAL_DOCUMENTS[
                    "r514_final_resolution_plan"
                ]["sha256"],
            },
            "PASS",
            {"high": 0, "medium": 0, "low": 0},
        ),
    },
    "r517_false_source_hash_plan_audit": {
        "commit": "dfb1b86938685899194f11449de481b9e14e45ee",
        "parent": "d564d1c078248cc0083ecda206db81b0c80752da",
        "path": (
            "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
            "517_wave60_r515_source_hash_authority_correction_plan_audit.md"
        ),
        "sha256": (
            "dd67ef3f771013d9950f3138a2a5d3d61a2b238c5acdc93f680b5d21c8d3a14b"
        ),
        "authority_json": _audit_payload(
            "R517",
            "R515_SOURCE_HASH_AUTHORITY_CORRECTION_PLAN",
            {
                "plan_commit": HISTORICAL_DOCUMENTS[
                    "r516_false_source_hash_plan"
                ]["commit"],
                "plan_sha256": HISTORICAL_DOCUMENTS[
                    "r516_false_source_hash_plan"
                ]["sha256"],
            },
            "REVISE",
            {"high": 1, "medium": 0, "low": 0},
        ),
    },
    "r519_false_attribution_resolution_plan_audit": {
        "commit": "5e2b2a6fcacc55bc3abc25af603dee0250e26561",
        "parent": "bbd5616b40923fdccc44ec0db97cd180b10925f3",
        "path": (
            "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
            "519_wave60_r517_false_attribution_resolution_plan_audit.md"
        ),
        "sha256": (
            "dab1dbc868aa013d47292f83992f274a3a161f6d63e7e365168e62943345dff9"
        ),
        "authority_json": _audit_payload(
            "R519",
            "R517_FALSE_ATTRIBUTION_RESOLUTION_PLAN",
            {
                "plan_commit": HISTORICAL_DOCUMENTS[
                    "r518_false_attribution_resolution_plan"
                ]["commit"],
                "plan_sha256": HISTORICAL_DOCUMENTS[
                    "r518_false_attribution_resolution_plan"
                ]["sha256"],
            },
            "PASS",
            {"high": 0, "medium": 0, "low": 0},
        ),
    },
}

EXPECTED_LIMITATIONS = [
    "synthetic_generator_only",
    "pipeline_transport_does_not_identify_target_effect",
    "conditional_intervals_without_multiplicity_correction",
    "replay_exact_pending_pair_finalize",
]
ACTIVATED_LIMITATIONS = [
    "synthetic_generator_only",
    "pipeline_transport_does_not_identify_target_effect",
    "conditional_intervals_without_multiplicity_correction",
    "replay_exact_adjudicated_by_r509_findings_resolution_chain",
]

TOP_KEYS = {
    "schema_version",
    "artifact_status",
    "activation_condition",
    "authority_chain",
    "attempt_binding",
    "config_binding",
    "source_bindings",
    "original_observation",
    "normalized_evidence",
    "conditional_corrected_view",
    "metrics_binding",
    "limitations",
    "scientific_decision",
    "decision_authority",
    "architecture_promoted",
    "gpu_used_or_queried",
}
EXPECTED_INVENTORY = {
    "primary_files": 65,
    "replay_files": 66,
    "pair_files": 10,
    "total_files": 141,
    "manifested_files": 138,
    "self_manifest_files": 3,
    "regular_files": 141,
    "nlink_one_files": 141,
    "unique_device_inode_pairs": 141,
    "metadata_and_hashes_match": True,
}


def require_exact_keys(value: Any, keys: Iterable[str], label: str) -> dict[str, Any]:
    wanted = set(keys)
    if not isinstance(value, dict) or set(value) != wanted:
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise AdjudicationError(
            f"{label} keyset drifted: expected={sorted(wanted)}, observed={observed}"
        )
    return value


def _require_hex(value: Any, length: int, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(f"[0-9a-f]{{{length}}}", value) is None:
        raise AdjudicationError(f"{label} is not {length} lowercase hex characters")
    return value


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdjudicationError(f"invalid JSON: {path}") from exc


def canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.PIPE
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise AdjudicationError(f"git authority lookup failed: {' '.join(args)}") from exc


def git_blob_sha256(commit: str, relative: str) -> str:
    _require_hex(commit, 40, "commit")
    try:
        blob = subprocess.check_output(
            ["git", "show", f"{commit}:{relative}"],
            cwd=REPO_ROOT,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        raise AdjudicationError(f"missing Git blob {commit}:{relative}") from exc
    return hashlib.sha256(blob).hexdigest()


def _require_commit(
    commit: str,
    parent: str,
    paths: Iterable[str],
    label: str,
    *,
    statuses: Mapping[str, str] | None = None,
) -> None:
    _require_hex(commit, 40, f"{label} commit")
    _require_hex(parent, 40, f"{label} parent")
    lineage = _git("rev-list", "--parents", "-n", "1", commit).split()
    if lineage != [commit, parent]:
        raise AdjudicationError(f"{label} direct parent drifted")
    records = _git(
        "diff-tree", "--no-commit-id", "--name-status", "-r", commit
    ).splitlines()
    expected = sorted(
        f"{(statuses or {}).get(path, 'A')}\t{path}" for path in paths
    )
    if sorted(records) != expected:
        raise AdjudicationError(f"{label} exclusive pathset drifted")


def parse_single_audit_json(text: str, label: str) -> dict[str, Any]:
    blocks = re.findall(r"```json[ \t]*\n(.*?)\n```", text, flags=re.DOTALL)
    if len(blocks) != 1:
        raise AdjudicationError(f"{label} must contain exactly one JSON block")
    try:
        payload = json.loads(blocks[0])
    except json.JSONDecodeError as exc:
        raise AdjudicationError(f"{label} authority JSON is invalid") from exc
    require_exact_keys(
        payload,
        {
            "schema_version",
            "audit_id",
            "scope",
            "target",
            "technical_verdict",
            "findings",
            "files_modified",
            "gpu_used_or_queried",
        },
        f"{label} authority",
    )
    require_exact_keys(payload["findings"], {"high", "medium", "low"}, f"{label} findings")
    return payload


def _validate_document(spec: Mapping[str, Any], label: str) -> dict[str, Any]:
    require_exact_keys(spec, {"commit", "parent", "path", "sha256"}, label)
    _require_commit(spec["commit"], spec["parent"], [spec["path"]], label)
    path = REPO_ROOT / spec["path"]
    if path.is_symlink() or not path.is_file():
        raise AdjudicationError(f"{label} physical document is absent or aliased")
    if file_sha256(path) != spec["sha256"]:
        raise AdjudicationError(f"{label} physical hash drifted")
    if git_blob_sha256(spec["commit"], spec["path"]) != spec["sha256"]:
        raise AdjudicationError(f"{label} Git blob drifted")
    return {key: spec[key] for key in ("commit", "path", "sha256")}


def _validate_audit(spec: Mapping[str, Any], label: str) -> dict[str, Any]:
    require_exact_keys(
        spec, {"commit", "parent", "path", "sha256", "authority_json"}, label
    )
    document = _validate_document(
        {key: spec[key] for key in ("commit", "parent", "path", "sha256")},
        label,
    )
    parsed = parse_single_audit_json(
        (REPO_ROOT / spec["path"]).read_text(encoding="utf-8"), label
    )
    if parsed != spec["authority_json"]:
        raise AdjudicationError(f"{label} authority JSON drifted")
    return {**document, "authority_json": parsed}


def validate_static_authorities() -> dict[str, Any]:
    chain: dict[str, Any] = {}
    for label in (
        "r509_result_audit",
        "r511_base_plan_audit",
        "r513_first_resolution_plan_audit",
        "r515_final_resolution_plan_audit",
        "r517_false_source_hash_plan_audit",
        "r519_false_attribution_resolution_plan_audit",
    ):
        chain[label] = _validate_audit(HISTORICAL_AUDITS[label], label)
    for label in (
        "r510_base_plan",
        "r512_first_resolution_plan",
        "r514_final_resolution_plan",
        "r516_false_source_hash_plan",
        "r518_false_attribution_resolution_plan",
    ):
        chain[label] = _validate_document(HISTORICAL_DOCUMENTS[label], label)
    return chain


def validate_config_and_sources() -> tuple[dict[str, Any], dict[str, str]]:
    audit = {
        "commit": CONFIG_BINDING["audit"]["commit"],
        "parent": CONFIG_BINDING["commit"],
        "path": CONFIG_BINDING["audit"]["path"],
        "sha256": CONFIG_BINDING["audit"]["sha256"],
        "authority_json": _audit_payload(
            "R508",
            "CONFIG",
            {
                "config_commit": CONFIG_BINDING["commit"],
                "config_sha256": CONFIG_BINDING["physical_sha256"],
            },
            "PASS",
            {"high": 0, "medium": 0, "low": 0},
        ),
    }
    _validate_audit(audit, "R508 config audit")
    _require_commit(
        CONFIG_BINDING["commit"],
        "1bfc0de3ebed7d66a8378e4ec24c38bed52aa37a",
        [CONFIG_RELATIVE],
        "R508 config",
        statuses={CONFIG_RELATIVE: "M"},
    )
    if file_sha256(CONFIG) != CONFIG_BINDING["physical_sha256"]:
        raise AdjudicationError("config physical SHA-256 drifted")
    if (
        git_blob_sha256(CONFIG_BINDING["commit"], CONFIG_RELATIVE)
        != CONFIG_BINDING["physical_sha256"]
    ):
        raise AdjudicationError("config Git blob drifted")
    config = read_json(CONFIG)
    if config.get("source_sha256") != SOURCE_BINDINGS:
        raise AdjudicationError("config source bindings drifted")
    if config_self_binding_sha256(config, CONFIG_RELATIVE) != SOURCE_BINDINGS[
        CONFIG_RELATIVE
    ]:
        raise AdjudicationError("config self-binding drifted")
    for relative, expected in SOURCE_BINDINGS.items():
        if relative == CONFIG_RELATIVE:
            continue
        if file_sha256(REPO_ROOT / relative) != expected:
            raise AdjudicationError(f"frozen source drifted: {relative}")
    binding = deepcopy(CONFIG_BINDING)
    return binding, dict(SOURCE_BINDINGS)


def _physical_record(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise AdjudicationError(f"non-regular or aliased path: {path}")
    return {
        "bytes": metadata.st_size,
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "mode": f"{metadata.st_mode & 0o777:04o}",
        "sha256": file_sha256(path),
        "nlink": metadata.st_nlink,
        "device_inode": (metadata.st_dev, metadata.st_ino),
    }


def _manifest_record(record: Mapping[str, Any]) -> dict[str, Any]:
    require_exact_keys(record, {"bytes", "sha256", "owner", "group", "mode"}, "manifest record")
    return {
        "bytes": record["bytes"],
        "uid": record["owner"],
        "gid": record["group"],
        "mode": record["mode"],
        "sha256": record["sha256"],
    }


def require_physical_match(
    record: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    if {field: record.get(field) for field in expected} != dict(expected):
        raise AdjudicationError(f"physical metadata/hash drifted: {label}")


def require_single_link(record: Mapping[str, Any], label: str) -> None:
    if record.get("nlink") != 1:
        raise AdjudicationError(f"hardlink detected: {label}")


def validate_attempt_inventory(attempt: Path = ATTEMPT) -> dict[str, Any]:
    if attempt != ATTEMPT or attempt.is_symlink() or attempt.resolve(strict=True) != ATTEMPT:
        raise AdjudicationError("only the canonical physical v4 attempt is admissible")
    if not stat.S_ISDIR(attempt.lstat().st_mode):
        raise AdjudicationError("attempt is not a physical directory")
    children = {path.name for path in attempt.iterdir()}
    if children != {"primary", "replay", "pair"}:
        raise AdjudicationError("attempt top-level roster drifted")

    counts: dict[str, int] = {}
    all_records: dict[str, dict[str, Any]] = {}
    for role in ("primary", "replay", "pair"):
        root = attempt / role
        if root.is_symlink() or not stat.S_ISDIR(root.lstat().st_mode):
            raise AdjudicationError(f"{role} is not a physical directory")
        manifest_path = root / "artifact_manifest.json"
        manifest = read_json(manifest_path)
        require_exact_keys(
            manifest,
            {"schema_version", "terminal", "files", "classes", "self_reference"}
            | ({"run_role"} if role != "pair" else set()),
            f"{role} manifest",
        )
        if manifest["self_reference"] != {
            "path": "artifact_manifest.json",
            "hashes_omitted": True,
        }:
            raise AdjudicationError(f"{role} manifest self-reference drifted")
        expected_files = set(manifest["files"]) | {"artifact_manifest.json"}
        expected_dirs = {
            str(parent)
            for relative in expected_files
            for parent in Path(relative).parents
            if str(parent) != "."
        }
        actual_files: set[str] = set()
        actual_dirs: set[str] = set()
        for path in root.rglob("*"):
            metadata = path.lstat()
            relative = path.relative_to(root).as_posix()
            if stat.S_ISLNK(metadata.st_mode):
                raise AdjudicationError(f"symlink in attempt: {role}/{relative}")
            if stat.S_ISDIR(metadata.st_mode):
                actual_dirs.add(relative)
            elif stat.S_ISREG(metadata.st_mode):
                actual_files.add(relative)
            else:
                raise AdjudicationError(f"special node in attempt: {role}/{relative}")
        if actual_files != expected_files or actual_dirs != expected_dirs:
            raise AdjudicationError(f"{role} closed-world roster drifted")
        counts[role] = len(actual_files)
        for relative in sorted(actual_files):
            key = f"{role}/{relative}"
            record = _physical_record(root / relative)
            if relative == "artifact_manifest.json":
                expected = SELF_MANIFEST_METADATA[key]
            else:
                expected = _manifest_record(manifest["files"][relative])
            require_physical_match(record, expected, key)
            require_single_link(record, key)
            all_records[key] = record

    if counts != {"primary": 65, "replay": 66, "pair": 10}:
        raise AdjudicationError("attempt file counts drifted")
    inodes = {record["device_inode"] for record in all_records.values()}
    if len(all_records) != 141 or len(inodes) != 141:
        raise AdjudicationError("attempt inode identity drifted")
    return dict(EXPECTED_INVENTORY)


def validate_target_hashes(attempt: Path = ATTEMPT) -> dict[str, str]:
    observed = {
        relative: file_sha256(attempt / relative) for relative in TARGET_HASHES
    }
    if observed != TARGET_HASHES:
        raise AdjudicationError("R509 target hashes drifted")
    return observed


def _validate_pair_attestation(attempt: Path) -> None:
    pair = attempt / "pair"
    freeze = read_json(pair / "replay_finalize_freeze.json")
    expected_freeze = {
        "schema_version": "wave60-replay-finalize-v1",
        "phase": "replay_finalize",
        "primary_evaluation_attestation_sha256": file_sha256(
            attempt / "primary/evaluation/evaluation_attestation.json"
        ),
        "replay_evaluation_attestation_sha256": file_sha256(
            attempt / "replay/evaluation/evaluation_attestation.json"
        ),
        "primary_root_manifest_sha256": TARGET_HASHES[
            "primary/artifact_manifest.json"
        ],
        "replay_root_manifest_sha256": TARGET_HASHES[
            "replay/artifact_manifest.json"
        ],
        "replay_comparison_sha256": TARGET_HASHES["pair/replay_comparison.json"],
        "final_analysis_sha256": TARGET_HASHES["pair/final_analysis.json"],
        "pair_status_sha256": TARGET_HASHES["pair/pair_status.json"],
    }
    if freeze != expected_freeze:
        raise AdjudicationError("pair finalize freeze drifted")
    attestation = read_json(pair / "replay_finalize_attestation.json")
    wave60_runner.verify_wave60_attestation(attestation)
    expected_payload = {
        "scope": "pair",
        "freeze_sha256": file_sha256(pair / "replay_finalize_freeze.json"),
        "receipt_sha256": file_sha256(pair / "replay_finalize_receipt.json"),
        "journal_sha256": file_sha256(pair / "journals/replay_finalize.json"),
        "runtime_sha256": file_sha256(pair / "runtime.json"),
        "primary_evaluation_attestation_sha256": expected_freeze[
            "primary_evaluation_attestation_sha256"
        ],
        "replay_evaluation_attestation_sha256": expected_freeze[
            "replay_evaluation_attestation_sha256"
        ],
    }
    if (
        attestation.get("schema_version") != "wave60-replay-finalize-v1"
        or attestation.get("phase") != "replay_finalize"
        or attestation.get("payload") != expected_payload
    ):
        raise AdjudicationError("pair finalize attestation drifted")


def validate_historical_terminal(attempt: Path = ATTEMPT) -> None:
    status = read_json(attempt / "pair/pair_status.json")
    with patch.object(wave60_runner, "git_commit", return_value=EXECUTION_COMMIT):
        primary_binding = wave60_runner.validate_evaluated_root(
            attempt / "primary", "primary"
        )
        replay_binding = wave60_runner.validate_evaluated_root(
            attempt / "replay", "replay"
        )
        wave60_runner.validate_pair_status_against_roots(attempt, status)
    if (
        primary_binding != TARGET_HASHES["primary/artifact_manifest.json"]
        or replay_binding != TARGET_HASHES["replay/artifact_manifest.json"]
        or status
        != {
            "schema_version": "wave60-pair-status-v1",
            "terminal": "COMPLETE",
            "primary_terminal": "EVALUATED_IMMUTABLE",
            "replay_terminal": "EVALUATED_IMMUTABLE",
            "primary_terminal_binding_sha256": primary_binding,
            "replay_terminal_binding_sha256": replay_binding,
            "any_truth_accessed": True,
            "recovery_allowed": False,
            "created_at": status.get("created_at"),
        }
        or not isinstance(status["created_at"], str)
    ):
        raise AdjudicationError("pair terminal semantics drifted")
    _validate_pair_attestation(attempt)


def _comparison_checks(comparison: Mapping[str, Any]) -> dict[str, bool]:
    groups = (
        "exact_json_md",
        "exact_npz",
        "functional_states",
        "secret_hashes",
        "operational_semantic",
    )
    checks: dict[str, bool] = {}
    for group in groups:
        values = comparison.get(group)
        if not isinstance(values, dict):
            raise AdjudicationError(f"comparison group missing: {group}")
        for name, value in values.items():
            if not isinstance(value, bool):
                raise AdjudicationError(f"comparison check is not boolean: {group}:{name}")
            checks[f"{group}:{name}"] = value
    return checks


def validate_unique_historical_mismatch(comparison: Mapping[str, Any]) -> None:
    checks = _comparison_checks(comparison)
    false_checks = sorted(name for name, value in checks.items() if not value)
    if (
        len(checks) != 36
        or sum(checks.values()) != 35
        or false_checks != ["operational_semantic:preparation_receipt.json"]
        or comparison.get("status") != "MISMATCH"
        or comparison.get("mismatches")
        != ["operational:preparation_receipt.json"]
    ):
        raise AdjudicationError("historical mismatch is not the unique R509 common mode")


def validate_receipt_normalization(
    preparations: Mapping[str, Mapping[str, Any]],
    generations: Mapping[str, Mapping[str, Any]],
    generation_hashes: Mapping[str, str],
    freeze_hashes: Mapping[str, str],
) -> dict[str, str]:
    require_exact_keys(preparations, {"primary", "replay"}, "preparation receipts")
    require_exact_keys(generations, {"primary", "replay"}, "generation receipts")
    require_exact_keys(generation_hashes, {"primary", "replay"}, "generation hashes")
    require_exact_keys(freeze_hashes, {"primary", "replay"}, "freeze hashes")
    for role in ("primary", "replay"):
        if preparations[role].get("generation_receipt_sha256") != generation_hashes[
            role
        ]:
            raise AdjudicationError(f"{role} local generation receipt link drifted")
    if (
        len(set(freeze_hashes.values())) != 1
        or any(
            preparations[role].get("preparation_freeze_sha256")
            != freeze_hashes[role]
            for role in ("primary", "replay")
        )
        or any(preparations[role].get("next_state") != "PREPARED" for role in preparations)
    ):
        raise AdjudicationError("preparation freeze/local receipt linkage drifted")
    modes = {role: generations[role].get("execution_mode") for role in generations}
    if modes != {"primary": "recovery", "replay": "replay"}:
        raise AdjudicationError("generation receipt roles drifted")
    normalized = deepcopy(generations)
    for payload in normalized.values():
        payload.pop("execution_mode", None)
    if (
        set(generations["primary"]) != set(generations["replay"])
        or normalized["primary"] != normalized["replay"]
    ):
        raise AdjudicationError("generation receipts differ beyond execution_mode")
    return modes


def adjudicate_attempt(attempt: Path = ATTEMPT) -> dict[str, Any]:
    inventory = validate_attempt_inventory(attempt)
    targets = validate_target_hashes(attempt)
    validate_historical_terminal(attempt)

    primary = attempt / "primary"
    replay = attempt / "replay"
    pair = attempt / "pair"
    published_comparison = read_json(pair / "replay_comparison.json")
    recomputed = wave60_runner.compare_evaluated_roots(primary, replay)
    if recomputed != published_comparison:
        raise AdjudicationError("historical replay comparison does not recompose")
    validate_unique_historical_mismatch(recomputed)

    preparations = {
        role: read_json(attempt / role / "preparation_receipt.json")
        for role in ("primary", "replay")
    }
    generations = {
        role: read_json(attempt / role / "generation_receipt.json")
        for role in ("primary", "replay")
    }
    generation_hashes = {
        role: file_sha256(attempt / role / "generation_receipt.json")
        for role in ("primary", "replay")
    }
    freeze_hashes = {
        role: file_sha256(attempt / role / "preparation_freeze.json")
        for role in ("primary", "replay")
    }
    modes = validate_receipt_normalization(
        preparations, generations, generation_hashes, freeze_hashes
    )

    primary_analysis = read_json(primary / "evaluation/analysis.json")
    replay_analysis = read_json(replay / "evaluation/analysis.json")
    if primary_analysis != replay_analysis:
        raise AdjudicationError("primary/replay analyses drifted")
    analysis_hash = file_sha256(primary / "evaluation/analysis.json")
    if analysis_hash != "f1378cb22f45e10580c18d3f0d5d12d8b6a4ed19d39cea6fd612bf0121bdb5e0":
        raise AdjudicationError("analysis hash drifted")
    original_conditions, original_patterns = finalize_patterns(primary_analysis, False)
    normalized_conditions, normalized_patterns = finalize_patterns(primary_analysis, True)
    final_analysis = read_json(pair / "final_analysis.json")
    if (
        final_analysis.get("conditions") != original_conditions
        or final_analysis.get("patterns") != original_patterns
        or original_patterns != {"incompatibility": False, "harm": False}
        or normalized_patterns != {"incompatibility": False, "harm": False}
        or final_analysis.get("scientific_decision") is not None
        or final_analysis.get("decision_authority") != "user"
        or final_analysis.get("limitations") != EXPECTED_LIMITATIONS
    ):
        raise AdjudicationError("scientific finalization drifted")
    diagnostics = primary_analysis.get("diagnostics", {}).get("control_deltas", {})
    if (
        diagnostics.get("posterior_incompatibility", {}).get("ci95_high")
        != 0.0015481381506090807
        or diagnostics.get("harm", {}).get("ci95_high")
        != 0.009468438538205979
    ):
        raise AdjudicationError("core control diagnostics drifted")

    return {
        "attempt_binding": {
            "path": ATTEMPT_RELATIVE,
            "target_sha256": targets,
            "physical_inventory": inventory,
            "self_manifest_metadata": deepcopy(SELF_MANIFEST_METADATA),
        },
        "original_observation": {
            "status": "MISMATCH",
            "replay_exact": False,
            "mismatches": ["operational:preparation_receipt.json"],
            "conditions": original_conditions,
            "patterns": original_patterns,
            "limitations": list(EXPECTED_LIMITATIONS),
        },
        "normalized_evidence": {
            "historical_check_count": 36,
            "historical_true_count": 35,
            "normalized_check_count": 36,
            "normalized_all_true": True,
            "local_generation_receipt_sha256": generation_hashes,
            "preparation_freeze_sha256": freeze_hashes["primary"],
            "generation_execution_modes": modes,
            "generation_receipts_equal_except_execution_mode": True,
            "primary_scientific_hashes": recomputed["primary_scientific_hashes"],
            "replay_scientific_hashes": recomputed["replay_scientific_hashes"],
        },
        "conditional_corrected_view": {
            "normalized_replay_exact": True,
            "conditions": normalized_conditions,
            "patterns": normalized_patterns,
        },
        "metrics_binding": {
            "primary_analysis_sha256": analysis_hash,
            "replay_analysis_sha256": file_sha256(
                replay / "evaluation/analysis.json"
            ),
            "r509_numeric_recomputation": {
                "actions": 14,
                "metric_arrays": 56,
                "pair_tokens": 301,
                "bootstrap_replicates": 5000,
                "r509_report_sha256": HISTORICAL_AUDITS["r509_result_audit"][
                    "sha256"
                ],
                "values_unchanged": True,
            },
        },
    }


def validate_implementation_authority(
    implementation_commit: str,
    implementation_audit_commit: str,
    implementation_audit_sha256: str,
) -> dict[str, Any]:
    _require_hex(implementation_commit, 40, "implementation commit")
    _require_hex(implementation_audit_commit, 40, "implementation audit commit")
    _require_hex(implementation_audit_sha256, 64, "implementation audit SHA-256")
    _require_commit(
        implementation_commit,
        HISTORICAL_AUDITS["r519_false_attribution_resolution_plan_audit"][
            "commit"
        ],
        IMPLEMENTATION_PATHS,
        "R520 implementation",
    )
    files: dict[str, str] = {}
    for relative in IMPLEMENTATION_PATHS:
        physical = file_sha256(REPO_ROOT / relative)
        if git_blob_sha256(implementation_commit, relative) != physical:
            raise AdjudicationError(f"implementation physical/blob drift: {relative}")
        files[relative] = physical
    audit_spec = {
        "commit": implementation_audit_commit,
        "parent": implementation_commit,
        "path": IMPLEMENTATION_AUDIT_PATH,
        "sha256": implementation_audit_sha256,
        "authority_json": _audit_payload(
            "R521",
            "R509_REPLAY_NORMALIZATION_RESOLUTION_IMPLEMENTATION",
            {"implementation_commit": implementation_commit, "files": files},
            "PASS",
            {"high": 0, "medium": 0, "low": 0},
        ),
    }
    audit = _validate_audit(audit_spec, "R521 implementation audit")
    return {
        "r520_implementation": {
            "commit": implementation_commit,
            "files": files,
        },
        "r521_implementation_audit": audit,
    }


def build_correction_payload(
    implementation_commit: str,
    implementation_audit_commit: str,
    implementation_audit_sha256: str,
) -> dict[str, Any]:
    static_chain = validate_static_authorities()
    config_binding, source_bindings = validate_config_and_sources()
    implementation_chain = validate_implementation_authority(
        implementation_commit,
        implementation_audit_commit,
        implementation_audit_sha256,
    )
    evidence = adjudicate_attempt()
    authority_chain = {
        "r509_result_audit": static_chain["r509_result_audit"],
        "r510_base_plan": static_chain["r510_base_plan"],
        "r511_base_plan_audit": static_chain["r511_base_plan_audit"],
        "r512_first_resolution_plan": static_chain["r512_first_resolution_plan"],
        "r513_first_resolution_plan_audit": static_chain[
            "r513_first_resolution_plan_audit"
        ],
        "r514_final_resolution_plan": static_chain["r514_final_resolution_plan"],
        "r515_final_resolution_plan_audit": static_chain[
            "r515_final_resolution_plan_audit"
        ],
        "r516_false_source_hash_plan": static_chain[
            "r516_false_source_hash_plan"
        ],
        "r517_false_source_hash_plan_audit": static_chain[
            "r517_false_source_hash_plan_audit"
        ],
        "r518_false_attribution_resolution_plan": static_chain[
            "r518_false_attribution_resolution_plan"
        ],
        "r519_false_attribution_resolution_plan_audit": static_chain[
            "r519_false_attribution_resolution_plan_audit"
        ],
        **implementation_chain,
    }
    payload = {
        "schema_version": "wave60-v4-replay-normalization-correction-v1",
        "artifact_status": "CANDIDATE_PENDING_R523_AUDIT",
        "activation_condition": {
            "required_audit_id": "R523",
            "required_audit_path": ARTIFACT_AUDIT_PATH,
            "required_scope": "WAVE60_V4_REPLAY_NORMALIZATION_CORRECTION",
            "required_verdict": "PASS",
            "required_findings": {"high": 0, "medium": 0, "low": 0},
            "authority_effect": "ACTIVATES_CONDITIONAL_CORRECTED_VIEW",
        },
        "authority_chain": authority_chain,
        "attempt_binding": evidence["attempt_binding"],
        "config_binding": config_binding,
        "source_bindings": source_bindings,
        "original_observation": evidence["original_observation"],
        "normalized_evidence": evidence["normalized_evidence"],
        "conditional_corrected_view": evidence["conditional_corrected_view"],
        "metrics_binding": evidence["metrics_binding"],
        "limitations": {
            "original": list(EXPECTED_LIMITATIONS),
            "if_activated": list(ACTIVATED_LIMITATIONS),
            "replaced": {
                "from": "replay_exact_pending_pair_finalize",
                "to": "replay_exact_adjudicated_by_r509_findings_resolution_chain",
            },
        },
        "scientific_decision": None,
        "decision_authority": "user",
        "architecture_promoted": False,
        "gpu_used_or_queried": False,
    }
    validate_correction_payload(payload)
    return payload


def validate_correction_payload(payload: Any) -> dict[str, Any]:
    require_exact_keys(payload, TOP_KEYS, "correction")
    if (
        payload["schema_version"]
        != "wave60-v4-replay-normalization-correction-v1"
        or payload["artifact_status"] != "CANDIDATE_PENDING_R523_AUDIT"
        or payload["scientific_decision"] is not None
        or payload["decision_authority"] != "user"
        or payload["architecture_promoted"] is not False
        or payload["gpu_used_or_queried"] is not False
    ):
        raise AdjudicationError("correction top-level semantics drifted")
    activation = require_exact_keys(
        payload["activation_condition"],
        {
            "required_audit_id",
            "required_audit_path",
            "required_scope",
            "required_verdict",
            "required_findings",
            "authority_effect",
        },
        "activation condition",
    )
    expected_activation = {
        "required_audit_id": "R523",
        "required_audit_path": ARTIFACT_AUDIT_PATH,
        "required_scope": "WAVE60_V4_REPLAY_NORMALIZATION_CORRECTION",
        "required_verdict": "PASS",
        "required_findings": {"high": 0, "medium": 0, "low": 0},
        "authority_effect": "ACTIVATES_CONDITIONAL_CORRECTED_VIEW",
    }
    if activation != expected_activation:
        raise AdjudicationError("activation condition drifted")

    chain = require_exact_keys(
        payload["authority_chain"],
        {
            "r509_result_audit",
            "r510_base_plan",
            "r511_base_plan_audit",
            "r512_first_resolution_plan",
            "r513_first_resolution_plan_audit",
            "r514_final_resolution_plan",
            "r515_final_resolution_plan_audit",
            "r516_false_source_hash_plan",
            "r517_false_source_hash_plan_audit",
            "r518_false_attribution_resolution_plan",
            "r519_false_attribution_resolution_plan_audit",
            "r520_implementation",
            "r521_implementation_audit",
        },
        "authority chain",
    )
    for label in (
        "r510_base_plan",
        "r512_first_resolution_plan",
        "r514_final_resolution_plan",
        "r516_false_source_hash_plan",
        "r518_false_attribution_resolution_plan",
    ):
        require_exact_keys(chain[label], {"commit", "path", "sha256"}, label)
    for label in (
        "r509_result_audit",
        "r511_base_plan_audit",
        "r513_first_resolution_plan_audit",
        "r515_final_resolution_plan_audit",
        "r517_false_source_hash_plan_audit",
        "r519_false_attribution_resolution_plan_audit",
        "r521_implementation_audit",
    ):
        require_exact_keys(
            chain[label], {"commit", "path", "sha256", "authority_json"}, label
        )
        parse_target = chain[label]["authority_json"]
        require_exact_keys(
            parse_target,
            {
                "schema_version",
                "audit_id",
                "scope",
                "target",
                "technical_verdict",
                "findings",
                "files_modified",
                "gpu_used_or_queried",
            },
            f"{label} authority JSON",
        )
    implementation = require_exact_keys(
        chain["r520_implementation"], {"commit", "files"}, "R520 implementation"
    )
    require_exact_keys(implementation["files"], IMPLEMENTATION_PATHS, "implementation files")
    for label, spec in HISTORICAL_DOCUMENTS.items():
        expected = {key: spec[key] for key in ("commit", "path", "sha256")}
        if chain[label] != expected:
            raise AdjudicationError(f"historical document binding drifted: {label}")
    for label, spec in HISTORICAL_AUDITS.items():
        expected = {
            key: spec[key]
            for key in ("commit", "path", "sha256", "authority_json")
        }
        if chain[label] != expected:
            raise AdjudicationError(f"historical audit binding drifted: {label}")
    implementation_commit = _require_hex(
        implementation["commit"], 40, "R520 implementation commit"
    )
    for relative, digest in implementation["files"].items():
        _require_hex(digest, 64, f"R520 file digest {relative}")
    r521 = chain["r521_implementation_audit"]
    _require_hex(r521["commit"], 40, "R521 commit")
    _require_hex(r521["sha256"], 64, "R521 SHA-256")
    expected_r521_json = _audit_payload(
        "R521",
        "R509_REPLAY_NORMALIZATION_RESOLUTION_IMPLEMENTATION",
        {
            "implementation_commit": implementation_commit,
            "files": implementation["files"],
        },
        "PASS",
        {"high": 0, "medium": 0, "low": 0},
    )
    if (
        r521["path"] != IMPLEMENTATION_AUDIT_PATH
        or r521["authority_json"] != expected_r521_json
    ):
        raise AdjudicationError("R521 binding drifted")

    attempt = require_exact_keys(
        payload["attempt_binding"],
        {"path", "target_sha256", "physical_inventory", "self_manifest_metadata"},
        "attempt binding",
    )
    require_exact_keys(attempt["target_sha256"], TARGET_HASHES, "target hashes")
    require_exact_keys(
        attempt["physical_inventory"],
        {
            "primary_files",
            "replay_files",
            "pair_files",
            "total_files",
            "manifested_files",
            "self_manifest_files",
            "regular_files",
            "nlink_one_files",
            "unique_device_inode_pairs",
            "metadata_and_hashes_match",
        },
        "physical inventory",
    )
    require_exact_keys(
        attempt["self_manifest_metadata"], SELF_MANIFEST_METADATA, "self manifests"
    )
    for relative, record in attempt["self_manifest_metadata"].items():
        require_exact_keys(record, {"bytes", "uid", "gid", "mode", "sha256"}, relative)
    if (
        attempt["path"] != ATTEMPT_RELATIVE
        or attempt["target_sha256"] != TARGET_HASHES
        or attempt["physical_inventory"] != EXPECTED_INVENTORY
        or attempt["self_manifest_metadata"] != SELF_MANIFEST_METADATA
    ):
        raise AdjudicationError("attempt binding values drifted")

    config = require_exact_keys(
        payload["config_binding"],
        {"path", "commit", "physical_sha256", "self_binding_sha256", "audit"},
        "config binding",
    )
    require_exact_keys(config["audit"], {"audit_id", "commit", "path", "sha256"}, "config audit")
    if config != CONFIG_BINDING:
        raise AdjudicationError("config binding values drifted")
    require_exact_keys(payload["source_bindings"], SOURCE_BINDINGS, "source bindings")
    if payload["source_bindings"] != SOURCE_BINDINGS:
        raise AdjudicationError("source binding values drifted")

    original = require_exact_keys(
        payload["original_observation"],
        {"status", "replay_exact", "mismatches", "conditions", "patterns", "limitations"},
        "original observation",
    )
    normalized = require_exact_keys(
        payload["normalized_evidence"],
        {
            "historical_check_count",
            "historical_true_count",
            "normalized_check_count",
            "normalized_all_true",
            "local_generation_receipt_sha256",
            "preparation_freeze_sha256",
            "generation_execution_modes",
            "generation_receipts_equal_except_execution_mode",
            "primary_scientific_hashes",
            "replay_scientific_hashes",
        },
        "normalized evidence",
    )
    require_exact_keys(normalized["local_generation_receipt_sha256"], {"primary", "replay"}, "local receipts")
    require_exact_keys(normalized["generation_execution_modes"], {"primary", "replay"}, "generation modes")
    require_exact_keys(normalized["primary_scientific_hashes"], SCIENTIFIC_HASHES, "primary scientific hashes")
    require_exact_keys(normalized["replay_scientific_hashes"], SCIENTIFIC_HASHES, "replay scientific hashes")
    corrected = require_exact_keys(
        payload["conditional_corrected_view"],
        {"normalized_replay_exact", "conditions", "patterns"},
        "conditional corrected view",
    )
    metrics = require_exact_keys(
        payload["metrics_binding"],
        {"primary_analysis_sha256", "replay_analysis_sha256", "r509_numeric_recomputation"},
        "metrics binding",
    )
    require_exact_keys(
        metrics["r509_numeric_recomputation"],
        {"actions", "metric_arrays", "pair_tokens", "bootstrap_replicates", "r509_report_sha256", "values_unchanged"},
        "R509 numeric recomputation",
    )
    limitations = require_exact_keys(
        payload["limitations"], {"original", "if_activated", "replaced"}, "limitations"
    )
    require_exact_keys(limitations["replaced"], {"from", "to"}, "replaced limitation")
    if (
        original["status"] != "MISMATCH"
        or original["replay_exact"] is not False
        or original["mismatches"] != ["operational:preparation_receipt.json"]
        or original["patterns"] != {"incompatibility": False, "harm": False}
        or corrected["normalized_replay_exact"] is not True
        or corrected["patterns"] != {"incompatibility": False, "harm": False}
        or {
            name: {**conditions, "replay_exact": True}
            for name, conditions in original["conditions"].items()
        }
        != corrected["conditions"]
        or normalized["historical_check_count"] != 36
        or normalized["historical_true_count"] != 35
        or normalized["normalized_check_count"] != 36
        or normalized["normalized_all_true"] is not True
        or normalized["preparation_freeze_sha256"]
        != "ed47e512452ee13689427bca97166c323233c2a8825d89d18cb9c91b73efe6f9"
        or normalized["generation_execution_modes"]
        != {"primary": "recovery", "replay": "replay"}
        or normalized["generation_receipts_equal_except_execution_mode"] is not True
        or normalized["primary_scientific_hashes"] != SCIENTIFIC_HASHES
        or normalized["replay_scientific_hashes"] != SCIENTIFIC_HASHES
        or any(
            re.fullmatch(r"[0-9a-f]{64}", value or "") is None
            for value in normalized["local_generation_receipt_sha256"].values()
        )
        or normalized["local_generation_receipt_sha256"]
        != {
            "primary": "847ec250397a459eeb25e4ca6bb10774551e40dbadceaeeac8cb27527c60be44",
            "replay": "9088275ca41be5f82d14bb7c227eed36b443e080062786bc77dd8e38483e3e13",
        }
        or metrics["primary_analysis_sha256"] != SCIENTIFIC_HASHES["evaluation/analysis.json"]
        or metrics["replay_analysis_sha256"] != SCIENTIFIC_HASHES["evaluation/analysis.json"]
        or metrics["r509_numeric_recomputation"]
        != {
            "actions": 14,
            "metric_arrays": 56,
            "pair_tokens": 301,
            "bootstrap_replicates": 5000,
            "r509_report_sha256": HISTORICAL_AUDITS["r509_result_audit"]["sha256"],
            "values_unchanged": True,
        }
        or limitations["original"] != EXPECTED_LIMITATIONS
        or limitations["if_activated"] != ACTIVATED_LIMITATIONS
        or limitations["replaced"]
        != {
            "from": "replay_exact_pending_pair_finalize",
            "to": "replay_exact_adjudicated_by_r509_findings_resolution_chain",
        }
    ):
        raise AdjudicationError("correction scientific semantics drifted")
    return payload


def publish_correction(payload: Mapping[str, Any], output: Path = OUTPUT) -> Path:
    validate_correction_payload(payload)
    if output != OUTPUT or OUTPUT.is_relative_to(ATTEMPT):
        raise AdjudicationError("correction output is not the canonical external path")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.fchmod(descriptor, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if read_json(output) != payload:
            raise AdjudicationError("published correction failed self-validation")
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check-attempt", "build", "publish", "validate"))
    parser.add_argument("--implementation-commit")
    parser.add_argument("--implementation-audit-commit")
    parser.add_argument("--implementation-audit-sha256")
    parser.add_argument("--artifact", type=Path, default=OUTPUT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.mode == "check-attempt":
        print(json.dumps(adjudicate_attempt(), sort_keys=True))
        return 0
    if args.mode == "validate":
        validate_correction_payload(read_json(args.artifact))
        return 0
    required = (
        args.implementation_commit,
        args.implementation_audit_commit,
        args.implementation_audit_sha256,
    )
    if any(value is None for value in required):
        raise SystemExit("build/publish require implementation and R521 bindings")
    payload = build_correction_payload(*required)
    if args.mode == "build":
        sys.stdout.buffer.write(canonical_bytes(payload))
    else:
        publish_correction(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
