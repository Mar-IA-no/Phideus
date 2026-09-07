#!/usr/bin/env python3
"""Build a mapping candidate from the prepared public surface only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


FIXED = {
    "gpu_used_or_queried": False,
    "architecture_promoted": False,
    "scientific_decision": None,
    "decision_authority": "user",
}
PUBLIC_GRAPH_NAMES = {
    "n_nodes.npy", "edge_index.npy", "observed_log_ratio.npy", "edge_valid.npy", "path_index.npy",
    "path_sign.npy", "path_valid.npy", "edge_variance.npy", "edge_offsets.npy", "node_offsets.npy",
    "path_offsets.npy", "unit_key.npy", "corrected_log_ratio.npy", "reliability.npy",
}
GRAPH_DIRECTORIES = {
    "raw_generic__seed=104729", "raw_generic__seed=130363",
    "raw_typed__seed=104729", "raw_typed__seed=130363",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path, opened: list[str], public: Path) -> Any:
    opened.append(path.relative_to(public).as_posix())
    return json.loads(path.read_text(encoding="utf-8"))


def load_array(path: Path, opened: list[str], public: Path) -> np.ndarray:
    opened.append(path.relative_to(public).as_posix())
    return np.load(path, allow_pickle=False)


def verify_manifest(public: Path, opened: list[str]) -> dict[str, Any]:
    manifest = read_json(public / "manifest.json", opened, public)
    if manifest.get("schema_version") != "mapping-prepared-public-manifest-v1":
        raise ValueError("public manifest schema mismatch")
    actual = sorted(
        path.relative_to(public).as_posix()
        for path in public.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    expected = {"protocol.json", "w49_contract.json", "graph_states.json"}
    expected |= {f"w54/{name}.npy" for name in ("ensemble_logits", "per_seed_logits", "unit_key", "cluster_key", "split_role")}
    expected |= {f"graph/{directory}/{name}" for directory in GRAPH_DIRECTORIES for name in PUBLIC_GRAPH_NAMES}
    if actual != sorted(expected) or actual != manifest.get("pathset") or actual != sorted(manifest.get("files", {})):
        raise RuntimeError("public manifest pathset mismatch")
    expected_pathset_hash = hashlib.sha256("\n".join(actual).encode()).hexdigest()
    if manifest.get("pathset_sha256") != expected_pathset_hash:
        raise RuntimeError("public manifest pathset digest mismatch")
    for relative, receipt in manifest["files"].items():
        path = public / relative
        if not path.is_file() or sha256_file(path) != receipt["sha256"]:
            raise RuntimeError(f"public manifest mismatch: {relative}")
    return manifest


def array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def build(public: Path, output: Path) -> None:
    public = public.resolve(strict=True)
    output = output.resolve()
    opened: list[str] = []
    manifest = verify_manifest(public, opened)
    protocol = read_json(public / "protocol.json", opened, public)
    if protocol.get("schema_version") != "proportional-mapping-prepared-protocol-v1":
        raise ValueError("prepared protocol schema mismatch")
    w49 = read_json(public / "w49_contract.json", opened, public)
    graph_states = read_json(public / "graph_states.json", opened, public)["states"]

    set_shapes = {}
    for name in ("ensemble_logits", "per_seed_logits", "unit_key", "cluster_key", "split_role"):
        values = load_array(public / "w54" / f"{name}.npy", opened, public)
        set_shapes[name] = {"dtype": values.dtype.str, "shape": list(values.shape), "sha256": array_digest(values)}

    graph_public: dict[str, Any] = {}
    parity: dict[str, Any] = {}
    parity_fields = (
        "n_nodes", "edge_index", "observed_log_ratio", "edge_valid", "path_index",
        "path_sign", "path_valid", "edge_variance", "edge_offsets", "node_offsets",
        "path_offsets", "unit_key",
    )
    for row in graph_states:
        state = row["state"]
        directory = public / "graph" / row["directory"]
        fields = {}
        for name in (*parity_fields, "corrected_log_ratio", "reliability"):
            values = load_array(directory / f"{name}.npy", opened, public)
            fields[name] = {"dtype": values.dtype.str, "shape": list(values.shape), "sha256": array_digest(values)}
        graph_public[state] = {"directory": row["directory"], "fields": fields}
    for seed in (104729, 130363):
        generic = graph_public[f"raw_generic|seed={seed}"]["fields"]
        typed = graph_public[f"raw_typed|seed={seed}"]["fields"]
        parity[str(seed)] = {
            name: generic[name]["sha256"] == typed[name]["sha256"] for name in parity_fields
        }

    candidate = {
        "schema_version": "proportional-mapping-candidate-v1",
        "query": protocol["query"],
        "builder_input": "prepared/public",
        "builder_may_emit_mapping_decision": False,
        "unit_namespaces": protocol["unit_namespaces"],
        "declared_cross_domain_unit_bridge": None,
        "lines": {
            "eiv": {
                "observation": w49["observation_fields"],
                "output": "family_scores + structural_set + conformal_cutoff",
                "target_authority": "evaluator-only W50 oracle",
            },
            "set_valued": {
                "observation": "four ensemble logits; per-seed logits only for uncertainty features",
                "output": "mass over 15 nonempty family sets",
                "target_authority": "evaluator-only boolean family set",
            },
            "relational": {
                "observation": "typed graph arrays and observed edge log-ratios",
                "output": "corrected relation + reliability, then WLS/IRLS",
                "target_authority": "evaluator-only clean relation and quotient modulo gauge",
            },
        },
        "adapters": {
            "eiv": ["visible_fixture_to_eiv_scores", "conformal_structural_reader"],
            "set_valued": ["logits_to_marginal_or_joint_mass", "mass_to_hard_or_contextual_action"],
            "relational": ["graph_to_generic_or_typed_relation", "relation_to_wls_or_irls_quotient"],
        },
        "public_facts": {
            "w49": w49,
            "w54_shapes": set_shapes,
            "graph": graph_public,
            "graph_representation_input_parity": parity,
            "public_manifest_sha256": sha256_file(public / "manifest.json"),
            "protocol_sha256": sha256_file(public / "protocol.json"),
            "public_file_count": len(manifest["files"]),
        },
        **FIXED,
    }
    native = {
        "schema_version": "proportional-native-contracts-v1",
        "relational": {
            "factorial": ["GENERIC", "TYPED", "WLS", "IRLS"],
            "unit": "master_id with paired views",
            "targets": ["clean_log_ratio", "x_true modulo mean-zero gauge"],
            "required_states": sorted(graph_public),
        },
        "set_valued": {
            "factorial": ["MARGINAL", "JOINT", "HARD_MAP_SET", "CONTEXTUAL_PROPOSER_GUARD"],
            "unit": "pair_token",
            "target": "nonempty boolean set over four families",
            "posterior_shape": [384, 15],
            "action_shape": [384, 24],
        },
        **FIXED,
    }
    write_json(output / "mapping_candidate.json", candidate)
    write_json(output / "native_contracts.json", native)
    write_json(
        output / "builder_access_receipt.json",
        {
            "schema_version": "proportional-mapping-builder-access-v1",
            "root_kind": "prepared_public_only",
            "opened": sorted(set(opened)),
            **FIXED,
        },
    )


def build_test_fixture(public: Path, output: Path) -> None:
    """Build only the TEST_ONLY candidate; never accepts a scientific root."""
    public = public.resolve(strict=True)
    output = output.resolve()
    if "tmp" not in public.parts or "tmp" not in output.parts:
        raise ValueError("TEST_ONLY builder roots must be temporary")
    if public.name != "public" or (public.parent / "test_receipt.json").exists() is False:
        raise ValueError("TEST_ONLY prepared receipt missing")
    receipt = json.loads((public.parent / "test_receipt.json").read_text())
    if receipt.get("fixture_mode") != "TEST_ONLY":
        raise ValueError("not a TEST_ONLY fixture")
    payload = json.loads((public / "payload.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "test_candidate.json",
        {
            "schema_version": "mapping-test-only-candidate-v1",
            "fixture_mode": "TEST_ONLY",
            "public_payload": payload,
            "test_decision": "BUILT",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.public, args.output)


if __name__ == "__main__":
    main()
