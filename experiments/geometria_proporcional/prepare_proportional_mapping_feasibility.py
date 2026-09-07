#!/usr/bin/env python3
"""Prepare public/private CPU surfaces for proportional mapping feasibility."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments/geometria_proporcional/configs/proportional_mapping_feasibility_v1.json"
PUBLIC_GRAPH_FIELDS = (
    "n_nodes",
    "edge_index",
    "observed_log_ratio",
    "edge_valid",
    "path_index",
    "path_sign",
    "path_valid",
    "edge_variance",
)
GRAPH_CONTAINER_FIELDS = ("edge_offsets", "node_offsets", "path_offsets")
GRAPH_PRIVATE_FIELDS = (
    "x_true",
    "clean_log_ratio",
    "causal_corruption_mask",
    "master_id",
    "view_id",
    "split",
    "mechanism",
    "x_hat_wls",
    "x_hat_irls",
    "relation_rmse",
    "wls_quotient_rmse",
    "irls_quotient_rmse",
    "irls_converged",
    "irls_iterations",
)
FIXED = {
    "gpu_used_or_queried": False,
    "architecture_promoted": False,
    "scientific_decision": None,
    "decision_authority": "user",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unit_key(namespace: str, value: str) -> str:
    return hashlib.sha256(namespace.encode("utf-8") + b"\0" + value.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    path.write_text(text + "\n", encoding="utf-8")


def save_array(path: Path, values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.lib.format.write_array(handle, np.asarray(values), allow_pickle=False)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_manifest(root: Path, schema: str) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"):
        relative = path.relative_to(root).as_posix()
        row: dict[str, Any] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if path.suffix == ".npy":
            array = np.load(path, allow_pickle=False, mmap_mode="r")
            row.update({"dtype": array.dtype.str, "shape": list(array.shape)})
        files[relative] = row
    return {
        "schema_version": schema,
        "pathset": sorted(files),
        "pathset_sha256": hashlib.sha256("\n".join(sorted(files)).encode()).hexdigest(),
        "files": files,
        **FIXED,
    }


def inspect_jsonl(path: Path, id_field: str) -> dict[str, Any]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"empty JSONL source: {path}")
    keysets = sorted({tuple(sorted(row)) for row in rows})
    identifiers = [str(row[id_field]) for row in rows]
    return {
        "rows": len(rows),
        "keysets": [list(keys) for keys in keysets],
        "id_unique": len(set(identifiers)),
        "id_digest": hashlib.sha256("\n".join(sorted(identifiers)).encode()).hexdigest(),
    }


def verify_sources(config: dict[str, Any]) -> list[dict[str, Any]]:
    bindings = []
    for source_id, relative, expected in config["source_bindings"]:
        path = ROOT / relative
        actual = sha256_file(path) if path.is_file() else None
        bindings.append(
            {
                "id": source_id,
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "status": "PASS" if actual == expected else "FAIL",
            }
        )
    plan = config["plan"]
    actual_plan = sha256_file(ROOT / plan["path"])
    bindings.append(
        {
            "id": "PLAN",
            "path": plan["path"],
            "expected_sha256": plan["sha256"],
            "actual_sha256": actual_plan,
            "status": "PASS" if actual_plan == plan["sha256"] else "FAIL",
        }
    )
    return bindings


def _prepare_protocol(public: Path, config: dict[str, Any]) -> None:
    """Serialize every recipe needed by B/E; those processes never open sources."""
    policy = json.loads(
        (ROOT / "data/geometria_proporcional/wave52_policy_transport_v1/policy_manifest.json").read_text()
    )
    platt = json.loads(
        (ROOT / "data/geometria_proporcional/wave53_uncertainty_policy_v1/platt_calibrator.json").read_text()
    )
    selection = json.loads(
        (ROOT / "data/geometria_proporcional/wave54_joint_set_v1/selection_freeze.json").read_text()
    )
    graph = json.loads(
        (ROOT / "data/geometria_proporcional/proportional_graph_neural_smoke_v1/resolved_config.json").read_text()
    )["graph"]
    theta = selection["selected_models"]["joint_full"]["theta"]
    payload = {
        "schema_version": "proportional-mapping-prepared-protocol-v1",
        "query": config["query"],
        "unit_namespaces": {
            "eiv": "w49-fixture",
            "set_valued": "w54-pair",
            "relational": "graph-view",
        },
        "common_contract": {
            "unit_bijection": None,
            "observation_schema": {
                "eiv": "continuous fixture tuple with covariance",
                "set_valued": "four ensemble logits",
                "relational": "typed graph with edge log-ratios",
            },
            "target_schema": {
                "eiv": "compatible parametric family set",
                "set_valued": "nonempty boolean family set",
                "relational": "continuous relation and quotient modulo gauge",
            },
            "score_semantics": {
                "eiv": "family score and conformal structural set",
                "set_valued": "probability mass over fifteen sets",
                "relational": "edge correction and reliability",
            },
            "executor": {"eiv": "conformal", "set_valued": "reader", "relational": "WLS_or_IRLS"},
            "reader": {"eiv": "structural_set", "set_valued": "hard_or_contextual", "relational": "quotient"},
        },
        "set_recipe": {
            "levels": policy["levels"],
            "rank_permutations": policy["rank_permutations"],
            "platt": {"coefficient": platt["coefficient"], "intercept": platt["intercept"]},
            "joint_theta": theta,
            "selection_contract": {
                "best_independent": selection["best_independent"],
                "sealed_monitor_accessed": selection["sealed_monitor_accessed"],
            },
            "reader": config["set_reader"],
        },
        "graph_recipe": {
            "weight_floor": graph["weight_floor"],
            "huber_delta": graph["huber_delta"],
            "irls_iterations": graph["irls_iterations"],
            "irls_damping": graph["irls_damping"],
        },
        "controls": config["controls"],
        "source_policy": config["source_policy"],
        "authority": {
            "utility": "SYNTHETIC_EXTERNAL",
            "monitor_or_lockbox_opened": False,
            "builder_phase": "PUBLIC_ONLY_BEFORE_PRIVATE_EVALUATION",
        },
        **FIXED,
    }
    write_json(public / "protocol.json", payload)


def _prepare_w49(public: Path, private: Path) -> None:
    visible: dict[str, Any] = {}
    predictions: dict[str, Any] = {}
    targets: dict[str, Any] = {}
    for split in ("train", "val"):
        visible_path = ROOT / f"data/geometria_proporcional/wave49/visible/{split}.jsonl"
        pred_path = ROOT / f"data/geometria_proporcional/wave49/predictions/{split}.jsonl"
        target_path = ROOT / f"data/geometria_proporcional/wave50_prospective_v1/authorized_labels/{split}.jsonl"
        visible[split] = inspect_jsonl(visible_path, "fixture_id")
        pred_rows = read_jsonl(pred_path)
        predictions[split] = {
            **inspect_jsonl(pred_path, "fixture_id"),
            "selectors": sorted({str(row["selector"]) for row in pred_rows}),
            "families": sorted({key for row in pred_rows for key in row["family_scores"]}),
        }
        target_rows = read_jsonl(target_path)
        targets[split] = {
            **inspect_jsonl(target_path, "fixture_id"),
            "pair_token_unique": len({str(row["pair_token"]) for row in target_rows}),
            "fixture_keys": [unit_key("w49-fixture", str(row["fixture_id"])) for row in target_rows],
            "pair_keys": [unit_key("w50-pair", str(row["pair_token"])) for row in target_rows],
            "target": [row["oracle_compatible_set"] for row in target_rows],
            "oracle_status": [row["oracle_status"] for row in target_rows],
        }
    write_json(
        public / "w49_contract.json",
        {
            "schema_version": "mapping-w49-public-v1",
            "observation_fields": [
                "fixture_id", "x", "y", "n", "covariance", "coordinate_semantics", "domain",
            ],
            "visible": visible,
            "predictions": predictions,
            **FIXED,
        },
    )
    write_json(
        private / "w49_targets.json",
        {"schema_version": "mapping-w49-private-v1", "splits": targets, **FIXED},
    )


def _prepare_w54(public: Path, private: Path) -> None:
    source = ROOT / "data/geometria_proporcional/wave54_joint_set_inputs_v1/fit_select_bundle.npz"
    with np.load(source, allow_pickle=False) as data:
        expected = {
            "pair_token", "cluster_id", "target", "per_seed_logits", "ensemble_logits",
            "design_stratum", "cardinality", "split_role",
        }
        if set(data.files) != expected:
            raise ValueError(f"Wave 54 keyset mismatch: {sorted(set(data.files) ^ expected)}")
        pair_keys = np.asarray([unit_key("w54-pair", str(value)) for value in data["pair_token"]])
        cluster_keys = np.asarray([unit_key("w54-cluster", str(value)) for value in data["cluster_id"]])
        for name, values in {
            "ensemble_logits": data["ensemble_logits"].astype(np.float64),
            "per_seed_logits": data["per_seed_logits"].astype(np.float64),
            "unit_key": pair_keys,
            "cluster_key": cluster_keys,
            "split_role": data["split_role"].astype(str),
        }.items():
            save_array(public / "w54" / f"{name}.npy", values)
        for name, values in {
            "target": data["target"].astype(bool),
            "design_stratum": data["design_stratum"].astype(str),
            "cardinality": data["cardinality"].astype(np.int64),
            "unit_key": pair_keys,
            "cluster_key": cluster_keys,
        }.items():
            save_array(private / "w54" / f"{name}.npy", values)


def _prepare_graph(public: Path, private: Path, config: dict[str, Any]) -> None:
    sources = {source_id: relative for source_id, relative, _ in config["source_bindings"]}
    state_rows: list[dict[str, Any]] = []
    for state_name, source_id in sorted(config["graph_states"].items()):
        source_path = ROOT / sources[source_id]
        state_dir = state_name.replace("|", "__")
        with np.load(source_path, allow_pickle=False) as data:
            required = set(PUBLIC_GRAPH_FIELDS) | set(GRAPH_CONTAINER_FIELDS) | set(GRAPH_PRIVATE_FIELDS) | {
                "corrected_log_ratio", "reliability"
            }
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"{state_name} missing fields: {sorted(missing)}")
            keys = np.asarray(
                [
                    unit_key("graph-view", f"{master}\0{view}")
                    for master, view in zip(data["master_id"].astype(str), data["view_id"].astype(str), strict=True)
                ]
            )
            public_arrays = {
                **{name: data[name] for name in PUBLIC_GRAPH_FIELDS + GRAPH_CONTAINER_FIELDS},
                "corrected_log_ratio": data["corrected_log_ratio"].astype(np.float64),
                "reliability": data["reliability"].astype(np.float64),
                "unit_key": keys,
            }
            private_arrays = {
                **{name: data[name] for name in GRAPH_PRIVATE_FIELDS},
                "edge_offsets": data["edge_offsets"],
                "node_offsets": data["node_offsets"],
                "unit_key": keys,
            }
            for name, values in public_arrays.items():
                save_array(public / "graph" / state_dir / f"{name}.npy", values)
            for name, values in private_arrays.items():
                save_array(private / "graph" / state_dir / f"{name}.npy", values)
            state_rows.append(
                {
                    "state": state_name,
                    "source_id": source_id,
                    "directory": state_dir,
                    "views": int(len(data["n_nodes"])),
                    "edges": int(len(data["observed_log_ratio"])),
                    "nodes": int(len(data["x_true"])),
                }
            )
    write_json(public / "graph_states.json", {"schema_version": "mapping-graph-public-v1", "states": state_rows, **FIXED})


def prepare(config_path: Path, run_dir: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "proportional-mapping-feasibility-config-v1":
        raise ValueError("config schema mismatch")
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    bindings = verify_sources(config)
    source_status = "PASS" if all(row["status"] == "PASS" for row in bindings) else "FAIL"
    write_json(
        run_dir / "source_inventory.json",
        {
            "schema_version": "proportional-mapping-source-inventory-v1",
            "source_status": source_status,
            "bindings": bindings,
            **FIXED,
        },
    )
    if source_status != "PASS":
        raise RuntimeError("source hash verification failed")
    public = run_dir / "prepared/public"
    private = run_dir / "prepared/private_dev"
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    _prepare_protocol(public, config)
    _prepare_w49(public, private)
    _prepare_w54(public, private)
    _prepare_graph(public, private, config)
    write_json(public / "manifest.json", file_manifest(public, "mapping-prepared-public-manifest-v1"))
    write_json(private / "manifest.json", file_manifest(private, "mapping-prepared-private-manifest-v1"))


def prepare_test_fixture(
    fixture_root: Path,
    output: Path,
    expected_manifest_sha256: str,
) -> None:
    """Test-only public/private split; it cannot write a scientific run."""
    fixture_root = fixture_root.resolve(strict=True)
    output = output.resolve()
    if "tmp" not in fixture_root.parts or "tmp" not in output.parts:
        raise ValueError("TEST_ONLY fixtures and outputs must live under a temporary root")
    manifest = fixture_root / "fixture_manifest.json"
    if sha256_file(manifest) != expected_manifest_sha256:
        raise RuntimeError("TEST_ONLY trust-root mismatch")
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    if spec.get("schema_version") != "mapping-private-invariance-fixture-v1":
        raise ValueError("TEST_ONLY fixture schema mismatch")
    if output.exists():
        raise FileExistsError(output)
    (output / "public").mkdir(parents=True)
    (output / "private_dev").mkdir(parents=True)
    write_json(output / "public/payload.json", spec["public"])
    write_json(output / "private_dev/payload.json", spec["private"])
    write_json(
        output / "test_receipt.json",
        {"schema_version": "mapping-test-only-receipt-v1", "fixture_mode": "TEST_ONLY", "test_decision": "PREPARED"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.config.resolve(strict=True), args.output.resolve())


if __name__ == "__main__":
    main()
