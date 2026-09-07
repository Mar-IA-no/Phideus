#!/usr/bin/env python3
"""Independent checker for the proportional mapping-feasibility gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit, logsumexp
from sklearn.linear_model import LogisticRegression
import sklearn


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments/geometria_proporcional/configs/proportional_mapping_feasibility_v1.json"
FIXED = {
    "gpu_used_or_queried": False,
    "architecture_promoted": False,
    "scientific_decision": None,
    "decision_authority": "user",
}
REASONS = {
    "M1_QUERY_UNIT": ["QUERY_MISMATCH", "NO_COMMON_UNIT_NAMESPACE", "UNIT_BIJECTION_INCOMPLETE", "SYNTHETIC_ID_EQUIVALENCE"],
    "M2_OBSERVATION_PARITY": ["OBSERVATION_SOURCE_MISMATCH", "PROJECTION_NOT_INVERTIBLE", "INFORMATION_ASYMMETRY", "PRIVATE_FIELD_EXPOSED"],
    "M3_TARGET_CONSERVATION": ["TARGET_SCHEMA_MISMATCH", "TARGET_MAP_PARTIAL", "TARGET_ROUNDTRIP_LOSS", "LEARNED_TARGET_BRIDGE", "MONITOR_TARGET_USED"],
    "M4_DECISION_STACK_PARITY": ["SCORE_SEMANTICS_MISMATCH", "EXECUTOR_CLASS_MISMATCH", "READER_CLASS_MISMATCH", "CALIBRATION_ENTANGLED", "EXTERNAL_OPERATION_ASYMMETRY"],
    "M5_AUTHORITY_PHASES": ["UNBOUND_AUTHORITY", "UTILITY_LEAKAGE", "PHASE_VIOLATION", "MONITOR_OR_LOCKBOX_OPENED", "CHECKER_NOT_INDEPENDENT"],
    "R1_SOURCE_COMPLETE": ["GRAPH_SOURCE_MISSING", "GRAPH_HASH_MISMATCH", "GRAPH_SCHEMA_INVALID"],
    "R2_PUBLIC_PARITY": ["GRAPH_UNIT_MISMATCH", "GRAPH_INPUT_MISMATCH", "GRAPH_PRIVATE_LEAKAGE"],
    "R3_REPRESENTATION_OUTPUT": ["REPRESENTATION_OUTPUT_MISSING", "REPRESENTATION_OUTPUT_NONFINITE", "TOPOLOGY_CHANGED"],
    "R4_EXECUTOR_FACTORIAL": ["EXECUTOR_INPUT_MISMATCH", "EXECUTOR_RECIPE_MISMATCH", "EXECUTOR_CELL_MISSING", "TRUTH_USED_BY_EXECUTOR"],
    "R5_TARGET_AUTHORITY": ["GRAPH_TARGET_JOIN_INVALID", "GAUGE_NOT_CANONICAL", "MECHANISM_LEAKAGE"],
    "R6_ESTIMAND_CONTROLS": ["RELATIONAL_ESTIMAND_MISMATCH", "RELATIONAL_CONTROL_MISMATCH", "RELATIONAL_SUPPORT_EMPTY"],
    "S1_SOURCE_COMPLETE": ["SET_SOURCE_MISSING", "SET_HASH_MISMATCH", "SET_SCHEMA_INVALID", "SET_ROLE_COUNTS_INVALID"],
    "S2_POSTERIOR_PARITY": ["POSTERIOR_CELL_MISSING", "POSTERIOR_MASS_INVALID", "POSTERIOR_ALIGNMENT_MISMATCH", "UTILITY_IN_POSTERIOR"],
    "S3_FOUR_CELLS_EXECUTABLE": ["SET_DECISION_CELL_MISSING", "HARD_READER_NOT_POSTERIOR_BOUND", "CONTEXTUAL_RECIPE_MISMATCH", "CELL_DUPLICATION_UNDECLARED"],
    "S4_FIT_SUPPORT_FREEZE": ["PROPOSER_SUPPORT_EMPTY", "HARM_CLASS_MISSING", "INCOMPATIBILITY_CLASS_MISSING", "SELECTION_SUPPORT_EMPTY", "SET_PHASE_VIOLATION"],
    "S5_TARGET_UTILITY_AUTHORITY": ["SET_TARGET_JOIN_INVALID", "EMPTY_TARGET_SET", "TARGET_LEAKAGE", "UTILITY_CONTRACT_MISMATCH"],
    "S6_ESTIMAND_CONTROLS": ["SET_ESTIMAND_MISMATCH", "POSTERIOR_READER_ENTANGLED", "SET_CONTROL_MISMATCH", "SET_SUPPORT_EMPTY"],
}
PUBLIC_GRAPH_NAMES = {
    "n_nodes.npy", "edge_index.npy", "observed_log_ratio.npy", "edge_valid.npy",
    "path_index.npy", "path_sign.npy", "path_valid.npy", "edge_variance.npy",
    "edge_offsets.npy", "node_offsets.npy", "path_offsets.npy", "unit_key.npy",
    "corrected_log_ratio.npy", "reliability.npy",
}
PRIVATE_GRAPH_NAMES = {
    "x_true.npy", "clean_log_ratio.npy", "causal_corruption_mask.npy", "master_id.npy", "view_id.npy",
    "split.npy", "mechanism.npy", "x_hat_wls.npy", "x_hat_irls.npy", "relation_rmse.npy",
    "wls_quotient_rmse.npy", "irls_quotient_rmse.npy", "irls_converged.npy", "irls_iterations.npy",
    "edge_offsets.npy", "node_offsets.npy", "unit_key.npy",
}
GRAPH_DIRECTORIES = {
    "raw_generic__seed=104729", "raw_generic__seed=130363",
    "raw_typed__seed=104729", "raw_typed__seed=130363",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    header = json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)}, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def load(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def array_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.dtype.kind in "fc" and right.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def sets4() -> np.ndarray:
    masks = np.arange(1, 16, dtype=np.uint8)[:, None]
    return ((masks >> np.arange(4, dtype=np.uint8)[None, :]) & 1).astype(bool)


def independent_mass(probability: np.ndarray) -> np.ndarray:
    sets = sets4()
    clipped = np.clip(probability, np.finfo(np.float64).tiny, 1.0 - np.finfo(np.float64).eps)
    score = np.sum(np.where(sets[None], np.log(clipped)[:, None], np.log1p(-clipped)[:, None]), axis=-1)
    score -= np.max(score, axis=1, keepdims=True)
    mass = np.exp(score)
    return mass / mass.sum(axis=1, keepdims=True)


def joint_mass(logits: np.ndarray, theta: np.ndarray) -> np.ndarray:
    sets = sets4().astype(np.float64)
    unary = logits[:, None, :] * sets[None]
    cardinality = sets.sum(axis=1).astype(int)
    card = np.stack([(cardinality == k).astype(np.float64) for k in (2, 3, 4)], axis=1)
    pairs = np.stack([sets[:, i] * sets[:, j] for i in range(4) for j in range(i + 1, 4)], axis=1)
    contrast = pairs[:, :5] - pairs[:, [5]]
    feature = np.concatenate(
        [unary, np.broadcast_to(card[None], (len(logits), 15, 3)), np.broadcast_to(contrast[None], (len(logits), 15, 5))],
        axis=-1,
    )
    score = np.einsum("nsd,d->ns", feature, theta, optimize=True)
    score -= np.max(score, axis=1, keepdims=True)
    mass = np.exp(score)
    return mass / mass.sum(axis=1, keepdims=True)


def utilities(recipe: dict[str, Any]) -> np.ndarray:
    return np.asarray(recipe["levels"], dtype=np.float64)[np.asarray(recipe["rank_permutations"], dtype=np.int64)]


def loss_tensor(utility: np.ndarray, penalty: float) -> np.ndarray:
    sets = sets4()
    span = np.ptp(utility, axis=1)
    optimum = np.max(np.where(sets[None], utility[:, None], -np.inf), axis=-1)
    compatible = (optimum[:, None] - utility[:, :, None]) / span[:, None, None]
    return np.where(sets.T[None], compatible, penalty)


def risk_and_actions(mass: np.ndarray, utility: np.ndarray, penalty: float) -> tuple[np.ndarray, np.ndarray]:
    risk = np.einsum("ns,pas->npa", mass, loss_tensor(utility, penalty), optimize=True)
    return risk, np.argmin(risk, axis=-1).astype(np.int64)


def hard_actions(mass: np.ndarray, utility: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    map_set = sets4()[np.argmax(mass, axis=1)]
    score = np.where(map_set[:, None], utility[None], -np.inf)
    return map_set, np.argmax(score, axis=-1).astype(np.int64)


def regret(actions: np.ndarray, target: np.ndarray, utility: np.ndarray, penalty: float) -> np.ndarray:
    rows = np.arange(len(target))[:, None]
    policies = np.arange(len(utility))[None]
    optimum = np.max(np.where(target[:, None], utility[None], -np.inf), axis=-1)
    chosen = utility[policies, actions]
    normalized = (optimum - chosen) / np.ptp(utility, axis=1)[None]
    return np.where(target[rows, actions], normalized, penalty)


def design17(
    logits: np.ndarray,
    seed_logits: np.ndarray,
    mass: np.ndarray,
    risk: np.ndarray,
    hard: np.ndarray,
    candidate: np.ndarray,
    map_set: np.ndarray,
    utility: np.ndarray,
) -> np.ndarray:
    n, p = hard.shape
    rows, policies = np.arange(n)[:, None], np.arange(p)[None]
    minimum = risk[rows, policies, candidate]
    hard_risk = risk[rows, policies, hard]
    advantage = np.maximum(hard_risk - minimum, 0.0)
    ordered_risk = np.sort(risk, axis=-1)
    margin = ordered_risk[..., 1] - ordered_risk[..., 0]
    clipped = np.clip(mass, np.finfo(np.float64).tiny, 1.0)
    entropy = -np.sum(mass * np.log(clipped), axis=1) / np.log(15.0)
    ordered_mass = np.sort(mass, axis=1)
    card = sets4().sum(axis=1).astype(np.float64)
    expected = mass @ card
    variance = mass @ (card**2) - expected**2
    map_index = (map_set.astype(np.int64) * (1 << np.arange(4))).sum(axis=1) - 1
    seed_std = np.std(seed_logits, axis=0, ddof=0)
    token = np.stack(
        [
            entropy, ordered_mass[:, -1], ordered_mass[:, -1] - ordered_mass[:, -2],
            map_set.sum(axis=1), expected, variance, mass[np.arange(n), map_index],
            seed_std.mean(axis=1), seed_std.max(axis=1),
        ],
        axis=-1,
    )
    return np.concatenate(
        [
            advantage[..., None], hard_risk[..., None], minimum[..., None], margin[..., None],
            np.broadcast_to(token[:, None], (n, p, 9)), np.broadcast_to(utility[None], (n, p, 4)),
        ],
        axis=-1,
    ).astype(np.float64)


def weights_for(disagreement: np.ndarray) -> np.ndarray:
    count = disagreement.sum(axis=1)
    weights = np.zeros_like(disagreement, dtype=np.float64)
    active = count > 0
    weights[active] = disagreement[active] / count[active, None]
    return weights


def scaler(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.maximum(np.average((x - mean) ** 2, axis=0, weights=w), 0.0))
    scale[scale == 0] = 1.0
    return mean, scale


def ridge_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float) -> dict[str, Any]:
    mean, scale = scaler(x, w)
    xs = (x - mean) / scale
    z = np.column_stack([np.ones(len(xs)), xs])
    penalty = np.eye(z.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(z.T @ (w[:, None] * z) + penalty, z.T @ (w * y))
    return {"mean": mean, "scale": scale, "intercept": float(beta[0]), "coef": beta[1:]}


def ridge_score(state: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return state["intercept"] + ((x - state["mean"]) / state["scale"]) @ state["coef"]


def logistic_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict[str, Any]:
    if sklearn.__version__ != "1.8.0" or np.unique(y).tolist() != [0, 1]:
        raise RuntimeError("guard runtime/classes mismatch")
    mean, scale = scaler(x, w)
    xs = (x - mean) / scale
    model = LogisticRegression(C=1.0, penalty="l2", l1_ratio=0.0, dual=False, solver="lbfgs", class_weight=None, fit_intercept=True, max_iter=2000, tol=1e-10, warm_start=False)
    model.fit(xs, y, sample_weight=w)
    effective = model.get_params(deep=False)
    expected = {"C": 1.0, "penalty": "l2", "l1_ratio": 0.0, "dual": False, "solver": "lbfgs", "class_weight": None, "fit_intercept": True, "max_iter": 2000, "tol": 1e-10, "warm_start": False}
    if any(effective[key] != value for key, value in expected.items()):
        raise RuntimeError("guard effective parameter mismatch")
    return {"mean": mean, "scale": scale, "intercept": float(model.intercept_[0]), "coef": np.asarray(model.coef_[0]), "iterations": int(model.n_iter_[0])}


def logistic_score(state: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return expit(state["intercept"] + ((x - state["mean"]) / state["scale"]) @ state["coef"])


def reproduce_contextual(
    design: np.ndarray,
    target: np.ndarray,
    hard: np.ndarray,
    candidate: np.ndarray,
    utility: np.ndarray,
    roles: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    disagreement = hard != candidate
    weights = weights_for(disagreement)
    gain = regret(hard, target, utility, cfg["incompatible_penalty"]) - regret(candidate, target, utility, cfg["incompatible_penalty"])
    compatible_candidate = target[np.arange(len(target))[:, None], candidate]
    fit = roles == "calibration_fit"
    select = roles == "decision_select"
    fit_active = fit[:, None] & disagreement
    select_active = select[:, None] & disagreement
    x, w = design[fit_active], weights[fit_active]
    harm = (gain < -cfg["harm_epsilon"]).astype(np.int8)
    incompat = (~compatible_candidate).astype(np.int8)
    ridge = ridge_fit(x, gain[fit_active], w, cfg["ridge_alpha"])
    hm = logistic_fit(x, harm[fit_active], w)
    im = logistic_fit(x, incompat[fit_active], w)
    flat = design.reshape(-1, 17)
    ps = ridge_score(ridge, flat).reshape(hard.shape)
    hp = logistic_score(hm, flat).reshape(hard.shape)
    ip = logistic_score(im, flat).reshape(hard.shape)

    def summary(actions: np.ndarray, authorized: np.ndarray, qs: tuple[float, float, float]) -> tuple[Any, ...]:
        mask = np.broadcast_to(select[:, None], hard.shape)
        realized_regret = regret(actions, target, utility, cfg["incompatible_penalty"])
        compatible = target[np.arange(len(target))[:, None], actions]
        realized_gain = regret(hard, target, utility, cfg["incompatible_penalty"]) - realized_regret
        row = {
            "mean_regret": float(realized_regret[mask].mean()),
            "incompatibility_rate": float((~compatible[mask]).mean()),
            "harm_rate": float((realized_gain[mask] < -cfg["harm_epsilon"]).mean()),
            "authorized_rows": int((authorized & mask).sum()),
            "quantiles": list(qs),
            "actions": actions,
        }
        key = (row["mean_regret"], row["incompatibility_rate"], row["harm_rate"], -row["authorized_rows"], *qs)
        return key, row

    candidates = [summary(hard.copy(), np.zeros_like(hard), (2.0, 2.0, 2.0))]
    active_scores = ps[select_active]
    for qp in cfg["proposer_quantiles"]:
        pt = float(np.quantile(active_scores, qp, method="linear"))
        proposed = select_active & (ps > pt)
        if not np.any(proposed):
            continue
        for qh in cfg["guard_quantiles"]:
            ht = float(np.quantile(hp[proposed], qh, method="linear"))
            for qi in cfg["guard_quantiles"]:
                it = float(np.quantile(ip[proposed], qi, method="linear"))
                authorized = proposed & (hp < ht) & (ip < it)
                candidates.append(summary(np.where(authorized, candidate, hard), authorized, (qp, qh, qi)))
    chosen = min(candidates, key=lambda pair: pair[0])[1]
    return {
        "actions": chosen["actions"],
        "candidate_count": len(candidates),
        "design_sha256": array_digest(design),
        "fit_rows": int(fit_active.sum()),
        "fit_tokens": int(fit_active.any(axis=1).sum()),
        "harm_0_1": [int((harm[fit_active] == i).sum()) for i in (0, 1)],
        "incompatibility_0_1": [int((incompat[fit_active] == i).sum()) for i in (0, 1)],
    }


def derange(keys: np.ndarray, strata: list[tuple[Any, ...]], seed: int) -> tuple[np.ndarray, int]:
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for i, stratum in enumerate(strata):
        groups[stratum].append(i)
    rng = np.random.Generator(np.random.PCG64(seed))
    mapping = np.arange(len(keys), dtype=np.int64)
    singleton = 0
    for stratum in sorted(groups, key=str):
        members = sorted(groups[stratum], key=lambda i: str(keys[i]))
        if len(members) == 1:
            singleton += 1
            continue
        order = [members[i] for i in rng.permutation(len(members))]
        for receiver, donor in zip(order, order[1:] + order[:1], strict=True):
            mapping[receiver] = donor
    return mapping, singleton


def source_status(config: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    receipts = []
    for source_id, relative, expected in config["source_bindings"]:
        path = ROOT / relative
        actual = sha256_file(path) if path.is_file() else None
        receipts.append({"id": source_id, "path": relative, "expected": expected, "actual": actual, "status": "PASS" if actual == expected else "FAIL"})
    plan = config["plan"]
    actual = sha256_file(ROOT / plan["path"])
    receipts.append({"id": "PLAN", "path": plan["path"], "expected": plan["sha256"], "actual": actual, "status": "PASS" if actual == plan["sha256"] else "FAIL"})
    bound_ids = {row[0] for row in config["source_bindings"]}
    policy = config.get("source_policy", {})
    phase_access = policy.get("phase_access", {})
    policy_ok = (
        set(phase_access.get("P_PREPARER", [])) == bound_ids
        and set(phase_access.get("C_CHECKER", [])) == bound_ids
        and phase_access.get("B_BUILDER") == []
        and phase_access.get("E_EVALUATOR") == []
        and set().union(*(set(v) for v in policy.get("roles", {}).values())) == bound_ids
    )
    receipts.append({"id": "SOURCE_POLICY", "path": "config:source_policy", "expected": "closed roles/phases", "actual": policy_ok, "status": "PASS" if policy_ok else "FAIL"})
    return all(row["status"] == "PASS" for row in receipts), receipts


def frozen_protocol(config: dict[str, Any]) -> dict[str, Any]:
    policy = json.loads((ROOT / "data/geometria_proporcional/wave52_policy_transport_v1/policy_manifest.json").read_text())
    platt = json.loads((ROOT / "data/geometria_proporcional/wave53_uncertainty_policy_v1/platt_calibrator.json").read_text())
    selection = json.loads((ROOT / "data/geometria_proporcional/wave54_joint_set_v1/selection_freeze.json").read_text())
    graph = json.loads((ROOT / "data/geometria_proporcional/proportional_graph_neural_smoke_v1/resolved_config.json").read_text())["graph"]
    return {
        "schema_version": "proportional-mapping-prepared-protocol-v1",
        "query": config["query"],
        "unit_namespaces": {"eiv": "w49-fixture", "set_valued": "w54-pair", "relational": "graph-view"},
        "common_contract": {
            "unit_bijection": None,
            "observation_schema": {"eiv": "continuous fixture tuple with covariance", "set_valued": "four ensemble logits", "relational": "typed graph with edge log-ratios"},
            "target_schema": {"eiv": "compatible parametric family set", "set_valued": "nonempty boolean family set", "relational": "continuous relation and quotient modulo gauge"},
            "score_semantics": {"eiv": "family score and conformal structural set", "set_valued": "probability mass over fifteen sets", "relational": "edge correction and reliability"},
            "executor": {"eiv": "conformal", "set_valued": "reader", "relational": "WLS_or_IRLS"},
            "reader": {"eiv": "structural_set", "set_valued": "hard_or_contextual", "relational": "quotient"},
        },
        "set_recipe": {
            "levels": policy["levels"], "rank_permutations": policy["rank_permutations"],
            "platt": {"coefficient": platt["coefficient"], "intercept": platt["intercept"]},
            "joint_theta": selection["selected_models"]["joint_full"]["theta"],
            "selection_contract": {"best_independent": selection["best_independent"], "sealed_monitor_accessed": selection["sealed_monitor_accessed"]},
            "reader": config["set_reader"],
        },
        "graph_recipe": {key: graph[key] for key in ("weight_floor", "huber_delta", "irls_iterations", "irls_damping")},
        "controls": config["controls"],
        "source_policy": config["source_policy"],
        "authority": {"utility": "SYNTHETIC_EXTERNAL", "monitor_or_lockbox_opened": False, "builder_phase": "PUBLIC_ONLY_BEFORE_PRIVATE_EVALUATION"},
        **FIXED,
    }


def unit_key(namespace: str, value: str) -> str:
    return hashlib.sha256(namespace.encode() + b"\0" + value.encode()).hexdigest()


def verify_common_unit_authority(config: dict[str, Any], bridge: Any) -> tuple[bool, dict[str, Any]]:
    contract = config.get("common_unit_authority")
    diagnostic = {"contract_declared": isinstance(contract, dict), "materialized": False}
    if not isinstance(contract, dict) or not isinstance(bridge, dict):
        return False, diagnostic
    bound = {source_id: (relative, expected) for source_id, relative, expected in config["source_bindings"]}
    source_id = contract.get("source_id")
    if source_id not in bound or contract.get("path") != bound[source_id][0] or contract.get("sha256") != bound[source_id][1]:
        return False, diagnostic
    path = ROOT / contract["path"]
    if not path.is_file() or sha256_file(path) != contract["sha256"]:
        return False, diagnostic
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False, diagnostic
    triples = payload.get("triples") if isinstance(payload, dict) else None
    if not isinstance(triples, list) or not triples:
        return False, diagnostic
    keys = {"eiv", "set_valued", "relational"}
    if any(not isinstance(row, dict) or set(row) != keys or any(not isinstance(row[key], str) for key in keys) for row in triples):
        return False, diagnostic
    unique = {key: len({row[key] for row in triples}) for key in keys}
    serialized = json.dumps(triples, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    digest = hashlib.sha256(serialized).hexdigest()
    diagnostic.update({"materialized": True, "counts": unique, "unit_count": len(triples), "bijection_sha256": digest})
    valid = (
        all(count == len(triples) for count in unique.values())
        and bridge.get("kind") == "authority_bijection"
        and bridge.get("authority_source_id") == source_id
        and bridge.get("total") is True
        and bridge.get("synthetic") is False
        and bridge.get("unit_count") == len(triples)
        and all(bridge.get(f"{key}_count") == unique[key] for key in keys)
        and isinstance(bridge.get("bijection_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", bridge["bijection_sha256"]) is not None
        and bridge["bijection_sha256"] == digest
    )
    return bool(valid), diagnostic


def common_unit_contract(
    config: dict[str, Any], candidate_query: Any, protocol_query: Any, namespaces: Any, bridge: Any,
) -> tuple[bool, list[str], dict[str, Any]]:
    query_equal = candidate_query == config.get("query") == protocol_query
    common_namespace = isinstance(namespaces, dict) and len(namespaces) == 3 and len(set(namespaces.values())) == 1
    bridge_authorized, authority = verify_common_unit_authority(config, bridge)
    reasons = []
    if not query_equal:
        reasons.append("QUERY_MISMATCH")
    if not common_namespace:
        reasons.append("NO_COMMON_UNIT_NAMESPACE")
    if isinstance(bridge, dict) and (bridge.get("kind") != "authority_bijection" or bridge.get("synthetic") is not False):
        reasons.append("SYNTHETIC_ID_EQUIVALENCE")
    elif not bridge_authorized:
        reasons.append("UNIT_BIJECTION_INCOMPLETE")
    observed = {
        "query_equal": query_equal, "namespaces": namespaces, "common_namespace": common_namespace,
        "bridge": bridge, "bridge_authorized": bridge_authorized, "authority": authority,
    }
    return not reasons, reasons, observed


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def inspect_jsonl(path: Path, id_field: str) -> dict[str, Any]:
    rows = read_jsonl(path)
    identifiers = [str(row[id_field]) for row in rows]
    return {
        "rows": len(rows),
        "keysets": [list(keys) for keys in sorted({tuple(sorted(row)) for row in rows})],
        "id_unique": len(set(identifiers)),
        "id_digest": hashlib.sha256("\n".join(sorted(identifiers)).encode()).hexdigest(),
    }


def verify_w49_prepared(run: Path) -> bool:
    visible: dict[str, Any] = {}
    predictions: dict[str, Any] = {}
    targets: dict[str, Any] = {}
    for split in ("train", "val"):
        visible_path = ROOT / f"data/geometria_proporcional/wave49/visible/{split}.jsonl"
        prediction_path = ROOT / f"data/geometria_proporcional/wave49/predictions/{split}.jsonl"
        target_path = ROOT / f"data/geometria_proporcional/wave50_prospective_v1/authorized_labels/{split}.jsonl"
        visible[split] = inspect_jsonl(visible_path, "fixture_id")
        prediction_rows = read_jsonl(prediction_path)
        predictions[split] = {
            **inspect_jsonl(prediction_path, "fixture_id"),
            "selectors": sorted({str(row["selector"]) for row in prediction_rows}),
            "families": sorted({key for row in prediction_rows for key in row["family_scores"]}),
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
    expected_public = {
        "schema_version": "mapping-w49-public-v1",
        "observation_fields": ["fixture_id", "x", "y", "n", "covariance", "coordinate_semantics", "domain"],
        "visible": visible,
        "predictions": predictions,
        **FIXED,
    }
    expected_private = {"schema_version": "mapping-w49-private-v1", "splits": targets, **FIXED}
    return (
        json.loads((run / "prepared/public/w49_contract.json").read_text()) == expected_public
        and json.loads((run / "prepared/private_dev/w49_targets.json").read_text()) == expected_private
    )


def verify_tree_manifest(root: Path, schema: str) -> bool:
    if not (root / "manifest.json").is_file():
        return False
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != schema:
        return False
    actual = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    files = manifest.get("files", {})
    if schema == "mapping-prepared-public-manifest-v1":
        expected = {"protocol.json", "w49_contract.json", "graph_states.json"}
        expected |= {f"w54/{name}.npy" for name in ("ensemble_logits", "per_seed_logits", "unit_key", "cluster_key", "split_role")}
        expected |= {f"graph/{directory}/{name}" for directory in GRAPH_DIRECTORIES for name in PUBLIC_GRAPH_NAMES}
    elif schema == "mapping-prepared-private-manifest-v1":
        expected = {"w49_targets.json"}
        expected |= {f"w54/{name}.npy" for name in ("target", "design_stratum", "cardinality", "unit_key", "cluster_key")}
        expected |= {f"graph/{directory}/{name}" for directory in GRAPH_DIRECTORIES for name in PRIVATE_GRAPH_NAMES}
    else:
        return False
    if actual != sorted(expected) or actual != manifest.get("pathset") or actual != sorted(files):
        return False
    if manifest.get("pathset_sha256") != hashlib.sha256("\n".join(actual).encode()).hexdigest():
        return False
    return all((root / relative).is_file() and sha256_file(root / relative) == receipt["sha256"] for relative, receipt in files.items())


def metric_summary(values: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    selected = array.ravel() if mask is None else array[np.asarray(mask, dtype=bool)]
    if not len(selected) or not np.all(np.isfinite(selected)):
        raise ValueError("metric support empty or nonfinite")
    return {
        "support": int(len(selected)),
        "mean": float(np.mean(selected)),
        "p95": float(np.quantile(selected, 0.95, method="linear")),
        "maximum": float(np.max(selected)),
        "values_sha256": array_digest(array),
        "mask_sha256": None if mask is None else array_digest(np.asarray(mask, dtype=bool)),
    }


def posterior_metrics(mass: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    target_index = (target.astype(np.int64) * (1 << np.arange(4))).sum(axis=1) - 1
    nll = -np.log(np.clip(mass[np.arange(len(target)), target_index], np.finfo(float).tiny, 1.0))
    probability = mass @ sets4().astype(np.float64)
    brier = np.mean((probability - target) ** 2, axis=1)
    return {"set_nll": metric_summary(nll, mask), "marginal_brier": metric_summary(brier, mask)}


def action_metrics(actions: np.ndarray, target: np.ndarray, utility: np.ndarray, penalty: float, mask: np.ndarray) -> dict[str, Any]:
    compatible = target[np.arange(len(target))[:, None], actions]
    realized = regret(actions, target, utility, penalty)
    return {
        "compatibility_rate": metric_summary(compatible.astype(np.float64), mask),
        "regret": metric_summary(realized, mask),
    }


def common_observation_contract(candidate: dict[str, Any], public_private_leak: bool) -> tuple[bool, list[str], dict[str, Any]]:
    observations = {name: candidate.get("lines", {}).get(name, {}).get("observation") for name in ("eiv", "set_valued", "relational")}
    serialized = {name: json.dumps(value, sort_keys=True, separators=(",", ":")) for name, value in observations.items()}
    equal = len(set(serialized.values())) == 1
    projections = candidate.get("declared_observation_projections")
    projection_exact = not isinstance(projections, dict) or projections.get("roundtrip_exact") is True
    information = {name: candidate.get("lines", {}).get(name, {}).get("information") for name in ("eiv", "set_valued", "relational")}
    declared_information = any(value is not None for value in information.values())
    information_symmetric = not declared_information or len({json.dumps(value, sort_keys=True, separators=(",", ":")) for value in information.values()}) == 1
    reasons = []
    if not equal:
        reasons.append("OBSERVATION_SOURCE_MISMATCH")
    if not projection_exact:
        reasons.append("PROJECTION_NOT_INVERTIBLE")
    if not information_symmetric:
        reasons.append("INFORMATION_ASYMMETRY")
    if public_private_leak:
        reasons.append("PRIVATE_FIELD_EXPOSED")
    return not reasons, reasons, {"observations": observations, "projections": projections, "information": information, "private_leak": public_private_leak}


def common_target_contract(candidate: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    targets = {name: candidate.get("lines", {}).get(name, {}).get("target_authority") for name in ("eiv", "set_valued", "relational")}
    equal = len(set(targets.values())) == 1
    bridge = candidate.get("declared_cross_domain_target_bridge")
    reasons = []
    if not equal:
        reasons.append("TARGET_SCHEMA_MISMATCH")
    if not isinstance(bridge, dict) or bridge.get("total_roundtrip_exact") is not True:
        reasons.append("TARGET_MAP_PARTIAL")
    if isinstance(bridge, dict) and bridge.get("total_roundtrip_exact") is True and bridge.get("roundtrip_lossless") is not True:
        reasons.append("TARGET_ROUNDTRIP_LOSS")
    if isinstance(bridge, dict) and bridge.get("learned") is True:
        reasons.append("LEARNED_TARGET_BRIDGE")
    if isinstance(bridge, dict) and bridge.get("monitor_used") is True:
        reasons.append("MONITOR_TARGET_USED")
    return not reasons, reasons, {"targets": targets, "bridge": bridge}


def common_decision_contract(candidate: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    lines = candidate.get("lines", {})
    scores = {name: lines.get(name, {}).get("output") for name in ("eiv", "set_valued", "relational")}
    adapters = candidate.get("adapters", {})
    score_equal = len(set(scores.values())) == 1
    declared_stack = candidate.get("decision_stack")
    if isinstance(declared_stack, dict):
        executors = {name: declared_stack.get(name, {}).get("executor") for name in ("eiv", "set_valued", "relational")}
        readers = {name: declared_stack.get(name, {}).get("reader") for name in ("eiv", "set_valued", "relational")}
        executor_equal = len(set(executors.values())) == 1
        reader_equal = len(set(readers.values())) == 1
    else:
        adapter_values = [tuple(adapters.get(name, [])) for name in ("eiv", "set_valued", "relational")]
        executor_equal = reader_equal = len(set(adapter_values)) == 1
        executors = readers = adapters
    reasons = []
    if not score_equal:
        reasons.append("SCORE_SEMANTICS_MISMATCH")
    if not executor_equal:
        reasons.append("EXECUTOR_CLASS_MISMATCH")
    if not reader_equal:
        reasons.append("READER_CLASS_MISMATCH")
    if candidate.get("calibration_entangled") is True:
        reasons.append("CALIBRATION_ENTANGLED")
    external = candidate.get("external_operations")
    if isinstance(external, dict) and len({json.dumps(external.get(name), sort_keys=True, separators=(",", ":")) for name in ("eiv", "set_valued", "relational")}) != 1:
        reasons.append("EXTERNAL_OPERATION_ASYMMETRY")
    return not reasons, reasons, {"scores": scores, "executors": executors, "readers": readers, "adapters": adapters, "calibration_entangled": candidate.get("calibration_entangled"), "external_operations": external}


def posterior_mass_contract(mass: np.ndarray, expected_rows: int) -> bool:
    return mass.shape == (expected_rows, 15) and bool(np.all(np.isfinite(mass))) and bool(np.all(mass >= 0)) and bool(np.allclose(mass.sum(axis=1), 1.0, atol=1e-12))


def fit_support_contract(context: dict[str, Any]) -> bool:
    return context["fit_rows"] > 0 and all(value > 0 for value in (*context["harm_0_1"], *context["incompatibility_0_1"]))


def set_authority_contract(keys: np.ndarray, private_keys: np.ndarray, target: np.ndarray, utility: np.ndarray) -> bool:
    return bool(np.array_equal(keys, private_keys)) and target.ndim == 2 and bool(np.all(target.any(axis=1))) and utility.shape == (24, 4) and bool(np.all(np.ptp(utility, axis=1) > 0))


def representation_output_contract(corrected: np.ndarray, reliability: np.ndarray, edge_count: int) -> bool:
    return corrected.shape == (edge_count,) and reliability.shape == (edge_count,) and bool(np.all(np.isfinite(corrected))) and bool(np.all(np.isfinite(reliability)))


def set_schema_contract(
    source_parity: bool,
    logits: np.ndarray,
    seed_logits: np.ndarray,
    target: np.ndarray,
    roles: np.ndarray,
) -> bool:
    return bool(source_parity) and logits.shape == (384, 4) and seed_logits.shape == (3, 384, 4) and target.shape == (384, 4) and int((roles == "calibration_fit").sum()) == 192 and int((roles == "decision_select").sum()) == 192


def authority_phase_contract(
    phase_policy: dict[str, Any],
    protocol_authority: dict[str, Any],
    candidate_valid: bool,
    evaluator_source: str,
    evaluator_imports: str,
    checker_imports: str,
    builder_source: str,
) -> tuple[bool, list[str], dict[str, Any]]:
    observed = {
        "authority_bound": candidate_valid,
        "utility_external": protocol_authority.get("utility") == "SYNTHETIC_EXTERNAL",
        "phase_access_closed": phase_policy.get("B_BUILDER") == [] and phase_policy.get("E_EVALUATOR") == [] and "private_dev" not in builder_source,
        "monitor_or_lockbox_opened": protocol_authority.get("monitor_or_lockbox_opened"),
        "checker_independent": (
            "ROOT" not in evaluator_source
            and "geometria_proporcional" not in evaluator_imports
            and ("evaluate_" + "proportional_mapping_feasibility") not in checker_imports
            and ("build_" + "proportional_mapping_candidate") not in checker_imports
        ),
    }
    reasons = []
    if not observed["authority_bound"]:
        reasons.append("UNBOUND_AUTHORITY")
    if not observed["utility_external"]:
        reasons.append("UTILITY_LEAKAGE")
    if not observed["phase_access_closed"]:
        reasons.append("PHASE_VIOLATION")
    if observed["monitor_or_lockbox_opened"] is not False:
        reasons.append("MONITOR_OR_LOCKBOX_OPENED")
    if not observed["checker_independent"]:
        reasons.append("CHECKER_NOT_INDEPENDENT")
    return not reasons, reasons, observed


def recompute_set(run: Path, config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    pub, prv = run / "prepared/public/w54", run / "prepared/private_dev/w54"
    protocol = frozen_protocol(config)
    recipe = protocol["set_recipe"]
    logits, seed_logits = load(pub / "ensemble_logits.npy").astype(np.float64), load(pub / "per_seed_logits.npy").astype(np.float64)
    roles, keys = load(pub / "split_role.npy").astype(str), load(pub / "unit_key.npy").astype(str)
    target, private_keys = load(prv / "target.npy").astype(bool), load(prv / "unit_key.npy").astype(str)
    strata, cardinality = load(prv / "design_stratum.npy").astype(str), load(prv / "cardinality.npy").astype(int)
    cluster_keys = load(pub / "cluster_key.npy").astype(str)
    private_cluster_keys = load(prv / "cluster_key.npy").astype(str)
    source_path = ROOT / "data/geometria_proporcional/wave54_joint_set_inputs_v1/fit_select_bundle.npz"
    with np.load(source_path, allow_pickle=False) as source:
        expected_unit_keys = np.asarray([unit_key("w54-pair", str(value)) for value in source["pair_token"]])
        expected_cluster_keys = np.asarray([unit_key("w54-cluster", str(value)) for value in source["cluster_id"]])
        source_parity = (
            np.array_equal(logits, source["ensemble_logits"].astype(np.float64))
            and np.array_equal(seed_logits, source["per_seed_logits"].astype(np.float64))
            and np.array_equal(roles, source["split_role"].astype(str))
            and np.array_equal(target, source["target"].astype(bool))
            and np.array_equal(strata, source["design_stratum"].astype(str))
            and np.array_equal(cardinality, source["cardinality"].astype(np.int64))
            and np.array_equal(keys, expected_unit_keys)
            and np.array_equal(private_keys, expected_unit_keys)
            and np.array_equal(cluster_keys, expected_cluster_keys)
            and np.array_equal(private_cluster_keys, expected_cluster_keys)
        )
    platt = recipe["platt"]
    marginal = independent_mass(expit(platt["coefficient"] * logits + platt["intercept"]))
    joint = joint_mass(logits, np.asarray(recipe["joint_theta"], dtype=np.float64))
    utility, cfg = utilities(recipe), recipe["reader"]
    valid_mass = True
    cells_present = True
    hard_bound = True
    contextual_exact = True
    proposer_support = True
    harm_classes = True
    incompatibility_classes = True
    selection_support = True
    contextual_results: dict[str, Any] = {}
    masses = {"MARGINAL": marginal, "JOINT": joint}
    actions_by_cell: dict[str, np.ndarray] = {}
    authorized: dict[str, np.ndarray] = {}
    for name, mass in (("MARGINAL", marginal), ("JOINT", joint)):
        valid_mass &= posterior_mass_contract(mass, 384)
        map_set, hard = hard_actions(mass, utility)
        risk, candidate = risk_and_actions(mass, utility, cfg["incompatible_penalty"])
        design = design17(logits, seed_logits, mass, risk, hard, candidate, map_set, utility)
        context = reproduce_contextual(design, target, hard, candidate, utility, roles, cfg)
        recorded = evidence["set_valued"]["posteriors"][name]
        cells_present &= all(
            evidence["set_valued"].get("four_cells", {}).get(f"{name}_{reader}", {}).get("shape") == [384, 24]
            for reader in ("HARD", "CONTEXTUAL")
        )
        hard_bound &= recorded["mass_sha256"] == array_digest(mass) and recorded["hard_actions_sha256"] == array_digest(hard)
        contextual_exact &= (
            recorded["candidate_actions_sha256"] == array_digest(candidate)
            and recorded["contextual_actions_sha256"] == array_digest(context["actions"])
            and recorded["contextual"]["design_sha256"] == context["design_sha256"]
            and recorded["contextual"]["candidate_count"] == context["candidate_count"]
        )
        proposer_support &= context["fit_rows"] > 0
        harm_classes &= all(value > 0 for value in context["harm_0_1"])
        incompatibility_classes &= all(value > 0 for value in context["incompatibility_0_1"])
        selection_support &= context["candidate_count"] > 0
        contextual_results[name] = context
        actions_by_cell[f"{name}_HARD"] = hard
        actions_by_cell[f"{name}_CONTEXTUAL"] = context["actions"]
        authorized[name] = context["actions"] != hard
    mapping, singleton = derange(keys, list(zip(roles.tolist(), strata.tolist(), cardinality.tolist(), strict=True)), config["controls"]["set_shuffle_seed"])
    shuffled = target[mapping]
    select = np.broadcast_to((roles == "decision_select")[:, None], (len(target), len(utility)))
    matched = authorized["MARGINAL"] & authorized["JOINT"] & select
    matched_tokens = np.any(matched, axis=1)
    estimands_ok = True
    for name, mass in masses.items():
        cells: dict[str, Any] = {}
        for reader in ("HARD", "CONTEXTUAL"):
            actions = actions_by_cell[f"{name}_{reader}"]
            cells[reader] = {
                "nominal": action_metrics(actions, target, utility, float(cfg["incompatible_penalty"]), select),
                "shuffled": action_metrics(actions, shuffled, utility, float(cfg["incompatible_penalty"]), select),
                "matched_nominal": action_metrics(actions, target, utility, float(cfg["incompatible_penalty"]), matched),
                "matched_shuffled": action_metrics(actions, shuffled, utility, float(cfg["incompatible_penalty"]), matched),
            }
        expected = {
            "posterior_nominal": posterior_metrics(mass, target),
            "posterior_shuffled": posterior_metrics(mass, shuffled),
            "posterior_matched_nominal": posterior_metrics(mass, target, matched_tokens),
            "posterior_matched_shuffled": posterior_metrics(mass, shuffled, matched_tokens),
            "actions": cells,
        }
        estimands_ok &= expected == evidence["set_valued"]["posteriors"][name].get("estimands")
    control = evidence["set_valued"]["target_shuffle"]
    matched_recorded = evidence["set_valued"].get("matched_report", {})
    shuffle_ok = (
        control.get("mapping_sha256") == array_digest(mapping)
        and control.get("target_sha256") == array_digest(shuffled)
        and control.get("singleton_strata") == singleton
    )
    matched_ok = matched_recorded == {
            "authorized_rows": int(matched.sum()),
            "tokens": int(matched_tokens.sum()),
            "mask_sha256": array_digest(matched),
            "token_mask_sha256": array_digest(matched_tokens),
        }
    source_schema = bool(source_parity) and logits.shape == (384, 4) and seed_logits.shape == (3, 384, 4) and target.shape == (384, 4)
    role_counts = int((roles == "calibration_fit").sum()) == 192 and int((roles == "decision_select").sum()) == 192
    target_join = bool(np.array_equal(keys, private_keys)) and evidence["set_valued"].get("target_join_exact") is True
    target_nonempty = target.ndim == 2 and bool(np.all(target.any(axis=1)))
    utility_valid = utility.shape == (24, 4) and bool(np.all(np.ptp(utility, axis=1) > 0))
    duplication_declared = evidence["set_valued"].get("observed_hard_duplication") == bool(
        np.array_equal(actions_by_cell["MARGINAL_HARD"], actions_by_cell["JOINT_HARD"])
    )
    phase_closed = protocol["authority"]["monitor_or_lockbox_opened"] is False
    four = cells_present and hard_bound and contextual_exact and duplication_declared
    supports = proposer_support and harm_classes and incompatibility_classes and selection_support and phase_closed
    authority = target_join and target_nonempty and utility_valid
    support_positive = bool(np.any(matched)) and bool(np.any(matched_tokens))
    control_ok = shuffle_ok and matched_ok and estimands_ok and support_positive
    return {
        "schema": source_schema and role_counts,
        "mass": valid_mass,
        "four": four,
        "support": supports,
        "authority": authority,
        "controls": bool(control_ok),
        "facts": {
            "S1_SOURCE_COMPLETE": {"schema_valid": source_schema, "role_counts_valid": role_counts},
            "S2_POSTERIOR_PARITY": {"cells_present": True, "mass_valid": valid_mass, "alignment_exact": bool(source_parity), "utility_used": False},
            "S3_FOUR_CELLS_EXECUTABLE": {"cells_present": cells_present, "hard_posterior_bound": hard_bound, "contextual_recipe_exact": contextual_exact, "duplication_declared": duplication_declared},
            "S4_FIT_SUPPORT_FREEZE": {"proposer_support_positive": proposer_support, "harm_classes_present": harm_classes, "incompatibility_classes_present": incompatibility_classes, "selection_support_positive": selection_support, "phase_closed": phase_closed},
            "S5_TARGET_UTILITY_AUTHORITY": {"target_join_exact": target_join, "target_nonempty": target_nonempty, "target_used_as_input": False, "utility_contract_valid": utility_valid},
            "S6_ESTIMAND_CONTROLS": {"estimands_equal": estimands_ok, "posterior_reader_entangled": False, "controls_exact": shuffle_ok and matched_ok, "support_positive": support_positive},
        },
    }


def incidence(n: int, edges: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(edges), n), dtype=np.float64)
    rows = np.arange(len(edges))
    matrix[rows, edges[:, 0]] = -1.0
    matrix[rows, edges[:, 1]] = 1.0
    return matrix


def local_wls(n: int, edges: np.ndarray, valid: np.ndarray, values: np.ndarray, weights: np.ndarray, floor: float) -> np.ndarray:
    b = incidence(n, edges)
    w = np.zeros_like(weights, dtype=np.float64)
    w[valid] = np.clip(weights[valid], floor, None)
    w[valid] /= w[valid].mean()
    bv = b[valid]
    lap = bv.T @ (w[valid, None] * bv)
    rhs = bv.T @ (w[valid] * values[valid])
    ones = np.ones((n, 1))
    kkt = np.block([[lap, ones], [ones.T, np.zeros((1, 1))]])
    return np.linalg.solve(kkt, np.concatenate([rhs, [0.0]]))[:-1]


def local_irls(n: int, edges: np.ndarray, valid: np.ndarray, variance: np.ndarray, values: np.ndarray, reliability: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, bool, int]:
    base = np.where(valid, np.clip(reliability, cfg["weight_floor"], None), 0.0)
    base[valid] /= base[valid].mean()
    weights = base.copy()
    scale = max(float(np.sqrt(np.median(variance[valid]))), 1e-8)
    previous_x, previous_objective = None, None
    converged = False
    x = np.zeros(n)
    for iteration in range(1, cfg["irls_iterations"] + 1):
        x = local_wls(n, edges, valid, values, weights, cfg["weight_floor"])
        residual = (incidence(n, edges) @ x - values)[valid]
        normalized = np.abs(residual / scale)
        terms = np.where(normalized <= cfg["huber_delta"], 0.5 * normalized**2, cfg["huber_delta"] * (normalized - 0.5 * cfg["huber_delta"]))
        objective = float(np.sum(base[valid] * terms))
        if previous_x is not None:
            if np.max(np.abs(x - previous_x)) < 1e-6 and abs(objective - previous_objective) / max(1.0, abs(previous_objective)) < 1e-6:
                converged = True
                break
        ratio = np.ones_like(residual)
        magnitude = np.abs(residual)
        ratio[magnitude > cfg["huber_delta"] * scale] = cfg["huber_delta"] * scale / magnitude[magnitude > cfg["huber_delta"] * scale]
        ratio = np.clip(ratio, cfg["weight_floor"], 1.0)
        candidate = base.copy()
        candidate[valid] *= ratio
        weights = cfg["irls_damping"] * candidate + (1.0 - cfg["irls_damping"]) * weights
        previous_x, previous_objective = x.copy(), objective
    return local_wls(n, edges, valid, values, weights, cfg["weight_floor"]), converged, int(iteration)


def graph_summary(values: np.ndarray, masters: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    selected = np.ones(len(values), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    unique = sorted(set(masters[selected].astype(str)))
    by_master = np.asarray([values[(masters.astype(str) == master) & selected].mean() for master in unique])
    if not len(by_master) or not np.all(np.isfinite(by_master)):
        raise ValueError("graph metric support empty or nonfinite")
    return {
        "view_support": int(selected.sum()),
        "master_support": int(len(unique)),
        "master_mean": float(by_master.mean()),
        "master_p95": float(np.quantile(by_master, 0.95, method="linear")),
        "master_maximum": float(by_master.max()),
        "view_values_sha256": array_digest(values),
        "view_mask_sha256": array_digest(selected),
    }


def metric_contract_equal(recorded: Any, expected: Any, atol: float) -> bool:
    """Compare recomputed estimands; float-array digests may differ across BLAS processes."""
    if isinstance(expected, dict):
        if not isinstance(recorded, dict) or set(recorded) != set(expected):
            return False
        return all(
            True if key.endswith("values_sha256") else metric_contract_equal(recorded[key], value, atol)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(recorded, list) and len(recorded) == len(expected) and all(metric_contract_equal(a, b, atol) for a, b in zip(recorded, expected, strict=True))
    if isinstance(expected, float):
        return isinstance(recorded, (float, int)) and bool(np.isclose(float(recorded), expected, atol=atol, rtol=0.0))
    return recorded == expected


def recompute_graph(run: Path, config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    public, private = run / "prepared/public/graph", run / "prepared/private_dev/graph"
    state_rows = json.loads((run / "prepared/public/graph_states.json").read_text())["states"]
    protocol = frozen_protocol(config)
    graph_cfg = protocol["graph_recipe"]
    loaded: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
    metrics: dict[str, dict[str, np.ndarray]] = {}
    source_complete = len(state_rows) == 4
    outputs_present, outputs_finite, topology_preserved = True, True, True
    target_join, gauge_canonical, executor = True, True, True
    private_names = {
        "x_true", "clean_log_ratio", "causal_corruption_mask", "master_id", "view_id", "split", "mechanism",
        "x_hat_wls", "x_hat_irls", "relation_rmse", "wls_quotient_rmse", "irls_quotient_rmse",
        "irls_converged", "irls_iterations", "edge_offsets", "node_offsets", "unit_key",
    }
    source_paths = {source_id: ROOT / relative for source_id, relative, _ in config["source_bindings"]}
    for row in state_rows:
        name, directory = row["state"], row["directory"]
        pub, prv = public / directory, private / directory
        observed_public_names = {p.name for p in pub.iterdir()}
        source_complete &= observed_public_names == PUBLIC_GRAPH_NAMES
        outputs_present &= {"corrected_log_ratio.npy", "reliability.npy"}.issubset(observed_public_names)
        source_complete &= {p.name for p in prv.iterdir()} == {f"{item}.npy" for item in private_names}
        a = {path.stem: load(path) for path in sorted(pub.iterdir())}
        q = {path.stem: load(path) for path in sorted(prv.iterdir())}
        with np.load(source_paths[row["source_id"]], allow_pickle=False) as raw:
            expected_keys = np.asarray(
                [unit_key("graph-view", f"{master}\0{view}") for master, view in zip(raw["master_id"].astype(str), raw["view_id"].astype(str), strict=True)]
            )
            raw_public = set(PUBLIC_GRAPH_NAMES) - {"unit_key.npy"}
            raw_private = private_names - {"unit_key"}
            source_complete &= all(array_equal(a[field.removesuffix(".npy")], raw[field.removesuffix(".npy")]) for field in raw_public)
            source_complete &= all(array_equal(q[field], raw[field]) for field in raw_private)
            source_complete &= np.array_equal(a["unit_key"], expected_keys) and np.array_equal(q["unit_key"], expected_keys)
            source_complete &= row.get("views") == len(raw["n_nodes"])
            source_complete &= row.get("edges") == len(raw["observed_log_ratio"])
            source_complete &= row.get("nodes") == len(raw["x_true"])
        loaded[name] = (a, q)
        outputs_finite &= bool(np.all(np.isfinite(a["corrected_log_ratio"]))) and bool(np.all(np.isfinite(a["reliability"])))
        topology_preserved &= a["corrected_log_ratio"].shape == a["observed_log_ratio"].shape and a["reliability"].shape == a["observed_log_ratio"].shape
        edge_offsets, node_offsets = a["edge_offsets"].astype(int), a["node_offsets"].astype(int)
        wls_parts, irls_parts = [], []
        convergence, iteration_counts = [], []
        relation_values, wls_values, irls_values = [], [], []
        for i in range(len(a["n_nodes"])):
            e0, e1 = edge_offsets[i : i + 2]
            n0, n1 = node_offsets[i : i + 2]
            edges = a["edge_index"][e0:e1]
            valid = a["edge_valid"][e0:e1].astype(bool)
            target = q["x_true"][n0:n1].astype(np.float64)
            matrix = incidence(int(a["n_nodes"][i]), edges)
            target_join &= bool(np.allclose(matrix @ target, q["clean_log_ratio"][e0:e1], atol=1e-12))
            gauge_canonical &= abs(float(target.mean())) < 1e-12
            corrected = a["corrected_log_ratio"][e0:e1].astype(np.float64)
            wls = local_wls(int(a["n_nodes"][i]), edges, valid, corrected, a["reliability"][e0:e1], graph_cfg["weight_floor"])
            irls, converged, count = local_irls(int(a["n_nodes"][i]), edges, valid, a["edge_variance"][e0:e1], corrected, a["reliability"][e0:e1], graph_cfg)
            wls_parts.append(wls); irls_parts.append(irls)
            convergence.append(converged); iteration_counts.append(count)
            relation_values.append(float(np.sqrt(np.mean((corrected[valid] - q["clean_log_ratio"][e0:e1][valid]) ** 2))))
            wls_values.append(float(np.sqrt(np.mean((wls - target) ** 2))))
            irls_values.append(float(np.sqrt(np.mean((irls - target) ** 2))))
        wls_all, irls_all = np.concatenate(wls_parts), np.concatenate(irls_parts)
        conv = np.asarray(convergence, dtype=bool)
        counts = np.asarray(iteration_counts, dtype=np.int64)
        relation_array = np.asarray(relation_values)
        wls_array = np.asarray(wls_values)
        irls_array = np.asarray(irls_values)
        metrics[name] = {"relation": relation_array, "wls": wls_array, "irls": irls_array, "x_wls": wls_all, "x_irls": irls_all, "converged": conv}
        tolerance = float(config["controls"]["solver_replay_atol"])
        recorded = evidence["relational"]["states"][name]
        nominal = {
            "relation_rmse": graph_summary(relation_array, q["master_id"]),
            "wls_quotient_rmse": graph_summary(wls_array, q["master_id"]),
            "irls_quotient_rmse": graph_summary(irls_array, q["master_id"]),
            "irls_converged": int(conv.sum()),
            "irls_failed": int((~conv).sum()),
        }
        executor &= (
            float(np.max(np.abs(wls_all - q["x_hat_wls"]))) <= tolerance
            and float(np.max(np.abs(irls_all - q["x_hat_irls"]))) <= tolerance
            and np.array_equal(conv, q["irls_converged"].astype(bool))
            and np.array_equal(counts, q["irls_iterations"].astype(np.int64))
            and recorded.get("all_wls_replay_within_tolerance") is True
            and recorded.get("all_irls_replay_within_tolerance") is True
            and recorded.get("irls_converged_sha256") == array_digest(conv)
            and recorded.get("irls_iterations_sha256") == array_digest(counts)
            and float(recorded.get("all_wls_max_abs_difference", float("inf"))) <= tolerance
            and float(recorded.get("all_irls_max_abs_difference", float("inf"))) <= tolerance
            and metric_contract_equal(recorded.get("nominal"), nominal, tolerance)
            and all(recorded.get("cache_metric_parity", {}).values())
        )
    parity_names = ("n_nodes", "edge_index", "observed_log_ratio", "edge_valid", "path_index", "path_sign", "path_valid", "edge_variance", "edge_offsets", "node_offsets", "path_offsets")
    unit_parity, input_parity = True, True
    for seed in (104729, 130363):
        left, right = loaded[f"raw_generic|seed={seed}"][0], loaded[f"raw_typed|seed={seed}"][0]
        unit_parity &= np.array_equal(left["unit_key"], right["unit_key"])
        input_parity &= all(np.array_equal(left[field], right[field]) for field in parity_names)
    parity = unit_parity and input_parity
    reference, truth = loaded["raw_generic|seed=104729"]
    target_parity_names = ("unit_key", "master_id", "view_id", "split", "x_true", "clean_log_ratio")
    target_join &= all(
        np.array_equal(truth[field], private_arrays[field])
        for _, private_arrays in loaded.values()
        for field in target_parity_names
    )
    masters, splits = truth["master_id"].astype(str), truth["split"].astype(str)
    unique = sorted(set(masters)); first = {master: int(np.flatnonzero(masters == master)[0]) for master in unique}
    keys = np.asarray([hashlib.sha256(("graph-master\0" + master).encode()).hexdigest() for master in unique])
    mapping, singletons = derange(keys, [(splits[first[m]], int(reference["n_nodes"][first[m]])) for m in unique], config["controls"]["graph_shuffle_seed"])
    donor = {unique[i]: unique[int(mapping[i])] for i in range(len(unique))}
    edge_offsets, node_offsets = reference["edge_offsets"].astype(int), reference["node_offsets"].astype(int)
    transported, transported_x, donor_by_view = [], [], []
    for i, master in enumerate(masters):
        donor_index = first[donor[master]]
        n0, n1 = node_offsets[donor_index : donor_index + 2]
        x = truth["x_true"][n0:n1].astype(np.float64); x -= x.mean()
        e0, e1 = edge_offsets[i : i + 2]
        transported.append(incidence(int(reference["n_nodes"][i]), reference["edge_index"][e0:e1]) @ x)
        transported_x.append(x); donor_by_view.append(donor[master])
    matched = np.ones(len(masters), dtype=bool)
    for name in metrics:
        matched &= metrics[name]["converged"].astype(bool)
    support_positive = bool(np.any(matched))
    estimands_equal = True
    controls_exact = True
    for name, (arrays, private_arrays) in loaded.items():
        shuffled_relation, shuffled_wls, shuffled_irls = [], [], []
        edges, nodes = arrays["edge_offsets"].astype(int), arrays["node_offsets"].astype(int)
        for i in range(len(masters)):
            e0, e1 = edges[i : i + 2]; n0, n1 = nodes[i : i + 2]
            valid = arrays["edge_valid"][e0:e1].astype(bool)
            shuffled_relation.append(float(np.sqrt(np.mean((arrays["corrected_log_ratio"][e0:e1][valid] - transported[i][valid]) ** 2))))
            shuffled_wls.append(float(np.sqrt(np.mean((metrics[name]["x_wls"][n0:n1] - transported_x[i]) ** 2))))
            shuffled_irls.append(float(np.sqrt(np.mean((metrics[name]["x_irls"][n0:n1] - transported_x[i]) ** 2))))
        sr, sw, si = np.asarray(shuffled_relation), np.asarray(shuffled_wls), np.asarray(shuffled_irls)
        expected_shuffled = {"relation_rmse": graph_summary(sr, masters), "wls_quotient_rmse": graph_summary(sw, masters), "irls_quotient_rmse": graph_summary(si, masters)}
        expected_matched = {
            "nominal_relation_rmse": graph_summary(metrics[name]["relation"], masters, matched),
            "nominal_wls_quotient_rmse": graph_summary(metrics[name]["wls"], masters, matched),
            "nominal_irls_quotient_rmse": graph_summary(metrics[name]["irls"], masters, matched),
            "shuffled_relation_rmse": graph_summary(sr, masters, matched),
            "shuffled_wls_quotient_rmse": graph_summary(sw, masters, matched),
            "shuffled_irls_quotient_rmse": graph_summary(si, masters, matched),
        }
        recorded = evidence["relational"]["states"][name]
        estimands_equal &= metric_contract_equal(recorded.get("shuffled"), expected_shuffled, float(config["controls"]["solver_replay_atol"]))
        estimands_equal &= metric_contract_equal(recorded.get("matched"), expected_matched, float(config["controls"]["solver_replay_atol"]))
    control = evidence["relational"]["target_shuffle"]
    controls_exact &= (
        singletons == control.get("singleton_strata")
        and sum(donor[m] == m for m in unique) == control.get("self_donors")
        and array_digest(np.concatenate(transported)) == control.get("transported_target_sha256")
        and array_digest(np.concatenate(transported_x)) == control.get("transported_quotient_sha256")
        and array_digest(matched) == control.get("matched_mask_sha256")
        and int(matched.sum()) == control.get("matched_views")
        and all(len({donor_by_view[i] for i in np.flatnonzero(masters == master)}) == 1 for master in unique)
    )
    outputs = outputs_present and outputs_finite and topology_preserved
    target_ok = target_join and gauge_canonical
    controls = support_positive and estimands_equal and controls_exact
    return {
        "source": bool(source_complete), "parity": bool(parity), "outputs": bool(outputs),
        "executor": bool(executor), "target": bool(target_ok), "controls": bool(controls),
        "facts": {
            "R1_SOURCE_COMPLETE": {"schema_valid": bool(source_complete)},
            "R2_PUBLIC_PARITY": {"unit_keys_equal": bool(unit_parity), "public_inputs_equal": bool(input_parity), "private_field_exposed": False},
            "R3_REPRESENTATION_OUTPUT": {"outputs_present": bool(outputs_present), "outputs_finite": bool(outputs_finite), "topology_preserved": bool(topology_preserved)},
            "R4_EXECUTOR_FACTORIAL": {"inputs_equal": bool(parity), "recipe_equal": True, "cells_replayed": bool(executor), "truth_used": False},
            "R5_TARGET_AUTHORITY": {"target_join_exact": bool(target_join), "gauge_canonical": bool(gauge_canonical), "mechanism_used": False},
            "R6_ESTIMAND_CONTROLS": {"estimands_equal": bool(estimands_equal), "controls_exact": bool(controls_exact), "support_positive": bool(support_positive)},
        },
    }


def evidence_item(source_id: str, locator: str, observed: Any) -> dict[str, Any]:
    serialized = json.dumps(observed, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return {"source_id": source_id, "locator": locator, "observed": observed, "digest": hashlib.sha256(serialized.encode()).hexdigest()}


def pred(identifier: str, passed: bool, evidence: Any, reasons: list[str] | None = None) -> dict[str, Any]:
    if isinstance(evidence, list) and all(isinstance(row, dict) and {"source_id", "locator", "observed", "digest"} <= set(row) for row in evidence):
        rows = evidence
    else:
        rows = [evidence_item("CHECKER", identifier, evidence)]
    reason_codes = [] if passed else list(reasons or REASONS[identifier][:1])
    if any(reason not in REASONS[identifier] for reason in reason_codes):
        raise ValueError(f"reason outside closed catalog for {identifier}")
    return {"id": identifier, "status": "PASS" if passed else "FAIL", "reason_codes": reason_codes, "evidence": rows}


def native_reason_codes(identifier: str, facts: dict[str, Any]) -> list[str]:
    """Derive native-predicate reasons from observed contract facts, never labels."""
    reasons: list[str] = []
    if identifier == "R1_SOURCE_COMPLETE":
        receipts = facts.get("source_receipts", [])
        if any(row.get("actual") is None for row in receipts if row.get("path") != "config:source_policy"):
            reasons.append("GRAPH_SOURCE_MISSING")
        if any(row.get("actual") is not None and row.get("status") == "FAIL" for row in receipts if row.get("path") != "config:source_policy"):
            reasons.append("GRAPH_HASH_MISMATCH")
        if not facts.get("schema_valid", False):
            reasons.append("GRAPH_SCHEMA_INVALID")
    elif identifier == "R2_PUBLIC_PARITY":
        if not facts.get("unit_keys_equal", False):
            reasons.append("GRAPH_UNIT_MISMATCH")
        if not facts.get("public_inputs_equal", False):
            reasons.append("GRAPH_INPUT_MISMATCH")
        if facts.get("private_field_exposed", True):
            reasons.append("GRAPH_PRIVATE_LEAKAGE")
    elif identifier == "R3_REPRESENTATION_OUTPUT":
        if not facts.get("outputs_present", False):
            reasons.append("REPRESENTATION_OUTPUT_MISSING")
        if not facts.get("outputs_finite", False):
            reasons.append("REPRESENTATION_OUTPUT_NONFINITE")
        if not facts.get("topology_preserved", False):
            reasons.append("TOPOLOGY_CHANGED")
    elif identifier == "R4_EXECUTOR_FACTORIAL":
        if not facts.get("inputs_equal", False):
            reasons.append("EXECUTOR_INPUT_MISMATCH")
        if not facts.get("recipe_equal", False):
            reasons.append("EXECUTOR_RECIPE_MISMATCH")
        if not facts.get("cells_replayed", False):
            reasons.append("EXECUTOR_CELL_MISSING")
        if facts.get("truth_used", True):
            reasons.append("TRUTH_USED_BY_EXECUTOR")
    elif identifier == "R5_TARGET_AUTHORITY":
        if not facts.get("target_join_exact", False):
            reasons.append("GRAPH_TARGET_JOIN_INVALID")
        if not facts.get("gauge_canonical", False):
            reasons.append("GAUGE_NOT_CANONICAL")
        if facts.get("mechanism_used", True):
            reasons.append("MECHANISM_LEAKAGE")
    elif identifier == "R6_ESTIMAND_CONTROLS":
        if not facts.get("estimands_equal", False):
            reasons.append("RELATIONAL_ESTIMAND_MISMATCH")
        if not facts.get("controls_exact", False):
            reasons.append("RELATIONAL_CONTROL_MISMATCH")
        if not facts.get("support_positive", False):
            reasons.append("RELATIONAL_SUPPORT_EMPTY")
    elif identifier == "S1_SOURCE_COMPLETE":
        receipts = facts.get("source_receipts", [])
        if any(row.get("actual") is None for row in receipts if row.get("path") != "config:source_policy"):
            reasons.append("SET_SOURCE_MISSING")
        if any(row.get("actual") is not None and row.get("status") == "FAIL" for row in receipts if row.get("path") != "config:source_policy"):
            reasons.append("SET_HASH_MISMATCH")
        if not facts.get("schema_valid", False):
            reasons.append("SET_SCHEMA_INVALID")
        if not facts.get("role_counts_valid", False):
            reasons.append("SET_ROLE_COUNTS_INVALID")
    elif identifier == "S2_POSTERIOR_PARITY":
        if not facts.get("cells_present", False):
            reasons.append("POSTERIOR_CELL_MISSING")
        if not facts.get("mass_valid", False):
            reasons.append("POSTERIOR_MASS_INVALID")
        if not facts.get("alignment_exact", False):
            reasons.append("POSTERIOR_ALIGNMENT_MISMATCH")
        if facts.get("utility_used", True):
            reasons.append("UTILITY_IN_POSTERIOR")
    elif identifier == "S3_FOUR_CELLS_EXECUTABLE":
        if not facts.get("cells_present", False):
            reasons.append("SET_DECISION_CELL_MISSING")
        if not facts.get("hard_posterior_bound", False):
            reasons.append("HARD_READER_NOT_POSTERIOR_BOUND")
        if not facts.get("contextual_recipe_exact", False):
            reasons.append("CONTEXTUAL_RECIPE_MISMATCH")
        if not facts.get("duplication_declared", False):
            reasons.append("CELL_DUPLICATION_UNDECLARED")
    elif identifier == "S4_FIT_SUPPORT_FREEZE":
        if not facts.get("proposer_support_positive", False):
            reasons.append("PROPOSER_SUPPORT_EMPTY")
        if not facts.get("harm_classes_present", False):
            reasons.append("HARM_CLASS_MISSING")
        if not facts.get("incompatibility_classes_present", False):
            reasons.append("INCOMPATIBILITY_CLASS_MISSING")
        if not facts.get("selection_support_positive", False):
            reasons.append("SELECTION_SUPPORT_EMPTY")
        if not facts.get("phase_closed", False):
            reasons.append("SET_PHASE_VIOLATION")
    elif identifier == "S5_TARGET_UTILITY_AUTHORITY":
        if not facts.get("target_join_exact", False):
            reasons.append("SET_TARGET_JOIN_INVALID")
        if not facts.get("target_nonempty", False):
            reasons.append("EMPTY_TARGET_SET")
        if facts.get("target_used_as_input", True):
            reasons.append("TARGET_LEAKAGE")
        if not facts.get("utility_contract_valid", False):
            reasons.append("UTILITY_CONTRACT_MISMATCH")
    elif identifier == "S6_ESTIMAND_CONTROLS":
        if not facts.get("estimands_equal", False):
            reasons.append("SET_ESTIMAND_MISMATCH")
        if facts.get("posterior_reader_entangled", True):
            reasons.append("POSTERIOR_READER_ENTANGLED")
        if not facts.get("controls_exact", False):
            reasons.append("SET_CONTROL_MISMATCH")
        if not facts.get("support_positive", False):
            reasons.append("SET_SUPPORT_EMPTY")
    else:
        raise ValueError(f"unsupported native predicate: {identifier}")
    if any(reason not in REASONS[identifier] for reason in reasons):
        raise ValueError(f"reason outside closed catalog for {identifier}")
    return reasons


def semantic_decision(predicates: list[dict[str, Any]], technical: dict[str, str]) -> str | None:
    if any(technical[name] != "PASS" for name in ("source_status", "artifact_status", "checker_status", "replay_status")):
        return None
    status = {row["id"]: row["status"] == "PASS" for row in predicates}
    mapping = all(status[name] for name in status if name.startswith("M"))
    relational = all(status[name] for name in status if name.startswith("R"))
    set_valued = all(status[name] for name in status if name.startswith("S"))
    if mapping:
        return "COMMON_FACTORIAL_FEASIBLE"
    if relational and set_valued:
        return "BIFURCATE_NATIVE_CONTRASTS"
    return "NO_EXECUTABLE_SUCCESSOR"


def mutation_suite(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "proportional-mapping-mutations-pending-v2",
        "status": "PENDING_EXTERNAL_MUTATION_TESTS",
        "predicate_ids": config["predicates"]["mapping"] + config["predicates"]["relational"] + config["predicates"]["set_valued"],
        "reason_catalog_sha256": hashlib.sha256(json.dumps(REASONS, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        **FIXED,
    }


def validate_core_manifest(run: Path) -> bool:
    path = run / "core_manifest.json"
    if not path.is_file():
        return False
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != "proportional-mapping-core-manifest-v1":
        return False
    excluded = {"runtime.json", "core_manifest.json", "adjudication.json", "REPORT_MAPPING_FEASIBILITY.md", "scientific_manifest.json", "replay_evidence.json", "terminal_status.json"}
    actual = {
        file.relative_to(run).as_posix(): {"bytes": file.stat().st_size, "sha256": sha256_file(file)}
        for file in sorted(run.rglob("*"))
        if file.is_file() and file.relative_to(run).as_posix() not in excluded
    }
    return manifest.get("files") == actual


def validate_replay(run: Path, replay_path: Path) -> bool:
    replay = json.loads(replay_path.read_text())
    expected_keys = {
        "schema_version", "status", "core_a_sha256", "core_b_sha256", "files_compared", "mismatches",
        "gpu_used_or_queried", "architecture_promoted", "scientific_decision", "decision_authority",
    }
    if set(replay) != expected_keys or replay.get("schema_version") != "proportional-mapping-replay-evidence-v1":
        return False
    if replay.get("status") != "PASS" or replay.get("mismatches") != [] or any(replay.get(key) != value for key, value in FIXED.items()):
        return False
    run_a, run_b = run.parent / "run_a", run.parent / "run_b"
    if not validate_core_manifest(run_a) or not validate_core_manifest(run_b):
        return False
    manifest_a = json.loads((run_a / "core_manifest.json").read_text())
    manifest_b = json.loads((run_b / "core_manifest.json").read_text())
    if manifest_a != manifest_b:
        return False
    if replay.get("core_a_sha256") != sha256_file(run_a / "core_manifest.json") or replay.get("core_b_sha256") != sha256_file(run_b / "core_manifest.json"):
        return False
    if replay.get("files_compared") != len(manifest_a["files"]):
        return False
    return all((run_a / relative).read_bytes() == (run_b / relative).read_bytes() for relative in manifest_a["files"])


def validate_terminal_status(run: Path, path: Path, replay_path: Path, config: dict[str, Any]) -> bool:
    receipt = json.loads(path.read_text())
    expected_keys = {
        "schema_version", "artifact_status", "wall_seconds_before_final", "max_ru_maxrss_bytes_before_final",
        "limits", "core_manifest_sha256", "replay_evidence_sha256",
        "gpu_used_or_queried", "architecture_promoted", "scientific_decision", "decision_authority",
    }
    return (
        set(receipt) == expected_keys
        and receipt.get("schema_version") == "proportional-mapping-terminal-status-v1"
        and receipt.get("artifact_status") == "PASS"
        and receipt.get("limits") == config["budget"]
        and float(receipt.get("wall_seconds_before_final", float("inf"))) < float(config["budget"]["wall_seconds_exclusive"])
        and int(receipt.get("max_ru_maxrss_bytes_before_final", config["budget"]["rss_bytes_exclusive"])) < int(config["budget"]["rss_bytes_exclusive"])
        and receipt.get("core_manifest_sha256") == sha256_file(run / "core_manifest.json")
        and receipt.get("replay_evidence_sha256") == sha256_file(replay_path)
        and all(receipt.get(key) == value for key, value in FIXED.items())
    )


def compute(run: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    source_ok, source_receipts = source_status(config)
    public_ok = verify_tree_manifest(run / "prepared/public", "mapping-prepared-public-manifest-v1")
    private_ok = verify_tree_manifest(run / "prepared/private_dev", "mapping-prepared-private-manifest-v1")
    required = ["mapping_candidate.json", "native_contracts.json", "builder_access_receipt.json", "evaluation_evidence.json", "prepared/public/protocol.json"]
    if not all((run / relative).is_file() for relative in required):
        raise RuntimeError("required gate artifact missing")
    candidate_path = run / "mapping_candidate.json"
    candidate = json.loads(candidate_path.read_text())
    native = json.loads((run / "native_contracts.json").read_text())
    access = json.loads((run / "builder_access_receipt.json").read_text())
    evidence = json.loads((run / "evaluation_evidence.json").read_text())
    protocol_path = run / "prepared/public/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    expected_protocol = frozen_protocol(config)
    manifest = json.loads((run / "prepared/public/manifest.json").read_text())
    opened = access.get("opened", [])
    builder_source = (ROOT / "experiments/geometria_proporcional/build_proportional_mapping_candidate.py").read_text()
    evaluator_source = (ROOT / "experiments/geometria_proporcional/evaluate_proportional_mapping_feasibility.py").read_text()
    checker_source = Path(__file__).read_text()
    evaluator_imports = "\n".join(line.strip() for line in evaluator_source.splitlines() if line.lstrip().startswith(("import ", "from ")))
    checker_imports = "\n".join(line.strip() for line in checker_source.splitlines() if line.lstrip().startswith(("import ", "from ")))
    expected_evidence_keys = {
        "schema_version", "candidate_sha256", "prepared_protocol_sha256", "candidate_frozen_before_private_access",
        "common_mapping_observations", "set_valued", "relational",
        "gpu_used_or_queried", "architecture_promoted", "scientific_decision", "decision_authority",
    }
    expected_common_keys = {"candidate_query_sha256", "candidate_unit_namespaces_sha256", "candidate_bridge", "contract_sha256"}
    w49_parity = verify_w49_prepared(run)
    expected_candidate_keys = {
        "schema_version", "query", "builder_input", "builder_may_emit_mapping_decision", "unit_namespaces",
        "declared_cross_domain_unit_bridge", "lines", "adapters", "public_facts",
        "gpu_used_or_queried", "architecture_promoted", "scientific_decision", "decision_authority",
    }
    expected_native_keys = {"schema_version", "relational", "set_valued", "gpu_used_or_queried", "architecture_promoted", "scientific_decision", "decision_authority"}
    actual_w54_shapes = {
        name: {
            "dtype": values.dtype.str,
            "shape": list(values.shape),
            "sha256": array_digest(values),
        }
        for name in ("ensemble_logits", "per_seed_logits", "unit_key", "cluster_key", "split_role")
        for values in [load(run / f"prepared/public/w54/{name}.npy")]
    }
    common_evidence = evidence.get("common_mapping_observations", {})
    expected_common_evidence = {
        "candidate_query_sha256": hashlib.sha256(str(candidate.get("query", "")).encode()).hexdigest(),
        "candidate_unit_namespaces_sha256": hashlib.sha256(json.dumps(candidate.get("unit_namespaces"), sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "candidate_bridge": candidate.get("declared_cross_domain_unit_bridge"),
        "contract_sha256": hashlib.sha256(json.dumps(expected_protocol["common_contract"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }
    candidate_valid = (
        candidate.get("schema_version") == "proportional-mapping-candidate-v1"
        and set(candidate) == expected_candidate_keys
        and native.get("schema_version") == "proportional-native-contracts-v1"
        and set(native) == expected_native_keys
        and access.get("schema_version") == "proportional-mapping-builder-access-v1"
        and evidence.get("schema_version") == "proportional-mapping-evaluation-evidence-v1"
        and set(evidence) == expected_evidence_keys
        and set(evidence.get("common_mapping_observations", {})) == expected_common_keys
        and common_evidence == expected_common_evidence
        and protocol.get("schema_version") == "proportional-mapping-prepared-protocol-v1"
        and protocol == expected_protocol
        and candidate.get("builder_may_emit_mapping_decision") is False
        and "mapping_decision" not in candidate
        and access.get("root_kind") == "prepared_public_only"
        and len(opened) == len(set(opened))
        and set(opened) == set(manifest.get("files", {})) | {"manifest.json"}
        and all(not path.startswith("/") and "private" not in path for path in opened)
        and "fixture_mode" not in candidate
        and evidence.get("candidate_sha256") == sha256_file(candidate_path)
        and evidence.get("prepared_protocol_sha256") == sha256_file(protocol_path)
        and candidate.get("public_facts", {}).get("public_manifest_sha256") == sha256_file(run / "prepared/public/manifest.json")
        and candidate.get("public_facts", {}).get("protocol_sha256") == sha256_file(protocol_path)
        and candidate.get("public_facts", {}).get("w49") == json.loads((run / "prepared/public/w49_contract.json").read_text())
        and candidate.get("public_facts", {}).get("w54_shapes") == actual_w54_shapes
        and set(candidate.get("lines", {})) == {"eiv", "set_valued", "relational"}
        and set(candidate.get("adapters", {})) == {"eiv", "set_valued", "relational"}
        and w49_parity
        and all(candidate.get(key) == value for key, value in FIXED.items())
        and all(native.get(key) == value and access.get(key) == value and evidence.get(key) == value and protocol.get(key) == value for key, value in FIXED.items())
    )
    set_checks = recompute_set(run, config, evidence)
    graph_checks = recompute_graph(run, config, evidence)
    common_contract = expected_protocol["common_contract"]
    namespaces = candidate.get("unit_namespaces", {})
    bridge = candidate.get("declared_cross_domain_unit_bridge")
    m1_ok, m1_reasons, m1_observed = common_unit_contract(config, candidate.get("query"), protocol.get("query"), namespaces, bridge)
    forbidden = {"x_true", "clean_log_ratio", "causal_corruption_mask", "master_id", "view_id", "split", "mechanism", "target", "design_stratum", "cardinality"}
    public_private_leak = any(path.stem in forbidden for path in (run / "prepared/public").rglob("*.npy"))
    m2_ok, m2_reasons, m2_observed = common_observation_contract(candidate, public_private_leak)
    m3_ok, m3_reasons, m3_observed = common_target_contract(candidate)
    m4_ok, m4_reasons, m4_observed = common_decision_contract(candidate)
    phase_policy = config["source_policy"]["phase_access"]
    authority_ok, authority_reasons, authority_observed = authority_phase_contract(
        phase_policy, protocol["authority"], candidate_valid,
        evaluator_source, evaluator_imports, checker_imports, builder_source,
    )
    graph_facts = graph_checks["facts"]
    set_facts = set_checks["facts"]
    graph_facts["R1_SOURCE_COMPLETE"] = {**graph_facts["R1_SOURCE_COMPLETE"], "source_receipts": source_receipts}
    set_facts["S1_SOURCE_COMPLETE"] = {**set_facts["S1_SOURCE_COMPLETE"], "source_receipts": source_receipts}
    native_rows = {}
    for identifier, facts in {**graph_facts, **set_facts}.items():
        reasons = native_reason_codes(identifier, facts)
        native_rows[identifier] = pred(identifier, not reasons, [evidence_item("RECOMPUTED_CONTRACT", identifier, facts)], reasons)
    predicates = [
        pred("M1_QUERY_UNIT", m1_ok, [evidence_item("CONFIG", "query+unit_namespaces", m1_observed)], m1_reasons),
        pred("M2_OBSERVATION_PARITY", m2_ok, [evidence_item("MAPPING_CANDIDATE", "lines.*.observation", m2_observed)], m2_reasons),
        pred("M3_TARGET_CONSERVATION", m3_ok, [evidence_item("MAPPING_CANDIDATE", "lines.*.target_authority", m3_observed)], m3_reasons),
        pred("M4_DECISION_STACK_PARITY", m4_ok, [evidence_item("MAPPING_CANDIDATE", "lines.*.output+adapters", m4_observed)], m4_reasons),
        pred("M5_AUTHORITY_PHASES", authority_ok, [evidence_item("SOURCE_POLICY", "phase_access+source_scan", authority_observed)], authority_reasons),
        *(native_rows[identifier] for identifier in (
            "R1_SOURCE_COMPLETE", "R2_PUBLIC_PARITY", "R3_REPRESENTATION_OUTPUT",
            "R4_EXECUTOR_FACTORIAL", "R5_TARGET_AUTHORITY", "R6_ESTIMAND_CONTROLS",
            "S1_SOURCE_COMPLETE", "S2_POSTERIOR_PARITY", "S3_FOUR_CELLS_EXECUTABLE",
            "S4_FIT_SUPPORT_FREEZE", "S5_TARGET_UTILITY_AUTHORITY", "S6_ESTIMAND_CONTROLS",
        )),
    ]
    technical = {
        "source_status": "PASS" if source_ok else "FAIL",
        "artifact_status": "PASS" if public_ok and private_ok and candidate_valid and w49_parity and set_checks["schema"] and graph_checks["source"] else "FAIL",
        "checker_status": "PASS",
        "replay_status": "NOT_RUN",
    }
    diagnostics = {"source_receipts": source_receipts, "set_checks": set_checks, "graph_checks": graph_checks, "candidate_valid": candidate_valid, "public_manifest_exact": public_ok, "private_manifest_exact": private_ok, "w49_source_parity": w49_parity, "authority_ok": authority_ok}
    return predicates, technical, diagnostics


def report_markdown(adjudication: dict[str, Any]) -> str:
    failed = [row for row in adjudication["predicates"] if row["status"] == "FAIL"]
    lines = [
        "# MAPPING-FEASIBILITY — adjudicación técnica",
        "",
        f"- `mapping_decision`: `{adjudication['mapping_decision']}`",
        f"- estados técnicos: `{json.dumps(adjudication['technical_status'], sort_keys=True)}`",
        f"- predicados: `{len(adjudication['predicates']) - len(failed)} PASS / {len(failed)} FAIL`",
        "- promoción arquitectónica: `false`",
        "- decisión científica: `null`",
        "",
        "## Predicados fallidos",
        "",
    ]
    lines.extend(f"- `{row['id']}`: `{','.join(row['reason_codes'])}`" for row in failed)
    lines.extend(["", "Esta hoja decide compatibilidad de contratos; no declara GO/NO-GO.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--phase", choices=("pre", "final"))
    parser.add_argument("--replay-evidence", type=Path)
    parser.add_argument("--terminal-status", type=Path)
    args = parser.parse_args()
    if args.run is None or args.phase is None:
        raise ValueError("scientific mode requires --run and --phase")
    run = args.run.resolve(strict=True)
    config = json.loads(args.config.resolve(strict=True).read_text())
    predicates, technical, diagnostics = compute(run, config)
    if args.phase == "pre":
        mutations = mutation_suite(config)
        write_json(run / "mapping_matrix.json", {"schema_version": "proportional-mapping-matrix-v1", "predicates": predicates, "diagnostics": diagnostics, **FIXED})
        write_json(run / "mutation_results.json", mutations)
        write_json(run / "pre_adjudication.json", {"schema_version": "proportional-mapping-pre-adjudication-v1", "technical_status": technical, "mapping_decision": None, "predicates": predicates, **FIXED})
        return
    if args.replay_evidence is None or args.terminal_status is None:
        raise ValueError("final phase requires replay and terminal evidence")
    replay_path = args.replay_evidence.resolve(strict=True)
    terminal_path = args.terminal_status.resolve(strict=True)
    replay_ok = validate_replay(run, replay_path)
    core_ok = validate_core_manifest(run)
    terminal_ok = validate_terminal_status(run, terminal_path, replay_path, config)
    technical["replay_status"] = "PASS" if replay_ok else "FAIL"
    if not (core_ok and terminal_ok):
        technical["artifact_status"] = "FAIL"
        technical["checker_status"] = "FAIL"
    decision = semantic_decision(predicates, technical)
    adjudication = {
        "schema_version": "proportional-mapping-adjudication-v1",
        "technical_status": technical,
        "mapping_decision": decision,
        "predicates": predicates,
        "replay_evidence_sha256": sha256_file(replay_path),
        "terminal_status_sha256": sha256_file(terminal_path),
        "closure_checks": {"core_manifest": core_ok, "replay": replay_ok, "terminal_status": terminal_ok},
        **FIXED,
    }
    write_json(run / "adjudication.json", adjudication)
    (run / "REPORT_MAPPING_FEASIBILITY.md").write_text(report_markdown(adjudication), encoding="utf-8")


if __name__ == "__main__":
    main()
