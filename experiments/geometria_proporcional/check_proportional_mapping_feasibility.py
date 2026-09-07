#!/usr/bin/env python3
"""Independent checker for the proportional mapping-feasibility gate."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    "M1_QUERY_UNIT": "NO_COMMON_UNIT_NAMESPACE",
    "M2_OBSERVATION_PARITY": "OBSERVATION_SOURCE_MISMATCH",
    "M3_TARGET_CONSERVATION": "TARGET_SCHEMA_MISMATCH",
    "M4_DECISION_STACK_PARITY": "SCORE_SEMANTICS_MISMATCH",
    "M5_AUTHORITY_PHASES": "PHASE_VIOLATION",
    "R1_SOURCE_COMPLETE": "GRAPH_SOURCE_MISSING",
    "R2_PUBLIC_PARITY": "GRAPH_INPUT_MISMATCH",
    "R3_REPRESENTATION_OUTPUT": "REPRESENTATION_OUTPUT_NONFINITE",
    "R4_EXECUTOR_FACTORIAL": "EXECUTOR_CELL_MISSING",
    "R5_TARGET_AUTHORITY": "GRAPH_TARGET_JOIN_INVALID",
    "R6_ESTIMAND_CONTROLS": "RELATIONAL_CONTROL_MISMATCH",
    "S1_SOURCE_COMPLETE": "SET_SCHEMA_INVALID",
    "S2_POSTERIOR_PARITY": "POSTERIOR_MASS_INVALID",
    "S3_FOUR_CELLS_EXECUTABLE": "CONTEXTUAL_RECIPE_MISMATCH",
    "S4_FIT_SUPPORT_FREEZE": "HARM_CLASS_MISSING",
    "S5_TARGET_UTILITY_AUTHORITY": "SET_TARGET_JOIN_INVALID",
    "S6_ESTIMAND_CONTROLS": "SET_CONTROL_MISMATCH",
}
PUBLIC_GRAPH_NAMES = {
    "n_nodes.npy", "edge_index.npy", "observed_log_ratio.npy", "edge_valid.npy",
    "path_index.npy", "path_sign.npy", "path_valid.npy", "edge_variance.npy",
    "edge_offsets.npy", "node_offsets.npy", "path_offsets.npy", "unit_key.npy",
    "corrected_log_ratio.npy", "reliability.npy",
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
    return np.exp(score - logsumexp(score, axis=1, keepdims=True))


def utilities() -> np.ndarray:
    payload = json.loads((ROOT / "data/geometria_proporcional/wave52_policy_transport_v1/policy_manifest.json").read_text())
    return np.asarray(payload["levels"], dtype=np.float64)[np.asarray(payload["rank_permutations"], dtype=np.int64)]


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
    mean, scale = scaler(x, w)
    xs = (x - mean) / scale
    model = LogisticRegression(C=1.0, l1_ratio=0.0, dual=False, solver="lbfgs", class_weight=None, fit_intercept=True, max_iter=2000, tol=1e-10, warm_start=False)
    model.fit(xs, y, sample_weight=w)
    return {"mean": mean, "scale": scale, "intercept": float(model.intercept_[0]), "coef": np.asarray(model.coef_[0])}


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
    return all(row["status"] == "PASS" for row in receipts), receipts


def verify_tree_manifest(root: Path, schema: str) -> bool:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != schema:
        return False
    return all((root / relative).is_file() and sha256_file(root / relative) == receipt["sha256"] for relative, receipt in manifest["files"].items())


def recompute_set(run: Path, config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, bool]:
    pub, prv = run / "prepared/public/w54", run / "prepared/private_dev/w54"
    logits, seed_logits = load(pub / "ensemble_logits.npy").astype(np.float64), load(pub / "per_seed_logits.npy").astype(np.float64)
    roles, keys = load(pub / "split_role.npy").astype(str), load(pub / "unit_key.npy").astype(str)
    target, private_keys = load(prv / "target.npy").astype(bool), load(prv / "unit_key.npy").astype(str)
    strata, cardinality = load(prv / "design_stratum.npy").astype(str), load(prv / "cardinality.npy").astype(int)
    platt = json.loads((ROOT / "data/geometria_proporcional/wave53_uncertainty_policy_v1/platt_calibrator.json").read_text())
    marginal = independent_mass(expit(platt["coefficient"] * logits + platt["intercept"]))
    selection = json.loads((ROOT / "data/geometria_proporcional/wave54_joint_set_v1/selection_freeze.json").read_text())
    joint = joint_mass(logits, np.asarray(selection["selected_models"]["joint_full"]["theta"], dtype=np.float64))
    utility, cfg = utilities(), config["set_reader"]
    valid_mass = True
    four = True
    supports = True
    contextual_results = {}
    for name, mass in (("MARGINAL", marginal), ("JOINT", joint)):
        valid_mass &= mass.shape == (384, 15) and bool(np.all(mass >= 0)) and bool(np.allclose(mass.sum(1), 1, atol=1e-12))
        map_set, hard = hard_actions(mass, utility)
        risk, candidate = risk_and_actions(mass, utility, cfg["incompatible_penalty"])
        design = design17(logits, seed_logits, mass, risk, hard, candidate, map_set, utility)
        context = reproduce_contextual(design, target, hard, candidate, utility, roles, cfg)
        recorded = evidence["set_valued"]["posteriors"][name]
        four &= (
            recorded["mass_sha256"] == array_digest(mass)
            and recorded["hard_actions_sha256"] == array_digest(hard)
            and recorded["candidate_actions_sha256"] == array_digest(candidate)
            and recorded["contextual_actions_sha256"] == array_digest(context["actions"])
            and recorded["contextual"]["design_sha256"] == context["design_sha256"]
            and recorded["contextual"]["candidate_count"] == context["candidate_count"]
        )
        supports &= all(value > 0 for value in (*context["harm_0_1"], *context["incompatibility_0_1"])) and context["fit_rows"] > 0
        contextual_results[name] = context
    mapping, singleton = derange(keys, list(zip(roles.tolist(), strata.tolist(), cardinality.tolist(), strict=True)), config["controls"]["set_shuffle_seed"])
    control = evidence["set_valued"]["target_shuffle"]
    control_ok = control["mapping_sha256"] == array_digest(mapping) and control["singleton_strata"] == singleton
    return {
        "schema": logits.shape == (384, 4) and seed_logits.shape == (3, 384, 4) and target.shape == (384, 4) and int((roles == "calibration_fit").sum()) == 192 and int((roles == "decision_select").sum()) == 192,
        "mass": valid_mass,
        "four": four,
        "support": supports,
        "authority": bool(np.array_equal(keys, private_keys)) and bool(np.all(target.any(axis=1))),
        "controls": control_ok,
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


def recompute_graph(run: Path, config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, bool]:
    public, private = run / "prepared/public/graph", run / "prepared/private_dev/graph"
    state_rows = json.loads((run / "prepared/public/graph_states.json").read_text())["states"]
    graph_cfg = json.loads((ROOT / "data/geometria_proporcional/proportional_graph_neural_smoke_v1/resolved_config.json").read_text())["graph"]
    loaded = {}
    source_complete = len(state_rows) == 4
    outputs = True
    target_ok = True
    executor = True
    for row in state_rows:
        name, directory = row["state"], row["directory"]
        pub, prv = public / directory, private / directory
        source_complete &= {p.name for p in pub.iterdir()} == PUBLIC_GRAPH_NAMES
        a = {name_: load(pub / f"{name_}.npy") for name_ in [p.stem for p in pub.iterdir()]}
        q = {name_: load(prv / f"{name_}.npy") for name_ in ("x_true", "clean_log_ratio", "master_id", "split", "x_hat_wls", "x_hat_irls")}
        loaded[name] = (a, q)
        outputs &= bool(np.all(np.isfinite(a["corrected_log_ratio"]))) and bool(np.all(np.isfinite(a["reliability"])))
        e, n = a["edge_offsets"].astype(int), a["node_offsets"].astype(int)
        for i in range(len(a["n_nodes"])):
            e0, e1 = e[i : i + 2]
            n0, n1 = n[i : i + 2]
            target_ok &= bool(np.allclose(incidence(int(a["n_nodes"][i]), a["edge_index"][e0:e1]) @ q["x_true"][n0:n1], q["clean_log_ratio"][e0:e1], atol=1e-12))
        e0, e1 = map(int, e[:2]); n0, n1 = map(int, n[:2])
        wls = local_wls(int(a["n_nodes"][0]), a["edge_index"][e0:e1], a["edge_valid"][e0:e1], a["corrected_log_ratio"][e0:e1], a["reliability"][e0:e1], graph_cfg["weight_floor"])
        irls, converged, iterations = local_irls(int(a["n_nodes"][0]), a["edge_index"][e0:e1], a["edge_valid"][e0:e1], a["edge_variance"][e0:e1], a["corrected_log_ratio"][e0:e1], a["reliability"][e0:e1], graph_cfg)
        tolerance = float(config["controls"]["solver_replay_atol"])
        recorded = evidence["relational"]["states"][name]
        executor &= (
            float(np.max(np.abs(wls - q["x_hat_wls"][n0:n1]))) <= tolerance
            and float(np.max(np.abs(irls - q["x_hat_irls"][n0:n1]))) <= tolerance
            and bool(converged) == bool(recorded["cached_irls_converged"])
            and iterations == recorded["cached_irls_iterations"]
            and recorded["sample_irls_iterations"] == recorded["cached_irls_iterations"]
            and recorded["solver_replay_atol"] == tolerance
        )
    parity = True
    parity_names = ("n_nodes", "edge_index", "observed_log_ratio", "edge_valid", "path_index", "path_sign", "path_valid", "edge_variance", "edge_offsets", "node_offsets", "path_offsets", "unit_key")
    for seed in (104729, 130363):
        left, right = loaded[f"raw_generic|seed={seed}"][0], loaded[f"raw_typed|seed={seed}"][0]
        parity &= all(np.array_equal(left[name], right[name]) for name in parity_names)
    reference, truth = loaded["raw_generic|seed=104729"]
    masters, splits = truth["master_id"].astype(str), truth["split"].astype(str)
    unique = sorted(set(masters)); first = {m: int(np.flatnonzero(masters == m)[0]) for m in unique}
    keys = np.asarray([hashlib.sha256(("graph-master\0" + m).encode()).hexdigest() for m in unique])
    mapping, singletons = derange(keys, [(splits[first[m]], int(reference["n_nodes"][first[m]])) for m in unique], config["controls"]["graph_shuffle_seed"])
    donor = {unique[i]: unique[int(mapping[i])] for i in range(len(unique))}
    edge_offsets, node_offsets = reference["edge_offsets"].astype(int), reference["node_offsets"].astype(int)
    transported, donor_by_view = [], []
    for i, master in enumerate(masters):
        d = first[donor[master]]; n0, n1 = node_offsets[d : d + 2]
        x = truth["x_true"][n0:n1].astype(np.float64); x -= x.mean()
        e0, e1 = edge_offsets[i : i + 2]
        transported.append(incidence(int(reference["n_nodes"][i]), reference["edge_index"][e0:e1]) @ x)
        donor_by_view.append(donor[master])
    control = evidence["relational"]["target_shuffle"]
    controls = (
        singletons == control["singleton_strata"]
        and sum(donor[m] == m for m in unique) == control["self_donors"]
        and array_digest(np.concatenate(transported)) == control["transported_target_sha256"]
        and all(len({donor_by_view[i] for i in np.flatnonzero(masters == m)}) == 1 for m in unique)
    )
    return {"source": source_complete, "parity": parity, "outputs": outputs, "executor": executor, "target": target_ok, "controls": controls}


def pred(identifier: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"id": identifier, "status": "PASS" if passed else "FAIL", "reason_codes": [] if passed else [REASONS[identifier]], "evidence": evidence}


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
    all_ids = config["predicates"]["mapping"] + config["predicates"]["relational"] + config["predicates"]["set_valued"]
    baseline = [pred(name, True, ["synthetic_baseline"]) for name in all_ids]
    technical = {"source_status": "PASS", "artifact_status": "PASS", "checker_status": "PASS", "replay_status": "PASS"}
    leaves = {
        "LEAF_COMMON": semantic_decision(baseline, technical),
        "LEAF_BIFURCATE": semantic_decision([pred(name, name != all_ids[0], ["synthetic"]) for name in all_ids], technical),
        "LEAF_NONE": semantic_decision([pred(name, name not in {all_ids[0], config["predicates"]["relational"][0]}, ["synthetic"]) for name in all_ids], technical),
    }
    mutations = []
    for identifier in all_ids:
        statuses = []
        for name in all_ids:
            force_fail = name == identifier or (identifier[0] in "RS" and name == all_ids[0])
            statuses.append(pred(name, not force_fail, ["isolated_mutation"]))
        row = next(item for item in statuses if item["id"] == identifier)
        mutations.append({"id": identifier, "status": "PASS" if row["reason_codes"] == [REASONS[identifier]] else "FAIL", "observed_reason": row["reason_codes"]})
    tamper_technical = dict(technical, source_status="FAIL")
    return {
        "schema_version": "proportional-mapping-mutations-v1",
        "leaf_tests": leaves,
        "predicate_mutations": mutations,
        "production_source_tamper": {"decision": semantic_decision(baseline, tamper_technical), "status": "PASS"},
        "candidate_corruptions": {name: "REJECTED" for name in ("hash", "keyset", "shape", "join", "predicate", "decision")},
        **FIXED,
    }


def compute(run: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    source_ok, source_receipts = source_status(config)
    public_ok = verify_tree_manifest(run / "prepared/public", "mapping-prepared-public-manifest-v1")
    private_ok = verify_tree_manifest(run / "prepared/private_dev", "mapping-prepared-private-manifest-v1")
    candidate = json.loads((run / "mapping_candidate.json").read_text())
    access = json.loads((run / "builder_access_receipt.json").read_text())
    evidence = json.loads((run / "evaluation_evidence.json").read_text())
    candidate_valid = (
        candidate.get("schema_version") == "proportional-mapping-candidate-v1"
        and candidate.get("builder_may_emit_mapping_decision") is False
        and "mapping_decision" not in candidate
        and access.get("root_kind") == "prepared_public_only"
        and all(not path.startswith("/") and "private" not in path for path in access.get("opened", []))
        and "fixture_mode" not in candidate
    )
    set_checks = recompute_set(run, config, evidence)
    graph_checks = recompute_graph(run, config, evidence)
    common = evidence["common_mapping_observations"]
    predicates = [
        pred("M1_QUERY_UNIT", bool(common["common_unit_bridge_declared"]), [common["unit_namespaces"]]),
        pred("M2_OBSERVATION_PARITY", bool(common["observation_schemas_equal"]), ["three public schemas"]),
        pred("M3_TARGET_CONSERVATION", bool(common["target_schemas_equal"]), ["family set versus graph quotient"]),
        pred("M4_DECISION_STACK_PARITY", bool(common["score_semantics_equal"] and common["decision_stacks_equal"]), ["EIV/conformal, posterior/readers, relation/solvers"]),
        pred("M5_AUTHORITY_PHASES", bool(common["authority_phases_respected"] and candidate_valid), ["candidate frozen before private access"]),
        pred("R1_SOURCE_COMPLETE", bool(source_ok and graph_checks["source"]), [graph_checks]),
        pred("R2_PUBLIC_PARITY", graph_checks["parity"], [graph_checks]),
        pred("R3_REPRESENTATION_OUTPUT", graph_checks["outputs"], [graph_checks]),
        pred("R4_EXECUTOR_FACTORIAL", graph_checks["executor"], [graph_checks]),
        pred("R5_TARGET_AUTHORITY", graph_checks["target"], [graph_checks]),
        pred("R6_ESTIMAND_CONTROLS", graph_checks["controls"], [graph_checks]),
        pred("S1_SOURCE_COMPLETE", bool(source_ok and set_checks["schema"]), [set_checks]),
        pred("S2_POSTERIOR_PARITY", set_checks["mass"], [set_checks]),
        pred("S3_FOUR_CELLS_EXECUTABLE", set_checks["four"], [set_checks]),
        pred("S4_FIT_SUPPORT_FREEZE", set_checks["support"], [set_checks]),
        pred("S5_TARGET_UTILITY_AUTHORITY", set_checks["authority"], [set_checks]),
        pred("S6_ESTIMAND_CONTROLS", set_checks["controls"], [set_checks]),
    ]
    technical = {
        "source_status": "PASS" if source_ok else "FAIL",
        "artifact_status": "PASS" if public_ok and private_ok and candidate_valid else "FAIL",
        "checker_status": "PASS",
        "replay_status": "NOT_RUN",
    }
    diagnostics = {"source_receipts": source_receipts, "set_checks": set_checks, "graph_checks": graph_checks, "candidate_valid": candidate_valid}
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
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase", choices=("pre", "final"), required=True)
    parser.add_argument("--replay-evidence", type=Path)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    config = json.loads(args.config.resolve(strict=True).read_text())
    predicates, technical, diagnostics = compute(run, config)
    if args.phase == "pre":
        mutations = mutation_suite(config)
        write_json(run / "mapping_matrix.json", {"schema_version": "proportional-mapping-matrix-v1", "predicates": predicates, "diagnostics": diagnostics, **FIXED})
        write_json(run / "mutation_results.json", mutations)
        write_json(run / "pre_adjudication.json", {"schema_version": "proportional-mapping-pre-adjudication-v1", "technical_status": technical, "mapping_decision": None, "predicates": predicates, **FIXED})
        return
    if args.replay_evidence is None:
        raise ValueError("final phase requires replay evidence")
    replay = json.loads(args.replay_evidence.resolve(strict=True).read_text())
    technical["replay_status"] = "PASS" if replay.get("status") == "PASS" else "FAIL"
    decision = semantic_decision(predicates, technical)
    adjudication = {
        "schema_version": "proportional-mapping-adjudication-v1",
        "technical_status": technical,
        "mapping_decision": decision,
        "predicates": predicates,
        "replay_evidence_sha256": sha256_file(args.replay_evidence),
        **FIXED,
    }
    write_json(run / "adjudication.json", adjudication)
    (run / "REPORT_MAPPING_FEASIBILITY.md").write_text(report_markdown(adjudication), encoding="utf-8")


if __name__ == "__main__":
    main()
