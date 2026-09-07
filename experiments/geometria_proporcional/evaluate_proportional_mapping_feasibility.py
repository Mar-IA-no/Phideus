#!/usr/bin/env python3
"""Evaluate frozen mapping candidates against development authority on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
import sklearn
FIXED = {
    "gpu_used_or_queried": False,
    "architecture_promoted": False,
    "scientific_decision": None,
    "decision_authority": "user",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_array(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def nonempty_sets(n_families: int = 4) -> np.ndarray:
    masks = np.arange(1, 1 << n_families, dtype=np.uint8)[:, None]
    return ((masks >> np.arange(n_families, dtype=np.uint8)[None]) & 1).astype(bool)


def independent_nonempty_mass(probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sets = nonempty_sets(4)
    clipped = np.clip(probability, np.finfo(np.float64).tiny, 1.0 - np.finfo(np.float64).eps)
    score = np.sum(
        np.where(sets[None], np.log(clipped)[:, None], np.log1p(-clipped)[:, None]), axis=-1
    )
    score -= np.max(score, axis=1, keepdims=True)
    mass = np.exp(score)
    mass /= mass.sum(axis=1, keepdims=True)
    return sets, mass


def posterior_mass(logits: np.ndarray, theta: np.ndarray) -> np.ndarray:
    sets = nonempty_sets(4).astype(np.float64)
    unary = logits[:, None, :] * sets[None]
    cardinality = sets.sum(axis=1).astype(int)
    card = np.stack([(cardinality == k).astype(np.float64) for k in (2, 3, 4)], axis=1)
    pairs = np.stack([sets[:, i] * sets[:, j] for i in range(4) for j in range(i + 1, 4)], axis=1)
    contrast = pairs[:, :5] - pairs[:, [5]]
    feature = np.concatenate(
        [
            unary,
            np.broadcast_to(card[None], (len(logits), 15, 3)),
            np.broadcast_to(contrast[None], (len(logits), 15, 5)),
        ],
        axis=-1,
    )
    score = np.einsum("nsd,d->ns", feature, theta, optimize=True)
    score -= np.max(score, axis=1, keepdims=True)
    mass = np.exp(score)
    return mass / mass.sum(axis=1, keepdims=True)


def ordinal_loss_tensor(sets: np.ndarray, utilities: np.ndarray, penalty: float) -> np.ndarray:
    span = np.ptp(utilities, axis=1)
    optimum = np.max(np.where(sets[None], utilities[:, None], -np.inf), axis=-1)
    compatible = (optimum[:, None] - utilities[:, :, None]) / span[:, None, None]
    return np.where(sets.T[None], compatible, float(penalty))


def utilities_from_recipe(recipe: dict[str, Any]) -> np.ndarray:
    levels = np.asarray(recipe["levels"], dtype=np.float64)
    ranks = np.asarray(recipe["rank_permutations"], dtype=np.int64)
    utilities = levels[ranks]
    if utilities.shape != (24, 4) or np.any(np.ptp(utilities, axis=1) <= 0):
        raise ValueError("utility contract mismatch")
    return utilities


def hard_map_actions(mass: np.ndarray, utilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sets = nonempty_sets(4)
    map_index = np.argmax(mass, axis=1)
    map_set = sets[map_index]
    scores = np.where(map_set[:, None, :], utilities[None, :, :], -np.inf)
    return map_set, np.argmax(scores, axis=-1).astype(np.int64)


def posterior_risk(mass: np.ndarray, utilities: np.ndarray, penalty: float) -> tuple[np.ndarray, np.ndarray]:
    risk = np.einsum("ns,pas->npa", mass, ordinal_loss_tensor(nonempty_sets(4), utilities, penalty), optimize=True)
    return risk, np.argmin(risk, axis=-1).astype(np.int64)


def constrained_regret(
    actions: np.ndarray,
    target: np.ndarray,
    utilities: np.ndarray,
    incompatible_penalty: float,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.int64)
    target = np.asarray(target, dtype=bool)
    utilities = np.asarray(utilities, dtype=np.float64)
    rows = np.arange(len(target))[:, None]
    policies = np.arange(len(utilities))[None, :]
    optimum = np.max(np.where(target[:, None, :], utilities[None, :, :], -np.inf), axis=-1)
    chosen = utilities[policies, actions]
    normalized = (optimum - chosen) / np.ptp(utilities, axis=1)[None, :]
    return np.where(target[rows, actions], normalized, float(incompatible_penalty))


def adapted_design(
    logits: np.ndarray,
    per_seed_logits: np.ndarray,
    mass: np.ndarray,
    risk: np.ndarray,
    hard_actions: np.ndarray,
    candidate_actions: np.ndarray,
    map_set: np.ndarray,
    utilities: np.ndarray,
) -> np.ndarray:
    n, p = hard_actions.shape
    rows = np.arange(n)[:, None]
    policies = np.arange(p)[None, :]
    minimum = risk[rows, policies, candidate_actions]
    hard_risk = risk[rows, policies, hard_actions]
    advantage = np.maximum(hard_risk - minimum, 0.0)
    ordered_risk = np.sort(risk, axis=-1)
    margin = ordered_risk[..., 1] - ordered_risk[..., 0]
    clipped = np.clip(mass, np.finfo(np.float64).tiny, 1.0)
    entropy = -np.sum(mass * np.log(clipped), axis=1) / np.log(15.0)
    ordered_mass = np.sort(mass, axis=1)
    top_mass = ordered_mass[:, -1]
    top_margin = ordered_mass[:, -1] - ordered_mass[:, -2]
    sets = nonempty_sets(4).astype(np.float64)
    cardinality = sets.sum(axis=1)
    expected_cardinality = mass @ cardinality
    cardinality_variance = mass @ (cardinality**2) - expected_cardinality**2
    map_cardinality = map_set.sum(axis=1).astype(np.float64)
    map_index = (map_set.astype(np.int64) * (1 << np.arange(4))).sum(axis=1) - 1
    map_mass = mass[np.arange(n), map_index]
    seed_std = np.std(per_seed_logits, axis=0, ddof=0)
    token = np.stack(
        [
            entropy,
            top_mass,
            top_margin,
            map_cardinality,
            expected_cardinality,
            cardinality_variance,
            map_mass,
            seed_std.mean(axis=1),
            seed_std.max(axis=1),
        ],
        axis=-1,
    )
    repeated = np.broadcast_to(token[:, None, :], (n, p, 9))
    utility = np.broadcast_to(utilities[None, :, :], (n, p, 4))
    design = np.concatenate(
        [advantage[..., None], hard_risk[..., None], minimum[..., None], margin[..., None], repeated, utility],
        axis=-1,
    ).astype(np.float64)
    if design.shape != (n, p, 17) or not np.all(np.isfinite(design)):
        raise ValueError("adapted contextual design invalid")
    return design


def disagreement_weights(disagreement: np.ndarray) -> np.ndarray:
    counts = disagreement.sum(axis=1)
    weights = np.zeros(disagreement.shape, dtype=np.float64)
    active = counts > 0
    weights[active] = disagreement[active] / counts[active, None]
    return weights


def weighted_scaler(x: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.average(x, axis=0, weights=weights)
    variance = np.average((x - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale == 0] = 1.0
    return mean, scale


def fit_ridge(x: np.ndarray, y: np.ndarray, weights: np.ndarray, alpha: float) -> dict[str, Any]:
    mean, scale = weighted_scaler(x, weights)
    xs = (x - mean) / scale
    design = np.column_stack([np.ones(len(xs)), xs])
    gram = design.T @ (weights[:, None] * design)
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(gram + penalty, design.T @ (weights * y))
    return {"mean": mean, "scale": scale, "intercept": float(beta[0]), "coef": beta[1:]}


def score_ridge(state: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return state["intercept"] + ((x - state["mean"]) / state["scale"]) @ state["coef"]


def fit_logistic(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    if sklearn.__version__ != "1.8.0" or np.unique(y).tolist() != [0, 1]:
        raise RuntimeError("guard runtime/classes mismatch")
    mean, scale = weighted_scaler(x, weights)
    xs = (x - mean) / scale
    model = LogisticRegression(
        C=1.0,
        penalty="l2",
        l1_ratio=0.0,
        dual=False,
        solver="lbfgs",
        class_weight=None,
        fit_intercept=True,
        max_iter=2000,
        tol=1e-10,
        warm_start=False,
    )
    model.fit(xs, y, sample_weight=weights)
    return {
        "mean": mean,
        "scale": scale,
        "intercept": float(model.intercept_[0]),
        "coef": np.asarray(model.coef_[0], dtype=np.float64),
        "iterations": int(model.n_iter_[0]),
    }


def score_logistic(state: dict[str, Any], x: np.ndarray) -> np.ndarray:
    linear = state["intercept"] + ((x - state["mean"]) / state["scale"]) @ state["coef"]
    return expit(linear)


def json_model(state: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in state.items()
    }


def select_contextual(
    design: np.ndarray,
    target: np.ndarray,
    hard: np.ndarray,
    candidate: np.ndarray,
    utilities: np.ndarray,
    split_role: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    disagreement = hard != candidate
    weights = disagreement_weights(disagreement)
    gain = constrained_regret(hard, target, utilities, config["incompatible_penalty"]) - constrained_regret(
        candidate, target, utilities, config["incompatible_penalty"]
    )
    rows = np.arange(len(target))[:, None]
    compatible_candidate = target[rows, candidate]
    fit_tokens = split_role.astype(str) == "calibration_fit"
    select_tokens = split_role.astype(str) == "decision_select"
    fit_active = fit_tokens[:, None] & disagreement
    select_active = select_tokens[:, None] & disagreement
    x_fit = design[fit_active]
    w_fit = weights[fit_active]
    if not len(x_fit) or np.any(w_fit <= 0):
        raise RuntimeError("empty proposer fitting support")
    ridge = fit_ridge(x_fit, gain[fit_active], w_fit, float(config["ridge_alpha"]))
    harm = (gain < -float(config["harm_epsilon"])).astype(np.int8)
    incompat = (~compatible_candidate).astype(np.int8)
    harm_model = fit_logistic(x_fit, harm[fit_active], w_fit)
    incompat_model = fit_logistic(x_fit, incompat[fit_active], w_fit)
    flat = design.reshape(-1, design.shape[-1])
    proposer_score = score_ridge(ridge, flat).reshape(hard.shape)
    harm_probability = score_logistic(harm_model, flat).reshape(hard.shape)
    incompat_probability = score_logistic(incompat_model, flat).reshape(hard.shape)
    candidates: list[dict[str, Any]] = []

    def evaluate(actions: np.ndarray, authorized: np.ndarray, qs: tuple[float, float, float]) -> dict[str, Any]:
        select = select_tokens[:, None]
        regret = constrained_regret(actions, target, utilities, config["incompatible_penalty"])
        compatible = target[rows, actions]
        realized = constrained_regret(hard, target, utilities, config["incompatible_penalty"]) - regret
        mask = np.broadcast_to(select, actions.shape)
        return {
            "mean_regret": float(regret[mask].mean()),
            "incompatibility_rate": float((~compatible[mask]).mean()),
            "harm_rate": float((realized[mask] < -float(config["harm_epsilon"])).mean()),
            "authorized_rows": int((authorized & mask).sum()),
            "quantiles": list(qs),
            "actions_sha256": array_digest(actions),
            "actions": actions,
        }

    hard_only = evaluate(hard.copy(), np.zeros_like(hard), (2.0, 2.0, 2.0))
    candidates.append(hard_only)
    active_scores = proposer_score[select_active]
    if not len(active_scores):
        raise RuntimeError("selection disagreement support empty")
    for qp in config["proposer_quantiles"]:
        proposer_threshold = float(np.quantile(active_scores, qp, method="linear"))
        proposed = select_active & (proposer_score > proposer_threshold)
        if not np.any(proposed):
            continue
        for qh in config["guard_quantiles"]:
            harm_threshold = float(np.quantile(harm_probability[proposed], qh, method="linear"))
            for qi in config["guard_quantiles"]:
                incompat_threshold = float(np.quantile(incompat_probability[proposed], qi, method="linear"))
                authorized = proposed & (harm_probability < harm_threshold) & (
                    incompat_probability < incompat_threshold
                )
                actions = np.where(authorized, candidate, hard)
                row = evaluate(actions, authorized, (float(qp), float(qh), float(qi)))
                row["thresholds"] = [proposer_threshold, harm_threshold, incompat_threshold]
                candidates.append(row)
    chosen = min(
        candidates,
        key=lambda row: (
            row["mean_regret"],
            row["incompatibility_rate"],
            row["harm_rate"],
            -row["authorized_rows"],
            *row["quantiles"],
        ),
    )
    actions = chosen.pop("actions")
    return {
        "actions": actions,
        "models": {"proposer": json_model(ridge), "harm_guard": json_model(harm_model), "incompatibility_guard": json_model(incompat_model)},
        "selected": chosen,
        "candidate_count": len(candidates),
        "fit_support": {
            "rows": int(fit_active.sum()),
            "tokens": int(fit_active.any(axis=1).sum()),
            "harm_0_1": [int((harm[fit_active] == value).sum()) for value in (0, 1)],
            "incompatibility_0_1": [int((incompat[fit_active] == value).sum()) for value in (0, 1)],
        },
        "design_sha256": array_digest(design),
        "disagreement_rows": int(disagreement.sum()),
    }


def deranged_indices(keys: np.ndarray, strata: list[tuple[Any, ...]], seed: int) -> tuple[np.ndarray, int]:
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, stratum in enumerate(strata):
        groups[stratum].append(index)
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


def set_posterior_metrics(mass: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    target_index = (target.astype(np.int64) * (1 << np.arange(4))).sum(axis=1) - 1
    nll = -np.log(np.clip(mass[np.arange(len(target)), target_index], np.finfo(float).tiny, 1.0))
    probability = mass @ nonempty_sets(4).astype(np.float64)
    brier = np.mean((probability - target) ** 2, axis=1)
    return {"set_nll": metric_summary(nll, mask), "marginal_brier": metric_summary(brier, mask)}


def set_action_metrics(
    actions: np.ndarray,
    target: np.ndarray,
    utility: np.ndarray,
    penalty: float,
    mask: np.ndarray,
) -> dict[str, Any]:
    compatible = target[np.arange(len(target))[:, None], actions]
    realized = constrained_regret(actions, target, utility, penalty)
    return {
        "compatibility_rate": metric_summary(compatible.astype(np.float64), mask),
        "regret": metric_summary(realized, mask),
    }


def evaluate_set(public: Path, private: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    logits = load_array(public / "w54/ensemble_logits.npy").astype(np.float64)
    per_seed = load_array(public / "w54/per_seed_logits.npy").astype(np.float64)
    split_role = load_array(public / "w54/split_role.npy").astype(str)
    unit_keys = load_array(public / "w54/unit_key.npy").astype(str)
    target = load_array(private / "w54/target.npy").astype(bool)
    design_stratum = load_array(private / "w54/design_stratum.npy").astype(str)
    cardinality = load_array(private / "w54/cardinality.npy").astype(np.int64)
    private_keys = load_array(private / "w54/unit_key.npy").astype(str)
    if not np.array_equal(unit_keys, private_keys) or target.shape != (384, 4) or not np.all(target.any(axis=1)):
        raise ValueError("set target join invalid")
    recipe = protocol["set_recipe"]
    platt = recipe["platt"]
    probability = expit(float(platt["coefficient"]) * logits + float(platt["intercept"]))
    _, marginal = independent_nonempty_mass(probability)
    selection = recipe["selection_contract"]
    if selection["best_independent"] != "independent_platt" or selection["sealed_monitor_accessed"] is not False:
        raise ValueError("Wave 54 selection contract mismatch")
    theta = np.asarray(recipe["joint_theta"], dtype=np.float64)
    joint = posterior_mass(logits, theta)
    utilities = utilities_from_recipe(recipe)
    reader_cfg = recipe["reader"]
    outputs: dict[str, Any] = {}
    action_arrays: dict[str, np.ndarray] = {}
    authorized_masks: dict[str, np.ndarray] = {}
    for name, mass in (("MARGINAL", marginal), ("JOINT", joint)):
        if mass.shape != (384, 15) or np.any(mass < 0) or not np.allclose(mass.sum(axis=1), 1.0, atol=1e-12):
            raise ValueError(f"{name} posterior invalid")
        map_set, hard = hard_map_actions(mass, utilities)
        risk, candidate = posterior_risk(mass, utilities, float(reader_cfg["incompatible_penalty"]))
        design = adapted_design(logits, per_seed, mass, risk, hard, candidate, map_set, utilities)
        contextual = select_contextual(design, target, hard, candidate, utilities, split_role, reader_cfg)
        actions = contextual.pop("actions")
        action_arrays[f"{name}_HARD"] = hard
        action_arrays[f"{name}_CONTEXTUAL"] = actions
        authorized_masks[name] = actions != hard
        outputs[name] = {
            "mass_shape": list(mass.shape),
            "mass_sha256": array_digest(mass),
            "hard_actions_sha256": array_digest(hard),
            "candidate_actions_sha256": array_digest(candidate),
            "contextual_actions_sha256": array_digest(actions),
            "map_set_sha256": array_digest(map_set),
            "contextual": contextual,
        }
    mapping, singleton = deranged_indices(
        unit_keys,
        list(zip(split_role.tolist(), design_stratum.tolist(), cardinality.tolist(), strict=True)),
        int(protocol["controls"]["set_shuffle_seed"]),
    )
    shuffled_target = target[mapping]
    select_mask = np.broadcast_to((split_role == "decision_select")[:, None], (len(target), len(utilities)))
    matched = authorized_masks["MARGINAL"] & authorized_masks["JOINT"] & select_mask
    matched_tokens = np.any(matched, axis=1)
    if not np.any(matched) or not np.any(matched_tokens):
        raise RuntimeError("set matched support empty")
    for name, mass in (("MARGINAL", marginal), ("JOINT", joint)):
        cells = {}
        for reader in ("HARD", "CONTEXTUAL"):
            actions = action_arrays[f"{name}_{reader}"]
            cells[reader] = {
                "nominal": set_action_metrics(actions, target, utilities, float(reader_cfg["incompatible_penalty"]), select_mask),
                "shuffled": set_action_metrics(actions, shuffled_target, utilities, float(reader_cfg["incompatible_penalty"]), select_mask),
                "matched_nominal": set_action_metrics(actions, target, utilities, float(reader_cfg["incompatible_penalty"]), matched),
                "matched_shuffled": set_action_metrics(actions, shuffled_target, utilities, float(reader_cfg["incompatible_penalty"]), matched),
            }
        outputs[name]["estimands"] = {
            "posterior_nominal": set_posterior_metrics(mass, target),
            "posterior_shuffled": set_posterior_metrics(mass, shuffled_target),
            "posterior_matched_nominal": set_posterior_metrics(mass, target, matched_tokens),
            "posterior_matched_shuffled": set_posterior_metrics(mass, shuffled_target, matched_tokens),
            "actions": cells,
        }
    return {
        "tokens": len(unit_keys),
        "roles": {role: int((split_role == role).sum()) for role in sorted(set(split_role))},
        "posteriors": outputs,
        "four_cells": {name: {"shape": list(values.shape), "sha256": array_digest(values)} for name, values in sorted(action_arrays.items())},
        "observed_hard_duplication": bool(np.array_equal(action_arrays["MARGINAL_HARD"], action_arrays["JOINT_HARD"])),
        "target_shuffle": {
            "mapping_sha256": array_digest(mapping),
            "target_sha256": array_digest(shuffled_target),
            "fixed_points": int((mapping == np.arange(len(mapping))).sum()),
            "singleton_strata": singleton,
        },
        "matched_report": {
            "authorized_rows": int(matched.sum()),
            "tokens": int(matched_tokens.sum()),
            "mask_sha256": array_digest(matched),
            "token_mask_sha256": array_digest(matched_tokens),
        },
        "target_join_exact": True,
    }


def graph_state_dirs(public: Path) -> list[dict[str, Any]]:
    return json.loads((public / "graph_states.json").read_text())["states"]


def incidence_matrix(n_nodes: int, edges: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(edges), n_nodes), dtype=np.float64)
    rows = np.arange(len(edges))
    matrix[rows, edges[:, 0]] = -1.0
    matrix[rows, edges[:, 1]] = 1.0
    return matrix


def solve_wls(
    n_nodes: int,
    edges: np.ndarray,
    valid: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    floor: float,
) -> np.ndarray:
    incidence = incidence_matrix(n_nodes, edges)
    normalized = np.zeros_like(weights, dtype=np.float64)
    normalized[valid] = np.clip(weights[valid], floor, None)
    normalized[valid] /= normalized[valid].mean()
    active = incidence[valid]
    laplacian = active.T @ (normalized[valid, None] * active)
    rhs = active.T @ (normalized[valid] * values[valid])
    ones = np.ones((n_nodes, 1), dtype=np.float64)
    kkt = np.block([[laplacian, ones], [ones.T, np.zeros((1, 1), dtype=np.float64)]])
    return np.linalg.solve(kkt, np.concatenate([rhs, [0.0]]))[:-1]


def solve_irls(
    n_nodes: int,
    edges: np.ndarray,
    valid: np.ndarray,
    variance: np.ndarray,
    values: np.ndarray,
    reliability: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, bool, int]:
    base = np.where(valid, np.clip(reliability, cfg["weight_floor"], None), 0.0)
    base[valid] /= base[valid].mean()
    weights = base.copy()
    scale = max(float(np.sqrt(np.median(variance[valid]))), 1e-8)
    incidence = incidence_matrix(n_nodes, edges)
    previous_x: np.ndarray | None = None
    previous_objective: float | None = None
    converged = False
    for iteration in range(1, int(cfg["irls_iterations"]) + 1):
        x_hat = solve_wls(n_nodes, edges, valid, values, weights, float(cfg["weight_floor"]))
        residual = (incidence @ x_hat - values)[valid]
        normalized = np.abs(residual / scale)
        terms = np.where(
            normalized <= float(cfg["huber_delta"]),
            0.5 * normalized**2,
            float(cfg["huber_delta"]) * (normalized - 0.5 * float(cfg["huber_delta"])),
        )
        objective = float(np.sum(base[valid] * terms))
        if previous_x is not None and previous_objective is not None:
            parameter_delta = float(np.max(np.abs(x_hat - previous_x)))
            objective_delta = abs(objective - previous_objective) / max(1.0, abs(previous_objective))
            if parameter_delta < 1e-6 and objective_delta < 1e-6:
                converged = True
                break
        magnitude = np.abs(residual)
        ratio = np.ones_like(residual)
        threshold = float(cfg["huber_delta"]) * scale
        ratio[magnitude > threshold] = threshold / magnitude[magnitude > threshold]
        ratio = np.clip(ratio, float(cfg["weight_floor"]), 1.0)
        candidate = base.copy()
        candidate[valid] *= ratio
        weights = float(cfg["irls_damping"]) * candidate + (1.0 - float(cfg["irls_damping"])) * weights
        previous_x, previous_objective = x_hat.copy(), objective
    final = solve_wls(n_nodes, edges, valid, values, weights, float(cfg["weight_floor"]))
    return final, converged, int(iteration)


def graph_summary(values: np.ndarray, masters: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    selected = np.ones(len(values), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    unique = sorted(set(masters[selected].astype(str)))
    master_values = np.asarray([values[(masters.astype(str) == master) & selected].mean() for master in unique])
    if not len(master_values) or not np.all(np.isfinite(master_values)):
        raise ValueError("graph metric support empty or nonfinite")
    return {
        "view_support": int(selected.sum()),
        "master_support": int(len(unique)),
        "master_mean": float(master_values.mean()),
        "master_p95": float(np.quantile(master_values, 0.95, method="linear")),
        "master_maximum": float(master_values.max()),
        "view_values_sha256": array_digest(values),
        "view_mask_sha256": array_digest(selected),
    }


def evaluate_graph(public: Path, private: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    state_rows = graph_state_dirs(public)
    graph_cfg = protocol["graph_recipe"]
    fields = (
        "n_nodes", "edge_index", "observed_log_ratio", "edge_valid", "path_index", "path_sign",
        "path_valid", "edge_variance", "edge_offsets", "node_offsets", "path_offsets", "unit_key",
    )
    states: dict[str, dict[str, Any]] = {}
    loaded: dict[str, dict[str, np.ndarray]] = {}
    for row in state_rows:
        name, directory = row["state"], row["directory"]
        pub = public / "graph" / directory
        prv = private / "graph" / directory
        arrays = {field: load_array(pub / f"{field}.npy") for field in fields}
        arrays.update({name_: load_array(pub / f"{name_}.npy") for name_ in ("corrected_log_ratio", "reliability")})
        private_arrays = {name_: load_array(prv / f"{name_}.npy") for name_ in GRAPH_EVALUATOR_PRIVATE_FIELDS}
        loaded[name] = {**arrays, **{f"private__{k}": v for k, v in private_arrays.items()}}
        edge_offsets = arrays["edge_offsets"].astype(int)
        node_offsets = arrays["node_offsets"].astype(int)
        target_ok, gauge_ok = True, True
        e_offsets = arrays["edge_offsets"].astype(int)
        n_offsets = arrays["node_offsets"].astype(int)
        wls_outputs: list[np.ndarray] = []
        irls_outputs: list[np.ndarray] = []
        convergence: list[bool] = []
        iterations: list[int] = []
        relation_rmse: list[float] = []
        wls_rmse: list[float] = []
        irls_rmse: list[float] = []
        for index in range(len(arrays["n_nodes"])):
            e0, e1 = e_offsets[index : index + 2]
            n0, n1 = n_offsets[index : index + 2]
            b = incidence_matrix(int(arrays["n_nodes"][index]), arrays["edge_index"][e0:e1])
            x = private_arrays["x_true"][n0:n1]
            target_ok &= bool(np.allclose(b @ x, private_arrays["clean_log_ratio"][e0:e1], atol=1e-12))
            gauge_ok &= bool(abs(float(x.mean())) < 1e-12)
            edges = arrays["edge_index"][e0:e1]
            valid = arrays["edge_valid"][e0:e1].astype(bool)
            corrected = arrays["corrected_log_ratio"][e0:e1].astype(np.float64)
            reliability = arrays["reliability"][e0:e1].astype(np.float64)
            variance = arrays["edge_variance"][e0:e1].astype(np.float64)
            wls = solve_wls(int(arrays["n_nodes"][index]), edges, valid, corrected, reliability, float(graph_cfg["weight_floor"]))
            irls, converged, count = solve_irls(int(arrays["n_nodes"][index]), edges, valid, variance, corrected, reliability, graph_cfg)
            wls_outputs.append(wls)
            irls_outputs.append(irls)
            convergence.append(converged)
            iterations.append(count)
            relation_rmse.append(float(np.sqrt(np.mean((corrected[valid] - private_arrays["clean_log_ratio"][e0:e1][valid]) ** 2))))
            wls_rmse.append(float(np.sqrt(np.mean((wls - x) ** 2))))
            irls_rmse.append(float(np.sqrt(np.mean((irls - x) ** 2))))
        wls_all = np.concatenate(wls_outputs)
        irls_all = np.concatenate(irls_outputs)
        converged_all = np.asarray(convergence, dtype=bool)
        iterations_all = np.asarray(iterations, dtype=np.int64)
        relation_array = np.asarray(relation_rmse, dtype=np.float64)
        wls_array = np.asarray(wls_rmse, dtype=np.float64)
        irls_array = np.asarray(irls_rmse, dtype=np.float64)
        solver_atol = float(protocol["controls"]["solver_replay_atol"])
        wls_difference = float(np.max(np.abs(wls_all - private_arrays["x_hat_wls"])))
        irls_difference = float(np.max(np.abs(irls_all - private_arrays["x_hat_irls"])))
        masters = private_arrays["master_id"].astype(str)
        states[name] = {
            "views": len(arrays["n_nodes"]),
            "edges": len(arrays["observed_log_ratio"]),
            "nodes": len(private_arrays["x_true"]),
            "target_incidence_exact": target_ok,
            "gauge_mean_zero": gauge_ok,
            "corrected_finite": bool(np.all(np.isfinite(arrays["corrected_log_ratio"]))),
            "reliability_finite": bool(np.all(np.isfinite(arrays["reliability"]))),
            "cached_wls_shape": list(private_arrays["x_hat_wls"].shape),
            "cached_irls_shape": list(private_arrays["x_hat_irls"].shape),
            "all_wls_max_abs_difference": wls_difference,
            "all_irls_max_abs_difference": irls_difference,
            "all_wls_replay_within_tolerance": wls_difference <= solver_atol,
            "all_irls_replay_within_tolerance": irls_difference <= solver_atol and bool(np.array_equal(converged_all, private_arrays["irls_converged"].astype(bool))) and bool(np.array_equal(iterations_all, private_arrays["irls_iterations"].astype(np.int64))),
            "solver_replay_atol": solver_atol,
            "wls_xhat_sha256": array_digest(wls_all),
            "irls_xhat_sha256": array_digest(irls_all),
            "irls_converged_sha256": array_digest(converged_all),
            "irls_iterations_sha256": array_digest(iterations_all),
            "cache_metric_parity": {
                "relation_rmse": bool(np.allclose(relation_array, private_arrays["relation_rmse"], atol=solver_atol, rtol=0.0)),
                "wls_quotient_rmse": bool(np.allclose(wls_array, private_arrays["wls_quotient_rmse"], atol=solver_atol, rtol=0.0)),
                "irls_quotient_rmse": bool(np.allclose(irls_array[converged_all], private_arrays["irls_quotient_rmse"][converged_all], atol=solver_atol, rtol=0.0)),
            },
            "nominal": {
                "relation_rmse": graph_summary(relation_array, masters),
                "wls_quotient_rmse": graph_summary(wls_array, masters),
                "irls_quotient_rmse": graph_summary(irls_array, masters),
                "irls_converged": int(converged_all.sum()),
                "irls_failed": int((~converged_all).sum()),
            },
        }
    parity: dict[str, Any] = {}
    for seed in (104729, 130363):
        left = loaded[f"raw_generic|seed={seed}"]
        right = loaded[f"raw_typed|seed={seed}"]
        parity[str(seed)] = {field: bool(np.array_equal(left[field], right[field])) for field in fields}
    reference = loaded["raw_generic|seed=104729"]
    master = reference["private__master_id"].astype(str)
    split = reference["private__split"].astype(str)
    n_nodes = reference["n_nodes"].astype(int)
    unique_masters = sorted(set(master))
    first_by_master = {value: int(np.flatnonzero(master == value)[0]) for value in unique_masters}
    master_keys = np.asarray([hashlib.sha256(("graph-master\0" + value).encode()).hexdigest() for value in unique_masters])
    strata = [(str(split[first_by_master[value]]), int(n_nodes[first_by_master[value]])) for value in unique_masters]
    master_mapping, singletons = deranged_indices(master_keys, strata, int(protocol["controls"]["graph_shuffle_seed"]))
    donor_for_master = {unique_masters[i]: unique_masters[int(master_mapping[i])] for i in range(len(unique_masters))}
    edge_offsets = reference["edge_offsets"].astype(int)
    node_offsets = reference["node_offsets"].astype(int)
    transported: list[np.ndarray] = []
    transported_x: list[np.ndarray] = []
    donor_by_view = []
    for index, receiver in enumerate(master):
        donor = donor_for_master[receiver]
        donor_index = first_by_master[donor]
        dn0, dn1 = node_offsets[donor_index : donor_index + 2]
        x = reference["private__x_true"][dn0:dn1].astype(np.float64)
        x = x - x.mean()
        e0, e1 = edge_offsets[index : index + 2]
        b = incidence_matrix(int(n_nodes[index]), reference["edge_index"][e0:e1])
        transported.append(b @ x)
        transported_x.append(x)
        donor_by_view.append(donor)
    paired_consistent = all(len(set(donor_by_view[i] for i in np.flatnonzero(master == value))) == 1 for value in unique_masters)
    transported_array = np.concatenate(transported)
    transported_x_array = np.concatenate(transported_x)
    matched_mask = np.ones(len(master), dtype=bool)
    for name in states:
        matched_mask &= loaded[name]["private__irls_converged"].astype(bool)
    if not np.any(matched_mask):
        raise RuntimeError("relational matched support empty")
    for name, row in states.items():
        arrays = loaded[name]
        e_offsets = arrays["edge_offsets"].astype(int)
        n_offsets = arrays["node_offsets"].astype(int)
        relation_values: list[float] = []
        wls_values: list[float] = []
        irls_values: list[float] = []
        for index in range(len(master)):
            e0, e1 = e_offsets[index : index + 2]
            n0, n1 = n_offsets[index : index + 2]
            valid = arrays["edge_valid"][e0:e1].astype(bool)
            relation_values.append(float(np.sqrt(np.mean((arrays["corrected_log_ratio"][e0:e1][valid] - transported[index][valid]) ** 2))))
            wls_values.append(float(np.sqrt(np.mean((arrays["private__x_hat_wls"][n0:n1] - transported_x[index]) ** 2))))
            irls_values.append(float(np.sqrt(np.mean((arrays["private__x_hat_irls"][n0:n1] - transported_x[index]) ** 2))))
        relation_values_array = np.asarray(relation_values)
        wls_values_array = np.asarray(wls_values)
        irls_values_array = np.asarray(irls_values)
        row["shuffled"] = {
            "relation_rmse": graph_summary(relation_values_array, master),
            "wls_quotient_rmse": graph_summary(wls_values_array, master),
            "irls_quotient_rmse": graph_summary(irls_values_array, master),
        }
        row["matched"] = {
            "nominal_relation_rmse": graph_summary(
                np.asarray([row_value for row_value in np.asarray(loaded[name]["private__relation_rmse"], dtype=np.float64)]), master, matched_mask
            ),
            "nominal_wls_quotient_rmse": graph_summary(np.asarray(loaded[name]["private__wls_quotient_rmse"], dtype=np.float64), master, matched_mask),
            "nominal_irls_quotient_rmse": graph_summary(np.asarray(loaded[name]["private__irls_quotient_rmse"], dtype=np.float64), master, matched_mask),
            "shuffled_relation_rmse": graph_summary(relation_values_array, master, matched_mask),
            "shuffled_wls_quotient_rmse": graph_summary(wls_values_array, master, matched_mask),
            "shuffled_irls_quotient_rmse": graph_summary(irls_values_array, master, matched_mask),
        }
    return {
        "states": states,
        "public_parity": parity,
        "four_executor_cells_present": all(
            row["all_wls_replay_within_tolerance"] and row["all_irls_replay_within_tolerance"] for row in states.values()
        ),
        "target_shuffle": {
            "masters": len(unique_masters),
            "views": len(master),
            "singleton_strata": singletons,
            "self_donors": sum(donor_for_master[value] == value for value in unique_masters),
            "paired_views_same_donor": paired_consistent,
            "transported_edges": len(transported_array),
            "transported_target_sha256": array_digest(transported_array),
            "transported_quotient_sha256": array_digest(transported_x_array),
            "matched_mask_sha256": array_digest(matched_mask),
            "matched_views": int(matched_mask.sum()),
        },
    }


GRAPH_EVALUATOR_PRIVATE_FIELDS = (
    "x_true", "clean_log_ratio", "causal_corruption_mask", "master_id", "view_id", "split", "mechanism",
    "x_hat_wls", "x_hat_irls", "relation_rmse", "wls_quotient_rmse", "irls_quotient_rmse", "irls_converged", "irls_iterations",
)


def evaluate(public: Path, private: Path, candidate: Path, output: Path) -> None:
    protocol_path = public / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema_version") != "proportional-mapping-prepared-protocol-v1":
        raise ValueError("prepared protocol schema mismatch")
    candidate_payload = json.loads(candidate.read_text())
    if candidate_payload.get("schema_version") != "proportional-mapping-candidate-v1":
        raise ValueError("candidate schema mismatch")
    candidate_before = sha256_file(candidate)
    set_evidence = evaluate_set(public, private, protocol)
    graph_evidence = evaluate_graph(public, private, protocol)
    if sha256_file(candidate) != candidate_before:
        raise RuntimeError("candidate changed during evaluation")
    evidence = {
        "schema_version": "proportional-mapping-evaluation-evidence-v1",
        "candidate_sha256": candidate_before,
        "prepared_protocol_sha256": sha256_file(protocol_path),
        "candidate_frozen_before_private_access": True,
        "common_mapping_observations": {
            "candidate_query_sha256": hashlib.sha256(candidate_payload["query"].encode()).hexdigest(),
            "candidate_unit_namespaces_sha256": hashlib.sha256(
                json.dumps(candidate_payload["unit_namespaces"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "candidate_bridge": candidate_payload["declared_cross_domain_unit_bridge"],
            "contract_sha256": hashlib.sha256(
                json.dumps(protocol["common_contract"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "set_valued": set_evidence,
        "relational": graph_evidence,
        **FIXED,
    }
    write_json(output / "evaluation_evidence.json", evidence)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(
        args.public.resolve(strict=True),
        args.private.resolve(strict=True),
        args.candidate.resolve(strict=True),
        args.output.resolve(strict=True),
    )


if __name__ == "__main__":
    main()
