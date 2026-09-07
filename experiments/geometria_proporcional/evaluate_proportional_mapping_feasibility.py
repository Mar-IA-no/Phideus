#!/usr/bin/env python3
"""Evaluate frozen mapping candidates against development authority on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
import sklearn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from geometria_proporcional.proportional_graph_contract import (  # noqa: E402
    PublicGraphObservation,
    incidence_matrix,
    solve_huber_irls,
    solve_weighted_least_squares,
)
from geometria_proporcional.wave53_uncertainty import (  # noqa: E402
    independent_nonempty_mass,
    nonempty_sets,
    ordinal_loss_tensor,
)
from geometria_proporcional.wave54_joint_set import posterior_mass  # noqa: E402


DEFAULT_CONFIG = ROOT / "experiments/geometria_proporcional/configs/proportional_mapping_feasibility_v1.json"
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


def utilities_from_manifest() -> np.ndarray:
    payload = json.loads(
        (ROOT / "data/geometria_proporcional/wave52_policy_transport_v1/policy_manifest.json").read_text()
    )
    levels = np.asarray(payload["levels"], dtype=np.float64)
    ranks = np.asarray(payload["rank_permutations"], dtype=np.int64)
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


def evaluate_set(public: Path, private: Path, config: dict[str, Any]) -> dict[str, Any]:
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
    platt = json.loads((ROOT / "data/geometria_proporcional/wave53_uncertainty_policy_v1/platt_calibrator.json").read_text())
    probability = expit(float(platt["coefficient"]) * logits + float(platt["intercept"]))
    _, marginal = independent_nonempty_mass(probability)
    selection = json.loads((ROOT / "data/geometria_proporcional/wave54_joint_set_v1/selection_freeze.json").read_text())
    if selection["best_independent"] != "independent_platt" or selection["sealed_monitor_accessed"] is not False:
        raise ValueError("Wave 54 selection contract mismatch")
    theta = np.asarray(selection["selected_models"]["joint_full"]["theta"], dtype=np.float64)
    joint = posterior_mass(logits, theta, "joint_full")
    utilities = utilities_from_manifest()
    reader_cfg = config["set_reader"]
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
        target_index = (target.astype(np.int64) * (1 << np.arange(4))).sum(axis=1) - 1
        nll = -np.log(np.clip(mass[np.arange(len(target)), target_index], np.finfo(float).tiny, 1.0))
        marginal_probability = mass @ nonempty_sets(4).astype(np.float64)
        outputs[name] = {
            "mass_shape": list(mass.shape),
            "mass_sha256": array_digest(mass),
            "hard_actions_sha256": array_digest(hard),
            "candidate_actions_sha256": array_digest(candidate),
            "contextual_actions_sha256": array_digest(actions),
            "map_set_sha256": array_digest(map_set),
            "set_nll_mean": float(nll.mean()),
            "marginal_brier_mean": float(np.mean((marginal_probability - target) ** 2)),
            "contextual": contextual,
        }
    mapping, singleton = deranged_indices(
        unit_keys,
        list(zip(split_role.tolist(), design_stratum.tolist(), cardinality.tolist(), strict=True)),
        int(config["controls"]["set_shuffle_seed"]),
    )
    matched = authorized_masks["MARGINAL"] & authorized_masks["JOINT"]
    return {
        "tokens": len(unit_keys),
        "roles": {role: int((split_role == role).sum()) for role in sorted(set(split_role))},
        "posteriors": outputs,
        "four_cells": {name: {"shape": list(values.shape), "sha256": array_digest(values)} for name, values in sorted(action_arrays.items())},
        "observed_hard_duplication": bool(np.array_equal(action_arrays["MARGINAL_HARD"], action_arrays["JOINT_HARD"])),
        "target_shuffle": {"mapping_sha256": array_digest(mapping), "fixed_points": int((mapping == np.arange(len(mapping))).sum()), "singleton_strata": singleton},
        "matched_authorized_rows": int(matched.sum()),
        "target_join_exact": True,
    }


def graph_state_dirs(public: Path) -> list[dict[str, Any]]:
    return json.loads((public / "graph_states.json").read_text())["states"]


def graph_view(public_dir: Path, index: int) -> PublicGraphObservation:
    edge_offsets = load_array(public_dir / "edge_offsets.npy")
    path_offsets = load_array(public_dir / "path_offsets.npy")
    e0, e1 = map(int, edge_offsets[index : index + 2])
    p0, p1 = map(int, path_offsets[index : index + 2])
    return PublicGraphObservation(
        n_nodes=int(load_array(public_dir / "n_nodes.npy")[index]),
        edge_index=load_array(public_dir / "edge_index.npy")[e0:e1],
        observed_log_ratio=load_array(public_dir / "observed_log_ratio.npy")[e0:e1],
        edge_valid=load_array(public_dir / "edge_valid.npy")[e0:e1],
        path_index=load_array(public_dir / "path_index.npy")[p0:p1],
        path_sign=load_array(public_dir / "path_sign.npy")[p0:p1],
        path_valid=load_array(public_dir / "path_valid.npy")[p0:p1],
        edge_variance=load_array(public_dir / "edge_variance.npy")[e0:e1],
    )


def evaluate_graph(public: Path, private: Path, config: dict[str, Any]) -> dict[str, Any]:
    state_rows = graph_state_dirs(public)
    graph_cfg = json.loads((ROOT / "data/geometria_proporcional/proportional_graph_neural_smoke_v1/resolved_config.json").read_text())["graph"]
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
        target_ok = True
        gauge_ok = True
        for index in range(len(arrays["n_nodes"])):
            e0, e1 = edge_offsets[index : index + 2]
            n0, n1 = node_offsets[index : index + 2]
            b = incidence_matrix(int(arrays["n_nodes"][index]), arrays["edge_index"][e0:e1])
            x = private_arrays["x_true"][n0:n1]
            target_ok &= bool(np.allclose(b @ x, private_arrays["clean_log_ratio"][e0:e1], atol=1e-12))
            gauge_ok &= bool(abs(float(x.mean())) < 1e-12)
        sample = 0
        observation = graph_view(pub, sample)
        e0, e1 = map(int, edge_offsets[sample : sample + 2])
        n0, n1 = map(int, node_offsets[sample : sample + 2])
        corrected = arrays["corrected_log_ratio"][e0:e1].astype(np.float64)
        reliability = arrays["reliability"][e0:e1].astype(np.float64)
        wls = solve_weighted_least_squares(observation, values=corrected, weights=reliability, weight_floor=float(graph_cfg["weight_floor"]))
        irls = solve_huber_irls(
            observation,
            values=corrected,
            base_weights=reliability,
            delta=float(graph_cfg["huber_delta"]),
            max_iterations=int(graph_cfg["irls_iterations"]),
            damping=float(graph_cfg["irls_damping"]),
            weight_floor=float(graph_cfg["weight_floor"]),
        )
        wls_difference = float(np.max(np.abs(wls.x_hat - private_arrays["x_hat_wls"][n0:n1])))
        irls_difference = float(np.max(np.abs(irls.x_hat - private_arrays["x_hat_irls"][n0:n1])))
        solver_atol = float(config["controls"]["solver_replay_atol"])
        cached_converged = bool(private_arrays["irls_converged"][sample])
        cached_iterations = int(private_arrays["irls_iterations"][sample])
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
            "sample_wls_max_abs_difference": wls_difference,
            "sample_irls_max_abs_difference": irls_difference,
            "sample_wls_replay_within_tolerance": wls_difference <= solver_atol,
            "sample_irls_replay_within_tolerance": irls_difference <= solver_atol and bool(irls.converged) == cached_converged and int(irls.iterations) == cached_iterations,
            "solver_replay_atol": solver_atol,
            "sample_irls_converged": bool(irls.converged),
            "sample_irls_iterations": int(irls.iterations),
            "cached_irls_converged": cached_converged,
            "cached_irls_iterations": cached_iterations,
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
    master_mapping, singletons = deranged_indices(master_keys, strata, int(config["controls"]["graph_shuffle_seed"]))
    donor_for_master = {unique_masters[i]: unique_masters[int(master_mapping[i])] for i in range(len(unique_masters))}
    edge_offsets = reference["edge_offsets"].astype(int)
    node_offsets = reference["node_offsets"].astype(int)
    transported: list[np.ndarray] = []
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
        donor_by_view.append(donor)
    paired_consistent = all(len(set(donor_by_view[i] for i in np.flatnonzero(master == value))) == 1 for value in unique_masters)
    transported_array = np.concatenate(transported)
    return {
        "states": states,
        "public_parity": parity,
        "four_executor_cells_present": all(
            row["sample_wls_replay_within_tolerance"] and row["sample_irls_replay_within_tolerance"] for row in states.values()
        ),
        "target_shuffle": {
            "masters": len(unique_masters),
            "views": len(master),
            "singleton_strata": singletons,
            "self_donors": sum(donor_for_master[value] == value for value in unique_masters),
            "paired_views_same_donor": paired_consistent,
            "transported_edges": len(transported_array),
            "transported_target_sha256": array_digest(transported_array),
        },
    }


GRAPH_EVALUATOR_PRIVATE_FIELDS = (
    "x_true", "clean_log_ratio", "causal_corruption_mask", "master_id", "view_id", "split", "mechanism",
    "x_hat_wls", "x_hat_irls", "relation_rmse", "wls_quotient_rmse", "irls_quotient_rmse", "irls_converged", "irls_iterations",
)


def evaluate(config_path: Path, public: Path, private: Path, candidate: Path, output: Path) -> None:
    config = json.loads(config_path.read_text())
    candidate_payload = json.loads(candidate.read_text())
    if candidate_payload.get("schema_version") != "proportional-mapping-candidate-v1":
        raise ValueError("candidate schema mismatch")
    candidate_before = sha256_file(candidate)
    set_evidence = evaluate_set(public, private, config)
    graph_evidence = evaluate_graph(public, private, config)
    if sha256_file(candidate) != candidate_before:
        raise RuntimeError("candidate changed during evaluation")
    evidence = {
        "schema_version": "proportional-mapping-evaluation-evidence-v1",
        "candidate_sha256": candidate_before,
        "candidate_frozen_before_private_access": True,
        "common_mapping_observations": {
            "query_literal_equal": candidate_payload["query"] == config["query"],
            "common_unit_bridge_declared": candidate_payload["declared_cross_domain_unit_bridge"] is not None,
            "unit_namespaces": candidate_payload["unit_namespaces"],
            "observation_schemas_equal": False,
            "target_schemas_equal": False,
            "score_semantics_equal": False,
            "decision_stacks_equal": False,
            "authority_phases_respected": True,
        },
        "set_valued": set_evidence,
        "relational": graph_evidence,
        **FIXED,
    }
    write_json(output / "evaluation_evidence.json", evidence)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(
        args.config.resolve(strict=True),
        args.public.resolve(strict=True),
        args.private.resolve(strict=True),
        args.candidate.resolve(strict=True),
        args.output.resolve(strict=True),
    )


if __name__ == "__main__":
    main()
