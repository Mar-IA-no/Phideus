"""CPU-only primitives for the frozen set-valued native pre-draw runner.

The module deliberately does not import ``wave52_policy`` because that source
loads torch at import time.  Its small NumPy decision formulas are reproduced
here and checked independently by the preflight suite.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import warnings
from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.special import expit
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .wave53_uncertainty import independent_nonempty_mass, nonempty_sets, ordinal_loss_tensor
from .wave54_joint_set import fit_joint_posterior, posterior_mass, target_set_indices


FEATURE_NAMES = (
    "advantage",
    "hard_risk",
    "minimum_risk",
    "action_risk_margin",
    "posterior_entropy_norm",
    "posterior_top_mass",
    "posterior_top_margin",
    "baseline_map_cardinality",
    "posterior_expected_cardinality",
    "posterior_cardinality_variance",
    "posterior_mass_baseline_map_set",
    "seed_std_mean",
    "seed_std_max",
    "utility_f0",
    "utility_f1",
    "utility_f2",
    "utility_f3",
)

MARGINAL_CONTRACT = {
    "C": 1.0,
    "l1_ratio": 0.0,
    "dual": False,
    "solver": "lbfgs",
    "class_weight": None,
    "fit_intercept": True,
    "max_iter": 1000,
    "random_state": 5301,
}

GUARD_CONTRACT = {
    "C": 1.0,
    "l1_ratio": 0.0,
    "dual": False,
    "solver": "lbfgs",
    "class_weight": None,
    "fit_intercept": True,
    "max_iter": 2000,
    "tol": 1e-10,
    "warm_start": False,
}

JOINT_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
PROPOSER_QUANTILES = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975)
GUARD_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8)
CONTROL_SEEDS = (53611, 53617, 53623, 53629, 53633)
HARM_EPSILON = 1e-12


def array_digest(values: np.ndarray) -> str:
    """Hash dtype, shape and C-order bytes without object serialization."""
    array = np.ascontiguousarray(values)
    if array.dtype.hasobject:
        raise TypeError("object arrays are not canonical")
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def validate_runtime_versions() -> dict[str, str]:
    observed = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }
    expected = {"numpy": "2.3.5", "scipy": "1.17.0", "sklearn": "1.8.0"}
    if observed != expected:
        raise RuntimeError(f"set-valued runtime version drift: {observed} != {expected}")
    return observed


def validate_logits_target(logits: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(target, dtype=bool)
    if x.ndim != 2 or x.shape[1] != 4 or y.shape != x.shape:
        raise ValueError("logits/target must align as [tokens,4]")
    if not len(x) or not np.all(np.isfinite(x)) or not np.all(y.any(axis=1)):
        raise ValueError("posterior fit requires finite logits and non-empty targets")
    return x, y


def utilities_from_manifest(payload: dict[str, Any]) -> np.ndarray:
    levels = np.asarray(payload.get("levels"), dtype=np.float64)
    permutations = np.asarray(payload.get("rank_permutations"), dtype=np.int64)
    if levels.shape != (4,) or permutations.shape != (24, 4):
        raise ValueError("utility catalogue shape mismatch")
    if set(map(tuple, permutations)) != set(itertools.permutations(range(4))):
        raise ValueError("utility catalogue must contain all 24 rank permutations")
    utilities = levels[permutations]
    if not np.all(np.isfinite(utilities)) or np.any(np.ptp(utilities, axis=1) <= 0.0):
        raise ValueError("utility catalogue has an invalid range")
    return utilities


def authorized_actions(target: np.ndarray, utilities: np.ndarray) -> np.ndarray:
    y = np.asarray(target, dtype=bool)
    u = np.asarray(utilities, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != 4 or u.shape != (24, 4) or not np.all(y.any(axis=1)):
        raise ValueError("authorized-action inputs are invalid")
    masked = np.where(y[:, None, :], u[None, :, :], -np.inf)
    return np.argmax(masked, axis=-1).astype(np.int64)


def constrained_regret(
    actions: np.ndarray,
    target: np.ndarray,
    utilities: np.ndarray,
    incompatible_penalty: float = 1.25,
) -> np.ndarray:
    a = np.asarray(actions, dtype=np.int64)
    y = np.asarray(target, dtype=bool)
    u = np.asarray(utilities, dtype=np.float64)
    if a.shape != (len(y), len(u)) or y.ndim != 2 or y.shape[1] != 4 or u.shape[1] != 4:
        raise ValueError("regret inputs do not align")
    if np.any((a < 0) | (a >= 4)) or not np.all(y.any(axis=1)):
        raise ValueError("regret actions/targets are invalid")
    optimum = np.max(np.where(y[:, None, :], u[None, :, :], -np.inf), axis=-1)
    chosen = u[np.arange(len(u))[None, :], a]
    span = np.ptp(u, axis=1)
    compatible = y[np.arange(len(y))[:, None], a]
    return np.where(compatible, (optimum - chosen) / span[None, :], float(incompatible_penalty))


def fit_marginal_state(logits: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    validate_runtime_versions()
    x, y = validate_logits_target(logits, target)
    flat_x = x.reshape(-1, 1)
    flat_y = y.reshape(-1).astype(np.int64)
    if not np.array_equal(np.unique(flat_y), np.asarray([0, 1])):
        raise RuntimeError("pooled Platt fit requires both classes")
    model = LogisticRegression(**MARGINAL_CONTRACT)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(flat_x, flat_y)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("pooled Platt fit emitted a convergence warning")
    if int(model.n_iter_[0]) >= int(MARGINAL_CONTRACT["max_iter"]):
        raise RuntimeError("pooled Platt fit did not converge")
    direct = model.predict_proba(flat_x)[:, 1]
    reconstructed = expit(float(model.coef_[0, 0]) * flat_x[:, 0] + float(model.intercept_[0]))
    np.testing.assert_allclose(direct, reconstructed, rtol=0.0, atol=2e-15)
    return {
        "kind": "pooled_platt",
        "contract": dict(MARGINAL_CONTRACT),
        "sklearn_version": sklearn.__version__,
        "classes": model.classes_.astype(int).tolist(),
        "coefficient": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
        "n_iter": int(model.n_iter_[0]),
        "n_rows": int(len(flat_y)),
        "positive_fraction": float(flat_y.mean()),
        "fit_probability_sha256": array_digest(direct),
    }


def marginal_probability(state: dict[str, Any], logits: np.ndarray) -> np.ndarray:
    if state.get("kind") != "pooled_platt" or state.get("contract") != MARGINAL_CONTRACT:
        raise ValueError("invalid pooled Platt state")
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 4 or not np.all(np.isfinite(x)):
        raise ValueError("marginal logits must be finite [tokens,4]")
    return expit(float(state["coefficient"]) * x + float(state["intercept"]))


def marginal_set_mass(state: dict[str, Any], logits: np.ndarray) -> np.ndarray:
    return independent_nonempty_mass(marginal_probability(state, logits))[1]


def posterior_fold_ids(
    pair_tokens: Iterable[str], design_stratum: Iterable[str], cardinality: np.ndarray
) -> np.ndarray:
    tokens = np.asarray(list(pair_tokens)).astype(str)
    strata = np.asarray(list(design_stratum)).astype(str)
    cards = np.asarray(cardinality, dtype=np.int64)
    if tokens.ndim != 1 or strata.shape != tokens.shape or cards.shape != tokens.shape:
        raise ValueError("posterior fold fields do not align")
    if len(np.unique(tokens)) != len(tokens):
        raise ValueError("posterior fold pair_token must be unique")
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, key in enumerate(zip(strata, cards, strict=True)):
        groups[(str(key[0]), int(key[1]))].append(index)
    assignment = np.full(len(tokens), -1, dtype=np.int64)
    for key in sorted(groups):
        ordered = sorted(
            groups[key],
            key=lambda i: (
                hashlib.sha256(b"set-fold-v1" + tokens[i].encode("utf-8")).digest(),
                tokens[i].encode("utf-8"),
            ),
        )
        if len(ordered) < 4:
            raise RuntimeError(f"posterior fold stratum lacks four rows: {key}")
        for rank, index in enumerate(ordered):
            assignment[index] = rank % 4
    if set(assignment.tolist()) != {0, 1, 2, 3}:
        raise RuntimeError("posterior fold assignment is incomplete")
    return assignment


def exact_set_metric_arrays(set_mass: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    mass = np.asarray(set_mass, dtype=np.float64)
    y = np.asarray(target, dtype=bool)
    sets = nonempty_sets(4)
    if mass.shape != (len(y), 15) or y.shape != (len(mass), 4):
        raise ValueError("set metric inputs do not align")
    np.testing.assert_allclose(mass.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    indices = target_set_indices(y)
    marginals = mass @ sets.astype(np.float64)
    target_mass = mass[np.arange(len(y)), indices]
    expected_cardinality = mass @ sets.sum(axis=1)
    return {
        "exact_set_nll": -np.log(np.clip(target_mass, np.finfo(np.float64).tiny, 1.0)),
        "marginal_brier": np.mean((marginals - y) ** 2, axis=1),
        "cardinality_abs_error": np.abs(expected_cardinality - y.sum(axis=1)),
        "target_set_mass": target_mass,
        "set_accuracy": (np.argmax(mass, axis=1) == indices).astype(np.float64),
    }


def fit_joint_cv(
    logits: np.ndarray,
    target: np.ndarray,
    fold_id: np.ndarray,
    regularization_grid: Iterable[float] = JOINT_GRID,
) -> dict[str, Any]:
    x, y = validate_logits_target(logits, target)
    folds = np.asarray(fold_id, dtype=np.int64)
    grid = tuple(float(value) for value in regularization_grid)
    if folds.shape != (len(x),) or set(folds.tolist()) != {0, 1, 2, 3}:
        raise ValueError("joint CV requires four complete folds")
    if grid != JOINT_GRID:
        raise ValueError("joint regularization grid drifted")
    oof_nll = np.full((len(grid), len(x)), np.nan, dtype=np.float64)
    oof_brier = np.full_like(oof_nll, np.nan)
    fold_theta = np.full((len(grid), 4, 12), np.nan, dtype=np.float64)
    fold_objective = np.full((len(grid), 4), np.nan, dtype=np.float64)
    fold_gradient_norm = np.full_like(fold_objective, np.nan)
    fold_iterations = np.full((len(grid), 4), -1, dtype=np.int64)
    fold_evaluations = np.full((len(grid), 4), -1, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for grid_index, regularization in enumerate(grid):
        for fold in range(4):
            train = folds != fold
            holdout = folds == fold
            if not np.any(train) or not np.any(holdout):
                raise RuntimeError("joint CV fold has empty train or holdout")
            fit = fit_joint_posterior(
                x[train],
                y[train],
                "joint_full",
                regularization,
                max_iter=2000,
                gtol=1e-9,
                ftol=1e-12,
            )
            mass = posterior_mass(x[holdout], fit["theta"], "joint_full")
            metrics = exact_set_metric_arrays(mass, y[holdout])
            oof_nll[grid_index, holdout] = metrics["exact_set_nll"]
            oof_brier[grid_index, holdout] = metrics["marginal_brier"]
            fold_theta[grid_index, fold] = np.asarray(fit["theta"], dtype=np.float64)
            fold_objective[grid_index, fold] = float(fit["objective"])
            fold_gradient_norm[grid_index, fold] = float(fit["gradient_norm"])
            fold_iterations[grid_index, fold] = int(fit["iterations"])
            fold_evaluations[grid_index, fold] = int(fit["function_evaluations"])
        if not np.all(np.isfinite(oof_nll[grid_index])) or not np.all(
            np.isfinite(oof_brier[grid_index])
        ):
            raise RuntimeError("joint CV left non-finite OOF metrics")
        rows.append(
            {
                "regularization": regularization,
                "mean_oof_exact_set_nll": float(oof_nll[grid_index].mean()),
                "mean_oof_marginal_brier": float(oof_brier[grid_index].mean()),
                "negative_regularization": -regularization,
            }
        )
    selected_index = min(
        range(len(rows)),
        key=lambda index: (
            rows[index]["mean_oof_exact_set_nll"],
            rows[index]["mean_oof_marginal_brier"],
            rows[index]["negative_regularization"],
        ),
    )
    selected_regularization = grid[selected_index]
    final = fit_joint_posterior(
        x,
        y,
        "joint_full",
        selected_regularization,
        max_iter=2000,
        gtol=1e-9,
        ftol=1e-12,
    )
    state = {
        "kind": "joint_full",
        "regularization_grid": list(grid),
        "selected_index": int(selected_index),
        "selected_regularization": selected_regularization,
        "selection_key": [
            "mean_oof_exact_set_nll",
            "mean_oof_marginal_brier",
            "negative_regularization",
        ],
        "optimizer": {"method": "L-BFGS-B", "max_iter": 2000, "gtol": 1e-9, "ftol": 1e-12},
        "grid_metrics": rows,
        "final_objective": float(final["objective"]),
        "final_gradient_norm": float(final["gradient_norm"]),
        "final_iterations": int(final["iterations"]),
        "final_function_evaluations": int(final["function_evaluations"]),
        "final_message": str(final["message"]),
    }
    arrays = {
        "fold_id": folds,
        "oof_exact_set_nll": oof_nll,
        "oof_marginal_brier": oof_brier,
        "fold_theta": fold_theta,
        "fold_objective": fold_objective,
        "fold_gradient_norm": fold_gradient_norm,
        "fold_iterations": fold_iterations,
        "fold_function_evaluations": fold_evaluations,
        "final_theta": np.asarray(final["theta"], dtype=np.float64),
        "final_interaction_coefficients": np.asarray(
            final["interaction_coefficients"], dtype=np.float64
        ),
    }
    return {"state": state, "arrays": arrays}


def joint_set_mass(state: dict[str, Any], theta: np.ndarray, logits: np.ndarray) -> np.ndarray:
    if state.get("kind") != "joint_full" or np.asarray(theta).shape != (12,):
        raise ValueError("invalid joint state")
    return posterior_mass(np.asarray(logits, dtype=np.float64), np.asarray(theta, dtype=np.float64), "joint_full")


def target_derangement_v1(
    pair_tokens: Iterable[str],
    fold_id: np.ndarray,
    design_stratum: Iterable[str],
    cardinality: np.ndarray,
    seed: int = 53602,
) -> dict[str, Any]:
    tokens = np.asarray(list(pair_tokens)).astype(str)
    folds = np.asarray(fold_id, dtype=np.int64)
    strata = np.asarray(list(design_stratum)).astype(str)
    cards = np.asarray(cardinality, dtype=np.int64)
    if not (tokens.shape == folds.shape == strata.shape == cards.shape):
        raise ValueError("target derangement fields do not align")
    if len(np.unique(tokens)) != len(tokens):
        raise ValueError("target derangement pair_token must be unique")
    groups: dict[tuple[int, str, int], list[int]] = defaultdict(list)
    for index in range(len(tokens)):
        groups[(int(folds[index]), str(strata[index]), int(cards[index]))].append(index)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    donor = np.full(len(tokens), -1, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    singletons: list[str] = []
    for key in sorted(groups):
        ordered_tokens = sorted(groups[key], key=lambda index: tokens[index].encode("utf-8"))
        if len(ordered_tokens) == 1:
            donor[ordered_tokens[0]] = ordered_tokens[0]
            singletons.append(str(tokens[ordered_tokens[0]]))
            continue
        random_keys = [int(rng.bit_generator.random_raw()) for _ in ordered_tokens]
        shuffled_order = [
            index
            for _, _, index in sorted(
                zip(random_keys, [tokens[index] for index in ordered_tokens], ordered_tokens, strict=True),
                key=lambda item: (item[0], item[1].encode("utf-8")),
            )
        ]
        shift = 1 + int(rng.integers(0, len(shuffled_order) - 1, endpoint=False))
        for position, receiver in enumerate(shuffled_order):
            donor[receiver] = shuffled_order[(position + shift) % len(shuffled_order)]
    if np.any(donor < 0):
        raise AssertionError("target derangement is incomplete")
    for receiver in sorted(range(len(tokens)), key=lambda index: tokens[index].encode("utf-8")):
        rows.append(
            {
                "receiver": str(tokens[receiver]),
                "donor": str(tokens[donor[receiver]]),
                "fold_id": int(folds[receiver]),
                "design_stratum": str(strata[receiver]),
                "cardinality": int(cards[receiver]),
            }
        )
    permutable = np.asarray([len(groups[(int(folds[i]), str(strata[i]), int(cards[i]))]) > 1 for i in range(len(tokens))])
    if np.any(donor[permutable] == np.flatnonzero(permutable)):
        raise AssertionError("target derangement has identity on a permutable row")
    return {
        "donor_index": donor,
        "rows": rows,
        "singletons": singletons,
        "permutable": permutable,
        "permutable_fraction": float(permutable.mean()),
    }


def hard_map_reader(set_mass: np.ndarray, utilities: np.ndarray) -> dict[str, np.ndarray]:
    mass = np.asarray(set_mass, dtype=np.float64)
    u = np.asarray(utilities, dtype=np.float64)
    sets = nonempty_sets(4)
    if mass.ndim != 2 or mass.shape[1] != 15 or u.shape != (24, 4):
        raise ValueError("hard MAP reader inputs are invalid")
    np.testing.assert_allclose(mass.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    map_index = np.argmax(mass, axis=1).astype(np.int64)
    map_set = sets[map_index]
    actions = np.argmax(np.where(map_set[:, None, :], u[None, :, :], -np.inf), axis=-1)
    return {
        "map_set_index": map_index,
        "map_set": map_set,
        "map_set_mass": mass[np.arange(len(mass)), map_index],
        "actions": actions.astype(np.int64),
    }


def posterior_decision(set_mass: np.ndarray, utilities: np.ndarray, penalty: float) -> dict[str, np.ndarray]:
    mass = np.asarray(set_mass, dtype=np.float64)
    losses = ordinal_loss_tensor(nonempty_sets(4), utilities, penalty)
    risk = np.einsum("ns,pas->npa", mass, losses, optimize=True)
    actions = np.argmin(risk, axis=-1).astype(np.int64)
    ordered = np.sort(risk, axis=-1)
    return {
        "action_risk": risk,
        "actions": actions,
        "minimum_risk": ordered[..., 0],
        "margin": ordered[..., 1] - ordered[..., 0],
    }


def contextual_design_map(
    *,
    ensemble_logits: np.ndarray,
    per_seed_logits: np.ndarray,
    set_mass: np.ndarray,
    action_risk: np.ndarray,
    hard_state: dict[str, np.ndarray],
    posterior_actions: np.ndarray,
    utilities: np.ndarray,
) -> dict[str, np.ndarray]:
    logits = np.asarray(ensemble_logits, dtype=np.float64)
    seed_logits = np.asarray(per_seed_logits, dtype=np.float64)
    mass = np.asarray(set_mass, dtype=np.float64)
    risk = np.asarray(action_risk, dtype=np.float64)
    hard = np.asarray(hard_state["actions"], dtype=np.int64)
    posterior = np.asarray(posterior_actions, dtype=np.int64)
    u = np.asarray(utilities, dtype=np.float64)
    n = len(logits)
    if logits.shape != (n, 4) or seed_logits.shape != (3, n, 4):
        raise ValueError("contextual logits shape mismatch")
    if mass.shape != (n, 15) or risk.shape != (n, 24, 4):
        raise ValueError("contextual posterior shape mismatch")
    if hard.shape != (n, 24) or posterior.shape != hard.shape or u.shape != (24, 4):
        raise ValueError("contextual action/utility shape mismatch")

    minimum = np.take_along_axis(risk, posterior[..., None], axis=-1)[..., 0]
    hard_risk = np.take_along_axis(risk, hard[..., None], axis=-1)[..., 0]
    advantage = hard_risk - minimum
    if np.any(advantage < -1e-12):
        raise AssertionError("posterior minimum risk exceeds hard risk invariant")
    ordered_risk = np.sort(risk, axis=-1)
    action_margin = ordered_risk[..., 1] - ordered_risk[..., 0]

    clipped = np.clip(mass, np.finfo(np.float64).tiny, 1.0)
    entropy = -np.sum(mass * np.log(clipped), axis=1) / np.log(15.0)
    ordered_mass = np.sort(mass, axis=1)
    top_mass = ordered_mass[:, -1]
    top_margin = ordered_mass[:, -1] - ordered_mass[:, -2]
    sets = nonempty_sets(4).astype(np.float64)
    set_cardinality = sets.sum(axis=1)
    expected_cardinality = mass @ set_cardinality
    cardinality_variance = mass @ (set_cardinality**2) - expected_cardinality**2
    baseline_cardinality = np.asarray(hard_state["map_set"], dtype=bool).sum(axis=1).astype(np.float64)
    baseline_mass = np.asarray(hard_state["map_set_mass"], dtype=np.float64)
    seed_std = np.std(seed_logits, axis=0, ddof=0)
    token_features = np.stack(
        [
            entropy,
            top_mass,
            top_margin,
            baseline_cardinality,
            expected_cardinality,
            cardinality_variance,
            baseline_mass,
            seed_std.mean(axis=1),
            seed_std.max(axis=1),
        ],
        axis=-1,
    )
    repeated = np.broadcast_to(token_features[:, None, :], (n, 24, 9))
    utility_features = np.broadcast_to(u[None, :, :], (n, 24, 4))
    design = np.concatenate(
        [
            advantage[..., None],
            hard_risk[..., None],
            minimum[..., None],
            action_margin[..., None],
            repeated,
            utility_features,
        ],
        axis=-1,
    )
    if design.shape != (n, 24, len(FEATURE_NAMES)) or not np.all(np.isfinite(design)):
        raise FloatingPointError("contextual MAP design is invalid")
    disagreement = hard != posterior
    counts = disagreement.sum(axis=1)
    weights = np.zeros(disagreement.shape, dtype=np.float64)
    active = counts > 0
    weights[active] = disagreement[active] / counts[active, None]
    return {
        "design": design,
        "advantage": advantage,
        "disagreement": disagreement,
        "weights": weights,
    }


def reader_public_data(
    *,
    ensemble_logits: np.ndarray,
    per_seed_logits: np.ndarray,
    set_mass: np.ndarray,
    utilities: np.ndarray,
    penalty: float,
) -> dict[str, np.ndarray]:
    hard = hard_map_reader(set_mass, utilities)
    posterior = posterior_decision(set_mass, utilities, penalty)
    design = contextual_design_map(
        ensemble_logits=ensemble_logits,
        per_seed_logits=per_seed_logits,
        set_mass=set_mass,
        action_risk=posterior["action_risk"],
        hard_state=hard,
        posterior_actions=posterior["actions"],
        utilities=utilities,
    )
    return {
        "map_set_index": hard["map_set_index"],
        "map_set": hard["map_set"],
        "map_set_mass": hard["map_set_mass"],
        "hard_actions": hard["actions"],
        "action_risk": posterior["action_risk"],
        "posterior_actions": posterior["actions"],
        "minimum_risk": posterior["minimum_risk"],
        "action_risk_margin": posterior["margin"],
        **design,
    }


def reader_training_data(
    *,
    ensemble_logits: np.ndarray,
    per_seed_logits: np.ndarray,
    target: np.ndarray,
    set_mass: np.ndarray,
    utilities: np.ndarray,
    penalty: float,
) -> dict[str, np.ndarray]:
    public = reader_public_data(
        ensemble_logits=ensemble_logits,
        per_seed_logits=per_seed_logits,
        set_mass=set_mass,
        utilities=utilities,
        penalty=penalty,
    )
    y = np.asarray(target, dtype=bool)
    hard_regret = constrained_regret(public["hard_actions"], y, utilities, penalty)
    candidate_regret = constrained_regret(
        public["posterior_actions"], y, utilities, penalty
    )
    gain = hard_regret - candidate_regret
    incompatibility = ~y[
        np.arange(len(y))[:, None], public["posterior_actions"]
    ]
    return {
        **public,
        "gain": gain,
        "harm": gain < -HARM_EPSILON,
        "incompatibility": incompatibility,
        "hard_regret": hard_regret,
        "candidate_regret": candidate_regret,
    }


def fit_weighted_scaler(values: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or w.shape != (len(x),) or not len(x):
        raise ValueError("weighted scaler inputs do not align")
    if np.any(w <= 0.0) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(w)):
        raise ValueError("weighted scaler requires finite rows and positive weights")
    mean = np.average(x, axis=0, weights=w)
    variance = np.average((x - mean) ** 2, axis=0, weights=w)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale == 0.0] = 1.0
    return {"mean": mean, "scale": scale}


def _scaled_design(state: dict[str, Any], values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    mean = np.asarray(state["mean"], dtype=np.float64)
    scale = np.asarray(state["scale"], dtype=np.float64)
    if x.ndim != 2 or mean.shape != (x.shape[1],) or scale.shape != mean.shape:
        raise ValueError("portable linear-state feature shape mismatch")
    if np.any(scale <= 0.0):
        raise ValueError("portable linear-state scale must be positive")
    return (x - mean) / scale


def fit_ridge_state(
    design: np.ndarray, target: np.ndarray, weights: np.ndarray, alpha: float = 1.0
) -> dict[str, Any]:
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or w.shape != (len(x),) or not len(x):
        raise ValueError("Ridge fit arrays do not align")
    if float(alpha) != 1.0 or np.any(w <= 0.0) or not all(
        np.all(np.isfinite(value)) for value in (x, y, w)
    ):
        raise ValueError("Ridge fit contract is invalid")
    scaler = fit_weighted_scaler(x, w)
    xs = (x - scaler["mean"]) / scaler["scale"]
    augmented = np.column_stack([np.ones(len(xs), dtype=np.float64), xs])
    normal = augmented.T @ (w[:, None] * augmented)
    normal[1:, 1:] += float(alpha) * np.eye(x.shape[1], dtype=np.float64)
    rhs = augmented.T @ (w * y)
    beta = np.linalg.solve(normal, rhs)
    prediction = augmented @ beta
    if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(prediction)):
        raise FloatingPointError("Ridge fit produced non-finite state")
    return {
        "kind": "ridge",
        "alpha": 1.0,
        "solver": "numpy.linalg.solve",
        "intercept_penalized": False,
        "mean": scaler["mean"].tolist(),
        "scale": scaler["scale"].tolist(),
        "coef": beta[1:].tolist(),
        "intercept": float(beta[0]),
        "fit_prediction_sha256": array_digest(prediction),
    }


def fit_logistic_state(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    target_name: str,
) -> dict[str, Any]:
    validate_runtime_versions()
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(target, dtype=np.int8)
    w = np.asarray(weights, dtype=np.float64)
    if target_name not in {"harm", "incompatibility"}:
        raise ValueError("unknown guard target")
    if x.ndim != 2 or y.shape != (len(x),) or w.shape != (len(x),) or not len(x):
        raise ValueError("guard fit arrays do not align")
    if np.any(w <= 0.0) or not all(np.all(np.isfinite(value)) for value in (x, y, w)):
        raise ValueError("guard fit arrays are invalid")
    if not np.array_equal(np.unique(y), np.asarray([0, 1], dtype=np.int8)):
        raise RuntimeError(f"{target_name} guard requires classes 0 and 1")
    scaler = fit_weighted_scaler(x, w)
    xs = (x - scaler["mean"]) / scaler["scale"]
    model = LogisticRegression(**GUARD_CONTRACT)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(xs, y, sample_weight=w)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError(f"{target_name} guard emitted a convergence warning")
    if not np.array_equal(model.classes_, np.asarray([0, 1])):
        raise RuntimeError(f"{target_name} guard class order drifted")
    probability = model.predict_proba(xs)[:, 1]
    reconstructed = expit(xs @ model.coef_[0] + float(model.intercept_[0]))
    np.testing.assert_allclose(probability, reconstructed, rtol=0.0, atol=2e-15)
    return {
        "kind": "logistic",
        "target_name": target_name,
        "contract": dict(GUARD_CONTRACT),
        "sklearn_version": sklearn.__version__,
        "classes": [0, 1],
        "mean": scaler["mean"].tolist(),
        "scale": scaler["scale"].tolist(),
        "coef": np.asarray(model.coef_[0], dtype=np.float64).tolist(),
        "intercept": float(model.intercept_[0]),
        "n_iter": np.asarray(model.n_iter_, dtype=np.int64).tolist(),
        "fit_probability_sha256": array_digest(probability),
    }


def score_linear_state(state: dict[str, Any], design: np.ndarray) -> np.ndarray:
    xs = _scaled_design(state, design)
    value = xs @ np.asarray(state["coef"], dtype=np.float64) + float(state["intercept"])
    if state.get("kind") == "ridge":
        output = value
    elif state.get("kind") == "logistic":
        if state.get("contract") != GUARD_CONTRACT or state.get("classes") != [0, 1]:
            raise ValueError("guard state recipe drifted")
        output = expit(value)
    else:
        raise ValueError("unknown portable linear-state kind")
    if not np.all(np.isfinite(output)):
        raise FloatingPointError("portable linear-state score is non-finite")
    return output


def fit_reader_states(data: dict[str, np.ndarray]) -> dict[str, Any]:
    active = np.asarray(data["disagreement"], dtype=bool)
    design = np.asarray(data["design"], dtype=np.float64)
    weights = np.asarray(data["weights"], dtype=np.float64)
    if design.shape[:2] != active.shape or weights.shape != active.shape:
        raise ValueError("reader fit data do not align")
    if not np.any(active):
        raise RuntimeError("reader fit has no disagreement")
    x = design[active]
    w = weights[active]
    states = {
        "proposer": fit_ridge_state(x, np.asarray(data["gain"])[active], w),
        "harm": fit_logistic_state(x, np.asarray(data["harm"])[active], w, target_name="harm"),
        "incompatibility": fit_logistic_state(
            x,
            np.asarray(data["incompatibility"])[active],
            w,
            target_name="incompatibility",
        ),
    }
    return {
        "states": states,
        "active_rows": int(active.sum()),
        "active_tokens": int(active.any(axis=1).sum()),
        "weight_total": float(w.sum()),
    }


def score_reader_states(
    states: dict[str, dict[str, Any]], design: np.ndarray, disagreement: np.ndarray
) -> dict[str, np.ndarray]:
    x = np.asarray(design, dtype=np.float64)
    active = np.asarray(disagreement, dtype=bool)
    if x.ndim != 3 or x.shape[:2] != active.shape or x.shape[2] != len(FEATURE_NAMES):
        raise ValueError("reader score arrays do not align")
    scores: dict[str, np.ndarray] = {}
    for name in ("proposer", "harm", "incompatibility"):
        values = np.full(active.shape, np.nan, dtype=np.float64)
        values[active] = score_linear_state(states[name], x[active])
        scores[name] = values
    return scores


def _assignment_optimum(cost: np.ndarray, rows: list[int], donors: list[int]) -> int | None:
    if not rows:
        return 0
    if len(rows) != len(donors):
        return None
    subset = np.asarray(cost[np.ix_(rows, donors)], dtype=np.int64)
    row_index, column_index = linear_sum_assignment(subset, maximize=True)
    selected = subset[row_index, column_index]
    if np.any(selected < 0):
        return None
    return int(selected.sum())


def maximum_hamming_control(
    *,
    gain: np.ndarray,
    harm: np.ndarray,
    incompatibility: np.ndarray,
    active: np.ndarray,
    pair_tokens: Iterable[str],
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(gain, dtype=np.float64)
    harm_y = np.asarray(harm, dtype=bool)
    incompat_y = np.asarray(incompatibility, dtype=bool)
    mask = np.asarray(active, dtype=bool)
    tokens = np.asarray(list(pair_tokens)).astype(str)
    if not (values.shape == harm_y.shape == incompat_y.shape == mask.shape):
        raise ValueError("matched-control targets do not align")
    if values.ndim != 2 or tokens.shape != (len(values),) or len(np.unique(tokens)) != len(tokens):
        raise ValueError("matched-control identity is invalid")
    if not np.all(np.isfinite(values[mask])):
        raise ValueError("matched-control gain must be finite")

    rng = np.random.Generator(np.random.PCG64(int(seed)))
    counts = mask.sum(axis=1)
    mapping = np.full(mask.shape, -1, dtype=np.int64)
    permutable = np.zeros(mask.shape, dtype=bool)
    strata_rows: list[dict[str, Any]] = []
    total_hamming = 0
    for policy in range(mask.shape[1]):
        observed_counts = sorted(np.unique(counts[mask[:, policy]]).astype(int).tolist())
        for disagreement_count in observed_counts:
            indices = np.flatnonzero(mask[:, policy] & (counts == disagreement_count))
            indices = np.asarray(
                sorted(indices.tolist(), key=lambda index: tokens[index].encode("utf-8")),
                dtype=np.int64,
            )
            n_rows = len(indices)
            if n_rows == 1:
                mapping[indices[0], policy] = indices[0]
                strata_rows.append(
                    {
                        "policy_index": int(policy),
                        "disagreement_count": int(disagreement_count),
                        "rows": 1,
                        "singleton": True,
                        "maximum_hamming": 0,
                    }
                )
                continue
            permutable[indices, policy] = True
            gain_bits = np.ascontiguousarray(values[indices, policy], dtype="<f8").view("<u8")
            signatures = np.column_stack(
                [
                    gain_bits,
                    harm_y[indices, policy].astype(np.uint64),
                    incompat_y[indices, policy].astype(np.uint64),
                ]
            )
            cost = np.sum(signatures[:, None, :] != signatures[None, :, :], axis=2).astype(np.int64)
            np.fill_diagonal(cost, -1_000_000)
            random_raw = np.asarray(
                [
                    [int(rng.bit_generator.random_raw()) for _ in range(n_rows)]
                    for _ in range(n_rows)
                ],
                dtype=np.uint64,
            )
            remaining_donors = list(range(n_rows))
            remaining_optimum = _assignment_optimum(cost, list(range(n_rows)), remaining_donors)
            if remaining_optimum is None:
                raise RuntimeError("matched-control stratum has no derangement")
            stratum_optimum = remaining_optimum
            for receiver in range(n_rows):
                donor_order = sorted(
                    remaining_donors,
                    key=lambda donor: (
                        int(random_raw[receiver, donor]),
                        tokens[indices[donor]].encode("utf-8"),
                    ),
                )
                selected_donor: int | None = None
                for donor in donor_order:
                    if donor == receiver:
                        continue
                    rest_donors = [item for item in remaining_donors if item != donor]
                    rest_optimum = _assignment_optimum(
                        cost, list(range(receiver + 1, n_rows)), rest_donors
                    )
                    if rest_optimum is not None and int(cost[receiver, donor]) + rest_optimum == remaining_optimum:
                        selected_donor = donor
                        break
                if selected_donor is None:
                    raise AssertionError("lexicographic matched assignment lost optimality")
                mapping[indices[receiver], policy] = indices[selected_donor]
                remaining_optimum -= int(cost[receiver, selected_donor])
                remaining_donors.remove(selected_donor)
            if remaining_optimum != 0:
                raise AssertionError("lexicographic matched assignment did not exhaust optimum")
            total_hamming += stratum_optimum
            strata_rows.append(
                {
                    "policy_index": int(policy),
                    "disagreement_count": int(disagreement_count),
                    "rows": int(n_rows),
                    "singleton": False,
                    "maximum_hamming": int(stratum_optimum),
                }
            )
    if np.any(mapping[mask] < 0):
        raise AssertionError("matched-control mapping is incomplete")
    receiver_rows, receiver_policies = np.where(permutable)
    if np.any(mapping[receiver_rows, receiver_policies] == receiver_rows):
        raise AssertionError("matched-control mapping has identity on permutable rows")

    transported_gain = values.copy()
    transported_harm = harm_y.copy()
    transported_incompatibility = incompat_y.copy()
    active_rows, active_policies = np.where(mask)
    donors = mapping[active_rows, active_policies]
    transported_gain[active_rows, active_policies] = values[donors, active_policies]
    transported_harm[active_rows, active_policies] = harm_y[donors, active_policies]
    transported_incompatibility[active_rows, active_policies] = incompat_y[
        donors, active_policies
    ]
    semantic_rows = [
        (str(tokens[row]), int(policy), str(tokens[mapping[row, policy]]), int(counts[row]))
        for row, policy in zip(active_rows.tolist(), active_policies.tolist(), strict=True)
    ]
    semantic_rows.sort(key=lambda item: (item[0].encode("utf-8"), item[1]))
    semantic_payload = json.dumps(
        semantic_rows, ensure_ascii=False, sort_keys=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    triplet_digest = hashlib.sha256()
    triplet_digest.update(np.ascontiguousarray(transported_gain[mask]).tobytes())
    triplet_digest.update(np.ascontiguousarray(transported_harm[mask]).tobytes())
    triplet_digest.update(np.ascontiguousarray(transported_incompatibility[mask]).tobytes())
    active_count = int(mask.sum())
    return {
        "mapping": mapping,
        "gain": transported_gain,
        "harm": transported_harm,
        "incompatibility": transported_incompatibility,
        "diagnostics": {
            "seed": int(seed),
            "active_rows": active_count,
            "strata": len(strata_rows),
            "singleton_rows": int(np.sum(mask & ~permutable)),
            "permutable_fraction": float(permutable[mask].mean()),
            "maximum_hamming": int(total_hamming),
            "mapping_sha256": hashlib.sha256(semantic_payload).hexdigest(),
            "target_triplet_sha256": triplet_digest.hexdigest(),
            "stratum_rows": strata_rows,
        },
    }


def fit_control_states(
    data: dict[str, np.ndarray], pair_tokens: Iterable[str]
) -> dict[str, Any]:
    active = np.asarray(data["disagreement"], dtype=bool)
    design = np.asarray(data["design"], dtype=np.float64)
    weights = np.asarray(data["weights"], dtype=np.float64)
    controls: list[dict[str, Any]] = []
    for seed in CONTROL_SEEDS:
        shuffled = maximum_hamming_control(
            gain=data["gain"],
            harm=data["harm"],
            incompatibility=data["incompatibility"],
            active=active,
            pair_tokens=pair_tokens,
            seed=seed,
        )
        x = design[active]
        w = weights[active]
        states = {
            "proposer": fit_ridge_state(x, shuffled["gain"][active], w),
            "harm": fit_logistic_state(x, shuffled["harm"][active], w, target_name="harm"),
            "incompatibility": fit_logistic_state(
                x,
                shuffled["incompatibility"][active],
                w,
                target_name="incompatibility",
            ),
        }
        controls.append({"seed": int(seed), "states": states, **shuffled})
    mapping_hashes = {row["diagnostics"]["mapping_sha256"] for row in controls}
    target_hashes = {row["diagnostics"]["target_triplet_sha256"] for row in controls}
    if len(mapping_hashes) != 5 or len(target_hashes) != 5:
        raise RuntimeError("NOT_EVALUABLE_CONTROL_DIVERSITY")
    if any(row["diagnostics"]["permutable_fraction"] < 0.8 for row in controls):
        raise RuntimeError("NOT_EVALUABLE_CONTROL_SUPPORT")
    return {"controls": controls}


def quantile_linear(values: np.ndarray, q: float) -> float:
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 1 or not len(raw) or not np.all(np.isfinite(raw)):
        raise ValueError("quantile requires a non-empty finite vector")
    if not 0.0 <= float(q) <= 1.0:
        raise ValueError("quantile must be in [0,1]")
    return float(np.quantile(raw, float(q), method="linear"))


def threshold_triplet(
    scores: dict[str, np.ndarray], disagreement: np.ndarray, quantiles: tuple[float, float, float]
) -> dict[str, float]:
    active = np.asarray(disagreement, dtype=bool)
    if not np.any(active):
        raise RuntimeError("threshold selection has no disagreement")
    qp, qh, qi = (float(value) for value in quantiles)
    return {
        "proposer_quantile": qp,
        "harm_quantile": qh,
        "incompatibility_quantile": qi,
        "proposer_threshold": quantile_linear(scores["proposer"][active], qp),
        "harm_threshold": quantile_linear(scores["harm"][active], qh),
        "incompatibility_threshold": quantile_linear(scores["incompatibility"][active], qi),
    }


def apply_threshold_triplet(
    scores: dict[str, np.ndarray],
    disagreement: np.ndarray,
    hard_actions: np.ndarray,
    candidate_actions: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, np.ndarray]:
    active = np.asarray(disagreement, dtype=bool)
    hard = np.asarray(hard_actions, dtype=np.int64)
    candidate = np.asarray(candidate_actions, dtype=np.int64)
    if not (active.shape == hard.shape == candidate.shape):
        raise ValueError("threshold application arrays do not align")
    override = (
        active
        & (np.asarray(scores["proposer"]) > float(thresholds["proposer_threshold"]))
        & (np.asarray(scores["harm"]) < float(thresholds["harm_threshold"]))
        & (
            np.asarray(scores["incompatibility"])
            < float(thresholds["incompatibility_threshold"])
        )
    )
    return {"override": override, "actions": np.where(override, candidate, hard).astype(np.int64)}


def action_metric_arrays(
    actions: np.ndarray,
    target: np.ndarray,
    utilities: np.ndarray,
    penalty: float,
) -> dict[str, np.ndarray]:
    a = np.asarray(actions, dtype=np.int64)
    y = np.asarray(target, dtype=bool)
    regret = constrained_regret(a, y, utilities, penalty)
    oracle = authorized_actions(y, utilities)
    incompatible = ~y[np.arange(len(y))[:, None], a]
    return {
        "accuracy": np.mean(a == oracle, axis=1),
        "incompatibility": np.mean(incompatible, axis=1),
        "regret": np.mean(regret, axis=1),
        "worst_regret": np.max(regret, axis=1),
        "regret_by_policy": regret,
        "incompatibility_by_policy": incompatible,
    }


def candidate_grid(
    scores: dict[str, np.ndarray],
    disagreement: np.ndarray,
    hard_actions: np.ndarray,
    candidate_actions: np.ndarray,
) -> dict[str, Any]:
    active = np.asarray(disagreement, dtype=bool)
    metadata: list[dict[str, Any]] = []
    action_rows: list[np.ndarray] = []
    override_rows: list[np.ndarray] = []
    for qp in PROPOSER_QUANTILES:
        for qh in GUARD_QUANTILES:
            for qi in GUARD_QUANTILES:
                threshold = threshold_triplet(scores, active, (qp, qh, qi))
                applied = apply_threshold_triplet(
                    scores, active, hard_actions, candidate_actions, threshold
                )
                metadata.append({"kind": "contextual", **threshold})
                action_rows.append(applied["actions"])
                override_rows.append(applied["override"])
    metadata.append(
        {
            "kind": "hard_only",
            "proposer_quantile": 2.0,
            "harm_quantile": 2.0,
            "incompatibility_quantile": 2.0,
            "proposer_threshold": None,
            "harm_threshold": None,
            "incompatibility_threshold": None,
        }
    )
    action_rows.append(np.asarray(hard_actions, dtype=np.int64).copy())
    override_rows.append(np.zeros(active.shape, dtype=bool))
    return {
        "metadata": metadata,
        "actions": np.asarray(action_rows, dtype=np.int64),
        "override": np.asarray(override_rows, dtype=bool),
    }


def evaluate_candidate_grid(
    grid: dict[str, Any],
    target: np.ndarray,
    utilities: np.ndarray,
    penalty: float,
    hard_actions: np.ndarray,
) -> dict[str, Any]:
    actions = np.asarray(grid["actions"], dtype=np.int64)
    overrides = np.asarray(grid["override"], dtype=bool)
    if actions.ndim != 3 or actions.shape[0] != 344 or overrides.shape != actions.shape:
        raise ValueError("candidate grid must contain 343 contextual rows plus hard-only")
    hard_regret = constrained_regret(hard_actions, target, utilities, penalty)
    metric_rows: list[dict[str, Any]] = []
    mean_regret = np.empty(len(actions), dtype=np.float64)
    incompatibility_rate = np.empty_like(mean_regret)
    harm_rate = np.empty_like(mean_regret)
    authorized_rows = np.empty(len(actions), dtype=np.int64)
    for index, candidate in enumerate(actions):
        metrics = action_metric_arrays(candidate, target, utilities, penalty)
        candidate_regret = metrics["regret_by_policy"]
        mean_regret[index] = float(metrics["regret"].mean())
        incompatibility_rate[index] = float(metrics["incompatibility_by_policy"].mean())
        harm_rate[index] = float(np.mean(candidate_regret > hard_regret + HARM_EPSILON))
        authorized_rows[index] = int(overrides[index].sum())
        meta = grid["metadata"][index]
        metric_rows.append(
            {
                "candidate_index": int(index),
                "kind": meta["kind"],
                "mean_regret": float(mean_regret[index]),
                "incompatibility_rate": float(incompatibility_rate[index]),
                "harm_rate": float(harm_rate[index]),
                "authorized_rows": int(authorized_rows[index]),
                "proposer_quantile": float(meta["proposer_quantile"]),
                "harm_quantile": float(meta["harm_quantile"]),
                "incompatibility_quantile": float(meta["incompatibility_quantile"]),
            }
        )
    selected_index = min(
        range(len(metric_rows)),
        key=lambda index: (
            metric_rows[index]["mean_regret"],
            metric_rows[index]["incompatibility_rate"],
            metric_rows[index]["harm_rate"],
            -metric_rows[index]["authorized_rows"],
            metric_rows[index]["proposer_quantile"],
            metric_rows[index]["harm_quantile"],
            metric_rows[index]["incompatibility_quantile"],
        ),
    )
    return {
        "selected_index": int(selected_index),
        "selected": {**grid["metadata"][selected_index], **metric_rows[selected_index]},
        "metrics": metric_rows,
        "arrays": {
            "mean_regret": mean_regret,
            "incompatibility_rate": incompatibility_rate,
            "harm_rate": harm_rate,
            "authorized_rows": authorized_rows,
        },
    }


def matched_control_actions(
    *,
    true_override: np.ndarray,
    scores: dict[str, np.ndarray],
    thresholds: dict[str, float],
    disagreement: np.ndarray,
    hard_actions: np.ndarray,
    candidate_actions: np.ndarray,
) -> dict[str, np.ndarray]:
    active = np.asarray(disagreement, dtype=bool)
    true = np.asarray(true_override, dtype=bool)
    hard = np.asarray(hard_actions, dtype=np.int64)
    candidate = np.asarray(candidate_actions, dtype=np.int64)
    if not (active.shape == true.shape == hard.shape == candidate.shape):
        raise ValueError("matched action arrays do not align")
    authorized = apply_threshold_triplet(scores, active, hard, candidate, thresholds)["override"]
    selected = np.zeros(active.shape, dtype=bool)
    valid = np.ones(len(active), dtype=bool)
    k = true.sum(axis=1).astype(np.int64)
    policies = np.arange(active.shape[1], dtype=np.int64)
    for token_index in range(len(active)):
        need = int(k[token_index])
        if need == 0:
            continue
        universe = np.flatnonzero(authorized[token_index])
        if len(universe) < need:
            valid[token_index] = False
            continue
        order = np.lexsort(
            (
                policies[universe],
                np.asarray(scores["incompatibility"])[token_index, universe],
                np.asarray(scores["harm"])[token_index, universe],
                -np.asarray(scores["proposer"])[token_index, universe],
            )
        )
        chosen = universe[order[:need]]
        selected[token_index, chosen] = True
    if np.any(selected.sum(axis=1)[valid] != k[valid]):
        raise AssertionError("matched action did not preserve valid Hamming count")
    return {
        "actions": np.where(selected, candidate, hard).astype(np.int64),
        "selected": selected,
        "authorized_universe": authorized,
        "match_valid": valid,
        "requested_k": k,
    }


def bootstrap_indices(n_tokens: int, replicates: int, seed: int) -> np.ndarray:
    if n_tokens < 1 or int(replicates) != 5000:
        raise ValueError("bootstrap requires positive support and 5000 replicates")
    return np.random.Generator(np.random.PCG64(int(seed))).integers(
        0, n_tokens, size=(int(replicates), n_tokens), dtype=np.int64
    )


def paired_delta_summary(
    first: np.ndarray, second: np.ndarray, indices: np.ndarray
) -> dict[str, float | int]:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    boot = np.asarray(indices, dtype=np.int64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("paired delta arrays must be aligned vectors")
    if boot.shape != (5000, len(left)) or np.any((boot < 0) | (boot >= len(left))):
        raise ValueError("paired bootstrap indices do not match support")
    delta = left - right
    draws = delta[boot].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "mean_diff": float(delta.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "n_tokens": int(len(delta)),
        "n_boot": 5000,
    }


def classify_loss_delta(
    summary: dict[str, float | int], *, allow_zero_upper: bool, support_ok: bool = True
) -> str:
    if not support_ok:
        return "NOT_EVALUABLE"
    high = float(summary["ci95_high"])
    low = float(summary["ci95_low"])
    satisfied = high <= 0.0 if allow_zero_upper else high < 0.0
    if satisfied:
        return "CONDITION_SATISFIED"
    if low > 0.0:
        return "ADVERSE"
    return "NOT_RESOLVED"
