#!/usr/bin/env python3
"""Independent checker for the CPU set-valued native preflight.

This file intentionally does not import the runner or the implementation module
under test.  It reconstructs the decisive arrays from portable states and raw
fixtures, using only NumPy/SciPy/scikit-learn plus the frozen W53/W54 primitives.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Callable
import zipfile

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import expit
import sklearn
from sklearn.linear_model import LogisticRegression


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geometria_proporcional.wave53_uncertainty import (  # noqa: E402
    independent_nonempty_mass,
    nonempty_sets,
    ordinal_loss_tensor,
)
from geometria_proporcional.wave54_joint_set import (  # noqa: E402
    centered_interactions,
    fit_joint_posterior,
    posterior_mass,
    target_set_indices,
)


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
CONTROL_SEEDS = (53611, 53617, 53623, 53629, 53633)
REPLAY_EXCLUDED = {"runtime.json", "replay_receipt.json", "artifact_manifest.json"}


class CheckFailure(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--reference", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def assert_equal(left: np.ndarray, right: np.ndarray, message: str) -> None:
    if not np.array_equal(np.asarray(left), np.asarray(right)):
        raise CheckFailure(message)


def assert_close(
    left: np.ndarray, right: np.ndarray, message: str, atol: float = 2e-12
) -> None:
    try:
        np.testing.assert_allclose(left, right, rtol=0.0, atol=atol)
    except AssertionError as error:
        raise CheckFailure(message) from error


def utilities(payload: dict[str, Any]) -> np.ndarray:
    levels = np.asarray(payload["levels"], dtype=np.float64)
    permutations = np.asarray(payload["rank_permutations"], dtype=np.int64)
    if levels.shape != (4,) or permutations.shape != (24, 4):
        raise CheckFailure("utility catalogue shape drifted")
    if set(map(tuple, permutations)) != set(itertools.permutations(range(4))):
        raise CheckFailure("utility catalogue permutations drifted")
    return levels[permutations]


def authorized_actions(target: np.ndarray, utility: np.ndarray) -> np.ndarray:
    return np.argmax(
        np.where(target[:, None, :], utility[None, :, :], -np.inf), axis=-1
    ).astype(np.int64)


def regret(
    actions: np.ndarray,
    target: np.ndarray,
    utility: np.ndarray,
    penalty: float,
) -> np.ndarray:
    optimum = np.max(
        np.where(target[:, None, :], utility[None, :, :], -np.inf), axis=-1
    )
    chosen = utility[np.arange(len(utility))[None, :], actions]
    compatible = target[np.arange(len(target))[:, None], actions]
    return np.where(
        compatible,
        (optimum - chosen) / np.ptp(utility, axis=1)[None, :],
        penalty,
    )


def set_metrics(mass: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    sets = nonempty_sets(4)
    indices = target_set_indices(target)
    marginals = mass @ sets.astype(np.float64)
    target_mass = mass[np.arange(len(target)), indices]
    expected_cardinality = mass @ sets.sum(axis=1)
    return {
        "exact_set_nll": -np.log(
            np.clip(target_mass, np.finfo(np.float64).tiny, 1.0)
        ),
        "marginal_brier": np.mean((marginals - target) ** 2, axis=1),
        "cardinality_abs_error": np.abs(expected_cardinality - target.sum(axis=1)),
        "target_set_mass": target_mass,
        "set_accuracy": (np.argmax(mass, axis=1) == indices).astype(np.float64),
    }


def action_metrics(
    actions: np.ndarray, target: np.ndarray, utility: np.ndarray, penalty: float
) -> dict[str, np.ndarray]:
    values = regret(actions, target, utility, penalty)
    oracle = authorized_actions(target, utility)
    incompatible = ~target[np.arange(len(target))[:, None], actions]
    return {
        "accuracy": np.mean(actions == oracle, axis=1),
        "incompatibility": np.mean(incompatible, axis=1),
        "regret": np.mean(values, axis=1),
        "worst_regret": np.max(values, axis=1),
        "regret_by_policy": values,
        "incompatibility_by_policy": incompatible,
    }


def fold_ids(tokens: np.ndarray, strata: np.ndarray, cardinality: np.ndarray) -> np.ndarray:
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, key in enumerate(zip(strata.astype(str), cardinality, strict=True)):
        groups[(str(key[0]), int(key[1]))].append(index)
    result = np.full(len(tokens), -1, dtype=np.int64)
    for key in sorted(groups):
        ordered = sorted(
            groups[key],
            key=lambda index: (
                hashlib.sha256(b"set-fold-v1" + str(tokens[index]).encode()).digest(),
                str(tokens[index]).encode(),
            ),
        )
        for rank, index in enumerate(ordered):
            result[index] = rank % 4
    return result


def target_derangement(
    tokens: np.ndarray,
    folds: np.ndarray,
    strata: np.ndarray,
    cards: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
    groups: dict[tuple[int, str, int], list[int]] = defaultdict(list)
    for index in range(len(tokens)):
        groups[(int(folds[index]), str(strata[index]), int(cards[index]))].append(index)
    rng = np.random.Generator(np.random.PCG64(seed))
    donor = np.full(len(tokens), -1, dtype=np.int64)
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda index: str(tokens[index]).encode())
        if len(ordered) == 1:
            donor[ordered[0]] = ordered[0]
            continue
        random_keys = [int(rng.bit_generator.random_raw()) for _ in ordered]
        shuffled = [
            index
            for _, _, index in sorted(
                zip(
                    random_keys,
                    [str(tokens[index]) for index in ordered],
                    ordered,
                    strict=True,
                ),
                key=lambda item: (item[0], item[1].encode()),
            )
        ]
        shift = 1 + int(rng.integers(0, len(shuffled) - 1, endpoint=False))
        for position, receiver in enumerate(shuffled):
            donor[receiver] = shuffled[(position + shift) % len(shuffled)]
    rows = [
        {
            "receiver": str(tokens[index]),
            "donor": str(tokens[donor[index]]),
            "fold_id": int(folds[index]),
            "design_stratum": str(strata[index]),
            "cardinality": int(cards[index]),
        }
        for index in sorted(range(len(tokens)), key=lambda item: str(tokens[item]).encode())
    ]
    permutable = np.asarray(
        [
            len(groups[(int(folds[i]), str(strata[i]), int(cards[i]))]) > 1
            for i in range(len(tokens))
        ],
        dtype=bool,
    )
    return donor, rows, permutable


def marginal_mass(state: dict[str, Any], logits: np.ndarray) -> np.ndarray:
    if state.get("kind") != "pooled_platt" or state.get("contract") != MARGINAL_CONTRACT:
        raise CheckFailure("marginal portable recipe drifted")
    probability = expit(float(state["coefficient"]) * logits + float(state["intercept"]))
    return independent_nonempty_mass(probability)[1]


def hard_reader(mass: np.ndarray, utility: np.ndarray) -> dict[str, np.ndarray]:
    sets = nonempty_sets(4)
    map_index = np.argmax(mass, axis=1)
    map_set = sets[map_index]
    actions = np.argmax(
        np.where(map_set[:, None, :], utility[None, :, :], -np.inf), axis=-1
    )
    return {
        "map_set_index": map_index,
        "map_set": map_set,
        "map_set_mass": mass[np.arange(len(mass)), map_index],
        "hard_actions": actions,
    }


def public_design(
    logits: np.ndarray,
    per_seed_logits: np.ndarray,
    mass: np.ndarray,
    utility: np.ndarray,
    penalty: float,
) -> dict[str, np.ndarray]:
    hard = hard_reader(mass, utility)
    losses = ordinal_loss_tensor(nonempty_sets(4), utility, penalty)
    risk = np.einsum("ns,pas->npa", mass, losses, optimize=True)
    posterior_actions = np.argmin(risk, axis=-1)
    minimum = np.take_along_axis(risk, posterior_actions[..., None], axis=-1)[..., 0]
    hard_risk = np.take_along_axis(risk, hard["hard_actions"][..., None], axis=-1)[..., 0]
    advantage = hard_risk - minimum
    if np.any(advantage < -1e-12):
        raise CheckFailure("posterior minimum-risk invariant failed")
    risk_ordered = np.sort(risk, axis=-1)
    clipped = np.clip(mass, np.finfo(np.float64).tiny, 1.0)
    entropy = -np.sum(mass * np.log(clipped), axis=1) / np.log(15.0)
    ordered_mass = np.sort(mass, axis=1)
    sets = nonempty_sets(4).astype(np.float64)
    cards = sets.sum(axis=1)
    expected = mass @ cards
    token_features = np.stack(
        [
            entropy,
            ordered_mass[:, -1],
            ordered_mass[:, -1] - ordered_mass[:, -2],
            hard["map_set"].sum(axis=1).astype(np.float64),
            expected,
            mass @ (cards**2) - expected**2,
            hard["map_set_mass"],
            np.std(per_seed_logits, axis=0, ddof=0).mean(axis=1),
            np.std(per_seed_logits, axis=0, ddof=0).max(axis=1),
        ],
        axis=-1,
    )
    design = np.concatenate(
        [
            advantage[..., None],
            hard_risk[..., None],
            minimum[..., None],
            (risk_ordered[..., 1] - risk_ordered[..., 0])[..., None],
            np.broadcast_to(token_features[:, None, :], (len(logits), 24, 9)),
            np.broadcast_to(utility[None, :, :], (len(logits), 24, 4)),
        ],
        axis=-1,
    )
    disagreement = hard["hard_actions"] != posterior_actions
    counts = disagreement.sum(axis=1)
    weights = np.zeros_like(disagreement, dtype=np.float64)
    active = counts > 0
    weights[active] = disagreement[active] / counts[active, None]
    return {
        **hard,
        "posterior_actions": posterior_actions,
        "action_risk": risk,
        "minimum_risk": minimum,
        "action_risk_margin": risk_ordered[..., 1] - risk_ordered[..., 0],
        "advantage": advantage,
        "design": design,
        "disagreement": disagreement,
        "weights": weights,
    }


def score_state(state: dict[str, Any], design: np.ndarray) -> np.ndarray:
    mean = np.asarray(state["mean"], dtype=np.float64)
    scale = np.asarray(state["scale"], dtype=np.float64)
    value = (design - mean) / scale @ np.asarray(state["coef"], dtype=np.float64)
    value += float(state["intercept"])
    if state["kind"] == "ridge":
        return value
    if state["kind"] == "logistic" and state.get("contract") == GUARD_CONTRACT:
        return expit(value)
    raise CheckFailure("portable model kind/recipe drifted")


def score_triplet(states: dict[str, Any], public: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    active = public["disagreement"]
    result = {}
    for name in ("proposer", "harm", "incompatibility"):
        values = np.full(active.shape, np.nan, dtype=np.float64)
        values[active] = score_state(states[name], public["design"][active])
        result[name] = values
    return result


def thresholds(scores: dict[str, np.ndarray], active: np.ndarray, qs: tuple[float, float, float]) -> dict[str, float]:
    return {
        "proposer_quantile": qs[0],
        "harm_quantile": qs[1],
        "incompatibility_quantile": qs[2],
        "proposer_threshold": float(np.quantile(scores["proposer"][active], qs[0], method="linear")),
        "harm_threshold": float(np.quantile(scores["harm"][active], qs[1], method="linear")),
        "incompatibility_threshold": float(np.quantile(scores["incompatibility"][active], qs[2], method="linear")),
    }


def apply_thresholds(
    scores: dict[str, np.ndarray],
    public: dict[str, np.ndarray],
    threshold: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    override = (
        public["disagreement"]
        & (scores["proposer"] > float(threshold["proposer_threshold"]))
        & (scores["harm"] < float(threshold["harm_threshold"]))
        & (scores["incompatibility"] < float(threshold["incompatibility_threshold"]))
    )
    actions = np.where(override, public["posterior_actions"], public["hard_actions"])
    return actions.astype(np.int64), override


def render_report(estimands: dict[str, Any]) -> str:
    lines = [
        "# Preflight CPU de la rama set-valued nativa", "",
        "Estado: `RUNNER_PREFLIGHT_VALID`.", "",
        "Este paquete valida implementación y replay sobre poblaciones históricas ya abiertas. "
        "No crea un draw prospectivo, no usa monitor o lockbox y no emite una decisión científica.",
        "", "## Contratos ejercitados", "",
        "- Posteriores: MARGINAL pooled Platt y JOINT `joint_full` con selección OOF propia.",
        "- Readers: HARD por set MAP y CONTEXTUAL con 17 features ligadas a cada posterior.",
        "- Controles: cinco transportes matched por posterior y soporte común explícito.",
        "- Incertidumbre: bootstrap pareado de 5.000 réplicas por `pair_token`.",
        "- Sensibilidad: checkpoints 17/29/43 como cortes históricos, no como población de seeds.",
        "", "## Tabla diagnóstica", "",
        "| ID | Instancia | Estado | N | Media izquierda-derecha | CI95 |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in estimands["rows"]:
        if "mean_diff" in row:
            interval = f"[{row['ci95_low']:.8g}, {row['ci95_high']:.8g}]"
            mean = f"{row['mean_diff']:.8g}"
        else:
            interval, mean = "n/a", "n/a"
        lines.append(
            f"| {row['id']} | {row['instance']} | {row['status']} | "
            f"{row.get('n_tokens', 0)} | {mean} | {interval} |"
        )
    lines.extend([
        "",
        "Las etiquetas anteriores son diagnósticos de implementación sobre datos abiertos. "
        "No acreditan cobertura prospectiva, generalización ni variabilidad de entrenamiento.",
        "",
    ])
    return "\n".join(lines)


def fit_ridge(design: np.ndarray, target: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray | float]:
    mean = np.average(design, axis=0, weights=weights)
    variance = np.average((design - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale == 0.0] = 1.0
    scaled = (design - mean) / scale
    augmented = np.column_stack([np.ones(len(scaled)), scaled])
    normal = augmented.T @ (weights[:, None] * augmented)
    normal[1:, 1:] += np.eye(design.shape[1])
    beta = np.linalg.solve(normal, augmented.T @ (weights * target))
    return {"mean": mean, "scale": scale, "coef": beta[1:], "intercept": float(beta[0])}


def fit_guard(
    design: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> dict[str, np.ndarray | float]:
    mean = np.average(design, axis=0, weights=weights)
    variance = np.average((design - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale == 0.0] = 1.0
    model = LogisticRegression(**GUARD_CONTRACT).fit(
        (design - mean) / scale, target.astype(np.int8), sample_weight=weights
    )
    return {
        "mean": mean,
        "scale": scale,
        "coef": model.coef_[0],
        "intercept": float(model.intercept_[0]),
        "n_iter": model.n_iter_,
    }


def compare_fitted_state(
    expected: dict[str, Any], observed: dict[str, np.ndarray | float], name: str
) -> None:
    for key in ("mean", "scale", "coef"):
        assert_close(np.asarray(expected[key]), np.asarray(observed[key]), f"{name} {key} drifted", 5e-12)
    assert_close(
        np.asarray([expected["intercept"]]),
        np.asarray([observed["intercept"]]),
        f"{name} intercept drifted",
        5e-12,
    )


def assignment_optimum(cost: np.ndarray, rows: list[int], donors: list[int]) -> int | None:
    if not rows:
        return 0
    if len(rows) != len(donors):
        return None
    subset = np.asarray(cost[np.ix_(rows, donors)], dtype=np.int64)
    row_index, column_index = linear_sum_assignment(subset, maximize=True)
    selected = subset[row_index, column_index]
    return None if np.any(selected < 0) else int(selected.sum())


def matched_map(
    gain: np.ndarray,
    harm: np.ndarray,
    incompatibility: np.ndarray,
    active: np.ndarray,
    tokens: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    counts = active.sum(axis=1)
    mapping = np.full(active.shape, -1, dtype=np.int64)
    for policy in range(active.shape[1]):
        for count in sorted(np.unique(counts[active[:, policy]]).tolist()):
            indices = np.asarray(
                sorted(
                    np.flatnonzero(active[:, policy] & (counts == count)).tolist(),
                    key=lambda index: str(tokens[index]).encode(),
                ),
                dtype=np.int64,
            )
            if len(indices) == 1:
                mapping[indices[0], policy] = indices[0]
                continue
            bits = np.ascontiguousarray(gain[indices, policy], dtype="<f8").view("<u8")
            signatures = np.column_stack(
                [bits, harm[indices, policy].astype(np.uint64), incompatibility[indices, policy].astype(np.uint64)]
            )
            cost = np.sum(signatures[:, None, :] != signatures[None, :, :], axis=2).astype(np.int64)
            np.fill_diagonal(cost, -1_000_000)
            random_raw = np.asarray(
                [[int(rng.bit_generator.random_raw()) for _ in indices] for _ in indices],
                dtype=np.uint64,
            )
            remaining = list(range(len(indices)))
            optimum = assignment_optimum(cost, list(range(len(indices))), remaining)
            if optimum is None:
                raise CheckFailure("matched map is infeasible")
            for receiver in range(len(indices)):
                order = sorted(
                    remaining,
                    key=lambda donor: (int(random_raw[receiver, donor]), str(tokens[indices[donor]]).encode()),
                )
                chosen = None
                for donor in order:
                    if donor == receiver:
                        continue
                    rest = [item for item in remaining if item != donor]
                    rest_optimum = assignment_optimum(cost, list(range(receiver + 1, len(indices))), rest)
                    if rest_optimum is not None and int(cost[receiver, donor]) + rest_optimum == optimum:
                        chosen = donor
                        break
                if chosen is None:
                    raise CheckFailure("matched map tie-break is infeasible")
                mapping[indices[receiver], policy] = indices[chosen]
                optimum -= int(cost[receiver, chosen])
                remaining.remove(chosen)
    return mapping


def matched_actions(
    true_override: np.ndarray,
    scores: dict[str, np.ndarray],
    public: dict[str, np.ndarray],
    threshold: dict[str, Any],
) -> dict[str, np.ndarray]:
    _, authorized = apply_thresholds(scores, public, threshold)
    selected = np.zeros_like(authorized)
    valid = np.ones(len(authorized), dtype=bool)
    requested = true_override.sum(axis=1).astype(np.int64)
    policies = np.arange(authorized.shape[1])
    for token_index, need in enumerate(requested.tolist()):
        if need == 0:
            continue
        universe = np.flatnonzero(authorized[token_index])
        if len(universe) < need:
            valid[token_index] = False
            continue
        order = np.lexsort(
            (
                policies[universe],
                scores["incompatibility"][token_index, universe],
                scores["harm"][token_index, universe],
                -scores["proposer"][token_index, universe],
            )
        )
        selected[token_index, universe[order[:need]]] = True
    return {
        "actions": np.where(selected, public["posterior_actions"], public["hard_actions"]).astype(np.int64),
        "selected": selected,
        "authorized_universe": authorized,
        "match_valid": valid,
        "requested_k": requested,
    }


class Checker:
    def __init__(self, artifact: Path, reference: Path | None):
        self.root = artifact.resolve(strict=True)
        self.reference = None if reference is None else reference.resolve(strict=True)
        self.config = read_json(self.root / "config.snapshot.json")
        self.paths = {
            role: REPO_ROOT / relative
            for role, relative, _ in self.config["source_bindings"]
        }
        self.posterior = load_npz(self.root / "prepared/posterior_fit_truth.npz")
        self.policy = load_npz(self.root / "prepared/policy_fit_truth.npz")
        self.public = load_npz(self.root / "prepared/decision_select_public.npz")
        self.truth = load_npz(self.root / "prepared/decision_select_truth.npz")
        self.posterior_states = read_json(self.root / "posterior_fit/states.json")
        self.posterior_arrays = load_npz(self.root / "posterior_fit/state_arrays.npz")
        self.posterior_oof = load_npz(self.root / "posterior_fit/oof_arrays.npz")
        self.policy_states = read_json(self.root / "policy_fit/states.json")
        self.fit_scores = load_npz(self.root / "policy_fit/fit_scores.npz")
        self.control_arrays = load_npz(self.root / "policy_fit/control_arrays.npz")
        self.selection_scores = load_npz(self.root / "decision_select/scores.npz")
        self.candidate_arrays = load_npz(self.root / "decision_select/candidate_metrics.npz")
        self.selection_freeze = read_json(self.root / "decision_select/selection_freeze.json")
        self.action_arrays = load_npz(self.root / "decision_select/action_arrays.npz")
        self.matches = load_npz(self.root / "apply_fixture/actions_and_matches.npz")
        self.raw = load_npz(self.root / "evaluate_fixture/diagnostic_arrays.npz")
        self.boot = load_npz(self.root / "evaluate_fixture/bootstrap_indices.npz")
        self.estimands = read_json(self.root / "evaluate_fixture/estimand_table.json")
        self.utility = utilities(read_json(self.paths["POLICY_MANIFEST"]))
        self.penalty = float(self.config["reader"]["penalty"])
        self.mass: dict[str, dict[str, np.ndarray]] = {}
        self.fit_public: dict[str, dict[str, np.ndarray]] = {}
        self.select_public: dict[str, dict[str, np.ndarray]] = {}
        self.select_scores: dict[str, dict[str, np.ndarray]] = {}

    def p1_source_and_scope(self) -> None:
        expected_versions = self.config["versions"]
        observed_versions = {
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "sklearn": sklearn.__version__,
        }
        if observed_versions != expected_versions:
            raise CheckFailure("runtime version drifted")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise CheckFailure("CUDA_VISIBLE_DEVICES is not exactly empty")
        if "torch" in sys.modules:
            raise CheckFailure("torch is loaded")
        bindings = read_json(self.root / "source_bindings.json")
        observed = {(row["role"], row["path"]): row["sha256"] for row in bindings["sources"]}
        forbidden = tuple(self.config["forbidden_input_path_fragments"])
        for role, relative, expected in self.config["source_bindings"]:
            if sha256_file(REPO_ROOT / relative) != expected:
                raise CheckFailure(f"source hash drifted: {role}")
            if observed.get((role, relative)) != expected:
                raise CheckFailure(f"receipt source drifted: {role}")
            if role.endswith("SOURCE") and any(value in relative.lower() for value in forbidden):
                raise CheckFailure(f"forbidden path bound: {role}")
        for relative in self.config["execution_sources"]:
            expected_execution_hash = observed.get(("EXECUTION_SOURCE", relative))
            if expected_execution_hash != sha256_file(REPO_ROOT / relative):
                raise CheckFailure(f"execution source hash drifted: {relative}")
            text = (REPO_ROOT / relative).read_text(encoding="utf-8")
            imported = []
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.append(node.module)
            if any(name == "torch" or name.startswith("torch.") or name.endswith("wave52_policy") for name in imported):
                raise CheckFailure(f"forbidden import in {relative}")
        checker_text = Path(__file__).read_text(encoding="utf-8")
        checker_imported = []
        for node in ast.walk(ast.parse(checker_text)):
            if isinstance(node, ast.Import):
                checker_imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                checker_imported.append(node.module)
        if any(
            name.endswith("proportional_set_valued_native")
            or name.endswith("run_proportional_set_valued_native_preflight")
            for name in checker_imported
        ):
            raise CheckFailure("checker imports tested code")
        runtime = read_json(self.root / "runtime.json")
        if runtime["status"] != "RUNNER_PREFLIGHT_VALID" or runtime["gpu_used_or_queried"]:
            raise CheckFailure("runtime scope/status invalid")
        canonical_config = read_json(
            REPO_ROOT
            / "experiments/geometria_proporcional/configs/proportional_set_valued_native_preflight_v1.json"
        )
        if self.config != canonical_config:
            raise CheckFailure("config snapshot differs from canonical config")

    def p2_prepared_phases(self) -> None:
        expected_keys = {
            "posterior_fit": {"pair_token", "ensemble_logits", "per_seed_logits", "design_stratum", "cardinality", "cluster_id", "split_role", "target"},
            "policy_fit": {"pair_token", "ensemble_logits", "per_seed_logits", "design_stratum", "cardinality", "cluster_id", "split_role", "target"},
            "decision_select_public": {"pair_token", "ensemble_logits", "per_seed_logits", "design_stratum", "cardinality"},
            "decision_select_truth": {"pair_token", "target"},
        }
        bundles = {
            "posterior_fit": self.posterior,
            "policy_fit": self.policy,
            "decision_select_public": self.public,
            "decision_select_truth": self.truth,
        }
        source_posterior = load_npz(self.paths["POSTERIOR_FIT_SOURCE"])
        posterior_mask = source_posterior["split_role"].astype(str) == "calibration_fit"
        expected_posterior = {}
        for key in expected_keys["posterior_fit"]:
            value = source_posterior[key]
            expected_posterior[key] = (
                value[:, posterior_mask].copy()
                if key == "per_seed_logits"
                else value[posterior_mask].copy()
            )
        expected_posterior["split_role"] = np.full(
            len(expected_posterior["pair_token"]), "posterior_fit"
        )
        source_policy = load_npz(self.paths["POLICY_FIT_SOURCE"])
        expected_policy = {key: source_policy[key].copy() for key in expected_keys["policy_fit"]}
        expected_policy["split_role"] = np.full(len(expected_policy["pair_token"]), "policy_fit")
        source_selection = load_npz(self.paths["SELECTION_TRUTH_SOURCE"])
        expected_public = {
            key: source_selection[key].copy()
            for key in expected_keys["decision_select_public"]
        }
        expected_truth = {
            key: source_selection[key].copy()
            for key in expected_keys["decision_select_truth"]
        }
        expected_bundles = {
            "posterior_fit": expected_posterior,
            "policy_fit": expected_policy,
            "decision_select_public": expected_public,
            "decision_select_truth": expected_truth,
        }
        for name, bundle in bundles.items():
            if set(bundle) != expected_keys[name]:
                raise CheckFailure(f"prepared schema drifted: {name}")
            expected_bundle = expected_bundles[name]
            for key in sorted(bundle):
                if (
                    bundle[key].dtype != expected_bundle[key].dtype
                    or bundle[key].shape != expected_bundle[key].shape
                    or not np.array_equal(bundle[key], expected_bundle[key])
                ):
                    raise CheckFailure(f"prepared provenance drifted: {name}/{key}")
        phases = [
            set(self.posterior["pair_token"].astype(str)),
            set(self.policy["pair_token"].astype(str)),
            set(self.public["pair_token"].astype(str)),
        ]
        expected_rows = self.config["opened_fixture_roles"]
        if (
            len(self.posterior["pair_token"]) != expected_rows["posterior_fit"]
            or len(self.policy["pair_token"]) != expected_rows["policy_fit"]
            or len(self.public["pair_token"]) != expected_rows["decision_select"]
        ):
            raise CheckFailure("prepared role count drifted")
        if any(len(tokens) != len(bundle["pair_token"]) for tokens, bundle in zip(phases, (self.posterior, self.policy, self.public), strict=True)):
            raise CheckFailure("prepared pair_token is not unique")
        if any(left & right for left, right in itertools.combinations(phases, 2)):
            raise CheckFailure("prepared physical phases overlap")
        assert_equal(self.public["pair_token"], self.truth["pair_token"], "public/truth identity drifted")
        if any(fragment in key.lower() for key in self.public for fragment in ("target", "truth", "oracle", "gain", "regret", "harm")):
            raise CheckFailure("public bundle leaks truth semantics")

    def p3_marginal_native(self) -> None:
        logits = self.public["ensemble_logits"].astype(np.float64)
        fit_logits = self.posterior["ensemble_logits"].astype(np.float64)
        shuffled_target = load_npz(
            self.root / "posterior_fit/target_shuffle_arrays.npz"
        )["target_shuffled"].astype(bool)
        self.mass["marginal"] = {}
        for role, state_role, fit_target in (
            ("real", "real", self.posterior["target"].astype(bool)),
            ("target_shuffled", "target_shuffled", shuffled_target),
        ):
            state = self.posterior_states["marginal"][state_role]
            if state["n_iter"] >= MARGINAL_CONTRACT["max_iter"] or state["classes"] != [0, 1]:
                raise CheckFailure("marginal convergence/classes invalid")
            model = LogisticRegression(**MARGINAL_CONTRACT).fit(
                fit_logits.reshape(-1, 1), fit_target.reshape(-1).astype(np.int64)
            )
            assert_close(
                np.asarray([state["coefficient"], state["intercept"]]),
                np.asarray([model.coef_[0, 0], model.intercept_[0]]),
                f"marginal {role} refit drifted",
                2e-14,
            )
            if state["n_iter"] != int(model.n_iter_[0]):
                raise CheckFailure(f"marginal {role} iteration metadata drifted")
            mass = marginal_mass(state, logits)
            assert_close(
                mass,
                self.selection_scores[f"marginal__set_mass_{role}"],
                f"marginal {role} mass drifted",
            )
            assert_close(mass.sum(axis=1), np.ones(len(mass)), "marginal mass not normalized")
            self.mass["marginal"][role] = mass

    def p4_joint_native(self) -> None:
        logits_fit = self.posterior["ensemble_logits"].astype(np.float64)
        target_real = self.posterior["target"].astype(bool)
        shuffled = load_npz(self.root / "posterior_fit/target_shuffle_arrays.npz")["target_shuffled"].astype(bool)
        folds = fold_ids(
            self.posterior["pair_token"], self.posterior["design_stratum"], self.posterior["cardinality"]
        )
        self.mass["joint"] = {}
        for role, target in (("real", target_real), ("target_shuffled", shuffled)):
            state = self.posterior_states["joint"][role]
            if tuple(state["regularization_grid"]) != JOINT_GRID or len(state["grid_metrics"]) != 6:
                raise CheckFailure(f"joint {role} grid drifted")
            prefix = f"joint_{role}"
            assert_equal(folds, self.posterior_oof[f"{prefix}__fold_id"], f"joint {role} fold drifted")
            oof_nll = np.empty((6, len(logits_fit)))
            oof_brier = np.empty_like(oof_nll)
            fold_theta = np.empty((6, 4, 12), dtype=np.float64)
            fold_objective = np.empty((6, 4), dtype=np.float64)
            fold_gradient = np.empty((6, 4), dtype=np.float64)
            fold_iterations = np.empty((6, 4), dtype=np.int64)
            fold_evaluations = np.empty((6, 4), dtype=np.int64)
            for grid_index, regularization in enumerate(JOINT_GRID):
                for fold in range(4):
                    train = folds != fold
                    holdout = folds == fold
                    fit = fit_joint_posterior(
                        logits_fit[train], target[train], "joint_full", regularization,
                        max_iter=2000, gtol=1e-9, ftol=1e-12,
                    )
                    metric = set_metrics(
                        posterior_mass(logits_fit[holdout], fit["theta"], "joint_full"),
                        target[holdout],
                    )
                    oof_nll[grid_index, holdout] = metric["exact_set_nll"]
                    oof_brier[grid_index, holdout] = metric["marginal_brier"]
                    fold_theta[grid_index, fold] = fit["theta"]
                    fold_objective[grid_index, fold] = fit["objective"]
                    fold_gradient[grid_index, fold] = fit["gradient_norm"]
                    fold_iterations[grid_index, fold] = fit["iterations"]
                    fold_evaluations[grid_index, fold] = fit["function_evaluations"]
            assert_close(oof_nll, self.posterior_oof[f"{prefix}__oof_exact_set_nll"], f"joint {role} OOF NLL drifted", 2e-10)
            assert_close(oof_brier, self.posterior_oof[f"{prefix}__oof_marginal_brier"], f"joint {role} OOF Brier drifted", 2e-10)
            for key, value, tolerance in (
                ("fold_theta", fold_theta, 2e-10),
                ("fold_objective", fold_objective, 2e-10),
                ("fold_gradient_norm", fold_gradient, 2e-10),
                ("fold_iterations", fold_iterations, 0.0),
                ("fold_function_evaluations", fold_evaluations, 0.0),
            ):
                if tolerance == 0.0:
                    assert_equal(value, self.posterior_oof[f"{prefix}__{key}"], f"joint {role} {key} drifted")
                else:
                    assert_close(value, self.posterior_oof[f"{prefix}__{key}"], f"joint {role} {key} drifted", tolerance)
            chosen = min(
                range(6),
                key=lambda index: (oof_nll[index].mean(), oof_brier[index].mean(), -JOINT_GRID[index]),
            )
            if chosen != state["selected_index"] or JOINT_GRID[chosen] != state["selected_regularization"]:
                raise CheckFailure(f"joint {role} lambda selection drifted")
            final = fit_joint_posterior(
                logits_fit, target, "joint_full", JOINT_GRID[chosen],
                max_iter=2000, gtol=1e-9, ftol=1e-12,
            )
            theta = self.posterior_arrays[f"{prefix}__final_theta"]
            assert_close(theta, final["theta"], f"joint {role} final theta drifted", 2e-10)
            assert_close(
                self.posterior_arrays[f"{prefix}__final_interaction_coefficients"],
                centered_interactions(final["theta"], "joint_full"),
                f"joint {role} final interactions drifted",
                2e-10,
            )
            for state_key, final_key in (
                ("final_objective", "objective"),
                ("final_gradient_norm", "gradient_norm"),
                ("final_iterations", "iterations"),
                ("final_function_evaluations", "function_evaluations"),
                ("final_message", "message"),
            ):
                if state[state_key] != final[final_key]:
                    raise CheckFailure(f"joint {role} {state_key} metadata drifted")
            mass = posterior_mass(self.public["ensemble_logits"], theta, "joint_full")
            assert_close(mass, self.selection_scores[f"joint__set_mass_{role}"], f"joint {role} selection mass drifted", 2e-11)
            self.mass["joint"][role] = mass

    def p5_target_shuffle(self) -> None:
        folds = fold_ids(self.posterior["pair_token"], self.posterior["design_stratum"], self.posterior["cardinality"])
        donor, rows, permutable = target_derangement(
            self.posterior["pair_token"], folds, self.posterior["design_stratum"], self.posterior["cardinality"], 53602
        )
        saved = load_npz(self.root / "posterior_fit/target_shuffle_arrays.npz")
        assert_equal(donor, saved["donor_index"], "target shuffle donor drifted")
        assert_equal(permutable, saved["permutable"], "target shuffle support drifted")
        assert_equal(self.posterior["target"][donor], saved["target_shuffled"], "target shuffle target drifted")
        if rows != read_json(self.root / "posterior_fit/target_shuffle_map.json"):
            raise CheckFailure("target shuffle semantic map drifted")
        fixture_rows = [
            ("a", 0, "FAR", 2), ("b", 0, "FAR", 2), ("c", 0, "FAR", 2),
            ("d", 0, "NEAR", 1), ("e", 1, "FAR", 2), ("f", 1, "FAR", 2),
        ]
        _, fixture_map, _ = target_derangement(
            np.asarray([x[0] for x in fixture_rows]), np.asarray([x[1] for x in fixture_rows]),
            np.asarray([x[2] for x in fixture_rows]), np.asarray([x[3] for x in fixture_rows]), 53602,
        )
        payload = (json.dumps(fixture_map, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        if hashlib.sha256(payload).hexdigest() != self.config["posterior"]["target_shuffle"]["fixture_sha256"]:
            raise CheckFailure("target shuffle canonical fixture drifted")

    def _mass_for_logits(self, posterior_name: str, logits: np.ndarray) -> np.ndarray:
        if posterior_name == "marginal":
            return marginal_mass(self.posterior_states["marginal"]["real"], logits)
        return posterior_mass(
            logits, self.posterior_arrays["joint_real__final_theta"], "joint_full"
        )

    def p6_hard_map_binding(self) -> None:
        for posterior_name in ("marginal", "joint"):
            public = public_design(
                self.public["ensemble_logits"], self.public["per_seed_logits"],
                self.mass[posterior_name]["real"], self.utility, self.penalty,
            )
            self.select_public[posterior_name] = public
            for key in ("map_set_index", "map_set", "hard_actions"):
                assert_equal(
                    public[key], self.selection_scores[f"{posterior_name}__{key}"],
                    f"{posterior_name} hard MAP {key} drifted",
                )
            assert_close(
                public["map_set_mass"], self.selection_scores[f"{posterior_name}__map_set_mass"],
                f"{posterior_name} hard MAP mass drifted",
            )
            assert_equal(
                public["hard_actions"], self.action_arrays[f"{posterior_name}__hard_actions"],
                f"{posterior_name} frozen HARD actions drifted",
            )

    def p7_contextual_design(self) -> None:
        if tuple(self.config["reader"]["feature_names"]) != FEATURE_NAMES:
            raise CheckFailure("feature-name contract drifted")
        schema = read_json(self.root / "policy_fit/feature_schema.json")
        if tuple(schema["feature_names"]) != FEATURE_NAMES or schema["hard_adapter"] != "HARD_MAP_SET":
            raise CheckFailure("feature schema file drifted")
        for posterior_name in ("marginal", "joint"):
            public = self.select_public[posterior_name]
            for key in ("posterior_actions", "disagreement", "weights"):
                assert_equal(
                    public[key], self.selection_scores[f"{posterior_name}__{key}"],
                    f"{posterior_name} contextual {key} drifted",
                )
            assert_close(
                public["design"], self.selection_scores[f"{posterior_name}__design"],
                f"{posterior_name} contextual design drifted",
            )
            active_tokens = public["disagreement"].any(axis=1)
            assert_close(
                public["weights"].sum(axis=1)[active_tokens],
                np.ones(active_tokens.sum()),
                f"{posterior_name} token weights drifted",
            )
            fit_mass = self._mass_for_logits(
                posterior_name, self.policy["ensemble_logits"]
            )
            fit_public = public_design(
                self.policy["ensemble_logits"], self.policy["per_seed_logits"],
                fit_mass, self.utility, self.penalty,
            )
            self.fit_public[posterior_name] = fit_public
            assert_close(
                fit_public["design"], self.fit_scores[f"{posterior_name}__design"],
                f"{posterior_name} fit design drifted",
            )

    def p8_model_states(self) -> None:
        target = self.policy["target"].astype(bool)
        expected_state_arrays: dict[str, np.ndarray] = {}
        for posterior_name in ("marginal", "joint"):
            public = self.fit_public[posterior_name]
            active = public["disagreement"]
            hard_values = regret(public["hard_actions"], target, self.utility, self.penalty)
            candidate_values = regret(public["posterior_actions"], target, self.utility, self.penalty)
            gain = hard_values - candidate_values
            harm = gain < -1e-12
            incompatibility = ~target[np.arange(len(target))[:, None], public["posterior_actions"]]
            for key, value in (("gain", gain), ("harm", harm), ("incompatibility", incompatibility)):
                assert_close(
                    value.astype(float), self.fit_scores[f"{posterior_name}__{key}"].astype(float),
                    f"{posterior_name} fit target {key} drifted",
                )
            x = public["design"][active]
            weights = public["weights"][active]
            true_states = self.policy_states[posterior_name]["true"]["states"]
            if (
                true_states["proposer"].get("alpha") != 1.0
                or true_states["proposer"].get("intercept_penalized") is not False
                or true_states["proposer"].get("solver") != "numpy.linalg.solve"
            ):
                raise CheckFailure(f"{posterior_name} Ridge recipe drifted")
            if (
                true_states["harm"].get("contract") != GUARD_CONTRACT
                or true_states["incompatibility"].get("contract") != GUARD_CONTRACT
                or true_states["harm"].get("target_name") != "harm"
                or true_states["incompatibility"].get("target_name") != "incompatibility"
                or true_states["harm"].get("classes") != [0, 1]
                or true_states["incompatibility"].get("classes") != [0, 1]
            ):
                raise CheckFailure(f"{posterior_name} guard recipe drifted")
            compare_fitted_state(true_states["proposer"], fit_ridge(x, gain[active], weights), f"{posterior_name} true proposer")
            compare_fitted_state(true_states["harm"], fit_guard(x, harm[active], weights), f"{posterior_name} true harm")
            compare_fitted_state(true_states["incompatibility"], fit_guard(x, incompatibility[active], weights), f"{posterior_name} true incompatibility")
            for model_name in ("proposer", "harm", "incompatibility"):
                state = true_states[model_name]
                prefix = f"{posterior_name}__true__{model_name}"
                for field in ("mean", "scale", "coef"):
                    expected_state_arrays[f"{prefix}__{field}"] = np.asarray(state[field], dtype=np.float64)
                expected_state_arrays[f"{prefix}__intercept"] = np.asarray([state["intercept"]], dtype=np.float64)
                if "n_iter" in state:
                    expected_state_arrays[f"{prefix}__n_iter"] = np.asarray(state["n_iter"], dtype=np.int64)
                score = np.full(active.shape, np.nan)
                score[active] = score_state(true_states[model_name], x)
                assert_close(
                    score[active], self.fit_scores[f"{posterior_name}__true__{model_name}"][active],
                    f"{posterior_name} true {model_name} portable score drifted",
                )
            for seed, control in zip(
                CONTROL_SEEDS, self.policy_states[posterior_name]["controls"], strict=True
            ):
                states = control["states"]
                prefix = f"{posterior_name}__control_{seed}"
                if (
                    states["proposer"].get("solver") != "numpy.linalg.solve"
                    or states["proposer"].get("intercept_penalized") is not False
                    or states["harm"].get("target_name") != "harm"
                    or states["incompatibility"].get("target_name") != "incompatibility"
                    or states["harm"].get("classes") != [0, 1]
                    or states["incompatibility"].get("classes") != [0, 1]
                ):
                    raise CheckFailure(f"{prefix} model recipe drifted")
                for model_name, state in states.items():
                    model_prefix = f"{prefix}__{model_name}"
                    for field in ("mean", "scale", "coef"):
                        expected_state_arrays[f"{model_prefix}__{field}"] = np.asarray(state[field], dtype=np.float64)
                    expected_state_arrays[f"{model_prefix}__intercept"] = np.asarray([state["intercept"]], dtype=np.float64)
                    if "n_iter" in state:
                        expected_state_arrays[f"{model_prefix}__n_iter"] = np.asarray(state["n_iter"], dtype=np.int64)
                control_gain = self.control_arrays[f"{prefix}__gain"]
                control_harm = self.control_arrays[f"{prefix}__harm"]
                control_incompat = self.control_arrays[f"{prefix}__incompatibility"]
                compare_fitted_state(states["proposer"], fit_ridge(x, control_gain[active], weights), f"{prefix} proposer")
                compare_fitted_state(states["harm"], fit_guard(x, control_harm[active], weights), f"{prefix} harm")
                compare_fitted_state(states["incompatibility"], fit_guard(x, control_incompat[active], weights), f"{prefix} incompatibility")
        saved_state_arrays = load_npz(self.root / "policy_fit/state_arrays.npz")
        if set(saved_state_arrays) != set(expected_state_arrays):
            raise CheckFailure("portable state-array inventory drifted")
        for key, value in expected_state_arrays.items():
            assert_close(value, saved_state_arrays[key], f"portable state array drifted: {key}", 0.0)

    def p9_selection(self) -> None:
        target = self.truth["target"].astype(bool)
        for posterior_name in ("marginal", "joint"):
            public = self.select_public[posterior_name]
            states = self.policy_states[posterior_name]["true"]["states"]
            scores = score_triplet(states, public)
            self.select_scores[posterior_name] = scores
            for model_name in scores:
                active = public["disagreement"]
                assert_close(
                    scores[model_name][active],
                    self.selection_scores[f"{posterior_name}__true__{model_name}"][active],
                    f"{posterior_name} decision score {model_name} drifted",
                )
            actions_rows = []
            override_rows = []
            metadata = []
            for qp in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975):
                for qh in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
                    for qi in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
                        threshold = thresholds(scores, public["disagreement"], (qp, qh, qi))
                        actions, override = apply_thresholds(scores, public, threshold)
                        actions_rows.append(actions); override_rows.append(override)
                        metadata.append({"kind": "contextual", **threshold})
            actions_rows.append(public["hard_actions"]); override_rows.append(np.zeros_like(public["disagreement"]))
            metadata.append({
                "kind": "hard_only", "proposer_quantile": 2.0, "harm_quantile": 2.0,
                "incompatibility_quantile": 2.0, "proposer_threshold": None,
                "harm_threshold": None, "incompatibility_threshold": None,
            })
            actions_array = np.asarray(actions_rows); override_array = np.asarray(override_rows)
            assert_equal(actions_array, self.candidate_arrays[f"{posterior_name}__actions"], f"{posterior_name} candidate actions drifted")
            assert_equal(override_array, self.candidate_arrays[f"{posterior_name}__override"], f"{posterior_name} candidate overrides drifted")
            hard_values = regret(public["hard_actions"], target, self.utility, self.penalty)
            rows = []
            for index, actions in enumerate(actions_array):
                metric = action_metrics(actions, target, self.utility, self.penalty)
                rows.append({
                    "mean_regret": float(metric["regret"].mean()),
                    "incompatibility_rate": float(metric["incompatibility_by_policy"].mean()),
                    "harm_rate": float(np.mean(metric["regret_by_policy"] > hard_values + 1e-12)),
                    "authorized_rows": int(override_array[index].sum()),
                    **metadata[index],
                })
            chosen = min(range(344), key=lambda i: (
                rows[i]["mean_regret"], rows[i]["incompatibility_rate"], rows[i]["harm_rate"],
                -rows[i]["authorized_rows"], rows[i]["proposer_quantile"], rows[i]["harm_quantile"], rows[i]["incompatibility_quantile"],
            ))
            freeze = self.selection_freeze[posterior_name]
            if (
                chosen != freeze["selected_index"]
                or freeze["candidate_metadata"] != metadata
                or any(freeze["selected"].get(key) != value for key, value in rows[chosen].items())
            ):
                raise CheckFailure(f"{posterior_name} selection key/grid drifted")
            for key in ("mean_regret", "incompatibility_rate", "harm_rate", "authorized_rows"):
                observed = self.candidate_arrays[f"{posterior_name}__{key}"]
                expected = np.asarray([row[key] for row in rows])
                assert_close(observed.astype(float), expected.astype(float), f"{posterior_name} candidate metric {key} drifted", 0.0)
            assert_equal(actions_array[chosen], self.action_arrays[f"{posterior_name}__contextual_actions"], f"{posterior_name} selected actions drifted")
            assert_equal(override_array[chosen], self.action_arrays[f"{posterior_name}__true_override"], f"{posterior_name} selected override drifted")

    def p10_matched_controls(self) -> None:
        target = self.policy["target"].astype(bool)
        tokens = self.policy["pair_token"].astype(str)
        for posterior_name in ("marginal", "joint"):
            fit_public = self.fit_public[posterior_name]
            active = fit_public["disagreement"]
            hard_values = regret(fit_public["hard_actions"], target, self.utility, self.penalty)
            candidate_values = regret(fit_public["posterior_actions"], target, self.utility, self.penalty)
            gain = hard_values - candidate_values
            harm = gain < -1e-12
            incompatibility = ~target[np.arange(len(target))[:, None], fit_public["posterior_actions"]]
            map_hashes = set()
            target_hashes = set()
            decision_public = self.select_public[posterior_name]
            true_override = self.action_arrays[f"{posterior_name}__true_override"]
            valid_masks = []
            controls = self.policy_states[posterior_name]["controls"]
            if [int(row["seed"]) for row in controls] != list(CONTROL_SEEDS):
                raise CheckFailure(f"{posterior_name} control seeds drifted")
            for control in controls:
                seed = int(control["seed"])
                prefix = f"{posterior_name}__control_{seed}"
                mapping = matched_map(gain, harm, incompatibility, active, tokens, seed)
                assert_equal(mapping, self.control_arrays[f"{prefix}__mapping"], f"{prefix} mapping drifted")
                rows, policies = np.where(active)
                donors = mapping[rows, policies]
                transported_gain = gain.copy(); transported_gain[rows, policies] = gain[donors, policies]
                transported_harm = harm.copy(); transported_harm[rows, policies] = harm[donors, policies]
                transported_incompat = incompatibility.copy(); transported_incompat[rows, policies] = incompatibility[donors, policies]
                for name, value in (("gain", transported_gain), ("harm", transported_harm), ("incompatibility", transported_incompat)):
                    assert_close(value.astype(float), self.control_arrays[f"{prefix}__{name}"].astype(float), f"{prefix} transported {name} drifted")
                counts = active.sum(axis=1)
                semantic_rows = [
                    (str(tokens[row]), int(policy), str(tokens[mapping[row, policy]]), int(counts[row]))
                    for row, policy in zip(rows.tolist(), policies.tolist(), strict=True)
                ]
                semantic_rows.sort(key=lambda item: (item[0].encode(), item[1]))
                semantic_payload = json.dumps(
                    semantic_rows, ensure_ascii=False, sort_keys=False,
                    separators=(",", ":"), allow_nan=False,
                ).encode()
                mapping_hash = hashlib.sha256(semantic_payload).hexdigest()
                triplet = hashlib.sha256()
                triplet.update(np.ascontiguousarray(transported_gain[active]).tobytes())
                triplet.update(np.ascontiguousarray(transported_harm[active]).tobytes())
                triplet.update(np.ascontiguousarray(transported_incompat[active]).tobytes())
                target_hash = triplet.hexdigest()
                diagnostics = control["diagnostics"]
                if diagnostics["mapping_sha256"] != mapping_hash or diagnostics["target_triplet_sha256"] != target_hash:
                    raise CheckFailure(f"{prefix} declared digest drifted")
                permutable = np.zeros_like(active)
                stratum_rows = []
                total_hamming = 0
                for policy in range(active.shape[1]):
                    for count in sorted(np.unique(counts[active[:, policy]]).tolist()):
                        indices = np.asarray(sorted(np.flatnonzero(active[:, policy] & (counts == count)).tolist(), key=lambda index: tokens[index].encode()))
                        singleton = len(indices) == 1
                        if not singleton:
                            permutable[indices, policy] = True
                        signatures = np.column_stack([
                            np.ascontiguousarray(gain[indices, policy], dtype="<f8").view("<u8"),
                            harm[indices, policy].astype(np.uint64),
                            incompatibility[indices, policy].astype(np.uint64),
                        ])
                        selected_cost = 0 if singleton else int(np.sum(signatures != signatures[[np.where(indices == mapping[index, policy])[0][0] for index in indices]]))
                        total_hamming += selected_cost
                        stratum_rows.append({
                            "policy_index": int(policy), "disagreement_count": int(count),
                            "rows": int(len(indices)), "singleton": singleton,
                            "maximum_hamming": selected_cost,
                        })
                expected_diag = {
                    "seed": seed, "active_rows": int(active.sum()),
                    "strata": len(stratum_rows), "singleton_rows": int(np.sum(active & ~permutable)),
                    "permutable_fraction": float(permutable[active].mean()),
                    "maximum_hamming": total_hamming, "mapping_sha256": mapping_hash,
                    "target_triplet_sha256": target_hash, "stratum_rows": stratum_rows,
                }
                if diagnostics != expected_diag:
                    raise CheckFailure(f"{prefix} diagnostics drifted")
                map_hashes.add(mapping_hash)
                target_hashes.add(target_hash)
                scores = score_triplet(control["states"], decision_public)
                selected = self.selection_freeze[posterior_name]["selected"]
                if selected["kind"] == "hard_only":
                    matched = {
                        "actions": decision_public["hard_actions"],
                        "selected": np.zeros_like(true_override),
                        "authorized_universe": np.zeros_like(true_override),
                        "match_valid": np.ones(len(true_override), dtype=bool),
                        "requested_k": np.zeros(len(true_override), dtype=np.int64),
                    }
                else:
                    q = (float(selected["proposer_quantile"]), float(selected["harm_quantile"]), float(selected["incompatibility_quantile"]))
                    threshold = thresholds(scores, decision_public["disagreement"], q)
                    frozen = self.selection_freeze[posterior_name]["control_thresholds"][str(seed)]
                    for key in ("proposer_threshold", "harm_threshold", "incompatibility_threshold"):
                        assert_close(np.asarray([threshold[key]]), np.asarray([frozen[key]]), f"{prefix} threshold drifted")
                    matched = matched_actions(true_override, scores, decision_public, threshold)
                for key, value in matched.items():
                    assert_equal(value, self.matches[f"{prefix}__{key}"], f"{prefix} match {key} drifted")
                valid_masks.append(matched["match_valid"])
            if len(map_hashes) != 5 or len(target_hashes) != 5:
                raise CheckFailure(f"{posterior_name} controls lack diversity")
            u_true = true_override.any(axis=1)
            common = u_true.copy()
            for valid in valid_masks:
                common &= valid
            assert_equal(u_true, self.matches[f"{posterior_name}__u_true"], f"{posterior_name} U_true drifted")
            assert_equal(common, self.matches[f"{posterior_name}__u_common"], f"{posterior_name} U_common drifted")

    @staticmethod
    def _paired(left: np.ndarray, right: np.ndarray, boot: np.ndarray) -> dict[str, float | int]:
        delta = left - right
        draws = delta[boot].mean(axis=1)
        low, high = np.percentile(draws, [2.5, 97.5])
        return {
            "mean_diff": float(delta.mean()), "ci95_low": float(low),
            "ci95_high": float(high), "n_tokens": int(len(delta)), "n_boot": 5000,
        }

    @staticmethod
    def _status(summary: dict[str, Any], allow_zero: bool) -> str:
        satisfied = summary["ci95_high"] <= 0.0 if allow_zero else summary["ci95_high"] < 0.0
        if satisfied:
            return "CONDITION_SATISFIED"
        if summary["ci95_low"] > 0.0:
            return "ADVERSE"
        return "NOT_RESOLVED"

    def p11_cell_and_estimand_parity(self) -> None:
        target = self.truth["target"].astype(bool)
        global_boot = np.random.Generator(np.random.PCG64(53641)).integers(
            0, len(target), size=(5000, len(target)), dtype=np.int64
        )
        assert_equal(global_boot, self.boot["global_pair_token_index"], "global bootstrap drifted")
        set_rows: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        action_rows: dict[str, dict[str, np.ndarray]] = {}
        expected_raw: dict[str, np.ndarray] = {
            "pair_token": self.public["pair_token"], "target": target
        }
        for posterior_name in ("marginal", "joint"):
            set_rows[posterior_name] = {}
            for role in ("real", "target_shuffled"):
                metric = set_metrics(self.mass[posterior_name][role], target)
                set_rows[posterior_name][role] = metric
                for key, value in metric.items():
                    assert_close(value, self.raw[f"{posterior_name}__{role}__{key}"], f"{posterior_name} {role} raw {key} drifted")
                    expected_raw[f"{posterior_name}__{role}__{key}"] = value
            hard = action_metrics(self.action_arrays[f"{posterior_name}__hard_actions"], target, self.utility, self.penalty)
            contextual = action_metrics(self.action_arrays[f"{posterior_name}__contextual_actions"], target, self.utility, self.penalty)
            action_rows[f"{posterior_name}_hard"] = hard
            action_rows[f"{posterior_name}_contextual"] = contextual
            for reader, metric in (("hard", hard), ("contextual", contextual)):
                for key, value in metric.items():
                    assert_close(value.astype(float), self.raw[f"{posterior_name}__{reader}__{key}"].astype(float), f"{posterior_name} {reader} raw {key} drifted")
                    expected_raw[f"{posterior_name}__{reader}__{key}"] = value
            for seed in CONTROL_SEEDS:
                control_metric = action_metrics(
                    self.matches[f"{posterior_name}__control_{seed}__actions"],
                    target, self.utility, self.penalty,
                )
                for key, value in control_metric.items():
                    raw_key = f"{posterior_name}__control_{seed}__{key}"
                    assert_close(value.astype(float), self.raw[raw_key].astype(float), f"control raw drifted: {raw_key}")
                    expected_raw[raw_key] = value
            support = self.matches[f"{posterior_name}__u_common"]
            if support.any():
                seed = 53642 if posterior_name == "marginal" else 53643
                expected = np.random.Generator(np.random.PCG64(seed)).integers(
                    0, int(support.sum()), size=(5000, int(support.sum())), dtype=np.int64
                )
                assert_equal(expected, self.boot[f"{posterior_name}__common_index"], f"{posterior_name} common bootstrap drifted")
                assert_equal(self.public["pair_token"][support], self.boot[f"{posterior_name}__common_pair_token"], f"{posterior_name} common token order drifted")
        if set(self.raw) != set(expected_raw):
            raise CheckFailure("diagnostic raw inventory drifted")
        assert_equal(self.public["pair_token"], self.boot["global_pair_token"], "global bootstrap token order drifted")
        if str(self.boot["global_index_sha256_utf8"][0]) != array_digest(global_boot):
            raise CheckFailure("global bootstrap digest drifted")
        expected_boot_keys = {
            "global_pair_token_index", "global_pair_token", "global_index_sha256_utf8",
            "marginal__common_index", "marginal__common_pair_token",
            "joint__common_index", "joint__common_pair_token",
        }
        if set(self.boot) != expected_boot_keys:
            raise CheckFailure("bootstrap artifact inventory drifted")
        expected_specs: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray | None, bool, bool]] = [
            ("SET_JOINT_NLL", "joint_minus_marginal", set_rows["joint"]["real"]["exact_set_nll"], set_rows["marginal"]["real"]["exact_set_nll"], global_boot, False, False),
            ("SET_JOINT_BRIER", "joint_minus_marginal", set_rows["joint"]["real"]["marginal_brier"], set_rows["marginal"]["real"]["marginal_brier"], global_boot, True, False),
        ]
        for posterior_name in ("marginal", "joint"):
            expected_specs.append(("SET_SHUFFLE", posterior_name, set_rows[posterior_name]["real"]["exact_set_nll"], set_rows[posterior_name]["target_shuffled"]["exact_set_nll"], global_boot, False, False))
            expected_specs.extend([
                ("READER_REGRET", posterior_name, action_rows[f"{posterior_name}_contextual"]["regret"], action_rows[f"{posterior_name}_hard"]["regret"], global_boot, False, False),
                ("READER_COMPAT", posterior_name, action_rows[f"{posterior_name}_contextual"]["incompatibility"], action_rows[f"{posterior_name}_hard"]["incompatibility"], global_boot, True, False),
                ("READER_WORST", posterior_name, action_rows[f"{posterior_name}_contextual"]["worst_regret"], action_rows[f"{posterior_name}_hard"]["worst_regret"], global_boot, True, False),
            ])
            support = self.matches[f"{posterior_name}__u_common"]
            controls = []
            for seed in CONTROL_SEEDS:
                actions = self.matches[f"{posterior_name}__control_{seed}__actions"]
                controls.append(action_metrics(actions, target, self.utility, self.penalty)["regret"])
            status = self.selection_freeze[posterior_name]["common_support_status"]
            if status == "EVALUABLE":
                expected_specs.append(("READER_CONTROL", posterior_name, action_rows[f"{posterior_name}_contextual"]["regret"][support], np.mean(np.stack(controls)[:, support], axis=0), self.boot[f"{posterior_name}__common_index"], False, False))
            else:
                expected_specs.append(("READER_CONTROL", posterior_name, np.asarray([]), np.asarray([]), None, False, False))
        expected_specs.append((
            "FACTOR_INTERACTION", "joint_reader_delta_minus_marginal_reader_delta",
            action_rows["joint_contextual"]["regret"] - action_rows["joint_hard"]["regret"],
            action_rows["marginal_contextual"]["regret"] - action_rows["marginal_hard"]["regret"],
            global_boot, True, True,
        ))
        actual = {(row["id"], row["instance"]): row for row in self.estimands["rows"]}
        if len(actual) != len(expected_specs):
            raise CheckFailure("estimand row inventory drifted")
        for row_id, instance, left, right, boot, allow_zero, descriptive in expected_specs:
            row = actual.get((row_id, instance))
            if row is None:
                raise CheckFailure(f"estimand missing: {row_id}/{instance}")
            if row.get("orientation") != "left_minus_right":
                raise CheckFailure(f"estimand orientation drifted: {row_id}/{instance}")
            if boot is None:
                if row["status"] != "NOT_EVALUABLE":
                    raise CheckFailure(f"estimand precedence drifted: {row_id}/{instance}")
                continue
            summary = self._paired(left, right, boot)
            for key, value in summary.items():
                assert_close(np.asarray([row[key]], float), np.asarray([value], float), f"estimand {row_id}/{instance} {key} drifted", 2e-12)
            expected_status = "DESCRIPTIVE_ONLY" if descriptive else self._status(summary, allow_zero)
            if row["status"] != expected_status:
                raise CheckFailure(f"estimand status drifted: {row_id}/{instance}")
        sensitivity = load_npz(self.root / "evaluate_fixture/sensitivity_arrays.npz")
        expected_sensitivity: dict[str, np.ndarray] = {}
        sensitivity_rows = []
        cards = self.public["cardinality"].astype(np.int64)
        for checkpoint_index, checkpoint in enumerate(self.config["checkpoint_epochs"]):
            checkpoint_logits = self.public["per_seed_logits"][checkpoint_index]
            for posterior_name in ("marginal", "joint"):
                mass = self._mass_for_logits(posterior_name, checkpoint_logits)
                pdata = public_design(
                    checkpoint_logits, self.public["per_seed_logits"], mass,
                    self.utility, self.penalty,
                )
                scores = score_triplet(
                    self.policy_states[posterior_name]["true"]["states"], pdata
                )
                selected = self.selection_freeze[posterior_name]["selected"]
                contextual = (
                    pdata["hard_actions"]
                    if selected["kind"] == "hard_only"
                    else apply_thresholds(scores, pdata, selected)[0]
                )
                for reader, actions in (("hard", pdata["hard_actions"]), ("contextual", contextual)):
                    metric = action_metrics(actions, target, self.utility, self.penalty)
                    prefix = f"checkpoint_{checkpoint}__{posterior_name}__{reader}"
                    expected_sensitivity[f"{prefix}__actions"] = actions
                    expected_sensitivity[f"{prefix}__regret_by_policy"] = metric["regret_by_policy"]
                    expected_sensitivity[f"{prefix}__incompatibility_by_policy"] = metric["incompatibility_by_policy"]
                    for card in sorted(np.unique(cards).tolist()):
                        mask = cards == card
                        sensitivity_rows.append({
                            "checkpoint_epoch": int(checkpoint),
                            "checkpoint_is_population_seed": False,
                            "posterior": posterior_name, "reader": reader,
                            "cardinality": int(card), "n_tokens": int(mask.sum()),
                            "mean_regret": float(metric["regret"][mask].mean()),
                            "mean_incompatibility": float(metric["incompatibility"][mask].mean()),
                            "mean_accuracy": float(metric["accuracy"][mask].mean()),
                        })
        if set(sensitivity) != set(expected_sensitivity):
            raise CheckFailure("sensitivity raw inventory drifted")
        for key, value in expected_sensitivity.items():
            assert_close(value.astype(float), sensitivity[key].astype(float), f"sensitivity array drifted: {key}", 0.0)
        diagnostic_metrics = read_json(self.root / "evaluate_fixture/diagnostic_metrics.json")
        if diagnostic_metrics.get("checkpoint_sensitivity") != sensitivity_rows:
            raise CheckFailure("sensitivity summary rows drifted")
        duplications = read_json(self.root / "evaluate_fixture/cell_duplications.json")
        cell_actions = {
            "marginal_hard": self.action_arrays["marginal__hard_actions"],
            "marginal_contextual": self.action_arrays["marginal__contextual_actions"],
            "joint_hard": self.action_arrays["joint__hard_actions"],
            "joint_contextual": self.action_arrays["joint__contextual_actions"],
        }
        expected_duplications = []
        for left, right in itertools.combinations(sorted(cell_actions), 2):
            left_actions, right_actions = cell_actions[left], cell_actions[right]
            expected_duplications.append({
                "left": left, "right": right,
                "actions_exact": bool(np.array_equal(left_actions, right_actions)),
                "action_position_equal_fraction": float(np.mean(left_actions == right_actions)),
                "regret_exact": bool(np.array_equal(action_rows[left]["regret_by_policy"], action_rows[right]["regret_by_policy"])),
            })
        if duplications != {
            "schema_version": "proportional-cell-duplications-v1",
            "cells_retained_even_if_equal": True,
            "comparisons": expected_duplications,
        }:
            raise CheckFailure("cell duplication inventory drifted")
        patterns = self.estimands.get("patterns", {})
        def satisfied(row_id: str, instance: str | None = None) -> bool:
            values = [
                row for row in self.estimands["rows"]
                if row["id"] == row_id and (instance is None or row["instance"] == instance)
            ]
            return bool(values) and all(row["status"] == "CONDITION_SATISFIED" for row in values)
        expected_patterns = {
            "JOINT_PATTERN_PRESENT": all(
                (satisfied("SET_JOINT_NLL"), satisfied("SET_JOINT_BRIER"), satisfied("SET_SHUFFLE", "joint"))
            ),
            "CONTEXTUAL_PATTERN_PRESENT": {
                name: all(satisfied(row_id, name) for row_id in ("READER_REGRET", "READER_COMPAT", "READER_WORST", "READER_CONTROL"))
                for name in ("marginal", "joint")
            },
            "interpretation": "OPENED_DATA_IMPLEMENTATION_DIAGNOSTIC_ONLY",
        }
        if patterns != expected_patterns:
            raise CheckFailure("diagnostic pattern logic drifted")

    def p12_raw_and_replay(self) -> None:
        manifest = read_json(self.root / "artifact_manifest.json")
        actual = {
            path.relative_to(self.root).as_posix(): path
            for path in self.root.rglob("*")
            if path.is_file() and path.name != "artifact_manifest.json"
        }
        recorded = {row["path"]: row for row in manifest["files"]}
        if set(actual) != set(recorded):
            raise CheckFailure("artifact manifest inventory drifted")
        for relative, path in actual.items():
            row = recorded[relative]
            if row["sha256"] != sha256_file(path) or row["bytes"] != path.stat().st_size:
                raise CheckFailure(f"artifact manifest hash/size drifted: {relative}")
            if path.suffix == ".npz":
                with zipfile.ZipFile(path) as archive:
                    infos = archive.infolist()
                    names = [info.filename for info in infos]
                    if names != sorted(names) or any(not name.endswith(".npy") for name in names):
                        raise CheckFailure(f"NPZ key order/layout is noncanonical: {relative}")
                    for info in infos:
                        if (
                            info.date_time != (1980, 1, 1, 0, 0, 0)
                            or info.compress_type != zipfile.ZIP_DEFLATED
                            or info.external_attr != (0o600 << 16)
                        ):
                            raise CheckFailure(f"NPZ ZIP metadata is noncanonical: {relative}/{info.filename}")
                arrays = load_npz(path)
                if any(value.dtype.hasobject for value in arrays.values()):
                    raise CheckFailure(f"NPZ object array is forbidden: {relative}")
        if self.reference is not None:
            current = {p.relative_to(self.root).as_posix(): p for p in self.root.rglob("*") if p.is_file() and p.relative_to(self.root).as_posix() not in REPLAY_EXCLUDED}
            reference = {p.relative_to(self.reference).as_posix(): p for p in self.reference.rglob("*") if p.is_file() and p.relative_to(self.reference).as_posix() not in REPLAY_EXCLUDED}
            if set(current) != set(reference) or any(sha256_file(current[k]) != sha256_file(reference[k]) for k in current):
                raise CheckFailure("replay differs byte-exactly")
            if read_json(self.root / "replay_receipt.json")["byte_exact"] is not True:
                raise CheckFailure("replay receipt does not assert exactness")

    def p13_claim_boundary(self) -> None:
        claims = self.config["fixed_claims"]
        if any(claims[key] for key in ("fresh_draw_created_or_opened", "monitor_or_lockbox_opened", "gpu_used_or_queried", "torch_imported", "architecture_promoted")):
            raise CheckFailure("fixed claim boundary drifted")
        if claims["scientific_decision"] is not None or claims["decision_authority"] != "user":
            raise CheckFailure("scientific decision authority drifted")
        if self.config["maximum_status"] != "RUNNER_PREFLIGHT_VALID":
            raise CheckFailure("maximum status drifted")
        report = (self.root / "REPORT.md").read_text(encoding="utf-8")
        if report != render_report(self.estimands):
            raise CheckFailure("report differs from canonical diagnostic rendering")
        status_documents = [
            self.selection_freeze,
            read_json(self.root / "apply_fixture/action_freeze.json"),
            self.estimands,
            read_json(self.root / "evaluate_fixture/diagnostic_metrics.json"),
        ]
        if any(document.get("status") != "OPENED_DATA_IMPLEMENTATION_DIAGNOSTIC" for document in status_documents):
            raise CheckFailure("structured diagnostic status drifted")
        action_freeze = status_documents[1]
        if action_freeze.get("truth_keys_received_by_applier") != [] or not action_freeze.get("posteriors"):
            raise CheckFailure("target-blind action boundary drifted")

    def p14_cost_contract(self) -> None:
        runtime = read_json(self.root / "runtime.json")
        if runtime["peak_rss_bytes"] > self.config["budgets"]["peak_rss_bytes"]:
            raise CheckFailure("peak RSS budget exceeded")
        if runtime["wall_seconds"] > self.config["budgets"]["runner_primary_plus_replay_seconds"]:
            raise CheckFailure("single runner wall budget exceeded")
        receipt = read_json(self.root / "replay_receipt.json")
        if receipt.get("mode") == "replay" and (
            not receipt.get("within_hard_budget")
            or receipt["primary_plus_replay_wall_seconds"] > self.config["budgets"]["runner_primary_plus_replay_seconds"]
        ):
            raise CheckFailure("primary plus replay wall budget exceeded")
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        if peak > self.config["budgets"]["peak_rss_bytes"]:
            raise CheckFailure("checker peak RSS budget exceeded")


PREDICATES: tuple[tuple[str, str, str], ...] = (
    ("P1_SOURCE_AND_SCOPE", "SOURCE_OR_SCOPE_INVALID", "p1_source_and_scope"),
    ("P2_PREPARED_PHASES", "PHASE_BUNDLE_INVALID", "p2_prepared_phases"),
    ("P3_MARGINAL_NATIVE", "MARGINAL_RECIPE_INVALID", "p3_marginal_native"),
    ("P4_JOINT_NATIVE", "JOINT_RECIPE_INVALID", "p4_joint_native"),
    ("P5_TARGET_SHUFFLE", "TARGET_SHUFFLE_INVALID", "p5_target_shuffle"),
    ("P6_HARD_MAP_BINDING", "HARD_POSTERIOR_BINDING_INVALID", "p6_hard_map_binding"),
    ("P7_CONTEXTUAL_DESIGN", "CONTEXTUAL_DESIGN_INVALID", "p7_contextual_design"),
    ("P8_MODEL_STATES", "CONTEXTUAL_STATE_INVALID", "p8_model_states"),
    ("P9_SELECTION", "SELECTION_PROTOCOL_INVALID", "p9_selection"),
    ("P10_MATCHED_CONTROLS", "MATCHED_CONTROL_INVALID", "p10_matched_controls"),
    ("P11_CELL_AND_ESTIMAND_PARITY", "CELL_ESTIMAND_MISMATCH", "p11_cell_and_estimand_parity"),
    ("P12_RAW_AND_REPLAY", "RAW_OR_REPLAY_INVALID", "p12_raw_and_replay"),
    ("P13_CLAIM_BOUNDARY", "CLAIM_BOUNDARY_INVALID", "p13_claim_boundary"),
    ("P14_COST_CONTRACT", "COST_CONTRACT_INVALID", "p14_cost_contract"),
)


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    results = []
    try:
        checker = Checker(args.artifact, args.reference)
    except Exception as error:
        payload = {
            "status": "FAIL",
            "reason_code": "RAW_OR_REPLAY_INVALID",
            "detail": f"checker bootstrap failed: {type(error).__name__}: {error}",
            "predicates": [],
        }
        print(json.dumps(payload, sort_keys=True))
        return 1
    failed = False
    for predicate_id, reason_code, method_name in PREDICATES:
        if failed:
            results.append({"id": predicate_id, "status": "NOT_RUN", "reason_code": None})
            continue
        try:
            getattr(checker, method_name)()
        except Exception as error:
            failed = True
            results.append(
                {
                    "id": predicate_id,
                    "status": "FAIL",
                    "reason_code": reason_code,
                    "detail": f"{type(error).__name__}: {error}",
                }
            )
        else:
            results.append({"id": predicate_id, "status": "PASS", "reason_code": None})
    elapsed = time.monotonic() - started
    budget = float(checker.config["budgets"]["auxiliary_checker_and_mutations_seconds"])
    if elapsed > budget and not failed:
        failed = True
        results[-1] = {
            "id": "P14_COST_CONTRACT",
            "status": "FAIL",
            "reason_code": "COST_CONTRACT_INVALID",
            "detail": f"checker exceeded auxiliary budget: {elapsed} > {budget}",
        }
    payload = {
        "schema_version": "proportional-independent-check-v1",
        "status": "FAIL" if failed else "PASS",
        "reason_code": next(
            (row["reason_code"] for row in results if row["status"] == "FAIL"), None
        ),
        "predicates": results,
        "wall_seconds": elapsed,
        "torch_imported": "torch" in sys.modules,
        "gpu_used_or_queried": False,
    }
    print(json.dumps(payload, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
