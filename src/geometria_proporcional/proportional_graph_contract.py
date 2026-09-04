"""CPU contract for the local proportional-coherence graph benchmark.

The model-facing observation and the private synthetic authority are separate
types.  Classical solvers in this module establish the benchmark mechanics;
they do not adjudicate a neural architecture.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
from scipy.linalg import null_space

SCHEMA_VERSION = "proportional-graph-contract-v1"
PUBLIC_ARRAY_FIELDS = frozenset(
    {
        "n_nodes",
        "edge_index",
        "observed_log_ratio",
        "edge_valid",
        "path_index",
        "path_sign",
        "path_valid",
        "edge_variance",
    }
)
FORBIDDEN_PUBLIC_FIELDS = frozenset(
    {
        "q",
        "x_true",
        "clean_log_ratio",
        "base_noise",
        "corruption_delta",
        "causal_corruption_mask",
        "corruption_mechanism",
        "seed",
        "lineage_id",
        "split",
        "master_id",
        "view_id",
        "oracle_weight",
    }
)


@dataclass(frozen=True)
class ProportionalGraphConfig:
    schema_version: str = SCHEMA_VERSION
    masters: int = 256
    train_fraction: float = 0.5
    calibration_fraction: float = 0.125
    validation_fraction: float = 0.125
    n_min: int = 8
    n_max: int = 16
    extra_edge_probability: float = 0.28
    noise_sigma: float = 0.04
    corruption_rate: float = 0.15
    corruption_amplitude_min: float = 0.6
    corruption_amplitude_max: float = 1.4
    huber_delta: float = 1.5
    irls_iterations: int = 100
    irls_damping: float = 0.5
    weight_floor: float = 1e-3
    bootstrap_replicates: int = 2000
    seed: int = 20260903

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        if self.masters < 4:
            raise ValueError("masters must be at least 4")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be in (0, 1)")
        if (
            not 0 < self.calibration_fraction < 1
            or not 0 < self.validation_fraction < 1
        ):
            raise ValueError(
                "calibration_fraction and validation_fraction must be in (0, 1)"
            )
        if (
            self.train_fraction + self.calibration_fraction + self.validation_fraction
            >= 1
        ):
            raise ValueError(
                "train, calibration, and validation fractions must leave a test partition"
            )
        if not 2 <= self.n_min <= self.n_max:
            raise ValueError("require 2 <= n_min <= n_max")
        if not 0 <= self.extra_edge_probability <= 1:
            raise ValueError("extra_edge_probability must be in [0, 1]")
        if self.noise_sigma <= 0:
            raise ValueError("noise_sigma must be positive")
        if not 0 < self.corruption_rate < 1:
            raise ValueError("corruption_rate must be in (0, 1)")
        if not 0 < self.corruption_amplitude_min <= self.corruption_amplitude_max:
            raise ValueError("invalid corruption amplitude range")
        if (
            self.huber_delta <= 0
            or self.irls_iterations < 1
            or not 0 < self.irls_damping <= 1
        ):
            raise ValueError("invalid IRLS recipe")
        if not 0 < self.weight_floor < 1:
            raise ValueError("weight_floor must be in (0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProportionalGraphConfig":
        return cls(**dict(data))


@dataclass(frozen=True)
class PublicGraphObservation:
    n_nodes: int
    edge_index: np.ndarray
    observed_log_ratio: np.ndarray
    edge_valid: np.ndarray
    path_index: np.ndarray
    path_sign: np.ndarray
    path_valid: np.ndarray
    edge_variance: np.ndarray

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "n_nodes": np.asarray(self.n_nodes, dtype=np.int64),
            "edge_index": self.edge_index,
            "observed_log_ratio": self.observed_log_ratio,
            "edge_valid": self.edge_valid,
            "path_index": self.path_index,
            "path_sign": self.path_sign,
            "path_valid": self.path_valid,
            "edge_variance": self.edge_variance,
        }


@dataclass(frozen=True)
class PrivateGraphAuthority:
    x_true: np.ndarray
    clean_log_ratio: np.ndarray
    base_noise: np.ndarray
    corruption_delta: np.ndarray
    causal_corruption_mask: np.ndarray
    corruption_mechanism: str
    split: str
    view_id: str
    master_id: str
    lineage_id: str
    seed: int
    topology_family: str

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "x_true": self.x_true,
            "clean_log_ratio": self.clean_log_ratio,
            "base_noise": self.base_noise,
            "corruption_delta": self.corruption_delta,
            "causal_corruption_mask": self.causal_corruption_mask,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "corruption_mechanism": self.corruption_mechanism,
            "split": self.split,
            "view_id": self.view_id,
            "master_id": self.master_id,
            "lineage_id": self.lineage_id,
            "seed": self.seed,
            "topology_family": self.topology_family,
        }


@dataclass(frozen=True)
class GraphView:
    public: PublicGraphObservation
    private: PrivateGraphAuthority


@dataclass(frozen=True)
class SolverResult:
    x_hat: np.ndarray
    reconstructed_log_ratio: np.ndarray
    weights: np.ndarray
    quotient_rmse: float
    relation_rmse: float
    weighted_residual_rmse: float
    laplacian_rank: int
    laplacian_condition: float
    converged: bool
    iterations: int


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def public_schema_hash() -> str:
    payload = canonical_json(
        {"schema_version": SCHEMA_VERSION, "fields": sorted(PUBLIC_ARRAY_FIELDS)}
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_public_arrays(arrays: Mapping[str, Any]) -> None:
    keys = set(arrays)
    forbidden = keys & FORBIDDEN_PUBLIC_FIELDS
    if forbidden:
        raise ValueError(
            f"private fields leaked into public observation: {sorted(forbidden)}"
        )
    if keys != PUBLIC_ARRAY_FIELDS:
        missing = sorted(PUBLIC_ARRAY_FIELDS - keys)
        extra = sorted(keys - PUBLIC_ARRAY_FIELDS)
        raise ValueError(f"public schema mismatch: missing={missing}, extra={extra}")


def incidence_matrix(n_nodes: int, edge_index: np.ndarray) -> np.ndarray:
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edge_index must have shape [n_edges, 2]")
    if np.any(edges < 0) or np.any(edges >= n_nodes):
        raise ValueError("edge endpoint outside node range")
    if np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("self edges are not allowed")
    matrix = np.zeros((len(edges), n_nodes), dtype=np.float64)
    rows = np.arange(len(edges))
    matrix[rows, edges[:, 0]] = -1.0
    matrix[rows, edges[:, 1]] = 1.0
    return matrix


def _connected_edges(
    n_nodes: int,
    extra_edge_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    order = rng.permutation(n_nodes)
    for pos in range(1, n_nodes):
        child = int(order[pos])
        parent = int(order[int(rng.integers(0, pos))])
        edges.add(tuple(sorted((parent, child))))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if (i, j) not in edges and rng.random() < extra_edge_probability:
                edges.add((i, j))
    return np.asarray(sorted(edges), dtype=np.int64)


def _path_incidence(
    n_nodes: int, edge_index: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {tuple(edge): idx for idx, edge in enumerate(edge_index.tolist())}
    path_index: list[tuple[int, int, int]] = []
    path_sign: list[tuple[float, float, float]] = []

    def edge_and_sign(src: int, dst: int) -> tuple[int, float] | None:
        key = tuple(sorted((src, dst)))
        if key not in lookup:
            return None
        return lookup[key], 1.0 if src < dst else -1.0

    for target, (i, j) in enumerate(edge_index.tolist()):
        for k in range(n_nodes):
            if k in (i, j):
                continue
            first = edge_and_sign(i, k)
            second = edge_and_sign(k, j)
            if first is None or second is None:
                continue
            path_index.append((target, first[0], second[0]))
            path_sign.append((1.0, first[1], second[1]))
    if not path_index:
        return np.empty((0, 3), dtype=np.int64), np.empty((0, 3), dtype=np.float64)
    return np.asarray(path_index, dtype=np.int64), np.asarray(
        path_sign, dtype=np.float64
    )


def exact_path_closure(observation: PublicGraphObservation) -> np.ndarray:
    if len(observation.path_index) == 0:
        return np.empty(0, dtype=np.float64)
    path_valid = observation.path_valid.copy()
    path_valid &= np.all(observation.edge_valid[observation.path_index], axis=1)
    values = observation.observed_log_ratio[observation.path_index[path_valid]]
    signed = values * observation.path_sign[path_valid]
    return signed[:, 0] - signed[:, 1] - signed[:, 2]


def cycle_residual_rms(
    edge_index: np.ndarray,
    values: np.ndarray,
    n_nodes: int,
    edge_valid: np.ndarray | None = None,
) -> float:
    valid = (
        np.ones(len(edge_index), dtype=bool)
        if edge_valid is None
        else np.asarray(edge_valid, dtype=bool)
    )
    if valid.shape != (len(edge_index),):
        raise ValueError("edge_valid must align with edges")
    incidence = incidence_matrix(n_nodes, edge_index)[valid]
    basis = null_space(incidence.T).T
    if basis.size == 0:
        return 0.0
    residual = basis @ np.asarray(values, dtype=np.float64)[valid]
    return float(np.sqrt(np.mean(residual * residual)))


def _node_distances(n_nodes: int, edge_index: np.ndarray, anchor: int) -> np.ndarray:
    adjacency: list[list[int]] = [[] for _ in range(n_nodes)]
    for i, j in edge_index.tolist():
        adjacency[i].append(j)
        adjacency[j].append(i)
    distance = np.full(n_nodes, n_nodes + 1, dtype=np.int64)
    distance[anchor] = 0
    frontier = [anchor]
    while frontier:
        current = frontier.pop(0)
        for neighbor in adjacency[current]:
            if distance[neighbor] > distance[current] + 1:
                distance[neighbor] = distance[current] + 1
                frontier.append(neighbor)
    return distance


def _corruption_indices(
    mechanism: str,
    edge_index: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_edges = len(edge_index)
    if mechanism == "iid":
        return np.sort(rng.choice(n_edges, size=count, replace=False))
    if mechanism == "grouped":
        n_nodes = int(edge_index.max()) + 1
        anchor = int(rng.integers(0, n_nodes))
        distance = _node_distances(n_nodes, edge_index, anchor)
        jitter = rng.uniform(0.0, 1e-3, size=n_edges)
        score = (
            np.minimum(distance[edge_index[:, 0]], distance[edge_index[:, 1]]) + jitter
        )
        return np.sort(np.argsort(score)[:count])
    raise ValueError(f"unknown corruption mechanism: {mechanism}")


def _partition_for_master(index: int, config: ProportionalGraphConfig) -> str:
    train_end = int(round(config.masters * config.train_fraction))
    calibration_end = train_end + int(
        round(config.masters * config.calibration_fraction)
    )
    validation_end = calibration_end + int(
        round(config.masters * config.validation_fraction)
    )
    if index < train_end:
        return "train"
    if index < calibration_end:
        return "calibration"
    if index < validation_end:
        return "validation"
    return "test"


def generate_graph_views(config: ProportionalGraphConfig) -> list[GraphView]:
    """Generate atomic train/validation masters and paired IID/OOD test views."""
    root = np.random.SeedSequence(config.seed)
    sequences = root.spawn(config.masters)
    views: list[GraphView] = []
    for master_index, sequence in enumerate(sequences):
        rng = np.random.default_rng(sequence)
        local_seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
        split = _partition_for_master(master_index, config)
        n_nodes = int(rng.integers(config.n_min, config.n_max + 1))
        edge_index = _connected_edges(n_nodes, config.extra_edge_probability, rng)
        incidence = incidence_matrix(n_nodes, edge_index)
        x_true = rng.normal(0.0, 1.0, size=n_nodes)
        x_true -= x_true.mean()
        clean = incidence @ x_true
        noise = rng.normal(0.0, config.noise_sigma, size=len(edge_index))
        base_observation = clean + noise
        count = max(
            1,
            min(
                len(edge_index) - 1,
                int(round(config.corruption_rate * len(edge_index))),
            ),
        )
        magnitudes = rng.uniform(
            config.corruption_amplitude_min,
            config.corruption_amplitude_max,
            size=count,
        )
        magnitudes *= rng.choice(np.asarray([-1.0, 1.0]), size=count)
        path_index, path_sign = _path_incidence(n_nodes, edge_index)
        master_id = hashlib.sha256(
            f"{config.schema_version}:{config.seed}:{master_index}".encode("utf-8")
        ).hexdigest()[:24]
        mechanisms = ("iid", "grouped") if split == "test" else ("iid",)
        for mechanism in mechanisms:
            selected = _corruption_indices(mechanism, edge_index, count, rng)
            corruption_delta = np.zeros(len(edge_index), dtype=np.float64)
            corruption_delta[selected] = magnitudes
            observed = base_observation + corruption_delta
            mask = np.zeros(len(edge_index), dtype=bool)
            mask[selected] = True
            view_id = hashlib.sha256(
                f"{master_id}:{mechanism}".encode("utf-8")
            ).hexdigest()[:24]
            public = PublicGraphObservation(
                n_nodes=n_nodes,
                edge_index=edge_index.copy(),
                observed_log_ratio=observed,
                edge_valid=np.ones(len(edge_index), dtype=bool),
                path_index=path_index.copy(),
                path_sign=path_sign.copy(),
                path_valid=np.ones(len(path_index), dtype=bool),
                edge_variance=np.full(
                    len(edge_index), config.noise_sigma**2, dtype=np.float64
                ),
            )
            private = PrivateGraphAuthority(
                x_true=x_true.copy(),
                clean_log_ratio=clean.copy(),
                base_noise=noise.copy(),
                corruption_delta=corruption_delta,
                causal_corruption_mask=mask,
                corruption_mechanism=mechanism,
                split=split,
                view_id=view_id,
                master_id=master_id,
                lineage_id=master_id,
                seed=local_seed,
                topology_family="random_tree_plus_edges",
            )
            validate_graph_view(GraphView(public=public, private=private))
            views.append(GraphView(public=public, private=private))
    return views


def validate_graph_view(view: GraphView) -> None:
    public = view.public
    private = view.private
    validate_public_arrays(public.arrays())
    n_edges = len(public.edge_index)
    if public.edge_index.shape != (n_edges, 2):
        raise ValueError("invalid edge_index shape")
    if not np.all(public.edge_index[:, 0] < public.edge_index[:, 1]):
        raise ValueError("public edges must use canonical orientation i < j")
    if len({tuple(edge) for edge in public.edge_index.tolist()}) != n_edges:
        raise ValueError("duplicate edges")
    for values in (
        public.observed_log_ratio,
        public.edge_valid,
        public.edge_variance,
        private.clean_log_ratio,
        private.base_noise,
        private.corruption_delta,
        private.causal_corruption_mask,
    ):
        if len(values) != n_edges:
            raise ValueError("edge-aligned array has the wrong length")
    if public.path_index.shape != public.path_sign.shape:
        raise ValueError("path_index and path_sign must have the same shape")
    if public.path_index.ndim != 2 or public.path_index.shape[1] != 3:
        raise ValueError("path arrays must have shape [n_paths, 3]")
    if len(public.path_valid) != len(public.path_index):
        raise ValueError("path_valid has the wrong length")
    if np.any(public.edge_variance <= 0):
        raise ValueError("edge variances must be positive")
    incidence = incidence_matrix(public.n_nodes, public.edge_index)
    if np.linalg.matrix_rank(incidence) != public.n_nodes - 1:
        raise ValueError("public graph is not connected")
    if not np.allclose(incidence @ private.x_true, private.clean_log_ratio, atol=1e-12):
        raise ValueError("private clean relations violate the incidence convention")
    if not np.array_equal(
        private.causal_corruption_mask, private.corruption_delta != 0
    ):
        raise ValueError("causal corruption mask and delta disagree")
    expected = private.clean_log_ratio + private.base_noise + private.corruption_delta
    if not np.allclose(expected, public.observed_log_ratio, atol=1e-12):
        raise ValueError(
            "public observation does not match the private generative decomposition"
        )


def solve_weighted_least_squares(
    observation: PublicGraphObservation,
    values: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    *,
    weight_floor: float = 1e-3,
) -> SolverResult:
    y = np.asarray(
        observation.observed_log_ratio if values is None else values,
        dtype=np.float64,
    )
    incidence_all = incidence_matrix(observation.n_nodes, observation.edge_index)
    if len(y) != len(incidence_all):
        raise ValueError("values must align with edges")
    valid = np.asarray(observation.edge_valid, dtype=bool)
    if valid.shape != y.shape or not np.any(valid):
        raise ValueError("edge_valid must select at least one edge")
    incidence = incidence_all[valid]
    y_valid = y[valid]
    raw_weights = (
        np.ones(len(y), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if raw_weights.shape != y.shape or not np.all(np.isfinite(raw_weights)):
        raise ValueError("weights must be finite and edge-aligned")
    normalized = np.zeros_like(raw_weights)
    normalized[valid] = np.clip(raw_weights[valid], weight_floor, None)
    normalized[valid] /= normalized[valid].mean()
    valid_weights = normalized[valid]
    laplacian = incidence.T @ (valid_weights[:, None] * incidence)
    if np.linalg.matrix_rank(incidence, tol=1e-10) != observation.n_nodes - 1:
        raise ValueError("valid-edge subgraph is disconnected")
    rhs = incidence.T @ (valid_weights * y_valid)
    ones = np.ones((observation.n_nodes, 1), dtype=np.float64)
    kkt = np.block([[laplacian, ones], [ones.T, np.zeros((1, 1))]])
    solution = np.linalg.solve(kkt, np.concatenate([rhs, np.zeros(1)]))
    x_hat = solution[:-1]
    reconstructed = incidence_all @ x_hat
    residual = reconstructed[valid] - y_valid
    eigenvalues = np.linalg.eigvalsh(laplacian)
    positive = eigenvalues[eigenvalues > 1e-10]
    condition = (
        float(positive.max() / positive.min()) if len(positive) else float("inf")
    )
    return SolverResult(
        x_hat=x_hat,
        reconstructed_log_ratio=reconstructed,
        weights=normalized,
        quotient_rmse=float("nan"),
        relation_rmse=float("nan"),
        weighted_residual_rmse=float(
            np.sqrt(np.average(residual * residual, weights=valid_weights))
        ),
        laplacian_rank=int(np.linalg.matrix_rank(laplacian, tol=1e-10)),
        laplacian_condition=condition,
        converged=True,
        iterations=1,
    )


def solve_huber_irls(
    observation: PublicGraphObservation,
    *,
    values: np.ndarray | None = None,
    base_weights: np.ndarray | None = None,
    delta: float = 1.5,
    max_iterations: int = 20,
    damping: float = 0.5,
    weight_floor: float = 1e-3,
    tolerance: float = 1e-6,
) -> SolverResult:
    if not 0 < damping <= 1:
        raise ValueError("damping must be in (0, 1]")
    valid = np.asarray(observation.edge_valid, dtype=bool)
    y = np.asarray(
        observation.observed_log_ratio if values is None else values,
        dtype=np.float64,
    )
    if y.shape != observation.observed_log_ratio.shape:
        raise ValueError("values must align with edges")
    if base_weights is None:
        base = np.where(valid, 1.0, 0.0)
    else:
        base = np.asarray(base_weights, dtype=np.float64)
        if base.shape != y.shape or not np.all(np.isfinite(base)):
            raise ValueError("base_weights must be finite and edge-aligned")
        base = np.where(valid, np.clip(base, weight_floor, None), 0.0)
        base[valid] /= base[valid].mean()
    weights = base.copy()
    variance = np.asarray(observation.edge_variance, dtype=np.float64)
    if variance.shape != weights.shape:
        raise ValueError("edge_variance must align with edges")
    # Keep the robust objective fixed across iterations. Re-estimating scale from
    # each residual vector makes IRLS chase a moving target and can oscillate.
    scale = max(float(np.sqrt(np.median(variance[valid]))), 1e-8)
    threshold = delta * scale
    previous_x: np.ndarray | None = None
    previous_objective: float | None = None
    result: SolverResult | None = None
    converged = False
    for iteration in range(1, max_iterations + 1):
        result = solve_weighted_least_squares(
            observation,
            values=y,
            weights=weights,
            weight_floor=weight_floor,
        )
        residual = result.reconstructed_log_ratio[valid] - y[valid]
        normalized_residual = residual / scale
        magnitude_normalized = np.abs(normalized_residual)
        huber_terms = np.where(
            magnitude_normalized <= delta,
            0.5 * normalized_residual * normalized_residual,
            delta * (magnitude_normalized - 0.5 * delta),
        )
        objective = float(np.sum(base[valid] * huber_terms))
        if previous_x is not None and previous_objective is not None:
            solution_change = float(np.max(np.abs(result.x_hat - previous_x)))
            objective_change = abs(objective - previous_objective) / max(
                1.0, abs(previous_objective)
            )
            if solution_change < tolerance and objective_change < tolerance:
                converged = True
                break
        magnitude = np.abs(residual)
        candidate = np.ones_like(magnitude)
        large = magnitude > threshold
        candidate[large] = threshold / magnitude[large]
        candidate = np.clip(candidate, weight_floor, 1.0)
        updated = np.zeros_like(weights)
        target_weights = base[valid] * candidate
        updated[valid] = (1.0 - damping) * weights[valid] + damping * target_weights
        previous_x = result.x_hat.copy()
        previous_objective = objective
        weights = updated
    assert result is not None
    final = solve_weighted_least_squares(
        observation,
        values=y,
        weights=weights,
        weight_floor=weight_floor,
    )
    return SolverResult(
        **{
            **asdict(final),
            "converged": converged,
            "iterations": iteration,
        }
    )


def score_solver(
    result: SolverResult, authority: PrivateGraphAuthority
) -> SolverResult:
    aligned_true = authority.x_true - authority.x_true.mean()
    aligned_hat = result.x_hat - result.x_hat.mean()
    quotient_rmse = float(np.sqrt(np.mean((aligned_hat - aligned_true) ** 2)))
    relation_rmse = float(
        np.sqrt(
            np.mean((result.reconstructed_log_ratio - authority.clean_log_ratio) ** 2)
        )
    )
    return SolverResult(
        **{
            **asdict(result),
            "quotient_rmse": quotient_rmse,
            "relation_rmse": relation_rmse,
        }
    )


def solve_oracle_weights(view: GraphView, weight_floor: float = 1e-3) -> SolverResult:
    weights = np.where(view.private.causal_corruption_mask, weight_floor, 1.0)
    return score_solver(
        solve_weighted_least_squares(
            view.public, weights=weights, weight_floor=weight_floor
        ),
        view.private,
    )


def permute_nodes(
    observation: PublicGraphObservation,
    old_to_new: np.ndarray,
) -> PublicGraphObservation:
    permutation = np.asarray(old_to_new, dtype=np.int64)
    if permutation.shape != (observation.n_nodes,) or set(permutation.tolist()) != set(
        range(observation.n_nodes)
    ):
        raise ValueError("old_to_new must be a node permutation")
    transformed = permutation[observation.edge_index]
    direction = np.where(transformed[:, 0] < transformed[:, 1], 1.0, -1.0)
    canonical_edges = np.sort(transformed, axis=1)
    order = np.lexsort((canonical_edges[:, 1], canonical_edges[:, 0]))
    canonical_edges = canonical_edges[order]
    values = (observation.observed_log_ratio * direction)[order]
    variance = observation.edge_variance[order]
    edge_valid = observation.edge_valid[order]
    path_index, path_sign = _path_incidence(observation.n_nodes, canonical_edges)
    return PublicGraphObservation(
        n_nodes=observation.n_nodes,
        edge_index=canonical_edges,
        observed_log_ratio=values,
        edge_valid=edge_valid,
        path_index=path_index,
        path_sign=path_sign,
        path_valid=np.ones(len(path_index), dtype=bool),
        edge_variance=variance,
    )


def reverse_orientations(
    observation: PublicGraphObservation,
    reverse_mask: np.ndarray,
) -> PublicGraphObservation:
    mask = np.asarray(reverse_mask, dtype=bool)
    if mask.shape != (len(observation.edge_index),):
        raise ValueError("reverse_mask must align with edges")
    edges = observation.edge_index.copy()
    edges[mask] = edges[mask][:, ::-1]
    values = observation.observed_log_ratio.copy()
    values[mask] *= -1.0
    path_sign = observation.path_sign.copy()
    if len(path_sign):
        path_sign *= np.where(mask[observation.path_index], -1.0, 1.0)
    return PublicGraphObservation(
        n_nodes=observation.n_nodes,
        edge_index=edges,
        observed_log_ratio=values,
        edge_valid=observation.edge_valid.copy(),
        path_index=observation.path_index.copy(),
        path_sign=path_sign,
        path_valid=observation.path_valid.copy(),
        edge_variance=observation.edge_variance.copy(),
    )


def result_arrays(result: SolverResult) -> dict[str, np.ndarray]:
    return {
        "x_hat": result.x_hat,
        "reconstructed_log_ratio": result.reconstructed_log_ratio,
        "weights": result.weights,
        "quotient_rmse": np.asarray(result.quotient_rmse),
        "relation_rmse": np.asarray(result.relation_rmse),
        "weighted_residual_rmse": np.asarray(result.weighted_residual_rmse),
        "laplacian_rank": np.asarray(result.laplacian_rank),
        "laplacian_condition": np.asarray(result.laplacian_condition),
        "converged": np.asarray(result.converged),
        "iterations": np.asarray(result.iterations),
    }
