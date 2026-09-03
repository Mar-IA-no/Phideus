"""Contract tests for the local proportional-coherence graph benchmark."""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from geometria_proporcional.proportional_graph_contract import (  # noqa: E402
    FORBIDDEN_PUBLIC_FIELDS,
    PUBLIC_ARRAY_FIELDS,
    ProportionalGraphConfig,
    cycle_residual_rms,
    exact_path_closure,
    generate_graph_views,
    incidence_matrix,
    permute_nodes,
    reverse_orientations,
    score_solver,
    solve_huber_irls,
    solve_weighted_least_squares,
    validate_public_arrays,
)


@pytest.fixture(scope="module")
def views():
    return generate_graph_views(
        ProportionalGraphConfig(masters=16, n_min=8, n_max=10, seed=1234)
    )


def test_generation_is_deterministic_and_test_views_are_paired():
    config = ProportionalGraphConfig(masters=12, n_min=8, n_max=8, seed=77)
    first = generate_graph_views(config)
    second = generate_graph_views(config)
    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert a.private.metadata() == b.private.metadata()
        for key in PUBLIC_ARRAY_FIELDS:
            assert np.array_equal(a.public.arrays()[key], b.public.arrays()[key])
    test = [view for view in first if view.private.split == "test"]
    by_lineage = {}
    for view in test:
        by_lineage.setdefault(view.private.lineage_id, []).append(view)
    assert by_lineage
    for pair in by_lineage.values():
        assert {view.private.corruption_mechanism for view in pair} == {"iid", "grouped"}
        assert len({int(view.private.causal_corruption_mask.sum()) for view in pair}) == 1
        assert np.array_equal(pair[0].public.edge_index, pair[1].public.edge_index)
        assert np.array_equal(pair[0].private.clean_log_ratio, pair[1].private.clean_log_ratio)
        assert np.array_equal(pair[0].private.base_noise, pair[1].private.base_noise)
        assert np.array_equal(
            np.sort(pair[0].private.corruption_delta[pair[0].private.causal_corruption_mask]),
            np.sort(pair[1].private.corruption_delta[pair[1].private.causal_corruption_mask]),
        )
        assert pair[0].private.master_id == pair[1].private.master_id
        assert pair[0].private.view_id != pair[1].private.view_id


def test_public_schema_is_exact_and_rejects_private_fields(views):
    for view in views:
        arrays = view.public.arrays()
        assert set(arrays) == PUBLIC_ARRAY_FIELDS
        assert not (set(arrays) & FORBIDDEN_PUBLIC_FIELDS)
        validate_public_arrays(arrays)
    leaked = dict(views[0].public.arrays())
    leaked["x_true"] = np.zeros(views[0].public.n_nodes)
    with pytest.raises(ValueError, match="leaked"):
        validate_public_arrays(leaked)


def test_incidence_orientation_and_gauge_identity(views):
    view = views[0]
    incidence = incidence_matrix(view.public.n_nodes, view.public.edge_index)
    clean = incidence @ view.private.x_true
    shifted = incidence @ (view.private.x_true + 7.25)
    assert np.allclose(clean, view.private.clean_log_ratio, atol=1e-12)
    assert np.allclose(clean, shifted, atol=1e-12)
    assert np.linalg.matrix_rank(incidence) == view.public.n_nodes - 1


def test_clean_paths_and_cycles_close(views):
    for view in views:
        clean_observation = type(view.public)(
            **{
                **view.public.__dict__,
                "observed_log_ratio": view.private.clean_log_ratio,
            }
        )
        assert np.allclose(exact_path_closure(clean_observation), 0.0, atol=1e-11)
        assert cycle_residual_rms(
            view.public.edge_index,
            view.private.clean_log_ratio,
            view.public.n_nodes,
        ) < 1e-11


def test_wls_recovers_clean_potentials_modulo_gauge(views):
    view = views[0]
    clean_observation = type(view.public)(
        **{
            **view.public.__dict__,
            "observed_log_ratio": view.private.clean_log_ratio,
        }
    )
    result = solve_weighted_least_squares(clean_observation)
    assert np.allclose(result.x_hat, view.private.x_true, atol=1e-10)
    assert result.laplacian_rank == view.public.n_nodes - 1
    assert np.isfinite(result.laplacian_condition)


def test_coherent_orientation_reversal_preserves_solution(views):
    view = views[0]
    baseline = solve_weighted_least_squares(view.public)
    mask = np.zeros(len(view.public.edge_index), dtype=bool)
    mask[::2] = True
    reversed_observation = reverse_orientations(view.public, mask)
    transformed = solve_weighted_least_squares(reversed_observation)
    assert np.allclose(baseline.x_hat, transformed.x_hat, atol=1e-10)
    assert np.allclose(
        exact_path_closure(view.public),
        exact_path_closure(reversed_observation),
        atol=1e-12,
    )
    assert np.allclose(
        baseline.weighted_residual_rmse,
        transformed.weighted_residual_rmse,
        atol=1e-12,
    )


def test_masks_govern_edges_and_paths(views):
    view = next(view for view in views if len(view.public.path_index) > 1)
    removable = None
    for edge in range(len(view.public.edge_index)):
        valid = np.ones(len(view.public.edge_index), dtype=bool)
        valid[edge] = False
        matrix = incidence_matrix(view.public.n_nodes, view.public.edge_index)[valid]
        if np.linalg.matrix_rank(matrix) == view.public.n_nodes - 1:
            removable = edge
            break
    assert removable is not None
    valid = np.ones(len(view.public.edge_index), dtype=bool)
    valid[removable] = False
    ordinary = type(view.public)(**{**view.public.__dict__, "edge_valid": valid})
    corrupted_values = view.public.observed_log_ratio.copy()
    corrupted_values[removable] = 1e6
    adversarial = type(view.public)(
        **{
            **view.public.__dict__,
            "edge_valid": valid,
            "observed_log_ratio": corrupted_values,
        }
    )
    assert np.allclose(
        solve_weighted_least_squares(ordinary).x_hat,
        solve_weighted_least_squares(adversarial).x_hat,
        atol=1e-10,
    )

    path_valid = view.public.path_valid.copy()
    path_valid[0] = False
    masked_paths = type(view.public)(**{**view.public.__dict__, "path_valid": path_valid})
    assert len(exact_path_closure(masked_paths)) == len(exact_path_closure(view.public)) - 1


def test_node_permutation_is_equivariant(views):
    view = views[0]
    permutation = np.random.default_rng(8).permutation(view.public.n_nodes)
    original = solve_weighted_least_squares(view.public)
    transformed = solve_weighted_least_squares(permute_nodes(view.public, permutation))
    expected = np.empty_like(original.x_hat)
    expected[permutation] = original.x_hat
    assert np.allclose(expected, transformed.x_hat, atol=1e-10)


def test_huber_irls_converges_on_preflight_distribution():
    config = ProportionalGraphConfig(
        masters=64,
        n_min=8,
        n_max=16,
        irls_iterations=100,
        seed=991,
    )
    results = [
        score_solver(
            solve_huber_irls(
                view.public,
                delta=config.huber_delta,
                max_iterations=config.irls_iterations,
                damping=config.irls_damping,
                weight_floor=config.weight_floor,
            ),
            view.private,
        )
        for view in generate_graph_views(config)
    ]
    assert all(result.converged for result in results)
    assert all(np.isfinite(result.quotient_rmse) for result in results)


def test_preflight_runner_preserves_public_private_boundary(tmp_path: Path):
    output = tmp_path / "preflight"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/geometria_proporcional/run_proportional_graph_preflight.py"),
            "--output",
            str(output),
            "--masters",
            "8",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["masters"] == 8
    assert manifest["counts"]["views"] > 8
    assert (output / "PREFLIGHT_REPORT.md").exists()
    assert (output / "target_authority_table.json").exists()
    assert manifest["environment"]["python"]
    assert manifest["source_files"]
    public_file = next((output / "public").glob("*.npz"))
    with np.load(public_file, allow_pickle=False) as data:
        assert set(data.files) == PUBLIC_ARRAY_FIELDS
        assert not (set(data.files) & FORBIDDEN_PUBLIC_FIELDS)
    for relative, record in manifest["files"].items():
        payload = (output / relative).read_bytes()
        assert len(payload) == record["bytes"]
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]

    first_public = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (output / "public").glob("*.npz")
    }
    subprocess.run(
        [str(output / "replay.sh")],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    replayed_public = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (output / "public").glob("*.npz")
    }
    assert replayed_public == first_public
    replayed_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert replayed_manifest["config"] == manifest["config"]
