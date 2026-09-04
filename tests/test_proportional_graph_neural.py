"""Tests for the local proportional graph neural smoke."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.geometria_proporcional.run_proportional_graph_neural_smoke import (  # noqa: E402
    _bootstrap_result,
    _contrast_values,
    _view_tensors,
)

from geometria_proporcional.proportional_graph_contract import (  # noqa: E402
    ProportionalGraphConfig,
    generate_graph_views,
    incidence_matrix,
    permute_nodes,
    reverse_orientations,
    solve_huber_irls,
)
from geometria_proporcional.proportional_graph_neural import (  # noqa: E402
    EdgewiseMLP,
    GenericMessagePassing,
    ProportionalPathMixer,
    differentiable_wls,
    direct_centered_decoder,
    exact_closure_only,
    materialize_closure_evidence,
    observation_tensors,
    parameter_count,
    shuffled_path_tensors,
)


def _view():
    return generate_graph_views(
        ProportionalGraphConfig(masters=8, n_min=8, n_max=8, seed=3301)
    )[0]


def test_factorial_models_are_exactly_parameter_matched_and_shape_safe():
    view = _view()
    tensors = observation_tensors(view.public)
    models = [
        ProportionalPathMixer(hidden_dim=16, evidence=evidence, mixer=mixer)
        for evidence in ("raw", "closure")
        for mixer in ("generic", "typed")
    ]
    assert len({parameter_count(model) for model in models}) == 1
    for model in models:
        inputs = (
            materialize_closure_evidence(tensors)
            if model.evidence == "closure"
            else tensors
        )
        output = model(inputs)
        assert output.corrected_log_ratio.shape == view.public.observed_log_ratio.shape
        assert output.reliability.shape == view.public.observed_log_ratio.shape
        assert torch.all(output.reliability > 0)
        assert torch.all(output.reliability <= 1)


def test_all_factorial_models_are_equivariant_to_edge_orientation_convention():
    view = _view()
    mask = np.zeros(len(view.public.edge_index), dtype=bool)
    mask[::2] = True
    reversed_observation = reverse_orientations(view.public, mask)
    for evidence in ("raw", "closure"):
        for mixer in ("generic", "typed"):
            model = ProportionalPathMixer(
                hidden_dim=16, evidence=evidence, mixer=mixer
            ).eval()
            original_inputs = observation_tensors(view.public)
            transformed_inputs = observation_tensors(reversed_observation)
            if evidence == "closure":
                original_inputs = materialize_closure_evidence(original_inputs)
                transformed_inputs = materialize_closure_evidence(transformed_inputs)
            with torch.no_grad():
                original = model(original_inputs)
                transformed = model(transformed_inputs)
            expected = original.corrected_log_ratio.clone()
            expected[torch.as_tensor(mask)] *= -1
            assert torch.allclose(expected, transformed.corrected_log_ratio, atol=1e-6)
            assert torch.allclose(
                original.reliability, transformed.reliability, atol=1e-6
            )


def test_all_factorial_models_are_equivariant_to_node_permutation():
    view = _view()
    permutation = np.random.default_rng(987).permutation(view.public.n_nodes)
    permuted = permute_nodes(view.public, permutation)
    transformed_edges = permutation[view.public.edge_index]
    direction = np.where(transformed_edges[:, 0] < transformed_edges[:, 1], 1.0, -1.0)
    canonical_edges = np.sort(transformed_edges, axis=1)
    order = np.lexsort((canonical_edges[:, 1], canonical_edges[:, 0]))
    for evidence in ("raw", "closure"):
        for mixer in ("generic", "typed"):
            torch.manual_seed(11)
            model = ProportionalPathMixer(
                hidden_dim=16, evidence=evidence, mixer=mixer
            ).eval()
            original_inputs = observation_tensors(view.public)
            permuted_inputs = observation_tensors(permuted)
            if evidence == "closure":
                original_inputs = materialize_closure_evidence(original_inputs)
                permuted_inputs = materialize_closure_evidence(permuted_inputs)
            with torch.no_grad():
                original = model(original_inputs)
                transformed = model(permuted_inputs)
            expected_values = original.corrected_log_ratio * torch.as_tensor(
                direction, dtype=original.corrected_log_ratio.dtype
            )
            expected_weights = original.reliability
            assert torch.allclose(
                expected_values[order], transformed.corrected_log_ratio, atol=1e-6
            )
            assert torch.allclose(
                expected_weights[order], transformed.reliability, atol=1e-6
            )


def test_generic_mixer_remains_node_equivariant_with_nonzero_swap_logit():
    view = _view()
    permutation = np.random.default_rng(42).permutation(view.public.n_nodes)
    permuted = permute_nodes(view.public, permutation)
    transformed_edges = permutation[view.public.edge_index]
    direction = np.where(transformed_edges[:, 0] < transformed_edges[:, 1], 1.0, -1.0)
    canonical_edges = np.sort(transformed_edges, axis=1)
    order = np.lexsort((canonical_edges[:, 1], canonical_edges[:, 0]))
    model = ProportionalPathMixer(hidden_dim=16, mixer="generic").eval()
    model.swap_logit.data.fill_(2.0)
    with torch.no_grad():
        original = model(observation_tensors(view.public))
        transformed = model(observation_tensors(permuted))
    expected = original.corrected_log_ratio * torch.as_tensor(
        direction, dtype=original.corrected_log_ratio.dtype
    )
    assert torch.allclose(expected[order], transformed.corrected_log_ratio, atol=1e-6)
    assert torch.allclose(
        original.reliability[order], transformed.reliability, atol=1e-6
    )


def test_closure_arm_requires_externally_materialized_evidence():
    model = ProportionalPathMixer(hidden_dim=16, evidence="closure", mixer="typed")
    with np.testing.assert_raises_regex(ValueError, "materialized outside"):
        model(observation_tensors(_view().public))


def test_path_shuffle_is_deterministic_joint_and_nontrivial():
    tensors = observation_tensors(_view().public)
    first = shuffled_path_tensors(tensors, seed=19)
    second = shuffled_path_tensors(tensors, seed=19)
    assert torch.equal(first["path_index"], second["path_index"])
    assert torch.equal(first["path_index"][:, 0], tensors["path_index"][:, 0])
    before = sorted(map(tuple, tensors["path_index"][:, 1:].tolist()))
    after = sorted(map(tuple, first["path_index"][:, 1:].tolist()))
    assert before == after
    assert not torch.equal(first["path_index"], tensors["path_index"])
    assert torch.all(torch.any(first["path_index"] != tensors["path_index"], dim=1))
    assert torch.all(first["path_index"][:, 0, None] != first["path_index"][:, 1:])
    before_joint = sorted(
        zip(
            map(tuple, tensors["path_index"][:, 1:].tolist()),
            map(tuple, tensors["path_sign"][:, 1:].tolist()),
        )
    )
    after_joint = sorted(
        zip(
            map(tuple, first["path_index"][:, 1:].tolist()),
            map(tuple, first["path_sign"][:, 1:].tolist()),
        )
    )
    assert before_joint == after_joint


def test_ineligible_path_shuffle_preserves_original_evidence_for_exclusion():
    tensors = observation_tensors(_view().public)
    tiny = {
        **tensors,
        "path_index": tensors["path_index"][:1],
        "path_sign": tensors["path_sign"][:1],
        "path_valid": tensors["path_valid"][:1],
    }
    shuffled = shuffled_path_tensors(tiny, seed=3)
    assert not bool(shuffled["path_shuffle_eligible"])
    assert torch.equal(shuffled["path_index"], tiny["path_index"])
    assert torch.equal(shuffled["path_sign"], tiny["path_sign"])
    assert torch.equal(shuffled["path_valid"], tiny["path_valid"])


def test_path_shuffle_is_paired_across_corruptions_and_varies_by_seed():
    views = generate_graph_views(
        ProportionalGraphConfig(
            masters=32,
            n_min=8,
            n_max=9,
            extra_edge_probability=0.8,
            seed=909,
        )
    )
    by_master: dict[str, list] = {}
    for view in views:
        if view.private.split == "test":
            by_master.setdefault(view.private.master_id, []).append(view)
    arm = {
        "architecture": "path_mixer",
        "evidence": "closure",
        "mixer": "typed",
        "mix_paths": True,
        "path_shuffle": True,
    }
    changed_between_seeds = False
    for pair in by_master.values():
        assert len(pair) == 2
        first_seed = [_view_tensors(view, arm, 17) for view in pair]
        assert torch.equal(first_seed[0]["path_index"], first_seed[1]["path_index"])
        assert torch.equal(first_seed[0]["path_sign"], first_seed[1]["path_sign"])
        second_seed = _view_tensors(pair[0], arm, 19)
        changed_between_seeds |= not torch.equal(
            first_seed[0]["path_index"], second_seed["path_index"]
        )
    assert changed_between_seeds


def test_no_mix_control_keeps_every_parameter_active():
    model = ProportionalPathMixer(hidden_dim=16, mix_paths=False)
    output = model(observation_tensors(_view().public))
    (output.corrected_log_ratio.square().mean() + output.reliability.mean()).backward()
    inactive = {
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is None or not torch.any(parameter.grad != 0)
    }
    assert inactive == set()
    assert not any(
        name.startswith(("path_mlp", "path_score")) or name == "swap_logit"
        for name in inactive
    )


def test_generic_message_passing_is_capacity_matched_and_equivariant():
    view = _view()
    path_model = ProportionalPathMixer(hidden_dim=16)
    generic = GenericMessagePassing(hidden_dim=16).eval()
    assert parameter_count(path_model) == parameter_count(generic)
    assert {
        key: tuple(value.shape) for key, value in path_model.state_dict().items()
    } == {key: tuple(value.shape) for key, value in generic.state_dict().items()}
    generic.load_state_dict(path_model.state_dict())

    mask = np.zeros(len(view.public.edge_index), dtype=bool)
    mask[::2] = True
    reversed_observation = reverse_orientations(view.public, mask)
    with torch.no_grad():
        original = generic(observation_tensors(view.public))
        reversed_output = generic(observation_tensors(reversed_observation))
    expected = original.corrected_log_ratio.clone()
    expected[torch.as_tensor(mask)] *= -1
    assert torch.allclose(expected, reversed_output.corrected_log_ratio, atol=1e-6)
    assert torch.allclose(original.reliability, reversed_output.reliability, atol=1e-6)

    permutation = np.random.default_rng(43).permutation(view.public.n_nodes)
    permuted = permute_nodes(view.public, permutation)
    transformed_edges = permutation[view.public.edge_index]
    direction = np.where(transformed_edges[:, 0] < transformed_edges[:, 1], 1.0, -1.0)
    canonical_edges = np.sort(transformed_edges, axis=1)
    order = np.lexsort((canonical_edges[:, 1], canonical_edges[:, 0]))
    with torch.no_grad():
        transformed = generic(observation_tensors(permuted))
    expected = original.corrected_log_ratio * torch.as_tensor(
        direction, dtype=original.corrected_log_ratio.dtype
    )
    assert torch.allclose(expected[order], transformed.corrected_log_ratio, atol=1e-6)
    assert torch.allclose(
        original.reliability[order], transformed.reliability, atol=1e-6
    )


def test_generic_message_passing_ignores_invalid_edges_in_valid_context():
    view = _view()
    valid = view.public.edge_valid.copy()
    valid[0] = False
    baseline = replace(view.public, edge_valid=valid)
    adversarial_values = view.public.observed_log_ratio.copy()
    adversarial_values[0] = 1e6
    adversarial = replace(
        view.public,
        edge_valid=valid,
        observed_log_ratio=adversarial_values,
    )
    model = GenericMessagePassing(hidden_dim=16).eval()
    with torch.no_grad():
        first = model(observation_tensors(baseline))
        second = model(observation_tensors(adversarial))
    valid_tensor = torch.as_tensor(valid)
    assert torch.equal(
        first.corrected_log_ratio[valid_tensor],
        second.corrected_log_ratio[valid_tensor],
    )
    assert torch.equal(
        first.reliability[valid_tensor], second.reliability[valid_tensor]
    )


def test_edge_mlp_is_strictly_edge_local_and_direct_decoder_is_centered():
    view = _view()
    model = EdgewiseMLP(hidden_dim=16).eval()
    tensors = observation_tensors(view.public)
    changed = {**tensors, "path_index": tensors["path_index"].flip(0)}
    with torch.no_grad():
        original = model(tensors)
        transformed = model(changed)
    assert torch.equal(original.corrected_log_ratio, transformed.corrected_log_ratio)
    decoded = direct_centered_decoder(
        view.public,
        view.private.clean_log_ratio,
        np.ones_like(view.private.clean_log_ratio),
    )
    assert decoded.shape == view.private.x_true.shape
    assert abs(decoded.mean()) < 1e-12


def test_bootstrap_refuses_incomplete_solver_estimand():
    result = _bootstrap_result(
        np.asarray([0.1, np.nan, -0.2]),
        indices_for=lambda n: np.zeros((8, n), dtype=np.int64),
        require_complete=True,
    )
    assert result["status"] == "NOT_EVALUABLE_SOLVER_FAILURE"
    assert result["n_total"] == 3
    assert result["n_complete"] == 2
    assert result["missing"] == 1
    assert result["mean"] is None


def test_shuffle_contrast_excludes_ineligible_master_without_dropping_evidence():
    rows = {
        "closure_typed": {
            "a": {
                "master_id": "m",
                "split": "test",
                "mechanism": "iid",
                "relation_rmse": 0.1,
                "path_shuffle_eligible": True,
            }
        },
        "closure_typed_path_shuffle": {
            "b": {
                "master_id": "m",
                "split": "test",
                "mechanism": "iid",
                "relation_rmse": 0.2,
                "path_shuffle_eligible": False,
            }
        },
    }
    values = _contrast_values(
        rows,
        {"closure_typed": 1.0, "closure_typed_path_shuffle": -1.0},
        "relation_rmse",
        "iid",
        ["m"],
    )
    assert np.isnan(values[0])


def test_neural_lineage_seed_is_disjoint_from_classical_preflight():
    neural = ProportionalGraphConfig(masters=8, seed=2026090317)
    classical = ProportionalGraphConfig(masters=8, seed=20260903)
    neural_ids = {view.private.master_id for view in generate_graph_views(neural)}
    classical_ids = {view.private.master_id for view in generate_graph_views(classical)}
    assert neural_ids.isdisjoint(classical_ids)


def test_differentiable_wls_recovers_clean_graph_and_backpropagates():
    view = _view()
    incidence = incidence_matrix(view.public.n_nodes, view.public.edge_index)
    clean = torch.tensor(
        incidence @ view.private.x_true, dtype=torch.float32, requires_grad=True
    )
    weights = torch.ones_like(clean, requires_grad=True)
    x_hat = differentiable_wls(view.public, clean, weights)
    target = torch.tensor(view.private.x_true, dtype=torch.float32)
    loss = torch.mean((x_hat - target) ** 2)
    loss.backward()
    assert torch.allclose(x_hat, target, atol=1e-5)
    assert clean.grad is not None and torch.all(torch.isfinite(clean.grad))
    assert weights.grad is not None and torch.all(torch.isfinite(weights.grad))


def test_exact_closure_control_and_huber_interface_accept_corrected_state():
    view = _view()
    clean_public = type(view.public)(
        **{
            **view.public.__dict__,
            "observed_log_ratio": view.private.clean_log_ratio.copy(),
        }
    )
    corrected, reliability = exact_closure_only(clean_public)
    assert np.allclose(corrected, view.private.clean_log_ratio, atol=1e-12)
    result = solve_huber_irls(
        view.public,
        values=corrected,
        base_weights=reliability,
        max_iterations=100,
    )
    assert result.converged
    assert np.all(np.isfinite(result.x_hat))


def test_huber_irls_functionally_uses_supplied_base_weights():
    view = _view()
    unweighted = solve_huber_irls(view.public, max_iterations=5000, damping=1.0)
    oracle_weights = np.where(view.private.causal_corruption_mask, 1e-3, 1.0)
    weighted = solve_huber_irls(
        view.public,
        base_weights=oracle_weights,
        max_iterations=5000,
        damping=1.0,
    )
    assert unweighted.converged and weighted.converged
    assert np.max(np.abs(unweighted.x_hat - weighted.x_hat)) > 1e-3


def test_runner_writes_reusable_artifacts_on_tiny_development_smoke(tmp_path: Path):
    config = {
        "schema_version": "proportional-graph-neural-smoke-v1",
        "artifact_schema_version": "proportional-graph-neural-artifact-v1",
        "analysis_universe": {"require_path_shuffle_eligible": True},
        "graph": {
            **ProportionalGraphConfig(
                masters=16,
                train_fraction=0.5,
                calibration_fraction=0.125,
                validation_fraction=0.125,
                n_min=4,
                n_max=5,
                extra_edge_probability=0.8,
                irls_iterations=100,
                seed=123,
            ).to_dict()
        },
        "training": {
            "batch_size": 4,
            "device": "cpu",
            "epochs": 1,
            "grad_clip": 5.0,
            "hidden_dim": 8,
            "latency_repeats": 3,
            "learning_rate": 0.001,
            "max_rss_gib": 8.0,
            "max_seconds": 300,
            "seeds": [17],
            "torch_threads": 1,
            "weight_decay": 0.0001,
        },
        "loss": {"closure_l1": 0.05, "quotient_mse": 1.0, "relation_mse": 1.0},
        "inference": {
            "contrast_direction": "negative_favors_positive_term_first_named_arm",
            "primary_order": [
                "typed_effect_raw",
                "typed_effect_closure",
                "factorial_interaction",
            ],
            "secondary_family_order": [
                "closure_effect_generic",
                "closure_effect_typed",
                "typed_closure_vs_path_shuffle",
                "path_mixing_vs_pair_state",
                "typed_path_vs_generic_message_passing",
            ],
            "secondary_multiplicity_method": (
                "holm_all_metrics_slices_and_solver_interactions"
            ),
            "secondary_family_scope": (
                "all_estimable_two_sided_bootstrap_tail_probabilities"
            ),
        },
        "arms": [
            {
                "name": "raw_generic",
                "architecture": "path_mixer",
                "evidence": "raw",
                "mixer": "generic",
                "mix_paths": True,
                "path_shuffle": False,
            },
            {
                "name": "raw_typed",
                "architecture": "path_mixer",
                "evidence": "raw",
                "mixer": "typed",
                "mix_paths": True,
                "path_shuffle": False,
            },
            {
                "name": "closure_generic",
                "architecture": "path_mixer",
                "evidence": "closure",
                "mixer": "generic",
                "mix_paths": True,
                "path_shuffle": False,
            },
            {
                "name": "closure_typed",
                "architecture": "path_mixer",
                "evidence": "closure",
                "mixer": "typed",
                "mix_paths": True,
                "path_shuffle": False,
            },
            {
                "name": "closure_typed_path_shuffle",
                "architecture": "path_mixer",
                "evidence": "closure",
                "mixer": "typed",
                "mix_paths": True,
                "path_shuffle": True,
            },
            {
                "name": "pair_state_no_mix",
                "architecture": "path_mixer",
                "evidence": "raw",
                "mixer": "generic",
                "mix_paths": False,
                "path_shuffle": False,
            },
            {
                "name": "generic_message_passing",
                "architecture": "generic_message_passing",
                "evidence": "raw",
                "mixer": "generic",
                "mix_paths": True,
                "path_shuffle": False,
            },
            {
                "name": "edge_mlp",
                "architecture": "edge_mlp",
                "evidence": "raw",
                "mixer": "generic",
                "mix_paths": False,
                "path_shuffle": False,
            },
        ],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "output"
    subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "experiments/geometria_proporcional/run_proportional_graph_neural_smoke.py"
            ),
            "--config",
            str(config_path),
            "--output",
            str(output),
            "--development",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema_version"] == "proportional-graph-neural-artifact-v1"
    dataset_manifest = json.loads((output / "dataset_manifest.json").read_text())
    assert len(dataset_manifest) == 20
    excluded_view_ids = {
        row["view_id"] for row in dataset_manifest if not row["analysis_included"]
    }
    assert excluded_view_ids
    assert all(
        row["analysis_exclusion_reason"] == "no_feasible_balanced_path_derangement"
        for row in dataset_manifest
        if row["view_id"] in excluded_view_ids
    )
    assert len(list((output / "checkpoints").glob("*.npz"))) == 8
    assert len(list((output / "raw_eval").glob("*.npz"))) == 11
    assert manifest["public_schema_sha256"]
    runtime = json.loads((output / "runtime_observation.json").read_text())
    assert len(runtime["by_model_run"]) == 8
    assert all(
        row["forward_latency"]["median_seconds"] > 0
        for row in runtime["by_model_run"].values()
    )
    with np.load(
        next((output / "raw_eval").glob("raw_typed*.npz")), allow_pickle=False
    ) as data:
        assert {
            "corrected_log_ratio",
            "reliability",
            "x_hat_wls",
            "x_hat_direct",
            "x_hat_irls",
            "edge_index",
            "edge_valid",
            "edge_variance",
            "path_index",
            "path_sign",
            "path_valid",
            "path_offsets",
            "n_nodes",
            "wls_laplacian_rank",
            "wls_condition",
            "irls_laplacian_rank",
            "irls_condition",
        } <= set(data.files)
    with np.load(
        next((output / "checkpoints").glob("*.npz")), allow_pickle=False
    ) as checkpoint:
        assert (
            checkpoint["format_version"].item() == "proportional-neural-checkpoint-v1"
        )
        assert any(key.startswith("model::") for key in checkpoint.files)
        assert any(key.startswith("optimizer::") for key in checkpoint.files)
    assert (output / "contrasts.json").exists()
    contrasts = json.loads((output / "contrasts.json").read_text())
    assert all(
        "solver_interaction_irls_minus_wls" in payload for payload in contrasts.values()
    )
    assert contrasts["typed_effect_raw"]["family_role"] == "primary"
    assert (
        contrasts["typed_effect_raw"]["metrics"]["relation_rmse"]["test|iid"][
            "p_holm_secondary_family"
        ]
        is None
    )
    assert contrasts["closure_effect_generic"]["family_role"] == "secondary"
    assert (
        contrasts["closure_effect_generic"]["metrics"]["relation_rmse"]["test|iid"][
            "p_holm_secondary_family"
        ]
        is not None
    )
    report = (output / "SMOKE_REPORT.md").read_text(encoding="utf-8")
    frozen_contrast_order = (
        config["inference"]["primary_order"]
        + config["inference"]["secondary_family_order"]
    )
    first_report_positions = [
        report.index(f"| `{contrasts[name]['family_role']}` | `{name}` |")
        for name in frozen_contrast_order
    ]
    assert first_report_positions == sorted(first_report_positions)
    metrics_by_key = json.loads((output / "per_view_metrics.json").read_text())
    assert all(
        row["view_id"] not in excluded_view_ids
        for rows in metrics_by_key.values()
        for row in rows
    )
    compute_contract = json.loads((output / "compute_contract.json").read_text())
    assert (
        "common train, validation and test analysis universe"
        in compute_contract["operation_contract"]["path_shuffle_exclusion"]
    )
    assert (output / "bootstrap_indices.npz").exists()
    assert (output / "replay.sh").exists()
    original_hashes = {
        path.relative_to(output): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output.rglob("*")
        if path.is_file() and path.name != "runtime_observation.json"
    }
    replayed = tmp_path / "replayed"
    subprocess.run(
        [str(output / "replay.sh"), str(replayed)], cwd=REPO_ROOT, check=True
    )
    assert original_hashes == {
        path.relative_to(output): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output.rglob("*")
        if path.is_file() and path.name != "runtime_observation.json"
    }
    assert {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file() and path.name != "runtime_observation.json"
    } == {
        path.relative_to(replayed): path.read_bytes()
        for path in replayed.rglob("*")
        if path.is_file() and path.name != "runtime_observation.json"
    }
    sentinel = replayed / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")
    refused = subprocess.run(
        [str(output / "replay.sh"), str(replayed)], cwd=REPO_ROOT, capture_output=True
    )
    assert refused.returncode != 0
    assert sentinel.read_text(encoding="utf-8") == "preserve"
