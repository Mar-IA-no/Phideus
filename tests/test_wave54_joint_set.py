import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from geometria_proporcional.wave53_uncertainty import independent_nonempty_mass
from geometria_proporcional.wave54_joint_set import (
    centered_interactions,
    empirical_set_prior,
    expected_regret_from_mass,
    feature_tensor,
    fit_joint_posterior,
    marginal_probability,
    nll_and_gradient,
    parameter_size,
    posterior_mass,
    reference_parameters,
    target_set_indices,
)


def load_script(name: str):
    path = Path(__file__).parents[1] / "experiments/geometria_proporcional" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_full_parameterization_has_identified_zero_sum_interactions():
    theta = np.arange(12, dtype=float)
    interaction = centered_interactions(theta, "joint_full")
    assert parameter_size("joint_full") == 12
    assert len(interaction) == 6
    assert interaction.sum() == 0.0


def test_reference_posterior_matches_independent_raw_bernoulli():
    logits = np.array([[0.3, -1.1, 2.0, 0.7], [-2.0, 0.4, 0.1, 1.5]])
    expected = independent_nonempty_mass(expit(logits))[1]
    for structure in ("joint_unary", "joint_unary_cardinality", "joint_full"):
        actual = posterior_mass(logits, reference_parameters(structure), structure)
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_joint_gradient_matches_finite_difference():
    rng = np.random.default_rng(54)
    logits = rng.normal(size=(7, 4))
    target = rng.random((7, 4)) > 0.5
    target[~target.any(axis=1), 0] = True
    features = feature_tensor(logits, "joint_full")
    target_index = target_set_indices(target)
    reference = reference_parameters("joint_full")
    theta = reference + rng.normal(scale=0.2, size=len(reference))
    value, gradient = nll_and_gradient(theta, features, target_index, 0.1, reference)
    assert np.isfinite(value)
    numeric = np.empty_like(theta)
    eps = 1e-6
    for i in range(len(theta)):
        left = theta.copy()
        right = theta.copy()
        left[i] -= eps
        right[i] += eps
        lv = nll_and_gradient(left, features, target_index, 0.1, reference)[0]
        rv = nll_and_gradient(right, features, target_index, 0.1, reference)[0]
        numeric[i] = (rv - lv) / (2 * eps)
    np.testing.assert_allclose(gradient, numeric, rtol=2e-5, atol=2e-7)


def test_fit_reduces_regularized_objective_and_is_finite():
    rng = np.random.default_rng(55)
    logits = rng.normal(size=(80, 4))
    target = logits > 0.0
    target[~target.any(axis=1), np.argmax(logits[~target.any(axis=1)], axis=1)] = True
    structure = "joint_full"
    reference = reference_parameters(structure)
    features = feature_tensor(logits, structure)
    indices = target_set_indices(target)
    initial = nll_and_gradient(reference, features, indices, 0.01, reference)[0]
    result = fit_joint_posterior(logits, target, structure, 0.01)
    assert result["objective"] < initial
    assert result["gradient_norm"] < 1e-5
    assert np.all(np.isfinite(result["theta"]))


def test_empirical_prior_is_smoothed_and_normalized():
    target = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=bool)
    prior = empirical_set_prior(target, alpha=1.0)
    assert np.all(prior > 0.0)
    np.testing.assert_allclose(prior.sum(), 1.0)
    assert prior[0] > prior[1] > prior[2]


def test_joint_mass_produces_marginals_and_expected_regret_actions():
    mass = np.full((3, 15), 1.0 / 15.0)
    marginal = marginal_probability(mass)
    assert marginal.shape == (3, 4)
    utilities = np.array([[1.0, 0.6, 0.2, -0.2], [-0.2, 0.2, 0.6, 1.0]])
    result = expected_regret_from_mass(mass, utilities, 1.25)
    assert result["actions"].shape == (3, 2)
    assert result["action_risk"].shape == (3, 2, 4)
    assert np.all(result["margin"] >= 0.0)


def test_regularization_selection_prefers_larger_lambda_on_exact_tie():
    runner = load_script("run_wave54_joint_set.py")
    rows = [
        {"regularization": 0.1, "primary_nll": 1.0},
        {"regularization": 1.0, "primary_nll": 1.0},
    ]
    assert runner.select_regularization(rows, "primary")["regularization"] == 1.0


def test_preparer_metadata_filters_noncanonical_and_unselected_rows(tmp_path):
    preparer = load_script("prepare_wave54_inputs.py")
    rows = [
        {
            "pair_token": "keep",
            "calibration_population": "origin_translation_break",
            "design_stratum": "wrong",
            "oracle_compatible_set": ["a"],
        },
        {
            "pair_token": "keep",
            "calibration_population": "canonical_preserving",
            "design_stratum": "NEAR_RIVAL",
            "oracle_compatible_set": ["a", "b"],
        },
        {
            "pair_token": "ignore",
            "calibration_population": "canonical_preserving",
            "design_stratum": "EASY",
            "oracle_compatible_set": ["a"],
        },
    ]
    path = tmp_path / "labels.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    assert preparer.load_metadata(path, {"keep"}) == {
        "keep": {"design_stratum": "NEAR_RIVAL", "cardinality": 2}
    }


def test_experiment_wrappers_reject_lockbox_and_overlapping_paths(tmp_path):
    preparer = load_script("prepare_wave54_inputs.py")
    runner = load_script("run_wave54_joint_set.py")
    source = tmp_path / "source"
    nested = source / "nested"
    source.mkdir()
    assert preparer.paths_overlap(source, nested)
    assert runner.paths_overlap(nested, source)
    for module in (preparer, runner):
        try:
            module.reject_lockbox_paths([tmp_path / "sealed_lockbox" / "data"])
        except ValueError:
            pass
        else:
            raise AssertionError("lockbox path was not rejected")


def test_preparer_rejects_duplicate_or_overlapping_bundle_tokens():
    preparer = load_script("prepare_wave54_inputs.py")
    with np.testing.assert_raises(RuntimeError):
        preparer.validate_token_separation(
            {"pair_token": np.array(["a", "a"])},
            {"pair_token": np.array(["b"])},
        )
    with np.testing.assert_raises(RuntimeError):
        preparer.validate_token_separation(
            {"pair_token": np.array(["a", "b"])},
            {"pair_token": np.array(["b", "c"])},
        )


def test_runner_rejects_bundle_overlap():
    runner = load_script("run_wave54_joint_set.py")
    with np.testing.assert_raises(RuntimeError):
        runner.validate_bundle_separation(
            {"pair_token": np.array(["fit", "shared"])},
            {"pair_token": np.array(["shared", "monitor"])},
        )


def test_force_archives_existing_output_instead_of_deleting_it(tmp_path):
    runner = load_script("run_wave54_joint_set.py")
    output = tmp_path / "result"
    output.mkdir()
    (output / "evidence.txt").write_text("keep", encoding="utf-8")
    archived = runner.prepare_output_directory(output, force=True)
    assert archived is not None
    assert (archived / "evidence.txt").read_text(encoding="utf-8") == "keep"
    assert output.is_dir()
    assert not any(output.iterdir())


def test_runner_output_guard_rejects_directory_containing_execution_source(tmp_path):
    runner = load_script("run_wave54_joint_set.py")
    source_root = tmp_path / "src"
    execution_source = source_root / "package" / "primitive.py"
    execution_source.parent.mkdir(parents=True)
    execution_source.write_text("# source\n", encoding="utf-8")
    with np.testing.assert_raises(ValueError):
        runner.validate_output_path(
            source_root,
            [tmp_path / "external-input"],
            [execution_source],
            repo_root=tmp_path,
        )
