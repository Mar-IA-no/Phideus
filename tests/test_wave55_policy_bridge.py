from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from geometria_proporcional.wave55_policy_bridge import (
    HARD_ONLY,
    action_metric_arrays,
    algebraic_sign,
    bridge_actions,
    override_diagnostics,
    select_gamma,
)


UTILITIES = np.asarray([[1.0, 0.6, 0.2, -0.2], [-0.2, 0.2, 0.6, 1.0]])
REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_hard_only_is_exact_and_numeric_gate_is_strict() -> None:
    risk = np.asarray([[[0.4, 0.1, 0.8, 0.9], [0.2, 0.3, 0.1, 0.7]]])
    hard = np.asarray([[0, 0]])
    identity = bridge_actions(risk, hard, HARD_ONLY)
    assert np.array_equal(identity["actions"], hard)
    assert not identity["override"].any()

    gated = bridge_actions(risk, hard, 0.1)
    assert gated["override"].tolist() == [[True, False]]
    assert gated["actions"].tolist() == [[1, 0]]


def test_select_gamma_enforces_constraints_and_prefers_conservatism_on_tie() -> None:
    rows = [
        {"gamma": 0.0, "accuracy": 0.88, "compatible": 0.91, "regret": 0.10},
        {"gamma": 0.2, "accuracy": 0.90, "compatible": 0.91, "regret": 0.11},
        {"gamma": 0.4, "accuracy": 0.90, "compatible": 0.91, "regret": 0.11},
        {"gamma": HARD_ONLY, "accuracy": 0.90, "compatible": 0.91, "regret": 0.12},
    ]
    selected = select_gamma(rows, hard_accuracy=0.90, hard_compatible=0.91)
    assert selected["selected"]["gamma"] == 0.4
    assert selected["grid"][0]["feasible"] is False


def test_override_diagnostics_uses_non_neutral_precision_denominator() -> None:
    target = np.asarray([[True, False, True, False]])
    hard = np.asarray([[2, 0]])
    bridge = {
        "actions": np.asarray([[0, 1]]),
        "override": np.asarray([[True, True]]),
    }
    out = override_diagnostics(bridge, hard, target, UTILITIES, 1.25)
    assert out["n_overrides"] == 2
    assert out["n_non_neutral_overrides"] == 2
    assert out["override_precision"] == pytest.approx(0.5)
    assert out["beneficial_fraction_all"] == pytest.approx(0.5)
    assert out["harmful_fraction_all"] == pytest.approx(0.5)


def test_empty_override_metrics_are_json_safe() -> None:
    target = np.asarray([[True, False, True, False]])
    hard = np.asarray([[0, 0]])
    bridge = {"actions": hard.copy(), "override": np.zeros_like(hard, dtype=bool)}
    out = override_diagnostics(bridge, hard, target, UTILITIES, 1.25)
    assert out["override_precision"] is None
    assert out["regret_conditioned_on_override"] is None


def test_override_diagnostics_respects_primary_token_mask() -> None:
    target = np.asarray([
        [True, False, True, False],
        [True, False, True, False],
    ])
    hard = np.asarray([[2, 0], [0, 0]])
    bridge = {
        "actions": np.asarray([[0, 1], [1, 1]]),
        "override": np.ones((2, 2), dtype=bool),
    }
    out = override_diagnostics(
        bridge,
        hard,
        target,
        UTILITIES,
        1.25,
        token_mask=np.asarray([True, False]),
    )
    assert out["n_overrides"] == 2
    assert out["override_precision"] == pytest.approx(0.5)


def test_action_metrics_average_policies_within_token() -> None:
    target = np.asarray([[True, False, True, False]])
    actions = np.asarray([[0, 2]])
    out = action_metric_arrays(actions, target, UTILITIES, 1.25)
    assert out["accuracy"].shape == (1,)
    assert out["compatible"].tolist() == [1.0]
    assert out["regret"].tolist() == [0.0]


@pytest.mark.parametrize("value, expected", [(-1e-3, -1), (0.0, 0), (1e-13, 0), (1e-3, 1)])
def test_algebraic_sign(value: float, expected: int) -> None:
    assert algebraic_sign(value) == expected


def test_primary_redraw_is_rejected_after_a_key_bearing_failure(tmp_path: Path) -> None:
    prep = load_script("wave55_prepare_test", "experiments/geometria_proporcional/prepare_wave55_fresh.py")
    primary = tmp_path / "wave55_policy_bridge_fresh_v1"
    failed = tmp_path / "wave55_policy_bridge_fresh_v1.failed_20260903T000000Z"
    key = failed / "benchmark/sealed/generation_secret.json"
    key.parent.mkdir(parents=True)
    key.write_text('{"key_hex":"' + "00" * 32 + '"}')
    args = argparse.Namespace(
        replay_secrets_from=None,
        recovery_secrets_from=None,
        reference_dir=None,
        force=False,
    )
    config = {
        "primary_output_name": "wave55_policy_bridge_fresh_v1",
        "replay_output_name": "wave55_policy_bridge_fresh_v1_replay",
        "output_parent_relative": ".",
    }
    with pytest.raises(RuntimeError, match="already drew keys"):
        prep.validate_invocation(args, primary, config, repo_root=tmp_path)


def test_replay_requires_distinct_primary_as_reference_and_key_source(tmp_path: Path) -> None:
    prep = load_script("wave55_prepare_replay_test", "experiments/geometria_proporcional/prepare_wave55_fresh.py")
    primary = tmp_path / "wave55_policy_bridge_fresh_v1"
    replay = tmp_path / "wave55_policy_bridge_fresh_v1_replay"
    primary.mkdir()
    args = argparse.Namespace(
        replay_secrets_from=primary,
        recovery_secrets_from=None,
        reference_dir=primary,
        force=False,
    )
    config = {
        "primary_output_name": primary.name,
        "replay_output_name": replay.name,
        "output_parent_relative": ".",
    }
    assert prep.validate_invocation(args, replay, config, repo_root=tmp_path) == "replay"
    replay.mkdir()
    args.reference_dir = replay
    with pytest.raises(ValueError, match="distinct primary"):
        prep.validate_invocation(args, replay, config, repo_root=tmp_path)


def test_output_parent_is_canonical(tmp_path: Path) -> None:
    prep = load_script("wave55_prepare_parent_test", "experiments/geometria_proporcional/prepare_wave55_fresh.py")
    args = argparse.Namespace(
        replay_secrets_from=None,
        recovery_secrets_from=None,
        reference_dir=None,
        force=False,
    )
    config = {
        "primary_output_name": "wave55_policy_bridge_fresh_v1",
        "replay_output_name": "wave55_policy_bridge_fresh_v1_replay",
        "output_parent_relative": "canonical",
    }
    with pytest.raises(ValueError, match="directly under"):
        prep.validate_invocation(
            args,
            tmp_path / "elsewhere/wave55_policy_bridge_fresh_v1",
            config,
            repo_root=tmp_path,
        )


def test_theta_semantic_freeze_crosscheck() -> None:
    runner = load_script("wave55_runner_test", "experiments/geometria_proporcional/run_wave55_policy_bridge.py")
    theta = {
        "joint_full": np.asarray([1.0, 2.0]),
        "joint_unary_cardinality": np.asarray([3.0]),
        "joint_full_target_shuffled": np.asarray([4.0]),
    }
    freeze = {
        "selected_models": {name: {"theta": value.tolist()} for name, value in theta.items()},
        "selections": {
            "joint_full": {"primary": {"theta": theta["joint_full"].tolist()}},
            "joint_unary_cardinality": {"primary": {"theta": theta["joint_unary_cardinality"].tolist()}},
        },
    }
    runner.verify_primary_theta(theta, freeze)
    freeze["selections"]["joint_full"]["primary"]["theta"][0] = 99.0
    with pytest.raises(RuntimeError, match="not the primary"):
        runner.verify_primary_theta(theta, freeze)
