from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/geometria_proporcional/run_proportional_graph_signed_tail_gate.py"
SPEC = importlib.util.spec_from_file_location("signed_tail_gate", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_config_and_source_are_frozen_cpu_contracts() -> None:
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    MODULE.verify_source(cfg)
    assert cfg["execution"]["torch_threads"] == 1
    assert cfg["realizations"] == {
        "risk_fit_seed": 2026091109,
        "risk_calibration_seed": 2026091117,
        "policy_selection_seed": 2026091129,
        "adjudication_seed": 2026091137,
        "min_eligible_masters": 220,
    }


def test_interfaces_use_distinct_upper_bound_mechanics() -> None:
    mu = np.asarray([[-0.30, -0.10, 0.20, 0.40], [-0.10, -0.20, 0.30, 0.50]])
    signed_risk = np.asarray([[0.10, 0.05, 0.0, 0.0], [0.20, 0.10, 0.0, 0.0]])
    signed_action, signed_upper = MODULE.signed_action(mu, signed_risk, 0.02)
    absolute_action, absolute_upper = MODULE.absolute_action(mu, np.ones_like(mu), 0.15)
    np.testing.assert_array_equal(signed_action, np.asarray([1, 2]))
    np.testing.assert_array_equal(absolute_action, np.asarray([1, 2]))
    np.testing.assert_allclose(signed_upper, mu + signed_risk + 0.02)
    np.testing.assert_allclose(absolute_upper, mu + 0.15)


def test_interface_firewall_falls_back_to_exact_identity() -> None:
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    actions = {
        interface: {
            family: np.ones((4, 8), dtype=np.int64)
            for family in MODULE.FAMILY_BASES
        }
        for interface in MODULE.INTERFACES
    }
    controls = {
        interface: np.ones((4, cfg["control_replicates"], 8), dtype=np.int64)
        for interface in MODULE.INTERFACES
    }
    firewall = {"interfaces": {}}
    for interface in MODULE.INTERFACES:
        firewall["interfaces"][interface] = {"arms": {}}
        for arm_index, arm in enumerate(cfg["arms"]):
            firewall["interfaces"][interface]["arms"][arm] = {
                "families": {
                    family: {"deploy": arm_index != 0}
                    for family in MODULE.FAMILY_BASES
                },
                MODULE.CONTROL_BASE: [
                    {"deploy": replicate % 2 == 0}
                    for replicate in range(cfg["control_replicates"])
                ],
            }
    deployed, deployed_controls = MODULE.apply_firewall(cfg, actions, controls, firewall)
    for interface in MODULE.INTERFACES:
        for family in MODULE.FAMILY_BASES:
            np.testing.assert_array_equal(deployed[interface][family][0], 0)
            np.testing.assert_array_equal(deployed[interface][family][1:], 1)
        np.testing.assert_array_equal(deployed_controls[interface][:, 1::2], 0)


def test_frozen_runtime_config_cannot_be_replaced(tmp_path: Path) -> None:
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    (tmp_path / "resolved_config.json").write_text(json.dumps(cfg))
    MODULE.verify_resolved_config(cfg, tmp_path)
    altered = {**cfg, "quantile": 0.8}
    try:
        MODULE.verify_resolved_config(altered, tmp_path)
    except AssertionError as error:
        assert "differs" in str(error)
    else:
        raise AssertionError("altered config passed freeze verification")


def test_phase_freeze_rejects_unmanifested_files(tmp_path: Path) -> None:
    role = tmp_path / "risk_fit"
    role.mkdir()
    (role / "expected.txt").write_text("frozen")
    MODULE.write_phase_freeze(tmp_path, "risk_fit", ("risk_fit",), {}, [])
    MODULE.verify_phase(tmp_path, "risk_fit")
    (role / "unexpected.txt").write_text("late")
    try:
        MODULE.verify_phase(tmp_path, "risk_fit")
    except AssertionError as error:
        assert "file set" in str(error)
    else:
        raise AssertionError("unmanifested file passed freeze verification")
