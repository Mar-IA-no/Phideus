from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/geometria_proporcional/run_proportional_graph_mean_ranking_attribution.py"
SPEC = importlib.util.spec_from_file_location("mean_attribution", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_config_and_sources_are_cpu_frozen():
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    r368_cfg, _, models = MODULE.verify_sources(cfg)
    assert r368_cfg["source_manifest_sha256"] == "814aced29af86ddb0d0f4e39611ab2fc31058304b7397c27d9319327485e281c"
    assert set(models["arms"]) == set(cfg["arms"])
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_fixed_proposal_uses_alpha_quarter_action_code():
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    state = {"delta": np.zeros((1, 7, 4))}
    proposal = MODULE.proposal_for(cfg, state, 0, "fixed_alpha_0.25")
    np.testing.assert_array_equal(proposal, np.ones(7, dtype=np.int64))
    assert cfg["alphas"][proposal[0]] == 0.25


def test_common_proposal_is_shared_before_mean_family_ranking():
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    state = {
        "delta": np.zeros((1, 3, 4)),
        "mu": np.asarray([[[0.3, 0.1, 0.2, 0.4], [0.0, -0.1, 0.2, 0.3], [0.4, 0.3, 0.2, 0.1]]]),
        "predicted": {"public_base_selected_action": np.zeros((1, 3, 4))},
    }
    proposal = MODULE.proposal_for(cfg, state, 0, "r368_common")
    np.testing.assert_array_equal(proposal, np.asarray([2, 2, 4]))
    assert set(MODULE.MODEL_FAMILY) == set((*MODULE.MAIN, *MODULE.CONTROLS))
