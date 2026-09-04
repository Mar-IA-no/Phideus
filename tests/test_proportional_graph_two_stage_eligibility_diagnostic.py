from __future__ import annotations
import importlib.util, os, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; SCRIPT = ROOT / "experiments/geometria_proporcional/run_proportional_graph_two_stage_eligibility_diagnostic.py"; SPEC = importlib.util.spec_from_file_location("two_stage", SCRIPT); assert SPEC and SPEC.loader; MODULE = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = MODULE; SPEC.loader.exec_module(MODULE)

def test_config_and_source_are_cpu_frozen():
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG); _, source = MODULE.verify_source(cfg); assert source["miscoverage"] == 0.10; assert os.environ["CUDA_VISIBLE_DEVICES"] == ""

def test_filtered_action_preserves_mean_order_and_budget():
    order = np.asarray([3, 1, 4, 0, 2]); proposal = np.asarray([1, 2, 3, 4, 1]); eligible = np.asarray([True, False, True, True, True]); action = MODULE.filtered_action(order, proposal, eligible, 2); np.testing.assert_array_equal(action, [0, 0, 0, 4, 1])

def test_rejected_filter_cannot_reorder_remaining_views():
    order = np.asarray([2, 0, 3, 1]); proposal = np.ones(4, dtype=np.int64); eligible = np.asarray([True, True, False, True]); action = MODULE.filtered_action(order, proposal, eligible, 3); assert np.flatnonzero(action).tolist() == [0, 1, 3]
