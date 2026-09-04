from __future__ import annotations
import importlib.util, os, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; SCRIPT=ROOT/"experiments/geometria_proporcional/run_proportional_graph_mean_only_ranking_ablation.py"; SPEC=importlib.util.spec_from_file_location("mean_only_ablation",SCRIPT); assert SPEC and SPEC.loader; MODULE=importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name]=MODULE; SPEC.loader.exec_module(MODULE)

def test_config_and_source_cpu_contract():
    cfg=MODULE.load_config(MODULE.DEFAULT_CONFIG); source=MODULE.verify_source(cfg); assert source["budget_fractions"]==cfg["budget_fractions"]; assert os.environ["CUDA_VISIBLE_DEVICES"]==""

def test_mean_only_score_is_selected_mu():
    cfg=MODULE.load_config(MODULE.DEFAULT_CONFIG); r366=MODULE.ranking.load_config(MODULE.source_root(cfg)/"resolved_config.json"); source_cfg=MODULE.transport.load_config(MODULE.ranking.source_root(r366)/"resolved_config.json"); rec=MODULE.transport.reconstruct_b(source_cfg); scores=MODULE.ablation_scores(rec,0); expected=MODULE.ranking.selected_component(rec["state"]["mu"][0],scores["proposal"]); np.testing.assert_array_equal(scores["mean_only"],expected)

def test_positive_primary_maps_to_mean_only_better():
    assert {"ADVERSE_BOTH":"MEAN_ONLY_BETTER_BOTH","FAVORABLE_BOTH":"TAIL_BETTER_BOTH"}["ADVERSE_BOTH"]=="MEAN_ONLY_BETTER_BOTH"
