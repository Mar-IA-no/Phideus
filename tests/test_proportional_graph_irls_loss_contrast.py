from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "experiments/geometria_proporcional/run_proportional_graph_irls_loss_contrast.py"
)
SPEC = importlib.util.spec_from_file_location("irls_loss_contrast", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

from geometria_proporcional.proportional_graph_contract import (  # noqa: E402
    ProportionalGraphConfig,
    generate_graph_views,
)


def setup_case():
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    source = MODULE.smoke._load_config((ROOT / cfg["source_config"]).resolve())
    cfg["source"] = source
    arm = next(a for a in source["arms"] if a["name"] == "raw_generic")
    small = ProportionalGraphConfig(
        masters=16,
        train_fraction=0.5,
        calibration_fraction=0.125,
        validation_fraction=0.125,
        n_min=5,
        n_max=6,
    )
    views = [v for v in generate_graph_views(small) if v.private.split == "train"]
    scale = MODULE.smoke._input_scale(views)
    model = MODULE.smoke._model_for_arm(arm, source, scale)
    return cfg, arm, views, model


def test_official_config_is_frozen_cpu_only() -> None:
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    assert cfg["surrogate"]["steps"] == 64
    assert cfg["training"]["torch_threads"] == 1
    assert cfg["training"]["epochs"] == 5


def test_post_irls_objective_only_gradients_correction_head() -> None:
    cfg, arm, views, model = setup_case()
    MODULE.configure_head(model)
    loss, components = MODULE.objective(
        model, views[0], arm, cfg["seeds"][0], cfg, "post_irls"
    )
    loss.backward()
    assert components["quotient_mse"] is not None
    assert all(
        (parameter.grad is not None) == name.startswith("correction_head.")
        for name, parameter in model.named_parameters()
    )


def test_gradient_scale_probe_is_finite_and_non_mutating() -> None:
    cfg, arm, views, local = setup_case()
    post = MODULE.smoke._model_for_arm(
        arm, cfg["source"], MODULE.smoke._input_scale(views)
    )
    post.load_state_dict(local.state_dict())
    before = [p.detach().clone() for p in local.parameters()]
    cfg["training"]["batch_size"] = 4
    cfg["training"]["scale_probe_batches"] = 1
    record = MODULE.probe_scale(
        {"local_relation": local, "post_irls": post}, views, arm, cfg["seeds"][0], cfg
    )
    assert np.isfinite(record["post_irls_multiplier"])
    assert record["post_irls_multiplier"] > 0
    assert all(
        np.isfinite(v) and v > 0
        for values in record["gradient_norms"].values()
        for v in values
    )
    assert all((old == new).all() for old, new in zip(before, local.parameters()))


def test_batch_order_is_shared_and_deterministic() -> None:
    first = MODULE.batch_order(32, "raw_generic", 104729, 0)
    second = MODULE.batch_order(32, "raw_generic", 104729, 0)
    np.testing.assert_array_equal(first, second)
    assert sorted(first.tolist()) == list(range(32))
