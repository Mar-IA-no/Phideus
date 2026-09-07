"""CPU fixtures for the set-valued native preflight."""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.geometria_proporcional import proportional_set_valued_native as native
from src.geometria_proporcional.wave53_uncertainty import nonempty_sets
from src.geometria_proporcional.wave54_joint_set import (
    feature_tensor,
    nll_and_gradient,
    reference_parameters,
    target_set_indices,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class SetValuedNativeTests(unittest.TestCase):
    def test_import_does_not_load_torch(self) -> None:
        environment = dict(os.environ)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        script = (
            "import sys; "
            "import src.geometria_proporcional.proportional_set_valued_native; "
            "assert 'torch' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, env=environment, check=True)

    def test_hard_uses_set_map_and_deterministic_ties(self) -> None:
        sets = nonempty_sets(4)
        mass = np.full((1, 15), 0.0)
        mass[0, 0] = 0.40  # {0}, the MAP set.
        mass[0, 1] = 0.35
        mass[0, 5] = 0.25  # {1,2}; marginal threshold instead yields {1}.
        utility = np.tile(np.asarray([1.0, 1.0, 0.0, -1.0]), (24, 1))
        hard = native.hard_map_reader(mass, utility)
        self.assertEqual(int(hard["map_set_index"][0]), 0)
        self.assertTrue(np.array_equal(hard["map_set"][0], sets[0]))
        self.assertTrue(np.all(hard["actions"] == 0))
        marginal_threshold_set = (mass @ sets.astype(float))[0] >= 0.5
        self.assertFalse(np.array_equal(marginal_threshold_set, hard["map_set"][0]))

    def test_joint_gradient_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(12)
        logits = rng.normal(size=(9, 4))
        target = rng.random((9, 4)) < 0.45
        target[~target.any(axis=1), 0] = True
        features = feature_tensor(logits, "joint_full")
        target_index = target_set_indices(target)
        theta = reference_parameters("joint_full") + rng.normal(scale=0.1, size=12)
        value, gradient = nll_and_gradient(
            theta, features, target_index, 0.1, reference_parameters("joint_full")
        )
        self.assertTrue(np.isfinite(value))
        epsilon = 1e-6
        finite = np.empty_like(theta)
        for index in range(len(theta)):
            shift = np.zeros_like(theta); shift[index] = epsilon
            plus = nll_and_gradient(theta + shift, features, target_index, 0.1, reference_parameters("joint_full"))[0]
            minus = nll_and_gradient(theta - shift, features, target_index, 0.1, reference_parameters("joint_full"))[0]
            finite[index] = (plus - minus) / (2 * epsilon)
        np.testing.assert_allclose(gradient, finite, rtol=0.0, atol=2e-9)

    def test_joint_shuffled_runs_own_grid(self) -> None:
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(48, 4))
        target = rng.random((48, 4)) < (0.25 + 0.45 / (1.0 + np.exp(-logits)))
        empty = ~target.any(axis=1)
        target[empty, np.argmax(logits[empty], axis=1)] = True
        folds = np.arange(48, dtype=np.int64) % 4
        real = native.fit_joint_cv(logits, target, folds)
        shuffled = native.fit_joint_cv(logits, target[np.roll(np.arange(48), 1)], folds)
        self.assertEqual(real["state"]["selected_regularization"], 0.1)
        self.assertEqual(shuffled["state"]["selected_regularization"], 0.01)

    def test_target_shuffle_fixture_digest(self) -> None:
        rows = [
            ("a", 0, "FAR", 2), ("b", 0, "FAR", 2), ("c", 0, "FAR", 2),
            ("d", 0, "NEAR", 1), ("e", 1, "FAR", 2), ("f", 1, "FAR", 2),
        ]
        result = native.target_derangement_v1(
            [row[0] for row in rows], np.asarray([row[1] for row in rows]),
            [row[2] for row in rows], np.asarray([row[3] for row in rows]),
        )
        payload = (
            json.dumps(result["rows"], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        ).encode()
        import hashlib

        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "d7aa2f128b6d42dbe7448415dcd8d4d69ca0ad8311394a5b1209ca9579e03904",
        )

    def test_portable_reader_states(self) -> None:
        rng = np.random.default_rng(18)
        design = rng.normal(size=(60, len(native.FEATURE_NAMES)))
        weights = rng.uniform(0.2, 1.0, size=60)
        regression = native.fit_ridge_state(design, rng.normal(size=60), weights)
        classification = native.fit_logistic_state(
            design, np.arange(60) % 2, weights, target_name="harm"
        )
        self.assertTrue(np.all(np.isfinite(native.score_linear_state(regression, design))))
        probability = native.score_linear_state(classification, design)
        self.assertTrue(np.all((probability > 0.0) & (probability < 1.0)))

    def test_advantage_preserves_tolerated_negative_roundoff(self) -> None:
        mass = np.full((1, 15), 1.0 / 15.0)
        risk = np.full((1, 24, 4), 2.0)
        risk[:, :, 0] = 1.0 - 5e-13
        risk[:, :, 1] = 1.0
        hard = {
            "actions": np.zeros((1, 24), dtype=np.int64),
            "map_set": np.broadcast_to(nonempty_sets(4)[0], (1, 4)),
            "map_set_mass": np.asarray([1.0 / 15.0]),
        }
        result = native.contextual_design_map(
            ensemble_logits=np.zeros((1, 4)),
            per_seed_logits=np.zeros((3, 1, 4)),
            set_mass=mass,
            action_risk=risk,
            hard_state=hard,
            posterior_actions=np.ones((1, 24), dtype=np.int64),
            utilities=np.tile(np.asarray([1.0, 0.6, 0.2, -0.2]), (24, 1)),
        )
        self.assertTrue(np.all(result["advantage"] < 0.0))
        np.testing.assert_allclose(result["design"][..., 0], result["advantage"], rtol=0.0, atol=0.0)

    def test_matched_common_support_is_exact_intersection(self) -> None:
        active = np.ones((3, 4), dtype=bool)
        hard = np.zeros((3, 4), dtype=np.int64)
        candidate = np.ones((3, 4), dtype=np.int64)
        true = np.asarray([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=bool)
        scores = {
            "proposer": np.asarray([[4, 3, 2, 1], [4, 3, 2, 1], [4, 3, 2, 1]], dtype=float),
            "harm": np.zeros((3, 4)),
            "incompatibility": np.zeros((3, 4)),
        }
        threshold = {"proposer_threshold": 2.5, "harm_threshold": 1.0, "incompatibility_threshold": 1.0}
        matched = native.matched_control_actions(
            true_override=true, scores=scores, thresholds=threshold,
            disagreement=active, hard_actions=hard, candidate_actions=candidate,
        )
        np.testing.assert_array_equal(matched["match_valid"], np.asarray([True, True, True]))
        np.testing.assert_array_equal(matched["selected"].sum(axis=1), true.sum(axis=1))

    def test_assignment_is_optimal_and_row_order_stable(self) -> None:
        tokens = np.asarray(["a", "b", "c", "d"])
        active = np.ones((4, 1), dtype=bool)
        gain = np.asarray([[0.0], [1.0], [0.0], [1.0]])
        harm = np.asarray([[False], [False], [True], [True]])
        incompat = np.asarray([[False], [True], [True], [False]])
        result = native.maximum_hamming_control(
            gain=gain, harm=harm, incompatibility=incompat, active=active,
            pair_tokens=tokens, seed=53611,
        )
        signatures = np.column_stack(
            [gain[:, 0].view(np.uint64), harm[:, 0].astype(np.uint64), incompat[:, 0].astype(np.uint64)]
        )
        cost = np.sum(signatures[:, None, :] != signatures[None, :, :], axis=2)
        np.fill_diagonal(cost, -1_000_000)
        row, column = linear_sum_assignment(cost, maximize=True)
        observed = result["mapping"][:, 0]
        self.assertEqual(int(cost[np.arange(4), observed].sum()), int(cost[row, column].sum()))
        order = np.asarray([2, 0, 3, 1])
        reordered = native.maximum_hamming_control(
            gain=gain[order], harm=harm[order], incompatibility=incompat[order],
            active=active[order], pair_tokens=tokens[order], seed=53611,
        )
        semantic = {tokens[i]: tokens[observed[i]] for i in range(4)}
        semantic_reordered = {
            tokens[order][i]: tokens[order][reordered["mapping"][i, 0]] for i in range(4)
        }
        self.assertEqual(semantic, semantic_reordered)


if __name__ == "__main__":
    unittest.main()
