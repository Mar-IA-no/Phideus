"""Abstract-only fixtures for the final geometric-decision audit checker."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.atencion_armonica import audit_geometric_decision_core as core
from experiments.atencion_armonica import audit_geometric_decision as audit


def test_independent_entropy_ari_and_canonical_arithmetic_are_separate():
    partitions = [((0, 1), (2, 3)), ((0, 2), (1, 3))]
    value = core.independent_targets(partitions, np.array([4, 4, 9, 9], np.int64))
    np.testing.assert_allclose(value["u64"], [[0, 0], [.5, .5]], atol=1e-12, rtol=0)
    np.testing.assert_allclose(value["ari"], [1, -.5], atol=1e-12, rtol=0)
    canonical = {"u64": value["u64"].copy(), "u32": value["u64"].astype(np.float32)}
    canonical["tM"] = canonical["u64"].sum(1, dtype=np.float64)
    canonical["tD"] = canonical["u32"].astype(np.float64).sum(1, dtype=np.float64)
    core.canonical_target_arithmetic(canonical)
    for name in ("tM", "tD"):
        wrong_dtype = {key: value.copy() for key, value in canonical.items()}
        wrong_dtype[name] = wrong_dtype[name].astype(np.float32)
        with pytest.raises(ValueError, match="canonical"):
            core.canonical_target_arithmetic(wrong_dtype)
    canonical["u32"][1, 0] = np.nextafter(canonical["u32"][1, 0], np.float32(1))
    with pytest.raises(ValueError, match="canonical"):
        core.canonical_target_arithmetic(canonical)


@pytest.mark.parametrize("bad", [0.5, True, np.float64(1)])
def test_candidate_and_event_identities_are_never_coerced(bad):
    with pytest.raises(ValueError):
        core.canonical_partition(((0, bad), (2, 3)), 4)
    with pytest.raises(ValueError):
        core.order_record(np.array([1., 2.]), [0, bad])
    with pytest.raises(ValueError):
        core.labels_by_event({"canonical_to_observed": [0, bad]},
                             {"canonical_to_observed": [0, 1]}, np.array([1, 2]))


def test_signature_tie_break_accepts_transport_order_and_tau_counts_pairs():
    roster = [((0, 2), (1, 3)), ((0, 1), (2, 3))]
    assert core.choose_energy(np.array([1., 1.], np.float64), roster) == 1
    record = core.order_record(np.array([2., 1., 1.]))
    assert record["order"] == [1, 2, 0] and record["tie_blocks"] == [[1, 2], [0]]
    tau = core.kendall_tau_b(np.array([0., 0., 1.]), np.array([0., 1., 1.]))
    assert tau == {"status": "DEFINED", "value": .5, "concordant": 1, "discordant": 0,
                   "score_only_ties": 1, "target_only_ties": 1, "double_ties": 0, "pair_count": 3}


def test_strata_include_singletons_and_reject_overlap_or_missing_support():
    ps = [((0,), (1, 2)), ((0, 1), (2,)), ((0, 2), (1,))]
    target = core.independent_targets(ps, np.array([0, 0, 1]))
    definitions = {"singleton": [0], "rest": [1, 2]}
    actual = core.stratum_metrics(ps, target, np.array([.3, .2, .1]), definitions,
                                  np.zeros((3, 2), np.float64))
    assert actual["singleton"]["chosen"] == 0
    assert actual["singleton"]["tau_status"] == "INSUFFICIENT_PAIRS"
    for bad in ({"a": [0], "b": [0, 1, 2]}, {"a": [0, 1]}):
        with pytest.raises(ValueError, match="partition"):
            core.stratum_metrics(ps, target, np.array([.3, .2, .1]), bad)


def test_observable_strata_and_all_sham_channels_membership_donors_masks_support():
    partitions = [((0,), (1, 2)), ((0, 1), (2,)), ((0,), (1,), (2,))]
    available = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1]], np.bool_)
    strata = core.observable_strata(partitions, available, ["A", "B", "A"])
    assert strata["full"] == {"all": [0, 1, 2]}
    assert strata["k"] == {"k=2": [0, 1], "k=3": [2]}
    assert sorted(i for rows in strata["sizes_available_branch"].values() for i in rows) == [0, 1, 2]
    evidence = np.arange(18, dtype=np.float32).reshape(3, 6)
    sham = {"donors": [1, 0, 2], "changed_mask": [True, True, False], "status": "INPUT_CHANGED",
        "strata": [
            {"sizes": [1, 1, 1], "candidate_ids": [2], "donors": [2], "shift": None,
             "changed_fraction": 0.0, "status": "NO_PERMUTATION"},
            {"sizes": [1, 2], "candidate_ids": [0, 1], "donors": [1, 0], "shift": 1,
             "changed_fraction": 1.0, "status": "INPUT_CHANGED"},
        ]}
    core.validate_sham(partitions, evidence, sham)
    mutations = (("donors", [0, 1, 2]), ("changed_mask", [False, True, False]),
                 ("status", "INPUT_UNCHANGED"))
    for field, value in mutations:
        bad = copy.deepcopy(sham); bad[field] = value
        with pytest.raises(ValueError): core.validate_sham(partitions, evidence, bad)
    for field, value in (("sizes", [1, 4]), ("candidate_ids", [1, 0]),
                         ("donors", [0, 1]), ("shift", True),
                         ("changed_fraction", 0.5), ("status", "INPUT_UNCHANGED")):
        bad = copy.deepcopy(sham); bad["strata"][1][field] = value
        with pytest.raises(ValueError): core.validate_sham(partitions, evidence, bad)


def transport_fixture():
    partitions = [((0,), (1, 2)), ((0, 1), (2,))]
    routed = {"groups": np.arange(18, dtype=np.float32).reshape(2, 9),
              "globals": np.arange(34, dtype=np.float32).reshape(2, 17),
              "incidence": np.array([[1/3, 2/3], [2/3, 1/3]], np.float32),
              "evidence": np.arange(16, dtype=np.float32).reshape(2, 8)}
    baseline = np.array([[.2, .3], [.7, .1]], np.float64)
    moved = baseline[::-1].copy()
    c, g, h = np.array([1, 0]), np.array([1, 0]), np.arange(7, -1, -1)
    arrays = {"candidate_order": c.astype(np.int64), "group_order": g.astype(np.int64),
        "channel_order": h.astype(np.int64),
        "weight_column_order": np.r_[np.arange(33), 33+h].astype(np.int64),
        "transported_inputs/groups": routed["groups"][g], "transported_inputs/globals": routed["globals"][c],
        "transported_inputs/incidence": routed["incidence"][np.ix_(c, g)],
        "transported_inputs/evidence": routed["evidence"][np.ix_(c, h)],
        "baseline/components": baseline, "baseline/energy": baseline.sum(1, dtype=np.float64),
        "baseline/offsets": np.array([0, 2], np.int64), "transported_components": moved,
        "restored_components": moved[::-1], "transported_energy": moved.sum(1, dtype=np.float64),
        "restored_energy": moved[::-1].sum(1, dtype=np.float64), "batched_components": baseline.copy()}
    return partitions, routed, arrays


@pytest.mark.parametrize("field", ["candidate_order", "channel_order", "transported_inputs/evidence",
                                    "restored_components", "transported_energy", "batched_components"])
def test_transport_recomputes_all_coordinate_numeric_and_singleton_families(field):
    partitions, routed, arrays = transport_fixture()
    expected = core.transport_diagnostic(arrays, partitions, routed, 1, arrays["batched_components"])
    assert expected["same_exact_choice"] and expected["singleton_vs_batch_same_exact_choice"]
    bad = {key: value.copy() for key, value in arrays.items()}
    bad[field].flat[0] += 1
    with pytest.raises(ValueError):
        core.transport_diagnostic(bad, partitions, routed, 1, arrays["batched_components"])


def test_transport_rejects_parent_prediction_slice_substitution():
    partitions, routed, arrays = transport_fixture()
    parent = arrays["batched_components"].copy(); parent[0, 0] += 1
    with pytest.raises(ValueError, match="parent prediction"):
        core.transport_diagnostic(arrays, partitions, routed, 1, parent)


def test_roundtrip_and_coordinate_mutations_are_detected():
    before = {"partitions": [((0, 1), (2, 3))], "canonical_to_observed": [0, 1, 2, 3]}
    after = {"partitions": [((0, 2), (1, 3))], "canonical_to_observed": [0, 2, 1, 3]}
    report = core.roundtrip_comparison(before, after, energy_before=np.array([1.]), energy_after=np.array([1.]),
        evidence_before=np.zeros((1, 8), np.float32), evidence_after=np.ones((1, 8), np.float32),
        choice_before=0, choice_after=0)
    assert report["common_candidates"] == 1 and report["same_exact_choice_by_event"] is True
    coordinates = {"original": np.array([0, 1], np.float32), "shifted64": np.array([1, 2], np.float64),
        "shifted32": np.array([1, 2], np.float32), "q_center": np.array([-.5, .5], np.float32),
        "q_probe": np.array([-.5, .5], np.float32)}
    assert core.coordinate_comparison(coordinates)["event_count"] == 2
    coordinates["q_probe"] = coordinates["q_probe"].astype(np.float64)
    with pytest.raises(ValueError, match="dtype"):
        core.coordinate_comparison(coordinates)


def test_primary_exact_indices_distribution_interpolation_and_empty_support():
    values = np.full((3, 8, 3, 3), np.nan, np.float64)
    values[:2] = np.arange(2 * 8 * 9, dtype=np.float64).reshape(2, 8, 3, 3) / 1000
    result = core.primary(values, np.array([True, True, False]))
    expected_indices = np.random.Generator(np.random.PCG64(core.BOOTSTRAP_SEED)).integers(
        0, 2, size=(core.BOOTSTRAP_COUNT, 2), dtype=np.int64)
    np.testing.assert_array_equal(result["arrays"]["bootstrap_indices"], expected_indices)
    direct = result["arrays"]["scene_contrasts"][expected_indices].mean(1, dtype=np.float64)
    np.testing.assert_allclose(result["arrays"]["bootstrap_distribution"], direct, atol=1e-12, rtol=0)
    empty = core.primary(np.full((2, 8, 3, 3), np.nan), np.zeros(2, bool))
    assert empty["arrays"]["arm_scene_means"].shape == (0, 8)
    assert empty["summary"]["eligible_scenes"] == 0
    bad = values.copy(); bad[0, 0, 0, 0] = -1
    with pytest.raises(ValueError): core.primary(bad, np.array([True, True, False]))


def test_complete_descriptive_tables_are_rebuilt_with_presence_and_missing_support():
    learned = np.zeros((3, 2, 8, 3, 3, 9), np.float64)
    classical = np.zeros((3, 4, 9), np.float64)
    learned[1] = np.nan; classical[1] = np.nan
    summary = core.aggregate_summary(learned, classical, np.array([True, False, True]),
                                     ["pool", "neighbor", "absent"])
    assert summary["eligible_scenes"] == 2
    assert summary["cell_descriptive"]["defined_scenes"][0][0][0][0][0] == 2
    assert summary["presence"]["neighbor"]["eligible_scenes"] == 0
    assert "presence" not in summary["presence"]["pool"]


def test_authenticated_reader_returns_same_hashed_bytes_and_rejects_symlink(tmp_path):
    raw = core.encoded({"schema": "fixture", "value": 1})
    (tmp_path / "value.json").write_bytes(raw)
    ref = {"path": "value.json", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    coverage = core.Coverage(); store = core.AuditStore("fixture", tmp_path, coverage)
    assert store.read(ref) == raw and store.json(ref)["value"] == 1
    assert coverage.summary()["files_by_store"] == {"fixture": 1}
    (tmp_path / "alias.json").symlink_to(tmp_path / "value.json")
    with pytest.raises(ValueError, match="symlink"):
        store.read({**ref, "path": "alias.json"})
    (tmp_path / "value.json").write_bytes(raw[:-1])
    with pytest.raises(ValueError, match="changed"):
        store.read(ref)


@pytest.mark.parametrize("probe_count", range(5))
def test_structural_precommit_cut_preserves_exact_zero_to_four_probe_roster(probe_count):
    rows = [{"scene_id": i, "partitions": [((0,), (1,))] if i < probe_count else []}
            for i in range(512)]
    probes = list(range(probe_count))
    cut = core.select_structural_cut(rows, probes)
    assert cut["probe_count"] == probe_count and cut["probe_scene_ids"] == probes
    assert {0, 511}.issubset(cut["scene_ids"])


def test_schema_dispatch_rejects_unknown_and_omitted_required_fields():
    with pytest.raises(ValueError, match="no reviewed dispatch"):
        audit.expect_schema({"schema": "invented"}, "invented", label="fixture")
    with pytest.raises(ValueError, match="omits"):
        audit.expect_schema({"schema": "geometric-decision-batch-evaluation-v1"},
                            "geometric-decision-batch-evaluation-v1", label="fixture")


def test_core_has_no_scientific_or_model_import_or_filesystem_side_effect():
    source = Path(core.__file__).read_text()
    assert "geometric_decision_metrics" not in source
    assert "operator_objective_core" not in source
    assert "torch" not in source
    assert "reconstruct_truth" not in source
    assert "cuda" not in source.lower()
