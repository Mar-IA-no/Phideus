"""Exact source delta and JSON boundaries; not an end-to-end campaign test.

Only structural partitions and the already excluded arithmetic q8 fixture.
The integrated 512-scene/45-output test is a separate required verification.
"""
import ast
from copy import deepcopy
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from src.atencion_armonica import generative_evidence_fresh_inference as original
from src.atencion_armonica import generative_evidence_fresh_inference_json as fixed
from src.atencion_armonica import generative_evidence_fresh_store as stores
from src.atencion_armonica.partial_compatibility_cache import encoded

ROOT = Path(__file__).resolve().parents[2]
TEMP = ROOT/'.agent-work/phideus-json-contract-tests-20260909'
FREEZE = {'path': 'fixture/no-campaign-authority.json', 'sha256': 'f'*64}


def tree(name):
    return ast.parse((ROOT/'src/atencion_armonica'/name).read_text())


def without_doc(value):
    assert isinstance(value.body[0], ast.Expr)
    assert isinstance(value.body[0].value, ast.Constant)
    value.body = value.body[1:]
    return value


def test_inference_diff_is_exactly_six_canonical_comparisons_and_one_import():
    path = Path(original.__file__)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '4b9007c350a9016eb0bb46aa8d023cbe6a839ecdaa3d11d7ae24fd6eea10d7da'
    old = tree(path.name)
    wanted = {
        (140, 'store.json(ref)'), (145, 'store.json(ref)'),
        (487, "fit['inventory']"), (612, 'VerifiedBytes(ROOT).json(source_ref)'),
        (616, "fit['inventory']"), (632, 'choice_record'),
    }
    seen = set()
    for node in ast.walk(old):
        if isinstance(node, ast.ImportFrom) and node.module == 'partial_compatibility_cache':
            assert [a.name for a in node.names] == ['feature_record']
            node.names.insert(0, ast.alias(name='encoded'))
        if not isinstance(node, ast.Compare):
            continue
        key = node.lineno, ast.unparse(node.left)
        if key in wanted:
            assert len(node.ops) == len(node.comparators) == 1
            assert isinstance(node.ops[0], ast.NotEq)
            assert key not in seen
            seen.add(key)
            node.left = ast.Call(func=ast.Name(id='encoded', ctx=ast.Load()), args=[node.left], keywords=[])
            node.comparators = [ast.Call(func=ast.Name(id='encoded', ctx=ast.Load()),
                                        args=node.comparators, keywords=[])]
    assert seen == wanted
    assert ast.dump(without_doc(old)) == ast.dump(without_doc(tree(Path(fixed.__file__).name)))


def test_evaluation_diff_is_only_the_successor_import():
    name = 'generative_evidence_fresh_evaluation.py'
    assert hashlib.sha256((ROOT/'src/atencion_armonica'/name).read_bytes()).hexdigest() == '4706c2f1b5bda852c3aebe8368c4f78376637659f8ea6b2d213060a4a3a37978'
    old = tree(name)
    seen = 0
    for node in ast.walk(old):
        if isinstance(node, ast.ImportFrom) and node.module == 'generative_evidence_fresh_inference':
            assert [a.name for a in node.names] == ['verify_predictions', 'read_prediction']
            node.module = 'generative_evidence_fresh_inference_json'
            seen += 1
    assert seen == 1
    assert ast.dump(without_doc(old)) == ast.dump(without_doc(tree('generative_evidence_fresh_evaluation_json.py')))


@pytest.fixture
def store(request, monkeypatch):
    # Only fixture path admission changes. The class and every serializer stay real.
    monkeypatch.setattr(stores, 'TEMPORARY', TEMP)
    return stores.FreshObservableStore(TEMP/f'run-{os.getpid()}'/request.node.name,
                                       binding={'test_freeze': FREEZE})


def test_actual_old_failure_leaves_exact_payload_that_successor_reuses(store):
    value = {'partitions': [((0, 1, 2, 3), (4, 5, 6, 7))], 'status': 'STRUCTURAL_FIXTURE'}
    path = store.path('iid/source.json')
    with pytest.raises(ValueError, match='stored JSON changed after publication'):
        original._fixed_json(store, 'iid/source.json', value)
    before = path.read_bytes()
    assert before == encoded(value)
    ref = fixed._fixed_json(store, 'iid/source.json', value)
    assert path.read_bytes() == before
    assert store.json(ref) != value
    assert encoded(store.json(ref)) == encoded(value)
    assert fixed._fixed_json(store, 'iid/source.json', value) == ref


@pytest.mark.parametrize('changed', [
    {'partition': ((0, 1), (3, 2)), 'number': 1},
    {'partition': ((2, 3), (0, 1)), 'number': 1},
    {'partition': ((0, 1), (2, 3)), 'number': 1.0},
    {'partition': ((0, 1), (2, 3)), 'number': True},
    {'partition': ((0, 1), (2, 3)), 'number': 2},
])
def test_json_canonical_equality_rejects_content_order_and_numeric_type_changes(store, changed):
    value = {'partition': ((0, 1), (2, 3)), 'number': 1}
    ref = fixed._fixed_json(store, 'iid/value.json', value)
    with pytest.raises(ValueError, match='cannot replace'):
        fixed._fixed_json(store, 'iid/value.json', changed)
    assert store.reference(store.path('iid/value.json')) == ref


def test_json_nan_is_rejected_before_publication(store):
    with pytest.raises(ValueError):
        fixed._fixed_json(store, 'iid/invalid.json', {'value': float('nan')})
    assert not store.path('iid/invalid.json').exists()


def test_source_value_and_real_observable_choices_keep_their_original_bytes(store):
    from experiments.atencion_armonica.test_generative_evidence import fixture
    q, z, _, _, ps, fits = fixture()
    scene = {'pools': {'fixture': {'partitions': ps}},
             'inventory': {'candidates': [{'partition': p} for p in ps]},
             'partitions': ps, 'canonical_to_observed': np.arange(8),
             'q32': q, 'status': 'ELIGIBLE'}
    features = {'path': 'fixture/features.json', 'sha256': 'a'*64}
    logits = {'fixture': {'path': 'fixture/logits.json', 'sha256': 'b'*64}}
    expected = original._source_record_value(store, 'iid', 0, scene, features, logits)
    assert encoded(fixed._source_record_value(store, 'iid', 0, scene, features, logits)) == encoded(expected)
    source = fixed._source_record(store, 'iid', 0, scene, features, logits)
    assert (ROOT/source['path']).read_bytes() == encoded(expected)
    assert fixed._source_record(store, 'iid', 0, scene, features, logits) == source
    choice = fixed.references.observable_choices(ps, fits, {cp: z for cp in fixed.ge.CHECKPOINTS}, np.arange(8))
    records = [{'scene_id': 0, 'choice': choice}]
    ref, value = fixed._choices(store, 'iid', records)
    assert (ROOT/ref['path']).read_bytes() == encoded(value)
    assert fixed._choices(store, 'iid', deepcopy(records))[0] == ref


def test_empty_candidate_choices_still_roundtrip_historical_partitions(store):
    z = np.zeros((8, 8), np.float32)
    choice = fixed.references.observable_choices([], [], {cp: z for cp in fixed.ge.CHECKPOINTS}, np.arange(8))
    assert choice['base'] is choice['extended'] is None
    ref, value = fixed._choices(store, 'iid', [{'scene_id': 0, 'choice': choice}])
    assert (ROOT/ref['path']).read_bytes() == encoded(value)
