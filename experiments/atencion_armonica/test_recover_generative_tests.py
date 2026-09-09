"""CPU integration fixtures; no real draws, models, fits, or privileged parse."""
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

from experiments.atencion_armonica import recover_generative_tests as r
from src.atencion_armonica import generative_evidence_fresh_data as data
from src.atencion_armonica import generative_evidence_fresh_store as stores
from src.atencion_armonica.generative_evidence_storage import write_json
from src.atencion_armonica.partial_compatibility_cache import encoded


def ref(name):
    return {'path': name, 'sha256': hashlib.sha256(name.encode()).hexdigest()}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    # pytest --basetemp must remain in the project's dedicated test area.
    assert tmp_path.is_relative_to(r.ROOT/'.agent-work')
    fresh, control = tmp_path/'fresh', tmp_path/'control'
    fresh.mkdir()
    control.mkdir()
    monkeypatch.setattr(r.old, 'FRESH', fresh)
    monkeypatch.setattr(r.old, 'FINAL', fresh/'test_completion.json')
    monkeypatch.setattr(r, 'CONTROL', control)
    monkeypatch.setattr(r, 'MANIFEST', control/'manifest.json')
    monkeypatch.setattr(r, 'RECEIPT', control/'namespace-reconciliation.json')
    monkeypatch.setattr(r, 'BINDING', fresh/'binding.json')
    monkeypatch.setattr(r, 'FINAL', fresh/'test_completion.json')
    monkeypatch.setattr(r.old, 'COMMON_LOCK', control/'common.lock')
    monkeypatch.setattr(stores, 'CANONICAL', fresh)
    monkeypatch.setattr(data, 'DRAW_ROOT', fresh/'draws')
    freeze_path = control/'freeze.json'
    write_json(freeze_path, {'fixture': 'freeze'})
    freeze = r.reference(freeze_path)
    monkeypatch.setattr(data, '_authority', lambda fr, check: {'freeze': freeze, 'exclusions': {}})
    monkeypatch.setattr(data.exclusions, 'duplicate_matches',
                        lambda *args: {'duplicate': False, 'fingerprint': 'fixture-alias'})
    parsed = []
    real_json = data.VerifiedBytes.json
    def observable_json(reader, record):
        assert Path(record['path']).name != 'sidecar.json', 'privileged parse attempted'
        parsed.append(record['path'])
        return real_json(reader, record)
    monkeypatch.setattr(data.VerifiedBytes, 'json', observable_json)
    calls = []
    def draw(split, sid, seed):
        calls.append(sid)
        # arange8 is a previously excluded alias. No new geometric fixture.
        return ({'scene_id': sid, 'split_seed': seed, 'log_f': list(range(8))},
                {'not_a_real_truth': True})
    files = data._DrawFiles(data.DRAW_ROOT, freeze)
    def fixture_json(path, value):
        # Populate test bytes without 2049 disk fsyncs per fixture. The recovery
        # binding/receipt writers remain the real atomic writers below.
        with Path(path).open('xb') as handle:
            handle.write(encoded(value))
    with monkeypatch.context() as population:
        population.setattr(data.storage, 'write_json', fixture_json)
        data._produce_verified(files, 'iid', {}, check=lambda: None, draw=draw)
    assert calls == list(range(512))
    state = {'seconds': r.ORIGINAL_SECONDS, 'next_stage': 1, 'last_status': 'FAILED',
             'previous_exit': ref('original-exit'), 'completions': [{
                 'stage': r.old.stage_roster()[0], 'output': freeze,
                 'launch': ref('freeze-launch'), 'exit': ref('freeze-exit'),
                 'worker': ref('freeze-worker')}]}
    original = {'manifest': ref('original-manifest'), 'last_exit': ref('original-exit'),
        'state': state, 'freeze': freeze, 'iid': r.reference(fresh/'draws/iid/index.json'),
        'value': {'stages': r.old.stage_roster(), 'limits': {'total_seconds': r.old.LIMIT_SECONDS},
                  'output': {'path': r.FINAL.relative_to(r.ROOT).as_posix(),
                             'status': r.old.FINAL_STATUS}}}
    monkeypatch.setattr(r, 'verify_original', lambda: original)
    manifest_value = r._manifest_value(original)
    write_json(r.MANIFEST, manifest_value)
    manifest = r.reference(r.MANIFEST)
    return SimpleNamespace(original=original, manifest=manifest, fresh=fresh,
                           control=control, files=files, calls=calls, parsed=parsed)


def test_producer_then_real_store_reproduces_bug_and_recovery_never_redraws(setup):
    s = setup
    with pytest.raises(ValueError, match='cannot adopt an unbound'):
        stores.FreshObservableStore(s.fresh, binding={'test_freeze': s.original['freeze']})
    assert r._namespace_state(s.manifest)[0] == 'A'
    before = {p.relative_to(s.fresh): p.read_bytes() for p in s.fresh.rglob('*') if p.is_file()}
    receipt = r.reconcile_namespace(s.manifest, lambda: None)
    assert r._namespace_state(s.manifest) == ('C', receipt)
    assert r.reconcile_namespace(s.manifest, lambda: None) == receipt
    def never_draw(*args):
        pytest.fail('recovery redrew an existing scene')
    index = data._produce_verified(s.files, 'iid', {}, check=lambda: None, draw=never_draw)
    assert index == {'path': 'iid/index.json', 'sha256': s.original['iid']['sha256']}
    assert all((s.fresh/p).read_bytes() == raw for p, raw in before.items())
    assert len(s.calls) == 512


def test_pause_after_binding_resumes_B_then_C_allows_partial_scientific_files(setup):
    s = setup
    def pause_in_B():
        if r.BINDING.exists() and not r.RECEIPT.exists():
            raise InterruptedError('fixture B pause')
    with pytest.raises(InterruptedError):
        r.reconcile_namespace(s.manifest, pause_in_B)
    assert r._namespace_state(s.manifest) == ('B', None)
    def pause_in_C():
        if r.RECEIPT.exists():
            raise InterruptedError('fixture C pause')
    with pytest.raises(InterruptedError):
        r.reconcile_namespace(s.manifest, pause_in_C)
    receipt = r.reference(r.RECEIPT)
    assert r.reconcile_namespace(s.manifest, lambda: None) == receipt
    # Preserve one real observable feature archive using the frozen codec.
    # No backbone, fit, prediction or truth is needed for this partial prefix.
    from src.atencion_armonica import generative_evidence_fresh_inference as frozen_inference
    from src.atencion_armonica.partial_compatibility_cache import feature_record
    observation = {'scene_id': 0, 'split_seed': 2026090982, 'log_f': list(range(8))}
    features = feature_record(observation)
    store = stores.FreshObservableStore(s.fresh, binding={'test_freeze': s.original['freeze']})
    filename = 'iid/source/features/00000.npz'
    feature_ref = frozen_inference._fixed_arrays(store, filename, features)
    assert r.reconcile_namespace(s.manifest, lambda: None) == receipt
    assert frozen_inference._fixed_arrays(store, filename, features) == feature_ref
    with pytest.raises(ValueError, match='bootstrap tree'):
        r._strict_bootstrap_tree(True)


@pytest.mark.parametrize('extra', ['prediction_seal.json', 'foreign.bin'])
def test_A_rejects_extra_files_before_binding(setup, extra):
    write_json(setup.fresh/extra, {'fixture': True})
    with pytest.raises(ValueError, match='bootstrap tree'):
        r.reconcile_namespace(setup.manifest, lambda: None)
    assert not r.BINDING.exists() and not r.RECEIPT.exists()


def test_B_rejects_incompatible_binding(setup):
    s = setup
    write_json(r.BINDING, {'test_freeze': ref('other')})
    with pytest.raises(ValueError, match='binding differs'):
        r.reconcile_namespace(s.manifest, lambda: None)


def test_B_with_correct_binding_rejects_prior_scientific_file(setup):
    write_json(r.BINDING, {'test_freeze': setup.original['freeze']})
    write_json(setup.fresh/'predictions.json', {'fixture': 'unreceipted output'})
    assert r._namespace_state(setup.manifest) == ('B', None)
    with pytest.raises(ValueError, match='bootstrap tree'):
        r.reconcile_namespace(setup.manifest, lambda: None)
    assert not r.RECEIPT.exists()


@pytest.mark.parametrize('member', ['binding', 'receipt', 'scene'])
def test_symlinks_are_rejected_including_dangling_bindings(setup, member):
    if member == 'binding':
        r.BINDING.symlink_to(setup.control/'missing-target')
    elif member == 'receipt':
        r.RECEIPT.symlink_to(setup.control/'missing-target')
    else:
        (setup.fresh/'draws/iid/00000/foreign-link').symlink_to(setup.control/'freeze.json')
    with pytest.raises(ValueError):
        r.reconcile_namespace(setup.manifest, lambda: None)


def test_forged_receipt_cannot_open_C(setup):
    write_json(r.BINDING, {'test_freeze': setup.original['freeze']})
    write_json(r.RECEIPT, {'status': 'pretend-complete'})
    with pytest.raises(ValueError, match='receipt differs'):
        r.reconcile_namespace(setup.manifest, lambda: None)


def test_wrong_root_relative_draw_reference_is_rejected(setup, monkeypatch):
    monkeypatch.setattr(data, 'FreshObservations', lambda *args, **kwargs:
        SimpleNamespace(files=setup.files, reference=setup.original['iid']))
    with pytest.raises(ValueError, match='observations differ'):
        r.reconcile_namespace(setup.manifest, lambda: None)


def test_metadata_C_does_not_construct_store_or_authenticate_observations(setup, monkeypatch):
    receipt = r.reconcile_namespace(setup.manifest, lambda: None)
    def forbidden(*args, **kwargs):
        pytest.fail('metadata parent entered scientific/constructing port')
    monkeypatch.setattr(stores, 'FreshObservableStore', forbidden)
    monkeypatch.setattr(data, 'FreshObservations', forbidden)
    assert r._namespace_state(setup.manifest) == ('C', receipt)


def test_initialize_does_not_enter_observable_or_constructing_ports(setup, monkeypatch):
    s = setup
    r.MANIFEST.rename(s.control/'previous-fixture-manifest.json')
    def forbidden(*args, **kwargs):
        pytest.fail('initialize entered a scientific or constructing port')
    monkeypatch.setattr(stores, 'FreshObservableStore', forbidden)
    monkeypatch.setattr(data, 'FreshObservations', forbidden)
    monkeypatch.setattr(r, '_authenticate_iid', forbidden)
    monkeypatch.setattr(r.old, '_training', lambda: SimpleNamespace(_lock=lambda path: nullcontext()))
    result = r.initialize_recovery()
    assert r.read_manifest(result)['original_seconds'] == r.ORIGINAL_SECONDS
    assert not r.BINDING.exists() and not r.RECEIPT.exists()


@pytest.mark.parametrize('name', ['PLAN', 'AUDIT'])
def test_sources_reject_drift_from_approved_plan_or_audit(tmp_path, monkeypatch, name):
    assert tmp_path.is_relative_to(r.ROOT/'.agent-work')
    changed = tmp_path/'changed.json'
    write_json(changed, {'fixture': 'not the approved document'})
    monkeypatch.setattr(r, name, changed)
    with pytest.raises(ValueError, match='audited version'):
        r.sources()


def launch(s, stage_index, used, previous, *, status, seconds, bootstrap='C', output=None):
    stage = r.old.stage_roster()[stage_index]
    attempt = f'attempt-{len(r._launches()):04d}'
    unit = 'phideus-generative-test-'+f'{len(r._launches()):016x}'
    remaining = math.floor(r.old.LIMIT_SECONDS-used)
    availability = {'fixture_gpu': True} if stage['device'] == 'cuda:0' else None
    path = s.control/f'{attempt}.launch.json'
    write_json(path, {'schema': 'generative-evidence-fresh-test-recovery-launch-v1',
        'manifest': s.manifest, 'stage': stage, 'previous_exit': previous,
        'unit': unit, 'command': r._service_command(unit, remaining-r.old.GRACE, stage, attempt),
        'availability': availability, 'used_seconds': used, 'remaining_seconds': remaining,
        'runtime_seconds': remaining-r.old.GRACE, 'bootstrap_state': bootstrap,
        'reconciliation': r.reference(r.RECEIPT) if bootstrap == 'C' else None})
    launch_ref = r.reference(path)
    worker = {'launch': launch_ref, 'stage': stage, 'status': status, 'seconds': seconds,
              'peak_rss_bytes': 100, 'peak_reserved_bytes': 0, 'cuda_initialized': False,
              'availability': availability}
    code = 0 if status == r.old._success_status(stage) else 75 if status == 'PAUSED_RECOVERABLE' else 1
    worker.update({'output': output} if code == 0 else {'reason': 'fixture'})
    write_json(s.control/f'{attempt}.worker.json', worker)
    end = s.control/f'{attempt}.exit.json'
    write_json(end, {'launch': launch_ref, 'process_returncode': code,
                     'seconds': seconds, 'terminal': True})
    return r.reference(end)


def test_two_chains_all_thirteen_completions_and_pause_time_are_verified(setup, monkeypatch):
    s = setup
    r.reconcile_namespace(s.manifest, lambda: None)
    monkeypatch.setattr(r.old, '_availability_ok', lambda value: value == {'fixture_gpu': True})
    output = {i: ref('output-'+str(i if i % 3 != 0 else i-1)) for i in range(1, 13)}
    monkeypatch.setattr(r.old, '_stage_reference', lambda stage: output[stage['index']])
    used, previous = r.ORIGINAL_SECONDS, s.original['last_exit']
    previous = launch(s, 1, used, previous, status='PAUSED_RECOVERABLE', seconds=3., bootstrap='A')
    used += 3.
    for i in range(1, 13):
        previous = launch(s, i, used, previous, status=r.old._success_status(r.old.stage_roster()[i]),
                          seconds=2., output=output[i])
        used += 2.
    state = r.recovery_accumulated(s.manifest)
    assert state['seconds'] == r.ORIGINAL_SECONDS+27.
    assert len(state['completions']) == state['next_stage'] == 13
    assert state['completions'][0] == s.original['state']['completions'][0]
    r.old._publish_final(s.manifest, state)
    assert r.verified_tests() == r.reference(r.FINAL)


def test_new_failed_attempt_never_advances(setup, monkeypatch):
    s = setup
    r.reconcile_namespace(s.manifest, lambda: None)
    monkeypatch.setattr(r.old, '_availability_ok', lambda v: True)
    end = launch(s, 1, r.ORIGINAL_SECONDS, s.original['last_exit'], status='FAILED', seconds=2.)
    assert r.recovery_accumulated(s.manifest)['last_status'] == 'FAILED'
    launch(s, 1, r.ORIGINAL_SECONDS+2., end, status='PAUSED_RECOVERABLE', seconds=2.)
    with pytest.raises(ValueError, match='eligible stage'):
        r.recovery_accumulated(s.manifest)


@pytest.mark.parametrize('missing', ['worker', 'exit'])
def test_missing_worker_or_exit_cannot_be_skipped(setup, monkeypatch, missing):
    s = setup
    monkeypatch.setattr(r.old, '_availability_ok', lambda v: True)
    launch(s, 1, r.ORIGINAL_SECONDS, s.original['last_exit'],
           status='PAUSED_RECOVERABLE', seconds=2., bootstrap='A')
    path = s.control/f'attempt-0000.{missing}.json'
    path.rename(s.control/f'preserved-missing-{missing}.json')
    with pytest.raises(RuntimeError, match='unreconciled recovery'):
        r.recovery_accumulated(s.manifest)


def test_exhausted_budget_stops_before_any_command(setup, monkeypatch):
    monkeypatch.setattr(r, 'recovery_accumulated', lambda ref: {
        **setup.original['state'], 'last_status': 'PAUSED_RECOVERABLE',
        'seconds': r.old.LIMIT_SECONDS-10})
    training = SimpleNamespace(_lock=lambda path: nullcontext(), check_disk=lambda: pytest.fail('past budget'))
    monkeypatch.setattr(r.old, '_training', lambda: training)
    with pytest.raises(RuntimeError, match='time exhausted'):
        r.run_recovery()
    assert not r._launches()


def test_command_only_changes_worker_module_and_preserves_resource_guards():
    stage = r.old.stage_roster()[1]
    unit = 'phideus-generative-test-'+'a'*16
    original = r.old._service_command(unit, 123, stage, 'attempt-0000')
    modified = r._service_command(unit, 123, stage, 'attempt-0000')
    changes = [(a, b) for a, b in zip(original, modified) if a != b]
    assert changes == [('experiments.atencion_armonica.run_generative_tests',
                        'experiments.atencion_armonica.recover_generative_tests')]
    assert len(original) == len(modified)


@pytest.mark.parametrize('mutation', [None, 'manifest', 'receipt', 'freeze', 'index', 'failure', 'nonterminal'])
def test_original_authority_rejects_wrong_exact_parents(monkeypatch, mutation):
    """Exercise the real exact-failure validator with metadata-only fake bytes."""
    values = {r.ORIGINAL_MANIFEST: r.ORIGINAL_MANIFEST_SHA,
              r.old._freeze().FREEZE: r.FREEZE_SHA,
              r.old.FRESH/'draws/iid/index.json': r.IID_SHA,
              **{r.old.CONTROL/name: sha for name, sha in r.ORIGINAL_FILES.items()}}
    target = {'manifest': r.ORIGINAL_MANIFEST, 'receipt': r.old.CONTROL/'attempt-0001.worker.json',
              'freeze': r.old._freeze().FREEZE, 'index': r.old.FRESH/'draws/iid/index.json'}.get(mutation)
    if target:
        values[target] = '0'*64
    monkeypatch.setattr(r, 'reference', lambda p: {
        'path': Path(p).relative_to(r.ROOT).as_posix(), 'sha256': values[Path(p)]})
    monkeypatch.setattr(r.old, 'read_manifest', lambda ref: {'fixture': True})
    monkeypatch.setattr(r.old, '_launches', lambda: [
        r.old.CONTROL/'attempt-0000.launch.json', r.old.CONTROL/'attempt-0001.launch.json'])
    def accumulated(ref):
        if mutation == 'nonterminal':
            raise RuntimeError('unreconciled original prefix')
        return {'next_stage': 1, 'completions': [{}], 'last_status': 'FAILED',
                'seconds': r.ORIGINAL_SECONDS}
    monkeypatch.setattr(r.old, 'test_accumulated', accumulated)
    monkeypatch.setattr(r, '_read', lambda ref: {
        'status': 'FAILED', 'reason': 'another failure' if mutation == 'failure' else r.FAILURE})
    if mutation is None:
        assert r.verify_original()['state']['seconds'] == r.ORIGINAL_SECONDS
    else:
        with pytest.raises((ValueError, RuntimeError)):
            r.verify_original()


@pytest.mark.parametrize('pause', [False, True])
def test_cpu_worker_verifies_receipt_then_delegates_frozen_stage(setup, monkeypatch, pause):
    s = setup
    stage = r.old.stage_roster()[2]
    launch_ref = ref('fixture-launch')
    manifest = {'manifest': s.manifest, 'unit': 'fixture-unit', 'runtime_seconds': 100}
    monkeypatch.setattr(r, '_validate_tail', lambda attempt: (launch_ref, manifest, stage))
    monkeypatch.setattr(r, 'read_manifest', lambda ref: {})
    order = []
    monkeypatch.setattr(r, 'verify_reconciliation', lambda *args, **kwargs: order.append('receipt'))
    output = ref('fixture-evaluation')
    monkeypatch.setattr(r.old, '_stage_reference', lambda s: output)
    def execute(current, original, check, progress):
        assert current == stage and original == s.original['value']
        order.append('original-executor')
        if pause:
            signal.raise_signal(signal.SIGTERM)
            check()
        return output
    monkeypatch.setattr(r.old, '_execute_stage', execute)
    def check(started, seconds, stopped, **kwargs):
        if stopped:
            raise InterruptedError('fixture cooperative pause')
    monkeypatch.setattr(r.old, '_training', lambda: SimpleNamespace(
        _verify_service=lambda *args: None, _worker_check=check))
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False),
        set_num_threads=lambda n: None, set_num_interop_threads=lambda n: None,
        use_deterministic_algorithms=lambda value: None)
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        monkeypatch.setenv(name, '1')
    assert r.worker_stage('attempt-0000') == (75 if pause else 0)
    value = json.loads((s.control/'attempt-0000.worker.json').read_text())
    assert value['status'] == ('PAUSED_RECOVERABLE' if pause else 'EVALUATED')
    assert value['cuda_initialized'] is False
    assert order == ['receipt', 'original-executor']
