"""Metadata-only JSON amendment supervision tests, separate from science.

No CUDA, real draw/forward/fit or q/sidecar parse. Namespace/payload fixtures
are opaque; the separate integration test exercises real scientific codecs.
"""
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

from experiments.atencion_armonica import resume_generative_json_tests as r
from experiments.atencion_armonica import run_generative_training as training
from src.atencion_armonica.generative_evidence_storage import write_json
from src.atencion_armonica.partial_compatibility_cache import encoded

REAL_VERIFY_PREVIOUS = r.verify_previous
GPU = {'processes': '', 'inventory': 'NVIDIA GeForce RTX 3090, fixture-not-a-device\n'}


def ref(name):
    return {'path': name, 'sha256': hashlib.sha256(name.encode()).hexdigest()}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    assert tmp_path.is_relative_to(r.ROOT/'.agent-work')
    fresh, control = tmp_path/'fresh', tmp_path/'control'
    control.mkdir()
    monkeypatch.setattr(r.old, 'FRESH', fresh)
    monkeypatch.setattr(r.old, 'FINAL', fresh/'test_completion.json')
    monkeypatch.setattr(r.old, 'COMMON_LOCK', control/'common.lock')
    monkeypatch.setattr(r, 'CONTROL', control)
    monkeypatch.setattr(r, 'MANIFEST', control/'manifest.json')
    monkeypatch.setattr(r, 'FINAL', fresh/'test_completion.json')
    # Setup-only bytes: no expensive fsync for thousands of opaque fixtures.
    # Manifest/launch/worker/exit publication uses the real atomic writer.
    def opaque(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle:
            handle.write(value)
    opaque(fresh/'binding.json', encoded({'fixture': True}))
    opaque(fresh/'draws/iid/index.json', encoded({'fixture': True}))
    for i in range(512):
        for name in ('intent', 'observation', 'sidecar', 'draw'):
            opaque(fresh/f'draws/iid/{i:05d}/{name}.json', b'opaque-not-parsed\n')
    def payload(path):
        opaque(path, b'opaque-array-not-decoded\n')
        return {**r.reference(path), 'bytes': path.stat().st_size}
    features = [payload(fresh/f'iid/source/features/{i:05d}.npz') for i in range(512)]
    opaque(fresh/'iid/source/features.json', encoded({'scene_ids': list(range(512)), 'records': features}))
    opaque(fresh/'iid/source/scenes/00000.json', b'opaque-source-not-parsed\n')
    for cp in (2026090721, 2026090722, 2026090723):
        opaque(fresh/f'iid/source/logits-{cp}.json', encoded({
            'logits': payload(fresh/f'iid/source/logits-{cp}.npz')}))
    monkeypatch.setattr(r, 'SOURCE_PREFIX', {name: r.reference(fresh/name)['sha256'] for name in r.SOURCE_PREFIX})
    freeze_path = control/'freeze.json'
    write_json(freeze_path, {'fixture': 'freeze'})
    freeze = r.reference(freeze_path)
    state = {'seconds': r.INHERITED_SECONDS, 'next_stage': 1, 'last_status': 'FAILED',
        'previous_exit': ref('preceding-terminal-exit'), 'completions': [{
            'stage': r.old.stage_roster()[0], 'launch': ref('freeze-launch'),
            'exit': ref('freeze-exit'), 'worker': ref('freeze-worker'), 'output': freeze}]}
    original = {'manifest': ref('original-manifest'), 'freeze': freeze,
        'iid': r.reference(fresh/'draws/iid/index.json'),
        'value': {'limits': {'total_seconds': r.old.LIMIT_SECONDS},
            'output': {'path': r.FINAL.relative_to(r.ROOT).as_posix(), 'status': r.old.FINAL_STATUS}}}
    context = {'manifest': ref('preceding-manifest'), 'state': state,
               'original': original, 'freeze': freeze, 'reconciliation': ref('reconciliation')}
    monkeypatch.setattr(r, 'verify_previous', lambda: context)
    monkeypatch.setattr(r.old, '_training', lambda: SimpleNamespace(
        _lock=lambda path: nullcontext(), check_disk=lambda: None,
        _valid_gpu_availability=training._valid_gpu_availability))
    parsed = []
    read = r._read
    def metadata_only(value):
        assert Path(value['path']).name not in ('observation.json', 'sidecar.json', '00000.json')
        parsed.append(value['path'])
        return read(value)
    monkeypatch.setattr(r, '_read', metadata_only)
    return SimpleNamespace(fresh=fresh, control=control, context=context, parsed=parsed)


def test_initialize_metadata_only_and_later_growth_is_not_rejected(setup, monkeypatch):
    from src.atencion_armonica import generative_evidence_fresh_data as data
    from src.atencion_armonica import generative_evidence_fresh_inference_json as inference
    def forbidden(*a, **kw):
        pytest.fail('initialize called a scientific producer or consumer')
    monkeypatch.setattr(data, 'produce_test', forbidden)
    monkeypatch.setattr(data, 'FreshObservations', forbidden)
    monkeypatch.setattr(inference, 'prepare_and_predict', forbidden)
    manifest = r.initialize()
    value = r.read_manifest(manifest)
    assert value['inherited_seconds'] == r.INHERITED_SECONDS
    assert len(value['preserved_sources']) == 520
    assert value['stages'] == r.old.stage_roster()[1:]
    assert r.accumulated(manifest)['next_stage'] == 1
    folder = setup.fresh/'iid/00000'
    folder.mkdir()
    write_json(folder/'fixture-progress.json', {'fixture': True})
    assert r.read_manifest(manifest) == value
    with pytest.raises(FileExistsError):
        r.initialize()


@pytest.mark.parametrize('change', ['extra-file', 'extra-dir', 'symlink', 'missing', 'bytes'])
def test_initial_prefix_rejects_ambiguity(setup, change):
    path = setup.fresh/'iid/source/features/00000.npz'
    if change == 'extra-file':
        write_json(setup.fresh/'other.json', {})
    elif change == 'extra-dir':
        (setup.fresh/'other').mkdir()
    elif change == 'symlink':
        (setup.fresh/'link').symlink_to(path)
    elif change == 'missing':
        path.rename(setup.control/'held-payload')
    else:
        path.write_bytes(b'corrupted own fixture')
    with pytest.raises((ValueError, FileNotFoundError)):
        r.initialize()
    assert not r.MANIFEST.exists()


def test_manifest_reauthenticates_preserved_bytes_and_sources(setup, monkeypatch):
    manifest = r.initialize()
    path = setup.fresh/'iid/source/features/00000.npz'
    before = path.read_bytes()
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='preserved'):
        r.read_manifest(manifest)
    path.write_bytes(before)
    monkeypatch.setattr(r, 'CODECS', {**r.CODECS,
        'src/atencion_armonica/partial_compatibility_cache.py': '0'*64})
    with pytest.raises(ValueError, match='codec'):
        r.read_manifest(manifest)


def add_attempt(manifest, *, status=None, seconds=2., mutate=None, missing=None):
    state = r.accumulated(manifest)
    stage = r.old.stage_roster()[state['next_stage']]
    attempt = f'attempt-{len(r._launches()):04d}'
    unit = 'phideus-generative-test-'+f'{len(r._launches()):016x}'
    remaining = math.floor(r.old.LIMIT_SECONDS-state['seconds'])
    launch = {'schema': r.LAUNCH_SCHEMA, 'manifest': manifest, 'stage': stage,
        'previous_exit': state['previous_exit'], 'unit': unit,
        'command': r._service_command(unit, remaining-r.old.GRACE, stage, attempt),
        'availability': GPU if stage['device'] == 'cuda:0' else None,
        'used_seconds': state['seconds'], 'remaining_seconds': remaining,
        'runtime_seconds': remaining-r.old.GRACE}
    if mutate:
        mutate(launch)
    path = r.CONTROL/f'{attempt}.launch.json'
    write_json(path, launch)
    launch_ref = r.reference(path)
    status = status or r.old._success_status(stage)
    code = 75 if status == 'PAUSED_RECOVERABLE' else 1 if status == 'FAILED' else 0
    worker = {'launch': launch_ref, 'stage': stage, 'status': status, 'seconds': seconds,
        'peak_rss_bytes': 100, 'peak_reserved_bytes': 0, 'cuda_initialized': False,
        'availability': launch['availability']}
    if code == 0:
        output = r.old._stage_path(stage)
        if not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            write_json(output, {'fixture': 'typed output only', 'split': stage['split']})
        worker['output'] = r.old._stage_reference(stage)
    else:
        worker['reason'] = 'fixture stop'
    if missing != 'worker':
        write_json(r.CONTROL/f'{attempt}.worker.json', worker)
    if missing != 'exit':
        write_json(r.CONTROL/f'{attempt}.exit.json', {
            'launch': launch_ref, 'process_returncode': code, 'seconds': seconds, 'terminal': True})
    return attempt


def test_ledger_inherits_two_failures_pause_and_thirteen_completions(setup):
    manifest = r.initialize()
    add_attempt(manifest, status='PAUSED_RECOVERABLE', seconds=3.)
    state = r.accumulated(manifest)
    assert state['seconds'] == r.INHERITED_SECONDS+3 and state['next_stage'] == 1
    for i in range(1, 13):
        add_attempt(manifest)
        assert r.accumulated(manifest)['next_stage'] == i+1
    state = r.accumulated(manifest)
    assert state['seconds'] == r.INHERITED_SECONDS+27
    assert len(state['completions']) == 13
    r.old._publish_final(manifest, state)
    assert r.verified_tests() == r.reference(r.FINAL)


@pytest.mark.parametrize('field', ['manifest', 'stage', 'previous_exit', 'used_seconds',
                                  'remaining_seconds', 'runtime_seconds', 'command', 'availability'])
def test_launch_chain_tampering_fails(setup, field):
    manifest = r.initialize()
    def mutate(value):
        value[field] = None
    add_attempt(manifest, mutate=mutate)
    with pytest.raises((ValueError, TypeError)):
        r.accumulated(manifest)


@pytest.mark.parametrize('missing', ['worker', 'exit'])
def test_missing_terminal_receipt_never_relaunches(setup, monkeypatch, missing):
    manifest = r.initialize()
    add_attempt(manifest, missing=missing)
    monkeypatch.setattr(r.old, '_execute_command', lambda *a: pytest.fail('unexpected relaunch'))
    with pytest.raises(RuntimeError, match='unreconciled'):
        r.run()


def test_new_failed_attempt_is_terminal(setup, monkeypatch):
    manifest = r.initialize()
    add_attempt(manifest, status='FAILED')
    assert r.accumulated(manifest)['last_status'] == 'FAILED'
    monkeypatch.setattr(r.old, '_execute_command', lambda *a: pytest.fail('unexpected retry'))
    with pytest.raises(RuntimeError, match='cannot be retried'):
        r.run()


def test_budget_counts_failed_and_paused_parent_time(setup):
    manifest = r.initialize()
    add_attempt(manifest, status='PAUSED_RECOVERABLE', seconds=r.old.LIMIT_SECONDS)
    assert r.accumulated(manifest)['seconds'] > r.old.LIMIT_SECONDS
    with pytest.raises(RuntimeError, match='budget'):
        r.run()


def test_parent_executes_exact_twelve_successor_stages_and_publishes_final(setup, monkeypatch):
    r.initialize()
    launched, probes = [], []
    def availability():
        probes.append(True)
        return GPU
    monkeypatch.setattr(r.old, '_profile', lambda: SimpleNamespace(gpu_availability=availability))
    def execute(command, unit, launch_ref, exit_path):
        launch = r._read(launch_ref)
        attempt = exit_path.name.removesuffix('.exit.json')
        assert r._validate_tail(attempt) == (launch_ref, launch, launch['stage'])
        assert command == launch['command'] and unit == launch['unit']
        stage = launch['stage']
        launched.append(stage)
        output = r.old._stage_path(stage)
        if not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            write_json(output, {'fixture': 'typed output only', 'split': stage['split']})
        write_json(r.CONTROL/f'{attempt}.worker.json', {
            'launch': launch_ref, 'stage': stage, 'status': r.old._success_status(stage),
            'seconds': 1., 'peak_rss_bytes': 100, 'peak_reserved_bytes': 0,
            'cuda_initialized': False, 'availability': launch['availability'],
            'output': r.old._stage_reference(stage)})
        end = {'launch': launch_ref, 'process_returncode': 0, 'seconds': 2., 'terminal': True}
        write_json(exit_path, end)
        return end
    monkeypatch.setattr(r.old, '_execute_command', execute)
    result = r.run()
    assert launched == r.old.stage_roster()[1:]
    assert len(probes) == 4
    assert result == r.verified_tests()
    assert r._read(result)['accumulated_seconds'] == r.INHERITED_SECONDS+24


@pytest.mark.parametrize('index', [1, 2, 3])
def test_dispatch_calls_static_json_successors_not_original_ports(monkeypatch, index):
    from src.atencion_armonica import generative_evidence_fresh_data as data
    from src.atencion_armonica import generative_evidence_fresh_inference as original
    from src.atencion_armonica import generative_evidence_fresh_inference_json as inference
    from src.atencion_armonica import generative_evidence_fresh_evaluation_json as evaluation
    monkeypatch.setattr(original, 'prepare_and_predict', lambda *a, **kw: pytest.fail('original inference called'))
    events = []
    freeze = ref('fixture-freeze')
    output = ref('fixture-output')
    def record(name):
        def call(split, **kwargs):
            assert split == 'iid' and kwargs['freeze_ref'] == freeze
            events.append(name)
            return output
        return call
    monkeypatch.setattr(data, 'produce_test', record('produce'))
    monkeypatch.setattr(inference, 'prepare_and_predict', record('predict'))
    monkeypatch.setattr(evaluation, 'evaluate_test', record('evaluate'))
    monkeypatch.setattr(evaluation, 'replay_test', record('replay'))
    assert r._execute_stage(r.old.stage_roster()[index], {'freeze': freeze}, lambda: None, print) == output
    assert events == {1: ['produce', 'predict'], 2: ['evaluate'], 3: ['replay']}[index]


@pytest.mark.parametrize('mutation', [None, 'manifest', 'receipt', 'count', 'failure', 'time', 'nonterminal'])
def test_previous_validator_checks_exact_terminal_authority(monkeypatch, mutation):
    values = {r.previous.MANIFEST: r.PREVIOUS_MANIFEST_SHA,
              **{r.previous.CONTROL/name: sha for name, sha in r.PREVIOUS_FILES.items()}}
    if mutation in ('manifest', 'receipt'):
        values[r.previous.MANIFEST if mutation == 'manifest' else
               r.previous.CONTROL/'attempt-0000.worker.json'] = '0'*64
    monkeypatch.setattr(r, 'reference', lambda path: {
        'path': Path(path).relative_to(r.ROOT).as_posix(), 'sha256': values[Path(path)]})
    monkeypatch.setattr(r.previous, '_launches', lambda: [r.previous.CONTROL/'attempt-0000.launch.json']
                        * (2 if mutation == 'count' else 1))
    def prior(ref):
        if mutation == 'nonterminal':
            raise RuntimeError('unreconciled previous')
        return {'seconds': r.INHERITED_SECONDS+(1 if mutation == 'time' else 0),
                'next_stage': 1, 'last_status': 'FAILED', 'completions': [{}]}
    monkeypatch.setattr(r.previous, 'recovery_accumulated', prior)
    monkeypatch.setattr(r, '_read', lambda ref: {'status': 'FAILED',
        'reason': 'other failure' if mutation == 'failure' else r.FAILURE})
    monkeypatch.setattr(r.previous, 'verify_reconciliation', lambda *a, **kw: ref('reconciliation'))
    monkeypatch.setattr(r.previous, 'verify_original', lambda: {'freeze': ref('freeze')})
    if mutation is None:
        assert REAL_VERIFY_PREVIOUS()['state']['seconds'] == r.INHERITED_SECONDS
    else:
        with pytest.raises((ValueError, RuntimeError)):
            REAL_VERIFY_PREVIOUS()


@pytest.mark.parametrize('pause', [False, True])
def test_worker_checks_authority_before_successor_dispatch_and_handles_signal(setup, monkeypatch, pause):
    stage = r.old.stage_roster()[2]
    launch_ref = ref('worker-launch')
    launch = {'manifest': ref('manifest'), 'unit': 'fixture-unit', 'runtime_seconds': 100}
    monkeypatch.setattr(r, '_validate_tail', lambda attempt: (launch_ref, launch, stage))
    events = []
    def authority(ref):
        events.append('authority')
        return {'fixture': 'amended manifest'}
    monkeypatch.setattr(r, 'read_manifest', authority)
    output = ref('fixture-evaluation')
    monkeypatch.setattr(r.old, '_stage_reference', lambda stage: output)
    def dispatch(stage, manifest, check, progress):
        assert manifest == {'fixture': 'amended manifest'} and events == ['authority']
        events.append('successor')
        if pause:
            signal.raise_signal(signal.SIGTERM)
            check()
        return output
    monkeypatch.setattr(r, '_execute_stage', dispatch)
    def check(started, seconds, stopped, **kwargs):
        if stopped:
            raise InterruptedError('fixture cooperative pause')
    monkeypatch.setattr(r.old, '_training', lambda: SimpleNamespace(
        _verify_service=lambda *a: None, _worker_check=check))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        cuda=SimpleNamespace(is_initialized=lambda: False), set_num_threads=lambda n: None,
        set_num_interop_threads=lambda n: None, use_deterministic_algorithms=lambda b: None))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        monkeypatch.setenv(name, '1')
    assert r.worker_stage('attempt-0000') == (75 if pause else 0)
    report = json.loads((r.CONTROL/'attempt-0000.worker.json').read_bytes())
    assert report['status'] == ('PAUSED_RECOVERABLE' if pause else 'EVALUATED')
    assert report['cuda_initialized'] is False and events == ['authority', 'successor']
