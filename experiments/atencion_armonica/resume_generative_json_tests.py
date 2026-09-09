"""One-off JSON comparison amendment for the preserved fresh-test campaign.

Metadata-only parent, inherited original and namespace-recovery failures,
unchanged resources and stage roster, and explicit static JSON successors.
This module never rewrites a preceding source, manifest, receipt or result.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import uuid

from experiments.atencion_armonica import recover_generative_tests as previous
from experiments.atencion_armonica import run_generative_tests as old
from src.atencion_armonica.generative_evidence_reuse import ROOT
from src.atencion_armonica.generative_evidence_storage import write_json

reference = previous.reference
_read = previous._read
CONTROL = old.TEMP/'test-json-control'
MANIFEST = CONTROL/'manifest.json'
FINAL = old.FINAL
PLAN = ROOT/'Biblioteca/Geometria_Proporcional_Ground_Truth/notes/20260909_fresh_json_recovery_plan.md'
AUDIT = ROOT/'Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/736_generative_json_recovery_plan_audit.md'
INHERITED_SECONDS = 250.42521479801508
FAILURE = 'ValueError: stored JSON changed after publication'
PREVIOUS_MANIFEST_SHA = 'a6ff16e73ed039ec4fd0d673cd0a2306e4a874410d86c26405e386a450b3a29f'
PREVIOUS_FILES = {
    'attempt-0000.launch.json': 'c8f47969a9c79d36dd3fea1a93ed1c840f9830a1731cff034445d75bac271283',
    'attempt-0000.worker.json': '422b32295254655ebe651592f0f4430f4ac4b26523d7fe5d4fbdfe93e6c8afeb',
    'attempt-0000.exit.json': 'b608b1b4a1efaaba5bf857040573515cefb781a0213bd71a2182377c37ca49bb',
    'namespace-reconciliation.json': '804a6723f81f2b70a533b72e098bdb72c8e159ae97f88ca100315f6bca47ee81',
}
CODECS = {
    'src/atencion_armonica/partial_compatibility_cache.py': '8f6b8b091a9575e8d70fa13c897b3b2686a4f85da20019dacc2178cf2575b52b',
    'src/atencion_armonica/generative_evidence_storage.py': '78d30f8b1ead6207292f3b0fb46e968ef9d6e035ff9b277dc3d1e844fa269786',
}
SOURCE_PREFIX = {
    'iid/source/features.json': 'd049321a3418386c8bec489dc459559425c8b86651e5b1c626cca3dad2a585f5',
    'iid/source/scenes/00000.json': 'e99dd274eb319c55471f7d5d7fb2c762705817d6834b87dd0cba070861624521',
    'iid/source/logits-2026090721.json': 'bb46332b1b036b2866ebcd227647477443b41796e1529c6a033a1ebc6126e0c8',
    'iid/source/logits-2026090722.json': '5fe2292cc4fdecc13866e18cefe3791d793e5771509f2d57852a8b2b08a67695',
    'iid/source/logits-2026090723.json': 'e22b613192be33bbf675aa3d9128566fd25c6d1e1eb3d76560b48bcd8c79a8fe',
}
NEW_SOURCES = (
    'src/atencion_armonica/generative_evidence_fresh_inference_json.py',
    'src/atencion_armonica/generative_evidence_fresh_evaluation_json.py',
    'experiments/atencion_armonica/test_generative_evidence_json_contract.py',
    'experiments/atencion_armonica/test_generative_evidence_json_integration.py',
    'experiments/atencion_armonica/test_resume_generative_json_tests.py',
)
LAUNCH_SCHEMA = 'generative-evidence-json-recovery-launch-v1'
LAUNCH_KEYS = {'schema', 'manifest', 'stage', 'previous_exit', 'unit', 'command',
               'availability', 'used_seconds', 'remaining_seconds', 'runtime_seconds'}


def sources():
    pinned = {**CODECS,
        PLAN.relative_to(ROOT).as_posix(): '4ec0849aa89cdc9e6cbe3b235488f7463ea6604ddafb35f4097463a6ad9db8fa',
        AUDIT.relative_to(ROOT).as_posix(): 'fc400ce45f5e37a66baa1b5159c88df4a824e38483722ad4c53cf7a9f142a0af'}
    paths = {Path(__file__).resolve(), *(ROOT/name for name in (*NEW_SOURCES, *pinned))}
    result = {}
    for path in sorted(paths):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f'JSON recovery source missing: {path}')
        ref = reference(path)
        if ref['path'] in pinned and ref['sha256'] != pinned[ref['path']]:
            raise ValueError('JSON recovery plan, audit or critical codec differs')
        result[ref['path']] = ref['sha256']
    return result


def verify_previous():
    manifest_ref = reference(previous.MANIFEST)
    if manifest_ref['sha256'] != PREVIOUS_MANIFEST_SHA:
        raise ValueError('namespace recovery manifest differs')
    if [p.name for p in previous._launches()] != ['attempt-0000.launch.json']:
        raise ValueError('JSON recovery requires the exact single namespace-recovery attempt')
    for name, sha in PREVIOUS_FILES.items():
        if reference(previous.CONTROL/name)['sha256'] != sha:
            raise ValueError(f'namespace recovery receipt differs: {name}')
    # These validators authenticate original manifests, both prior terminal
    # chains, their sources, freeze and namespace without parsing q/sidecars.
    state = previous.recovery_accumulated(manifest_ref)
    failed = _read(reference(previous.CONTROL/'attempt-0000.worker.json'))
    if (state['seconds'] != INHERITED_SECONDS or state['next_stage'] != 1
            or state['last_status'] != 'FAILED' or len(state['completions']) != 1
            or failed.get('status') != 'FAILED' or failed.get('reason') != FAILURE):
        raise ValueError('preceding state is not the exact terminal JSON failure')
    reconciliation = previous.verify_reconciliation(manifest_ref, deep=False, check=None)
    original = previous.verify_original()
    return {'manifest': manifest_ref, 'state': state, 'original': original,
            'reconciliation': reconciliation, 'freeze': original['freeze']}


def preserved_sources():
    """Authenticate prefix bytes, but never decode observed arrays or a scene."""
    result = []
    for name, sha in sorted(SOURCE_PREFIX.items()):
        ref = reference(old.FRESH/name)
        if ref['sha256'] != sha:
            raise ValueError(f'preserved IID source differs: {name}')
        result.append(ref)
    features = _read(reference(old.FRESH/'iid/source/features.json'))
    if features.get('scene_ids') != list(range(512)) or len(features.get('records', [])) != 512:
        raise ValueError('preserved feature roster differs')
    records = [(value, old.FRESH/f'iid/source/features/{i:05d}.npz')
               for i, value in enumerate(features['records'])]
    for cp in (2026090721, 2026090722, 2026090723):
        receipt = _read(reference(old.FRESH/f'iid/source/logits-{cp}.json'))
        records.append((receipt['logits'], old.FRESH/f'iid/source/logits-{cp}.npz'))
    for value, path in records:
        ref = reference(path)
        if (set(value) != {'path', 'sha256', 'bytes'} or type(value['bytes']) is not int
                or value['path'] != ref['path'] or value['sha256'] != ref['sha256']
                or value['bytes'] != path.stat().st_size):
            raise ValueError('preserved source payload identity or bytes differ')
        result.append(ref)
    return sorted(result, key=lambda r: r['path'])


def strict_initial_tree(preserved):
    """Only initialize may require the exact pre-amendment tree."""
    if not old.FRESH.is_dir() or old.FRESH.is_symlink():
        raise ValueError('missing preserved fresh tree')
    expected = {'binding.json', 'draws/iid/index.json',
        *(f'draws/iid/{i:05d}/{name}.json' for i in range(512)
          for name in ('intent', 'observation', 'sidecar', 'draw')),
        *((ROOT/ref['path']).relative_to(old.FRESH).as_posix() for ref in preserved)}
    directories = {p.as_posix() for name in expected for p in Path(name).parents if p != Path('.')}
    actual_files, actual_dirs = set(), set()
    for path in old.FRESH.rglob('*'):
        if path.is_symlink():
            raise ValueError('initial fresh tree contains a symlink')
        name = path.relative_to(old.FRESH).as_posix()
        if path.is_file():
            actual_files.add(name)
        elif path.is_dir():
            actual_dirs.add(name)
        else:
            raise ValueError('initial fresh tree contains a special node')
    if actual_files != expected or actual_dirs != directories:
        raise ValueError('initial fresh tree differs from the exact preserved IID prefix')


def _manifest_value(context):
    original = context['original']
    return {'schema': 'generative-evidence-json-recovery-v1',
        'stage': 'JSON_REPRESENTATION_AMENDED_NOT_PROMOTED',
        'plan': reference(PLAN), 'plan_audit': reference(AUDIT), 'sources': sources(),
        'previous_manifest': context['manifest'], 'previous_exit': context['state']['previous_exit'],
        'original_manifest': original['manifest'], 'reconciliation': context['reconciliation'],
        'inherited_seconds': context['state']['seconds'], 'freeze': context['freeze'],
        'iid_index': original['iid'], 'preserved_sources': preserved_sources(),
        'stages': old.stage_roster()[1:], 'limits': original['value']['limits'],
        'output': original['value']['output'],
        'policy': {'six_json_comparisons_only': True, 'post_draw_source_amendment': True,
                   'no_redraw': True, 'no_adaptive_tuning': True, 'promotion': False}}


def initialize():
    training = old._training()
    with training._lock(old.COMMON_LOCK), training._lock(CONTROL/'operator.lock'):
        if MANIFEST.exists():
            raise FileExistsError('JSON recovery manifest already exists')
        if list(CONTROL.glob('attempt-*')):
            raise ValueError('cannot initialize over a prior attempt')
        value = _manifest_value(verify_previous())
        strict_initial_tree(value['preserved_sources'])
        CONTROL.mkdir(parents=True, exist_ok=True)
        write_json(MANIFEST, value)
        ref = reference(MANIFEST)
        read_manifest(ref)
        return ref


def read_manifest(ref):
    if not previous._valid_ref(ref) or ref['path'] != MANIFEST.relative_to(ROOT).as_posix():
        raise ValueError('unexpected JSON recovery manifest')
    value = _read(ref)
    if value != _manifest_value(verify_previous()):
        raise ValueError('JSON amendment sources, preserved prefix or authority differs')
    return value


def _service_command(unit, runtime_seconds, stage, attempt):
    command = old._service_command(unit, runtime_seconds, stage, attempt)
    token = 'experiments.atencion_armonica.run_generative_tests'
    if command.count(token) != 1:
        raise ValueError('historical command has no unique worker module')
    command[command.index(token)] = 'experiments.atencion_armonica.resume_generative_json_tests'
    return command


def _launches():
    return sorted(CONTROL.glob('attempt-*.launch.json'))


def _validate_launch(launch, manifest_ref, state, attempt):
    if state['next_stage'] >= 13 or state['last_status'] == 'FAILED':
        raise ValueError('JSON recovery launch follows no eligible stage')
    stage = old.stage_roster()[state['next_stage']]
    remaining = math.floor(old.LIMIT_SECONDS-state['seconds'])
    if remaining <= old.GRACE+10:
        raise RuntimeError('inherited test budget exhausted')
    availability = launch.get('availability')
    if (set(launch) != LAUNCH_KEYS or launch.get('schema') != LAUNCH_SCHEMA
            or launch.get('manifest') != manifest_ref or launch.get('stage') != stage
            or launch.get('previous_exit') != state['previous_exit']
            or launch.get('used_seconds') != state['seconds']
            or launch.get('remaining_seconds') != remaining
            or launch.get('runtime_seconds') != remaining-old.GRACE
            or launch.get('command') != _service_command(launch.get('unit'), remaining-old.GRACE, stage, attempt)
            or (stage['device'] == 'cpu' and availability is not None)
            or (stage['device'] == 'cuda:0' and not old._availability_ok(availability))):
        raise ValueError('JSON recovery launch chain, stage, resources or command differs')
    return stage


def _prefix(manifest_ref, paths):
    read_manifest(manifest_ref)
    context = verify_previous()
    all_paths = _launches()
    if ([p.name for p in all_paths] != [f'attempt-{i:04d}.launch.json' for i in range(len(all_paths))]
            or all_paths[:len(paths)] != list(paths)):
        raise ValueError('JSON recovery attempts are not a contiguous exact prefix')
    state = {**context['state'], 'completions': list(context['state']['completions']),
             'last_status': 'EXACT_JSON_FAILURE_ACCEPTED'}
    for path in paths:
        launch_ref = reference(path)
        launch = _read(launch_ref)
        attempt = path.name.removesuffix('.launch.json')
        stage = _validate_launch(launch, manifest_ref, state, attempt)
        exit_path = CONTROL/f'{attempt}.exit.json'
        worker_path = CONTROL/f'{attempt}.worker.json'
        if not exit_path.exists() or not worker_path.exists():
            raise RuntimeError(f'unreconciled JSON recovery attempt: {launch.get("unit")}')
        exit_ref, worker_ref = reference(exit_path), reference(worker_path)
        end, worker = _read(exit_ref), _read(worker_ref)
        if (end.get('launch') != launch_ref or end.get('terminal') is not True
                or not old._number(end.get('seconds')) or worker.get('launch') != launch_ref):
            raise ValueError('JSON recovery attempt lacks matching terminal receipts')
        status = worker.get('status')
        output = old._stage_reference(stage) if status == old._success_status(stage) else None
        if not old._worker_schema(worker, end.get('process_returncode'), stage, output):
            raise ValueError('JSON recovery worker receipt is untyped or names another output')
        state['seconds'] += end['seconds']
        state['previous_exit'] = exit_ref
        state['last_status'] = status
        if status == old._success_status(stage):
            state['completions'].append({'stage': stage, 'launch': launch_ref,
                'exit': exit_ref, 'worker': worker_ref, 'output': output})
            state['next_stage'] += 1
            state['last_status'] = 'COMPLETE'
    return state


def accumulated(manifest_ref):
    return _prefix(manifest_ref, _launches())


def _validate_tail(attempt):
    old._attempt_name(attempt)
    path = CONTROL/f'{attempt}.launch.json'
    paths = _launches()
    if not paths or paths[-1] != path:
        raise ValueError('JSON recovery worker is not the unique ledger tail')
    ref = reference(path)
    launch = _read(ref)
    state = _prefix(launch['manifest'], paths[:-1])
    return ref, launch, _validate_launch(launch, launch['manifest'], state, attempt)


def _execute_stage(stage, manifest, check, progress):
    freeze = manifest['freeze']
    if stage['kind'] == 'predict':
        from src.atencion_armonica import generative_evidence_fresh_data as data
        from src.atencion_armonica import generative_evidence_fresh_inference_json as inference
        data.produce_test(stage['split'], freeze_ref=freeze, check=check)
        return inference.prepare_and_predict(stage['split'], freeze_ref=freeze,
            device=stage['device'], check=check, progress=progress)
    from src.atencion_armonica import generative_evidence_fresh_evaluation_json as evaluation
    if stage['kind'] == 'evaluate':
        return evaluation.evaluate_test(stage['split'], freeze_ref=freeze, check=check)
    if stage['kind'] == 'replay':
        return evaluation.replay_test(stage['split'], freeze_ref=freeze, check=check)
    raise ValueError('JSON recovery cannot replace the original freeze or add stages')


def worker_stage(attempt):
    launch_ref, launch, stage = _validate_tail(attempt)
    started, stopped, last_disk = time.monotonic(), [], [0.]
    def stop(signum, frame):
        stopped.append(signum)
    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    torch = None
    availability = None
    report = {"launch": launch_ref, "stage": stage, "status": "INCOMPLETE"}
    code = 1
    try:
        old._training()._verify_service(launch["unit"], launch["runtime_seconds"])
        manifest = read_manifest(launch["manifest"])
        import torch as torch_module
        torch = torch_module
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA initialized before recovery worker device checks")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        if (any(os.environ.get(name) != "1" for name in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
                or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"):
            raise RuntimeError("recovery worker deterministic environment differs")
        if stage["device"] == "cuda:0":
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
                raise RuntimeError("GPU recovery stage lacks its exact visible device")
            availability = old._profile().gpu_availability()
            if not old._availability_ok(availability):
                raise RuntimeError("GPU unavailable before recovery")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.set_device(0)
            fraction = old.MEMORY_LIMIT/torch.cuda.get_device_properties(0).total_memory
            if not 0 < fraction <= 1:
                raise RuntimeError("recovery GPU fraction is invalid")
            torch.cuda.set_per_process_memory_fraction(fraction, 0)
            torch.cuda.reset_peak_memory_stats(0)
        elif os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("CPU recovery stage must hide CUDA")
        def check():
            now = time.monotonic()
            due = now-last_disk[0] >= 30
            active = torch if stage["device"] == "cuda:0" and torch.cuda.is_initialized() else None
            old._training()._worker_check(started, launch["runtime_seconds"], stopped,
                                          torch=active, check_storage=due)
            if due:
                last_disk[0] = now
        def progress(value):
            print(value if isinstance(value, str) else json.dumps(value, sort_keys=True), flush=True)
        check()
        output = _execute_stage(stage, manifest, check, progress)
        expected = old._stage_reference(stage)
        if output != expected:
            raise ValueError("recovered stage returned another output")
        check()
        report.update(status=old._success_status(stage), output=expected)
        code = 0
    except InterruptedError as exc:
        report.update(status="PAUSED_RECOVERABLE", reason=str(exc))
        code = 75
    except BaseException as exc:
        report.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        initialized = bool(torch is not None and torch.cuda.is_initialized())
        peak = torch.cuda.max_memory_reserved(0) if initialized and stage["device"] == "cuda:0" else 0
        report.update(seconds=time.monotonic()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_reserved_bytes=peak, cuda_initialized=initialized, availability=availability)
        write_json(CONTROL/f"{attempt}.worker.json", report)
    return code


def verified_tests():
    manifest_ref = reference(MANIFEST)
    state = accumulated(manifest_ref)
    if state['seconds'] > old.LIMIT_SECONDS:
        raise RuntimeError('JSON recovered tests exceeded the inherited four-hour budget')
    expected = old._final_value(manifest_ref, state)
    ref = reference(FINAL)
    if _read(ref) != expected:
        raise ValueError('final index differs from all three typed execution chains')
    return ref


def run():
    training = old._training()
    with training._lock(old.COMMON_LOCK), training._lock(CONTROL/'operator.lock'):
        manifest_ref = reference(MANIFEST)
        read_manifest(manifest_ref)
        while True:
            state = accumulated(manifest_ref)
            if state['seconds'] > old.LIMIT_SECONDS:
                raise RuntimeError('JSON recovery exceeded the inherited four-hour budget')
            if state['next_stage'] == 13:
                old._publish_final(manifest_ref, state)
                return verified_tests()
            if state['last_status'] == 'FAILED':
                raise RuntimeError('new failed JSON recovery attempt cannot be retried automatically')
            remaining = math.floor(old.LIMIT_SECONDS-state['seconds'])
            if remaining <= old.GRACE+10:
                raise RuntimeError('inherited fresh-test time exhausted')
            training.check_disk()
            stage = old.stage_roster()[state['next_stage']]
            availability = None
            if stage['device'] == 'cuda:0':
                availability = old._profile().gpu_availability()
                if not old._availability_ok(availability):
                    raise RuntimeError('GPU unavailable before JSON recovery launch')
            attempt = f'attempt-{len(_launches()):04d}'
            unit = 'phideus-generative-test-'+uuid.uuid4().hex[:16]
            command = _service_command(unit, remaining-old.GRACE, stage, attempt)
            launch_path = CONTROL/f'{attempt}.launch.json'
            write_json(launch_path, {'schema': LAUNCH_SCHEMA, 'manifest': manifest_ref,
                'stage': stage, 'previous_exit': state['previous_exit'], 'unit': unit,
                'command': command, 'availability': availability,
                'used_seconds': state['seconds'], 'remaining_seconds': remaining,
                'runtime_seconds': remaining-old.GRACE})
            end = old._execute_command(command, unit, reference(launch_path), CONTROL/f'{attempt}.exit.json')
            if end.get('process_returncode') == 75:
                raise InterruptedError('JSON recovery paused at a durable boundary')
            if end.get('process_returncode') != 0:
                raise RuntimeError('JSON recovery stage did not complete')
            updated = accumulated(manifest_ref)
            if updated['next_stage'] != state['next_stage']+1:
                raise RuntimeError('JSON recovery exit did not complete its typed stage')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--initialize', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--attempt', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.attempt is None:
            parser.error('worker requires --attempt')
        sys.exit(worker_stage(args.attempt))
    if args.attempt is not None:
        parser.error('--attempt is worker-only')
    print(json.dumps(initialize() if args.initialize else run(), sort_keys=True))


if __name__ == '__main__':
    main()
