"""Real storage→45-output seal→evaluation→replay on existing OPEN aliases.

This is NOT fresh-test evidence. Sampler/observation authority, backbone
execution and fitting are replaced by the already excluded OPEN scenes 0/26
and their preserved logits/factors. Every serializer, source reconstruction,
input codec, selected CPU head, prediction verifier, seal and metric/replay
operation remains real. No new observation and no CUDA are produced.
"""
from copy import deepcopy
import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.atencion_armonica import generative_evidence_fresh_data as data
from src.atencion_armonica import generative_evidence_fresh_evaluation_json as evaluation
from src.atencion_armonica import generative_evidence_fresh_inference_json as fresh
from src.atencion_armonica import generative_evidence_fresh_store as stores
from src.atencion_armonica.generative_evidence_preparation import ProfileFits
from src.atencion_armonica.generative_evidence_reuse import OpenReuse
from src.atencion_armonica.generative_evidence_test_freeze import FREEZE as REAL_FREEZE
from src.atencion_armonica.partial_compatibility_cache import encoded

TEMP = fresh.ROOT/'.agent-work/phideus-json-integration-tests-20260909'
FREEZE = {'path': 'fixture/open-alias-not-fresh-authority', 'sha256': 'f'*64}


def hashes(folder):
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob('*') if p.is_file()}


def test_complete_alias_pipeline_preserves_codecs_truth_gate_and_replay(monkeypatch):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    root = TEMP/f'run-{os.getpid()}'/'fresh'
    monkeypatch.setattr(stores, 'TEMPORARY', TEMP)
    monkeypatch.setattr(fresh, 'CANONICAL', root)
    monkeypatch.setattr(evaluation, 'CANONICAL', root)
    reuse = OpenReuse()
    shard = reuse.shard('train', 0)
    scenes = {i: shard.scene(i) for i in (0, 26)}
    profile = ProfileFits()
    fitted = {i: profile.fitted('train', i, scenes[i]) for i in scenes}
    assert scenes[0]['partitions'] and not scenes[26]['partitions']
    # OPEN truth is already development evidence, never a fresh sidecar.
    opened_truths = [json.loads(line) for line in shard.data.read('sidecars.jsonl').splitlines()]
    manifest = json.loads(REAL_FREEZE.read_bytes())
    # Preserve real selected heads and TRAIN normalizers; replace only the
    # fixture authority, not source/choice/verifier logic or neural computation.
    frozen = {'freeze': FREEZE, 'manifest': manifest, 'exclusions': {}}
    monkeypatch.setattr(fresh, '_freeze_authority', lambda ref, check: frozen)
    calls = {'backbone': 0, 'fits': 0, 'truth': 0}
    sealed_before_truth = []

    def alias(i):
        return 0 if i == 0 else 26

    class Observations:
        def __init__(self, split, **kwargs):
            assert split == 'iid' and kwargs['freeze_ref'] == FREEZE
            self.index = {'records': [{'path': f'{i}/draw.json', 'sha256': 'b'*64}
                                      for i in range(512)]}
            self.files = SimpleNamespace(root=root/'fixture-observation-authority',
                                         reader=SimpleNamespace(json=self.read))

        def observation(self, i):
            return {**deepcopy(scenes[alias(i)]['observation']), 'scene_id': i,
                    'split_seed': fresh.cache.SPLITS['iid'][1]}

        def read(self, ref):
            i = int(ref['path'].split('/')[0])
            if ref['path'].endswith('/draw.json'):
                return {'sidecar': {'path': f'{i}/sidecar.json', 'sha256': 'c'*64}}
            assert ref['path'].endswith('/sidecar.json')
            # The evaluator's real verifier must already have accepted this.
            seal = json.loads((root/'iid/prediction_seal.json').read_bytes())
            assert seal['prediction_count'] == 45 and seal['status'] == fresh.SEAL_STATUS
            sealed_before_truth.append(seal['prediction_index'])
            calls['truth'] += 1
            return {**deepcopy(opened_truths[alias(i)]), 'scene_id': i,
                    'split_seed': fresh.cache.SPLITS['iid'][1]}

    monkeypatch.setattr(data, 'FreshObservations', Observations)

    def forbidden(*args, **kwargs):
        pytest.fail('fixture attempted a new sampler, backbone, fit or model load')

    from src.atencion_armonica import learned_partition_data
    monkeypatch.setattr(learned_partition_data, '_draw_scene', forbidden)
    runtime = {'torch': reuse.common['runtime']['torch'], 'numpy': reuse.common['runtime']['numpy'],
               'cuda': 'fixture-not-CUDA', 'cudnn': 1, 'device': 'NVIDIA GeForce RTX 3090'}
    monkeypatch.setattr(fresh, '_gpu_runtime', lambda *args: runtime)

    def forward(checkpoint, records, supplied_runtime, device):
        assert device == 'cpu' and supplied_runtime == runtime and len(records) == 512
        calls['backbone'] += 1
        return [scenes[alias(i)]['logits'][checkpoint['seed']].copy() for i in range(512)]

    monkeypatch.setattr(fresh, '_checkpoint_forward', forward)

    def cached_fit(q, partitions, fitter):
        calls['fits'] += 1
        i = 0 if partitions else 26
        assert np.array_equal(q, scenes[i]['q32'])
        assert encoded(partitions) == encoded(scenes[i]['partitions'])
        return deepcopy(fitted[i])

    monkeypatch.setattr(fresh.ge.law, 'fit_candidates', cached_fit)
    monkeypatch.setattr(fresh.ge.law.GroupFitter, 'fit', forbidden)
    def pause_after_first_scene(value):
        if json.loads(value)['scene_id'] == 0:
            raise InterruptedError('fixture pause after durable fit and observable')

    with pytest.raises(InterruptedError, match='fixture pause'):
        fresh.prepare_and_predict('iid', freeze_ref=FREEZE, device='cpu', check=lambda: None,
                                  progress=pause_after_first_scene)
    paused = hashes(root)
    assert calls == {'backbone': 3, 'fits': 1, 'truth': 0}
    assert not (root/'iid/prediction_seal.json').exists()
    seal_ref = fresh.prepare_and_predict('iid', freeze_ref=FREEZE, device='cpu', check=lambda: None,
                                          progress=lambda value: None)
    assert calls == {'backbone': 3, 'fits': 512, 'truth': 0}
    completed = hashes(root)
    assert all(completed[name] == sha for name, sha in paused.items())
    verified = fresh.verify_predictions('iid', freeze_ref=FREEZE, check=lambda: None)
    assert verified['seal'] == seal_ref and len(verified['records']) == 45
    assert len(verified['choices']['records']) == 512
    before = hashes(root)
    monkeypatch.setattr(fresh, '_checkpoint_forward', forbidden)
    monkeypatch.setattr(fresh.ge.law, 'fit_candidates', forbidden)
    monkeypatch.setattr(fresh, '_model', forbidden)
    assert fresh.prepare_and_predict('iid', freeze_ref=FREEZE, device='cpu', check=lambda: None,
                                      progress=lambda value: None) == seal_ref
    assert hashes(root) == before
    assert calls['truth'] == 0

    # Exact fixture files only: preserve bytes, temporarily remove one required
    # output/seal, and restore in finally. Every negative must precede truth.
    payload = fresh.ROOT/verified['records'][0]['prediction']['path']
    for path in (payload, root/'iid/prediction_seal.json'):
        for corrupt in (False, True):
            held = path.with_name(path.name+'.fixture-held')
            path.rename(held)
            try:
                if corrupt:
                    fresh.storage.atomic_bytes(path, b'corrupt fixture payload\n')
                with pytest.raises((ValueError, FileNotFoundError)):
                    evaluation.evaluate_test('iid', freeze_ref=FREEZE, check=lambda: None)
                assert calls['truth'] == 0
            finally:
                if corrupt and path.exists():
                    path.rename(root.parent/(path.name+'.invalid-preserved'))
                held.rename(path)
    assert hashes(root) == before
    assert not (root/'iid/evaluation').exists()
    ref = evaluation.evaluate_test('iid', freeze_ref=FREEZE, check=lambda: None)
    assert calls['truth'] == 512
    after = hashes(root)
    assert evaluation.replay_test('iid', freeze_ref=FREEZE, check=lambda: None) == ref
    assert calls['truth'] == 1024 and hashes(root) == after
    assert len({encoded(r) for r in sealed_before_truth}) == 1
    assert all(after[name] == sha for name, sha in before.items())
    summary = json.loads((root/'iid/evaluation/summary.json').read_bytes())
    assert summary['learned']['primary']['output_count'] == 1
    assert summary['systems']['count'] == 512
    with np.load(root/'iid/evaluation/metrics.npz', allow_pickle=False) as arrays:
        assert arrays['readouts'].shape == (512, 45, 13)
        assert np.isfinite(arrays['readouts'][0]).all()
        assert np.isnan(arrays['readouts'][1:]).all()
    assert not torch.cuda.is_initialized()
