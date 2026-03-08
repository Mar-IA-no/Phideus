#!/usr/bin/env python3
"""
S2-P0: French Lombard Dataset — Manifest + Split + Segment Index + Alignment Audit

Produces:
  data/lombard/manifest.json          — clip-level manifest (9120 entries)
  data/lombard/segment_index.json     — window-level index (deterministic)
  data/lombard/alignment_audit.json   — F0 lag + voiced fraction + clipping stats

Protocol (frozen in P0, used in P1/P2c/P2m):
  sr=16kHz, segment=2.0s, hop=0.5s
  Split: by speaker, gender balanced, seed=42
  Lag correction applied BEFORE freezing segment_index

Usage:
  python experiments/bias_control/escalon2/s2_p0_manifest.py \
      --lombard-dir data/lombard/FLombard \
      --output-dir data/lombard \
      --audit-clips 50 --workers 14
"""

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Protocol constants (frozen) ──────────────────────────────────────────────

PROTOCOL_VERSION = "s2-p0-v1"
SAMPLE_RATE = 16000
SEGMENT_LEN = 2.0       # seconds
SEGMENT_HOP = 0.5       # seconds
SEGMENT_SAMPLES = int(SEGMENT_LEN * SAMPLE_RATE)  # 32000
HOP_SAMPLES = int(SEGMENT_HOP * SAMPLE_RATE)      # 8000

NOISE_MAP = {
    'silence': 'noise0',
    'noise1': 'noise1',
    'noise2': 'noise2',
    'noise3': 'noise3',
}


# ── Segmentation rule (deterministic, versioned) ────────────────────────────

def segment_windows(duration_sec, seg_len=SEGMENT_LEN, hop=SEGMENT_HOP):
    """Generate fixed windows. Same function used in P1, P2c, P2m."""
    windows = []
    t = 0.0
    while t + seg_len <= duration_sec:
        windows.append((round(t, 4), round(t + seg_len, 4)))
        t += hop
    return windows


# ── Split assignment ────────────────────────────────────────────────────────

def assign_splits(speakers_info, seed=42):
    """
    Assign speakers to train/val/test splits with gender balance.

    With 38 speakers (20F/18M): 28 train / 5 val / 5 test.
    Val/test: aim for 2-3 of each gender.
    """
    rng = np.random.RandomState(seed)

    males = sorted([s for s, info in speakers_info.items() if info['gender'] == 'Male'])
    females = sorted([s for s, info in speakers_info.items() if info['gender'] == 'Female'])

    rng.shuffle(males)
    rng.shuffle(females)

    # Test: 3F + 2M = 5
    test_speakers = set(females[:3]) | set(males[:2])
    # Val: 2F + 3M = 5
    val_speakers = set(females[3:5]) | set(males[2:5])
    # Train: rest (15F + 13M = 28)
    train_speakers = set(females[5:]) | set(males[5:])

    splits = {}
    for s in train_speakers:
        splits[s] = 'train'
    for s in val_speakers:
        splits[s] = 'validation'
    for s in test_speakers:
        splits[s] = 'test'

    return splits


# ── Parse transcription files ───────────────────────────────────────────────

def parse_transcriptions(txt_dir):
    """Parse txt files to get sentence text and metadata per clip."""
    clip_meta = {}
    for txt_file in sorted(Path(txt_dir).glob("*.txt")):
        speaker_id = txt_file.stem
        with open(txt_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                rec_id = int(row['Id'])
                list_id = int(row['List'])
                sentence_num = int(row['Sentence'])
                text = row['Text']
                condition = row['Condition']
                session = int(row['Session'])

                noise_cond = NOISE_MAP.get(condition, condition)

                clip_meta[(speaker_id, rec_id)] = {
                    'list_id': list_id,
                    'sentence_num': sentence_num,
                    'text': text,
                    'noise_condition': noise_cond,
                    'session': session,
                }
    return clip_meta


def build_sentence_id_map(clip_meta):
    """Create normalized integer sentence_id from (list_id, sentence_num)."""
    unique_sentences = sorted(set(
        (m['list_id'], m['sentence_num']) for m in clip_meta.values()
    ))
    return {pair: idx for idx, pair in enumerate(unique_sentences)}


# ── Voiced fraction computation ─────────────────────────────────────────────

def compute_voiced_fraction_energy(audio, sr=SAMPLE_RATE, frame_len=0.025, hop_len=0.010):
    """
    Estimate voiced fraction using short-term energy + ZCR.
    Returns fraction of frames classified as voiced.
    """
    frame_samples = int(frame_len * sr)
    hop_samples = int(hop_len * sr)
    n_frames = max(1, (len(audio) - frame_samples) // hop_samples + 1)

    energies = np.zeros(n_frames)
    zcrs = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop_samples
        frame = audio[start:start + frame_samples]
        if len(frame) < frame_samples:
            break
        energies[i] = np.mean(frame ** 2)
        zcrs[i] = np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0

    if energies.max() == 0:
        return 0.0

    energy_norm = energies / (energies.max() + 1e-10)
    # Voiced: energy above threshold AND zcr below threshold
    voiced = (energy_norm > 0.01) & (zcrs < 0.3)
    return float(voiced.mean())


# ── Alignment audit (F0 lag measurement) ────────────────────────────────────

def audit_single_clip(args):
    """Audit a single clip: clipping, voiced fraction, F0 lag."""
    speech_path, egg_path, clip_id = args
    result = {'clip_id': clip_id}

    try:
        speech, sr1 = sf.read(speech_path)
        egg, sr2 = sf.read(egg_path)
    except Exception as e:
        result['error'] = str(e)
        return result

    result['sr_speech'] = sr1
    result['sr_egg'] = sr2
    result['frames_speech'] = len(speech)
    result['frames_egg'] = len(egg)
    result['frame_match'] = len(speech) == len(egg)
    result['duration_sec'] = len(speech) / sr1

    # Clipping
    result['clipping_speech'] = int((np.abs(speech) > 0.99).sum())
    result['clipping_egg'] = int((np.abs(egg) > 0.99).sum())

    # Voiced fraction (both modalities)
    result['voiced_fraction_speech'] = compute_voiced_fraction_energy(speech, sr1)
    result['voiced_fraction_egg'] = compute_voiced_fraction_energy(egg, sr2)

    # F0 lag via cross-correlation of energy envelopes
    # (More robust than raw waveform xcorr, lighter than full F0 extraction)
    frame_len = int(0.025 * sr1)
    hop = int(0.010 * sr1)
    n_frames = min(
        (len(speech) - frame_len) // hop + 1,
        (len(egg) - frame_len) // hop + 1,
    )
    if n_frames > 10:
        speech_env = np.array([
            np.sqrt(np.mean(speech[i*hop:i*hop+frame_len]**2))
            for i in range(n_frames)
        ])
        egg_env = np.array([
            np.sqrt(np.mean(egg[i*hop:i*hop+frame_len]**2))
            for i in range(n_frames)
        ])

        # Normalize
        speech_env = (speech_env - speech_env.mean()) / (speech_env.std() + 1e-10)
        egg_env = (egg_env - egg_env.mean()) / (egg_env.std() + 1e-10)

        # Cross-correlation (max lag = 50 frames = 500ms)
        max_lag = min(50, n_frames // 2)
        xcorr = np.correlate(speech_env, egg_env[max_lag:-max_lag], mode='full') if n_frames > 2 * max_lag else np.array([0])

        if len(xcorr) > 0:
            best_lag_idx = np.argmax(xcorr)
            center = len(xcorr) // 2
            lag_frames = best_lag_idx - center
            lag_ms = lag_frames * (hop / sr1) * 1000
            result['lag_frames'] = int(lag_frames)
            result['lag_ms'] = round(lag_ms, 2)
            result['xcorr_peak'] = round(float(xcorr.max() / n_frames), 4)

    return result


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='S2-P0: Build manifest + segment index + alignment audit')
    parser.add_argument('--lombard-dir', default='data/lombard/FLombard',
                        help='Path to extracted FLombard directory')
    parser.add_argument('--output-dir', default='data/lombard',
                        help='Output directory for manifest, segment_index, audit')
    parser.add_argument('--audit-clips', type=int, default=50,
                        help='Number of clips to audit for alignment (0 to skip)')
    parser.add_argument('--workers', type=int, default=14,
                        help='Parallel workers for audit')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    lombard = Path(args.lombard_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # ── 1. Read speakers ────────────────────────────────────────────────
    speakers_info = {}
    with open(lombard / 'speakers.txt') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                speakers_info[parts[0]] = {
                    'gender': parts[1],
                    'age': int(parts[2]),
                }

    n_male = sum(1 for v in speakers_info.values() if v['gender'] == 'Male')
    n_female = sum(1 for v in speakers_info.values() if v['gender'] == 'Female')
    logger.info(f"Speakers: {len(speakers_info)} ({n_female}F / {n_male}M)")

    # ── 2. Assign splits ────────────────────────────────────────────────
    speaker_splits = assign_splits(speakers_info, seed=args.seed)

    for split_name in ['train', 'validation', 'test']:
        spks = sorted(s for s, sp in speaker_splits.items() if sp == split_name)
        genders = [speakers_info[s]['gender'][0] for s in spks]
        logger.info(f"  {split_name}: {len(spks)} speakers ({genders.count('F')}F/{genders.count('M')}M): {spks}")

    # ── 3. Parse transcriptions ─────────────────────────────────────────
    clip_meta = parse_transcriptions(lombard / 'txt')
    sentence_id_map = build_sentence_id_map(clip_meta)
    logger.info(f"Transcription entries: {len(clip_meta)}, unique sentences: {len(sentence_id_map)}")

    # ── 4. Build clip-level manifest ────────────────────────────────────
    wav_dir = lombard / 'process' / 'wav'
    egg_dir = lombard / 'process' / 'egg'

    manifest = []
    duration_issues = 0

    for speaker_id in sorted(os.listdir(wav_dir)):
        spk_wav_dir = wav_dir / speaker_id
        if not spk_wav_dir.is_dir():
            continue

        # Build rec_id mapping from txt file
        txt_path = lombard / 'txt' / f'{speaker_id}.txt'
        rec_to_filename = {}
        if txt_path.exists():
            with open(txt_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    rec_id = int(row['Id'])
                    list_id = int(row['List'])
                    sentence_num = int(row['Sentence'])
                    session = int(row['Session'])
                    condition = NOISE_MAP.get(row['Condition'], row['Condition'])

                    fname = f"{speaker_id}_s{session}_l{list_id}_sen{sentence_num}_{condition}.wav"
                    rec_to_filename[rec_id] = {
                        'filename': fname,
                        'list_id': list_id,
                        'sentence_num': sentence_num,
                        'session': session,
                        'noise_condition': condition,
                    }

        for wav_file in sorted(os.listdir(spk_wav_dir)):
            if not wav_file.endswith('.wav'):
                continue

            speech_path = str(spk_wav_dir / wav_file)
            egg_path = str(egg_dir / speaker_id / wav_file)

            if not os.path.exists(egg_path):
                logger.warning(f"Missing EGG: {egg_path}")
                continue

            info = sf.info(speech_path)
            duration = info.duration

            if duration < SEGMENT_LEN:
                duration_issues += 1
                continue

            # Parse filename for metadata
            parts = wav_file.replace('.wav', '').split('_')
            spk = parts[0]
            session = int(parts[1].replace('s', ''))
            list_id = int(parts[2].replace('l', ''))
            sentence_num = int(parts[3].replace('sen', ''))
            noise_cond = parts[4]

            sentence_id = sentence_id_map.get((list_id, sentence_num), -1)

            clip_id = wav_file.replace('.wav', '')

            manifest.append({
                'clip_id': clip_id,
                'speaker_id': spk,
                'gender': speakers_info[spk]['gender'],
                'age': speakers_info[spk]['age'],
                'noise_condition': noise_cond,
                'session': session,
                'list_id': list_id,
                'sentence_num': sentence_num,
                'sentence_id': sentence_id,
                'speech_path': f"process/wav/{spk}/{wav_file}",
                'egg_path': f"process/egg/{spk}/{wav_file}",
                'duration_sec': round(duration, 4),
                'split': speaker_splits[spk],
            })

    logger.info(f"Manifest: {len(manifest)} clips ({duration_issues} excluded for duration < {SEGMENT_LEN}s)")

    # Stats per split
    for split_name in ['train', 'validation', 'test']:
        clips = [c for c in manifest if c['split'] == split_name]
        total_dur = sum(c['duration_sec'] for c in clips)
        noise0 = sum(1 for c in clips if c['noise_condition'] == 'noise0')
        logger.info(f"  {split_name}: {len(clips)} clips, {total_dur/3600:.2f}h, {noise0} noise0")

    # Save manifest
    manifest_path = output / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump({
            'protocol_version': PROTOCOL_VERSION,
            'sample_rate': SAMPLE_RATE,
            'segment_len': SEGMENT_LEN,
            'segment_hop': SEGMENT_HOP,
            'n_speakers': len(speakers_info),
            'n_clips': len(manifest),
            'seed': args.seed,
            'splits': {
                s: sorted(sp for sp, split in speaker_splits.items() if split == s)
                for s in ['train', 'validation', 'test']
            },
            'clips': manifest,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {manifest_path}")

    # ── 5. Build segment index ──────────────────────────────────────────
    segment_index = []

    for clip in manifest:
        windows = segment_windows(clip['duration_sec'])
        for seg_idx, (start, end) in enumerate(windows):
            segment_index.append({
                'clip_id': clip['clip_id'],
                'segment_idx': seg_idx,
                'start_sec': start,
                'end_sec': end,
                'speaker_id': clip['speaker_id'],
                'gender': clip['gender'],
                'noise_condition': clip['noise_condition'],
                'sentence_id': clip['sentence_id'],
                'split': clip['split'],
            })

    logger.info(f"Segment index: {len(segment_index)} segments")

    for split_name in ['train', 'validation', 'test']:
        segs = [s for s in segment_index if s['split'] == split_name]
        noise0 = sum(1 for s in segs if s['noise_condition'] == 'noise0')
        logger.info(f"  {split_name}: {len(segs)} segments ({noise0} noise0)")

    seg_index_path = output / 'segment_index.json'
    with open(seg_index_path, 'w') as f:
        json.dump({
            'protocol_version': PROTOCOL_VERSION,
            'segmentation_rule': 'segment_windows(duration, seg_len=2.0, hop=0.5)',
            'segment_len': SEGMENT_LEN,
            'segment_hop': SEGMENT_HOP,
            'lag_correction_samples': 0,
            'n_segments': len(segment_index),
            'segments': segment_index,
        }, f, indent=2)
    logger.info(f"Saved: {seg_index_path}")

    # ── 6. Alignment audit ──────────────────────────────────────────────
    if args.audit_clips > 0:
        logger.info(f"Running alignment audit on {args.audit_clips} clips...")

        # Sample clips stratified by speaker
        rng = np.random.RandomState(args.seed)
        speakers_list = sorted(set(c['speaker_id'] for c in manifest))
        clips_per_speaker = max(1, args.audit_clips // len(speakers_list))

        audit_tasks = []
        for spk in speakers_list:
            spk_clips = [c for c in manifest if c['speaker_id'] == spk]
            rng.shuffle(spk_clips)
            for clip in spk_clips[:clips_per_speaker]:
                speech_path = str(lombard / clip['speech_path'])
                egg_path = str(lombard / clip['egg_path'])
                audit_tasks.append((speech_path, egg_path, clip['clip_id']))
                if len(audit_tasks) >= args.audit_clips:
                    break
            if len(audit_tasks) >= args.audit_clips:
                break

        # Run audit in parallel
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            audit_results = list(pool.map(audit_single_clip, audit_tasks))

        # Aggregate
        lags = [r['lag_ms'] for r in audit_results if 'lag_ms' in r]
        voiced_speech = [r['voiced_fraction_speech'] for r in audit_results if 'voiced_fraction_speech' in r]
        voiced_egg = [r['voiced_fraction_egg'] for r in audit_results if 'voiced_fraction_egg' in r]
        clipping_s = sum(r.get('clipping_speech', 0) for r in audit_results)
        clipping_e = sum(r.get('clipping_egg', 0) for r in audit_results)
        frame_matches = sum(1 for r in audit_results if r.get('frame_match', False))

        aggregate = {
            'n_audited': len(audit_results),
            'frame_match_rate': f"{frame_matches}/{len(audit_results)}",
            'total_clipping_speech': clipping_s,
            'total_clipping_egg': clipping_e,
        }

        if lags:
            lag_arr = np.array(lags)
            aggregate['lag_ms'] = {
                'mean': round(float(lag_arr.mean()), 3),
                'std': round(float(lag_arr.std()), 3),
                'min': round(float(lag_arr.min()), 3),
                'max': round(float(lag_arr.max()), 3),
                'median': round(float(np.median(lag_arr)), 3),
                'abs_mean': round(float(np.abs(lag_arr).mean()), 3),
            }
            if abs(lag_arr.mean()) < 5.0:
                aggregate['lag_correction_samples'] = 0
                aggregate['lag_verdict'] = 'No systematic lag detected (mean < 5ms)'
            else:
                correction = int(round(lag_arr.mean() / 1000 * SAMPLE_RATE))
                aggregate['lag_correction_samples'] = correction
                aggregate['lag_verdict'] = f'Systematic lag detected: {lag_arr.mean():.1f}ms = {correction} samples'

        if voiced_speech:
            vs = np.array(voiced_speech)
            ve = np.array(voiced_egg)
            aggregate['voiced_fraction_speech'] = {
                'mean': round(float(vs.mean()), 4),
                'std': round(float(vs.std()), 4),
                'min': round(float(vs.min()), 4),
                'p10': round(float(np.percentile(vs, 10)), 4),
                'median': round(float(np.median(vs)), 4),
            }
            aggregate['voiced_fraction_egg'] = {
                'mean': round(float(ve.mean()), 4),
                'std': round(float(ve.std()), 4),
                'min': round(float(ve.min()), 4),
                'p10': round(float(np.percentile(ve, 10)), 4),
                'median': round(float(np.median(ve)), 4),
            }
            # Threshold: p10 of speech voiced fraction in audited clips
            threshold = round(float(np.percentile(vs, 10)), 4)
            aggregate['voiced_threshold'] = threshold
            aggregate['voiced_threshold_rule'] = 'percentile 10 of speech voiced_fraction in audit sample'

        audit_output = {
            'protocol_version': PROTOCOL_VERSION,
            'aggregate': aggregate,
            'per_clip': audit_results,
        }

        audit_path = output / 'alignment_audit.json'
        with open(audit_path, 'w') as f:
            json.dump(audit_output, f, indent=2)
        logger.info(f"Saved: {audit_path}")

        # Print summary
        logger.info("=== ALIGNMENT AUDIT SUMMARY ===")
        logger.info(f"  Frame matches: {aggregate['frame_match_rate']}")
        logger.info(f"  Clipping: speech={clipping_s}, egg={clipping_e}")
        if 'lag_ms' in aggregate:
            logger.info(f"  Lag: mean={aggregate['lag_ms']['mean']}ms, std={aggregate['lag_ms']['std']}ms")
            logger.info(f"  Verdict: {aggregate.get('lag_verdict', 'N/A')}")
        if 'voiced_fraction_speech' in aggregate:
            logger.info(f"  Voiced (speech): mean={aggregate['voiced_fraction_speech']['mean']}, p10={aggregate['voiced_fraction_speech']['p10']}")
            logger.info(f"  Voiced (egg): mean={aggregate['voiced_fraction_egg']['mean']}, p10={aggregate['voiced_fraction_egg']['p10']}")
            logger.info(f"  Voiced threshold: {aggregate.get('voiced_threshold', 'N/A')}")

    # ── 7. Summary ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("S2-P0 COMPLETE")
    logger.info(f"  Manifest: {len(manifest)} clips -> {manifest_path}")
    logger.info(f"  Segment index: {len(segment_index)} segments -> {seg_index_path}")
    if args.audit_clips > 0:
        logger.info(f"  Alignment audit: {args.audit_clips} clips -> {audit_path}")
    logger.info(f"  Protocol version: {PROTOCOL_VERSION}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
