"""
Gate 6 — Exp 0: Transkun Baseline Verification

Verifies that Transkun v2 transcribes MAESTRO segments faithfully.
Tests both 4s (Phideus regime) and 16s (Transkun regime) segments.

Generates human-audible outputs:
  - Original audio clips (WAV)
  - Transkun transcription rendered as audio (WAV via FluidSynth)
  - GT MIDI rendered as audio (WAV via FluidSynth)
  - Piano roll comparison plots (PNG)
  - Per-segment and aggregate metrics (JSON)

All human-facing artifacts go to resultados_compartir/07_gate6_amt/exp0_transkun_baseline/

Usage:
    python experiments/bias_control/gate6/test_transkun_baseline.py \
        --maestro-dir data/maestro_v3/maestro-v3.0.0 \
        --output data/gate6_results/transkun_baseline \
        --device cuda
"""

import argparse
import json
import os
import sys
import time
import subprocess
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import numpy as np
import torch
import pretty_midi
import mir_eval

SHARE_DIR = Path('Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/resultados_compartir/07_gate6_amt/exp0_transkun_baseline')
SOUNDFONT = '/usr/share/sounds/sf2/default-GM.sf2'

# Number of "showcase" segments to render as audio + plots
N_SHOWCASE = 5


# ── Transkun loading ──────────────────────────────────────────────────────

def load_transkun(device: str = 'cuda'):
    """Load pretrained Transkun v2 model."""
    import pkg_resources
    import moduleconf

    weight_path = pkg_resources.resource_filename('transkun', 'pretrained/2.0.pt')
    conf_path = pkg_resources.resource_filename('transkun', 'pretrained/2.0.conf')

    conf_manager = moduleconf.parseFromFile(conf_path)
    TransKun = conf_manager['Model'].module.TransKun
    conf = conf_manager['Model'].config

    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)

    model = TransKun(conf=conf).to(device)
    if 'best_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['best_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    return model


# ── MAESTRO segment sampling ──────────────────────────────────────────────

def load_maestro_metadata(maestro_dir: Path):
    """Load MAESTRO v3 metadata JSON."""
    json_path = maestro_dir / 'maestro-v3.0.0.json'
    if not json_path.exists():
        csv_path = maestro_dir / 'maestro-v3.0.0.csv'
        if csv_path.exists():
            import csv as csv_mod
            with open(csv_path) as f:
                reader = csv_mod.DictReader(f)
                entries = list(reader)
            return entries
        raise FileNotFoundError(f"No metadata found in {maestro_dir}")

    with open(json_path) as f:
        data = json.load(f)

    n = len(data['canonical_composer'])
    entries = []
    for i in range(n):
        entry = {k: v[str(i)] for k, v in data.items()}
        entries.append(entry)
    return entries


def sample_validation_segments(
    maestro_dir: Path,
    n_short: int = 50,
    n_long: int = 50,
    seed: int = 42,
):
    """Sample validation segments for baseline testing."""
    metadata = load_maestro_metadata(maestro_dir)

    val_entries = [e for e in metadata if e.get('split') == 'validation']
    if not val_entries:
        print("WARNING: No validation split found, using all entries")
        val_entries = metadata

    rng = np.random.RandomState(seed)
    rng.shuffle(val_entries)

    segments = []

    for i in range(n_short):
        entry = val_entries[i % len(val_entries)]
        audio_path = maestro_dir / entry['audio_filename']
        midi_path = maestro_dir / entry['midi_filename']
        duration = float(entry.get('duration', 300))
        composer = entry.get('canonical_composer', 'Unknown')
        title = entry.get('canonical_title', 'Unknown')

        max_start = max(0, duration - 4.0)
        start = rng.uniform(0, max_start) if max_start > 0 else 0

        segments.append({
            'audio_path': str(audio_path),
            'midi_path': str(midi_path),
            'start_time': start,
            'duration': 4.0,
            'segment_type': '4s',
            'piece_idx': i % len(val_entries),
            'composer': composer,
            'title': title,
        })

    for i in range(n_long):
        entry = val_entries[i % len(val_entries)]
        audio_path = maestro_dir / entry['audio_filename']
        midi_path = maestro_dir / entry['midi_filename']
        duration = float(entry.get('duration', 300))
        composer = entry.get('canonical_composer', 'Unknown')
        title = entry.get('canonical_title', 'Unknown')

        max_start = max(0, duration - 16.0)
        start = rng.uniform(0, max_start) if max_start > 0 else 0

        segments.append({
            'audio_path': str(audio_path),
            'midi_path': str(midi_path),
            'start_time': start,
            'duration': 16.0,
            'segment_type': '16s',
            'piece_idx': i % len(val_entries),
            'composer': composer,
            'title': title,
        })

    return segments


# ── Audio loading ─────────────────────────────────────────────────────────

def load_audio_segment(
    audio_path: str,
    start_time: float,
    duration: float,
    target_sr: int = 44100,
) -> torch.Tensor:
    """Load audio segment at 44.1kHz for Transkun."""
    import pydub
    import soxr

    audio = pydub.AudioSegment.from_file(audio_path)
    y = np.array(audio.get_array_of_samples(), dtype=np.float32) / 2**15
    y = y.reshape(-1, audio.channels)
    sr = audio.frame_rate

    if sr != target_sr:
        y = soxr.resample(y, sr, target_sr)

    start_sample = int(start_time * target_sr)
    end_sample = start_sample + int(duration * target_sr)

    if end_sample > len(y):
        pad_len = end_sample - len(y)
        y = np.pad(y, ((0, pad_len), (0, 0)), mode='constant')

    segment = y[start_sample:end_sample]
    return torch.from_numpy(segment).float()


# ── MIDI rendering ────────────────────────────────────────────────────────

def transkun_notes_to_pretty_midi(notes, segment_duration: float) -> pretty_midi.PrettyMIDI:
    """Convert Transkun Note objects to a PrettyMIDI object."""
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0, name='Piano')

    for note in notes:
        if note.pitch < 0:
            continue
        onset = max(0, note.start)
        offset = min(note.end, segment_duration)
        if offset <= onset:
            continue
        vel = max(1, min(127, int(note.velocity)))
        piano.notes.append(pretty_midi.Note(
            velocity=vel, pitch=note.pitch,
            start=onset, end=offset,
        ))

    pm.instruments.append(piano)
    return pm


def midi_segment_to_pretty_midi(
    midi_path: str, start_time: float, end_time: float,
) -> pretty_midi.PrettyMIDI:
    """Extract a segment from a MIDI file as PrettyMIDI."""
    src = pretty_midi.PrettyMIDI(str(midi_path))
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0, name='Piano')

    for inst in src.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            if note.end <= start_time or note.start >= end_time:
                continue
            onset = max(note.start, start_time) - start_time
            offset = min(note.end, end_time) - start_time
            if offset <= onset:
                continue
            piano.notes.append(pretty_midi.Note(
                velocity=note.velocity, pitch=note.pitch,
                start=onset, end=offset,
            ))

    pm.instruments.append(piano)
    return pm


def render_midi_to_wav(pm: pretty_midi.PrettyMIDI, output_path: str, sr: int = 44100):
    """Render PrettyMIDI to WAV using FluidSynth."""
    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
        tmp_midi = tmp.name
        pm.write(tmp_midi)

    try:
        subprocess.run([
            'fluidsynth', '-ni', SOUNDFONT, tmp_midi,
            '-F', output_path, '-r', str(sr),
            '-g', '1.0',
        ], check=True, capture_output=True, timeout=30)
    finally:
        Path(tmp_midi).unlink(missing_ok=True)


def save_audio_segment_wav(
    audio_path: str, start_time: float, duration: float,
    output_path: str, sr: int = 44100,
):
    """Save an audio segment as WAV."""
    import soundfile as sf

    audio_tensor = load_audio_segment(audio_path, start_time, duration, sr)
    audio_np = audio_tensor.numpy()
    if audio_np.ndim == 2 and audio_np.shape[1] == 1:
        audio_np = audio_np[:, 0]
    elif audio_np.ndim == 2:
        audio_np = audio_np.mean(axis=1)
    sf.write(output_path, audio_np, sr)


# ── Piano roll plotting ───────────────────────────────────────────────────

def plot_piano_roll_comparison(
    notes_est, notes_ref, duration: float,
    output_path: str, title: str = '',
    metrics: dict = None,
):
    """Plot GT vs estimated piano rolls side by side."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True, sharey=True)

    for ax, notes, label, color in [
        (axes[0], notes_ref, 'Ground Truth', '#2196F3'),
        (axes[1], notes_est, 'Transkun v2', '#FF5722'),
    ]:
        for note in notes:
            if hasattr(note, 'start'):
                onset, offset, pitch = note.start, note.end, note.pitch
            else:
                onset, offset, pitch = note[0], note[1], note[2]
            if pitch < 21 or pitch > 108:
                continue
            rect = Rectangle(
                (onset, pitch - 0.4), offset - onset, 0.8,
                facecolor=color, edgecolor='none', alpha=0.7,
            )
            ax.add_patch(rect)
        ax.set_xlim(0, duration)
        ax.set_ylim(20, 109)
        ax.set_ylabel(label, fontsize=11)
        ax.set_yticks([21, 33, 45, 57, 69, 81, 93, 105])
        ax.set_yticklabels(['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7'])
        ax.grid(axis='y', alpha=0.2)

    axes[1].set_xlabel('Time (s)')

    suptitle = title
    if metrics:
        f1_note = metrics.get('note_onset_F1', 0)
        f1_offset = metrics.get('note_offset_F1', 0)
        f1_vel = metrics.get('note_offset_vel_F1', 0)
        suptitle += f'\nOnset F1={f1_note:.3f}  |  Note+Off F1={f1_offset:.3f}  |  Note+Off+Vel F1={f1_vel:.3f}'

    fig.suptitle(suptitle, fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Main evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_segments(
    model,
    segments,
    device: str = 'cuda',
    output_dir: Path = None,
    share_dir: Path = None,
):
    """Transcribe segments, evaluate, and generate human-facing artifacts."""
    from experiments.bias_control.gate6.evaluation import (
        evaluate_note_level,
        transkun_notes_to_mir_eval,
        midi_to_mir_eval,
        notes_to_piano_roll,
        evaluate_frame_level,
        evaluate_onset_level,
    )

    results_4s = []
    results_16s = []

    # Track best/worst for showcase selection
    all_results_with_notes = []

    for idx, seg in enumerate(segments):
        t0 = time.time()

        # Load audio
        audio = load_audio_segment(
            seg['audio_path'], seg['start_time'], seg['duration'],
            target_sr=model.fs,
        )
        audio = audio.to(device)

        # Transcribe
        notes_est = model.transcribe(audio)

        # Convert estimated notes to mir_eval format
        est_intervals, est_pitches, est_velocities = transkun_notes_to_mir_eval(
            notes_est, segment_start=0.0, segment_end=seg['duration'],
        )

        # Load ground truth MIDI
        ref_intervals, ref_pitches, ref_velocities = midi_to_mir_eval(
            seg['midi_path'],
            start_time=seg['start_time'],
            end_time=seg['start_time'] + seg['duration'],
        )

        # Note-level evaluation
        note_metrics = evaluate_note_level(
            est_intervals, est_pitches, est_velocities,
            ref_intervals, ref_pitches, ref_velocities,
        )

        # Frame-level evaluation
        est_pr = notes_to_piano_roll(notes_est, seg['duration'], hop_sec=0.01)
        ref_notes_for_pr = []
        for i in range(len(ref_intervals)):
            ref_notes_for_pr.append((
                ref_intervals[i, 0], ref_intervals[i, 1],
                int(round(mir_eval.util.hz_to_midi(ref_pitches[i]))),
            ))
        ref_pr = notes_to_piano_roll(ref_notes_for_pr, seg['duration'], hop_sec=0.01)

        frame_metrics = evaluate_frame_level(est_pr, ref_pr)
        onset_metrics = evaluate_onset_level(est_pr, ref_pr)

        elapsed = time.time() - t0

        result = {
            **note_metrics,
            **frame_metrics,
            **onset_metrics,
            'segment_type': seg['segment_type'],
            'piece_idx': seg['piece_idx'],
            'start_time': seg['start_time'],
            'duration': seg['duration'],
            'composer': seg['composer'],
            'title': seg['title'],
            'inference_time': elapsed,
        }

        if seg['segment_type'] == '4s':
            results_4s.append(result)
        else:
            results_16s.append(result)

        all_results_with_notes.append({
            'result': result,
            'notes_est': notes_est,
            'seg': seg,
        })

        # Progress
        if (idx + 1) % 10 == 0 or (idx + 1) == len(segments):
            print(f"  [{idx+1}/{len(segments)}] {seg['segment_type']} "
                  f"onset_F1={note_metrics['note_onset_F1']:.3f} "
                  f"note+off_F1={note_metrics['note_offset_F1']:.3f} "
                  f"frame_F1={frame_metrics['frame_F1']:.3f} "
                  f"({elapsed:.1f}s)")

    # ── Generate showcase artifacts ────────────────────────────────────
    if share_dir is not None:
        print(f"\nGenerating human-facing artifacts...")
        generate_showcase(all_results_with_notes, share_dir)

    return results_4s, results_16s


def generate_showcase(all_results_with_notes, share_dir: Path):
    """Generate audio + plots for best/worst/median segments."""
    import soundfile as sf

    audio_dir = share_dir / 'audio_samples'
    plot_dir = share_dir / 'piano_roll_plots'
    metrics_dir = share_dir / 'metrics'
    audio_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Separate by regime
    for regime in ['4s', '16s']:
        regime_items = [x for x in all_results_with_notes
                        if x['result']['segment_type'] == regime]
        if not regime_items:
            continue

        # Sort by note+offset F1
        regime_items.sort(key=lambda x: x['result']['note_offset_F1'])

        # Pick showcase: worst, median, best + 2 random
        n = len(regime_items)
        showcase_indices = [
            ('worst', 0),
            ('median', n // 2),
            ('best', n - 1),
        ]
        # Add 2 random
        rng = np.random.RandomState(42)
        random_idxs = rng.choice(n, size=min(2, n), replace=False)
        for ri, ridx in enumerate(random_idxs):
            showcase_indices.append((f'random{ri+1}', int(ridx)))

        for label, item_idx in showcase_indices:
            item = regime_items[item_idx]
            seg = item['seg']
            notes_est = item['notes_est']
            result = item['result']

            tag = f"{regime}_{label}"
            composer_short = seg['composer'].split()[-1][:10]
            f1_str = f"{result['note_offset_F1']:.0%}"

            print(f"  {tag}: {composer_short} F1={f1_str}")

            # 1. Save original audio clip
            try:
                save_audio_segment_wav(
                    seg['audio_path'], seg['start_time'], seg['duration'],
                    str(audio_dir / f'{tag}_original.wav'),
                )
            except Exception as e:
                print(f"    WARN: Could not save original audio: {e}")

            # 2. Render GT MIDI as audio
            try:
                gt_pm = midi_segment_to_pretty_midi(
                    seg['midi_path'], seg['start_time'],
                    seg['start_time'] + seg['duration'],
                )
                render_midi_to_wav(gt_pm, str(audio_dir / f'{tag}_gt_midi.wav'))
            except Exception as e:
                print(f"    WARN: Could not render GT MIDI: {e}")

            # 3. Render Transkun transcription as audio
            try:
                est_pm = transkun_notes_to_pretty_midi(notes_est, seg['duration'])
                render_midi_to_wav(est_pm, str(audio_dir / f'{tag}_transkun.wav'))
            except Exception as e:
                print(f"    WARN: Could not render Transkun MIDI: {e}")

            # 4. Piano roll comparison plot
            try:
                gt_notes = []
                src_midi = pretty_midi.PrettyMIDI(str(seg['midi_path']))
                for inst in src_midi.instruments:
                    if inst.is_drum:
                        continue
                    for note in inst.notes:
                        if note.end <= seg['start_time'] or note.start >= seg['start_time'] + seg['duration']:
                            continue
                        gt_notes.append(pretty_midi.Note(
                            velocity=note.velocity, pitch=note.pitch,
                            start=max(note.start - seg['start_time'], 0),
                            end=min(note.end - seg['start_time'], seg['duration']),
                        ))

                title = f"{seg['composer']} — {seg['title'][:40]}\n{regime} segment ({label})"
                plot_piano_roll_comparison(
                    notes_est, gt_notes, seg['duration'],
                    str(plot_dir / f'{tag}_pianoroll.png'),
                    title=title, metrics=result,
                )
            except Exception as e:
                print(f"    WARN: Could not generate plot: {e}")

            # 5. Per-segment metrics JSON
            with open(metrics_dir / f'{tag}_metrics.json', 'w') as f:
                json.dump({
                    'segment': {
                        'composer': seg['composer'],
                        'title': seg['title'],
                        'start_time': seg['start_time'],
                        'duration': seg['duration'],
                        'regime': regime,
                        'rank': label,
                    },
                    'metrics': {k: v for k, v in result.items()
                                if isinstance(v, (int, float))},
                }, f, indent=2)

    # 6. Summary table as text
    write_summary_table(all_results_with_notes, share_dir)


def write_summary_table(all_results_with_notes, share_dir: Path):
    """Write a human-readable summary table."""
    lines = []
    lines.append("# Gate 6 Exp 0 — Transkun v2 Baseline Verification")
    lines.append("")
    lines.append("Transkun v2 (12.9M params) transcribing MAESTRO v3.0.0 validation segments.")
    lines.append("")

    for regime in ['4s', '16s']:
        items = [x['result'] for x in all_results_with_notes
                 if x['result']['segment_type'] == regime]
        if not items:
            continue

        lines.append(f"## {regime} segments (N={len(items)})")
        lines.append("")
        lines.append(f"| Metric | Mean | Std | Min | Max |")
        lines.append(f"|--------|------|-----|-----|-----|")

        for key in ['note_onset_F1', 'note_offset_F1', 'note_offset_vel_F1',
                     'frame_F1', 'onset_F1']:
            vals = [r[key] for r in items if key in r]
            if vals:
                lines.append(f"| {key} | {np.mean(vals):.3f} | {np.std(vals):.3f} | "
                             f"{np.min(vals):.3f} | {np.max(vals):.3f} |")

        times = [r['inference_time'] for r in items]
        lines.append(f"| inference_time (s) | {np.mean(times):.2f} | {np.std(times):.2f} | "
                     f"{np.min(times):.2f} | {np.max(times):.2f} |")
        lines.append("")

    lines.append("## Audio samples")
    lines.append("")
    lines.append("For each regime (4s, 16s), 5 samples: worst, median, best by Note+Offset F1, plus 2 random.")
    lines.append("")
    lines.append("Each sample has 3 WAV files:")
    lines.append("- `*_original.wav` — original MAESTRO recording")
    lines.append("- `*_gt_midi.wav` — ground truth MIDI rendered via FluidSynth")
    lines.append("- `*_transkun.wav` — Transkun transcription rendered via FluidSynth")
    lines.append("")
    lines.append("Compare `_original.wav` vs `_transkun.wav` to hear transcription quality.")
    lines.append("Compare `_gt_midi.wav` vs `_transkun.wav` to isolate MIDI-level differences.")

    with open(share_dir / 'SUMMARY.md', 'w') as f:
        f.write('\n'.join(lines))


# ── Summary printing ──────────────────────────────────────────────────────

def print_summary(results: list, label: str):
    """Print summary statistics for a set of results."""
    if not results:
        print(f"\n{label}: No results")
        return

    keys = [
        'note_onset_F1', 'note_offset_F1', 'note_offset_vel_F1',
        'frame_F1', 'onset_F1',
    ]

    print(f"\n{'='*60}")
    print(f"  {label} (N={len(results)})")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for key in keys:
        vals = [r[key] for r in results if key in r]
        if vals:
            print(f"  {key:<25} {np.mean(vals):>8.3f} {np.std(vals):>8.3f} "
                  f"{np.min(vals):>8.3f} {np.max(vals):>8.3f}")

    times = [r['inference_time'] for r in results]
    print(f"  {'inference_time (s)':<25} {np.mean(times):>8.2f} {np.std(times):>8.2f}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Gate 6 Exp 0: Transkun Baseline')
    parser.add_argument('--maestro-dir', type=str, required=True,
                        help='Path to MAESTRO v3.0.0 directory')
    parser.add_argument('--output', type=str, default='data/gate6_results/transkun_baseline',
                        help='Output directory for JSON results')
    parser.add_argument('--share-dir', type=str, default=str(SHARE_DIR),
                        help='Output directory for human-facing artifacts')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n-short', type=int, default=50,
                        help='Number of 4s segments')
    parser.add_argument('--n-long', type=int, default=50,
                        help='Number of 16s segments')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    maestro_dir = Path(args.maestro_dir)
    output_dir = Path(args.output)
    share_dir = Path(args.share_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    share_dir.mkdir(parents=True, exist_ok=True)

    print(f"Gate 6 — Exp 0: Transkun Baseline Verification")
    print(f"  MAESTRO:    {maestro_dir}")
    print(f"  Device:     {args.device}")
    print(f"  Segments:   {args.n_short}x4s + {args.n_long}x16s")
    print(f"  Results:    {output_dir}")
    print(f"  Share dir:  {share_dir}")

    # Load model
    print("\nLoading Transkun v2...")
    model = load_transkun(args.device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Params: {n_params:.1f}M")

    # Sample segments
    print("\nSampling validation segments...")
    segments = sample_validation_segments(
        maestro_dir, args.n_short, args.n_long, args.seed,
    )
    print(f"  Total: {len(segments)} segments")

    # Evaluate
    print("\nTranscribing and evaluating...")
    results_4s, results_16s = evaluate_segments(
        model, segments, args.device, output_dir,
        share_dir=share_dir,
    )

    # Print summaries
    print_summary(results_4s, "4s segments (Phideus regime)")
    print_summary(results_16s, "16s segments (Transkun regime)")

    # Save full results
    from experiments.bias_control.gate6.evaluation import save_results

    all_results = {
        'segments_4s': results_4s,
        'segments_16s': results_16s,
        'summary_4s': {k: float(np.mean([r[k] for r in results_4s]))
                       for k in ['note_onset_F1', 'note_offset_F1',
                                  'note_offset_vel_F1', 'frame_F1']
                       if results_4s},
        'summary_16s': {k: float(np.mean([r[k] for r in results_16s]))
                        for k in ['note_onset_F1', 'note_offset_F1',
                                   'note_offset_vel_F1', 'frame_F1']
                        if results_16s},
    }

    save_results(
        all_results,
        output_dir / 'exp0_baseline.json',
        experiment='gate6_exp0_transkun_baseline',
        config={
            'n_short': args.n_short,
            'n_long': args.n_long,
            'device': args.device,
        },
        seed=args.seed,
    )

    print(f"\nJSON results:  {output_dir / 'exp0_baseline.json'}")
    print(f"Human artifacts: {share_dir}")
    print(f"  audio_samples/  — {N_SHOWCASE}x2 regimes x 3 WAVs = {N_SHOWCASE*2*3} files")
    print(f"  piano_roll_plots/ — comparison PNGs")
    print(f"  metrics/ — per-segment JSONs")
    print(f"  SUMMARY.md — human-readable table")


if __name__ == '__main__':
    main()
