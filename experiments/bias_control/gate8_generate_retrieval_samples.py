#!/usr/bin/env python3
"""
Generate qualitative retrieval samples for Gate 8 / conditioned projection checkpoints.

Output package mirrors the spirit of Gate 6 sample bundles, but adapted to retrieval:
for selected audio queries, write
  - original query audio
  - ground-truth MIDI segment (.mid + rendered .wav)
  - top-1 retrieved MIDI segment (.mid + rendered .wav)
  - comparison piano-roll plot
  - per-sample metrics JSON
  - package SUMMARY.md

Selection policy:
  - best: strongest exact-match retrieval among sampled queries
  - median: query closest to median GT rank
  - worst: largest GT rank
  - random1/random2: deterministic random leftovers
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.bias_control.gate5b.checkpoint_loader import load_model_from_checkpoint
from experiments.bias_control.gate5b.render_midi_audio import (
    render_midi_to_audio,
    write_notes_to_midi,
)
from experiments.bias_control.evaluate_structured_pool import extract_all_embeddings
from src.bias_control.datasets.maestro_segments import MaestroSegmentDataset, collate_segments
from src.utils.midi_utils import Note, notes_to_piano_roll, parse_midi

LOGGER = logging.getLogger("gate8_samples")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to best_model.pt",
    )
    parser.add_argument(
        "--maestro-dir",
        type=str,
        required=True,
        help="Path to MAESTRO v3.0.0 root",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where the qualitative package will be written",
    )
    parser.add_argument(
        "--segment-len",
        type=float,
        default=4.0,
        help="Segment length in seconds (default: 4.0)",
    )
    parser.add_argument(
        "--hop",
        type=float,
        default=1.0,
        help="Hop in seconds for the evaluation dataset (default: 1.0)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Sample rate for query audio and MIDI rendering (default: 24000)",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=8,
        help="Embedding extraction batch size (default: 8)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers for embedding extraction (default: 4)",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=500,
        help="How many validation queries to score when selecting examples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for query sampling and random examples",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device (default: cuda)",
    )
    parser.add_argument(
        "--soundfont",
        type=str,
        default="/usr/share/sounds/sf2/TimGM6mb.sf2",
        help="SoundFont for MIDI rendering",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="auto",
        choices=["auto", "fluidsynth", "prettymidi"],
        help="Renderer backend",
    )
    return parser.parse_args()


def _piece_label(seg) -> str:
    return f"{seg.canonical_composer} — {seg.canonical_title}"


def _segment_relative_notes(
    piece_notes: Sequence[Note],
    start_time: float,
    segment_seconds: float,
    min_pitch: int = 21,
    max_pitch: int = 108,
) -> List[Tuple[int, float, float, int]]:
    out: List[Tuple[int, float, float, int]] = []
    end_time = start_time + segment_seconds
    for note in piece_notes:
        if note.offset <= start_time or note.onset >= end_time:
            continue
        if note.pitch < min_pitch or note.pitch > max_pitch:
            continue

        onset_rel = max(0.0, note.onset - start_time)
        offset_rel = min(segment_seconds, note.offset - start_time)
        if offset_rel <= onset_rel:
            continue

        out.append((int(note.pitch), float(onset_rel), float(offset_rel), int(note.velocity)))
    return out


def _choose_soundfont(path_str: str) -> Path | None:
    candidates = [
        Path(path_str),
        Path("/usr/share/sounds/sf2/TimGM6mb.sf2"),
        Path("/usr/share/sounds/sf2/FluidR3_GM.sf2"),
        Path("venv/lib/python3.13/site-packages/pretty_midi/TimGM6mb.sf2"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _score_queries(
    audio_embs: torch.Tensor,
    midi_embs: torch.Tensor,
    dataset: MaestroSegmentDataset,
    query_indices: Sequence[int],
) -> List[Dict]:
    audio_norm = F.normalize(audio_embs, dim=1)
    midi_norm = F.normalize(midi_embs, dim=1)
    all_cases: List[Dict] = []

    for query_idx in query_indices:
        seg = dataset.segments[query_idx]
        sims = midi_norm @ audio_norm[query_idx]

        top2_vals, top2_idx = torch.topk(sims, k=min(2, sims.shape[0]))
        top1_idx = int(top2_idx[0].item())
        top1_similarity = float(top2_vals[0].item())
        gt_similarity = float(sims[query_idx].item())
        gt_rank = int((sims > gt_similarity).sum().item()) + 1

        if top1_idx == query_idx and len(top2_vals) > 1:
            margin_vs_next = gt_similarity - float(top2_vals[1].item())
        else:
            margin_vs_next = gt_similarity - top1_similarity

        top1_seg = dataset.segments[top1_idx]
        all_cases.append(
            {
                "query_idx": int(query_idx),
                "query_piece_idx": int(seg.piece_idx),
                "query_segment_idx": int(seg.segment_idx),
                "query_start_time": float(seg.start_time),
                "query_label": _piece_label(seg),
                "top1_idx": top1_idx,
                "top1_piece_idx": int(top1_seg.piece_idx),
                "top1_segment_idx": int(top1_seg.segment_idx),
                "top1_start_time": float(top1_seg.start_time),
                "top1_label": _piece_label(top1_seg),
                "top1_similarity": top1_similarity,
                "gt_similarity": gt_similarity,
                "gt_rank": gt_rank,
                "top1_correct": bool(top1_idx == query_idx),
                "top1_same_piece": bool(top1_seg.piece_idx == seg.piece_idx),
                "top1_same_composer": bool(
                    top1_seg.canonical_composer == seg.canonical_composer
                ),
                "margin_vs_next": float(margin_vs_next),
            }
        )

    return all_cases


def _select_examples(cases: Sequence[Dict], seed: int) -> List[Tuple[str, Dict]]:
    rng = np.random.RandomState(seed)
    used: set[int] = set()

    rank1 = [c for c in cases if c["gt_rank"] == 1]
    if rank1:
        best = max(rank1, key=lambda c: (c["margin_vs_next"], c["gt_similarity"]))
    else:
        best = min(cases, key=lambda c: (c["gt_rank"], -c["gt_similarity"]))
    used.add(best["query_idx"])

    sorted_by_rank = sorted(cases, key=lambda c: (c["gt_rank"], c["top1_similarity"]))
    median_rank = np.median([c["gt_rank"] for c in sorted_by_rank])
    median = min(
        [c for c in sorted_by_rank if c["query_idx"] not in used],
        key=lambda c: (abs(c["gt_rank"] - median_rank), c["query_idx"]),
    )
    used.add(median["query_idx"])

    worst = max(
        [c for c in cases if c["query_idx"] not in used],
        key=lambda c: (c["gt_rank"], c["top1_similarity"] - c["gt_similarity"]),
    )
    used.add(worst["query_idx"])

    leftovers = [c for c in cases if c["query_idx"] not in used]
    rng.shuffle(leftovers)
    random_cases = leftovers[:2]

    selection = [("best", best), ("median", median), ("worst", worst)]
    for i, case in enumerate(random_cases, start=1):
        selection.append((f"random{i}", case))
    return selection


def _plot_pianorolls(
    gt_notes: Sequence[Tuple[int, float, float, int]],
    pred_notes: Sequence[Tuple[int, float, float, int]],
    duration: float,
    out_path: Path,
    title: str,
) -> None:
    gt_roll = notes_to_piano_roll(
        [Note(p, s, e, v) for p, s, e, v in gt_notes],
        duration=duration,
        frame_rate=50.0,
    )
    pred_roll = notes_to_piano_roll(
        [Note(p, s, e, v) for p, s, e, v in pred_notes],
        duration=duration,
        frame_rate=50.0,
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True, sharey=True)
    axes[0].imshow(gt_roll.T, origin="lower", aspect="auto", cmap="magma")
    axes[0].set_title("Ground truth MIDI")
    axes[1].imshow(pred_roll.T, origin="lower", aspect="auto", cmap="viridis")
    axes[1].set_title("Retrieved MIDI (top-1)")
    axes[1].set_xlabel("Frames @ 50 fps")
    for ax in axes:
        ax.set_ylabel("Pitch (A0-C8)")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio_samples"
    midi_dir = output_dir / "midi_samples"
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "piano_roll_plots"
    for path in [audio_dir, midi_dir, metrics_dir, plots_dir]:
        path.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    LOGGER.info("Loading checkpoint: %s", checkpoint_path)
    model, meta = load_model_from_checkpoint(str(checkpoint_path), device=device)

    dataset = MaestroSegmentDataset(
        maestro_dir=args.maestro_dir,
        segment_len=args.segment_len,
        hop=args.hop,
        sample_rate=args.sample_rate,
        split="validation",
    )
    LOGGER.info(
        "Validation dataset: %d segments from %d pieces",
        len(dataset),
        len({seg.piece_idx for seg in dataset.segments}),
    )

    query_count = min(args.n_queries, len(dataset))
    rng = np.random.RandomState(args.seed)
    query_indices = np.sort(rng.choice(len(dataset), size=query_count, replace=False))

    LOGGER.info("Extracting embeddings on %s (batch_size=%d, workers=%d)...",
                device, args.embed_batch_size, args.num_workers)
    audio_embs, midi_embs = extract_all_embeddings(
        model,
        dataset,
        device=device,
        batch_size=args.embed_batch_size,
        num_workers=args.num_workers,
    )

    LOGGER.info("Scoring %d sampled queries against full validation MIDI set...", query_count)
    cases = _score_queries(audio_embs, midi_embs, dataset, query_indices)
    selected = _select_examples(cases, args.seed)

    soundfont = _choose_soundfont(args.soundfont)
    if soundfont is None:
        LOGGER.warning("No soundfont found. Falling back to pretty_midi synth only.")
    else:
        LOGGER.info("Using soundfont: %s", soundfont)

    piece_cache: Dict[str, Sequence[Note]] = {}

    def get_piece_notes(midi_path: Path) -> Sequence[Note]:
        key = str(midi_path)
        if key not in piece_cache:
            piece_cache[key] = parse_midi(midi_path)
        return piece_cache[key]

    renderer_tally: Dict[str, int] = {}
    sample_rows: List[Dict] = []

    for label, case in selected:
        query_seg = dataset.segments[case["query_idx"]]
        top1_seg = dataset.segments[case["top1_idx"]]

        query_audio = dataset._load_audio_segment(query_seg).numpy()
        query_wav_path = audio_dir / f"4s_{label}_original.wav"
        sf.write(query_wav_path, query_audio.astype(np.float32), args.sample_rate)

        gt_notes = _segment_relative_notes(
            get_piece_notes(query_seg.midi_path),
            start_time=query_seg.start_time,
            segment_seconds=args.segment_len,
        )
        pred_notes = _segment_relative_notes(
            get_piece_notes(top1_seg.midi_path),
            start_time=top1_seg.start_time,
            segment_seconds=args.segment_len,
        )

        gt_mid_path = midi_dir / f"4s_{label}_gt.mid"
        pred_mid_path = midi_dir / f"4s_{label}_retrieved.mid"
        write_notes_to_midi(gt_notes, gt_mid_path)
        write_notes_to_midi(pred_notes, pred_mid_path)

        gt_wav_path = audio_dir / f"4s_{label}_gt_midi.wav"
        pred_wav_path = audio_dir / f"4s_{label}_retrieved.wav"
        renderer_gt = render_midi_to_audio(
            gt_mid_path,
            gt_wav_path,
            renderer=args.renderer,
            soundfont=str(soundfont) if soundfont else None,
            sample_rate=args.sample_rate,
        )
        renderer_pred = render_midi_to_audio(
            pred_mid_path,
            pred_wav_path,
            renderer=args.renderer,
            soundfont=str(soundfont) if soundfont else None,
            sample_rate=args.sample_rate,
        )
        renderer_tally[renderer_gt] = renderer_tally.get(renderer_gt, 0) + 1
        renderer_tally[renderer_pred] = renderer_tally.get(renderer_pred, 0) + 1

        plot_title = (
            f"{label.upper()} | query={case['query_label']} seg={case['query_segment_idx']} "
            f"| retrieved={case['top1_label']} seg={case['top1_segment_idx']}"
        )
        plot_path = plots_dir / f"4s_{label}_pianoroll.png"
        _plot_pianorolls(gt_notes, pred_notes, args.segment_len, plot_path, plot_title)

        case_payload = {
            **case,
            "tag": label,
            "query_audio_wav": str(query_wav_path),
            "gt_midi_path": str(gt_mid_path),
            "retrieved_midi_path": str(pred_mid_path),
            "gt_midi_wav": str(gt_wav_path),
            "retrieved_midi_wav": str(pred_wav_path),
            "piano_roll_plot": str(plot_path),
            "renderer_gt": renderer_gt,
            "renderer_retrieved": renderer_pred,
            "gt_note_count": len(gt_notes),
            "retrieved_note_count": len(pred_notes),
            "segment_seconds": args.segment_len,
        }
        sample_rows.append(case_payload)
        with open(metrics_dir / f"4s_{label}_metrics.json", "w") as f:
            json.dump(case_payload, f, indent=2)

    summary = {
        "checkpoint": str(checkpoint_path),
        "meta": meta,
        "selection_seed": args.seed,
        "n_validation_segments": len(dataset),
        "n_queries_scored": query_count,
        "fullset_a2m_top1_accuracy_over_sampled_queries": float(
            np.mean([c["top1_correct"] for c in cases])
        ),
        "median_gt_rank_over_sampled_queries": float(np.median([c["gt_rank"] for c in cases])),
        "renderer_tally": renderer_tally,
        "selected_samples": sample_rows,
    }

    with open(output_dir / "sample_selection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _copy_if_exists(checkpoint_path.parent / "config.json", output_dir / "config.json")
    _copy_if_exists(checkpoint_path.parent / "final_results.json", output_dir / "final_results.json")

    summary_md = output_dir / "SUMMARY.md"
    with open(summary_md, "w") as f:
        f.write("# Gate 8 — Retrieval Samples (`a4r-pcd`)\n\n")
        f.write("Qualitative package for Gate 8 conditioned-projection retrieval.\n\n")
        f.write("## Selection protocol\n\n")
        f.write(
            f"- Validation split: 4s segments, hop={args.hop:.1f}s, sample_rate={args.sample_rate} Hz\n"
        )
        f.write(f"- Query scoring set: {query_count} deterministic queries (seed={args.seed})\n")
        f.write("- Retrieval direction for sample selection: Audio -> MIDI top-1 over full validation set\n")
        f.write("- Selection tags:\n")
        f.write("  - `best`: strongest exact-match retrieval among sampled queries\n")
        f.write("  - `median`: query closest to median GT rank\n")
        f.write("  - `worst`: largest GT rank\n")
        f.write("  - `random1`, `random2`: deterministic random leftovers\n\n")
        f.write("## Summary metrics\n\n")
        f.write(
            f"- Full-set sampled-query top-1 accuracy: {summary['fullset_a2m_top1_accuracy_over_sampled_queries']:.1%}\n"
        )
        f.write(
            f"- Median GT rank across sampled queries: {summary['median_gt_rank_over_sampled_queries']:.1f}\n"
        )
        f.write(f"- Renderer tally: `{renderer_tally}`\n")
        f.write(f"- Gate metric reported in checkpoint metadata: `best_S={meta.get('best_S')}`\n\n")
        f.write("## Package contents\n\n")
        f.write("- `audio_samples/`: original query audio, GT MIDI render, retrieved MIDI render\n")
        f.write("- `midi_samples/`: cropped GT/retrieved segment MIDIs\n")
        f.write("- `metrics/`: per-sample JSON with ranks, similarities and identities\n")
        f.write("- `piano_roll_plots/`: GT vs retrieved MIDI visual comparison\n")
        f.write("- `sample_selection_summary.json`: full selection summary\n")
        f.write("- `config.json`, `final_results.json`: copied from checkpoint directory\n\n")
        f.write("## Selected samples\n\n")
        f.write("| Tag | GT rank | Top-1 correct | Query | Retrieved |\n")
        f.write("|-----|---------|---------------|-------|-----------|\n")
        for row in sample_rows:
            f.write(
                f"| {row['tag']} | {row['gt_rank']} | "
                f"{'yes' if row['top1_correct'] else 'no'} | "
                f"{row['query_label']} (seg {row['query_segment_idx']}) | "
                f"{row['top1_label']} (seg {row['top1_segment_idx']}) |\n"
            )

    LOGGER.info("Done. Package written to %s", output_dir)


if __name__ == "__main__":
    main()
