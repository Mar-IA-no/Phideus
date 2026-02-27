#!/usr/bin/env python3
"""Inference sweep for Test11 event decoders (midi2events / audio2events)."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.bias_control.gate5b.event_decoder_model import (
    ConditionedEventTransformerDecoder,
    EventDecoderConfig,
)
from experiments.bias_control.gate5b.midi_event_codec import MidiEventCodec, MidiEventCodecConfig
from experiments.bias_control.gate5b.render_midi_audio import render_midi_to_audio, tokens_to_midi


CODEC_CFG = MidiEventCodecConfig(
    step_ms=20,
    segment_seconds=4.0,
    min_pitch=21,
    max_pitch=108,
    velocity_bins=32,
    max_seq_len=512,
)

MODEL_CFG = EventDecoderConfig(
    z_dim=256,
    d_model=512,
    n_heads=8,
    n_layers=8,
    d_ff=2048,
    dropout=0.1,
    n_cond_tokens=16,
    max_seq_len=CODEC_CFG.max_seq_len,
    vocab_size=512,  # replaced at runtime
    pad_id=0,
)

SWEEP_CONFIGS_BROAD: List[Dict[str, Any]] = [
    {"name": "deterministic", "deterministic": True, "temperature": 1.0, "top_k": 0, "top_p": 1.0},
    {"name": "t055_k04_p080", "deterministic": False, "temperature": 0.55, "top_k": 4, "top_p": 0.80},
    {"name": "t060_k08_p085", "deterministic": False, "temperature": 0.60, "top_k": 8, "top_p": 0.85},
    {"name": "t065_k12_p088", "deterministic": False, "temperature": 0.65, "top_k": 12, "top_p": 0.88},
    {"name": "t070_k12_p090", "deterministic": False, "temperature": 0.70, "top_k": 12, "top_p": 0.90},
    {"name": "t075_k16_p092", "deterministic": False, "temperature": 0.75, "top_k": 16, "top_p": 0.92},
    {"name": "t085_k24_p095", "deterministic": False, "temperature": 0.85, "top_k": 24, "top_p": 0.95},
]

SWEEP_CONFIGS_D0_PERCEPTUAL_FINE_V1: List[Dict[str, Any]] = [
    {"name": "t078_k16_p092", "deterministic": False, "temperature": 0.78, "top_k": 16, "top_p": 0.92},
    {"name": "t082_k20_p093", "deterministic": False, "temperature": 0.82, "top_k": 20, "top_p": 0.93},
    {"name": "t085_k24_p095", "deterministic": False, "temperature": 0.85, "top_k": 24, "top_p": 0.95},
    {"name": "t088_k24_p095", "deterministic": False, "temperature": 0.88, "top_k": 24, "top_p": 0.95},
    {"name": "t090_k28_p096", "deterministic": False, "temperature": 0.90, "top_k": 28, "top_p": 0.96},
    {"name": "t092_k32_p097", "deterministic": False, "temperature": 0.92, "top_k": 32, "top_p": 0.97},
    {"name": "t095_k32_p098", "deterministic": False, "temperature": 0.95, "top_k": 32, "top_p": 0.98},
    {"name": "t100_k40_p099", "deterministic": False, "temperature": 1.00, "top_k": 40, "top_p": 0.99},
]

SWEEP_CONFIGS_D0_PERCEPTUAL_FINE_V2: List[Dict[str, Any]] = [
    {"name": "t090_k28_p096", "deterministic": False, "temperature": 0.90, "top_k": 28, "top_p": 0.96},
    {"name": "t092_k32_p097", "deterministic": False, "temperature": 0.92, "top_k": 32, "top_p": 0.97},
    {"name": "t094_k32_p097", "deterministic": False, "temperature": 0.94, "top_k": 32, "top_p": 0.97},
    {"name": "t096_k36_p098", "deterministic": False, "temperature": 0.96, "top_k": 36, "top_p": 0.98},
    {"name": "t098_k36_p098", "deterministic": False, "temperature": 0.98, "top_k": 36, "top_p": 0.98},
    {"name": "t100_k40_p099", "deterministic": False, "temperature": 1.00, "top_k": 40, "top_p": 0.99},
    {"name": "t102_k40_p099", "deterministic": False, "temperature": 1.02, "top_k": 40, "top_p": 0.99},
    {"name": "t104_k44_p099", "deterministic": False, "temperature": 1.04, "top_k": 44, "top_p": 0.99},
    {"name": "t100_k48_p995", "deterministic": False, "temperature": 1.00, "top_k": 48, "top_p": 0.995},
    {"name": "t098_k48_p995", "deterministic": False, "temperature": 0.98, "top_k": 48, "top_p": 0.995},
]

SWEEP_CONFIGS_AUDIO_FOCUS_A4R: List[Dict[str, Any]] = [
    {"name": "deterministic", "deterministic": True, "temperature": 1.0, "top_k": 0, "top_p": 1.0},
    {"name": "t070_k08_p080", "deterministic": False, "temperature": 0.70, "top_k": 8, "top_p": 0.80},
    {"name": "t075_k12_p085", "deterministic": False, "temperature": 0.75, "top_k": 12, "top_p": 0.85},
    {"name": "t080_k16_p088", "deterministic": False, "temperature": 0.80, "top_k": 16, "top_p": 0.88},
    {"name": "t085_k24_p092", "deterministic": False, "temperature": 0.85, "top_k": 24, "top_p": 0.92},
    {"name": "t090_k48_p095", "deterministic": False, "temperature": 0.90, "top_k": 48, "top_p": 0.95},
    {"name": "t095_k48_p097", "deterministic": False, "temperature": 0.95, "top_k": 48, "top_p": 0.97},
    {"name": "t100_k64_p098", "deterministic": False, "temperature": 1.00, "top_k": 64, "top_p": 0.98},
]

SWEEP_PRESETS: Dict[str, List[Dict[str, Any]]] = {
    "broad": SWEEP_CONFIGS_BROAD,
    "d0_perceptual_fine_v1": SWEEP_CONFIGS_D0_PERCEPTUAL_FINE_V1,
    "d0_perceptual_fine_v2": SWEEP_CONFIGS_D0_PERCEPTUAL_FINE_V2,
    "audio_focus_a4r": SWEEP_CONFIGS_AUDIO_FOCUS_A4R,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _build_decoder(codec: MidiEventCodec, device: torch.device) -> ConditionedEventTransformerDecoder:
    cfg = EventDecoderConfig(**{**asdict(MODEL_CFG), "vocab_size": codec.vocab_size, "pad_id": codec.pad_id})
    return ConditionedEventTransformerDecoder(config=cfg).to(device)


def _notes_to_roll(
    notes: Sequence[Tuple[int, float, float, int]],
    n_frames: int = 188,
    sr: int = 24000,
    hop: int = 512,
) -> np.ndarray:
    pr = np.zeros((n_frames, 88), dtype=np.float32)
    for pitch, onset, offset, velocity in notes:
        if pitch < 21 or pitch > 108:
            continue
        start = int(np.floor(onset * sr / hop))
        end = int(np.ceil(offset * sr / hop))
        start = max(0, min(start, n_frames - 1))
        end = max(start + 1, min(end, n_frames))
        pr[start:end, pitch - 21] = max(pr[start:end, pitch - 21].max(), velocity / 127.0)
    return pr


def _frame_f1(pred_roll: np.ndarray, true_roll: np.ndarray, thr: float = 0.1) -> Tuple[float, float, float]:
    pred = pred_roll > thr
    truth = true_roll > thr
    tp = float(np.logical_and(pred, truth).sum())
    fp = float(np.logical_and(pred, np.logical_not(truth)).sum())
    fn = float(np.logical_and(np.logical_not(pred), truth).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return f1, precision, recall


def _loop_flags(notes: Sequence[Tuple[int, float, float, int]]) -> Tuple[bool, int]:
    if not notes:
        return False, 0
    pitches = [n[0] for n in notes]
    unique_pitches = len(set(pitches))
    return (len(pitches) >= 40 and unique_pitches <= 2), unique_pitches


def _resolve_device(device: str) -> torch.device:
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. Aborting instead of silently falling back to CPU."
        )
    return requested


@torch.no_grad()
def run_config(
    decoder: ConditionedEventTransformerDecoder,
    codec: MidiEventCodec,
    z_val: np.ndarray,
    tok_val: np.ndarray,
    out_dir: Path,
    cfg: Dict[str, Any],
    n_samples: int,
    n_eval: int,
    seed: int,
    renderer: str,
    soundfont: str | None,
    device: torch.device,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    dev = device
    rng = np.random.RandomState(seed)
    sample_idxs = np.sort(rng.choice(len(z_val), size=min(n_samples, len(z_val)), replace=False))
    eval_rng = np.random.RandomState(seed + 100)
    eval_idxs = eval_rng.choice(len(z_val), size=min(n_eval, len(z_val)), replace=False)

    loop_count = 0
    uniq_pitch_vals: List[int] = []
    f1_vals: List[float] = []
    p_vals: List[float] = []
    r_vals: List[float] = []
    tok_len_vals: List[int] = []

    renderer_stats = {"fluidsynth": 0, "prettymidi": 0, "failed": 0}

    # Generate review samples with files
    for out_i, idx in enumerate(sample_idxs):
        z_i = torch.from_numpy(z_val[idx: idx + 1]).float().to(dev)
        gen_tokens = decoder.generate(
            z=z_i,
            bos_id=codec.bos_id,
            eos_id=codec.eos_id,
            max_len=codec.cfg.max_seq_len,
            temperature=float(cfg["temperature"]),
            top_k=int(cfg["top_k"]),
            top_p=float(cfg["top_p"]),
            deterministic=bool(cfg["deterministic"]),
        )[0].cpu().numpy()

        pred_notes = codec.decode_tokens(gen_tokens)
        true_notes = codec.decode_tokens(tok_val[idx])

        loop_flag, uniq = _loop_flags(pred_notes)
        loop_count += int(loop_flag)
        uniq_pitch_vals.append(uniq)
        tok_len_vals.append(int(len(gen_tokens)))

        pred_roll = _notes_to_roll(pred_notes)
        true_roll = _notes_to_roll(true_notes)
        f1, p, r = _frame_f1(pred_roll, true_roll)
        f1_vals.append(f1)
        p_vals.append(p)
        r_vals.append(r)

        pred_mid = samples_dir / f"pred_{out_i:02d}.mid"
        truth_mid = samples_dir / f"truth_{out_i:02d}.mid"
        pred_wav = samples_dir / f"pred_{out_i:02d}.wav"
        truth_wav = samples_dir / f"truth_{out_i:02d}.wav"

        tokens_to_midi(codec, gen_tokens, pred_mid)
        tokens_to_midi(codec, tok_val[idx], truth_mid)
        used_pred = render_midi_to_audio(pred_mid, pred_wav, renderer, soundfont, sample_rate=24000)
        used_truth = render_midi_to_audio(truth_mid, truth_wav, renderer, soundfont, sample_rate=24000)
        renderer_stats[used_pred] = renderer_stats.get(used_pred, 0) + 1
        renderer_stats[used_truth] = renderer_stats.get(used_truth, 0) + 1

    # Evaluate metrics on separate subset (no files)
    for idx in eval_idxs:
        z_i = torch.from_numpy(z_val[idx: idx + 1]).float().to(dev)
        gen_tokens = decoder.generate(
            z=z_i,
            bos_id=codec.bos_id,
            eos_id=codec.eos_id,
            max_len=codec.cfg.max_seq_len,
            temperature=float(cfg["temperature"]),
            top_k=int(cfg["top_k"]),
            top_p=float(cfg["top_p"]),
            deterministic=bool(cfg["deterministic"]),
        )[0].cpu().numpy()
        pred_notes = codec.decode_tokens(gen_tokens)
        true_notes = codec.decode_tokens(tok_val[idx])
        pred_roll = _notes_to_roll(pred_notes)
        true_roll = _notes_to_roll(true_notes)
        f1, p, r = _frame_f1(pred_roll, true_roll)
        f1_vals.append(f1)
        p_vals.append(p)
        r_vals.append(r)

    metrics = {
        "config": cfg,
        "n_samples_written": int(len(sample_idxs)),
        "n_eval": int(len(eval_idxs)),
        "frame_f1_mean": float(np.mean(f1_vals)) if f1_vals else float("nan"),
        "frame_precision_mean": float(np.mean(p_vals)) if p_vals else float("nan"),
        "frame_recall_mean": float(np.mean(r_vals)) if r_vals else float("nan"),
        "loop_fraction_samples": float(loop_count / max(1, len(sample_idxs))),
        "unique_pitch_mean_samples": float(np.mean(uniq_pitch_vals)) if uniq_pitch_vals else float("nan"),
        "token_len_mean_samples": float(np.mean(tok_len_vals)) if tok_len_vals else float("nan"),
        "renderer_usage": renderer_stats,
        "samples_dir": str(samples_dir),
    }

    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def run_arm(
    arm: str,
    task: str,
    root_out: Path,
    sweep_configs: Sequence[Dict[str, Any]],
    n_samples: int,
    n_eval: int,
    seed: int,
    renderer: str,
    soundfont: str | None,
    device: str,
) -> None:
    arm_dir = root_out / arm
    arm_dir.mkdir(parents=True, exist_ok=True)

    out_model_dir = Path("data/gate5b_results") / arm / "test11_perceptual_models"
    ckpt_path = out_model_dir / f"{task}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for {arm}: {ckpt_path}")

    val_emb_path = Path("data/gate5b_results") / arm / "embeddings_normal.npz"
    tok_val_path = Path("data/gate5b_results") / "targets_event_val.npz"
    if not val_emb_path.exists():
        raise FileNotFoundError(f"Missing val embeddings for {arm}: {val_emb_path}")
    if not tok_val_path.exists():
        raise FileNotFoundError(f"Missing val targets: {tok_val_path}")

    emb_key = "midi_embs" if task == "midi2events" else "audio_embs"
    z_val = np.load(val_emb_path)[emb_key].astype(np.float32)
    tok_val = np.load(tok_val_path)["data"].astype(np.int32)

    codec = MidiEventCodec(CODEC_CFG)
    dev = _resolve_device(device)
    decoder = _build_decoder(codec, dev)
    ckpt = torch.load(ckpt_path, map_location=dev)
    decoder.load_state_dict(ckpt["state_dict"])
    decoder.eval()

    summary_rows: List[Dict[str, Any]] = []
    for idx, cfg in enumerate(sweep_configs):
        cfg_dir = arm_dir / "configs" / f"{idx:02d}_{cfg['name']}"
        logger.info("[%s][%s] config %d/%d -> %s", arm, task, idx + 1, len(sweep_configs), cfg["name"])
        metrics = run_config(
            decoder=decoder,
            codec=codec,
            z_val=z_val,
            tok_val=tok_val,
            out_dir=cfg_dir,
            cfg=cfg,
            n_samples=n_samples,
            n_eval=n_eval,
            seed=seed,
            renderer=renderer,
            soundfont=soundfont,
            device=device,
        )
        summary_rows.append({
            "config_id": f"{idx:02d}_{cfg['name']}",
            **metrics,
        })
        logger.info(
            "[%s][%s] done %s | f1=%.4f loop=%.3f tok_len=%.1f",
            arm,
            task,
            cfg["name"],
            metrics["frame_f1_mean"],
            metrics["loop_fraction_samples"],
            metrics["token_len_mean_samples"],
        )

    summary_rows_sorted = sorted(
        summary_rows,
        key=lambda r: (r["frame_f1_mean"], -r["loop_fraction_samples"]),
        reverse=True,
    )
    with (arm_dir / "summary_sorted.json").open("w") as f:
        json.dump(summary_rows_sorted, f, indent=2)

    best = summary_rows_sorted[0] if summary_rows_sorted else {}
    with (arm_dir / "best_config.txt").open("w") as f:
        f.write(json.dumps(best, indent=2))
    logger.info("[%s][%s] best=%s f1=%.4f", arm, task, best.get("config_id"), float(best.get("frame_f1_mean", float("nan"))))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference sweep for Test11 event decoders")
    p.add_argument("--task", choices=["midi2events", "audio2events"], default="midi2events")
    p.add_argument("--arms", nargs="+", default=["D0", "a4r"], choices=["D0", "a4r", "d4a4"])
    p.add_argument(
        "--config-preset",
        choices=["broad", "d0_perceptual_fine_v1", "d0_perceptual_fine_v2", "audio_focus_a4r"],
        default="broad",
    )
    p.add_argument("--output-dir", default="")
    p.add_argument("--n-samples", type=int, default=6)
    p.add_argument("--n-eval", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--renderer", choices=["auto", "fluidsynth", "prettymidi"], default="auto")
    p.add_argument("--soundfont", type=str, default="/usr/share/sounds/sf2/default-GM.sf2")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.strip() or f"data/gate5b_results/test11_{args.task}_inference_sweep"
    root_out = Path(out_dir)
    root_out.mkdir(parents=True, exist_ok=True)
    sweep_configs = SWEEP_PRESETS[args.config_preset]

    for arm in args.arms:
        run_arm(
            arm=arm,
            task=args.task,
            root_out=root_out,
            sweep_configs=sweep_configs,
            n_samples=args.n_samples,
            n_eval=args.n_eval,
            seed=args.seed,
            renderer=args.renderer,
            soundfont=args.soundfont,
            device=args.device,
        )


if __name__ == "__main__":
    main()
