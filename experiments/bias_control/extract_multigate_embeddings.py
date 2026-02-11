#!/usr/bin/env python3
"""
Extract and cache embeddings from 4 checkpoints (DEC-005 Track 1, Script 2).

Gate 6 Phase 1 diagnostic: loads Gate 2, RB0, RA5, and R1-rescue checkpoints,
extracts audio + MIDI embeddings for all validation segments, and saves them
to a single .npz file for downstream hubness analysis and visualization.

Output: data/bias_control_medium/evaluations/gate6/multigate_embeddings.npz
  Keys: {label}_audio, {label}_midi  (8 arrays, each [N, 256])
  Plus metadata: piece_indices, composers, start_times
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "bias_control"))

from src.bias_control.architectures.cross_modal_model import CrossModalModel
from src.bias_control.datasets.maestro_segments import MaestroSegmentDataset
from evaluate_structured_pool import extract_all_embeddings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded paths (diagnostic script, no argparse needed)
# ---------------------------------------------------------------------------
CHECKPOINTS = {
    "gate2": ROOT / "data/bias_control_medium/training_outputs/gate2/checkpoint_epoch45.pt",
    "RB0":   ROOT / "data/bias_control_medium/training_outputs/gate4_RB0/checkpoint_epoch5_base.pt",
    "RA5":   ROOT / "data/bias_control_medium/training_outputs/gate4_runA/checkpoint_epoch5_base.pt",
    "R1":    ROOT / "data/bias_control_medium/training_outputs/gate4_R1rescue/checkpoint_epoch5_base.pt",
}

MAESTRO_DIR = ROOT / "data/maestro_v3/maestro-v3.0.0"
OUTPUT_PATH = ROOT / "data/bias_control_medium/evaluations/gate6/multigate_embeddings.npz"


def load_model(checkpoint_path: Path, device: torch.device) -> CrossModalModel:
    """Load CrossModalModel from a checkpoint with strict=False validation."""
    model = CrossModalModel(audio_encoder='lite')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(
        checkpoint['model_state_dict'], strict=False
    )
    logger.info(f"  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if unexpected:
        logger.warning(f"  Unexpected keys: {unexpected}")
    assert len(unexpected) == 0, f"Checkpoint has unexpected keys: {unexpected}"
    model = model.to(device)
    model.eval()
    return model


def main():
    t_start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load dataset ONCE (all checkpoints share the same val segments)
    # ------------------------------------------------------------------
    logger.info(f"Loading validation dataset from {MAESTRO_DIR}")
    dataset = MaestroSegmentDataset(
        maestro_dir=MAESTRO_DIR,
        segment_len=4.0,
        hop=1.0,
        split='validation'
    )
    logger.info(f"Validation segments: {len(dataset)}")

    # ------------------------------------------------------------------
    # 2. Build metadata arrays from dataset.segments
    # ------------------------------------------------------------------
    piece_indices = np.array([seg.piece_idx for seg in dataset.segments])
    composers = np.array([seg.canonical_composer for seg in dataset.segments])
    start_times = np.array([seg.start_time for seg in dataset.segments])

    # ------------------------------------------------------------------
    # 3. Extract embeddings for each checkpoint
    # ------------------------------------------------------------------
    results = {}
    for label, ckpt_path in CHECKPOINTS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Extracting: {label}  ({ckpt_path.name})")
        logger.info(f"{'='*60}")

        model = load_model(ckpt_path, device)

        with torch.no_grad():
            audio_embs, midi_embs = extract_all_embeddings(
                model, dataset, device,
                batch_size=64,
                num_workers=8
            )

        # extract_all_embeddings returns torch tensors -> convert to numpy
        results[f"{label}_audio"] = audio_embs.numpy()
        results[f"{label}_midi"] = midi_embs.numpy()
        logger.info(f"  audio: {results[f'{label}_audio'].shape}, "
                     f"midi: {results[f'{label}_midi'].shape}")

        # Free model memory before loading the next one
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 4. Save everything to a single .npz
    # ------------------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        **results,
        piece_indices=piece_indices,
        composers=composers,
        start_times=start_times,
    )

    elapsed = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"Saved: {OUTPUT_PATH}")
    logger.info(f"Arrays: {len(results)} embedding arrays + 3 metadata arrays")
    for key in sorted(results):
        logger.info(f"  {key}: {results[key].shape}")
    logger.info(f"Metadata: piece_indices={piece_indices.shape}, "
                f"composers={composers.shape}, start_times={start_times.shape}")
    logger.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
