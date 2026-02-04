"""
Bias Control: Cross-Modal Learning with Domain Adversarial Training

This module implements a cross-modal retrieval system between Audio and MIDI
using learned embeddings with soft matching, abandoning exact hash matching.

Architecture:
    Audio → MERT (frozen) → Projection → Embedding (256d)
    MIDI  → Transformer   → Projection → Embedding (256d)

    Loss: VICReg + DANN (optional)

Gates:
    0: Data Integrity - Verify alignment
    1: Intra-Modal Baselines - Audio→Audio, MIDI→MIDI
    2: Foundation - VICReg cross-modal baseline
    2.5: Embedding Analysis - t-SNE/UMAP diagnostics
    3: DANN - Domain adversarial training
    4: Ratio Auxiliary - Multi-view with ratios
"""

from .architectures.cross_modal_model import CrossModalModel
from .datasets.maestro_segments import MaestroSegmentDataset

__version__ = "1.0.0"
__all__ = ["CrossModalModel", "MaestroSegmentDataset"]
