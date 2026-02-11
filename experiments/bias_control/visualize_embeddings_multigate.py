#!/usr/bin/env python3
"""Comparative embedding visualizations across 4 checkpoints.

DEC-005 Track 1 — Gate 6 Phase 1, Script 4.

Generates 3 figures comparing Gate 2, RB0, RA5, and R1-rescue embeddings:
  1. UMAP 2x2 grid (audio cyan vs MIDI magenta)
  2. Cross-modal bridges (pair connections colored by distance)
  3. Similarity heatmaps (50-segment cosine sim matrices)

Requires: multigate_embeddings.npz (from Script 2)
Style: dark theme (#0a0a0a), audio=#00e5ff, MIDI=#ff1493

Output: data/bias_control_medium/evaluations/gate6/fig_*.png
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from umap import UMAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
EMB_PATH = ROOT / "data/bias_control_medium/evaluations/gate6/multigate_embeddings.npz"
OUTPUT_DIR = ROOT / "data/bias_control_medium/evaluations/gate6"

LABELS = ["gate2", "RB0", "RA5", "R1"]
LABEL_DISPLAY = {"gate2": "Gate 2 (baseline)", "RB0": "RB0 (control)",
                 "RA5": "RA5 (ratio baseline)", "R1": "R1 (ratio enriched)"}

# Style
DARK_BG = "#0a0a0a"
COLOR_AUDIO = "#00e5ff"
COLOR_MIDI = "#ff1493"
N_SUB = 500  # subsample for UMAP
N_HEAT = 50  # segments for heatmaps
SEED = 42


def setup_dark_axes(ax):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors="white", which="both", labelsize=7)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#333333")


def glow_text(ax, x, y, text, fontsize=10, color="white"):
    ax.text(x, y, text, fontsize=fontsize, color=color,
            transform=ax.transAxes, va="top",
            path_effects=[pe.withStroke(linewidth=3, foreground="#000000")])


def main():
    logger.info(f"Loading embeddings from {EMB_PATH}")
    data = np.load(EMB_PATH, allow_pickle=True)

    # Subsample for UMAP
    rng = np.random.RandomState(SEED)
    n_total = len(data["gate2_audio"])
    idx = rng.choice(n_total, min(N_SUB, n_total), replace=False)
    idx.sort()

    logger.info(f"Total segments: {n_total}, subsampled: {len(idx)}")

    # ---------------------------------------------------------------
    # Figure 1: UMAP 2x2 grid
    # ---------------------------------------------------------------
    logger.info("Computing UMAP (joint embedding for consistent scale)...")

    # Concatenate ALL checkpoints' subsampled embeddings for joint UMAP
    all_embs = []
    all_labels_type = []  # "audio" or "midi"
    all_labels_ckpt = []  # checkpoint label
    for label in LABELS:
        audio = data[f"{label}_audio"][idx]
        midi = data[f"{label}_midi"][idx]
        all_embs.append(audio)
        all_embs.append(midi)
        all_labels_type.extend(["audio"] * len(audio))
        all_labels_type.extend(["midi"] * len(midi))
        all_labels_ckpt.extend([label] * len(audio))
        all_labels_ckpt.extend([label] * len(midi))

    all_embs = np.vstack(all_embs)
    all_labels_type = np.array(all_labels_type)
    all_labels_ckpt = np.array(all_labels_ckpt)

    reducer = UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=SEED)
    umap_coords = reducer.fit_transform(all_embs)
    logger.info(f"UMAP done: {umap_coords.shape}")

    n_sub = len(idx)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("UMAP: Audio vs MIDI Embeddings by Checkpoint",
                 color="white", fontsize=16,
                 path_effects=[pe.withStroke(linewidth=3, foreground="#000000")])

    for ax_idx, (ax, label) in enumerate(zip(axes.flat, LABELS)):
        setup_dark_axes(ax)
        mask = all_labels_ckpt == label

        audio_mask = mask & (all_labels_type == "audio")
        midi_mask = mask & (all_labels_type == "midi")

        ax.scatter(umap_coords[midi_mask, 0], umap_coords[midi_mask, 1],
                   s=4, alpha=0.4, c=COLOR_MIDI, label="MIDI", zorder=2)
        ax.scatter(umap_coords[audio_mask, 0], umap_coords[audio_mask, 1],
                   s=4, alpha=0.4, c=COLOR_AUDIO, label="Audio", zorder=3)

        ax.set_title(LABEL_DISPLAY[label], fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        if ax_idx == 0:
            ax.legend(fontsize=8, loc="upper left",
                      facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white",
                      markerscale=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = OUTPUT_DIR / "fig_umap_multigate.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logger.info(f"Saved: {fig_path}")

    # ---------------------------------------------------------------
    # Figure 2: Cross-modal bridges
    # ---------------------------------------------------------------
    logger.info("Generating cross-modal bridges...")

    # Use fewer points for bridge clarity
    n_bridge = min(200, n_sub)
    bridge_idx = idx[:n_bridge]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Cross-Modal Bridges (Audio-MIDI Pairs)",
                 color="white", fontsize=16,
                 path_effects=[pe.withStroke(linewidth=3, foreground="#000000")])

    for ax_idx, (ax, label) in enumerate(zip(axes.flat, LABELS)):
        setup_dark_axes(ax)

        # Get UMAP coords for this checkpoint's audio and midi
        ckpt_mask = all_labels_ckpt == label
        audio_mask = ckpt_mask & (all_labels_type == "audio")
        midi_mask = ckpt_mask & (all_labels_type == "midi")

        audio_coords = umap_coords[audio_mask][:n_bridge]
        midi_coords = umap_coords[midi_mask][:n_bridge]

        # Compute distances for coloring
        dists = np.linalg.norm(audio_coords - midi_coords, axis=1)
        dists_norm = (dists - dists.min()) / (dists.max() - dists.min() + 1e-12)

        # Draw bridges (lines)
        for i in range(n_bridge):
            color = plt.cm.RdYlGn_r(dists_norm[i])  # red=far, green=close
            ax.plot([audio_coords[i, 0], midi_coords[i, 0]],
                    [audio_coords[i, 1], midi_coords[i, 1]],
                    color=color, alpha=0.3, linewidth=0.5, zorder=1)

        ax.scatter(audio_coords[:, 0], audio_coords[:, 1],
                   s=6, c=COLOR_AUDIO, alpha=0.6, zorder=3)
        ax.scatter(midi_coords[:, 0], midi_coords[:, 1],
                   s=6, c=COLOR_MIDI, alpha=0.6, zorder=3)

        mean_dist = np.mean(dists)
        glow_text(ax, 0.02, 0.98, f"mean dist: {mean_dist:.2f}", fontsize=8)
        ax.set_title(LABEL_DISPLAY[label], fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = OUTPUT_DIR / "fig_bridges_multigate.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logger.info(f"Saved: {fig_path}")

    # ---------------------------------------------------------------
    # Figure 3: Similarity heatmaps
    # ---------------------------------------------------------------
    logger.info("Generating similarity heatmaps...")

    heat_idx = idx[:N_HEAT]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(f"Cosine Similarity [Audio x MIDI] ({N_HEAT} segments)",
                 color="white", fontsize=14,
                 path_effects=[pe.withStroke(linewidth=3, foreground="#000000")])

    for ax, label in zip(axes, LABELS):
        setup_dark_axes(ax)

        audio = data[f"{label}_audio"][heat_idx]
        midi = data[f"{label}_midi"][heat_idx]

        # Normalize
        audio_n = audio / (np.linalg.norm(audio, axis=1, keepdims=True) + 1e-12)
        midi_n = midi / (np.linalg.norm(midi, axis=1, keepdims=True) + 1e-12)
        sim = audio_n @ midi_n.T

        im = ax.imshow(sim, cmap="magma", vmin=0, vmax=1, aspect="equal")
        ax.set_title(LABEL_DISPLAY[label], fontsize=10)
        ax.set_xlabel("MIDI seg", fontsize=8)
        ax.set_ylabel("Audio seg", fontsize=8)

        # Diagonal brightness metric
        diag_mean = np.mean(np.diag(sim))
        off_diag = sim[~np.eye(N_HEAT, dtype=bool)].mean()
        glow_text(ax, 0.02, 0.98, f"diag={diag_mean:.3f}\noff={off_diag:.3f}",
                  fontsize=7)

    # Colorbar
    fig.colorbar(im, ax=axes, shrink=0.8, label="cosine similarity",
                 pad=0.02)
    # Make colorbar label white
    fig.axes[-1].yaxis.label.set_color("white")
    fig.axes[-1].tick_params(colors="white")

    plt.tight_layout(rect=[0, 0, 0.92, 0.93])
    fig_path = OUTPUT_DIR / "fig_heatmaps_multigate.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logger.info(f"Saved: {fig_path}")

    logger.info("\nAll visualizations complete.")


if __name__ == "__main__":
    main()
