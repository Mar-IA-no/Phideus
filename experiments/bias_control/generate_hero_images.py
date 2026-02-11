#!/usr/bin/env python3
"""Generate 3 hero composite images for DEC-005 visual curation.

Creates:
  HERO_01_drift_frozen.png       — drift table + UMAP Gate2 vs R1
  HERO_02_bridges_separation.png — bridges + similarity distributions
  HERO_03_prered_nogo.png        — ROC + scatter with NO-GO verdict

Uses existing figures and JSON data to compose new visual artifacts.
Style: dark theme bg=#0a0a0a, audio=#00e5ff, MIDI=#ff1493, glow text.

Output: Documents/BIAS_CONTROL/RESULTADOS_CURADOS/EXHIBIT/
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
GATE6_DIR = ROOT / "data/bias_control_medium/evaluations/gate6"
GATE42_DIR = ROOT / "data/bias_control_medium/evaluations/gate42"
OUTPUT_DIR = ROOT / "Documents/BIAS_CONTROL/04_DIAGNOSTICO_GATE_6_Y_GATE_4_2/CURADURIA_VISUAL"

# Style
DARK_BG = "#0a0a0a"
COLOR_AUDIO = "#00e5ff"
COLOR_MIDI = "#ff1493"
COLOR_CORRECT = "#00ff88"
COLOR_INCORRECT = "#ff4444"
COLOR_NOGO = "#ff3333"
COLOR_GO = "#33ff33"
GLOW = [pe.withStroke(linewidth=3, foreground="#000000")]


def glow_text(ax, x, y, text, fontsize=12, color="white", ha="left", va="top",
              transform=None, weight="normal"):
    """Text with dark glow for readability on dark background."""
    if transform is None:
        transform = ax.transAxes
    ax.text(x, y, text, fontsize=fontsize, color=color, ha=ha, va=va,
            transform=transform, weight=weight,
            path_effects=[pe.withStroke(linewidth=3, foreground="#000000")])


def setup_dark_axes(ax, hide_ticks=False):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors="white", which="both", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#333333")
    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])


def generate_hero_01():
    """HERO_01: Audio Encoder Frozen — Drift table + UMAP Gate2 vs R1."""
    logger.info("Generating HERO_01: Drift + UMAP...")

    with open(GATE6_DIR / "layer_drift.json") as f:
        drift_data = json.load(f)

    fig = plt.figure(figsize=(20, 10))
    fig.patch.set_facecolor(DARK_BG)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.05)

    # --- LEFT: Drift table ---
    ax_table = fig.add_subplot(gs[0])
    ax_table.set_facecolor(DARK_BG)
    ax_table.axis("off")

    modules = [
        ("Audio CNN", "Audio CNN"),
        ("Audio Transformer", "Audio Transformer"),
        ("Audio PosEmb", "Audio PosEmb"),
        ("Audio Projection", "Audio Projection"),
        ("MIDI Embedding", "MIDI Embedding"),
        ("MIDI Transformer", "MIDI Transformer"),
        ("MIDI Projection", "MIDI Projection"),
        ("MIDI OutputNorm", "MIDI OutputNorm"),
    ]

    checkpoints = ["RB0", "RA5", "R1"]

    # Title
    glow_text(ax_table, 0.5, 0.97, "Layer Drift vs Gate 2 Baseline",
              fontsize=16, ha="center", weight="bold")
    glow_text(ax_table, 0.5, 0.93, "(relative change in weights after fine-tuning)",
              fontsize=10, ha="center", color="#888888")

    # Table header
    y_start = 0.85
    y_step = 0.075
    col_x = [0.05, 0.45, 0.62, 0.79]

    glow_text(ax_table, col_x[0], y_start, "Module", fontsize=11, weight="bold",
              color="#aaaaaa")
    for i, ckpt in enumerate(checkpoints):
        glow_text(ax_table, col_x[i + 1], y_start, ckpt, fontsize=11, weight="bold",
                  color="#aaaaaa", ha="center")

    # Horizontal line
    ax_table.plot([0.03, 0.92], [y_start - 0.025, y_start - 0.025],
                  color="#333333", linewidth=1, transform=ax_table.transAxes)

    for row_idx, (module_key, display_name) in enumerate(modules):
        y = y_start - (row_idx + 1) * y_step

        # Module name
        is_audio = display_name.startswith("Audio")
        name_color = COLOR_AUDIO if is_audio else COLOR_MIDI
        glow_text(ax_table, col_x[0], y, display_name, fontsize=10, color=name_color)

        for col_idx, ckpt in enumerate(checkpoints):
            if module_key in drift_data[ckpt]["summary"]:
                val = drift_data[ckpt]["summary"][module_key]["rel_change_mean"]
                pct = val * 100

                # Color coding: green=0%, yellow=low, red=high
                if pct < 0.01:
                    cell_color = "#00ff88"
                    txt = "0.0%"
                elif pct < 5:
                    cell_color = "#88ff00"
                    txt = f"{pct:.1f}%"
                elif pct < 15:
                    cell_color = "#ffaa00"
                    txt = f"{pct:.1f}%"
                else:
                    cell_color = "#ff4444"
                    txt = f"{pct:.1f}%"

                glow_text(ax_table, col_x[col_idx + 1], y, txt,
                          fontsize=11, color=cell_color, ha="center", weight="bold")

    # Key insight box
    box_y = y_start - (len(modules) + 1.5) * y_step
    ax_table.add_patch(plt.Rectangle((0.03, box_y - 0.03), 0.89, 0.07,
                                      transform=ax_table.transAxes,
                                      facecolor="#1a0a0a", edgecolor=COLOR_NOGO,
                                      linewidth=1.5, alpha=0.8))
    glow_text(ax_table, 0.5, box_y + 0.02,
              "59.7M audio parameters: ZERO change across all fine-tuned models",
              fontsize=11, ha="center", color=COLOR_NOGO, weight="bold")

    # --- RIGHT: UMAP Gate2 vs R1 (load existing figure and crop) ---
    ax_umap = fig.add_subplot(gs[1])
    ax_umap.set_facecolor(DARK_BG)
    ax_umap.axis("off")

    # Load the 2x2 UMAP figure
    umap_img = mpimg.imread(str(GATE6_DIR / "fig_umap_multigate.png"))
    h, w = umap_img.shape[:2]

    # Create 2 sub-panels: Gate2 (top-left) and R1 (bottom-right) from the 2x2 grid
    gs_inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1],
                                                 hspace=0.08)

    # Gate 2 panel (top-left quadrant of original)
    ax_g2 = fig.add_subplot(gs_inner[0])
    ax_g2.set_facecolor(DARK_BG)
    ax_g2.imshow(umap_img[:h//2, :w//2])
    ax_g2.axis("off")
    glow_text(ax_g2, 0.02, 0.95, "Gate 2 (baseline)", fontsize=12,
              color=COLOR_GO, weight="bold")

    # R1 panel (bottom-right quadrant of original)
    ax_r1 = fig.add_subplot(gs_inner[1])
    ax_r1.set_facecolor(DARK_BG)
    ax_r1.imshow(umap_img[h//2:, w//2:])
    ax_r1.axis("off")
    glow_text(ax_r1, 0.02, 0.95, "R1 (ratio enriched)", fontsize=12,
              color=COLOR_NOGO, weight="bold")

    # Caption at bottom
    fig.text(0.5, 0.02,
             "Audio encoder frozen. Fine-tuning only moves MIDI side.",
             fontsize=14, color="white", ha="center", weight="bold",
             path_effects=[pe.withStroke(linewidth=4, foreground="#000000")])

    fig.text(0.5, -0.01,
             "Source: layer_drift.json, fig_umap_multigate.png | DEC-005 Gate 6 Phase 1",
             fontsize=8, color="#555555", ha="center")

    output = OUTPUT_DIR / "HERO_01_drift_frozen.png"
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor=DARK_BG,
                pad_inches=0.3)
    plt.close(fig)
    logger.info(f"Saved: {output}")


def generate_hero_02():
    """HERO_02: Bridges + Separation — Gate2 vs R1 comparison."""
    logger.info("Generating HERO_02: Bridges + Separation...")

    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor(DARK_BG)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.15)

    # --- TOP: Bridges Gate2 vs R1 ---
    bridges_img = mpimg.imread(str(GATE6_DIR / "fig_bridges_multigate.png"))
    h, w = bridges_img.shape[:2]

    gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0],
                                               wspace=0.05)

    # Gate 2 bridges (top-left quadrant)
    ax_b_g2 = fig.add_subplot(gs_top[0])
    ax_b_g2.set_facecolor(DARK_BG)
    ax_b_g2.imshow(bridges_img[:h//2, :w//2])
    ax_b_g2.axis("off")
    glow_text(ax_b_g2, 0.5, 0.98, "Gate 2 — Bridges (mean: 3.27)",
              fontsize=12, ha="center", weight="bold", color=COLOR_GO)

    # R1 bridges (bottom-right quadrant)
    ax_b_r1 = fig.add_subplot(gs_top[1])
    ax_b_r1.set_facecolor(DARK_BG)
    ax_b_r1.imshow(bridges_img[h//2:, w//2:])
    ax_b_r1.axis("off")
    glow_text(ax_b_r1, 0.5, 0.98, "R1 — Bridges (mean: 4.68, +43%)",
              fontsize=12, ha="center", weight="bold", color=COLOR_NOGO)

    # --- BOTTOM: Similarity distributions Gate2 vs R1 ---
    sim_img = mpimg.imread(str(GATE6_DIR / "fig_similarity_distributions.png"))
    h2, w2 = sim_img.shape[:2]

    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1],
                                               wspace=0.05)

    # Gate 2 similarity (leftmost panels)
    ax_s_g2 = fig.add_subplot(gs_bot[0])
    ax_s_g2.set_facecolor(DARK_BG)
    ax_s_g2.imshow(sim_img[:, :w2//4])
    ax_s_g2.axis("off")
    glow_text(ax_s_g2, 0.5, 0.98, "Gate 2 — Separation: 0.479",
              fontsize=12, ha="center", weight="bold", color=COLOR_GO)

    # R1 similarity (rightmost panels)
    ax_s_r1 = fig.add_subplot(gs_bot[1])
    ax_s_r1.set_facecolor(DARK_BG)
    ax_s_r1.imshow(sim_img[:, 3*w2//4:])
    ax_s_r1.axis("off")
    glow_text(ax_s_r1, 0.5, 0.98, "R1 — Separation: 0.395 (-18%)",
              fontsize=12, ha="center", weight="bold", color=COLOR_NOGO)

    # Caption
    fig.text(0.5, 0.01,
             "Fine-tuning lengthens cross-modal bridges and degrades separation.",
             fontsize=14, color="white", ha="center", weight="bold",
             path_effects=[pe.withStroke(linewidth=4, foreground="#000000")])

    fig.text(0.5, -0.02,
             "Source: hubness_analysis.json, fig_bridges_multigate.png, "
             "fig_similarity_distributions.png | DEC-005 Gate 6",
             fontsize=8, color="#555555", ha="center")

    output = OUTPUT_DIR / "HERO_02_bridges_separation.png"
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor=DARK_BG,
                pad_inches=0.3)
    plt.close(fig)
    logger.info(f"Saved: {output}")


def generate_hero_03():
    """HERO_03: Pre-Red NO-GO — ROC + Scatter with verdict."""
    logger.info("Generating HERO_03: Pre-Red NO-GO...")

    with open(GATE42_DIR / "h426_prered_results.json") as f:
        prered = json.load(f)

    fig = plt.figure(figsize=(20, 10))
    fig.patch.set_facecolor(DARK_BG)
    gs = gridspec.GridSpec(1, 2, wspace=0.08)

    # --- LEFT: ROC curves ---
    roc_img = mpimg.imread(str(GATE42_DIR / "fig_roc_p0_p1.png"))
    ax_roc = fig.add_subplot(gs[0])
    ax_roc.set_facecolor(DARK_BG)
    ax_roc.imshow(roc_img)
    ax_roc.axis("off")

    # Overlay AUC annotations
    p0_auc = prered["P0_oracle"]["auc"]
    p1_auc = prered["P1_real"]["auc"]
    glow_text(ax_roc, 0.05, 0.15,
              f"P0 Oracle AUC: {p0_auc:.3f}\nP1 Real AUC: {p1_auc:.3f}",
              fontsize=13, color="#ffaa00", weight="bold")

    # GO threshold line annotation
    glow_text(ax_roc, 0.05, 0.05,
              "GO threshold: AUC >= 0.80 (P0), >= 0.70 (P1)",
              fontsize=10, color="#666666")

    # --- RIGHT: Scatter ---
    scatter_img = mpimg.imread(str(GATE42_DIR / "fig_similarity_scatter.png"))
    ax_scatter = fig.add_subplot(gs[1])
    ax_scatter.set_facecolor(DARK_BG)
    ax_scatter.imshow(scatter_img)
    ax_scatter.axis("off")

    # Overlay delta_sim annotations
    p0_delta = prered["P0_oracle"]["delta_sim"]
    p1_delta = prered["P1_real"]["delta_sim"]
    glow_text(ax_scatter, 0.05, 0.15,
              f"P0 delta_sim: {p0_delta:+.4f}\nP1 delta_sim: {p1_delta:+.4f}",
              fontsize=13, color="#ffaa00", weight="bold")

    # NO-GO verdict box
    verdict_y = 0.93
    for ax in [ax_roc, ax_scatter]:
        pass  # verdict goes on the figure level

    # Large NO-GO verdict across the top
    fig.text(0.5, 0.96,
             "VEREDICTO:  NO-GO",
             fontsize=28, color=COLOR_NOGO, ha="center", weight="bold",
             path_effects=[pe.withStroke(linewidth=5, foreground="#000000")])

    fig.text(0.5, 0.92,
             "H4.2-6 (dual-domain harmonic ratios via CQT) — ELIMINADA",
             fontsize=14, color="#ff8888", ha="center",
             path_effects=[pe.withStroke(linewidth=3, foreground="#000000")])

    # Caption
    fig.text(0.5, 0.01,
             "CQT cannot discriminate, even under oracle conditions.",
             fontsize=14, color="white", ha="center", weight="bold",
             path_effects=[pe.withStroke(linewidth=4, foreground="#000000")])

    fig.text(0.5, -0.02,
             "Source: h426_prered_results.json, fig_roc_p0_p1.png, "
             "fig_similarity_scatter.png | DEC-005 Track 2",
             fontsize=8, color="#555555", ha="center")

    output = OUTPUT_DIR / "HERO_03_prered_nogo.png"
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor=DARK_BG,
                pad_inches=0.3)
    plt.close(fig)
    logger.info(f"Saved: {output}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")

    generate_hero_01()
    generate_hero_02()
    generate_hero_03()

    logger.info("\nAll 3 hero images generated successfully.")


if __name__ == "__main__":
    main()
