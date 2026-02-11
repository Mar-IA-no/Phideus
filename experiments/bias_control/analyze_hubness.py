#!/usr/bin/env python3
"""Hubness analysis across Gate 2 and fine-tuned checkpoints.

DEC-005 Track 1 — Gate 6 Phase 1, Script 3.

Tests hypotheses:
  H6.3: If k-occurrence skewness grows with fine-tuning -> hubness is a problem
  H6.4: If correct/incorrect sim separation is smaller for A2M than M2A -> intrinsic asymmetry

Requires: multigate_embeddings.npz (from Script 2)

Output:
    data/bias_control_medium/evaluations/gate6/hubness_analysis.json
    data/bias_control_medium/evaluations/gate6/fig_hubness_distribution.png
    data/bias_control_medium/evaluations/gate6/fig_similarity_distributions.png
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
EMB_PATH = ROOT / "data/bias_control_medium/evaluations/gate6/multigate_embeddings.npz"
OUTPUT_DIR = ROOT / "data/bias_control_medium/evaluations/gate6"

LABELS = ["gate2", "RB0", "RA5", "R1"]
K = 10  # top-K for hubness

# Dark theme
DARK_BG = "#0a0a0a"
COLOR_AUDIO = "#00e5ff"
COLOR_MIDI = "#ff1493"
COLOR_CORRECT = "#00ff88"
COLOR_INCORRECT = "#ff4444"


def setup_dark_axes(ax):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors="white", which="both")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#333333")


def cosine_sim_matrix(A, B):
    """Compute cosine similarity matrix [N_A, N_B]."""
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_norm @ B_norm.T


def compute_hubness(sim_matrix, k=K):
    """Compute k-occurrence for each column (MIDI embedding).

    For each audio query (row), find top-k MIDI neighbors.
    k-occurrence[j] = how many audio queries have MIDI j in their top-k.
    """
    n_queries = sim_matrix.shape[0]
    n_keys = sim_matrix.shape[1]
    k_occ = np.zeros(n_keys, dtype=int)

    for i in range(n_queries):
        topk = np.argsort(sim_matrix[i])[-k:]
        for j in topk:
            k_occ[j] += 1

    return k_occ


def compute_per_piece_recall(sim_matrix, piece_indices, k=K):
    """Compute per-piece Recall@K.

    Returns dict: piece_idx -> recall@K
    Each segment's correct match is the same index (diagonal).
    """
    n = sim_matrix.shape[0]
    unique_pieces = np.unique(piece_indices)
    piece_recalls = {}

    for piece in unique_pieces:
        piece_mask = piece_indices == piece
        piece_indices_list = np.where(piece_mask)[0]
        if len(piece_indices_list) == 0:
            continue

        hits = 0
        for idx in piece_indices_list:
            topk = np.argsort(sim_matrix[idx])[-k:]
            if idx in topk:
                hits += 1
        piece_recalls[int(piece)] = hits / len(piece_indices_list)

    return piece_recalls


def compute_sim_distributions(sim_matrix):
    """Get correct (diagonal) and incorrect similarities."""
    n = sim_matrix.shape[0]
    correct_sims = np.array([sim_matrix[i, i] for i in range(n)])
    # Sample incorrect: for each query, take mean of non-diagonal
    incorrect_sims = []
    rng = np.random.RandomState(42)
    for i in range(n):
        # Sample 50 random incorrect indices
        candidates = np.delete(np.arange(n), i)
        sample = rng.choice(candidates, min(50, len(candidates)), replace=False)
        incorrect_sims.extend(sim_matrix[i, sample].tolist())
    return correct_sims, np.array(incorrect_sims)


def main():
    logger.info(f"Loading embeddings from {EMB_PATH}")
    data = np.load(EMB_PATH, allow_pickle=True)
    piece_indices = data["piece_indices"]
    logger.info(f"Segments: {len(piece_indices)}, Pieces: {len(np.unique(piece_indices))}")

    all_results = {}

    for label in LABELS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Analyzing: {label}")
        logger.info(f"{'='*50}")

        audio = data[f"{label}_audio"]
        midi = data[f"{label}_midi"]
        logger.info(f"  Shapes: audio={audio.shape}, midi={midi.shape}")

        # A2M: audio queries -> MIDI keys
        sim_a2m = cosine_sim_matrix(audio, midi)
        # M2A: MIDI queries -> audio keys
        sim_m2a = cosine_sim_matrix(midi, audio)

        # 1. Hubness (A2M direction)
        k_occ_a2m = compute_hubness(sim_a2m, K)
        k_occ_m2a = compute_hubness(sim_m2a, K)

        hub_stats_a2m = {
            "mean": float(np.mean(k_occ_a2m)),
            "std": float(np.std(k_occ_a2m)),
            "max": int(np.max(k_occ_a2m)),
            "skewness": float(skew(k_occ_a2m)),
            "n_hubs_3sigma": int(np.sum(k_occ_a2m > np.mean(k_occ_a2m) + 3 * np.std(k_occ_a2m))),
        }
        hub_stats_m2a = {
            "mean": float(np.mean(k_occ_m2a)),
            "std": float(np.std(k_occ_m2a)),
            "max": int(np.max(k_occ_m2a)),
            "skewness": float(skew(k_occ_m2a)),
            "n_hubs_3sigma": int(np.sum(k_occ_m2a > np.mean(k_occ_m2a) + 3 * np.std(k_occ_m2a))),
        }
        logger.info(f"  A2M hubness: mean={hub_stats_a2m['mean']:.1f}, skew={hub_stats_a2m['skewness']:.2f}, "
                     f"max={hub_stats_a2m['max']}, hubs(3σ)={hub_stats_a2m['n_hubs_3sigma']}")
        logger.info(f"  M2A hubness: mean={hub_stats_m2a['mean']:.1f}, skew={hub_stats_m2a['skewness']:.2f}, "
                     f"max={hub_stats_m2a['max']}, hubs(3σ)={hub_stats_m2a['n_hubs_3sigma']}")

        # 2. Per-piece recall
        piece_recall_a2m = compute_per_piece_recall(sim_a2m, piece_indices, K)
        piece_recall_m2a = compute_per_piece_recall(sim_m2a, piece_indices, K)

        mean_a2m = np.mean(list(piece_recall_a2m.values()))
        mean_m2a = np.mean(list(piece_recall_m2a.values()))
        logger.info(f"  Per-piece R@{K}: A2M mean={mean_a2m:.3f}, M2A mean={mean_m2a:.3f}")

        # 3. Similarity distributions
        correct_a2m, incorrect_a2m = compute_sim_distributions(sim_a2m)
        correct_m2a, incorrect_m2a = compute_sim_distributions(sim_m2a)

        sep_a2m = float(np.mean(correct_a2m) - np.mean(incorrect_a2m))
        sep_m2a = float(np.mean(correct_m2a) - np.mean(incorrect_m2a))
        logger.info(f"  Sim separation: A2M={sep_a2m:.4f}, M2A={sep_m2a:.4f}")

        all_results[label] = {
            "hubness_a2m": hub_stats_a2m,
            "hubness_m2a": hub_stats_m2a,
            "per_piece_recall_a2m_mean": round(mean_a2m, 4),
            "per_piece_recall_m2a_mean": round(mean_m2a, 4),
            "sim_separation_a2m": round(sep_a2m, 4),
            "sim_separation_m2a": round(sep_m2a, 4),
            "correct_sim_a2m_mean": round(float(np.mean(correct_a2m)), 4),
            "correct_sim_m2a_mean": round(float(np.mean(correct_m2a)), 4),
            "incorrect_sim_a2m_mean": round(float(np.mean(incorrect_a2m)), 4),
            "incorrect_sim_m2a_mean": round(float(np.mean(incorrect_m2a)), 4),
            "k_occ_a2m": k_occ_a2m.tolist(),
            "k_occ_m2a": k_occ_m2a.tolist(),
            "per_piece_recall_a2m": {str(k): round(v, 4) for k, v in piece_recall_a2m.items()},
            "per_piece_recall_m2a": {str(k): round(v, 4) for k, v in piece_recall_m2a.items()},
        }

    # --- Per-piece drift vs Gate 2 ---
    if "gate2" in all_results:
        gate2_a2m = all_results["gate2"]["per_piece_recall_a2m"]
        gate2_m2a = all_results["gate2"]["per_piece_recall_m2a"]
        for label in ["RB0", "RA5", "R1"]:
            if label not in all_results:
                continue
            ft_a2m = all_results[label]["per_piece_recall_a2m"]
            ft_m2a = all_results[label]["per_piece_recall_m2a"]

            pieces_lost_a2m = sum(1 for p in gate2_a2m if ft_a2m.get(p, 0) < gate2_a2m[p])
            pieces_gained_m2a = sum(1 for p in gate2_m2a if ft_m2a.get(p, 0) > gate2_m2a[p])
            # Correlation: pieces that lose A2M vs gain M2A
            common_pieces = set(gate2_a2m.keys()) & set(gate2_m2a.keys()) & set(ft_a2m.keys()) & set(ft_m2a.keys())
            delta_a2m_vals = [ft_a2m[p] - gate2_a2m[p] for p in common_pieces]
            delta_m2a_vals = [ft_m2a[p] - gate2_m2a[p] for p in common_pieces]
            if len(delta_a2m_vals) > 2:
                corr = float(np.corrcoef(delta_a2m_vals, delta_m2a_vals)[0, 1])
            else:
                corr = 0.0

            all_results[label]["vs_gate2"] = {
                "pieces_lost_a2m": pieces_lost_a2m,
                "pieces_gained_m2a": pieces_gained_m2a,
                "delta_a2m_m2a_correlation": round(corr, 4),
            }
            logger.info(f"  {label} vs Gate2: {pieces_lost_a2m} pieces lost A2M, "
                        f"{pieces_gained_m2a} gained M2A, corr={corr:.3f}")

    # Strip large arrays before saving JSON
    results_for_json = {}
    for label, res in all_results.items():
        res_copy = {k: v for k, v in res.items() if k not in ("k_occ_a2m", "k_occ_m2a")}
        results_for_json[label] = res_copy

    json_path = OUTPUT_DIR / "hubness_analysis.json"
    with open(json_path, "w") as f:
        json.dump(results_for_json, f, indent=2)
    logger.info(f"\nSaved: {json_path}")

    # --- Figures ---
    logger.info("Generating figures...")

    # Fig 1: K-occurrence distribution (4 panels, A2M direction)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(f"MIDI Hub Distribution (A2M, K={K})", color="white", fontsize=14)

    for ax, label in zip(axes.flat, LABELS):
        setup_dark_axes(ax)
        k_occ = np.array(all_results[label]["k_occ_a2m"])
        stats = all_results[label]["hubness_a2m"]

        ax.hist(k_occ, bins=range(0, min(int(k_occ.max()) + 2, 80)),
                color=COLOR_MIDI, alpha=0.7, edgecolor="#333333")
        ax.axvline(stats["mean"], color=COLOR_AUDIO, linestyle="--", linewidth=1.5,
                   label=f"mean={stats['mean']:.1f}")
        ax.axvline(stats["mean"] + 3 * stats["std"], color="#ff8800", linestyle=":",
                   linewidth=1, label=f"3σ={stats['mean'] + 3 * stats['std']:.0f}")
        ax.set_title(f"{label} (skew={stats['skewness']:.2f}, hubs={stats['n_hubs_3sigma']})",
                     fontsize=10)
        ax.set_xlabel("k-occurrence")
        ax.set_ylabel("count")
        ax.legend(fontsize=7, facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path = OUTPUT_DIR / "fig_hubness_distribution.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logger.info(f"Saved: {fig_path}")

    # Fig 2: Similarity distributions (correct vs incorrect, A2M vs M2A)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Similarity Distributions: Correct vs Incorrect", color="white", fontsize=14)

    for col, label in enumerate(LABELS):
        audio = data[f"{label}_audio"]
        midi = data[f"{label}_midi"]

        # Subsample for efficiency (1000 segments)
        rng = np.random.RandomState(42)
        n = min(1000, len(audio))
        idx = rng.choice(len(audio), n, replace=False)
        audio_sub = audio[idx]
        midi_sub = midi[idx]

        sim_a2m = cosine_sim_matrix(audio_sub, midi_sub)
        sim_m2a = cosine_sim_matrix(midi_sub, audio_sub)

        correct_a2m = [sim_a2m[i, i] for i in range(n)]
        incorrect_a2m = [sim_a2m[i, j] for i in range(min(200, n))
                         for j in rng.choice(np.delete(np.arange(n), i), 20, replace=False)]

        correct_m2a = [sim_m2a[i, i] for i in range(n)]
        incorrect_m2a = [sim_m2a[i, j] for i in range(min(200, n))
                         for j in rng.choice(np.delete(np.arange(n), i), 20, replace=False)]

        for row, (correct, incorrect, direction) in enumerate([
            (correct_a2m, incorrect_a2m, "A2M"),
            (correct_m2a, incorrect_m2a, "M2A"),
        ]):
            ax = axes[row, col]
            setup_dark_axes(ax)
            ax.hist(incorrect, bins=60, density=True, alpha=0.5, color=COLOR_INCORRECT,
                    label="Incorrect")
            ax.hist(correct, bins=60, density=True, alpha=0.7, color=COLOR_CORRECT,
                    label="Correct")
            sep = np.mean(correct) - np.mean(incorrect)
            ax.set_title(f"{label} {direction} (sep={sep:.3f})", fontsize=9)
            if col == 0:
                ax.set_ylabel("density", fontsize=8)
            if row == 1:
                ax.set_xlabel("cosine sim", fontsize=8)
            if col == 0 and row == 0:
                ax.legend(fontsize=6, facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path = OUTPUT_DIR / "fig_similarity_distributions.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logger.info(f"Saved: {fig_path}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("HUBNESS ANALYSIS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Checkpoint':<10} {'A2M skew':>10} {'M2A skew':>10} {'A2M sep':>10} {'M2A sep':>10}")
    for label in LABELS:
        r = all_results[label]
        logger.info(f"{label:<10} {r['hubness_a2m']['skewness']:>10.2f} "
                     f"{r['hubness_m2a']['skewness']:>10.2f} "
                     f"{r['sim_separation_a2m']:>10.4f} "
                     f"{r['sim_separation_m2a']:>10.4f}")

    # H6.3 check
    g2_skew = all_results["gate2"]["hubness_a2m"]["skewness"]
    for label in ["RB0", "RA5", "R1"]:
        ft_skew = all_results[label]["hubness_a2m"]["skewness"]
        logger.info(f"  H6.3: {label} A2M skew={ft_skew:.2f} vs Gate2={g2_skew:.2f} "
                     f"-> {'WORSE' if ft_skew > g2_skew else 'better'}")

    # H6.4 check
    for label in LABELS:
        a2m_sep = all_results[label]["sim_separation_a2m"]
        m2a_sep = all_results[label]["sim_separation_m2a"]
        logger.info(f"  H6.4: {label} A2M sep={a2m_sep:.4f} vs M2A sep={m2a_sep:.4f} "
                     f"-> {'A2M weaker' if a2m_sep < m2a_sep else 'symmetric'}")


if __name__ == "__main__":
    main()
