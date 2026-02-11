#!/usr/bin/env python3
"""Compare parameter drift between Gate 2 baseline and fine-tuned checkpoints.

DEC-005 Track 1 — Gate 6 Phase 1, Script 1.

Tests hypotheses:
  H6.1: Asymmetric forgetting — audio_projection drift >> midi_projection drift
  H6.2: Deep layer sensitivity — transformer layers 3-4 drift > layers 0-1

No GPU required. Loads state_dicts on CPU only.

Output: data/bias_control_medium/evaluations/gate6/layer_drift.json
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]

BASELINE = ROOT / "data/bias_control_medium/training_outputs/gate2/checkpoint_epoch45.pt"
CHECKPOINTS = {
    "RB0": ROOT / "data/bias_control_medium/training_outputs/gate4_RB0/checkpoint_epoch5_base.pt",
    "RA5": ROOT / "data/bias_control_medium/training_outputs/gate4_runA/checkpoint_epoch5_base.pt",
    "R1":  ROOT / "data/bias_control_medium/training_outputs/gate4_R1rescue/checkpoint_epoch5_base.pt",
}
OUTPUT = ROOT / "data/bias_control_medium/evaluations/gate6/layer_drift.json"

MODULE_GROUPS = {
    "audio_encoder.feature_extractor": "Audio CNN",
    "audio_encoder.pos_embedding":     "Audio PosEmb",
    "audio_encoder.transformer":       "Audio Transformer",
    "audio_projection":                "Audio Projection",
    "midi_encoder.event_embedding":    "MIDI Embedding",
    "midi_encoder.output_norm":        "MIDI OutputNorm",
    "midi_encoder.pos_encoding":       "MIDI PosEncoding",
    "midi_encoder.transformer":        "MIDI Transformer",
    "midi_projection":                 "MIDI Projection",
}


def classify_param(name: str) -> str:
    for prefix, group in MODULE_GROUPS.items():
        if name.startswith(prefix):
            return group
    return "Other"


def compute_drift(base_sd, ft_sd):
    """Compute per-parameter drift metrics between two state_dicts."""
    results = {}
    shared_keys = set(base_sd.keys()) & set(ft_sd.keys())

    for key in sorted(shared_keys):
        w_base = base_sd[key].float()
        w_ft = ft_sd[key].float()
        if w_base.shape != w_ft.shape:
            continue

        delta = w_ft - w_base
        l2 = delta.norm().item()
        base_norm = w_base.norm().item()
        rel_change = l2 / (base_norm + 1e-12)

        # Cosine similarity (flatten)
        if w_base.numel() > 1:
            cos_sim = F.cosine_similarity(
                w_base.flatten().unsqueeze(0),
                w_ft.flatten().unsqueeze(0)
            ).item()
        else:
            cos_sim = 1.0 if torch.allclose(w_base, w_ft) else 0.0

        results[key] = {
            "group": classify_param(key),
            "shape": list(w_base.shape),
            "numel": w_base.numel(),
            "l2_distance": round(l2, 6),
            "cosine_sim": round(cos_sim, 6),
            "relative_change": round(rel_change, 6),
            "base_norm": round(base_norm, 6),
        }

    return results


def aggregate_by_group(param_results):
    """Aggregate drift metrics by module group."""
    groups = {}
    for key, metrics in param_results.items():
        g = metrics["group"]
        if g not in groups:
            groups[g] = {"params": [], "l2_sum": 0.0, "numel": 0}
        groups[g]["params"].append((key, metrics))
        groups[g]["l2_sum"] += metrics["l2_distance"]
        groups[g]["numel"] += metrics["numel"]

    summary = {}
    for g, data in groups.items():
        l2_vals = [m["l2_distance"] for _, m in data["params"]]
        rel_vals = [m["relative_change"] for _, m in data["params"]]
        cos_vals = [m["cosine_sim"] for _, m in data["params"]]
        worst = max(data["params"], key=lambda x: x[1]["relative_change"])

        summary[g] = {
            "n_params": len(data["params"]),
            "total_numel": data["numel"],
            "l2_mean": round(sum(l2_vals) / len(l2_vals), 6),
            "l2_max": round(max(l2_vals), 6),
            "rel_change_mean": round(sum(rel_vals) / len(rel_vals), 6),
            "rel_change_max": round(max(rel_vals), 6),
            "cosine_sim_mean": round(sum(cos_vals) / len(cos_vals), 6),
            "cosine_sim_min": round(min(cos_vals), 6),
            "worst_param": worst[0],
            "worst_rel_change": round(worst[1]["relative_change"], 6),
        }

    return summary


def print_report(label, summary):
    """Print formatted drift report for one checkpoint."""
    print(f"\n{'='*70}")
    print(f"  {label} vs Gate 2 (baseline)")
    print(f"{'='*70}")
    print(f"{'Module':<22} {'Params':>6} {'L2 mean':>10} {'RelΔ mean':>10} "
          f"{'RelΔ max':>10} {'Cos min':>10}")
    print("-" * 70)

    # Sort: audio groups first, then midi
    order = [
        "Audio CNN", "Audio PosEmb", "Audio Transformer", "Audio Projection",
        "MIDI Embedding", "MIDI OutputNorm", "MIDI PosEncoding",
        "MIDI Transformer", "MIDI Projection", "Other",
    ]
    for g in order:
        if g not in summary:
            continue
        s = summary[g]
        print(f"{g:<22} {s['n_params']:>6} {s['l2_mean']:>10.4f} "
              f"{s['rel_change_mean']:>10.4f} {s['rel_change_max']:>10.4f} "
              f"{s['cosine_sim_min']:>10.6f}")

    # Hypothesis check
    ap = summary.get("Audio Projection", {})
    mp = summary.get("MIDI Projection", {})
    if ap and mp:
        print(f"\n  H6.1 check: Audio Proj relΔ={ap['rel_change_mean']:.4f} "
              f"vs MIDI Proj relΔ={mp['rel_change_mean']:.4f} "
              f"→ ratio={ap['rel_change_mean']/(mp['rel_change_mean']+1e-12):.2f}x")

    at = summary.get("Audio Transformer", {})
    mt = summary.get("MIDI Transformer", {})
    if at and mt:
        print(f"  H6.2 check: Audio Trans relΔ={at['rel_change_mean']:.4f} "
              f"vs MIDI Trans relΔ={mt['rel_change_mean']:.4f}")


def main():
    print("Loading Gate 2 baseline checkpoint...")
    base_sd = torch.load(BASELINE, map_location="cpu")["model_state_dict"]
    print(f"  {len(base_sd)} parameters")

    all_results = {}

    for label, path in CHECKPOINTS.items():
        print(f"\nLoading {label} checkpoint from {path.name}...")
        ft_sd = torch.load(path, map_location="cpu")["model_state_dict"]

        # Verify overlap
        shared = set(base_sd.keys()) & set(ft_sd.keys())
        only_base = set(base_sd.keys()) - set(ft_sd.keys())
        only_ft = set(ft_sd.keys()) - set(base_sd.keys())
        print(f"  Shared: {len(shared)}, Only in base: {len(only_base)}, "
              f"Only in {label}: {len(only_ft)}")

        if only_base:
            print(f"  Keys only in Gate 2: {sorted(only_base)}")
        if only_ft:
            print(f"  Keys only in {label}: {sorted(only_ft)}")

        param_drift = compute_drift(base_sd, ft_sd)
        summary = aggregate_by_group(param_drift)
        print_report(label, summary)

        all_results[label] = {
            "summary": summary,
            "shared_params": len(shared),
            "only_in_baseline": sorted(only_base),
            "only_in_finetuned": sorted(only_ft),
        }

    # Sanity check: drift must be > 0 for at least audio_projection
    for label, res in all_results.items():
        ap = res["summary"].get("Audio Projection", {})
        if ap and ap.get("l2_mean", 0) == 0:
            print(f"\n⚠️  WARNING: {label} Audio Projection has zero drift — "
                  f"possible checkpoint loading bug!")

    # Save JSON
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUTPUT}")


if __name__ == "__main__":
    main()
