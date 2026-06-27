#!/usr/bin/env python3
"""E3-P4: Probe Regime on frozen P2 latents.

This script tests the Chapter 10 activation hypothesis:
does the METHOD OF READING change the structure recovered from the same latent?

Two frozen baselines are expected:
  - L0-Flat Canonical
  - L0-CQTShift Ratio-Aware

This is eval-only. No retraining.
"""

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.escalon3.eval_escalon3 import (
    extract_embeddings,
    sample_structured_pool,
    find_positives_scale_ood,
    find_positives_equiv_ood,
    _build_key_index,
    _build_equiv_key_index,
    build_lissajous_index,
)
from src.bias_control.encoders.projection import ProjectionHead
from src.escalon3.encoders import (
    build_lissajous_audio_encoder,
    build_lissajous_image_encoder,
)
from src.escalon3.lissajous_dataset import (
    LissajousDataset,
    load_ood_dataset,
    load_reduced_gallery_dataset,
)


PHI = (1.0 + math.sqrt(5.0)) / 2.0
EPS = 1e-8


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _audio_key(encoder) -> str:
    return "audio_cqt" if getattr(encoder, "input_kind", "waveform") == "cqt" else "audio"


def _float_key(value: float) -> str:
    return f"{value:.10f}"


def _json_default(obj):
    if isinstance(obj, (Path,)):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _candidate_meta(dataset, candidate_indices: Sequence[int]) -> Dict[str, List]:
    return {
        "candidate_indices": [int(i) for i in candidate_indices],
        "candidate_ratio_ids": [int(dataset.ratio_id[i]) for i in candidate_indices],
        "candidate_equiv_ids": [int(dataset.equiv_id[i]) for i in candidate_indices],
        "candidate_scene_ids": [str(dataset.scene_id[i]) for i in candidate_indices],
    }


def _resolve_query_indices(n_total: int, n_queries: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    return rng.sample(range(n_total), min(n_queries, n_total))


def precompute_probe_pools(
    split_name: str,
    query_ds,
    gallery_ds,
    pool_size: int,
    n_queries: int,
    seed: int,
    positive_finder=None,
) -> Dict:
    """Precompute fixed candidate pools for all probe families.

    For IID:
      - query_ds is gallery_ds (test split)
      - positive = same scene index
    For OOD:
      - positive_finder resolves gallery positives
    """
    rng = random.Random(seed)
    if split_name == "iid":
        gallery_index = build_lissajous_index(gallery_ds)
    else:
        gallery_index = build_lissajous_index(gallery_ds)

    query_ids = _resolve_query_indices(len(query_ds), n_queries, seed)
    pools = {}

    for q_idx in query_ids:
        if split_name == "iid":
            candidates, pos_pos = sample_structured_pool(q_idx, gallery_index, pool_size, rng)
            pool = {
                "positive_positions": [int(pos_pos)],
                **_candidate_meta(gallery_ds, candidates),
            }
        else:
            positive_gallery_indices = positive_finder(q_idx)
            if not positive_gallery_indices:
                continue

            pos_gal_idx = rng.choice(positive_gallery_indices)
            candidates, _ = sample_structured_pool(pos_gal_idx, gallery_index, pool_size, rng)
            positive_set = set(int(i) for i in positive_gallery_indices)
            positive_positions = [i for i, c in enumerate(candidates) if int(c) in positive_set]
            if not positive_positions:
                continue

            pool = {
                "positive_positions": [int(i) for i in positive_positions],
                **_candidate_meta(gallery_ds, candidates),
            }

        pools[int(q_idx)] = pool

    return {
        "split_name": split_name,
        "query_ids": [int(i) for i in pools.keys()],
        "pools": pools,
    }


class ProbeEngine:
    """Hyperspherical probe traversal on S^(D-1)."""

    def __init__(self, dim: int, n_directions: int = 4, k_walk: int = 32, seed: int = 42):
        self.dim = dim
        self.n_directions = n_directions
        self.k_walk = k_walk
        gen = torch.Generator()
        gen.manual_seed(seed)
        base = torch.randn(64, dim, generator=gen)
        self.global_dirs = F.normalize(base, dim=-1)

    def _tangent_directions(self, u: torch.Tensor) -> torch.Tensor:
        directions = []
        for g in self.global_dirs:
            d = g.to(u.device) - torch.dot(g.to(u.device), u) * u
            norm = d.norm()
            if norm > 1e-6:
                directions.append(d / norm.clamp_min(EPS))
            if len(directions) == self.n_directions:
                break
        if len(directions) < self.n_directions:
            raise RuntimeError("Failed to construct tangent directions")
        return torch.stack(directions, dim=0)

    def generate_walk(self, query: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (probes [M,K,D], tangents [M,D])."""
        u = F.normalize(query, dim=0)
        tangents = self._tangent_directions(u)
        steps = torch.arange(1, self.k_walk + 1, device=query.device, dtype=query.dtype)
        angles = steps * alpha
        cos_t = torch.cos(angles)[None, :, None]
        sin_t = torch.sin(angles)[None, :, None]
        probes = cos_t * u[None, None, :] + sin_t * tangents[:, None, :]
        probes = F.normalize(probes, dim=-1)
        return probes, tangents

    def probe_scores(self, query: torch.Tensor, candidates_norm: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return aggregated scores and top-1 candidate indices per (m, t)."""
        probes, _ = self.generate_walk(query, alpha)
        flat = probes.reshape(-1, query.numel())
        sims = flat @ candidates_norm.T
        scores = sims.max(dim=0).values
        top_idx = sims.argmax(dim=1).reshape(self.n_directions, self.k_walk)
        return scores, top_idx


def _build_probe_families() -> Dict[str, Dict]:
    simple = [2 * math.pi * x for x in (1 / 2, 1 / 3, 1 / 4, 1 / 5)]
    complex_ = [2 * math.pi * x for x in (2 / 5, 3 / 7, 5 / 8)]
    phi_alpha = 2 * math.pi / PHI
    noble_alpha = 2 * math.pi / (PHI ** 2)

    rational_refs = simple + complex_ + [phi_alpha, noble_alpha]

    def draw_irrational(seed: int) -> float:
        rng = random.Random(seed)
        while True:
            frac = rng.random()
            alpha = 2 * math.pi * frac
            too_close = any(abs(((alpha - ref + math.pi) % (2 * math.pi)) - math.pi) < 0.05 for ref in rational_refs)
            if not too_close:
                return alpha

    irr1 = draw_irrational(42)
    irr2 = draw_irrational(43)
    irr3 = draw_irrational(44)

    return {
        "cosine": {"type": "direct", "gate_metric": False, "has_walk": False},
        "euclidean_raw": {"type": "direct_raw", "gate_metric": False, "has_walk": False},
        "rational_simple": {"type": "multi_alpha", "alphas": simple, "gate_metric": "mean"},
        "rational_complex": {"type": "multi_alpha", "alphas": complex_, "gate_metric": "mean"},
        "phi": {"type": "single_alpha", "alpha": phi_alpha, "gate_metric": True, "has_walk": True},
        "noble": {"type": "single_alpha", "alpha": noble_alpha, "gate_metric": True, "has_walk": True},
        "irrational_random_1": {"type": "single_alpha", "alpha": irr1, "gate_metric": False, "has_walk": True},
        "irrational_random_2": {"type": "single_alpha", "alpha": irr2, "gate_metric": False, "has_walk": True},
        "irrational_random_3": {"type": "single_alpha", "alpha": irr3, "gate_metric": False, "has_walk": True},
    }


def _hit_at_k(ranked_positions: Sequence[int], k: int) -> float:
    if not ranked_positions:
        return 0.0
    arr = np.array(ranked_positions)
    return float((arr < k).mean())


def _normalized_entropy(labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    unique, counts = np.unique(np.asarray(labels), return_counts=True)
    if len(unique) <= 1:
        return 0.0
    probs = counts / counts.sum()
    entropy = -(probs * np.log(probs + EPS)).sum()
    return float(entropy / math.log(len(unique)))


def _first_relocking_depth(labels: Sequence[int], consecutive: int = 3, max_depth: int = 32) -> float:
    if len(labels) < consecutive:
        return float(max_depth + 1)
    for idx in range(len(labels) - consecutive + 1):
        window = labels[idx:idx + consecutive]
        if len(set(window)) == 1:
            return float(idx + 1)
    return float(max_depth + 1)


def _walk_metrics_for_direction(visited_labels: Sequence[int], candidate_class_ids: Sequence[int], max_depth: int) -> Dict[str, float]:
    if not visited_labels:
        return {
            "locking_selectivity": 0.0,
            "coverage_uniformity": 0.0,
            "relocking_depth": float(max_depth + 1),
            "basin_exposure": 0.0,
        }

    label_arr = np.asarray(visited_labels)
    _, counts = np.unique(label_arr, return_counts=True)
    dominant = counts.max()
    distinct_visited = len(np.unique(label_arr))
    distinct_pool = len(np.unique(np.asarray(candidate_class_ids)))

    return {
        "locking_selectivity": float(dominant / len(label_arr)),
        "coverage_uniformity": _normalized_entropy(visited_labels),
        "relocking_depth": _first_relocking_depth(visited_labels, max_depth=max_depth),
        "basin_exposure": float(distinct_visited / max(distinct_pool, 1)),
    }


def _aggregate_direction_metrics(direction_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    keys = direction_metrics[0].keys()
    out = {}
    for key in keys:
        values = np.array([m[key] for m in direction_metrics], dtype=np.float64)
        out[key] = float(values.mean())
        out[f"{key}_std"] = float(values.std())
    return out


def _direct_scores(
    family_name: str,
    query_raw: torch.Tensor,
    query_norm: torch.Tensor,
    candidates_raw: torch.Tensor,
    candidates_norm: torch.Tensor,
    k_walk: int,
    n_directions: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if family_name == "cosine":
        scores = query_norm @ candidates_norm.T
    elif family_name == "euclidean_raw":
        scores = -torch.norm(candidates_raw - query_raw.unsqueeze(0), dim=1)
    else:
        raise ValueError(f"Unknown direct family: {family_name}")

    top_idx = int(scores.argmax().item())
    trace = torch.full((n_directions, k_walk), top_idx, dtype=torch.long)
    return scores, trace


def _candidate_class_key(split_name: str) -> str:
    return "candidate_equiv_ids" if split_name == "equiv_ood" else "candidate_ratio_ids"


def _metric_name(split_name: str) -> str:
    if split_name == "iid":
        return "scene_hit_at_10"
    if split_name == "scale_ood":
        return "ratio_hit_at_10"
    if split_name == "equiv_ood":
        return "equiv_hit_at_10"
    raise ValueError(split_name)


def _success_positions(split_name: str, pool: Dict, query_ds, q_idx: int) -> List[int]:
    if split_name == "iid":
        return pool["positive_positions"]

    if split_name == "scale_ood":
        target = int(query_ds.ratio_id[q_idx])
        return [i for i, rid in enumerate(pool["candidate_ratio_ids"]) if int(rid) == target]

    if split_name == "equiv_ood":
        target = int(query_ds.equiv_id[q_idx])
        return [i for i, eid in enumerate(pool["candidate_equiv_ids"]) if int(eid) == target]

    raise ValueError(split_name)


def _compute_hit_at_10(split_name: str, sorted_idx: List[int], pool: Dict, query_ds, q_idx: int, k_hit: int) -> Tuple[float, int]:
    positive_positions = _success_positions(split_name, pool, query_ds, q_idx)
    if not positive_positions:
        return 0.0, -1
    best_rank = min(sorted_idx.index(pos) + 1 for pos in positive_positions)
    return float(best_rank <= k_hit), best_rank


def evaluate_single_alpha(
    family_name: str,
    alpha: Optional[float],
    query_embs: torch.Tensor,
    candidate_embs: torch.Tensor,
    split_name: str,
    query_ds,
    pool_cache: Dict,
    query_ids: Sequence[int],
    probe_engine: ProbeEngine,
    k_hit: int,
) -> Tuple[Dict, Dict]:
    metric_name = _metric_name(split_name)
    hits = []
    ranks = []
    dir_metrics_all = []
    exemplar_traces = {}

    cand_raw_norm = None

    for q_idx in query_ids:
        pool = pool_cache["pools"].get(int(q_idx))
        if pool is None:
            continue

        cand_idx = pool["candidate_indices"]
        query_raw = query_embs[q_idx]
        candidate_raw = candidate_embs[cand_idx]
        query_norm = F.normalize(query_raw, dim=0)
        candidate_norm = F.normalize(candidate_raw, dim=-1)

        if family_name in {"cosine", "euclidean_raw"}:
            scores, trace_idx = _direct_scores(
                family_name, query_raw, query_norm, candidate_raw, candidate_norm,
                probe_engine.k_walk, probe_engine.n_directions)
        else:
            scores, trace_idx = probe_engine.probe_scores(query_norm, candidate_norm, alpha)

        sorted_idx = scores.argsort(descending=True).tolist()
        hit, best_rank = _compute_hit_at_10(split_name, sorted_idx, pool, query_ds, q_idx, k_hit)
        hits.append(hit)
        ranks.append(best_rank)

        class_key = _candidate_class_key(split_name)
        class_ids = pool[class_key]
        q_direction_metrics = []
        q_trace = []
        for m in range(trace_idx.shape[0]):
            visited = [int(class_ids[int(trace_idx[m, t].item())]) for t in range(trace_idx.shape[1])]
            q_direction_metrics.append(
                _walk_metrics_for_direction(visited, class_ids, max_depth=probe_engine.k_walk)
            )
            q_trace.append(visited)
        dir_metrics_all.append(_aggregate_direction_metrics(q_direction_metrics))
        exemplar_traces[int(q_idx)] = q_trace

    if not hits:
        result = {
            metric_name: 0.0,
            "n_queries": 0,
            "mean_rank": 0.0,
            "mrr": 0.0,
            "alpha": alpha,
        }
        empty_activation = {
            "locking_selectivity": 0.0,
            "locking_selectivity_std": 0.0,
            "coverage_uniformity": 0.0,
            "coverage_uniformity_std": 0.0,
            "relocking_depth": 0.0,
            "relocking_depth_std": 0.0,
            "basin_exposure": 0.0,
            "basin_exposure_std": 0.0,
        }
        return {**result, **empty_activation}, exemplar_traces

    rank_arr = np.array([r for r in ranks if r >= 0], dtype=np.float64)
    result = {
        metric_name: float(np.mean(hits)),
        "n_queries": len(hits),
        "mean_rank": float(rank_arr.mean()) if len(rank_arr) else 0.0,
        "mrr": float(np.mean(1.0 / rank_arr)) if len(rank_arr) else 0.0,
    }
    if alpha is not None:
        result["alpha"] = float(alpha)

    # Aggregate per-query activation metrics
    act_keys = dir_metrics_all[0].keys()
    for key in act_keys:
        result[key] = float(np.mean([m[key] for m in dir_metrics_all]))

    return result, exemplar_traces


def evaluate_family(
    family_name: str,
    family_cfg: Dict,
    query_embs: torch.Tensor,
    candidate_embs: torch.Tensor,
    split_name: str,
    query_ds,
    pool_cache: Dict,
    query_ids: Sequence[int],
    probe_engine: ProbeEngine,
    k_hit: int,
    max_exemplars: int = 50,
) -> Tuple[Dict, Dict]:
    if family_cfg["type"] in {"direct", "direct_raw"}:
        metrics, traces = evaluate_single_alpha(
            family_name,
            None,
            query_embs,
            candidate_embs,
            split_name,
            query_ds,
            pool_cache,
            query_ids,
            probe_engine,
            k_hit,
        )
        metrics["walk_metrics_valid"] = False
        exemplars = {str(k): v for k, v in list(traces.items())[:max_exemplars]}
        return metrics, exemplars

    if family_cfg["type"] == "single_alpha":
        metrics, traces = evaluate_single_alpha(
            family_name,
            family_cfg["alpha"],
            query_embs,
            candidate_embs,
            split_name,
            query_ds,
            pool_cache,
            query_ids,
            probe_engine,
            k_hit,
        )
        metrics["walk_metrics_valid"] = True
        exemplars = {str(k): v for k, v in list(traces.items())[:max_exemplars]}
        return metrics, exemplars

    if family_cfg["type"] == "multi_alpha":
        member_results = {}
        member_traces = {}
        for alpha in family_cfg["alphas"]:
            alpha_key = _float_key(alpha)
            metrics, traces = evaluate_single_alpha(
                family_name,
                alpha,
                query_embs,
                candidate_embs,
                split_name,
                query_ds,
                pool_cache,
                query_ids,
                probe_engine,
                k_hit,
            )
            member_results[alpha_key] = metrics
            member_traces[alpha_key] = {str(k): v for k, v in list(traces.items())[:max_exemplars]}

        numeric_keys = [k for k in next(iter(member_results.values())).keys() if k != "alpha"]
        family_mean = {}
        for key in numeric_keys:
            values = [mr[key] for mr in member_results.values() if isinstance(mr[key], (int, float))]
            family_mean[key] = float(np.mean(values))

        metric_name = _metric_name(split_name)
        best_alpha = max(member_results.items(), key=lambda kv: kv[1][metric_name])[0]
        return {
            "family_mean": family_mean,
            "best_alpha": best_alpha,
            "members": member_results,
            "walk_metrics_valid": True,
        }, member_traces

    raise ValueError(f"Unknown family type: {family_cfg['type']}")


def attach_activation_gain(results: Dict) -> None:
    """Add activation_gain against cosine within each split/direction."""
    for split_data in results["splits"].values():
        for dir_data in split_data.values():
            if "cosine" not in dir_data:
                continue
            metric_name = None
            for key in ("scene_hit_at_10", "ratio_hit_at_10", "equiv_hit_at_10"):
                if key in dir_data["cosine"]:
                    metric_name = key
                    break
            if metric_name is None:
                continue

            cosine_val = dir_data["cosine"][metric_name]
            for family_name, family_result in dir_data.items():
                if family_name == "cosine":
                    family_result["activation_gain"] = 0.0
                    continue
                if "family_mean" in family_result:
                    family_result["family_mean"]["activation_gain"] = (
                        family_result["family_mean"][metric_name] - cosine_val
                    )
                    for member in family_result["members"].values():
                        member["activation_gain"] = member[metric_name] - cosine_val
                else:
                    family_result["activation_gain"] = family_result[metric_name] - cosine_val


def summarize_for_csv(results: Dict) -> List[Dict]:
    rows = []
    for split_name, split_data in results["splits"].items():
        for direction, dir_data in split_data.items():
            for family_name, family_result in dir_data.items():
                row = {
                    "split": split_name,
                    "direction": direction,
                    "family": family_name,
                }
                if "family_mean" in family_result:
                    row.update({
                        "metric_value": family_result["family_mean"].get(_metric_name(split_name), 0.0),
                        "activation_gain": family_result["family_mean"].get("activation_gain", 0.0),
                        "coverage_uniformity": family_result["family_mean"].get("coverage_uniformity", 0.0),
                        "relocking_depth": family_result["family_mean"].get("relocking_depth", 0.0),
                        "best_alpha": family_result["best_alpha"],
                    })
                else:
                    row.update({
                        "metric_value": family_result.get(_metric_name(split_name), 0.0),
                        "activation_gain": family_result.get("activation_gain", 0.0),
                        "coverage_uniformity": family_result.get("coverage_uniformity", 0.0),
                        "relocking_depth": family_result.get("relocking_depth", 0.0),
                        "best_alpha": "",
                    })
                rows.append(row)
    return rows


def assess_gate(results: Dict) -> Dict:
    split = results["splits"]["scale_ood"]["a2i"]
    equiv = results["splits"]["equiv_ood"]["a2i"]

    phi_scale = split["phi"]["ratio_hit_at_10"]
    noble_scale = split["noble"]["ratio_hit_at_10"]
    cosine_scale = split["cosine"]["ratio_hit_at_10"]
    irr_scales = [split[f"irrational_random_{i}"]["ratio_hit_at_10"] for i in (1, 2, 3)]
    rat_scale_cov = split["rational_simple"]["family_mean"]["coverage_uniformity"]
    rat_scale_relock = split["rational_simple"]["family_mean"]["relocking_depth"]

    phi_equiv = equiv["phi"]["equiv_hit_at_10"]
    noble_equiv = equiv["noble"]["equiv_hit_at_10"]
    cosine_equiv = equiv["cosine"]["equiv_hit_at_10"]
    irr_equivs = [equiv[f"irrational_random_{i}"]["equiv_hit_at_10"] for i in (1, 2, 3)]
    rat_equiv_cov = equiv["rational_simple"]["family_mean"]["coverage_uniformity"]
    rat_equiv_relock = equiv["rational_simple"]["family_mean"]["relocking_depth"]

    candidates = {
        "phi": {
            "scale_gain": phi_scale - cosine_scale,
            "equiv_gain": phi_equiv - cosine_equiv,
            "scale_pass_random": phi_scale > max(irr_scales),
            "equiv_pass_random": phi_equiv > max(irr_equivs),
            "scale_cov_delta": split["phi"]["coverage_uniformity"] - rat_scale_cov,
            "scale_relock_delta": split["phi"]["relocking_depth"] - rat_scale_relock,
            "equiv_cov_delta": equiv["phi"]["coverage_uniformity"] - rat_equiv_cov,
            "equiv_relock_delta": equiv["phi"]["relocking_depth"] - rat_equiv_relock,
        },
        "noble": {
            "scale_gain": noble_scale - cosine_scale,
            "equiv_gain": noble_equiv - cosine_equiv,
            "scale_pass_random": noble_scale > max(irr_scales),
            "equiv_pass_random": noble_equiv > max(irr_equivs),
            "scale_cov_delta": split["noble"]["coverage_uniformity"] - rat_scale_cov,
            "scale_relock_delta": split["noble"]["relocking_depth"] - rat_scale_relock,
            "equiv_cov_delta": equiv["noble"]["coverage_uniformity"] - rat_equiv_cov,
            "equiv_relock_delta": equiv["noble"]["relocking_depth"] - rat_equiv_relock,
        },
    }

    opened_by = []
    for name, vals in candidates.items():
        hit_gain_ok = vals["scale_gain"] >= 0.03 or vals["equiv_gain"] >= 0.03
        random_ok = vals["scale_pass_random"] and vals["equiv_pass_random"]
        activation_ok = (
            vals["scale_cov_delta"] >= 0.05
            or vals["scale_relock_delta"] >= 3.0
            or vals["equiv_cov_delta"] >= 0.05
            or vals["equiv_relock_delta"] >= 3.0
        )
        if hit_gain_ok and random_ok and activation_ok:
            opened_by.append(name)

    return {
        "candidate_summary": candidates,
        "open_p5_p6": bool(opened_by),
        "opened_by": opened_by,
    }


def build_model_from_run(run_dir: Path, device: torch.device):
    config = json.loads((run_dir / "config.json").read_text())
    ckpt = torch.load(run_dir / "best_model.pt", map_location=device)

    audio_encoder_name = config.get("audio_encoder", "baseline")
    image_encoder_name = config.get("image_encoder", "baseline")
    proj_dim = int(config.get("proj_dim", 256))

    audio_enc = build_lissajous_audio_encoder(audio_encoder_name, output_dim=256).to(device)
    image_enc = build_lissajous_image_encoder(image_encoder_name, output_dim=256).to(device)
    proj_audio = ProjectionHead(input_dim=256, hidden_dim=512, output_dim=proj_dim, n_layers=3).to(device)
    proj_image = ProjectionHead(input_dim=256, hidden_dim=512, output_dim=proj_dim, n_layers=3).to(device)

    audio_enc.load_state_dict(ckpt["audio_enc"])
    image_enc.load_state_dict(ckpt["image_enc"])
    proj_audio.load_state_dict(ckpt["proj_audio"])
    proj_image.load_state_dict(ckpt["proj_image"])

    audio_enc.eval()
    image_enc.eval()
    proj_audio.eval()
    proj_image.eval()

    return config, audio_enc, image_enc, proj_audio, proj_image


def build_or_load_pool_cache(
    args,
    data_dir: Path,
    output_dir: Path,
) -> Dict:
    if args.pool_cache:
        cache = torch.load(args.pool_cache, map_location="cpu")
        torch.save(cache, output_dir / "probe_pool_cache.pt")
        with open(output_dir / "query_ids.json", "w") as f:
            json.dump({k: v["query_ids"] for k, v in cache["splits"].items()}, f, indent=2)
        return cache

    test_ds = LissajousDataset(data_dir / "test")
    gallery_ds = load_reduced_gallery_dataset(data_dir)
    gallery_key_idx = _build_key_index(gallery_ds)
    gallery_equiv_idx = _build_equiv_key_index(gallery_ds)

    scale_ds = load_ood_dataset(data_dir, "scale_ood")
    equiv_ds = load_ood_dataset(data_dir, "equiv_ood")

    cache = {
        "config": {
            "pool_size": args.pool_size,
            "seed": args.seed,
            "n_queries_iid": min(args.n_queries_iid, len(test_ds)),
            "n_queries_ood": min(args.n_queries_ood, len(scale_ds)),
        },
        "splits": {}
    }

    cache["splits"]["iid"] = precompute_probe_pools(
        "iid",
        query_ds=test_ds,
        gallery_ds=test_ds,
        pool_size=args.pool_size,
        n_queries=args.n_queries_iid,
        seed=args.seed,
    )
    cache["splits"]["scale_ood"] = precompute_probe_pools(
        "scale_ood",
        query_ds=scale_ds,
        gallery_ds=gallery_ds,
        pool_size=args.pool_size,
        n_queries=args.n_queries_ood,
        seed=args.seed,
        positive_finder=lambda q_idx: find_positives_scale_ood(q_idx, scale_ds, gallery_key_idx),
    )
    cache["splits"]["equiv_ood"] = precompute_probe_pools(
        "equiv_ood",
        query_ds=equiv_ds,
        gallery_ds=gallery_ds,
        pool_size=args.pool_size,
        n_queries=args.n_queries_ood,
        seed=args.seed,
        positive_finder=lambda q_idx: find_positives_equiv_ood(q_idx, equiv_ds, gallery_equiv_idx),
    )

    torch.save(cache, output_dir / "probe_pool_cache.pt")
    with open(output_dir / "query_ids.json", "w") as f:
        json.dump({k: v["query_ids"] for k, v in cache["splits"].items()}, f, indent=2)
    return cache


def main():
    parser = argparse.ArgumentParser(description="E3-P4: Probe regime on frozen P2 latents")
    parser.add_argument("--data", default="data/escalon3/bundled")
    parser.add_argument("--run", required=True, help="Path to P2 run dir with best_model.pt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pool-cache", default=None)
    parser.add_argument("--k-hit", type=int, default=10)
    parser.add_argument("--k-walk", type=int, default=32)
    parser.add_argument("--m-directions", type=int, default=4)
    parser.add_argument("--pool-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-queries-iid", type=int, default=480)
    parser.add_argument("--n-queries-ood", type=int, default=500)
    parser.add_argument("--max-exemplars", type=int, default=50)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    run_dir = Path(args.run)
    data_dir = Path(args.data)

    config, audio_enc, image_enc, proj_audio, proj_image = build_model_from_run(run_dir, device)
    probe_families = _build_probe_families()
    probe_engine = ProbeEngine(
        dim=int(config["proj_dim"]),
        n_directions=args.m_directions,
        k_walk=args.k_walk,
        seed=args.seed,
    )

    cache = build_or_load_pool_cache(args, data_dir, output_dir)

    # Datasets / embeddings
    test_ds = LissajousDataset(data_dir / "test")
    gallery_ds = load_reduced_gallery_dataset(data_dir)
    scale_ds = load_ood_dataset(data_dir, "scale_ood")
    equiv_ds = load_ood_dataset(data_dir, "equiv_ood")

    # Extract embeddings (returned on CPU by extract_embeddings) then move to GPU for fast probe math
    print("Extracting embeddings...")
    test_a, test_i = extract_embeddings(
        audio_enc, image_enc, proj_audio, proj_image, test_ds, device,
        num_workers=args.num_workers,
    )
    gallery_a, gallery_i = extract_embeddings(
        audio_enc, image_enc, proj_audio, proj_image, gallery_ds, device,
        num_workers=args.num_workers,
    )
    scale_a, scale_i = extract_embeddings(
        audio_enc, image_enc, proj_audio, proj_image, scale_ds, device,
        num_workers=args.num_workers,
    )
    equiv_a, equiv_i = extract_embeddings(
        audio_enc, image_enc, proj_audio, proj_image, equiv_ds, device,
        num_workers=args.num_workers,
    )

    # Move embeddings to GPU for fast probe computation
    probe_device = device
    test_a, test_i = test_a.to(probe_device), test_i.to(probe_device)
    gallery_a, gallery_i = gallery_a.to(probe_device), gallery_i.to(probe_device)
    scale_a, scale_i = scale_a.to(probe_device), scale_i.to(probe_device)
    equiv_a, equiv_i = equiv_a.to(probe_device), equiv_i.to(probe_device)
    print(f"Embeddings on {probe_device}")

    split_payloads = {
        "iid": {
            "query_ds": test_ds,
            "candidate_ds": test_ds,
            "query_audio": test_a,
            "query_image": test_i,
            "cand_audio": test_a,
            "cand_image": test_i,
        },
        "scale_ood": {
            "query_ds": scale_ds,
            "candidate_ds": gallery_ds,
            "query_audio": scale_a,
            "query_image": scale_i,
            "cand_audio": gallery_a,
            "cand_image": gallery_i,
        },
        "equiv_ood": {
            "query_ds": equiv_ds,
            "candidate_ds": gallery_ds,
            "query_audio": equiv_a,
            "query_image": equiv_i,
            "cand_audio": gallery_a,
            "cand_image": gallery_i,
        },
    }

    results = {
        "run_dir": str(run_dir),
        "baseline_name": run_dir.name,
        "probe_config": {
            "k_hit": args.k_hit,
            "k_walk": args.k_walk,
            "m_directions": args.m_directions,
            "pool_size": args.pool_size,
            "seed": args.seed,
            "probe_families": probe_families,
            "gate_metrics": {
                "primary": [
                    "scale_ood a2i ratio_hit@10",
                    "equiv_ood a2i equiv_hit@10",
                ],
                "rational_gate_value": "family_mean",
            },
        },
        "splits": {},
    }
    exemplars = {}

    # Count total evaluations for progress bar
    n_total_evals = len(split_payloads) * 2 * len(probe_families)
    pbar = tqdm(total=n_total_evals, desc="P4 Probe Regime")

    for split_name, payload in split_payloads.items():
        results["splits"][split_name] = {}
        exemplars[split_name] = {}
        query_ids = cache["splits"][split_name]["query_ids"]
        pool_cache = cache["splits"][split_name]
        for direction in ("a2i", "i2a"):
            if direction == "a2i":
                query_embs = payload["query_audio"]
                candidate_embs = payload["cand_image"]
            else:
                query_embs = payload["query_image"]
                candidate_embs = payload["cand_audio"]

            results["splits"][split_name][direction] = {}
            exemplars[split_name][direction] = {}
            for family_name, family_cfg in probe_families.items():
                pbar.set_postfix_str(f"{split_name}/{direction}/{family_name}")
                family_result, family_exemplars = evaluate_family(
                    family_name,
                    family_cfg,
                    query_embs=query_embs,
                    candidate_embs=candidate_embs,
                    split_name=split_name,
                    query_ds=payload["query_ds"],
                    pool_cache=pool_cache,
                    query_ids=query_ids,
                    probe_engine=probe_engine,
                    k_hit=args.k_hit,
                    max_exemplars=args.max_exemplars,
                )
                results["splits"][split_name][direction][family_name] = family_result
                exemplars[split_name][direction][family_name] = family_exemplars
                pbar.update(1)

    pbar.close()
    attach_activation_gain(results)
    gate = assess_gate(results)
    results["gate_assessment"] = gate

    with open(output_dir / "probe_results.json", "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    with open(output_dir / "probe_config.json", "w") as f:
        json.dump(results["probe_config"], f, indent=2, default=_json_default)
    with open(output_dir / "probe_exemplars.json", "w") as f:
        json.dump(exemplars, f, indent=2, default=_json_default)

    csv_rows = summarize_for_csv(results)
    with open(output_dir / "probe_family_summary.csv", "w", newline="") as f:
        fieldnames = [
            "split", "direction", "family", "metric_value",
            "activation_gain", "coverage_uniformity",
            "relocking_depth", "best_alpha",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # Persist local copy of query ids for convenience
    with open(output_dir / "query_ids.json", "w") as f:
        json.dump({k: v["query_ids"] for k, v in cache["splits"].items()}, f, indent=2)

    print(f"Saved P4 probe results to {output_dir}")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
