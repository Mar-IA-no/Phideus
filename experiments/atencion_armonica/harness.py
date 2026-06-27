"""Harness de métricas — Fase 0 Atención Armónica.

Todas las métricas pairwise se computan SOLO sobre el triángulo superior (i<j) de pares
VÁLIDOS (excluye diagonal, padding, near-collisions sub-ε). Codex r2: nunca diagonal ni padding.

    F1 pairwise (umbral 0.5)        — PRIMARIA, sin knobs
    AP / AUPRC, ROC-AUC             — secundarias threshold-free
    ARI (clustering por τ)          — τ se elige SOLO en val (nunca en test)

extract_valid_pairs: de [B,N,N] a flat (logit, target) sobre i<j válidos, con mixture index.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import (
    adjusted_rand_score, average_precision_score, f1_score, roc_auc_score,
)


def extract_valid_pairs(
    logits: torch.Tensor, target: torch.Tensor, pair_valid: torch.Tensor,
    token_mask: torch.Tensor,
) -> List[Dict]:
    """Por cada muestra del batch, extrae los pares i<j VÁLIDOS.

    Returns: lista (una entrada por muestra) de dicts {logit:[P], target:[P]} (numpy).
    Solo i<j (triángulo superior), solo pair_valid True. pair_valid ya excluye diagonal,
    near-collisions; acá además se restringe a i<j y a tokens válidos.
    """
    B, N, _ = logits.shape
    iu, ju = np.triu_indices(N, k=1)               # i<j
    out = []
    lg = logits.detach().cpu().numpy()
    tg = target.detach().cpu().numpy()
    pv = pair_valid.detach().cpu().numpy()
    tm = token_mask.detach().cpu().numpy()
    for b in range(B):
        # doble protección (Codex): pair_valid ∩ ambos tokens válidos (excluye padding)
        tmask_pair = tm[b][:, None] & tm[b][None, :]   # [N,N]
        valid = (pv[b] & tmask_pair)[iu, ju]           # [P_upper]
        out.append({
            "logit": lg[b][iu, ju][valid].astype(np.float64),
            "target": tg[b][iu, ju][valid].astype(np.float64),
        })
    return out


def pairwise_metrics(logit: np.ndarray, target: np.ndarray, thresh: float = 0.5) -> Dict:
    """F1 (umbral), AP, ROC-AUC sobre arrays flat de pares válidos i<j."""
    if len(target) == 0:
        return {"f1": float("nan"), "ap": float("nan"), "roc_auc": float("nan"), "n_pairs": 0}
    prob = 1.0 / (1.0 + np.exp(-logit))
    pred = (prob >= thresh).astype(np.int64)
    tgt = target.astype(np.int64)
    f1 = f1_score(tgt, pred, zero_division=0)
    # AP / ROC-AUC requieren ambas clases presentes
    if tgt.min() == tgt.max():
        ap = float("nan")
        roc = float("nan")
    else:
        ap = average_precision_score(tgt, prob)
        roc = roc_auc_score(tgt, prob)
    # baselines interpretables (Codex Bajo #5): predecir todo-mismo / todo-distinto
    all_same = f1_score(tgt, np.ones_like(tgt), zero_division=0)
    all_diff = f1_score(tgt, np.zeros_like(tgt), zero_division=0)
    return {
        "f1": float(f1), "ap": float(ap), "roc_auc": float(roc),
        "all_same_f1": float(all_same), "all_diff_f1": float(all_diff),
        "n_pairs": int(len(tgt)),
    }


def cluster_from_pairs(
    logit_mat: np.ndarray, token_mask: np.ndarray, tau: float, pair_valid: np.ndarray,
) -> np.ndarray:
    """Connected components sobre la matriz de pares binarizada a τ. Devuelve labels [N_valid].

    Solo sobre tokens válidos. Umbral τ sobre prob = sigmoid(logit). Los pares INVÁLIDOS
    (near-collision sub-ε, nunca supervisados) se fuerzan a NO-edge (Codex Alto #2). Simétrico.
    """
    valid_idx = np.where(token_mask)[0]
    n = len(valid_idx)
    if n == 0:
        return np.array([], dtype=np.int64)
    sub = logit_mat[np.ix_(valid_idx, valid_idx)]
    pv_sub = pair_valid[np.ix_(valid_idx, valid_idx)]
    prob = 1.0 / (1.0 + np.exp(-sub))
    adj = (prob >= tau) & pv_sub                     # excluir edges en pares inválidos
    np.fill_diagonal(adj, True)                      # cada nodo consigo mismo
    adj = adj | adj.T                                # simétrico
    _, labels = connected_components(adj, directed=False)
    return labels


def ari_for_mixture(
    logit_mat: np.ndarray, token_mask: np.ndarray, true_source: np.ndarray,
    tau: float, pair_valid: np.ndarray,
) -> float:
    """ARI del clustering inducido (τ) vs la partición verdadera por fuente, sobre tokens válidos."""
    valid_idx = np.where(token_mask)[0]
    if len(valid_idx) < 2:
        return float("nan")
    pred_labels = cluster_from_pairs(logit_mat, token_mask, tau, pair_valid)
    true_labels = true_source[valid_idx]
    return float(adjusted_rand_score(true_labels, pred_labels))


def random_baseline_f1(target: np.ndarray, seed: int = 0) -> float:
    """F1 esperado de predicción aleatoria con la prior de la clase positiva."""
    if len(target) == 0:
        return float("nan")
    rng = np.random.RandomState(seed)
    p_pos = target.mean()
    pred = (rng.rand(len(target)) < p_pos).astype(np.int64)
    return float(f1_score(target.astype(np.int64), pred, zero_division=0))


def bootstrap_diff_ci(
    per_mix_a: List[Dict], per_mix_b: List[Dict], metric: str = "f1",
    n_boot: int = 1000, seed: int = 42,
) -> Dict:
    """Bootstrap PAREADO sobre mezclas de test: resamplea mezclas (mismas para a y b),
    poolea sus pares, computa metric para a y b, acumula la diferencia (a - b).

    per_mix_a/b: listas alineadas por mezcla, cada entrada {logit:[P], target:[P]}.
    Devuelve mean_diff, ci95, frac_positive.
    """
    assert len(per_mix_a) == len(per_mix_b)
    # alineación explícita por mixture_id si los dicts lo traen (Codex Bajo #4):
    if per_mix_a and "mixture_id" in per_mix_a[0] and "mixture_id" in per_mix_b[0]:
        ids_a = [m["mixture_id"] for m in per_mix_a]
        ids_b = [m["mixture_id"] for m in per_mix_b]
        assert ids_a == ids_b, "bootstrap pareado: listas desalineadas por mixture_id"
    M = len(per_mix_a)
    rng = np.random.RandomState(seed)

    def pooled_metric(per_mix, idxs):
        lg = np.concatenate([per_mix[i]["logit"] for i in idxs]) if len(idxs) else np.array([])
        tg = np.concatenate([per_mix[i]["target"] for i in idxs]) if len(idxs) else np.array([])
        return pairwise_metrics(lg, tg)[metric]

    point_a = pooled_metric(per_mix_a, list(range(M)))
    point_b = pooled_metric(per_mix_b, list(range(M)))
    diffs = np.empty(n_boot)
    for t in range(n_boot):
        idxs = rng.randint(0, M, size=M)
        diffs[t] = pooled_metric(per_mix_a, idxs) - pooled_metric(per_mix_b, idxs)
    diffs = diffs[np.isfinite(diffs)]
    lo, hi = np.percentile(diffs, [2.5, 97.5]) if len(diffs) else (float("nan"), float("nan"))
    return {
        "metric": metric,
        "point_a": float(point_a), "point_b": float(point_b),
        "mean_diff": float(point_a - point_b),
        "ci95_lo": float(lo), "ci95_hi": float(hi),
        "frac_positive": float((diffs > 0).mean()) if len(diffs) else float("nan"),
        "n_mixtures": M,
    }
