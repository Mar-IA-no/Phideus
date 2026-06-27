"""Dataset + collate + splits de runs (ID/OOD-poly/OOD-regime) — Fase 0 Atención Armónica.

Lee el pool (mixtures.jsonl) y arma los tres runs del plan:
    ID          train/val/test = poly{1,2,3} × {easy,hard} (split por mezcla)
    OOD-poly    train/val = poly{1,2} ; test = poly{3}
    OOD-regime  train/val = easy(poly{1,2,3}) ; test = hard(poly{1,2,3})

Split por MEZCLA (no por pico). Split fijo por SPLIT_SEED — NO depende del seed del modelo
(los seeds de modelo solo varían init de pesos). Mismos splits para los 5 modelos.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.atencion_armonica.peak_tokens import N_PAIR_CONT_FEATS, mixture_to_arrays

SPLIT_SEED = 7            # congelado: define los splits, independiente del seed de modelo
VAL_FRAC = 0.10
TEST_FRAC_ID = 0.17       # ID: ~24k/3k/6k de 36k


def load_pool(jsonl_path: str | Path) -> List[Dict]:
    """Carga todas las mezclas del JSONL."""
    recs = []
    with Path(jsonl_path).open() as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def _split_indices(n: int, fracs: Tuple[float, ...], seed: int) -> List[np.ndarray]:
    """Permuta [0,n) y la parte en bloques según fracs (deben sumar ≤ 1; el resto es el primer bloque)."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    sizes = [int(round(fr * n)) for fr in fracs]
    first = n - sum(sizes)
    blocks = []
    start = 0
    for sz in [first] + sizes:
        blocks.append(perm[start:start + sz])
        start += sz
    return blocks


def make_run_splits(pool: List[Dict], run: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Devuelve (train, val, test) según el run. Split por mezcla, determinístico (SPLIT_SEED)."""
    by_cell: Dict[Tuple[int, str], List[Dict]] = {}
    for m in pool:
        by_cell.setdefault((m["polyphony"], m["regime"]), []).append(m)

    train, val, test = [], [], []

    if run == "ID":
        # split cada celda en train/val/test
        for (poly, regime), recs in by_cell.items():
            tr, va, te = _split_indices(len(recs), (VAL_FRAC, TEST_FRAC_ID), SPLIT_SEED)
            train += [recs[i] for i in tr]
            val += [recs[i] for i in va]
            test += [recs[i] for i in te]

    elif run == "OOD-poly":
        # train/val de poly 1,2 (todos los regímenes); test = poly 3
        for (poly, regime), recs in by_cell.items():
            if poly in (1, 2):
                tr, va = _split_indices(len(recs), (VAL_FRAC,), SPLIT_SEED)
                train += [recs[i] for i in tr]
                val += [recs[i] for i in va]
            elif poly == 3:
                test += recs

    elif run == "OOD-regime":
        # train/val de easy (todas las polifonías); test = hard
        for (poly, regime), recs in by_cell.items():
            if regime == "easy":
                tr, va = _split_indices(len(recs), (VAL_FRAC,), SPLIT_SEED)
                train += [recs[i] for i in tr]
                val += [recs[i] for i in va]
            elif regime == "hard":
                test += recs

    else:
        raise ValueError(f"Unknown run: {run}")

    return train, val, test


class GroupingDataset(Dataset):
    """Cada item: arrays de una mezcla (computados on-the-fly, N≤24 es barato)."""

    def __init__(self, records: List[Dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        return mixture_to_arrays(self.records[idx])


def collate_grouping(batch: List[Dict]) -> Dict:
    """Pad N a N_max del batch. Construye máscaras de token y de par (excluye padding,
    diagonal y near-collisions)."""
    B = len(batch)
    N_max = max(item["n_peaks"] for item in batch)
    F = N_PAIR_CONT_FEATS

    tokens = torch.zeros(B, N_max, 2, dtype=torch.float32)
    token_mask = torch.zeros(B, N_max, dtype=torch.bool)
    pair_cont = torch.zeros(B, N_max, N_max, F, dtype=torch.float32)
    ratio_class_id = torch.zeros(B, N_max, N_max, dtype=torch.long)
    target = torch.zeros(B, N_max, N_max, dtype=torch.float32)
    pair_valid = torch.zeros(B, N_max, N_max, dtype=torch.bool)

    polyphony, regime, mixture_id, n_peaks = [], [], [], []

    for b, item in enumerate(batch):
        N = item["n_peaks"]
        tokens[b, :N] = torch.from_numpy(item["tokens"])
        token_mask[b, :N] = True
        pair_cont[b, :N, :N] = torch.from_numpy(item["pair_cont"])
        ratio_class_id[b, :N, :N] = torch.from_numpy(item["ratio_class_id"])
        target[b, :N, :N] = torch.from_numpy(item["target"])
        pair_valid[b, :N, :N] = torch.from_numpy(item["pair_valid"])
        polyphony.append(item["polyphony"])
        regime.append(item["regime"])
        mixture_id.append(item["mixture_id"])
        n_peaks.append(N)

    return {
        "tokens": tokens,                       # [B,N_max,2]
        "token_mask": token_mask,               # [B,N_max]
        "pair_cont": pair_cont,                 # [B,N_max,N_max,F]
        "ratio_class_id": ratio_class_id,       # [B,N_max,N_max]
        "target": target,                       # [B,N_max,N_max]
        "pair_valid": pair_valid,               # [B,N_max,N_max] (ya excluye diag y near-collision; padding excluido abajo)
        "polyphony": polyphony,
        "regime": regime,
        "mixture_id": mixture_id,
        "n_peaks": n_peaks,
    }
