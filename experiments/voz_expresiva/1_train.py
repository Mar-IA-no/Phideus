#!/usr/bin/env python3
"""Fase 1 training: LOSO 10-fold × {N-strict, N-adapt} × {none, concat, film, xattn} × 3 seeds.

Asume caches WavLM y Familia A pre-generados. Lee desde
`data/voz_expresiva/wavlm_cache/` y `data/voz_expresiva/descriptors_cache/`.

Outputs (en --output-dir, default data/voz_expresiva/1/):
    uar_results.json          # 1 record por (norm, config, fold, seed)
    calib_manifest.json       # 25 utts de calib N-adapt por fold (seed 42)
    embeddings/<fold>_<config>_<norm>_<seed>.npy  # [N_test_utts_eval, 1024]
    predictions/<fold>_<config>_<norm>_<seed>.json # y_true, y_pred per test utt

Run (largo, en tmux):
    python experiments/voz_expresiva/1_train.py \\
        --cache-root data/voz_expresiva \\
        --output-dir data/voz_expresiva/1 \\
        --epochs 30 --batch-size 64 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.voz_expresiva.esd_dataset import (  # noqa: E402
    EMOTION_TO_LABEL,
    ESDCachedDataset,
    collate_padded,
    compute_descriptor_zscore,
    load_cache,
)
from src.voz_expresiva.wavlm_injection import WavLMInjectionClassifier  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


SEEDS = (42, 123, 456)
CONFIGS = ("none", "concat", "film", "xattn")  # 'none' = WavLM-only baseline
NORMS = ("strict", "adapt")
N_CALIB = 25
CALIB_SEED = 42
N_CLASSES = 5
EARLY_STOP_PATIENCE = 5


# ---------------------------------------------------------------------------
# Speaker pool helpers
# ---------------------------------------------------------------------------

def get_speaker_pool(utterances: List[Dict]) -> List[str]:
    """Return sorted list of unique speaker_ids present in the index."""
    return sorted({u["speaker_id"] for u in utterances})


# ---------------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------------

def get_fold_speakers(
    fold_idx: int, speaker_pool: List[str],
) -> Tuple[str, str, List[str]]:
    """LOSO fold k: test = pool[k], val = pool[(k+1)%N], train = rest."""
    test_spk = speaker_pool[fold_idx]
    val_spk = speaker_pool[(fold_idx + 1) % len(speaker_pool)]
    train_spks = [s for s in speaker_pool if s not in (test_spk, val_spk)]
    return test_spk, val_spk, train_spks


def utterances_of(utterances: List[Dict], speaker_id: str) -> List[int]:
    return [u["row_idx"] for u in utterances if u["speaker_id"] == speaker_id]


def utterances_of_many(utterances: List[Dict], speakers: List[str]) -> List[int]:
    s = set(speakers)
    return [u["row_idx"] for u in utterances if u["speaker_id"] in s]


# ---------------------------------------------------------------------------
# Calibration manifest (N-adapt traceability)
# ---------------------------------------------------------------------------

def _speaker_calib_seed(spk: str, base_seed: int) -> int:
    """Stable deterministic seed derived from (base_seed, speaker_id).

    Independent of speaker_pool order; adding/removing speakers does NOT
    change the calibration set of the others.
    """
    h = sha256(f"{base_seed}:{spk}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)  # 32-bit seed


def build_calib_manifest(
    utterances: List[Dict],
    output_path: Path,
    speaker_pool: List[str],
) -> Dict[str, Dict]:
    """Pre-compute calib utts for each fold (test speaker × seed 42). Persistent.

    Uses speaker-derived effective seed (B2 fix): each speaker gets an
    independent RNG seeded by sha256(f"{CALIB_SEED}:{speaker_id}"). This
    avoids the prior bug where reinstantiating RandomState(CALIB_SEED) in
    the loop selected the SAME 25 sentence_ids for every speaker (all
    speakers share identical inventory order).

    Returns dict {test_speaker: {calib_seed (base), calib_seed_effective,
                                  n_calib, calib_row_idx, sentence_ids,
                                  emotions, calib_hash}}.
    """
    if output_path.exists():
        logger.info("Loading existing calib_manifest from %s", output_path)
        cached = json.loads(output_path.read_text())
        # Validate compatibility with current speaker_pool and seeding policy
        cached_pool = sorted(cached.keys())
        if cached_pool != sorted(speaker_pool):
            raise RuntimeError(
                f"Stale calib_manifest at {output_path}: cached pool {cached_pool} "
                f"differs from current speaker_pool {sorted(speaker_pool)}. "
                f"Remove the file to regenerate, or use a different --output-dir."
            )
        for spk in speaker_pool:
            entry = cached[spk]
            expected_eff = _speaker_calib_seed(spk, CALIB_SEED)
            actual_eff = entry.get("calib_seed_effective")
            if actual_eff is None or int(actual_eff) != expected_eff:
                raise RuntimeError(
                    f"Stale calib_manifest at {output_path}: speaker {spk} has "
                    f"calib_seed_effective={actual_eff}, expected {expected_eff} "
                    f"under current seeding policy. Remove the file to regenerate."
                )
        logger.info("Cached manifest validated against current seeding policy")
        return cached

    manifest = {}
    for spk in speaker_pool:
        spk_utts = [u for u in utterances if u["speaker_id"] == spk]
        spk_seed = _speaker_calib_seed(spk, CALIB_SEED)
        rng = np.random.RandomState(spk_seed)
        idx_in_speaker = rng.choice(len(spk_utts), size=N_CALIB, replace=False)
        idx_in_speaker = sorted(idx_in_speaker.tolist())

        calib_records = [spk_utts[i] for i in idx_in_speaker]
        calib_row_ids = [u["row_idx"] for u in calib_records]
        calib_sids = [u["sentence_id"] for u in calib_records]
        calib_emotions = [u["emotion"] for u in calib_records]
        # Hash for trazabilidad
        h_input = ",".join([f"{row}:{sid}:{emo}" for row, sid, emo in
                            zip(calib_row_ids, calib_sids, calib_emotions)])
        calib_hash = sha256(h_input.encode("utf-8")).hexdigest()[:16]

        manifest[spk] = {
            "calib_seed": CALIB_SEED,
            "calib_seed_effective": int(spk_seed),
            "n_calib": N_CALIB,
            "calib_row_idx": calib_row_ids,
            "sentence_ids": calib_sids,
            "emotions": calib_emotions,
            "calib_hash": calib_hash,
        }

    output_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote calib_manifest with %d speakers to %s", len(manifest), output_path)
    return manifest


# ---------------------------------------------------------------------------
# Z-score policies
# ---------------------------------------------------------------------------

def build_zscore_maps(
    norm_condition: str,
    train_indices: List[int],
    val_indices: List[int],
    test_eval_indices: List[int],
    calib_indices: List[int],
    descriptors: np.ndarray,
    lengths: np.ndarray,
    utterances: List[Dict],
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Build per-row z-score (mean, std) maps based on norm condition.

    N-strict:
        train: per-speaker stats from train_indices for that speaker
        val: per-speaker stats from train_indices for val speaker — NOT AVAILABLE
             → val gets NO normalization (mismatched train regime).
             Actually for consistency we apply train-pooled stats to val.
        test_eval: NO per-speaker normalization. Apply train-pooled stats only.

    N-adapt:
        train: per-speaker stats from each train speaker's own utts
        val: train-pooled stats (val speaker also gets train-pool — not used differently)
        test_eval: per-test-speaker stats computed from calib_indices (the 25 utts).
                   calib_indices excluded from test_eval.

    Returns:
        (mean_map, std_map): each {row_idx: ndarray[12]}.
    """
    # Per-train-speaker stats
    train_by_spk: Dict[str, List[int]] = {}
    for idx in train_indices:
        spk = utterances[idx]["speaker_id"]
        train_by_spk.setdefault(spk, []).append(idx)

    per_spk_stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for spk, idxs in train_by_spk.items():
        mu, sd = compute_descriptor_zscore(descriptors, lengths, idxs)
        per_spk_stats[spk] = (mu, sd)

    # Train pool: aggregate stats over all train indices (mean of means weighted by count)
    # Simpler: use direct compute over all train frames
    train_pool_mu, train_pool_sd = compute_descriptor_zscore(
        descriptors, lengths, train_indices,
    )

    mean_map: Dict[int, np.ndarray] = {}
    std_map: Dict[int, np.ndarray] = {}

    # Train: per-speaker
    for idx in train_indices:
        spk = utterances[idx]["speaker_id"]
        mu, sd = per_spk_stats[spk]
        mean_map[idx] = mu
        std_map[idx] = sd

    # Val: train pool stats
    for idx in val_indices:
        mean_map[idx] = train_pool_mu
        std_map[idx] = train_pool_sd

    if norm_condition == "strict":
        # Test eval: train pool stats (no per-speaker)
        for idx in test_eval_indices:
            mean_map[idx] = train_pool_mu
            std_map[idx] = train_pool_sd
    elif norm_condition == "adapt":
        # Test eval: per-test-speaker stats from calib
        calib_mu, calib_sd = compute_descriptor_zscore(
            descriptors, lengths, calib_indices,
        )
        for idx in test_eval_indices:
            mean_map[idx] = calib_mu
            std_map[idx] = calib_sd

    return mean_map, std_map


# ---------------------------------------------------------------------------
# Training loop per single run
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_run(
    config: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
    """Train one (fold, norm, config, seed) run.

    Returns:
        metrics dict, y_true [N_test], y_pred [N_test], embeddings [N_test, 1024].
    """
    model = WavLMInjectionClassifier(
        mechanism=config, feature_dim=1024, descriptor_dim=12, n_classes=N_CLASSES,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = torch.nn.CrossEntropyLoss(weight=None)  # class_weight applied later via sample weighting if needed

    best_val_uar = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        # Train
        model.train()
        for batch in train_loader:
            features = batch["features"].to(device, non_blocking=True)
            descriptor = batch["descriptor"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(features, descriptor if config != "none" else None, mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        # Val
        val_uar = _evaluate_uar(model, val_loader, device, config)
        if val_uar > best_val_uar:
            best_val_uar = val_uar
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test eval
    y_true, y_pred, embeddings = _evaluate_full(model, test_loader, device, config)
    test_uar = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    test_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    test_acc = float((y_true == y_pred).mean())

    return {
        "val_uar": float(best_val_uar),
        "test_uar": test_uar,
        "test_f1_macro": test_f1,
        "test_accuracy": test_acc,
        "n_test": int(len(y_true)),
        "epochs_trained": epoch + 1,
    }, y_true, y_pred, embeddings


@torch.no_grad()
def _evaluate_uar(model, loader, device, config) -> float:
    model.eval()
    all_true, all_pred = [], []
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        descriptor = batch["descriptor"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].numpy()

        logits = model(features, descriptor if config != "none" else None, mask)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_true.append(labels)
        all_pred.append(preds)
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return float(recall_score(y_true, y_pred, average="macro", zero_division=0))


@torch.no_grad()
def _evaluate_full(model, loader, device, config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_true, all_pred, all_emb = [], [], []
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        descriptor = batch["descriptor"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].numpy()

        # Get embedding then classify
        emb = model.get_embedding(features, descriptor if config != "none" else None, mask)
        logits = model.classifier(emb)
        preds = logits.argmax(dim=-1).cpu().numpy()

        all_true.append(labels)
        all_pred.append(preds)
        all_emb.append(emb.cpu().numpy().astype(np.float32))
    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_emb, axis=0),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", required=True, help="data/voz_expresiva/")
    p.add_argument("--wavlm-subdir", default="wavlm_cache",
                   help="Subdir of --cache-root for WavLM cache (default: wavlm_cache)")
    p.add_argument("--desc-subdir", default="descriptors_cache",
                   help="Subdir of --cache-root for descriptor cache (default: descriptors_cache)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit-folds", type=int, default=None,
                   help="Debug: run only N folds")
    p.add_argument("--limit-seeds", type=int, default=None,
                   help="Debug: run only first N seeds")
    p.add_argument(
        "--limit-norms", default=None,
        choices=("strict", "adapt"),
        help="Restrict to a single norm condition. Used for partial reruns "
             "(e.g. EN N-adapt only post calib_manifest fix).",
    )
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "embeddings").mkdir(exist_ok=True)
    (out_dir / "predictions").mkdir(exist_ok=True)

    logger.info("Loading caches...")
    cache = load_cache(
        args.cache_root,
        wavlm_subdir=args.wavlm_subdir,
        desc_subdir=args.desc_subdir,
    )
    utterances = cache["utterances"]
    descriptors = cache["descriptors"]
    lengths = cache["wavlm_lengths"]

    speaker_pool = get_speaker_pool(utterances)
    logger.info("Speaker pool (%d): %s", len(speaker_pool), speaker_pool)

    # Calib manifest
    calib_manifest = build_calib_manifest(
        utterances, out_dir / "calib_manifest.json", speaker_pool,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    folds_to_run = range(len(speaker_pool))
    if args.limit_folds:
        folds_to_run = range(args.limit_folds)
    seeds_to_run = SEEDS
    if args.limit_seeds:
        seeds_to_run = SEEDS[: args.limit_seeds]
    norms_to_run = NORMS
    if args.limit_norms:
        norms_to_run = (args.limit_norms,)
        logger.info("Restricted to norm condition: %s", args.limit_norms)

    all_results = []
    t0_global = time.time()

    for fold_idx in folds_to_run:
        test_spk, val_spk, train_spks = get_fold_speakers(fold_idx, speaker_pool)
        test_indices_full = utterances_of(utterances, test_spk)
        val_indices = utterances_of(utterances, val_spk)
        train_indices = utterances_of_many(utterances, train_spks)
        calib_indices = calib_manifest[test_spk]["calib_row_idx"]
        calib_set = set(calib_indices)

        logger.info(
            "Fold %d/%d: test=%s val=%s train=%d test_full=%d calib=%d",
            fold_idx + 1, len(speaker_pool), test_spk, val_spk,
            len(train_indices), len(test_indices_full), len(calib_indices),
        )

        for seed in seeds_to_run:
            for norm in norms_to_run:
                # Build test eval indices
                if norm == "strict":
                    test_eval_indices = test_indices_full
                else:  # adapt
                    test_eval_indices = [i for i in test_indices_full if i not in calib_set]

                # Build z-score maps
                mean_map, std_map = build_zscore_maps(
                    norm, train_indices, val_indices, test_eval_indices,
                    calib_indices, descriptors, lengths, utterances,
                )

                # Datasets
                train_ds = ESDCachedDataset(
                    cache["wavlm_features"], lengths, descriptors, utterances,
                    indices=train_indices, per_row_zscore=(mean_map, std_map),
                )
                val_ds = ESDCachedDataset(
                    cache["wavlm_features"], lengths, descriptors, utterances,
                    indices=val_indices, per_row_zscore=(mean_map, std_map),
                )
                test_ds = ESDCachedDataset(
                    cache["wavlm_features"], lengths, descriptors, utterances,
                    indices=test_eval_indices, per_row_zscore=(mean_map, std_map),
                )

                train_loader = DataLoader(
                    train_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, collate_fn=collate_padded,
                    pin_memory=True,
                )
                val_loader = DataLoader(
                    val_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate_padded,
                    pin_memory=True,
                )
                test_loader = DataLoader(
                    test_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate_padded,
                    pin_memory=True,
                )

                for config in CONFIGS:
                    t_run_start = time.time()
                    set_seed(seed)

                    metrics, y_true, y_pred, embeddings = train_one_run(
                        config, train_loader, val_loader, test_loader,
                        device, args.epochs,
                    )

                    elapsed = time.time() - t_run_start

                    # Save embeddings + predictions
                    tag = f"fold{fold_idx}_{config}_{norm}_seed{seed}"
                    np.save(out_dir / "embeddings" / f"{tag}.npy", embeddings)
                    (out_dir / "predictions" / f"{tag}.json").write_text(json.dumps({
                        "fold_idx": fold_idx, "test_speaker": test_spk,
                        "config": config, "norm_condition": norm, "seed": seed,
                        "row_idx": [int(test_eval_indices[i]) for i in range(len(y_true))],
                        "y_true": y_true.tolist(),
                        "y_pred": y_pred.tolist(),
                    }))

                    record = {
                        "fold_idx": fold_idx,
                        "test_speaker": test_spk,
                        "val_speaker": val_spk,
                        "config": config,
                        "norm_condition": norm,
                        "seed": int(seed),
                        "calib_seed": CALIB_SEED if norm == "adapt" else None,
                        "calib_seed_effective": (
                            int(calib_manifest[test_spk]["calib_seed_effective"])
                            if norm == "adapt" else None
                        ),
                        "n_calib": N_CALIB if norm == "adapt" else None,
                        "calib_hash": calib_manifest[test_spk]["calib_hash"] if norm == "adapt" else None,
                        **metrics,
                        "wall_seconds": float(elapsed),
                    }
                    all_results.append(record)

                    logger.info(
                        "  %s norm=%s config=%s seed=%d → UAR=%.3f F1=%.3f (%.1fs)",
                        test_spk, norm, config, seed,
                        record["test_uar"], record["test_f1_macro"], elapsed,
                    )

                    # Flush results periodically
                    (out_dir / "uar_results.json").write_text(json.dumps(all_results, indent=2))

    total = time.time() - t0_global
    logger.info("All runs done in %.1f h", total / 3600)
    (out_dir / "uar_results.json").write_text(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
