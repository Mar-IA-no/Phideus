#!/usr/bin/env python3
"""Fase 0B — clasificador clásico descriptor-only sobre ESD, LOSO CV.

Configuraciones: 8 feature subsets × 2 norm conditions × 2 classifiers × 10 folds.
Pregunta operativa: ¿A+D supera D-only en clasificación speaker-independent?

Outputs:
    data/voz_expresiva/0B/uar_results.json     # UAR per (norm, config, clf, speaker)
    data/voz_expresiva/0B/predictions.npz      # y_true, y_pred, meta per task

Run:
    python experiments/voz_expresiva/0B_classify.py \\
        --input data/esd/descriptors_0A_en.npz \\
        --output-dir data/voz_expresiva/0B \\
        --workers 14
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configs and grids (frozen per plan)
# ---------------------------------------------------------------------------

CONFIGS: Dict[str, List[str]] = {
    "D-only": ["D"],
    "A-only": ["A"],
    "B-only": ["B"],
    "C-only": ["C"],
    "A+B": ["A", "B"],
    "A+D": ["A", "D"],
    "C+D": ["C", "D"],
    "A+B+D": ["A", "B", "D"],
}

LOGREG_GRID = [0.01, 0.1, 1.0, 10.0]                        # 4 points (C)
SVM_GRID = [(C, g) for C in (0.1, 1.0, 10.0) for g in ("scale", "auto")]  # 6 points

NORM_CONDITIONS = ("strict", "adapt")
CLASSIFIERS = ("logreg", "svm_rbf")
ADAPT_REPEATS = 3
ADAPT_SEEDS = (42, 43, 44)
N_CALIB = 25
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Per-speaker z-score (label-agnostic)
# ---------------------------------------------------------------------------

def zscore_within_groups(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Per-group z-score using each group's own stats. NaN-safe."""
    X = X.astype(np.float64)
    out = np.empty_like(X)
    for g in np.unique(groups):
        mask = groups == g
        sub = X[mask]
        mu = np.nanmean(sub, axis=0, keepdims=True)
        sd = np.nanstd(sub, axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        out[mask] = (sub - mu) / sd
    return out


def impute_nans(X: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """Impute NaNs with provided per-column medians."""
    X = X.copy()
    bad = ~np.isfinite(X)
    if bad.any():
        col_idx = np.where(bad.any(axis=0))[0]
        for c in col_idx:
            X[bad[:, c], c] = medians[c]
    return X


# ---------------------------------------------------------------------------
# Model fit + grid search on single val speaker
# ---------------------------------------------------------------------------

def _fit_and_score_logreg(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> Tuple[np.ndarray, float, dict]:
    """Grid search over LogReg C on val, evaluate on test. Returns (y_pred, val_uar, best_params)."""
    best_C = None
    best_val_uar = -1.0
    for C in LOGREG_GRID:
        clf = LogisticRegression(
            C=C, penalty="l2", solver="lbfgs",
            max_iter=2000, class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=1,
        )
        clf.fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)
        uar = recall_score(y_val, y_val_pred, average="macro", zero_division=0)
        if uar > best_val_uar:
            best_val_uar = uar
            best_C = C
    # Refit best on train, evaluate on test
    clf = LogisticRegression(
        C=best_C, penalty="l2", solver="lbfgs",
        max_iter=2000, class_weight="balanced",
        random_state=RANDOM_SEED, n_jobs=1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred, best_val_uar, {"C": best_C}


def _fit_and_score_svm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> Tuple[np.ndarray, float, dict]:
    """Grid search over SVM RBF (C, gamma) on val, evaluate on test."""
    best_params = None
    best_val_uar = -1.0
    for C, gamma in SVM_GRID:
        clf = SVC(
            C=C, gamma=gamma, kernel="rbf",
            class_weight="balanced", random_state=RANDOM_SEED,
        )
        clf.fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)
        uar = recall_score(y_val, y_val_pred, average="macro", zero_division=0)
        if uar > best_val_uar:
            best_val_uar = uar
            best_params = {"C": C, "gamma": gamma}
    # Refit
    clf = SVC(
        C=best_params["C"], gamma=best_params["gamma"], kernel="rbf",
        class_weight="balanced", random_state=RANDOM_SEED,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred, best_val_uar, best_params


def fit_and_score(
    clf_name: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> Tuple[np.ndarray, float, dict]:
    if clf_name == "logreg":
        return _fit_and_score_logreg(X_train, y_train, X_val, y_val, X_test, y_test)
    if clf_name == "svm_rbf":
        return _fit_and_score_svm(X_train, y_train, X_val, y_val, X_test, y_test)
    raise ValueError(f"Unknown clf: {clf_name}")


# ---------------------------------------------------------------------------
# Single task = one (fold, norm, config, classifier)
# ---------------------------------------------------------------------------

def _build_features(
    families: Dict[str, np.ndarray], family_keys: List[str],
) -> np.ndarray:
    return np.concatenate([families[k] for k in family_keys], axis=1)


def run_task(
    fold_idx: int,
    test_speaker: str,
    val_speaker: str,
    train_speakers: List[str],
    norm_condition: str,
    config_name: str,
    family_keys: List[str],
    clf_name: str,
    families: Dict[str, np.ndarray],
    speaker_ids: np.ndarray,
    emotion_labels: np.ndarray,
) -> dict:
    """Run one (fold, norm, config, clf) experiment. Returns a result dict."""
    X = _build_features(families, family_keys)
    y = emotion_labels

    train_mask = np.isin(speaker_ids, train_speakers)
    val_mask = speaker_ids == val_speaker
    test_mask = speaker_ids == test_speaker

    X_train_raw = X[train_mask].copy()
    y_train = y[train_mask].copy()
    X_val_raw = X[val_mask].copy()
    y_val = y[val_mask].copy()
    X_test_raw_full = X[test_mask].copy()
    y_test_full = y[test_mask].copy()

    # Impute NaNs using train medians (computed before per-speaker norm)
    with np.errstate(invalid="ignore"):
        medians = np.nanmedian(X_train_raw, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    X_train_raw = impute_nans(X_train_raw, medians)
    X_val_raw = impute_nans(X_val_raw, medians)
    X_test_raw_full = impute_nans(X_test_raw_full, medians)

    # Per-speaker z-score on train (always)
    X_train_z = zscore_within_groups(X_train_raw, speaker_ids[train_mask])

    # Z-score val: depends on norm condition (same logic as test)
    # N-strict val: no per-speaker norm; just StandardScaler from train below
    # N-adapt val: per-speaker norm using val_speaker stats from random calib
    if norm_condition == "strict":
        X_val_z_input = X_val_raw.copy()
    elif norm_condition == "adapt":
        rng_val = np.random.RandomState(RANDOM_SEED + fold_idx * 100)
        calib_idx = rng_val.choice(len(X_val_raw), size=min(N_CALIB, len(X_val_raw)), replace=False)
        calib = X_val_raw[calib_idx]
        mu = np.mean(calib, axis=0)
        sd = np.std(calib, axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        # Remove calibration from val (no leakage between calibration and val)
        mask_remain = np.ones(len(X_val_raw), dtype=bool)
        mask_remain[calib_idx] = False
        X_val_raw = X_val_raw[mask_remain]
        y_val = y_val[mask_remain]
        X_val_z_input = (X_val_raw - mu) / sd
    else:
        raise ValueError(f"Unknown norm: {norm_condition}")

    # Fit StandardScaler on per-speaker-normalized train, apply to val + test
    scaler = StandardScaler().fit(X_train_z)
    X_train_scaled = scaler.transform(X_train_z)
    X_val_scaled = scaler.transform(X_val_z_input)

    # --- N-strict test path ---
    if norm_condition == "strict":
        X_test_input = X_test_raw_full.copy()  # no per-speaker norm
        X_test_scaled = scaler.transform(X_test_input)
        y_pred, val_uar, best_params = fit_and_score(
            clf_name, X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test_full,
        )
        test_uar = recall_score(y_test_full, y_pred, average="macro", zero_division=0)
        test_f1 = f1_score(y_test_full, y_pred, average="macro", zero_division=0)
        test_acc = accuracy_score(y_test_full, y_pred)
        return {
            "fold_idx": fold_idx,
            "test_speaker": test_speaker,
            "val_speaker": val_speaker,
            "norm_condition": "strict",
            "config_name": config_name,
            "clf_name": clf_name,
            "best_params": best_params,
            "val_uar": float(val_uar),
            "test_uar": float(test_uar),
            "test_f1_macro": float(test_f1),
            "test_accuracy": float(test_acc),
            "n_test": int(len(y_test_full)),
            "y_true": y_test_full.tolist(),
            "y_pred": y_pred.tolist(),
        }

    # --- N-adapt test path: 3 repeats with different calibration samples ---
    repeat_results = []
    for rep_seed in ADAPT_SEEDS:
        rng = np.random.RandomState(rep_seed + fold_idx * 100)
        calib_idx = rng.choice(len(X_test_raw_full), size=min(N_CALIB, len(X_test_raw_full)),
                               replace=False)
        calib = X_test_raw_full[calib_idx]
        mu = np.mean(calib, axis=0)
        sd = np.std(calib, axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)

        eval_mask = np.ones(len(X_test_raw_full), dtype=bool)
        eval_mask[calib_idx] = False
        X_test_eval = X_test_raw_full[eval_mask]
        y_test_eval = y_test_full[eval_mask]
        X_test_eval_norm = (X_test_eval - mu) / sd
        X_test_scaled = scaler.transform(X_test_eval_norm)

        y_pred, val_uar, best_params = fit_and_score(
            clf_name, X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test_eval,
        )
        rep_uar = recall_score(y_test_eval, y_pred, average="macro", zero_division=0)
        rep_f1 = f1_score(y_test_eval, y_pred, average="macro", zero_division=0)
        rep_acc = accuracy_score(y_test_eval, y_pred)
        repeat_results.append({
            "rep_seed": int(rep_seed),
            "best_params": best_params,
            "val_uar": float(val_uar),
            "test_uar": float(rep_uar),
            "test_f1_macro": float(rep_f1),
            "test_accuracy": float(rep_acc),
            "n_test": int(len(y_test_eval)),
            "y_true": y_test_eval.tolist(),
            "y_pred": y_pred.tolist(),
        })

    # Aggregate per-speaker: mean across 3 repeats
    uars = [r["test_uar"] for r in repeat_results]
    f1s = [r["test_f1_macro"] for r in repeat_results]
    accs = [r["test_accuracy"] for r in repeat_results]
    return {
        "fold_idx": fold_idx,
        "test_speaker": test_speaker,
        "val_speaker": val_speaker,
        "norm_condition": "adapt",
        "config_name": config_name,
        "clf_name": clf_name,
        "test_uar": float(np.mean(uars)),
        "test_uar_std_rep": float(np.std(uars)),
        "test_f1_macro": float(np.mean(f1s)),
        "test_accuracy": float(np.mean(accs)),
        "repeats": repeat_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="NPZ from 0A_extract.py")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--workers", type=int, default=14)
    args = p.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s", in_path)
    d = np.load(in_path, allow_pickle=True)
    families = {
        "A": d["family_A_pooled"].astype(np.float64),
        "B": d["family_B"].astype(np.float64),
        "C": d["family_C_pooled"].astype(np.float64),
        "D": d["family_D_egemaps"].astype(np.float64),
    }
    speaker_ids = d["speaker_ids"]
    emotion_labels = d["emotion_labels"]

    speakers = sorted(set(speaker_ids.tolist()))
    n_speakers = len(speakers)
    logger.info("Speakers: %s", speakers)
    logger.info("Emotions: %s", sorted(set(emotion_labels.tolist())))
    logger.info("N utterances: %d", len(speaker_ids))

    # Build all tasks
    tasks = []
    for fold_idx, test_spk in enumerate(speakers):
        val_spk = speakers[(fold_idx + 1) % n_speakers]
        train_spks = [s for s in speakers if s not in (test_spk, val_spk)]
        for norm_cond in NORM_CONDITIONS:
            for cfg_name, fam_keys in CONFIGS.items():
                for clf in CLASSIFIERS:
                    tasks.append({
                        "fold_idx": fold_idx,
                        "test_speaker": test_spk,
                        "val_speaker": val_spk,
                        "train_speakers": train_spks,
                        "norm_condition": norm_cond,
                        "config_name": cfg_name,
                        "family_keys": fam_keys,
                        "clf_name": clf,
                    })
    logger.info("Total tasks: %d", len(tasks))

    t0 = time.time()
    results = Parallel(n_jobs=args.workers, backend="loky", verbose=10)(
        delayed(run_task)(
            t["fold_idx"], t["test_speaker"], t["val_speaker"], t["train_speakers"],
            t["norm_condition"], t["config_name"], t["family_keys"], t["clf_name"],
            families, speaker_ids, emotion_labels,
        )
        for t in tasks
    )
    elapsed = time.time() - t0
    logger.info("All tasks done in %.1f min", elapsed / 60)

    # Save UAR results (excluding y_true/y_pred from light JSON)
    light = []
    predictions = []
    for r in results:
        light_entry = {k: v for k, v in r.items() if k not in ("y_true", "y_pred", "repeats")}
        if r["norm_condition"] == "adapt":
            light_entry["repeats_summary"] = [
                {k: rep[k] for k in ("rep_seed", "test_uar", "test_f1_macro",
                                     "test_accuracy", "best_params", "val_uar")}
                for rep in r["repeats"]
            ]
        light.append(light_entry)

        if r["norm_condition"] == "strict":
            predictions.append({
                "norm_condition": "strict",
                "config_name": r["config_name"],
                "clf_name": r["clf_name"],
                "test_speaker": r["test_speaker"],
                "y_true": r["y_true"],
                "y_pred": r["y_pred"],
            })
        else:
            # For adapt, store the 3 repeats' y_true/y_pred
            for rep in r["repeats"]:
                predictions.append({
                    "norm_condition": "adapt",
                    "config_name": r["config_name"],
                    "clf_name": r["clf_name"],
                    "test_speaker": r["test_speaker"],
                    "rep_seed": rep["rep_seed"],
                    "y_true": rep["y_true"],
                    "y_pred": rep["y_pred"],
                })

    (out_dir / "uar_results.json").write_text(json.dumps(light, indent=2))
    logger.info("Saved %s", out_dir / "uar_results.json")

    # Save predictions in NPZ for downstream analysis
    np.savez_compressed(
        out_dir / "predictions.npz",
        predictions=np.array(json.dumps(predictions), dtype=object),
    )
    logger.info("Saved %s", out_dir / "predictions.npz")
    logger.info("Done. Elapsed: %.1f min", elapsed / 60)


if __name__ == "__main__":
    main()
