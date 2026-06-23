#!/usr/bin/env python3
"""Fase 0A — extract descriptors over ESD.

Reads ESD inventory via `ESDLoader`, computes all 4 descriptor families per
utterance in parallel, and persists the result as a single NPZ.

Output NPZ schema (keys):
    speaker_ids        : (N,) U4         "0011", "0012", ...
    emotion_labels     : (N,) U10        "Angry", "Happy", ...
    sentence_ids       : (N,) U10
    languages          : (N,) U2         "EN" / "ZH"
    family_A_pooled    : (N, 48)         48-dim Phideus-ratio (pooled)
    family_B           : (N, 9)          9-dim voice quality
    family_C_pooled    : (N, 32)         32-dim A4-16k (pooled)
    family_D_egemaps   : (N, 88)         eGeMAPSv02 functionals
    compound_pooled    : (N, 89)         A + B + C
    family_A_names     : (48,) str       column names
    family_B_names     : (9,)  str
    family_C_names     : (32,) str
    family_D_names     : (88,) str
    voice_quality_kind : (9,)  str       "direct" | "proxy"
    family_index       : (3, 3) int      [[A_lo, A_hi], [B_lo, B_hi], [C_lo, C_hi]]
    meta               : str             json metadata blob (sr, hop, dataset, etc.)

Run:
    python experiments/voz_expresiva/0A_extract.py \\
        --esd-root data/esd/raw/Emotional\\ Speech\\ Dataset \\
        --output data/esd/descriptors_0A_en.npz \\
        --language EN --workers 14
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np

# Ensure repo root on path so `import src.voz_expresiva` works when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.voz_expresiva import ESDLoader, compute_all_descriptors  # noqa: E402
from src.voz_expresiva.compound_descriptor import (  # noqa: E402
    build_feature_names,
    family_index_for_compound,
    voice_quality_kind_array,
)
from src.voz_expresiva.esd_loader import ESDUtterance  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _worker(utt_dict: dict) -> dict:
    """Worker-side: rebuild utterance, run extraction, return dict of arrays + meta."""
    utt_path = Path(utt_dict["wav_path"])
    try:
        vec = compute_all_descriptors(utt_path, include_egemaps=True)
        return {
            "ok": True,
            **utt_dict,
            "family_A": vec.family_A,
            "family_B": vec.family_B,
            "family_C": vec.family_C,
            "family_D": vec.family_D,
        }
    except Exception as exc:
        logger.warning("FAIL %s: %s", utt_path.name, exc)
        return {"ok": False, **utt_dict, "error": str(exc)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--esd-root", required=True,
                   help="ESD root containing speaker dirs 0001/..0020/")
    p.add_argument("--output", required=True, help="Output NPZ path")
    p.add_argument("--language", default="EN", choices=("EN", "ZH", "ALL"))
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--limit", type=int, default=None,
                   help="Optional max number of utterances (debug)")
    p.add_argument("--speakers", nargs="*", default=None,
                   help="Explicit speaker list (overrides --language)")
    p.add_argument("--emotions", nargs="*", default=None,
                   help="Explicit emotion list (default: all 5)")
    args = p.parse_args()

    loader = ESDLoader(
        root_dir=args.esd_root,
        language=args.language,
        speakers=args.speakers,
        emotions=args.emotions,
    )
    inventory = loader.list_all()
    if args.limit:
        inventory = inventory[: args.limit]

    summary = loader.summary()
    logger.info("ESD inventory: %d utterances over %d speakers, %d emotions",
                summary["n_utterances"], summary["n_speakers"], summary["n_emotions"])
    if args.limit:
        logger.info("Limit applied → %d utterances", len(inventory))

    # Convert ESDUtterance instances to plain dicts for pickling
    payloads = [
        {
            "wav_path": str(u.wav_path),
            "speaker_id": u.speaker_id,
            "emotion": u.emotion,
            "sentence_id": u.sentence_id,
            "language": u.language,
        }
        for u in inventory
    ]

    results: list[dict] = []
    failures: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, payload): payload for payload in payloads}
        n_done = 0
        n_total = len(futures)
        for fut in as_completed(futures):
            res = fut.result()
            n_done += 1
            if res["ok"]:
                results.append(res)
            else:
                failures.append(res)
            if n_done % max(1, n_total // 50) == 0:
                logger.info("Progress %d/%d (ok=%d, fail=%d)",
                            n_done, n_total, len(results), len(failures))

    if not results:
        logger.error("No successful extractions. Aborting.")
        sys.exit(1)

    # Sort by (speaker, emotion, sentence) for reproducibility
    results.sort(key=lambda r: (r["speaker_id"], r["emotion"], r["sentence_id"]))

    speaker_ids = np.array([r["speaker_id"] for r in results])
    emotion_labels = np.array([r["emotion"] for r in results])
    sentence_ids = np.array([r["sentence_id"] for r in results])
    languages = np.array([r["language"] for r in results])

    A = np.stack([r["family_A"] for r in results], axis=0)        # (N, 48)
    B = np.stack([r["family_B"] for r in results], axis=0)        # (N, 9)
    C = np.stack([r["family_C"] for r in results], axis=0)        # (N, 32)
    D = np.stack([r["family_D"] for r in results], axis=0)        # (N, 88)
    compound = np.concatenate([A, B, C], axis=1)                  # (N, 89)

    names = build_feature_names()
    fam_idx = family_index_for_compound()
    family_index_arr = np.array([
        [fam_idx["A"][0], fam_idx["A"][1]],
        [fam_idx["B"][0], fam_idx["B"][1]],
        [fam_idx["C"][0], fam_idx["C"][1]],
    ], dtype=np.int32)

    meta = {
        "dataset": "ESD",
        "language": args.language,
        "speakers": sorted(set(speaker_ids.tolist())),
        "emotions": sorted(set(emotion_labels.tolist())),
        "n_utterances": int(A.shape[0]),
        "n_failed": len(failures),
        "sample_rate": 16000,
        "f0_hop_samples": 160,
        "extractor_version": "voz_expresiva_0A_v1",
        "pooling_stats": ["mean", "std", "max", "min"],
        "family_dims": {
            "A": int(A.shape[1]),
            "B": int(B.shape[1]),
            "C": int(C.shape[1]),
            "D": int(D.shape[1]),
            "compound": int(compound.shape[1]),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        speaker_ids=speaker_ids,
        emotion_labels=emotion_labels,
        sentence_ids=sentence_ids,
        languages=languages,
        family_A_pooled=A,
        family_B=B,
        family_C_pooled=C,
        family_D_egemaps=D,
        compound_pooled=compound,
        family_A_names=np.array(names["A"]),
        family_B_names=np.array(names["B"]),
        family_C_names=np.array(names["C"]),
        family_D_names=np.array(names["D"]),
        voice_quality_kind=np.array(voice_quality_kind_array()),
        family_index=family_index_arr,
        meta=np.array(json.dumps(meta)),
    )

    logger.info("Saved %s — %d utterances, %d failed", out_path, A.shape[0], len(failures))
    if failures:
        log_path = out_path.with_suffix(".failures.json")
        log_path.write_text(json.dumps(failures, indent=2))
        logger.info("Failure log → %s", log_path)


if __name__ == "__main__":
    main()
