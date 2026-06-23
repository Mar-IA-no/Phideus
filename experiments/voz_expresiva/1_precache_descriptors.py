#!/usr/bin/env python3
"""Pre-cache Familia A frame-level (V4-lin + H-series) alineada a 50 Hz para Fase 1.

Lee `wavlm_index.json` para conocer T_max y T_50Hz por utt. Genera:
    data/voz_expresiva/descriptors_cache/family_A.npy  (memmap [N, T_max, 12])

CPU only, paralelo con 14 workers (ProcessPoolExecutor).

Run:
    python experiments/voz_expresiva/1_precache_descriptors.py \\
        --wavlm-index data/voz_expresiva/wavlm_cache/wavlm_index.json \\
        --output-dir data/voz_expresiva/descriptors_cache \\
        --workers 14
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.voz_expresiva.descriptor_precache import (  # noqa: E402
    FAMILY_A_DIM,
    align_to_wavlm_length,
    extract_family_A_frame_level,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _worker(payload: dict) -> dict:
    """Worker: extract descriptor for one utt, align to T_50Hz."""
    try:
        desc = extract_family_A_frame_level(payload["wav_path"])
        desc_aligned = align_to_wavlm_length(desc, payload["T_50Hz"])
        return {
            "row_idx": payload["row_idx"],
            "ok": True,
            "desc": desc_aligned,
        }
    except Exception as exc:
        return {
            "row_idx": payload["row_idx"],
            "ok": False,
            "error": str(exc),
            "T_50Hz": payload["T_50Hz"],
        }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wavlm-index", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--workers", type=int, default=14)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = json.loads(Path(args.wavlm_index).read_text())
    N = index["N"]
    T_max = index["T_max"]
    utterances = index["utterances"]

    logger.info("Loaded WavLM index: N=%d, T_max=%d", N, T_max)

    # Allocate memmap
    desc_path = out_dir / "family_A.npy"
    desc_mm = np.memmap(
        desc_path, dtype=np.float32, mode="w+",
        shape=(N, T_max, FAMILY_A_DIM),
    )
    # Init with NaN to make missing data obvious
    desc_mm[:] = np.nan
    desc_mm.flush()

    payloads = [
        {
            "row_idx": u["row_idx"],
            "wav_path": u["wav_path"],
            "T_50Hz": u["T_50Hz"],
        }
        for u in utterances
    ]

    failures = []
    t0 = time.time()
    n_done = 0
    n_total = len(payloads)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, p): p for p in payloads}
        for fut in as_completed(futures):
            res = fut.result()
            n_done += 1
            if res["ok"]:
                row = res["row_idx"]
                desc = res["desc"]
                if desc.shape[0] != T_max:
                    # Pad if shorter (shouldn't happen after align)
                    pad = np.full(
                        (T_max - desc.shape[0], FAMILY_A_DIM), np.nan, dtype=np.float32,
                    )
                    desc = np.vstack([desc, pad])
                desc_mm[row, :, :] = desc[:T_max]
            else:
                failures.append(res)

            if n_done % max(1, n_total // 50) == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta = (n_total - n_done) / rate if rate > 0 else 0
                logger.info("Progress %d/%d (%.1f utt/s, ETA %.1f min, %d fails)",
                            n_done, n_total, rate, eta / 60, len(failures))

    desc_mm.flush()
    elapsed = time.time() - t0
    size_mb = (N * T_max * FAMILY_A_DIM * 4) / 1e6
    logger.info("Done in %.1f min. Cache size: %.1f MB. Failures: %d",
                elapsed / 60, size_mb, len(failures))

    if failures:
        (out_dir / "descriptor_failures.json").write_text(json.dumps(failures, indent=2))
        logger.warning("Wrote failure log to %s", out_dir / "descriptor_failures.json")


if __name__ == "__main__":
    main()
