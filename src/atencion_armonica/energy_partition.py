"""Amplitude-only exact-cover diagnostic; no frequency, labels, torch or GPU.

Public generator prior: each source has 4..8 positive partials and unit
squared-amplitude energy. This is NOT a general-purpose harmonic grouper.
"""

from __future__ import annotations

import math

import numpy as np


def accessible_amplitudes(amplitudes: np.ndarray) -> np.ndarray:
    """Decode precisely the float32 log-amplitude channel of compute_tokens."""
    values = np.asarray(amplitudes, dtype=np.float64)
    return np.exp(np.log(np.maximum(values, 1e-12)).astype(np.float32).astype(np.float64))


def canonical_partition(blocks) -> tuple[tuple[int, ...], ...]:
    """Quotient out block labels, not the identities of observed peaks."""
    return tuple(sorted(tuple(sorted(int(i) for i in block)) for block in blocks))


def _indices(mask: int, n: int) -> tuple[int, ...]:
    return tuple(i for i in range(n) if mask & (1 << i))


def _subsets(energies: np.ndarray):
    n = len(energies)
    # At most 4096 entries per half; fsum is the authoritative summation.
    return sorted(
        (math.fsum(float(energies[i]) for i in _indices(mask, n)), mask)
        for mask in range(1 << n)
    )


def valid_partition(amplitudes, blocks, k: int, tolerance: float) -> bool:
    """Validate witnesses without truth; no claim of uniqueness."""
    n = len(amplitudes)
    if len(blocks) != k or any(not 4 <= len(block) <= 8 for block in blocks):
        return False
    flattened = [i for block in blocks for i in block]
    if sorted(flattened) != list(range(n)):
        return False
    return all(
        abs(math.fsum(float(amplitudes[i]) * float(amplitudes[i]) for i in block) - 1.0) <= tolerance
        for block in blocks
    )


def solve_energy_partition(
    amplitudes,
    *,
    tolerance: float,
    max_candidates: int = 20_000,
    max_nodes: int = 50_000,
) -> dict:
    """Only observation is an unordered amplitude vector; kwargs are public.

    UNIQUE/NO_PARTITION require exhaustive search. MULTIPLE returns two
    witnesses, not the full solution set. LIMIT never certifies uniqueness.
    """
    if np.iscomplexobj(amplitudes):
        raise ValueError("amplitudes must be real")
    a = np.asarray(amplitudes, dtype=np.float64)
    if a.ndim != 1 or not len(a) or not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise ValueError("expected a nonempty finite positive 1-D amplitude vector")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if any(type(cap) is not int or cap < 1 for cap in (max_candidates, max_nodes)):
        raise ValueError("caps must be positive integers")
    result = {
        "status": None, "n": len(a), "k": None, "k_candidates": [],
        "total_energy": None, "candidate_masks": [], "candidate_count": 0,
        "candidates_exhaustive": False, "nodes": 0,
        "search_exhaustive": False, "solutions": [],
    }
    if len(a) > 24:
        result["status"] = "LIMIT_INPUT_SIZE"
        return result
    # Scalar multiplication matches the final witness checker exactly.
    energies = np.array([float(x) * float(x) for x in a])
    if not np.all(np.isfinite(energies)):
        raise ValueError("squared amplitudes must remain finite")
    total = math.fsum(energies)
    ks = [k for k in range(math.ceil(len(a) / 8), len(a) // 4 + 1)
          if abs(total - k) <= k * tolerance]
    result.update(total_energy=total, k_candidates=ks)
    if len(ks) != 1:
        result["status"] = "PRIOR_VIOLATION" if not ks else "AMBIGUOUS_K"
        return result
    k = result["k"] = ks[0]
    middle = len(a) // 2
    left = _subsets(energies[:middle])
    right = _subsets(energies[middle:])
    right_sums = np.array([entry[0] for entry in right])
    # Conservative MITM window; final fsum/tolerance, not this pad, admits a set.
    pad = 64 * np.finfo(np.float64).eps * max(1.0, total)
    candidates = set()
    for left_sum, left_mask in left:
        low = np.searchsorted(right_sums, 1 - tolerance - pad - left_sum, side="left")
        high = np.searchsorted(right_sums, 1 + tolerance + pad - left_sum, side="right")
        for pos in range(int(low), int(high)):
            mask = left_mask | (right[pos][1] << middle)
            if not 4 <= mask.bit_count() <= 8:
                continue
            energy = math.fsum(float(energies[i]) for i in _indices(mask, len(a)))
            if abs(energy - 1) > tolerance or mask in candidates:
                continue
            if len(candidates) == max_candidates:
                result.update(status="LIMIT_CANDIDATES", candidate_count=len(candidates),
                              candidate_masks=sorted(candidates))
                return result
            candidates.add(mask)
    candidate_masks = sorted(candidates)
    result.update(candidate_count=len(candidates), candidate_masks=candidate_masks,
                  candidates_exhaustive=True)
    by_peak = [[mask for mask in candidate_masks if mask & (1 << i)]
               for i in range(len(a))]
    solutions = set()
    limited = False

    def search(remaining: int, blocks: tuple[int, ...]):
        nonlocal limited
        if limited or len(solutions) >= 2:
            return
        if result["nodes"] >= max_nodes:
            limited = True
            return
        result["nodes"] += 1
        missing = k - len(blocks)
        if not remaining:
            if not missing:
                partition = canonical_partition(_indices(mask, len(a)) for mask in blocks)
                if not valid_partition(a, partition, k, tolerance):
                    raise AssertionError("invalid exact-cover witness")
                solutions.add(partition)
            return
        if missing <= 0 or not 4 * missing <= remaining.bit_count() <= 8 * missing:
            return
        first = (remaining & -remaining).bit_length() - 1
        for mask in by_peak[first]:
            if mask & remaining == mask:
                search(remaining ^ mask, blocks + (mask,))
            if limited or len(solutions) >= 2:
                break

    search((1 << len(a)) - 1, ())
    result["solutions"] = [list(map(list, blocks)) for blocks in sorted(solutions)]
    result["search_exhaustive"] = not limited and len(solutions) < 2
    result["status"] = ("MULTIPLE" if len(solutions) >= 2 else
                        "LIMIT_NODES" if limited else
                        "UNIQUE" if solutions else "NO_PARTITION")
    return result
