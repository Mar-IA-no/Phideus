"""Observable-only shared-source grid witnesses, never physical certificates."""
from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np

CENTS = 1200/np.log(2)
TIE_CENTS = 1e-9
GRID_VERSION = "shared-source-grid-v1"


class SourceFitter:
    def __init__(self):
        self.grid = np.geomspace(1e-5, .02, 1025, dtype=np.float64)
        self.grid.flags.writeable = False
        self.grid_hash = hashlib.sha256(self.grid.astype("<f8").tobytes()).hexdigest()
        self.templates = {}

    def prepare(self, size):
        if size not in range(3, 9):
            raise ValueError("templates require 3..8 members")
        if size not in self.templates:
            indices = np.array(list(combinations(range(1, 9), size)), dtype=np.int64)
            values = np.log(indices[:, None, :])+.5*np.log1p(self.grid[None, :, None]*indices[:, None, :]**2)
            means = values.mean(axis=-1)
            centered = values-means[:, :, None]
            for array in (indices, means, centered):
                array.flags.writeable = False
            self.templates[size] = indices, means, centered
        return self.templates[size]

    def _witness(self, residuals, indices, offsets, grid_indices):
        minimum = float(residuals.min())
        near = residuals <= minimum+TIE_CENTS
        # C-order is lexicographic indices followed by increasing beta.
        assignment, cell = np.argwhere(near)[0]
        outside = residuals[~near]
        return {"minimum_cents": minimum, "co_minimum_count": int(near.sum()),
                "next_level_gap_cents": float(outside.min()-minimum) if len(outside) else None,
                "witness_indices": indices[assignment].tolist(),
                "witness_beta": float(self.grid[grid_indices[cell]]),
                "witness_grid_index": int(grid_indices[cell]),
                "witness_offset": float(offsets[assignment, cell]),
                "witness_residual_cents": float(residuals[assignment, cell]),
                "index_authority": "WITNESS_NOT_IDENTIFIED",
                "offset_authority": "CENTERED_LOG_GAUGE"}

    def fit(self, observed_q, members):
        q = np.asarray(observed_q)
        selected = np.asarray(members)
        if (q.ndim != 1 or q.dtype != np.float32 or not np.isfinite(q).all()
                or not 1 <= len(q) <= 32 or selected.ndim != 1
                or not np.issubdtype(selected.dtype, np.integer) or len(selected) == 0
                or len(np.unique(selected)) != len(selected) or np.any(selected < 0)
                or np.any(selected >= len(q))):
            raise ValueError("expected finite observed float32 q and unique valid member indices")
        selected = np.sort(selected)
        ordered = selected[np.argsort(q[selected], kind="stable")]
        result = {"member_indices": sorted(selected.tolist()), "frequency_order_members": ordered.tolist(),
                  "observed_q32": q[ordered].tolist(), "size": len(selected),
                  "grid_version": GRID_VERSION, "grid_sha256": self.grid_hash}
        if len(selected) < 3:
            return {**result, "status": "UNDERCONSTRAINED"}
        if len(selected) > 8:
            return {**result, "status": "OUTSIDE_DECLARED_CARDINALITY"}
        observed = q[ordered].astype(np.float64)
        indices, means, templates = self.prepare(len(selected))
        centered = observed-observed.mean()
        differences = templates-centered[None, None, :]
        residuals = CENTS*np.sqrt(np.mean(differences*differences, axis=-1))
        offsets = observed.mean()-means
        fine = self._witness(residuals, indices, offsets, np.arange(1025))
        coarse = self._witness(residuals[:, ::4], indices, offsets[:, ::4], np.arange(0, 1025, 4))
        gap = coarse["minimum_cents"]-fine["minimum_cents"]
        if gap < -TIE_CENTS:
            raise ArithmeticError("nested grid refinement worsened the minimum")
        return {**result, "status": "GRID_WITNESS_APPROXIMATE", "fine": fine, "coarse": coarse,
                "coarse_minus_fine_cents": gap, "continuous_minimum_status": "NOT_COMPUTED"}


class GroupFitCache:
    """A scene identity cannot silently be rebound to another observation."""
    def __init__(self, fitter=None):
        self.fitter = fitter if fitter is not None else SourceFitter()
        self.observation_hashes, self.values = {}, {}
        self.hits = self.misses = 0

    def fit(self, split, scene_id, q, members):
        q = np.asarray(q)
        selected = np.asarray(members)
        if (q.dtype != np.float32 or q.ndim != 1 or not np.isfinite(q).all()
                or not 1 <= len(q) <= 32 or selected.ndim != 1 or not len(selected)
                or not np.issubdtype(selected.dtype, np.integer)
                or len(np.unique(selected)) != len(selected)
                or np.any(selected < 0) or np.any(selected >= len(q))):
            raise ValueError("cache requires the actual finite float32 observation")
        identity = (split, scene_id)
        digest = hashlib.sha256(q.astype("<f4").tobytes()).hexdigest()
        if identity in self.observation_hashes and self.observation_hashes[identity] != digest:
            raise ValueError("scene identity rebound to different ordered observation")
        self.observation_hashes[identity] = digest
        key = (split, scene_id, tuple(sorted(members)), self.fitter.grid_hash)
        if key not in self.values:
            self.values[key] = self.fitter.fit(q, members)
            self.misses += 1
        else:
            self.hits += 1
        return self.values[key]
