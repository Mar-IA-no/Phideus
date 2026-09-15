"""Bounded synchronous resource guard; CPU mode never queries a GPU.

Artifact scans are rate-limited, not repeated per tensor. The runner calls a
forced check at phase boundaries; counters are guard observations, not a claim
of continuous physical monitoring or a sandbox against arbitrary allocations.
"""
from __future__ import annotations

from pathlib import Path
import resource
import shutil
import time

GIB = 1024**3


class ResourceGuard:
    def __init__(self, root, *, ram_bytes=8*GIB, artifact_bytes=16*GIB,
                 minimum_free_bytes=2*GIB, gpu_peak=None, vram_bytes=6*GIB,
                 clock=time.monotonic):
        self.root = Path(root)
        self.ram_bytes, self.artifact_bytes = ram_bytes, artifact_bytes
        self.minimum_free_bytes, self.vram_bytes = minimum_free_bytes, vram_bytes
        if any(type(v) is not int or v <= 0 for v in (ram_bytes, artifact_bytes, minimum_free_bytes, vram_bytes)):
            raise ValueError("resource limits must be positive integer bytes")
        if gpu_peak is not None and not callable(gpu_peak):
            raise ValueError("GPU metrics require an explicitly admitted callback")
        self.gpu_peak, self.clock = gpu_peak, clock
        self.last_disk = None
        self.observed = {"ram_peak_bytes": 0, "artifact_bytes": 0,
                         "free_bytes": None, "vram_peak_bytes": None}

    def __call__(self, *, force=False):
        self.observed["ram_peak_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
        if self.observed["ram_peak_bytes"] > self.ram_bytes:
            raise MemoryError("process RSS peak exceeded measurement budget")
        now = self.clock()
        if force or self.last_disk is None or now-self.last_disk >= 30:
            total = 0
            for path in self.root.rglob("*"):
                if path.is_symlink():
                    raise ValueError("resource accounting cannot follow symlinks")
                if path.is_file():
                    total += path.stat().st_size
            self.observed["artifact_bytes"] = total
            self.observed["free_bytes"] = shutil.disk_usage(self.root).free
            self.last_disk = now
        if self.observed["artifact_bytes"] > self.artifact_bytes:
            raise OSError("measurement artifact budget exceeded")
        if self.observed["free_bytes"] < self.minimum_free_bytes:
            raise OSError("insufficient disk safety margin")
        if self.gpu_peak is not None:
            peak = self.gpu_peak()
            if type(peak) is not int or peak < 0:
                raise ValueError("invalid admitted GPU resource observation")
            self.observed["vram_peak_bytes"] = peak
            if peak > self.vram_bytes:
                raise MemoryError("GPU reserved-memory peak exceeded measurement budget")
        return dict(self.observed)
