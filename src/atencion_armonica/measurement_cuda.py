"""Explicitly admitted frozen-model CUDA service; importing it never touches CUDA.

Models remain resident across units; weights, features, kernels and decisions
are unchanged. The caller owns a GPU-marked measured phase and a current device
ownership check. No implicit CPU fallback or remote dispatch is available.
"""
from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import importlib.metadata
import platform

import numpy as np

from .measurement_reuse import CHECKPOINTS, read_reference
from .partial_compatibility_cache import encoded


class CUDABackend:
    SETTINGS = {"tf32": False, "backbone_dtype": "float32", "fit_dtype": "float64",
                "grid": [257, 65, 4], "assignment_batch": 8}

    def __init__(self, controller, reused, *, source_snapshot, verify_gpu_ownership):
        if not callable(verify_gpu_ownership):
            raise ValueError("current GPU ownership verification is required")
        self.controller, self.reused = controller, deepcopy(reused)
        self.source_snapshot = deepcopy(source_snapshot)
        self.verify_gpu_ownership = verify_gpu_ownership
        self.torch, self.runtime = None, None
        self.models, self.heads = {}, {}
        self.fitter = None
        self.ownership = None
        self._phase, self._closed = None, False
        self.runtime_ref = None
        self._final_peak = 0

    def _assert_phase(self, *, require_receipt=True):
        if self._closed:
            raise PermissionError("closed CUDA service requires a fresh ownership check")
        active = self.controller._active
        if active is None:
            raise PermissionError("CUDA requires an active measured phase")
        started = self.controller.store.json(active)
        if not started["gpu"] or started["phase"] not in ("profile_gpu", "test_prediction"):
            raise PermissionError("phase does not admit measurement CUDA")
        self._verify_start_source(started)
        if self.torch is not None:
            if encoded(active) != encoded(self._phase):
                raise PermissionError("CUDA service cannot carry ownership into another phase")
            if not self.ownership or (require_receipt and self.runtime_ref is None):
                raise PermissionError("initialized CUDA service has no ownership receipt")
        return active

    def _verify_start_source(self, started):
        if started["phase"] == "profile_gpu":
            source = started["identity"]["source_snapshot"]
        else:
            receipt = self.controller.store.json(started["identity"]["freeze"])
            freeze, _ = self.controller.store._payload(receipt["payload"], receipt["identity"])
            source = freeze["source_snapshot"]
        if encoded(source) != encoded(self.source_snapshot):
            raise PermissionError("GPU execution snapshot differs from the admitted phase")

    def _start(self, check):
        check()
        active = self._assert_phase()
        if self.torch is not None:
            return
        self.ownership = self.verify_gpu_ownership()
        if not isinstance(self.ownership, dict) or not self.ownership:
            raise PermissionError("ownership verification returned no evidence")
        import torch
        actual = {"python": platform.python_version(), "numpy": np.__version__,
                  "scipy": importlib.metadata.version("scipy"),
                  "scikit-learn": importlib.metadata.version("scikit-learn"),
                  "torch": torch.__version__, "torch_distribution": importlib.metadata.version("torch")}
        if encoded(actual) != encoded(self.reused["runtime"]):
            raise ValueError("runtime differs from the frozen backbones")
        from .structured_source_runner import gpu_runtime
        self.runtime = gpu_runtime()  # Device query only after the explicit ownership gate.
        self.torch = torch
        self._phase = active
        self.runtime_ref = self.controller.store.publish_json(
            active["path"].rsplit("/", 1)[0]+"/cuda-runtime.json", {
                "schema": "measurement-cuda-execution-v1", "start": active,
                "source_snapshot": self.source_snapshot, "runtime": self.runtime,
                "ownership": self.ownership, "settings": self.SETTINGS})
        check()

    def execution(self, check):
        self._start(check)
        return deepcopy(self.runtime_ref)

    def validate_execution(self, ref):
        """Authenticate old producing attempts by CPU, without acquiring a new GPU."""
        store = self.controller.store
        row = store.json(ref)
        if (set(row) != {"schema", "start", "source_snapshot", "runtime", "ownership", "settings"}
                or row["schema"] != "measurement-cuda-execution-v1"
                or encoded(row["source_snapshot"]) != encoded(self.source_snapshot)
                or encoded(row["settings"]) != encoded(self.SETTINGS)
                or not isinstance(row["ownership"], dict) or not row["ownership"]
                or set(row["runtime"]) != {"torch", "numpy", "cuda", "cudnn", "device"}
                or row["runtime"]["torch"] != self.reused["runtime"]["torch"]
                or row["runtime"]["numpy"] != self.reused["runtime"]["numpy"]
                or row["runtime"]["device"] != "NVIDIA GeForce RTX 3090"):
            raise ValueError("GPU execution provenance differs")
        matches = [s for s, r in self.controller._starts() if encoded(r) == encoded(row["start"])]
        if len(matches) != 1 or not matches[0]["gpu"] or matches[0]["phase"] not in ("profile_gpu", "test_prediction"):
            raise ValueError("GPU execution is not bound to an admitted producer attempt")
        self._verify_start_source(matches[0])
        return row

    def forward(self, checkpoint_seed, features, check):
        self._start(check)
        if checkpoint_seed not in CHECKPOINTS:
            raise ValueError("unknown frozen backbone")
        if checkpoint_seed not in self.models:
            from .partial_compatibility_learning import ARMS
            from .pairformer import MODEL_CONFIGS, build_model
            cp = next(c for c in self.reused["checkpoints"] if c["seed"] == checkpoint_seed)
            ref = next(r for r in self.reused["references"] if r["path"] == cp["checkpoint"]["path"])
            # Trusted historical pickle is deserialized only after exact byte authentication.
            state = self.torch.load(BytesIO(read_reference(ref)), map_location="cpu", weights_only=False)
            binding = state["binding"]
            if (binding["runtime"] != self.reused["runtime"] or binding["arm"] != "pairs_descriptors"
                    or binding["seed"] != cp["seed"]
                    or binding["model_config"] != MODEL_CONFIGS[ARMS["pairs_descriptors"][0]]
                    or state["steps"] != 3200 or state["next_epoch"] != 50 or state["next_batch"] != 0):
                raise ValueError("backbone is not the declared frozen last-epoch state")
            model = build_model(ARMS["pairs_descriptors"][0])
            model.load_state_dict(state["model"], strict=True)
            self.models[checkpoint_seed] = model.to("cuda:0").eval()
            del state
        from .partial_compatibility_inference import collect_logits
        result = collect_logits(self.models[checkpoint_seed], [features], device="cuda:0")[0]
        check()
        return result

    def fit(self, q32, partitions, check):
        self._start(check)
        from .observable_source_rivals import GroupFitter, Grid, fit_candidates
        if self.fitter is None:
            self.fitter = GroupFitter(Grid(257, 65, 4), device="cuda", assignment_batch=8)
        # Boundaries between small historical batches, without changing its math.
        delegate = self.fitter
        class CheckedFitter:
            def fit(self, observations, branch):
                check()
                result = delegate.fit(observations, branch)
                check()
                return result
        result = fit_candidates(q32, partitions, CheckedFitter())
        check()
        return result

    def predict(self, head, row, check):
        self._start(check)
        key = head["arm"], head["checkpoint_seed"], head["reader_seed"]
        canonical = next(h for h in self.reused["heads"] if
                         (h["arm"], h["checkpoint_seed"], h["reader_seed"]) == key)
        if encoded(canonical["record"]) != encoded(head["record"]):
            raise ValueError("head differs from frozen selected roster")
        if key not in self.heads:
            from .measurement_reuse import head_arrays
            from .geometric_decision_model import GeometricDecisionHead
            arrays = head_arrays(read_reference(canonical["array_reference"]))
            model = GeometricDecisionHead(head["reader_seed"], head["arm"].rsplit("_", 1)[0])
            model.load_state_dict({k: self.torch.from_numpy(a.copy()) for k, a in arrays.items()}, strict=True)
            self.heads[key] = model.to("cuda:0").eval()
        from .geometric_decision_inference import predict
        return predict(self.heads[key], [row], device="cuda:0", check=check)

    def peak_reserved(self):
        if self._closed:
            return self._final_peak  # Last measured value, no CUDA query after closure.
        if self.torch is None:
            return 0
        self._assert_phase()
        return int(self.torch.cuda.max_memory_reserved(0))

    def close(self):
        """Call within the GPU phase, before its final guard/accounting check."""
        if self.torch is not None:
            # Cleanup also covers a failed runtime-receipt publication after
            # device initialization. Phase and ownership remain mandatory.
            self._assert_phase(require_receipt=False)
        self.models.clear()
        self.heads.clear()
        self.fitter = None
        if self.torch is not None:
            self.torch.cuda.synchronize(0)
            self._final_peak = int(self.torch.cuda.max_memory_reserved(0))
            self.torch.cuda.empty_cache()
        self._closed = True
