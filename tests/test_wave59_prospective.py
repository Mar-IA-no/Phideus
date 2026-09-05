from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
EXPERIMENTS = REPO_ROOT / "experiments/geometria_proporcional"
for path in (SRC, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _wave59_phase_worker as worker  # noqa: E402
import run_wave59_hgb_guard_bracket as runner  # noqa: E402
from geometria_proporcional.wave59_hgb_guard_bracket import (  # noqa: E402
    inference_safe_view,
)


WAVE57 = (
    REPO_ROOT
    / "data/geometria_proporcional/wave57_contextual_tail_guard_fresh_v1/phases"
)
POLICY_MANIFEST = (
    REPO_ROOT
    / "data/geometria_proporcional/wave52_policy_transport_v1/policy_manifest.json"
)
CONFIG = EXPERIMENTS / "configs/wave59_fresh_hgb_guard_bracket.json"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def save_npz(path: Path, arrays: dict[str, np.ndarray], *, secret: bool = False) -> None:
    np.savez(path, **arrays)
    if secret:
        path.chmod(0o600)


@pytest.fixture(scope="module")
def historical_physical_pipeline(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not WAVE57.is_dir() or not POLICY_MANIFEST.is_file():
        pytest.skip("canonical Wave 57 artifacts are unavailable")
    root = tmp_path_factory.mktemp("wave59-physical")
    prepared = root / "prepared"
    prepared.mkdir()
    fit = load_npz(WAVE57 / "fit.complete/analytics.complete/gate_fit_bundle.npz")
    validation = load_npz(
        WAVE57 / "select.complete/analytics.complete/gate_select_bundle.npz"
    )
    monitor = load_npz(
        WAVE57 / "adjudicate.complete/analytics.complete/sealed_monitor_bundle.npz"
    )
    save_npz(prepared / "gate_fit_bundle.npz", fit, secret=True)
    save_npz(prepared / "gate_select_truth_bundle.npz", validation, secret=True)
    save_npz(prepared / "sealed_monitor_truth_bundle.npz", monitor, secret=True)
    save_npz(
        prepared / "gate_select_inference_bundle.npz",
        inference_safe_view(validation),
    )
    save_npz(
        prepared / "sealed_monitor_inference_bundle.npz",
        inference_safe_view(monitor),
    )
    return runner.execute(prepared, POLICY_MANIFEST, root / "output", CONFIG)


def test_physical_workers_are_unprivileged_and_truth_is_denied(
    historical_physical_pipeline: Path,
) -> None:
    for phase in ("fit", "calibrate_scores", "validate", "monitor_apply", "monitor_evaluate"):
        receipt = json.loads(
            (historical_physical_pipeline / phase / "access_receipt.json").read_text(
                encoding="utf-8"
            )
        )
        assert receipt["effective_uid"] == 65534
        assert receipt["effective_gid"] == 65534
        assert receipt["process_security"] == {
            "effective_capabilities_hex": "0000000000000000",
            "no_new_privileges": 1,
            "supplementary_groups": [],
        }
        assert all(row["denied"] for row in receipt["forbidden_probes"])


def test_historical_main_policies_reproduce_wave58_metrics(
    historical_physical_pipeline: Path,
) -> None:
    analysis = json.loads(
        (
            historical_physical_pipeline / "monitor_evaluate/analysis.json"
        ).read_text(encoding="utf-8")
    )
    assert len(analysis["contrasts"]["factorial"]) == 36
    assert analysis["scientific_decision"] is None
    mean_regret = analysis["contrasts"]["mean_vs_hard"]["regret"]
    tail_worst = analysis["contrasts"]["tail_vs_hard"]["worst_regret"]
    assert mean_regret["mean_diff"] == pytest.approx(-0.014490286855482936)
    assert tail_worst["mean_diff"] == pytest.approx(-0.02342047930283224)
    manifest = json.loads(
        (
            historical_physical_pipeline / "fit/model_states_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert len(manifest["models"]) == 16


def test_stage_allowlist_rejects_truth_in_calibration(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    for name in worker.PHASE_FILES["calibrate_scores"] - {"phase_request.json"}:
        (stage / name).write_bytes(b"placeholder")
    (stage / "truth_bundle.npz").write_bytes(b"forbidden")
    worker.write_json(
        stage / "phase_request.json",
        {
            "phase": "calibrate_scores",
            "allowed_files": sorted(path.name for path in stage.iterdir()),
            "sha256": {},
        },
    )
    with pytest.raises(RuntimeError, match="allowlist"):
        worker.validate_stage(stage, "calibrate_scores")
