"""Pure forecast equivalence and explicit accounting; no experimental data."""
import subprocess
import sys

import pytest

from src.atencion_armonica.geometric_decision_remaining_cost import UNITS, projection, allocate_recovery


def test_pure_projection_matches_frozen_profile_arithmetic():
    # Privileged operator import belongs only to this CPU fixture, never runner.
    from experiments.atencion_armonica.profile_geometric_decision_closing import projection as original
    observed = {"observed_path_with_closing_seconds": 100.,
        "observable_recovery_with_closing_seconds": 50., "projected_profile_bytes": 100}
    for factor in (0.003, 1., 19.8):
        timings = {key: (i+1)*factor for i, key in enumerate(UNITS)}
        assert projection(timings, observed, overhead=2., profile_bytes=101) == original(
            timings, observed, overhead=2., profile_bytes=101)


def test_cost_module_does_not_import_truth_reader():
    result = subprocess.run([sys.executable, "-c",
        "import sys; from src.atencion_armonica.geometric_decision_remaining_cost import projection; "
        "assert 'src.atencion_armonica.generative_evidence_supervision' not in sys.modules"],
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_recovery_allocation_preserves_total_work_and_separate_reservations():
    timings = dict.fromkeys(UNITS, 1.)
    observed = {"observable_recovery_with_closing_seconds": 50.}
    forecast = {"fresh_seconds": 965., "evaluation_seconds": 1357.5,
        "projected_new_bytes": 16100, "test_authority": False}
    actual = allocate_recovery(forecast, observed, timings)
    assert actual["transferred_postseal_recovery_seconds"] == 210.
    assert actual["fresh_seconds"]+actual["evaluation_seconds"] == 965.+1357.5
    assert actual["operation_seconds"] == {"prospective-observables": 965.,
        "observable-replay": 210., "metrics-and-replay": 1147.5}
    assert actual["evaluation_inventory_count"]+actual["postseal_recovery_inventory_count"] == 5
    assert actual["projected_new_bytes"] == forecast["projected_new_bytes"]
    with pytest.raises(ValueError, match="loses metric budget"):
        allocate_recovery({**forecast, "evaluation_seconds": 210.}, observed, timings)
