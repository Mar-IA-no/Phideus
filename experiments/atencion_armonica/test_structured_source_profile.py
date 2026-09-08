"""Resource-profile mechanics with an explicitly mocked non-scientific receipt."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.atencion_armonica import structured_source_gate as gate
from src.atencion_armonica import structured_source_profile as profile
from src.atencion_armonica.structured_source_metrics import SEEDS


class ProfileTests(unittest.TestCase):
    def test_cpu_profile_covers_maximum_n_and_conservative_no_hit_projection(self):
        common = {"mechanical_fixture": True, "checkpoints": [
            {"seed": s, "threshold": t} for s, t in zip(SEEDS, (.55, .65, .6))]}
        with tempfile.TemporaryDirectory() as tmp, patch.object(gate, "ROOT", Path(tmp)), \
                patch.object(gate, "common_binding", return_value=common), patch.object(gate, "verify_audit"):
            ref = profile.cpu_preflight(Path(tmp)/"mechanical", audit={"fixture_not_an_authorization": True})
            root, _ = gate.bundle_reference(ref, "cpu_preflight", common)
            report = json.loads((root/"report.json").read_text())
            self.assertEqual(report["namespace"], "MECHANICAL_NOT_PROSPECTIVE")
            self.assertEqual([r["n"] for r in report["case_statistics"]], [32, 32])
            self.assertEqual(report["maximum_unique_evaluable_groups_per_scene"], 6*31)
            self.assertEqual(report["grid_cells_by_size"], {str(m): math.comb(8, m)*1025 for m in range(3, 9)})
            bound = 2*256*186*max(max(v) for v in report["fit_seconds_by_size"].values())
            self.assertGreaterEqual(report["projected_cpu_phase_seconds"], bound)
            self.assertEqual(report["projected_five_phase_cpu_seconds"], 5*report["projected_cpu_phase_seconds"])
            for i in range(2):
                fixture = json.loads((root/f"fixture_{i}.json").read_text())
                self.assertEqual(fixture["observation"]["split_seed"], 2026090732)
                self.assertEqual(set(fixture["payloads"]), {str(s) for s in SEEDS})


if __name__ == "__main__":
    unittest.main()
