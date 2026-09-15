"""CPU fixtures only; no campaign sampler, checkpoints, test roster or CUDA."""
import itertools

import numpy as np
import pytest

from src.atencion_armonica import measurement_metrics as metrics
from src.atencion_armonica import measurement_sensor as sensor


def partitions(n):
    if n == 0:
        yield np.array([], dtype=np.int64)
        return
    def extend(prefix):
        if len(prefix) == n:
            yield np.array(prefix, dtype=np.int64)
        else:
            for label in range(max(prefix)+2):
                yield from extend(prefix+[label])
    yield from extend([0])


@pytest.mark.parametrize("n", [0, 1, 7, 8, 32, 33, 100])
def test_gate_before_kernel(n):
    f = np.arange(n, dtype=np.float64)+100.
    result = sensor.operator_observation(f)
    if 8 <= n <= 32:
        assert result["status"] == "ELIGIBLE"
        assert result["q32"].dtype == np.float32
        np.testing.assert_array_equal(result["q32"], (np.log(f)-np.log(f).mean()).astype(np.float32))
    else:
        assert result["status"] == "OUTSIDE_OPERATOR_DOMAIN"
        assert result["q32"] is None
    np.testing.assert_array_equal(result["frequencies"], f)


def test_renderer_exact_formula_prefix_and_snr():
    f, db, phases = np.array([400., 733.]), np.array([-12., 0.]), np.array([.1, .7])
    noise = np.random.default_rng(4).normal(size=24000)
    result = sensor.render_audio(f, db, phases, noise)
    amplitude = 10.**(db/20.)
    amplitude *= np.sqrt(.02/(amplitude@amplitude))
    t = np.arange(24000)/24000
    expected = sum(a*np.sin(2*np.pi*hz*t+p) for a, hz, p in zip(amplitude, f, phases))
    np.testing.assert_array_equal(result["clean"], expected.astype(np.float32))
    assert amplitude[0]/amplitude[1] == pytest.approx(10**(-12/20))
    for name, n, snr in sensor.CONDITIONS:
        row = result["conditions"][name]
        scale = np.sqrt(np.mean(expected[:n]**2)/np.mean(noise[:n]**2))*10**(-snr/20)
        np.testing.assert_allclose(row["noise_scale"], scale, rtol=1e-14)
        np.testing.assert_array_equal(row["waveform"], (expected[:n]+scale*noise[:n]).astype(np.float32))
        assert row["snr_db_float64"] == pytest.approx(snr)


def test_zero_and_invalid_rms():
    row = sensor.mix_at_snr(np.zeros(10), np.ones(10), 40)
    assert row["status"] == "ZERO_CLEAN_ENERGY" and row["snr_db_float64"] is None
    assert not row["waveform"].any()
    with pytest.raises(ValueError, match="zero noise"):
        sensor.mix_at_snr(np.ones(10), np.zeros(10), 40)
    with pytest.raises(ValueError):
        sensor.mix_at_snr(np.array([np.nan]), np.ones(1), 40)


def test_emission_band_alias_and_no_truth_repair():
    result = sensor.render_audio([40., 11000., 14000.], [-3.]*3, [.3]*3, np.ones(24000))
    assert result["outside_detection_band"].tolist() == [True, True, True]
    assert result["aliased_emission"].tolist() == [False, False, True]
    t = np.arange(24000)/24000
    detected = sensor.detect_peaks(np.sin(2*np.pi*15000*t).astype(np.float32), height=-30., prominence=6.)
    assert len(detected["frequencies"]) >= 1
    assert np.min(abs(detected["frequencies"]-9000)) < 1
    assert metrics.correspondence([15000.], detected["frequencies"])["detected_to_emitted"].max() == -1


def test_detector_resolved_tones_gain_and_silence():
    t = np.arange(24000)/24000
    x = np.sin(2*np.pi*400.3*t)+.6*np.sin(2*np.pi*913.7*t+.4)
    a = sensor.detect_peaks(x, height=-30., prominence=6.)
    b = sensor.detect_peaks(x*.03125, height=-30., prominence=6.)
    np.testing.assert_allclose(a["frequencies"], [400.3, 913.7], atol=.02)
    np.testing.assert_allclose(a["frequencies"], b["frequencies"], atol=1e-10)
    silent = sensor.detect_peaks(np.zeros(3000), height=-30., prominence=6.)
    assert silent["status"] == "SILENT" and len(silent["frequencies"]) == 0
    assert np.all(silent["spectrum_db"] == -300)
    assert a["nfft"] == 131072 and silent["nfft"] == 16384


def test_ambiguous_leaves_and_isolated_vertices():
    c = metrics.correspondence([100., 101., 300., 900.], [100.5, 300., 1200.])
    assert c["detected_to_emitted"].tolist() == [-1, 2, -1]
    assert c["emitted_status"].tolist() == ["ambiguous", "ambiguous", "unique_tolerance_match", "missing"]
    assert c["detected_status"].tolist() == ["ambiguous", "unique_tolerance_match", "spurious/unassigned"]
    assert len(c["components"]) == 4
    d = metrics.correspondence([100., 100.], [100., 100.])
    assert d["detected_to_emitted"].tolist() == [-1, -1]


@pytest.mark.parametrize("n,m", [(0, 0), (0, 3), (3, 0), (2, 3), (3, 4), (4, 3)])
def test_assignment_against_exhaustive_partial_assignments(n, m):
    a = 100*2**(np.arange(n)*13/1200)
    b = 100*2**((np.arange(m)*19+4)/1200)
    distance = metrics.cents_distance(a, b)
    best = 10*(n+m)
    for k in range(min(n, m)+1):
        for left in itertools.combinations(range(n), k):
            for right in itertools.permutations(range(m), k):
                cost = sum(distance[i, j] for i, j in zip(left, right))+10*(n+m-2*k)
                best = min(best, cost)
    result = metrics.detection_cost(a, b)
    assert result["cost"] == pytest.approx(best)
    assert result["cost"] == pytest.approx(result["localization"]+10*(result["missing"]+result["spurious"]))


def test_assignment_edge_at_cutoff_is_unassigned():
    a, b = [100.], [101.]
    cutoff = float(metrics.cents_distance(a, b)[0, 0])
    result = metrics.detection_cost(a, b, cutoff=cutoff)
    assert result["assignment"] == [] and result["cost"] == cutoff


@pytest.mark.parametrize("match", [[0, 1, 2, 3], [0, -1, 2, -1], [0, 1, -1, -1], [-1]*4])
def test_unrestricted_oracle_and_accounting_exhaustive(match):
    truth = np.array([0, 0, 1, 1])
    candidates = list(partitions(4))
    scores = [metrics.pair_score(truth, match, p)["F"] for p in candidates]
    result = metrics.evaluate_candidates(truth, match, candidates, 0)
    assert result["U"] == max(scores)
    for selected, score in enumerate(scores):
        r = metrics.evaluate_candidates(truth, match, candidates, selected)
        assert r["F"] == score and r["F"] <= r["C"] <= r["U"]
        assert sum(r[k] for k in ("correspondence_loss", "coverage_loss", "choice_loss")) == pytest.approx(1-score)
    empty = metrics.evaluate_candidates(truth, match, [], None)
    assert empty["F"] == empty["C"] == 0 and empty["U"] == result["U"]


def test_pairs_include_ambiguous_and_missing_relations():
    result = metrics.pair_score([0, 0, 1, 1], [0, 1, -1, -1], [0, 0, 0, 0])
    assert result == {"T": 2, "P": 6, "TP": 1, "F": .25}
    assert metrics.pair_score([0], [0], [0])["F"] == 1
    with pytest.raises(ValueError, match="one-to-one"):
        metrics.pair_score([0, 0], [0, 0], [0, 0])


def test_common_support_bits_small_n_and_permutation():
    result = metrics.common_support_metrics([0, 0], [1, 0], [0, 1])
    assert result["n"] == 2 and result["VI_bits"] == 1 and result["ARI"] == 0
    assert result["emitted_indices"].tolist() == [0, 1]
    result = metrics.common_support_metrics([3, 3, 4, 4], [3, 1, -1, 0, 2], [9, 8, 9, 8, 9])
    assert result["ARI"] == 1 and result["VI_bits"] == 0
    result = metrics.common_support_metrics([0, 0], [-1, 0], [0, 0])
    assert result["n"] == 1 and result["ARI"] is None and result["VI_bits"] is None


def test_detector_does_not_have_truth_or_model_port():
    import inspect
    assert tuple(inspect.signature(sensor.detect_peaks).parameters) == ("waveform", "height", "prominence")
    assert tuple(inspect.signature(sensor.render_audio).parameters) == ("frequencies", "amplitude_db", "phases", "noise")
