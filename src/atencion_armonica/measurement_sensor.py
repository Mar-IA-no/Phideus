"""Pure audio sensor for the paired measurement protocol; no truth or model IO.

The campaign owns split admission, RNG provenance, calibration and persistence.
This module never draws scenes, selects thresholds, loads models or uses CUDA.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

SAMPLE_RATE = 24000
CONDITIONS = (("nominal", 24000, 40.), ("short", 3000, 40.),
              ("noisy", 24000, 20.))
DETECTOR_GRID = tuple((h, p) for h in (-40., -30., -20.) for p in (3., 6., 12.))


def vector(values, name, *, positive=False, nonempty=False):
    result = np.asarray(values, dtype=np.float64)
    if (result.ndim != 1 or not np.isfinite(result).all()
            or (nonempty and not len(result))
            or (positive and np.any(result <= 0))):
        raise ValueError(f"invalid {name}")
    return result


def mix_at_snr(clean, noise, snr_db):
    clean, noise = vector(clean, "clean", nonempty=True), vector(noise, "noise")
    if clean.shape != noise.shape or not np.isfinite(snr_db):
        raise ValueError("invalid noise extent or SNR")
    clean_rms = float(np.sqrt(np.mean(clean**2)))
    noise_rms = float(np.sqrt(np.mean(noise**2)))
    if not np.isfinite([clean_rms, noise_rms]).all():
        raise ValueError("nonfinite RMS")
    if clean_rms == 0:
        scale, actual_snr, status = 0., None, "ZERO_CLEAN_ENERGY"
    elif noise_rms == 0:
        raise ValueError("zero noise RMS for nonzero clean energy")
    else:
        scale = clean_rms / noise_rms * 10.**(-snr_db/20.)
        scaled_rms = float(np.sqrt(np.mean((noise*scale)**2)))
        if not np.isfinite(scale) or not np.isfinite(scaled_rms) or scaled_rms <= 0:
            raise ValueError("invalid scaled noise RMS")
        actual_snr, status = float(20*np.log10(clean_rms/scaled_rms)), "OK"
    waveform = (clean + noise*scale).astype(np.float32)
    if not np.isfinite(waveform).all():
        raise ValueError("nonfinite delivered waveform")
    return {"waveform": waveform, "noise_scale": scale, "status": status,
            "snr_db_float64": actual_snr, "clean_rms": clean_rms,
            "delivered_error_rms": float(np.sqrt(np.mean(
                (waveform.astype(np.float64)-clean)**2)))}


def render_audio(frequencies, amplitude_db, phases, noise):
    """Render explicit events; caller archives parameters and the supplied noise."""
    f = vector(frequencies, "frequencies", positive=True, nonempty=True)
    db, phase = vector(amplitude_db, "amplitude dB"), vector(phases, "phases")
    noise = vector(noise, "noise")
    if (db.shape != f.shape or phase.shape != f.shape or noise.shape != (24000,)
            or np.any((db < -12) | (db > 0))
            or np.any((phase < 0) | (phase >= 2*np.pi))):
        raise ValueError("invalid render parameters")
    amplitude = 10.**(db/20.)
    gain = float(np.sqrt(.02/np.sum(amplitude**2)))
    amplitude *= gain
    time = np.arange(24000, dtype=np.float64)/SAMPLE_RATE
    clean = np.sum(amplitude[:, None] * np.sin(
        2*np.pi*f[:, None]*time[None, :] + phase[:, None]), axis=0)
    if not np.isfinite(clean).all():
        raise ValueError("nonfinite clean waveform")
    return {"clean": clean.astype(np.float32), "amplitude": amplitude,
            "common_gain": gain, "aliased_emission": f >= SAMPLE_RATE/2,
            "outside_detection_band": (f < 50) | (f > 10000),
            "conditions": {name: mix_at_snr(clean[:n], noise[:n], snr)
                           for name, n, snr in CONDITIONS}}


def detect_peaks(waveform, *, height, prominence):
    """All local peaks and discard reasons plus retained events, never truth top-k."""
    x = vector(waveform, "waveform", nonempty=True)
    if len(x) < 3 or (height, prominence) not in DETECTOR_GRID:
        raise ValueError("invalid detector configuration")
    nfft = 4*(1 << (len(x)-1).bit_length())
    window = .5 - .5*np.cos(2*np.pi*np.arange(len(x))/len(x))
    magnitude = np.abs(np.fft.rfft(x*window, n=nfft))
    if not np.isfinite(magnitude).all():
        raise ValueError("nonfinite spectrum")
    maximum = float(magnitude.max())
    db = (np.full_like(magnitude, -300.) if maximum == 0 else
          20*np.log10(np.maximum(magnitude/maximum, 1e-15)))
    bins, properties = find_peaks(db, height=height, prominence=prominence,
                                  plateau_size=(None, None))
    alpha, beta, gamma = db[bins-1], db[bins], db[bins+1]
    denominator = alpha - 2*beta + gamma
    offset = np.zeros(len(bins), dtype=np.float64)
    valid = (denominator != 0) & (properties["plateau_sizes"] == 1)
    np.divide(.5*(alpha-gamma), denominator, out=offset, where=valid)
    valid &= np.isfinite(offset) & (np.abs(offset) <= .5)
    offset[~valid] = 0
    frequency = (bins+offset)*SAMPLE_RATE/nfft
    retained = (frequency >= 50) & (frequency <= 10000)
    indices = np.flatnonzero(retained)
    order = indices[np.lexsort((bins[indices], frequency[indices]))]
    reasons = np.where(retained, "RETAINED", "OUTSIDE_DETECTION_BAND")
    return {"frequencies": frequency[order], "retained_peak_indices": order,
            "bins": bins, "offset": offset, "refinement_valid": valid,
            "all_frequencies": frequency, "retained": retained,
            "reason": reasons, "properties": properties, "spectrum_db": db,
            "spectrum_magnitude": magnitude, "nfft": nfft,
            "status": "SILENT" if maximum == 0 else "OK"}


def operator_observation(frequencies):
    """Pre-kernel gate. No padding, truncation, feature construction or CUDA."""
    f = vector(frequencies, "detected frequencies", positive=True)
    if np.any(np.diff(f) < 0):
        raise ValueError("detected events must be sorted")
    if not 8 <= len(f) <= 32:
        return {"status": "OUTSIDE_OPERATOR_DOMAIN", "frequencies": f.copy(),
                "q32": None}
    log_f = np.log(f)
    return {"status": "ELIGIBLE", "frequencies": f.copy(),
            "q32": (log_f-log_f.mean()).astype(np.float32)}
