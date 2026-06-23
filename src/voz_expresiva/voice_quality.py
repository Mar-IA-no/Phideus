"""Familia B — Voice Quality wrapper.

Compone 9 medidas voice-quality per utterance, distinguiendo explícitamente:

    Medidas directas (7d) — implementación parselmouth/Praat estándar:
        HNR              — Harmonics-to-Noise Ratio (dB)
        CPP              — Cepstral Peak Prominence (dB)
        jitter_local     — perturbación de período (fracción)
        shimmer_local    — perturbación de amplitud (fracción)
        F2_F1            — ratio F2/F1 (VTL-invariante)
        F3_F1            — ratio F3/F1
        alpha_ratio      — log10(energy>1kHz / energy<1kHz)

    Proxies acústicos (2d) — implementación rápida, NO equivalente clínico completo:
        H1_H2_proxy      — diferencia de amplitud (dB) entre los primeros 2 armónicos
        H1_A3_proxy      — diferencia entre H1 y pico en F3-band

Ambos proxies REQUIEREN F0 confiable. Para frames sin F0 se devuelve NaN; el agregador
posterior decide cómo tratarlos (típicamente: ignorar NaN → mean sobre voiced frames).

Separación familia D (eGeMAPS) aparte en `compute_egemaps_functionals()`.

Tipos:
    - Single utterance → dict[str, float] con 9 entries (Familia B)
    - eGeMAPS → dict[str, float] con 88 entries (Familia D)

Uso pensado para CPU + multiprocessing.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

VOICE_QUALITY_DIRECT_KEYS = (
    "HNR",
    "CPP",
    "jitter_local",
    "shimmer_local",
    "F2_F1",
    "F3_F1",
    "alpha_ratio",
)

VOICE_QUALITY_PROXY_KEYS = (
    "H1_H2_proxy",
    "H1_A3_proxy",
)

VOICE_QUALITY_ALL_KEYS = VOICE_QUALITY_DIRECT_KEYS + VOICE_QUALITY_PROXY_KEYS

VOICE_QUALITY_KIND = (
    {k: "direct" for k in VOICE_QUALITY_DIRECT_KEYS}
    | {k: "proxy" for k in VOICE_QUALITY_PROXY_KEYS}
)


# ---------------------------------------------------------------------------
# Parselmouth-backed direct measures
# ---------------------------------------------------------------------------

def _safe_float(x) -> float:
    """Coerce any parselmouth scalar to float; map invalid to NaN."""
    try:
        v = float(x)
        if not np.isfinite(v):
            return float("nan")
        return v
    except (TypeError, ValueError):
        return float("nan")


def compute_voice_quality_direct(
    wav_path: Path | str,
    sr_target: int = 16000,
    f0_floor: float = 75.0,
    f0_ceil: float = 500.0,
) -> Dict[str, float]:
    """Compute the 7 direct voice-quality measures from a WAV file.

    Returns NaN for any measure that fails (silent frames, no voicing, etc.).
    """
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(str(wav_path))
    if snd.sampling_frequency != sr_target:
        snd = snd.resample(sr_target)

    out: Dict[str, float] = {}

    # --- HNR (mean over the whole sound) ---
    try:
        harmonicity = snd.to_harmonicity_cc(
            minimum_pitch=f0_floor, time_step=0.01, silence_threshold=0.1
        )
        out["HNR"] = _safe_float(call(harmonicity, "Get mean", 0, 0))
    except Exception:
        out["HNR"] = float("nan")

    # --- Pitch & PointProcess (needed for jitter, shimmer, CPP support) ---
    try:
        pitch = snd.to_pitch_cc(time_step=0.01, pitch_floor=f0_floor, pitch_ceiling=f0_ceil)
        point_process = call([snd, pitch], "To PointProcess (cc)")
    except Exception:
        pitch = None
        point_process = None

    # --- Jitter / Shimmer local ---
    if point_process is not None:
        try:
            out["jitter_local"] = _safe_float(call(
                point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
            ))
        except Exception:
            out["jitter_local"] = float("nan")
        try:
            out["shimmer_local"] = _safe_float(call(
                [snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
            ))
        except Exception:
            out["shimmer_local"] = float("nan")
    else:
        out["jitter_local"] = float("nan")
        out["shimmer_local"] = float("nan")

    # --- CPP (Hillenbrand-style, manual via librosa — Praat signature varies between versions) ---
    try:
        out["CPP"] = _compute_cpp_manual(str(wav_path), sr_target=sr_target,
                                         f0_floor=f0_floor, f0_ceil=f0_ceil)
    except Exception:
        out["CPP"] = float("nan")

    # --- Formants F1, F2, F3 → ratios ---
    try:
        formant = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5,
                                      maximum_formant=5500)
        # Average formant values over voiced frames using pitch object as voicing proxy
        n_frames = formant.get_number_of_frames()
        F1_vals, F2_vals, F3_vals = [], [], []
        for i in range(1, n_frames + 1):
            t = formant.get_time_from_frame_number(i)
            f1 = formant.get_value_at_time(1, t)
            f2 = formant.get_value_at_time(2, t)
            f3 = formant.get_value_at_time(3, t)
            if np.isfinite(f1) and f1 > 0:
                F1_vals.append(f1)
            if np.isfinite(f2) and f2 > 0:
                F2_vals.append(f2)
            if np.isfinite(f3) and f3 > 0:
                F3_vals.append(f3)
        f1_mean = float(np.mean(F1_vals)) if F1_vals else float("nan")
        f2_mean = float(np.mean(F2_vals)) if F2_vals else float("nan")
        f3_mean = float(np.mean(F3_vals)) if F3_vals else float("nan")
        out["F2_F1"] = f2_mean / f1_mean if np.isfinite(f1_mean) and f1_mean > 0 else float("nan")
        out["F3_F1"] = f3_mean / f1_mean if np.isfinite(f1_mean) and f1_mean > 0 else float("nan")
    except Exception:
        out["F2_F1"] = float("nan")
        out["F3_F1"] = float("nan")

    # --- Alpha-ratio: 10 * log10( energy_high / energy_low ) in dB ---
    try:
        spec = snd.to_spectrum()
        # Praat Spectrum.Get band energy returns linear energy (Pa^2 s)
        e_low = _safe_float(call(spec, "Get band energy", 50.0, 1000.0))
        e_high = _safe_float(call(spec, "Get band energy", 1000.0, 5000.0))
        if np.isfinite(e_low) and np.isfinite(e_high) and e_low > 0 and e_high > 0:
            # alpha-ratio per eGeMAPS = 10 * log10(E_high / E_low) in dB
            out["alpha_ratio"] = 10.0 * float(np.log10(e_high / e_low))
        else:
            out["alpha_ratio"] = float("nan")
    except Exception:
        out["alpha_ratio"] = float("nan")

    return out


def _compute_cpp_manual(
    wav_path: str,
    sr_target: int = 16000,
    f0_floor: float = 75.0,
    f0_ceil: float = 500.0,
    n_fft: int = 2048,
    hop_length: int = 160,
    time_smooth: int = 1,
    quef_smooth: int = 1,
) -> float:
    """Hillenbrand-style CPP (smoothed), averaged over voiced frames.

    Algorithm (Hillenbrand et al. 1994; refined by Heman-Ackah 2003 / ASHA 2018):
      1) log-power spectrum per frame
      2) real cepstrum via inverse rFFT
      3) temporal smoothing (rectangular average over `time_smooth` frames)
         and quefrency smoothing (rectangular over `quef_smooth` samples)
      4) per frame: find peak in quefrency range [sr/F0_ceil, sr/F0_floor]
      5) linear regression of the WHOLE valid cepstrum range
         (q >= 1 ms .. n_quef//2)
      6) CPP = peak_value - regression_at_peak_quefrency (dB)
      7) average over voiced frames

    Returns scalar dB value, NaN on failure.
    """
    import librosa
    import warnings as _warnings

    wav, sr = librosa.load(wav_path, sr=sr_target, mono=True)
    if wav.size < n_fft:
        return float("nan")

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        _, voiced_flag, _ = librosa.pyin(
            wav, sr=sr, fmin=f0_floor, fmax=f0_ceil,
            frame_length=n_fft, hop_length=hop_length, fill_na=0.0,
        )

    S = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, center=True))
    log_S = 20.0 * np.log10(S + 1e-10)                   # (n_freq, n_frames) in dB

    cepstrum = np.fft.irfft(log_S, axis=0)                # (n_fft_eff, n_frames)
    n_quef = cepstrum.shape[0]

    # Temporal smoothing across frames
    if time_smooth > 1 and cepstrum.shape[1] >= time_smooth:
        kernel_t = np.ones(time_smooth) / time_smooth
        cepstrum = np.apply_along_axis(
            lambda x: np.convolve(x, kernel_t, mode="same"), 1, cepstrum
        )
    # Quefrency smoothing
    if quef_smooth > 1 and n_quef >= quef_smooth:
        kernel_q = np.ones(quef_smooth) / quef_smooth
        cepstrum = np.apply_along_axis(
            lambda x: np.convolve(x, kernel_q, mode="same"), 0, cepstrum
        )

    # Peak search range = F0 quefrency band
    q_peak_min = max(int(sr / f0_ceil), 2)
    q_peak_max = min(int(sr / f0_floor), n_quef // 2 - 1)

    # Regression range = from 1 ms quefrency to n_quef//2
    q_reg_min = max(int(0.001 * sr), q_peak_min)
    q_reg_max = min(n_quef // 2 - 1, q_peak_max + int(0.005 * sr))

    if q_peak_max <= q_peak_min + 2 or q_reg_max <= q_reg_min + 4:
        return float("nan")

    q_reg_grid = np.arange(q_reg_min, q_reg_max + 1, dtype=np.float64)

    cpp_values: list[float] = []
    n_frames = min(cepstrum.shape[1], len(voiced_flag) if voiced_flag is not None else cepstrum.shape[1])
    for t in range(n_frames):
        if voiced_flag is not None and not voiced_flag[t]:
            continue
        ceps_t = cepstrum[:, t]
        peak_region = ceps_t[q_peak_min:q_peak_max + 1]
        if not np.all(np.isfinite(peak_region)):
            continue
        peak_rel = int(np.argmax(peak_region))
        peak_q = q_peak_min + peak_rel
        peak_val = float(peak_region[peak_rel])

        reg_region = ceps_t[q_reg_min:q_reg_max + 1]
        if not np.all(np.isfinite(reg_region)):
            continue
        try:
            slope, intercept = np.polyfit(q_reg_grid, reg_region.astype(np.float64), 1)
        except Exception:
            continue
        trend_at_peak = slope * peak_q + intercept
        cpp_values.append(peak_val - float(trend_at_peak))

    if not cpp_values:
        return float("nan")
    return float(np.mean(cpp_values))


# ---------------------------------------------------------------------------
# Acoustic proxies (H1-H2, H1-A3)
# ---------------------------------------------------------------------------

def compute_voice_quality_proxies(
    wav_path: Path | str,
    sr_target: int = 16000,
    n_fft: int = 2048,
    hop_length: int = 160,
    f0_floor: float = 75.0,
    f0_ceil: float = 500.0,
) -> Dict[str, float]:
    """Compute H1-H2 and H1-A3 acoustic proxies (averaged over voiced frames).

    Both rely on F0 + formant tracking via Praat; harmonic amplitudes are estimated
    from the STFT magnitude at expected harmonic bins (±2 bin local peak search).
    NO formant correction is applied — declared as proxy in the plan.

    Returns NaN for frames without reliable F0; final value = mean over voiced frames.
    """
    import librosa
    import parselmouth
    from parselmouth.praat import call

    out = {"H1_H2_proxy": float("nan"), "H1_A3_proxy": float("nan")}

    try:
        wav, sr = librosa.load(str(wav_path), sr=sr_target, mono=True)
        if wav.size == 0:
            return out

        # F0 via PYIN (returns NaN for unvoiced)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f0, voiced_flag, _ = librosa.pyin(
                wav, sr=sr, fmin=f0_floor, fmax=f0_ceil,
                frame_length=n_fft, hop_length=hop_length, fill_na=np.nan,
            )
        if voiced_flag is None or not np.any(voiced_flag):
            return out

        # F3 via Praat for A3 anchor
        snd = parselmouth.Sound(wav.astype(np.float32), sampling_frequency=sr)
        formant = snd.to_formant_burg(time_step=hop_length / sr,
                                      max_number_of_formants=5,
                                      maximum_formant=5500)

        # STFT magnitude
        S = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, center=True))
        freq_res = sr / n_fft
        n_bins = S.shape[0]
        n_frames = min(S.shape[1], len(f0))

        H1_H2 = []
        H1_A3 = []
        for t_idx in range(n_frames):
            if not voiced_flag[t_idx] or not np.isfinite(f0[t_idx]):
                continue
            f0_hz = float(f0[t_idx])

            # H1 = peak around F0 bin; H2 = around 2*F0
            def _peak_amp(freq_hz: float, search: int = 2) -> float:
                if freq_hz <= 0 or freq_hz >= sr / 2:
                    return float("nan")
                bin_c = int(round(freq_hz / freq_res))
                lo = max(0, bin_c - search)
                hi = min(n_bins, bin_c + search + 1)
                if lo >= hi:
                    return float("nan")
                return float(S[lo:hi, t_idx].max())

            h1 = _peak_amp(f0_hz)
            h2 = _peak_amp(2 * f0_hz)

            # F3 at this time → A3
            t_sec = t_idx * hop_length / sr
            try:
                f3_hz = formant.get_value_at_time(3, t_sec)
            except Exception:
                f3_hz = float("nan")
            a3 = _peak_amp(f3_hz, search=4) if np.isfinite(f3_hz) and f3_hz > 0 else float("nan")

            if np.isfinite(h1) and h1 > 0 and np.isfinite(h2) and h2 > 0:
                H1_H2.append(20 * np.log10(h1 + 1e-12) - 20 * np.log10(h2 + 1e-12))
            if np.isfinite(h1) and h1 > 0 and np.isfinite(a3) and a3 > 0:
                H1_A3.append(20 * np.log10(h1 + 1e-12) - 20 * np.log10(a3 + 1e-12))

        if H1_H2:
            out["H1_H2_proxy"] = float(np.mean(H1_H2))
        if H1_A3:
            out["H1_A3_proxy"] = float(np.mean(H1_A3))
    except Exception as exc:
        logger.debug("VQ proxies failed for %s: %s", wav_path, exc)

    return out


def compute_voice_quality(wav_path: Path | str, sr_target: int = 16000) -> Dict[str, float]:
    """Compute all 9 voice quality measures (7 direct + 2 proxies) for one WAV.

    Returns a flat dict keyed by VOICE_QUALITY_ALL_KEYS. NaN for failed measures.
    """
    out = compute_voice_quality_direct(wav_path, sr_target=sr_target)
    out.update(compute_voice_quality_proxies(wav_path, sr_target=sr_target))
    # Force the canonical key order
    return {k: out.get(k, float("nan")) for k in VOICE_QUALITY_ALL_KEYS}


# ---------------------------------------------------------------------------
# eGeMAPSv02 functionals — Familia D, kept separate
# ---------------------------------------------------------------------------

_smile_singleton: Optional[object] = None


def _get_smile():
    """Lazy-init openSMILE eGeMAPSv02 extractor (singleton)."""
    global _smile_singleton
    if _smile_singleton is None:
        import opensmile
        _smile_singleton = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    return _smile_singleton


def compute_egemaps_functionals(wav_path: Path | str) -> Dict[str, float]:
    """Extract eGeMAPSv02 functionals (88d per utterance) via openSMILE.

    Already utterance-level — NO additional pooling needed downstream.
    """
    smile = _get_smile()
    df = smile.process_file(str(wav_path))
    # df has shape (1, 88). Returned as dict for consistency with other extractors.
    row = df.iloc[0].to_dict()
    return {str(k): float(v) for k, v in row.items()}


def egemaps_feature_names() -> tuple[str, ...]:
    """Return canonical 88 eGeMAPSv02 functional names in order."""
    smile = _get_smile()
    return tuple(smile.feature_names)
