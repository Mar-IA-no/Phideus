"""
Audio-domain ratio descriptors for Gate 4.3 / Fase 5.

Descriptors computed from raw audio waveform via STFT:
  - A4: Local log-frequency deltas (analog of D4 for audio)
  - A7: Rational Attractor (tests Phideus hypothesis directly)
  - A8: Onset-weighted chroma (inspired by Escalón 1 Route A)
  - A9: IDF-weighted rational attractor (inspired by Escalón 1 Route B)

All are pure signal-processing — no learnable parameters.
All operations use torch for GPU compatibility.
"""

import torch
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# A4: Local Log-Frequency Deltas
# ---------------------------------------------------------------------------

# 8 log-spaced frequency bands (STFT bin ranges, n_fft=2048, sr=24000)
# freq_res = 24000/2048 ≈ 11.72 Hz/bin
A4_BAND_EDGES = [
    (4, 8),       # ~47-94 Hz
    (8, 16),      # ~94-188 Hz
    (16, 32),     # ~188-375 Hz
    (32, 64),     # ~375-750 Hz
    (64, 128),    # ~750-1500 Hz
    (128, 256),   # ~1500-3000 Hz
    (256, 512),   # ~3000-6000 Hz
    (512, 1025),  # ~6000-12000 Hz
]


def compute_audio_descriptor_a4(audio: torch.Tensor, target_length: int = None,
                                 n_fft: int = 2048, hop_length: int = 512
                                 ) -> torch.Tensor:
    """
    Local log-frequency deltas — analog of D4 for audio side.

    Args:
        audio: [B, 96000] raw waveform
        target_length: T' (CNN output temporal dim, typically 2400).
                       If None, returns at native STFT resolution (~188 frames).
        n_fft: STFT window size
        hop_length: STFT hop

    Returns:
        [B, target_length, 8] — normalized temporal deltas per freq band
    """
    B = audio.size(0)
    device = audio.device

    # STFT with explicit params for reproducibility
    window = torch.hann_window(n_fft, device=device)
    stft_out = torch.stft(
        audio, n_fft=n_fft, hop_length=hop_length,
        win_length=n_fft, window=window,
        center=True, return_complex=True,
    )
    magnitude = stft_out.abs()  # [B, n_fft//2+1, T_stft]

    # Log magnitude
    log_mag = torch.log1p(magnitude)  # [B, 1025, T_stft]

    # Group into 8 log-freq bands by averaging bins
    T_stft = log_mag.size(2)
    bands = []
    for lo, hi in A4_BAND_EDGES:
        band_mean = log_mag[:, lo:hi, :].mean(dim=1)  # [B, T_stft]
        bands.append(band_mean)
    banded = torch.stack(bands, dim=1)  # [B, 8, T_stft]

    # Temporal delta: diff along time axis, pad first frame with zeros
    delta = banded[:, :, 1:] - banded[:, :, :-1]  # [B, 8, T_stft-1]
    zero_pad = torch.zeros(B, 8, 1, device=device)
    delta = torch.cat([zero_pad, delta], dim=2)  # [B, 8, T_stft]

    # Normalize per band per sample (zero mean, unit std)
    mean = delta.mean(dim=2, keepdim=True)
    std = delta.std(dim=2, keepdim=True).clamp(min=1e-8)
    delta = (delta - mean) / std  # [B, 8, T_stft]

    # Interpolate to target_length (or keep native STFT resolution)
    if target_length is not None:
        delta = F.interpolate(
            delta, size=target_length, mode='linear', align_corners=False,
        )  # [B, 8, target_length]

    return delta.transpose(1, 2)  # [B, T, 8]


def compute_audio_band_energy(audio: torch.Tensor,
                               n_fft: int = 2048, hop_length: int = 512
                               ) -> torch.Tensor:
    """
    Segment-level spectral envelope: mean log-magnitude per A4 band.

    Unlike compute_audio_descriptor_a4 (which returns z-scored temporal deltas
    with mean=0 by construction), this returns a non-degenerate 8-dim summary
    suitable for segment-level conditioning.

    Uses the same STFT + banding as A4 but stops before delta/z-score.

    Args:
        audio: [B, 96000] raw waveform

    Returns:
        [B, 8] — mean log-magnitude per frequency band
    """
    device = audio.device

    window = torch.hann_window(n_fft, device=device)
    stft_out = torch.stft(
        audio, n_fft=n_fft, hop_length=hop_length,
        win_length=n_fft, window=window,
        center=True, return_complex=True,
    )
    log_mag = torch.log1p(stft_out.abs())  # [B, 1025, T_stft]

    bands = []
    for lo, hi in A4_BAND_EDGES:
        band_mean = log_mag[:, lo:hi, :].mean(dim=1)  # [B, T_stft]
        bands.append(band_mean)
    banded = torch.stack(bands, dim=1)  # [B, 8, T_stft]

    return banded.mean(dim=2)  # [B, 8]


# ---------------------------------------------------------------------------
# A7: Rational Attractor
# ---------------------------------------------------------------------------

# 12 just-intonation attractors (log2, octave-folded to [0, 1))
A7_ATTRACTORS = torch.tensor([
    math.log2(1 / 1),      # 0.000  unison/octave
    math.log2(16 / 15),    # 0.093  minor 2nd
    math.log2(9 / 8),      # 0.170  major 2nd
    math.log2(6 / 5),      # 0.263  minor 3rd
    math.log2(5 / 4),      # 0.322  major 3rd
    math.log2(4 / 3),      # 0.415  perfect 4th
    math.log2(7 / 5),      # 0.485  tritone
    math.log2(3 / 2),      # 0.585  perfect 5th
    math.log2(8 / 5),      # 0.678  minor 6th
    math.log2(5 / 3),      # 0.737  major 6th
    math.log2(7 / 4),      # 0.807  harmonic 7th
    math.log2(15 / 8),     # 0.907  major 7th
], dtype=torch.float32)

A7_SIGMA = 0.02  # Gaussian width for soft assignment
A7_NUM_PEAKS = 8
A7_MIN_FREQ_HZ = 50.0

# A10-cont: 32 uniform centers in log2-folded [0, 1) for ontology-free ratio descriptors
A10_CONT_NBINS = 32
A10_CONT_CENTERS = torch.linspace(0, 1 - 1 / 32, 32, dtype=torch.float32)


def _soft_assign_to_centers(
    log2_r: torch.Tensor,
    weight: torch.Tensor,
    centers: torch.Tensor,
    sigma: float = 0.02,
) -> torch.Tensor:
    """Soft Gaussian assignment to arbitrary centers on circular [0,1).

    Same math as A7/A10a JI assignment but parameterized over any center set.
    Used by A10d/A10e with uniform centers.

    Args:
        log2_r: [*, n_pairs] octave-folded ratios in [0, 1)
        weight: [*, n_pairs] weights per pair
        centers: [C] center positions in [0, 1)
        sigma: Gaussian width (default 0.02, same as A7_SIGMA)

    Returns:
        [*, C] soft activation per center
    """
    centers = centers.to(log2_r.device)
    dist = log2_r.unsqueeze(-1) - centers  # [*, n_pairs, C]
    dist = torch.min(dist.abs(), 1.0 - dist.abs())  # circular distance
    activation = torch.exp(-0.5 * (dist / sigma) ** 2)  # [*, n_pairs, C]
    weighted = activation * weight.unsqueeze(-1)
    return weighted.sum(dim=-2)  # [*, C]


def _compute_raw_attractor_activations(
    audio: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 512,
    sample_rate: int = 24000,
) -> torch.Tensor:
    """
    Shared logic for A7 and A9: STFT → peaks → pairwise ratios → soft assignment.

    Returns raw (unnormalized) attractor activations [B, T_stft, 12].
    Caller is responsible for normalization and interpolation.
    """
    device = audio.device
    freq_res = sample_rate / n_fft

    # STFT
    window = torch.hann_window(n_fft, device=device)
    stft_out = torch.stft(
        audio, n_fft=n_fft, hop_length=hop_length,
        win_length=n_fft, window=window,
        center=True, return_complex=True,
    )
    magnitude = stft_out.abs()  # [B, n_fft//2+1, T_stft]

    # Top-k peaks by magnitude
    k = A7_NUM_PEAKS
    peaks_mag, peaks_idx = torch.topk(magnitude, k=k, dim=1)  # [B, k, T_stft]

    # Convert bin indices to Hz
    freq_hz = peaks_idx.float() * freq_res  # [B, k, T_stft]

    # Sort peaks by frequency (topk returns by magnitude — we need freq order)
    sorted_idx = freq_hz.argsort(dim=1)  # [B, k, T_stft]
    freq_hz = freq_hz.gather(1, sorted_idx)
    peaks_mag = peaks_mag.gather(1, sorted_idx)

    # Mask peaks below minimum frequency (DC artifacts)
    freq_mask = (freq_hz >= A7_MIN_FREQ_HZ).float()  # [B, k, T_stft]
    peaks_mag = peaks_mag * freq_mask

    # Clamp frequencies for safe log2 (masked ones won't contribute due to mag=0)
    freq_hz = freq_hz.clamp(min=A7_MIN_FREQ_HZ)

    # Pairwise log2 ratios (upper triangle: i < j)
    idx_i, idx_j = torch.triu_indices(k, k, offset=1)  # C(8,2) = 28 pairs

    freq_i = freq_hz[:, idx_i, :]  # [B, 28, T_stft]
    freq_j = freq_hz[:, idx_j, :]  # [B, 28, T_stft]
    mag_i = peaks_mag[:, idx_i, :]
    mag_j = peaks_mag[:, idx_j, :]

    # Log2 ratio, octave-folded to [0, 1)
    log2_ratio = torch.log2(freq_j / (freq_i + 1e-8))  # [B, 28, T_stft]
    log2_ratio = log2_ratio % 1.0  # octave fold

    # Magnitude weighting: geometric mean of pair magnitudes
    pair_weight = torch.sqrt(mag_i * mag_j + 1e-8)  # [B, 28, T_stft]

    # Soft Gaussian assignment to 12 attractors
    attractors = A7_ATTRACTORS.to(device)  # [12]
    log2_ratio = log2_ratio.unsqueeze(-1)   # [B, 28, T_stft, 1]
    pair_weight = pair_weight.unsqueeze(-1)  # [B, 28, T_stft, 1]

    # Distance to each attractor (with wraparound)
    dist = log2_ratio - attractors  # [B, 28, T_stft, 12]
    dist = torch.min(dist.abs(), (1.0 - dist.abs()))

    activation = torch.exp(-0.5 * (dist / A7_SIGMA) ** 2)  # [B, 28, T_stft, 12]
    weighted = activation * pair_weight  # [B, 28, T_stft, 12]

    # Sum over all 28 pairs → [B, T_stft, 12]
    return weighted.sum(dim=1)  # [B, T_stft, 12]


def compute_audio_descriptor_a7(audio: torch.Tensor, target_length: int = None,
                                 n_fft: int = 2048, hop_length: int = 512,
                                 sample_rate: int = 24000,
                                 ) -> torch.Tensor:
    """
    Rational Attractor — tests Phideus hypothesis directly.

    Picks spectral peaks, computes pairwise log2 frequency ratios,
    and measures proximity to just-intonation attractors.

    Args:
        audio: [B, 96000] raw waveform
        target_length: T' (CNN output temporal dim, typically 2400).
                       If None, returns at native STFT resolution (~188 frames).

    Returns:
        [B, T, 12] — attractor activations per frame
    """
    # Raw activations [B, T_stft, 12]
    result = _compute_raw_attractor_activations(audio, n_fft, hop_length, sample_rate)

    # Normalize per frame (sum=1), handle silent frames
    frame_sum = result.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    result = result / frame_sum  # [B, T_stft, 12]

    # Interpolate to target_length (or keep native STFT resolution)
    if target_length is not None:
        result = result.transpose(1, 2)  # [B, 12, T_stft]
        result = F.interpolate(
            result, size=target_length, mode='linear', align_corners=False,
        )  # [B, 12, target_length]
        result = result.transpose(1, 2)  # [B, target_length, 12]

    return result  # [B, T, 12]


# ---------------------------------------------------------------------------
# A8: Onset-Weighted Chroma (inspired by Escalón 1 Route A)
# ---------------------------------------------------------------------------

def compute_audio_descriptor_a8(audio: torch.Tensor, target_length: int = None,
                                 n_fft: int = 2048, hop_length: int = 512,
                                 sample_rate: int = 24000,
                                 ) -> torch.Tensor:
    """
    Onset-Weighted Chroma — inspired by Escalón 1 Route A (event-based).

    Insight: onsets are the most informative moments for ratio extraction.
    Combines pitch-class energy (12 bins, octave-folded) with spectral flux
    gating to suppress stationary frames.

    Args:
        audio: [B, 96000] raw waveform
        target_length: T' or None for native STFT resolution.

    Returns:
        [B, T, 12] — onset-gated pitch class energy per frame
    """
    B = audio.size(0)
    device = audio.device
    freq_res = sample_rate / n_fft

    # STFT
    window = torch.hann_window(n_fft, device=device)
    stft_out = torch.stft(
        audio, n_fft=n_fft, hop_length=hop_length,
        win_length=n_fft, window=window,
        center=True, return_complex=True,
    )
    magnitude = stft_out.abs()  # [B, 1025, T_stft]
    T_stft = magnitude.size(2)

    # --- Chroma: map STFT bins to 12 pitch classes ---
    n_bins = n_fft // 2 + 1
    bin_freqs = torch.arange(n_bins, device=device).float() * freq_res
    valid_mask = bin_freqs >= 30.0  # skip DC / very low bins
    # Pitch class: round(12 * log2(freq / C1)) % 12, C1 ≈ 32.7 Hz
    log2_ratio = torch.log2(bin_freqs.clamp(min=30.0) / 32.7)
    pitch_classes = (torch.round(12.0 * log2_ratio) % 12).long()  # [n_bins]

    # Accumulate energy per pitch class
    chroma = torch.zeros(B, 12, T_stft, device=device)
    for pc in range(12):
        mask_pc = valid_mask & (pitch_classes == pc)
        if mask_pc.any():
            chroma[:, pc, :] = magnitude[:, mask_pc, :].sum(dim=1)

    # --- Spectral flux (onset strength) ---
    flux = torch.clamp(magnitude[:, :, 1:] - magnitude[:, :, :-1], min=0).sum(dim=1)
    flux = torch.cat([torch.zeros(B, 1, device=device), flux], dim=1)  # [B, T_stft]
    flux_max = flux.max(dim=1, keepdim=True).values.clamp(min=1e-8)
    flux = flux / flux_max  # [B, T_stft] normalized to [0, 1]

    # --- Gate chroma by onset strength ---
    chroma = chroma * flux.unsqueeze(1)  # [B, 12, T_stft]

    # Normalize per frame (sum=1)
    frame_sum = chroma.sum(dim=1, keepdim=True).clamp(min=1e-8)
    chroma = chroma / frame_sum

    # Interpolate
    if target_length is not None:
        chroma = F.interpolate(
            chroma, size=target_length, mode='linear', align_corners=False,
        )

    return chroma.transpose(1, 2)  # [B, T, 12]


# ---------------------------------------------------------------------------
# A9: IDF-Weighted Rational Attractor (inspired by Escalón 1 Route B)
# ---------------------------------------------------------------------------

def compute_audio_descriptor_a9(audio: torch.Tensor, target_length: int = None,
                                 n_fft: int = 2048, hop_length: int = 512,
                                 sample_rate: int = 24000,
                                 idf_threshold: float = 0.05,
                                 ) -> torch.Tensor:
    """
    IDF-Weighted Rational Attractor — inspired by Escalón 1 Route B (improved TF).

    Like A7 but with per-sample IDF weighting: common attractors (octave, fifth)
    are downweighted, rare attractors (tritone, harmonic 7th) are upweighted.

    Args:
        audio: [B, 96000] raw waveform
        target_length: T' or None for native STFT resolution.
        idf_threshold: activation threshold for "document frequency" computation.

    Returns:
        [B, T, 12] — IDF-weighted attractor activations per frame
    """
    # Raw activations [B, T_stft, 12] (shared with A7)
    raw = _compute_raw_attractor_activations(audio, n_fft, hop_length, sample_rate)

    # --- Per-sample IDF weighting ---
    # "Document frequency": fraction of frames where attractor has significant activation
    active_mask = (raw > idf_threshold).float()  # [B, T_stft, 12]
    df = active_mask.mean(dim=1)  # [B, 12]

    # IDF: log(1 / (df + eps)), clamped to avoid extremes
    idf = torch.log(1.0 / (df + 1e-3))  # [B, 12]
    idf = idf.clamp(min=0.0, max=5.0)

    # Apply IDF weighting
    result = raw * idf.unsqueeze(1)  # [B, T_stft, 12]

    # Normalize per frame (sum=1)
    frame_sum = result.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    result = result / frame_sum

    # Interpolate
    if target_length is not None:
        result = result.transpose(1, 2)
        result = F.interpolate(
            result, size=target_length, mode='linear', align_corners=False,
        )
        result = result.transpose(1, 2)

    return result  # [B, T, 12]


# ---------------------------------------------------------------------------
# A10: Recurrence-Based Descriptors
# ---------------------------------------------------------------------------
# Three descriptors operationalizing "natural harmony" via dynamical recurrence
# rather than spectral peak matching (A7/A9).
#
# A10a: Autocorrelation → 12 JI attractors (linear phase-locking, O(N log N))
# A10b: RQA → 12 JI attractors (non-linear phase-locking, O(N²) reduced)
# A10c: RQA generic 6d (non-linear dynamics control, no JI projection)
#
# All project onto A7_ATTRACTORS vocabulary for direct comparison.
# ---------------------------------------------------------------------------

# --- Shared helpers ---

def _normalized_autocorrelation(window: torch.Tensor) -> torch.Tensor:
    """FFT-based normalized autocorrelation. O(N log N).

    Args:
        window: [N] 1D signal

    Returns:
        [N] normalized autocorrelation (acf[0] = 1.0 for non-silent signals)
    """
    N = window.size(0)
    # Zero-pad to avoid circular correlation artifacts
    padded = F.pad(window, (0, N))
    X = torch.fft.rfft(padded)
    power = X * X.conj()
    acf = torch.fft.irfft(power)[:N].real
    # Normalize by acf[0] (energy)
    if acf[0].abs() < 1e-10:
        return torch.zeros(N, device=window.device)
    return acf / acf[0]


def _find_acf_peaks(
    acf: torch.Tensor,
    min_lag: int,
    max_lag: int,
    k: int = 8,
) -> tuple:
    """Find top-k peaks in autocorrelation by local maximum detection.

    Args:
        acf: [N] normalized autocorrelation
        min_lag: minimum lag to search (sr // max_freq)
        max_lag: maximum lag to search (sr // min_freq)
        k: number of peaks to return

    Returns:
        (lags: [k], heights: [k]) — zero-padded if fewer than k peaks found
    """
    device = acf.device
    N = acf.size(0)
    max_lag = min(max_lag, N - 2)
    if min_lag >= max_lag:
        return torch.zeros(k, device=device), torch.zeros(k, device=device)

    # Search region
    region = acf[min_lag:max_lag + 1]
    L = region.size(0)
    if L < 3:
        return torch.zeros(k, device=device), torch.zeros(k, device=device)

    # Local maxima: value > both neighbors
    is_peak = (region[1:-1] > region[:-2]) & (region[1:-1] > region[2:])
    peak_indices = torch.where(is_peak)[0] + 1  # offset within region
    peak_heights = region[peak_indices]

    n_peaks = peak_indices.size(0)
    if n_peaks == 0:
        return torch.zeros(k, device=device), torch.zeros(k, device=device)

    # Top-k by height
    n_take = min(n_peaks, k)
    topk_heights, topk_idx = peak_heights.topk(n_take)
    topk_lags = (peak_indices[topk_idx] + min_lag).float()

    # Zero-pad to k
    lags = torch.zeros(k, device=device)
    heights = torch.zeros(k, device=device)
    lags[:n_take] = topk_lags
    heights[:n_take] = topk_heights
    return lags, heights


def _delay_embedding(
    window: torch.Tensor,
    m: int = 3,
    tau: int = 1,
    stride: int = 16,
) -> torch.Tensor:
    """Phase space reconstruction via delay embedding.

    Args:
        window: [N] 1D signal
        m: embedding dimension
        tau: delay (in samples)
        stride: sub-sampling stride for reduced recipe

    Returns:
        [N_vec, m] embedded vectors
    """
    N = window.size(0)
    # Sub-sample first
    sub = window[::stride]  # [N_sub]
    N_sub = sub.size(0)
    max_start = N_sub - (m - 1) * tau
    if max_start <= 0:
        return torch.zeros(1, m, device=window.device)
    # Build embedding via unfold
    # Each vector: [sub[i], sub[i+tau], sub[i+2*tau], ...]
    indices = torch.arange(max_start, device=window.device).unsqueeze(1) + \
              torch.arange(m, device=window.device).unsqueeze(0) * tau
    return sub[indices]  # [N_vec, m]


def _recurrence_matrix(
    embedded: torch.Tensor,
    threshold_pct: float = 10.0,
) -> torch.Tensor:
    """Compute binary recurrence matrix with adaptive threshold.

    Args:
        embedded: [N_vec, m] phase space vectors
        threshold_pct: percentile of distances for threshold

    Returns:
        [N_vec, N_vec] boolean recurrence matrix
    """
    dists = torch.cdist(embedded.unsqueeze(0), embedded.unsqueeze(0)).squeeze(0)
    # Adaptive threshold: percentile of all pairwise distances
    flat = dists.flatten()
    k_idx = max(1, int(flat.size(0) * threshold_pct / 100.0))
    threshold = flat.kthvalue(k_idx).values
    return dists <= threshold


def _rqa_metrics(rp: torch.Tensor) -> torch.Tensor:
    """Compute 6 standard RQA metrics from a recurrence matrix.

    Args:
        rp: [N, N] boolean recurrence matrix

    Returns:
        [6] tensor: %REC, %DET, %LAM, Lmean, ENTR, TT
    """
    device = rp.device
    N = rp.size(0)
    if N < 2:
        return torch.zeros(6, device=device)

    total_points = N * N
    n_rec = rp.sum().float()
    rec_rate = n_rec / total_points

    # --- Diagonal lines (determinism) ---
    # Collect line lengths from diagonals (offset 1..N-1)
    diag_lines = []
    for offset in range(1, N):
        diag = torch.diagonal(rp, offset=offset)
        if diag.size(0) < 2:
            continue
        # Find consecutive runs of True
        d_int = diag.int()
        diff = d_int[1:] - d_int[:-1]
        starts = torch.where(diff == 1)[0] + 1
        if d_int[0] == 1:
            starts = torch.cat([torch.tensor([0], device=device), starts])
        ends = torch.where(diff == -1)[0] + 1
        if d_int[-1] == 1:
            ends = torch.cat([ends, torch.tensor([diag.size(0)], device=device)])
        lengths = ends - starts
        if lengths.numel() > 0:
            diag_lines.append(lengths[lengths >= 2])

    if diag_lines:
        all_diag = torch.cat(diag_lines).float()
    else:
        all_diag = torch.tensor([], device=device)

    n_det_points = all_diag.sum() if all_diag.numel() > 0 else torch.tensor(0.0, device=device)
    det = n_det_points / n_rec.clamp(min=1.0)
    l_mean = all_diag.mean() if all_diag.numel() > 0 else torch.tensor(0.0, device=device)

    # Entropy of line length distribution
    if all_diag.numel() > 0:
        counts = torch.bincount(all_diag.long())
        probs = counts.float() / counts.sum()
        probs = probs[probs > 0]
        entr = -(probs * probs.log()).sum()
    else:
        entr = torch.tensor(0.0, device=device)

    # --- Vertical lines (laminarity) ---
    vert_lines = []
    for col in range(N):
        v = rp[:, col].int()
        if v.sum() < 2:
            continue
        diff = v[1:] - v[:-1]
        starts = torch.where(diff == 1)[0] + 1
        if v[0] == 1:
            starts = torch.cat([torch.tensor([0], device=device), starts])
        ends = torch.where(diff == -1)[0] + 1
        if v[-1] == 1:
            ends = torch.cat([ends, torch.tensor([N], device=device)])
        lengths = ends - starts
        if lengths.numel() > 0:
            vert_lines.append(lengths[lengths >= 2])

    if vert_lines:
        all_vert = torch.cat(vert_lines).float()
    else:
        all_vert = torch.tensor([], device=device)

    n_lam_points = all_vert.sum() if all_vert.numel() > 0 else torch.tensor(0.0, device=device)
    lam = n_lam_points / n_rec.clamp(min=1.0)
    tt = all_vert.mean() if all_vert.numel() > 0 else torch.tensor(0.0, device=device)

    return torch.stack([rec_rate, det, lam, l_mean, entr, tt])


def _diagonal_recurrence_profile(rp: torch.Tensor) -> tuple:
    """Extract recurrence rate per diagonal offset ("period histogram").

    Used by A10b to find dominant periods from the recurrence matrix.

    Args:
        rp: [N, N] boolean recurrence matrix

    Returns:
        (lags: [k], heights: [k]) top-k diagonal recurrence rates
    """
    N = rp.size(0)
    device = rp.device
    k = A7_NUM_PEAKS

    if N < 3:
        return torch.zeros(k, device=device), torch.zeros(k, device=device)

    # Recurrence rate for each diagonal offset (1..N-1)
    n_diags = N - 1
    rates = torch.zeros(n_diags, device=device)
    for offset in range(1, N):
        diag = torch.diagonal(rp.float(), offset=offset)
        rates[offset - 1] = diag.mean()

    if n_diags < 3:
        return torch.zeros(k, device=device), torch.zeros(k, device=device)

    # Peak detection (same logic as _find_acf_peaks but on rates)
    is_peak = (rates[1:-1] > rates[:-2]) & (rates[1:-1] > rates[2:])
    peak_indices = torch.where(is_peak)[0] + 1  # offset within rates
    peak_heights = rates[peak_indices]

    n_peaks = peak_indices.size(0)
    if n_peaks == 0:
        return torch.zeros(k, device=device), torch.zeros(k, device=device)

    n_take = min(n_peaks, k)
    topk_heights, topk_idx = peak_heights.topk(n_take)
    # lags = offset + 1 (since rates[0] corresponds to diagonal offset 1)
    topk_lags = (peak_indices[topk_idx] + 1).float()

    lags = torch.zeros(k, device=device)
    heights = torch.zeros(k, device=device)
    lags[:n_take] = topk_lags
    heights[:n_take] = topk_heights
    return lags, heights


def _period_ratios_to_ji_activations(
    lags: torch.Tensor,
    heights: torch.Tensor,
    sigma: float = 0.02,
) -> torch.Tensor:
    """Convert period lags + heights to JI attractor activations.

    Computes pairwise period ratios, folds to [0,1) via log2,
    and soft-assigns to A7_ATTRACTORS.

    Args:
        lags: [k] period lags (0 = padding)
        heights: [k] corresponding heights/strengths
        sigma: Gaussian width for soft assignment

    Returns:
        [12] attractor activations
    """
    device = lags.device
    attractors = A7_ATTRACTORS.to(device)

    # Valid mask: only non-zero lags
    valid = lags > 0
    n_valid = valid.sum().item()
    if n_valid < 2:
        return torch.zeros(12, device=device)

    valid_lags = lags[valid]
    valid_heights = heights[valid]
    n = valid_lags.size(0)

    # Pairwise ratios (all pairs, i != j)
    activations = torch.zeros(12, device=device)
    for i in range(n):
        for j in range(i + 1, n):
            ratio = valid_lags[j] / valid_lags[i]
            if ratio < 1.0:
                ratio = valid_lags[i] / valid_lags[j]
            log2_r = torch.log2(ratio) % 1.0  # octave fold

            # Geometric mean weight
            weight = torch.sqrt(valid_heights[i] * valid_heights[j] + 1e-8)

            # Distance to each attractor (with wraparound)
            dist = (log2_r - attractors).abs()
            dist = torch.min(dist, 1.0 - dist)

            act = torch.exp(-0.5 * (dist / sigma) ** 2)
            activations = activations + act * weight

    return activations


# --- A10a: Autocorrelation → 12 JI attractors (VECTORIZED) ---

def compute_audio_descriptor_a10a(
    audio: torch.Tensor,
    target_length: int = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    sample_rate: int = 24000,
) -> torch.Tensor:
    """A10a: Autocorrelation-based JI attractor activations.

    Fully vectorized: no Python per-frame loops.
    Batch autocorrelation → batch peak detection → batch JI assignment.

    Args:
        audio: [B, T_samples] raw waveform
        target_length: T' for interpolation (None = native STFT resolution)

    Returns:
        [B, T, 12] attractor activations per frame
    """
    B = audio.size(0)
    device = audio.device
    k = A7_NUM_PEAKS

    min_lag = max(1, sample_rate // 4000)
    max_lag = min(sample_rate // 50, n_fft - 2)

    # Frame the audio
    pad = n_fft // 2
    audio_padded = F.pad(audio, (pad, pad), mode='reflect')
    frames = audio_padded.unfold(1, n_fft, hop_length)  # [B, T, n_fft]
    T_stft = frames.size(1)
    window = torch.hann_window(n_fft, device=device)
    frames = frames * window

    # Batch autocorrelation
    padded = F.pad(frames, (0, n_fft))
    X = torch.fft.rfft(padded)
    acf_all = torch.fft.irfft(X * X.conj())[..., :n_fft].real  # [B, T, n_fft]
    acf0 = acf_all[..., 0:1].clamp(min=1e-10)
    acf_all = acf_all / acf0

    # Energy gate mask
    energy = (frames ** 2).mean(dim=-1)  # [B, T]
    active = energy > 1e-10  # [B, T]

    # Vectorized peak detection on region [min_lag, max_lag]
    region = acf_all[:, :, min_lag:max_lag + 1]  # [B, T, L]
    L = region.size(2)
    if L < 3:
        result = torch.zeros(B, T_stft, 12, device=device)
    else:
        # Local maxima: value > both neighbors
        is_peak = (region[:, :, 1:-1] > region[:, :, :-2]) & \
                  (region[:, :, 1:-1] > region[:, :, 2:])  # [B, T, L-2]
        peak_vals = region[:, :, 1:-1] * is_peak.float()  # zero non-peaks

        # Top-k peaks per frame
        topk_heights, topk_idx = peak_vals.topk(min(k, L - 2), dim=-1)  # [B, T, k']
        k_actual = topk_heights.size(-1)
        topk_lags = (topk_idx + min_lag + 1).float()  # actual lag positions

        # Mask out zero-height "peaks" (frames with < k real peaks)
        valid_peak = topk_heights > 0  # [B, T, k']

        # Pairwise period ratios (vectorized)
        idx_i, idx_j = torch.triu_indices(k_actual, k_actual, offset=1, device=device)
        n_pairs = idx_i.size(0)

        lags_i = topk_lags[:, :, idx_i]  # [B, T, n_pairs]
        lags_j = topk_lags[:, :, idx_j]
        h_i = topk_heights[:, :, idx_i]
        h_j = topk_heights[:, :, idx_j]
        valid_i = valid_peak[:, :, idx_i]
        valid_j = valid_peak[:, :, idx_j]
        pair_valid = valid_i & valid_j  # [B, T, n_pairs]

        # Period ratio (ensure >= 1)
        ratio = torch.where(
            lags_j >= lags_i,
            lags_j / lags_i.clamp(min=1),
            lags_i / lags_j.clamp(min=1),
        )
        log2_r = torch.log2(ratio.clamp(min=1.0)) % 1.0  # octave fold [B, T, n_pairs]

        # Geometric mean weight
        weight = torch.sqrt(h_i * h_j + 1e-8) * pair_valid.float()  # [B, T, n_pairs]

        # Soft assignment to 12 attractors (vectorized)
        attractors = A7_ATTRACTORS.to(device)  # [12]
        dist = log2_r.unsqueeze(-1) - attractors  # [B, T, n_pairs, 12]
        dist = torch.min(dist.abs(), 1.0 - dist.abs())
        activation = torch.exp(-0.5 * (dist / A7_SIGMA) ** 2)  # [B, T, n_pairs, 12]
        weighted = activation * weight.unsqueeze(-1)
        result = weighted.sum(dim=2)  # [B, T, 12]

    # Apply energy gate
    result = result * active.unsqueeze(-1).float()

    # Normalize per frame
    frame_sum = result.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    result = result / frame_sum
    result = result * active.unsqueeze(-1).float()  # re-zero silent frames

    if target_length is not None:
        result = result.transpose(1, 2)
        result = F.interpolate(result, size=target_length, mode='linear', align_corners=False)
        result = result.transpose(1, 2)

    return result


# --- A10d: Autocorrelation → 32 uniform centers (ratio-native continuous) ---

def compute_audio_descriptor_a10d(
    audio: torch.Tensor,
    target_length: int = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    sample_rate: int = 24000,
) -> torch.Tensor:
    """A10d: Autocorrelation-based ratio descriptor with 32 uniform centers.

    Same pipeline as A10a (autocorr → peaks → pairwise ratios) but projects
    onto 32 uniformly spaced centers in log2 [0,1) instead of 12 JI attractors.
    Same σ=0.02, same circular distance — only centers differ.

    Args:
        audio: [B, T_samples] raw waveform
        target_length: T' for interpolation (None = native STFT resolution)

    Returns:
        [B, T, 32] soft activations per frame
    """
    n_bins = A10_CONT_NBINS
    B = audio.size(0)
    device = audio.device
    k = A7_NUM_PEAKS

    min_lag = max(1, sample_rate // 4000)
    max_lag = min(sample_rate // 50, n_fft - 2)

    # Frame the audio
    pad = n_fft // 2
    audio_padded = F.pad(audio, (pad, pad), mode='reflect')
    frames = audio_padded.unfold(1, n_fft, hop_length)  # [B, T, n_fft]
    T_stft = frames.size(1)
    window = torch.hann_window(n_fft, device=device)
    frames = frames * window

    # Batch autocorrelation
    padded = F.pad(frames, (0, n_fft))
    X = torch.fft.rfft(padded)
    acf_all = torch.fft.irfft(X * X.conj())[..., :n_fft].real
    acf0 = acf_all[..., 0:1].clamp(min=1e-10)
    acf_all = acf_all / acf0

    # Energy gate mask
    energy = (frames ** 2).mean(dim=-1)
    active = energy > 1e-10

    # Vectorized peak detection
    region = acf_all[:, :, min_lag:max_lag + 1]
    L = region.size(2)
    if L < 3:
        result = torch.zeros(B, T_stft, n_bins, device=device)
    else:
        is_peak = (region[:, :, 1:-1] > region[:, :, :-2]) & \
                  (region[:, :, 1:-1] > region[:, :, 2:])
        peak_vals = region[:, :, 1:-1] * is_peak.float()

        topk_heights, topk_idx = peak_vals.topk(min(k, L - 2), dim=-1)
        k_actual = topk_heights.size(-1)
        topk_lags = (topk_idx + min_lag + 1).float()
        valid_peak = topk_heights > 0

        # Pairwise period ratios
        idx_i, idx_j = torch.triu_indices(k_actual, k_actual, offset=1, device=device)
        n_pairs = idx_i.size(0)

        lags_i = topk_lags[:, :, idx_i]
        lags_j = topk_lags[:, :, idx_j]
        h_i = topk_heights[:, :, idx_i]
        h_j = topk_heights[:, :, idx_j]
        pair_valid = valid_peak[:, :, idx_i] & valid_peak[:, :, idx_j]

        ratio = torch.where(
            lags_j >= lags_i,
            lags_j / lags_i.clamp(min=1),
            lags_i / lags_j.clamp(min=1),
        )
        log2_r = torch.log2(ratio.clamp(min=1.0)) % 1.0
        weight = torch.sqrt(h_i * h_j + 1e-8) * pair_valid.float()

        # Soft assignment to 32 uniform centers (same σ as A7)
        result = _soft_assign_to_centers(log2_r, weight, A10_CONT_CENTERS)  # [B, T, 32]

    # Apply energy gate
    result = result * active.unsqueeze(-1).float()

    # Normalize per frame
    frame_sum = result.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    result = result / frame_sum
    result = result * active.unsqueeze(-1).float()

    if target_length is not None:
        result = result.transpose(1, 2)
        result = F.interpolate(result, size=target_length, mode='linear', align_corners=False)
        result = result.transpose(1, 2)

    return result


# --- A10b: RQA → 12 JI attractors (VECTORIZED) ---

def compute_audio_descriptor_a10b(
    audio: torch.Tensor,
    target_length: int = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    sample_rate: int = 24000,
    m: int = 3,
    tau: int = 1,
    stride: int = 16,
    threshold_pct: float = 10.0,
) -> torch.Tensor:
    """A10b: RQA-based JI attractor activations (vectorized).

    Batched delay embedding → batched cdist → batched diagonal profile →
    vectorized peak detection → vectorized JI assignment.

    Args:
        audio: [B, T_samples] raw waveform
        target_length: T' for interpolation (None = native STFT resolution)

    Returns:
        [B, T, 12] attractor activations per frame
    """
    B = audio.size(0)
    device = audio.device
    k = A7_NUM_PEAKS

    # Frame the audio
    pad = n_fft // 2
    audio_padded = F.pad(audio, (pad, pad), mode='reflect')
    frames = audio_padded.unfold(1, n_fft, hop_length)  # [B, T, n_fft]
    T_stft = frames.size(1)
    window = torch.hann_window(n_fft, device=device)
    frames = frames * window

    # Energy gate
    energy = (frames ** 2).mean(dim=-1)  # [B, T]
    active = energy > 1e-10

    # Batched delay embedding: sub-sample then unfold
    sub = frames[:, :, ::stride]  # [B, T, N_sub]
    N_sub = sub.size(2)
    N_vec = N_sub - (m - 1) * tau
    if N_vec < 3:
        result = torch.zeros(B, T_stft, 12, device=device)
        if target_length is not None:
            result = result.transpose(1, 2)
            result = F.interpolate(result, size=target_length, mode='linear', align_corners=False)
            result = result.transpose(1, 2)
        return result

    # Build embedding indices
    idx = torch.arange(N_vec, device=device).unsqueeze(1) + \
          torch.arange(m, device=device).unsqueeze(0) * tau  # [N_vec, m]
    embedded = sub[:, :, idx]  # [B, T, N_vec, m]

    # Reshape for batched cdist: [B*T, N_vec, m]
    BT = B * T_stft
    emb_flat = embedded.reshape(BT, N_vec, m)

    # Chunked processing to avoid memory explosion (Codex ALTA fix)
    # Without chunking: [BT, N_vec, N_vec] float32 at BT=12864 = ~707MB
    # With chunks of 512: ~32MB peak per chunk
    CDIST_CHUNK = 512
    k_idx = max(1, int(N_vec * N_vec * threshold_pct / 100.0))
    max_offset = N_vec - 1
    attractors = A7_ATTRACTORS.to(device)

    all_chunk_results = []
    for start in range(0, BT, CDIST_CHUNK):
        end = min(start + CDIST_CHUNK, BT)
        chunk_emb = emb_flat[start:end]  # [chunk_sz, N_vec, m]
        chunk_sz = end - start

        # Distance matrix + adaptive threshold + recurrence plot
        dists = torch.cdist(chunk_emb, chunk_emb)  # [chunk_sz, N_vec, N_vec]
        flat_dists = dists.reshape(chunk_sz, -1)
        thresholds = flat_dists.kthvalue(k_idx, dim=1).values
        rp = dists <= thresholds.unsqueeze(-1).unsqueeze(-1)
        del dists, flat_dists  # free memory

        # Diagonal recurrence profile
        rates = torch.zeros(chunk_sz, max_offset, device=device)
        for offset in range(1, max_offset + 1):
            diag = torch.diagonal(rp.float(), offset=offset, dim1=-2, dim2=-1)
            rates[:, offset - 1] = diag.mean(dim=-1)
        del rp

        # Peak detection on rates
        if max_offset < 3:
            chunk_result = torch.zeros(chunk_sz, 12, device=device)
        else:
            is_peak = (rates[:, 1:-1] > rates[:, :-2]) & \
                      (rates[:, 1:-1] > rates[:, 2:])
            peak_vals = rates[:, 1:-1] * is_peak.float()
            k_take = min(k, peak_vals.size(1))
            topk_heights, topk_idx = peak_vals.topk(k_take, dim=-1)
            topk_lags = ((topk_idx + 2).float() * stride)

            valid_peak = topk_heights > 0
            k_actual = k_take

            idx_i, idx_j = torch.triu_indices(k_actual, k_actual, offset=1, device=device)
            n_pairs = idx_i.size(0)
            if n_pairs == 0:
                chunk_result = torch.zeros(chunk_sz, 12, device=device)
            else:
                lags_i = topk_lags[:, idx_i]
                lags_j = topk_lags[:, idx_j]
                h_i = topk_heights[:, idx_i]
                h_j = topk_heights[:, idx_j]
                pair_valid = valid_peak[:, idx_i] & valid_peak[:, idx_j]

                ratio = torch.where(
                    lags_j >= lags_i,
                    lags_j / lags_i.clamp(min=1),
                    lags_i / lags_j.clamp(min=1),
                )
                log2_r = torch.log2(ratio.clamp(min=1.0)) % 1.0
                weight = torch.sqrt(h_i * h_j + 1e-8) * pair_valid.float()

                dist = log2_r.unsqueeze(-1) - attractors
                dist = torch.min(dist.abs(), 1.0 - dist.abs())
                activation = torch.exp(-0.5 * (dist / A7_SIGMA) ** 2)
                weighted = activation * weight.unsqueeze(-1)
                chunk_result = weighted.sum(dim=1)  # [chunk_sz, 12]

        all_chunk_results.append(chunk_result)

    result = torch.cat(all_chunk_results, dim=0).reshape(B, T_stft, 12)

    # Apply energy gate
    result = result * active.unsqueeze(-1).float()

    # Normalize per frame
    frame_sum = result.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    result = result / frame_sum
    result = result * active.unsqueeze(-1).float()

    if target_length is not None:
        result = result.transpose(1, 2)
        result = F.interpolate(result, size=target_length, mode='linear', align_corners=False)
        result = result.transpose(1, 2)

    return result


# --- A10e: RQA → 32 uniform centers (ratio-native continuous) ---

def compute_audio_descriptor_a10e(
    audio: torch.Tensor,
    target_length: int = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    sample_rate: int = 24000,
    m: int = 3,
    tau: int = 1,
    stride: int = 16,
    threshold_pct: float = 10.0,
) -> torch.Tensor:
    """A10e: RQA-based ratio descriptor with 32 uniform centers.

    Same pipeline as A10b (delay embedding → RQA → diagonal profile → peaks →
    pairwise ratios) but projects onto 32 uniformly spaced centers in log2 [0,1)
    instead of 12 JI attractors. Same σ=0.02, same circular distance.

    Args:
        audio: [B, T_samples] raw waveform
        target_length: T' for interpolation (None = native STFT resolution)

    Returns:
        [B, T, 32] soft activations per frame
    """
    n_bins = A10_CONT_NBINS
    B = audio.size(0)
    device = audio.device
    k = A7_NUM_PEAKS

    # Frame the audio
    pad = n_fft // 2
    audio_padded = F.pad(audio, (pad, pad), mode='reflect')
    frames = audio_padded.unfold(1, n_fft, hop_length)
    T_stft = frames.size(1)
    window = torch.hann_window(n_fft, device=device)
    frames = frames * window

    # Energy gate
    energy = (frames ** 2).mean(dim=-1)
    active = energy > 1e-10

    # Batched delay embedding
    sub = frames[:, :, ::stride]
    N_sub = sub.size(2)
    N_vec = N_sub - (m - 1) * tau
    if N_vec < 3:
        result = torch.zeros(B, T_stft, n_bins, device=device)
        if target_length is not None:
            result = result.transpose(1, 2)
            result = F.interpolate(result, size=target_length, mode='linear', align_corners=False)
            result = result.transpose(1, 2)
        return result

    idx = torch.arange(N_vec, device=device).unsqueeze(1) + \
          torch.arange(m, device=device).unsqueeze(0) * tau
    embedded = sub[:, :, idx]  # [B, T, N_vec, m]

    BT = B * T_stft
    emb_flat = embedded.reshape(BT, N_vec, m)

    # Chunked processing (same as A10b but with 32 uniform centers)
    CDIST_CHUNK = 512
    k_idx = max(1, int(N_vec * N_vec * threshold_pct / 100.0))
    max_offset = N_vec - 1
    centers = A10_CONT_CENTERS.to(device)

    all_chunk_results = []
    for start in range(0, BT, CDIST_CHUNK):
        end = min(start + CDIST_CHUNK, BT)
        chunk_emb = emb_flat[start:end]
        chunk_sz = end - start

        dists = torch.cdist(chunk_emb, chunk_emb)
        flat_dists = dists.reshape(chunk_sz, -1)
        thresholds = flat_dists.kthvalue(k_idx, dim=1).values
        rp = dists <= thresholds.unsqueeze(-1).unsqueeze(-1)
        del dists, flat_dists

        # Diagonal recurrence profile
        rates = torch.zeros(chunk_sz, max_offset, device=device)
        for offset in range(1, max_offset + 1):
            diag = torch.diagonal(rp.float(), offset=offset, dim1=-2, dim2=-1)
            rates[:, offset - 1] = diag.mean(dim=-1)
        del rp

        # Peak detection
        if max_offset < 3:
            chunk_result = torch.zeros(chunk_sz, n_bins, device=device)
        else:
            is_peak = (rates[:, 1:-1] > rates[:, :-2]) & \
                      (rates[:, 1:-1] > rates[:, 2:])
            peak_vals = rates[:, 1:-1] * is_peak.float()
            k_take = min(k, peak_vals.size(1))
            topk_heights, topk_idx = peak_vals.topk(k_take, dim=-1)
            topk_lags = ((topk_idx + 2).float() * stride)

            valid_peak = topk_heights > 0
            k_actual = k_take

            idx_i, idx_j = torch.triu_indices(k_actual, k_actual, offset=1, device=device)
            n_pairs = idx_i.size(0)
            if n_pairs == 0:
                chunk_result = torch.zeros(chunk_sz, n_bins, device=device)
            else:
                lags_i = topk_lags[:, idx_i]
                lags_j = topk_lags[:, idx_j]
                h_i = topk_heights[:, idx_i]
                h_j = topk_heights[:, idx_j]
                pair_valid = valid_peak[:, idx_i] & valid_peak[:, idx_j]

                ratio = torch.where(
                    lags_j >= lags_i,
                    lags_j / lags_i.clamp(min=1),
                    lags_i / lags_j.clamp(min=1),
                )
                log2_r = torch.log2(ratio.clamp(min=1.0)) % 1.0
                weight = torch.sqrt(h_i * h_j + 1e-8) * pair_valid.float()

                # Soft assignment to 32 uniform centers
                chunk_result = _soft_assign_to_centers(log2_r, weight, centers)

        all_chunk_results.append(chunk_result)

    result = torch.cat(all_chunk_results, dim=0).reshape(B, T_stft, n_bins)

    # Apply energy gate
    result = result * active.unsqueeze(-1).float()

    # Normalize per frame
    frame_sum = result.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    result = result / frame_sum
    result = result * active.unsqueeze(-1).float()

    if target_length is not None:
        result = result.transpose(1, 2)
        result = F.interpolate(result, size=target_length, mode='linear', align_corners=False)
        result = result.transpose(1, 2)

    return result


# --- A10c: RQA generic 6d (control — no JI projection, VECTORIZED) ---

def _batched_rqa_metrics(rp: torch.Tensor) -> torch.Tensor:
    """Vectorized RQA metrics for batched recurrence matrices.

    Computes 6 metrics without per-diagonal/per-column Python loops.
    Uses neighbor-based approximation for DET/LAM that captures the same
    structural information as line-counting but in O(N²) tensor ops.

    Args:
        rp: [BT, N, N] boolean recurrence matrices

    Returns:
        [BT, 6] tensor: %REC, %DET, %LAM, Lmean, ENTR, TT
    """
    BT, N, _ = rp.shape
    device = rp.device
    rp_f = rp.float()

    # 1. %REC: fraction of recurrent points
    n_rec = rp_f.sum(dim=(-2, -1))  # [BT]
    rec_rate = n_rec / (N * N)

    # 2. %DET: fraction of recurrent points on diagonal lines (length >= 2)
    # A point (i,j) is on a diagonal line if it AND a diagonal neighbor are recurrent
    # Diagonal neighbor: (i-1,j-1) or (i+1,j+1)
    has_prev_diag = F.pad(rp_f[:, 1:, 1:], (1, 0, 1, 0))  # [BT, N, N]
    has_next_diag = F.pad(rp_f[:, :-1, :-1], (0, 1, 0, 1))  # [BT, N, N]
    on_diag_line = rp_f * torch.clamp(has_prev_diag + has_next_diag, max=1.0)
    n_det = on_diag_line.sum(dim=(-2, -1))
    det = n_det / n_rec.clamp(min=1.0)

    # 3. %LAM: fraction of recurrent points on vertical lines (length >= 2)
    has_up = F.pad(rp_f[:, 1:, :], (0, 0, 1, 0))  # shifted up
    has_down = F.pad(rp_f[:, :-1, :], (0, 0, 0, 1))  # shifted down
    on_vert_line = rp_f * torch.clamp(has_up + has_down, max=1.0)
    n_lam = on_vert_line.sum(dim=(-2, -1))
    lam = n_lam / n_rec.clamp(min=1.0)

    # 4. Lmean: average diagonal continuation length
    # Count diagonal continuations: points where both (i,j) and (i+1,j+1) are True
    diag_cont = rp_f[:, :-1, :-1] * rp_f[:, 1:, 1:]  # [BT, N-1, N-1]
    n_cont = diag_cont.sum(dim=(-2, -1))
    # Lmean ≈ (n_det_points / n_starts) where n_starts = n_det - n_cont
    n_starts = (n_det - n_cont).clamp(min=1.0)
    l_mean = n_det / n_starts

    # 5. ENTR: entropy of diagonal recurrence profile
    # Compute per-diagonal recurrence rates, then entropy of the distribution
    max_offset = N - 1  # full offset range (consistent with A10b diagonal profile)
    rates = torch.zeros(BT, max_offset, device=device)
    for offset in range(1, max_offset + 1):
        diag = torch.diagonal(rp_f, offset=offset, dim1=-2, dim2=-1)  # [BT, N-offset]
        rates[:, offset - 1] = diag.mean(dim=-1)
    # Normalize to probability distribution
    rate_sum = rates.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    probs = rates / rate_sum
    # Shannon entropy
    log_probs = torch.log(probs.clamp(min=1e-10))
    entr = -(probs * log_probs).sum(dim=-1)

    # 6. TT (trapping time): average vertical continuation length
    vert_cont = rp_f[:, :-1, :] * rp_f[:, 1:, :]  # [BT, N-1, N]
    n_vert_cont = vert_cont.sum(dim=(-2, -1))
    n_vert_starts = (n_lam - n_vert_cont).clamp(min=1.0)
    tt = n_lam / n_vert_starts

    return torch.stack([rec_rate, det, lam, l_mean, entr, tt], dim=-1)  # [BT, 6]


def compute_audio_descriptor_a10c(
    audio: torch.Tensor,
    target_length: int = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    sample_rate: int = 24000,
    m: int = 3,
    tau: int = 1,
    stride: int = 16,
    threshold_pct: float = 10.0,
) -> torch.Tensor:
    """A10c: Generic RQA metrics (control for A10b, vectorized).

    Same delay embedding + recurrence as A10b, but extracts 6 RQA
    metrics instead of projecting onto JI attractors.

    Metrics: %REC, %DET, %LAM, Lmean, ENTR, TT

    Args:
        audio: [B, T_samples] raw waveform
        target_length: T' for interpolation (None = native STFT resolution)

    Returns:
        [B, T, 6] RQA metrics per frame, z-scored per sample
    """
    B = audio.size(0)
    device = audio.device

    # Frame the audio
    pad = n_fft // 2
    audio_padded = F.pad(audio, (pad, pad), mode='reflect')
    frames = audio_padded.unfold(1, n_fft, hop_length)  # [B, T, n_fft]
    T_stft = frames.size(1)
    window = torch.hann_window(n_fft, device=device)
    frames = frames * window

    # Energy gate
    energy = (frames ** 2).mean(dim=-1)  # [B, T]
    active = energy > 1e-10

    # Batched delay embedding
    sub = frames[:, :, ::stride]  # [B, T, N_sub]
    N_sub = sub.size(2)
    N_vec = N_sub - (m - 1) * tau
    if N_vec < 3:
        result = torch.zeros(B, T_stft, 6, device=device)
        if target_length is not None:
            result = result.transpose(1, 2)
            result = F.interpolate(result, size=target_length, mode='linear', align_corners=False)
            result = result.transpose(1, 2)
        return result

    idx = torch.arange(N_vec, device=device).unsqueeze(1) + \
          torch.arange(m, device=device).unsqueeze(0) * tau
    embedded = sub[:, :, idx]  # [B, T, N_vec, m]

    BT = B * T_stft
    emb_flat = embedded.reshape(BT, N_vec, m)

    # Chunked processing to avoid memory explosion (Codex ALTA fix)
    CDIST_CHUNK = 512
    k_idx = max(1, int(N_vec * N_vec * threshold_pct / 100.0))

    all_chunk_results = []
    for start in range(0, BT, CDIST_CHUNK):
        end = min(start + CDIST_CHUNK, BT)
        chunk_emb = emb_flat[start:end]
        chunk_sz = end - start

        dists = torch.cdist(chunk_emb, chunk_emb)
        flat_dists = dists.reshape(chunk_sz, -1)
        thresholds = flat_dists.kthvalue(k_idx, dim=1).values
        chunk_rp = dists <= thresholds.unsqueeze(-1).unsqueeze(-1)
        del dists, flat_dists

        chunk_result = _batched_rqa_metrics(chunk_rp)  # [chunk_sz, 6]
        del chunk_rp
        all_chunk_results.append(chunk_result)

    result = torch.cat(all_chunk_results, dim=0).reshape(B, T_stft, 6)

    # Apply energy gate
    result = result * active.unsqueeze(-1).float()

    # Z-score normalize per sample
    mean = result.mean(dim=1, keepdim=True)
    std = result.std(dim=1, keepdim=True).clamp(min=1e-8)
    result = (result - mean) / std
    result = result * active.unsqueeze(-1).float()  # re-zero silent

    if target_length is not None:
        result = result.transpose(1, 2)
        result = F.interpolate(result, size=target_length, mode='linear', align_corners=False)
        result = result.transpose(1, 2)

    return result
