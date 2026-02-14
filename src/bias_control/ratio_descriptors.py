"""
Ratio Descriptors for Gate 4.2.

D3: Temporal-rhythmic ratio descriptor (IOI ratios + duration ratios + pitch intervals).
D4: Local interval features for input augmentation.

D1/D2 are computed directly via gate4_ratio_auxiliary.py functions
(compute_batch_ratio_histograms, compute_batch_ratio_histograms_enriched).
"""

import torch
from typing import Optional


# ---------------------------------------------------------------------------
# D3: Temporal-Rhythmic Ratio Descriptor
# ---------------------------------------------------------------------------

def compute_descriptor_d3(
    midi_pitch: torch.Tensor,
    midi_onset: torch.Tensor,
    midi_duration_sec: torch.Tensor,
    midi_mask: Optional[torch.Tensor],
    n_ioi_bins: int = 64,
    n_dur_bins: int = 64,
    n_pitch_bins: int = 25,
) -> torch.Tensor:
    """
    Compute temporal-rhythmic ratio descriptor.

    Combines three sub-descriptors:
    - IOI ratio histogram [B, n_ioi_bins]
    - Duration ratio histogram [B, n_dur_bins]
    - Pitch interval histogram [B, n_pitch_bins]

    Args:
        midi_pitch: [B, N] MIDI pitch values (0-127)
        midi_onset: [B, N] onset times in seconds (relative to segment start, -1 = padding)
        midi_duration_sec: [B, N] note durations in seconds (0 = padding)
        midi_mask: [B, N] True = padding
        n_ioi_bins: bins for IOI ratio histogram
        n_dur_bins: bins for duration ratio histogram
        n_pitch_bins: bins for pitch interval histogram (default 25 = [-12, +12])

    Returns:
        [B, n_ioi_bins + n_dur_bins + n_pitch_bins] concatenated descriptor
    """
    # D3 requires onset and duration data
    if midi_onset is None or midi_duration_sec is None:
        raise ValueError(
            "compute_descriptor_d3 requires midi_onset and midi_duration_sec. "
            "Ensure MaestroSegmentDataset is up-to-date (collate_segments must "
            "include these fields)."
        )

    # Build valid mask from midi_mask and onset validity
    if midi_mask is not None:
        valid = ~midi_mask  # [B, N] True = real note
    else:
        valid = torch.ones_like(midi_pitch, dtype=torch.bool)

    # Also exclude padding markers from onset
    valid = valid & (midi_onset >= 0)

    ioi_hist = _compute_ioi_ratio_histogram(midi_onset, valid, n_bins=n_ioi_bins)
    dur_hist = _compute_duration_ratio_histogram(midi_duration_sec, valid, n_bins=n_dur_bins)
    pitch_hist = _compute_pitch_interval_histogram(midi_pitch, valid, n_bins=n_pitch_bins)

    return torch.cat([ioi_hist, dur_hist, pitch_hist], dim=1)


def _compute_ioi_ratio_histogram(
    onsets: torch.Tensor,
    valid: torch.Tensor,
    n_bins: int = 64,
    min_ioi: float = 0.01,
    ratio_min: float = 0.25,
    ratio_max: float = 4.0,
) -> torch.Tensor:
    """
    IOI (Inter-Onset Interval) ratio histogram.

    For consecutive valid notes, compute IOI_n / IOI_{n+1}.
    Soft Gaussian binning in [ratio_min, ratio_max].

    Args:
        onsets: [B, N] onset times in seconds
        valid: [B, N] True = real note
        n_bins: number of histogram bins

    Returns:
        [B, n_bins] normalized histogram
    """
    B, N = onsets.shape
    device = onsets.device

    if N < 3:
        return torch.zeros(B, n_bins, device=device)

    # Compute IOIs between consecutive notes: [B, N-1]
    ioi = onsets[:, 1:] - onsets[:, :-1]

    # Valid IOI: both notes valid and IOI > min_ioi
    valid_pair = valid[:, 1:] & valid[:, :-1]
    ioi_valid = valid_pair & (ioi > min_ioi)

    # Compute consecutive IOI ratios: IOI_n / IOI_{n+1} → [B, N-2]
    ioi_curr = ioi[:, :-1]  # [B, N-2]
    ioi_next = ioi[:, 1:]   # [B, N-2]

    # Both IOIs must be valid
    ratio_valid = ioi_valid[:, :-1] & ioi_valid[:, 1:]

    # Compute ratio (safe division)
    ratios = ioi_curr / (ioi_next + 1e-8)  # [B, N-2]
    ratios = ratios.clamp(ratio_min, ratio_max)

    # Soft Gaussian binning
    bin_edges = torch.linspace(ratio_min, ratio_max, n_bins + 1, device=device)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # [n_bins]
    sigma = (ratio_max - ratio_min) / n_bins

    # [B, N-2, 1] vs [1, 1, n_bins]
    ratios_exp = ratios.unsqueeze(2)
    centers_exp = bin_centers.view(1, 1, -1)
    kernel = torch.exp(-0.5 * ((ratios_exp - centers_exp) / sigma) ** 2)  # [B, N-2, n_bins]

    # Mask
    kernel = kernel * ratio_valid.unsqueeze(2).float()

    # Sum and normalize
    histogram = kernel.sum(dim=1)  # [B, n_bins]
    histogram = histogram / (histogram.sum(dim=1, keepdim=True) + 1e-8)

    return histogram


def _compute_duration_ratio_histogram(
    durations: torch.Tensor,
    valid: torch.Tensor,
    n_bins: int = 64,
    min_dur: float = 0.05,
    ratio_min: float = 0.25,
    ratio_max: float = 4.0,
) -> torch.Tensor:
    """
    Duration ratio histogram for consecutive notes.

    dur_n / dur_{n+1}, soft Gaussian binning.

    Args:
        durations: [B, N] note durations in seconds
        valid: [B, N] True = real note
        n_bins: number of histogram bins

    Returns:
        [B, n_bins] normalized histogram
    """
    B, N = durations.shape
    device = durations.device

    if N < 2:
        return torch.zeros(B, n_bins, device=device)

    # Both notes valid and duration > min
    dur_valid = valid & (durations > min_dur)
    pair_valid = dur_valid[:, :-1] & dur_valid[:, 1:]  # [B, N-1]

    dur_curr = durations[:, :-1]  # [B, N-1]
    dur_next = durations[:, 1:]   # [B, N-1]

    ratios = dur_curr / (dur_next + 1e-8)
    ratios = ratios.clamp(ratio_min, ratio_max)

    # Soft Gaussian binning
    bin_edges = torch.linspace(ratio_min, ratio_max, n_bins + 1, device=device)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    sigma = (ratio_max - ratio_min) / n_bins

    ratios_exp = ratios.unsqueeze(2)
    centers_exp = bin_centers.view(1, 1, -1)
    kernel = torch.exp(-0.5 * ((ratios_exp - centers_exp) / sigma) ** 2)
    kernel = kernel * pair_valid.unsqueeze(2).float()

    histogram = kernel.sum(dim=1)
    histogram = histogram / (histogram.sum(dim=1, keepdim=True) + 1e-8)

    return histogram


def _compute_pitch_interval_histogram(
    pitches: torch.Tensor,
    valid: torch.Tensor,
    n_bins: int = 25,
) -> torch.Tensor:
    """
    Pitch interval histogram for consecutive notes.

    Semitone differences in [-12, +12], hard binning (25 bins).

    Args:
        pitches: [B, N] MIDI pitch values (0-127)
        valid: [B, N] True = real note
        n_bins: number of bins (default 25 = [-12, +12] inclusive)

    Returns:
        [B, n_bins] normalized histogram
    """
    B, N = pitches.shape
    device = pitches.device

    if N < 2:
        return torch.zeros(B, n_bins, device=device)

    pair_valid = valid[:, :-1] & valid[:, 1:]  # [B, N-1]

    # Semitone intervals
    intervals = pitches[:, 1:].float() - pitches[:, :-1].float()  # [B, N-1]

    # Clamp to [-12, +12] and shift to [0, 24] for bin indexing
    half_range = n_bins // 2  # 12
    intervals_clamped = intervals.clamp(-half_range, half_range)
    bin_indices = (intervals_clamped + half_range).long()  # [0, 24]
    bin_indices = bin_indices.clamp(0, n_bins - 1)

    # Hard binning with scatter
    histogram = torch.zeros(B, n_bins, device=device)
    weight = pair_valid.float()  # [B, N-1]

    # Scatter add
    histogram.scatter_add_(1, bin_indices, weight)

    # Normalize
    histogram = histogram / (histogram.sum(dim=1, keepdim=True) + 1e-8)

    return histogram


# ---------------------------------------------------------------------------
# D4: Local Interval Features (for input augmentation)
# ---------------------------------------------------------------------------

def compute_local_interval_features(
    midi_pitch: torch.Tensor,
    midi_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Compute per-note local interval features for D4 input augmentation.

    Returns 4 features per note:
    1. semitone_prev: (pitch[i] - pitch[i-1]) / 24, 0 for first note
    2. semitone_next: (pitch[i+1] - pitch[i]) / 24, 0 for last note
    3. log_ratio_prev: (pitch[i] - pitch[i-1]) / 12, clamped [-2,2], /2
    4. log_ratio_next: (pitch[i+1] - pitch[i]) / 12, clamped [-2,2], /2

    All values are normalized to roughly [-1, 1]. Padding positions zeroed.

    Args:
        midi_pitch: [B, N] MIDI pitch values (0-127)
        midi_mask: [B, N] True = padding

    Returns:
        [B, N, 4] local interval features
    """
    B, N = midi_pitch.shape
    device = midi_pitch.device
    pitch_f = midi_pitch.float()

    # Compute intervals: pitch[i+1] - pitch[i] → [B, N-1]
    diff_forward = pitch_f[:, 1:] - pitch_f[:, :-1]  # [B, N-1]

    # Build pair-validity masks to avoid contamination at padding boundaries
    if midi_mask is not None:
        valid = ~midi_mask  # [B, N] True = real note
        # valid_prev_pair[i] = True iff note i AND note i-1 are both valid
        valid_prev_pair = valid[:, 1:] & valid[:, :-1]  # [B, N-1]
        # valid_next_pair[i] = True iff note i AND note i+1 are both valid
        valid_next_pair = valid[:, :-1] & valid[:, 1:]  # [B, N-1]
    else:
        valid_prev_pair = torch.ones(B, N - 1, device=device, dtype=torch.bool)
        valid_next_pair = valid_prev_pair

    # semitone_prev: for note i, interval from i-1 to i
    # diff_forward[i-1] = pitch[i] - pitch[i-1]
    semitone_prev = torch.zeros(B, N, device=device)
    semitone_prev[:, 1:] = (diff_forward / 24.0) * valid_prev_pair.float()

    # semitone_next: for note i, interval from i to i+1
    # diff_forward[i] = pitch[i+1] - pitch[i]
    semitone_next = torch.zeros(B, N, device=device)
    semitone_next[:, :-1] = (diff_forward / 24.0) * valid_next_pair.float()

    # log_ratio_prev: same interval but /12 and clamped
    log_ratio_prev = torch.zeros(B, N, device=device)
    log_ratio_prev[:, 1:] = (diff_forward / 12.0).clamp(-2.0, 2.0) / 2.0 * valid_prev_pair.float()

    # log_ratio_next
    log_ratio_next = torch.zeros(B, N, device=device)
    log_ratio_next[:, :-1] = (diff_forward / 12.0).clamp(-2.0, 2.0) / 2.0 * valid_next_pair.float()

    # Stack features: [B, N, 4]
    features = torch.stack([semitone_prev, semitone_next, log_ratio_prev, log_ratio_next], dim=2)

    # Zero out padding positions
    if midi_mask is not None:
        features = features * valid.unsqueeze(2).float()

    return features
