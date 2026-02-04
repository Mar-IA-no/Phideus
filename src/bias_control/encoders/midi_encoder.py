"""
MIDI Encoder: Transformer-based encoder for MIDI events.

Processes MIDI as a sequence of note events with pitch, velocity,
onset time, and duration information.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time-aware embeddings."""

    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] input embeddings
        Returns:
            [B, T, D] with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MIDIEventEmbedding(nn.Module):
    """
    Embedding layer for MIDI events.

    Each event has: pitch (0-127), velocity (0-127), duration_bucket (0-31)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        n_pitches: int = 128,
        n_velocities: int = 128,
        n_duration_buckets: int = 32,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Separate embeddings for each attribute
        self.pitch_embed = nn.Embedding(n_pitches, embed_dim // 2)
        self.velocity_embed = nn.Embedding(n_velocities, embed_dim // 4)
        self.duration_embed = nn.Embedding(n_duration_buckets, embed_dim // 4)

        # Combine embeddings
        self.combine = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        pitch: torch.Tensor,
        velocity: torch.Tensor,
        duration: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pitch: [B, T] pitch values (0-127)
            velocity: [B, T] velocity values (0-127)
            duration: [B, T] duration bucket indices (0-31)

        Returns:
            [B, T, D] event embeddings
        """
        p_emb = self.pitch_embed(pitch)  # [B, T, D/2]
        v_emb = self.velocity_embed(velocity)  # [B, T, D/4]
        d_emb = self.duration_embed(duration)  # [B, T, D/4]

        # Concatenate and combine
        combined = torch.cat([p_emb, v_emb, d_emb], dim=-1)  # [B, T, D]
        out = self.combine(combined)
        out = self.norm(out)

        return out


class MIDIEncoder(nn.Module):
    """
    Transformer encoder for MIDI sequences.

    Architecture:
        MIDI Events → Event Embedding → Positional Encoding → Transformer → Pooling
    """

    def __init__(
        self,
        embed_dim: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        aggregation: str = "mean",  # "mean", "cls", "attention"
    ):
        """
        Args:
            embed_dim: Embedding dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            dim_feedforward: FFN hidden dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            aggregation: How to pool sequence to single vector
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.aggregation = aggregation

        # Event embedding
        self.event_embedding = MIDIEventEmbedding(embed_dim=embed_dim)

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=embed_dim,
            max_len=max_seq_len,
            dropout=dropout
        )

        # CLS token for sequence representation (if using cls aggregation)
        if aggregation == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Attention pooling (if using attention aggregation)
        if aggregation == "attention":
            self.attention_pool = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.Tanh(),
                nn.Linear(embed_dim // 4, 1)
            )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True  # Pre-norm for better stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False
        )

        # Output norm
        self.output_norm = nn.LayerNorm(embed_dim)

        self.output_dim = embed_dim

    def forward(
        self,
        pitch: torch.Tensor,
        velocity: torch.Tensor,
        duration: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_sequence: bool = False
    ) -> torch.Tensor:
        """
        Encode MIDI sequence.

        Args:
            pitch: [B, T] pitch values
            velocity: [B, T] velocity values
            duration: [B, T] duration bucket indices
            padding_mask: [B, T] True for padding positions
            return_sequence: If True, return full sequence instead of pooled

        Returns:
            [B, D] pooled embedding or [B, T, D] if return_sequence
        """
        B, T = pitch.shape

        # Event embedding
        x = self.event_embedding(pitch, velocity, duration)  # [B, T, D]

        # Add CLS token if using cls aggregation
        if self.aggregation == "cls":
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
            x = torch.cat([cls_tokens, x], dim=1)  # [B, T+1, D]
            if padding_mask is not None:
                # Add False for CLS token position
                cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device)
                padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        # Positional encoding
        x = self.pos_encoding(x)

        # Transformer
        if padding_mask is not None:
            x = self.transformer(x, src_key_padding_mask=padding_mask)
        else:
            x = self.transformer(x)

        # Output norm
        x = self.output_norm(x)

        if return_sequence:
            return x

        # Aggregate to single vector
        if self.aggregation == "mean":
            if padding_mask is not None:
                # Masked mean
                mask = ~padding_mask.unsqueeze(-1)  # [B, T, 1]
                x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                x = x.mean(dim=1)
        elif self.aggregation == "cls":
            x = x[:, 0, :]  # CLS token
        elif self.aggregation == "attention":
            # Attention-weighted pooling
            weights = self.attention_pool(x)  # [B, T, 1]
            if padding_mask is not None:
                weights = weights.masked_fill(padding_mask.unsqueeze(-1), float("-inf"))
            weights = torch.softmax(weights, dim=1)
            x = (x * weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return x  # [B, D]

    def get_output_dim(self) -> int:
        """Return output embedding dimension."""
        return self.output_dim


def duration_to_bucket(duration_sec: torch.Tensor, n_buckets: int = 32) -> torch.Tensor:
    """
    Convert duration in seconds to bucket index.

    Uses log-scale buckets to handle wide range of durations.
    Bucket 0: duration < 0.05s
    Bucket 31: duration > 4s

    Args:
        duration_sec: Duration in seconds
        n_buckets: Number of buckets

    Returns:
        Bucket indices (0 to n_buckets-1)
    """
    # Log-scale boundaries from 0.05s to 4s
    min_dur = 0.05
    max_dur = 4.0

    # Clamp durations
    duration_sec = duration_sec.clamp(min=min_dur, max=max_dur)

    # Log scale
    log_dur = torch.log(duration_sec / min_dur)
    log_max = math.log(max_dur / min_dur)

    # Map to buckets
    buckets = (log_dur / log_max * (n_buckets - 1)).long()
    buckets = buckets.clamp(0, n_buckets - 1)

    return buckets


def piano_roll_to_events(
    piano_roll: torch.Tensor,
    time_resolution: float = 0.01,
    max_events: int = 2048
) -> Dict[str, torch.Tensor]:
    """
    Convert piano roll to event sequence.

    Args:
        piano_roll: [T, 128] or [B, T, 128] piano roll (velocity values)
        time_resolution: Time per frame in seconds
        max_events: Maximum number of events to return

    Returns:
        Dict with pitch, velocity, duration tensors [B, N] or [N]
    """
    if piano_roll.dim() == 2:
        # Single sample
        piano_roll = piano_roll.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    B, T, n_pitches = piano_roll.shape
    device = piano_roll.device

    batch_pitches = []
    batch_velocities = []
    batch_durations = []

    for b in range(B):
        pr = piano_roll[b]  # [T, 128]

        # Find note onsets (velocity > 0 and previous frame was 0)
        active = pr > 0
        onset = active & ~torch.cat([torch.zeros(1, n_pitches, device=device).bool(), active[:-1]], dim=0)

        # Get onset positions
        onset_positions = torch.nonzero(onset)  # [N, 2] (time, pitch)

        if onset_positions.size(0) == 0:
            # No notes - add dummy
            batch_pitches.append(torch.zeros(1, dtype=torch.long, device=device))
            batch_velocities.append(torch.zeros(1, dtype=torch.long, device=device))
            batch_durations.append(torch.zeros(1, dtype=torch.long, device=device))
            continue

        # Sort by time
        sort_idx = onset_positions[:, 0].argsort()
        onset_positions = onset_positions[sort_idx]

        # Limit to max_events
        if onset_positions.size(0) > max_events:
            onset_positions = onset_positions[:max_events]

        n_events = onset_positions.size(0)

        pitches = onset_positions[:, 1]
        times = onset_positions[:, 0]

        # Get velocities at onset
        velocities = pr[times, pitches].long().clamp(0, 127)

        # Compute durations
        durations = []
        for i in range(n_events):
            t, p = times[i].item(), pitches[i].item()
            # Find note end
            for t_end in range(t + 1, min(t + 500, T)):  # Max 5 seconds
                if pr[t_end, p] == 0:
                    break
            dur_sec = (t_end - t) * time_resolution
            durations.append(dur_sec)

        durations = torch.tensor(durations, device=device)
        duration_buckets = duration_to_bucket(durations)

        batch_pitches.append(pitches)
        batch_velocities.append(velocities)
        batch_durations.append(duration_buckets)

    # Pad to same length
    max_len = max(p.size(0) for p in batch_pitches)

    padded_pitches = torch.zeros(B, max_len, dtype=torch.long, device=device)
    padded_velocities = torch.zeros(B, max_len, dtype=torch.long, device=device)
    padded_durations = torch.zeros(B, max_len, dtype=torch.long, device=device)
    padding_mask = torch.ones(B, max_len, dtype=torch.bool, device=device)

    for b in range(B):
        n = batch_pitches[b].size(0)
        padded_pitches[b, :n] = batch_pitches[b]
        padded_velocities[b, :n] = batch_velocities[b]
        padded_durations[b, :n] = batch_durations[b]
        padding_mask[b, :n] = False

    result = {
        "pitch": padded_pitches,
        "velocity": padded_velocities,
        "duration": padded_durations,
        "padding_mask": padding_mask
    }

    if squeeze_output:
        result = {k: v.squeeze(0) for k, v in result.items()}

    return result
