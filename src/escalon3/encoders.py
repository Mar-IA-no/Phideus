"""Escalón 3: Audio and Image encoders for Lissajous scenes.

Baseline encoders (A0/I0):
  LissajousAudioEncoder: 4-layer 1D-CNN + AvgPool. ~462K params.
  LissajousImageEncoder: 4-layer 2D-CNN + AvgPool. ~389K params.

Sweep variants (audio only):
  LissajousAudioEncoderAttPool (A1): Same trunk + AttentiveStatsPool. ~530K params.
  LissajousAudioEncoderResAtt (A2): ResBlock trunk + AttentiveStatsPool. ~1.5M params.
  LissajousAudioEncoderMiniTransformer (A4): CNN + temporal compress + 2 Transformer layers. ~2.5M params.

Factory: build_lissajous_audio_encoder(name), build_lissajous_image_encoder(name).
"""

import math

import torch
import torch.nn as nn


# ============================================================
# Pooling modules
# ============================================================

class AttentiveStatsPool1d(nn.Module):
    """Attentive Statistics Pooling (Okabe 2018 / ECAPA-TDNN).

    [B, C, T] → attention-weighted mean || weighted std → [B, 2C]
    """

    def __init__(self, channels: int, bottleneck: int = None):
        super().__init__()
        bn = bottleneck or max(channels // 4, 32)
        self.attention = nn.Sequential(
            nn.Linear(channels, bn),
            nn.ReLU(),
            nn.Linear(bn, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, T] → [B, 2C]"""
        h = x.transpose(1, 2)                    # [B, T, C]
        w = self.attention(h)                     # [B, T, C]
        w = torch.softmax(w, dim=1)              # attention over time
        w = w.transpose(1, 2)                     # [B, C, T]
        mu = (x * w).sum(dim=2)                  # [B, C]
        var = ((x - mu.unsqueeze(2)) ** 2 * w).sum(dim=2)
        std = (var + 1e-8).sqrt()                # [B, C]
        return torch.cat([mu, std], dim=1)       # [B, 2C]


# ============================================================
# Building blocks
# ============================================================

class ResidualBlock1d(nn.Module):
    """Residual block: Conv → GN → GELU → Conv → GN + skip.

    Supports channel change (in_channels != out_channels) and stride > 1.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, pad),
            nn.GroupNorm(min(16, out_channels), out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, 1, pad),
            nn.GroupNorm(min(16, out_channels), out_channels),
        )
        # Skip connection
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv1d(in_channels, out_channels, 1, stride)
        else:
            self.skip = nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.skip(x))


# ============================================================
# A0: Baseline (original)
# ============================================================

class LissajousAudioEncoder(nn.Module):
    """A0: 4-layer 1D-CNN + AdaptiveAvgPool. ~462K params."""
    input_kind = "waveform"

    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.output_dim = output_dim
        self.features = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=10, stride=5, padding=0),
            nn.GroupNorm(16, 64), nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=0),
            nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=0),
            nn.GroupNorm(16, 256), nn.GELU(),
            nn.Conv1d(256, output_dim, kernel_size=4, stride=2, padding=0),
            nn.GroupNorm(16, output_dim), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.features(audio)
        return self.pool(x).squeeze(-1)


# ============================================================
# A1: Same trunk + Attentive Stats Pooling
# ============================================================

class LissajousAudioEncoderAttPool(nn.Module):
    """A1: Baseline CNN trunk + AttentiveStatsPool1d. ~530K params."""
    input_kind = "waveform"

    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.output_dim = output_dim
        self.features = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=10, stride=5, padding=0),
            nn.GroupNorm(16, 64), nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=0),
            nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=0),
            nn.GroupNorm(16, 256), nn.GELU(),
            nn.Conv1d(256, output_dim, kernel_size=4, stride=2, padding=0),
            nn.GroupNorm(16, output_dim), nn.GELU(),
        )
        self.pool = AttentiveStatsPool1d(output_dim)
        self.proj = nn.Linear(output_dim * 2, output_dim)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.features(audio)       # [B, D, T']
        x = self.pool(x)              # [B, 2D]
        return self.proj(x)           # [B, D]


# ============================================================
# A2: Residual trunk + Attentive Stats Pooling
# ============================================================

class LissajousAudioEncoderResAtt(nn.Module):
    """A2: ResBlock trunk + AttentiveStatsPool1d. ~1.5M params."""
    input_kind = "waveform"

    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.output_dim = output_dim

        # Stem: raw waveform → feature map
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=15, stride=5, padding=7),
            nn.GroupNorm(16, 64),
            nn.GELU(),
        )

        # Residual stages with progressive downsampling
        self.blocks = nn.Sequential(
            ResidualBlock1d(64, 64, kernel_size=3, stride=1),
            ResidualBlock1d(64, 128, kernel_size=3, stride=2),
            ResidualBlock1d(128, 128, kernel_size=3, stride=1),
            ResidualBlock1d(128, 256, kernel_size=3, stride=2),
            ResidualBlock1d(256, output_dim, kernel_size=3, stride=1),
        )

        self.pool = AttentiveStatsPool1d(output_dim)
        self.proj = nn.Linear(output_dim * 2, output_dim)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.stem(audio)           # [B, 64, ~17640]
        x = self.blocks(x)            # [B, D, ~4410]
        x = self.pool(x)              # [B, 2D]
        return self.proj(x)           # [B, D]


# ============================================================
# A4: CNN + Temporal Compress + Mini Transformer
# ============================================================

class LissajousAudioEncoderMiniTransformer(nn.Module):
    """A4: CNN feature extractor + temporal compression + 2 Transformer layers. ~2.5M params."""
    input_kind = "waveform"

    def __init__(self, output_dim: int = 256, n_transformer_layers: int = 2,
                 n_heads: int = 4):
        super().__init__()
        self.output_dim = output_dim

        # Same CNN trunk as baseline
        self.features = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=10, stride=5, padding=0),
            nn.GroupNorm(16, 64), nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=0),
            nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=0),
            nn.GroupNorm(16, 256), nn.GELU(),
            nn.Conv1d(256, output_dim, kernel_size=4, stride=2, padding=0),
            nn.GroupNorm(16, output_dim), nn.GELU(),
        )

        # Temporal compression: ~1100 tokens → ~275 tokens
        self.temporal_compress = nn.Sequential(
            nn.Conv1d(output_dim, output_dim, kernel_size=4, stride=4, padding=0),
            nn.GroupNorm(16, output_dim),
            nn.GELU(),
        )

        # Positional embedding (max 300 tokens after compression)
        self.pos_embedding = nn.Parameter(torch.randn(1, 300, output_dim) * 0.02)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=n_heads,
            dim_feedforward=output_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.features(audio)                  # [B, D, ~1100]
        x = self.temporal_compress(x)             # [B, D, ~275]
        x = x.transpose(1, 2)                    # [B, ~275, D]
        T = x.size(1)
        x = x + self.pos_embedding[:, :T, :]     # add positional
        x = self.transformer(x)                   # [B, T, D]
        return x.mean(dim=1)                      # [B, D]


# ============================================================
# Image encoder (unchanged)
# ============================================================

class LissajousImageEncoder(nn.Module):
    """I0: 4-layer 2D-CNN + AdaptiveAvgPool. ~389K params."""

    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.output_dim = output_dim
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 64), nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv2d(128, output_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, output_dim), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.features(image)
        x = self.pool(x)
        return x.squeeze(-1).squeeze(-1)


# ============================================================
# B1: CQT + Conv2d trunk (log-frequency representation test)
# ============================================================

class LissajousAudioEncoderCQTConv(nn.Module):
    """B1: CQT frontend (precomputed) + 2D conv trunk.

    Tests if log-frequency representation alone helps scale-OOD.
    Input: [B, 2, F, T] where F=252 freq bins, T=173 time frames (precomputed CQT).
    """
    input_kind = "cqt"

    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.output_dim = output_dim
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 64), nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv2d(128, output_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, output_dim), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 2, F, T] → [B, output_dim]"""
        x = self.features(x)
        return self.pool(x).squeeze(-1).squeeze(-1)


# ============================================================
# B2: CQT + Cross-Channel Shift-Correlation (ratio-aware)
# ============================================================

class LissajousAudioEncoderCQTShift(nn.Module):
    """B2: CQT + shared conv trunk + cross-channel frequency-shift correlation.

    Explicitly ratio-aware: compares the two XY channels at frequency offsets.
    In log-frequency, a fixed ratio = a fixed offset. This is the key inductive bias.
    """
    input_kind = "cqt"

    def __init__(self, output_dim: int = 256, delta_max: int = 24):
        super().__init__()
        self.output_dim = output_dim
        self.delta_max = delta_max
        n_deltas = 2 * delta_max + 1  # 49

        # Shared trunk for each channel
        self.trunk = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(2, 1), padding=1),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1),
            nn.GroupNorm(16, 64), nn.GELU(),
        )
        # After trunk: [B, 64, F'=63, T'=173]

        # Head: correlation profile → embedding
        self.head = nn.Sequential(
            nn.Linear(n_deltas * 2, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 2, F, T] → [B, output_dim]"""
        B = x.shape[0]
        # Separate channels and process through shared trunk
        fx = self.trunk(x[:, 0:1])  # [B, C, F', T']
        fy = self.trunk(x[:, 1:2])  # [B, C, F', T']

        C, Fp, Tp = fx.shape[1], fx.shape[2], fx.shape[3]

        # Cross-channel shift correlation.
        # Cosine-style normalization keeps offsets comparable even though
        # the valid support shrinks as |delta| grows.
        corrs = []
        eps = 1e-8
        for delta in range(-self.delta_max, self.delta_max + 1):
            if delta >= 0:
                fx_slice = fx[:, :, :Fp - delta, :]    # [B, C, Fp-|d|, T']
                fy_slice = fy[:, :, delta:, :]          # [B, C, Fp-|d|, T']
            else:
                fx_slice = fx[:, :, -delta:, :]         # [B, C, Fp-|d|, T']
                fy_slice = fy[:, :, :Fp + delta, :]     # [B, C, Fp-|d|, T']

            # Cosine-style similarity over channel+frequency axes, keep time.
            num = (fx_slice * fy_slice).sum(dim=(1, 2))  # [B, T']
            fx_norm = fx_slice.square().sum(dim=(1, 2)).sqrt()
            fy_norm = fy_slice.square().sum(dim=(1, 2)).sqrt()
            corr = num / (fx_norm * fy_norm).clamp_min(eps)
            corrs.append(corr)

        corr_map = torch.stack(corrs, dim=1)  # [B, n_deltas, T']

        # Stats pooling over time
        mu = corr_map.mean(dim=2)                         # [B, n_deltas]
        std = (corr_map.var(dim=2) + 1e-8).sqrt()        # [B, n_deltas]
        pooled = torch.cat([mu, std], dim=1)              # [B, 2*n_deltas]

        return self.head(pooled)                           # [B, output_dim]


# ============================================================
# B3: CQT Hybrid (absolute + relative branches)
# ============================================================

class LissajousAudioEncoderCQTHybrid(nn.Module):
    """B3: Dual-branch — absolute (cqtconv) + relative (cqtshift) fused.

    Tests if combining absolute spectral info with ratio-aware shift correlation helps.
    """
    input_kind = "cqt"

    def __init__(self, output_dim: int = 256, delta_max: int = 24):
        super().__init__()
        self.output_dim = output_dim
        self.abs_branch = LissajousAudioEncoderCQTConv(output_dim)
        self.rel_branch = LissajousAudioEncoderCQTShift(output_dim, delta_max)
        self.fuse = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 2, F, T] → [B, output_dim]"""
        z_abs = self.abs_branch(x)
        z_rel = self.rel_branch(x)
        return self.fuse(torch.cat([z_abs, z_rel], dim=1))


# ============================================================
# Factory functions
# ============================================================

AUDIO_ENCODERS = {
    'baseline': LissajousAudioEncoder,
    'attpool': LissajousAudioEncoderAttPool,
    'resatt': LissajousAudioEncoderResAtt,
    'cnnxfmr': LissajousAudioEncoderMiniTransformer,
    'cqtconv': LissajousAudioEncoderCQTConv,
    'cqtshift': LissajousAudioEncoderCQTShift,
    'cqthybrid': LissajousAudioEncoderCQTHybrid,
}

IMAGE_ENCODERS = {
    'baseline': LissajousImageEncoder,
}


def build_lissajous_audio_encoder(name: str, output_dim: int = 256) -> nn.Module:
    """Build audio encoder by name."""
    if name not in AUDIO_ENCODERS:
        raise ValueError(f"Unknown audio encoder '{name}'. Options: {list(AUDIO_ENCODERS.keys())}")
    return AUDIO_ENCODERS[name](output_dim=output_dim)


def build_lissajous_image_encoder(name: str, output_dim: int = 256) -> nn.Module:
    """Build image encoder by name."""
    if name not in IMAGE_ENCODERS:
        raise ValueError(f"Unknown image encoder '{name}'. Options: {list(IMAGE_ENCODERS.keys())}")
    return IMAGE_ENCODERS[name](output_dim=output_dim)
