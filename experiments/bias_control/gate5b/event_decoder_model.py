#!/usr/bin/env python3
"""Autoregressive event decoder for Test11 perceptual MIDI generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class EventDecoderConfig:
    z_dim: int = 256
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    n_cond_tokens: int = 16
    max_seq_len: int = 512
    vocab_size: int = 512
    pad_id: int = 0


class ConditionedEventTransformerDecoder(nn.Module):
    """Decode MIDI event tokens autoregressively from frozen embedding z."""

    def __init__(self, config: Optional[EventDecoderConfig] = None):
        super().__init__()
        self.cfg = config or EventDecoderConfig()

        self.z_to_memory = nn.Linear(
            self.cfg.z_dim,
            self.cfg.n_cond_tokens * self.cfg.d_model,
        )
        self.memory_norm = nn.LayerNorm(self.cfg.d_model)

        self.token_emb = nn.Embedding(
            self.cfg.vocab_size,
            self.cfg.d_model,
            padding_idx=self.cfg.pad_id,
        )
        self.pos_emb = nn.Parameter(torch.zeros(self.cfg.max_seq_len, self.cfg.d_model))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=self.cfg.d_model,
            nhead=self.cfg.n_heads,
            dim_feedforward=self.cfg.d_ff,
            dropout=self.cfg.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=self.cfg.n_layers)
        self.final_norm = nn.LayerNorm(self.cfg.d_model)
        self.head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _build_memory(self, z: torch.Tensor) -> torch.Tensor:
        bsz = z.shape[0]
        memory = self.z_to_memory(z).view(bsz, self.cfg.n_cond_tokens, self.cfg.d_model)
        return self.memory_norm(memory)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
            diagonal=1,
        )
        return mask

    def forward(self, z: torch.Tensor, input_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, z_dim]
            input_tokens: [B, T] (teacher-forcing input)

        Returns:
            logits: [B, T, vocab_size]
        """
        bsz, seq_len = input_tokens.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"input seq_len={seq_len} exceeds max_seq_len={self.cfg.max_seq_len}")

        memory = self._build_memory(z)

        tgt = self.token_emb(input_tokens)
        tgt = tgt + self.pos_emb[:seq_len].unsqueeze(0)

        tgt_mask = self._causal_mask(seq_len, input_tokens.device)
        tgt_key_padding_mask = input_tokens.eq(self.cfg.pad_id)

        hidden = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        hidden = self.final_norm(hidden)
        return self.head(hidden)

    @torch.no_grad()
    def generate(
        self,
        z: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 32,
        top_p: float = 0.95,
        min_len: int = 8,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Autoregressive sampling with temperature + top-k + top-p."""
        if max_len is None:
            max_len = self.cfg.max_seq_len
        max_len = min(max_len, self.cfg.max_seq_len)
        if max_len < 2:
            raise ValueError('max_len must be >= 2')

        bsz = z.shape[0]
        tokens = torch.full(
            (bsz, 1),
            int(bos_id),
            dtype=torch.long,
            device=z.device,
        )
        finished = torch.zeros(bsz, dtype=torch.bool, device=z.device)

        for _ in range(max_len - 1):
            logits = self.forward(z, tokens)[:, -1, :]
            if tokens.shape[1] < min_len:
                logits[:, eos_id] = float('-inf')
            next_token = sample_next_token(
                logits=logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                deterministic=deterministic,
            )

            next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)
            tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
            finished = finished | next_token.eq(eos_id)

            if bool(finished.all()) and tokens.shape[1] >= min_len:
                break

        return tokens


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 32,
    top_p: float = 0.95,
    deterministic: bool = False,
) -> torch.Tensor:
    """Sample next token ids from logits with optional top-k/top-p filtering."""
    if deterministic:
        return torch.argmax(logits, dim=-1)

    if temperature <= 0:
        temperature = 1.0

    x = logits / temperature

    if top_k > 0 and top_k < x.shape[-1]:
        kth_vals = torch.topk(x, top_k, dim=-1).values[:, -1].unsqueeze(-1)
        x = torch.where(x < kth_vals, torch.full_like(x, float('-inf')), x)

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(x, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > top_p
        remove[:, 0] = False
        sorted_logits = torch.where(remove, torch.full_like(sorted_logits, float('-inf')), sorted_logits)
        x = torch.full_like(x, float('-inf'))
        x.scatter_(1, sorted_idx, sorted_logits)

    probs = torch.softmax(x, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs_sum = probs.sum(dim=-1, keepdim=True)
    probs = torch.where(probs_sum > 0, probs / probs_sum, torch.full_like(probs, 1.0 / probs.shape[-1]))
    return torch.multinomial(probs, num_samples=1).squeeze(1)


def build_note_weight_vector(
    vocab_size: int,
    note_on_start: int,
    note_off_start: int,
    pitch_count: int,
    note_weight: float = 2.0,
) -> torch.Tensor:
    """Return CE class weights with extra weight for note on/off tokens."""
    w = torch.ones(vocab_size, dtype=torch.float32)
    w[note_on_start: note_on_start + pitch_count] = note_weight
    w[note_off_start: note_off_start + pitch_count] = note_weight
    return w
