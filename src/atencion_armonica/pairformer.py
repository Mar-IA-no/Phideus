"""Harmonic Pairformer — los 6 modelos de Fase 0 Atención Armónica.

A-naive  : token self-attention con bias de posición relativa (Δlog-f). Readout pairwise
           solo desde tokens. SIN pair features armónicas.
A-rich   : igual, pero el readout pairwise recibe ADEMÁS las pair features armónicas
           (mismas que B). SIN pair state, SIN triangle. Baseline DECISIVO (param-match con B).
B        : pair representation z[i,j] init desde pair features; bloques = token-attn sesgada
           por z + pair update (outer-product) + TRIANGLE multiplicative update. Readout desde z.
B-minus  : B sin el paso de mixing de pares (sin triangle). Quita módulo + params.
B-local  : B con un mixing LOCAL por par (mismos params que triangle, SIN suma sobre k).
           Param-matched a B → B vs B-local aísla la propagación de transitividad (Codex r1 #2).
B-shuffle: B con pair init shuffleado determinísticamente por mezcla (control negativo parcial).

Triangle update (Codex r1 #6): mask-aware (excluye padding y k∈{i,j}), normalizado por |K_ij|,
scale aprendible. Triangle y Local NO modifican la diagonal (self-pair; Codex r1/r2).
Simetrización de z tras cada pair update para TODOS los variants (Codex r1 #1).
Pair features (Codex r1 #3): anti-leakage, definidas en peak_tokens.compute_pair_features.

MODEL_CONFIGS abajo congela los dims por modelo (param-match A-rich vs B ~1.1%). El training
DEBE usar build_model(name) que toma el config congelado — NO pasar un d_model global (Codex r2 #3).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.atencion_armonica.peak_tokens import N_PAIR_CONT_FEATS, n_ratio_classes

NEG_INF = -1e9


# ---------------------------------------------------------------------------
# Bloques compartidos
# ---------------------------------------------------------------------------

class PairFeatureEmbed(nn.Module):
    """Pair features (cont [.,4] + ratio_class_id) → vector [B,N,N,pf_dim]. Simétrico por construcción."""

    def __init__(self, pf_dim: int = 32, class_emb_dim: int = 8):
        super().__init__()
        self.class_emb = nn.Embedding(n_ratio_classes(), class_emb_dim)
        self.proj = nn.Linear(N_PAIR_CONT_FEATS + class_emb_dim, pf_dim)

    def forward(self, pair_cont: torch.Tensor, ratio_class_id: torch.Tensor) -> torch.Tensor:
        emb = self.class_emb(ratio_class_id)                     # [B,N,N,class_emb_dim]
        x = torch.cat([pair_cont, emb], dim=-1)                  # [B,N,N,F+emb]
        return self.proj(x)                                      # [B,N,N,pf_dim]


def _attention(q, k, v, bias, key_mask):
    """q,k,v [B,H,N,dh]; bias [B,H,N,N] o None; key_mask [B,N] bool."""
    dh = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dh)   # [B,H,N,N]
    if bias is not None:
        scores = scores + bias
    scores = scores.masked_fill(~key_mask[:, None, None, :], NEG_INF)
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)                                    # [B,H,N,dh]


class TokenAttnLayer(nn.Module):
    """Self-attention sobre tokens con bias aditivo por par + FFN."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model {d_model} no divisible por n_heads {n_heads}"
        self.n_heads = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model), nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(self, h, bias, key_mask):
        B, N, D = h.shape
        qkv = self.qkv(self.norm1(h)).reshape(B, N, 3, self.n_heads, self.dh)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)                         # cada [B,H,N,dh]
        a = _attention(q, k, v, bias, key_mask)                     # [B,H,N,dh]
        a = a.transpose(1, 2).reshape(B, N, D)
        h = h + self.out(a)
        h = h + self.ffn(self.norm2(h))
        return h


class DlogfBias(nn.Module):
    """Bias de atención por-cabeza desde dlogf (posición relativa). Disponible a TODOS los modelos."""

    def __init__(self, n_heads: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, n_heads),
        )

    def forward(self, dlogf: torch.Tensor) -> torch.Tensor:
        # dlogf [B,N,N] → [B,H,N,N]
        b = self.net(dlogf.unsqueeze(-1))                           # [B,N,N,H]
        return b.permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Pieza Pairformer: pair update + triangle
# ---------------------------------------------------------------------------

class PairBiasFromZ(nn.Module):
    """z [B,N,N,c] → bias de atención por-cabeza [B,H,N,N]."""

    def __init__(self, pair_dim: int, n_heads: int):
        super().__init__()
        self.proj = nn.Linear(pair_dim, n_heads)

    def forward(self, z):
        return self.proj(z).permute(0, 3, 1, 2)


class PairUpdateFromTokens(nn.Module):
    """Comunicación token→pair: z[i,j] += g(lin(h_i) ⊙ lin(h_j))."""

    def __init__(self, d_model: int, pair_dim: int):
        super().__init__()
        self.li = nn.Linear(d_model, pair_dim)
        self.lj = nn.Linear(d_model, pair_dim)
        self.g = nn.Linear(pair_dim, pair_dim)
        self.norm = nn.LayerNorm(pair_dim)

    def forward(self, z, h):
        a = self.li(h)                                              # [B,N,pair]
        b = self.lj(h)                                              # [B,N,pair]
        outer = a[:, :, None, :] * b[:, None, :, :]                 # [B,N,N,pair]
        return z + self.g(F.gelu(self.norm(outer)))


class TriangleUpdate(nn.Module):
    """Triangle multiplicative update mask-aware, normalizado, simetrizado (Codex #6).

    z[i,j] ← z[i,j] + scale · (1/|K_ij|) Σ_{k válido, k≠i,j} a(z[i,k]) ⊙ b(z[k,j])
    """

    def __init__(self, pair_dim: int):
        super().__init__()
        self.a = nn.Linear(pair_dim, pair_dim)
        self.b = nn.Linear(pair_dim, pair_dim)
        self.out = nn.Linear(pair_dim, pair_dim)
        self.norm = nn.LayerNorm(pair_dim)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, z, token_mask):
        # z [B,N,N,c]; token_mask [B,N] bool
        A = self.a(z)                                               # [B,N,N,c]  (índices i,k)
        Bm = self.b(z)                                              # [B,N,N,c]  (índices k,j)
        mk = token_mask.float()                                     # [B,N]
        # enmascarar el eje k en ambos
        A = A * mk[:, None, :, None]                                # k es axis=2 de A
        Bm = Bm * mk[:, :, None, None]                              # k es axis=1 de Bm
        full = torch.einsum("bikc,bkjc->bijc", A, Bm)              # incluye k=i, k=j
        # restar contribuciones k=i y k=j
        # k=i: A[b,i,i,c] * Bm[b,i,j,c]
        diagA = torch.diagonal(A, dim1=1, dim2=2).permute(0, 2, 1)  # [B,N,c] = A[b,i,i]
        term_ki = diagA[:, :, None, :] * Bm                        # [B,N,N,c] (Bm[b,i,j] con i en axis1)
        # k=j: A[b,i,j,c] * Bm[b,j,j,c]
        diagB = torch.diagonal(Bm, dim1=1, dim2=2).permute(0, 2, 1)  # [B,N,c] = Bm[b,j,j]
        term_kj = A * diagB[:, None, :, :]                         # [B,N,N,c]
        summ = full - term_ki - term_kj
        # normalizar por |K_ij| = n_valid - 2  (i,j válidos)
        n_valid = mk.sum(dim=1)                                     # [B]
        count = (n_valid - 2.0).clamp(min=1.0)[:, None, None, None]
        tri = self.out(F.gelu(self.norm(summ / count)))
        # El triangle NO modifica la diagonal (self-pair i==j): la transitividad es sobre
        # triples DISTINTOS. En la diagonal, {i,j} colapsa y la resta term_ki/term_kj
        # doble-contaría. Zeroar la diagonal del aporte es lo correcto y testeable.
        Npk = z.shape[1]
        eye = torch.eye(Npk, dtype=torch.bool, device=z.device)
        tri = tri.masked_fill(eye[None, :, :, None], 0.0)
        # NOTA: la simetrización se hace en el forward del Pairformer (para TODOS los
        # variants, incl B-minus), no acá. Codex Alto #1.
        return z + self.scale * tri


class LocalUpdate(nn.Module):
    """Control B-local (Codex Alto #2): MISMA estructura de params que TriangleUpdate
    (a, b, out, norm, scale) pero SIN suma sobre k — operación puramente local por par.
    Aísla la propagación de transitividad (la suma sobre k) con params igualados a B.

    z[i,j] ← z[i,j] + scale · out(gelu(norm( a(z[i,j]) ⊙ b(z[i,j]) )))
    """

    def __init__(self, pair_dim: int):
        super().__init__()
        self.a = nn.Linear(pair_dim, pair_dim)
        self.b = nn.Linear(pair_dim, pair_dim)
        self.out = nn.Linear(pair_dim, pair_dim)
        self.norm = nn.LayerNorm(pair_dim)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, z, token_mask):
        local = self.a(z) * self.b(z)                               # [B,N,N,c] sin mezclar k
        loc = self.out(F.gelu(self.norm(local)))
        # MISMA política de diagonal que TriangleUpdate (Codex r2 #1): no modificar self-pair,
        # para que B vs B-local aísle SOLO la suma sobre k, no el tratamiento del diagonal.
        Npk = z.shape[1]
        eye = torch.eye(Npk, dtype=torch.bool, device=z.device)
        loc = loc.masked_fill(eye[None, :, :, None], 0.0)
        return z + self.scale * loc


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

class PairwiseReadout(nn.Module):
    """[h_i, h_j, h_i⊙h_j] (+ pair_feat opcional) → logit. Simetrizado promediando (i,j)/(j,i)."""

    def __init__(self, d_model: int, extra_pair_dim: int = 0, hidden: int = 64):
        super().__init__()
        in_dim = 3 * d_model + extra_pair_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )

    def forward(self, h, pair_feat: Optional[torch.Tensor] = None):
        B, N, D = h.shape
        hi = h[:, :, None, :].expand(B, N, N, D)
        hj = h[:, None, :, :].expand(B, N, N, D)
        feats = [hi, hj, hi * hj]
        if pair_feat is not None:
            feats.append(pair_feat)
        x = torch.cat(feats, dim=-1)
        logit = self.net(x).squeeze(-1)                            # [B,N,N]
        return 0.5 * (logit + logit.transpose(1, 2))


class TokenModel(nn.Module):
    """A-naive / A-rich. mechanism='naive' o 'rich'."""

    def __init__(self, mechanism: str, d_model: int = 96, n_heads: int = 4,
                 n_layers: int = 4, pf_dim: int = 32):
        super().__init__()
        assert mechanism in ("naive", "rich")
        self.mechanism = mechanism
        self.embed = nn.Linear(2, d_model)
        self.dlogf_bias = DlogfBias(n_heads)
        self.layers = nn.ModuleList(
            [TokenAttnLayer(d_model, n_heads) for _ in range(n_layers)]
        )
        if mechanism == "rich":
            self.pair_embed = PairFeatureEmbed(pf_dim)
            self.readout = PairwiseReadout(d_model, extra_pair_dim=pf_dim)
        else:
            self.readout = PairwiseReadout(d_model, extra_pair_dim=0)

    def forward(self, batch):
        h = self.embed(batch["tokens"])
        dlogf = batch["pair_cont"][..., 0]                         # [B,N,N]
        bias = self.dlogf_bias(dlogf)
        km = batch["token_mask"]
        for layer in self.layers:
            h = layer(h, bias, km)
        if self.mechanism == "rich":
            pf = self.pair_embed(batch["pair_cont"], batch["ratio_class_id"])
            return self.readout(h, pf)
        return self.readout(h)


class Pairformer(nn.Module):
    """B (pair_mix='triangle') / B-minus ('none') / B-local ('local') / B-shuffle (triangle+shuffle).

    pair_mix ∈ {'triangle','none','local'}; shuffle_pair_init = control negativo (determinístico).
    """

    def __init__(self, d_model: int = 88, n_heads: int = 4, n_layers: int = 4,
                 pair_dim: int = 56, pair_mix: str = "triangle",
                 shuffle_pair_init: bool = False):
        super().__init__()
        assert pair_mix in ("triangle", "none", "local")
        self.pair_mix = pair_mix
        self.shuffle_pair_init = shuffle_pair_init
        self.embed = nn.Linear(2, d_model)
        self.pair_embed = PairFeatureEmbed(pair_dim)
        self.dlogf_bias = DlogfBias(n_heads)
        self.pair_bias = nn.ModuleList([PairBiasFromZ(pair_dim, n_heads) for _ in range(n_layers)])
        self.token_layers = nn.ModuleList([TokenAttnLayer(d_model, n_heads) for _ in range(n_layers)])
        self.pair_updates = nn.ModuleList([PairUpdateFromTokens(d_model, pair_dim) for _ in range(n_layers)])
        if pair_mix == "triangle":
            self.mixers = nn.ModuleList([TriangleUpdate(pair_dim) for _ in range(n_layers)])
        elif pair_mix == "local":
            self.mixers = nn.ModuleList([LocalUpdate(pair_dim) for _ in range(n_layers)])
        else:
            self.mixers = None
        self.pair_norm = nn.LayerNorm(pair_dim)
        self.readout = nn.Sequential(
            nn.Linear(pair_dim, 64), nn.GELU(), nn.Linear(64, 1),
        )

    @staticmethod
    def _shuffle_pair_init(z, token_mask, mixture_id):
        """Permuta z dentro de los pares válidos (i<j), DETERMINÍSTICO por mezcla (seed=mixture_id),
        preservando máscara y re-simetrizando. Estable entre forwards/epochs (Codex Medio)."""
        B, N, _, c = z.shape
        out = z.clone()
        for b in range(B):
            valid = token_mask[b].nonzero(as_tuple=True)[0]
            nv = valid.numel()
            if nv <= 1:
                continue
            # pares i<j entre válidos, MISMO orden que la list-comp original (row-major triu)
            ii, jj = torch.triu_indices(nv, nv, offset=1, device=valid.device)
            iu = valid[ii]
            ju = valid[jj]
            vals = z[b, iu, ju]                                      # [P,c] — un gather vectorizado
            g = torch.Generator(device="cpu").manual_seed(int(mixture_id[b]))
            perm = torch.randperm(iu.numel(), generator=g).to(z.device)
            permuted = vals[perm]
            out[b, iu, ju] = permuted
            out[b, ju, iu] = permuted                               # re-simetrizar
        return out

    def forward(self, batch):
        h = self.embed(batch["tokens"])
        km = batch["token_mask"]
        pair_token_mask = km[:, :, None] & km[:, None, :]           # [B,N,N]
        z = self.pair_embed(batch["pair_cont"], batch["ratio_class_id"])
        if self.shuffle_pair_init:
            z = self._shuffle_pair_init(z, km, batch["mixture_id"])
        z = z * pair_token_mask[..., None]
        dlogf = batch["pair_cont"][..., 0]
        dlogf_b = self.dlogf_bias(dlogf)

        for li in range(len(self.token_layers)):
            bias = dlogf_b + self.pair_bias[li](z)
            h = self.token_layers[li](h, bias, km)
            z = self.pair_updates[li](z, h)
            if self.mixers is not None:
                z = self.mixers[li](z, km)
            # Simetrización para TODOS los variants tras el pair update (Codex Alto #1):
            z = 0.5 * (z + z.transpose(1, 2))
            z = z * pair_token_mask[..., None]

        logit = self.readout(self.pair_norm(z)).squeeze(-1)        # [B,N,N]
        return 0.5 * (logit + logit.transpose(1, 2))               # simétrico


MODEL_NAMES = ("A-naive", "A-rich", "B", "B-minus", "B-local", "B-shuffle")

# Config de arquitectura CONGELADO por modelo (param-match A-rich vs B ~1.1%, Codex r2 #3).
# El training usa build_model(name) que toma esto — NO pasar un d_model global.
MODEL_CONFIGS = {
    "A-naive":   {"d_model": 96, "n_heads": 4, "n_layers": 4, "pf_dim": 32},
    "A-rich":    {"d_model": 96, "n_heads": 4, "n_layers": 4, "pf_dim": 32},
    "B":         {"d_model": 88, "n_heads": 4, "n_layers": 4, "pair_dim": 56},
    "B-minus":   {"d_model": 88, "n_heads": 4, "n_layers": 4, "pair_dim": 56},
    "B-local":   {"d_model": 88, "n_heads": 4, "n_layers": 4, "pair_dim": 56},
    "B-shuffle": {"d_model": 88, "n_heads": 4, "n_layers": 4, "pair_dim": 56},
}


def build_model(name: str, **overrides) -> nn.Module:
    """Construye un modelo con su config CONGELADO (MODEL_CONFIGS). overrides solo para tests."""
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {name}")
    cfg = dict(MODEL_CONFIGS[name])
    cfg.update(overrides)
    if name == "A-naive":
        return TokenModel("naive", **cfg)
    if name == "A-rich":
        return TokenModel("rich", **cfg)
    if name == "B":
        return Pairformer(pair_mix="triangle", shuffle_pair_init=False, **cfg)
    if name == "B-minus":
        return Pairformer(pair_mix="none", shuffle_pair_init=False, **cfg)
    if name == "B-local":
        return Pairformer(pair_mix="local", shuffle_pair_init=False, **cfg)
    if name == "B-shuffle":
        return Pairformer(pair_mix="triangle", shuffle_pair_init=True, **cfg)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
