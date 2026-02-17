# Ranking Unificado de Descriptores — Phideus Bias Control

> Documento vivo. Se actualiza con cada nuevo screening.
> Última actualización: 2026-02-17 (Gate 4.4 en curso)

---

## Screening @ 5 epochs (foundation + freeze-policy run-d)

Protocolo estándar: foundation_locked_e25.pt, freeze-policy run-d, batch-size 16, seed 42.
Métrica principal: **S = min(A2M_R@10, M2A_R@10)** sobre structured pool (13,532 segmentos, 500 piezas).

| # | Brazo | Familia | Mecanismo | Best S | Best Ep | A2M | M2A | hard_neg | vs D0 | Gate |
|---|-------|---------|-----------|--------|---------|-----|-----|----------|-------|------|
| 1 | **d4a4** | Dual (MIDI+Audio) | concat | **69.8%** | 5 | 69.8% | 70.6% | 91.6% | **+9.6pp** | 4.3 |
| 2 | **a4r** | Audio (log-freq) | reverse cross-att | **68.6%** | 5 | 68.6% | 69.0% | 91.6% | **+8.4pp** | 4.3-F5 |
| 3 | **t3-tri** | Third Tower | trilinear bridge | **65.0%** | 5 | 65.4% | 65.0% | 90.6% | +4.8pp | 4.4 |
| 4 | d4r | MIDI (intervals) | reverse cross-att | 64.2% | 5 | 64.2% | 64.4% | 93.2% | +4.0pp | 4.3-F5 |
| 5 | D4 | MIDI (intervals) | concat | 63.6% | 5 | 63.6% | 64.4% | 91.2% | +3.4pp | 4.3 |
| 5 | A4 | Audio (log-freq) | concat | 63.6% | 5 | 65.8% | 63.6% | 92.4% | +3.4pp | 4.3 |
| 7 | A4x | Audio (log-freq) | cross-att | 62.6% | 5 | 64.0% | 62.6% | 92.4% | +2.4pp | 4.3 |
| 8 | A7x | Audio (attractor) | cross-att | 62.2% | 5 | 62.2% | 63.8% | 92.0% | +2.0pp | 4.3 |
| 9 | **D0** | — | **baseline** | **60.2%** | 3 | 60.4% | 60.2% | 90.0% | — | 4.3 |
| 10 | D4x | MIDI (intervals) | cross-att | 60.0% | 4 | 60.0% | 60.4% | 91.4% | -0.2pp | 4.3 |
| 11 | a9 | Audio (IDF-attractor) | concat | 58.8% | 5 | 58.8% | 60.8% | 90.4% | -1.4pp | 4.3-F5 |
| 11 | A7 | Audio (attractor) | concat | 58.8% | 5 | 60.2% | 58.8% | 90.2% | -1.4pp | 4.3 |
| 13 | **moe-a4** | MoE | expert routing | **58.2%** | 5 | 61.8% | 58.2% | 91.4% | -2.0pp | 4.4 |
| 14 | a8 | Audio (onset-chroma) | concat | 57.4% | 5 | 60.4% | 57.4% | 90.6% | -2.8pp | 4.3-F5 |
| 15 | d4a4cm | Dual (cross-modal) | concat | 52.4% | 5 | 52.4% | 56.6% | 89.6% | -7.8pp | 4.3 |

### Gate 4.4 — Resultados parciales (pendiente e5)

| Brazo | Familia | Mecanismo | S@e3 | S@e5 | Gate |
|-------|---------|-----------|------|------|------|
| film-a4 | FiLM | feature modulation (audio) | 59.2% | pendiente | 4.4 |
| t3-wt | Third Tower | weighted bridge | 47.6% | pendiente | 4.4 |
| t3-anc | Third Tower | anchor bridge | 40.2% | pendiente | 4.4 |
| film-d4 | FiLM | feature modulation (MIDI) | pendiente | pendiente | 4.4 |
| film-dual | FiLM | feature modulation (dual) | pendiente | pendiente | 4.4 |
| moe-dual | MoE | expert routing (dual) | pendiente | pendiente | 4.4 |

---

## Runs largos (30 epochs, scratch o foundation)

| Descriptor | Protocolo | Best S | Best Ep | A2M | M2A | hard_neg | Tiempo total |
|-----------|-----------|--------|---------|-----|-----|----------|-------------|
| **d4a4** | scratch, run-d, seed 42 | **83.6%** | 30 | 83.6% | 84.2% | 95.2% | ~15.5h |
| d4a4 | scratch, multi-seed (5) | **84.1% ±2.3pp** | 30 | — | — | — | ~78h total |
| **a4r** | scratch, run-d, seed 42 | **82.0%** | 29 | 82.6% | 82.0% | 94.4% | 12.3h |
| **d4a4r** | scratch, run-d, seed 42 | **74.4%** | 30 | 74.4% | 74.8% | 92.0% | 12.4h |
| **d4-a4r** | foundation, run-d, seed 42 | **en curso** | — | — | — | — | ~16h est. |

### Curvas epoch-by-epoch (runs scratch 30ep)

#### d4a4 (benchmark)
| Epoch | S | A2M | M2A | hard_neg |
|-------|---|-----|-----|----------|
| 10 | 74.6% | — | — | 93.0% |
| 15 | 65.8% | — | — | 91.0% |
| 20 | 75.6% | — | — | 93.6% |
| 25 | 82.2% | — | — | 95.4% |
| 28 | 82.8% | — | — | 94.8% |
| 29 | 82.6% | — | — | 95.2% |
| 30 | **83.6%** | 83.6% | 84.2% | 95.2% |

#### a4r
| Epoch | S | A2M | M2A | hard_neg |
|-------|---|-----|-----|----------|
| 5 | 61.8% | 61.8% | 62.2% | 91.0% |
| 10 | 69.0% | 71.8% | 69.0% | 90.2% |
| 15 | 77.2% | 77.2% | 77.2% | 94.0% |
| 20 | 77.6% | 77.6% | 77.8% | 94.8% |
| 25 | 80.4% | 81.4% | 80.4% | 94.6% |
| 28 | 81.8% | 83.2% | 81.8% | 94.4% |
| 29 | **82.0%** | 82.6% | 82.0% | 94.4% |
| 30 | 80.2% | 82.2% | 80.2% | 94.6% |

#### d4a4r
| Epoch | S | A2M | M2A | hard_neg |
|-------|---|-----|-----|----------|
| 5 | 43.8% | 43.8% | 47.8% | 84.0% |
| 10 | 56.0% | 56.0% | 61.6% | 90.2% |
| 15 | 67.8% | 68.8% | 67.8% | 90.8% |
| 20 | 71.4% | 72.4% | 71.4% | 92.6% |
| 25 | 74.2% | 76.4% | 74.2% | 92.0% |
| 28 | 74.2% | 76.0% | 74.2% | 93.0% |
| 29 | 73.6% | 75.8% | 73.6% | 92.4% |
| 30 | **74.4%** | 74.4% | 74.8% | 92.0% |

---

## Observaciones empíricas

Patrones observados en los datos. No constituyen juicio GO/NO-GO — las decisiones las toma el equipo.

1. **Concat > Cross-attention** para descriptores fuertes (D4, A4)
2. **Reverse cross-att > Standard cross-att**: a4r=68.6% vs A4x=62.6% (+6.0pp)
3. **Same-modality > Cross-modal**: d4a4=69.8% vs d4a4cm=52.4% (+17.4pp)
4. **Efecto superaditivo en d4a4**: D4(+3.4) + A4(+3.4) = d4a4(+9.6), no 6.8
5. **Log-freq > Attractor**: A4 supera A7 en todos los mecanismos
6. **d4a4r (dual reverse) en 30ep**: 74.4% vs d4a4=83.6%
7. **moe-a4**: aux loss cayó de 0.075 a 0.001 entre e3 y e5

---

## Referencia histórica

- **Gate 4.0/4.1**: Audio encoder 100% frozen. Descriptores ratio sin efecto.
- **Gate 4.2**: Primer resultado positivo con D4 al descongelar audio encoder layers 2-3.

---

## Glosario de mecanismos

| Mecanismo | Código | Descripción |
|-----------|--------|-------------|
| concat | d0-d4, a4, a7 | Descriptor concatenado a features antes de proyección |
| cross-att | A4x, A7x, D4x | Q=features, K/V=descriptor (standard) |
| reverse cross-att | a4r, d4r | Q=descriptor, K/V=features (invertido) |
| dual concat | d4a4 | D4 en MIDI + A4 en audio, ambos concat |
| dual reverse | d4a4r | A4r en audio + D4r en MIDI, ambos reverse |
| dual mixed | d4-a4r | D4 concat en MIDI + A4r reverse en audio |
| cross-modal | d4a4cm | D4→audio, A4→MIDI (cruzado) |
| trilinear bridge | t3-tri | Third tower con producto trilineal audio×midi×ratio |
| anchor bridge | t3-anc | Third tower con anchor points ratio→(audio,midi) |
| weighted bridge | t3-wt | Third tower con weighted sum ratio-conditioned |
| FiLM | film-* | Feature-wise Linear Modulation (γ,β from descriptor) |
| MoE | moe-* | Mixture of Experts con routing condicionado por descriptor |
