# Ranking Unificado de Descriptores — Phideus Bias Control

> Documento vivo. Se actualiza con cada nuevo screening.
> Última actualización: 2026-02-19 00:30 UTC-3 (TODOS los runs 30ep COMPLETOS, results_unc/ sincronizado al 100%)

---

## Screening @ 5 epochs (foundation + freeze-policy run-d)

Protocolo estándar: foundation_locked_e25.pt, freeze-policy run-d, batch-size 16, seed 42.
Métrica principal: **S = min(A2M_R@10, M2A_R@10)** sobre structured pool (13,532 segmentos, 500 piezas).

| # | Brazo | Familia | Mecanismo | Best S | Best Ep | A2M | M2A | hard_neg | vs D0 | Gate |
|---|-------|---------|-----------|--------|---------|-----|-----|----------|-------|------|
| 1 | **d4a4** | Dual (MIDI+Audio) | concat | **69.8%** | 5 | 69.8% | 70.6% | 91.6% | **+9.6pp** | 4.3 |
| 2 | **a4r** | Audio (log-freq) | reverse cross-att | **68.6%** | 5 | 68.6% | 69.0% | 91.6% | **+8.4pp** | 4.3-F5 |
| 3 | **t3-wt** | Third Tower | weighted bridge | **67.6%** | 5 | 71.4% | 67.6% | 91.2% | **+7.4pp** | 4.4 |
| 4 | **t3-tri** | Third Tower | trilinear bridge | **65.0%** | 5 | 65.4% | 65.0% | 90.6% | +4.8pp | 4.4 |
| 5 | d4r | MIDI (intervals) | reverse cross-att | 64.2% | 5 | 64.2% | 64.4% | 93.2% | +4.0pp | 4.3-F5 |
| 6 | D4 | MIDI (intervals) | concat | 63.6% | 5 | 63.6% | 64.4% | 91.2% | +3.4pp | 4.3 |
| 6 | A4 | Audio (log-freq) | concat | 63.6% | 5 | 65.8% | 63.6% | 92.4% | +3.4pp | 4.3 |
| 8 | A4x | Audio (log-freq) | cross-att | 62.6% | 5 | 64.0% | 62.6% | 92.4% | +2.4pp | 4.3 |
| 9 | A7x | Audio (attractor) | cross-att | 62.2% | 5 | 62.2% | 63.8% | 92.0% | +2.0pp | 4.3 |
| 10 | **D0** | — | **baseline** | **60.2%** | 3 | 60.4% | 60.2% | 90.0% | — | 4.3 |
| 11 | moe-a4-v2 | MoE v2 | non-zero init + noise decay | 60.2% | 5 | 60.4% | 60.2% | 90.8% | 0.0pp | 4.4-MoE |
| 12 | D4x | MIDI (intervals) | cross-att | 60.0% | 4 | 60.0% | 60.4% | 91.4% | -0.2pp | 4.3 |
| 13 | moe-a4-v4 | MoE v4 | top-1 hard gating | 59.4% | 5 | 60.6% | 59.4% | 91.2% | -0.8pp | 4.4-MoE |
| 13 | film-dual | FiLM | modulation (dual) | 59.4% | 5 | 60.2% | 59.4% | 91.4% | -0.8pp | 4.4 |
| 15 | film-a4 | FiLM | modulation (audio) | 59.2% | 3 | 60.8% | 59.2% | 89.8% | -1.0pp | 4.4 |
| 15 | moe-dual | MoE | expert routing (dual) | 59.2% | 5 | 61.2% | 59.2% | 91.6% | -1.0pp | 4.4 |
| 15 | moe-a4-v3 | MoE v3 | entropy penalty | 59.2% | 5 | 60.6% | 59.2% | 91.2% | -1.0pp | 4.4-MoE |
| 18 | a9 | Audio (IDF-attractor) | concat | 58.8% | 5 | 58.8% | 60.8% | 90.4% | -1.4pp | 4.3-F5 |
| 18 | A7 | Audio (attractor) | concat | 58.8% | 5 | 60.2% | 58.8% | 90.2% | -1.4pp | 4.3 |
| 20 | film-d4 | FiLM | modulation (MIDI) | 58.6% | 5 | 61.0% | 58.6% | 91.8% | -1.6pp | 4.4 |
| 21 | moe-a4 | MoE | expert routing | 58.2% | 3 | 58.8% | 60.2% | 89.6% | -2.0pp | 4.4 |
| 22 | a8 | Audio (onset-chroma) | concat | 57.4% | 5 | 60.4% | 57.4% | 90.6% | -2.8pp | 4.3-F5 |
| 23 | d4a4cm | Dual (cross-modal) | concat | 52.4% | 5 | 52.4% | 56.6% | 89.6% | -7.8pp | 4.3 |
| 24 | t3-anc | Third Tower | anchor bridge | 42.2% | 5 | 42.2% | 42.2% | 89.4% | -18.0pp | 4.4 |

**24 brazos finalizados** (21 originales + 3 MoE v2/v3/v4).

### MoE v2/v3/v4 — Resultado final

Variantes diseñadas para resolver la inercia simétrica de moe-a4.
Diagnóstico original: zero-init + lb_weight débil (0.01) → routing uniforme → expertos idénticos → MoE inerte.

| Brazo | Mecanismo | S@e3 | S@e5 | aux@e5 | Resultado |
|-------|-----------|------|------|--------|-----------|
| moe-a4-v2 | Non-zero init + router noise decay | 58.6% | **60.2%** | 0.001 | Mejor MoE, empata D0 |
| moe-a4-v3 | v2 + entropy penalty | 59.8% | 59.2% | 0.157 | Bajó de e3→e5, aux activo pero insuficiente |
| moe-a4-v4 | v2 + top-1 hard gating | 59.2% | 59.4% | 0.001 | Hard gating no rompió simetría |

Conclusión: ninguno supera D0. Familia MoE no competitiva en screening 5ep.

---

## Runs largos (30 epochs, scratch)

| Descriptor | Protocolo | Best S | Best Ep | A2M | M2A | hard_neg | Tiempo total |
|-----------|-----------|--------|---------|-----|-----|----------|-------------|
| **d4a4** | scratch, run-d, seed 42 | **83.6%** | 30 | 83.6% | 84.2% | 95.2% | ~15.5h |
| d4a4 | scratch, multi-seed (5) | **84.1% ±2.3pp** | 30 | — | — | — | ~78h total |
| **a4r** | scratch, run-d, seed 42 | **82.0%** | 29 | 82.6% | 82.0% | 94.4% | 12.3h |
| **d4-a4r** | scratch, run-d, seed 42 | **79.8%** | 30 | 81.4% | 79.8% | 94.2% | 12.1h |
| **d4-a4r** | scratch, run-d, seed 42 | **79.8%** | 30 | 81.4% | 79.8% | 94.2% | 12.1h |
| **t3-wt** | scratch, run-d, seed 42 | **79.8%** | 30 | 82.4% | 79.8% | 94.8% | 24.8h |
| **d4a4r** | scratch, run-d, seed 42 | **74.4%** | 30 | 74.4% | 74.8% | 92.0% | 12.4h |
| **moe-dual** | scratch, run-d, seed 42 | **72.6%** | 30 | 72.8% | 72.6% | 93.4% | 26.8h |

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

#### d4-a4r
| Epoch | S | A2M | M2A | hard_neg |
|-------|---|-----|-----|----------|
| 5 | 62.2% | 62.2% | 62.6% | 90.8% |
| 10 | 58.8% | 58.8% | 60.6% | 89.0% |
| 15 | 72.2% | 72.2% | 72.2% | 91.0% |
| 20 | 77.6% | 77.6% | 77.6% | 94.2% |
| 25 | 79.2% | 80.4% | 79.2% | 94.2% |
| 30 | **79.8%** | 81.4% | 79.8% | 94.2% |

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

#### t3-wt
| Epoch | S | A2M | M2A | hard_neg |
|-------|---|-----|-----|----------|
| 5 | 40.0% | 40.0% | 46.6% | 86.2% |
| 10 | 57.6% | 57.6% | 58.0% | 92.0% |
| 15 | 66.2% | 66.2% | 68.2% | 92.2% |
| 20 | 77.6% | 79.2% | 77.6% | 92.6% |
| 25 | 79.4% | 81.0% | 79.4% | 93.8% |
| 30 | **79.8%** | 82.4% | 79.8% | 94.8% |

#### moe-dual
| Epoch | S | A2M | M2A | hard_neg |
|-------|---|-----|-----|----------|
| 5 | 42.4% | 42.4% | 49.6% | 87.2% |
| 10 | 63.6% | 63.6% | 65.0% | 91.0% |
| 15 | 67.8% | 68.8% | 67.8% | 93.4% |
| 20 | 69.8% | 71.2% | 69.8% | 92.8% |
| 25 | 71.2% | 71.2% | 71.4% | 92.8% |
| 30 | **72.6%** | 72.8% | 72.6% | 93.4% |

### Comparativa lado a lado (S por epoch)

| Epoch | d4a4 | a4r | d4-a4r | t3-wt | d4a4r | moe-dual |
|-------|------|-----|--------|-------|-------|----------|
| 5 | — | 61.8% | 62.2% | 40.0% | 43.8% | 42.4% |
| 10 | 74.6% | 69.0% | 58.8% | 57.6% | 56.0% | 63.6% |
| 15 | 65.8% | 77.2% | 72.2% | 66.2% | 67.8% | 67.8% |
| 20 | 75.6% | 77.6% | 77.6% | 77.6% | 71.4% | 69.8% |
| 25 | 82.2% | 80.4% | 79.2% | 79.4% | 74.2% | 71.2% |
| 30 | **83.6%** | 80.2% | **79.8%** | **79.8%** | **74.4%** | **72.6%** |

---

## Observaciones empíricas

Patrones observados en los datos. No constituyen juicio GO/NO-GO — las decisiones las toma el equipo.

1. **Concat > Cross-attention** para descriptores fuertes (D4, A4)
2. **Reverse cross-att > Standard cross-att**: a4r=68.6% vs A4x=62.6% (+6.0pp)
3. **Same-modality > Cross-modal**: d4a4=69.8% vs d4a4cm=52.4% (+17.4pp)
4. **Efecto superaditivo en d4a4**: D4(+3.4) + A4(+3.4) = d4a4(+9.6), no 6.8
5. **Log-freq > Attractor**: A4 supera A7 en todos los mecanismos
6. **d4a4 late bloomer a 30ep**: dip en e15 (65.8%) pero sube fuerte e20→e30 (+8pp). Único que mejora hasta e30 sin regresión
7. **a4r converge rápido**: lidera e10-e20 pero techo en e29 (82.0%) con regresión a 80.2% en e30
8. **d4-a4r intermedio**: empata a4r en e5 y e20, pero se estanca ~79-80% — no tiene la subida tardía de d4a4
9. **d4a4r (dual reverse) no competitivo**: -9.2pp vs d4a4 a 30ep. Reverse en ambas modalidades perjudica
10. **FiLM y MoE (Gate 4.4)**: todos en franja 58-60%, en/por debajo de D0=60.2%
11. **moe-a4 inercia simétrica**: lb→0 = routing uniforme (no colapso a 1 experto). Zero-init + lb_weight=0.01 insuficiente → expertos nunca se especializan → MoE inerte. Diagnóstico confirmado por Codex
12. **MoE v2/v3/v4**: ninguno supera D0. v2 empata (60.2%). Familia MoE agotada
13. **Third Tower**: t3-wt (#3, 67.6%) y t3-tri (#4, 65.0%) son los mejores brazos de Gate 4.4
14. **t3-wt 30ep**: S@e5=40.0% → S@e30=79.8%. Empata d4-a4r en 3er lugar. Crecimiento sostenido sin regresión
15. **moe-dual 30ep**: S@e30=72.6%. Plateau desde e20 (+2.8pp en 10 epochs). 6to de 6 runs largos
16. **t3-wt = d4-a4r a 30ep**: ambos 79.8%, pero t3-wt arranca mucho peor (40% vs 62% a e5) y recupera. Curvas muy diferentes, mismo destino

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
| MoE v2 | moe-*-v2 | MoE con non-zero init + router noise decay |
| MoE v3 | moe-*-v3 | v2 + entropy penalty (castiga routing uniforme) |
| MoE v4 | moe-*-v4 | v2 + top-1 hard gating (Switch Transformer) |
