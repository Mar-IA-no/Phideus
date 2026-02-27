<div align="center">

# Proyecto Estado Actual
### Phideus v5.0

![Program](https://img.shields.io/badge/Program-Research_Active-0A7E3B?style=for-the-badge)
![Current Focus](https://img.shields.io/badge/Focus-Escalon_1--C-1F6FEB?style=for-the-badge)
![Bias Control](https://img.shields.io/badge/BIAS_CONTROL-Gate_5B_EN_CURSO-F59E0B?style=for-the-badge)

</div>

> [!IMPORTANT]
> **Actualizado**: 2026-02-27
> **Estado**: Gate 5B activo con paquete local consolidado (`Test12/01/04/03/06/08/09/10` cerrados). Test 11 perceptual (decoder suite) cerrado; A/B pre-projection activo con avance en `D0` (`preproj_midi2events` cerrado, `preproj_audio2events` en entrenamiento). **Test 13G (generative encoder training) implementado** — primer test que re-entrena encoders con dual-objective (VICReg + reconstrucción PR).
> **Decisión operativa vigente**: (1) completar A/B pre-projection, (2) ejecutar Test 13G Phase A (D0 λ sweep), (3) cerrar bloque UNC de robustez (Test05 en `9/15` sync local + bloque `D0` corriendo; Test02 en cola `4/4`).
> **Encuadre estrategico**: Gate 5A deja de ser barrido bloqueante. Conditioned projections queda implementado como linea oportunista; Escalon 2 pasa a foco principal apenas cierre Gate 5B, con Gate 5A corriendo en paralelo cuando haya recursos libres.
> **Infraestructura**: estrategia distribuida LOCAL+UNC activa; foundation lock publicado (`v0.1.0-foundation`).

## Navegación rápida

- [Resumen Ejecutivo](#resumen-ejecutivo)
- [Estado por Gate](#estado-por-gate)
- [Hallazgos Causales del Corte](#hallazgos-causales-del-corte)
- [Plan Operativo Vigente](#plan-operativo-vigente)
- [Frentes y Documentos](#frentes-y-documentos)

---

## Resumen Ejecutivo

Gate 4.3 dejó una base fuerte (`d4a4=69.8%` a 5ep; `d4a4=83.6%` a 30ep), y Gate 4.4 completó el filtro arquitectural con evidencia comparable en toda la grilla corta. El bloque largo confirmó techo competitivo en variantes ratio-céntricas y habilitó selección robusta de checkpoints para validación científica.

Ese bloque de dinámica temporal queda encapsulado como **Gate 4.5 (LR Schedule Optimization)** y opera ahora como soporte de **Gate 5B**: checkpoints consolidados para evaluación científica (`D0`, `d4`, `d4a4`, `a4r`, `d4-a4r`) con loader universal y protocolo canónico fijo.

### Baseline oficial de comparación (histórico)

`Gate 2 - checkpoint_epoch45`

| Métrica | Valor |
|--------|-------|
| A2M R@10 (pool 256/500/seed42) | 34.4% |
| M2A R@10 (pool 256/500/seed42) | 37.6% |
| Hard negative accuracy | 80.4% |
| S=min(A2M,M2A) | 34.4% |

### Screening @5ep (ranking unificado, top del frente)

| Rank | Brazo | Gate/Familia | Best S | A2M | M2A | hard_neg | vs D0 |
|------|-------|--------------|--------|-----|-----|----------|-------|
| 1 | d4a4 | 4.3 Dual concat | 69.8% | 69.8% | 70.6% | 91.6% | +9.6pp |
| 2 | a4r | 4.3-F5 reverse | 68.6% | 68.6% | 69.0% | 91.6% | +8.4pp |
| 3 | t3-wt | 4.4 Third Tower | 67.6% | 71.4% | 67.6% | 91.2% | +7.4pp |
| 4 | t3-tri | 4.4 Third Tower | 65.0% | 65.4% | 65.0% | 90.6% | +4.8pp |
| 10 | D0 | 4.3 baseline | 60.2% | 60.4% | 60.2% | 90.0% | — |
| 11 | moe-a4-v2 | 4.4-MoE | 60.2% | 60.4% | 60.2% | 90.8% | 0.0pp |

Notas de cierre 4.4:
- `film-dual` cerró en `59.4%` (e5), `moe-dual` en `59.2%` (e5).
- `moe-a4-v2/v3/v4` no superan D0 (v2 empata D0).

### Runs largos (30ep, scratch) — todos cerrados

| Descriptor | Best S | Best Ep | A2M | M2A | hard_neg |
|-----------|--------|---------|-----|-----|----------|
| d4a4 | 83.6% | 30 | 83.6% | 84.2% | 95.2% |
| a4r | 82.0% | 29 | 82.6% | 82.0% | 94.4% |
| d4-a4r | 79.8% | 30 | 81.4% | 79.8% | 94.2% |
| t3-wt | 79.8% | 30 | 82.4% | 79.8% | 94.8% |
| d4a4r | 74.4% | 30 | 74.4% | 74.8% | 92.0% |
| moe-dual | 72.6% | 30 | 72.8% | 72.6% | 93.4% |

Multi-seed e30 (5 seeds): `d4a4 = 84.1% +/- 2.3pp`.

### Gate 4.5 + Gate 5B (corte operativo 2026-02-27)

| Bloque | Corridas | Estado |
|--------|----------|--------|
| Batch 60ep (cosine estándar) | `a4r` | **completado** (`S=79.4%` en e60) |
| Batch 60ep (cosine estándar) | `D0`, `d4a4` | **completados** (`D0=72.8%`, `d4a4=83.8%`) |
| Batch 60ep (cosine estándar) | `d4-a4r` | **completado** (`S=79.8%` en e55) |
| Batch 60ep (cosine estándar) | `moe-dual` | **dead por time limit** (`best S=73.0%` en e30, no sostenido) |
| Hold scheduler 50ep | `t3-wt` (`--lr-hold-fraction=0.5`) | **completado** (`S=81.2%` en e50) |
| Batch 60ep (cosine-tail) | `a4r` | **completado** (`S=80.6%` en e60) |
| Gate 5B Test12 (scoreboard) | `D0`, `d4a4`, `a4r`, `d4-a4r` | **cerrado** (`73.4%`, `83.8%`, `82.0%`, `79.8%`) |
| Gate 5B Test01 (causal ablation) | `D0`, `d4`, `d4a4`, `a4r`, `d4-a4r` | **cerrado** (A4/A4r causal fuerte; D4 marginal en duales) |
| Gate 5B Test04 (transposition) | `D0`, `d4a4`, `a4r`, `d4-a4r` | **cerrado** (robustez relativa mayor en modelos con A4/A4r) |
| Gate 5B Test03 (ratio probe) | `D0`, `d4a4`, `a4r`, `d4-a4r` | **cerrado** (sin “smoking gun” lineal; mejora vive en geometría de retrieval) |
| Gate 5B Test06 (RSA/CKA) | `D0`, `d4a4`, `a4r`, `d4-a4r` | **cerrado** (alineación cross-encoder aumenta fuerte con A4/A4r) |
| Gate 5B Test08 (ratio decoding) | `d4a4`, `a4r`, `d4-a4r` | **cerrado** (bandas 750+ Hz dominan sensibilidad) |
| Gate 5B Test10 (visualizaciones) | `D0`, `d4a4`, `a4r`, `d4-a4r` | **cerrado** (paquete visual v2: 24 PNG + 6 GIF) |
| Gate 5B Test09 (invariance suite) | `D0`, `d4a4`, `a4r`, `d4-a4r` | **cerrado** (temporal robusto; alta fragilidad a velocity/octava; robustez a ruido con patrón bimodal) |
| Gate 5B Test05 (multi-seed, UNC) | `a4r` seeds `42/123/456/789/1337` + `d4-a4r` seeds `42/123/456/789` | **parcial cerrado** (`9/15`) |
| Gate 5B Test05 (multi-seed, UNC) | bloque `D0` (`42/123/456/789/1337`) | **en ejecución** (reporte UNC: `4` running, `1` pending) |
| Gate 5B Test02 (parameter-matched, UNC) | `real`, `random`, `shuffled`, `zero` | **pending** (`4/4`, job `1143844`, `nice=1000`) |

---

## Estado por Gate

| Gate / Etapa | Estado | Resultado |
|--------------|--------|-----------|
| Gate 0 | Completado | GO |
| Gate 1 | Completado | GO (sanity intra-modal) |
| Gate 2 | Completado | GO (baseline canónico) |
| Gate 2.5 | Completado | Diagnóstico de separabilidad |
| Gate 3 (DANN) | Cerrado | NO-GO |
| Gate 4.1 | Cerrado | NO-GO (`R1-rescue` insuficiente) |
| Gate 6 (diagnóstico) | Completado | Causa raíz confirmada |
| Bloque A v1.1 | Cerrado | `D-02 e25` como foundation lock |
| Gate 4.2 ratio-céntrico | Cerrado | `D4 8ep` (`S=64.2%`) |
| Gate 4.3 ratio re-céntrico | Cerrado | 13 brazos + scratch; record 30ep `S=83.6%` |
| Gate 4.4 arquitecturas mayores | **Cerrado** | Screening 24 brazos + 30ep (`t3-wt`, `moe-dual`) |
| Gate 4.5 LR schedule optimization | **Cierre operativo** | resultados consolidados y usados en selección de checkpoints |
| Gate 5A | Replanteado | conditioned projections (implementado) + combinatorios `t3-wt` + dos slots TBD; ejecucion oportunista en paralelo con Escalon 2 |
| Gate 5B showcase científico | **En curso** | Paquete local cerrado + bloque UNC en progreso (T05 `9/15` sync local y bloque D0 corriendo; T02 `4/4` en cola) |

---

## Hallazgos Causales del Corte

1. **Dual same-modality es superaditivo**  
D4 y A4 por separado dan `+3.4pp`; juntos (`d4a4`) dan `+9.6pp`.

2. **Reverse cross-attention supera al cross-attention regular**  
Se observó en audio y MIDI (`A4r>A4x`, `D4r>D4x`).

3. **Cross-modal injection temprana no fue mecanismo ganador**  
`d4a4cm` quedó por debajo del baseline (`-7.8pp` vs D0).

4. **El mejor espacio no apareció por accidente**  
`d4a4-scratch` superó a `D-02` por más de 20pp en el mismo marco de evaluación.

5. **Third Tower weighted (`t3-wt`) mostró convergencia tardía real**  
Pasó de `S=40.0%` (e5 en 30ep scratch) a `S=79.8%` (e30).

6. **MoE mejoró transitoriamente, pero no sostuvo el pico en extendido**  
`moe-dual` llegó a `73.0%` (e30, 60ep stretched) y luego cayó a banda 69-72; terminó dead por time limit.

7. **En 5ep, FiLM/MoE quedaron en banda 58-60%**  
La familia 4.4 no desplazó a los ganadores de Gate 4.3 en screening corto.

8. **Gate 5B valida causalidad de la rama audio**
En Test01, ablacionar A4/A4r produce caídas masivas de `S` (32-78pp según modo/modelo), mientras que ablaciones D4 en modelos duales muestran efecto marginal o nulo.

9. **Gate 5B confirma invariancia relativa en modelos con A4**
En Test04 (cerrado), los modelos con A4/A4r retienen más `S` bajo transposición MIDI que `D0`, consistente con uso de señal relativa (ratios) y no solo pitch absoluto.

10. **Test09 cierra la lectura de robustez con un patrón no trivial**
Todos los arms son robustos a shifts temporales moderados, frágiles a escalado de velocity y transposición de octava; en ruido aparece patrón bimodal: `D0` domina en SNR 40-20 dB y `d4-a4r/a4r` retienen más `S` en 5 dB.

---

## Plan Operativo Vigente

Secuencia inmediata:

1. **A/B Pre-Projection test** (corriendo en tmux `preproj_ab`): diagnóstico de si el bottleneck de generación es la proyección z→256d o el encoder fundamental.
   - avance real: extracción pre-proj de `D0` completa; `preproj_midi2events` cerrado (CE `2.9449`, frame F1 `0.1250`, `shuffle_gap=1.1498`); `preproj_audio2events` en entrenamiento (último log: e9).
2. **Test 13G Phase A** (D0 λ sweep): cuando GPU se libere. Entrena encoders con VICReg + auxiliary MiniPRDecoder (1.92M params). Sweep λ ∈ {0.03, 0.1, 0.3} × 15 epochs.
3. **Test 13G Phase B** (D0 confirmatoria): si Phase A muestra señal, 30ep × 2 seeds + control.
4. **Test 13G sobre a4r**: repetir pipeline con descriptor augmentado.
5. Completar bloque `D0` de Test05 en UNC (`42/123/456/789` running, `1337` pending al corte reportado) y luego ejecutar Test02 parameter-matched (`4` modos).
6. Consolidar reporte científico de Gate 5B con separación explícita local vs UNC.

Marco estrategico inmediato:

1. Gate 5B sigue siendo el cierre principal de Escalon 1-C.
2. Una vez cerrado Gate 5B, Escalon 2 (Speech <-> EGG) pasa a ser el foco principal del programa.
3. Gate 5A continua como linea paralela y oportunista: no bloquea Escalon 2 y solo absorbe recursos libres para conditioned projections, combinatorios `t3-wt` y futuras hipotesis acotadas.

---

## Frentes y Documentos

| Documento | Rol |
|-----------|-----|
| `README.md` | Entrada principal del repositorio |
| `Documents/00_TRONCAL/HANDOFF.md` | Continuidad operativa |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md` | Plan maestro vigente |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/INDEX_BIAS_CONTROL.md` | Navegación del frente |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/RANKING_DESCRIPTORES_UNIFICADO.md` | Tabla canónica corta+larga |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/09_GATE_4_5_LR_SCHEDULE_OPTIMIZATION/README.md` | Gate 4.5 (scheduler/LR) |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_UNC.md` | Estrategia distribuida LOCAL+UNC |
| `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md` | Evolución histórica de representaciones |
| `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/CATALOGO_NARRATIVO_DESCRIPTORES_RATIOS_PHIDEUS.md` | Catálogo vivo de descriptores |

Nota operativa:
- Foundation lock publicado en GitHub Release: `v0.1.0-foundation` (`foundation_locked_e25.pt`, MD5 `ddb2ebf7075eec4dcec1628341ec4942`).

---

*Documento actualizado al corte operativo 2026-02-27 (Gate 5B activo con Test 13G implementado; A/B pre-projection corriendo; fase UNC en progreso parcial).*
