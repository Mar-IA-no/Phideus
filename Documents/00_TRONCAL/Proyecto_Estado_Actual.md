<div align="center">

# Proyecto Estado Actual
### Phideus v5.0

![Program](https://img.shields.io/badge/Program-Research_Active-0A7E3B?style=for-the-badge)
![Current Focus](https://img.shields.io/badge/Focus-Escalon_1--C-1F6FEB?style=for-the-badge)
![Bias Control](https://img.shields.io/badge/BIAS_CONTROL-Bloque_A_Activo_+_Gate_4.2_Planificado-F59E0B?style=for-the-badge)

</div>

> [!IMPORTANT]
> **Actualizado**: 2026-02-12  
> **Estado**: Escalon 1-C en etapa post-diagnostico (Gate 6 + Gate 4.2 pre-red completados, Bloque A v1.1 con S0/Run A/Run B/Run C/Run D cerrados y `Run D-02` en curso; Gate 4.2 ratio-centrico listo para screening tras foundation lock definitivo)  
> **Infraestructura**: linea `VibeTensor` en pausa hasta cerrar Bloque A de BIAS_CONTROL

## Navegacion rapida

- [Resumen Ejecutivo](#resumen-ejecutivo)
- [Estado por Gate](#estado-por-gate)
- [Hallazgo Causal Central](#hallazgo-causal-central)
- [Plan Operativo Vigente](#plan-operativo-vigente)
- [Protocolo Anti-Variable-Fantasma](#protocolo-anti-variable-fantasma)
- [Frentes del Proyecto](#frentes-del-proyecto)
- [Documentos Troncales](#documentos-troncales)

---

## Resumen Ejecutivo

### Baseline vigente (referencia oficial)

`Gate 2 - checkpoint_epoch45`

| Metrica | Valor |
|---------|-------|
| A2M R@10 (structured pool 256/500/seed42) | **34.4%** |
| M2A R@10 (structured pool 256/500/seed42) | **37.6%** |
| Hard negative accuracy | **80.4%** |
| MRR A2M / M2A | **0.138 / 0.158** |
| Score balanceado `S=min(A2M,M2A)` | **34.4** |

### Lectura operativa actual

1. Gate 3 (DANN) quedo cerrado por falta de mejora robusta.
2. Gate 4.1 quedo cerrado por umbral causal (`R1-rescue` insuficiente).
3. El diagnostico post Gate 4.1 quedo completado:
   - Gate 6: explica la degradacion.
   - Gate 4.2 pre-red: NO-GO para extractor CQT de ratios audio.
4. La etapa activa es el **Bloque A v1.1** (`S0/A/B/C/D`) para recuperar rendimiento sin romper comparabilidad.
5. Gate 4.2 ratio-centrico queda integrado como siguiente etapa condicionada, con plan final consolidado.

### Bloque A v1.1 (corte operativo)

| Etapa | Estado | A2M R@10 | M2A R@10 | hard_neg | S=min(A2M,M2A) |
|-------|--------|----------|----------|----------|----------------|
| S0 (control) | Completado | 34.4% | 37.6% | 80.4% | 34.4% |
| Run A (adapter) | Completado | 30.0% | 38.6% | 76.8% | 30.0% |
| Run B (partial unfreeze) | Completado | 43.2% | 43.4% | 85.2% | 43.2% |
| Run C (hybrid) | Completado | 49.4% | 51.0% | 88.4% | 49.4% |
| Run D (full unfreeze) | Completado | 51.0% | 51.8% | 89.2% | 51.0% |
| Run D-02 (full unfreeze, 30 ep) | En curso | — | — | — | — |

Lectura:
1. Run B (ep3) establecio foundation provisional fuerte (`S=43.2%`, asimetria 0.2pp).
2. Run C cerro con mejor checkpoint en epoch 5 (`S=49.4%`, `hard_neg=88.4%`).
3. Run D cerro en epoch 5 (`S=51.0%`, `A2M=51.0%`, `M2A=51.8%`, `hard_neg=89.2%`).
4. Foundation provisional actual: `Run D epoch5`.
5. Siguiente decision experimental: cerrar `Run D-02` y resolver lock final `C5 vs D5 vs D-02(best)` con desempate robusto.

### Cuadros de arquitectura y configuracion (preflight por run)

Fuente: `data/bias_control_medium/training_outputs/bloqueA_runA_log.txt`, `data/bias_control_medium/training_outputs/bloqueA_runB_log.txt`, `data/bias_control_medium/training_outputs/bloqueA_runC_log.txt`, `data/bias_control_medium/training_outputs/bloqueA_runD/training.log`, `data/bias_control_medium/training_outputs/bloqueA_runD-02/training.log`.

#### Run A (adapter bottleneck)

| Module Group | Trainable | Frozen | Status |
|---|---:|---:|---|
| Audio Adapters | 528,640 | 0 | TRAIN |
| Audio CNN | 0 | 3,158,528 | FROZEN |
| Audio PosEmb | 0 | 6,144,000 | FROZEN |
| Audio Projection | 920,832 | 0 | TRAIN |
| Audio Transformer | 0 | 50,384,896 | FROZEN |
| MIDI Embedding | 316,928 | 0 | TRAIN |
| MIDI OutputNorm | 1,024 | 0 | TRAIN |
| MIDI Projection | 658,688 | 0 | TRAIN |
| MIDI Transformer | 12,609,536 | 0 | TRAIN |
| **TOTAL** | **15,035,648** | **59,687,424** | |

LR por grupo: `adapters=5e-4`, `midi_encoder=5e-5`, `projections=1e-4`.

#### Run B (partial unfreeze)

| Module Group | Trainable | Frozen | Status |
|---|---:|---:|---|
| Audio CNN | 0 | 3,158,528 | FROZEN |
| Audio PosEmb | 0 | 6,144,000 | FROZEN |
| Audio Projection | 920,832 | 0 | TRAIN |
| Audio Transformer | 25,192,448 | 25,192,448 | MIXED |
| MIDI Embedding | 316,928 | 0 | TRAIN |
| MIDI OutputNorm | 1,024 | 0 | TRAIN |
| MIDI Projection | 658,688 | 0 | TRAIN |
| MIDI Transformer | 12,609,536 | 0 | TRAIN |
| **TOTAL** | **39,699,456** | **34,494,976** | |

LR por grupo: `audio_layers_2_3=1e-5`, `midi_encoder=5e-5`, `projections=1e-4`.

#### Run C (hybrid)

| Module Group | Trainable | Frozen | Status |
|---|---:|---:|---|
| Audio Adapters | 264,320 | 0 | TRAIN |
| Audio CNN | 0 | 3,158,528 | FROZEN |
| Audio PosEmb | 0 | 6,144,000 | FROZEN |
| Audio Projection | 920,832 | 0 | TRAIN |
| Audio Transformer | 25,192,448 | 25,192,448 | MIXED |
| MIDI Embedding | 316,928 | 0 | TRAIN |
| MIDI OutputNorm | 1,024 | 0 | TRAIN |
| MIDI Projection | 658,688 | 0 | TRAIN |
| MIDI Transformer | 12,609,536 | 0 | TRAIN |
| **TOTAL** | **39,963,776** | **34,494,976** | |

LR por grupo: `adapters=5e-4`, `audio_layers_2_3=1e-5`, `midi_encoder=5e-5`, `projections=1e-4`.

#### Run D (full unfreeze, split-LR)

| Module Group | Trainable | Frozen | Status |
|---|---:|---:|---|
| Audio CNN | 0 | 3,158,528 | FROZEN |
| Audio PosEmb | 0 | 6,144,000 | FROZEN |
| Audio Projection | 920,832 | 0 | TRAIN |
| Audio Transformer | 50,384,896 | 0 | TRAIN |
| MIDI Embedding | 316,928 | 0 | TRAIN |
| MIDI OutputNorm | 1,024 | 0 | TRAIN |
| MIDI Projection | 658,688 | 0 | TRAIN |
| MIDI Transformer | 12,609,536 | 0 | TRAIN |
| **TOTAL** | **64,891,904** | **9,302,528** | |

LR por grupo: `audio_layers_0_1=5e-6`, `audio_layers_2_3=1e-5`, `midi_encoder=5e-5`, `projections=1e-4`.

---

## Estado por Gate

| Gate / Etapa | Estado | Resultado |
|--------------|--------|-----------|
| Gate 0 | Completado | GO |
| Gate 1 | Completado | GO (sanity) |
| Gate 2 | Completado | GO (baseline de referencia) |
| Gate 2.5 | Completado | Diagnostico de separabilidad |
| Gate 3 (DANN) | Cerrado | NO-GO (no mejora estable) |
| Gate 4 base | Completado | Senal mixta |
| Gate 4.1 | Cerrado | `R1-rescue` no supera umbral |
| Gate 6 (diagnostico) | Completado | Causa raiz confirmada |
| Gate 4.2 pre-red (H4.2-6) | Completado | NO-GO (AUC ~ chance) |
| Bloque A v1.1 (S0/A/B/C/D) | Activo | S0/Run A/Run B/Run C/Run D cerrados; mejor single-seed D(ep5) |
| Foundation lock C/D | En cierre | Desempate final pendiente entre C5 y D5 para habilitar screening Gate 4.2 |
| Gate 4.2 ratio-centrico (post Bloque A) | Planificado | Implementacion puede correr en paralelo; screening solo tras foundation definitivo |
| Gate2R-lite | Backlog post Gate 4.2 | Higiene metodologica (no bloqueante para screening ratio-centrico) |
| Gate 5 | Hold | Opcional |

---

## Hallazgo Causal Central

### Que se encontro

En los checkpoints fine-tuned post Gate 2, el `audio encoder` permanecio efectivamente congelado, mientras cambiaban sobre todo `midi_encoder` y `projection heads`.

### Evidencia sintetica

| Modelo | Separation (correcto - incorrecto) | Bridge distance |
|--------|-------------------------------------|-----------------|
| Gate 2 | **0.479** | **3.27** |
| RB0 | 0.396 | 4.50 |
| RA5 | 0.419 | 4.47 |
| R1 | 0.395 | 4.68 |

### Implicacion

La degradacion de A2M no fue "porque ratios si/no" en abstracto, sino por un regimen de ajuste asimetrico. Por eso la siguiente iteracion se centra en adapter/unfreeze controlado con gates cuantitativos.

---

## Plan Operativo Vigente

Documento canonico:
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/05_PLAN_POST_DIAGNOSTICO_BLOQUE_A/PLAN_EJECUCION_POST_DEC005_v1.1.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/06_GATE_4_2_RATIO_CENTRICO/PLANES/plan_gate_4.2.md`

Secuencia vigente:
1. `S0` (eval-only) para control de reproducibilidad.
2. `Run A` (adapters con audio base congelado) - completado, INCONCLUSO.
3. `Run B` (partial unfreeze de capas altas de audio) - completado (mejor ep3).
4. `Run C` (hibrido adapters + partial unfreeze) - completado (mejor ep5).
5. `Run D` (full-unfreeze) - completado (mejor ep5).
6. `Run D-02` (full-unfreeze desde cero, 30 epocas) - en curso para robustecer foundation lock.
7. `Gate 4.2` - codigo en paralelo habilitado; screening post-foundation lock final.
8. `Gate2R-lite` - backlog de higiene metodologica posterior a Gate 4.2.

Criterio primario de screening:
- `S=min(A2M,M2A)` y `hard_neg` sobre protocolo canónico (`pool=256`, `queries=500`, `seed=42`).

Estado de colaboracion:
- `COLLAB OFF` salvo activacion explicita del usuario.

---

## Protocolo Anti-Variable-Fantasma

Para evitar que vuelva a escaparse un factor estructural (como el `audio encoder` congelado), se adopta esta regla minima antes de cada ola de training:

1. **Inventario de trainables pre-run**
   - Reportar por modulo: total params, trainables, frozen.
2. **Sanity de drift post-run corto**
   - Verificar que los modulos esperados realmente cambian (`rel_change > 0` donde corresponda).
3. **Control de comparabilidad obligatorio**
   - Misma configuracion canonica de evaluacion y misma semilla para comparaciones causales.
4. **Gate de corte temprano por evidencia**
   - Si run de control no reproduce baseline o drift contradice hipotesis, no se escala.
5. **Trazabilidad documental inmediata**
   - Registrar decision, evidencia y proximo paso en roadmap + bitacora.

---

## Frentes del Proyecto

| Frente | Estado | Documento eje |
|--------|--------|---------------|
| BIAS_CONTROL | Activo (principal) | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md` |
| Pre-analisis de ratios (historico) | Pausado | `Documents/01_FRENTES_ACTIVOS/ESCALON_1/Plan_implementacion.md` |
| UOEMD revisionismo | Cerrado (NO-GO) | `Documents/03_FRENTES_CERRADOS/UOEMD/UOEMD_Revisionismo/ROADMAP.md` |
| VibeTensor spike | Pausado | `Documents/02_FRENTES_PAUSADOS/VIBETENSOR_SPIKE_PLAN/VIBETENSOR_SPIKE_PLAN.md` |

---

## Documentos Troncales

| Documento | Rol |
|-----------|-----|
| `README.md` | Entrada principal del repositorio |
| `Documents/00_TRONCAL/INDICE_DOCUMENTACION.md` | Mapa global de documentos |
| `Documents/00_TRONCAL/HANDOFF.md` | Hand-off operativo para continuidad entre sesiones/instancias |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md` | Estado tecnico y plan detallado de BIAS_CONTROL |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/INDEX_BIAS_CONTROL.md` | Navegación por fases y orden documental canónico |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/04_DIAGNOSTICO_GATE_6_Y_GATE_4_2/INFORME_DEC005_DIAGNOSTICO_COMPLETO.md` | Evidencia completa del diagnostico cerrado |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/05_PLAN_POST_DIAGNOSTICO_BLOQUE_A/PLAN_EJECUCION_POST_DEC005_v1.1.md` | Plan activo de ejecucion |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/06_GATE_4_2_RATIO_CENTRICO/PLANES/plan_gate_4.2.md` | Plan final de Gate 4.2 ratio-centrico (post Bloque A) |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/04_DIAGNOSTICO_GATE_6_Y_GATE_4_2/CURADURIA_VISUAL/INDEX_VISUAL.md` | Curaduria visual y snapshot de resultados |

Nota de operación:
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/resultados_compartir/` se usa como espejo local de visualizaciones y no se versiona en git.

---

*Documento actualizado: 2026-02-12 (Escalon 1-C post-diagnostico, Bloque A v1.1 con `Run D-02` en curso y lock final `C5 vs D5 vs D-02(best)` pendiente antes de screening Gate 4.2)*
