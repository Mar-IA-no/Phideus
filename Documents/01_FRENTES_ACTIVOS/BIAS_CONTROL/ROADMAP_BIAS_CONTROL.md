<div align="center">

# Roadmap BIAS_CONTROL
### Escalon 1-C de Phideus: de baseline cross-modal a diagnostico causal y plan de recuperacion

![Version](https://img.shields.io/badge/Version-2.2-111827?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-MAESTRO_v3.0.0-1F6FEB?style=for-the-badge)
![Fase](https://img.shields.io/badge/Fase-Escalon_1--C-F59E0B?style=for-the-badge)
![Estado](https://img.shields.io/badge/Estado-Gate_5B_CERRADO-0A7E3B?style=for-the-badge)

</div>

> [!IMPORTANT]
> **Fecha de corte**: 2026-03-05
> **Estado del programa**: Gate 4.3 y Gate 4.4 permanecen cerrados. Gate 4.5 queda en cierre operativo y **Gate 5B ya quedó completamente cerrado** como línea principal de Escalón 1-C. `Test11` ya no es solo un hallazgo parcial: cerró `4/4` con retención `d4a4=0.770 > d4-a4r=0.748 > a4r=0.712 > D0=0.597`. `Test05` quedó cerrado en `results_unc` (`15/15`), `Test02` cerró `4/4` y `13G-B` cerró `4/4` sin ventaja descriptor-guided en decodificabilidad pre-pooling.
> **Siguiente paso operativo**: (1) sostener Gate 5B como bloque cerrado y usar la tesis “ventaja geométrica, no de feature richness” como lectura canónica, (2) monitorear Gate 6 `Exp C` en sus dos planos activos (corrida local `a4r` + resubmisión UNC `1144560`), (3) abrir `Exp A` cuando haya slot en UNC y (4) mantener Escalón 2 como foco principal sin bloquearse por Gate 5A ni por Gate 6.
> **Roadmap post Gate 4.5**: Gate 5 sigue en dos lineas paralelas, pero con nuevo encuadre: Linea A queda replanteada como exploracion oportunista (conditioned projections + combinatorios de alta prioridad, sin bloquear Escalon 2) y Linea B ya quedó como cierre científico consolidado. Gate 6 pasa a alojar la nueva línea AMT.
> **Nota de foco**: `Documents/02_FRENTES_PAUSADOS/VIBETENSOR_SPIKE_PLAN/` queda desacoplado y no bloquea el cierre de BIAS_CONTROL.

---

## Navegacion

- [1. Estado Ejecutivo](#1-estado-ejecutivo)
- [2. Marco Teorico y Objetivo Cientifico](#2-marco-teorico-y-objetivo-cientifico)
- [3. Protocolo Canonico de Evaluacion](#3-protocolo-canonico-de-evaluacion)
- [4. Arquitectura y Configuracion Base](#4-arquitectura-y-configuracion-base)
- [5. Gates Ejecutados (0 a 4.1)](#5-gates-ejecutados-0-a-41)
- [6. Diagnostico Post Gate 4.1 (Decision de diagnostico)](#6-diagnostico-post-gate-41-decision-de-diagnostico)
- [7. Plan Aprobado de Ejecucion (v1.1)](#7-plan-aprobado-de-ejecucion-v11)
- [7.6 Integracion de Gate 4.2](#76-integracion-de-gate-42-post-bloque-a)
- [7.9 Exploracion Foundation y Visualizacion](#79-exploracion-foundation-y-visualizacion)
- [7.10 Bifurcacion Metodologica Gate 4.3](#710-bifurcacion-metodologica-gate-43)
- [7.11 Resultados Gate 4.3 (13 brazos + scratch)](#711-resultados-gate-43-13-brazos--scratch)
- [7.12 Gate 4.4: Arquitecturas Mayores](#712-gate-44-arquitecturas-mayores)
- [8. Gate 4.5: LR Schedule Optimization](#8-gate-45-lr-schedule-optimization)
- [9. Gate 5: Dos Lineas Paralelas](#9-gate-5-dos-lineas-paralelas)
- [10. Riesgos Tecnicos y Criterios de Corte](#10-riesgos-tecnicos-y-criterios-de-corte)
- [11. Artefactos de Verdad](#11-artefactos-de-verdad)

---

## Mapa Documental

- Índice de fase/documentos: `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/INDEX_BIAS_CONTROL.md`
- Carpeta espejo local para compartir visuales: `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/resultados_compartir/` (no versionada)
- Gate 4.2 ratio-centrico (plan final): `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/06_GATE_4_2_RATIO_CENTRICO/plan_gate_4.2.md`
- Gate 4.3 ratio re-centrico (plan bifurcado): `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/07_GATE_4_3_RATIO_RE_CENTRICO/INFORME_GATE_4_3_RATIO_RE_CENTRICO.md`
- Gate 4.4 arquitecturas mayores (third tower + FiLM + MoE): `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/08_GATE_4_4_ARQUITECTURAS_MAYORES/README.md`
- Gate 4.5 LR schedule optimization (extended runs): `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/09_GATE_4_5_LR_SCHEDULE_OPTIMIZATION/README.md`
- Gate 5 Linea A (replanteo estrategico: conditioned projections + combinatorios oportunistas): `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/10_GATE_5_LINEA_A_BARRIDO/README.md`
- Gate 5 Linea B (showcase cientifico): `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/README.md`
- Gate 6 AMT (validación downstream): `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/12_GATE_6_AMT/README.md`

---

## 1. Estado Ejecutivo

### 1.1 Que esta cerrado y que esta abierto

**Cerrado**:
- Gate 0: integridad de datos y alineacion.
- Gate 1: baselines intra-modal (diagnostico, no bloqueante de pipeline).
- Gate 2: baseline cross-modal robusto (checkpoint de referencia vigente).
- Gate 2.5: probes diagnosticos de separabilidad modal.
- Gate 3: linea DANN cerrada (no mejora estable sobre Gate 2).
- Gate 4.1: cerrado por criterio pre-registrado (senal marginal insuficiente).
- Decision de diagnostico post Gate 4.1 (DEC-005): completada, sin training.
- Gate 4.3: cerrado con 13 brazos + run largo `d4a4-scratch` (record del bloque 30ep, `S=83.6%`).
- Gate 5B: showcase cientifico cerrado (`Test05`, `Test02`, `Test11`, `13G-A`, `13G-B` ya integrados en la lectura canónica del frente).

**Abierto**:
- Gate 5A — linea replanteada: conditioned projections implementado, combinatorios `t3-wt` pendientes y ejecucion oportunista en paralelo con recursos libres.
- Gate 6 AMT — validación downstream: `Exp 0` completo en local, `Exp C` activo (corrida local `a4r` + resubmisión UNC `1144560`), `Exp A` listo para submitir y `Exp B` bloqueado por `Exp A`.

**En cierre operativo**:
- Gate 4.5 — LR Schedule Optimization (bloque usado como soporte de checkpoints canónicos para Gate 5B).

### 1.2 Baseline oficial vigente

**Modelo de referencia actual**: `data/bias_control_medium/training_outputs/gate2/checkpoint_epoch45.pt`

Structured pool canonico (pool=256, queries=500, seed=42):
- A2M R@10: **34.4%**
- M2A R@10: **37.6%**
- Hard negative accuracy (same-piece-diff-time): **80.4%**
- MRR A2M/M2A: **0.138 / 0.158**
- Score balanceado primario: `S = min(A2M, M2A) = 34.4%`

### 1.3 Mensaje tecnico de alto nivel

El ciclo completo mostro que:
1. El sistema aprende retrieval cross-modal real sobre MAESTRO (Gate 2).
2. Forzar invariancia modal con DANN no dio mejora estable ni causal (Gate 3).
3. Las variantes de ratio auxiliary probadas no superaron el umbral de promocion (Gate 4.1).
4. El diagnostico post Gate 4.1 identifico la causa estructural dominante: **fine-tuning asimetrico con audio encoder congelado**.
5. La siguiente iteracion mantiene control experimental estricto y separa explicitamente paradigmas MIDI temperado vs audio no temperado.

---

## 2. Marco Teorico y Objetivo Cientifico

### 2.1 Problema cientifico

BIAS_CONTROL evalua si un modelo audio<->MIDI puede alinear identidad temporal de segmentos musicales en presencia de:
- alta similitud intra-pieza,
- variabilidad interpretativa,
- sesgos de modalidad,
- shortcuts por firma de pieza.

El objetivo no es solo subir recall bruto, sino **maximizar recuperacion bidireccional con separacion robusta contra negativos duros**.

### 2.2 Hipotesis historicas del programa

- Hipotesis base (Gate 2): VICReg + encoder audio estable + encoder MIDI entrenable produce espacio cross-modal util.
- Hipotesis de sesgo modal (Gate 3): DANN podria reducir shortcut modal y mejorar retrieval.
- Hipotesis de ratio auxiliary (Gate 4.1): una vista de ratios puede aportar senal extra.
- Hipotesis post-diagnostico: la degradacion principal viene de un regimen que mueve MIDI/projections y no mueve audio.

### 2.3 Criterio de verdad operacional

El criterio de cierre por experimento es cuantitativo y pre-registrado en structured pool canonico:
- A2M R@10
- M2A R@10
- Hard negative accuracy
- `S = min(A2M, M2A)` como metrica de equilibrio

Este criterio evita optimizaciones unidireccionales (por ejemplo, mejorar M2A a costa de degradar A2M).

---

## 3. Protocolo Canonico de Evaluacion

### 3.1 Structured pool (decision metric)

Por query:
- 1 positivo
- 64 hard negatives (misma pieza, distinto tiempo)
- 32 semi-hard (mismo compositor, otra pieza)
- 159 random

Total por query: 256 candidatos.

Configuracion fija de comparabilidad:
- `pool_size=256`
- `n_queries=500`
- `seed=42`
- mismo split de validacion

### 3.2 Metricas primarias y secundarias

Primarias:
- A2M R@10
- M2A R@10
- Hard negative accuracy
- `S = min(A2M, M2A)`

Secundarias:
- R@1, R@5, R@20
- MRR
- mean rank
- separacion de similitudes (correct vs incorrect)
- bridge distance en UMAP

### 3.3 Regla de interpretacion

- No se acepta conclusion por metricas de monitoreo interno de training sin structured pool.
- No se comparan runs con pools no equivalentes como si fueran definitivos.
- Toda recomendacion de escalado debe referenciar explicitamente metrica primaria.

---

## 4. Arquitectura y Configuracion Base

### 4.1 Arquitectura (checkpoint baseline)

- Audio encoder (`MERTEncoderLite`):
  - CNN feature extractor (4 Conv1d + GroupNorm + GELU)
  - Transformer encoder (4 capas, d_model=1024)
  - Positional embedding aprendible
- MIDI encoder (Transformer, 4 capas, d_model=512)
- Projection heads:
  - Audio projection: 1024 -> 512 -> 256
  - MIDI projection: 512 -> 512 -> 256

### 4.2 Tamaño aproximado de parametros

- Audio encoder total: ~59.7M
- MIDI encoder total: ~13.9M
- Proyecciones: ~1.6M
- Total aproximado: ~74-75M parametros

### 4.3 Configuracion de regimen usada en el ciclo principal

- `segment_len=4.0`
- `hop=1.0`
- `batch_size=16`
- Entrenamiento acotado en varios runs con `max_batches_per_epoch=1000`
- Evaluacion de monitoreo con `max_val_batches=846`

---

## 5. Gates Ejecutados (0 a 4.1)

## 5.1 Gate 0 — Integridad de datos (PASS)

Objetivo: validar consistencia de slicing y alineacion.

Resultado funcional:
- Integridad de pipeline confirmada.
- Segmentacion operativa estable para ciclo Gate 2+.

## 5.2 Gate 1 — Baselines intra-modal (diagnostico)

Metricas historicas reportadas:
- Audio->Audio R@10: 98.4%
- MIDI->MIDI R@10: 100%

Lectura tecnica:
- Las rutas intra-modal no eran el cuello de botella del problema cross-modal.

## 5.3 Gate 2 — Baseline cross-modal (PASS, referencia vigente)

Entrenamiento historico en dos fases:
- Fase 1: 200 batches/epoch
- Fase 2: 1000 batches/epoch

Checkpoint seleccionado: `checkpoint_epoch45.pt`

Resultados clave:
- Gap de similitud (aligned-random): **0.478**
- Structured pool A2M/M2A R@10: **34.4% / 37.6%**
- Hard neg accuracy: **80.4%**

Conclusion Gate 2:
- Baseline solido para comparar intervenciones posteriores.

## 5.4 Gate 2.5 — Probes diagnosticos

Hallazgos reportados:
- Domain separability: **92.7%** (shortcut modal fuerte)
- Sin colapso de embedding (dead dims = 0)

Decision historica:
- Abrir Gate 3 para testear DANN como contramedida.

## 5.5 Gate 3 — DANN (linea cerrada)

Se evaluaron multiples regimenes (A/B/C/D), incluyendo normalizacion y schedules distintos de lambda adversarial.

Comparativa definitiva en structured pool (resumen):
- Gate 2 baseline: A2M 34.4, M2A 37.6, hard neg 80.4
- Mejor punto DANN observado (transitorio): comparable en algunos cortes
- Regimen sostenido (Run D de DANN): inferior al baseline

Lectura de cierre:
- DANN no produjo mejora robusta y reproducible sobre Gate 2.
- La linea queda cerrada para esta fase del roadmap.

## 5.6 Gate 4 / 4.1 — Ratio auxiliary y control causal (cerrado)

### Gate 4 Run A (ratio auxiliary)

Structured pool:
- Ep5: A2M 31.4, M2A 40.6, hard neg 79.0
- Ep30: A2M 29.2, M2A 36.4, hard neg 74.8

Lectura:
- Mejor punto temprano, degradacion con entrenamiento prolongado.
- Senal asimetrica: mejora M2A sin sostener A2M.

### Gate 4.1 (control causal y rescue)

Checkpoints de referencia:
- Gate 2: 34.4 / 37.6, hard neg 80.4
- RB0 (control sin ratio): 30.2 / 38.2, hard neg 77.6
- RA5 (ratio baseline): 31.4 / 40.6, hard neg 79.0
- R1-rescue (descriptor enriquecido): 31.0 / 40.2, hard neg 78.8

Cierre formal:
- La mejora relativa no alcanzo umbral de promocion pre-registrado.
- Gate 4.1 se cierra sin abrir expansion de variantes.

---

## 6. Diagnostico Post Gate 4.1 (Decision de diagnostico)

## 6.1 Alcance

La Decision de diagnostico post Gate 4.1 (DEC-005) se ejecuto en modo **diagnostic-only**:
- sin habilitar entrenamientos nuevos,
- con foco en causa raiz de la degradacion.

Tracks ejecutados:
1. Retroanalisis de checkpoints (Gate 6, fase diagnostica).
2. Pre-red dual-domain de ratios audio+MIDI (H4.2-6).

## 6.2 Hallazgo estructural principal

Layer drift analysis:
- Audio encoder (CNN + Transformer + pos embedding): drift ~0% en checkpoints fine-tuned.
- Cambios concentrados en MIDI encoder y proyecciones.

Consecuencia observable:
- Puentes cross-modal mas largos en fine-tuning.
- Caida de separacion entre correctos e incorrectos.

Metricas sinteticas (curaduria visual):

| Modelo | Separation | Bridge distance |
|---|---:|---:|
| Gate 2 | **0.479** | **3.27** |
| RB0 | 0.396 | 4.50 |
| RA5 | 0.419 | 4.47 |
| R1 | 0.395 | 4.68 |

Interpretacion:
- El regimen mueve el espacio MIDI mientras deja audio casi fijo.
- Eso explica degradacion A2M y asimetria creciente.

## 6.3 H4.2-6 pre-red (dual-domain ratios): NO-GO

Resultados oficiales:

| Fase | AUC | delta_sim | Veredicto |
|---|---:|---:|---|
| P0 (oracle) | 0.559 | +0.034 | NO-GO |
| P1 (real) | 0.502 | -0.004 | NO-GO |

Conclusion:
- La extraccion CQT propuesta no entrega senal discriminativa robusta para esta hipotesis.
- Se elimina H4.2-6 como via de training en esta iteracion.

## 6.4 Decision operativa derivada

Matriz aplicada:
- Drift asimetrico: SI
- H4.2-6: NO-GO

Siguiente via prioritaria:
- Adapter/unfreezing controlado del audio encoder + S-control.

---

## 7. Plan Aprobado de Ejecucion (v1.1)

Documento canonico:
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/05_PLAN_POST_DIAGNOSTICO_BLOQUE_A/PLAN_EJECUCION_POST_DEC005_v1.1.md`

Estado:
- Plan validado por Claude + Codex.
- Aprobado por usuario para ejecucion.

## 7.1 Avance real de ejecucion (corte actual)

| Etapa | Estado | A2M R@10 | M2A R@10 | hard_neg | S=min(A2M,M2A) | Decision |
|---|---|---:|---:|---:|---:|---|
| S0 (control) | Completado | 34.4% | 37.6% | 80.4% | 34.4% | Control reproducido |
| Run A (adapter) | Completado | 30.0% | 38.6% | 76.8% | 30.0% | INCONCLUSO |
| Run B (partial unfreeze) | Completado | 43.2% | 43.4% | 85.2% | 43.2% | Mejor hasta ahora (ep3) |
| Run C (hybrid) | Completado | 49.4% | 51.0% | 88.4% | 49.4% | Runner-up actual (ep5) |
| Run D (full unfreeze) | Completado | 51.0% | 51.8% | 89.2% | 51.0% | Mejor single-seed (ep5) |
| Run D-02 (full unfreeze, 30 ep) | Completado (best ep25; empate S con ep26) | 61.8% | 62.4% | 90.4% | 61.8% | Extension larga desde cero (misma recipe de Run D) |

Notas:
1. Run A tuvo interrupcion por caida de servidor durante epoch 5 y se completo con resume desde `checkpoint_epoch4`.
2. La traza canonica por epoca quedo en `eval_per_epoch/eval_epoch1..5.json`.
3. Run B mejor checkpoint: epoch 3 (`S=43.2%`, `hard_neg=85.2%`). Epoch 5 quedo en `S=42.4%`, `hard_neg=86.8%`.
4. Run C cerro en epoch 5 (`S=49.4%`, `A2M=49.4%`, `M2A=51.0%`, `hard_neg=88.4%`).
5. Run D cerro en epoch 5 (`S=51.0%`, `A2M=51.0%`, `M2A=51.8%`, `hard_neg=89.2%`).
6. Re-evaluacion multi-seed (`42/123/456/789`) ejecutada en `e25` y `e26`; `e26` mejora levemente media, `e25` muestra mayor estabilidad de gap.
7. Lock formal resuelto: `foundation_locked_e25.pt` como checkpoint inmutable para Gate 4.2.
8. `explore_foundation.py` ejecutado sobre checkpoint bloqueado; resultados en `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/resultados_compartir/`.
9. Gate 4.2 cerrado: `D4 8ep` confirma `S_best=64.2%` (e7) y `hard_neg_best=91.6%`.
10. Gate 4.3 cerrado: 13 brazos completados (incluye Fase 5 en UNC), mejor 5ep en `d4a4=69.8%`.
11. `d4a4-scratch` 30ep completado con `S=83.6%` (record del bloque 30ep), multi-seed e30 `84.1% +/- 2.3pp`.
12. `Gate2R-lite` se mantiene en backlog post Gate 4.5 (no bloqueante).

### 7.1.b Cuadros de arquitectura y configuracion por run (preflight real)

Fuente: `data/bias_control_medium/training_outputs/bloqueA_runA_log.txt`, `data/bias_control_medium/training_outputs/bloqueA_runB_log.txt`, `data/bias_control_medium/training_outputs/bloqueA_runC_log.txt`, `data/bias_control_medium/training_outputs/bloqueA_runD/training.log`.
Nota: `Run D-02` reutilizó la misma politica de freeze/LR de `Run D`; traza cerrada en `data/bias_control_medium/training_outputs/bloqueA_runD-02/training.log`.

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

## 7.2 Secuencia de Bloque A

1. `Run S0` (eval-only): reproducir baseline Gate 2 sin entrenamiento.
2. `Run A` (adapters con audio base congelado).
3. `Run B` (partial unfreeze de capas altas de audio transformer).
4. `Run C` (hibrido: adapters en capas bajas + unfreeze capas altas).
5. `Run D` (full-unfreeze) condicional, ejecutado y completado (ep5).
6. `Run D-02` (full-unfreeze desde cero, 30 epocas) completado y usado para lock final.

## 7.3 Gate de screening (5 epocas)

Definiciones:
- `S_control = S_S0`
- `hard_neg_control = hard_neg_S0`

Criterios:
- SCALE: `S_run - S_control >= +1.5pp` y `hard_neg_run >= hard_neg_control - 1pp`
- GO fuerte: SCALE + `S_run >= 33.4%`
- DROP: `S_run < 30%` o `hard_neg_run < 75%`

## 7.4 Regla de disciplina

- Screening sin early stopping en 5 epocas.
- Checkpoint en todas las epocas (comportamiento por default del training loop actual).
- Escalado a 15-30 epocas solo para ganador claro.

## 7.5 Costos estimados (v1.1)

- Screening completo: ~8-9h
- Pipeline completo (con escalado + bloque visual/generativo): ~15-22h

## 7.6 Integracion de Gate 4.2 (post Bloque A)

Gate 4.2 queda formalmente integrado al roadmap de BIAS_CONTROL como etapa siguiente condicionada al cierre de Bloque A:

1. Comparativa final C/D/D-02 consolidada (con A/B/C como referencia historica).
2. Foundation lock definitivo resuelto con desempate robusto multi-seed: `foundation_locked_e25.pt`.
3. Politica de freeze consolidada para screening Gate 4.2.
4. Cerrar extension `D4` (8 epocas) como validacion de persistencia de mejora temprana.
5. Abrir Gate 4.3/4.4 segun bifurcacion definida en la seccion 7.10.

## 7.7 Paralelizacion permitida (DEC-007)

- La implementacion de codigo de Gate 4.2 puede avanzar en paralelo mientras se cierra foundation lock.
- Esta paralelizacion no habilita screening antes del foundation definitivo.
- Regla operativa: paralelo para desarrollo (dataset/descriptors/training script), serial para decision cientifica (screening/confirmacion).

## 7.8 Gate2R-lite (backlog post Gate 4.2)

- Se agenda `Gate2R-lite` como higiene metodologica posterior a Gate 4.2.
- Definicion: repetir baseline Gate 2 con `MERTEncoderLite` pero incluyendo audio params en optimizer desde epoch 1.
- No bloquea Gate 4.2 porque la pregunta de Gate 4.2 es relativa (`D0 vs Dx`) dentro del mismo foundation.

Documento operativo de Gate 4.2:
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/06_GATE_4_2_RATIO_CENTRICO/plan_gate_4.2.md`

## 7.9 Exploracion Foundation y Visualizacion

- `experiments/bias_control/explore_foundation.py` implementado y ejecutado (6 probes en ~29s) sobre `foundation_locked_e25.pt`.
- Resultados clave de exploracion: retrieval full-set (`A2M top-1 15%`, `M2A top-1 5%`), pair-alignment (`gap=0.702`, `Cohen's d=3.07`), per-piece overall (`5.6%`) y outputs en `resultados_compartir/`.
- Regla operativa mantenida: usar siempre checkpoint inmutable post-lock para nuevas corridas de exploracion.
- Visualizaciones 3D publicadas en `https://altermundi.github.io/Phideus/` (adaptacion sobre `https://github.com/bbycroft/llm-viz`).

## 7.10 Bifurcacion metodologica Gate 4.3

Decision de roadmap (2026-02-14):
- **Gate 4.2** conserva el run `D4` extendido a 8 epocas dentro de la misma fase y queda cerrado.
- Resultado de cierre: `D4 8ep` alcanza `S_best=64.2%` (e7), confirma techo de `D4 3ep` y mejora `hard_neg` a `91.6%`.
- Con precondicion cumplida, queda abierto **Gate 4.3**.
- Gate 4.3 deja de ser un barrido `D0..D10` y pasa a una matriz factorial corta, causal y bifurcada.

Bifurcacion epistemologica explicita:
1. **Linea MIDI (temperada)**: descriptores basados en eventos MIDI discretos (12-TET).
2. **Linea Audio (armonia natural)**: descriptores sobre estructura espectral continua/no temperada.
3. **Linea Dual**: combinaciones MIDI+Audio para medir si hay sinergia.
4. **Mecanismos**: concat vs cross-attention vs reverse cross-attention.

Regla operativa de comparabilidad:
- Todos los brazos de Gate 4.3 se ejecutan **fresh** desde `foundation_locked_e25.pt`.
- No usar `--resume` para comparar brazos (se evita sesgo de scheduler/LR por cambio de `total_steps`).

## 7.11 Resultados Gate 4.3 (13 brazos + scratch)

Fuentes:
- `data/bias_control_medium/training_outputs/gate43/gate43_20260214_1000/`
- `data/bias_control_medium/training_outputs/gate43/gate43_d4a4_scratch_30ep/`
- `data/bias_control_medium/training_outputs/gate43/gate43_d4a4_scratch_30ep/multiseed/`

Estado: **Gate 4.3 cerrado**.

### Tabla final Gate 4.3 (sorted by S)

| Rank | Brazo | Mecanismo | Best ep | Best S | hard_neg | vs D0 |
|------|-------|-----------|---------|--------|----------|-------|
| **1** | **d4a4** | **Dual same-mod concat** | **e5** | **69.8%** | **91.6%** | **+9.6pp** |
| 2 | A4r | Reverse cross-att audio | e5 | 68.6% | 91.6% | +8.4pp |
| 3 | D4r | Reverse cross-att midi | e5 | 64.2% | 93.2% | +4.0pp |
| 4 | D4 | MIDI intervals concat | e5 | 63.6% | 91.2% | +3.4pp |
| 4 | A4 | Audio log-freq concat | e5 | 63.6% | 92.4% | +3.4pp |
| 6 | A4x | Audio cross-attention | e5 | 62.6% | 92.4% | +2.4pp |
| 7 | A7x | Audio attractor cross-att | e5 | 62.2% | 92.0% | +2.0pp |
| 8 | D0 | Baseline (sin descriptor) | e3 | 60.2% | 90.0% | — |
| 9 | D4x | MIDI cross-attention | e5 | 60.0% | 91.4% | -0.2pp |
| 10 | A7 | Audio attractor concat | e5 | 58.8% | 90.2% | -1.4pp |
| 10 | A9 | IDF attractor concat | e5 | 58.8% | 90.4% | -1.4pp |
| 12 | A8 | Onset-chroma concat | e5 | 57.4% | 90.6% | -2.8pp |
| 13 | d4a4cm | Dual cross-modal concat | e5 | 52.4% | 89.6% | -7.8pp |

### Hallazgos clave

1. **Dual superaditivo**: `d4a4` supera ampliamente la suma lineal de mejoras individuales.
2. **Reverse > cross-att regular** en ambos dominios (`A4r>A4x`, `D4r>D4x`).
3. **Same-mod > cross-modal**: `d4a4cm` quedó 17.4pp por debajo de `d4a4`.
4. **A4 y D4** sostienen el núcleo descriptorial más robusto.
5. **A8/A9** no desplazaron a A4 en esta etapa.

### d4a4-scratch (30ep, completo)

| Epoch | S | hard_neg | MRR_avg |
|------:|---:|---------:|--------:|
| 10 | 74.6% | 93.0% | 0.336 |
| 15 | 65.8% | 91.0% | 0.316 |
| 20 | 75.6% | 93.6% | 0.370 |
| 25 | 82.2% | 95.4% | 0.430 |
| 28 | 82.8% | 94.8% | 0.444 |
| 29 | 82.6% | 95.2% | 0.443 |
| 30 | 83.6% | 95.2% | 0.444 |

Resultado final:
- **S=83.6%** (record del bloque 30ep; superado luego en Gate 4.5 por `83.8%`).
- **+21.8pp** vs D-02 best (`61.8%`).
- Multi-seed e30: **84.1% +/- 2.3pp**.

## 7.12 Gate 4.4: Arquitecturas Mayores

Gate 4.4 queda definido como bloque arquitectural mayor (Third Tower + FiLM + MoE) y se mantiene **cerrado**.

Estado operativo (2026-02-23):
- Implementación Gate 4.4 cerrada y validada en `experiments/bias_control/gate43_scratch/gate43_scratch_training.py`.
- Screening 5ep **cerrado** en 24 brazos (21 originales + `moe-a4-v2/v3/v4`) con protocolo fijo: foundation + `run-d`.
- Runs largos scratch 30ep **cerrados** para `t3-wt` y `moe-dual`.
- La optimizacion temporal/scheduler se separa en Gate 4.5 (seccion 8).

### Tabla final Gate 4.4 (structured eval, 5ep)

| Brazo | Familia | Best S | Best Ep | A2M | M2A | hard_neg | vs D0 |
|------|---------|--------|---------|-----|-----|----------|-------|
| `t3-wt` | Third Tower | 67.6% | 5 | 71.4% | 67.6% | 91.2% | +7.4pp |
| `t3-tri` | Third Tower | 65.0% | 5 | 65.4% | 65.0% | 90.6% | +4.8pp |
| `moe-a4-v2` | MoE v2 | 60.2% | 5 | 60.4% | 60.2% | 90.8% | 0.0pp |
| `film-dual` | FiLM | 59.4% | 5 | 60.2% | 59.4% | 91.4% | -0.8pp |
| `moe-a4-v4` | MoE v4 | 59.4% | 5 | 60.6% | 59.4% | 91.2% | -0.8pp |
| `film-a4` | FiLM | 59.2% | 3 | 60.8% | 59.2% | 89.8% | -1.0pp |
| `moe-dual` | MoE | 59.2% | 5 | 61.2% | 59.2% | 91.6% | -1.0pp |
| `moe-a4-v3` | MoE v3 | 59.2% | 5 | 60.6% | 59.2% | 91.2% | -1.0pp |
| `film-d4` | FiLM | 58.6% | 5 | 61.0% | 58.6% | 91.8% | -1.6pp |
| `moe-a4` | MoE | 58.2% | 3 | 58.8% | 60.2% | 89.6% | -2.0pp |
| `t3-anc` | Third Tower | 42.2% | 5 | 42.2% | 42.2% | 89.4% | -18.0pp |

Notas de lectura:
- D0 de referencia corta: `S=60.2%`.
- En MoE v2/v3/v4, ninguna variante supera D0 (v2 empata).
- En FiLM, ninguna variante supera D0.
- En Third Tower, `t3-wt` y `t3-tri` quedan como mejores señales de la familia.

### Runs largos 30ep (scratch, run-d) — cierre del bloque

| Descriptor | Best S | Best Ep | A2M | M2A | hard_neg | Estado |
|-----------|--------|---------|-----|-----|----------|--------|
| `d4a4` | 83.6% | 30 | 83.6% | 84.2% | 95.2% | COMPLETADO |
| `a4r` | 82.0% | 29 | 82.6% | 82.0% | 94.4% | COMPLETADO |
| `d4-a4r` | 79.8% | 30 | 81.4% | 79.8% | 94.2% | COMPLETADO |
| `t3-wt` | 79.8% | 30 | 82.4% | 79.8% | 94.8% | COMPLETADO |
| `d4a4r` | 74.4% | 30 | 74.4% | 74.8% | 92.0% | COMPLETADO |
| `moe-dual` | 72.6% | 30 | 72.8% | 72.6% | 93.4% | COMPLETADO |

Tres propuestas de **cambio arquitectonico mayor**:

1. **Third Tower / Ratio Bridge**: tratar ratios como modalidad propia con encoder independiente. Tres torres convergen en espacio latente.
2. **FiLM estructural (audio/midi/dual)**: Feature-wise Linear Modulation aplicada a capas internas de encoder para condicionar representación con descriptores robustos.
3. **MoE con Ratio Expert**: Mixture of Experts con un experto dedicado a ratio processing.

Las tres líneas usan los mejores descriptores y mecanismos determinados en Gate 4.3.

Documentacion: `08_GATE_4_4_ARQUITECTURAS_MAYORES/README.md`

---

## 8. Gate 4.5: LR Schedule Optimization

Decision operativa (2026-02-25): Gate 4.5 queda como bloque de soporte metodológico para selección de checkpoints, mientras el frente activo se desplaza a Gate 5B.

Pregunta central:
- con arquitectura y descriptores fijos, cual scheduler/ventana temporal extrae mejor performance?

Schedulers bajo prueba:
1. cosine stretched (`--epochs 60`)
2. trapezoidal hold (`--lr-hold-fraction 0.5`)
3. cosine-tail (`--lr-cosine-ref-epochs 30 --lr-floor 0.10 --lr-tail-end 0.02`)

Tabla de runs Gate 4.5:

| Run | Scheduler | Estado | Best S | Best ep | Delta vs 30ep |
|-----|-----------|--------|--------|---------|---------------|
| d4a4 60ep | cosine stretched | COMPLETO | 83.8% | e50 | +0.2pp |
| a4r 60ep | cosine stretched | COMPLETO | 79.4% | e60 | -2.6pp |
| D0 60ep | cosine stretched | COMPLETO | 72.8% | e50 | +12.6pp |
| t3-wt 50ep | trapezoidal hold | COMPLETO | 81.2% | e50 | +1.4pp |
| d4-a4r 60ep | cosine stretched | COMPLETO | 79.8% | e55 | ±0.0pp |
| moe-dual 60ep | cosine stretched | DEAD (time limit) | 73.0% | e30 | +0.4pp (no sostenido) |
| D0 60ep | cosine-tail | CIERRE OPERATIVO (usado en Gate 5B) | 73.4% | e50 | +13.2pp vs D0@30ep |
| d4a4 60ep | cosine-tail | CIERRE OPERATIVO (no canónico para Gate 5B) | 83.4% | e30 | -0.4pp (referencia interna) |
| a4r 60ep | cosine-tail | COMPLETO | 80.6% | e60 | -1.4pp |
| d4-a4r 60ep | cosine-tail | FUERA DE RUTA CRÍTICA | — | — | — |

Documentacion: `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/09_GATE_4_5_LR_SCHEDULE_OPTIMIZATION/README.md`

---

## 9. Gate 5: Dos Lineas Paralelas

**Renumeracion** (2026-02-15): Gate 5 pasa de "opcional" a componente central del roadmap, con dos lineas.

## 9.1 Gate 5 Linea A — Replanteo estrategico y ejecucion oportunista

Gate 5A ya no se lee como un barrido amplio y bloqueante antes de Escalon 2. Ese framing pertenecia al roadmap original; hoy la prioridad real es otra.

El cierre cientifico de Escalon 1-C ya quedó resuelto en Gate 5B. Gate 5A queda vivo como una linea paralela, util para aprovechar ventanas de GPU local o slots UNC libres, sin bloquear la transicion a Escalon 2 ni la validacion downstream de Gate 6.

### Caja 1 — Ya explorado / parcialmente cerrado

| Item | Estado | Lectura actual |
|---|---|---|
| Concat same-modality | probado | `d4a4` y familia dual ya fueron medidas y comparadas |
| Reverse cross-attention | probado | `a4r` y `d4-a4r` quedaron entre los mejores brazos |
| FiLM per-layer | probado | negativo, por debajo de `D0` |
| Cross-modal bidireccional (`d4a4cm`) | probado | negativo fuerte (`52.4%`, 17.4pp por debajo de `d4a4`) |
| Third Tower | probado | `t3-wt` mostro valor real como mecanismo complementario |

Nota clave:
- cross-modal injection no esta "pendiente" en abstracto: el caso bidireccional ya se ejecuto y fue negativo. Lo que podria reabrirse mas adelante son variantes unidireccionales especificas (`CM-a`, `CM-m`) si aparece una hipotesis nueva.

### Caja 2 — Alta prioridad oportunista

| Componente | Estado | Lectura operativa |
|---|---|---|
| Conditioned projections (`a4r-ctrl`, `a4r-pca`, `a4r-pcm`, `a4r-pcd`, `a4r-pcd-zero`) | implementado y verificado | ataca el bottleneck de proyeccion diagnosticado por Pre-Proj A/B |
| `t3-wt-vanilla` | diseno listo | control barato para aislar contribucion de tower weighted |
| `t3-wt-a4r` | diseno listo | combinatorio entre dos mecanismos con valor demostrado |
| C3 / C4 | TBD | reservados para hipotesis nuevas del usuario |

### Caja 3 — Backlog legacy de baja prioridad

- barrido amplio de descriptores no probados (`D3`, `D8`, `D9`, `A1-A6`, etc.);
- variantes cross-modal unidireccionales sin hipotesis nueva fuerte;
- deep injection por capas (AdaLN / familia afine), a revaluar solo si conditioned projections no mueve nada.

### Regla de ejecucion

1. Gate 5B ya no marca la ruta critica: queda como bloque cerrado.
2. Gate 5A corre cuando hay recursos libres.
3. Gate 5A no bloquea la transicion a Escalon 2.

Documentacion: `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/10_GATE_5_LINEA_A_BARRIDO/README.md`

## 9.2 Gate 5 Linea B — Showcase Cientifico

Best model → train largo → bateria de 13 tests cientificos ordenados por relevancia para la tesis Phideus.

Estado operativo (2026-03-01):
- Test12 (scoreboard canónico) **cerrado**:
  - `D0=73.4%`, `d4a4=83.8%`, `a4r=82.0%`, `d4-a4r=79.8%`.
- Test01 (causal ablation) **cerrado**:
  - A4/A4r causal dominante (caídas grandes al ablacionar audio descriptor).
  - D4 marginal/casi nulo en duales (`d4a4`, `d4-a4r`).
  - Incluye corrida `d4` puro (efecto pequeño, no causal robusto en inferencia).
- Test04 (transposition) **cerrado**.
- Test03 (ratio probe), Test06 (RSA/CKA), Test08 (ratio decoding) y Test10 (visualizaciones) **cerrados**.
- Test09 (invariance suite) **cerrado** en `D0`, `d4a4`, `a4r`, `d4-a4r`:
  - temporal shift: robustez aceptable (peor caso entre `-3.6pp` y `-7.2pp`);
  - velocity scaling y octave transposition: fragilidad alta en todos los arms;
  - audio noise: patrón bimodal (`D0` domina en 40-20 dB; `a4r/d4-a4r` retienen mejor en 5 dB).
- Estado UNC / sync local:
  - Test05 (multi-seed) **cerrado en repo**: `15/15` corridas disponibles en `results_unc` para `D0`, `a4r` y `d4-a4r`.
  - Lectura multi-seed vigente:
    - `d4a4 = 84.1%±2.3pp` (referencia multi-seed ya cerrada),
    - `d4-a4r = 81.2%±2.5pp`,
    - `a4r = 80.7%±1.9pp`,
    - `D0 = 75.2%±2.3pp`.
  - Test02 (parameter-matched) **cerrado 4/4**:
    - `real = 83.0%`,
    - `zero = 75.0%`,
    - `random = 73.6%`,
    - `shuffled = 73.6%*`.
    Con exactamente los mismos parámetros entrenables, las ablaciones sin descriptor real caen a banda `D0`: la mejora de `d4a4` es causal.
- Test 11 Pre-Proj A/B (completo): diagnóstico de bottleneck z→256d vs encoder fundamental. Resultado principal:
  - `D0` retention ratio `0.597`;
  - `a4r` retention ratio `0.712` (**+19% relativo**);
  - la proyeccion MIDI 512→256 destruye aproximadamente `81-88%` de la informacion condicionante.
- **Test 13G-A (Phase A cerrada)**: Generative Encoder Training — primer test que modifica el encoder training con dual-objective (VICReg + reconstrucción piano-roll). Resultado observacional:
  - `best_S≈64.4-64.6%`,
  - `audio_f1≈0.114`,
  - `midi_f1≈0.118`,
  - `λ` irrelevante en `0.03/0.1/0.3`.
  Inferencia operativa: la limitación está en la compresión a `z=256`, no en elegir otro `λ`. Las `Phase B/C` originales quedan canceladas.
- **Test 13G-B (cerrado)**: decoder post-hoc sobre features pre-pooling congeladas del encoder de audio.
  - Resultado: `D0 pool-188 = 0.1089`, `d4a4 = 0.1037`, `a4r = 0.1024`.
  - Lectura: la decodificabilidad pre-pooling es baja y genérica (`F1≈0.10` en todos); la ventaja descriptor-guided no aparece en esta tarea.
  - Guardrail interpretativo: la ventaja de descriptores permanece en la geometría de retrieval, no en una reconstrucción de piano-roll más nítida.

Tests imprescindibles para publicacion (top 5):
1. Causal ablation (zero-out injection)
2. Parameter-matched ablations (control de ruido)
3. RatioProbeDecoder + cross-decoding
4. Invariancia a transposicion MIDI
5. Multi-seed replication

Tests exploratorios de nueva frontera:
- Test 11 Pre-Proj A/B: bottleneck de proyección vs encoder
- **Test 13G**: `13G-A` como falsación del camino `z=256 -> PR`; `13G-B` como probing ya cerrado, útil para delimitar el límite generativo de las features pre-pooling

Documentacion: `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/README.md`

## 9.3 Gate 6 — AMT with Descriptor Conditioning

> **Nota historica**: Gate 6 originalmente fue diagnóstico RSA/CKA (2026-02). Esa fase fue absorbida por Gate 5B Test 06. Gate 6 se reasigna a AMT (Automatic Music Transcription).

**Pregunta central**: ¿La ventaja de los descriptores es puramente geométrica (solo retrieval) o se traduce a tareas musicales concretas?

**SOTA elegido**: Transkun v2 (12.9M params, F1=92.94% en MAESTRO v3, Semi-CRF, MIT license).

### Estado operativo al corte

| Bloque | Estado | Nota |
|--------|--------|------|
| `Exp 0` | **COMPLETO (LOCAL)** | baseline `Transkun` ya verificado sobre segmentos de `4s` y `16s` |
| `Exp C` | **ACTIVO** | corrida local `a4r` en curso (`best_F1=0.1485` @ `e35`) + array UNC reenviado como `job 1144560` tras fix de paths |
| `Exp A` | **LISTO PARA SUBMITIR** | `transkun` ya instalado en UNC; script validado para Mendieta |
| `Exp B` | **BLOQUEADO** | depende de validar `Exp A`; A4 siempre desde audio degradado |

### Hallazgo arquitectónico fijado

Transkun **no** usa “event tracks” como tokens independientes concatenados al estilo asumido inicialmente. El backbone real opera con tensores `[B, T, F+90, D]`, donde `90` corresponde a embeddings posicionales de `88` notas y `2` pedales en la dimensión de frecuencia. Por eso la inyección A4 quedó redefinida como:

1. extensión en la dimensión de frecuencia con `8` tracks A4, o
2. FiLM/adapters después de cada `BasicBlock`.

### Experimentos

| Exp | Pregunta | Método | Régimen | ETA |
|-----|----------|--------|---------|-----|
| **0** | ¿Transkun transcribe nuestros segmentos? | Inference pretrained | Ambos | **completo** |
| **A** | ¿A4 aporta info que SOTA no tiene? | Inyectar A4 en Transkun (tracks en frecuencia + FiLM) | 44.1kHz/16s | ~5 días UNC |
| **B** | ¿Más útil bajo degradación? | Transkun+A4 con ruido/filtrado | 44.1kHz/16s | ~4.5 días UNC |
| **C** | ¿Features VICReg decodifican música? | AMT decoder 38M sobre features congeladas | 24kHz/4s | ~16h UNC |

### Exp 0: baseline ya establecido

`Transkun v2` ya se verificó localmente sobre `100` segmentos de validación MAESTRO (`50x4s + 50x16s`):

| Régimen | note_onset_F1 | note+offset_F1 | note+offset+velocity_F1 | frame_F1 |
|---------|---------------|----------------|--------------------------|----------|
| `4s` | `0.938` | `0.667` | `0.607` | `0.784` |
| `16s` | `0.972` | `0.729` | `0.718` | `0.814` |

Lectura: el baseline es suficientemente sano como para usar `Transkun` como banco de pruebas de `Exp A/B`; no aparece una falla básica de setup que invalide la línea.

### Exp A: Configuraciones con control param-matched

| Config | Inyección | Freeze | Control |
|--------|-----------|--------|---------|
| `baseline` | ninguna | todo | — |
| `finetune-noA4` | 8 tracks constantes (=0) | base congelada | **control param-matched** |
| `A4-event` | 8 tracks A4 | base congelada | comparar vs finetune-noA4 |
| `A4-adapter` | FiLM por layer | base congelada | comparar vs adapter-noA4 |
| `adapter-noA4` | FiLM con input=0 | base congelada | **control param-matched** |

### Exp C: Arms (checkpoints Gate 5B congelados)

| Arm | Pre-pooling | Nota |
|-----|-------------|------|
| D0 | [B, 2400, 1024] | Baseline sin descriptor |
| d4a4 | [B, 2400, 1024] | Concat A4+D4 (checkpoint existente) |
| a4r | [B, 188, 1024] | Reverse cross-att A4 |
| d4-a4r | [B, 188, 1024] | Mixed reverse A4+D4 |

### Orden de ejecución

1. **Fase 0**: Setup + inspección Transkun (LOCAL) — **completada**
2. **Fase 1**: Exp 0 baseline verification (LOCAL) — **completada**
3. **Fase 2**: Exp C — AMT decoder (local + UNC, no requiere modificar Transkun) — **activo**
4. **Fase 3**: Exp A — Transkun+A4 (UNC) — **pendiente de entorno**
5. **Fase 4**: Exp B — Degraded (UNC) — **bloqueada por Exp A**

Documentacion:
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/12_GATE_6_AMT/README.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/12_GATE_6_AMT/Explicacion_gate6.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/12_GATE_6_AMT/Briefing_para_claude_unc.md`

---

## 10. Riesgos Tecnicos y Criterios de Corte

## 10.1 Riesgos activos

1. **No reproducir baseline en S0**.
   - Impacto: invalida comparabilidad.
   - Mitigacion: bloquear A/B/C hasta resolver.

2. **Catastrophic forgetting en unfreeze parcial**.
   - Impacto: caida adicional de A2M.
   - Mitigacion: LR audio bajo (1e-5), gate estricto, no escalar sin evidencia.

3. **Mejora unilateral (solo M2A)**.
   - Impacto: progreso falso por asimetria.
   - Mitigacion: usar `S=min(A2M, M2A)` como criterio principal.

4. **Overfitting narrativo por visualizaciones**.
   - Impacto: decisiones por evidencia cualitativa no robusta.
   - Mitigacion: visualizaciones como soporte, no como criterio primario.

5. **Ejecutar screening de Gate 4.2 antes de foundation lock**.
   - Impacto: invalida comparabilidad causal entre descriptores.
   - Mitigacion: bloquear screening hasta cierre formal A/B/C/D/D-02 + freeze policy definitiva.

## 9.4 Gate 7 — MERT-large Linear Probe

> **Fecha apertura**: 2026-03-05. **Estado**: IMPLEMENTADO — pendiente ejecución.

**Motivación**: Gate 6 Exp C plateau (F1≈0.157) compatible con techo del encoder. Gate 5B mostró que la ventaja de A4 es geométrica, no de feature richness. Queda la ambigüedad: ¿es la limitación el encoder, el objetivo de entrenamiento, o A4 es genuinamente complementario para encoders más fuertes?

**Pregunta central**: ¿Cuánto de A4 está linealmente accesible en MERTLite-D0 vs MERT-v1-95M vs MERT-v1-330M?

**IMPORTANTE**: Solo reduce la ambigüedad del lado audio. Gate 7 NO resuelve sola la cuestión cross-modal.

| Encoder | Params | Origen |
|---------|--------|--------|
| MERTLite-D0 | ~60M | Entrenado VICReg MAESTRO (régimen cross-modal) |
| MERT-v1-95M | ~95M | HF foundation model, sin régimen cross-modal |
| MERT-v1-330M | ~330M | HF foundation model, **test principal** |

**Protocolo**: Ridge regression cerrado, 5 group splits por pieza (80/20), 8 bandas A4 + global, nulls (shuffled_between + dummy).

**Exp 7.0b** (opcional): curva R² vs layer depth en MERT-330M.

**Exp 7.1** (diferida): mini Test02 con MERT-large. Se diseña solo post Exp 7.0 según patrón.

**Documentación**: `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/13_GATE_7_MERT_PROBE/README.md`

---

## 10.2 Criterios de corte global

- Si Bloque A no supera control y baseline con evidencia robusta, cerrar rama y re-evaluar estrategia.
- No reabrir hipotesis H4.2-6 sin nueva evidencia fuerte independiente.

## 10.3 Protocolo anti-variable-fantasma (obligatorio)

Para evitar repetir errores estructurales (como descubrir tarde que un modulo clave estaba congelado), cada ola de entrenamiento debe pasar este checklist:

1. **Inventario pre-run de trainables**
   - reporte por modulo: parametros totales, trainables y frozen.
2. **Drift sanity tras run corto**
   - verificar que los modulos que debian moverse efectivamente muestran drift > 0.
3. **Gate de reproducibilidad**
   - validar S0/control antes de escalar A/B/C o fases largas.
4. **Comparabilidad estricta**
   - misma configuracion canonica (`pool=256`, `queries=500`, `seed=42`).
5. **Trazabilidad inmediata**
   - registrar en bitacora + roadmap cualquier desvio detectado y su impacto sobre decisiones.

---

## 11. Artefactos de Verdad

## 11.1 Baseline y evaluaciones

- `data/bias_control_medium/training_outputs/gate2/checkpoint_epoch45.pt`
- `data/bias_control_medium/evaluations/structured_pool_epoch45.json`
- `data/bias_control_medium/training_outputs/bloqueA_runA/checkpoint_epoch5.pt`
- `data/bias_control_medium/training_outputs/bloqueA_runA/final_results.json`
- `data/bias_control_medium/training_outputs/bloqueA_runA/eval_per_epoch/eval_epoch5.json`
- `data/bias_control_medium/training_outputs/bloqueA_runD/eval_per_epoch/eval_epoch5.json`
- `data/bias_control_medium/training_outputs/bloqueA_runD-02/eval_per_epoch/eval_epoch25.json`
- `data/bias_control_medium/training_outputs/bloqueA_runD-02/training.log`
- `data/bias_control_medium/training_outputs/bloqueA_runD-02/final_results.json`
- `data/bias_control_medium/training_outputs/bloqueA_runD-02/multiseed_reeval.json`
- `data/bias_control_medium/training_outputs/foundation_locked_e25.pt`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/resultados_compartir/explore_summary.json`
- `data/bias_control_medium/evaluations/gate4/RA5_ep5.json`
- `data/bias_control_medium/evaluations/gate4/RB0_ep5.json`
- `data/bias_control_medium/evaluations/gate4/R1rescue_ep5.json`

## 11.2 Diagnostico post Gate 4.1

- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/04_DIAGNOSTICO_GATE_6_Y_GATE_4_2/INFORME_DEC005_DIAGNOSTICO_COMPLETO.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/04_DIAGNOSTICO_GATE_6_Y_GATE_4_2/CURADURIA_VISUAL/INDEX_VISUAL.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/04_DIAGNOSTICO_GATE_6_Y_GATE_4_2/CURADURIA_VISUAL/SNAPSHOT_DEC005.md`
- `data/bias_control_medium/evaluations/gate6/layer_drift.json`
- `data/bias_control_medium/evaluations/gate6/hubness_analysis.json`
- `data/bias_control_medium/evaluations/gate42/h426_prered_results.json`

## 11.3 Plan operativo vigente

- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/05_PLAN_POST_DIAGNOSTICO_BLOQUE_A/PLAN_EJECUCION_POST_DEC005_v1.1.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/05_PLAN_POST_DIAGNOSTICO_BLOQUE_A/PLAN_EJECUCION_POST_DEC005_CODEX.md` (historial v1.0)
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/06_GATE_4_2_RATIO_CENTRICO/plan_gate_4.2.md` (plan final Gate 4.2, version ratio-centrica)
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/07_GATE_4_3_RATIO_RE_CENTRICO/plan_gate_4.3.md` (bloque causal bifurcado)
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/08_GATE_4_4_ARQUITECTURAS_MAYORES/README.md` (third tower + FiLM + MoE)
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/09_GATE_4_5_LR_SCHEDULE_OPTIMIZATION/README.md` (optimizacion de scheduler y ventana temporal)
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/10_GATE_5_LINEA_A_BARRIDO/README.md` (replanteo Gate 5A: conditioned projections + combinatorios oportunistas)
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/README.md` (13+ tests cientificos)
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/12_GATE_6_AMT/README.md` (Gate 6 AMT: validación downstream)
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/12_GATE_6_AMT/Explicacion_gate6.md` (lectura narrativa del nuevo frente)
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/12_GATE_6_AMT/Briefing_para_claude_unc.md` (briefing operativo para UNC)
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/13_GATE_7_MERT_PROBE/README.md` (Gate 7: MERT-large linear probe)
- `experiments/bias_control/gate7/mert_large_feature_extractor.py` (HF MERT wrapper)
- `experiments/bias_control/gate7/mert_large_probe.py` (script principal: probe + nulls + plots)
- `experiments/bias_control/slurm/gate7_mert_probe.sh` (SLURM para UNC)
- `experiments/bias_control/gate5b/test13g_generative_encoder.py` (Test 13G: dual-objective generative encoder)
- `experiments/bias_control/gate5b/test13g_posthoc_decoder.py` (Test 13G-B: decoder post-hoc pre-pooling)
- `experiments/bias_control/gate5b/test11_preproj_ab_test.py` (Test 11 Pre-Proj A/B)
- `experiments/bias_control/gate6/README.md` (overview operativo de scripts Gate 6)
- `data/gate5b_results/d0/test13g/pr_validation_gate.json` (gate de validación PR targets)
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/Explicacion_test_13G_faseB.md` (lectura metodológica de la nueva fase generativa)
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/INFORME_COMPLETO_GATE5B.md` (informe exhaustivo del corte Gate 5B)
- `README.md` (entrada principal + links de visualizaciones 3D de arquitectura)

---

## Cierre

Este roadmap queda actualizado al corte operativo 2026-03-05 (Gate 5B completamente cerrado; Gate 6 activo; Gate 7 implementado y pendiente ejecución).

Foco inmediato:
1. Tratar `Test05` como cierre estadístico y `Test02` como cierre causal ya consolidados.
2. Mantener `Test11` como hallazgo mecanístico principal del frente.
3. Leer Gate 6 AMT como validación downstream activa: `Exp 0` completo, `Exp C` corriendo en local y reenviado en UNC, `Exp A` listo, `Exp B` bloqueado por `Exp A`.
3. Leer `13G-B` como cierre negativo útil de la línea generativa, no como soporte para una claim descriptor-guided.
4. Abrir Escalón 2 como foco principal, con Gate 5A limitado a ventanas oportunistas.
5. Mantener sincronía documental entre troncal, frente y transversales.
