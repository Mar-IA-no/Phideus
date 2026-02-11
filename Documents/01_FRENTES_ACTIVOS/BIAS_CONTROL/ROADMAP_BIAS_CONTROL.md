<div align="center">

# Roadmap BIAS_CONTROL
### Escalon 1-C de Phideus: de baseline cross-modal a diagnostico causal y plan de recuperacion

![Version](https://img.shields.io/badge/Version-2.1-111827?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-MAESTRO_v3.0.0-1F6FEB?style=for-the-badge)
![Fase](https://img.shields.io/badge/Fase-Escalon_1--C-F59E0B?style=for-the-badge)
![Estado](https://img.shields.io/badge/Estado-Bloque_A_Run_A_INCONCLUSO-0A7E3B?style=for-the-badge)

</div>

> [!IMPORTANT]
> **Fecha de corte**: 2026-02-11  
> **Estado del programa**: Gate 4.1 cerrado, diagnostico post Gate 4.1 completado, `S0` y `Run A` de Bloque A v1.1 completados (Run A clasificado INCONCLUSO).  
> **Siguiente paso operativo**: ejecutar `Run B` y luego `Run C` bajo protocolo canonico identico.  
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
- [8. Gate 5 y Gate 6: Estado en el Roadmap](#8-gate-5-y-gate-6-estado-en-el-roadmap)
- [9. Riesgos Tecnicos y Criterios de Corte](#9-riesgos-tecnicos-y-criterios-de-corte)
- [10. Artefactos de Verdad](#10-artefactos-de-verdad)

---

## Mapa Documental

- Índice de fase/documentos: `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/INDEX_BIAS_CONTROL.md`
- Carpeta espejo local para compartir visuales: `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/resultados_compartir/` (no versionada)

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

**Abierto**:
- Nueva ola de experimentos post-diagnostico: Bloque A (adapter/unfreezing controlado), ya definido en `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/05_PLAN_POST_DIAGNOSTICO_BLOQUE_A/PLAN_EJECUCION_POST_DEC005_v1.1.md`.

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
5. La siguiente iteracion debe atacar esa causa con control experimental estricto (plan v1.1 aprobado).

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
- Regimen sostenido (Run D): inferior al baseline

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
| Run B (partial unfreeze) | Pendiente | - | - | - | - | Siguiente |
| Run C (hybrid) | Pendiente | - | - | - | - | Luego de Run B |

Notas:
1. Run A tuvo interrupcion por caida de servidor durante epoch 5 y se completo con resume desde `checkpoint_epoch4`.
2. La traza canonica por epoca quedo en `eval_per_epoch/eval_epoch1..5.json`.
3. La decision de Bloque A no se cierra hasta completar B y C con el mismo protocolo.

## 7.2 Secuencia de Bloque A

1. `Run S0` (eval-only): reproducir baseline Gate 2 sin entrenamiento.
2. `Run A` (adapters con audio base congelado).
3. `Run B` (partial unfreeze de capas altas de audio transformer).
4. `Run C` (hibrido: adapters en capas bajas + unfreeze capas altas).

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

---

## 8. Gate 5 y Gate 6: Estado en el Roadmap

## 8.1 Gate 5

Permanece opcional y no bloquea el cierre tecnico de Escalon 1-C.
Se mantiene en backlog condicionado a evidencia futura.

## 8.2 Gate 6

En este roadmap, Gate 6 ya tuvo una fase diagnostica ejecutada dentro de la Decision de diagnostico post Gate 4.1.

Punto clave:
- Si se necesita una fase 2 de Gate 6 (RSA/CKA/probes extendidos), debe abrirse como decision separada con alcance y costo definidos.

---

## 9. Riesgos Tecnicos y Criterios de Corte

## 9.1 Riesgos activos

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

## 9.2 Criterios de corte global

- Si Bloque A no supera control y baseline con evidencia robusta, cerrar rama y re-evaluar estrategia.
- No reabrir hipotesis H4.2-6 sin nueva evidencia fuerte independiente.

## 9.3 Protocolo anti-variable-fantasma (obligatorio)

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

## 10. Artefactos de Verdad

## 10.1 Baseline y evaluaciones

- `data/bias_control_medium/training_outputs/gate2/checkpoint_epoch45.pt`
- `data/bias_control_medium/evaluations/structured_pool_epoch45.json`
- `data/bias_control_medium/training_outputs/bloqueA_runA/checkpoint_epoch5.pt`
- `data/bias_control_medium/training_outputs/bloqueA_runA/final_results.json`
- `data/bias_control_medium/training_outputs/bloqueA_runA/eval_per_epoch/eval_epoch5.json`
- `data/bias_control_medium/evaluations/gate4/RA5_ep5.json`
- `data/bias_control_medium/evaluations/gate4/RB0_ep5.json`
- `data/bias_control_medium/evaluations/gate4/R1rescue_ep5.json`

## 10.2 Diagnostico post Gate 4.1

- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/04_DIAGNOSTICO_GATE_6_Y_GATE_4_2/INFORME_DEC005_DIAGNOSTICO_COMPLETO.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/04_DIAGNOSTICO_GATE_6_Y_GATE_4_2/CURADURIA_VISUAL/INDEX_VISUAL.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/04_DIAGNOSTICO_GATE_6_Y_GATE_4_2/CURADURIA_VISUAL/SNAPSHOT_DEC005.md`
- `data/bias_control_medium/evaluations/gate6/layer_drift.json`
- `data/bias_control_medium/evaluations/gate6/hubness_analysis.json`
- `data/bias_control_medium/evaluations/gate42/h426_prered_results.json`

## 10.3 Plan operativo vigente

- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/05_PLAN_POST_DIAGNOSTICO_BLOQUE_A/PLAN_EJECUCION_POST_DEC005_v1.1.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/05_PLAN_POST_DIAGNOSTICO_BLOQUE_A/PLAN_EJECUCION_POST_DEC005_CODEX.md` (historial v1.0)

---

## Cierre

Este roadmap queda actualizado como documento troncal de BIAS_CONTROL para el estado actual del proyecto:
- historial tecnico completo,
- decisiones cerradas,
- diagnostico causal ejecutado,
- plan de siguiente ola aprobado con criterios de corte reproducibles.

El foco ahora ya no es "probar mas variantes" sin control, sino ejecutar Bloque A con disciplina experimental y comparabilidad estricta.
