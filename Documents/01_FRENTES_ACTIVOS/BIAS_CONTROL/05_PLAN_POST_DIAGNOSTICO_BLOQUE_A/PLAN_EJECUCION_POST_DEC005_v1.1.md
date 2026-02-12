# Plan de Ejecucion Post Diagnostico de BIAS_CONTROL — v1.1

Version: 1.1 (consolidada Claude + Codex, con addendum operativo 2026-02-12)
Fecha: 2026-02-11
Estado: aprobado por usuario, en ejecucion (S0/Run A/Run B completados; Run C en curso; Run D condicional)
Base: `PLAN_EJECUCION_POST_DEC005_CODEX.md` (v1.0 de Codex) + 4 ajustes validados en COLLAB/DIALOGUE.md

---

## 1) Objetivo

Cerrar BIAS_CONTROL con evidencia causal sobre si el adapter/unfreezing controlado del audio encoder recupera rendimiento cross-modal. Evitar iteraciones largas sin senal.

## 2) Estado de partida

**Baseline oficial:** Gate 2 epoch 45 (structured pool canonico 256/500/seed42)
- A2M R@10: 34.4%
- M2A R@10: 37.6%
- hard_neg_acc: 80.4%
- S = min(A2M, M2A) = 34.4%

**Hallazgo del diagnostico post Gate 4.1 (Retroanalisis de degradacion por fine-tuning):** Audio encoder completamente congelado en fine-tuning. Solo MIDI + projections cambian. Esto causa degradacion de A2M y alargamiento de puentes cross-modal.

### 2.1) Avance de ejecucion (corte actual)

| Etapa | Estado | A2M R@10 | M2A R@10 | hard_neg | S=min(A2M,M2A) | Decision |
|---|---|---:|---:|---:|---:|---|
| S0 (control) | Completado | 34.4% | 37.6% | 80.4% | 34.4% | Control valido |
| Run A (adapter) | Completado | 30.0% | 38.6% | 76.8% | 30.0% | INCONCLUSO |
| Run B (partial unfreeze) | Completado | 43.2% | 43.4% | 85.2% | 43.2% | Mejor checkpoint provisional (ep3) |
| Run C (hybrid) | En curso | 35.0%* | 38.2%* | 79.6%* | 35.0%* | Pendiente cierre 5 ep |

Notas:
1. Run A se interrumpio por caida de servidor en epoch 5 y se completo con resume desde `checkpoint_epoch4`.
2. `training_history.json` del resume refleja solo epoch 5; la serie completa por epoca esta en `eval_per_epoch/eval_epoch1..5.json`.
3. `*` indica corte parcial al cierre de epoch 2 de Run C.
4. La decision de Bloque A se cierra recien despues de comparar A/B/C bajo el mismo protocolo.
5. Segun DEC-007, `Run D` es condicional y no bloquea implementacion de codigo Gate 4.2.

### 2.2) Addendum operativo (2026-02-12)

1. Orden de decision cientifica:
   - Cerrar Run C -> comparativa A/B/C -> ejecutar Run D solo si aplica -> foundation lock final.
2. Gate 4.2:
   - Implementacion de codigo permitida en paralelo.
   - Screening bloqueado hasta foundation definitivo.
3. Gate2R-lite:
   - Se agenda como higiene metodologica post Gate 4.2 (no bloqueante para la pregunta causal ratio vs control).

## 3) Arquitectura del modelo

| Componente | Tipo | Capas | Params | Dim |
|---|---|---|---|---|
| Audio encoder CNN | Conv1d x4 | 4 | ~3.2M | 1024 |
| Audio encoder Transformer | TransformerEncoder | 4 | ~50.4M | 1024 |
| Audio PosEmbedding | Learnable | 1 | ~6.1M | 1024 |
| Audio Projection | MLP | 3 | ~0.9M | 1024->512->256 |
| MIDI Embedding | Event embed + norm | — | ~0.3M | 512 |
| MIDI Transformer | TransformerEncoder | 4 | ~12.6M | 512 |
| MIDI Projection | MLP | 3 | ~0.7M | 512->512->256 |
| **Total** | | | **~74.2M** | |

## 4) Bloque A — Adapter/Unfreezing controlado

### Secuencia de ejecucion

```
Run S0 (eval-only)  ->  Run A (adapter)  ->  Run B (partial unfreeze)  ->  Run C (hybrid)  ->  Run D (condicional)
      |                      |                      |                         |                         |
 Confirmar baseline     5 epochs              5 epochs                  5 epochs                  5 epochs
 sin training           screening             screening                 screening                 solo si aplica
```

### Run S0 — Control de reproducibilidad (eval-only)

- **Accion:** Evaluar `checkpoint_epoch45` con structured pool canonico. SIN ENTRENAMIENTO.
- **Objetivo:** Confirmar que las metricas reportadas de Gate 2 son reproducibles bajo el evaluador actual.
- **Criterio bloqueante:** Si difiere >1pp del baseline reportado, investigar evaluador antes de continuar.
- **Tiempo estimado:** ~5 minutos.

### Run A — Adapter bottleneck

- **Que se modifica:** Se insertan 4 adapter bottleneck (1 por capa transformer del audio encoder).
- **Adapter architecture:** Linear(1024 -> 64) -> GELU -> Linear(64 -> 1024) + conexion residual.
- **Parametros nuevos:** ~131K por adapter, ~524K total (0.9% del audio encoder).
- **Que queda frozen:** Todo el audio encoder original (CNN + Transformer + PosEmb). Los adapters se agregan como modulos nuevos.
- **`adapter_dim`:** Configurable. Default 64, escalable a 128 si screening inconcluso.

| Grupo de parametros | LR | Justificacion |
|---|---|---|
| Adapters audio (nuevos) | 5e-4 | Random init, necesitan aprender rapido |
| MIDI encoder | 5e-5 | Misma que Gate 4 |
| Audio projection | 1e-4 | Misma que Gate 4 |
| MIDI projection | 1e-4 | Misma que Gate 4 |

- **Epochs:** 5 (screening). Sin early stopping. Correr las 5 completas.
- **Checkpoints:** Guardar todas las epochs (checkpoint_epoch1..5).
- **Evaluacion:** Mejor epoch + ultimo epoch con structured pool canonico.

### Run B — Partial unfreeze (ultimas 2 capas audio transformer)

- **Que se descongela:** `audio_encoder.transformer.layers[2]` y `audio_encoder.transformer.layers[3]`.
- **Que queda frozen:** CNN, PosEmb, capas transformer 0-1.
- **Parametros desbloqueados:** ~25M (50% del audio transformer).

| Grupo de parametros | LR | Justificacion |
|---|---|---|
| Audio transformer capas 2-3 | 1e-5 | 10x mas bajo que MIDI para evitar catastrophic forgetting |
| MIDI encoder | 5e-5 | Misma que Gate 4 |
| Audio projection | 1e-4 | Misma que Gate 4 |
| MIDI projection | 1e-4 | Misma que Gate 4 |

- **Epochs:** 5 (screening). Sin early stopping.
- **Checkpoints:** Guardar todas las epochs.

### Run C — Hibrido (adapters capas 0-1 + unfreeze capas 2-3)

- **Que se descongela:** Capas transformer 2-3 completas.
- **Que se agrega:** Adapters bottleneck en capas 0-1 (misma arch que Run A).
- **Que queda frozen:** CNN, PosEmb, capas transformer 0-1 (pesos originales).

| Grupo de parametros | LR | Justificacion |
|---|---|---|
| Adapters capas 0-1 (nuevos) | 5e-4 | Random init |
| Audio transformer capas 2-3 | 1e-5 | Proteger de forgetting |
| MIDI encoder | 5e-5 | Misma que Gate 4 |
| Audio projection | 1e-4 | Misma que Gate 4 |
| MIDI projection | 1e-4 | Misma que Gate 4 |

- **Epochs:** 5 (screening). Sin early stopping.
- **Checkpoints:** Guardar todas las epochs.

## 5) Criterios de gate (screening a 5 epochs)

### Nivel 1 — Relativo (comparacion contra control S0)

**Definicion de variables de control:** `S_control = S_S0` y `hard_neg_control = hard_neg_S0` (valores obtenidos de la evaluacion eval-only del Run S0).

| Decision | Criterio |
|---|---|
| **SCALE** | `S_run - S_control >= +1.5pp` AND `hard_neg_run >= hard_neg_control - 1pp` |
| **GO fuerte** | SCALE + `S_run >= 33.4%` |
| **DROP** | `S_run < 30%` OR `hard_neg_run < 75%` |
| **INCONCLUSO** | Todo lo intermedio. No escalar automaticamente; decidir con Codex. |

### Nivel 2 — Absoluto (objetivo de negocio, solo para fase larga)

| Decision | Criterio |
|---|---|
| **EXITO** | `S > 34.4%` AND `hard_neg_acc >= 80.4%` AND `bridge_dist < 3.27` |
| **MEJORA parcial** | Supera S0 pero no alcanza baseline completo |
| **FRACASO** | No supera S0 tras 15-30 epochs |

## 6) Regimen de comparabilidad (obligatorio)

No se acepta resultado fuera de este protocolo:
- pool_size = 256
- n_queries = 500
- seed = 42
- Mismo split que Gate 2
- Mismas metricas primarias

**Metricas primarias:** A2M R@10, M2A R@10, hard_neg_acc, S = min(A2M, M2A)
**Metricas secundarias:** MRR A2M/M2A, mean rank, separacion, bridge distance

## 7) Bloque B — Gate 4.2 ratio-centrico (post-foundation lock)

Gate 4.2 no depende de que Bloque A sea "inconcluso". Se ejecuta como etapa siguiente una vez cerrado foundation lock.

Reglas:
1. Implementacion de codigo habilitada en paralelo al cierre de Bloque A.
2. Screening bloqueado hasta foundation definitivo (A/B/C y Run D condicional si aplica).
3. Comparabilidad estricta con protocolo canonico (`pool=256`, `queries=500`, `seed=42`).
4. Clausula anti-goalpost vigente: decisiones de promotion/confirmacion segun umbrales pre-registrados de `S` y `hard_neg`.

## 8) Bloque C — Paquete visual y generativo

### C1. Retrievals pareados para escucha

1. 10 segmentos fijos (seed=42): 2 faciles, 6 medios, 2 dificiles (segun recall per-piece de Gate 2).
2. Top-3 retrievals A2M y M2A por modelo evaluado.
3. MIDI retrieved renderizado a audio via FluidSynth (SoundFont TimGM6mb).
4. Paquete: `{query_audio, gt_midi_rendered, retrieved_top1..top3}` + manifiesto JSON.
5. **Mismos 10 segmentos para todos los modelos.** Manifiesto unico con orden estable para comparacion A/B ciega.

### C2. UMAP comparativo

1. Reutilizar scripts existentes (`extract_multigate_embeddings.py`, `visualize_embeddings_multigate.py`).
2. Extender a checkpoints S0, A, B, C.
3. Bridges + similarity distributions por modelo.

### C3. Tabla comparativa final

1. Todas las metricas primarias y secundarias por run.
2. Columna delta vs Gate 2.
3. Columna de decision (KEEP/SCALE/DROP).

## 9) ETA estimado

| Fase | Tiempo | Notas |
|---|---|---|
| Implementar adapters + training loop | 2-3 horas | Nuevo modulo `adapter.py` + patch en training |
| Run S0 (eval-only) | ~5 min | Solo evaluacion |
| Run A (5 epochs, 1000 batches/epoch) | ~100 min | Adapters overhead minimo |
| Run B (5 epochs) | ~90 min | Sin overhead adicional |
| Run C (5 epochs) | ~100 min | Adapters + unfreeze |
| Evaluacion structured pool x3 | ~10 min | A, B, C |
| Decision screening | 10 min | Tabla + gate |
| **Subtotal screening** | **~8-9 horas** | Implementacion + 3 runs + eval |
| Escalado del ganador (15-30 epochs) | ~5-10 horas | Solo si hay ganador claro |
| Bloque C artefactos | ~2-3 horas | Retrievals + UMAP + tabla |
| **Total completo** | **~15-22 horas** | Todo incluido |

## 10) Entregables por ciclo

Por cada run:
1. JSON de resultados structured pool.
2. Resumen corto en docs curados.
3. Decision explicita: KEEP / SCALE / DROP.

Al cierre del bloque:
1. Tabla comparativa final contra Gate 2.
2. Recomendacion de continuidad (seguir, cerrar rama, o rediseniar).

## 11) Gobernanza

- Este plan no habilita entrenamientos automaticamente; define secuencia y criterios.
- Toda ejecucion requiere aprobacion explicita del usuario.
- Collab mode: se usara para validacion inter-agente de resultados.
- Se guarda checkpoint en todas las epochs por default (comportamiento del training loop actual).

---

> Historial: v1.0 (Codex, 2026-02-11) -> v1.1 (Claude + Codex, 2026-02-11)
> Cambios v1.0 -> v1.1: S0 eval-only, gate 2 niveles, sin early-stop screening, ETA realista, adapter_dim configurable.
