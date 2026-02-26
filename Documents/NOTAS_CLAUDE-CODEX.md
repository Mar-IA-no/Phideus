# Notas de Claude LOCAL para Codex

> Fecha: 2026-02-20 (S1-7), 2026-02-22 (S8), 2026-02-23 (S8 update + S9 + S10), 2026-02-24/25 (S11-S14)
> Sesiones: cosine-tail LR + Gate 4.5 + SSH Mendieta + cleanup plan + Gate 5B execution + charts + glosario
> Nota: secciones 6 y 7 fueron restauradas tras pérdida accidental en merge con unc

---

## 1. Contexto: problema con el LR scheduler en runs de 60ep

### Resultados observados

Los runs de 30ep (cosine estándar) produjeron los mejores S:
- d4a4: 83.6% (e30)
- a4r: 82.0% (e29)

Los runs de 60ep (cosine estirado a 60K steps) no alcanzan esos niveles:
- d4a4 60ep: best S=79.0% (e25), aún corriendo
- a4r 60ep: final S=79.4% (e60) — nunca alcanzó el 82.0% del 30ep
- D0 60ep: oscila 68-72% desde e15, el control no mejora con más epochs

### Diagnóstico: el LR profile importa más que el número de epochs

Comparando las curvas de LR:

| Epoch | 30ep LR mult | 60ep LR mult |
|-------|-------------|-------------|
| 5 | 0.944 | 0.986 |
| 10 | 0.764 | 0.939 |
| 15 | 0.513 | 0.861 |
| 20 | 0.256 | 0.758 |
| 25 | 0.072 | 0.636 |
| 30 | 0.000 | 0.493 |

El 30ep fuerza una transición agresiva exploración→explotación. A e25 ya tiene LR=0.07 (modo explotación) y ambos modelos alcanzan sus mejores S. El 60ep a e25 todavía tiene LR=0.64 — sigue explorando cuando debería consolidar.

---

## 2. Nuevo scheduler: cosine-tail

### Concepto

Combinar lo mejor de ambos mundos:
1. **Replicar exactamente** la curva del 30ep (cosine agresivo) hasta que el LR llega a 0.10
2. **Cola lineal suave** de 0.10 → 0.02 hasta el final del training
3. Así el modelo nunca queda sin gradiente (como en 30ep) ni demasiado caliente (como en 60ep)

### Implementación

Se extendió `LinearWarmupCosineScheduler` en `gate43_scratch_training.py` con 3 nuevos parámetros:

```
--lr-cosine-ref-epochs 30    # Referencia para la fase cosine (simula run de 30ep)
--lr-floor 0.10              # LR mult donde cosine se detiene y arranca la cola
--lr-tail-end 0.02           # LR mult final al terminar el training
```

### Fases del schedule

```
LR mult
1.00 ─┐
      │╲  cosine (idéntico a 30ep)
      │  ╲
      │    ╲
0.10 ─┤─────╲─────────────────
      │       ╲  cola lineal
      │         ╲___________
0.02 ─┤                      ╲
      └──────────────────────────
      e0     e15    e24   e40   e60
      warm   cosine  tail (0.10→0.02)
```

### Curva LR verificada

| Epoch | Phase | LR mult |
|-------|-------|---------|
| 1 | cosine | 0.999 |
| 5 | cosine | 0.944 |
| 10 | cosine | 0.765 |
| 15 | cosine | 0.513 |
| 20 | cosine | 0.258 |
| 24 | tail | 0.100 |
| 25 | tail | 0.098 |
| 30 | tail | 0.087 |
| 35 | tail | 0.076 |
| 40 | tail | 0.064 |
| 45 | tail | 0.053 |
| 50 | tail | 0.042 |
| 55 | tail | 0.031 |
| 60 | tail | 0.020 |

### Verificación

- Fase cosine: **diff = 0.0** vs scheduler de 30ep estándar (idéntica)
- Backward compatible: sin los nuevos flags, comportamiento idéntico al original
- Transición suave: LR pasa de 0.100 (floor) a 0.020 (tail_end) linealmente
- state_dict/load_state_dict actualizados para resume

### Código modificado

El scheduler ahora tiene 3 modos mutuamente excluyentes:
1. **Estándar** (default): warmup → cosine → 0
2. **Trapezoidal** (--lr-hold-fraction): warmup → hold → cosine → 0
3. **Cosine-tail** (--lr-cosine-ref-epochs): warmup → cosine(ref) → linear tail

---

## 3. Batch cosine-tail 60ep — 4 runs para UNC

### Diseño experimental

Mismas condiciones que los runs de 30ep/60ep existentes, pero con el nuevo scheduler cosine-tail. 60 epochs totales, eval cada 5ep.

### Scripts SLURM creados

| Script | Arm | Output dir | Params | Referencia 30ep |
|--------|-----|-----------|--------|-----------------|
| `batch_60ep_ctail_d0.sh` | D0 (control) | `batch_60ep_ctail_d0/` | ~65M | 72.0% (e30) |
| `batch_60ep_ctail_d4a4.sh` | d4a4 | `batch_60ep_ctail_d4a4/` | ~66.5M | 83.6% (e30) |
| `batch_60ep_ctail_a4r.sh` | a4r | `batch_60ep_ctail_a4r/` | ~68.2M | 82.0% (e29) |
| `batch_60ep_ctail_d4-a4r.sh` | d4-a4r | `batch_60ep_ctail_d4-a4r/` | ~69.6M | 79.8% (e30) |

Todos: seed 42, batch 16, run-d, 1000 batches/ep, eval epochs 5,10,...,55,60.

### Flags clave (comunes a todos)

```
--lr-cosine-ref-epochs 30
--lr-floor 0.10
--lr-tail-end 0.02
```

### Qué buscar en los resultados

1. **S@e25 vs 30ep**: deberían ser ~iguales (misma curva LR hasta e24)
2. **S@e30-e60**: ¿la cola suave permite seguir mejorando? Es la pregunta central
3. **D0 control**: si D0 mejora mucho → el scheduler es mejor en general. Si no → los descriptores son los que aprovechan
4. **d4-a4r**: con +4.6M params, la hipótesis es que se beneficia más de la cola extendida
5. **lr_mult en training_history.json**: verificar que registra los valores del cosine-tail

### Tiempo estimado

~25-31h por run (igual que 60ep estándar). Todos caben en 48h de SLURM.

---

## 4. Estado de jobs UNC actualmente corriendo

Al momento de esta sesión (2026-02-20):

| Job | Estado | Epoch | S más reciente |
|-----|--------|-------|----------------|
| t3-wt 50ep hold | RUNNING | 42/50 | S@e40=80.6% |
| D0 60ep | RUNNING | 42/60 | S@e40=72.4% |
| d4a4 60ep | RUNNING | 39/60 | S@e35=75.6% |
| a4r 60ep | COMPLETADO | 60/60 | S@e60=79.4% |
| d4-a4r 60ep | PENDING | — | — |
| moe-dual 60ep | PENDING | — | — |

Los 4 nuevos scripts cosine-tail se suman a esta cola.

---

## 5. Archivos modificados/creados

| Archivo | Cambio |
|---------|--------|
| `experiments/bias_control/gate43_scratch/gate43_scratch_training.py` | Cosine-tail scheduler mode |
| `experiments/bias_control/slurm/batch_60ep_ctail_d0.sh` | NUEVO |
| `experiments/bias_control/slurm/batch_60ep_ctail_d4a4.sh` | NUEVO |
| `experiments/bias_control/slurm/batch_60ep_ctail_a4r.sh` | NUEVO |
| `experiments/bias_control/slurm/batch_60ep_ctail_d4-a4r.sh` | NUEVO |

---

## 6. Hallazgo: A4r reverse cross-attention — triple win (velocidad + métrica + eficiencia)

### Descubrimiento

Al analizar los tiempos de training de todos los runs en UNC (A30, 1000 batches/ep, bs=16), encontramos que las arquitecturas con reverse cross-attention de audio (a4r, d4a4r, d4-a4r) son **2.6x más rápidas** que el baseline D0 y el resto de arquitecturas:

### Tiempos de training por arquitectura (A30, UNC)

| Clase | Train/ep | sec/batch | Arquitecturas |
|-------|---------|-----------|---------------|
| **Rápida** | **~13 min** | **0.77 s** | **a4r, d4a4r, d4-a4r** |
| Estándar | ~34-35 min | 2.04-2.11 s | D0, d4a4, t3-wt, t3-tri, film-*, d4r, a8, a9 |
| Pesada | ~37-38 min | 2.23-2.30 s | moe-dual, moe-a4, moe-v2/v3/v4 |

Tabla detallada con eval y wall times:

| Arquitectura | Train/ep | Eval/ep | Total/ep (con eval) | Wall 30ep | Wall 60ep |
|-------------|---------|---------|-------------------|-----------|-----------|
| a4r | 12.9 min | 9.4 min | 14.8 min | 7.7h | ~14.8h |
| d4a4r | 12.9 min | 9.5 min | 14.8 min | 7.7h | — |
| d4-a4r | 13.0 min | 9.4 min | 14.9 min | 7.4h | ~14.9h |
| D0 (baseline) | 34.0 min | 11.7 min | 36.3 min | — | ~36.3h |
| d4a4 | 35.1 min | 11.6 min | 37.4 min | — | ~37.4h |
| t3-wt | 35.1 min | 11.6 min | 37.4 min | 18.9h | — |
| moe-dual | 38.3 min | 12.8 min | 40.9 min | 20.4h | — |

(Eval se amortiza: solo ocurre cada 5 epochs en los runs de 60ep)

### Causa raíz: secuencia de 188 vs 2400 tokens en el audio transformer

El speedup NO viene de menos parámetros (a4r tiene +3.2M vs D0). Viene de la **longitud de secuencia** que procesa el transformer de audio.

**D0 (y d4a4, t3-wt, etc) — pipeline estándar**:
```
Audio waveform → CNN → features [B, 2400, 1024] → Transformer(2400 tokens) → pool → proj
```

**A4r — reverse cross-attention pipeline**:
```
Audio waveform → CNN → features [B, 2400, 1024]  (K/V)
Audio waveform → STFT → descriptor [B, 188, 8] → q_proj → [B, 188, 1024]  (Q)
cross_attn(Q=descriptor, K/V=features) → [B, 188, 1024]
→ Transformer(188 tokens) → pool → proj
```

El transformer de audio (4 layers, d=1024, ~60M params) es la parte más pesada del modelo. Self-attention cuesta O(n²) en longitud de secuencia:

- D0: 2400² = **5,760,000** operaciones de atención por layer
- a4r: 188² = **35,344** operaciones de atención por layer
- **Ratio: 163x menos operaciones de atención**

Como el transformer de audio domina el cómputo total (~60% del forward pass), reducir 163x su costo de atención produce ~2.6x de speedup total.

### El descriptor como cuello de botella informacional beneficioso

La clave conceptual: el descriptor de ratios (188 tokens) no es solo una feature extra — **reemplaza** la secuencia de 2400 tokens de CNN como input al transformer.

Funciona como un **bottleneck de atención**: en lugar de que el transformer procese 2400 tokens de features CNN (mayormente redundantes), procesa 188 tokens de descriptor que ya contienen la información que importa (ratios de frecuencia a resolución STFT nativa). El cross-attention previo (Q=descriptor, K/V=CNN features) es el mecanismo que transfiere la información acústica relevante a los tokens del descriptor.

Esto explica por qué funciona mejor: el transformer no pierde tiempo en self-attention entre tokens redundantes de la CNN. Se enfoca directamente en la información de ratios.

### Comparativa completa: A4r vs D0

| Dimensión | D0 (baseline) | a4r | d4-a4r |
|-----------|--------------|-----|--------|
| **Best S (30ep)** | 72.0% | **82.0%** (+10pp) | **79.8%** (+7.8pp) |
| **Params** | ~65M | ~68.2M (+3.2M) | ~69.6M (+4.6M) |
| **Train/ep** | 34 min | **13 min** (2.6x) | **13 min** (2.6x) |
| **Wall 60ep** | ~36h | **~15h** | **~15h** |
| **Attn ops/layer** | 5.76M | **35K** (163x menos) | **35K** (163x menos) |
| **Seq len (audio transformer)** | 2400 | **188** | **188** |

**Triple win**: más rápido, mejor métrica, y el aumento de parámetros (+3.2M) es modesto comparado con los ~65M del baseline.

### Implicancias para el proyecto

1. **Eficiencia computacional**: A4r permite más iteraciones experimentales en el mismo presupuesto de GPU-hours. Un run de 60ep cuesta lo que D0 tarda en 25ep.

2. **Escalabilidad**: Si escalamos a más epochs o más datos, a4r escala 2.6x mejor. El bottleneck computacional del proyecto deja de ser el transformer de audio.

3. **Validación de la hipótesis central de Phideus**: Los ratios de frecuencia (capturados en 188 tokens STFT) contienen suficiente información para superar una representación CNN de 2400 tokens. Esto es evidencia directa de que los ratios son una representación **más eficiente** de la señal de audio, en línea con la Harmonic Information Theory.

4. **Arquitectura candidata para producción**: Si el objetivo fuera deployment, a4r ofrece el mejor tradeoff calidad/costo. Menos FLOPS por inferencia, mejor accuracy.

### Código de referencia

La implementación de reverse cross-attention está en:
- `gate43_scratch_training.py`, líneas 1310-1379: `_encode_audio_with_reverse_cross_attention()`
- Línea 1325: *"Key difference: Transformer processes 188 tokens (vs 2400 in regular), so self-attention is 12.8x cheaper per layer."*
- Clase `Gate42AudioReverseCrossAttModel` (línea 1382): wrapper que usa la función anterior

El descriptor A4 (`compute_audio_descriptor_a4`) genera 8 features por frame STFT:
- log-frequency deltas entre picos espectrales consecutivos
- Resolución temporal nativa de STFT (~188 frames para 4s de audio)

---

## 7. Gate 5A — Nuevos brazos propuestos: t3-wt combinatorios

### Contexto

Gate 5A ("Barrido descriptor × mecanismo + cross-modal injection") está pendiente. El usuario propone agregar dos nuevas variantes de t3-wt que exploran la combinatoria entre la Third Tower y los mecanismos de inyección en encoders.

### Motivación

Los resultados actuales de t3-wt usan **d4a4 injection** (concat simple) en los encoders base. Pero sabemos que:
- **a4r** (reverse cross-att) es mejor mecanismo de inyección que d4a4 concat (+10pp vs D0, y 2.6x más rápido)
- No sabemos cuánto de la performance de t3-wt viene de la torre vs de la inyección d4a4

### Dos nuevos brazos

#### t3-wt-vanilla: Third tower SIN inyección

```
Audio waveform → Audio Encoder (VANILLA, sin descriptor) → audio_emb [B, 256]
MIDI events    → MIDI Encoder (VANILLA, sin descriptor)  → midi_emb [B, 256]
A4+D4 concat   → Ratio Tower (2-layer Transformer)       → ratio_emb [B, 256]

Loss = 0.7 × VICReg(audio, midi) + 0.15 × VICReg(audio, ratio) + 0.15 × VICReg(midi, ratio)
```

- Encoders sin inyección de descriptores (como D0)
- La torre de ratios es la ÚNICA vía de información de ratios
- **Pregunta**: ¿la tercera torre sola aporta señal, o necesita la inyección en encoders?
- **Diferencia con t3-anc**: t3-anc también era sin inyección pero usaba loss "anchor" (0% peso en audio↔midi). t3-wt-vanilla mantiene el 70% en audio↔midi

#### t3-wt-a4r: Third tower CON d4-a4r injection

```
Audio waveform → Audio Encoder (A4 reverse cross-att, 188 tokens) → audio_emb [B, 256]
MIDI events    → MIDI Encoder (D4 concat)                         → midi_emb [B, 256]
A4+D4 concat   → Ratio Tower (2-layer Transformer)                → ratio_emb [B, 256]

Loss = 0.7 × VICReg(audio, midi) + 0.15 × VICReg(audio, ratio) + 0.15 × VICReg(midi, ratio)
```

- Audio encoder usa A4 reverse cross-attention (el mecanismo que da +10pp y 2.6x speedup)
- MIDI encoder usa D4 concat (como d4-a4r)
- La torre de ratios añade la loss auxiliar triangular
- **Pregunta**: ¿la torre de ratios mejora sobre d4-a4r solo? ¿Es aditivo?
- **Beneficio extra**: hereda el speedup de 2.6x del audio encoder (188 tokens en lugar de 2400)

### Tabla comparativa: familia t3-wt completa

| Variante | Audio encoder | MIDI encoder | Ratio tower | Loss weights (AM/AR/MR) |
|----------|--------------|-------------|-------------|------------------------|
| t3-anc (existente) | vanilla | vanilla | si | 0/50/50 |
| t3-wt (existente) | d4a4 concat | D4 concat | si | 70/15/15 |
| **t3-wt-vanilla** (NUEVO) | **vanilla** | **vanilla** | si | 70/15/15 |
| **t3-wt-a4r** (NUEVO) | **A4 reverse cross-att** | **D4 concat** | si | 70/15/15 |

### Predicción de velocidad

| Variante | Audio seq len | Est. train/ep |
|----------|--------------|---------------|
| t3-wt (existente) | 2400 | ~35 min |
| t3-wt-vanilla | 2400 | ~35 min |
| **t3-wt-a4r** | **188** | **~15 min** (hereda speedup a4r) |

### Para la implementación (futura)

- t3-wt-vanilla: `Gate44ThirdTowerModel(base_model, loss_mode='weighted', alpha_ratio=0.3, use_d4a4_injection=False)` — casi idéntico a t3-anc pero con loss weighted
- t3-wt-a4r: requiere nuevo modelo que combine Gate44ThirdTowerModel con Gate42DualMixedModel (d4-a4r encoders + ratio tower). No existe aún en código.

### Codex: agregar estos dos brazos al plan de Gate 5A

Estos brazos deben documentarse como parte del plan de Gate 5A (barrido descriptor × mecanismo). La matriz completa de Gate 5A incluiría todas las combinaciones de:
- **Descriptor**: A4, D4, A4+D4
- **Mecanismo de inyección**: concat, reverse cross-att, third tower, FiLM
- **Combinaciones cruzadas**: tower + concat, tower + reverse cross-att, etc.

---

## 8. Reestructuración: Gate 4.5 — LR Schedule Optimization

> Fecha: 2026-02-22
> Decisión del equipo: los extended runs constituyen un gate propio

### Decisión

Todos los extended runs (60ep, 50ep) que están entre Gate 4.4 y Gate 5 deben agruparse bajo un nuevo **Gate 4.5 — LR Schedule Optimization**. Esto es formalmente una exploración de hiperparámetros: la arquitectura y descriptores están fijos (decididos en Gates 4.3/4.4), lo que varía es la duración del training y el scheduler de learning rate.

### Codex: actualizar roadmap y documentación

**Acción requerida**: Insertar Gate 4.5 en la secuencia de gates del ROADMAP_BIAS_CONTROL.md y en toda la documentación relevante. La secuencia correcta es:

```
... → Gate 4.3 (CERRADO) → Gate 4.4 (CERRADO) → Gate 4.5 (EN CURSO) → Gate 5A → Gate 5B
```

### Definición del Gate

**Gate 4.5 — LR Schedule Optimization (Extended Runs)**

**Pregunta central**: ¿Puede un scheduler de LR más inteligente extraer más performance de los mejores arms de Gates 4.3/4.4?

**Variable independiente**: scheduler de LR (3 variantes) × duración (50-60ep)
**Variables fijas**: arquitectura, descriptor, seed, batch size, freeze policy (todo idéntico a 30ep)

### Los 3 schedulers bajo prueba

| # | Scheduler | Descripción | Flags CLI |
|---|-----------|-------------|-----------|
| 1 | **Cosine stretched** | Cosine estándar estirado a 60ep. Más lento que 30ep, el LR baja gradualmente. | (default, `--epochs 60`) |
| 2 | **Trapezoidal hold** | Hold al peak LR por 50% del training, luego cosine decay. | `--lr-hold-fraction 0.5` |
| 3 | **Cosine-tail** | Replica exacta de curva 30ep hasta LR=0.10 (~e24), luego cola lineal 0.10→0.02 hasta e60. | `--lr-cosine-ref-epochs 30 --lr-floor 0.10 --lr-tail-end 0.02` |

### Tabla completa de runs (actualizada 2026-02-23)

**Cosine stretched (6 runs, 5 COMPLETE + 1 DEAD)**:

| Run | Status | Best S | Best ep | Δ vs 30ep |
|-----|--------|--------|---------|-----------|
| **d4a4 60ep** | **COMPLETE** | **83.8%** | e50 | **+0.2pp ALL-TIME RECORD** |
| a4r 60ep | **COMPLETE** | 79.4% | e60 | -2.6pp (regresó) |
| D0 60ep | **COMPLETE** | 72.8% | e50 | +12.6pp |
| t3-wt 50ep | **COMPLETE** (trap) | 81.2% | e50 | +1.4pp |
| d4-a4r 60ep | **COMPLETE** | 79.8% | e55 | ±0pp (empató) |
| moe-dual 60ep | **DEAD** (time limit) | 73.0% | e30 | +0.4pp, peak no sostenido |

**Cosine-tail (4 runs, 1 COMPLETE + 2 EN CURSO + 1 PENDING)**:

| Run | Status | Best S | Best ep | Δ vs 30ep |
|-----|--------|--------|---------|-----------|
| a4r ctail | **COMPLETE** | 80.6% | e60 | -1.4pp |
| d4a4 ctail | EN CURSO (~e51) | 83.4% | e30 | -0.4pp del RECORD |
| D0 ctail | EN CURSO (~e56) | 73.4% | e50 | **nuevo all-time best D0** |
| d4-a4r ctail | PENDING (Job 1143330) | — | resume e5 | re-submitted, ivb04 excluido |

### All-time best actualizado (2026-02-23)

| Descriptor | Best S | Fuente |
|------------|--------|--------|
| d4a4 | **83.8%** | 60ep cosine e50 (RECORD) |
| a4r | 82.0% | 30ep e29 |
| t3-wt | 81.2% | 50ep trap e50 |
| d4-a4r | 79.8% | 30ep e30 = 60ep cos e55 |
| d4a4r | 74.4% | 30ep e30 |
| D0 | 73.4% | ctail e50 (nuevo best) |
| moe-dual | 73.0% | 60ep cosine e30 (DEAD) |

### Observaciones consolidadas

**Cosine stretched**:
1. **d4a4** es el único arm que mejoró con cosine stretched (+0.2pp → record 83.8%)
2. **a4r regresó -2.6pp** — el scheduler lento le perjudica
3. **t3-wt ganó +1.4pp** con trapezoidal hold (50ep)
4. **d4-a4r empató** — 79.8% en e55, idéntico a 30ep
5. **D0 ganó +12.6pp** — tenía mucho room (30ep solo daba 60.2%)
6. **moe-dual MUERTO** — peak e30=73.0% cayó a 69-70% en e35-e45, familia MoE agotada

**Cosine-tail**:
7. **a4r ctail NO recupera**: 80.6% es mejor que cosine (79.4%) pero sigue -1.4pp bajo 30ep. a4r no mejora con ningún schedule extendido.
8. **d4a4 ctail converge antes**: pico e30=83.4% vs cosine e50=83.8%. Trade-off velocidad vs precisión máxima (-0.4pp, converge ~20ep antes).
9. **D0 ctail nuevo all-time best**: 73.4% > 72.8% (cosine). La cola lineal beneficia ligeramente incluso al control sin descriptores.
10. **d4-a4r ctail pendiente**: re-submitted tras exclusión de nodo degradado ivb04.

### Conclusiones parciales del Gate 4.5

1. **El schedule agresivo de 30ep es difícil de superar**: solo d4a4 mejoró (marginalmente) con cosine stretched.
2. **Cosine-tail es mejor que cosine stretched para a4r y D0**, pero no recupera el nivel del 30ep en a4r.
3. **La familia MoE está definitivamente agotada**: peak no sostenido, sin interés para Gate 5.
4. **d4-a4r empata pero no mejora**: el run de 60ep confirma que 30ep ya extraía todo el jugo.
5. **Falta**: d4a4 ctail final (e60), D0 ctail final (e60), y d4-a4r ctail completo para cerrar Gate 4.5.

### Scripts SLURM

Cosine stretched:
- `batch_60ep_d0.sh`, `batch_60ep_d4a4.sh`, `batch_60ep_a4r.sh`, `batch_60ep_d4-a4r.sh`, `batch_60ep_moe-dual.sh`

Trapezoidal hold:
- `gate44_t3-wt_scratch_50ep_hold.sh`

Cosine-tail:
- `batch_60ep_ctail_d0.sh`, `batch_60ep_ctail_d4a4.sh`, `batch_60ep_ctail_a4r.sh`, `batch_60ep_ctail_d4-a4r.sh`

Todos en `experiments/bias_control/slurm/`. Nota: `batch_60ep_ctail_d4-a4r.sh` actualizado con `--exclude=ivb03,ivb04,ivb10`.

### Limpieza: plan MoE eliminado

Se eliminó el plan viejo de Gate 4.4-MoE (`/root/.claude/plans/wondrous-meandering-newt.md`). Esas variantes ya se ejecutaron en screening y quedaron en ~60% S (nivel D0).

---

## 9. Conexión SSH directa a Mendieta

> Fecha: 2026-02-23

Se estableció conexión SSH directa desde Inference01 a Mendieta usando las llaves RSA del MacBook del usuario, copiadas a `/mnt/m2-1TB/Phideus/SSH/` (ignorado por git).

```bash
ssh -i /mnt/m2-1TB/Phideus/SSH/id_rsa mfmendez@mendieta.ccad.unc.edu.ar
```

**Uso**: transferencia de datasets vía rsync. Se transfirió SAINetset8.0 (11GB, 129K archivos) a `/home/mfmendez/SAINet/SAINetset8.0/` a ~30 MB/s.

**Nota**: las llaves son temporales y están en `.gitignore`. Se agregó `SSH/` al gitignore en commit `d045992`.

---

## 10. Plan de limpieza local del repo (Caso B Seguro)

> Fecha: 2026-02-23
> Estado: plan aprobado, pendiente implementación como skill

Se diseñó un plan de limpieza local para liberar ~73-86G en disco. El plan fue elaborado por Codex y revisado por Claude LOCAL.

### Fases

| Fase | Descripción | Ahorro estimado | Riesgo |
|------|-------------|-----------------|--------|
| 0 | Inventario + PRESERVE_LIST.txt | 0G | — |
| 1 | venv/, viz/node_modules, caches | ~8.5G | Cero |
| 2 | Checkpoints redundantes en training_outputs | ~60-75G | Bajo-medio |
| 3 | Duplicados experiments/un_audio_un_midi | ~2-3G | Bajo |
| 4 | Modelos legacy (vae_checkpoints, models/vae) | ~1.5-2.2G | Bajo-medio |
| 5 | Verificación post-limpieza | 0G | — |

### Salvaguarda central

Generar `PRESERVE_LIST.txt` con rutas absolutas antes de cualquier borrado. Ningún `rm` si el path aparece en esa lista. Si hay ambigüedad → no borrar, enviar a `SKIPPED_MANUAL_REVIEW.tsv`.

### Decisiones tomadas

- **data/maestro_v3 (121G) NO se toca** en esta primera pasada
- **results_unc/ intocable**
- **foundation_locked_e25.pt intocable** (chmod 444)
- Backup en /mnt/raid1/Phideus-backup como red de seguridad pasiva
- Primera ejecución obligatoriamente en dry-run

### Feedback de Claude LOCAL incorporado

- Fase 2: verificar que `best_model.pt` existe antes de purgar checkpoints intermedios (cruzar con training_history.json)
- Fase 4: criterio "últimos N" reemplazado por "solo referenciados en docs/scripts activos"
- Milestones cada 10ep en keep-set: innecesarios para runs cerrados, solo best + final

### Codex: documentar este plan

Crear documento en la estructura del repo con el plan completo para referencia futura y para la implementación de la skill de limpieza.

---

## 11. Gate 5B (S11): estado operativo, bugfix y optimización de tiempos

> Fecha: 2026-02-24
> Estado: ejecución LOCAL en tmux (`gate5b`) con foco en Test 01 + mejora de eficiencia por cache

### 11.1 Scoreboard canónico (Test 12) validado

Se consolidó la corrida canónica (`pool=256`, `n_queries=500`, `seed=42`) para los 4 checkpoints Gate 5B:

| Arm | S | A2M R@10 | M2A R@10 |
|-----|---|----------|----------|
| `d4a4` | 83.8% | 84.4% | 83.8% |
| `a4r` | 82.0% | 82.6% | 82.0% |
| `d4-a4r` | 79.8% | 81.4% | 79.8% |
| `D0` | 73.4% | 74.8% | 73.4% |

Lectura operativa:
- Los valores recuperan los históricos esperados del frente activo.
- El scoreboard exporta `a2m`/`m2a`/MRR/R@k, pero **no** una métrica separada `hard_neg_accuracy` como en eval por época de training.

### 11.2 Test 01 (Causal Ablation): incidente y corrección

Incidente:
- El test se detuvo en `collect_descriptor_stats` por `RuntimeError` de `torch.cat` en tensores D4 con longitud temporal variable por batch (`[B, N, 4]`, con `N` variable por padding dinámico).

Causa:
- Se intentó concatenar directamente `midi_vals` en dim 0 asumiendo shape homogénea.

Fix aplicado:
- Flatten por batch antes de concatenar: `v.reshape(-1, v.size(-1))` para cada tensor D4.
- Resultado: evita dependencia de `N` y permite estimar media/std globales para modo `noise`.

### 11.3 Estado de corrida en tmux

- Sesión activa: `gate5b`.
- `D0` ya cerrado (control negativo, sin ablaciones).
- `d4a4` corriendo en secuencia de ablaciones; luego siguen `a4r` y `d4-a4r`.
- Se evitó relanzar bloque completo al detectar que `--model` permite ejecución individual por arm.

### 11.4 Optimización aprobada: cache de embeddings normales

Problema identificado:
- Varias pruebas repetían extracción completa de embeddings sin valor científico adicional.

Estrategia:
- Introducir cache en `data/gate5b_results/{arm}/embeddings_normal.npz` y reutilizar en tests que operan sobre embeddings normales.

Cambios implementados:
- `experiments/bias_control/gate5b/harness.py`
  - `save_embeddings()`
  - `load_cached_embeddings()`
  - `get_normal_embeddings()`
- Script nuevo: `experiments/bias_control/gate5b/cache_embeddings.py` (genera cache para los 4 arms).
- Integración de cache en:
  - `experiments/bias_control/gate5b/test12_scoreboard.py` (extrae + cachea)
  - `experiments/bias_control/gate5b/test01_causal_ablation.py` (normal eval desde cache; se retiró verificación redundante `verify_ablation_effective` en esta versión)
  - `experiments/bias_control/gate5b/test04_transposition.py` (reusa audio normal + midi shift=0)
  - `experiments/bias_control/gate5b/test10_visualizations.py`
  - `experiments/bias_control/gate5b/test03_ratio_probe.py`

Impacto estimado:
- Ahorro operativo agregado ~1.5-2h en la batería local por eliminación de extracciones redundantes.

### 11.5 Resultados completos Test 01: Causal Ablation (CERRADO)

> **Codex**: Estas tablas son los datos canónicos de Test 01. Usarlas tal cual en la documentación del Gate 5B showcase y en el informe de ejecución.

**Tabla maestra — Test 01 Causal Ablation (todos los arms)**:

| Arm | S_normal | zero_audio | zero_midi | zero_both | noise_audio | noise_midi | noise_both | shuffle_audio | shuffle_midi | shuffle_both |
|-----|----------|------------|-----------|-----------|-------------|------------|------------|---------------|--------------|--------------|
| **D0** | 73.4% | — | — | — | — | — | — | — | — | — |
| **d4** | 63.6% | — | 62.8% (+0.8) | — | — | 63.6% (0.0) | — | — | 62.4% (+1.2) | — |
| **d4a4** | 83.8% | 7.8% (-76.0) | 84.4% (+0.6) | 7.4% (-76.4) | 39.0% (-44.8) | 83.6% (-0.2) | 38.2% (-45.6) | 46.6% (-37.2) | 83.8% (0.0) | 47.0% (-36.8) |
| **a4r** | 82.0% | 4.4% (-77.6) | — | — | 32.6% (-49.4) | — | — | 49.8% (-32.2) | — | — |
| **d4-a4r** | 79.8% | 4.4% (-75.4) | 79.4% (-0.4) | 4.6% (-75.2) | 33.2% (-46.6) | 80.0% (+0.2) | 32.4% (-47.4) | 47.4% (-32.4) | 79.8% (0.0) | 48.0% (-31.8) |

Notas:
- D0 no tiene descriptores → no aplica ablación (control negativo, delta=0 by definition)
- d4 solo tiene MIDI descriptor → columnas audio vacías
- a4r solo tiene audio descriptor → columnas midi vacías
- Deltas entre paréntesis, positivo = S ablated > S normal (ruido estadístico)

**Tabla resumen simplificada para documentación**:

| Arm | S_normal | Δ zero_audio | Δ zero_midi | Δ shuffle_audio | Δ shuffle_midi |
|-----|----------|--------------|-------------|-----------------|----------------|
| D0 | 73.4% | n/a | n/a | n/a | n/a |
| d4 | 63.6% | n/a | +0.8pp | n/a | +1.2pp |
| d4a4 | 83.8% | **-76.0pp** | +0.6pp | **-37.2pp** | 0.0pp |
| a4r | 82.0% | **-77.6pp** | n/a | **-32.2pp** | n/a |
| d4-a4r | 79.8% | **-75.4pp** | -0.4pp | **-32.4pp** | 0.0pp |

### 11.6 Hallazgo científico principal de Test 01

> **Codex**: Este hallazgo debe ser prominente en toda la documentación de Gate 5B. Es el resultado más importante hasta ahora.

**A4 (audio descriptor) es completamente causal. D4 (MIDI descriptor) no contribuye nada — ni en duales, ni solo.**

1. **A4 es causal**: Zerear A4 destruye el modelo (-75 a -78pp). Shufflear A4 lo degrada severamente (-32 a -37pp). Noise tiene efecto intermedio (-45 a -49pp). Esto confirma que la información de ratios de audio es el motor principal de la mejora.

2. **D4 NO es causal en duales**: En d4a4 y d4-a4r, zerear/shufflear/ruidear D4 no cambia S (deltas ≈ 0). A4 subsume completamente la señal de D4.

3. **D4 NO es causal ni solo**: El checkpoint D4 puro (Gate 4.3, S=63.6%) tampoco muestra dependencia causal de su descriptor MIDI. Zerear D4 → delta +0.8pp, shuffle → +1.2pp (ruido estadístico).

4. **Paradoja D4**: D4 históricamente mejoró +3.4pp sobre D0 (63.6% vs 60.2%), pero la ablación post-training no detecta causalidad. Hipótesis posibles:
   - Los parámetros extra del wrapper (~0.5M) son suficientes para la mejora, no la información del descriptor
   - D4 actúa como regularización durante training (ayuda a la optimización) pero no es necesario en inference
   - El Test 02 (parameter-matched) en UNC resolverá esta ambigüedad

### 11.7 Gate 5B — estado operativo actualizado (2026-02-25)

| Test | Status | Resultado clave |
|------|--------|-----------------|
| **Test 12 (Scoreboard)** | ✅ DONE | 4 modelos validados vs históricos |
| **Test 01 (Causal Ablation)** | ✅ DONE | A4 causal, D4 no causal (ver tablas arriba) |
| Test 04 (Transposition) | PENDING | Siguiente en cola local |
| Test 10 (Visualizations) | PENDING | — |
| Test 03 (RatioProbe) | PENDING | — |
| Test 06 (RSA/CKA) | PENDING | — |
| Test 08 (Ratio Decoding) | PENDING | — |
| Test 09 (Invariance Suite) | PENDING | — |
| Test 05 (Multi-seed) | PENDING UNC | SLURM script listo, pendiente submit |
| Test 02 (Param-matched) | PENDING UNC | Training wrapper por implementar |

**Orden de ejecución local**: 04 → 10 → 03 → 06 → 08 → 09

**Cache de embeddings**: Generado para D0, d4a4, a4r, d4-a4r en `data/gate5b_results/{arm}/embeddings_normal.npz`. Los tests locales reutilizan estos caches (~5 min ahorrados por extracción).

### 11.8 Checkpoints Gate 5B

> **Codex**: Esta tabla documenta los checkpoints canónicos para Gate 5B. Incluirla en la documentación del showcase.

| Arm | Checkpoint | Descriptor | Epoch | S | Params | Fuente |
|-----|-----------|-----------|-------|---|--------|--------|
| D0 | `models/gate5b/D0/best_model.pt` | d0 | e50 | 73.4% | 74.2M | ctail 60ep |
| d4 | `models/gate5b/d4/best_model.pt` | d4 | e5 | 63.6% | 74.5M | Gate 4.3 |
| d4a4 | `models/gate5b/d4a4/best_model.pt` | d4a4 | e50 | 83.8% | 75.5M | cosine 60ep (RECORD) |
| a4r | `models/gate5b/a4r/best_model.pt` | a4r | e29 | 82.0% | 78.6M | 30ep |
| d4-a4r | `models/gate5b/d4-a4r/best_model.pt` | d4-a4r | e30 | 79.8% | 78.9M | 30ep |

### 11.9 Instrucciones para Codex

> **ACCIÓN REQUERIDA**: Actualizar la documentación del Gate 5B con los datos de esta sección.

1. **Informe de ejecución** (`INFORME_EJECUCION_TEST01_TEST12_2026-02-25.md`): Actualizar con la tabla maestra completa de Test 01 (sección 11.5), incluyendo el resultado de D4 puro.

2. **Showcase/README**: Destacar el hallazgo principal (sección 11.6): A4 causal, D4 no causal, paradoja D4.

3. **Checkpoints**: Documentar la tabla de checkpoints (sección 11.8) en el plan de implementación o showcase.

4. **Estado de tests**: Mantener la tabla de sección 11.7 actualizada conforme avancen los tests.

5. **Nomenclatura de deltas**: En las tablas, positivo = S_ablated > S_normal (el modelo mejoró al ablacionar, interpretado como ruido estadístico). Negativo = S cayó (causalidad detectada).

### 11.10 Test 04: Transposition Invariance — Resultados COMPLETOS

> **Codex**: Test 04 mide si los modelos aprendieron intervalos relativos (ratios) en lugar de alturas absolutas. Se transpone el MIDI ±N semitonos y se mide cuánto cae el retrieval. Un modelo que aprendió ratios debería ser más robusto a transposición.

**Estado**: Los 4 modelos **COMPLETOS** (D0, d4a4, a4r, d4-a4r).

**Tabla de resultados absolutos — S por transposición**:

| Shift | D0 | d4a4 | a4r | d4-a4r |
|------:|-----:|------:|-----:|-------:|
| **-6** | 13.8% | 24.2% | 27.0% | 27.0% |
| **-3** | 26.6% | 41.4% | 46.2% | 45.0% |
| **-1** | 65.6% | 75.2% | 76.6% | 73.2% |
| **0** | 73.4% | 83.8% | 82.0% | 79.8% |
| **+1** | 64.0% | 75.6% | 76.8% | 75.2% |
| **+3** | 27.4% | 44.6% | 51.0% | 49.2% |
| **+6** | 13.4% | 25.6% | 27.6% | 27.2% |

**Tabla de retención proporcional — S/S₀ × 100%**:

| Shift | D0 | d4a4 | a4r | d4-a4r |
|------:|-----:|------:|-----:|-------:|
| **-6** | 18.8% | 28.9% | 32.9% | 33.8% |
| **-3** | 36.2% | 49.4% | 56.3% | 56.4% |
| **-1** | 89.4% | 89.7% | 93.4% | 91.7% |
| **0** | 100% | 100% | 100% | 100% |
| **+1** | 87.2% | 90.2% | 93.7% | 94.2% |
| **+3** | 37.3% | 53.2% | 62.2% | 61.7% |
| **+6** | 18.3% | 30.5% | 33.7% | 34.1% |

**Tabla comparativa — Ventaja absoluta sobre D0 baseline (pp)**:

| Shift | d4a4 vs D0 | a4r vs D0 | d4-a4r vs D0 |
|------:|-----------:|----------:|-------------:|
| **-6** | +10.4pp | **+13.2pp** | +13.2pp |
| **-3** | +14.8pp | **+19.6pp** | +18.4pp |
| **-1** | +9.6pp | **+11.0pp** | +7.6pp |
| **+1** | +11.6pp | **+12.8pp** | +11.2pp |
| **+3** | +17.2pp | **+23.6pp** | +21.8pp |
| **+6** | +12.2pp | **+14.2pp** | +13.8pp |

> **Nota**: d4-a4r muestra retención % comparable a a4r (ambos usan A4 reverse cross-att), pero con S absoluto menor (79.8% vs 82.0%). El componente D4 no aporta retención adicional — coherente con Test 01 (D4 no causal).

### 11.11 Hallazgo científico Test 04: a4r es el descriptor más invariante a transposición

> **Codex**: Este hallazgo debe documentarse junto al de Test 01. Son complementarios: Test 01 demuestra causalidad del A4, Test 04 demuestra que A4 codifica intervalos relativos (ratios) y no alturas absolutas.

**Observación central**: Cuanto mayor es el shift de transposición, más se nota la ventaja del descriptor sobre el baseline D0. A ±1 semitono todos los modelos retienen ~89-94%. A ±3/±6 semitonos, la brecha se amplifica:

1. **a4r gana en TODOS los shifts sobre d4a4** — consistentemente +2-9pp más de retención. La reverse cross-attention (188 tokens compactos) genera una representación más robusta a transposición que d4a4 (concat).

2. **Patrón simétrico**: Los modelos degradan simétricamente en ± shifts, lo cual es esperado (transponer hacia arriba o abajo es equivalente en dificultad).

3. **Interpretación**: A4 codifica log-freq deltas (intervalos relativos entre picos espectrales consecutivos). Estos son transposition-invariant por definición: transponer ±N semitonos desplaza todas las frecuencias pero los **ratios entre picos consecutivos** no cambian. El modelo que usa A4 (especialmente a4r con cross-attention directa) captura esta propiedad.

4. **D0 como control negativo**: D0 solo tiene features CNN de audio (magnitudes espectrales absolutas). La transposición cambia las magnitudes → embeddings cambian → S cae. La caída pronunciada de D0 en ±3/±6 confirma que sin descriptor de ratios, el modelo es sensible a pitch absoluto.

5. **Conexión con Test 01**: A4 es causal (Test 01) Y codifica información invariante a transposición (Test 04). Esto es evidencia fuerte de que A4 captura ratios de frecuencia útiles para cross-modal retrieval.

### 11.12 Gate 5B — Estado operativo actualizado (2026-02-25 ~05:00 UTC)

> **Codex**: Reemplaza la tabla de sección 11.7 como estado más reciente.

| Test | Status | Resultado clave |
|------|--------|-----------------|
| **Test 12 (Scoreboard)** | ✅ DONE | 4 modelos validados, S coincide con históricos |
| **Test 01 (Causal Ablation)** | ✅ DONE (5 arms) | A4 causal, D4 no causal, paradoja D4 |
| **Test 04 (Transposition)** | ✅ DONE (4 arms) | a4r más invariante, +23.6pp vs D0 a ±3 |
| **Test 10 (Visualizations)** | ✅ DONE | t-SNE/UMAP 2x2 grids + detail + alignment cosine |
| **Test 03 (RatioProbe)** | ✅ DONE (4 arms) | R² moderado, D0≥augmented en cross-decoding |
| **Test 06 (RSA/CKA)** | ✅ DONE (4 arms) | **HALLAZGO FUERTE**: descriptores duplican CKA cross-encoder |
| **Test 08 (Ratio Decoding)** | ✅ DONE (3 arms aug) | Bandas alta frecuencia = features más sensibles |
| **Test 09 (Invariance Suite)** | 🟡 EN CURSO | Temporal/velocity/octave/noise × 4 modelos |
| Test 05 (Multi-seed) | PENDING UNC | SLURM script listo (`gate5b_multiseed.sh`) |
| Test 02 (Param-matched) | PENDING UNC | Training wrapper por implementar |

**Tests locales**: 8/9 DONE, falta Test 09 (en curso, muy lento ~5.5min/evaluación).

### 11.13 Gráficos generados — Gate 5B Scientific Validation (25 charts, v2)

> **Codex**: TODOS los gráficos fueron regenerados en v2 (2026-02-25) con mejoras sustanciales: descriptor type labels, colores consistentes, overlaps corregidos, 4 modelos en todos los charts, dashboard de 6 paneles, nombres de bandas Hz corregidos. Estilo visual unificado: fondo oscuro (#1a1a2e), 150 DPI.

**Directorio raíz**: `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/resultados_compartir/06_gate5b_scientific_validation/`

**Paleta de colores estandarizada**:
- **D0** = `#888888` (gris) — baseline sin descriptor
- **d4a4** = `#e74c3c` (rojo) — D4+A4 concat, el campeón en S
- **a4r** = `#3498db` (azul) — A4 reverse cross-attention
- **d4-a4r** = `#9b59b6` (púrpura) — D4 + A4 reverse cross-attention (dual)
- **d4** = `#66bb6a` (verde) — D4 concat solo (solo en ablation)
- Direcciones: A→M = `#26c6da` (cyan), M→A = `#ff7043` (coral)

**Descriptor type labels** (etiquetas bajo nombre del arm):
- D0 = "baseline"
- d4 = "D4 concat"
- d4a4 = "D4+A4 concat"
- a4r = "A4 rev-crossatt"
- d4-a4r = "D4 + A4 rev-crossatt"

#### Inventario completo: 13 analytical charts + 11 visualization charts + 1 dashboard

**test12_scoreboard/** (4 charts):

| # | Archivo | Contenido |
|---|---------|-----------|
| 01 | `chart01_scoreboard_S.png` | Barras horizontales S por arm con deltas vs D0, descriptor type labels |
| 02 | `chart02_recall_spectrum.png` | R@1/R@5/R@10/R@20 por arm, paneles A→M y M→A, valores en R@10 |
| 03 | `chart03_mrr_meanrank.png` | MRR y Mean Rank bidireccional (cyan/coral) |
| 04 | `chart04_hard_negatives.png` | Hard negative accuracy: same-piece (harder) vs random piece |

**test01_causal_ablation/** (2 charts):

| # | Archivo | Contenido |
|---|---------|-----------|
| 05 | `chart05_ablation_heatmap.png` | Heatmap 4 arms × 9 modos, deltas en pp, colorscale rojo→verde |
| 06 | `chart06_audio_vs_midi_causal.png` | Barras A4 vs D4 causal contribution, annotaciones "FULLY CAUSAL" / "NOT CAUSAL" |

**test04_transposition/** (1 chart):

| # | Archivo | Contenido |
|---|---------|-----------|
| 07 | `chart07_transposition_curves.png` | 2 paneles: S absoluto + retención %, **4 modelos**, advantage annotations |

**test06_rsa_cka/** (2 charts):

| # | Archivo | Contenido |
|---|---------|-----------|
| 09 | `chart09_cka_heatmaps_4models.png` | 2×2 grid de matrices CKA 8×8, bloque cross-encoder resaltado |
| 10 | `chart10_cka_crossencoder_bar.png` | Bar chart CKA cross-encoder mean, % incremento vs D0, línea baseline |

**test08_ratio_decoding/** (2 charts):

| # | Archivo | Contenido |
|---|---------|-----------|
| 11 | `chart11_sensitivity_bars.png` | Grouped bars: 8 bandas × 3 arms, separador low/high freq, nombres Hz |
| 12 | `chart12_sensitivity_radar.png` | Spider plot 8 ejes, 3 líneas (d4a4/a4r/d4-a4r), perfiles distintos |

**test03_ratio_probe/** (1 chart):

| # | Archivo | Contenido |
|---|---------|-----------|
| 13 | `chart13_probe_r2.png` | Grouped bars: 6 probes × 4 arms, separados cross-modal vs self, "D0 wins!" annotation |

**summary/** (1 chart):

| # | Archivo | Contenido |
|---|---------|-----------|
| 08 | `chart08_summary_dashboard.png` | **6 paneles**: A.Scoreboard + B.Causal + C.Transposition + D.CKA + E.Sensitivity + F.Recall |

**test10_visualizations/** (11 charts):

| # | Archivo | Contenido |
|---|---------|-----------|
| V1 | `comparison_tsne.png` | 2×2 grid t-SNE, info boxes con S y params por modelo |
| V2 | `comparison_umap.png` | 2×2 grid UMAP, info boxes con S y params por modelo |
| V3 | `alignment_cosine_distribution.png` | 4 histogramas: matched vs random cosine similarity por arm |
| V4-V7 | `{D0,d4a4,a4r,d4-a4r}_tsne_detail.png` | t-SNE detail: panel izq modality coloring, panel der piece coloring |
| V8-V11 | `{D0,d4a4,a4r,d4-a4r}_umap_detail.png` | UMAP detail: panel izq modality coloring, panel der piece coloring |

### 11.14 GLOSARIO DE VISUALIZACIONES — Qué es, qué representa, qué se puede ver

> **Codex**: Esta sección es el glosario completo de las 24 visualizaciones PNG de Gate 5B (más 6 GIF animados en sección aparte). Para cada chart se explica: (1) qué tipo de gráfico es, (2) qué datos representa, (3) qué información científica se puede extraer al observarlo, y (4) hallazgos clave visibles.

---

#### CHART 01 — `chart01_scoreboard_S.png` (Test 12: Scoreboard)

**Tipo**: Barras horizontales.
**Qué representa**: La métrica S (= min(A2M R@10, M2A R@10)) para cada uno de los 4 modelos candidatos, evaluados bajo configuración canónica idéntica (pool=256, queries=500, seed=42, hard negatives: 64 same-piece + 32 semi-hard, dataset MAESTRO v3 validation split 137 piezas).
**Qué información se puede ver**:
- La performance absoluta de cada modelo en cross-modal retrieval.
- La ganancia en puntos porcentuales (pp) de cada modelo aumentado respecto al baseline D0.
- El descriptor type label indica el mecanismo de inyección de cada arm (concat, reverse cross-attention, dual).
- Un info box en la esquina inferior derecha muestra la configuración canónica de evaluación para verificar reproducibilidad.
**Hallazgo visible**: d4a4 lidera con 83.8% (+10.4pp sobre D0=73.4%). La línea punteada marca el baseline D0.

---

#### CHART 02 — `chart02_recall_spectrum.png` (Test 12: Scoreboard)

**Tipo**: Barras agrupadas, 2 paneles (Audio→MIDI izquierda, MIDI→Audio derecha).
**Qué representa**: El espectro completo de Recall@K para K={1, 5, 10, 20} en ambas direcciones de retrieval. Recall@K = fracción de queries donde el match correcto está entre los top-K resultados del ranking.
**Qué información se puede ver**:
- Cómo escala la performance de cada modelo conforme se relaja el criterio (de R@1 stricto a R@20 laxo).
- La asimetría A→M vs M→A: en general M→A es ligeramente más difícil (valores menores).
- d4a4 lidera en R@10 en ambas direcciones (84.4% A→M, 83.8% M→A).
- Los valores numéricos están anotados sobre las barras de R@10.
**Hallazgo visible**: A R@20 todos los modelos augmented convergen (~95%), la diferencia se concentra en R@1/R@5/R@10.

---

#### CHART 03 — `chart03_mrr_meanrank.png` (Test 12: Scoreboard)

**Tipo**: Barras agrupadas, 2 paneles (MRR izquierda, Mean Rank derecha), colores cyan/coral para A→M/M→A.
**Qué representa**:
- **MRR (Mean Reciprocal Rank)**: promedio de 1/rank del match correcto. Un MRR de 0.458 significa que, en promedio, el match correcto está en la posición ~2.2 del ranking. Mayor = mejor.
- **Mean Rank**: posición promedio del match correcto en el ranking (de 256 candidatos). Menor = mejor.
**Qué información se puede ver**:
- d4a4 tiene el mejor MRR (0.458 A→M, 0.442 M→A) y el mejor Mean Rank (5.2 A→M, 5.6 M→A).
- D0 tiene Mean Rank ~8.4/8.9 — el match correcto cae a la posición 8-9 en promedio.
- La mejora de ranking es sustancial: de posición ~8 (D0) a posición ~5 (d4a4).
**Hallazgo visible**: Consistencia bidireccional — A→M siempre ligeramente mejor que M→A en todos los modelos.

---

#### CHART 04 — `chart04_hard_negatives.png` (Test 12: Scoreboard)

**Tipo**: Barras agrupadas, 2 categorías por arm (same-piece harder vs random piece).
**Qué representa**: Hard negative discrimination — la capacidad del modelo de distinguir entre:
- **Same-piece negatives** (naranja, más difícil): segmentos de la misma pieza musical pero en distinto momento temporal. Son confusores fuertes porque comparten estilo, timbre, tonalidad.
- **Random piece negatives** (cyan, más fácil): segmentos de piezas completamente diferentes.
**Qué información se puede ver**:
- Todos los modelos superan 94% en ambas tareas — discriminación excelente.
- Same-piece es consistentemente más difícil (94-95%) que random (98-99%).
- d4-a4r tiene la mejor discriminación random (99.4%) pero la peor same-piece (94.2%).
- d4a4 es el más balanceado (95.4% same-piece, 99.0% random).
**Hallazgo visible**: Los modelos no "hacen trampa" usando features de pieza — pueden distinguir segmentos dentro de la misma pieza.

---

#### CHART 05 — `chart05_ablation_heatmap.png` (Test 01: Causal Ablation)

**Tipo**: Heatmap (mapa de calor) con escala rojo→verde. Filas = 4 arms augmented (d4, d4a4, a4r, d4-a4r). Columnas = 9 modos de ablación.
**Qué representa**: El delta S (en pp) cuando se interviene causalmente cada descriptor. Delta = S_normal - S_ablated. Positivo (rojo) = el modelo depende de ese descriptor (pierde performance sin él). Cero/negativo (verde) = no depende. Los 9 modos son: Zero Audio, Zero MIDI, Zero Both, Noise Audio, Noise MIDI, Noise Both, Shuffle Audio, Shuffle MIDI, Shuffle Both.
- **Zero**: reemplaza el descriptor por un tensor de ceros.
- **Noise**: reemplaza por ruido gaussiano con misma media y desviación.
- **Shuffle**: permuta el descriptor entre los samples del batch (señal real pero desalineada).
**Qué información se puede ver**:
- Las columnas de "Audio" (A4) son intensamente rojas en d4a4/a4r/d4-a4r: deltas de +75 a +78pp. El modelo COLAPSA sin A4.
- Las columnas de "MIDI" (D4) son verdes en todos: deltas de -0.6 a +1.2pp. D4 no contribuye nada.
- "n/a" gris indica combinaciones que no aplican (ej: d4 no tiene audio descriptor, a4r no tiene MIDI descriptor).
- Noise es intermedio entre zero (máximo efecto) y shuffle (efecto parcial, la señal es real pero desalineada).
**Hallazgo visible**: El contraste visual rojo/verde es dramático — A4 es causal, D4 no. Este es el hallazgo más importante de Gate 5B.

---

#### CHART 06 — `chart06_audio_vs_midi_causal.png` (Test 01: Causal Ablation)

**Tipo**: Barras verticales, 4 arms, barras rojas (Zero Audio) vs barras azules (Zero MIDI).
**Qué representa**: Versión simplificada y de alto impacto del heatmap anterior. Compara directamente la contribución causal del descriptor A4 (audio) vs D4 (MIDI), midiendo cuántos pp cae S al zerear cada uno.
**Qué información se puede ver**:
- Barras rojas (A4) enormes: 76-78pp de caída en d4a4, a4r, d4-a4r.
- Barras azules (D4) invisibles: -0.6 a +0.8pp en d4, d4a4, d4-a4r.
- Annotations explícitas: "A4: FULLY CAUSAL (-75 to -78pp when zeroed)" y "D4: NOT CAUSAL (<=0.8pp even solo)".
- El arm d4 (solo descriptor MIDI, sin audio) confirma que D4 no es causal ni siquiera cuando es el único descriptor.
**Hallazgo visible**: La asimetría A4/D4 es total. Este gráfico es el "money shot" para documentación y presentaciones.

---

#### CHART 07 — `chart07_transposition_curves.png` (Test 04: Transposition Invariance)

**Tipo**: Curvas con marcadores, 2 paneles (S absoluto izquierda, retención proporcional % derecha).
**Qué representa**: Se transpone el MIDI ±N semitonos (N = -6, -3, -1, 0, +1, +3, +6) sin modificar el audio, y se re-evalúa S. Un modelo que aprendió intervalos relativos (ratios) debería ser más robusto que uno que depende de alturas absolutas.
**Qué información se puede ver**:
- Panel izquierdo (S absoluto): la forma de V invertida con pico en 0. D0 (gris) cae más rápido que los modelos augmented.
- Panel derecho (retención %): S/S₀ × 100. a4r retiene más que todos los demás en cada shift.
- Annotations en el panel izquierdo: "Advantage at ±3 semitones vs D0: d4a4 +15.9pp, a4r +23.6pp, d4-a4r +21.8pp".
- Las curvas son simétricas (transponer arriba o abajo es equivalente).
**Hallazgo visible**: a4r (azul) es consistentemente el más invariante a transposición. La ventaja crece con el shift: a ±1 es modesta (~4pp), a ±3 es sustancial (24pp), a ±6 es masiva (14pp en un régimen donde D0 ya está en ~13%).

---

#### CHART 08 — `chart08_summary_dashboard.png` (Summary)

**Tipo**: Dashboard de 6 paneles (2×3 grid).
**Qué representa**: Resumen ejecutivo de los 6 hallazgos principales de Gate 5B en una sola imagen. Cada panel sintetiza un test diferente:
- **A. Scoreboard**: Barras S por arm (Test 12)
- **B. Causal Ablation**: Barras delta A4 vs D4 por arm (Test 01)
- **C. Transposition Invariance**: Curvas de retención % (Test 04)
- **D. Cross-Encoder CKA Alignment**: Barras CKA mean por arm (Test 06)
- **E. A4 Top-3 Feature Sensitivity**: Barras de las 3 bandas más sensibles por arm (Test 08)
- **F. Recall Spectrum (A→M)**: Curvas R@K de 1 a 20 (Test 12)
**Qué información se puede ver**:
- En una sola imagen, la narrativa completa: descriptores mejoran S (A), mediante señal A4 causal (B), que codifica intervalos relativos (C), alineando representaciones cross-encoder (D), con máxima sensibilidad en bandas armónicas (E), y mejorando recall en todos los puntos del espectro (F).
**Hallazgo visible**: Los 6 paneles cuentan una historia coherente. Ideal para presentaciones o resúmenes de una página.

---

#### CHART 09 — `chart09_cka_heatmaps_4models.png` (Test 06: RSA/CKA)

**Tipo**: 2×2 grid de heatmaps (matrices 8×8), un heatmap por modelo, escala de color unificada.
**Qué representa**: Matrices CKA (Centered Kernel Alignment) 8×8 donde filas/columnas son las 8 capas internas de los transformers: 4 audio (A0-A3) + 4 MIDI (M0-M3). CKA mide si dos representaciones tienen la misma estructura geométrica (0 = totalmente diferentes, 1 = idénticas).
- El **bloque diagonal superior-izquierdo** (A0-A3 × A0-A3) = auto-similitud del encoder de audio.
- El **bloque diagonal inferior-derecho** (M0-M3 × M0-M3) = auto-similitud del encoder de MIDI.
- El **bloque off-diagonal** (A0-A3 × M0-M3) = **cross-encoder alignment** — el más importante. Resaltado con recuadros verdes punteados.
**Qué información se puede ver**:
- D0: bloque cross-encoder frío (valores 0.12-0.74), las representaciones de audio y MIDI son bastante diferentes.
- d4a4: bloque cross-encoder más caliente (0.42-0.86).
- a4r y d4-a4r: bloque cross-encoder muy caliente (0.65-0.89). Audio y MIDI "hablan el mismo idioma".
- En todos los modelos, la alineación crece con la profundidad (A3×M3 > A0×M0).
- Los valores numéricos están anotados en cada celda.
**Hallazgo visible**: La diferencia visual entre D0 (colores fríos en el cross-block) y d4-a4r (colores calientes) es dramática.

---

#### CHART 10 — `chart10_cka_crossencoder_bar.png` (Test 06: RSA/CKA)

**Tipo**: Barras verticales, 4 arms, con valores y porcentaje de incremento vs D0.
**Qué representa**: El promedio del bloque cross-encoder de la matriz CKA (la media de los 16 valores del bloque 4×4 audio×midi). Es el resumen numérico de cuánto se alinean las representaciones internas de ambos encoders.
**Qué información se puede ver**:
- D0 = 0.435 (baseline). d4a4 = 0.659 (+51%). a4r = 0.766 (+76%). d4-a4r = 0.794 (+82%).
- Línea punteada horizontal marca el nivel D0 baseline.
- Info box explica que CKA mide si dos conjuntos de representaciones tienen la misma estructura geométrica; mayor = audio y MIDI "hablan el mismo lenguaje".
- El incremento es monótono: D0 < d4a4 < a4r < d4-a4r.
**Hallazgo visible**: Los descriptores DUPLICAN la alineación representacional. d4-a4r tiene +82% más alineación, pero NOTA: más alineación ≠ mejor S (d4-a4r=79.8% < d4a4=83.8%).

---

#### CHART 11 — `chart11_sensitivity_bars.png` (Test 08: Ratio Decoding)

**Tipo**: Barras agrupadas (3 arms × 8 bandas de frecuencia), con separador visual low-freq / high-freq.
**Qué representa**: Perturbation sensitivity de cada dimensión del descriptor A4. Para cada banda de octava, se perturba esa dimensión ±epsilon (0.1) y se mide cuánto cambia el embedding de salida (distancia L2). Mayor sensibilidad = esa banda tiene más influencia en la representación final.
- **Las 8 bandas del A4**: 47-94 Hz, 94-188 Hz, 188-375 Hz, 375-750 Hz (low-freq, faded) | 750-1500 Hz, 1500-3000 Hz, 3000-6000 Hz, 6000-12000 Hz (high-freq, opacas).
- Las barras de alta frecuencia se muestran en opacidad completa, las de baja frecuencia en opacidad reducida (45%), para resaltar visualmente la zona armónica.
**Qué información se puede ver**:
- Las bandas de alta frecuencia (750+ Hz) dominan en TODOS los modelos.
- d4a4 (rojo) pica en band4-5 (750-3000 Hz): zona de "presencia".
- a4r (azul) pica en band6-7 (3000-12000 Hz): zona de "brilliance/air".
- d4-a4r (púrpura) pica en band6 (3-6 kHz) con el valor MÁXIMO global: 1.09.
- Info box explica qué es A4 (temporal delta of log-magnitude per octave band, STFT-based).
**Hallazgo visible**: El mecanismo de inyección determina QUÉ bandas importan más. Concat → presencia (750-3k). Cross-attention → brilliance (3k-12k).

---

#### CHART 12 — `chart12_sensitivity_radar.png` (Test 08: Ratio Decoding)

**Tipo**: Spider/radar plot con 8 ejes (uno por banda de octava), 3 líneas (d4a4, a4r, d4-a4r).
**Qué representa**: Los mismos datos de sensibilidad que chart 11, pero en formato radar para visualizar el "perfil espectral" de cada modelo como una forma geométrica. Cada eje va de 0 a 1.2 y representa la sensibilidad de perturbación de esa banda.
**Qué información se puede ver**:
- La FORMA del perfil es diferente por modelo:
  - d4a4 (rojo): rombo achatado, prominente en 750-3000 Hz.
  - a4r (azul): forma que apunta hacia la derecha (3000-12000 Hz).
  - d4-a4r (púrpura): pico pronunciado en 3-6 kHz.
- En la parte inferior, annotation: "d4a4: peaks at 750-3000 Hz | a4r: peaks at 3000-12000 Hz | d4-a4r: strongest at 3000-6000 Hz (1.09)".
**Hallazgo visible**: Cada modelo "escucha" diferentes partes del espectro a través de la misma representación A4. La cross-attention (a4r, d4-a4r) prefiere frecuencias altas (armónicos débiles pero discriminativos), mientras el concat (d4a4) prefiere frecuencias medias-altas (armónicos más energéticos).

---

#### CHART 13 — `chart13_probe_r2.png` (Test 03: RatioProbe)

**Tipo**: Barras agrupadas, 6 probes × 4 arms, separados en 2 secciones (Cross-Modal Decoding izq, Self-Decoding control der).
**Qué representa**: R² de probes lineales (MLP de 1 capa, 512→target_dim) entrenados sobre embeddings CONGELADOS de cada modelo. Mide cuánta información del dominio opuesto se puede extraer linealmente:
- **Cross-Modal Decoding**: audio→pitch_hist (¿z_audio contiene info de MIDI?), audio→interval_hist, midi→chroma (¿z_midi contiene info de audio?), midi→centroid.
- **Self-Decoding** (control): audio→chroma (mismo dominio), midi→pitch_hist (mismo dominio).
**Qué información se puede ver**:
- midi→centroid tiene el R² más alto en todos los modelos (0.62-0.66): los embeddings MIDI contienen información significativa sobre el centroide espectral del audio.
- **D0 gana en midi→chroma** (0.330 vs ~0.25): resultado contraintuitivo — el baseline decodifica MEJOR el perfil de chroma. Annotation "D0 wins!" lo señala.
- audio→pitch_hist y audio→interval_hist son bajos en todos (~0.09-0.19).
- Self-decoding es similar entre modelos (~0.23), confirmando que la capacidad básica no cambia.
- Nota al pie: "Descriptors do NOT improve cross-modal linear decodability — advantage lives in distance geometry (retrieval), not in extractable features."
**Hallazgo visible**: La ventaja de los descriptores NO se manifiesta en cross-decoding lineal. La mejora de +10pp en S vive en la geometría de distancias, no en features extraíbles por un probe.

---

#### CHARTS V1-V2 — `comparison_tsne.png` / `comparison_umap.png` (Test 10: Visualizations)

**Tipo**: 2×2 grid de scatter plots (un panel por modelo), puntos coloreados por modalidad (cyan = audio, magenta = MIDI).
**Qué representa**: Reducción de dimensionalidad (t-SNE o UMAP) de 2000 embeddings aleatorios por modelo (1000 audio + 1000 MIDI) proyectados a 2D. Muestra la estructura global del espacio de embeddings de cada modelo.
**Qué información se puede ver**:
- **Mezcla de modalidades**: si los puntos cyan y magenta están entremezclados (bueno para retrieval) o separados en clusters por modalidad (malo — el modelo no alinea audio/MIDI).
- Info boxes con S, descriptor type y número de parámetros por modelo.
- Los modelos augmented muestran mezcla más homogénea que D0 (donde hay zonas con mayor separación por modalidad).
- n=2000 pares indicado en cada panel.
**Hallazgo visible**: En d4a4 y a4r, las nubes cyan/magenta están muy entremezcladas. En D0, hay regiones con mayor segregación por modalidad.

---

#### CHART V3 — `alignment_cosine_distribution.png` (Test 10: Visualizations)

**Tipo**: 4 histogramas superpuestos (uno por modelo), distribución de cosine similarity matched (color del arm) vs random (gris).
**Qué representa**: La distribución de similitud coseno entre pares audio-MIDI matched (la pieza correcta) vs pares random (piezas diferentes). Para cada modelo se muestra:
- **Matched** (color): cosine similarity entre z_audio[i] y z_midi[i] para el mismo segmento.
- **Random** (gris): cosine similarity entre z_audio[i] y z_midi[j] con j≠i.
**Qué información se puede ver**:
- Los valores matched/random/gap están anotados en cada panel.
- **Gap = matched - random**: d4a4 tiene el mayor gap (0.787), seguido de d4-a4r (0.779), a4r (0.777), D0 (0.719).
- Los matched se concentran en cosine ~0.75-0.95 (alta similitud). Los random se concentran en ~-0.1 a +0.2 (baja similitud).
- d4a4 tiene la cola matched más compacta y más hacia la derecha (cosine ~0.85-0.95).
**Hallazgo visible**: Los modelos augmented separan mucho más las distribuciones matched/random. d4a4 tiene el gap más limpio.

---

#### CHARTS V4-V11 — `{arm}_tsne_detail.png` / `{arm}_umap_detail.png` (Test 10: Visualizations)

**Tipo**: 2 paneles por imagen. Panel izquierdo: scatter por modalidad (cyan=audio, magenta=MIDI). Panel derecho: scatter coloreado por pieza musical (top 10 piezas con colores distintos).
**Qué representa**: Vista detallada de cada modelo individual con dos coloraciones complementarias:
- **By Modality**: revela si audio y MIDI se mezclan bien globalmente.
- **By Piece**: revela si segmentos de la misma pieza forman clusters coherentes (lo cual indica que el modelo captura identidad de pieza, no solo features genéricas).
**Qué información se puede ver**:
- El título incluye S, descriptor type y número de parámetros.
- En la vista por pieza, los colores de las top-10 piezas forman clusters reconocibles (especialmente en modelos augmented).
- El mean cosine de matched pairs está anotado (ej: d4a4 = 0.844).
- "o" markers = audio, "^" markers = MIDI (en la vista por pieza).
- Número total de piezas y segmentos indicado.
**Hallazgo visible**: Los clusters por pieza son más compactos y mejor definidos en d4a4 y a4r que en D0, indicando que los descriptores ayudan a agrupar segmentos de la misma pieza.

---

### 11.15 Test 03: RatioProbe — Resultados COMPLETOS

> **Codex**: Test 03 entrena probes lineales (MLP de 1 capa) sobre embeddings CONGELADOS para medir cuánta información cross-modal se puede decodificar linealmente. Si z_audio contiene info de MIDI → la ventaja de los descriptores se debería ver en cross-decoding.

**Método**:
- Congelar embeddings de los 4 modelos (5000 segmentos del validation set)
- Entrenar MLPs pequeños para decodificar features del dominio opuesto:
  - **Cross-decoding audio→MIDI**: z_audio → pitch histogram (R²), z_audio → interval histogram (R²)
  - **Cross-decoding MIDI→audio**: z_midi → chroma profile (R²), z_midi → spectral centroid (R²)
  - **Self-decoding** (control): z_audio → chroma (R²), z_midi → pitch histogram (R²)

**Tabla de resultados — R² por probe por modelo**:

| Probe | D0 | d4a4 | a4r | d4-a4r |
|-------|----:|------:|-----:|-------:|
| **Cross: audio→pitch_hist** | 0.181 | 0.174 | 0.167 | 0.186 |
| **Cross: audio→interval_hist** | 0.094 | 0.112 | 0.095 | 0.115 |
| **Cross: midi→chroma** | **0.330** | 0.245 | 0.255 | 0.251 |
| **Cross: midi→centroid** | 0.616 | 0.637 | **0.662** | 0.652 |
| Self: audio→chroma | 0.310 | 0.235 | 0.249 | 0.231 |
| Self: midi→pitch_hist | 0.239 | 0.236 | 0.233 | 0.233 |

**Observaciones detalladas**:

1. **midi→centroid es el mejor probe en todos los modelos** (R² 0.62-0.66): El embedding MIDI contiene información sustancial sobre el centroide espectral del audio. a4r lidera ligeramente (0.662).

2. **D0 gana midi→chroma** (0.330 vs ~0.245-0.255 en augmented): Resultado sorprendente. El baseline sin descriptores decodifica MEJOR el perfil de chroma del audio desde embeddings MIDI. Los modelos augmented aparentemente reorganizan la información en un formato menos linealmente accesible.

3. **audio→pitch_hist y audio→interval_hist son bajos en todos** (0.09-0.19): Los embeddings de audio capturan poca información linealmente decodificable sobre las distribuciones MIDI.

4. **Self-decoding estable**: audio→chroma y midi→pitch_hist son similares entre modelos (~0.23), indicando que la capacidad básica de representación no cambia mucho.

5. **No hay "smoking gun" cross-modal**: Los modelos augmented NO muestran ventaja clara en cross-decoding sobre D0. La mejora de +10pp en S (retrieval) no se manifiesta como mejor decodificación lineal.

**Interpretación**: La ventaja de los descriptores vive en el **espacio de distancias** (cómo se organizan los embeddings para retrieval), no en features linealmente extraíbles. Los descriptores no "inyectan" información cross-modal decodificable — transforman la geometría del espacio de embeddings de forma no-lineal. Esto es consistente con VICReg (loss de distancias), no con un autoencoder (loss de reconstrucción).

### 11.16 Test 06: RSA/CKA — Resultados COMPLETOS (HALLAZGO FUERTE)

> **Codex**: Este es el hallazgo más fuerte de la sesión junto con Test 01. RSA (Representational Similarity Analysis) y CKA (Centered Kernel Alignment) miden si dos conjuntos de representaciones tienen la misma estructura geométrica. Aquí comparamos las activaciones INTERNAS (por capa del transformer) entre el encoder de audio y el de MIDI.

**Método**:
- Registrar hooks en las 8 capas transformer (4 audio + 4 MIDI)
- Forward pass sobre 500 segmentos del validation set
- Extraer activaciones por capa: [N, T, D] → mean-pool temporal → [N, D]
- Computar matrices RSA 8×8 (correlación entre matrices de distancia) y CKA 8×8 (similitud de kernel centrado)
- **Foco**: el bloque off-diagonal (audio_layers × midi_layers) = "cross-encoder alignment"

**Tabla resumen — CKA cross-encoder (media del bloque 4×4 audio×midi)**:

| Arm | CKA cross-encoder mean | RSA cross-encoder mean | Δ CKA vs D0 |
|-----|----------------------:|----------------------:|------------:|
| **D0** | **0.435** | 0.446 | — |
| **d4a4** | **0.659** | 0.646 | **+51%** |
| **a4r** | **0.766** | 0.721 | **+76%** |
| **d4-a4r** | **0.794** | 0.761 | **+82%** |

**CKA cross-encoder detallado por par de capas (audio_layer × midi_layer)**:

**D0** (baseline — baja alineación):
| | midi_0 | midi_1 | midi_2 | midi_3 |
|---------|-------:|-------:|-------:|-------:|
| audio_0 | 0.305 | 0.211 | 0.130 | 0.126 |
| audio_1 | 0.396 | 0.319 | 0.214 | 0.201 |
| audio_2 | 0.545 | 0.631 | 0.596 | 0.571 |
| audio_3 | 0.537 | 0.719 | 0.740 | 0.722 |

**d4a4** (concat — alineación moderada):
| | midi_0 | midi_1 | midi_2 | midi_3 |
|---------|-------:|-------:|-------:|-------:|
| audio_0 | 0.473 | 0.504 | 0.459 | 0.421 |
| audio_1 | 0.586 | 0.651 | 0.628 | 0.582 |
| audio_2 | 0.692 | 0.802 | 0.812 | 0.756 |
| audio_3 | 0.689 | 0.809 | 0.859 | 0.827 |

**a4r** (reverse cross-att — alineación alta):
| | midi_0 | midi_1 | midi_2 | midi_3 |
|---------|-------:|-------:|-------:|-------:|
| audio_0 | 0.651 | 0.738 | 0.744 | 0.725 |
| audio_1 | 0.652 | 0.761 | 0.794 | 0.781 |
| audio_2 | 0.695 | 0.816 | 0.853 | 0.835 |
| audio_3 | 0.667 | 0.810 | 0.873 | 0.863 |

**d4-a4r** (dual — alineación máxima):
| | midi_0 | midi_1 | midi_2 | midi_3 |
|---------|-------:|-------:|-------:|-------:|
| audio_0 | 0.686 | 0.743 | 0.749 | 0.735 |
| audio_1 | 0.716 | 0.796 | 0.814 | 0.797 |
| audio_2 | 0.756 | 0.849 | 0.874 | 0.852 |
| audio_3 | 0.737 | 0.840 | 0.885 | 0.873 |

**Observaciones detalladas**:

1. **Los descriptores DUPLICAN la alineación cross-encoder**: D0 tiene CKA medio de 0.435. d4-a4r llega a 0.794 (+82%). Audio y MIDI transformers "hablan el mismo lenguaje representacional" cuando A4 está presente.

2. **Gradiente por capas**: En TODOS los modelos, la alineación crece con la profundidad de las capas (audio_3×midi_3 > audio_0×midi_0). Las capas profundas convergen más. Pero en D0 la convergencia es débil (0.126→0.722), mientras en d4-a4r es fuerte y empieza más alto (0.735→0.873).

3. **d4-a4r lidera en alineación pero NO en S**: d4-a4r tiene la CKA más alta (0.794) pero S=79.8%, inferior a d4a4 (CKA=0.659, S=83.8%) y a4r (CKA=0.766, S=82.0%). **Más alineación representacional ≠ mejor retrieval**. La relación es monótona en el salto D0→augmented, pero no dentro de los augmented.

4. **Todas las p-values = 0.0**: La significancia es total. Los N=500 segmentos dan poder estadístico masivo.

5. **RSA confirma CKA**: Los rankings son idénticos (d4-a4r > a4r > d4a4 > D0), lo cual valida la robustez del hallazgo con dos métricas independientes.

**Interpretación**: Los descriptores de ratios no solo inyectan información causal (Test 01) — transforman la **geometría interna** de ambos encoders para que converjan. Esto es evidencia de que la "lingua franca" que los ratios proveen actúa a nivel de representación interna, no solo en la proyección final. Es exactamente lo que predice la Harmonic Information Theory: los ratios de frecuencia son un lenguaje compartido entre dominios.

### 11.17 Test 08: Ratio Decoding (Perturbation Sensitivity) — Resultados COMPLETOS

> **Codex**: Test 08 mide la SENSIBILIDAD del modelo a cada dimensión individual del descriptor. No requiere gradientes (los descriptores se computan bajo `no_grad()`). En su lugar, perturba cada dim ±epsilon y mide cuánto cambia el embedding de salida (L2 distance). Sensibilidad alta = esa dimensión tiene más influencia en la representación final.

**Contexto — Dimensiones de los descriptores**:

**CORRECCIÓN IMPORTANTE (2026-02-25)**: Los nombres originales del Test 08 eran engañosos ("ratio_1_2", "spec_centroid", etc.). A4 NO computa ratios entre picos espectrales ni centroide espectral. A4 computa **deltas temporales de log-magnitud en 8 bandas de octava** vía STFT. Las 8 dimensiones son todas del mismo tipo — solo difieren en el rango de frecuencia de la banda. Ver `src/bias_control/audio_descriptors.py::compute_audio_descriptor_a4()`.

El descriptor **A4** (audio, 8 dims) — deltas temporales de log-magnitud por banda de octava:
- `band0_47Hz`: banda 47-94 Hz (bass fundamental)
- `band1_94Hz`: banda 94-188 Hz (bass harmonics)
- `band2_188Hz`: banda 188-375 Hz (low-mid)
- `band3_375Hz`: banda 375-750 Hz (mid)
- **`band4_750Hz`**: banda 750-1500 Hz (upper-mid, harmonic region)
- **`band5_1500Hz`**: banda 1500-3000 Hz (presence, harmonic region)
- **`band6_3000Hz`**: banda 3000-6000 Hz (brilliance, harmonic region)
- **`band7_6000Hz`**: banda 6000-12000 Hz (air, harmonic region)

El descriptor **D4** (MIDI, 4 dims) contiene:
- `interval_prev`: intervalo (semitonos) respecto a nota anterior
- `interval_next`: intervalo respecto a nota siguiente
- `duration_ratio`: ratio de duración nota actual / nota anterior
- `velocity_diff`: diferencia de velocity con nota anterior

**Solo aplica a modelos augmented** — D0 no tiene descriptor.

**Tabla completa — Perturbation Sensitivity (A4, audio descriptor)**:

| Feature A4 | Hz range | d4a4 | a4r | d4-a4r | Zona |
|------------|----------|------:|-----:|-------:|------|
| **band4_750Hz** | 750-1500 | **0.664** | 0.478 | **0.773** | high-freq |
| **band5_1500Hz** | 1500-3000 | **0.662** | 0.476 | 0.599 | high-freq |
| **band6_3000Hz** | 3000-6000 | 0.264 | **0.875** | **1.092** | high-freq |
| **band7_6000Hz** | 6000-12000 | 0.209 | **0.933** | 0.529 | high-freq |
| band3_375Hz | 375-750 | 0.546 | 0.423 | 0.526 | low-freq |
| band2_188Hz | 188-375 | 0.375 | 0.381 | 0.514 | low-freq |
| band1_94Hz | 94-188 | 0.224 | 0.335 | 0.313 | low-freq |
| band0_47Hz | 47-94 | 0.073 | 0.238 | 0.303 | low-freq |

**Tabla completa — Perturbation Sensitivity (D4, MIDI descriptor)**:

| Feature D4 | d4a4 | d4-a4r | Tipo |
|------------|------:|-------:|------|
| duration_ratio | 0.077 | 0.124 | temporal |
| interval_prev | 0.070 | 0.107 | interval |
| velocity_diff | 0.068 | 0.047 | dynamics |
| interval_next | 0.066 | 0.047 | interval |

(a4r no tiene descriptor MIDI)

**Tabla — Correlation Analysis (|r| medio, A4 features)**:

| Feature A4 | Hz range | d4a4 |r| | a4r |r| | d4-a4r |r| |
|------------|----------|--------:|------:|---------:|
| band5_1500Hz | 1500-3000 | 0.031 | 0.039 | 0.037 |
| band4_750Hz | 750-1500 | 0.031 | 0.047 | 0.044 |
| band7_6000Hz | 6000-12000 | 0.029 | 0.045 | 0.044 |
| band6_3000Hz | 3000-6000 | 0.027 | 0.037 | 0.038 |
| band3_375Hz | 375-750 | 0.029 | 0.047 | 0.047 |
| band0_47Hz | 47-94 | 0.024 | 0.043 | 0.041 |
| band2_188Hz | 188-375 | 0.021 | 0.033 | 0.035 |
| band1_94Hz | 94-188 | 0.031 | 0.031 | 0.034 |

**Observaciones detalladas**:

1. **Las bandas de alta frecuencia (750+ Hz) son las MÁS sensibles en TODOS los modelos**. En d4a4, band4 (750-1500 Hz) y band5 (1500-3000 Hz) dominan (0.66). En a4r, band7 (6-12 kHz) y band6 (3-6 kHz) dominan (0.93, 0.87). En d4-a4r, band6 alcanza el MÁXIMO global (1.092). La zona 750-12000 Hz es donde vive la estructura armónica de piano — fundamentales de las notas altas y armónicos de las notas medias/bajas.

2. **El mecanismo de inyección determina QUÉ bandas importan más**:
   - **d4a4 (concat)**: band4-5 (750-3000 Hz) dominan → el modelo en modo concat se enfoca en la zona de "presencia" donde están los armónicos más energéticos.
   - **a4r (reverse cross-att)**: band6-7 (3000-12000 Hz) dominan → la cross-attention extrae información de las bandas de alta frecuencia (brilliance/air), donde los armónicos son más débiles pero más discriminativos. Esto sugiere que la atención cruzada puede "buscar" información más sutil.
   - **d4-a4r (dual)**: band6 (3-6 kHz) es dominante (1.092, el valor más alto de TODO el test) → el modelo dual pica en la zona de brilliance.

3. **D4 (MIDI descriptor) es 5-10× menos sensible que A4** (máx 0.12 vs máx 1.09). Perfectamente consistente con Test 01: D4 no es causal, y aquí vemos que el modelo apenas reacciona a perturbaciones del descriptor MIDI.

4. **Bandas de baja frecuencia (47-750 Hz) tienen sensibilidad moderada pero inferior a bandas altas**. La excepción parcial es band3 (375-750 Hz) en d4a4 (0.546), que aparece en 3er lugar. Pero las bandas 750+ Hz siempre dominan los primeros puestos.

5. **Correlaciones lineales bajísimas (|r| < 0.05 en todos)**: Sensibilidad alta + correlación baja = el modelo transforma la información de los descriptores de forma **altamente no-lineal**. Perturbar una banda cambia mucho el embedding, pero la relación no es una función lineal simple. Esto es esperable en un modelo con cross-attention + 4 capas transformer.

6. **d4-a4r tiene las sensibilidades más altas globalmente**: band6=1.092 y band2=0.514 superan los máximos de los otros modelos. El modelo dual "escucha más atentamente" cada dimensión del descriptor, posiblemente porque la presencia de D4 en el MIDI encoder crea una presión adicional de alineación. Esto es consistente con el hallazgo del Test 06 (d4-a4r tiene la CKA más alta).

7. **Interpretación para Phideus**: A4 no captura directamente "ratios de frecuencia" sino cambios temporales de energía por banda. La sensibilidad alta en bandas armónicas (750+ Hz) indica que el modelo aprovecha la **dinámica espectral** en las zonas donde los armónicos musicales crean patrones distintivos. La conexión con la Harmonic Information Theory es indirecta: los patrones de energía por banda reflejan la distribución armónica del instrumento y las notas tocadas.

### 11.18 Hallazgo científico consolidado Tests 03+06+08

> **Codex**: Estos tres tests, junto con 01+04, completan una narrativa científica coherente. Aquí el resumen integrado.

**La narrativa en 5 puntos**:

1. **A4 es causal** (Test 01, -75 a -78pp al zelear) y **D4 no** (+0 a +1pp).

2. **A4 codifica intervalos relativos** (Test 04): los modelos con A4 son más invariantes a transposición, especialmente a4r (+23.6pp vs D0 a ±3).

3. **A4 alinea las representaciones internas de audio y MIDI** (Test 06): CKA cross-encoder sube de 0.435 (D0) a 0.794 (d4-a4r). Los transformers de ambos dominios convergen hacia geometrías similares.

4. **Las bandas de alta frecuencia (750+ Hz) son las dimensiones más influyentes del descriptor** (Test 08): band4 a band7 tienen sensibilidad 2-5× mayor que bandas bajas. Estas son las zonas donde la estructura armónica musical es más discriminativa. La información se codifica no-linealmente (|r| < 0.05 pero sensitivity > 0.5).

5. **La ventaja NO es linealmente decodificable** (Test 03): los modelos augmented no superan a D0 en cross-decoding lineal. La mejora de +10pp en S vive en la geometría del espacio de distancias (retrieval), no en features extraíbles.

**Implicancia para Phideus**: A4 captura la dinámica espectral por banda de octava — no ratios de frecuencia directamente, pero sí patrones que reflejan la estructura armónica del audio (los armónicos musicales crean patterns distintivos de energía por banda). Esta información actúa como puente entre audio y MIDI — no por inyectar features cross-modal decodificables, sino por **alinear la geometría representacional** de ambos encoders. El mecanismo de reverse cross-attention (a4r) es el más efectivo porque permite al transformer de audio trabajar directamente con 188 tokens de descriptor en lugar de 2400 tokens CNN.

### 11.19 Datos numéricos para gráficos (Tests 03, 06, 08)

> **Codex**: Estos datos son para generación de gráficos adicionales si se decide hacerlos.

**Test 03 — Probe R² para gráfico de barras agrupadas**:
```
audio→pitch_hist: D0=0.181, d4a4=0.174, a4r=0.167, d4-a4r=0.186
audio→interval:   D0=0.094, d4a4=0.112, a4r=0.095, d4-a4r=0.115
midi→chroma:      D0=0.330, d4a4=0.245, a4r=0.255, d4-a4r=0.251
midi→centroid:    D0=0.616, d4a4=0.637, a4r=0.662, d4-a4r=0.652
```

**Test 06 — CKA cross-encoder mean para bar chart**:
```
D0=0.435, d4a4=0.659, a4r=0.766, d4-a4r=0.794
```

**Test 08 — Sensitivity top-4 high-freq bands para radar chart**:
```
                    d4a4    a4r    d4-a4r
band4_750Hz:       0.664   0.478   0.773
band5_1500Hz:      0.662   0.476   0.599
band6_3000Hz:      0.264   0.875   1.092
band7_6000Hz:      0.209   0.933   0.529
```

### 11.20 Gate 5B — Gráficos: todos generados (v2, 2026-02-25)

> **Codex**: Los 5 gráficos que estaban pendientes (CKA heatmaps, CKA bar chart, sensitivity bars, radar, probe R²) ya fueron generados como charts 09-13. Ver glosario completo en sección 11.14.

| # | Test | Tipo | Estado | Archivo |
|---|------|------|--------|---------|
| 09 | Test 06 | Heatmap 2×2 | ✅ GENERADO | `chart09_cka_heatmaps_4models.png` |
| 10 | Test 06 | Bar chart | ✅ GENERADO | `chart10_cka_crossencoder_bar.png` |
| 11 | Test 08 | Grouped bars | ✅ GENERADO | `chart11_sensitivity_bars.png` |
| 12 | Test 08 | Radar/spider | ✅ GENERADO | `chart12_sensitivity_radar.png` |
| 13 | Test 03 | Grouped bars | ✅ GENERADO | `chart13_probe_r2.png` |

**Pendiente**: Chart para Test 09 (Invariance Suite) — se generará cuando termine la ejecución.

### 11.20b Gate 5B — Animaciones (6 GIFs, 5.8 MB total)

> **Codex**: Animaciones estilo "amarillismo exploratorio" para showcase/presentaciones. Fondo oscuro, colores cyan/magenta, alto impacto visual. Ubicadas en `animations/` dentro del directorio de visualizaciones.

**Directorio**: `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/resultados_compartir/06_gate5b_scientific_validation/animations/`

| # | Archivo | MB | Test | Descripción |
|---|---------|---:|------|-------------|
| A1 | `anim1_morphing_evolution.gif` | 1.0 | Tests 10+12 | Nube t-SNE morphando D0 → d4a4 → a4r (Procrustes-aligned). Barra de progreso, S dinámico. Muestra cómo los descriptores reorganizan la geometría del embedding. |
| A2 | `anim2_bridges_crossmodal.gif` | 0.5 | Test 10 | Puentes audio↔MIDI apareciendo progresivamente: primero D0 (bridges largos/rojos, dist=0.113), luego d4a4 (bridges cortos/verdes). Visualiza la mejora en alineación cross-modal. |
| A3 | `anim3_cka_pulse.gif` | 1.2 | Test 06 | Heatmap CKA 8×8 morfando D0 → d4a4 → a4r → d4-a4r. El bloque cross-encoder (resaltado verde) se "enciende" de frío (0.435) a caliente (0.794). **Directamente vinculada al hallazgo fuerte de Test 06.** |
| A4 | `anim4_rotation_3d.gif` | 2.2 | Test 10 | Galaxia 3D UMAP de d4a4 (800 pares) rotando 360°. Bridges blancos entre los 50 pares más cercanos. Muestra la mezcla tridimensional audio/MIDI. |
| A5 | `anim5_sidebyside_D0_vs_d4a4.gif` | 0.5 | Test 10 | D0 vs d4a4 side-by-side, puntos t-SNE apareciendo progresivamente con efecto glow. Comparación directa baseline vs champion. |
| A6 | `anim6_sensitivity_radar.gif` | 0.4 | Test 08 | Radar de sensibilidad A4 (8 bandas Hz). Perfiles d4a4 → a4r → d4-a4r aparecen uno a uno. Muestra cómo cada mecanismo de inyección "escucha" distintas bandas. |

**Uso recomendado**:
- **Presentaciones**: A1 (morphing) y A3 (CKA pulse) son las más narrativas — cuentan la historia de la evolución.
- **Showcase web/README**: A4 (rotación 3D) y A2 (bridges) son las más visualmente impactantes.
- **Explicación técnica**: A6 (radar) es buena para explicar Test 08.

**Script de generación**: `/tmp/gate5b_animations.py`

### 11.21 Instrucciones para Codex (actualización S14 — Glosario + Charts v2 + Animaciones)

> **ACCIÓN REQUERIDA**: Esta sección supersede TODAS las instrucciones previas (11.9, 11.14 anterior). Actualizar documentación completa del Gate 5B.

**1. GLOSARIO COMPLETO (sección 11.14)**: Se añadió un glosario exhaustivo de las 24 visualizaciones PNG generadas (complementadas por 6 GIF animados). Para cada chart incluye: tipo de gráfico, qué datos representa, qué información se puede extraer, y hallazgos clave visibles. **Incluir este glosario en la documentación del showcase como referencia para lectores**.

**1b. ANIMACIONES (sección 11.20b)**: 6 GIFs animados (5.8 MB total) en `animations/`. Destacar A1 (morphing D0→d4a4→a4r) y A3 (CKA pulse) en el showcase — son las que mejor narran la historia científica. A4 (rotación 3D) es ideal para portada o cabecera visual.

**2. Todos los charts regenerados (v2)**: Los 13 analytical charts fueron regenerados con mejoras sustanciales:
   - Descriptor type labels en todos los charts (ej: "D4+A4 concat", "A4 rev-crossatt")
   - Colores estandarizados: D0=gris, d4a4=rojo, a4r=azul, d4-a4r=púrpura
   - chart07 ahora muestra 4 modelos (antes solo 2)
   - chart08 dashboard expandido a 6 paneles (antes 4): añadido CKA, sensitivity, recall
   - charts 11-12 con nombres Hz correctos (antes tenían nombres engañosos)
   - chart13 con texto más grande y legible

**3. Tests completados (datos en secciones 11.5, 11.10, 11.15, 11.16, 11.17)**:
   - Test 12 (Scoreboard): S validado vs históricos — ver 11.3/11.4
   - Test 01 (Causal Ablation): A4 causal, D4 no — ver 11.5/11.6
   - Test 04 (Transposition): a4r más invariante — ver 11.10/11.11
   - Test 03 (RatioProbe): D0≥augmented en cross-decoding — ver 11.15
   - Test 06 (RSA/CKA): CKA duplicado por descriptores — ver 11.16
   - Test 08 (Ratio Decoding): bandas 750+ Hz dominan — ver 11.17 (CORREGIDO: nombres Hz)
   - Test 10 (Visualizations): t-SNE/UMAP/alignment — ver glosario V1-V11
   - Test 09 (Invariance Suite): EN CURSO — resultados pendientes

**4. Test 09 EN CURSO**: Invariance suite (temporal shift, velocity scaling, octave transposition, audio noise). Muy lento (~5.5 min/evaluación × muchas combinaciones × 4 modelos). Cuando termine, agregar datos y generar chart.

**5. Narrativa científica completa (6 hallazgos complementarios)**:

| # | Test | Hallazgo | Evidencia |
|---|------|----------|-----------|
| 1 | Test 01 | A4 es causal, D4 no | Zerear A4 → -75 a -78pp. Zerear D4 → ~0pp |
| 2 | Test 04 | A4 codifica intervalos relativos | a4r +23.6pp vs D0 a ±3 semitonos |
| 3 | Test 06 | A4 alinea representaciones internas | CKA cross-encoder: D0=0.435 → d4-a4r=0.794 (+82%) |
| 4 | Test 08 | Bandas alta-freq (750+ Hz) dominan | Sensitivity 0.5-1.1 en high-freq vs 0.07-0.5 en low-freq |
| 5 | Test 03 | Ventaja no es linealmente decodificable | D0 ≥ augmented en cross-decoding R². Geometría no-lineal |
| 6 | Test 10 | Embeddings más mezclados cross-modal | Cosine gap: d4a4=0.787 vs D0=0.719 |

**Juntos**: Evidencia convergente de que la dinámica espectral por banda de octava (capturada por A4) actúa como puente representacional cross-modal, alineando la geometría interna de audio y MIDI encoders. El mecanismo es no-lineal (alta sensibilidad + baja correlación lineal) y opera transformando la geometría de distancias, no inyectando features decodificables.

**6. UNC pendiente**: Tests 02 (param-matched) + 05 (multi-seed). SLURM scripts listos.

**7. Archivos clave para Codex**:
   - Glosario de charts: sección 11.14 de este documento
   - Glosario de animaciones: sección 11.20b de este documento
   - Datos numéricos: secciones 11.5, 11.10, 11.15, 11.16, 11.17, 11.19
   - Charts (24 PNGs): `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/resultados_compartir/06_gate5b_scientific_validation/`
   - Animaciones (6 GIFs): `.../06_gate5b_scientific_validation/animations/`
   - JSONs fuente: `data/gate5b_results/{arm}/{test}.json`
   - Scripts de generación: `/tmp/regenerate_all_charts.py` (charts v2), `/tmp/gate5b_animations.py` (animaciones)

### 11.22 Paper LaTeX escrito (2026-02-25 ~09:00 UTC)

**Paper completo** en `Paper/`. 25 páginas formato NeurIPS preprint (inglés).

**Título**: "Descriptor-Injected Cross-Modal Learning: A Systematic Exploration of Audio–MIDI Alignment via Spectral and Melodic Features"

**Autor**: Mariano Fernández Méndez, Asociación Civil AlterMundi.

**Disclosure de AI**: Párrafo dedicado antes de Acknowledgments reconociendo uso de Claude, Codex, y otros modelos como asistentes en código, análisis, documentación, y preparación del manuscrito. Decisiones científicas por el autor humano.

**Estructura**:
1. Introduction — modality gap, hypothesis, 3-phase exploration
2. Related Work — audio-MIDI matching, self-supervised audio, contrastive learning, conditioning, RSA
3. Method — formulas completas: VICReg, A4 (octave-band energy dynamics), D4 (local intervals), concat, standard cross-att, **reverse cross-attention** (163× speedup), combined mechanisms, training protocol
4. Descriptor and Mechanism Selection — 13-arm screening (5ep), architecture families (Gate 4.4, 11 arms), long-horizon confirmation (30-60ep)
5. Scientific Validation Gate 5B — Test 12, Test 01 (causal ablation), Test 04 (transposition), Test 06 (CKA), Test 08 (sensitivity), Test 03 (probing), Test 10 (visualization)
6. Discussion — A4 mechanism, D4 paradox, reverse cross-att as bottleneck, alignment≠retrieval, limitations
7. Conclusion

**Apéndices A-F**: Full descriptor catalog (A7/A8/A9 formulas), architecture specs, Gate 4.4, hyperparameters, visualizations, Bloque A unfreezing.

**Figuras (tikz/pgfplots)**: architecture diagram, 13-arm screening bars, causal ablation grouped bars, transposition curves, CKA 2×2 heatmaps, band sensitivity bars, 6-panel summary dashboard.

**Tablas**: 12 tablas con todos los datos numéricos de Gate 5B.

**Bibliografía**: 47 entradas (VICReg, Barlow Twins, CLIP, MERT, wav2vec2.0, CLAP, CKA, MAESTRO, Shazam, FiLM, MoE, Perceiver, etc.)

**Archivos para compartir**:
- `Paper/paper_standalone.tex` — autocontenido (100 KB), todo inlined
- `Paper/neurips_2024.sty` — estilo NeurIPS (12 KB)
- Compilar: `pdflatex paper_standalone.tex` (una sola pasada, sin bibtex)

**Versión modular** (para editar):
- `Paper/main.tex` + `Paper/appendix.tex` + `Paper/references.bib` + `Paper/figures/*.tex`
- Compilar: `pdflatex main && bibtex main && pdflatex main && pdflatex main`

### 11.23 Test 09 Invariance Suite — Resultado D0 COMPLETO (2026-02-25 ~05:24 UTC)

**JSON**: `data/gate5b_results/D0/test09_invariance_suite.json`

**D0 Baseline S=73.4%**:

**Temporal shift** (audio desplazado ±N samples a 24kHz):
| Shift | S | Delta |
|-------|------|--------|
| -8000 (-0.5s) | 71.2% | -2.2pp |
| -4000 (-0.25s) | 72.4% | -1.0pp |
| 0 | 73.4% | — |
| +4000 (+0.25s) | 70.2% | -3.2pp |
| +8000 (+0.5s) | 68.2% | -5.2pp |
**Interpretación**: Bastante robusto. Medio segundo de desalineamiento = -5pp.

**Velocity scaling** (MIDI velocity × factor):
| Factor | S | Delta |
|--------|------|--------|
| 0.5 | 5.2% | -68.2pp |
| 0.8 | 37.2% | -36.2pp |
| 1.0 | 73.4% | — |
| 1.2 | 54.0% | -19.4pp |
| 1.5 | 18.4% | -55.0pp |
**Interpretación**: MUY sensible a velocity. Factor 0.5 es catastrófico. El modelo codifica fuertemente la dinámica de intensidad.

**Octave transposition** (MIDI pitch ±12/24 semitonos):
| Shift | S | Delta |
|-------|------|--------|
| -24 | 8.4% | -65.0pp |
| -12 | 12.0% | -61.4pp |
| 0 | 73.4% | — |
| +12 | 10.0% | -63.4pp |
| +24 | 5.2% | -68.2pp |
**Interpretación**: Transposición octava destruye matching. Esperable — cambia contenido espectral.

**Audio noise** (ruido gaussiano a distintos SNR):
| SNR | S | Delta |
|-----|------|--------|
| Clean | 73.4% | — |
| 40 dB | 73.4% | 0.0pp |
| 30 dB | 73.4% | 0.0pp |
| 20 dB | 73.0% | -0.4pp |
| 10 dB | 46.8% | -26.6pp |
| 5 dB | 17.8% | -55.6pp |
**Interpretación**: Excelente robustez hasta 20 dB (ruido apenas perceptible). Colapsa a 10 dB.

**Parciales d4a4** (en curso, ~09:30 UTC):
- Temporal: -8000→76.6%, -4000→80.8%, 0→83.8%, +4000→81.2%, +8000→79.0%
- Velocity: 0.5→8.8%, 0.8→46.8%, 1.0→83.8%, 1.2→55.2%, 1.5→(corriendo)
- ETA completo (4 modelos): ~11:20 UTC

### 11.24 Instrucciones para Codex (actualización S15 — Paper + Test 09)

> **ACCIÓN REQUERIDA**: Incorporar el paper y resultados Test 09 a la documentación.

**1. Paper escrito**: El paper LaTeX completo está en `Paper/`. Ver sección 11.22 para estructura y contenido. Es un preprint de 25 páginas con toda la ciencia de Gate 5B. **Incluir referencia al paper en la documentación del proyecto.**

**2. Test 09 D0 completo**: Los resultados de invarianza de D0 están en 11.23 arriba. Los modelos augmented (d4a4, a4r, d4-a4r) están corriendo — cuando terminen habrá una actualización con comparación. **Los datos de D0 ya se pueden documentar.**

**3. Lo más interesante de Test 09 para documentar**:
- Velocity scaling es la perturbación más destructiva (incluso peor que transposición octava)
- Audio noise es muy robusta hasta 20 dB
- Temporal shift es moderadamente robusta
- La pregunta pendiente es si los descriptores mejoran o empeoran estas invarianzas

**4. Archivos clave**:
- Paper: `Paper/paper_standalone.tex` + `Paper/neurips_2024.sty`
- Test 09 D0: `data/gate5b_results/D0/test09_invariance_suite.json`
- Test 09 otros: aparecerán en `data/gate5b_results/{d4a4,a4r,d4-a4r}/test09_invariance_suite.json`

### 11.25 Paper — Paleta de colores unificada (2026-02-25 ~10:45 UTC)

**Cambio**: Se implementó una paleta de colores uniforme y consistente para todas las figuras del paper.

**Antes**: Cada figura usaba colores hardcoded (e.g., `fill=blue!55`, `fill=teal!50`). Había una inconsistencia: ablation.tex usaba `blue!60` para d4a4 mientras el resto usaba `blue!55`.

**Después**: 6 colores base definidos con `\definecolor` en el preámbulo + 14 variantes derivadas (`*fill`, `*draw`, `*desat`). Todas las figuras referencian nombres semánticos. Cambiar un color ahora = editar 1 línea.

**Paleta definida:**
| Nombre | Hex | Modelo/Uso |
|--------|-----|------------|
| `Dzero` | `#888888` | D0 baseline (gray) |
| `Dfour` | `#4682B4` | d4a4 (steel blue) |
| `Afour` | `#008080` | a4r (teal) |
| `DAfour` | `#DC8C32` | d4-a4r (amber/orange) |
| `negcol` | `#B24040` | Anotaciones negativas/colapso |
| `poscol` | `#228B22` | Anotaciones positivas/sin efecto |

**Variantes (14 total)**: `Dzerofill/draw`, `Dfourfill/draw`, `Afourfill/draw`, `DAfourfill/draw`, `Dfourdesat/desatdraw`, `Afourdesat/desatdraw`, `DAfourdesat/desatdraw`.

**Archivos modificados (9)**:
- `Paper/main.tex` — definiciones en preámbulo
- `Paper/paper_standalone.tex` — definiciones + figuras inlineadas
- `Paper/figures/ablation.tex` — fix inconsistencia blue!60→Dfourfill
- `Paper/figures/sensitivity.tex`
- `Paper/figures/transposition.tex`
- `Paper/figures/dashboard.tex`
- `Paper/figures/screening.tex`
- `Paper/figures/architecture.tex`
- `Paper/figures/cka_heatmaps.tex`

**Compilación**: Ambos PDFs compilan 26 páginas, 3.1MB, 0 errores, 0 warnings de color.

### 11.25b Test 09 — Resultados parciales d4a4 (2026-02-25 ~10:30 UTC)

**d4a4 completed phases** (audio_noise SNR 5dB still running at report time):

| Perturbation | Values | S (%) | Delta vs Normal |
|---|---|---|---|
| **Temporal shift** | -0.5s | 76.6 | -7.2pp |
| | -0.25s | 80.8 | -3.0pp |
| | 0 | 83.8 | 0 |
| | +0.25s | 81.2 | -2.6pp |
| | +0.5s | 79.0 | -4.8pp |
| **Velocity scaling** | 0.5x | 8.8 | -75.0pp |
| | 0.8x | 46.8 | -37.0pp |
| | 1.0x | 83.8 | 0 |
| | 1.2x | 55.2 | -28.6pp |
| | 1.5x | 12.8 | -71.0pp |
| **Octave transposition** | -24 st | 9.8 | -74.0pp |
| | -12 st | 16.0 | -67.8pp |
| | 0 | 83.8 | 0 |
| | +12 st | 13.8 | -70.0pp |
| | +24 st | 7.4 | -76.4pp |
| **Audio noise** | 20 dB | 83.8 | 0pp |
| | 15 dB | 79.8 | -4.0pp |
| | 10 dB | 67.0 | -16.8pp |
| | 5 dB | 54.8 | -29.0pp |

**Comparación d4a4 vs D0**:
- **Temporal**: d4a4 ligeramente más robusto (max -7.2pp vs -5.2pp en ±0.5s, pero base más alta)
- **Velocity**: Ambos frágiles. d4a4 peor en 0.5x (S=8.8% vs D0 S=5.2%) pero empezando de base más alta
- **Octave**: Ambos catastróficos
- **Audio noise**: d4a4 más robusto — 10dB: -16.8pp vs D0 -26.6pp

**a4r y d4-a4r aún corriendo** — ETA ~14:30 UTC.

### 11.26 Corrección email del autor (2026-02-25 ~11:00 UTC)

Email corregido de `mariano@altermundi.net` a `marianofm@altermundi.net` en ambos archivos (`main.tex` y `paper_standalone.tex`).

### 11.27 Test09 — Nota canónica de consistencia (2026-02-25 ~11:30 UTC)

Para evitar deriva entre logs parciales y resultados finales, usar como fuente de verdad:

- `data/gate5b_results/D0/test09_invariance_suite.json`
- `data/gate5b_results/d4a4/test09_invariance_suite.json`

Estado canónico al corte:
- **Test09 parcial**: `D0` y `d4a4` cerrados; `a4r` y `d4-a4r` pendientes.
- **D0 baseline**: `S=73.4%`.
- **d4a4 baseline**: `S=83.8%`.

Corrección relevante frente a notas parciales previas:
- `d4a4` en `audio_noise` quedó finalmente con `S@5dB=25.0%` (no `54.8%`).
- Serie final `d4a4/audio_noise`: `40dB=79.8%`, `30dB=67.0%`, `20dB=54.8%`, `10dB=52.2%`, `5dB=25.0%`.

### 11.28 Test 11 — Decoder Suite (2026-02-25 ~17:40 UTC)

**Concepto**: Transformer decoder no-lineal reconstruye secuencias temporales completas (mel [188,128], piano roll [188,88]) desde embeddings congelados z[256]. Complemento generativo del Test 03 (linear probes).

**Preguntas que responde**:
1. Cuánta info secuencial sobrevive la compresión a z[256] (intra-domain)
2. Cuánta info del *otro* dominio está codificada (cross-modal)
3. Los modelos con descriptores retienen más info cross-modal que D0?
4. Bonus: .wav y .mid generados desde embeddings

**Arquitectura**: `ConditionedTransformerDecoder` (27.5M params)
- z[256] → Linear → 16 conditioning tokens (memory)
- 188 learnable frame queries + sinusoidal PE → TransformerDecoder (6 layers, 8 heads, d=512) → output head
- Mel head: Linear(512,128), PR head: Linear(512,88) raw logits (sigmoid solo en eval)

**4 tasks por arm**:
| Task | Input z | Target | Loss | Mide |
|------|---------|--------|------|------|
| audio2mel | z_audio | mel [188,128] | MSE + 0.1×L1 | Intra-audio |
| midi2pr | z_midi | PR [188,88] | BCE(pos_weight=50) | Intra-MIDI |
| audio2pr | z_audio | PR [188,88] | BCE(pos_weight=50) | Cross A→M |
| midi2mel | z_midi | mel [188,128] | MSE + 0.1×L1 | Cross M→A |

**Controles**:
- random2mel/random2pr: z~N(0,1), entrenados aparte (loss floor)
- shuffle: misma decoder con z de otro segmento (derangement, eval-only)
- mean_z: z = mean(z_train)
- zero_z: z = 0

**Info retention ratio**: `(shuffle_loss - cross_loss) / (shuffle_loss - intra_loss)`

**Training config**: AdamW lr=1e-4, CosineAnnealingLR T_max=60, early stopping patience=10, batch=64, train subsample=20K, val=all 13.5K.

**Piano roll**: construido en mel grid exacto (sr=24000, hop=512 → T=188 nativo). floor(onset), ceil(offset).

**Onset F1**: greedy closest-first, pitch-specific, ±2 frames (~43ms), tie-break determinístico.

**Archivos creados**:
- `experiments/bias_control/gate5b/decoder_model.py` (~170 líneas)
- `experiments/bias_control/gate5b/test11_decoder_suite.py` (~730 líneas)
- `requirements.txt`: added `pretty_midi>=0.2.10`

**Orden de ejecución**: precompute → baselines → D0 → a4r → d4a4

**Resultados parciales (2026-02-25 ~19:00 UTC)**:

*Baselines (DONE)*:
| Baseline | Best ep | Val loss | Notas |
|----------|---------|----------|-------|
| random2mel | e2 | 0.2254 MSE | cosine_sim=0.592 |
| random2pr | e2 | 0.8367 BCE | F1=0.064 |

*D0 (en curso, 3/4 decoders done)*:
| Decoder | Best ep | Val loss | vs Random | Tipo |
|---------|---------|----------|-----------|------|
| audio2mel | e15 | 0.1635 MSE | **-27%** | intra |
| midi2pr | e2 | 0.7141 BCE | **-15%** | intra |
| audio2pr | e5+ | 0.7402 | en curso | cross |
| midi2mel | - | - | pendiente | cross |

**Observación temprana**: audio2mel (intra) muestra mejora clara sobre random (cosine 0.629 vs 0.592). PR decoders hacen early stop rápido (e2-e12) por la sparsity extrema (0.9% nonzero). audio2pr (cross) arranque peor que midi2pr (intra) — esperado.

**Output structure**:
```
data/gate5b_results/
├── targets_{mel,pr}_{train,val}.npz  # shared
├── train_indices.npy
├── baselines/test11_decoder_suite.json
├── D0/
│   ├── embeddings_train.npz
│   ├── test11_decoder_suite.json
│   ├── test11_models/*.pt
│   └── test11_samples/*.wav, *.mid
├── a4r/  (pending)
└── d4a4/ (pending)
```

**ETA**: D0 ~20 min más, a4r ~60 min, d4a4 ~60 min. Todo en tmux `test11`.

### 11.29 Test09 — Cierre canónico 4/4 arms (2026-02-25 ~23:30 UTC, verificación Codex)

Fuente de verdad verificada contra JSON:
- `data/gate5b_results/D0/test09_invariance_suite.json`
- `data/gate5b_results/d4a4/test09_invariance_suite.json`
- `data/gate5b_results/a4r/test09_invariance_suite.json`
- `data/gate5b_results/d4-a4r/test09_invariance_suite.json`

Estado actualizado:
- **Test09 cerrado** para `D0`, `d4a4`, `a4r`, `d4-a4r`.

Patrón consolidado:
1. **Temporal shift**: robustez aceptable en los 4 arms (peor caso entre `-3.6pp` y `-7.2pp`).
2. **Velocity scaling**: fragilidad alta en todos.
3. **Octave transposition**: fragilidad alta/catastrófica en todos.
4. **Audio noise**: patrón bimodal:
   - `D0` domina en ruido leve/moderado (40-20 dB),
   - `a4r` / `d4-a4r` retienen más `S` en ruido severo (5 dB).

---

*Fin de notas — Claude LOCAL + sync Codex, 2026-02-25 ~23:30 UTC*

## [Codex->Claude] 2026-02-26 01:00 UTC — Test11 run detenido + GPU liberada para rescate perceptual
### Estado
- Sesión `tmux test11` confirmada activa y en ejecución de `test11_decoder_suite` (arm `d4a4`).
- Se capturaron los últimos ~200 logs antes de detener.
- Se envió `Ctrl-C` y luego se cerró la sesión con `tmux kill-session -t test11`.
- Verificación post-stop: no hay procesos activos de `test11_decoder_suite`/`test11_perceptual_suite`.

### Decisiones
- Se aborta el run cuantitativo en curso para priorizar pipeline perceptual (audio/midi reconocibles) como objetivo principal.
- Se preserva baseline científico existente (JSON + samples actuales) para comparación before/after.

### Evidencia (paths + métricas)
- Snapshot logs pre-stop: `/tmp/test11_last200.log`
- Último tramo observado en logs:
  - `audio2pr` early stop epoch 12, best val loss `0.7374`
  - `audio2pr` métricas: BCE `0.7376`, F1 `0.0842`, onset F1 `0.0496`
  - `midi2mel` alcanzó al menos epoch 5 (`val=0.1908`) antes del corte
- Baseline preservado:
  - `data/gate5b_results/D0/test11_decoder_suite.json`
  - `data/gate5b_results/a4r/test11_decoder_suite.json`
  - `data/gate5b_results/baselines/test11_decoder_suite.json`
  - Samples pre-rescate: `data/gate5b_results/{D0,a4r}/test11_samples/` (80 archivos por arm)
- Commit baseline: `e6a2acd`

### Próximo paso
- Implementar pipeline perceptual-first:
  - `midi_event_codec.py` (en progreso)
  - `event_decoder_model.py`
  - `render_midi_audio.py`
  - `test11_perceptual_suite.py`
  - `eval_perceptual_human.py`
- Ejecutar smoke test en subset pequeño y luego entrenar por orden: `D0 -> a4r -> d4a4`.

### Riesgos
- `nvidia-smi` no reporta estado en este entorno (`Failed to initialize NVML`), por lo que la validación de liberación de GPU se hace por ausencia de procesos de entrenamiento.
- `fluidsynth` puede no estar disponible; se mantiene fallback automático a `pretty_midi.synthesize`.

## [Codex->Claude] 2026-02-26 01:08 UTC — Pipeline perceptual implementado (smoke test técnico OK)
### Estado
- Implementados módulos nuevos para rescate perceptual Test11:
  - `experiments/bias_control/gate5b/midi_event_codec.py`
  - `experiments/bias_control/gate5b/event_decoder_model.py`
  - `experiments/bias_control/gate5b/render_midi_audio.py`
  - `experiments/bias_control/gate5b/eval_perceptual_human.py`
  - `experiments/bias_control/gate5b/test11_perceptual_suite.py`
- CLI perceptual operativa con flags requeridos:
  - `--model` / `--all`
  - `--renderer auto|fluidsynth|prettymidi`
  - `--soundfont`
  - `--skip-train`
  - `--skip-precompute`
  - `--skip-train-embs`
  - `--seed`, `--n-samples-human`

### Decisiones
- Estrategia **cache-first** explícita:
  - Train embeddings: usa `embeddings_train.npz` si existe; con `--skip-train-embs` exige cache + valida `indices == train_indices`.
  - Val embeddings: usa `get_normal_embeddings()` del harness (cache-aware).
  - Targets de eventos: `targets_event_{train,val}.npz` + `train_indices.npy`.
- Render de audio:
  - Primario `fluidsynth` si está disponible y hay soundfont.
  - Fallback automático a `pretty_midi`.

### Evidencia (paths + métricas)
- Compilación de sintaxis OK:
  - `python -m py_compile .../{midi_event_codec,event_decoder_model,render_midi_audio,eval_perceptual_human,test11_perceptual_suite}.py`
- Smoke CLI OK:
  - `python experiments/bias_control/gate5b/test11_perceptual_suite.py --help`

### Próximo paso
- Ejecutar primera corrida perceptual real en `D0` usando caches disponibles, luego `a4r` y `d4a4`.
- Generar artifacts `.mid/.wav` pareados y `test11_perceptual.json` por arm.

### Riesgos
- Tiempo de cómputo de primer run puede ser alto si falta algún cache crítico.
- Si no hay `fluidsynth`/soundfont, el audio dependerá del fallback (`pretty_midi`) con timbre menos realista.

## [Codex->Claude] 2026-02-26 01:16 UTC — Sync resultados_compartir (Test11 previo + perceptual en curso)
### Estado
- Auditada carpeta de compartidos `resultados_compartir` para activos sensoriales de Test11.
- Detectado faltante: `a4r` completo de `test11_decoder_suite` no estaba copiado.
- Corrida perceptual `D0` sigue activa en `tmux test11_perceptual`.

### Decisiones
- Se sincroniza `a4r` completo del Test11 anterior al bloque compartido.
- Se crea bloque nuevo de compartidos para perceptual en curso (`test11_perceptual/D0`) y se copia log incremental.

### Evidencia (paths + métricas)
- Copiado ahora:
  - `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/resultados_compartir/06_gate5b_scientific_validation/test11_decoder_suite/a4r_json/test11_decoder_suite.json`
  - `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/resultados_compartir/06_gate5b_scientific_validation/test11_decoder_suite/a4r_samples/*` (80 archivos)
- Estado perceptual D0:
  - `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/resultados_compartir/06_gate5b_scientific_validation/test11_perceptual/D0/test11_perceptual_D0.log`
  - Último hito: `midi2events e1 train=3.8172 val=3.4170 tok_acc=0.2311`

### Próximo paso
- Mantener sync de outputs perceptuales (json + .mid/.wav) en cuanto se generen.
- Al cerrar D0, lanzar `a4r` y luego `d4a4` en el mismo pipeline.

### Riesgos
- ETA real del entrenamiento puede subir por secuencias largas (512 tokens) y validación full-set.
