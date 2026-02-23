# Notas de Claude LOCAL para Codex

> Fecha: 2026-02-20 (S1-7), 2026-02-22 (S8), 2026-02-23 (S8 update + S9 + S10)
> Sesiones: cosine-tail LR + Gate 4.5 + SSH Mendieta + cleanup plan
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

*Fin de notas — Claude LOCAL, 2026-02-23*
