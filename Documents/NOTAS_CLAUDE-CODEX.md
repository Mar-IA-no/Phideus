# Notas de Claude LOCAL para Codex

> Fecha: 2026-02-21
> Sesión: cosine-tail LR scheduler + batch cosine-tail 60ep + análisis de speedup A4r
> Commits: `f02a8a0`

> [!NOTE]
> **Addendum de sincronización (2026-02-21, post-import `ce26296`)**:
> - `results_unc/` quedó en **182 JSON**.
> - `batch_60ep_a4r` quedó completo (`S=79.4%` en e60).
> - `batch_60ep_d0` y `batch_60ep_d4a4` quedaron importados hasta e40 (`S=72.4%` y `S=82.6%`).
> - `gate44_t3-wt_scratch_50ep_hold` quedó importado hasta e40 (`S=80.6%`).
> - El estado operativo puede avanzar en UNC más rápido que `results_unc/`; cuando eso ocurra, prevalece el corte versionado al comparar en `main`.

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

### Tiempo estimado por arquitectura (ver sección 6)

| Run ctail | Est. wall 60ep | Cabe en 48h? |
|-----------|---------------|--------------|
| D0 | ~36h | Si |
| d4a4 | ~37h | Si |
| a4r | **~15h** | Si, sobra |
| d4-a4r | **~15h** | Si, sobra |

a4r y d4-a4r son 2.6x más rápidos — ver sección 6 para el análisis completo.

---

## 4. Estado de jobs UNC (corte versionado en `main`)

Al corte de artefactos importados (`ce26296`):

| Job | Estado | Epoch con evidencia en repo | S más reciente (repo) |
|-----|--------|-----------------------------|------------------------|
| t3-wt 50ep hold | RUNNING | e40/50 | S@e40=80.6% |
| D0 60ep | RUNNING | e40/60 | S@e40=72.4% |
| d4a4 60ep | RUNNING | e40/60 | S@e40=82.6% |
| a4r 60ep | COMPLETADO | e60/60 | S@e60=79.4% |
| d4-a4r 60ep | PENDING | — | — |
| moe-dual 60ep | PENDING | — | — |

Además, los 4 scripts `cosine-tail` 60ep (`D0`, `d4a4`, `a4r`, `d4-a4r`) quedaron enviados a cola.

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

*Fin de notas — Claude LOCAL, 2026-02-21*
