# Plan: Rediseño S2-P2-main — Descriptores con Armonía Natural

## Contexto

S2-P2-control (D0) COMPLETO: S=77.8% @ ep25, CI=[72.0%, 80.8%]. GPU libre.

**Problema**: El plan original de P2-main solo tenía V4 (log2 ratios, perceptual) + A4-16k. Ningún descriptor testeaba la tesis fuerte de Phideus sobre armonía natural (ratios lineales, serie armónica física). Codex identificó que hay que separar 3 hipótesis distintas y agregar descriptores genuinamente naturales.

**Directiva epistemológica (2026-03-08)**: Los descriptores primarios deben derivarse de invariantes físicos del fenómeno medido. Descriptores perceptuales/logarítmicos quedan como controles de comparación, no como default.

## Tres hipótesis a separar (Codex)

- **Hipótesis A**: La dinámica temporal del oscilador glotal contiene invariantes relacionales útiles (V4-lin, V4-log)
- **Hipótesis B**: La estructura armónica natural intra-frame contiene invariantes alineados con la tesis fuerte de Phideus (H-series)
- **Hipótesis C**: Las mejoras vienen de descriptores genéricos no-ratio (A4-16k)

## Correcciones Codex (ronda revisión del plan)

### C1: Fuga cross-modal por F0 compartido — CORREGIDO

**Problema**: Usar F0 extraído de speech para ambas modalidades introduce fuga cross-modal. El encoder EGG recibiría información derivada del speech pareado, debilitando la interpretación causal.

**Solución**: F0 se extrae **per-modality**. Cada señal usa su propio F0:
- **Speech**: PYIN (librosa.pyin, frame_length=2048, fmin=50, fmax=500)
- **EGG**: Autocorrelación adaptada (la señal EGG es cuasi-periódica con baseline de impedancia, PYIN no funciona bien). Implementar `extract_f0_egg()` con detección de período por autocorrelación + threshold de periodicidad.

El F0 de speech y el F0 de EGG son estimaciones independientes del mismo fenómeno (deberían coincidir en voiced regions, pero no se copian). Esto mantiene la limpieza causal: cada encoder solo ve información derivada de su propia señal.

**Nota**: Si autocorrelación en EGG da voicing problemático (~14-17% con PYIN), se documenta explícitamente como hipótesis abierta: "H-series en EGG podría no funcionar igual que en speech porque el sensor captura un perfil armónico distinto". Esto es una observación empírica, no un motivo para fusionar F0s.

### C2: Interfaz eval/cache — CORREGIDO

**Problema**: `extract_embeddings_lombard()` solo recibe `model(waveform)` y no pasa clip_id/segment_idx al modelo, que los necesitaría para lookup en el F0 cache.

**Solución**: Modificar eval_escalon2.py (aceptar que hay que tocarlo). Cambio mínimo y retrocompatible:
- `extract_embeddings_lombard()` recibe un parámetro opcional `descriptor_fn: Callable = None`
- Si `descriptor_fn is None` → comportamiento idéntico a D0
- Si `descriptor_fn is not None` → para cada batch, computa `desc = descriptor_fn(batch)` y pasa `model(waveform, descriptor=desc)`
- El batch ya contiene `clip_id`, `segment_idx`, `f0`, `voiced` (del dataset augmentado)

Cambio estimado: ~15 líneas en `extract_embeddings_lombard()`. No rompe D0 (default None = path actual).

### C3: Alineación temporal descriptor → 800 tokens — CONGELADO

**Problema**: Los descriptores se computan a resolución menor que los 800 tokens del CNN y la política de interpolación no estaba especificada.

**Política congelada**:
- **Resolución nativa de descriptores**: hop_length=160 @ 16kHz = 10ms/frame → 201 frames por 2s segment
- **Interpolación**: `F.interpolate(desc.transpose(1,2), size=T_cnn, mode='linear', align_corners=False).transpose(1,2)` donde `T_cnn` = dimensión temporal real del output CNN (calculado dinámicamente, no hardcodeado a 800)
- **Centering**: descriptores se computan con `center=True` en STFT (consistente con audio_descriptors.py:59)
- **Todos los descriptors usan la misma función de interpolación** → comparables y reproducibles
- La interpolación ocurre DENTRO de cada función de descriptor (no en el caller)

### C4: Z-score per-batch → estadísticas congeladas — CORREGIDO

**Problema**: Z-score por batch hace que el descriptor de un sample dependa de otros samples del batch. Cambia entre training y eval. No determinístico.

**Solución**: Normalización per-segment (no per-batch):
- **V4-lin/V4-log**: Ratios ya están en rango acotado por clipping. No necesitan z-score. `(ratio - 1.0)` para lin, ratio crudo para log. Rango fijo.
- **H-series**: Harmonic ratios `log(Hn/H1 + 1e-3)` se normalizan con estadísticas congeladas del train set, **separadas por modalidad**. Precomputar mean/std de cada feature sobre los segments de train (una vez, al inicio de la primera ejecución). Guardar en `data/lombard/h_series_norm_speech.json` y `data/lombard/h_series_norm_egg.json`. Estas stats se cargan y aplican de forma fija en train y eval.
- **A4-16k**: Z-score per-segment per-band (como audio_descriptors.py:82-83 ya hace). Determinístico por segmento.
- **Regla general**: NINGÚN descriptor depende de otros samples del batch. Todo es per-segment o con stats congeladas.

### C5: Protocolo de screening honesto — CORREGIDO

**Problema**: "Top 2 @ 10ep" se vuelve decisorio en la práctica, contradiciendo la directiva epistemológica.

**Nuevo protocolo**:
- **Smoke 1-3ep × 50 batches**: Solo verificar que no colapsa, shapes correctas, DriftSentinel OK. ~25 min total.
- **Full 30ep directamente** para: D0 (ya tenemos), V4-lin, H-series, A4-16k. Estos 3 representan cada familia (temporal natural, armónica natural, control espectral). Son los que responden las preguntas centrales.
- **V4-log y V4-lin+H**: Se corren DESPUÉS si los primeros 3 muestran señal. V4-log solo importa si V4-lin funciona (para comparar escala). V4-lin+H solo importa si ambos componentes funcionan (para ver complementariedad).
- **NO hay corte intermedio**. 30ep para las 3 familias primarias. ~165 min × 3 = ~8h secuencial.
- **Solo el usuario decide** interpretación con los resultados completos.

### C6: H-series — búsqueda local de pico — CORREGIDO

**Problema**: `round(n * F0 / freq_res)` con n_fft=1024 @ 16kHz (freq_res=15.625 Hz/bin) es demasiado grueso para F0 bajos. Error de 1 bin es significativo.

**Solución**: Usar n_fft=2048 (freq_res=7.8 Hz/bin) para H-series + **búsqueda local de pico** alrededor del bin esperado:
```python
expected_bin = round(n * F0 / freq_res)
search_range = 2  # ±2 bins
local_slice = magnitude[..., max(0, expected_bin-search_range):expected_bin+search_range+1]
H_n = local_slice.max(dim=-1).values  # pico local
```
Esto mitiga el leakage espectral y el error de cuantización de bin. Con n_fft=2048, costo es ~2x más STFT pero sigue siendo <8ms/batch en GPU.

### C7: H-series speech vs EGG — hipótesis explícita

**Reconocido**: H-series en EGG no es automáticamente equivalente a H-series en speech. EGG captura la fuente glótica directamente (rica en armónicos pares, H2/H1 alto), mientras que speech tiene el filtro del tracto vocal superpuesto (formantes modifican las amplitudes relativas de armónicos).

**Tratamiento**: Se documenta como hipótesis abierta explícita, no como supuesto silencioso:
- "Los ratios de amplitud armónica (H-series) capturan perfiles DIFERENTES en speech y EGG. La pregunta es si esas diferencias contienen información complementaria para alineación, o si son ruido."
- Si H-series funciona mejor en speech que en EGG (o viceversa), eso es un resultado informativo, no un bug.

## Arms finales (corregidos)

| Arm | Descriptor | Familia | Base | Dims | Prioridad |
|-----|-----------|---------|------|------|-----------|
| D0 | ninguno (DONE, S=77.8%) | — | — | 0 | — |
| V4-lin | F0 ratios lineales | Temporal | **NATURAL** | 4 | **Primaria** |
| H-series | Amplitud relativa de armónicos | Armónica | **NATURAL** | 8 | **Primaria** |
| A4-16k | Band energy deltas | Espectral | No-ratio | 8 | **Primaria** |
| V4-log | F0 ratios log2 | Temporal | Perceptual | 4 | Secundaria |
| V4-lin+H | V4-lin + H-series combinado | Combinado | **NATURAL** | 12 | Secundaria |

Primarios se corren full 30ep. Secundarios se corren después si hay señal.

### Qué responde cada comparación

- V4-lin > D0 → dinámica relacional del oscilador ayuda
- V4-lin > V4-log → la escala física lineal importa (solo si V4-lin muestra señal)
- H-series > D0 → estructura armónica natural intra-frame aporta señal
- H-series > A4-16k → la mejora no viene solo de forma espectral genérica
- V4-lin+H > ambos solos → dinámica temporal y estructura armónica son complementarias
- A4-16k > D0 pero V4/H no → descriptor genérico ayuda, sin evidencia a favor de tesis natural

## Especificación de descriptores

### V4-lin (4 dims, NATURAL) — Familia 1: dinámica temporal

```
1. ratio_prev = F0[t] / F0[t-1]           (lineal, neutral=1.0 si unvoiced)
2. ratio_next = F0[t+1] / F0[t]           (lineal, neutral=1.0 si unvoiced)
3. voicing_strength                        (float [0,1], smoothed 3-frame avg)
4. period_regularity                       (1 - std_local de ratios, clipped [0,1])
```

Normalización: ratios clipped [0.5, 2.0], luego `(ratio - 1.0)` → rango [-0.5, 1.0]. Determinístico per-segment.
F0 extraído per-modality: PYIN para speech, autocorrelación para EGG.

### V4-log (4 dims, perceptual) — Familia 1 control

```
1. log2(F0[t] / F0[t-1])                  (neutral=0.0 si unvoiced)
2. log2(F0[t+1] / F0[t])                  (neutral=0.0 si unvoiced)
3. voicing_strength                        (idéntico a V4-lin)
4. period_regularity                       (idéntico a V4-lin)
```

Clipped [-1, 1] (±1 octava). Determinístico per-segment.

### H-series (8 dims, NATURAL) — Familia 2: armónica intra-frame

```
1-5. log(H_{n+1}/H_1 + 1e-3)  para n=1..5   (normalizados con stats congeladas de train)
  6. harmonic_concentration = sum(H1..H6) / total_energy   [0,1]
  7. harmonic_deviation = std(log(Hn/H1))                  (normalizados con stats congeladas)
  8. voicing_strength
```

Extracción: STFT n_fft=2048 en GPU + **búsqueda local de pico** (±2 bins) alrededor de `round(n * F0 / freq_res)`.
Normalización: stats (mean/std per feature) precomputadas sobre train set, **separadas por modalidad**: `h_series_norm_speech.json` y `h_series_norm_egg.json`. Speech y EGG tienen perfiles armónicos distintos (EGG: H2/H1~0.5, speech: H2/H1~0.1), mezclar sus stats contaminaría la escala de cada sensor.
Unvoiced → todo 0. Requiere F0 precomputado per-modality.
**Hipótesis explícita**: H-series captura perfiles armónicos diferentes en speech vs EGG. Esto es un resultado, no un bug.

### A4-16k (8 dims, no-ratio) — Familia 3: control de dinámica espectral

```
8 bandas log-freq @ 16kHz (n_fft=1024, hop=160):
  [47-94, 94-188, 188-375, 375-750, 750-1500, 1500-3000, 3000-6000, 6000-8000] Hz
Temporal delta + z-score per-segment per-band.
```

Adaptación directa de `compute_audio_descriptor_a4()`. NO requiere F0. Determinístico per-segment.

**Nota de interpretación**: A4-16k mide **dinámica espectral local** (cómo cambia la energía por banda), no forma espectral estática. Si H-series > A4-16k, la conclusión correcta es "la estructura armónica natural supera este control dinámico particular", no "supera cualquier descriptor espectral genérico". Esta distinción se documenta en los resultados.

## Alineación temporal (política congelada)

| Parámetro | Valor |
|-----------|-------|
| Resolución nativa descriptores | hop=160 @ 16kHz = 10ms → 201 frames/2s |
| Target length | `T_cnn` = dim temporal real del output CNN (~800) |
| Interpolación | `F.interpolate(mode='linear', align_corners=False)` |
| Centering STFT | `center=True` |
| Dónde ocurre | Dentro de cada función de descriptor |
| Consistencia | Todos los descriptors usan la misma función |

## Inyección: Input Augmentation (patrón Gate42)

```
CNN features [B, T', 512] + descriptor [B, T', D]
  → concat → Linear(512+D, 512) → Transformer
```

- Subclase `SpeechEGGEncoderAug(SpeechEGGEncoder)` con parámetro `descriptor_dim`
- **SIN LayerNorm** en la proyección de inyección. El encoder base no tiene LN entre CNN y Transformer (`speech_egg_encoder.py:70-76`), así que agregar LN cambiaría la distribución de activaciones y rompería la identidad al arranque.
- Near-identity init: W = [I | 0], bias = 0 → ep0 produce output **exactamente idéntico** a D0 (Linear sin LN con identity weights = passthrough perfecto)
- **Ambos encoders** reciben descriptores (cada uno de su propia modalidad)

## F0: precomputar per-modality, cachear por clip

- **Speech**: PYIN (librosa.pyin, frame_length=2048, hop=160, fmin=50, fmax=500)
- **EGG**: Autocorrelación adaptada (búsqueda de período en rango [2ms, 20ms] → F0 [50, 500] Hz, con threshold de periodicidad)
- Almacenar per-clip (2280 clips noise0 × 2 modalidades). ~20 MB total.
- Cache: `data/lombard/f0_cache_noise0.npz` con keys `f0_speech_{clip_id}`, `voiced_speech_{clip_id}`, `f0_egg_{clip_id}`, `voiced_egg_{clip_id}`
- Dataset extendido (`LombardSegmentDatasetAug`) carga F0 cache y slice por segment.

**Confound explícito**: PYIN (speech) y autocorrelación (EGG) son estimadores distintos. V4-lin compara no solo sensores sino también métodos de estimación de F0. Si la señal aparece, un control posterior con estimador parejo (ej: autocorrelación en ambos) sería informativo. No es blocker pero se documenta.

### Fórmula de slice del cache F0 (versionada)

```python
# Parámetros congelados
F0_HOP_LENGTH = 160         # samples @ 16kHz
F0_HOP_SEC = 0.01           # 160 / 16000 = 10ms
F0_CENTER = True            # center=True en PYIN/STFT

# Mapeo segment_start_sec → frame range
# Con center=True, el frame 0 está centrado en sample 0.
# frame_time[i] = i * hop_sec
frame_start = round(segment_start_sec / F0_HOP_SEC)
frame_end = frame_start + round(SEGMENT_LEN / F0_HOP_SEC) + 1  # +1 por fencepost
# → 201 frames para segment de 2.0s con hop 10ms

# Esta fórmula se guarda en el cache metadata y se usa idénticamente en
# precompute_f0.py, lombard_segments_aug.py y train_escalon2_descriptors.py
```

Metadata del cache incluye: `{hop_length: 160, sr: 16000, center: true, fmin: 50, fmax: 500, frame_length: 2048, segment_len: 2.0, frames_per_segment: 201}`.

## Archivos a crear (5 nuevos)

| # | Archivo | Líneas est. | Propósito |
|---|---------|------------|-----------|
| 1 | `src/bias_control/vocal_descriptors.py` | ~350 | compute_v4_linear, compute_v4_log, compute_h_series, compute_a4_16k, extract_f0_egg |
| 2 | `src/bias_control/encoders/speech_egg_encoder_aug.py` | ~80 | SpeechEGGEncoderAug con descriptor injection |
| 3 | `src/bias_control/datasets/lombard_segments_aug.py` | ~120 | LombardSegmentDatasetAug con F0 cache per-modality |
| 4 | `experiments/bias_control/escalon2/precompute_f0.py` | ~250 | PYIN (speech) + autocorrelación (EGG) + cache a NPZ |
| 5 | `experiments/bias_control/escalon2/train_escalon2_descriptors.py` | ~650 | Training con --descriptor flag |

## Archivos existentes a modificar (mínimo)

| Archivo | Cambio | Líneas |
|---------|--------|--------|
| `experiments/bias_control/escalon2/eval_escalon2.py` | Agregar `descriptor_fn` param a `extract_embeddings_lombard()` | ~15 |

Cambio retrocompatible: `descriptor_fn=None` → path D0 intacto.

## Archivos existentes reutilizados (sin modificación)

| Componente | Archivo |
|-----------|---------|
| VICRegLoss | `src/RNA/vicreg.py` |
| ProjectionHead | `src/bias_control/encoders/projection.py` |
| DriftSentinel | `src/bias_control/training/preflight.py` |
| Pool builder + retrieval | `experiments/bias_control/escalon2/eval_escalon2.py` (evaluate_retrieval_lombard, build_lombard_pool) |
| Base encoder | `src/bias_control/encoders/speech_egg_encoder.py` |
| Base dataset | `src/bias_control/datasets/lombard_segments.py` |

## Orden de ejecución

```
1. Implementar vocal_descriptors.py + speech_egg_encoder_aug.py (unit-testeable)
2. Implementar precompute_f0.py
3. Ejecutar precompute_f0 → f0_cache_noise0.npz (~60 min, tmux)
4. Implementar lombard_segments_aug.py
5. Modificar eval_escalon2.py (descriptor_fn param, ~15 líneas)
6. Implementar train_escalon2_descriptors.py
7. Smoke test: 3ep × 50 batches × 3 arms primarios (~15 min total)
8. Full 30ep: V4-lin, H-series, A4-16k (~165 min × 3 = ~8h secuencial)
9. Si hay señal → V4-log, V4-lin+H como arms secundarios
```

## Protocolo de screening (corregido)

- **Smoke 1-3ep × 50 batches**: Verificar no collapse, shapes correctas, DriftSentinel OK. ~15 min total.
- **Full 30ep directamente** para 3 arms primarios (V4-lin, H-series, A4-16k). Sin corte intermedio.
- **Arms secundarios** (V4-log, V4-lin+H) solo si los primarios muestran señal.
- **Solo el usuario decide** interpretación. Claude reporta observaciones, NO declara GO/NO-GO.

## Verificación

| Paso | Verificación |
|------|-------------|
| F0 precompute | Voiced fraction speech vs EGG reportada. F0 agreement en voiced frames medido. |
| Smoke | No NaN/inf, variance > 0, DriftSentinel pass, shapes correctas |
| Smoke | Descriptor dims: V4=4, H-series=8, A4-16k=8 |
| Smoke | Descriptors determinísticos (mismo input → mismo output) |
| Full | CI bootstrap, same pool/eval que D0, checkpoints every epoch |
| Full | Config guardado con descriptor type, dim, F0 method, norm stats path en JSON |
| Full | H-series norm stats congeladas per-modality (speech/egg separadas, no recalculadas entre runs) |
| Full | F0 cache slice usa fórmula versionada (frame_start = round(start_sec / 0.01)) |
