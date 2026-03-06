# Plan: Escalón 2 — Speech ↔ EGG Cross-Modal Alignment

## Contexto

Escalón 1-C (Audio↔MIDI) cerró con evidencia causal: descriptores mejoran retrieval +9.4pp (Test 02), reorganizan geometría (+82% CKA, Test 06), sin enriquecer decodificabilidad individual (Test 13G-B inverted ranking). Multi-seed record: d4a4=84.1%±2.3pp. Gate 7.1a mostró que un encoder frozen más fuerte no mejora VICReg retrieval (D0_mert=75.0% ≈ D0_lite=75.2%).

**Escalón 2 = primera prueba fuera de música.** Speech ↔ EGG (electroglotógrafo): mismo oscilador (cuerdas vocales), sensores distintos (micrófono vs electrodo en garganta). F0 de voz es **continua** (no cuantizada a semitonos como MIDI) — primera oportunidad de trabajar con ratios reales.

**Hipótesis formal (H3b)**: La representación relacional puede transferirse a dos sensores físicos distintos del mismo fenómeno vocal, superando baseline lineal.

### Dataset: French Lombard (Zenodo 15533059)
- 836 MB, 40 speakers (20M/20F), 9120 clips, ~7.5h
- Speech + EGG simultáneos a 44.1 kHz (raw) / 16 kHz (processed)
- 4 condiciones de ruido (0, 65, 75, 85 dB SPL)
- 60 sentencias por speaker, 3 sesiones
- CC BY-NC-SA 4.0
- **DESCONOCIDO**: si Speech/EGG son canales stereo en un archivo o archivos separados

### Correcciones de Codex incorporadas (rondas 1 y 2)

**Ronda 1:**
1. NO definir S_floor en Fase 0 — sale después de baseline + control
2. Baseline lineal ANTES de diseñar descriptor
3. Auditoría de alineación temporal como tarea explícita
4. Hard negatives más ricos: mismo speaker/diff utterance + mismo texto/diff speaker
5. Encoders austeros: baseline lineal → D0 simétrico pequeño → descriptor → solo después SOTA frozen
6. Primer entregable real = manifest + split + protocolo + baseline lineal, NO un modelo

**Ronda 2:**
7. Fijar protocolo canónico exacto en P0 (sr, ventana, hop, definición de positivo) — sin esto P1 y P2 no son comparables
8. Eval harness es código NUEVO, no "adaptar índices" — evaluate_structured_pool.py está hardcodeado a piece/composer/audio+midi
9. compute_audio_band_energy() necesita variante 16kHz explícita (A4_BAND_EDGES son para sr=24000)
10. Política de ruido: piloto limpio (condición 0 dB) primero, O métricas estratificadas — no mezclar todo desde día 1
11. Auditoría de alineación: medir lag y caracterizarlo/corregirlo en regiones voiced con features comparables, NO cross-corr cruda del waveform como criterio binario
12. Criterios P1: el criterio real es retrieval con pool canónico y CI, no solo CCA corr > 0
13. P2-control necesita mini-run de throughput/VRAM (20 batches) antes del run largo
14. Andamiaje documental: crear `Documents/01_FRENTES_ACTIVOS/ESCALON_2/` y contracts/ en P0

**Ronda 3:**
15. R@10 random con pool_size=128 = 10/128 = 7.8%, no 0.78% — el gate estaba aflojado por factor 10
16. Hard negative más importante faltaba: **mismo clip / distinta ventana** (L1). Sin esto, el modelo resuelve identidad de clip/sentence, no alineación temporal fina.
17. Manifest clip-level no alcanza: necesita segment index window-level derivado canónicamente (regla determinista versionada que genera exactamente las mismas ventanas en P1, P2c y P2m)
18. L2 "same speaker + diff condition" no existe en piloto limpio (0 dB). Pool del piloto necesita estratos propios.
19. Gate de P1 necesita más que "supera random": exigir ganancia no trivial + mirar desempeño por estrato duro
20. No prometer "no se modifica ningún archivo existente" — docs troncales sí se tocan

**Ronda 4 (cierres metodológicos):**
21. CI bootstrap debe ser grouped por speaker (o al mínimo por clip), no naïve por query — con 5 test speakers, CIs naïve son demasiado optimistas
22. max_batches_per_epoch=1000 probablemente excede el dataset real en piloto limpio (0 dB). Regla: época = full pass, o max derivado del N real de segmentos
23. Lag correction debe quedar versionado en el protocolo (segment_index con offset, o compensación fija en loader). Sin esto el positivo canónico queda mal definido
24. Política de ventanas no válidas: clips < 2s, ventanas con poca región voiced, ventanas en silencio. Definir regla en P0 para que P1/P2 usen exactamente la misma población
25. Fallback de estratos en pool: si un estrato no tiene suficientes candidatos (ej: L3 sin match de sentence_id), política explícita + logging de counts reales
26. V4 PYIN/autocorrelación puede ser lento on-the-fly — benchmarkear extracción y precomputar/cachear si es necesario
27. Seeds/splits: pilot con split fijo 30/5/5. Si hay señal, repetir en speaker splits adicionales antes de claim fuerte
28. No asumir "0 dB = silencio" hasta verificar — llamarlo "condición limpia / no-added-noise"

### Preguntas resueltas (por Codex)
- **Positivo canónico**: misma ventana temporal del mismo clip (speech[t0:t1] ↔ egg[t0:t1])
- **Same text matching**: por `sentence_id` normalizado, NO por string de transcript
- **Primer piloto**: condición limpia (0 dB SPL) primero. Métricas estratificadas por condición después.
- **Segment index**: regla determinista de segmentación versionada, derivada del manifest clip-level. Mismas ventanas en P1/P2c/P2m.

---

## Protocolo canónico (fijado en P0, usado en todas las fases)

| Parámetro | Valor | Rationale |
|-----------|-------|-----------|
| Sample rate | 16 kHz | Usar archivos de Process/. Cubre speech (F0+formants hasta 8kHz) |
| Ventana (segment) | 2.0 s = 32000 samples | Sentencias cortas (~2-5s), suficiente voiced material |
| Hop | 0.5 s = 8000 samples | ~3-5 segments/clip, solapamiento moderado |
| Definición de positivo | Misma ventana temporal del mismo clip: speech[t0:t1] ↔ egg[t0:t1] |
| Segmentación | Regla determinista versionada: `segment_windows(duration, seg_len=2.0, hop=0.5)` genera lista fija de (start, end). Mismas ventanas en P1, P2c, P2m. |
| Split | Por speaker: 30 train / 5 val / 5 test (gender balanced) |
| Condición de ruido piloto | Condición limpia (no-added-noise) primero. Agregar las 4 después con métricas estratificadas. Verificar codificación real en P0. |
| Pool size | 128 |
| R@10 random | 10/128 = **7.8%** |
| Queries | min(500, test_segments) |
| Métrica primaria | S = min(Speech2EGG@10, EGG2Speech@10) |
| CI | **Grouped bootstrap** por speaker (o mínimo por clip), 1000 resamples. NO naïve por query. |
| Epoch sizing | Época = full pass del dataset real. Si N_segments < 1000×bs, NO repetir — una época es un barrido completo. |
| Lag correction | Si P0 detecta lag sistemático, se incorpora como offset fijo en segment_index o compensación en loader. Versionado en protocolo. |
| Ventanas válidas | Clips < 2s se excluyen. Ventanas con voiced_fraction < umbral (definido en P0 tras distribución) se excluyen. Misma población en P1/P2c/P2m. |
| Seeds/splits | Pilot: split fijo 30/5/5 seed=42. Grouped bootstrap con 5 test speakers = suficiente para pilot, NO para claim fuerte. Si señal, repetir en splits de speakers adicionales antes de claim. |
| STFT params (16kHz) | n_fft=1024, hop_length=256, freq_res=15.625 Hz/bin |

### Manifest: dos niveles
1. **Clip-level manifest** (`manifest.json`): un entry por clip original (9120 clips). Contiene speaker_id, noise_condition, sentence_id, paths, duration, split.
2. **Segment index** (`segment_index.json`): derivado canónicamente del manifest por la regla determinista de segmentación. Un entry por ventana. Contiene clip_id, segment_idx, start_time, end_time. Este es el que usa la evaluación.

---

## Fases

### S2-P0: Data Ingestion + Manifest + Protocolo + Alignment Audit + Docs (~1 sesión)

**Entregable**: manifest.json, split por speaker, protocolo canónico congelado, reporte de alineación, andamiaje documental.

#### S2-P0.1: Descargar y extraer
```bash
wget https://zenodo.org/records/15533059/files/FLombard.zip -O data/lombard/FLombard.zip
cd data/lombard && unzip FLombard.zip
```

#### S2-P0.2: Descubrir estructura
- Abrir archivo sample, verificar canales (mono/stereo)
- Si stereo: identificar cuál es speech, cuál es EGG
- Si archivos separados: mapear convención de nombres
- Parsear Txt/ para transcripciones → mapear a speaker_id, noise_condition, sentence_id

#### S2-P0.3: Construir manifest (dos niveles)

**Nivel 1: Clip-level** (`data/lombard/manifest.json`):
Fields: clip_id, speaker_id, gender, noise_condition, sentence_id (entero normalizado), session, speech_path, egg_path, duration_sec, split.

**Nivel 2: Segment index** (`data/lombard/segment_index.json`):
Derivado determinísticamente del manifest por regla versionada:
```python
def segment_windows(duration_sec, seg_len=2.0, hop=0.5):
    """Genera ventanas fijas. Misma función en P1, P2c, P2m."""
    windows = []
    t = 0.0
    while t + seg_len <= duration_sec:
        windows.append((t, t + seg_len))
        t += hop
    return windows
```
El segment index hereda speaker_id, noise_condition, sentence_id del clip padre. Cada entry tiene: clip_id, segment_idx, start_sec, end_sec.

#### S2-P0.4: Split por speaker
30 train (15M/15F) / 5 val / 5 test. Balance gender.
Noise conditions se balancean automáticamente.

#### S2-P0.5: Política de ventanas válidas

Definir en P0 para que P1/P2 usen exactamente la misma población:
- Clips con `duration_sec < 2.0` → excluidos del segment_index
- Ventanas con `voiced_fraction < THRESHOLD` → excluidas. THRESHOLD se define con regla predefinida usando solo datos de train/audit (ej: percentil 10 de la distribución de voiced_fraction en train), **no afinado contra resultados de retrieval**.
- La lista de segmentos válidos queda congelada en segment_index.json
- **Lag correction se aplica ANTES de congelar segment_index** — si el audit detecta lag sistemático, el offset ya está incorporado cuando se genera el índice final

#### S2-P0.6: Auditoría de alineación temporal

**NO** cross-correlación cruda del waveform bruto como criterio binario. Medir en regiones voiced con features comparables (contornos F0):

1. **Extraer F0 de ambas modalidades** en regiones voiced (PYIN para speech, autocorrelación para EGG) en 50 clips.
2. **Medir lag** entre contornos F0 (cross-correlación de los contornos F0, no de waveforms). Reportar distribución de lags.
3. **Caracterizar y corregir** si hay lag sistemático. Si el lag es consistente (ej: media ± 1 sample), definir `LAG_CORRECTION_SAMPLES` y documentarlo en `alignment_audit.json`. Este offset se aplica en el loader o se incorpora en segment_index — el positivo canónico debe quedar bien definido.
4. **Consistencia sample rate**: verificar ambos canales.
5. **Clipping**: contar samples |x| > 0.99.
6. **Voiced fraction**: energía + ZCR → distribución por clip. Usada para definir THRESHOLD de ventanas válidas (S2-P0.5).
7. **Silencios**: detectar leading/trailing silence. Registrar voiced region start/end.

Output: `data/lombard/alignment_audit.json` con per-clip y aggregate stats, incluyendo `lag_correction_samples` si aplica.

#### S2-P0.7: Andamiaje documental
Crear:
- `Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md` — doc principal del escalón
- `Documents/00_TRONCAL/ROADMAP_GENERAL/contracts/` — directorio para schemas (vacío o con templates)
- Actualizar `Documents/00_TRONCAL/Proyecto_Estado_Actual.md` — registrar apertura Escalón 2
- Actualizar `Documents/00_TRONCAL/ROADMAP_GENERAL/Rosetta_triplescaloneta.md` — estado

#### Archivos a crear
| Archivo | Líneas est. | Tipo |
|---------|------------|------|
| `experiments/bias_control/escalon2/__init__.py` | 5 | Código |
| `experiments/bias_control/escalon2/s2_p0_manifest.py` | ~450 | Código |
| `experiments/bias_control/escalon2/README.md` | ~150 | Doc |
| `Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md` | ~100 | Doc |

#### Verificación P0
- File count verificado contra metadata del dataset
- Canales speech/EGG correctamente separados (waveform visual check)
- Lag F0 distribution reportada (sin criterio binario)
- Manifest con sentence_id normalizado
- Split 30/5/5, gender balanced
- Voiced fraction distribution generada

---

### S2-P1: Baseline Lineal + Pool Canónico (~1 sesión)

**Pregunta**: "¿Speech↔EGG tiene señal cross-modal usable con features simples?"
**Entregable**: Retrieval numbers con pool canónico y CI bootstrap.

**Condición de ruido**: solo clips de condición **0 dB** (piloto limpio). Agregar las 4 condiciones con métricas estratificadas es paso posterior.

#### S2-P1.1: Extracción de features simples

Extraer por **ventana de 2s** (protocolo canónico, no "clip o 500ms"):

**Speech** (~20 dims): mean/std de 10 mel bands (librosa.feature.melspectrogram) + F0 stats (mean/std/range via PYIN)
**EGG** (~20 dims): mean/std de 10 frequency bands + F0 stats (autocorrelación/PYIN)

**NO** usar compute_audio_band_energy() existente (band edges hardcodeadas a sr=24000). Escribir una variante 16kHz directamente en el script de baseline, con:
```python
BAND_EDGES_16K = [
    (3, 6),     # ~47-94 Hz
    (6, 12),    # ~94-188 Hz
    (12, 24),   # ~188-375 Hz
    (24, 48),   # ~375-750 Hz
    (48, 96),   # ~750-1500 Hz
    (96, 192),  # ~1500-3000 Hz
    (192, 384), # ~3000-6000 Hz
    (384, 513), # ~6000-8000 Hz
]
```

#### S2-P1.2: Pool canónico + hard negatives

Construir pool builder NUEVO (no adaptar evaluate_structured_pool.py):

```python
def build_lombard_pool(segment_index, query_segment, pool_size=128):
    """
    Pool builder con semántica window/clip/speaker/sentence_id.
    Positivo: misma ventana temporal del mismo clip (speech[t0:t1] ↔ egg[t0:t1]).
    """
```

**Hard negative hierarchy — piloto limpio (condición 0 dB)**:
| Nivel | Descripción | N en pool | Confound que testea |
|-------|------------|-----------|---------------------|
| L1 (hardest) | Mismo clip / distinta ventana **no solapada** (separación temporal >= 2.0s) | 16 | ¿Resuelve por identidad de clip? |
| L2 (hard) | Mismo speaker / distinta utterance | 16 | ¿Resuelve por identidad de speaker? |
| L3 (semi-hard) | Distinto speaker / mismo sentence_id | 16 | ¿Resuelve por contenido verbal? |
| L4 (random) | Distinto speaker / distinta utterance | 80 | Baseline |

**Cuando se agreguen las 4 condiciones de ruido** (S2-P2.5), se añade un nivel:
- **L2b**: Mismo speaker / misma utterance / distinta condición de ruido — testea robustez al efecto Lombard

#### S2-P1.3: Métodos lineales + retrieval
1. **CCA**: correlaciones canónicas speech↔EGG.
2. **Ridge**: predecir features EGG desde speech (y viceversa). R².
3. **Retrieval CCA**: proyectar a espacio CCA compartido, cosine similarity, evaluar con pool canónico.

#### S2-P1.4: CI bootstrap (grouped)
**Grouped bootstrap por speaker** (o mínimo por clip), 1000 resamples. Con solo 5 test speakers, bootstrap naïve por query daría CIs demasiado optimistas por correlación intra-speaker.

#### S2-P1.5: Pool fallback + logging
Si un estrato no tiene suficientes candidatos para un query (ej: L3 sin match de sentence_id en test set), política:
- Rellenar con L4 (random) hasta completar pool_size
- Loguear counts reales por estrato (agregado global + per-query si falla)
- Reportar N_queries con pool completo vs parcial

#### Archivo a crear
| Archivo | Líneas est. |
|---------|------------|
| `experiments/bias_control/escalon2/s2_p1_baseline_linear.py` | ~600 |

Incluye: feature extraction + pool builder + CCA/Ridge + retrieval eval + CI bootstrap. Todo en un script.

#### Verificación P1
- **Criterio real**: Retrieval con pool canónico y CI bootstrap.
- R@10 random = 10/128 = **7.8%**. El baseline lineal debe superar esto con ganancia no trivial y CI clara.
- Mirar desempeño por **estrato duro** (L1: mismo clip / diff ventana): si la ganancia viene toda de L4 random y L1 está en azar, la señal es espuria.
- CCA/Ridge R² son auxiliares informativos, no criterio de avance.
- Si retrieval CCA no supera 7.8% con CI clara → problema fundamental, no avanzar a P2.

---

### S2-P2-control: D0 Neural con Mini-Run + Run Completo (~1 sesión)

**Entregable**: S_control y hard_neg_control con CI bootstrap.

#### S2-P2-control-mini: Throughput benchmark (20 batches)

Antes del run largo, verificar:
- VRAM usage con bs=64
- Batches/min estable
- Shapes correctas
- No OOM

Si bs=64 no entra, reducir. Loguear: VRAM peak, batches/min, time estimates para 30ep.

#### Arquitectura: dos encoders idénticos

Basado en MERTEncoderLite pattern, d=512:

```
Waveform [B, 32000]  (2s @ 16kHz)
  → Conv1d(1, 256, k=10, s=5) → GN → GELU
  → Conv1d(256, 256, k=3, s=2) → GN → GELU
  → Conv1d(256, 256, k=3, s=2) → GN → GELU
  → Conv1d(256, 512, k=3, s=2) → GN → GELU
  → [B, 512, T']  (T' ≈ 800)
  → Positional Embedding
  → Transformer(4 layers, 8 heads, d=512)
  → Mean pooling → [B, 512]
```

~15M params por encoder, ~30M total. Ambos trainable from scratch.

```
Speech → SpeechEncoder → ProjectionHead(512→512→256) → z_speech
EGG    → EGGEncoder    → ProjectionHead(512→512→256) → z_egg
Loss = VICReg(z_speech, z_egg, λ_inv=10, λ_var=10, λ_cov=1)
```

#### Configuración de training
Usa el protocolo canónico fijado en P0:
- sr=16kHz, segment=2s, hop=0.5s
- Condición de ruido: **condición limpia solamente** (piloto)
- Epochs: 30, batch_size: ajustado por mini-run
- **Epoch = full pass del dataset real**. Si N_train_segments/bs < 1000, una época es un barrido completo — NO repetir batches artificialmente.
- LR: 5e-4 (encoders), proj LR: 1e-3
- Warmup: 500 steps, scheduler: warmup → cosine
- Grad clip: 1.0, weight decay: 0.01
- Structured eval: epochs 5, 10, 15, 20, 25, 28, 29, 30

#### Evaluación
Usa el **mismo pool builder** de P1 (build_lombard_pool). Misma definición de positivo, misma hard negative hierarchy, mismo CI bootstrap.

#### Anti-ghost
DriftSentinel + preflight reutilizables. Ambos encoders trainables → todos los params deben driftear.

#### Archivos a crear
| Archivo | Líneas est. |
|---------|------------|
| `src/bias_control/datasets/lombard_segments.py` | ~350 |
| `src/bias_control/encoders/speech_egg_encoder.py` | ~80 |
| `experiments/bias_control/escalon2/train_escalon2.py` | ~800 |
| `experiments/bias_control/escalon2/eval_escalon2.py` | ~400 |

**eval_escalon2.py es NUEVO** — no reutiliza evaluate_structured_pool.py. Contiene:
- `build_lombard_index()`: índice por speaker, sentence_id, noise_condition
- `build_lombard_pool()`: pool con hard negative hierarchy (4 niveles)
- `extract_embeddings_lombard()`: extracción de embeddings speech+egg
- `evaluate_retrieval_lombard()`: S = min(S2E@10, E2S@10) con CI

---

### S2-P2-main: Descriptor Vocal Relacional V4 (~1 sesión)

**Solo después de que S2-P2-control establezca S_control.**

#### V4: F0 Ratio Descriptor (4 dims)
Primera vez con ratios reales continuos:

1. Extraer contorno F0 con PYIN (speech) / autocorrelación (EGG)
2. En frames voiced:
   - `log2_ratio_prev = log2(F0[t] / F0[t-1])`
   - `log2_ratio_next = log2(F0[t+1] / F0[t])`
   - `voicing_strength`
   - `period_regularity`
3. Output: `[B, T', 4]`

**V4 se inyecta en AMBOS encoders** (ambas modalidades tienen F0).

**Caching**: PYIN/autocorrelación puede ser lento on-the-fly. Benchmarkear extracción en P2-control-mini. Si > 10ms/segment, precomputar V4 por segmento y cachear en disco (`data/lombard/v4_cache/`).

#### A4 adaptado a 16kHz

Variante nueva `compute_audio_descriptor_a4_16k()` en `vocal_descriptors.py` con BAND_EDGES_16K. No modifica audio_descriptors.py existente.

#### Inyección: concat + linear (patrón Gate42InputAugModel)
```
CNN features [B, T', 512] + descriptor [B, T', D] → concat → Linear(512+D, 512) → LN → Transformer
```

#### Arms de screening (3ep cada uno, condición 0 dB)
| Arm | Descriptor | Dims |
|-----|-----------|------|
| D0 | ninguno | 0 |
| V4 | F0 ratios | 4 |
| A4-16k | Band energy deltas | 8 |
| V4+A4 | combinado | 12 |

Full training (30ep) del ganador + D0 control.

#### Archivo a crear
| Archivo | Líneas est. |
|---------|------------|
| `src/bias_control/vocal_descriptors.py` | ~250 |

---

### S2-P2.5: Agregar condiciones de ruido + métricas estratificadas

Después del piloto limpio (0 dB), agregar las 4 condiciones:
- Repetir D0 + ganador con datos mezclados (4 condiciones)
- Reportar métricas **estratificadas por condición de ruido**:
  - S por condición (0, 65, 75, 85 dB)
  - Hard_neg por condición
  - Delta descriptor vs D0 por condición

Esto previene mezclar sensor shift con efecto Lombard en la interpretación.

---

### S2-P3 (opcional): Arm SOTA Frozen

Solo si S2-P2 muestra señal. WavLM/HuBERT frozen en speech, encoder pequeño en EGG. No se diseña — depende de resultados.

---

## Infraestructura

### Código nuevo (NO reutilización directa de evaluate_structured_pool.py)

| Componente | Status |
|-----------|--------|
| Pool builder para Lombard | NUEVO (`eval_escalon2.py`) |
| Band edges 16kHz | NUEVO (en `vocal_descriptors.py` y `s2_p1_baseline_linear.py`) |
| Dataset loader | NUEVO (`lombard_segments.py`, patrón de maestro_segments.py) |
| Encoder speech/egg | NUEVO (wrapper de MERTEncoderLite, `speech_egg_encoder.py`) |
| Training loop | NUEVO (`train_escalon2.py`, patrón de train_gate71.py) |

### Reutilización real (sin modificaciones)

| Componente | Archivo fuente |
|-----------|---------------|
| VICRegLoss | `src/RNA/vicreg.py` |
| ProjectionHead | `src/bias_control/encoders/projection.py` |
| DriftSentinel + preflight | `src/bias_control/training/preflight.py` |
| LinearWarmupCosineScheduler | `experiments/bias_control/gate71/train_gate71.py` |

### Patrón (código nuevo sigue el mismo diseño)

| Componente | Fuente del patrón |
|-----------|-------------------|
| MERTEncoderLite → SpeechEGGEncoder | `src/bias_control/encoders/mert_encoder.py` L221-315 |
| MaestroSegmentDataset → LombardSegmentDataset | `src/bias_control/datasets/maestro_segments.py` |
| Training loop gate71 → train_escalon2 | `experiments/bias_control/gate71/train_gate71.py` |

---

## Archivos nuevos (total)

| Archivo | Líneas | Fase |
|---------|--------|------|
| `experiments/bias_control/escalon2/__init__.py` | 5 | P0 |
| `experiments/bias_control/escalon2/README.md` | 150 | P0 |
| `experiments/bias_control/escalon2/s2_p0_manifest.py` | 450 | P0 |
| `Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md` | 100 | P0 |
| `experiments/bias_control/escalon2/s2_p1_baseline_linear.py` | 600 | P1 |
| `src/bias_control/datasets/lombard_segments.py` | 350 | P2c |
| `src/bias_control/encoders/speech_egg_encoder.py` | 80 | P2c |
| `experiments/bias_control/escalon2/train_escalon2.py` | 800 | P2c |
| `experiments/bias_control/escalon2/eval_escalon2.py` | 400 | P2c |
| `src/bias_control/vocal_descriptors.py` | 250 | P2m |
| **Total** | **~3185** | |

**Docs troncales que se actualizan**: `Proyecto_Estado_Actual.md`, `Rosetta_triplescaloneta.md`, `INDICE_DOCUMENTACION.md`.

---

## Verificación end-to-end

| Fase | Verificación | Criterio |
|------|-------------|----------|
| P0 | File count vs metadata | Count matches |
| P0 | Canales speech/EGG | Separación correcta (visual check) |
| P0 | Lag F0 distribution | Reportada, lag corregido si sistemático |
| P0 | Split | 30/5/5, gender balanced |
| P0 | Segment index | Generado determinísticamente, mismas ventanas reproducibles |
| P0 | Voiced fraction | Distribution reportada |
| P0 | Docs | ESCALON_2/README.md existe |
| P1 | **Retrieval CCA con pool canónico y CI** | **R@10 > 7.8% (random) con ganancia no trivial y CI clara** |
| P1 | **Desempeño por estrato duro** | **L1 (mismo clip/diff ventana) > azar** |
| P1 | CCA/Ridge R² (auxiliar) | Reportado, no es criterio de avance |
| P2c-mini | Throughput benchmark 20 batches | bs y time calibrados, no OOM |
| P2c | Preflight + DriftSentinel | Anti-ghost pass |
| P2c | VICReg loss decrece, std > 0.5 | No colapso |
| P2c | **S_control con CI** | **Establecido** |
| P2m | V4 3ep screening vs D0 3ep | Señal de descriptor |
| P2m | Full 30ep con CI | S_run - S_control comparado (usuario decide) |
| P2.5 | Métricas estratificadas por condición ruido | 4 S values + L2b (diff condition) reportados |

---

## Orden de ejecución

```
S2-P0 (datos + manifest + audit + docs + protocolo canónico)
    ↓
S2-P1 (baseline lineal + pool canónico, solo condición 0 dB)
    ↓
S2-P2-control-mini (20 batches, throughput/VRAM check)
    ↓
S2-P2-control (D0 neural 30ep, condición 0 dB)
    ↓
S2-P2-main (V4 screening 3ep + full 30ep, condición 0 dB)
    ↓
S2-P2.5 (agregar 4 condiciones, métricas estratificadas)
    ↓
[DECISION GATE: usuario decide con evidencia]
    ↓
S2-P3 (opcional: SOTA frozen)
```

Cada fase bloquea la siguiente. No se salta ninguna.
Primera implementación: S2-P0 completo.
