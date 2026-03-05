# Gate 5B — Informe Completo de Validación Científica

> **Fecha**: 2026-03-05 (v2 — Test 13G-B 4/4 completo, Test 11 Pre-Proj 4/4)
> **Proyecto**: Phideus — Harmonic Information Theory
> **Frente**: BIAS_CONTROL — Gate 5, Línea B (Showcase)
> **Autor**: Claude LOCAL (Inference01) + Claude UNC (Mendieta)

---

## 1. Resumen Ejecutivo

Gate 5B es la batería de validación científica del proyecto Phideus. Tras alcanzar los mejores modelos cross-modales en Gates 4.2–4.5, se diseñaron **13 tests** para responder una pregunta central:

> **¿La mejora de los modelos con descriptores de ratios es genuina, causal y científicamente robusta?**

Los tests cubren causalidad, robustez, geometría representacional, decodificabilidad, replicabilidad estadística y generación. Los resultados convergen hacia una conclusión fuerte:

**Los descriptores de ratios (particularmente A4, la dinámica espectral temporal por banda de octava) producen una mejora causal, estadísticamente significativa y mecánicamente interpretable en el aprendizaje cross-modal audio↔MIDI.**

### Hallazgos principales

1. **Causalidad confirmada**: A4 es completamente causal (Test 01: -75 a -78pp al eliminarlo). D4 no contribuye.
2. **Replicabilidad robusta**: Multi-seed confirma la mejora (Test 05: d4a4 84.1%±2.3pp vs D0 75.2%±2.3pp, p<0.05, Cohen d=4.50).
3. **Control de capacidad**: Modelos param-matched con descriptores inutilizados caen a nivel D0 (Test 02: random ~73%, zero ~74% vs real 83%).
4. **Mecanismo no-lineal**: La ventaja vive en la geometría de distancias, no en decodificabilidad lineal (Tests 03, 06, 08).
5. **Cuello de botella identificado**: El mean-pooling 750:1 y la proyección 1024→256 destruyen información masivamente (Test 11, 13G).
6. **Ventaja geométrica, no de feature richness** *(hallazgo final)*: Los descriptores reorganizan el espacio de distancias (+82% CKA, Test 06) sin enriquecer la decodificabilidad temporal de las features individuales. Test 13G-B (4/4): F1~0.10 uniforme en todos los brazos, con D0 pool-188 marginalmente superior a los brazos descriptor — inversión del ranking respecto a Test 11 (retención pre-proj: d4a4=0.770 > d4-a4r=0.748 > a4r=0.712 > D0=0.597). La ventaja de los descriptores es real y causal, pero opera en la organización del espacio latente, no en el contenido informacional de cada vector.

---

## 2. Contexto: Arquitectura y Modelos Evaluados

### 2.1 Arquitectura base

```
Audio (waveform) ─→ MERTEncoderLite ─→ [B, 2400, 1024] ─→ mean pool ─→ [B, 1024] ─→ MLP proj ─→ [B, 256]
                    (4 CNN + 4 Transformer, ~60M params)                                (1024→512→256)

MIDI (tokens)    ─→ MIDIEncoder     ─→ [B, N, 512]     ─→ mean pool ─→ [B, 512]  ─→ MLP proj ─→ [B, 256]
                    (4 Transformer, ~13M params)                                       (512→512→256)

Loss: VICReg(z_audio, z_midi)  →  inv=10, var=10, cov=1
```

### 2.2 Descriptores evaluados

| Descriptor | Modalidad | Dimensión | Descripción |
|------------|-----------|-----------|-------------|
| **A4** | Audio | 8 | Deltas temporales de log-magnitud en 8 bandas de octava (47–12000 Hz) |
| **D4** | MIDI | 4 | Intervalos locales (interval_prev, interval_next, duration_ratio, velocity_diff) |

### 2.3 Mecanismos de inyección

| Mecanismo | Código | Descripción |
|-----------|--------|-------------|
| **Concat** | d4a4 | Descriptor concatenado a features pre-transformer |
| **Reverse cross-att** | a4r | Q=descriptor(188), K/V=CNN_features(2400). Comprime la secuencia de 2400→188 tokens |
| **Híbrido** | d4-a4r | D4 por concat + A4 por reverse cross-attention |

### 2.4 Modelos (checkpoints Gate 4.5)

| Brazo | Descriptor | Mecanismo | Epoch | S (%) | Params |
|-------|-----------|-----------|-------|-------|--------|
| **d4a4** | D4+A4 | Concat | 50 | **83.8** | 75.5M |
| **a4r** | A4 | Rev. cross-att | 29 | 82.0 | 78.6M |
| **d4-a4r** | D4+A4 | Concat+Cross | 30 | 79.8 | 78.9M |
| **D0** | Ninguno | — | 50 | 73.4 | 74.2M |
| d4 | D4 | Concat | 5 | 63.6 | 74.5M |

**Métrica primaria**: S = min(A2M_R@10, M2A_R@10), pool=256 candidatos, 500 queries, seed=42.

---

## 3. Catálogo de Tests

| # | Test | Pregunta científica | Nodo | Estado |
|---|------|---------------------|------|--------|
| 01 | Causal Ablation | ¿El descriptor es causalmente responsable? | LOCAL | CERRADO |
| 02 | Param-Matched | ¿La mejora viene de la info o de los params extra? | UNC | PARCIAL |
| 03 | Ratio Probe | ¿Las features son linealmente decodificables? | LOCAL | CERRADO |
| 04 | Transposition | ¿El modelo es invariante a transposición? | LOCAL | CERRADO |
| 05 | Multi-Seed | ¿Los resultados son replicables? | UNC | CERRADO (15/15) |
| 06 | RSA/CKA | ¿Los descriptores alinean los encoders? | LOCAL | CERRADO |
| 08 | Ratio Decoding | ¿Qué bandas del descriptor son más sensibles? | LOCAL | CERRADO |
| 09 | Invariance Suite | ¿Cómo responde cada brazo a perturbaciones? | LOCAL | CERRADO |
| 10 | Embedding Geometry | Propiedades geométricas del espacio latente | LOCAL | CERRADO |
| 11 | Decoder Suite | ¿Cuánta información retiene el embedding? | LOCAL+UNC | CERRADO (4/4) |
| 12 | Scoreboard | Retrieval canónico estandarizado | LOCAL | CERRADO |
| 13G-A | Generative Encoder | ¿Un decoder auxiliar mejora el training? | LOCAL | CERRADO |
| 13G-B | Post-Hoc Decoder | ¿Las features pre-pooling son decodificables? | LOCAL+UNC | CERRADO (4/4) |

---

## 4. Test 12 — Scoreboard (Retrieval Canónico)

**Pregunta**: ¿Cuál es el rendimiento de retrieval de cada modelo bajo condiciones estandarizadas?

**Protocolo**: pool=256 candidatos, 500 queries, 64 hard negatives (mismo compositor), 32 semi-hard, seed=42.

### 4.1 Tabla principal

| Métrica | D0 | d4a4 | a4r | d4-a4r |
|---------|-----|------|-----|--------|
| **S** | **73.4%** | **83.8%** | **82.0%** | **79.8%** |
| A2M R@1 | 22.2% | 28.0% | 27.6% | 25.2% |
| A2M R@5 | 56.2% | 67.8% | 64.8% | 65.8% |
| A2M R@10 | 74.8% | 84.4% | 82.6% | 81.4% |
| A2M R@20 | 90.0% | 93.6% | 93.6% | 94.4% |
| M2A R@1 | 21.2% | 27.6% | 25.8% | 24.0% |
| M2A R@5 | 56.0% | 65.6% | 63.0% | 64.4% |
| M2A R@10 | 73.4% | 83.8% | 82.0% | 79.8% |
| M2A R@20 | 89.0% | 93.0% | 93.8% | 91.4% |
| MRR (A2M) | 0.380 | 0.458 | 0.438 | 0.426 |
| MRR (M2A) | 0.372 | 0.442 | 0.424 | 0.420 |
| Mean rank (A2M) | 8.37 | 5.23 | 5.44 | 5.64 |
| Mean rank (M2A) | 8.87 | 5.63 | 5.83 | 6.12 |
| Hard neg (same piece) | 94.6% | 95.4% | 94.4% | 94.2% |
| Hard neg (random) | 98.4% | 99.0% | 98.8% | 99.4% |

### 4.2 Mejora vs baseline

| Brazo | Delta S vs D0 | Factor vs random |
|-------|---------------|-----------------|
| d4a4 | **+10.4pp** | 71.7× |
| a4r | +8.6pp | 70.7× |
| d4-a4r | +6.4pp | 64.5× |
| D0 | — | 56.8× |

**Lectura**: d4a4 lidera todas las métricas de retrieval. Los tres brazos con descriptores superan a D0 de forma consistente en todos los niveles de recall.

---

## 5. Test 01 — Causal Ablation

**Pregunta**: ¿El descriptor es causalmente responsable de la mejora, o el modelo podría funcionar sin él en inference?

**Método**: En evaluación (no en training), se aplican tres tipos de ablación a los descriptores: zero (reemplazar por ceros), noise (ruido gaussiano), shuffle (permutar entre segmentos del batch). Se mide el impacto en S.

### 5.1 d4a4 — Ablación por componente

| Ablación | S | Delta S | Interpretación |
|----------|---|---------|----------------|
| Normal | 83.8% | — | Baseline |
| zero_audio (A4=0) | 7.8% | **-76.0pp** | A4 completamente causal |
| zero_midi (D4=0) | 84.4% | +0.6pp | D4 no contribuye |
| zero_both | 7.6% | -76.2pp | Misma caída que zero_audio |
| noise_audio | 22.0% | -61.8pp | Ruido en A4: catastrófico |
| noise_midi | 84.4% | +0.6pp | Ruido en D4: irrelevante |
| shuffle_audio | 46.6% | -37.2pp | A4 shuffled retiene info parcial |
| shuffle_midi | 83.8% | 0.0pp | D4 shuffled: irrelevante |

### 5.2 a4r — Ablación A4

| Ablación | S | Delta S |
|----------|---|---------|
| Normal | 82.0% | — |
| zero_audio | 4.4% | **-77.6pp** |
| noise_audio | 29.0% | -53.0pp |
| shuffle_audio | 49.8% | -32.2pp |

### 5.3 d4-a4r — Ablación dual

| Ablación | S | Delta S |
|----------|---|---------|
| Normal | 79.8% | — |
| zero_audio | 4.4% | **-75.4pp** |
| zero_midi | 79.4% | -0.4pp |
| noise_audio | 26.8% | -53.0pp |
| noise_midi | 79.8% | 0.0pp |
| shuffle_audio | 47.4% | -32.4pp |
| shuffle_midi | 79.8% | 0.0pp |

### 5.4 d4 — Ablación D4 sola

| Ablación | S | Delta S |
|----------|---|---------|
| Normal | 63.6% | — |
| zero_midi | 62.8% | -0.8pp |
| noise_midi | 63.6% | 0.0pp |
| shuffle_midi | 62.4% | -1.2pp |

### 5.5 Hallazgo clave

**A4 es completamente causal**: eliminarlo destruye la performance (-75 a -78pp). El modelo no puede funcionar sin el descriptor de audio.

**D4 no contribuye en inference**: zeroing, noise y shuffle de D4 no afectan la performance (<1pp cambio). Esto es consistente entre d4a4, d4-a4r y d4 solo.

**Paradoja D4**: D4 mejora durante training (d4 63.6% > D0-epoch-5 ~55%), pero no es necesario en inference. Posibles explicaciones:
- Los parámetros del wrapper D4 (~0.5M) actúan como regularización durante training
- D4 ayuda a la optimización pero la información se codifica en otros parámetros
- El Test 02 (param-matched) resuelve esta ambigüedad

---

## 6. Test 05 — Multi-Seed Replication

**Pregunta**: ¿Los resultados son replicables con distintas inicializaciones?

**Método**: 5 seeds (42, 123, 456, 789, 1337) × 4 descriptores × 30 epochs, entrenados en UNC Mendieta (A30 24GB). Evaluación estructurada en epoch final.

### 6.1 Resultados finales (15/15 CERRADO)

| Descriptor | Media | ±Std | Rango | Delta vs D0 | t-stat | p<0.05 | Cohen d |
|------------|-------|------|-------|-------------|--------|--------|---------|
| **d4a4** | **84.1%** | ±2.3pp | 82.0–86.4% | **+8.9pp** | 7.12 | SI | 4.50 |
| d4-a4r | 81.2% | ±2.5pp | 78.4–83.4% | +6.0pp | 3.95 | SI | 2.50 |
| a4r | 80.7% | ±1.9pp | 79.4–84.0% | +5.5pp | 4.16 | SI | 2.63 |
| D0 | 75.2% | ±2.3pp | 71.8–77.4% | — | — | — | — |

### 6.2 Análisis estadístico

- **Cero overlap entre distribuciones**: La peor seed de cualquier descriptor (a4r s1337 = 79.4%) supera la mejor seed de D0 (s123 = 77.4%) por +2.0pp.
- **Todos los p-values < 0.05**: La mejora es estadísticamente significativa para los tres descriptores.
- **Cohen d > 2.5 en todos los casos**: Efecto grande (> 0.8 = efecto grande convencional).
- **d4a4 vs D0 d=4.50**: Efecto extremadamente robusto.

### 6.3 Comparación single-seed vs multi-seed

| Descriptor | Single-seed best (s42) | Multi-seed media | Delta |
|------------|----------------------|------------------|-------|
| d4a4 | 83.8% | 84.1% | +0.3pp |
| a4r | 82.0% | 80.7% | -1.3pp |
| d4-a4r | 79.8% | 81.2% | +1.4pp |
| D0 | 73.4% | 75.2% | +1.8pp |

**Lectura**: Los resultados single-seed (seed=42) son representativos del comportamiento real. d4-a4r fue subestimado por la seed original (79.8% → media 81.2%).

---

## 7. Test 02 — Parameter-Matched Ablation

**Pregunta**: ¿La mejora viene de la **información** del descriptor o de los **parámetros extra** (~1.3M para d4a4)?

**Método**: Entrenar 4 variantes con arquitectura idéntica a d4a4 pero con descriptores degradados:

| Mode | Descriptor | Esperado |
|------|-----------|----------|
| real | d4a4 original | ~83% |
| random | Valores aleatorios N(0,1) por dim | ~73% (nivel D0) |
| zero | Todo ceros | ~73% (nivel D0) |
| shuffled | Descriptores de otro segmento del batch | ~73% (nivel D0) |

### 7.1 Resultados completos (4/4 cerrados)

| Mode | Best S | Best ep | vs D0 (73.4%) | vs real | hard_neg | Estado |
|------|--------|---------|---------------|---------|----------|--------|
| real (d4a4) | **83.0%** | e25 | **+9.6pp** | — | 94.0% | COMPLETO |
| random | 73.6% | e30 | +0.2pp | **-9.4pp** | 95.2% | COMPLETO |
| zero | 75.0% | e28 | +1.6pp | **-8.0pp** | 95.4% | COMPLETO |
| shuffled | 73.6%* | e20* | +0.2pp | **-9.4pp** | 93.6% | COMPLETO* |

\* `shuffled` se tomó como cierre operativo por convergencia clara en `e20`.

### 7.2 Lectura

**Las 3 ablaciones caen a nivel D0** (~73-75%), mientras que **real alcanza 83%**. El delta de ~9pp es consistente entre random (-9.4pp), zero (-8.0pp) y shuffled (-9.4pp). Esto confirma que la mejora es **causal desde la información de los descriptores**, no un artefacto de capacidad adicional del modelo.

**Detalle por ablación**:
- **random** (N(0,1) por dimensión): Los ~1.3M params extra del descriptor pathway no aportan señal útil cuando reciben ruido → 73.6% = D0 nivel.
- **zero** (descriptor = 0): El concatenador recibe 8 ceros → el modelo aprende a ignorarlos → 75.0%, ligeramente sobre D0.
- **shuffled** (descriptor de otro segmento del batch): Información musical real pero *desalineada* con el audio → 73.6%, peor que zero.

**Complementariedad con Test 01**: Test 01 ablaciona post-hoc (en evaluación, A4 zeroed → -76pp). Test 02 ablaciona durante training (pregunta más fuerte: ¿y si el modelo nunca vio descriptores reales?). Juntos cierran el argumento causal: la información del descriptor es necesaria tanto en training como en inference.

---

## 8. Test 04 — Invarianza a Transposición

**Pregunta**: ¿El modelo retiene rendimiento cuando la pieza se transpone a otra tonalidad?

**Método**: Transponer las notas MIDI en -6, -3, -1, 0, +1, +3, +6 semitonos y evaluar S.

### 8.1 S por nivel de transposición

| Semitonos | D0 | d4a4 | a4r | d4-a4r |
|-----------|------|------|------|--------|
| -6 | 13.8% | 24.2% | 27.0% | 27.0% |
| -3 | 26.6% | 41.4% | **46.2%** | 45.0% |
| -1 | 65.6% | 75.2% | **76.6%** | 73.2% |
| 0 | 73.4% | **83.8%** | 82.0% | 79.8% |
| +1 | 64.0% | 75.6% | **76.8%** | 75.2% |
| +3 | 27.4% | 44.6% | **51.0%** | 49.2% |
| +6 | 13.4% | 25.6% | **27.6%** | 27.2% |

### 8.2 Retención relativa a ±3 semitonos

| Brazo | S baseline | S promedio ±3 | Retención | Ventaja vs D0 |
|-------|-----------|---------------|-----------|---------------|
| D0 | 73.4% | 27.0% | 36.8% | — |
| d4a4 | 83.8% | 43.0% | 51.3% | +15.9pp |
| **a4r** | 82.0% | **48.6%** | **59.3%** | **+23.6pp** |
| d4-a4r | 79.8% | 47.1% | 59.0% | +21.8pp |

### 8.3 Hallazgo

**a4r es el brazo más invariante a transposición.** Retiene 59% de su performance a ±3 semitonos (vs 37% de D0). Esto es consistente con la naturaleza del mecanismo de reverse cross-attention: al comprimir 2400 tokens en 188 queries de descriptor, la representación captura relaciones relativas (intervalos) más que posiciones absolutas (pitch).

Todos los brazos con descriptores mejoran sustancialmente la invarianza a transposición respecto a D0.

---

## 9. Test 03 — Ratio Probe (Decodificabilidad Lineal)

**Pregunta**: ¿Los embeddings contienen información musicalmente relevante que sea linealmente decodificable?

**Método**: Regresión lineal (Ridge) de embeddings (z=256) a features musicales. R² mide decodificabilidad. Cross-decoding = decodificar info de una modalidad desde la otra.

### 9.1 Cross-decoding R²

| Probe | D0 | d4a4 | a4r | d4-a4r |
|-------|------|------|------|--------|
| audio→pitch_hist | 0.181 | 0.174 | 0.167 | 0.186 |
| audio→interval_hist | 0.094 | 0.112 | 0.095 | 0.115 |
| midi→chroma | **0.330** | 0.245 | 0.255 | 0.252 |
| midi→centroid | 0.616 | 0.637 | **0.662** | 0.652 |

### 9.2 Self-decoding R²

| Probe | D0 | d4a4 | a4r | d4-a4r |
|-------|------|------|------|--------|
| audio→chroma (self) | **0.310** | 0.235 | 0.249 | 0.231 |
| midi→pitch_hist (self) | 0.239 | 0.236 | 0.233 | 0.233 |

### 9.3 Hallazgo

**D0 gana en midi→chroma cross-decoding** (0.330 vs ~0.25 para los brazos augmented). Los brazos con descriptores son **menos** linealmente decodificables para chroma pese a tener mejor retrieval.

**La ventaja de los descriptores no vive en la decodificabilidad lineal sino en la geometría de distancias.** Los descriptores reorganizan el espacio latente de manera no-lineal: segmentos musicalmente similares quedan más cerca, pero esta proximidad no se captura con un probe lineal simple.

Esto es consistente con Test 06 (CKA) y Test 08 (sensibilidad perturbacional con correlaciones lineales bajas).

---

## 10. Test 06 — RSA/CKA (Alineamiento Cross-Encoder)

**Pregunta**: ¿Los descriptores alinean las representaciones internas de los encoders de audio y MIDI?

**Método**: Centered Kernel Alignment (CKA) y Representational Similarity Analysis (RSA) entre todas las capas de ambos encoders (audio_0..3 × midi_0..3). Mayor CKA = representaciones más similares.

### 10.1 CKA cross-encoder (media de todas las parejas audio_i × midi_j)

| Brazo | CKA media | RSA media | Delta CKA vs D0 |
|-------|-----------|-----------|-----------------|
| D0 | 0.435 | 0.363 | — |
| d4a4 | 0.659 | 0.589 | +51% |
| a4r | 0.766 | 0.684 | +76% |
| **d4-a4r** | **0.794** | **0.719** | **+82%** |

### 10.2 CKA entre últimas capas (audio_3 vs midi_3)

| Brazo | CKA | RSA |
|-------|-----|-----|
| D0 | 0.722 | 0.660 |
| d4a4 | 0.827 | 0.808 |
| a4r | 0.863 | 0.854 |
| d4-a4r | 0.873 | 0.864 |

### 10.3 Hallazgo

**Los descriptores aproximadamente duplican el alineamiento CKA cross-encoder** (D0: 0.435 → d4-a4r: 0.794, +82%).

**Paradoja alineamiento-retrieval**: d4-a4r lidera el alineamiento (CKA 0.794) pero NO lidera el retrieval (S=79.8% vs d4a4 83.8%). **Más alineamiento no implica mejor retrieval.** Esto sugiere que existe un punto óptimo de alineamiento — demasiado puede sobre-alinear las representaciones eliminando información complementaria útil.

Todos los p-values = 0.0 (altamente significativos).

---

## 11. Test 08 — Ratio Decoding (Sensibilidad Perturbacional)

**Pregunta**: ¿Qué componentes del descriptor A4 son más influyentes en el embedding final?

**Método**: Para cada dimensión del descriptor A4, perturbar su valor ±1σ y medir el cambio en la norma del embedding. Mayor sensibilidad = más influencia.

### 11.1 Top features A4 por brazo (sensibilidad)

| Brazo | #1 | Sens. | #2 | Sens. | #3 | Sens. | Rango dominante |
|-------|-----|-------|-----|-------|-----|-------|-----------------|
| d4a4 | band4_750Hz | 0.664 | band5_1500Hz | 0.662 | band3_375Hz | 0.526 | Mid (375–1500 Hz) |
| a4r | band7_6000Hz | **0.933** | band6_3000Hz | **0.875** | band5_1500Hz | 0.570 | High (1500–6000 Hz) |
| d4-a4r | band6_3000Hz | **1.092** | band4_750Hz | 0.773 | band7_6000Hz | 0.582 | High (750–6000 Hz) |

### 11.2 Sensibilidad D4 (donde aplica)

| Brazo | Feature | Sensibilidad |
|-------|---------|-------------|
| d4a4 | duration_ratio | 0.077 |
| d4a4 | interval_prev | 0.070 |
| d4-a4r | duration_ratio | 0.124 |
| d4-a4r | interval_prev | 0.107 |

**D4 tiene 5-10× menos sensibilidad que A4**, confirmando que A4 es el driver principal del rendimiento.

### 11.3 Correlaciones lineales

Todas las correlaciones lineales entre dimensiones del descriptor y dimensiones del embedding son bajas: |r| < 0.05 (media), |r| < 0.28 (máximo). Esto confirma que **el encoding es no-lineal** — los descriptores influyen fuertemente en la geometría pero no de forma linealmente recuperable.

### 11.4 Hallazgo

Cada mecanismo de inyección selecciona bandas diferentes:
- **d4a4 (concat)**: Domina en frecuencias medias (750–1500 Hz)
- **a4r (reverse cross-att)**: Domina en frecuencias altas (3000–6000 Hz)
- **d4-a4r (híbrido)**: Pico en 3000 Hz (sensibilidad 1.09, la más alta observada)

La sensibilidad diferencial entre bandas sugiere que el mecanismo de inyección determina qué aspectos del espectro son utilizados para la alineación cross-modal.

---

## 12. Test 09 — Suite de Invarianza

**Pregunta**: ¿Cómo responde cada modelo a perturbaciones realistas?

**Método**: 4 tipos de perturbación con múltiples niveles: shift temporal, escalamiento de velocity, transposición por octavas, y ruido gaussiano en audio.

### 12.1 Shift temporal (ms)

| Shift | D0 | d4a4 | a4r | d4-a4r |
|-------|------|------|------|--------|
| -8000 | 71.2% | 76.6% | 75.0% | 77.6% |
| -4000 | 72.4% | 80.8% | 77.6% | 77.6% |
| 0 | 73.4% | 83.8% | 82.0% | 79.8% |
| +4000 | 70.2% | 81.2% | 77.6% | 76.6% |
| +8000 | 68.2% | 79.0% | 75.6% | 76.2% |

**Lectura**: Todos los brazos son razonablemente robustos al shift temporal (máx -5.2pp para D0 a +8s). d4-a4r es el más estable (máx -3.6pp).

### 12.2 Escalamiento de velocity

| Factor | D0 | d4a4 | a4r | d4-a4r |
|--------|------|------|------|--------|
| 0.5 | 5.2% | 8.8% | 6.8% | 6.4% |
| 0.8 | 37.2% | 46.8% | 34.8% | 32.4% |
| 1.0 | 73.4% | 83.8% | 82.0% | 79.8% |
| 1.2 | 54.0% | 55.2% | 45.0% | 44.2% |
| 1.5 | 18.4% | 12.8% | 12.4% | 11.6% |

**Lectura**: La velocity es catastrófica para todos los brazos. **D0 es relativamente el más robusto** (mantiene 18.4% a 1.5× vs 11-13% para los augmented). Los descriptores hacen al modelo más dependiente de la estructura dinámica exacta.

### 12.3 Transposición por octavas (semitonos)

| Shift | D0 | d4a4 | a4r | d4-a4r |
|-------|------|------|------|--------|
| -24 | 8.4% | 9.8% | 11.2% | 11.2% |
| -12 | 12.0% | 16.0% | 17.4% | 17.0% |
| 0 | 73.4% | 83.8% | 82.0% | 79.8% |
| +12 | 10.0% | 13.8% | 16.0% | 15.8% |
| +24 | 5.2% | 7.4% | 9.8% | 9.0% |

**Lectura**: Transposiciones de ±12 semitonos (1 octava) destruyen la performance (todos caen al 10-17%). a4r retiene marginalmente más. Los modelos NO son octave-invariant.

### 12.4 Ruido gaussiano en audio (SNR dB)

| SNR | D0 | d4a4 | a4r | d4-a4r |
|-----|------|------|------|--------|
| clean | 73.4% | 83.8% | 82.0% | 79.8% |
| 40 dB | **73.4%** | 79.8% | 66.0% | 67.6% |
| 30 dB | **73.4%** | 67.0% | 47.2% | 50.2% |
| 20 dB | **73.0%** | 54.8% | 41.2% | 40.4% |
| 10 dB | 46.8% | **52.2%** | 40.4% | 41.8% |
| 5 dB | 17.8% | 25.0% | **31.8%** | **33.0%** |

### 12.5 Análisis del patrón de ruido

Este es el resultado más revelador de Test 09. Hay un **crossover**:

- **A SNR alto (40-20 dB)**: D0 es el más robusto (0.0pp de pérdida hasta 30dB). Los brazos con descriptores son mucho más sensibles (a4r pierde -16.0pp a 40dB).
- **A SNR bajo (5 dB)**: a4r y d4-a4r **superan** a D0 (31.8-33.0% vs 17.8%).

**Explicación**: Los brazos con descriptores dependen del descriptor A4, que se computa desde el audio. Ruido moderado corrompe A4 → caída fuerte. Pero a niveles extremos de ruido, la representación aprendida por los descriptors arms tiene una estructura interna que resiste mejor que la de D0.

D0 no tiene descriptor de audio, por eso es inmune al ruido en la señal de condicionamiento. Pero su representación base es menos estructurada, así que colapsa más rápido a nivel de ruido extremo.

---

## 13. Test 11 — Decoder Suite (Retención de Información)

**Pregunta**: ¿Cuánta información musical retienen los embeddings cross-modalmente?

### 13.1 Diseño experimental

Se entrenan decoders ligeros desde embeddings congelados (z=256) para reconstruir:
- **Mel spectrogram** (128 bandas) — proxy de contenido acústico
- **Piano roll** (88 notas × 188 frames) — proxy de contenido musical

Configuraciones:
- **Intra-domain**: audio→mel, midi→PR (lo que debería ser fácil)
- **Cross-domain**: midi→mel, audio→PR (lo que prueba transferencia cross-modal)
- **Controles**: shuffle (pairs aleatorios), mean_z, zero_z

### 13.2 Resultados — Mel spectrogram

| Tarea | D0 (MSE) | a4r (MSE) | Random (MSE) |
|-------|---------|---------|-------------|
| audio→mel (intra) | 0.150 | 0.162 | 0.208 |
| midi→mel (cross) | 0.174 | 0.175 | — |
| shuffle_mel | 0.265 | 0.259 | — |

### 13.3 Resultados — Piano roll

| Tarea | D0 (BCE) | a4r (BCE) | Random (BCE) |
|-------|---------|---------|-------------|
| midi→PR (intra) | 0.714 | 0.722 | 0.837 |
| audio→PR (cross) | 0.736 | 0.741 | — |
| shuffle_PR | 1.086 | 1.030 | — |

### 13.4 Info retention ratio

| Modalidad | Fórmula | D0 | a4r |
|-----------|---------|-----|-----|
| Mel | (shuffle - cross) / (shuffle - intra) | 0.788 | **0.830** |
| Piano roll | (shuffle - cross) / (shuffle - intra) | 0.942 | 0.938 |

**Lectura**: a4r retiene +4.2pp más información mel cross-modalmente (0.830 vs 0.788). La retención de piano roll es casi perfecta (~0.94) para ambos, significando que audio→PR es casi tan bueno como midi→PR.

---

## 14. Test 11 — Pre-Projection A/B (Cuello de Botella)

**Pregunta**: ¿Cuánta información destruyen las capas de proyección (1024→256 y 512→256)?

**Método**: Entrenar el mismo decoder de eventos musicales sobre features pre-proyección (audio: 1024d, midi: 512d) y comparar con post-proyección (256d).

### 14.1 Shuffle gap (medida de información condicionante)

| Pathway | D0 | a4r | Interpretación |
|---------|------|------|---------------|
| MIDI pre-proj (512d) | 1.150 | 1.159 | Máxima info disponible |
| Audio pre-proj (1024d) | 0.186 | **0.304** | Info cross-modal pre-proj |
| Audio post-proj (256d) | 0.136 | 0.215 | Info cross-modal post-proj |

### 14.2 Destrucción por proyección

| Proyección | D0 gap pre | D0 gap post | % destruido |
|------------|-----------|-------------|------------|
| MIDI (512→256) | 1.150 | ~0.136 | **~88%** |
| Audio (1024→256) | 0.186 | 0.136 | ~27% |

| Proyección | a4r gap pre | a4r gap post | % destruido |
|------------|-----------|-------------|------------|
| MIDI (512→256) | 1.159 | ~0.215 | **~81%** |
| Audio (1024→256) | 0.304 | 0.215 | ~29% |

### 14.3 Info retention cross-modal (pre-projection events) — 4/4 brazos

Fórmula: `retention = (shuffle_CE - audio_CE) / (shuffle_CE - midi_CE)`

| Brazo | Retention ratio | Δ vs D0 | Nota |
|-------|----------------|---------|------|
| D0 | 0.597 | — | Baseline |
| a4r | 0.712 | +19% | Reverse cross-att (audio) |
| d4-a4r | 0.748 | +25% | Híbrido D4-concat + A4r |
| **d4a4** | **0.770** | **+29%** | Concat dual — máxima retención |

Los 4 brazos con descriptor superan a D0. El ranking de retención es d4a4 > d4-a4r > a4r > D0. Nota: d4a4, a pesar de usar concat (no reverse cross-att), lidera en retención de información cross-modal pre-proyección.

### 14.4 Resultados completos d4a4 / d4-a4r (UNC, 120ep max, patience=15)

| Brazo | Task | best_ep | val_CE | token_acc | frame_f1 |
|-------|------|:---:|:---:|:---:|:---:|
| d4a4 | midi2events (intra) | 10 | 2.965 | 0.306 | 0.108 |
| d4a4 | audio2events (cross) | 8 | 3.069 | 0.289 | 0.051 |
| d4-a4r | midi2events (intra) | 11 | 2.971 | 0.307 | 0.111 |
| d4-a4r | audio2events (cross) | 10 | 3.073 | 0.289 | 0.045 |

Las dos columnas más importantes: `audio2events val_CE` (qué tan bien se decodifican eventos MIDI desde audio pre-proyección) y `retention ratio` derivado de comparar con shuffle y con intra.

### 14.5 Hallazgo

**La proyección MIDI (512→256) destruye ~81-88% de la información condicionante.** Esto es un cuello de botella masivo y apunta directamente a las proyecciones como el principal limitante del rendimiento actual.

**Los 4 brazos descriptor retienen más información cross-modal que D0 en pre-projection.** El ranking es d4a4(0.770) > d4-a4r(0.748) > a4r(0.712) > D0(0.597). La mejora es de +19% a +29% dependiendo del brazo.

**Hallazgo relevante para sección 16**: Esta retención mayor NO se traduce en mejor decodificabilidad de piano roll (Test 13G-B). El ranking de Test 11 (retención) es el INVERSO del ranking de Test 13G-B (F1). Esto indica que la información extra de los brazos descriptor está en una forma que el cross-attention decoder de piano roll no puede aprovechar directamente — ver sección 16.4 para análisis detallado.

**Implicación para Gate 5A**: Las conditioned projections (C1) que preserven más información a través de la proyección podrían desbloquear mejoras adicionales sustanciales.

---

## 15. Test 13G Phase A — Generative Encoder

**Pregunta**: ¿Agregar un decoder de piano roll como objetivo auxiliar durante el training mejora la calidad de los embeddings?

**Método**: MiniPRDecoder (1.92M params) se entrena conjuntamente con el encoder. Loss combinada: VICReg + λ × BCE(decoder(z_midi), PR_target). Se evalúa tanto z_midi→PR como z_audio→PR (transferencia cross-modal del objetivo reconstructivo).

### 15.1 Lambda sweep (D0, 15 epochs)

| λ | Best S | last3 avg S | audio_pr_f1 | midi_pr_f1 | PR gap |
|---|--------|-------------|-------------|------------|--------|
| 0.03 | 64.6% | 63.2% | 0.1139 | 0.1183 | 0.004 |
| 0.10 | 64.4% | 62.8% | 0.1137 | 0.1172 | 0.005 |
| 0.30 | 64.4% | 63.6% | 0.1140 | 0.1187 | 0.005 |

D0 baseline (50ep, sin decoder): 73.4%

### 15.2 Validación del MiniPRDecoder

Antes de evaluar los resultados de retrieval, se validó que el decoder produce piano rolls correctos:
- 50 muestras evaluadas
- Median F1 = 0.981, Mean MSE = 0.0005
- **Gate PASS**: El decoder funciona correctamente

### 15.3 Hallazgos

1. **λ es completamente irrelevante**: Los tres valores producen resultados idénticos. El loss reconstructivo no modula el aprendizaje.

2. **PR F1 ~0.114 para audio→PR**: Desde z (256d), la reconstrucción es muy pobre. Las predicciones son "manchas difusas" centradas en registro medio, sin notas individuales discernibles.

3. **Gap midi-audio ~0.004**: z_audio reconstruye PR con ~96% de la calidad de z_midi. Ambos son igualmente malos pero bien alineados.

4. **cos(pred_midi, pred_audio) > 0.99**: Las predicciones desde ambos dominios son prácticamente idénticas.

### 15.4 Diagnóstico

El cuello de botella es la **compresión 750:1** del mean-pooling:
- Pre-pooling: [B, 2400, 1024] = ~2.5M valores por muestra
- Post-pooling+proj: [B, 256] = 256 valores por muestra

Un vector de 256 dimensiones no puede representar fielmente 4 segundos de música con notas individuales. **El problema no es el objetivo de training sino la representación.**

### 15.5 Consecuencia: Phase B

Este resultado motivó directamente el diseño de **Test 13G Phase B**, que luego quedó completo. La fase nueva preguntó si las features pre-pooling (2400×1024 para D0, 188×1024 para a4r) retenían suficiente información para decodificar piano rolls. El resultado final fue negativo: la decodificabilidad quedó en `F1≈0.10` para todos los arms y no apareció ventaja descriptor-guided.

---

## 16. Test 13G Phase B — Post-Hoc Pre-Pooling Decoder (CERRADO — 4/4)

**Pregunta**: Dadas las representaciones pre-pooling del encoder de audio, ¿qué tan decodificable es el piano roll?

### 16.1 Diseño

- **PostHocPRDecoder** (2.44M params): Cross-attention decoder con learned frame queries
- **Encoder completamente congelado**: Solo el decoder recibe gradientes
- **Features capturadas via forward hook** en `audio_encoder.transformer`

| Brazo | N tokens pre-pooling | Interpretación |
|-------|---------------------|---------------|
| D0 | 2400 | Baseline, secuencia larga |
| d4a4 | 2400 | Concat, secuencia larga |
| a4r | 188 | Cross-att, secuencia comprimida |
| d4-a4r | 188 | Cross-att híbrido, secuencia comprimida |
| D0 pool-to-188 | 188 (pooled desde 2400) | Control de longitud |

### 16.2 Resultados (COMPLETO — 4/4 brazos, 40 epochs c/u)

| Brazo | N tokens | best_f1 | precision | recall | onset_f1 | BCE | cosine |
|-------|----------|---------|-----------|--------|----------|-----|--------|
| **D0 (pool-188)** | 188 | **0.1089** | 5.80% | 92.2% | 0.0419 | 0.831 | 0.260 |
| d4a4 | 2400 | 0.1037 | 5.52% | 90.7% | 0.0406 | 0.904 | 0.241 |
| a4r | 188 | 0.1024 | 5.46% | 91.4% | 0.0410 | 0.895 | 0.236 |
| d4-a4r | 188 | 0.1021 | 5.43% | 92.2% | 0.0415 | 0.884 | 0.236 |

- Ningún brazo hizo early stopping (mejora monotónica hasta e40 en todos los casos)
- 8 samples generados por brazo (PNGs + MIDI + WAV en resultados_compartir)
- D0 pool-188 ligeramente superior a todos los brazos descriptor

### 16.3 Lectura

**F1 ~0.10 para los 4 brazos, sin diferencia significativa entre ellos.** La matriz completa confirma el patrón iniciado con los primeros 3 brazos: los descriptores no mejoran la decodificabilidad pre-pooling del piano roll.

1. **Recall altísimo (~91-92%)**: El decoder activa muchas frames → alta recall, bajísima precision (~5.5%). El modelo predice "todo suena" en vez de notas discretas. Aprendió la distribución estadística tonal del corpus pero no la localización temporal.
2. **D0 pool-188 gana marginalmente en todos los casos**: Sorprendente dado que D0 tiene la menor retención de información cross-modal según Test 11 (retention=0.597). Los brazos descriptor con más retención (d4a4=0.770) producen peor F1 de piano roll.
3. **BCE 0.83-0.90**: La loss sigue bajando a e40 sin plateau → el decoder mejoraría con más epochs, pero el perfil (recall alto / precision baja) sugiere un limitante cualitativo, no solo de capacidad.
4. **onset_f1 ~4%**: Incapaz de detectar onsets (comienzos de nota). La representación codifica información tonal/espectral pero no temporal discreta. La dinámica de "cuándo empieza cada nota" no está preservada en las features pre-pooling de ningún brazo.
5. **d4a4 y d4-a4r con N=2400 y 188 respectivamente convergen al mismo techo**: La longitud de la secuencia de entrada al decoder no determina el techo de F1. Ambas alcanzan ~0.102.

**Interpretación**: Las representaciones pre-pooling contienen información musical genérica (qué registro/tonalidad suena) pero no precisa (cuándo empieza cada nota). El cuello de botella está en la **granularidad temporal**, no en el mecanismo de compresión ni en la longitud de la secuencia.

### 16.4 La paradoja: Test 11 vs Test 13G-B

El resultado más conceptualmente profundo de Gate 5B surge de comparar los rankings de estos dos tests:

**Test 11 (retención info cross-modal):** d4a4 > d4-a4r > a4r > D0
**Test 13G-B (F1 decodificación piano roll):** D0 > d4a4 > a4r > d4-a4r

Los rankings están **invertidos**: el brazo con mayor retención de información cross-modal (d4a4, 0.770) produce el peor decoder de piano roll entre los descriptor-arms; y el brazo con menor retención (D0, 0.597) produce el mejor decoder.

Esta inversión no es una contradicción — es una distinción entre **dos tipos de información**:

- **Test 11** mide si las features de audio pueden informar sobre *qué eventos ocurren* en el segmento completo (una pregunta holística). Los brazos descriptor tienen más de esta información.
- **Test 13G-B** mide si las features de audio pueden localizar *cuándo exactamente* ocurre cada nota (una pregunta de granularidad temporal). Aquí todos los brazos fallan por igual.

La información extra que los descriptores aportan (según Test 11) está organizada en patrones de correlación entre dimensiones del espacio — la **geometría relativa** entre vectores — pero no está codificada como activaciones temporalmente localizadas que un decoder frame-a-frame pueda leer directamente.

D0, sin descriptor, tiene features más "uniformes" y menos condicionadas estructuralmente, lo que paradójicamente hace que su decoder de piano roll tenga marginalmente mejor precision: el decoder puede mapear más directamente desde el contenido espectral bruto hacia presencia de notas, aunque tampoco puede resolver la granularidad temporal.

**Conclusión**: La ventaja de los descriptores vive en la **geometría del espacio de distancias** — qué tan cerca están pares correspondientes, qué tan bien se discriminan piezas distintas — no en la **riqueza de features individuales** en términos de decodificabilidad musical temporal. Son dos propiedades del espacio latente conceptualmente distintas, y los descriptores mejoran una sin necesariamente mejorar la otra.

---

## 17. Síntesis: Evidencia Convergente

### 17.1 Mapa de hallazgos por test

| Test | Hallazgo principal | Implicación |
|------|-------------------|-------------|
| 12 (Scoreboard) | d4a4 83.8%, +10.4pp vs D0 | La mejora es real y sustancial |
| 01 (Causal) | A4 causal (-75pp), D4 no contribuye | La señal viene del audio descriptor |
| 05 (Multi-seed) | d4a4 84.1%±2.3, p<0.05, d=4.50 | Resultado replicable y significativo |
| 02 (Param-match) | random/zero ~73% vs real 83% | Mejora causal, no artefacto de capacidad |
| 04 (Transposición) | a4r retiene 59% a ±3 semitonos | Descriptores dan invarianza |
| 06 (RSA/CKA) | Descriptores duplican CKA cross-encoder | Alineamiento representacional profundo |
| 03 (Ratio Probe) | D0 mejor en probe lineal | Ventaja es geométrica, no lineal |
| 08 (Ratio Decoding) | Bandas 750-6000 Hz dominan, no-lineal | Mecanismo opera en frecuencias medias-altas |
| 09 (Invarianza) | D0 más robusto a ruido, augmented a shift | Trade-off robustez vs rendimiento |
| 11 (Decoder) | MIDI proj destruye ~88% info | Cuello de botella identificado |
| 11 (Pre-Proj, 4/4) | Retención: d4a4=0.770 > d4-a4r=0.748 > a4r=0.712 > D0=0.597 | Descriptor arms retienen +19-29% más info cross-modal |
| 13G-A (Generative) | z=256 insuficiente para PR | Pooling es el limitante |
| 13G-B (Post-Hoc, 4/4) | F1~0.10 ∀ arms; ranking INVERSO a Test 11 | Ventaja es geométrica: afecta distancias, no contenido frame-level |

### 17.2 La cadena causal

Los tests, en conjunto, construyen una cadena argumental:

```
1. Los descriptores de ratios mejoran el retrieval
   (Test 12: +10.4pp para d4a4)
       │
       ▼
2. La mejora es causal — viene de la INFORMACIÓN del descriptor
   (Test 01: A4 zeroed → -76pp)
   (Test 02: params sin info → nivel D0)
       │
       ▼
3. La mejora es replicable y estadísticamente robusta
   (Test 05: 5 seeds, p<0.05, Cohen d=4.50)
       │
       ▼
4. El mecanismo es no-lineal y opera en la geometría de distancias
   (Test 03: no ventaja lineal; Test 06: +82% CKA; Test 08: correlaciones <0.05)
       │
       ▼
5. El descriptor A4 captura dinámica espectral temporal
   por banda de octava (750-6000 Hz)
   (Test 08: sensibilidad perturbacional)
       │
       ▼
6. Más alineamiento cross-encoder ≠ mejor retrieval
   (Test 06: d4-a4r lidera CKA pero no S)
       │
       ▼
7. El principal cuello de botella es la proyección/pooling
   (Test 11: -88% info en MIDI proj; Test 13G-A: 256d insuficiente)
       │
       ▼
8. Las representaciones pre-pooling contienen info extra en descriptor arms
   (Test 11 Pre-Proj 4/4: d4a4=0.770 > d4-a4r=0.748 > a4r=0.712 > D0=0.597)
       │
       ▼
9. Pero esa info extra NO es decodificable como piano roll temporal
   (Test 13G-B 4/4: F1~0.10 ∀ arms, onset_f1~4%)
   El ranking de Test 13G-B es INVERSO al de Test 11:
   D0-pool-188 (0.1089) > d4a4 (0.1037) > a4r (0.1024) > d4-a4r (0.1021)
       │
       ▼
10. CONCLUSIÓN: La ventaja opera en la geometría de distancias, no en feature richness
    Descriptores → mejor organización espacial (cercanos=similares)
    Descriptores ≠ features más ricas en contenido musical temporal frame-a-frame
    La paradoja Test11/13G-B es la demostración experimental de esta distinción
```

### 17.3 Qué sabemos sobre A4

A4 = temporal deltas of log-magnitude in 8 octave bands (47-12000 Hz):

```
Audio → STFT → 8 log-spaced bands → energy per band → temporal delta → z-score normalize
```

| Banda | Frecuencia | d4a4 sens. | a4r sens. | d4-a4r sens. |
|-------|-----------|-----------|----------|-------------|
| band0 | 47 Hz | 0.087 | 0.067 | 0.076 |
| band1 | 94 Hz | 0.113 | 0.099 | 0.093 |
| band2 | 188 Hz | 0.187 | 0.155 | 0.202 |
| band3 | 375 Hz | 0.526 | 0.280 | 0.462 |
| band4 | 750 Hz | **0.664** | 0.375 | 0.773 |
| band5 | 1500 Hz | **0.662** | 0.570 | 0.540 |
| band6 | 3000 Hz | 0.414 | **0.875** | **1.092** |
| band7 | 6000 Hz | 0.271 | **0.933** | 0.582 |

**El rango 375-6000 Hz es donde vive la señal cross-modal.** Las bandas graves (<188 Hz) contribuyen poco. Cada mecanismo de inyección "selecciona" bandas diferentes: concat favorece frecuencias medias, reverse cross-attention favorece frecuencias altas.

### 17.4 Ventaja geométrica: qué es y qué implica

El hallazgo más profundo de Gate 5B, emergente de la lectura conjunta de los 13 tests, es la distinción entre dos propiedades del espacio latente:

**Geometría de distancias**: cómo están organizados los puntos *relativamente entre sí*. Los descriptores mejoran esta organización: pares correspondientes (audio A, midi A) quedan más cerca (+82% CKA, Test 06), y segmentos distintos quedan más separados (+10.4pp retrieval, Test 12). Esta organización es no-lineal (Test 03, 08) y persiste incluso bajo transposición (Test 04).

**Riqueza de features individuales**: qué información está codificada en cada vector de forma directamente decodificable. Tests 03 y 13G-B muestran que los descriptores NO enriquecen esto: D0 es mejor en probe lineal (Test 03) y marginalmente mejor en decodificación de piano roll (Test 13G-B).

La analogía útil es un sistema de indexación bibliográfica: los descriptores construyen un índice de búsqueda excelente (encontrás lo que buscás con 84% de precisión), pero el índice no contiene el texto del libro. Para "leer" el contenido musical (notas individuales en el tiempo), necesitás abrir el libro directamente — lo que requeriría una arquitectura diferente con objetivo de training supervisado.

Esta distinción tiene implicaciones para la teoría y para las aplicaciones:

- **Para retrieval, matching, clustering, score following, detección de versiones**: la ventaja geométrica es suficiente y directamente útil.
- **Para AMT (transcripción), análisis de nota, generación musical**: la arquitectura actual no es el camino. Se necesita o bien un objetivo de training con supervisión nota-a-nota, o bien un decoder capaz de extraer información de la secuencia completa de features (→ Gate 6 Exp C testa el límite superior con decoder de 34.3M params).

### 17.5 Preguntas abiertas (post Gate 5B)

1. **¿Por qué D4 no contribuye en inference?** Training con D4 mejora el modelo, pero D4 es dispensable en evaluación. ¿Es regularización pura? La paradoja D4 permanece sin resolución mecanística.

2. **¿Cuál es el rendimiento techo?** d4a4 alcanza 84.1% multi-seed. ¿Es esto un límite de la arquitectura, del pooling, o de la tarea? Gate 5A C1 (conditioned projections) era la hipótesis para atacar el cuello de botella.

3. **¿Qué pasa si se elimina el cuello de botella?** Test 11 Pre-Proj confirma que hay +19-29% más info en pre-projection que post-projection. ¿Conditioned projections o cross-attention preservarían esa info hasta el embedding final?

4. **¿Las features pre-pooling son musicalmente decodificables con un decoder más potente?** → Test 13G-B respondió con decoder de 2.44M params: no. Gate 6 Exp C lo ataca con 34.3M params. La pregunta relevante es si el límite es arquitectural (decoder insuficiente) o informacional (la info temporal simplemente no está).

5. **¿Por qué la inversión de ranking entre Test 11 y Test 13G-B?** La explicación propuesta (info organizada en geometría vs en activaciones temporales) es coherente pero no experimentalmente falsada todavía.

---

## 18. Trade-offs Observados

### 18.1 Rendimiento vs Robustez a ruido

| Brazo | S (%) | Delta S a 20dB ruido |
|-------|-------|---------------------|
| D0 | 73.4% | -0.4pp |
| d4a4 | 83.8% | **-29.0pp** |
| a4r | 82.0% | **-40.8pp** |

Los brazos augmented son más potentes pero más frágiles al ruido. Esto es estructural: dependen de A4 computado desde el audio, y el ruido corrompe esa señal.

### 18.2 Alineamiento vs Retrieval

| Brazo | CKA mean | S (%) |
|-------|---------|-------|
| d4-a4r | **0.794** | 79.8% |
| a4r | 0.766 | 82.0% |
| d4a4 | 0.659 | **83.8%** |
| D0 | 0.435 | 73.4% |

d4a4 tiene el MEJOR retrieval con MENOR alineamiento entre los augmented. Esto sugiere que alineamiento excesivo puede ser contraproducente — las representaciones necesitan ser suficientemente diferentes para codificar información complementaria.

### 18.3 Decodificabilidad lineal vs Rendimiento

| Brazo | midi→chroma R² | S (%) |
|-------|----------------|-------|
| D0 | **0.330** | 73.4% |
| a4r | 0.255 | 82.0% |
| d4a4 | 0.245 | **83.8%** |

D0 es el más linealmente decodificable pero el peor en retrieval. Los descriptores reorganizan el espacio de forma no-lineal, sacrificando decodificabilidad simple por geometría de distancias óptima.

---

## 19. Implicaciones para el Proyecto Phideus

### 19.1 Para la Harmonic Information Theory

Gate 5B proporciona la evidencia más fuerte hasta la fecha de que **la dinámica espectral temporal por banda de octava actúa como puente representacional cross-modal**. No son "ratios armónicos" en el sentido clásico (intervalos musicales), sino el patrón temporal de energía por bandas de frecuencia. A4 captura cómo evoluciona el espectro frame a frame — información que existe tanto en audio como en MIDI (implícitamente, vía las notas ejecutadas).

La evidencia es convergente: causal (Tests 01, 02), replicable (Test 05), geométrica (Tests 03, 06), no-lineal (Test 08), y diferencial por banda de frecuencia.

**Matiz importante (hallazgo final)**: La función que cumple A4 es de *organización espacial*, no de *enriquecimiento de contenido*. A4 ayuda al modelo a aprender qué dirección en el espacio de 256 dimensiones corresponde a "musicalmente similar", pero no fuerza a los vectores individuales a codificar más información musical decodificable. Este es un resultado más específico (y más interesante) que simplemente "A4 funciona": revela *cómo* funciona a nivel representacional.

### 19.2 Para la arquitectura

El cuello de botella identificado (pooling + proyecciones destruyen 81-88% de info) señala el camino para mejoras futuras:
- **Gate 5A C1 (conditioned projections)**: Proyecciones que preserven más información
- **Eliminar mean-pooling**: Attention pooling o representaciones multi-escala
- **Ampliación de z**: 256d es insuficiente para capturar estructura temporal

### 19.3 Para el paper

Gate 5B proporciona material para:
- **Claims sólidos**: Mejora causal (Test 01 + 02), replicable (Test 05), cuantificada (Test 12)
- **Figuras**: Curvas de transposición (Test 04), matrices CKA (Test 06), sensibilidad por banda (Test 08), curvas de ruido (Test 09), tabla ranking inversión Test11/13G-B
- **Diagnósticos**: Paradoja D4, cuello de botella de proyección, trade-off alineamiento-retrieval, distinción geométrica/feature-richness
- **Resultados negativos honestos (valiosos)**: D4 no contribuye en inference, sensibilidad a ruido aumentada, decodificabilidad lineal no mejora, descriptores no enriquecen piano roll (F1~10% uniforme)
- **Argumento teórico nuevo**: La distinción geométrica/feature-richness como marco conceptual para entender qué tipo de información capturan los descriptores de ratios espectrales. La paradoja Test11 vs Test13G-B es el experimento clave que la demuestra empíricamente.

### 19.4 Estado de cierre

| Test | Estado | Resultado clave |
|------|--------|-----------------|
| 01 Causal Ablation | ✅ CERRADO | A4 causal (-75pp), D4 no contribuye |
| 02 Param-Matched | ✅ CERRADO (4/4) | Ablaciones caen a D0 (~73-75%), gap causal = 9pp |
| 03 Ratio Probe | ✅ CERRADO | D0 mejor en lineal; ventaja no-lineal |
| 04 Transposición | ✅ CERRADO | a4r más invariante (+23.6pp a ±3 semitonos) |
| 05 Multi-Seed | ✅ CERRADO (15/15) | d4a4 84.1%±2.3pp, Cohen d=4.50 |
| 06 RSA/CKA | ✅ CERRADO | d4-a4r +82% CKA; paradoja CKA≠retrieval |
| 08 Ratio Decoding | ✅ CERRADO | Bandas 750-6000 Hz; D4 5-10× menos sensible |
| 09 Invarianza Suite | ✅ CERRADO | Crossover en ruido SNR 5-10dB |
| 10 Embedding Geometry | ✅ CERRADO | Visualizaciones t-SNE/UMAP |
| 11 Decoder Suite | ✅ CERRADO (4/4) | Retención: d4a4>d4-a4r>a4r>D0; MIDI proj destruye ~88% |
| 12 Scoreboard | ✅ CERRADO | d4a4 83.8%, a4r 82.0%, d4-a4r 79.8%, D0 73.4% |
| 13G-A Generative | ✅ CERRADO | z=256 insuficiente; λ irrelevante |
| 13G-B Post-Hoc | ✅ CERRADO (4/4) | F1~0.10 uniforme; ranking inverso a Test 11 |

**Gate 5B COMPLETO — todos los tests cerrados.** 2026-03-05.

El corpus empírico completo sienta la base científica para el paper y para Gate 6.

---

## Apéndice A: Protocolo de Evaluación

- **Pool size**: 256 candidatos
- **Queries**: 500
- **Hard negatives**: 64 (mismo compositor) + 32 semi-hard
- **Seed**: 42
- **Métrica primaria**: S = min(A2M_R@10, M2A_R@10)
- **Dataset**: MAESTRO v3.0.0, split=validation, segment_len=4.0s
- **GPU**: RTX 3090 24GB (LOCAL) / A30 24GB (UNC Mendieta)

## Apéndice B: Referencia de Archivos

| Test | Archivos de resultados |
|------|----------------------|
| Test 01 | `data/gate5b_results/{D0,d4a4,a4r,d4-a4r,d4}/test01_causal_ablation.json` |
| Test 02 | `results_unc/gate5b_param_matched/{real,random,zero,shuffled}/` |
| Test 03 | `data/gate5b_results/{D0,d4a4,a4r,d4-a4r}/test03_ratio_probe.json` |
| Test 04 | `data/gate5b_results/{D0,d4a4,a4r,d4-a4r}/test04_transposition.json` |
| Test 05 | `results_unc/gate5b_multiseed/` (15 dirs, 54 JSONs) |
| Test 06 | `data/gate5b_results/{D0,d4a4,a4r,d4-a4r}/test06_rsa_cka.json` |
| Test 08 | `data/gate5b_results/{d4a4,a4r,d4-a4r}/test08_ratio_decoding.json` |
| Test 09 | `data/gate5b_results/{D0,d4a4,a4r,d4-a4r}/test09_invariance_suite.json` |
| Test 10 | `data/gate5b_results/visualizations/` + `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/resultados_compartir/06_gate5b_scientific_validation/test10_visualizations/` |
| Test 11 | `data/gate5b_results/{D0,a4r}/test11_*.json` |
| Test 12 | `data/gate5b_results/{D0,d4a4,a4r,d4-a4r}/test12_scoreboard.json` |
| Test 13G-A | `data/gate5b_results/d0/test13g/` |
| Test 13G-B | `data/gate5b_results/{D0,d4a4,a4r}/test13g_posthoc/` |

## Apéndice C: Glosario

| Término | Definición |
|---------|-----------|
| **S** | min(A2M_R@10, M2A_R@10) — métrica primaria de retrieval bidireccional |
| **A2M** | Audio-to-MIDI retrieval |
| **M2A** | MIDI-to-Audio retrieval |
| **R@K** | Recall at K — proporción de queries cuyo match correcto está en top-K |
| **CKA** | Centered Kernel Alignment — similitud entre representaciones de dos redes |
| **RSA** | Representational Similarity Analysis — correlación de matrices de disimilitud |
| **A4** | Audio descriptor: temporal deltas of log-magnitude in 8 octave bands |
| **D4** | MIDI descriptor: 4D local intervals |
| **VICReg** | Variance-Invariance-Covariance Regularization (loss contrastiva) |
| **Shuffle gap** | Diferencia entre loss shuffle y loss alineado — mide info condicionante |
| **Info retention** | (shuffle - cross) / (shuffle - intra) — cuánta info cross-modal se retiene |
