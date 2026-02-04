# Escalón 1: MAESTRO (Audio ↔ MIDI) - Informe Final de Resultados

**Fecha**: 2026-02-04
**Estado**: ✗ NO-GO (pero científicamente informativo)
**Autor**: Claude Code + Mar-IA-no

---

## Resumen Ejecutivo

El Escalón 1 buscaba demostrar que es posible aprender representación cross-modal entre Audio real y MIDI usando "ratio language" (constelaciones de ratios de frecuencia estilo Shazam).

**Resultado**: El experimento produjo un **NO-GO informativo**:
- ✓ Las distribuciones de tokens son compatibles (cosine > 0.95)
- ✓ El algoritmo Shazam funciona correctamente (Oracle 90.9%)
- ✗ Los hashes Audio↔MIDI NO coinciden para el mismo contenido musical
- ✗ El "ratio language" NO es cross-modalmente identificador a nivel de tokens

---

## Cronología del Experimento

### Fase 1: Implementación Inicial (2026-02-04)

**Archivos creados:**
```
src/utils/midi_utils.py              # Parseo MIDI, piano roll, constellation tokens
src/RNA/vicreg.py                    # VICReg loss + encoder
src/RNA/barlow_twins.py              # Barlow Twins loss + encoder
src/analizador/analizador_maestro.py # Extracción constellation audio+MIDI

experiments/maestro/
├── gate0_harness.py                 # Métricas + controles negativos
├── gate1_ingest.py                  # Descarga + segmentación MAESTRO
├── gate2_baselines.py               # Chroma + CCA baselines
├── gate3_cross_modal.py             # Training VICReg/Barlow
├── gate4_ratio_tokens.py            # Training constellation
├── gate5_moco.py                    # MoCo queue + hard negatives
└── run_maestro_experiment.py        # Orquestador principal
```

**Dataset MAESTRO v3.0.0:**
- Descargado y descomprimido (121GB)
- 1,276 pares Audio-MIDI de piano
- Ubicación: `data/maestro_v3/maestro-v3.0.0/`

### Fase 2: Prueba con Pocos Pares

En lugar de ejecutar todo MAESTRO, se decidió validar primero con 10 pares seleccionados.

**Directorio de trabajo:**
```
experiments/un_audio_un_midi/
├── Un par/                    # Par inicial de prueba
├── Varios_pares/              # 10 pares para validación
└── *.py                       # Scripts de prueba
```

### Fase 3: Extractor V1 - Problema Detectado

**Script**: `test_single_pair.py`

**Resultados V1 (problemáticos):**
| Métrica | Audio | MIDI |
|---------|-------|------|
| Tokens totales | 2.6M | 7K |
| Mean log_ratio | 0.086 | 1.32 |
| Cosine similarity | 0.13 | - |

**Diagnóstico**: El extractor de audio colapsaba a ratio≈1 (tokens triviales).

### Fase 4: Extractor V2 - Correcciones GPT5.2Think

**Script**: `test_single_pair_v2_parallel.py`

**Correcciones aplicadas:**
1. `peaks_per_frame_max = 8` (antes saturaba a ~62)
2. Forzar diversidad: 50% targets cercanos, 50% lejanos
3. `min_ratio = 1.05` (evitar ratio≈1)
4. Targets solo de frames FUTUROS (estilo Shazam)
5. MIDI pseudo-TF con 6 armónicos (decaimiento 1/h)
6. Envolvente ADSR simplificada
7. Paralelización con 14 workers

**Resultados V2 (mejora dramática):**
| Métrica | V1 (antes) | V2 (después) |
|---------|------------|--------------|
| Cosine similarity | 0.13 | **0.96** |
| Token ratio (A/M) | 356x | **1.07x** |
| Audio mean log_ratio | 0.086 | **0.956** |
| MIDI mean log_ratio | 1.32 | **0.942** |

### Fase 5: Validación con 10 Pares

**Script**: `test_varios_pares_pre_red.py`

**Resultados Token Compatibility (10 piezas, 110 segmentos):**
| Métrica | Valor | Umbral GO | Estado |
|---------|-------|-----------|--------|
| Avg Cosine | 0.957 | > 0.9 | ✓ PASS |
| Avg Token Ratio | 1.16x | 0.5-2.0 | ✓ PASS |
| Retrieval Recall@1 | 4.5% | > 4.5% | ≈ LÍMITE |
| Self vs Cross Gap | 0.009 | > 0.05 | ✗ FAIL |

**Problema identificado**: Los histogramas 1D no capturan identidad temporal.

### Fase 6: Mejoras de Matching

**Script**: `test_varios_pares_pre_red_v2.py`

**Técnicas probadas:**
1. Hashes 2D/3D: `(dt_bin, log_ratio_bin, f_anchor_coarse)`
2. TF-IDF weighting
3. Histograma 2D (log_ratio × delta_t)

**Resultados por método:**
| Método | Recall@1 | Gap (same piece) |
|--------|----------|------------------|
| hist_1d | 4.5% | 0.009 |
| hist_2d | **8.2%** | 0.007 |
| hash_overlap | 4.5% | 0.017 |
| tfidf_cosine | 5.5% | 0.010 |

**Conclusión**: Mejora marginal pero insuficiente.

### Fase 7: Implementación Shazam Real

**Recomendación GPT5.2Think**: Usar offset-consensus voting en lugar de similaridad global.

**Scripts creados:**
- `test_shazam_oracle.py` - Test MIDI vs MIDI (verificar algoritmo)
- `test_shazam_crossmodal.py` - Test Audio vs MIDI (cross-modal)

**Configuración Shazam:**
```python
DT_BIN_SIZE = 2          # frames (~46ms)
LOG_RATIO_BIN_SIZE = 1/24  # ~50 cents
N_ANCHOR_BANDS = 8
OFFSET_BIN_SIZE = 4      # frames (~92ms)
```

**Características:**
- DB indexada por pieza completa (no segmentos)
- Tiempos absolutos en DB y queries
- IDF weighting
- Stoplist (hashes en >50% de piezas)
- Cap de matches por hash (100)

### Fase 8: Resultados Finales

#### Test ORACLE (MIDI vs MIDI)
```
Piece Accuracy (top-1): 90.9%
Recall@3:               94.5%
Recall@5:               95.5%
Offset Accuracy (<1s):  90.0%
Offset MAE:             0.14s

✓ ORACLE PASS: Voting implementation is correct!
```

#### Test CROSS-MODAL (Audio vs MIDI)
```
Piece Accuracy (top-1): 15.5%
Recall@3:               33.6%
Recall@5:               50.9%
Offset Accuracy (<1s):  2.7%
Offset MAE:             30.87s

Random baseline: top-1 = 10.0%
Improvement: 1.5x over random

✗ CROSS-MODAL NO-GO
```

---

## Análisis de Resultados

### ¿Por qué el Oracle funciona pero Cross-Modal no?

| Aspecto | Oracle (MIDI vs MIDI) | Cross-Modal (Audio vs MIDI) |
|---------|----------------------|----------------------------|
| Misma fuente | ✓ Sí | ✗ No |
| Hashes coinciden | ✓ Exactamente | ✗ Parcialmente |
| Piece Accuracy | 90.9% | 15.5% |
| Offset MAE | 0.14s | 30.87s |

**Interpretación**:
- El algoritmo Shazam funciona perfectamente cuando los hashes coinciden
- Los hashes de Audio y MIDI **no coinciden** para el mismo contenido musical
- La "compatibilidad de distribuciones" (cosine 0.95) no implica "coincidencia de hashes individuales"

### ¿Qué significa esto para el "Ratio Language"?

1. **Nivel macro (distribución)**: ✓ Compatible
   - Las estadísticas globales de ratios son similares
   - Cosine similarity > 0.95

2. **Nivel micro (tokens individuales)**: ✗ No compatible
   - Los hashes específicos no coinciden
   - Un intervalo de quinta en Audio no produce el mismo hash que en MIDI

### Posibles causas de la incompatibilidad

1. **Timing differences**: El peak picking en Audio vs pseudo-TF MIDI no detecta los mismos "eventos"
2. **Harmonic content**: El Audio tiene armónicos reales, el pseudo-TF tiene armónicos sintéticos
3. **Noise/dynamics**: El Audio tiene ruido y dinámica que el MIDI no tiene
4. **Quantization mismatch**: Los bins de ratio/delta_t no alinean eventos cross-modalmente

---

## Archivos Generados

### Scripts de Prueba
```
experiments/un_audio_un_midi/
├── test_single_pair.py               # V1 original (obsoleto)
├── test_single_pair_v2.py            # V2 corregido (secuencial)
├── test_single_pair_v2_parallel.py   # V2 paralelo (14 workers)
├── test_varios_pares_pre_red.py      # Validación 10 pares (hist 1D)
├── test_varios_pares_pre_red_v2.py   # Hashes 2D/3D + TF-IDF
├── test_varios_pares_shazam.py       # Shazam original (lento)
├── test_varios_pares_shazam_gpu.py   # Shazam con GPU
├── test_shazam_oracle.py             # Oracle MIDI vs MIDI
└── test_shazam_crossmodal.py         # Cross-modal final
```

### Resultados
```
experiments/un_audio_un_midi/Varios_pares/
├── results/                    # Pre-red V1
├── results_v2/                 # Pre-red V2 (hashes)
├── results_shazam_gpu/         # Shazam GPU
└── results_crossmodal/         # Resultados finales
    ├── crossmodal_results.json
    └── crossmodal_results.png
```

---

## Conclusiones

### Hipótesis Evaluadas

| Hipótesis | Estado | Evidencia |
|-----------|--------|-----------|
| H1: Distribuciones compatibles | ✓ VALIDADA | cosine > 0.95 |
| H2: Shazam voting funciona | ✓ VALIDADA | Oracle 90.9% |
| H3: Cross-modal identification | ✗ NO VALIDADA | 15.5% vs 90.9% |

### Lecciones Aprendidas

1. **Compatibilidad de distribuciones ≠ Compatibilidad de tokens**
   - Marginales iguales no garantiza información de instancia

2. **El extractor V2 resolvió el problema de colapso**
   - De ratio≈1 a distribución balanceada
   - Pero esto no fue suficiente para cross-modal

3. **El algoritmo Shazam es correcto**
   - Oracle demuestra 90.9% accuracy
   - El problema es la representación, no el algoritmo

4. **El "ratio language" tiene limitaciones fundamentales**
   - Captura estadística global pero no identidad cross-modal
   - Los mismos intervalos musicales no producen los mismos hashes en Audio vs MIDI

### Opciones Futuras

1. **Aceptar NO-GO**: Publicar resultados negativos (valor científico)
2. **Cambiar representación**: Abandonar ratios, probar spectrograms + contrastive
3. **Mejorar extractor**: Buscar alineación más fina de peak picking
4. **Cambiar hipótesis**: H3' = "Audio y MIDI requieren aprendizaje, no matching directo"

---

## Criterios GO/NO-GO Finales

| Criterio | Umbral | Resultado | Estado |
|----------|--------|-----------|--------|
| Token Compatibility | cosine > 0.9 | 0.957 | ✓ PASS |
| Oracle (MIDI vs MIDI) | Piece Acc > 80% | 90.9% | ✓ PASS |
| Cross-Modal Piece Acc | > 50% | 15.5% | ✗ FAIL |
| Cross-Modal Offset | MAE < 3s | 30.87s | ✗ FAIL |

**VEREDICTO FINAL**: ✗ NO-GO para cross-modal identification con ratio language

---

## Referencias

- Plan original: `Documents/ESCALON_1/Plan_implementacion.md`
- Recomendaciones GPT5.2Think: `Documents/ESCALON_1/Prueba_de_pocos_pares_GPT5.2Think.md`
- Modificaciones al plan: `Documents/ESCALON_1/escalon_1_plan_modificaciones.md`
- Dataset MAESTRO: https://magenta.tensorflow.org/datasets/maestro
