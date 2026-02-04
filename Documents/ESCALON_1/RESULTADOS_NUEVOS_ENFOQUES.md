# Escalón 1: Nuevos Enfoques - RESULTADOS GO

**Fecha**: 2026-02-04
**Estado**: ✓ **GO** - Cross-modal identification funciona con nuevos extractores

---

## Resumen Ejecutivo

Tras el resultado NO-GO inicial (15.5% accuracy), implementamos dos nuevos enfoques basados en recomendaciones de GPT5.2Think:

| Enfoque | Piece Accuracy | Recall@5 | Improvement | Status |
|---------|---------------|----------|-------------|--------|
| **Route A: Event-Based** | **71.4%** | 100% | 7.1x | ✓ GO |
| **Route B: Improved TF** | **80.0%** | 100% | 8.0x | ✓ GO |
| *Anterior (extractor original)* | *15.5%* | *50.9%* | *1.5x* | *NO-GO* |

**Conclusión**: El problema NO era el "ratio language" sino cómo se extraían y hasheaban los tokens.

---

## Diagnóstico Previo: COLISIÓN GENÉRICA

Antes de implementar las mejoras, ejecutamos un diagnóstico que reveló:

```
overlap_aligned:    66.23%
overlap_random:     65.13%
Gap (aligned-random): 1.10%  ← ¡Casi cero discriminabilidad!
```

**Problema identificado**: Los hashes coincidían mucho (66%) pero **igual para cualquier par**. Los top 10 hashes aparecían en 100% de las piezas.

---

## Route A: Event-Based Ratio Language

### Concepto

Convertir ambas modalidades a **eventos musicales** (onset + pitch) y construir ratio language sobre intervalos.

### Implementación

```
src/extractors/event_based_extractor.py
```

**Extractor de eventos desde Audio:**
- CQT con 84 bins (7 octavas)
- Onset detection via spectral flux
- Peak picking para estimar pitch
- Alineación temporal audio→MIDI

**Tokens extraídos:**
1. **T_chord** (tipo 1): Intervalos dentro de acordes (dt=0)
2. **T_seq** (tipo 2): Intervalos melódicos consecutivos
3. **T_pair** (tipo 3): Constelaciones Shazam en tiempo-pitch

**Hash structure (20 bits):**
```
(type, dt_bin, dp_bin, pc_anchor)
- type: 2 bits (1-3)
- dt_bin: 6 bits (0-63)
- dp_bin: 7 bits (-36 to +36 semitones)
- pc_anchor: 4 bits (pitch class 0-11)
```

### Resultados

- **Tokens/pieza**: ~1,800 (muy eficiente)
- **Hash diversity**: ~900 unique hashes
- **Piece Accuracy**: 71.4%
- **Recall@5**: 100%

---

## Route B: Improved TF-Constellations

### Concepto

Mejorar el extractor TF original con:
1. **Onset anchoring**: Solo usar frames cerca de onsets como anchors
2. **Harmonic folding**: Mapear frecuencias a pitch class (octave-invariant)
3. **IDF más agresivo**: Stoplist threshold de 30% (antes 50%)

### Implementación

```
src/extractors/improved_tf_extractor.py
```

**Mejoras clave:**

```python
# 1. Onset anchoring
def detect_onsets_spectral_flux(cqt_db):
    """Solo permite anchors cerca de onsets detectados."""
    ...

# 2. Harmonic folding
def fold_to_octave(freq):
    """Mapea frecuencia a pitch class."""
    return (9 + bin_idx) % 12

# 3. Hash con pitch class
h = (dt_bin << 10) | (lr_folded_bin << 4) | pc_anchor
```

### Resultados

- **Tokens/pieza**: ~52,000
- **Hash diversity**: ~3,500 unique hashes
- **Piece Accuracy**: 80.0%
- **Recall@5**: 100%

---

## Comparación de Enfoques

### Overlap Analysis (Pre-retrieval)

| Métrica | Route A | Route B | Anterior |
|---------|---------|---------|----------|
| overlap_aligned | 21.88% | 71.46% | 66.23% |
| overlap_random | 12.16% | 63.29% | 65.13% |
| **Gap** | **9.71%** | **8.17%** | **1.10%** |

**Route A tiene mejor ratio señal/ruido** aunque Route B tiene más overlap absoluto.

### Retrieval Performance

| Métrica | Route A | Route B |
|---------|---------|---------|
| Piece Accuracy | 71.4% | **80.0%** |
| Recall@3 | 100% | 100% |
| Recall@5 | 100% | 100% |
| Offset MAE | 0.00s | 0.00s |

**Route B gana por ~8.6 puntos porcentuales** en piece accuracy.

### Eficiencia

| Métrica | Route A | Route B |
|---------|---------|---------|
| Tokens/pieza | **1,800** | 52,000 |
| Unique hashes | **900** | 3,500 |
| Ratio (B/A) | 1x | 29x |

**Route A es 29× más eficiente** en tokens.

---

## Conclusiones Científicas

### ¿Por qué funcionan los nuevos enfoques?

1. **Onset anchoring**: Reduce hashes "genéricos" de frames sin eventos musicales
2. **Harmonic folding / Pitch class**: Hace los hashes octave-invariant, crucial para piano
3. **IDF agresivo**: Elimina hashes que aparecen en todas las piezas

### ¿Qué enfoque elegir?

| Criterio | Ganador |
|----------|---------|
| Accuracy | Route B (+8.6%) |
| Eficiencia | Route A (29× menos tokens) |
| Interpretabilidad | Route A (eventos musicales) |
| Facilidad de implementación | Route B (reutiliza extractor) |

**Recomendación**:
- Para **investigación**: Route A (más interpretable)
- Para **producción**: Route B (mejor accuracy)

### Hipótesis Revisada

| Hipótesis | Estado Anterior | Estado Actual |
|-----------|----------------|---------------|
| H3: Cross-modality | ❌ NO VALIDADA | ✓ **VALIDADA** |

El "ratio language" **SÍ funciona** para cross-modal Audio↔MIDI cuando:
1. Los anchors se condicionan a onsets
2. Los hashes son octave-invariant
3. Se aplica IDF agresivo

---

## Archivos Generados

### Código
```
src/extractors/
├── __init__.py
├── event_based_extractor.py    # Route A
└── improved_tf_extractor.py    # Route B
```

### Scripts de Evaluación
```
experiments/un_audio_un_midi/
├── diagnose_hash_collision.py  # Diagnóstico inicial
├── compare_routes.py           # Comparación overlap
└── test_retrieval_routes.py    # Test Shazam final
```

### Resultados
```
experiments/un_audio_un_midi/Varios_pares/
├── diagnosis/
│   ├── diagnosis_results.json
│   └── diagnosis_results.png
├── route_comparison/
│   ├── route_comparison.json
│   └── route_comparison.png
└── retrieval_routes/
    └── retrieval_results.json
```

---

## Próximos Pasos

1. **Validar en dataset completo MAESTRO** (1276 piezas vs 10 actuales)
2. **Optimizar Route B** para reducir tokens manteniendo accuracy
3. **Probar combinación A+B** (eventos + TF mejorado)
4. **Documentar y publicar** resultados positivos

---

## Referencias

- Diagnóstico GPT5.2Think: `Documents/ESCALON_1/Extractor_nuevos_enfoques_GPT5.2Think.md`
- Resultados anteriores: `Documents/ESCALON_1/RESULTADOS_ESCALON_1.md`
- Plan original: `Documents/ESCALON_1/Plan_implementacion.md`
