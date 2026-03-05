# Escalón 1: Nuevos Enfoques - RESULTADOS PRELIMINARES

**Fecha**: 2026-02-04
**Estado**: 🟡 **RESULTADOS PRELIMINARES PROMETEDORES** (pendiente validación)

---

## ⚠️ IMPORTANTE: Limitaciones del Experimento Actual

**Este es un experimento piloto con muestra muy pequeña:**
- **N = 10 pares** audio-MIDI
- **7-10 queries** generadas por ruta
- **Sin replicación** con otras muestras
- **Sin validación estadística** (intervalos de confianza, bootstrapping)

**Los resultados son prometedores pero NO demuestran H3.** Para validar la hipótesis necesitamos:
1. Auditoría del experimento (verificar que se hizo correctamente)
2. Replicación con otra muestra independiente
3. Validación con dataset completo o gran porción (100+ piezas)
4. Ejecución del pipeline completo del Escalón 1

---

## Resumen del Experimento Piloto

Tras el resultado NO-GO inicial (15.5% accuracy), implementamos dos nuevos enfoques basados en recomendaciones de GPT5.2Think:

| Enfoque | Piece Accuracy | Recall@5 | n_queries | Status |
|---------|---------------|----------|-----------|--------|
| **Route A: Event-Based** | 71.4% | 100% | 7 | Prometedor |
| **Route B: Improved TF** | 80.0% | 100% | 10 | Prometedor |
| *Anterior (extractor original)* | *15.5%* | *50.9%* | *55* | *NO-GO* |

**Hipótesis de trabajo**: El problema estaba en cómo se extraían y hasheaban los tokens.

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

### Resultados Preliminares (N=10)

- **Tokens/pieza**: ~1,800 (muy eficiente)
- **Hash diversity**: ~900 unique hashes
- **Piece Accuracy**: 71.4% (5/7 queries)
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

### Resultados Preliminares (N=10)

- **Tokens/pieza**: ~52,000
- **Hash diversity**: ~3,500 unique hashes
- **Piece Accuracy**: 80.0% (8/10 queries)
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
| n_queries | 7 | 10 |

### Eficiencia

| Métrica | Route A | Route B |
|---------|---------|---------|
| Tokens/pieza | **1,800** | 52,000 |
| Unique hashes | **900** | 3,500 |
| Ratio (B/A) | 1x | 29x |

**Route A es 29× más eficiente** en tokens.

---

## Análisis Crítico

### ¿Por qué podrían funcionar los nuevos enfoques?

1. **Onset anchoring**: Reduce hashes "genéricos" de frames sin eventos musicales
2. **Harmonic folding / Pitch class**: Hace los hashes octave-invariant, crucial para piano
3. **IDF agresivo**: Elimina hashes que aparecen en todas las piezas

### Riesgos y Limitaciones

| Riesgo | Descripción |
|--------|-------------|
| **Overfitting a muestra** | 10 pares pueden tener características especiales |
| **Sesgo de selección** | Las 10 piezas pueden ser "fáciles" |
| **Sin negativos duros** | No se probó NEG_SAME_COMPOSER |
| **Sin validación cruzada** | Sin split train/test |

### Estado de Hipótesis

| Hipótesis | Estado |
|-----------|--------|
| H3: Cross-modality | ⏳ **PENDIENTE VALIDACIÓN** |

**Los resultados preliminares son prometedores**, pero para validar H3 necesitamos:
1. Auditoría del experimento
2. Replicación con muestra independiente
3. Validación a escala

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

## Próximos Pasos REQUERIDOS

### Fase 1: Auditoría del Experimento
- [ ] Verificar correctitud de extractores
- [ ] Validar alineación audio-MIDI
- [ ] Revisar generación de queries

### Fase 2: Replicación
- [ ] Seleccionar 10-20 pares nuevos (diferentes piezas)
- [ ] Ejecutar mismo experimento
- [ ] Comparar resultados

### Fase 3: Validación a Escala
- [ ] Procesar 100+ piezas de MAESTRO
- [ ] Aplicar protocolo de evaluación completo (NEG_RANDOM, NEG_SAME_PIECE, NEG_SAME_COMPOSER)
- [ ] Calcular intervalos de confianza

### Fase 4: Pipeline Completo Escalón 1
- [ ] Gate 0: Setup harness con controles negativos
- [ ] Gate 1: Ingesta y alineación (ya hecho parcialmente)
- [ ] Gate 2: Baselines sin DL
- [ ] Gate 3: Modelo cross-modal (VICReg/Barlow)
- [ ] Gate 4: Ratio tokens (nuevos extractores)
- [ ] Gate 5: MoCo con negativos duros

---

## Referencias

- Diagnóstico GPT5.2Think: `Documents/01_FRENTES_ACTIVOS/ESCALON_1/Extractor_nuevos_enfoques_GPT5.2Think.md`
- Plan original: `Documents/01_FRENTES_ACTIVOS/ESCALON_1/Plan_implementacion.md`
