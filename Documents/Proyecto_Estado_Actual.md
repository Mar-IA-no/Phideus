# Proyecto Estado Actual - Phideus v5.0

**Actualizado**: 2026-02-04
**Estado**: 🟡 **Escalón 1 MAESTRO - EN PROGRESO** (resultados preliminares prometedores)

---

## Resumen Ejecutivo

### Estado de Hipótesis

| Hipótesis | Estado | Evidencia |
|-----------|--------|-----------|
| H1: Estructura de ratios | **VALIDADA** | Distribuciones no aleatorias |
| H2: Aprendibilidad | **VALIDADA** | VAE/HRM val_loss < 0.5 |
| H3: Cross-modality | 🟡 **PENDIENTE** | Resultados preliminares prometedores (N=10) |

### Situación Actual (2026-02-04)

El experimento piloto del Escalón 1 (N=10 pares) mostró resultados prometedores:

| Enfoque | Piece Accuracy | n_queries | Status |
|---------|---------------|-----------|--------|
| Route A (Event-Based) | 71.4% | 7 | Prometedor |
| Route B (Improved TF) | 80.0% | 10 | Prometedor |
| *Extractor original* | *15.5%* | *55* | *NO-GO* |

**⚠️ IMPORTANTE**: N=10 pares es insuficiente para validar H3. Se requiere validación rigurosa.

---

## 🟡 ESCALÓN 1: MAESTRO (Audio ↔ MIDI) - EN PROGRESO

### Estado: Experimento Piloto Completado, Validación Pendiente

### Resultados Preliminares (N=10 pares)

| Extractor | Piece Accuracy | n_queries | Gap |
|-----------|---------------|-----------|-----|
| V1 (original) | 15.5% | 55 | 1.10% |
| Route A (Event-Based) | 71.4% | 7 | 9.71% |
| Route B (Improved TF) | 80.0% | 10 | 8.17% |

### Diagnóstico del Problema Original

El diagnóstico reveló **COLISIÓN GENÉRICA**:
- overlap_aligned: 66.23%
- overlap_random: 65.13%
- Gap: **1.10%** (casi cero discriminabilidad)

Los hashes coincidían mucho pero **igual para cualquier par**.

### Nuevos Extractores Implementados

**Route A: Event-Based Ratio Language**
- Convertir Audio→eventos (onset+pitch) via CQT + onset detection
- Convertir MIDI→eventos (directo de notas)
- Ratio language sobre intervalos musicales

**Route B: Improved TF-Constellations**
1. **Onset anchoring**: Solo anchors cerca de onsets detectados
2. **Harmonic folding**: Frecuencias a pitch class (octave-invariant)
3. **IDF agresivo**: Stoplist threshold 30% (antes 50%)

### Archivos Creados

```
src/extractors/
├── __init__.py
├── event_based_extractor.py    # Route A
└── improved_tf_extractor.py    # Route B

experiments/un_audio_un_midi/
├── diagnose_hash_collision.py  # Diagnóstico
├── compare_routes.py           # Comparación overlap
└── test_retrieval_routes.py    # Test Shazam (N=10)
```

### Próximos Pasos REQUERIDOS

Ver: `Documents/ESCALON_1/PLAN_VALIDACION_H3.md`

1. **Fase A: Auditoría** - Verificar correctitud del experimento
2. **Fase B: Replicación** - Probar con 10-20 pares nuevos
3. **Fase C: Escala** - Validar con 100+ piezas
4. **Fase D: Pipeline Completo** - Ejecutar Gates 0-5

---

## Estado de Hipótesis Actualizado

### H1: Estructura de Ratios ✓
Las señales (audio, vibración, MIDI) contienen distribuciones de ratios estructuradas y no aleatorias.

### H2: Aprendibilidad ✓
Redes neuronales pueden aprender estas distribuciones (VAE val_loss < 0.5).

### H3: Cross-Modality 🟡 PENDIENTE

**Estado**: Resultados preliminares prometedores, pendiente validación.

Experimento piloto (N=10):
- Route A: 71.4% accuracy (7 queries)
- Route B: 80.0% accuracy (10 queries)

**Para validar H3 se requiere**:
1. Auditoría del experimento piloto
2. Replicación con muestra independiente
3. Validación a escala (100+ piezas)
4. Pipeline completo del Escalón 1

---

## 🔴 REVISIONISMO UOEMD - COMPLETADO (NO-GO)

*(Sin cambios - el dataset UOEMD sigue siendo NO-GO)*

### Fases Completadas

| Fase | Descripción | Resultado |
|------|-------------|-----------|
| 0 | Tests sintéticos | ✓ Funcionan |
| 1 | Extractor v2.2 | ✓ Gap pre-red 0.691 |
| 2 | Re-entrenamiento | ✗ Gap post-red 0.007 |
| 3A | Constellation tokens | ✗ Top-1 = 0.78% (random) |

### Conclusión UOEMD

El dataset UOEMD (128 muestras, motor diésel) no demostró cross-modality. Posibles razones:
- Dataset muy pequeño (128 vs 1276 en MAESTRO)
- Audio de motor vs vibración de motor puede no compartir estructura armónica
- Los nuevos enfoques deberían probarse en UOEMD para confirmar

---

## Observaciones Preliminares (Pendiente Validación)

Basadas en experimento piloto N=10:

1. **El extractor parece importar más que la arquitectura**
   - Mismo algoritmo Shazam: 15.5% → 80% (pendiente confirmar)

2. **Onset anchoring parece crítico**
   - Frames sin eventos generan hashes "genéricos"

3. **Harmonic folding (octave-invariance)**
   - Potencialmente esencial para música tonal

4. **IDF agresivo**
   - Stoplist con threshold bajo (30%) parece reducir ruido

---

## Próximos Pasos

### Inmediato: Validación H3

1. **Auditoría del experimento** (1-2 días)
2. **Replicación con muestra nueva** (1 día)
3. **Validación a escala 100+ piezas** (2-3 días)
4. **Pipeline completo Escalón 1** (5-7 días)

### Futuro (si H3 se valida)

1. Probar mejoras en UOEMD
2. Escalón 2: Speech ↔ EGG
3. Escalón 3: ECG ↔ PPG

---

## Referencias

- **Plan de validación**: `Documents/ESCALON_1/PLAN_VALIDACION_H3.md`
- **Resultados preliminares**: `Documents/ESCALON_1/RESULTADOS_NUEVOS_ENFOQUES.md`
- **Recomendaciones GPT**: `Documents/ESCALON_1/Extractor_nuevos_enfoques_GPT5.2Think.md`
- **Plan original**: `Documents/ESCALON_1/Plan_implementacion.md`
- **Dataset MAESTRO**: `data/maestro_v3/maestro-v3.0.0/` (121GB)
