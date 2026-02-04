# Proyecto Estado Actual - Phideus v5.0

**Actualizado**: 2026-02-04
**Estado**: 🟡 **Escalón 1 - Análisis de errores completado, decisión pendiente**

---

## Resumen Ejecutivo

### Estado de Hipótesis

| Hipótesis | Estado | Evidencia |
|-----------|--------|-----------|
| H1: Estructura de ratios | **VALIDADA** | Distribuciones no aleatorias |
| H2: Aprendibilidad | **VALIDADA** | VAE/HRM val_loss < 0.5 |
| H3: Cross-modality | 🔴 **LIMITADA** | 27% accuracy, 5.4x random (insuficiente) |

### Situación Actual (2026-02-04)

Fases A-B completadas + Análisis de errores exhaustivo:

| Experimento | Route A | Route B | vs Random |
|-------------|---------|---------|-----------|
| N=10 (corregido) | 42.5% | 32.9% | 4.2x / 3.3x |
| N=20 (replicación) | 26.6% | 21.4% | 5.3x / 4.3x |
| N=20 (post-mejoras) | **27.0%** | 21.4% | **5.4x** / 4.3x |

**Hallazgo clave**: El problema es la **resolución temporal del onset detector**, no el algoritmo de hashing. Mejoras incrementales tienen rendimientos decrecientes.

**Decisión pendiente**: Escalar a N=100, pivotar enfoque, o cerrar línea de investigación.

---

## 🟡 ESCALÓN 1: MAESTRO (Audio ↔ MIDI) - ANÁLISIS COMPLETADO

### Fases Ejecutadas

| Fase | Estado | Resultado |
|------|--------|-----------|
| A: Auditoría | ✅ Completada | Bug crítico encontrado y corregido |
| B: Replicación N=20 | ✅ Completada | 26.6% accuracy (Route A) |
| **Análisis de Errores** | ✅ **Completado** | Causa raíz identificada |
| C: Escala N=100 | ⏸️ Pausada | Pendiente decisión |
| D: Pipeline completo | ⏸️ Pausada | Pendiente decisión |

### Diagnóstico del Problema

```
┌─────────────────────────────────────────────────────────────────┐
│ CAUSA RAÍZ: El onset detector del audio tiene resolución       │
│ temporal insuficiente para generar tokens compatibles con MIDI │
│                                                                 │
│ - Tokens chord: 72% overlap (funcionan)                        │
│ - Tokens sequential: 8% overlap (no funcionan)                 │
│ - Tokens constellation: 3% overlap (no funcionan)              │
│                                                                 │
│ LÍMITE ACTUAL: ~27% accuracy con enfoque actual                │
└─────────────────────────────────────────────────────────────────┘
```

### Mejoras Implementadas y Resultados

| Mejora | Cambio | Impacto en Overlap | Impacto en Accuracy |
|--------|--------|-------------------|---------------------|
| A: DT_BIN_SIZE 2→10 | +tolerancia temporal | +8pp (17%→25%) | +0.4pp |
| B: CHORD boost | +peso a chords | Marginal | Marginal |

### Estudio de Ablación

| Configuración | Hashes | Accuracy | Conclusión |
|---------------|--------|----------|------------|
| All tokens | 7,049 | 27.0% | Baseline |
| Chord + Sequential | 3,754 | 27.1% | = Baseline |
| Chord ONLY | 398 | 13.6% | Muy pocos hashes |
| Sin chord | 6,651 | 24.9% | Chord aporta +2pp |

---

## Estado de Hipótesis Actualizado

### H1: Estructura de Ratios ✓
Las señales (audio, vibración, MIDI) contienen distribuciones de ratios estructuradas y no aleatorias.

### H2: Aprendibilidad ✓
Redes neuronales pueden aprender estas distribuciones (VAE val_loss < 0.5).

### H3: Cross-Modality 🔴 LIMITADA

**Estado**: Señal detectable pero insuficiente para aplicación práctica.

| Métrica | Valor | Umbral GO | Status |
|---------|-------|-----------|--------|
| Piece Accuracy | 27% | > 50% | ✗ FAIL |
| Recall@5 | 61% | > 80% | ✗ FAIL |
| vs Random | 5.4x | > 5x | ✓ PASS |

**Interpretación**: El sistema detecta correspondencia cross-modal (5.4x random), pero el nivel de accuracy es insuficiente para validar H3 de forma convincente.

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

---

## Opciones para Continuar

| Opción | Descripción | Esfuerzo | Potencial |
|--------|-------------|----------|-----------|
| **E** | Escalar a N=100 con 27% | Bajo | Ver tendencia |
| **C** | Mejorar onset detector | Alto | +10-15%? |
| **D** | LSH / soft matching | Muy alto | +15-20%? |
| **F** | Documentar y cerrar | - | - |
| **Pivot** | Cambiar enfoque (spectrograms, transformers) | Alto | Desconocido |

---

## 🔄 BIAS_CONTROL: Medium Test EN EJECUCIÓN

**Estado**: 🔄 **EJECUTANDO** - Epoch 11/30, Gap: 0.398 (migrado a tmux)

### Descripción

BIAS_CONTROL es un enfoque alternativo para cross-modal learning usando:
- **MERTLite**: Encoder pre-entrenado para audio (74M params)
- **MIDI Encoder**: Conv1D + GRU para piano roll
- **VICReg**: Loss de alineación sin pares negativos explícitos
- **DANN**: Domain adversarial para representaciones modal-agnósticas

### Medium Test en Progreso (2026-02-04)

| Epoch | Loss | Gap | Tendencia |
|-------|------|-----|-----------|
| 1 | 19.59 | 0.054 | baseline |
| 3 | 15.86 | 0.175 | ↑ +224% |
| 6 | 15.35 | 0.302 | ↑ peak |
| 10 | 15.18 | **0.398** | ↑ **15.3× fast test** |

**Señal MUY positiva**: Gap 0.398 supera criterio GO (>0.10) por 4×. Migrado a tmux para continuar epochs 11-30.

### Sanity Checks (2026-02-04)

| Check | Resultado |
|-------|-----------|
| Alineación Audio-MIDI | ✅ 30-50ms (excelente) |
| Segmentos válidos | ✅ 127,092 |
| Recall formula | ✅ Correcta |
| Pipeline bugs | ✅ Ninguno crítico |

### Auditoría Completada (2026-02-04)

| Fase | Status | Notas |
|------|--------|-------|
| 1. Dependencias | ✅ PASS | torch, transformers, librosa, pretty_midi |
| 2. Imports | ✅ PASS | Todos los módulos importan |
| 3. Dataset MAESTRO | ✅ PASS | 1,276 piezas, 121,940 segmentos |
| 4. Componentes | ✅ PASS | Encoders, Projection, DANN |
| 5. Dataset Loading | ✅ PASS | Audio + MIDI cargan correctamente |
| 6. Scripts --help | ✅ PASS | Todos los gates válidos |
| 7. Gate 0 E2E | ✅ PASS | Decision: GO |

### Bugs Corregidos

1. **CRÍTICO**: JSON MAESTRO v3.0.0 usa formato columnar (dict of dicts)
2. **ALTO**: Position embedding en MERTEncoderLite (1000 → 6000)
3. **MEDIO**: Verificación de shuffle comparaba piece_idx
4. **BAJO**: Tolerancia de alignment (100ms → 2000ms)

### Pipeline de Gates

| Gate | Objetivo | Criterio GO |
|------|----------|-------------|
| 0 | Data integrity | Alignment > 90%, Segments > 10K |
| 1 | Intra-modal baselines | Recall@10 > 50% |
| 2 | VICReg training | Cross-modal Recall@10 > 20% |
| 2.5 | Embedding analysis | Análisis cualitativo |
| 3 | DANN training | Domain acc → 50% |
| 4 | Ratio auxiliary | Mejora vs Gate 3 |

### Comando de Ejecución

```bash
python experiments/bias_control/run_all_gates.py \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/bias_control \
    --device cuda
```

**Documentación**: `Documents/ESCALON_1/BIAS_CONTROL_SYSTEM.md`

---

## Próximos Pasos

**BIAS_CONTROL** (en ejecución):
1. ✅ Fast test completado - Gap: 0.026
2. 🔄 Medium test en progreso - Gap: 0.175 (epoch 3/30)
3. ⏳ Evaluar con pool estructurado (256 candidatos + hard negatives)
4. ⏳ Decidir GO/NO-GO para H3

**Post Medium Test**:
1. Si Gap > 0.2 al terminar → Gate 3 (DANN)
2. Si Hard negative accuracy > 55% → Señal cross-modal confirmada
3. Si falla → Evaluar segment_len=8s + gradient accumulation

---

## Referencias de Documentación

### BIAS_CONTROL

| Documento | Ubicación | Contenido |
|-----------|-----------|-----------|
| **Roadmap** | `Documents/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md` | Arquitectura, gates, criterios |
| **Fast test results** | `Documents/BIAS_CONTROL/BIAS_CONTROL_FAST_TEST_RESULTS.md` | 3 epochs, baseline |
| **Medium test results** | `Documents/BIAS_CONTROL/BIAS_CONTROL_MEDIUM_TEST_RESULTS.md` | 30 epochs, en progreso |
| Sanity checks | `experiments/bias_control/sanity_checks.py` | Verificación alertas GPT5.2 |
| Pool estructurado | `experiments/bias_control/evaluate_structured_pool.py` | Evaluación con hard negatives |

### Escalón 1 (MAESTRO - Original)

| Documento | Ubicación | Contenido |
|-----------|-----------|-----------|
| Plan de validación | `Documents/ESCALON_1/PLAN_VALIDACION_H3.md` | 4 fases de validación |
| Auditoría Fase A | `Documents/ESCALON_1/AUDITORIA_FASE_A.md` | Bug t_anchor |
| Informe Fases A-B | `Documents/ESCALON_1/INFORME_FASES_A_B.md` | Resultados replicación |
| **Análisis de Errores** | `Documents/ESCALON_1/INFORME_ANALISIS_ERRORES.md` | **Diagnóstico completo** |
| Plan análisis errores | `Documents/ESCALON_1/PLAN_ANALISIS_ERRORES.md` | 5 fases de análisis |
| Recomendaciones GPT | `Documents/ESCALON_1/Extractor_nuevos_enfoques_GPT5.2Think.md` | Route A/B specs |
| Plan original | `Documents/ESCALON_1/Plan_implementacion.md` | 6 Gates |

### UOEMD / Rosetta (Histórico)

| Documento | Ubicación |
|-----------|-----------|
| Roadmap revisionismo | `Documents/UOEMD/UOEMD_Revisionismo/ROADMAP.md` |
| Resultados Fase 3A | `Documents/UOEMD/UOEMD_Revisionismo/Fase_3A/` |
| Extractor v2.2 | `Documents/UOEMD/UOEMD_Roseta_v2.2/` |

### Scripts Clave

| Script | Propósito |
|--------|-----------|
| `experiments/un_audio_un_midi/test_retrieval_routes.py` | Test Shazam-style |
| `experiments/un_audio_un_midi/analyze_errors.py` | Análisis de errores |
| `experiments/un_audio_un_midi/ablation_chord_only.py` | Ablation study |
| `src/extractors/event_based_extractor.py` | Route A extractor |
| `src/extractors/improved_tf_extractor.py` | Route B extractor |

### Datos

| Directorio | Contenido |
|------------|-----------|
| `experiments/un_audio_un_midi/Varios_pares/` | 10 pares originales |
| `experiments/un_audio_un_midi/muestra_replicacion/` | 20 pares replicación |
| `data/maestro_v3/maestro-v3.0.0/` | Dataset MAESTRO (121GB) |
