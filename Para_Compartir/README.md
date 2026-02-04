# Escalón 1: Cross-Modal Audio ↔ MIDI - Resultados y Documentación

**Fecha**: 2026-02-04
**Proyecto**: Phideus v5.0 - Harmonic Information Theory
**Experimento**: Escalón 1 - MAESTRO Dataset

---

## Resumen Ejecutivo

Este directorio contiene toda la documentación, scripts y resultados del experimento **Escalón 1**, que investigó la hipótesis de cross-modality entre Audio y MIDI usando el dataset MAESTRO.

### Resultado Principal

| Hipótesis | Estado | Evidencia |
|-----------|--------|-----------|
| H3: Cross-modality | 🔴 **LIMITADA** | 27% accuracy, 5.4x random |

**Diagnóstico**: El sistema detecta correspondencia cross-modal (5.4x sobre random), pero la accuracy es insuficiente para validar H3 de forma convincente. La causa raíz es la **resolución temporal del onset detector**.

---

## Contenido del Directorio

### Documents/

| Archivo | Descripción |
|---------|-------------|
| `INDICE_DOCUMENTACION.md` | Mapa completo de toda la documentación |
| `Proyecto_Estado_Actual.md` | Estado global del proyecto Phideus |
| `Plan_implementacion.md` | Plan original de 6 Gates |
| `PLAN_VALIDACION_H3.md` | Plan de 4 fases de validación |
| `AUDITORIA_FASE_A.md` | Bug crítico encontrado (t_anchor) |
| `INFORME_FASES_A_B.md` | Resultados de auditoría y replicación |
| `PLAN_ANALISIS_ERRORES.md` | Plan de análisis de errores |
| **`INFORME_ANALISIS_ERRORES.md`** | **Diagnóstico completo - DOCUMENTO PRINCIPAL** |
| `Extractor_nuevos_enfoques_GPT5.2Think.md` | Especificaciones de Route A y B |
| `RESULTADOS_NUEVOS_ENFOQUES.md` | Resultados preliminares N=10 |
| `RESULTADOS_ESCALON_1.md` | Resumen de resultados |

### Scripts/

| Script | Propósito |
|--------|-----------|
| `test_retrieval_routes.py` | Test principal de retrieval Shazam-style |
| `analyze_errors.py` | Análisis de distribución de errores |
| `analyze_overlap_deep.py` | Análisis de componentes de tokens |
| `ablation_chord_only.py` | Estudio de ablación por tipo de token |
| `diagnose_hash_collision.py` | Diagnóstico de colisiones de hash |
| `compare_routes.py` | Comparación de overlap entre rutas |

### Extractores/

| Archivo | Descripción |
|---------|-------------|
| `event_based_extractor.py` | **Route A**: Event-Based (27% accuracy) |
| `improved_tf_extractor.py` | **Route B**: Improved TF (21% accuracy) |

### Resultados/

| Directorio | Contenido |
|------------|-----------|
| `error_analysis/` | JSON con métricas de análisis de errores |
| `retrieval_improved/` | JSON con resultados de retrieval post-mejoras |

---

## Métricas Clave

### Evolución de Resultados

| Fase | Route A | Route B | vs Random |
|------|---------|---------|-----------|
| N=10 (con bug) | 71.4% | 80.0% | - |
| N=10 (corregido) | 42.5% | 32.9% | 4.2x / 3.3x |
| N=20 (replicación) | 26.6% | 21.4% | 5.3x / 4.3x |
| N=20 (post-mejoras) | **27.0%** | 21.4% | **5.4x** / 4.3x |

### Overlap Cross-Modal por Tipo de Token

| Tipo | Descripción | Overlap |
|------|-------------|---------|
| Chord | Notas simultáneas | **72%** ✓ |
| Sequential | Notas consecutivas | 8% ✗ |
| Constellation | Pares lejanos | 3% ✗ |

### Estudio de Ablación

| Configuración | Hashes | Accuracy |
|---------------|--------|----------|
| All tokens | 7,049 | 27.0% |
| Chord + Sequential | 3,754 | 27.1% |
| Chord ONLY | 398 | 13.6% |

---

## Diagnóstico Principal

```
┌─────────────────────────────────────────────────────────────────┐
│ CAUSA RAÍZ: El onset detector del audio tiene resolución       │
│ temporal insuficiente para generar tokens compatibles con MIDI │
│                                                                 │
│ - MIDI genera DT = 1-3 frames (timing perfecto)                │
│ - Audio genera DT = 5-20 frames (onset detection limitado)     │
│ - Los hashes resultantes son diferentes                        │
│                                                                 │
│ LÍMITE ACTUAL: ~27% accuracy con enfoque actual                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Cómo Reproducir

### Requisitos

```bash
pip install librosa numpy pretty_midi mido
```

### Ejecutar Test de Retrieval

```bash
cd /mnt/m2-1TB/Phideus
source venv/bin/activate

python experiments/un_audio_un_midi/test_retrieval_routes.py \
    --input-dir experiments/un_audio_un_midi/muestra_replicacion
```

### Ejecutar Análisis de Errores

```bash
python experiments/un_audio_un_midi/analyze_errors.py --route A
python experiments/un_audio_un_midi/ablation_chord_only.py
```

---

## Conclusiones

1. **H3 parcialmente soportada**: 5.4x random indica señal real, pero 27% accuracy es insuficiente
2. **Cuello de botella identificado**: Resolución temporal del onset detector
3. **Tokens chord funcionan**: 72% overlap cross-modal
4. **Mejoras incrementales saturadas**: +8pp overlap → +0.4pp accuracy

### Opciones para Continuar

| Opción | Descripción | Esfuerzo |
|--------|-------------|----------|
| Escalar N=100 | Ver tendencia | Bajo |
| Mejorar onset detector | Atacar causa raíz | Alto |
| LSH / soft matching | Tolerancia en hash | Muy alto |
| Documentar y cerrar | Resultado negativo válido | - |

---

## Contacto

Proyecto Phideus - Harmonic Information Theory Research
