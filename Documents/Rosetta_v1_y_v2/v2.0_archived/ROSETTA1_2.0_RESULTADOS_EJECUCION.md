# Rosetta1 2.0 - Resultados de Ejecución

**Fecha de ejecución**: 2026-01-28
**Estado**: COMPLETADO - Resultado NO-GO

---

## Resumen Ejecutivo

La ejecución completa del pipeline Rosetta1 2.0 reveló que **el modelo NO demuestra cross-modality real**. Los controles negativos implementados funcionaron correctamente y expusieron que el baseline original (cos_sim = 0.766) era un resultado espurio.

### Veredicto: NO-GO

| Criterio | Umbral | Resultado | Estado |
|----------|--------|-----------|--------|
| var(z_private) | > 0.1 | ~10⁻⁹ | ❌ FAIL |
| z_private diff | > 0.5 | **1.21** | ✅ PASS |
| Cross-recon Pearson | > 0.75 | 0.62 | ❌ FAIL |
| Δcorr (aligned - shuffled) | ≥ 0.15 | **0.002** | ❌ CRITICAL FAIL |
| Retrieval Top-1 | > 15% | 0.78% | ❌ FAIL |
| Shuffled retrieval ≈ random | ~0.8% | 0.78% | ✅ PASS |
| Silhouette Score | > 0.25 | -0.07 | ❌ FAIL |

**5 de 7 criterios fallaron**, incluyendo el más crítico: la diferenciación entre pares alineados y shuffled.

---

## Fase 1: Baseline Congelado

### Comando Ejecutado
```bash
python experiments/freeze_baseline.py \
    --checkpoint data/training_outputs/roseta_full/best_model.pt \
    --data data/datasets/roseta_full.npz \
    --output artifacts/baseline
```

### Métricas del Baseline
```json
{
  "cosine_similarity": 0.7650,
  "retrieval_accuracy": 0.0014,
  "z_private_audio_var": 4.85e-09,
  "z_private_vib_var": 1.30e-08,
  "z_private_diff": 0.0061,
  "n_samples": 128
}
```

### Diagnóstico Baseline
- ✅ Cosine similarity alto (0.765) - parecía prometedor
- ❌ z_private variance ~0 - **COLAPSADO**
- ❌ z_private diff muy bajo (0.006) - modalidades no diferenciadas
- ❌ Retrieval casi aleatorio (0.14%)

**Conclusión**: El baseline confirma el diagnóstico GPT5.2Pro - z_private no funciona.

---

## Fase 2: Re-entrenamiento con Fix z_private

### Comando Ejecutado
```bash
python experiments/run_roseta_experiment.py \
    --phase full \
    --data data/datasets/roseta_full.npz \
    --output data/training_outputs/roseta_v2 \
    --beta-kl-private 0.01 \
    --dropout-shared 0.5 \
    --lambda-diff 0.1 \
    --epochs 100 \
    --batch-size 8
```

### Parámetros del Fix
| Parámetro | Valor | Propósito |
|-----------|-------|-----------|
| beta-kl-private | 0.01 | Reducir presión KL en z_private |
| dropout-shared | 0.5 | Forzar uso de z_private en decoder |
| lambda-diff | 0.1 | Penalizar si z_priv_audio ≈ z_priv_vib |

### Resultados del Entrenamiento

#### Evolución de Métricas (epochs 1 → 100)
| Métrica | Epoch 1 | Epoch 100 | Cambio |
|---------|---------|-----------|--------|
| Val Loss | 5.14 | 4.85 | -5.6% |
| cos_sim | 0.011 | 0.627 | +5600% |
| z_priv diff | 1.08 | 1.21 | +12% |
| z_priv var | ~0 | ~0 | Sin cambio |

#### Cross-Retrieval por Condición
| Condición | Pearson Correlation |
|-----------|---------------------|
| HH (Healthy) | **0.876** |
| FB (Fault) | 0.714 |
| RU (Fault) | 0.526 |

### Problema Identificado

A pesar del fix:
- ✅ z_private diff mejoró (0.006 → 1.21)
- ❌ z_private variance sigue en ~0
- ⚠️ Cosine similarity bajó (0.765 → 0.627)

El dropout y diff loss lograron separar las representaciones, pero no lograron que z_private capture información útil.

---

## Fase 3: Evaluación con Controles Negativos

### 3.1 Cross-Reconstruction

#### Comando
```bash
python experiments/evaluate_cross_reconstruction.py \
    --model data/training_outputs/roseta_v2/best_model.pt \
    --data data/datasets/roseta_full.npz \
    --run-all-controls
```

#### Resultados Críticos

| Condición | Audio→Vib | Vib→Audio | Retrieval |
|-----------|-----------|-----------|-----------|
| **aligned** | 0.6154 | 0.5378 | 0.30% |
| **shuffled** | 0.6182 | 0.5397 | 0.30% |
| random_z | 0.6152 | 0.5527 | 0.10% |
| shuffled+random | 0.6154 | 0.5472 | 0.00% |

#### Análisis

**HALLAZGO CRÍTICO**: `aligned ≈ shuffled`

- Δcorr (Audio→Vib) = 0.6154 - 0.6182 = **-0.0028**
- Δcorr (Vib→Audio) = 0.5378 - 0.5397 = **-0.0019**

Esto significa que el modelo genera reconstrucciones de **calidad idéntica** sin importar si los pares audio-vibración son correctos o aleatorios.

**Implicación**: El modelo NO aprende la correspondencia cross-modal. Está generando un "histograma promedio" que funciona igual para cualquier input.

### 3.2 Retrieval

#### Comando
```bash
python experiments/evaluate_retrieval.py \
    --model data/training_outputs/roseta_v2/best_model.pt \
    --data data/datasets/roseta_full.npz \
    --output data/evaluations/retrieval_v2
```

#### Resultados

| Modo | Top-1 | Top-5 | MRR |
|------|-------|-------|-----|
| Global (A→V) | 0.78% | 3.91% | 0.042 |
| Shuffled | 0.78% | - | - |
| Random baseline | 1.56% | - | - |
| Teórico random | 0.78% | - | - |

**Diagnóstico**: Retrieval Top-1 = Random teórico. El embedding z_shared NO contiene información útil para matching.

### 3.3 Separación de Regímenes

#### Comando
```bash
python experiments/evaluate_regime_separation.py \
    --model data/training_outputs/roseta_v2/best_model.pt \
    --data data/datasets/roseta_full.npz \
    --output data/evaluations/regime_separation_v2
```

#### Resultados

| Modalidad | Tipo | Silhouette | Linear Probe |
|-----------|------|------------|--------------|
| Audio | Binary (H vs F) | -0.069 | AUC: 0.589 |
| Vibration | Binary | 0.246 | AUC: - |
| Audio | Multiclass (8) | -0.116 | Acc: 11.7% |
| Vibration | Multiclass | - | - |

**Diagnóstico**:
- Silhouette negativo = clusters superpuestos (no hay separación)
- Linear Probe AUC 0.59 = apenas mejor que random (0.50)
- Multiclass 11.7% ≈ random (12.5% para 8 clases)

**Conclusión**: Las condiciones de operación (healthy vs faults) NO se separan en el espacio latente z_shared.

---

## Diagnóstico Consolidado

### ¿Por qué falló?

1. **El modelo aprende un "atajo" (shortcut)**
   - En lugar de aprender la correspondencia real audio↔vibración
   - Aprende a generar el histograma promedio del dataset
   - Este histograma promedio tiene correlación ~0.6 con cualquier muestra

2. **z_private sigue colapsado**
   - A pesar del fix (beta=0.01, dropout, diff loss)
   - La varianza sigue siendo ~10⁻⁹
   - El decoder ignora z_private completamente

3. **InfoNCE no es suficiente**
   - La loss contrastiva alinea los espacios
   - Pero no garantiza que la información sea útil para reconstrucción
   - Los embeddings son similares pero no informativos

4. **Dataset posiblemente pequeño**
   - Solo 128 archivos totales
   - Con split 70/15/15 → ~11 archivos para training
   - Insuficiente para aprender correspondencias complejas

### Evidencia del Shortcut

```
Cross-recon con pares CORRECTOS:    0.615
Cross-recon con pares ALEATORIOS:   0.618
                                    -----
Diferencia:                         0.003 (insignificante)
```

Si el modelo aprendiera la correspondencia real, los pares shuffled deberían tener correlación mucho menor.

---

## Artefactos Generados

### Directorios
```
artifacts/baseline/
├── checkpoint.pt          # 38 MB
├── latents.npz           # 46 KB
├── metrics.json          # Métricas baseline
└── README.md

data/training_outputs/roseta_v2/
├── best_model.pt         # 38 MB
├── final_model.pt        # 38 MB
├── results.json          # Training history
└── roseta_experiment_report.md

data/evaluations/
├── retrieval_v2/
│   └── REPORT_RETRIEVAL.md
└── regime_separation_v2/
    ├── REPORT_REGIME_SEPARATION.md
    └── regime_separation_pca.png
```

---

## Conclusión

### Estado de las Hipótesis

| Hipótesis | Estado Pre-2.0 | Estado Post-2.0 |
|-----------|----------------|-----------------|
| H1: Estructura de ratios | ✅ Validada | ✅ Sin cambio |
| H2: Aprendibilidad | ✅ Validada | ✅ Sin cambio |
| H3: Cross-modality | ⚠️ Pendiente | ❌ **NO VALIDADA** |

### Veredicto Final

**NO-GO**: Rosetta1 2.0 demuestra que el resultado original (cos_sim = 0.766) era un falso positivo. Los controles negativos revelan que:

1. El modelo no aprende correspondencia cross-modal real
2. Los embeddings no son útiles para retrieval
3. Las condiciones no se separan en el espacio latente

### Valor del Experimento

A pesar del resultado negativo, Rosetta1 2.0 fue exitoso metodológicamente:

- ✅ Los controles negativos funcionaron y detectaron el problema
- ✅ La metodología es sólida y reproducible
- ✅ Ahora sabemos que el approach actual no funciona

Esto permite evitar publicar claims falsos y redirigir esfuerzos hacia soluciones reales.

---

## Próximos Pasos Sugeridos

1. **Investigar uso completo del dataset**
   - ¿Estamos usando todos los datos disponibles?
   - ¿El split es demasiado agresivo?

2. **Revisar arquitectura**
   - Considerar modelos más simples (sin VAE)
   - Probar arquitecturas contrastivas puras (sin reconstrucción)

3. **Aumentar datos**
   - Buscar más datasets de audio+vibración pareados
   - Data augmentation

4. **Redefinir el claim**
   - Si cross-modality no se puede demostrar, ¿qué SÍ se puede demostrar?
   - Enfocarse en lo que funciona (H1, H2)

---

*Documentado por Claude Code - 2026-01-28*
