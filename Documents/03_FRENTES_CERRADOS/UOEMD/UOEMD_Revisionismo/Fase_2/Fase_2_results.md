# Fase 2: Resultados del Re-entrenamiento con Extractor v2.2

**Fecha de ejecución**: 2026-01-30
**Decisión final**: **NO-GO** - H3 no validada
**Branch**: `feature/extractor-v22`

---

## Resumen Ejecutivo

La Fase 2 probó si histogramas más discriminativos (gap pre-red = 0.691, 172× mejor que v1) habilitarían al modelo RosetaVAE para aprender correspondencia real audio ↔ vibración.

**Resultado**: El modelo sigue produciendo embeddings similares para pares alineados y shuffled. La mejora del extractor (172×) se redujo a solo 3.5× después del modelo.

| Métrica | Fase 1 (pre-red) | Fase 2 (post-red) | Ratio |
|---------|------------------|-------------------|-------|
| Gap aligned-shuffled | 0.691 | 0.007 | **1%** |

**Diagnóstico**: El VAE colapsa la información discriminativa del histograma.

---

## 1. Configuración Experimental

### 1.1 Extractor v2.2 (config_002 - óptima de Fase 1)

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `top_k_peaks` | 8 | Máximo de picos por frame |
| `min_prominence` | 0.1 | Prominencia mínima normalizada |
| `temporal_stability` | 0.7 | 70% de frames en ventana |
| `temporal_window` | 10 frames | ~0.5s dependiendo de hop |
| `use_warped_bins` | False | Bins uniformes |
| `n_fft` | 4096 | Resolución frecuencial ~10 Hz |
| `hop_length` | 1024 | 75% overlap |

**Comando de generación**:
```bash
python src/analizador/analizador_roseta.py \
    --input-dir data/datasets/UOEMD/raw/2_CSV_Data_Files \
    --output data/datasets/roseta_v22_full.npz \
    --top-k-peaks 8 \
    --min-prominence 0.1 \
    --temporal-stability 0.7 \
    --workers 12
```

### 1.2 Dataset Generado

| Propiedad | Valor |
|-----------|-------|
| Archivo | `data/datasets/roseta_v22_full.npz` |
| Tamaño | 9.4 MB |
| Archivos procesados | 128 |
| Frames totales | 52,096 |
| Condiciones | HH, RU, RM, FB, SW, VU, BR, KA (16 c/u) |
| Healthy | 16 (12.5%) |
| Fallas | 112 (87.5%) |
| Shape por archivo | [407, 256, 3] (audio y vibración) |

### 1.3 Modelo RosetaVAE

| Parámetro | Valor | Propósito |
|-----------|-------|-----------|
| `epochs` | 100 | Entrenamiento completo |
| `batch_size` | 8 | Balance memoria/gradiente |
| `max_frames` | 100 | Truncamiento temporal |
| `lr` | 0.001 | Learning rate |
| `z_shared_dim` | 32 | Dimensión embedding compartido |
| `z_private_dim` | 16 | Dimensión embedding privado |
| `beta_kl` | 1.0 | Peso KL total |
| `beta_kl_private` | **0.01** | Fix z_private collapse |
| `dropout_shared` | **0.5** | Fuerza uso de z_private |
| `lambda_diff` | **0.1** | Separa z_private entre dominios |
| `lambda_infonce` | 1.0 | Peso contrastivo |
| `diff_margin` | 1.0 | Margen para diff loss |

**Fixes aplicados para z_private collapse**:
1. `beta_kl_private=0.01`: KL más bajo permite varianza en z_private
2. `dropout_shared=0.5`: Fuerza al decoder a usar z_private
3. `lambda_diff=0.1`: Penaliza z_private similares entre audio/vib

**Comando de entrenamiento**:
```bash
python experiments/run_roseta_experiment.py \
    --data data/datasets/roseta_v22_full.npz \
    --output data/training_outputs/roseta_v22 \
    --all-data \
    --epochs 100 \
    --batch-size 8 \
    --beta-kl-private 0.01 \
    --dropout-shared 0.5 \
    --lambda-diff 0.1 \
    --lambda-infonce 1.0
```

### 1.4 Split de Datos

| Split | Archivos | Porcentaje |
|-------|----------|------------|
| Train | 89 | 70% |
| Val | 19 | 15% |
| Test | 20 | 15% |

**Estrategia**: GroupSplit por archivo (sin leakage temporal)

---

## 2. Resultados del Entrenamiento

### 2.1 Convergencia

| Métrica | Mejor (Epoch 77) | Final (Epoch 100) |
|---------|------------------|-------------------|
| Val Loss | **3.9774** | 4.3406 |
| Recon Audio | - | 0.0014 |
| Recon Vibration | - | 0.0012 |
| KL Total | - | 1.1259 |
| KL Shared | - | 1.1028 |
| KL Private | - | 0.0231 |
| InfoNCE | - | 3.2349 |
| Diff Loss | - | 0.0018 |

### 2.2 Métricas de Alineamiento (Validación)

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| Cosine Similarity | 0.5808 | Moderado |
| L2 Distance | 4.9899 | - |
| Retrieval Accuracy | 0.0095 | Bajo (~1%) |

### 2.3 z_private Variance (Fix Assessment)

| Métrica | Valor | Criterio | Estado |
|---------|-------|----------|--------|
| var(z_private_audio) | 0.0045 | > 0.1 | **FAIL** |
| var(z_private_vib) | 0.0041 | > 0.1 | **FAIL** |
| z_private_diff | 0.7572 | > 0.5 | PASS |

**Nota**: Aunque la diferencia entre z_private de audio y vibración es buena (0.76), la varianza absoluta sigue siendo muy baja (<0.005), indicando que z_private no modela variación significativa.

---

## 3. Evaluación por Condición (Phase 2)

| Condición | Recon Audio | Recon Vib | Cos Sim | Retrieval Acc |
|-----------|-------------|-----------|---------|---------------|
| HH (Healthy) | 0.0017 | 0.0008 | 0.6379 | 0.63% |
| RU (Rotor Unbalance) | 0.0011 | 0.0009 | 0.6538 | 1.44% |
| RM (Rotor Misalignment) | 0.0012 | 0.0009 | 0.6575 | 1.69% |
| FB (Faulty Bearing) | 0.0012 | 0.0012 | 0.6416 | 1.31% |
| SW (Shaft Wear) | 0.0011 | 0.0009 | 0.6507 | 1.31% |
| VU (Voltage Unbalance) | 0.0014 | 0.0010 | 0.6544 | 1.69% |
| BR (Broken Rotor) | 0.0014 | 0.0009 | 0.6360 | 1.19% |
| KA (Phase Unbalance) | 0.0013 | 0.0010 | 0.6104 | 1.75% |

**Observaciones**:
- Cosine similarity similar entre todas las condiciones (0.61-0.66)
- Healthy no es distinguible de fallas en el espacio z_shared
- Retrieval accuracy uniformemente bajo (~1%)

---

## 4. Cross-Reconstruction (Phase 3)

| Condición | Cross MSE | Pearson Correlation |
|-----------|-----------|---------------------|
| HH | 0.0008 | 0.3111 |
| RU | 0.0009 | 0.1918 |
| FB | 0.0012 | 0.1505 |

**Target**: Pearson > 0.7
**Resultado**: Máximo 0.31 (HH) → **FAIL**

---

## 5. Evaluación con Controles Negativos (Protocolo P0)

### 5.1 Cross-Reconstruction: Aligned vs Shuffled

| Control | Pearson A→V | Pearson V→A | Retrieval Top-1 |
|---------|-------------|-------------|-----------------|
| **Aligned** | 0.2018 | 0.2107 | **0.1130** |
| Shuffled | 0.1929 | 0.2050 | 0.0340 |
| Random z | 0.1949 | 0.1952 | 0.0010 |
| Shuffled+Random | 0.1912 | 0.2003 | 0.0000 |

**Gap (aligned - shuffled)**: 0.2018 - 0.1929 = **0.0089** (A→V)
**Criterio**: > 0.15
**Estado**: **FAIL (CRÍTICO)**

### 5.2 Interpretación del Test de Control

```
Cross-Reconstruction Pearson:
  aligned:   0.2018 (A→V), 0.2107 (V→A)
  shuffled:  0.1929 (A→V), 0.2050 (V→A)
  Δ ≈ 0.007 - 0.009 ≈ 0
```

**Conclusión**: El modelo puede reconstruir vibración desde audio CON CUALQUIER audio, no necesita el par correcto. Esto indica que z_shared codifica información genérica (histograma promedio de la condición), no la identidad del par.

---

## 6. Retrieval Global (WP4)

### 6.1 Audio → Vibración

| Métrica | Valor |
|---------|-------|
| Top-1 Accuracy | 10.94% |
| Top-5 Accuracy | 24.22% |
| Top-10 Accuracy | 43.75% |
| MRR | 0.1975 |
| Mean Rank | 22.9 |

### 6.2 Vibración → Audio

| Métrica | Valor |
|---------|-------|
| Top-1 Accuracy | 5.47% |
| Top-5 Accuracy | 21.09% |
| MRR | 0.1489 |

### 6.3 Intra-Condition Retrieval (Hard Negatives)

| Condición | N | Top-1 | Top-5 | MRR |
|-----------|---|-------|-------|-----|
| BR | 16 | 18.75% | 75.00% | 0.4528 |
| FB | 16 | 31.25% | 75.00% | 0.4860 |
| HH | 16 | 6.25% | 62.50% | 0.2775 |
| KA | 16 | 25.00% | 62.50% | 0.4180 |
| RM | 16 | 37.50% | 81.25% | 0.5903 |
| RU | 16 | **50.00%** | 87.50% | 0.6406 |
| SW | 16 | 37.50% | 81.25% | 0.5694 |
| VU | 16 | 37.50% | 87.50% | 0.5948 |

**Promedio Intra-Condition**: Top-1 = 30.47%, MRR = 0.5037

### 6.4 Baselines

| Baseline | Top-1 | MRR |
|----------|-------|-----|
| Random teórico | 0.78% (1/128) | - |
| Shuffled | 1.56% | 0.0459 |
| Random embeddings | 1.56% | 0.0608 |

### 6.5 Validación

| Check | Criterio | Resultado | Estado |
|-------|----------|-----------|--------|
| Top-1 > 10× random | 7.8% | 10.94% (14×) | **PASS** |
| Top-1 > 15% | 15% | 10.94% | FAIL |
| Shuffled near random | ~0.78% | 1.56% | PASS |

---

## 7. Regime Separation (WP5)

### 7.1 Binary Classification (Healthy vs Fault)

| Modalidad | Silhouette | Linear Probe AUC | Accuracy |
|-----------|------------|------------------|----------|
| Audio z_shared | -0.1445 | 0.7764 ± 0.14 | 87.51% |
| Vib z_shared | -0.1808 | 0.8060 ± 0.08 | - |

### 7.2 Multi-class Classification (8 Condiciones)

| Modalidad | Silhouette | Linear Probe Acc | Random |
|-----------|------------|------------------|--------|
| Audio z_shared | -0.3115 | 18.71% ± 6% | 12.5% |
| Vib z_shared | -0.3450 | 13.23% | 12.5% |

### 7.3 Interpretación

| Métrica | Valor | Umbral | Interpretación |
|---------|-------|--------|----------------|
| Silhouette (binary) | -0.14 | > 0.3 | Weak separation |
| Silhouette (multiclass) | -0.31 | > 0.25 | Very weak |
| Linear Probe AUC | 0.78 | > 0.75 | Moderate |

**Conclusión**: z_shared tiene cierta capacidad para distinguir healthy vs fault (AUC 0.78), pero no separa bien las condiciones específicas ni forma clusters claros (Silhouette negativo).

---

## 8. Criterios GO/NO-GO

| Criterio | Umbral | Resultado | Estado |
|----------|--------|-----------|--------|
| **Gap aligned-shuffled** | **> 0.15** | **0.007** | **FAIL (CRÍTICO)** |
| Retrieval Top-1 | > 10× random | 10.94% vs 0.78% (14×) | PASS |
| Silhouette score | > 0.3 | -0.14 | FAIL |
| var(z_private) | > 0.1 | 0.0043 | FAIL |

**Decisión**: **NO-GO** (criterio crítico fallido)

Según ROADMAP_FINAL_EXTRACCION_RATIOS.md (Sección 10.4):
> "NO-GO Automático: aligned ≈ shuffled (gap < 2×)"

El gap de 0.007 está muy por debajo del umbral de 0.15.

---

## 9. Comparación con Experimentos Anteriores

| Experimento | Gap Pre-Red | Gap Post-Red | Retrieval Top-1 |
|-------------|-------------|--------------|-----------------|
| Rosetta1 2.0 (v1 extractor) | 0.004 | 0.002 | 0.78% |
| **Rosetta v2.2** | **0.691** | **0.007** | **10.94%** |
| **Mejora** | **172×** | **3.5×** | **14×** |

### Análisis del Colapso

```
Mejora del extractor:     172× (0.004 → 0.691)
Mejora del modelo:        3.5× (0.002 → 0.007)
Ratio de capitalización:  3.5 / 172 = 2%
```

**El modelo solo capitalizó el 2% de la mejora del extractor.**

---

## 10. Diagnóstico

### 10.1 Por Qué Falló

1. **InfoNCE insuficiente**: La pérdida contrastiva no fuerza discriminación suficiente entre pares
2. **z_shared genérico**: El embedding compartido captura "histograma promedio" de la condición, no la identidad del par específico
3. **z_private colapsado**: A pesar de los fixes, var < 0.01 indica que z_private no modela variación significativa
4. **Arquitectura VAE**: El decoder puede reconstruir desde cualquier z, no necesita el z correcto

### 10.2 Evidencia del Problema

El test clave es la comparación aligned vs shuffled en cross-reconstruction:

```python
# Si el modelo aprendió correspondencia real:
pearson(aligned) >> pearson(shuffled)

# Lo que observamos:
pearson(aligned)  = 0.2018
pearson(shuffled) = 0.1929
Δ = 0.0089 ≈ 0  # NO HAY DIFERENCIA
```

Esto significa que el modelo genera una reconstrucción igualmente buena (o mala) sin importar si el par es correcto.

### 10.3 El Histograma No Es Suficiente

Aunque el Extractor v2.2 produce histogramas discriminativos pre-red (gap 0.69):
- Los histogramas siguen siendo una **representación agregada**
- Pierden la información de **"quién se relaciona con quién"**
- Múltiples configuraciones de picos pueden producir el mismo histograma

---

## 11. Archivos Generados

| Archivo | Tamaño | Descripción |
|---------|--------|-------------|
| `data/datasets/roseta_v22_full.npz` | 9.4 MB | Dataset con extractor v2.2 |
| `data/training_outputs/roseta_v22/best_model.pt` | 38 MB | Mejor modelo (epoch 77) |
| `data/training_outputs/roseta_v22/final_model.pt` | 38 MB | Modelo final (epoch 100) |
| `data/training_outputs/roseta_v22/results.json` | 109 KB | Métricas completas |
| `data/training_outputs/roseta_v22/roseta_experiment_report.md` | 1.7 KB | Reporte automático |
| `data/evaluations/retrieval/REPORT_RETRIEVAL.md` | - | Métricas de retrieval |
| `data/evaluations/regime_separation/REPORT_REGIME_SEPARATION.md` | - | Métricas de separación |
| `data/evaluations/regime_separation/regime_separation_pca.png` | - | Visualización PCA |

---

## 12. Decisión y Próximos Pasos

### Decisión

Siguiendo el árbol de decisiones del ROADMAP_FINAL_EXTRACCION_RATIOS.md (Sección 10.5):

```
Fase 2 NO-GO (parcial)
        │
        ▼
┌───────────────────┐
│ ITERAR GRUPO 1:   │
│ - Constellations  │ ← Seleccionado
│ - Multi-banda     │
│ - TF-IDF          │
└─────────┬─────────┘
          │
     Si falla
          │
          ▼
┌───────────────────┐
│ GRUPO 2:          │
│ - PRISM-JEPA      │ ← Contingencia
│ - Log-spec + CNN  │
│ - Transformer     │
└───────────────────┘
```

### Fase 3A: Ratio Constellations (Grupo 1C)

**Justificación**: El histograma pierde "quién se relaciona con quién". Las constellations preservan esta estructura.

**Implementación**:
```python
token = {
    'log_ratio': np.log2(target.freq / anchor.freq),
    'delta_t': target.time - anchor.time,
    'weight': np.sqrt(anchor.amp * target.amp),
    'band_id': get_band_id(anchor.freq)
}
```

**Criterio GO**: Gap aligned-shuffled > 0.15

### Fase 3B: PRISM-JEPA (Grupo 2D) - Contingencia

**Justificación**: El VAE tiene shortcut de reconstrucción. JEPA predice en espacio latente sin decoder.

**Criterio GO**: Retrieval Top-1 > 15%

### Fallback: Publicar H1/H2

Si ambas fases fallan:
- H1 (Estructura): **VALIDADA**
- H2 (Aprendibilidad): **VALIDADA**
- H3 (Cross-modality): **NO VALIDADA** bajo múltiples enfoques

---

## 13. Lecciones Aprendidas

1. **Mejorar el descriptor no garantiza mejorar el modelo**: 172× mejora pre-red → solo 3.5× post-red
2. **Los controles negativos son esenciales**: Sin ellos, habríamos declarado éxito basándonos en métricas absolutas
3. **El VAE colapsa información**: El decoder puede funcionar sin el z correcto
4. **El histograma es una representación lossy**: Pierde relaciones estructurales entre picos
5. **z_private collapse es difícil de resolver**: Los fixes mejoraron pero no resolvieron (var ~0.004 vs objetivo 0.1)

---

*Informe generado: 2026-01-30*
*Fase 2 del Revisionismo de Extracción de Ratios*
*Proyecto Phideus v5.0*
