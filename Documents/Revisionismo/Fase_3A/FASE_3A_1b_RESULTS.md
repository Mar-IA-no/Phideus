# Fase 3A-1b: Corrección del Extractor y Re-entrenamiento

**Fecha**: 2026-02-01
**Estado**: **NO-GO** - H3 (cross-modality) no validada

---

## Resumen Ejecutivo

La Fase 3A-1b intentó corregir las causas raíz identificadas en la auditoría:
1. Extractor de constellations con datos no discriminativos
2. Configuración de training incorrecta

**Resultado**: A pesar de las mejoras, el modelo **no puede aprender correspondencia cross-modal** con representación de tokens sparse.

---

## Cambios Implementados

### Extractor de Constellations (v2)

| Parámetro | v1 (original) | v2 (corregido) | v3 (agresivo) |
|-----------|---------------|----------------|---------------|
| n_frequency_bands | 8 | 16 | 16 |
| target_zone_hz | 2000 | 1000 | 2000 |
| target_zone_frames | 5 | 3 | 5 |
| max_anchors | 12 | 16 | 20 |
| max_targets_per_anchor | 4 | 4 | 5 |
| min_prominence | 0.2 | 0.2 | **0.1** |
| temporal_stability | 0.6 | 0.6 | **0.4** |
| Band assignment | log-scale | **linear** | linear |
| Min anchor freq | - | **50 Hz** | 50 Hz |

### Resultados de Extracción

| Dataset | Sparse frames | Avg tokens/frame | Gap aligned-shuffled |
|---------|---------------|------------------|----------------------|
| v1 | 91.8% | 11.2 | -0.0019 |
| v2 | 82.5% | 10.1 | 0.0268 |
| **v3** | **18.4%** | **Audio: 19.7, Vib: 33.7** | **0.0294** |

**Conclusión**: Más tokens mejoran sparsity pero no discriminabilidad pre-red.

---

## Configuración de Training Corregida

| Parámetro | v1 (incorrecto) | v2 (corregido) |
|-----------|-----------------|----------------|
| dropout_shared | 0 | **0.5** |
| beta_kl_private | None | **0.01** |
| lambda_diff | 0 | **0.1** |
| epochs | 100 | 50 |

### Training Results (JEPA-lite, mlp encoder)

```
Epoch 1: val_loss=7.98, cos_sim=0.991
Epoch 25: val_loss=4.76, cos_sim=0.992
Epoch 50: val_loss=4.02, cos_sim=0.991 (best: 4.02)
```

- **cos_sim > 0.99**: Embeddings audio-vib son casi idénticos
- Esto parece bueno pero indica **colapso** - el modelo mapea todo al mismo punto

---

## Evaluación de Retrieval

### Resultados Finales

| Métrica | Valor | Umbral | Estado |
|---------|-------|--------|--------|
| **Top-1 Global** | **0.78%** | > 15% | **FAIL** |
| Top-5 Global | 5.47% | - | - |
| MRR | 0.048 | - | - |
| Random baseline | 0.78% | - | - |

### Intra-Condition Retrieval

| Condición | Top-1 | Top-5 | MRR |
|-----------|-------|-------|-----|
| BR | 6.25% | 50.0% | 0.26 |
| **FB** | **18.75%** | 56.25% | 0.36 |
| HH | 0.00% | 37.5% | 0.19 |
| KA | 0.00% | 37.5% | 0.19 |
| **RM** | **18.75%** | 43.75% | 0.33 |
| RU | 0.00% | 31.25% | 0.18 |
| SW | 0.00% | 31.25% | 0.17 |
| VU | 0.00% | 50.0% | 0.19 |

**Nota**: FB y RM muestran mejor retrieval, pero en general el modelo no discrimina.

### Negative Controls

| Control | Top-1 | Esperado |
|---------|-------|----------|
| Shuffled pairs | 0.00% | ~0.78% |
| Random embeddings | 1.56% | ~0.78% |

---

## Análisis de Causas

### Por qué la representación de tokens NO funciona

1. **Pérdida de información**: El histograma denso [T, 256, 3] captura la distribución completa de ratios. Los tokens sparse [T, K×M, 5] solo capturan los picos más prominentes.

2. **Discriminabilidad pre-red**:
   - Histograma v2.2: gap = **0.691** (172× random)
   - Constellation v3: gap = **0.029** (~1× random)

3. **El problema no es la arquitectura ni el training**: Es la representación de entrada.

4. **Shazam funciona diferente**: Shazam usa tokens para **fingerprinting**, no para **cross-modal alignment**. El matching es exact-hash, no embedding-similarity.

---

## Lecciones Aprendidas

### Lo que funcionó (parcialmente)
- Extractor v2.2 con histogramas densos: gap = 0.691
- Linear bands vs log-scale: mejor distribución
- Más tokens reduce sparsity

### Lo que NO funcionó
- Tokens sparse para cross-modal learning
- JEPA-lite sin decoder
- Constellation extraction en general

### Insight fundamental

> **La hipótesis H3 (cross-modality via ratio histograms) requiere representaciones densas.**
>
> Los tokens sparse eliminan la información de distribución que permite discriminar.
> Esto explica por qué Fase 2 (histogramas, gap=0.007) también falló - el problema está antes de la red.

---

## Decisión GO/NO-GO

### Criterios

| Criterio | Umbral | Resultado | Estado |
|----------|--------|-----------|--------|
| Top-1 Retrieval | > 15% | 0.78% | **FAIL** |
| Gap aligned-shuffled | > 0.10 | 0.029 | **FAIL** |
| Varianza embeddings | > 0.01 | ~0.99 (collapse) | **FAIL** |

### Veredicto: **NO-GO**

La hipótesis H3 (cross-modality via harmonic ratios) **no puede validarse** con las representaciones actuales.

---

## Recomendaciones

### Opción A: Aceptar resultado negativo
- Publicar resultados: "Harmonic ratios no transfieren cross-modal"
- Valor científico en resultados negativos bien documentados

### Opción B: Cambiar representación fundamentalmente
- Abandonar ratio histograms/tokens
- Probar representaciones alternativas:
  - Raw spectrograms con contrastive learning
  - Learned embeddings (no hand-crafted features)
  - Multi-scale wavelets

### Opción C: Cambiar hipótesis
- H3' alternativa: "Audio y vibración NO comparten estructura armónica"
- Buscar otras propiedades compartidas (temporal patterns, energy profiles)

---

## Archivos Generados

| Archivo | Descripción |
|---------|-------------|
| `data/datasets/roseta_constellation_v3.npz` | Dataset con tokens agresivos |
| `data/training_outputs/constellation_v3_jepa_mlp/` | Modelo entrenado |
| `data/evaluations/retrieval_v3_jepa/` | Métricas de retrieval |

## Commits Relacionados

- Extractor v2: defaults mejorados, linear bands
- Dataset v3: más tokens, constraints relajados
- Training: dropout=0.5, beta_kl_private=0.01

---

## Resumen Final

La Fase 3A completa (incluyendo 3A-1b) demuestra que:

1. **H1 (Estructura)**: VALIDADA - Señales contienen distribuciones estructuradas
2. **H2 (Aprendibilidad)**: VALIDADA - Redes pueden aprenderlas (val_loss < 0.5)
3. **H3 (Cross-modality)**: **NO VALIDADA** - Gap aligned-shuffled ≈ 0

> **El problema NO es la arquitectura (VAE, JEPA) ni el training (dropout, beta_kl).**
> **El problema es la representación: ni histogramas ni tokens capturan correspondencia cross-modal.**

---

*Fase 3A-1b del Revisionismo de Extracción de Ratios*
*Proyecto Phideus v5.0 - Febrero 2026*
