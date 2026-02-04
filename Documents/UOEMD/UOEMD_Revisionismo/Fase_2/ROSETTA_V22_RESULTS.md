# Rosetta v2.2 Results - Fase 2

**Fecha**: 2026-01-30
**Estado**: **NO-GO** - H3 no validada
**Decisión**: El Extractor v2.2 mejoró la discriminabilidad pre-red pero el modelo no aprende cross-modality

---

## Resumen Ejecutivo

La Fase 2 probó si histogramas más discriminativos (gap pre-red = 0.69 vs 0.004) habilitarían al modelo para aprender correspondencia real audio ↔ vibración.

**Resultado**: El modelo sigue produciendo embeddings similares para pares alineados y shuffled. La mejora en el extractor no se traduce en aprendizaje cross-modal.

---

## Configuración Experimental

### Extractor v2.2 (config_002)
| Parámetro | Valor |
|-----------|-------|
| Top-K peaks | 8 |
| Min prominence | 0.1 |
| Temporal stability | 0.7 |
| Warped bins | No |
| **Gap pre-red** | **0.691** (172× mejor que v1) |

### Modelo RosetaVAE
| Parámetro | Valor |
|-----------|-------|
| Epochs | 100 |
| Batch size | 8 |
| beta_kl_private | 0.01 (fix z_private collapse) |
| dropout_shared | 0.5 |
| lambda_diff | 0.1 |
| lambda_infonce | 1.0 |
| z_shared_dim | 32 |
| z_private_dim | 16 |

---

## Resultados Detallados

### 1. Cross-Reconstruction con Controles Negativos

| Control | Pearson A→V | Pearson V→A | Retrieval Top-1 |
|---------|-------------|-------------|-----------------|
| Aligned | 0.2018 | 0.2107 | **0.1130** |
| Shuffled | 0.1929 | 0.2050 | 0.0340 |
| Random z | 0.1949 | 0.1952 | 0.0010 |
| Shuf+Rand | 0.1912 | 0.2003 | 0.0000 |

**Δcorr (aligned - shuffled) = 0.0073**
**Criterio: > 0.15** → **FAIL**

### 2. Retrieval Global

| Métrica | Audio→Vib | Vib→Audio |
|---------|-----------|-----------|
| Top-1 | 10.94% | 5.47% |
| Top-5 | 24.22% | 21.09% |
| Top-10 | 43.75% | - |
| MRR | 0.1975 | 0.1489 |

**Random baseline: 0.78%**
**Criterio: Top-1 > 10× random (7.8%)** → **MARGINAL PASS** (10.94%)

### 3. Regime Separation

| Métrica | Audio | Vibration |
|---------|-------|-----------|
| Silhouette (binary) | -0.1445 | -0.1808 |
| Linear Probe AUC | 0.7764 | 0.8060 |
| Silhouette (multiclass) | -0.3115 | -0.3450 |

**Criterio: Silhouette > 0.3** → **FAIL**

### 4. z_private Variance

| Métrica | Valor |
|---------|-------|
| var(z_private_audio) | 0.0045 |
| var(z_private_vib) | 0.0041 |
| z_private_diff | 0.7572 |

**Criterio: var > 0.1** → **FAIL** (aunque mejoró vs v1)

---

## Criterios GO/NO-GO

| Criterio | Umbral | Resultado | Estado |
|----------|--------|-----------|--------|
| **Gap aligned-shuffled** | **> 0.15** | **0.0073** | **FAIL (CRÍTICO)** |
| Retrieval Top-1 | > 10× random | 10.94% vs 0.78% (14×) | PASS |
| Silhouette score | > 0.3 | -0.14 | FAIL |
| var(z_private) | > 0.1 | 0.0043 | FAIL |

**Decisión: NO-GO** (criterio crítico fallido)

---

## Comparación con Rosetta1 2.0

| Métrica | Rosetta 2.0 (v1 extractor) | Rosetta v2.2 | Cambio |
|---------|---------------------------|--------------|--------|
| Gap pre-red | 0.004 | 0.691 | +172× |
| Gap post-red (modelo) | 0.002 | 0.007 | +3.5× |
| Retrieval Top-1 | 0.78% | 10.94% | +14× |
| Pearson cross-recon | 0.31 | 0.20 | -35% |
| Silhouette | N/A | -0.14 | - |

### Interpretación

1. **El extractor mejoró**: Gap pre-red subió 172×
2. **El modelo no capitalizó**: Gap post-red solo mejoró 3.5×
3. **Retrieval mejoró**: Pero aún por debajo de criterio robusto (15%)
4. **Cross-reconstruction empeoró**: Correlación más baja

---

## Diagnóstico

### Por qué falló

El modelo no está aprendiendo correspondencia audio ↔ vibración porque:

1. **InfoNCE no discrimina suficientemente**: Las reconstrucciones de pares shuffled son casi igual de buenas que alineados
2. **z_shared genérico**: El embedding compartido captura "histograma promedio" de la condición, no la identidad del par
3. **z_private colapsado**: Aunque mejoró, var < 0.01 indica que z_private no modela variación privada

### Evidencia clave

```
Cross-Reconstruction Pearson:
  aligned:   0.2018 (A→V), 0.2107 (V→A)
  shuffled:  0.1929 (A→V), 0.2050 (V→A)

  Δ = 0.007 - 0.006 ≈ 0
```

**Conclusión**: El modelo puede reconstruir vibración desde audio CON CUALQUIER audio, no necesita el par correcto.

---

## Opciones para Fase 3

### Opción A: Ajustar hiperparámetros del modelo
- Aumentar lambda_infonce (10.0)
- Probar temperature más baja en InfoNCE
- Agregar hard negative mining

**Probabilidad de éxito**: Baja (el problema parece arquitectural)

### Opción B: Probar config_012 (máximo gap = 0.702)
- K=8, prom=0.05, stab=0.9, warped=Yes

**Probabilidad de éxito**: Muy baja (gap similar a config_002)

### Opción C: Cambiar a Grupo 2 (arquitectura diferente)
- Log-spectrogram + conv encoder
- JEPA predictivo
- Transformer cross-modal

**Probabilidad de éxito**: Moderada (cambio fundamental de enfoque)

### Opción D: Abandonar H3
- Publicar H1/H2 como resultados válidos
- Documentar que cross-modality NO se valida con ratio-histograms + VAE

**Probabilidad de éxito**: N/A (no es "éxito" pero es científicamente válido)

---

## Archivos Generados

| Archivo | Descripción |
|---------|-------------|
| `data/datasets/roseta_v22_full.npz` | Dataset con extractor v2.2 |
| `data/training_outputs/roseta_v22/best_model.pt` | Mejor modelo (epoch 77) |
| `data/evaluations/retrieval/REPORT_RETRIEVAL.md` | Métricas retrieval |
| `data/evaluations/regime_separation/REPORT_REGIME_SEPARATION.md` | Métricas separación |

---

## Conclusión

**H3 (Cross-modality) sigue sin validarse.**

El Extractor v2.2 produjo histogramas significativamente más discriminativos (gap pre-red 172× mayor), pero esta mejora no se tradujo en aprendizaje cross-modal. El modelo RosetaVAE sigue generando embeddings z_shared que no distinguen entre pares alineados y shuffled.

Esto sugiere que el problema no es solo la representación de entrada (histogramas), sino potencialmente:
1. La arquitectura VAE con InfoNCE
2. La naturaleza del dataset UOEMD
3. La hipótesis misma de que ratios armónicos codifican correspondencia cross-modal

---

## Decisión Tomada: Fase 3

**Fecha**: 2026-01-30

Siguiendo el árbol de decisiones del ROADMAP_FINAL_EXTRACCION_RATIOS.md, se decide continuar con:

### Fase 3A: Ratio Constellations (Grupo 1C)

El diagnóstico indica que el VAE colapsa la información discriminativa. El histograma pierde la información de "quién se relaciona con quién". Las Constellations preservan esta estructura.

**Implementación**: Tokens sparse (log_ratio, delta_t, weight, band_id) estilo Shazam
**Criterio GO**: Gap aligned-shuffled > 0.15

### Fase 3B: PRISM-JEPA (Grupo 2D) - Contingencia

Si Constellations falla, cambiar a arquitectura sin decoder (elimina shortcut de reconstrucción).

**Implementación**: Peak-tokens + ratio-slots + predicción latente
**Criterio GO**: Retrieval Top-1 > 15%

### Fallback: Publicar H1/H2

Si ambas fases fallan, documentar resultados negativos como contribución válida.

---

*Generado: 2026-01-30 por Fase 2 del Revisionismo de Extracción de Ratios*
