# Proyecto Estado Actual - Phideus v5.0

**Actualizado**: 2026-01-31
**Estado**: Fase 2 completada (NO-GO) - Próxima: Fase 3A (Ratio Constellations)

---

## Resumen Ejecutivo

### Estado de Hipótesis

| Hipótesis | Estado | Evidencia |
|-----------|--------|-----------|
| H1: Estructura de ratios | **VALIDADA** | Distribuciones no aleatorias |
| H2: Aprendibilidad | **VALIDADA** | VAE/HRM val_loss < 0.5 |
| H3: Cross-modality | **NO VALIDADA** | Gap aligned-shuffled = 0.007 (necesario > 0.15) |

### Conclusión Final

**El Extractor v2.2 mejoró la discriminabilidad pre-red 172×, pero esto no se tradujo en aprendizaje cross-modal.** El modelo RosetaVAE sigue generando embeddings que no distinguen pares alineados de shuffled.

---

## Fase 2: Re-entrenamiento con Extractor v2.2

### Configuración

**Extractor v2.2 (config_002)**:
- Top-K peaks: 8
- Min prominence: 0.1
- Temporal stability: 0.7
- Gap pre-red: **0.691** (172× mejor que v1)

**Modelo RosetaVAE**:
- 100 epochs, batch=8
- beta_kl_private: 0.01 (fix z_private collapse)
- dropout_shared: 0.5
- lambda_diff: 0.1

### Resultados

| Criterio | Umbral | Resultado | Estado |
|----------|--------|-----------|--------|
| **Gap aligned-shuffled** | **> 0.15** | **0.007** | **FAIL (CRÍTICO)** |
| Retrieval Top-1 | > 10× random | 10.94% vs 0.78% (14×) | PASS |
| Silhouette score | > 0.3 | -0.14 | FAIL |
| var(z_private) | > 0.1 | 0.0043 | FAIL |

**Decisión: NO-GO** - El criterio crítico (gap aligned-shuffled) falló.

### Cross-Reconstruction con Controles

| Control | Pearson A→V | Pearson V→A | Retrieval Top-1 |
|---------|-------------|-------------|-----------------|
| Aligned | 0.2018 | 0.2107 | 0.1130 |
| Shuffled | 0.1929 | 0.2050 | 0.0340 |
| Random z | 0.1949 | 0.1952 | 0.0010 |

**Conclusión**: El modelo puede reconstruir vibración desde audio CON CUALQUIER audio. No necesita el par correcto.

---

## Comparación: Fase 1 vs Fase 2

| Métrica | Fase 1 (pre-red) | Fase 2 (post-red) | Ratio |
|---------|------------------|-------------------|-------|
| Gap aligned-shuffled | 0.691 | 0.007 | 1% |
| Retrieval Top-1 | N/A | 10.94% | - |
| Pearson cross-recon | N/A | 0.20 | - |

**Interpretación**: La mejora del extractor (172×) se redujo a casi nada (3.5×) después del modelo. El VAE colapsa la información discriminativa.

---

## Diagnóstico

### Por qué falló H3

1. **InfoNCE insuficiente**: No fuerza discriminación entre pares
2. **z_shared genérico**: Captura "histograma promedio" de condición
3. **z_private colapsado**: var < 0.01 indica no modela variación privada
4. **Posiblemente**: Los ratio-histograms no codifican identidad de par

### Evidencia

```
Cross-Reconstruction Pearson:
  aligned:   0.2018 (A→V), 0.2107 (V→A)
  shuffled:  0.1929 (A→V), 0.2050 (V→A)
  Δ = 0.007 ≈ 0
```

---

## Logros del Proyecto

1. **H1 VALIDADA**: Las señales contienen distribuciones de ratios estructuradas
2. **H2 VALIDADA**: Redes neuronales pueden aprenderlas (val_loss < 0.5)
3. **Analizador 5.0**: Demostró que representación > arquitectura
4. **VAE Rehabilitado**: De val_loss 4212 → 0.456
5. **Metodología robusta**: Controles negativos que detectaron el problema
6. **Extractor v2.2**: Histogramas discriminativos (gap 172× mejor)

---

## Documentación

### Revisionismo

| Fase | Documento | Descripción |
|------|-----------|-------------|
| General | `Documents/Revisionismo/ROADMAP.md` | Roadmap del revisionismo |
| Fase 0 | `Documents/Revisionismo/Fase_0/Fase_0_results.md` | Auditoría inicial |
| Fase 1 | `Documents/Revisionismo/Fase_1/Fase_1_results.md` | Extractor v2.2 (GO) |
| Fase 2 | `Documents/Revisionismo/Fase_2/Fase_2_results.md` | Re-entrenamiento (NO-GO) |
| Fase 3A | `Documents/Revisionismo/Fase_3A/Fase_3A.md` | Plan Ratio Constellations |

### Analizador

| Documento | Contenido |
|-----------|-----------|
| `Documents/Revisionismo/Analizador/SPEC_ANALIZADOR_5.0.md` | Especificación técnica |
| `Documents/Revisionismo/Analizador/INFORME_REVISIONISMO_EXTRACCION_RATIOS.md` | Diagnóstico y roadmap |

### Histórico

| Documento | Contenido |
|-----------|-----------|
| `Documents/Rosetta_v1_y_v2/ROSETTA1_2.0_IMPLEMENTATION_PLAN.md` | Plan Rosetta 2.0 |
| `Documents/Rosetta_v1_y_v2/ROSETTA1_2.0_RESULTADOS_FULL.md` | Resultados (NO-GO) |

---

## Decisión: Siguiente Fase

**Fecha decisión**: 2026-01-31
**Diagnóstico**: El VAE colapsa la información discriminativa del histograma (172× mejora pre-red → solo 3.5× post-red)

### Plan Aprobado: Fase 3A - Ratio Constellations

**Plan completo**: `Documents/Revisionismo/Fase_3A/Fase_3A.md`

#### Concepto Principal

Cambiar de histograma denso [T, 256, 3] a **tokens sparse** estilo Shazam:

```python
token = {
    'log_ratio': np.log2(target.freq / anchor.freq),
    'delta_t': target.time - anchor.time,
    'weight': np.sqrt(anchor.amp * target.amp),
    'anchor_band': get_band_id(anchor.freq),
    'target_band': get_band_id(target.freq)
}
# Output: [T, 48, 5] en lugar de [T, 256, 3]
```

#### 6 Configuraciones a Probar

| Config | Encoder | Decoder |
|--------|---------|---------|
| C1 | MLP+Attention | Histograma |
| C2 | MLP+Attention | Tokens |
| C3 | Transformer | Histograma |
| C4 | Transformer | Tokens |
| C5 | MLP+Attention | **JEPA-lite (sin decoder)** |
| C6 | Transformer | **JEPA-lite (sin decoder)** |

#### Mejoras Incorporadas (críticas GPT5.2Think)

1. **Attention pooling** en vez de mean pooling (preserva relaciones)
2. **Variantes JEPA-lite** sin decoder (evita shortcut reconstructivo)
3. **Hard negatives intra-condición** como métrica principal
4. **Auditoría de evaluación** previa (resolver inconsistencia 10.94% vs 0.78%)

#### Criterios GO/NO-GO

| Criterio | Umbral |
|----------|--------|
| **Gap aligned-shuffled (intra-cond)** | **> 0.10** |
| Gap aligned-shuffled (global) | > 0.15 |
| Retrieval Top-1 (intra-cond) | > 2× random |

### Fase 3B: PRISM-JEPA (Si 3A falla)

Peak-tokens + ratio-slots + predicción latente SIN decoder.

### Fallback: Publicar H1/H2

Documentar H1/H2 como contribución válida, H3 como resultado negativo.

---

## Artefactos Generados (Fase 2)

| Archivo | Descripción |
|---------|-------------|
| `data/datasets/roseta_v22_full.npz` | Dataset con extractor v2.2 (9.4 MB) |
| `data/training_outputs/roseta_v22/best_model.pt` | Mejor modelo (38 MB) |
| `data/training_outputs/roseta_v22/results.json` | Métricas de entrenamiento |

---

*Última actualización: 2026-01-31 - Fase 2 NO-GO, próxima: Fase 3A (Ratio Constellations)*
