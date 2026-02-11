# Fase 3A Sweep Results - Ratio Constellations

**Fecha**: 2026-02-01
**Branch**: feature/extractor-v22

---

## Resumen Ejecutivo

**RESULTADO: NO-GO**

Todas las 6 configuraciones de Ratio Constellations fallaron en alcanzar los criterios GO/NO-GO.

| Config | Encoder | Decoder | Params | Best Val Loss | Top-1 | MRR | Status |
|--------|---------|---------|--------|---------------|-------|-----|--------|
| **C1** | MLP+Attn | Histogram | 1.63M | 2.94 | 0.78% | 0.036 | FAIL |
| **C2** | MLP+Attn | Token | 1.51M | 16.05 | 0.78% | 0.043 | FAIL |
| **C3** | Transformer | Histogram | 1.81M | 3.00 | 0.78% | 0.036 | FAIL |
| **C4** | Transformer | Token | 1.68M | 16.16 | 0.78% | 0.039 | FAIL |
| **C5** | MLP+Attn | JEPA-lite | 727K | 1.95 | **1.56%** | 0.045 | FAIL |
| **C6** | Transformer | JEPA-lite | 904K | 3.11 | 0.78% | 0.041 | FAIL |

**Random baseline**: 0.78% (1/128 samples)

---

## Criterios GO/NO-GO

| Criterio | Umbral | Mejor Resultado | Status |
|----------|--------|-----------------|--------|
| **Gap aligned-shuffled (intra-cond)** | > 0.10 | N/A* | FAIL |
| Top-1 Retrieval | > 15% | 1.56% (C5) | FAIL |
| Top-1 vs Random | > 2× | 2× (C5) | MARGINAL |

*Gap no calculado porque retrieval está en nivel random.

---

## Análisis

### Observaciones

1. **Colapso de embeddings**: Todos los modelos muestran cos_sim ≈ 1.0 durante entrenamiento, indicando que los embeddings se vuelven idénticos.

2. **JEPA-lite ligeramente mejor**: C5 (MLP+JEPA) muestra 2× random (1.56% vs 0.78%), pero sigue siendo insuficiente.

3. **Val loss engañosa**: C5 tiene la mejor val_loss (1.95) pero no se traduce en mejor retrieval.

4. **Dataset pequeño**: Solo 128 muestras (16 por condición) puede ser insuficiente para aprender correspondencia.

### Posibles Causas

1. **Tokens constellation demasiado sparse**: Promedio 11-13 tokens/frame vs 256 bins del histograma puede perder información crítica.

2. **Attention pooling colapsa información**: El pooling attention puede estar promediando información discriminativa.

3. **InfoNCE no es suficiente**: La pérdida contrastiva puede no ser suficiente para aprender correspondencia cross-modal.

4. **Problema de datos**: El dataset UOEMD puede no tener suficiente variabilidad para que los pares audio-vibración sean distinguibles.

---

## Siguiente Paso: Fase 3B o Cierre

### Opción A: Fase 3B - PRISM-JEPA (más investigación)
- Cambiar arquitectura radicalmente
- Usar predicción directa en espacio latente sin reconstrucción
- Agregar más regularización

### Opción B: Publicar Resultados Negativos
- Documentar H3 como no validada
- Publicar H1/H2 como contribución válida
- El resultado negativo es científicamente válido

### Opción C: Buscar Más Datos
- El dataset de 128 muestras puede ser insuficiente
- Buscar datasets audio-vibración más grandes

---

## Archivos Generados

```
data/training_outputs/
├── constellation_C1_mlp_hist/
├── constellation_C2_mlp_token/
├── constellation_C3_trans_hist/
├── constellation_C4_trans_token/
├── constellation_C5_mlp_jepa/
└── constellation_C6_trans_jepa/

data/evaluations/
├── constellation_C1/
├── constellation_C2_mlp_token/
├── constellation_C3_trans_hist/
├── constellation_C4_trans_token/
├── constellation_C5_mlp_jepa/
└── constellation_C6_trans_jepa/
```

---

*Fase 3A del Revisionismo de Extracción de Ratios*
*Proyecto Phideus v5.0 - Febrero 2026*
