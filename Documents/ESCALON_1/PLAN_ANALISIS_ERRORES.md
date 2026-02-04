# Plan de Análisis de Errores - Escalón 1

**Fecha**: 2026-02-04
**Objetivo**: Identificar causas de errores y oportunidades de mejora

---

## Contexto

Con 26% accuracy (N=20), el sistema falla en ~74% de los casos. Necesitamos entender:
1. ¿Qué piezas fallan consistentemente?
2. ¿Por qué fallan?
3. ¿Qué podemos mejorar?

---

## Fase 1: Caracterización de Errores

### 1.1 Clasificar piezas por dificultad

```python
# Generar para cada pieza:
# - accuracy individual
# - rank promedio del ground truth
# - características: duración, densidad de notas, tempo
```

**Preguntas**:
- ¿Las piezas largas son más difíciles?
- ¿Las piezas con muchas notas son más difíciles?
- ¿Hay correlación con el año/compositor?

### 1.2 Analizar distribución de errores

```python
# Para cada query incorrecta:
# - ¿Qué pieza fue predicha en lugar de la correcta?
# - ¿Qué score tuvo la correcta vs la predicha?
# - ¿Cuántos hashes coincidieron con la correcta?
```

**Preguntas**:
- ¿Los errores son "casi aciertos" (rank 2-3) o totalmente equivocados?
- ¿Hay piezas que "roban" queries de otras?

---

## Fase 2: Análisis de Hashes

### 2.1 Distribución de hashes

```python
# Analizar:
# - ¿Cuántos hashes únicos por pieza?
# - ¿Cuál es el document frequency de cada hash?
# - ¿Cuántos hashes son "stopwords" (aparecen en >X% de piezas)?
```

**Hipótesis**: Si muchos hashes son comunes, el IDF no es suficiente.

### 2.2 Overlap de hashes

```python
# Para pares aligned vs random:
# - ¿Cuántos hashes coinciden?
# - ¿Cuál es el gap real?
# - ¿Los hashes que coinciden son informativos o comunes?
```

**Métricas**:
- overlap_aligned_unique (excluyendo stopwords)
- overlap_random_unique
- gap_unique

### 2.3 Hashes discriminativos

```python
# Identificar hashes que:
# - Solo aparecen en 1-2 piezas (muy específicos)
# - Tienen alto IDF
# - Coinciden cross-modalmente
```

---

## Fase 3: Análisis de Alineación

### 3.1 Verificar alineación temporal

```python
# Para cada par audio-MIDI:
# - Correlacionar onsets detectados en audio vs MIDI
# - Calcular offset promedio
# - Identificar desalineaciones
```

**Visualización**: Plot de onsets audio vs MIDI para detectar problemas.

### 3.2 Verificar coincidencia de pitch

```python
# Para eventos detectados:
# - ¿El pitch estimado en audio coincide con MIDI?
# - ¿Cuál es el error promedio en semitonos?
```

---

## Fase 4: Ablations

### 4.1 Parámetros de IDF

| Stoplist % | Accuracy esperada |
|------------|-------------------|
| 50% (original) | baseline |
| 30% (actual) | actual |
| 20% | ? |
| 10% | ? |
| 5% | ? |

### 4.2 Parámetros de hashing

| Parámetro | Valores a probar |
|-----------|-----------------|
| DT_BIN_SIZE | 1, 2, 4, 8 frames |
| LOG_RATIO_BIN_SIZE | 1/12, 1/24, 1/48 octava |
| PAIR_WINDOW | 50, 86, 150 frames |
| FAN_OUT | 2, 4, 8 |

### 4.3 Filtros de calidad

- Solo usar hashes de anchors que son onsets (anchor_is_onset=True)
- Solo usar hashes con weight > umbral
- Solo usar hashes de frames con suficiente energía

---

## Fase 5: Mejoras Potenciales

### 5.1 Ensemble A + B

Combinar scores de Route A y Route B:
```python
score_ensemble = alpha * score_A + (1-alpha) * score_B
```

### 5.2 Re-ranking

Usar features adicionales para re-rankear candidatos:
- Similitud de histograma global
- Similitud de chroma
- Duración similar

### 5.3 Negative mining

Identificar piezas "confusoras" y entrenar para distinguirlas.

---

## Scripts a Crear

```
experiments/un_audio_un_midi/
├── analyze_errors.py          # Fase 1: Caracterización
├── analyze_hashes.py          # Fase 2: Distribución de hashes
├── analyze_alignment.py       # Fase 3: Alineación
├── run_ablations.py           # Fase 4: Ablations
└── test_improvements.py       # Fase 5: Mejoras
```

---

## Prioridades

1. **Alta**: Fase 2.1 (distribución de hashes) - ¿El IDF funciona bien?
2. **Alta**: Fase 1.2 (distribución de errores) - ¿Son errores sistemáticos?
3. **Media**: Fase 4.1 (ablation IDF) - Fácil de probar
4. **Media**: Fase 3.1 (alineación) - Puede revelar bugs
5. **Baja**: Fase 5 (mejoras) - Después de entender el problema

---

## Criterio de Éxito

Antes de volver a escalar, queremos:
- **Accuracy > 40%** en N=20 (actualmente 26.6%)
- **Recall@5 > 80%** (actualmente 64.2%)
- **Entender** por qué fallan las queries que fallan
