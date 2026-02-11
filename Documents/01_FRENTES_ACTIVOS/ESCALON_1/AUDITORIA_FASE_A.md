# Auditoría Fase A: Experimento Piloto (N=10)

**Fecha**: 2026-02-04
**Estado**: ✅ **COMPLETADA - BUG CRÍTICO CORREGIDO**

---

## Resumen de Auditoría

| Checkpoint | Estado | Detalle |
|------------|--------|---------|
| A.1: Extractores | ✓ OK | Código bien estructurado |
| A.2: Alineación | ⚠️ No testeada | Falta test explícito |
| A.3: Protocolo | ⚠️ **BUG ENCONTRADO** | t_abs usa dt incorrecto |
| A.4: Reproducibilidad | ✓ OK | Resultados idénticos |

---

## A.1: Verificación de Extractores

### Route A (`event_based_extractor.py`)
- ✓ Extracción de eventos desde MIDI correcta
- ✓ Extracción de eventos desde Audio via CQT + onset detection
- ✓ Generación de tokens (chord, sequential, constellation)
- ✓ Función de hash bien documentada (20 bits)

### Route B (`improved_tf_extractor.py`)
- ✓ Onset detection via spectral flux
- ✓ Harmonic folding (pitch class)
- ✓ Generación de tokens mejorados
- ✓ Función de hash con octave-invariance

---

## A.2: Verificación de Alineación

**Estado**: No hay test explícito de alineación temporal.

La función `align_audio_events_to_midi()` existe pero no se verifica que:
- Los onsets de audio correspondan a los de MIDI
- El offset global sea correcto

**Recomendación**: Añadir visualización de alineación.

---

## A.3: Verificación del Protocolo de Evaluación

### ⚠️ BUG ENCONTRADO: Tiempo Absoluto Incorrecto

**Ubicación**: `test_retrieval_routes.py`, líneas 137-138 y 161

**Problema**:
```python
# Route A (línea 137-138):
t_abs = t.dt  # This is dt, not ideal but works for testing

# Route B (línea 161):
audio_hashes = [(tf_token_to_hash(t, use_folded=True), t.dt) for t in audio_tokens]
```

**Impacto**:
- `t_abs` debería ser el tiempo absoluto del anchor en frames
- Se usa `dt` (delta time entre anchor y target) en su lugar
- Esto causa que la segmentación de queries sea incorrecta
- El offset voting de Shazam no funciona como debería

**Por qué funciona parcialmente**:
- Los hashes coinciden independientemente del tiempo
- El matching directo (sin offset voting real) sigue funcionando
- Con solo 10 piezas, la probabilidad de colisión es baja

### Número de Queries Generadas

| Route | Queries | Esperado (~) |
|-------|---------|--------------|
| A | 7 | 50-100 |
| B | 10 | 50-100 |

Las queries son muy pocas debido al bug de segmentación.

---

## A.4: Verificación de Reproducibilidad

**Resultado**: ✓ PASS

Ejecución 1 (original):
- Route A: 71.4% (5/7)
- Route B: 80.0% (8/10)

Ejecución 2 (auditoría):
- Route A: 71.4% (5/7)
- Route B: 80.0% (8/10)

Los resultados son **100% reproducibles**.

---

## Conclusiones de la Auditoría

### Hallazgos Críticos

1. **Bug de tiempo absoluto**: El protocolo usa `dt` en lugar del tiempo real del anchor
2. **Pocas queries**: Solo 7-10 queries en lugar de 50-100 esperadas
3. **Sin alineación verificada**: No hay test de alineación audio-MIDI

### Por qué los resultados son prometedores a pesar de los bugs

1. El matching de hashes funciona por coincidencia directa (set intersection)
2. Con N=10 piezas pequeño, hay poca ambigüedad
3. Los hashes capturan información discriminativa real

### Impacto en la Validez

| Aspecto | Impacto |
|---------|---------|
| Precisión de los % | ⚠️ Cuestionable (pocas queries) |
| Tendencia general | ✓ Probablemente correcta |
| Comparación A vs B | ✓ Válida (mismo bug en ambas) |
| Validación de H3 | ⚠️ Insuficiente |

---

## Recomendaciones

### Corrección Inmediata (para Fase B)

1. Modificar tokens para incluir tiempo absoluto del anchor
2. Corregir generación de queries en `test_retrieval_routes.py`
3. Añadir test de alineación

### Para Validación Rigurosa (Fase C)

1. Usar protocolo Shazam completo con offset voting
2. Negativos duros (NEG_SAME_COMPOSER)
3. Bootstrap CI para intervalos de confianza

---

---

## CORRECCIÓN APLICADA Y RESULTADOS REALES

### Bug Corregido

Se añadió `t_anchor` (tiempo absoluto del anchor) a los dataclasses:
- `EventToken` en `event_based_extractor.py`
- `TFToken` en `improved_tf_extractor.py`

Se actualizó `test_retrieval_routes.py` para usar `t.t_anchor` en lugar de `t.dt`.

### Resultados ANTES vs DESPUÉS de la corrección

| Métrica | Antes (Bug) | Después (Corregido) |
|---------|-------------|---------------------|
| **Queries Route A** | 7 | **1177** |
| **Queries Route B** | 10 | **1175** |
| Route A Accuracy | 71.4% | **42.5%** |
| Route B Accuracy | 80.0% | **32.9%** |
| Route A Recall@5 | 100% | **83.3%** |
| Route B Recall@5 | 100% | **78.3%** |
| **Ganador** | Route B | **Route A** |

### Interpretación

1. **Los resultados anteriores estaban inflados**: Con 7-10 queries, la varianza era enorme
2. **Route A es mejor que Route B**: Contrario a lo que parecía
3. **Aún mejor que random**: 4.2x (Route A) y 3.3x (Route B)
4. **No alcanza umbral GO (>50%)**: Pero es prometedor

### Conclusión Actualizada

El experimento muestra que el cross-modal **funciona parcialmente**:
- **42.5% accuracy** (vs 10% random) es significativo
- No alcanza el 50% necesario para GO definitivo
- Necesita validación a escala para confirmar

---

## Archivos Relevantes

- `src/extractors/event_based_extractor.py` - Route A (modificado: +t_anchor)
- `src/extractors/improved_tf_extractor.py` - Route B (modificado: +t_anchor)
- `experiments/un_audio_un_midi/test_retrieval_routes.py` - Script evaluación (corregido)
