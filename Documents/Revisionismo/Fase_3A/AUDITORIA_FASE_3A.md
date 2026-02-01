# Auditoría Exhaustiva - Fase 3A Ratio Constellations

**Fecha**: 2026-02-01
**Ejecutada por**: Claude Code

---

## Resumen Ejecutivo

La auditoría identificó **dos causas raíz** que explican por qué todos los modelos de Fase 3A obtuvieron Top-1 ≈ 0.78% (random):

### Causa 1: Datos No Discriminativos (Checkpoint 1)

Los tokens constellation **NO son discriminativos** pre-red:

| Métrica | Valor | Umbral | Status |
|---------|-------|--------|--------|
| Gap intra-condición vs inter-condición | **-0.0019** | > 0.05 | **FAIL** |
| Gap aligned vs shuffled (audio-vib) | **0.0369** | > 0.05 | **FAIL** |

**Interpretación**: Los histogramas de log-ratios de diferentes archivos son prácticamente indistinguibles. El extractor de constellations no está produciendo representaciones discriminativas.

### Causa 2: Configuración Incorrecta de Training (Checkpoint 5)

Los modelos se entrenaron con configuración subóptima:

| Parámetro | Valor usado | Valor recomendado |
|-----------|-------------|-------------------|
| `dropout_shared` | **0** | 0.5 |
| `beta_kl_private` | **None** | 0.01 |

Esto contribuyó al **colapso de embeddings** detectado en Checkpoint 3 (varianza = 0.000418, muy baja).

---

## Resultados por Checkpoint

### Checkpoint 4: Métricas de Evaluación ✓ PASS

Las funciones de cálculo de métricas funcionan correctamente:
- Embeddings idénticos → Top-1 = 100%
- Embeddings shuffled → Top-1 ≈ 0.78% (random)
- Embeddings con ruido → Top-1 decrece monotónicamente

**Conclusión**: Los resultados NO-GO no son un artefacto de métricas rotas.

---

### Checkpoint 1: Integridad de Datos ✗ FAIL

**Estadísticas del dataset**:
- 128 archivos, 52,096 frames
- 79.8% de frames con <10 tokens válidos (muy sparse)
- log_ratio: [0.0, 2.585], delta_t: [-5.0, 5.0]

**Tests de discriminabilidad**:

| Test | Resultado | Status |
|------|-----------|--------|
| Shapes y rangos | OK | ✓ PASS |
| Correspondencia audio-vib | OK | ✓ PASS |
| **Discriminabilidad por condición** | Gap = -0.0019 | ✗ FAIL |
| **Discriminabilidad audio-vib** | Gap = 0.0369 | ✗ FAIL |

**Conclusión**: El problema está en el extractor de constellations, no en el modelo.

---

### Checkpoint 2: Collate/DataLoader ✓ PASS

El DataLoader mantiene correctamente la correspondencia audio-vibración:
- Pairing preservado entre accesos
- Orden preservado después de collate
- No hay mezcla de pares

**Conclusión**: El bug no está en el DataLoader.

---

### Checkpoint 3: Forward Pass ✗ FAIL (parcial)

| Test | Resultado | Status |
|------|-----------|--------|
| Mismo input → mismo output | diff = 0 | ✓ PASS |
| Diferente input → diferente output | cosine_sim = 0.978 | ✓ PASS (marginal) |
| **Varianza de embeddings** | **0.000418** | ✗ FAIL |

**Hallazgo crítico**: Los embeddings tienen varianza muy baja (0.000418 << 0.01), indicando **colapso de representación**. Todos los inputs producen embeddings muy similares.

---

### Checkpoint 5: Training ✗ FAIL

**Problemas de configuración**:
1. `dropout_shared = 0` (debería ser 0.5)
2. `beta_kl_private = None` (debería ser 0.01)

Estos parámetros son críticos para prevenir el colapso de embeddings. Sin regularización, el modelo converge a una solución trivial.

---

### Checkpoint 6: End-to-End ✓ PASS

El pipeline funciona correctamente con datos sintéticos:

| Test | Top-1 | Esperado | Status |
|------|-------|----------|--------|
| Datos idénticos | 100% | 100% | ✓ PASS |
| Datos shuffled | 1.6% | ~0.78% | ✓ PASS |
| Datos correlacionados | 100% | >50% | ✓ PASS |

**Conclusión**: El código de training/evaluation es correcto. El problema está en los datos reales.

---

## Diagnóstico Final

### Cadena Causal del Fallo

```
1. Extractor de constellations → Tokens NO discriminativos (gap ≈ 0)
                                    ↓
2. Modelo recibe datos sin información útil
                                    ↓
3. Configuración sin regularización (dropout=0, beta_kl=None)
                                    ↓
4. Modelo colapsa a solución trivial (varianza embeddings ≈ 0)
                                    ↓
5. Retrieval = random (Top-1 = 0.78%)
```

### Prioridad de Corrección

1. **CRÍTICO**: Mejorar el extractor de constellations para producir representaciones discriminativas
2. **IMPORTANTE**: Usar `dropout_shared=0.5` y `beta_kl_private=0.01` en training
3. **OPCIONAL**: Verificar con datos sintéticos discriminativos antes de re-entrenar

---

## Acciones Recomendadas

### Opción A: Mejorar Extractor (Fase 3A-1 bis)

El extractor actual produce histogramas de log-ratios casi idénticos entre archivos. Posibles mejoras:

1. **Aumentar resolución de bandas**: Actualmente anchor_band/target_band solo van de 2-7 (6 valores)
2. **Usar más tokens por frame**: 79.8% de frames tienen <10 tokens (muy sparse)
3. **Normalizar de forma diferente**: Los weights van de 31-4160, rango muy amplio
4. **Cambiar métricas de distancia**: Coseno puede no capturar diferencias relevantes

### Opción B: Cambiar Representación Completamente

Abandonar constellations y probar:
- Spectrograms mel-scale directos
- Learned embeddings (wav2vec, etc.)
- Otras representaciones que han funcionado en cross-modal

### Opción C: Re-entrenar con Config Correcta

Antes de cambiar el extractor, verificar que el problema no es solo la configuración:

```bash
python experiments/run_roseta_experiment.py \
    --data data/datasets/roseta_constellation.npz \
    --model jepa-lite --encoder-type mlp \
    --dropout-shared 0.5 \
    --beta-kl-private 0.01 \
    --epochs 100
```

Si sigue fallando con config correcta → confirma que el problema está en los datos.

---

## Archivos de Auditoría

```
experiments/audits/
├── checkpoint_1_data_integrity.py    # Verifica datos ✗ FAIL
├── checkpoint_2_collate_verification.py  # Verifica DataLoader ✓ PASS
├── checkpoint_3_forward_pass.py      # Verifica modelo ✗ FAIL (colapso)
├── checkpoint_4_metrics.py           # Verifica métricas ✓ PASS
├── checkpoint_5_training.py          # Verifica training ✗ FAIL (config)
├── checkpoint_6_e2e.py               # Verifica pipeline ✓ PASS
└── run_all_audits.py                 # Ejecuta todos
```

---

## Conclusión

Los resultados NO-GO de Fase 3A **NO son genuinos** - son el resultado de:

1. **Datos no discriminativos** producidos por el extractor de constellations
2. **Configuración incorrecta** de training (sin regularización)

Antes de declarar H3 como "no validada", se debe:
1. Corregir el extractor para producir representaciones discriminativas
2. Re-entrenar con configuración correcta
3. Re-evaluar los resultados

---

*Generado por experimentos/audits/run_all_audits.py*
*Proyecto Phideus v5.0 - Febrero 2026*
