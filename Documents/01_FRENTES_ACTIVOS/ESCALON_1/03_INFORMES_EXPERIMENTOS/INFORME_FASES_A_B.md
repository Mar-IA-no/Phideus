# Informe Fases A-B: Auditoría y Replicación

**Fecha**: 2026-02-04
**Estado**: ✅ Fases A y B completadas

---

## Resumen Ejecutivo

Las Fases A (Auditoría) y B (Replicación) revelaron que:

1. **Los resultados originales estaban inflados** por un bug en el protocolo
2. **Las métricas reales son más modestas** pero consistentes
3. **Se requieren mejoras** antes de validar a escala

---

## Fase A: Auditoría

### Bug Crítico Encontrado

El script `test_retrieval_routes.py` usaba `t.dt` (delta time) en lugar de `t.t_anchor` (tiempo absoluto del anchor).

**Impacto**:
- Antes: 7-10 queries generadas → 71-80% accuracy (inflado)
- Después: 1177 queries generadas → 32-42% accuracy (real)

### Correcciones Aplicadas

1. Añadido `t_anchor` a `EventToken` y `TFToken`
2. Actualizado script para usar tiempo absoluto correcto
3. Eliminado límite de 10 piezas en el script

### Resultados Corregidos (N=10)

| Route | Accuracy (Bug) | Accuracy (Corregido) | Queries |
|-------|----------------|---------------------|---------|
| A | 71.4% | **42.5%** | 1177 |
| B | 80.0% | **32.9%** | 1175 |

---

## Fase B: Replicación

### Metodología

- 20 piezas nuevas seleccionadas (diferentes a las 10 originales)
- Años diversos: 2004, 2008, 2009, 2011, 2013, 2014, 2015, 2017, 2018
- Seed fijo para reproducibilidad (seed=42)

### Resultados (N=20)

| Route | Accuracy | vs Random | Queries |
|-------|----------|-----------|---------|
| A | **26.6%** | 5.3x | 2357 |
| B | **21.4%** | 4.3x | 2361 |

### Comparación con Muestra Original

| Métrica | Original (N=10) | Replicación (N=20) |
|---------|-----------------|-------------------|
| Route A | 42.5% (4.2x) | 26.6% (5.3x) |
| Route B | 32.9% (3.3x) | 21.4% (4.3x) |
| Ganador | Route A | Route A |

### Conclusiones de Replicación

- ✅ **Route A consistentemente mejor que Route B** (confirmado)
- ✅ **Resultados replicables** entre muestras independientes
- ✅ **Significativamente mejor que random** (4-5x)
- ⚠️ **Accuracy baja con más piezas** (esperado pero preocupante)

---

## Análisis de Métricas

### ¿Son suficientes estas métricas?

| Contexto | Accuracy | Evaluación |
|----------|----------|------------|
| Random (N=10) | 10% | Baseline |
| Random (N=20) | 5% | Baseline |
| Route A (N=10) | 42.5% | Prometedor |
| Route A (N=20) | 26.6% | Insuficiente |
| **Shazam real** | **>95%** | **Objetivo** |

### Proyección a Escala

| N | Random | Proyección Route A | Evaluación |
|---|--------|-------------------|------------|
| 10 | 10% | 42.5% | 4.2x |
| 20 | 5% | 26.6% | 5.3x |
| 100 | 1% | ~15%? | ~15x? |
| 1000 | 0.1% | ~5%? | ~50x? |

El factor de mejora sobre random aumenta, pero la accuracy absoluta es baja.

### Decisión: NO escalar todavía

**Razón**: Con 26% accuracy en N=20, escalar solo confirmaría que "funciona parcialmente". Es más productivo mejorar primero.

---

## Estado de Hipótesis

| Hipótesis | Estado | Evidencia |
|-----------|--------|-----------|
| H1: Estructura | ✓ VALIDADA | Distribuciones no aleatorias |
| H2: Aprendibilidad | ✓ VALIDADA | VAE val_loss < 0.5 |
| H3: Cross-modality | 🟡 **PARCIAL** | 26% accuracy (5x random) |

**H3 muestra señal pero no validación convincente.**

---

## Archivos Modificados

### Extractores (bug fix)
- `src/extractors/event_based_extractor.py` - Añadido t_anchor
- `src/extractors/improved_tf_extractor.py` - Añadido t_anchor

### Scripts
- `experiments/un_audio_un_midi/test_retrieval_routes.py` - Corregido

### Datos
- `experiments/un_audio_un_midi/Varios_pares/` - 10 pares originales
- `experiments/un_audio_un_midi/muestra_replicacion/` - 20 pares nuevos

### Documentación
- `Documents/01_FRENTES_ACTIVOS/ESCALON_1/AUDITORIA_FASE_A.md`
- `Documents/01_FRENTES_ACTIVOS/ESCALON_1/INFORME_FASES_A_B.md` (este archivo)

---

## Próximo Paso: Análisis de Errores

Antes de escalar, se requiere análisis de errores para identificar mejoras.

Ver: `Documents/01_FRENTES_ACTIVOS/ESCALON_1/PLAN_ANALISIS_ERRORES.md`

---

## Commits Relacionados

1. `32c7913` - Fase A: Auditoría revela bug crítico
2. `16670df` - Fase B: Replicación con N=20 confirma resultados
