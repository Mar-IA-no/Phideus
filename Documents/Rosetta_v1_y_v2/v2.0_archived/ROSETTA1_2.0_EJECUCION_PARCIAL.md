# Rosetta1 2.0 - Ejecución Parcial (12.5% Dataset)

**Fecha**: 2026-01-28
**Estado**: INVÁLIDO - Ejecutado con datos insuficientes

---

## Error Crítico Identificado

El entrenamiento se ejecutó **sin el flag `--all-data`**, resultando en:

| Parámetro | Esperado | Ejecutado |
|-----------|----------|-----------|
| Archivos totales | 128 | 16 (solo HH) |
| % dataset usado | 100% | **12.5%** |
| Training samples | ~90 | **11** |
| Validation samples | ~19 | **2** |
| Test samples | ~19 | **3** |

## Resultados (No Concluyentes)

Con solo 11 archivos de entrenamiento, los resultados NO son válidos:

| Criterio | Resultado | Nota |
|----------|-----------|------|
| Cross-recon aligned vs shuffled | Δ = 0.002 | Insuficientes datos |
| Retrieval Top-1 | 0.78% (= random) | Insuficientes datos |
| var(z_private) | ~0 | Posible problema real |
| z_private diff | 1.21 ✅ | Único criterio pasado |

## Lección Aprendida

El flag `--all-data` es **obligatorio** para entrenamientos válidos. Sin él, el script filtra a solo datos "healthy" (HH = 16 archivos).

## Próximo Paso

Re-ejecutar Rosetta1 2.0 completo con `--all-data` para usar los 128 archivos disponibles.

---

*Este documento reemplaza el análisis anterior. Los resultados detallados en `ROSETTA1_2.0_RESULTADOS_EJECUCION.md` corresponden a esta ejecución parcial inválida.*
