# Proyecto Estado Actual - Phideus v5.0

**Actualizado**: 2026-02-04
**Estado**: Escalón 1 MAESTRO implementado - Pendiente: Descargar datos y ejecutar

---

## Resumen Ejecutivo

### Estado de Hipótesis

| Hipótesis | Estado | Evidencia |
|-----------|--------|-----------|
| H1: Estructura de ratios | **VALIDADA** | Distribuciones no aleatorias |
| H2: Aprendibilidad | **VALIDADA** | VAE/HRM val_loss < 0.5 |
| H3: Cross-modality | **PENDIENTE** | UOEMD NO-GO, **MAESTRO pendiente** |

### Situación Actual (2026-02-04)

**Revisionismo UOEMD completado con resultado NO-GO**. El dataset UOEMD (128 muestras, motor diésel) no demostró cross-modality ni con histogramas densos ni con constellation tokens.

**NUEVO**: Escalón 1 MAESTRO implementado completamente (6 Gates). El dataset MAESTRO (200h de piano, audio+MIDI alineados ~3ms) ofrece una prueba más robusta de la hipótesis H3.

---

## 🎹 ESCALÓN 1: MAESTRO (Audio ↔ MIDI)

### Objetivo

Demostrar cross-modal learning entre Audio real y MIDI usando ratio constellations, con el dataset MAESTRO v3.0.0 (120GB, ~200h de piano).

### Arquitectura de 6 Gates

| Gate | Descripción | Criterio GO |
|------|-------------|-------------|
| **0** | Harness + controles negativos | Oracle > 90%, random ~ 1/N |
| **1** | Ingesta MAESTRO | Corr energia-densidad > 0.7 |
| **2** | Baselines (chroma, CCA) | Piece Top-1 > 10× random |
| **3** | VICReg/Barlow dense | No colapso + Top-1 > baselines |
| **4** | Ratio tokens (Phideus Test) | Matching > random |
| **5** | MoCo + hard negatives | Mejora NEG-SAME-COMPOSER |

### Archivos Implementados

```
experiments/maestro/
├── gate0_harness.py          ✅ Métricas + controles negativos
├── gate1_ingest.py           ✅ Descarga + segmentación
├── gate2_baselines.py        ✅ Chroma + CCA baselines
├── gate3_cross_modal.py      ✅ Training VICReg/Barlow
├── gate4_ratio_tokens.py     ✅ Training constellation + baseline
├── gate5_moco.py             ✅ MoCo queue + hard negatives
└── run_maestro_experiment.py ✅ Script orquestador

src/
├── utils/midi_utils.py       ✅ Parseo MIDI, piano roll, tokens
├── RNA/vicreg.py             ✅ VICReg loss + encoder
├── RNA/barlow_twins.py       ✅ Barlow Twins loss + encoder
├── analizador/analizador_maestro.py  ✅ Extracción constellation
└── datasets/maestro_dataset.py       ✅ DataLoader MAESTRO
```

### Estado de Auditoría

- ✅ Todos los 6 Gates implementados
- ✅ Corrección de max_tokens (64 vs 48) aplicada
- ⚠️ Dependencias externas pendientes (`pretty_midi`, `mido`)
- ⏳ Pendiente: Descargar MAESTRO y ejecutar

### Próximo Paso

```bash
# 1. Instalar dependencias
pip install pretty_midi mido

# 2. Descargar MAESTRO (101GB)
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip

# 3. Ejecutar pipeline completo
python experiments/maestro/run_maestro_experiment.py --mode full
```

---

## Historial: Revisionismo UOEMD (Completado)

### Fase 2: Re-entrenamiento con Extractor v2.2

| Criterio | Umbral | Resultado | Estado |
|----------|--------|-----------|--------|
| **Gap aligned-shuffled** | **> 0.15** | **0.007** | **FAIL (CRÍTICO)** |
| Retrieval Top-1 | > 10× random | 10.94% vs 0.78% (14×) | PASS |
| Silhouette score | > 0.3 | -0.14 | FAIL |

**Decisión: NO-GO** - El modelo colapsa la información discriminativa del extractor.

### Fase 3A: Ratio Constellations

| Config | Encoder | Decoder | Top-1 | Status |
|--------|---------|---------|-------|--------|
| C1-C4 | MLP/Transformer | Histogram/Token | 0.78% | FAIL |
| **C5** | MLP | JEPA-lite | **1.56%** | FAIL |
| C6 | Transformer | JEPA-lite | 0.78% | FAIL |

**Decisión: NO-GO** - Tokens sparse también fallan en UOEMD.

### Conclusión UOEMD

El problema podría ser:
1. **Dataset insuficiente**: Solo 128 muestras de motor diésel
2. **Dominio difícil**: Audio-vibración de maquinaria tiene menos estructura armónica
3. **Representación**: Ni histogramas ni tokens capturan la correspondencia

**Solución**: Probar con MAESTRO (200h, piano, audio-MIDI alineados).

---

## Logros del Proyecto

1. **H1 VALIDADA**: Las señales contienen distribuciones de ratios estructuradas
2. **H2 VALIDADA**: Redes neuronales pueden aprenderlas (val_loss < 0.5)
3. **Analizador 5.0**: Demostró que representación > arquitectura
4. **VAE Rehabilitado**: De val_loss 4212 → 0.456
5. **Metodología robusta**: Controles negativos que detectaron el problema
6. **6 Gates MAESTRO**: Sistema completo para prueba rigurosa de H3

---

## Documentación

### Escalón 1 (MAESTRO)

| Documento | Descripción |
|-----------|-------------|
| `Documents/ESCALON_1/Plan_implementacion.md` | Plan original |
| `Documents/ESCALON_1/AUDITORIA_IMPLEMENTACION.md` | Auditoría y correcciones |

### Revisionismo UOEMD

| Fase | Documento | Estado |
|------|-----------|--------|
| Fase 0 | `Documents/UOEMD/UOEMD_Revisionismo/Fase_0/` | ✅ Completada |
| Fase 1 | `Documents/UOEMD/UOEMD_Revisionismo/Fase_1/` | ✅ GO |
| Fase 2 | `Documents/UOEMD/UOEMD_Revisionismo/Fase_2/` | ✅ NO-GO |
| Fase 3A | `Documents/UOEMD/UOEMD_Revisionismo/Fase_3A/` | ✅ NO-GO |

---

*Última actualización: 2026-02-04 - Escalón 1 MAESTRO implementado, pendiente ejecución*
