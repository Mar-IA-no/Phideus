# Proyecto Estado Actual - Phideus v5.0

**Actualizado**: 2026-02-06
**Estado**: 🔄 **BIAS_CONTROL Gate 3 (DANN) — Evaluación Comparativa** de 3 Runs

---

## Resumen Ejecutivo

### Estado de Hipótesis

| Hipótesis | Estado | Evidencia |
|-----------|--------|-----------|
| H1: Estructura de ratios | **VALIDADA** | Distribuciones no aleatorias |
| H2: Aprendibilidad | **VALIDADA** | VAE/HRM val_loss < 0.5 |
| H3: Cross-modality | 🟢 **PROMETEDOR** | BIAS_CONTROL: Gap 0.478, Hard neg acc 80.4% |

### Situación Actual (2026-02-06)

**BIAS_CONTROL Gate 3 (DANN)** — 3 Runs completados, evaluación comparativa en curso:

- **Gate 2 completado**: GO (Gap 0.478, Recall@10 34.4%, Hard neg acc 80.4%)
- **Run A (sin norm)**: Detenido ep10. Best ep7 (R@10 6.3%*, domain_acc 62.7%)
- **Run B (F.normalize)**: Completado ep10. Best ep6 (R@10 9.4%*, gap 0.482)
- **Run C (optimized, λ=0.8)**: Detenido ep27/30. Best ep4 (R@10 3.1%**, gap 0.469)
- **Evaluación comparativa**: En curso con pool estructurado (256 candidatos, 6 checkpoints)

*Pool ~3,200. **Pool ~13,536. No directamente comparables — evaluación homogénea en curso.

#### Diagnóstico Run C: λ_max=0.8 es excesivo

| Fase del training | Recall | Gap | Domain Acc | Lambda |
|-------------------|--------|-----|-----------|--------|
| Epochs 1-4 (warmup/ramp) | 2.5-3.1% | 0.40-0.47 | 50-70% | 0.0→0.3 |
| Epochs 8-27 (cap λ=0.8) | 1.9-2.8% | 0.32-0.41 | 53-72% | 0.80 |

**Conclusión**: Sobre-regularización adversarial. DANN destruye señal de retrieval sin lograr invariancia modal. Los mejores resultados son *anteriores* al cap de lambda.

#### Comparación de Runs (métricas de training, NO comparables directamente)

| Métrica | Gate 2 | Run A best | Run B best | Run C best |
|---------|--------|-----------|-----------|-----------|
| R@10 a2m | 2.6% | 6.3%* | 9.4%* | 3.1%** |
| Gap | 0.478 | 0.364 | 0.482 | 0.469 |
| Domain acc | 92.7% | 62.7% | 76.8% | 50.0% (ep1) |
| Normalización | - | No | Sí | Sí |
| Lambda schedule | - | linear 0→1 | linear 0→1 | warmup_ramp_cap 0.8 |

*200 val batches, pool ~3,200. **846 val batches, pool ~13,536.

#### Evaluación comparativa (en progreso)

`compare_gate3_checkpoints.py` ejecutando `evaluate_structured_pool.py` en 6 checkpoints con protocolo idéntico:
- Pool: 256 (64 hard + 32 semi-hard + 159 random + 1 positivo)
- 500 queries, seed 42
- Métricas: R@{1,5,10,20}, MRR, vs-random, hard neg accuracy

Resultados pendientes en: `data/bias_control_medium/evaluations/gate3_comparison/`

---

## 🟢 BIAS_CONTROL: Gate 2 COMPLETADO - GO

### Resultados Gate 2

**Checkpoint seleccionado**: `checkpoint_epoch45.pt`

#### Métricas de Pool Global (N=13,532)

| Dirección | R@1 | R@10 | vs Random |
|-----------|-----|------|-----------|
| Audio→MIDI | 0.8% | 2.5% | 34× |
| MIDI→Audio | 1.0% | 2.7% | 36× |

#### Métricas de Pool Estructurado (N=256, 500 queries)

| Dirección | R@1 | R@10 | MRR |
|-----------|-----|------|-----|
| Audio→MIDI | 4.4% | 34.4% | 0.138 |
| MIDI→Audio | 5.2% | 37.6% | 0.158 |

#### Hard Negative Analysis

| Test | Accuracy |
|------|----------|
| vs Same-Piece-Diff-Time | **80.4%** |
| vs Random | 87.0% |

**Interpretación**: 80.4% accuracy contra hard negatives (misma pieza, distinto tiempo) demuestra que el modelo aprende **identidad temporal**, no solo "firma de pieza".

### Diagnóstico Gate 2.5

| Probe | Resultado | Implicación |
|-------|-----------|-------------|
| Linear Separability (modal) | **92.7%** | Modal shortcut detectado |
| Silhouette (piece) | -0.111 | Pobre clustering por pieza |
| Dead Dimensions | 0/256 | Sin colapso |

**Recomendación**: Proceder a Gate 3 (DANN) para reducir separabilidad modal a ~50%.

---

## Pipeline de Gates BIAS_CONTROL

| Gate | Descripción | Estado | Resultado |
|------|-------------|--------|-----------|
| 0 | Data Integrity | ✅ Completado | GO |
| 1 | Intra-Modal Baselines | ✅ Completado | GO |
| **2** | **VICReg Training** | ✅ **Completado** | **GO** |
| **2.5** | **Embedding Analysis** | ✅ **Completado** | 92.7% separabilidad |
| **3** | **DANN Training** | 🔄 **Evaluando** | 3 Runs completados, comparación en curso |
| 4 | Ratio Auxiliary | ⏳ Pendiente | - |
| 5 | Curriculum (opcional) | ⏳ Pendiente | - |
| 6 | Retroanálisis | ⏳ Pendiente | Embeddings vs Representaciones |

### Próximos Pasos

1. 🔄 Analizar resultados de evaluación comparativa (6 checkpoints, pool estructurado)
2. ⏳ Decidir mejor checkpoint Gate 3 → GO/NO-GO
3. ⏳ Si necesario: Run D con F.normalize + λ_max=0.3-0.4
4. ⏳ Gate 4: Ratio Auxiliary View
5. ⏳ Gate 6: Retroanálisis embeddings vs representaciones de ratios

---

## 🟡 ESCALÓN 1: MAESTRO (Hashing) - PAUSADO

**Estado**: Pausado para priorizar BIAS_CONTROL

### Resumen

| Métrica | Valor | Status |
|---------|-------|--------|
| Piece Accuracy | 27% | ✗ Insuficiente |
| vs Random | 5.4× | ✓ Señal detectada |
| Causa raíz | Resolución temporal onset | Identificada |

El enfoque de hashing estilo Shazam alcanzó un límite de ~27% accuracy. BIAS_CONTROL ofrece mejor perspectiva.

---

## 🔴 REVISIONISMO UOEMD - COMPLETADO (NO-GO)

| Fase | Resultado |
|------|-----------|
| Fase 0: Tests sintéticos | ✓ GO |
| Fase 1: Extractor v2.2 | ✓ Gap 0.691 |
| Fase 2: Re-entrenamiento | ✗ Gap 0.007 |
| Fase 3A: Constellation tokens | ✗ Random level |

**Conclusión**: Dataset UOEMD (128 muestras) es insuficiente para validar H3.

---

## Archivos de Referencia

### BIAS_CONTROL

| Archivo | Descripción |
|---------|-------------|
| `Documents/BIAS_CONTROL/INFORME_GATE2_COMPLETO.md` | **Informe exhaustivo Gate 2** |
| `Documents/BIAS_CONTROL/INFORME_GATE3_DANN_SIN_NORM.md` | Informe Runs A/B |
| `Documents/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md` | Arquitectura y gates (v1.8) |
| `data/bias_control_medium/training_outputs/gate2/checkpoint_epoch45.pt` | Modelo Gate 2 |
| `data/bias_control_medium/evaluations/gate3_comparison/` | Evaluación comparativa Gate 3 |

### Scripts

| Script | Propósito |
|--------|-----------|
| `experiments/bias_control/gate2_foundation.py` | Training VICReg |
| `experiments/bias_control/gate3_dann.py` | Training DANN |
| `experiments/bias_control/evaluate_structured_pool.py` | Pool estructurado |
| `experiments/bias_control/compare_gate3_checkpoints.py` | **Comparación Gate 3** |
| `experiments/bias_control/gate2_5_embedding_analysis.py` | Análisis embeddings |

---

## Métricas Clave del Proyecto

### BIAS_CONTROL (mejor resultado actual)

```
┌────────────────────────────────────────────────────────────────┐
│        BIAS_CONTROL GATE 3 (DANN) - EVALUACIÓN                 │
├────────────────────────────────────────────────────────────────┤
│  Gate 2 baselines:                                             │
│    Gap: 0.478 | R@10 pool256: 34.4% | Hard neg: 80.4%        │
│    Domain probe: 92.7% (→ shortcut detectado)                  │
│                                                                 │
│  Gate 3 Run A (sin norm, ep10):                                │
│    Domain acc: 62.7% best | R@10: 6.3%* | Gap: 0.364          │
│                                                                 │
│  Gate 3 Run B (F.normalize, ep10):                             │
│    Domain acc: 76.8% | R@10: 9.4%* | Gap: 0.482              │
│                                                                 │
│  Gate 3 Run C (λ=0.8, ep27):                                  │
│    Domain acc: 53-72% | R@10: 3.1%** | Gap: 0.469→0.32       │
│    Diagnóstico: λ_max excesivo, sobre-regularización          │
│                                                                 │
│  *Pool 3.2K  **Pool 13.5K  → Evaluación homogénea en curso   │
│                                                                 │
│  Gate 2: GO | Gate 3: EVALUANDO                                │
└────────────────────────────────────────────────────────────────┘
```

---

*Documento actualizado: 2026-02-06 19:15 UTC (Gate 3 — 3 Runs completados, evaluación comparativa)*
