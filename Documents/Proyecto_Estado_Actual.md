# Proyecto Estado Actual - Phideus v5.0

**Actualizado**: 2026-02-06 ~20:30 UTC
**Estado**: 🔄 **BIAS_CONTROL Gate 3 (DANN) — Run D en curso** (último experimento DANN)

---

## Resumen Ejecutivo

### Estado de Hipótesis

| Hipótesis | Estado | Evidencia |
|-----------|--------|-----------|
| H1: Estructura de ratios | **VALIDADA** | Distribuciones no aleatorias |
| H2: Aprendibilidad | **VALIDADA** | VAE/HRM val_loss < 0.5 |
| H3: Cross-modality | 🟢 **PROMETEDOR** | BIAS_CONTROL: Gap 0.478, Hard neg acc 80.4% |

### Situación Actual (2026-02-06)

**BIAS_CONTROL Gate 3 (DANN)** — Evaluación comparativa completada, Run D (último) en curso:

- **Gate 2**: GO (Gap 0.478, Recall@10 34.4%, Hard neg acc 80.4%)
- **Evaluación comparativa**: ✅ COMPLETADA (6 checkpoints, pool estructurado)
- **Resultado**: DANN no aporta mejora significativa sobre Gate 2
- **Run D** (λ_max=0.3): 🔄 EN CURSO — último experimento DANN (ETA ~03:30 UTC Feb 7)

#### Evaluación Comparativa — Resultado Definitivo

| Checkpoint | R@10 a2m | R@10 m2a | Hard Neg | MRR a2m | Decision |
|-----------|---------|---------|----------|---------|----------|
| **gate2_ep45** | **34.4%** | 37.6% | **80.4%** | 0.138 | **GO** |
| runA_best_ep7 | 27.8% | 35.4% | 74.8% | 0.132 | GO |
| runB_ep5 | 24.6% | 32.0% | 70.4% | 0.112 | WEAK-GO |
| runB_ep10 | 29.8% | 34.6% | 73.6% | 0.130 | GO |
| **runC_best_ep4** | **34.6%** | **39.2%** | **81.2%** | **0.148** | **GO** |
| runC_ep13 | 32.2% | 38.0% | 76.6% | 0.144 | GO |

**Hallazgos clave**:
- Gate 2 (sin DANN) sigue siendo el mejor o empata con Run C ep4
- Run C ep4 (λ~0.3 transitorio) es el único checkpoint DANN competitivo
- λ alto (0.8) destruye retrieval sin lograr invariancia modal
- Las métricas de training eran engañosas (pools de tamaño diferente)

#### Run D — Último Experimento DANN

| Parámetro | Valor |
|-----------|-------|
| λ_max | **0.3** (sostenido, no transitorio) |
| Warmup | 1000 steps |
| Ramp | 3000 steps |
| Epochs | 15 |
| F.normalize | Sí |
| LR | midi=1e-4, proj=5e-4, domain=2e-4 |
| tmux | `gate3d` |
| ETA | ~03:30 UTC Feb 7 |

**Hipótesis**: Run C ep4 estaba en λ~0.3 *transitoriamente* cuando igualó a Gate 2. Run D mantiene λ=0.3 como cap sostenido para determinar si este régimen aporta algo.

**Decisión post-Run D**:
- Si Run D ≈ Gate 2 → DANN cerrado, avanzar a Gate 4
- Si Run D > Gate 2 → usar como checkpoint Gate 3

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

---

## Pipeline de Gates BIAS_CONTROL

| Gate | Descripción | Estado | Resultado |
|------|-------------|--------|-----------|
| 0 | Data Integrity | ✅ Completado | GO |
| 1 | Intra-Modal Baselines | ✅ Completado | GO |
| **2** | **VICReg Training** | ✅ **Completado** | **GO** |
| **2.5** | **Embedding Analysis** | ✅ **Completado** | 92.7% separabilidad |
| **3** | **DANN Training** | 🔄 **Run D en curso** | Runs A/B/C: DANN ≈ Gate 2 |
| 4 | Ratio Auxiliary | ⏳ Pendiente | - |
| 5 | Curriculum (opcional) | ⏳ Pendiente | - |
| 6 | Retroanálisis | ⏳ Pendiente | Embeddings vs Representaciones |

### Próximos Pasos

1. 🔄 Esperar Run D (~03:30 UTC Feb 7), evaluar con structured pool
2. ⏳ Cerrar Gate 3 definitivamente
3. ⏳ Gate 4: Ratio Auxiliary View (siguiente prioridad)
4. ⏳ Gate 6: Retroanálisis embeddings vs representaciones de ratios

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
| `Documents/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md` | Arquitectura y gates (v1.9) |
| `Documents/BIAS_CONTROL/Gate3_DANN_Results/` | **Evaluación comparativa completa (6 checkpoints)** |
| `data/bias_control_medium/training_outputs/gate2/checkpoint_epoch45.pt` | Modelo Gate 2 |

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
│        BIAS_CONTROL GATE 3 (DANN) - RESULTADOS                 │
├────────────────────────────────────────────────────────────────┤
│  Gate 2 baseline (mejor checkpoint actual):                    │
│    Gap: 0.478 | R@10 pool256: 34.4% | Hard neg: 80.4%        │
│    Domain probe: 92.7% (shortcut detectado)                    │
│                                                                 │
│  Evaluación comparativa (6 checkpoints, pool estructurado):    │
│    Gate 2 ≈ Run C ep4 (λ~0.3) >> todos los demás DANN        │
│    DANN no aporta mejora significativa en ningún régimen       │
│                                                                 │
│  Run D (λ_max=0.3 sostenido): EN CURSO                        │
│    Último experimento DANN. ETA ~03:30 UTC Feb 7               │
│    Si ≈ Gate 2 → DANN cerrado → Gate 4                        │
│                                                                 │
│  Gate 2: GO | Gate 3: EVALUANDO (Run D)                        │
└────────────────────────────────────────────────────────────────┘
```

---

*Documento actualizado: 2026-02-06 ~20:30 UTC (Gate 3 — Evaluación completada, Run D en curso)*
