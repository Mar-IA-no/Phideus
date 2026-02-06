# Proyecto Estado Actual - Phideus v5.0

**Actualizado**: 2026-02-05
**Estado**: 🔄 **BIAS_CONTROL Gate 3 (DANN) EN EJECUCIÓN** - Training 30 epochs

---

## Resumen Ejecutivo

### Estado de Hipótesis

| Hipótesis | Estado | Evidencia |
|-----------|--------|-----------|
| H1: Estructura de ratios | **VALIDADA** | Distribuciones no aleatorias |
| H2: Aprendibilidad | **VALIDADA** | VAE/HRM val_loss < 0.5 |
| H3: Cross-modality | 🟢 **PROMETEDOR** | BIAS_CONTROL: Gap 0.478, Hard neg acc 80.4% |

### Situación Actual (2026-02-05)

**BIAS_CONTROL Gate 3 (DANN)** en ejecución — epoch 8/30:

- **Gate 2 completado**: GO (Gap 0.478, Recall@10 34.4%, Hard neg acc 80.4%)
- **Gate 3 smoke test**: GO (métricas sin degradación, script validado)
- **Gate 3 training**: Epoch 8/30, **nuevo best en epoch 7** (domain_acc 62.7%)

#### Progreso Training Gate 3 (epochs 1-7 completados)

| Epoch | Loss | Domain Acc | R@10 (a2m) | Gap | Lambda |
|-------|------|-----------|------------|-----|--------|
| 1 | 14.108 | 67.6% | 6.2% | 0.387 | 0.03 |
| 2 | 14.082 | 74.0% | 5.5% | 0.335 | 0.07 |
| 3 | 14.069 | 77.4% | 6.6% | 0.398 | 0.10 |
| 4 | 14.048 | 65.0% | 5.2% | 0.378 | 0.13 |
| 5 | 14.031 | 65.2% | 6.1% | 0.367 | 0.17 |
| 6 | 14.025 | 65.8% | 6.8% | 0.386 | 0.20 |
| **7** | **13.992** | **62.7%** | **6.3%** | **0.364** | **0.23** |

**Tendencia**: Domain accuracy bajando de 77.4% → 62.7% (objetivo ~50%). R@10 estable 5-7% (muy por encima del baseline Gate 2 de 2.6%). Loss convergiendo. **Nuevo best guardado en epoch 7** (recall=0.073, domain_acc=62.7%).

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

### Auditoría Gate 2

| Check | Resultado |
|-------|-----------|
| A1: Dataset Structure | ✅ PASS |
| A2: Alignment | ❌ FAIL (método impreciso*) |
| A3: Checkpoint | ✅ PASS |
| B1: Model Loading | ✅ PASS |
| B2: Dimensions | ✅ PASS |
| B3: No Collapse | ✅ PASS |
| C1: Pool Global | ✅ PASS |
| C2: Pool Structured | ✅ PASS |
| D1: Shuffled Pairs | ❌ FAIL (esperado*) |
| D2: Oracle MIDI | ✅ PASS |

*Los 2 "FAIL" son falsos positivos explicados en el informe completo.

**Documentación**: `Documents/BIAS_CONTROL/INFORME_GATE2_COMPLETO.md`

---

## Pipeline de Gates BIAS_CONTROL

| Gate | Descripción | Estado | Resultado |
|------|-------------|--------|-----------|
| 0 | Data Integrity | ✅ Completado | GO |
| 1 | Intra-Modal Baselines | ✅ Completado | GO |
| **2** | **VICReg Training** | ✅ **Completado** | **GO** |
| **2.5** | **Embedding Analysis** | ✅ **Completado** | 92.7% separabilidad |
| **3** | **DANN Training** | 🔄 **Epoch 8/30** | Domain acc 62.7%, best epoch 7 |
| 4 | Ratio Auxiliary | ⏳ Pendiente | - |
| 5 | Curriculum (opcional) | ⏳ Pendiente | - |
| 6 | Retroanálisis | ⏳ Pendiente | Embeddings vs Representaciones |

### Gate 3: DANN Training

**Smoke test (piloto)**: GO - 1 epoch, 5 batches, métricas sin degradación.

**Training completo en progreso**:
```bash
tmux attach -t gate3  # Monitorear

python experiments/bias_control/gate3_dann.py \
    --checkpoint data/bias_control_medium/training_outputs/gate2/checkpoint_epoch45.pt \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/bias_control_medium/training_outputs/gate3 \
    --epochs 30 --batch-size 16 --segment-len 4.0 --hop 1.0 \
    --max-batches-per-epoch 1000 --max-val-batches 200 \
    --checkpoint-every 5 --dann-weight 0.01 --gate2-recall 0.026
```

### Próximos Pasos

1. 🔄 Completar Gate 3 training (~22 epochs restantes, ~10h)
2. ⏳ Evaluar: Domain accuracy → ~50% (actualmente 62.7%, tendencia bajando)
3. ⏳ Pool estructurado post-DANN con `evaluate_structured_pool.py`
4. ⏳ Decidir GO/NO-GO para Gate 4
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
| `Documents/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md` | Arquitectura y gates |
| `data/bias_control_medium/training_outputs/gate2/checkpoint_epoch45.pt` | Modelo seleccionado |
| `data/bias_control_medium/evaluations/structured_pool_epoch45.json` | Métricas pool estructurado |
| `data/bias_control_medium/evaluations/gate2_5/gate2_5_results.json` | Análisis de embeddings |
| `data/bias_control_medium/evaluations/audit_gate2/audit_gate2_results.json` | Auditoría completa |

### Scripts

| Script | Propósito |
|--------|-----------|
| `experiments/bias_control/gate2_foundation.py` | Training VICReg |
| `experiments/bias_control/gate3_dann.py` | Training DANN |
| `experiments/bias_control/evaluate_structured_pool.py` | Pool estructurado |
| `experiments/bias_control/gate2_5_embedding_analysis.py` | Análisis embeddings |
| `experiments/bias_control/audit_gate2_complete.py` | Auditoría |

---

## Métricas Clave del Proyecto

### BIAS_CONTROL (mejor resultado actual)

```
┌────────────────────────────────────────────────────────────────┐
│              BIAS_CONTROL GATE 3 (DANN) - LIVE                │
├────────────────────────────────────────────────────────────────┤
│  Gate 2 baselines:                                             │
│    Gap: 0.478 | R@10 pool256: 34.4% | Hard neg: 80.4%        │
│    Domain probe: 92.7% (→ shortcut detectado)                  │
│                                                                 │
│  Gate 3 training (epoch 7/30 best):                            │
│    Domain acc: 62.7%  (↓ desde 77.4%, objetivo ~50%)          │
│    R@10 (global): 6.3% (↑ vs Gate 2 baseline 2.6%)           │
│    Loss: 13.992 (convergiendo)                                 │
│    Lambda DANN: 0.23 (schedule 0→1)                            │
│                                                                 │
│  Gate 2: GO | Gate 3: EPOCH 8/30 (~10h restantes)             │
└────────────────────────────────────────────────────────────────┘
```

---

*Documento actualizado: 2026-02-05 23:30 UTC (Gate 3 epoch 8/30)*
