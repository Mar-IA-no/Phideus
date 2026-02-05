# Proyecto Estado Actual - Phideus v5.0

**Actualizado**: 2026-02-05
**Estado**: 🟢 **BIAS_CONTROL Gate 2 COMPLETADO** - GO a Gate 3 (DANN)

---

## Resumen Ejecutivo

### Estado de Hipótesis

| Hipótesis | Estado | Evidencia |
|-----------|--------|-----------|
| H1: Estructura de ratios | **VALIDADA** | Distribuciones no aleatorias |
| H2: Aprendibilidad | **VALIDADA** | VAE/HRM val_loss < 0.5 |
| H3: Cross-modality | 🟢 **PROMETEDOR** | BIAS_CONTROL: Gap 0.478, Hard neg acc 80.4% |

### Situación Actual (2026-02-05)

**BIAS_CONTROL Gate 2** completado con resultado **GO**:

| Métrica | Valor | Umbral GO | Status |
|---------|-------|-----------|--------|
| Gap (aligned - random) | **0.478** | > 0.15 | ✅ PASS (3.2×) |
| Recall@10 (pool estructurado) | **34.4%** | > 25% | ✅ PASS (1.4×) |
| Hard Negative Accuracy | **80.4%** | > 60% | ✅ PASS (1.3×) |
| Domain Probe (separabilidad) | **92.7%** | Diagnóstico | ⚠️ Necesita DANN |

**Próximo paso**: Gate 3 (DANN) para forzar embeddings modal-agnostic.

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
| 3 | DANN Training | ⏳ Pendiente | - |
| 4 | Ratio Auxiliary | ⏳ Pendiente | - |
| 5 | Curriculum (opcional) | ⏳ Pendiente | - |

### Próximos Pasos

1. ⏳ Ejecutar Gate 3 (DANN)
   ```bash
   python experiments/bias_control/gate3_dann.py \
       --model data/bias_control_medium/training_outputs/gate2/checkpoint_epoch45.pt \
       --maestro-dir data/maestro_v3/maestro-v3.0.0 \
       --output data/bias_control_medium/training_outputs/gate3 \
       --epochs 30 --batch-size 16 --segment-len 4.0 --hop 1.0
   ```
2. ⏳ Evaluar: Domain classifier accuracy → ~50%
3. ⏳ Verificar: Recall no empeora vs Gate 2

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

### BIAS_CONTROL Gate 2 (mejor resultado actual)

```
┌────────────────────────────────────────────────────────────────┐
│                    BIAS_CONTROL GATE 2                         │
├────────────────────────────────────────────────────────────────┤
│  Gap (aligned - random):       0.478    (3.2× sobre umbral GO) │
│  Hard Negative Accuracy:       80.4%    (1.3× sobre umbral GO) │
│  Recall@10 (pool 256):         34.4%    (1.4× sobre umbral GO) │
│  Domain Probe (separabilidad): 92.7%    (→ Necesita DANN)      │
│                                                                 │
│  DECISIÓN: GO a Gate 3 (DANN)                                  │
└────────────────────────────────────────────────────────────────┘
```

---

*Documento actualizado: 2026-02-05 (post Gate 2 completion)*
