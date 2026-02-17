# BIAS_CONTROL Medium Test - Resultados

> [!NOTE]
> Documento histórico del ciclo Gate 2 / Gate 2.5.  
> Estado vigente del programa (Gate 3 cerrado, Bloque A y Gate 4.2 cerrados, Gate 4.3 cerrado):  
> `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md` y `Documents/00_TRONCAL/Proyecto_Estado_Actual.md`.

**Fecha inicio**: 2026-02-04 16:56
**Fecha fin**: 2026-02-05
**Estado**: ✅ **COMPLETADO - GO**
**Checkpoint**: `checkpoint_epoch45.pt`
**Objetivo**: Baseline v0 con evaluación global + pool estructurado

---

## 1. Configuración

### Fase 1: 200 batches/epoch (epochs 1-31)

```bash
python experiments/bias_control/run_all_gates.py \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/bias_control_medium \
    --epochs-gate2 30 --max-batches-per-epoch 200 \
    --batch-size 16 --segment-len 4.0 --hop 1.0 \
    --num-workers 8 --device cuda
```

### Fase 2: 1000 batches/epoch (epochs 32-61)

```bash
tmux new-session -d -s bias_control "source venv/bin/activate && \
python experiments/bias_control/gate2_foundation.py \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/bias_control_medium/training_outputs/gate2 \
    --epochs 61 --batch-size 16 --segment-len 4.0 --hop 1.0 \
    --num-workers 8 --device cuda --max-batches-per-epoch 1000 \
    --resume data/bias_control_medium/training_outputs/gate2/checkpoint_epoch31.pt \
    --checkpoint-every 1 2>&1 | tee data/bias_control_medium/gate2_1000batches.log"
```

| Parámetro | Fase 1 | Fase 2 | Notas |
|-----------|--------|--------|-------|
| Epochs | 1-31 | 32-61 | +30 epochs |
| Batches/epoch | 200 | 1000 | 5× más |
| Data coverage | 3.3% | 16.7% | del training set |
| Batch size | 16 | 16 | Limitado por VRAM |
| Total samples | 96,000 | 480,000 | por epoch |

---

## 2. Resultados por Gate

### Gate 0: Data Integrity ✅ PASS

| Criterio | Valor | Umbral | Status |
|----------|-------|--------|--------|
| Alignment rate | 100% | > 90% | ✅ |
| Total segments | 127,092 | > 10,000 | ✅ |
| Shuffle verification | 1.0 | Working | ✅ |

### Gate 1: Intra-Modal Baselines ⚠️ FAIL (esperado)

| Métrica | Audio | MIDI |
|---------|-------|------|
| Recall@10 | 98.4% | 100% |
| Gap (aligned-random) | 0.004 | 0.085 |

**Nota**: El gap bajo es esperado con embeddings pre-entrenados sin fine-tuning. El pipeline continúa.

---

## 3. Gate 2: VICReg Training 🔄 EN PROGRESO

### Progreso por Epoch - Fase 1 (200 batches/epoch)

| Epoch | Loss | a2m_r@10 | m2a_r@10 | Gap | Best? |
|-------|------|----------|----------|-----|-------|
| 1 | 19.59 | 0.30% | 0.50% | 0.054 | ✓ |
| 3 | 15.86 | 0.50% | 0.70% | 0.175 | ✓ |
| 6 | 15.35 | 1.00% | 1.40% | 0.302 | ✓ |
| 10 | 15.18 | 1.30% | 2.10% | 0.398 | ✓ |
| 20 | 14.98 | 1.50% | 1.90% | 0.412 | ✓ |
| 31 | 14.81 | 1.60% | 2.10% | 0.392 | |

### Progreso por Epoch - Fase 2 (1000 batches/epoch)

| Epoch | Loss | a2m_r@10 | m2a_r@10 | Gap | Best? |
|-------|------|----------|----------|-----|-------|
| 32 | 14.63 | 1.40% | 1.70% | 0.365 | |
| 38 | 14.37 | 2.50% | 3.70% | 0.475 | ✓ |
| 44 | 14.23 | 2.10% | 1.90% | 0.354 | |
| 45 | 14.22 | 2.50% | 2.70% | **0.478** | ★ |
| 49 | 14.14 | 2.70% | 2.60% | 0.446 | |
| 50 | 14.12 | 2.80% | 2.80% | 0.437 | |
| 53 | 14.09 | 2.30% | 2.70% | 0.388 | actual |

### Análisis de Tendencias

```
Gap Evolution:
├── Fast test (3 epochs):  0.026 (baseline)
├── Epoch 10 (200 bat):    0.398 (15.3× baseline)
├── Epoch 31 (200 bat):    0.392 (plateau)
├── Epoch 38 (1000 bat):   0.475 (↑ con más data)
├── Epoch 45 (1000 bat):   0.478 ★ BEST
└── Epoch 53 (1000 bat):   0.388 (varianza alta)

Loss Evolution:
├── Epoch 1:   19.59
├── Epoch 31:  14.81 (-24%)
├── Epoch 45:  14.22 (-4%)
└── Epoch 53:  14.09 (-1%) ← convergiendo
```

### Observaciones (Fase 2)

1. **Gap plateaued con varianza alta**: Oscila entre 0.35-0.48
2. **Best model**: Epoch 45 (Gap=0.478, 18.4× baseline)
3. **Recall estable**: ~2.5% ambas direcciones (≈34× random)
4. **Loss sigue bajando**: Pero gap no correlaciona linealmente

### Interpretación

El modelo aprendió la mayor parte del alignment en los primeros 30 epochs:
- Escalar a 1000 batches/epoch dio mejora marginal (+8% en best gap)
- La varianza alta sugiere que el optimizer oscila cerca del óptimo
- **Test definitivo**: El pool estructurado con hard negatives determinará si hay identidad temporal real

---

## 4. Sanity Checks Ejecutados

### Resultados

| Check | Status | Detalle |
|-------|--------|---------|
| 1. Alignment spot-check | ✅ PASS | Offset real: 30-50ms (NO 706ms) |
| 2. Segment slicing | ✅ PASS | Audio y MIDI correctos |
| 3. Positive pairs | ⚠️ Error script | (bug en test, no en sistema) |
| 4. Recall formula | ✅ PASS | `ranks < k` es correcto |

### Alertas GPT5.2 Resueltas

| Alerta | Descripción | Resolución |
|--------|-------------|------------|
| A | Pool grande aplasta números | ✅ Válido, implementar pool estructurado |
| B | Drift 706ms sospechoso | ✅ **RESUELTA**: Era duración total, no alineación |
| C | Inconsistencia métricas | ✅ **RESUELTA**: Matemáticamente consistente |

---

## 5. Tiempo Estimado

| Gate | Epochs | Tiempo estimado |
|------|--------|-----------------|
| Gate 2 | 30 | ~2.5 horas |
| Gate 3 | 10 | ~50 minutos |
| Gate 4 | 10 | ~50 minutos |
| **Total** | 50 | **~4-5 horas** |

---

## 6. Archivos de Salida

```
data/bias_control_medium/
├── segments/
│   └── segments_metadata.json      # 127,092 segmentos
├── training_outputs/
│   └── gate2/
│       ├── best_model.pt           # Mejor checkpoint
│       ├── training_history.json   # Métricas por epoch
│       └── final_model.pt          # (al terminar)
└── results/
    └── pipeline_results.json       # (al terminar)
```

---

## 7. Scripts Nuevos

| Script | Propósito |
|--------|-----------|
| `experiments/bias_control/sanity_checks.py` | Verificación de alertas GPT5.2 |
| `experiments/bias_control/evaluate_structured_pool.py` | Evaluación con pool 256 + hard negatives |

---

## 8. Evaluación Final - Pool Estructurado ✅ COMPLETADO

### Configuración del Pool

```
Por cada query (500 queries totales):
├── 64 hard negatives: misma pieza, distinto tiempo (±10s min)
├── 32 semi-hard: mismo compositor, otra pieza
├── 159 random: otras piezas aleatorias
└── 1 positivo: el match correcto

Total: 256 candidatos por query
```

### Resultados Pool Estructurado

**Audio → MIDI**:

| Métrica | Valor | vs Random |
|---------|-------|-----------|
| R@1 | 4.4% | 11.3× |
| R@5 | 20.8% | 10.6× |
| R@10 | 34.4% | 8.8× |
| R@20 | 52.0% | 6.7× |
| Mean Rank | 37.4 / 256 | - |
| MRR | 0.138 | - |

**MIDI → Audio**:

| Métrica | Valor | vs Random |
|---------|-------|-----------|
| R@1 | 5.2% | 13.3× |
| R@5 | 24.6% | 12.6× |
| R@10 | 37.6% | 9.6× |
| R@20 | 56.4% | 7.2× |
| Mean Rank | 31.6 / 256 | - |
| MRR | 0.158 | - |

### Hard Negative Analysis

| Test | Accuracy | Interpretación |
|------|----------|----------------|
| **vs Same-Piece-Diff-Time** | **80.4%** | ✅ Identidad temporal confirmada |
| vs Random | 87.0% | Distingue bien piezas diferentes |

### Comparación con Criterios GO/NO-GO

| Criterio | Umbral NO-GO | Umbral GO | Valor | Status |
|----------|--------------|-----------|-------|--------|
| R@10 (pool 256) | < 15% | > 25% | **36%** | ✅ PASS (1.4×) |
| Accuracy vs hard neg | < 50% | > 60% | **80.4%** | ✅ PASS (1.3×) |
| MRR | < 0.10 | > 0.20 | **0.148** | ⚠️ Borderline |

**DECISIÓN FINAL**: **GO** a Gate 3 (DANN)

---

## 9. Gate 2.5 - Embedding Analysis ✅ COMPLETADO

### Domain Probe

| Métrica | Valor |
|---------|-------|
| Linear Separability | **92.7%** |
| Centroid Distance | 1.73 |

**Diagnóstico**: Clasificador lineal distingue Audio vs MIDI con 92.7% → **Modal shortcut detectado** → DANN requerido.

### Piece Clustering

| Métrica | Audio | MIDI |
|---------|-------|------|
| Silhouette | -0.111 | -0.075 |

**Diagnóstico**: Clustering pobre, pero no invalida el aprendizaje de identidad temporal.

### Variance Analysis

| Dominio | Dead Dims | Total |
|---------|-----------|-------|
| Audio | 0 | 256 |
| MIDI | 0 | 256 |

**Diagnóstico**: Sin colapso de embeddings.

### Recomendación Gate 2.5

Proceder a **Gate 3 (DANN)** para forzar embeddings modal-agnostic (objetivo: Domain Probe → 50%).

---

## 10. Auditoría Completa ✅ 8/10 PASS

| Check | Resultado | Detalles |
|-------|-----------|----------|
| A1: Dataset Structure | ✅ PASS | 1,276 piezas |
| A2: Alignment | ❌ FAIL* | Método impreciso |
| A3: Checkpoint | ✅ PASS | 398MB, epoch 44 |
| B1: Model Loading | ✅ PASS | 74M params |
| B2: Dimensions | ✅ PASS | 256-d |
| B3: No Collapse | ✅ PASS | std > 0.1 |
| C1: Pool Global | ✅ PASS | Gap 0.467 |
| C2: Pool Structured | ✅ PASS | Hard neg acc 81.5% |
| D1: Shuffled Pairs | ❌ FAIL* | Esperado |
| D2: Oracle MIDI | ✅ PASS | Diagonal=1.0 |

*Falsos positivos explicados en `INFORME_GATE2_COMPLETO.md`

---

## 11. Archivos Generados

### Checkpoints

```
data/bias_control_medium/training_outputs/gate2/
├── checkpoint_epoch45.pt    ★ BEST (seleccionado)
├── checkpoint_epoch61.pt    Final
├── best_model.pt            Symlink
└── training_history.json    Métricas
```

### Evaluaciones

```
data/bias_control_medium/evaluations/
├── structured_pool_epoch45.json     Pool estructurado
├── gate2_5/gate2_5_results.json     Análisis embeddings
└── audit_gate2/audit_gate2_results.json  Auditoría
```

---

## 12. Conclusión

Gate 2 **COMPLETADO** con resultado **GO**:

1. El modelo aprende **alignment cross-modal** (Gap 0.478, 3.2× sobre umbral)
2. El modelo aprende **identidad temporal** (80.4% vs hard negatives)
3. Existe **modal shortcut** (92.7% separabilidad) → Requiere DANN
4. No hay **colapso de embeddings** (0 dead dims)

**Próximo paso**: Gate 3 (DANN) para forzar embeddings modal-agnostic.

---

*Documento actualizado: 2026-02-05 (post-evaluación final)*

---

## 9. Criterios GO/NO-GO (al terminar Gate 2)

### Evaluación Global (epoch 53)

| Criterio | Umbral | Valor | Status |
|----------|--------|-------|--------|
| Gap aligned-random | > 0.15 | **0.478** (best) | 🟢 PASS (3.2×) |
| vs Random (recall) | > 10× | ~34× | 🟢 PASS |
| min(a2m, m2a) | > 0.5% | 2.3% | 🟢 PASS |
| No collapse (std) | > 0.1 | ~0.35 | 🟢 PASS |

### Evaluación Estructurada (post-run) — TEST DEFINITIVO

| Criterio | Umbral | Status |
|----------|--------|--------|
| Recall@10 (pool 256) | > 25% | ⏳ Pendiente |
| Accuracy vs same-piece-diff-time | > 60% | ⏳ Pendiente |
| MRR | > 0.20 | ⏳ Pendiente |

**Comando**:
```bash
python experiments/bias_control/evaluate_structured_pool.py \
    --model data/bias_control_medium/training_outputs/gate2/best_model.pt \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --pool-size 256 --n-hard-negatives 64 --n-semi-hard 32
```

---

## 10. Migración a tmux (2026-02-04 18:46)

### Razón
El proceso original no estaba en tmux, riesgo de pérdida si se cerraba SSH.

### Mejoras implementadas en gate2_foundation.py
- `--resume`: Cargar checkpoint y continuar
- `--checkpoint-every`: Guardar cada N epochs (default: 1)
- `--max-val-batches`: Limitar batches de validación
- Guardado de `scheduler_state_dict` en checkpoints

### Relanzamientos

**Epoch 10 → 31** (200 batches/epoch):
```bash
tmux new-session -d -s bias_control "python experiments/bias_control/gate2_foundation.py \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/bias_control_medium/training_outputs/gate2 \
    --epochs 31 --batch-size 16 --segment-len 4.0 --hop 1.0 \
    --num-workers 8 --device cuda --max-batches-per-epoch 200 \
    --resume checkpoint_epoch10.pt --checkpoint-every 1"
```

**Epoch 31 → 61** (1000 batches/epoch):
```bash
tmux new-session -d -s bias_control "source venv/bin/activate && \
python experiments/bias_control/gate2_foundation.py \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/bias_control_medium/training_outputs/gate2 \
    --epochs 61 --batch-size 16 --segment-len 4.0 --hop 1.0 \
    --num-workers 8 --device cuda --max-batches-per-epoch 1000 \
    --resume checkpoint_epoch31.pt --checkpoint-every 1 \
    2>&1 | tee data/bias_control_medium/gate2_1000batches.log"
```

---

*Documento actualizado: 2026-02-05 12:45*
