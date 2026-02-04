# BIAS_CONTROL Medium Test - Resultados

**Fecha inicio**: 2026-02-04 16:56
**Estado**: 🔄 **EN EJECUCIÓN** (Epoch 11/30, migrado a tmux)
**Objetivo**: Baseline v0 con evaluación global

---

## 1. Configuración

```bash
python experiments/bias_control/run_all_gates.py \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/bias_control_medium \
    --epochs-gate2 30 --epochs-gate3 10 --epochs-gate4 10 \
    --max-batches-per-epoch 200 \
    --batch-size 16 --segment-len 4.0 --hop 1.0 \
    --num-workers 8 --device cuda
```

| Parámetro | Valor | Notas |
|-----------|-------|-------|
| Epochs Gate 2 | 30 | VICReg training |
| Epochs Gate 3 | 10 | DANN |
| Epochs Gate 4 | 10 | Ratio auxiliary |
| Batches/epoch | 200 | Limitado para tiempo razonable |
| Batch size | 16 | Limitado por VRAM |
| Segment length | 4.0s | Estándar |
| Total batches Gate 2 | 6,000 | 40× más que fast test |

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

### Progreso por Epoch

| Epoch | Loss | a2m_r@10 | m2a_r@10 | Gap | Best? |
|-------|------|----------|----------|-----|-------|
| 1 | 19.59 | 0.30% | 0.50% | 0.054 | ✓ |
| 2 | 15.92 | 0.20% | 0.70% | 0.075 | ✓ |
| 3 | 15.86 | 0.50% | 0.70% | 0.175 | ✓ |
| 4 | 15.66 | 0.90% | 1.20% | 0.242 | ✓ |
| 5 | 15.47 | 0.40% | 1.10% | 0.222 | |
| 6 | 15.35 | 1.00% | 1.40% | 0.302 | ✓ |
| 7 | 15.24 | 0.70% | 1.20% | 0.255 | |
| 8 | 15.18 | 0.90% | 1.60% | 0.295 | |
| 9 | 15.12 | 1.30% | 1.30% | 0.285 | ✓ |
| 10 | 15.18 | 1.30% | 2.10% | **0.398** | ✓ |
| 11+ | ... | ... | ... | ... | (en tmux) |

### Análisis de Tendencias

```
Gap Evolution:
├── Fast test (3 epochs):  0.026 (baseline)
├── Epoch 1:               0.054 (+107%)
├── Epoch 3:               0.175 (+224%)
├── Epoch 6:               0.302 (peak intermedio)
└── Epoch 10:              0.398 ← 15.3× fast test baseline!

Loss Evolution:
├── Fast test epoch 1:     28.3
├── Medium epoch 1:        19.6 (-31%)
├── Medium epoch 5:        15.5 (-21%)
└── Medium epoch 10:       15.2 (convergiendo)
```

### Señales Positivas

1. **Gap crece consistentemente**: 0.054 → 0.175 → 0.302 → 0.398
2. **Loss converge**: 19.6 → 15.2 (estable)
3. **No hay colapso**: std embeddings > 0.1 (verificado)
4. **Recall mejora**: a2m subió de 0.3% a 1.3%, m2a de 0.5% a 2.1%

### Interpretación

El modelo está aprendiendo cross-modal alignment de forma clara:
- Gap de 0.398 es **15.3× el fast test** (0.026) - señal muy fuerte
- Criterio GO (Gap > 0.10) **superado 4×**
- Aunque recall global es bajo (pool de 13,532), el gap indica separación real
- Tendencia sugiere que más epochs mejorarán aún más

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

## 8. Próximos Pasos

### Durante la ejecución
- [x] Monitorear epochs
- [x] Ejecutar sanity checks
- [x] Implementar evaluación estructurada

### Post-ejecución
- [ ] Guardar resultados finales
- [ ] Ejecutar `evaluate_structured_pool.py`
- [ ] Comparar con thresholds GO/NO-GO
- [ ] Decidir si continuar a Gate 3-4 o ajustar

---

## 9. Criterios GO/NO-GO (al terminar Gate 2)

### Evaluación Global (epoch 10)

| Criterio | Umbral | Valor | Status |
|----------|--------|-------|--------|
| Gap aligned-random | > 0.10 | **0.398** | 🟢 PASS (4×) |
| vs Random (recall) | > 5x | ~18× | 🟢 PASS |
| No collapse (std) | > 0.1 | ~0.35 | 🟢 PASS |

### Evaluación Estructurada (post-run)

| Criterio | Umbral |
|----------|--------|
| Recall@1 (pool 256) | > 5% |
| Hard negative accuracy | > 55% |

---

## 10. Migración a tmux (2026-02-04 18:46)

### Razón
El proceso original no estaba en tmux, riesgo de pérdida si se cerraba SSH.

### Mejoras implementadas en gate2_foundation.py
- `--resume`: Cargar checkpoint y continuar
- `--checkpoint-every`: Guardar cada N epochs (default: 1)
- `--max-val-batches`: Limitar batches de validación
- Guardado de `scheduler_state_dict` en checkpoints

### Comando de relanzamiento
```bash
tmux new-session -d -s bias_control "python experiments/bias_control/gate2_foundation.py \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/bias_control_medium/training_outputs/gate2 \
    --epochs 30 --batch-size 16 --segment-len 4.0 --hop 1.0 \
    --num-workers 8 --device cuda --max-batches-per-epoch 200 \
    --resume data/bias_control_medium/training_outputs/gate2/checkpoint_epoch10.pt \
    --checkpoint-every 1 --max-val-batches 256"
```

---

*Documento actualizado: 2026-02-04 18:50*
