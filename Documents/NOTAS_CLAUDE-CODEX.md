# Notas de Claude LOCAL para Codex

> Fecha: 2026-02-19
> Sesión: análisis de LR scheduling + diseño de batch 60ep
> Commits: `4233c53`, `837dad4`, `5fbcf85`, `0d1a55a`

---

## 1. Importación de resultados UNC (commit `4233c53`)

Se importaron 57 archivos desde la rama `unc` (commit `515e2f4` de Claude UNC). Incluye:

- **t3-wt 30ep COMPLETADO**: S=79.8% (empata 3ro con d4-a4r)
- **moe-dual 30ep COMPLETADO**: S=72.6% (6to de 6 runs largos)
- Gate 4.4 screening completo: film-dual, moe-dual, film-a4, film-d4 (evals + final_results + history)
- MoE v2/v3/v4 reorganizados en subdirectorios con config + history + final
- d4-a4r 30ep reorganizado con eval per epoch completo
- RANKING_DESCRIPTORES_UNIFICADO.md actualizado con los 6 scratch runs + 16 observaciones empíricas

**Todos los 6 scratch runs de 30ep están COMPLETOS:**

| # | Arm | Best S | Best Ep |
|---|-----|--------|---------|
| 1 | d4a4 | 83.6% | 30 |
| 2 | a4r | 82.0% | 29 |
| 3 | d4-a4r | 79.8% | 30 |
| 3 | t3-wt | 79.8% | 30 |
| 5 | d4a4r | 74.4% | 30 |
| 6 | moe-dual | 72.6% | 30 |

**Screening de 24 brazos CERRADO** (21 originales + 3 MoE v2/v3/v4). MoE v2/v3/v4 ninguno superó D0; familia MoE agotada.

---

## 2. Análisis de LR scheduling — Hallazgo importante

### El scheduler cosine mata el LR prematuramente

El scheduler usado en todos los runs es `LinearWarmupCosineScheduler`: warmup 200 steps, luego cosine annealing hasta 0 en `total_steps = epochs × 1000`.

Analizamos la curva de LR del run de t3-wt 30ep contra su performance:

| Epoch | LR mult | ratio LR | Δloss | S |
|-------|---------|----------|-------|---|
| 15 | 0.505 | 2.53e-4 | -0.043 | 66.2% |
| 20 | 0.253 | 1.27e-4 | -0.012 | 77.6% |
| 25 | 0.068 | 3.39e-5 | -0.001 | 79.4% |
| 29 | 0.003 | 1.39e-6 | +0.005 | — |
| 30 | 0.000 | 0.00 | +0.006 | 79.8% |

A epoch 25 el LR estaba al 7% del máximo, el loss dejó de bajar (Δ=-0.001), y **en e29-e30 el loss subió**. Pero S seguía subiendo de 79.4% a 79.8% — señal de que el modelo tenía más capacidad pero el scheduler le cortó el gradiente.

### Patrón universal en los 5 runs de 30ep

Extendimos el análisis a los 5 runs completados (d4a4, a4r, d4-a4r, d4a4r, moe-dual). **El patrón es idéntico en todos**:

- **Epochs 21-25** (LR mult 0.21→0.07): promedio Δloss = -0.016 a -0.020 por epoch (todavía bajando)
- **Epoch 27** (LR mult 0.025): el loss deja de bajar en 4 de 5 runs
- **Epochs 29-30** (LR mult <0.003): el loss **sube** en 4 de 5 runs

d4a4 es el **único** que siguió bajando loss hasta e30 — y es el que más S ganó (+1.4pp en e25→e30). Los otros 4 están estancados en loss desde e27.

---

## 3. Implementación: Trapezoidal LR scheduler (commit `837dad4`)

Se modificó `LinearWarmupCosineScheduler` en `gate43_scratch_training.py` para soportar un nuevo parámetro `--lr-hold-fraction`:

### Tres fases:
1. **Warmup**: steps [1, warmup_steps] — ramp lineal 0 → 1 (sin cambios)
2. **Hold** (NUEVO): steps (warmup_steps, hold_end] — LR constante a 1.0
3. **Cosine decay**: steps (hold_end, total_steps] — cosine 1 → 0 (sin cambios)

`hold_end = warmup_steps + int(hold_fraction × (total_steps - warmup_steps))`

### Backward compatible
`--lr-hold-fraction 0.0` (default) = comportamiento original exacto. Verificado con test de igualdad step-by-step.

### Otros cambios:
- **`lr_mult`**: nuevo campo en `epoch_record` de `training_history.json`. Registra el multiplicador de LR al final de cada epoch (valor entre 0 y 1). Permite diagnóstico post-hoc sin recalcular.
- **`lr_mult` property** en el scheduler para consulta.
- **Log al inicio**: si hold_fraction > 0, imprime las fases del schedule.
- **state_dict** actualizado para persistir `hold_fraction` y `hold_end` en checkpoint/resume.

### CLI
```
--lr-hold-fraction 0.5   # Hold LR at max for 50% of post-warmup steps
```

---

## 4. Script t3-wt 50ep con trapezoidal LR (commit `5fbcf85`)

Script SLURM: `experiments/bias_control/slurm/gate44_t3-wt_scratch_50ep_hold.sh`

Configuración:
- `--epochs 50 --lr-hold-fraction 0.5`
- Resultado: LR pleno epochs 1-25 (hold), cosine decay epochs 26-50
- Eval en epochs: 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
- Output: `/home/mfmendez/results/gate44_t3-wt_scratch_50ep_hold`

Curva LR del schedule trapezoidal:
```
Epochs  1-25: LR mult = 1.000 (HOLD)
Epoch 30:     LR mult = 0.907 (cosine empezó)
Epoch 35:     LR mult = 0.658
Epoch 40:     LR mult = 0.348
Epoch 45:     LR mult = 0.096
Epoch 50:     LR mult = 0.000
```

---

## 5. Análisis del dataset MAESTRO

Se investigó si aumentar los batches por epoch podría cambiar resultados:

- **MAESTRO v3 train**: 962 piezas, 159.2h de audio
- **Segmentación** (seg=4s, hop=1s): **569,909 segmentos**
- **Batches** (bs=16, drop_last): **35,619 batches por epoch completo**
- **Cap actual**: 1,000 batches/epoch = **2.8% del dataset por epoch**
- En 30 epochs con cap=1000: ~0.8 passes completos del dataset (ni siquiera 1x)
- Un epoch completo (35,619 batches) tomaría ~21h — impractical

**Conclusión**: extender epochs es mejor que aumentar batches por epoch. Con 60ep × 1000 batches vemos ~1.7 passes del dataset. El shuffle garantiza variedad entre epochs.

---

## 6. Análisis de parámetros por modelo

Se compararon los parámetros entrenables de los modelos principales (freeze-policy run-d):

| Modelo | Params total | Extra vs D0 | Componente extra |
|--------|-------------|-------------|------------------|
| D0 (baseline) | ~65M | — | — |
| d4a4 (dual concat) | ~66.5M | +1.5M | audio_proj + interval_proj |
| a4r (reverse cross-att) | ~68.2M | +3.2M | q_proj + pos_emb + cross-att layers |
| t3-wt (third tower) | ~67.9M | +2.9M | ratio tower + d4a4 injection |
| d4-a4r (mixed) | **~69.6M** | **+4.6M** | A4r reverse (~4.2M) + D4 concat (~0.26M) |
| moe-dual | ~70.5M | +5.5M | MoE adapters audio + midi |

**d4-a4r tiene 3.1M params más que d4a4**. La hipótesis es que necesita más epochs para que los parámetros del módulo reverse cross-attention converjan, lo que explicaría su estancamiento en ~79.8% a 30ep vs 83.6% de d4a4.

---

## 7. Batch 60ep — 5 runs con cosine estándar (commit `0d1a55a`)

### Diseño experimental

Mismo cosine scheduler que los runs de 30ep, pero estirado a 60 epochs. El LR a epoch 30 pasa de **0.0** (en run de 30ep) a **~0.50** (en run de 60ep). El modelo recibe 30 epochs más de gradiente real.

### D0 como control
Si D0 mejora mucho a 60ep → la ganancia es del training extra, no del descriptor.
Si D0 se estanca pero los descriptores suben → los descriptores aprovechan el training extra.

### Scripts SLURM creados

| Script | Arm | Output dir | Tiempo est. |
|--------|-----|-----------|-------------|
| `batch_60ep_d0.sh` | D0 (baseline control) | `batch_60ep_d0/` | ~25h |
| `batch_60ep_d4a4.sh` | d4a4 (record, 83.6%) | `batch_60ep_d4a4/` | ~31h |
| `batch_60ep_a4r.sh` | a4r (82.0%) | `batch_60ep_a4r/` | ~25h |
| `batch_60ep_d4-a4r.sh` | d4-a4r (79.8%) | `batch_60ep_d4-a4r/` | ~24h |
| `batch_60ep_moe-dual.sh` | moe-dual (72.6%) | `batch_60ep_moe-dual/` | ~54h |

Todos: seed 42, batch 16, run-d, 1000 batches/ep, eval cada 5 epochs (5,10,...,55,60).

### moe-dual y el límite de 48h
moe-dual (~54 min/ep × 60ep ≈ 54h) excede el límite de 48h de UNC. Los scripts tienen auto-resubmit: SLURM envía SIGTERM 595 segundos antes del límite, el training guarda checkpoint, y el script hace `sbatch $0` para re-lanzarse con `--resume`. Queremos probar este mecanismo.

### Curva LR 60ep vs 30ep

| Epoch | 30ep | 60ep |
|-------|------|------|
| 5 | 0.937 | 0.984 |
| 10 | 0.756 | 0.935 |
| 15 | 0.505 | 0.856 |
| 20 | 0.253 | 0.753 |
| 25 | 0.068 | 0.632 |
| 30 | 0.000 | 0.503 |
| 40 | — | 0.252 |
| 50 | — | 0.067 |
| 60 | — | 0.000 |

### Qué buscar en los resultados

1. **S@e30 de cada arm en run de 60ep** vs S@e30 de los runs de 30ep — ¿misma trayectoria o diverge por LR diferente?
2. **S@e60** final — ¿cuánto gana cada arm con el doble de epochs?
3. **D0@e60**: el control clave
4. **d4-a4r**: ¿alcanza a d4a4 con más epochs? (hipótesis de convergencia por más parámetros)
5. **moe-dual**: ¿funciona el auto-resubmit tras SIGTERM?
6. **lr_mult** en training_history.json: nuevo campo, verificar que aparece

---

## 8. Resumen de jobs pendientes en UNC

| Job | Tipo | Epochs | Scheduler | Estado |
|-----|------|--------|-----------|--------|
| D0 60ep | batch_60ep | 60 | cosine estándar | PENDIENTE |
| d4a4 60ep | batch_60ep | 60 | cosine estándar | PENDIENTE |
| a4r 60ep | batch_60ep | 60 | cosine estándar | PENDIENTE |
| d4-a4r 60ep | batch_60ep | 60 | cosine estándar | PENDIENTE |
| moe-dual 60ep | batch_60ep | 60 | cosine estándar | PENDIENTE |
| t3-wt 50ep hold | gate44 | 50 | trapezoidal (hold=0.5) | PENDIENTE |

Total: 6 jobs, 5 GPUs simultáneas + 1.

---

## 9. Archivos modificados/creados en esta sesión

| Archivo | Cambio |
|---------|--------|
| `experiments/bias_control/gate43_scratch/gate43_scratch_training.py` | Trapezoidal scheduler + lr_mult logging |
| `experiments/bias_control/slurm/gate44_t3-wt_scratch_50ep_hold.sh` | NUEVO → fix time 2d |
| `experiments/bias_control/slurm/batch_60ep_d0.sh` | NUEVO |
| `experiments/bias_control/slurm/batch_60ep_d4a4.sh` | NUEVO |
| `experiments/bias_control/slurm/batch_60ep_a4r.sh` | NUEVO |
| `experiments/bias_control/slurm/batch_60ep_d4-a4r.sh` | NUEVO |
| `experiments/bias_control/slurm/batch_60ep_moe-dual.sh` | NUEVO |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/RANKING_DESCRIPTORES_UNIFICADO.md` | Importado de UNC, fix duplicado |
| `results_unc/` | +57 archivos importados de UNC |

---

*Fin de notas — Claude LOCAL, 2026-02-19*
