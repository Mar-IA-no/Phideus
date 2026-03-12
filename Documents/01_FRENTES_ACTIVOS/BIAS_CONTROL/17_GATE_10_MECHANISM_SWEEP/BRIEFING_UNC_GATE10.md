# Gate 10 — Mechanism Sweep: Briefing para Claude UNC

> Fecha: 2026-03-12
> Autor: Claude LOCAL
> Prioridad: ALTA (correr antes de multi-seed)

---

## Contexto

Gate 9 y A10 testaron 7 descriptores audio con reverse cross-attention. Todos convergen a ~69-71% S, sin diferenciación significativa entre descriptores. Esto sugiere que el mecanismo domina sobre el contenido.

Gate 10 desacopla las dos variables: cruza 3 descriptores representativos con 3 mecanismos de inyeccion.

## Resultados previos (referencia)

| Arm | Mecanismo | Best S | Nota |
|-----|-----------|--------|------|
| ctrl | ninguno | 79.2% | baseline |
| a4r-pca (Gate 8) | FiLM audio proj | 82.6% | record audio-only |
| a7r | rev_xattn | 70.4% | Gate 9 |
| a9r | rev_xattn | 71.6% | Gate 9 |
| a10ar | rev_xattn | 70.6% | A10 |
| a10dr | rev_xattn | 70.2% | A10 |
| d4a4 | concat dual | 84.1% | NO comparable (dual) |

## Diseño: 3 descriptors x 3 mecanismos = 9 runs

| Descriptor | Dim | concat | pca | attn_bias | rev_xattn (done) |
|-----------|-----|--------|-----|-----------|------------------|
| **a7** | 12 | `a7` RE-RUN | `a7-pca` | `a7-ab` | 70.4% |
| **a10a** | 12 | `a10a` | `a10a-pca` | `a10a-ab` | 70.6% |
| **a10d** | 32 | `a10d` | `a10d-pca` | `a10d-ab` | 70.2% |

**a7 concat es RE-RUN**: el historico de a7 concat es Gate 4.3 con 5ep, NO comparable con protocolo 30ep.

## Protocolo (identico a Gate 9)

```bash
--from-scratch --freeze-policy run-d --gate 10 --epochs 30 \
--seed 42 --max-batches-per-epoch 1000 --max-val-batches 846 \
--structured-eval-epochs 5 10 15 20 25 28 29 30 \
--num-workers 8 --embed-batch-size 16
```

## Batch sizes

- **concat y pca**: `--batch-size 16` (igual que Gate 9)
- **attn_bias**: `--batch-size 8` (bias tensor [B*8, T, T] es pesado)

Si B=8 da OOM en A30, bajar a B=4. En LOCAL (RTX 3090 24GB) B=4 funciona sin problemas.

## SLURM

El script `slurm/gate10_pilot.sh` esta listo como SBATCH array job:

```bash
sbatch slurm/gate10_pilot.sh
```

Esto lanza 9 jobs paralelos (array 0-8). Si se prefiere control individual:

```bash
# Concat (B=16) — 3 runs
for desc in a7 a10a a10d; do
  python experiments/bias_control/gate43_scratch/gate43_scratch_training.py \
    --descriptor $desc --batch-size 16 \
    --from-scratch --freeze-policy run-d --gate 10 --epochs 30 \
    --seed 42 --max-batches-per-epoch 1000 --max-val-batches 846 \
    --structured-eval-epochs 5 10 15 20 25 28 29 30 \
    --num-workers 8 --embed-batch-size 16 \
    --maestro-dir ${SCRATCH}/maestro-v3.0.0 \
    --output data/gate10_results/${desc}_seed42
done

# PCA (B=16) — 3 runs
for desc in a7-pca a10a-pca a10d-pca; do
  python experiments/bias_control/gate43_scratch/gate43_scratch_training.py \
    --descriptor $desc --batch-size 16 \
    --from-scratch --freeze-policy run-d --gate 10 --epochs 30 \
    --seed 42 --max-batches-per-epoch 1000 --max-val-batches 846 \
    --structured-eval-epochs 5 10 15 20 25 28 29 30 \
    --num-workers 8 --embed-batch-size 16 \
    --maestro-dir ${SCRATCH}/maestro-v3.0.0 \
    --output data/gate10_results/${desc}_seed42
done

# Attention Bias (B=8) — 3 runs
for desc in a7-ab a10a-ab a10d-ab; do
  python experiments/bias_control/gate43_scratch/gate43_scratch_training.py \
    --descriptor $desc --batch-size 8 \
    --from-scratch --freeze-policy run-d --gate 10 --epochs 30 \
    --seed 42 --max-batches-per-epoch 1000 --max-val-batches 846 \
    --structured-eval-epochs 5 10 15 20 25 28 29 30 \
    --num-workers 8 --embed-batch-size 16 \
    --maestro-dir ${SCRATCH}/maestro-v3.0.0 \
    --output data/gate10_results/${desc}_seed42
done
```

## Prerequisito: git pull origin main

El codigo de Gate 10 esta en `main`. Hacer pull antes de correr:

```bash
git pull origin main
```

## Output esperado

```
data/gate10_results/
├── a7_seed42/          # concat
├── a10a_seed42/        # concat
├── a10d_seed42/        # concat
├── a7-pca_seed42/      # FiLM projection
├── a10a-pca_seed42/
├── a10d-pca_seed42/
├── a7-ab_seed42/       # attention bias
├── a10a-ab_seed42/
└── a10d-ab_seed42/
```

Cada directorio contiene:
- `checkpoint_epoch{N}.pt` (full, para resume y eval)
- `checkpoint_epoch{N}_archive_base_not_for_eval.pt`
- `final_results.json`
- `training_history.json`
- Evals en `eval_epoch{N}.json` para epochs 5,10,15,20,25,28,29,30

## Verificacion post-run

Confirmar que todos los `final_results.json` tienen `"gate": "10"` y S > 0 en al menos alguna epoch.

## Estimacion de tiempo

- ~4-5h por run en A30 (basado en Gate 9: 246 min promedio)
- Total: ~40h si secuencial, ~5h si 9 GPUs en paralelo
- Los 9 pueden correr todos en paralelo si hay GPUs disponibles

## Gate 6 Exp A pendiente

Si sobran GPUs, se pueden resubmitir los 4 tasks pendientes de Gate 6 Exp A (tasks 3, 6, 9, 12 — screening seed=42). Pero Gate 10 tiene prioridad.

## Notas tecnicas

- **PCA (`-pca`)**: `base_model.audio_projection` queda frozen dentro del modelo. El forward usa `cond_audio_projection` (FiLM-conditioned). No es lo mismo que `pca` de Gate 8 (`gate5a_proj_cond.py`) — aca el descriptor es A7/A10a/A10d, no A4.
- **Attention Bias (`-ab`)**: Manual forward del Transformer con `need_weights=False`. Sin esto, los attention weights [B*8, 2400, 2400] causan OOM.
- **Flag `--gate 10`**: Nuevo. Overridea el gate label para trazabilidad. Guarda en arch_config y final_results.
