# Bitacora UNC — Phideus en Mendieta CCAD

> Registro operativo de todo lo que se ejecuta en el cluster UNC.
> Rama: `unc` | Agente: Claude Opus 4.6 en Mendieta

---

## Estado actual

| Componente | Estado | Fecha |
|-----------|--------|-------|
| Repo clonado | OK | 2026-02-16 |
| Conda env (phideus) | OK — Python 3.11, PyTorch 2.5.1+cu121 | 2026-02-16 |
| CUDA verificada | OK — A30, Driver 535, CUDA 12.2 | 2026-02-16 |
| SSH key GitHub | OK — push a rama `unc` funciona | 2026-02-16 |
| MAESTRO v3.0.0 | OK — 120GB descargado y descomprimido | 2026-02-16 |
| foundation_locked_e25.pt | OK — 288MB, MD5 `ddb2ebf7075eec4dcec1628341ec4942` | 2026-02-16 |
| segments_metadata.json | No necesario (no lo usa el pipeline de training) | — |

### Git

- Rama: `unc` (basada en `main`)
- Remote: `git@github.com:AlterMundi/Phideus.git` (SSH)
- Regla: yo solo pusheo a `unc`, nunca a `main`

---

## Gate 4.3 Fase 5 — 4 nuevos brazos

**Objetivo**: Evaluar a4r, d4r, a8, a9 (5 epochs desde foundation, freeze-policy run-d).

**Protocolo**: pool=256, queries=500, seed=42, batch_size=16, max_batches=1000.

**Comparar contra**: d4a4-scratch=83.6% S @ 30ep (multi-seed mean ~84%, rango 82.6-88.4%).

### Brazos

| ID | Descriptor | Mecanismo | Dim |
|----|-----------|-----------|-----|
| a4r | A4 log-freq (8d) | reverse cross-attention | 8 |
| d4r | D4 intervals (4d) | reverse cross-attention | 4 |
| a8 | onset-weighted chroma (12d) | concat | 12 |
| a9 | IDF-weighted attractor (12d) | concat | 12 |

### Scripts SLURM

| Script | Path | Funcion |
|--------|------|---------|
| Dry run | `experiments/bias_control/slurm/gate43_dryrun.sh` | 1 arm (a8), 1 epoch, 50 batches |
| Array job | `experiments/bias_control/slurm/gate43_fase5.sh` | 4 arms en paralelo |

### Jobs

| Job ID | Tipo | Estado | Resultado |
|--------|------|--------|-----------|
| 1142226 | dry run | FAILED | `set -u` + `LC_ALL` unbound en `/etc/profile`. Fix: quitar `-u` de `set -euo` |
| 1142227 | dry run v2 | FAILED | `--mem=0` bloqueó scheduling (nodos `mix` no aceptan nodo completo). Fix: `--mem=32G` |
| 1142228 | dry run v3 | OK | a8, 1ep, 50 batches. S=4.2% (esperado bajo con 50 batches). Copia MAESTRO: 22 min. |
| 1142230_0 | fase5 a4r | RUNNING | ivb09. Epoch 3/5 completo. |
| 1142230_1 | fase5 d4r | RUNNING | ivb20. Epoch 2/5 eval en curso. |
| 1142230_2 | fase5 a8 | RUNNING | ivb19. Epoch 2/5 training. |
| 1142230_3 | fase5 a9 | RUNNING | ivb05. Epoch 1/5 training (40%). |

### Lecciones aprendidas

1. **No usar `set -u`** en scripts sbatch — `/etc/profile` de Mendieta tiene variables no definidas (`LC_ALL`) que rompen con `-u`.
2. **`sbatch --test-only` es pesimista** — estimó 18h de espera, en realidad el job 1142226 entró en <1 min.
3. **`segments_metadata.json` no es bloqueante** — no lo usa gate42_training.py ni maestro_segments.py. Solo lo genera gate0_data_integrity.py como diagnóstico.
4. **`--mem=0` bloquea scheduling** — pide nodo completo, no entra en nodos `mix` con slots parciales. Usar `--mem=32G` para jobs normales.
5. **`PYTHONUNBUFFERED=1` obligatorio** — sin esto Python bufferea y no se ve progreso en logs hasta que termina el job.
6. **Output de Python va a stderr** — el training loguea via `logging` (stderr). Para monitorear: revisar `.err`, no `.out`.
7. **Copia MAESTRO a /scratch: 22 min** (1 nodo). Con 2+ nodos simultáneos se reparte el ancho de banda NFS y tarda más (~35-40 min estimado).
8. **Structured eval**: 13,532 segs, 846 batches, ~1.72 it/s en A30 = ~8 min/epoch.
9. **Copia MAESTRO con 1 solo nodo leyendo**: vuelve a ~22 min. Con 4 nodos simultáneos: hasta 34 min (a9 en ivb05).
10. **a4r ~2x más rápido por epoch** que d4r/a8/a9 (~13 min training vs ~33 min). Puede ser el descriptor o el nodo.

### Resultados parciales (en curso, 2026-02-16 06:00 UTC-3)

| Arm | Epoch | Loss | A2M | M2A | S | Hard Neg | Tiempo/ep |
|-----|-------|------|-----|-----|---|----------|-----------|
| a4r | 1 | 13.90 | 30.2% | 35.2% | 30.2% | 75.8% | 33.2 min |
| a4r | 2 | 13.57 | 33.0% | 45.0% | 33.0% | 79.8% | 31.8 min |
| a4r | 3 | 13.48 | 55.2% | 57.4% | 55.2% | 90.8% | 31.5 min |
| d4r | 1 | 13.96 | 49.0% | 52.0% | 49.0% | 89.2% | 58.6 min |
| a8 | 1 | 14.11 | 36.2% | 41.4% | 36.2% | 82.4% | 58.9 min |

**Observaciones tempranas**:
- d4r lidera epoch 1 (S=49.0%), pero a4r aceleró fuerte y lidera en epoch 3 (S=55.2%, +22pp de e2→e3).
- a8 epoch 1 S=36.2%, entre a4r y d4r.
- a9 aún sin eval, loss bajando rápido (14.69→13.62).

---

## Notas operativas

### Copiar MAESTRO a /scratch

Cada job copia ~120GB de NFS a SSD local. Medido: **22 min con 1 nodo**, ~35-40 min con 2+ nodos simultáneos (NFS bandwidth compartido ~5 GB/min). Es overhead necesario porque NFS es lento para I/O de training.

### Walltime

- Dry run: 1h (sobra)
- Fase 5 (5 epochs): 6h asignadas, estimado real ~3-4h
- Runs largos (>48h): necesitan `--signal=B:SIGTERM@595` + auto-resubmit

### Paths

```
Foundation:  ~/Repos/Phideus/data/bias_control_medium/training_outputs/foundation_locked_e25.pt
MAESTRO:     ~/Repos/Phideus/data/maestro_v3/maestro-v3.0.0/
Resultados:  ~/results/gate43_fase5/{a4r,d4r,a8,a9}/
Logs SLURM:  ~/Repos/Phideus/logs/
```
