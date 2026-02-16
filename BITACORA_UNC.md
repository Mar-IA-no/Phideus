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
| a4r 30ep | `experiments/bias_control/slurm/gate43_a4r_scratch_30ep.sh` | a4r scratch, 30 epochs, auto-resubmit |

### Jobs

| Job ID | Tipo | Estado | Resultado |
|--------|------|--------|-----------|
| 1142226 | dry run | FAILED | `set -u` + `LC_ALL` unbound en `/etc/profile`. Fix: quitar `-u` de `set -euo` |
| 1142227 | dry run v2 | FAILED | `--mem=0` bloqueó scheduling (nodos `mix` no aceptan nodo completo). Fix: `--mem=32G` |
| 1142228 | dry run v3 | OK | a8, 1ep, 50 batches. S=4.2% (esperado bajo con 50 batches). Copia MAESTRO: 22 min. |
| 1142230_0 | fase5 a4r | **DONE** | ivb09. Best S=68.6% @ e5. Walltime ~2h37m. |
| 1142230_1 | fase5 d4r | **DONE** | ivb20. Best S=64.2% @ e5. Walltime ~4h52m. |
| 1142230_2 | fase5 a8 | **DONE** | ivb19. Best S=57.4% @ e5. Walltime ~4h55m. |
| 1142230_3 | fase5 a9 | **DONE** | ivb05. Best S=58.8% @ e5. Walltime ~5h26m. |

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

### Resultados finales (2026-02-16 10:40 UTC-3)

| Arm | Epoch | Loss | A2M | M2A | S | Hard Neg | Tiempo/ep |
|-----|-------|------|-----|-----|---|----------|-----------|
| a4r | 1 | 13.90 | 30.2% | 35.2% | 30.2% | 75.8% | 33.2 min |
| a4r | 2 | 13.57 | 33.0% | 45.0% | 33.0% | 79.8% | 31.8 min |
| a4r | 3 | 13.48 | 55.2% | 57.4% | 55.2% | 90.8% | 31.5 min |
| a4r | 4 | 13.38 | 63.4% | 64.8% | 63.4% | 90.2% | 31.5 min |
| a4r | 5 | 13.33 | 68.6% | 69.0% | **68.6%** | 91.6% | 31.5 min |
| d4r | 1 | 13.96 | 49.0% | 52.0% | 49.0% | 89.2% | 58.6 min |
| d4r | 2 | 13.75 | 58.0% | 58.2% | 58.0% | 91.6% | 58.4 min |
| d4r | 3 | 13.66 | 62.4% | 62.4% | 62.4% | 91.8% | 58.3 min |
| d4r | 4 | 13.58 | 63.6% | 63.0% | 63.0% | 92.2% | 58.4 min |
| d4r | 5 | 13.53 | 64.2% | 64.4% | **64.2%** | 93.2% | 58.4 min |
| a8 | 1 | 14.11 | 36.2% | 41.4% | 36.2% | 82.4% | 58.9 min |
| a8 | 2 | 13.58 | 49.0% | 48.6% | 48.6% | 86.2% | 58.6 min |
| a8 | 3 | 13.50 | 46.4% | 50.2% | 46.4% | 86.4% | 58.6 min |
| a8 | 4 | 13.42 | 56.4% | 54.4% | 54.4% | 88.8% | 58.7 min |
| a8 | 5 | 13.39 | 60.4% | 57.4% | **57.4%** | 90.6% | 58.7 min |
| a9 | 1 | 14.02 | 28.0% | 33.0% | 28.0% | 79.4% | 58.3 min |
| a9 | 2 | 13.60 | 48.2% | 51.0% | 48.2% | 85.8% | 57.9 min |
| a9 | 3 | 13.52 | 49.2% | 53.6% | 49.2% | 87.6% | 57.9 min |
| a9 | 4 | 13.43 | 52.4% | 54.2% | 52.4% | 87.6% | 58.1 min |
| a9 | 5 | 13.40 | 58.8% | 60.8% | **58.8%** | 90.4% | 57.9 min |

### Ranking @ epoch 5

| # | Arm | Best S | Curva | Potencial 30ep |
|---|-----|--------|-------|----------------|
| 1 | **a4r** | **68.6%** | +7-8pp/ep, no se aplanó | Alto |
| 2 | d4r | 64.2% | Se aplanó e3-e5 (+1.8pp) | Medio-bajo |
| 3 | a9 | 58.8% | Salto tardío e4→e5 (+6.4pp) | Medio |
| 4 | a8 | 57.4% | Inestable (regression e3) | Medio-bajo |

### Análisis

- **Ninguno superó d4a4@5ep (69.8%)**. a4r se acerca (68.6%).
- **a4r** es el más prometedor: curva ascendente, no saturó, y ~2x más rápido por epoch (~31 min vs ~58 min). Candidato para escalar a 30 epochs.
- **d4r** arrancó fuerte (49% e1) pero se aplanó rápidamente. A e5 solo +1.8pp vs e3.
- **a8** tuvo regression en e3 (48.6→46.4) pero se recuperó. Curva inestable.
- **a9** subió lento pero pegó un salto de +6.4pp en e4→e5. Podría seguir subiendo con más epochs.

---

## a4r scratch 30 epochs

**Objetivo**: Escalar a4r (ganador Fase 5) a 30 epochs from scratch, comparar vs d4a4-scratch=83.6%.

**Script**: `experiments/bias_control/slurm/gate43_a4r_scratch_30ep.sh`

**Código**: `gate43_scratch_training.py` con soporte a4r (implementado por LOCAL, merge bc651e5).

**Config**: from-scratch, freeze-policy run-d, seed=42, batch_size=16, structured eval en epochs 5,10,15,20,25,28,29,30.

| Job ID | Estado | Nodo | Detalle |
|--------|--------|------|---------|
| 1142272 | PENDING | — | Walltime 48h, estimado real ~17h. |

**Resultados en**: `~/results/gate43_a4r_scratch_30ep/`

**Logs**: `~/Repos/Phideus/logs/a4r30_1142272.{out,err}`

---

## Notas operativas

### Copiar MAESTRO a /scratch

Cada job copia ~120GB de NFS a SSD local. Medido: **22 min con 1 nodo**, ~35-40 min con 2+ nodos simultáneos (NFS bandwidth compartido ~5 GB/min). Es overhead necesario porque NFS es lento para I/O de training.

### Walltime

- Dry run: 1h (sobra)
- Fase 5 (5 epochs): 6h asignadas. Real: a4r ~2h37m, d4r/a8/a9 ~5h
- Runs largos (>48h): necesitan `--signal=B:SIGTERM@595` + auto-resubmit

### Paths

```
Foundation:  ~/Repos/Phideus/data/bias_control_medium/training_outputs/foundation_locked_e25.pt
MAESTRO:     ~/Repos/Phideus/data/maestro_v3/maestro-v3.0.0/
Resultados:  ~/results/gate43_fase5/{a4r,d4r,a8,a9}/
a4r 30ep:    ~/results/gate43_a4r_scratch_30ep/
Logs SLURM:  ~/Repos/Phideus/logs/
```
