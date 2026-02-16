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

**Comparar contra**: d4a4=69.8% S (ganador Gate 4.3).

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
| 1142227 | dry run v2 | PENDING | Esperando recursos (nodos full, ivb13 completing) |

### Lecciones aprendidas

1. **No usar `set -u`** en scripts sbatch — `/etc/profile` de Mendieta tiene variables no definidas (`LC_ALL`) que rompen con `-u`.
2. **`sbatch --test-only` es pesimista** — estimó 18h de espera, en realidad el job 1142226 entró en <1 min.
3. **`segments_metadata.json` no es bloqueante** — no lo usa gate42_training.py ni maestro_segments.py. Solo lo genera gate0_data_integrity.py como diagnóstico.

---

## Notas operativas

### Copiar MAESTRO a /scratch

Cada job copia ~120GB de NFS a SSD local. Estimado: 10-15 min. Es overhead necesario porque NFS es lento para I/O de training.

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
