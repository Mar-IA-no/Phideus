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
| d4a4r 30ep | `experiments/bias_control/slurm/gate43_d4a4r_scratch_30ep.sh` | d4a4r dual reverse scratch, 30 epochs, auto-resubmit |

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
11. **`set -eo pipefail` + `ls | head` mata scripts**: Si el directorio no existe, `ls` falla con exit 2, pipefail lo propaga, `set -e` termina el script silenciosamente. Siempre usar `|| true` en pipelines que pueden fallar legítimamente.
12. **sacct MaxRSS incluye page cache**: Reporta 60GB+ con --mem=32G pero no es OOM real. SLURM en Mendieta no aplica cgroups de memoria estrictamente.

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
| 1142272 | FAILED | ivb? | 1s — LDAP glitch (`/usr/bin/id: cannot find name for user ID 11163`) |
| 1142275 | FAILED | ivb09 | 27 min — pipefail bug: `ls checkpoint*.pt` en dir inexistente + `set -eo pipefail` = exit 2 |
| 1142416 | CANCELLED | — | Cancelado: se había cambiado --mem=0 innecesariamente |
| **1142417** | **PENDING** | — | Fix: mkdir OUTDIR antes del ls + `\|\| true` en pipeline. --mem=32G restaurado. |

**Fix aplicado**: El bug era que `ls -t $OUTDIR/checkpoint_epoch*.pt | head -1` con `set -eo pipefail` mataba el script si el directorio no existía (primer run). Solución: crear `$OUTDIR` antes del check y agregar `|| true`.

**Resultados en**: `~/results/gate43_a4r_scratch_30ep/`

**Logs**: `~/Repos/Phideus/logs/a4r30_1142417.{out,err}`

---

## d4a4r scratch 30 epochs

**Objetivo**: Evaluar dual reverse cross-attention (A4r + D4r combinados) a 30 epochs from scratch.

**Script**: `experiments/bias_control/slurm/gate43_d4a4r_scratch_30ep.sh`

**Código**: `gate43_scratch_training.py` con soporte d4a4r (implementado por LOCAL, commit 72c818d, merge a unc).

**Config**: from-scratch, freeze-policy run-d, seed=42, batch_size=16, structured eval en epochs 5,10,15,20,25,28,29,30.

**Modelo d4a4r**:
- Audio: Q=descriptor A4 (188 tokens), K/V=CNN features (2400) → Transformer(188) → pool → proj
- MIDI: Q=intervals D4 (N tokens), K/V=event embeddings (N) → Transformer(N) → pool → proj
- ~5.5M params nuevos (A4r ~4.4M + D4r ~1.05M)

| Job ID | Estado | Nodo | Detalle |
|--------|--------|------|---------|
| **1142422** | **PENDING** | — | En cola por Priority |

**Benchmark a superar**: d4a4-scratch (concat) = 83.6% S @ 30ep

**Resultados en**: `~/results/gate43_d4a4r_scratch_30ep/`

**Logs**: `~/Repos/Phideus/logs/d4a4r30_1142422.{out,err}`

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
d4a4r 30ep:  ~/results/gate43_d4a4r_scratch_30ep/
Logs SLURM:  ~/Repos/Phideus/logs/
```

---

## Gate 5B (Test05 + Test02) - Corte operativo 2026-02-26 22:40 UTC-3

### Estado real verificado

- Cluster/host: `mendieta.ccad.unc.edu.ar` (`2026-02-26T22:39:56-03:00`).
- Job array `1143414` (Test05 multi-seed):
  - **Cerrados con artefacto final**: `a4r_seed42`, `a4r_seed123`, `d4-a4r_seed42`, `d4-a4r_seed123`.
  - **RUNNING**: `a4r_seed456 (e22)`, `d4-a4r_seed456 (e22)`, `a4r_seed789 (e20)`, `d4-a4r_seed789 (e21)`, `a4r_seed1337 (e21)`, `d4-a4r_seed1337 (e5)`.
  - **PENDING**: todos los `d0` (`seed 42/123/456/789/1337`).
- Job array `1143415` (Test02 param-matched `random/shuffled/zero`): **3/3 PENDING**.
- `sacct` marca varios tasks Test05 como `FAILED`, pero el substep `python` figura `COMPLETED` y existe `final_results.json` (fallo de post-proceso del wrapper SLURM al leer claves viejas).

### Resultados preliminares cerrados (Test05, ordenados por S)

| Rank | Run | Best epoch | S | A2M | M2A | hard_neg |
|------|-----|------------|---|-----|-----|----------|
| 1 | `a4r_seed123` | 30 | **84.0%** | 84.2% | 84.0% | 95.0% |
| 2 | `d4-a4r_seed123` | 27 | **83.4%** | 85.0% | 83.4% | 95.2% |
| 3 | `d4-a4r_seed42` | 29 | **83.2%** | 83.8% | 83.2% | 94.6% |
| 4 | `a4r_seed42` | 26 | **80.2%** | 80.6% | 80.2% | 93.6% |

### Observacion / Hipotesis / Inferencia

- Observacion:
  - Hay 4 seeds cerradas solo para `a4r` y `d4-a4r`; `d0` aun no comenzo.
  - Test02 aun no inicio (3 tasks en cola).
  - Los runs `a4r`/`d4-a4r` muestran banda alta de `S` (80.2%-84.0%) en seeds cerradas.
- Hipotesis:
  - La cola de prioridad esta postergando los brazos mas lentos (`d0` y param-matched), que son justamente los que cierran robustez estadistica.
- Inferencia (preliminar):
  - Aun no se puede cerrar Gate 5B de robustez: faltan los controles de multi-seed `d0` + todo Test02.

### Proximo paso unico recomendado

- Mantener ejecucion y monitoreo de `1143414/1143415` hasta completar primero los 6 tasks RUNNING, luego sincronizar cada cierre nuevo a `results_unc` y re-evaluar ETA del bloque pendiente (`d0` + Test02).

### Riesgos / bloqueantes

1. `Priority` en SLURM mantiene bloqueados `d0` y Test02.
2. Falso `FAILED` en wrappers puede confundir cierre si se mira solo `sacct`.
3. Sin `d0` multi-seed y sin param-matched no hay contraste robusto para afirmaciones causales.

### Evidencia (paths + logs + metricas + timestamp)

- Queue snapshot: `squeue -u mfmendez` a `2026-02-26T22:39:56-03:00`.
- Accounting: `sacct -j 1143414,1143415 --format=JobID,JobName,State,Elapsed,Start,End,NodeList -P`.
- Logs activos: `~/Repos/Phideus/logs/g5b-ms_1143414_*.{out,err}`.
- Resultados cerrados: `~/results/gate5b_multiseed/{a4r_seed42,a4r_seed123,d4-a4r_seed42,d4-a4r_seed123}/final_results.json`.
- Sync para pull en otro server:
  - `results_unc/gate5b_multiseed/` (36 JSON),
  - `results_unc/logs/g5b-ms_1143414_{1,2,4,5}.{out,err}`.
- Commit/push de sync: `758e5c2` en rama `unc` (`origin/unc`).

### ETA realista

- Cierre de los 6 tasks RUNNING actuales: ~4-14 horas (segun epoch actual).
- Cierre de bloque pendiente (`d0` multi-seed + Test02): ~2-4 dias, dominado por cola + runs largos (~19h por task lento).

---

## Gate 5B (Test05 + Test02) - Corte operativo 2026-02-27 03:26 UTC-3

### Estado real verificado

- Cluster/host: `mendieta.ccad.unc.edu.ar` (`2026-02-27T03:26:19-03:00`).
- Job array `1143414` (Test05 multi-seed):
  - **Cerrados con `final_results.json`**: `a4r_seed42`, `a4r_seed123`, `a4r_seed456`, `a4r_seed789`, `a4r_seed1337`, `d4-a4r_seed42`, `d4-a4r_seed123`, `d4-a4r_seed456`, `d4-a4r_seed789`.
  - **RUNNING**: `d4-a4r_seed1337` (`idx14`, `Epoch 18/30`, sin linea `CANONICAL` aun).
  - **PENDING**: `d0_seed42`, `d0_seed123`, `d0_seed456`, `d0_seed789`, `d0_seed1337`.
- Job array `1143415` (Test02 param-matched): **3/3 PENDING** (`random`, `shuffled`, `zero`).
- `sacct` mantiene `FAILED` en wrappers de tasks cerrados, con substep `python` en `COMPLETED` + artefactos finales presentes.

### Resultados cerrados (Test05, ordenados por S)

| Rank | Run | Best epoch | S | A2M | M2A | hard_neg |
|------|-----|------------|---|-----|-----|----------|
| 1 | `a4r_seed123` | 30 | **84.0%** | 84.2% | 84.0% | 95.0% |
| 2 | `d4-a4r_seed123` | 27 | **83.4%** | 85.0% | 83.4% | 95.2% |
| 3 | `d4-a4r_seed42` | 29 | **83.2%** | 83.8% | 83.2% | 94.6% |
| 4 | `a4r_seed456` | 29 | **80.4%** | 81.0% | 80.4% | 93.6% |
| 5 | `a4r_seed42` | 26 | **80.2%** | 80.6% | 80.2% | 93.6% |
| 6 | `a4r_seed789` | 26 | **79.6%** | 80.4% | 79.6% | 92.4% |
| 7 | `a4r_seed1337` | 29 | **79.4%** | 80.2% | 79.4% | 95.4% |
| 8 | `d4-a4r_seed789` | 29 | **78.6%** | 81.6% | 78.6% | 94.0% |
| 9 | `d4-a4r_seed456` | 25 | **78.4%** | 81.2% | 78.4% | 93.2% |

### Sync `results_unc` (listo para pull remoto)

- Actualizados 5 runs cerrados nuevos en `results_unc/gate5b_multiseed/`:
  - `a4r_seed456`, `a4r_seed789`, `a4r_seed1337`, `d4-a4r_seed456`, `d4-a4r_seed789`.
- Actualizados logs en `results_unc/logs/`:
  - `g5b-ms_1143414_{7,8,10,11,13}.{out,err}`.
- Volumen del corte: **55 archivos nuevos** (45 JSON + 10 logs).
- Commit/push de sync: **`8ae30a2`** en rama `unc` (`origin/unc`).

### Observacion / Hipotesis / Inferencia

- Observacion:
  - Test05 quedo practicamente cerrado para `a4r/d4-a4r` (9/10 runs con final).
  - Solo resta `d4-a4r_seed1337` en ejecucion.
  - Test02 y todos los `d0` siguen en cola por `Priority`.
- Hipotesis:
  - El bloqueo principal para cerrar robustez estadistica ya no es entrenamiento activo masivo, sino scheduling de los brazos de control (`d0` + param-matched).
- Inferencia (preliminar):
  - Aun no corresponde cierre de robustez en Gate 5B: falta cerrar `idx14` y ejecutar bloques de control pendientes.

### Proximo paso unico recomendado

- Mantener monitoreo de `1143414_14` hasta `final_results.json`, sincronizar ese cierre a `results_unc`, y re-estimar ETA del bloque pendiente (`d0` + Test02).

### Riesgos / bloqueantes

1. Cola `Priority` sin ETA firme para `d0` y `1143415`.
2. `FAILED` de wrapper puede inducir lectura falsa de estado si no se verifica `python COMPLETED` + artefactos.
3. Sin corridas de control (`d0`, param-matched) no hay contraste causal completo.

### Evidencia (paths + logs + metricas + timestamp)

- Queue snapshot: `squeue -u mfmendez` a `2026-02-27T03:26:19-03:00`.
- Accounting: `sacct -j 1143414,1143415 --format=JobID,JobName,State,Elapsed,Start,End,NodeList -P`.
- Progreso run activo: `~/Repos/Phideus/logs/g5b-ms_1143414_14.err` -> `Epoch 18/30`, sin `CANONICAL`.
- Resultados cerrados: `~/results/gate5b_multiseed/*/final_results.json` (9 runs).
- Sync remoto listo: `~/Repos/Phideus/results_unc/gate5b_multiseed/` + `~/Repos/Phideus/results_unc/logs/`.
- Commit de sync publicado: `8ae30a2` (branch `unc`).

### ETA realista

- Cierre `1143414_14`: ~5-7 horas (si mantiene ritmo observado).
- Inicio/cierre de `d0` + Test02: sin ETA confiable mientras persista `Priority`.
