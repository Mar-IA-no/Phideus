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

---

## Gate 5B — CERRADO (2026-03-05)

Todos los tests de Gate 5B completados entre sesiones anteriores y la sesión del 2026-03-05.

### Test05 Multi-seed — Cerrado 15/15

| Descriptor | Media S | ±Std | Seeds completados |
|-----------|---------|------|-------------------|
| d4a4 | 84.1% | ±2.3pp | 5 (Gate 4.5, referencia) |
| d4-a4r | 81.2% | ±2.5pp | 5 |
| a4r | 80.7% | ±1.9pp | 5 |
| D0 | 75.2% | ±2.3pp | 5 |

### Test02 Parameter-matched — Cerrado 4/4

| Condición | S |
|-----------|---|
| real | 83.0% |
| zero | 75.0% |
| random | 73.6% |
| shuffled | 73.6% |

Caída de 8.0-9.4pp con mismos parámetros entrenables → argumento causal fijado.

### Test11 Pre-Projection Information Retention — Cerrado 2/2

| Descriptor | midi2events CE | audio2events CE | Info Retention |
|-----------|---------------|-----------------|----------------|
| d4a4 | 2.965 | 3.069 | **0.770** |
| d4-a4r | 2.971 | 3.073 | **0.748** |

Jobs: 1144295_0 (d4a4), 1144295_1 (d4-a4r). COMPLETED.

### Test13G d4-a4r — Cerrado

| Descriptor | Best F1 |
|-----------|---------|
| D0 pool-188 | 0.1089 |
| d4a4 | 0.1037 |
| a4r | 0.1024 |
| **d4-a4r** | **0.1021** |

Job 1144296. COMPLETED. No hay ventaja descriptor-guided en decodificabilidad pre-pooling.

### Commits de cierre

| Commit | Contenido |
|--------|-----------|
| `94e68d3` | Fix Gate 6 scripts para Mendieta (set -eo, MAESTRO_SRC, --error) |
| `4908f9a` | Gate 6 section en RANKING, status update |
| `dce967b` | Gate 5B COMPLETO + fix MAESTRO_SRC en Gate 6 |
| `e66d633` | validate-sbatch skill + Test13G d4-a4r samples |
| `11dcc4b` | Update validate-sbatch skill: strategic options panel |

---

## Gate 6 — AMT (Automatic Music Transcription)

**Pregunta central**: ¿A4 aporta información útil para tareas downstream como AMT?

**Backbone**: Transkun v2 (state-of-the-art AMT).

### Dependencias instaladas

| Paquete | Versión | Rol |
|---------|---------|-----|
| transkun | 2.0.1 | Modelo AMT base |
| pretty_midi | 0.2.11 | Parsing MIDI |
| midi2audio | 0.1.1 | MIDI → audio |
| mir_eval | 0.8.2 | Evaluación nota a nota |
| moduleconf | 0.1.4 | Config de Transkun |

Transkun pretrained weights: 56MB, incluidos en paquete pip (no requiere transfer).

### Experimentos

| Exp | Nombre | Jobs | Estado |
|-----|--------|------|--------|
| **0** | Transkun baseline | 1 | **CERRADO** (LOCAL) — Note F1=0.8934 |
| **C** | VICReg decoder | 4 arms | **CERRADO** (LOCAL) — Best F1=0.1570 @ ep50, 244 min |
| **A** | Transkun + A4 | 15 | **PENDIENTE UNC** — scripts listos |
| **B** | Transkun degraded | 27 | **PENDIENTE UNC** — preflight en curso |

### Exp C — Intentos en UNC (cancelados)

Exp C se corrió en LOCAL (RTX 3090, 244 min). Los intentos en UNC sirvieron para descubrir problemas en scripts:

| Job ID | Estado | Duración | Error |
|--------|--------|----------|-------|
| 1144325_[0-3] | FAILED | ~13s c/u | `cp: cannot stat '/home/mfmendez/data/maestro_v3/...'` — path MAESTRO absoluto incorrecto |
| 1144560_[0-3] | CANCELLED | — | Cancelado: Exp C ya cerrado en LOCAL |
| 1144579 | CANCELLED | — | Dry-run short cancelado: ya no necesario |

**Lección**: MAESTRO_SRC debe ser `$REPO/data/maestro_v3/maestro-v3.0.0`, NUNCA path absoluto bajo `/home/mfmendez/data/`. Este error desperdició queue slots y motivó la creación de `/validate-sbatch`.

### Exp B — Campaña de preflight

Estrategia: correr 100 iteraciones en partición `short` (55 min) para medir throughput real en A30 y calibrar `--time` de los 27 jobs reales.

#### Job 1144581 — Preflight v1 (FAILED, 1:14)

**Error**: `AttributeError: module 'transkun' has no attribute '__version__'`

**Causa**: El check de imports usaba `transkun.__version__` que no existe en transkun 2.0.1.

**Fix**: Quitar referencia a `transkun.__version__` del check.

**Dato positivo**: MAESTRO OK, scripts OK, GPU OK (A30), weights OK.

#### Job 1144594 — Preflight v2 (FAILED, 17:33)

**Progreso**: Preflight checks PASS → MAESTRO staging 928s (~15 min) → Transkun cargado (66.3K new params) → Creating dataloaders → CRASH.

**Error 1 — ImportError**:
```
ImportError: cannot import name 'createPickle' from 'transkun.createDatasetMaestro'
```

**Análisis**: `transkun.createDatasetMaestro` es solo un script CLI (`if __name__ == "__main__"`), no expone `createPickle()` como función. El módulo disponible es `Data.createDatasetMaestroCSV(datasetPath, csvPath)`. El código en `transkun_a4_finetune.py:302` estaba roto desde origen — nunca corrió exitosamente en ningún server.

**Fix aplicado** en `transkun_a4_finetune.py`:
- Reemplazar `from transkun.createDatasetMaestro import createPickle` por `from transkun.Data import createDatasetMaestroCSV`
- Implementar lógica de split (train/validation/test) + pickle dump inline
- Posteriormente: cambiar a fail-fast (`FileNotFoundError`) para forzar precompute

**Error 2 — Memoria** (observación, no fue causa del crash):
```
Memory Utilized: 60.72 GB
Memory Efficiency: 126.49% of 48.00 GB
```

**Análisis de Codex**: 60.72 GB con `--mem=48G` no es conservador. Reducir `num_workers` de 4 a 2 podría ayudar.

**Recomendaciones de Codex** (todas implementadas):
1. Separar creación de pickles de los jobs de entrenamiento → precompute
2. Repetir preflight con fix aplicado
3. Medir MaxRSS estable y dónde ocurre el pico
4. Recién ahí fijar memoria final

#### Precompute de pickles (login node, 2026-03-06)

Generados offline en login node (~40 min, 1.4 GB RAM pico):

| Split | Piezas | Tamaño |
|-------|--------|--------|
| train | 962 | 353.7 MB |
| val | 137 | 40.3 MB |
| test | 177 | 46.4 MB |

`createDatasetMaestroCSV` parsea 1276 MIDIs (via NFS) + lee headers WAV. No carga audio. Los pickles contienen notas parseadas + índices espaciales para búsqueda O(log n) por rango temporal.

**Riesgo eliminado**: Race condition — 27 jobs simultáneos ya no intentan crear pickles; los consumen read-only desde scratch.

**Bug encontrado durante validación**: Nuestro precompute generó `validation.pickle` pero el código busca `val.pickle` (nombre que usa la versión CLI de transkun). Fix: `mv validation.pickle val.pickle`.

#### Refactoring de `create_transkun_dataloaders()`

Cambios aplicados:
1. `num_workers` default: 4 → 2 (reducir pico de RAM por CoW de workers)
2. Eliminada creación de pickles in-flight → `FileNotFoundError` si no existen
3. Removido import de `createDatasetMaestroCSV` (ya no necesario en runtime)

#### Job 1144627 — Preflight v3 (FAILED, 28:48) ← PARA DISCUSIÓN

**Timeline completo del job**:
```
20:09:15  Preflight checks (GPU, paths, pickles, imports)  → ALL PASS
20:10:06  MAESTRO staging start (cp -r 120GB NFS→scratch)
20:35:44  MAESTRO staging end (1538s ≈ 25 min)
20:35:45  Benchmark start: transkun_degraded.py --iterations 100
          Config: noise@10dB, finetune-degraded
20:35:4x  Transkun v2 loaded. New params: 66.3K
20:36:1x  Dataloaders created:
            train: 962 pieces, 24.4s load + 2.5s index
            val:   137 pieces, 1.1s load + 0.3s index
20:36:xx  "Training finetune-degraded for 100 iterations..."
20:38:03  CRASH — RuntimeError in training loop
```

**Fixes validados por este run**:
- [x] `createPickle` ImportError → precomputed pickles OK
- [x] `val.pickle` naming → OK
- [x] Todos los paths → OK
- [x] Todas las deps → OK
- [x] Transkun model load → OK (66.3K new params)
- [x] Dataloader creation → OK (pickles consumidos read-only)

---

##### PROBLEMA ABIERTO 1: `torch.stack` en collate (BLOCKER para Exp A y B)

**Error exacto** (`transkun_a4_finetune.py:477`):
```python
RuntimeError: stack expects each tensor to be equal size,
  but got [705600, 2] at entry 0 and [768000, 2] at entry 1
```

**Contexto**: El training loop itera sobre batches del DataLoader. Cada item del batch es un dict con `audioSlice` (tensor de audio stereo) + `notes` + metadata. El collate intenta `torch.stack` sobre los `audioSlice` de un batch de 4 items.

**Por qué fallan los tamaños**: Transkun segmenta MAESTRO en chunks de `segment_size=16.0` segundos con `segment_hop=8.0`. Pero los chunks cerca de los bordes de una pieza (inicio/fin) son más cortos que 16s. El `DatasetMaestroIterator.__getitem__` usa `readAudioSlice(audioPath, begin, end)` donde `begin` puede ser negativo y `end` puede exceder la duración → produce slices de largo variable.

**Dimensiones observadas**:
- `[705600, 2]` → 705600 samples / 44100 Hz ≈ **16.0s** stereo ← chunk completo
- `[768000, 2]` → 768000 samples / 44100 Hz ≈ **17.4s** stereo ← chunk extendido (borde?)

Nota: 768000/16000 = 48.0s a 16kHz, o 17.4s a 44.1kHz. Podría ser un mismatch de sample rate. **Verificar qué sample rate usan los WAVs de MAESTRO vs lo que espera Transkun**.

**Transkun nativo NO tiene este problema** porque usa su propio `collate_fn` importado:
```python
from transkun.Data import collate_fn
```
Esta `collate_fn` maneja padding internamente. **El bug probablemente está en que nuestro código usa `torch.stack` en algún lado en vez de dejar que el collate de Transkun maneje el padding.**

**Ubicación del bug**: `transkun_a4_finetune.py:477` — hay que leer esa línea exacta para ver si:
1. Estamos usando un collate custom que no padea, o
2. El collate de Transkun SÍ se usa pero algo posterior hace `torch.stack`

**Preguntas para discusión**:
1. ¿El `collate_fn` de Transkun que importamos maneja padding? ¿O devuelve listas sin stackear?
2. Si devuelve listas, ¿dónde en nuestro código hacemos `torch.stack` y por qué?
3. ¿Hay un mismatch de sample rate (44.1 kHz vs 16 kHz) que explique los tamaños raros?
4. ¿Transkun espera mono o stereo? Los tensores son `[N, 2]` (stereo). Si Transkun espera mono, el canal extra podría ser la causa del tamaño inesperado.

**Impacto**: Este bug afecta tanto a Exp A como a Exp B — ambos usan `train_loop` de `transkun_a4_finetune.py`.

---

##### PROBLEMA ABIERTO 2: Memoria (RESUELTO — page cache, no OOM)

**Datos del seff**:
```
Memory Utilized: 60.57 GB
Memory Efficiency: 126.18% of 48.00 GB
Job Wall-clock time: 00:28:48
State: FAILED (exit code 1)   ← exit 1, NO 137/SIGKILL
```

**Profile de memoria del script** (bash `mem_usage()` en puntos clave):
```
20:09:16  RSS=3704kB | free=61G    ← inicio, nodo limpio
20:10:06  RSS=3704kB | free=60G    ← pre-staging, sin cambio
20:35:44  RSS=3464kB | free=0G     ← post-staging: cp -r 120GB comió todo el free
20:35:45  RSS=3464kB | free=0G     ← pre-benchmark, proceso sigue en ~3.4 MB
```

**Análisis**:
- El proceso bash consume ~3.4 MB en todo momento — no hay leak
- `free=0G` aparece SOLO después de `cp -r` de 120 GB a scratch
- Linux usa TODA la RAM libre como page cache para I/O — es comportamiento normal
- El page cache se libera on-demand cuando procesos necesitan RAM (no es "memoria usada")
- SLURM `seff` reporta MaxRSS del cgroup que INCLUYE page cache → ~60 GB inflado
- El job corrió 28 min con "126% memory efficiency" sin ser killed → Mendieta NO aplica cgroups de memoria estrictamente
- Exit code 1 (Python error) no 137 (SIGKILL) confirma que no murió por OOM

**Conclusión**: `--mem=48G` funciona. El consumo real del proceso (Python + modelo + dataloaders + 2 workers) es probablemente ~10-20 GB. Los ~60 GB de MaxRSS son artefacto del page cache del `cp -r`.

**Verificación pendiente**: Si queremos medir el consumo real de Python, podemos agregar en el preflight:
```python
import resource
print(f"Python MaxRSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB")
```
Esto reporta el RSS del proceso Python sin page cache.

**Recomendación para `--mem` de los jobs reales**:
- `--mem=48G` es seguro (probado, no mata por OOM en Mendieta)
- `--mem=32G` PODRÍA funcionar pero no está testeado — el modelo + dataloaders + workers podrían usar >32 GB
- `--mem=60G` es innecesariamente alto y perjudica scheduling (un nodo de 64 GB no puede alojar 2 jobs de 60G)

---

##### RESUMEN DE ESTADO PARA DISCUSIÓN

```
Exp B Pipeline Status:
  [OK] SLURM script syntax, directives, env setup
  [OK] Paths (MAESTRO, scripts, pickles, logs, conda)
  [OK] Dependencies (transkun 2.0.1, pretty_midi, mir_eval, etc.)
  [OK] MAESTRO staging to scratch (~15-25 min dependiendo de NFS)
  [OK] Transkun model load (pretrained + 66.3K new params)
  [OK] Pickle load + dataloader creation (train 962 + val 137 pieces)
  [OK] Memory: --mem=48G funciona (60GB seff es page cache)
  [FAIL] Training loop: torch.stack en collate → audio chunks largo variable
  [PENDING] Throughput measurement (no se alcanzó por crash)
  [PENDING] --time calibration para los 27 jobs reales
```

**Para poder submitir los 42 jobs (Exp A + B)**:
1. Fixear `torch.stack` — requiere entender collate de Transkun vs nuestro código
2. Preflight v4 que complete las 100 iters
3. Leer throughput/iter → calcular `--time` óptimo
4. `/validate-sbatch` + submit

### Scripts SLURM Gate 6

| Script | Exp | Partición | Notas |
|--------|-----|-----------|-------|
| `gate6_vicreg_decoder.sh` | C | multi | Ya no necesario (cerrado en LOCAL) |
| `gate6_transkun_a4.sh` | A | multi | 15 jobs, --array=0-14, --mem=60G, --time=48h |
| `gate6_transkun_degraded.sh` | B | multi | 27 jobs, --array=0-26, --mem=60G, --time=TBD |
| `gate6_expB_preflight.sh` | B preflight | short | 100 iters, --mem=48G, 55 min |

### Lecciones Gate 6

13. **`transkun.__version__` no existe**: transkun 2.0.1 no expone `__version__`. No usar en checks.
14. **`createPickle` no es función**: `transkun.createDatasetMaestro` es solo CLI script. Usar `Data.createDatasetMaestroCSV()` directamente.
15. **Precomputar pickles obligatorio**: Race condition entre jobs paralelos. Generar offline, copiar a scratch con MAESTRO.
16. **Pickle naming**: transkun espera `val.pickle`, no `validation.pickle`. Verificar al generar.
17. **Page cache infla MaxRSS**: `cp -r` de 120GB llena page cache → SLURM reporta ~60GB "utilized". No es OOM real. SLURM en Mendieta no mata por cgroups de memoria. `--mem=48G` es funcional.
18. **Audio chunks de largo variable**: Transkun produce chunks stereo de largo variable. `torch.stack` en collate falla. Requiere padding/truncado.

---

## Herramientas operativas

### Skill `/validate-sbatch` (creado 2026-03-05)

5 fases obligatorias antes de cualquier `sbatch`:
1. Static analysis (shebang, set flags, SLURM directives, env order)
2. Path verification (resolución de variables, verificación en disco)
3. Dependency verification (pip packages, conda env, CUDA modules)
4. SLURM dry run (`--test-only`) + queue position
5. Final report con BLOCKERS/WARNINGS/VERDICT

7 opciones estratégicas post-validación (A-G):
- A: Direct submit
- B: Preflight en `short`
- C: Interactive debug (`srun`)
- D: Nabucodonosor (sin cola)
- E: Optimizar `--time` para backfill (`scontrol update`)
- F: Post-submit verification (`scontrol show job`)
- G: Post-completion audit (`seff`)

Ubicación: `~/.claude/skills/validate-sbatch/SKILL.md` + `.claude/skills/` en repo.

**Regla absoluta**: NUNCA ejecutar `sbatch` sin `/validate-sbatch` previo.

---

## 2026-03-08 — Gate 6 preflight v4 + Gate 8 setup

### Gate 6: Fix torch.stack y preflight v4

**Diagnóstico del bug** (Job 1144627 preflight v3):
```
RuntimeError: stack expects each tensor to be equal size,
but got [705600, 2] at entry 0 and [768000, 2] at entry 1
```

**Causa raíz**: MAESTRO v3 tiene archivos con sample rates mixtos:
- 705600 / 16s = 44100 Hz
- 768000 / 16s = 48000 Hz

Chunks de 16 segundos producen distinta cantidad de samples según el archivo.
Transkun resuelve esto en `collate_fn_batching` truncando al mínimo, pero nuestro código
usaba `collate_fn` (lista cruda) y hacía `torch.stack` directo en el training loop.

**Fix aplicado** en `transkun_a4_finetune.py` (líneas 479-484 y 380-384):
```python
slices_raw = [torch.from_numpy(sample['audioSlice']) for sample in batch]
min_len = min(s.shape[0] for s in slices_raw)
audioSlices = torch.stack([s[:min_len] for s in slices_raw]).to(device)
```

Fix en train loop Y en evaluate_transkun. Consistente con patrón nativo de Transkun.

**Preflight v4** (Job 1144693): submitido a `short` en ivb10, RUNNING.
- Resultado pendiente al momento de compactación de contexto.
- Si completa las 100 iters → leer throughput/iter → calibrar --time → submit Exp B + A.

### Gate 8: Conditioned Projections (migración de LOCAL)

LOCAL completó 2/5 brazos (ctrl S=79.2%, pcm S=80.0%). Migración de 3 restantes a UNC.

**Implementado**:
1. **`--resume` en gate5a_proj_cond.py**: Patrón gate43 completo
   - Valida optimizer_state_dict existe en checkpoint
   - Valida que arm coincida con --arm
   - Restaura model (strict=True), optimizer, scheduler state_dicts
   - Pasa start_epoch/initial_best_S a train_loop_gate42
2. **SLURM script** `gate8_conditioned_projections.sh`:
   - Array 0-2: pcd-zero, pcd, pca
   - `--partition=multi`, `--mem=32G`, `--time=2-00:00:00`, `--gres=gpu:1`
   - MAESTRO staging, auto-resume, SIGTERM handler, auto-resubmit
   - `--num-workers 8`, `--structured-eval-epochs 5 10 15 20 25 28 29 30`
3. **Dependencias verificadas**: todas presentes en repo

**Pendiente**: `/validate-sbatch` + submit.

### Merge main→unc

Commit `318bf37`: merge de main (commit `56411ad` Gate 8 migration).
3 conflictos resueltos:
- BITACORA_UNC.md: mantenida en Documents/ (nuestra reorganización)
- RANKING_DESCRIPTORES_UNIFICADO.md: --theirs (LOCAL maneja)
- ROADMAP_UNC.md: --theirs (LOCAL maneja)

### Lecciones aprendidas

19. **MAESTRO v3 tiene mixed sample rates** (44100 Hz y 48000 Hz). Siempre truncar a min_len antes de torch.stack en batches de audio.
20. **`collate_fn` vs `collate_fn_batching`** en Transkun: el primero devuelve lista cruda, el segundo trunca+stack. Si usamos `collate_fn`, hacer truncación manual en training loop.
21. **Transkun usa `torch.utils.checkpoint` internamente**: requiere `requires_grad=True` en el input audio. Sin esto, backward falla con "element 0 of tensors does not require grad". Fix: `.requires_grad_(True)` después de `.to(device)`.

---

## Sesión 2026-03-08 (continuación)

### Gate 6 — Preflight v4 y v5

**Job 1144693 (preflight v4)**: FAILED (43:06 wall-clock)
- Staging MAESTRO: ~35 min (NFS lento por carga del cluster)
- Fix torch.stack: FUNCIONÓ (pasó el batching variable-length)
- **Nuevo error**: `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`
- **Causa raíz**: Transkun usa `torch.utils.checkpoint` internamente. Este mecanismo requiere al menos un input con `requires_grad=True`. Nuestro `audioSlices` venía de `torch.from_numpy()` → sin grad. En el training nativo de Transkun no pasa porque todos los parámetros del modelo están entrenables; en nuestro caso el base_model está congelado.
- **Fix aplicado**: `.requires_grad_(True)` en audioSlices después de `.to(device)`

**Job 1144701 (preflight v5)**: PENDING — incluye ambos fixes (torch.stack + requires_grad)

### Gate 8 — Conditioned Projections: FALLO Y FIX

**Job 1144698** (array 0-2): submitido a multi
- Arms: pcd-zero, pcd, pca
- `--time=1-06:00:00` (30h), `--mem=32G`

**Task 0 (pcd-zero) FAILED** — exit code 2 tras 23 min (22 min staging + crash inmediato)

**Causa raíz**: `set -eo pipefail` + `ls -t $OUTDIR/checkpoint_epoch*.pt 2>/dev/null | head -1`
- `$OUTDIR` no existe en primera ejecución (no hay checkpoints previos)
- `ls` retorna exit code 2 (directorio no encontrado)
- `2>/dev/null` silencia el MENSAJE pero NO el exit code
- `pipefail` propaga exit code 2 del `ls` a través del pipe
- `set -e` mata el script con exit code 2
- **22 minutos de staging de MAESTRO desperdiciados**

**Fix**: `LAST_CKPT=$(ls -t "$OUTDIR"/checkpoint_epoch*.pt 2>/dev/null | head -1 || true)`
- `|| true` garantiza que la sustitución de comando siempre tenga exit 0

Tasks 1 y 2 cancelados preventivamente (iban a fallar igual).

**Job 1144707** (resubmit): array 0-2, `--mem=48G` (subido de 32G por seguridad page cache)
- Validado con `/validate-sbatch`, all PASS, en cola PENDING (Priority)

**Mejora a `/validate-sbatch`**: Agregada sección **1.4 Bash Traps under `set -eo pipefail`**
- Detecta `$(cmd 2>/dev/null | ...)` sin `|| true` como BLOCKER
- Documenta la trampa de `2>/dev/null` (silencia mensaje, no exit code)
- Regla: si un comando PUEDE fallar legítimamente → `|| true` o dentro de `if`

### Lección 22

**`2>/dev/null` NO protege de `pipefail`**: Es la trampa bash más peligrosa en scripts SLURM.
Da una falsa sensación de seguridad. El `2>/dev/null` oculta el mensaje de error pero el exit
code sigue siendo non-zero. Bajo `set -eo pipefail`, cualquier pipeline con un comando que
falle silenciosamente mata el script entero. Todas las command substitutions que pueden fallar
legítimamente (no hay checkpoints, directorio no existe, etc.) DEBEN tener `|| true`.

### Skill nueva: `/slurm-handbook`

Se creó una skill comprehensiva para operar con SLURM en CCAD/UNC. 888 líneas, 14 secciones.

**Ubicación en repo**: `Documents/Skills/slurm-handbook/SKILL.md`

**Contenido**:
1. Arquitectura HPC (login vs compute, hardware Mendieta, particiones, almacenamiento)
2. Script sbatch (template completo con orden obligatorio de setup)
3. 12+ trampas conocidas (todas aprendidas de errores reales en Mendieta)
4. Data staging (`rsync --info=progress2` vs `cp -r`, verificación post-copia)
5. Checkpoint & resume (patrón bash, SIGTERM handler, auto-resubmit con contador)
6. Array jobs (1D, 2D producto cartesiano, logging con %A_%a, requeue individual)
7. Memoria (sizing por tipo de job, page cache vs OOM, profiling)
8. Scheduling (backfill, calibración de --time, `scontrol update` sin perder cola)
9. Monitoreo (squeue formatos, scontrol, seff, interpretar PENDING reasons)
10. Debugging (estrategia de preflight en short, sesiones interactivas, Nabucodonosor)
11. Wiki CCAD (todos los links organizados: infra, tutoriales, primeros pasos, ayuda)
12. Checklist pre-submit (17 ítems de verificación rápida)
13. 4 templates listos para copiar (job simple, job largo con resume, array, preflight)
14. Referencia rápida de comandos

**Para LOCAL**: Instalar copiando a `~/.claude/skills/slurm-handbook/SKILL.md`. Es genérica para cualquier usuario de CCAD, no específica de Phideus. Invocable con `/slurm-handbook`.

---

## Auditoría profunda de scripts (2026-03-08)

Auditoría completa de SLURM + Python para los dos jobs pendientes.

### Fix 1: Gate 8 auto-resubmit (`gate8_conditioned_projections.sh:100-106`)

**Bug**: `[ -f "$OUTDIR/checkpoint_epoch"*.pt ]` — el glob fuera de comillas con `-f` no funciona
cuando hay múltiples checkpoint files (error "too many arguments", silenciado por `2>/dev/null`).
Si el training falla con checkpoints de varias épocas, el auto-resubmit no se dispara.

**Fix**: Reemplazado por `ls ... | wc -l || true` que cuenta archivos correctamente:
```bash
CKPT_COUNT=$(ls "$OUTDIR"/checkpoint_epoch*.pt 2>/dev/null | wc -l || true)
if [ "$CKPT_COUNT" -gt 0 ]; then ...
```

**Nota**: Job 1144707 ya encolado NO tiene este fix (SLURM copia script al submit).
Si falla a medio camino, resubmit manual.

### Fix 2: Gate 6 Exp B — Degradación nunca se aplicaba (CRÍTICO)

**Bug**: `DegradedCollateWrapper` definido en `transkun_degraded.py` pero NUNCA instanciado
ni conectado al DataLoader. Los flags `--degradation noise --level 10` se parseaban pero no
se aplicaban al audio. Los 27 jobs de Exp B habrían entrenado en audio LIMPIO.

**Fix** (2 archivos):
1. `transkun_a4_finetune.py`: `create_transkun_dataloaders` acepta `custom_collate_fn=None`
   (backward compatible, Exp A no se rompe)
2. `transkun_degraded.py`: Instancia `DegradedCollateWrapper` y lo pasa como `custom_collate_fn`
   para noise/lowpass. Branch `data_limit` no usa wrapper (correcto).

**Para el preflight v5**: No afecta crash/éxito (solo entrena en audio degradado vs limpio).
**Para Exp B real**: Fix es OBLIGATORIO antes de submit.

### Lecciones 23-24

23. **`[ -f "path"*.pt ]` con glob externo**: `-f` solo acepta UN archivo. Con múltiples matches,
    falla silenciosamente. Usar `ls ... | wc -l` para contar.

24. **Verificar que wrappers estén CONECTADOS al pipeline**: Un wrapper definido pero nunca
    instanciado es código muerto. Siempre rastrear el flujo de datos desde DataLoader hasta
    el training loop para confirmar que las transformaciones se aplican.

---

## Fix masivo: `cp -r` → `rsync` en todos los scripts (2026-03-08)

### Problema

`rsync -a --info=progress2` estaba documentado como best practice en MEMORY.md y en
`/slurm-handbook` (sección 4), pero **NINGUNO de los 31 scripts SLURM usaba rsync**.
Todos usaban `cp -r` silencioso para staging de MAESTRO (~120GB, 22-35 min sin output).

Esto significaba:
- Imposible saber si el staging está progresando o colgado
- Sin soporte incremental si se interrumpe
- Sin verificación de integridad
- No se puede monitorear remotamente desde login node

### Fix

Reemplazo masivo en 31 scripts con 2 patrones:

| Patrón | Archivos | Antes | Después |
|--------|----------|-------|---------|
| Destino explícito | 7 (Gate 6+, Gate 8) | `cp -r $MAESTRO_SRC $SCRATCH/maestro-v3.0.0` | `rsync -a --info=progress2 $MAESTRO_SRC/ $SCRATCH/maestro-v3.0.0/` |
| Destino implícito | 24 (Gates antiguos) | `cp -r $MAESTRO_SRC $SCRATCH/` | `rsync -a --info=progress2 $MAESTRO_SRC $SCRATCH/` |

**Trailing slash importa en rsync**:
- `SRC/` → copia contenidos al destino
- `SRC` (sin slash) → copia el directorio entero (crea `DEST/basename(SRC)/`)

### Skills actualizadas

- **`/validate-sbatch`**: Nueva sección **1.5 Data Staging** — detecta `cp -r` como WARNING
- **`/slurm-handbook`**: Sección 3 "Trampas conocidas" — elevado de AVISO a CRITICA. Sección 12 checklist actualizada.

### Lección 25

25. **Documentar best practices NO es suficiente — hay que aplicarlas**. `rsync > cp` estaba
    en MEMORY.md desde la primera sesión pero nunca se migró a los scripts reales. Las lessons
    learned deben traducirse en cambios concretos en el código, no solo en documentación.
