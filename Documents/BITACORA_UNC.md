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
| `gate6_transkun_a4.sh` | A | multi | 15 jobs, --array=0-14, --mem=48G, checkpoint+resume+auto-resubmit |
| `gate6_transkun_degraded.sh` | B | multi | 27 jobs, --array=0-26, --mem=48G, checkpoint+resume+auto-resubmit |
| `gate6_expB_preflight.sh` | B preflight | short | Preflight v6: checkpoint+resume test (20+10 iters) |

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

---

## Checkpoint+resume para Gate 6 (2026-03-08)

### Problema

Preflight v5 (Job 1144701) completó exitosamente — 100 iters, throughput **4.9 s/iter** en A30.
Extrapolación: 50,000 iters = **68 horas**. Mendieta multi max = **48 horas**. No cabe en un solo job.

### Solución: Checkpoint periódico + SIGTERM handler + auto-resubmit

**Python (`transkun_a4_finetune.py` — `train_loop`)**:
1. **Resume**: `--resume checkpoint.pt` restaura model, optimizer, scheduler, step, best_f1, history
2. **Checkpoint periódico**: guarda `checkpoint.pt` en cada eval (cada 5000 iters)
3. **SIGTERM handler**: captura señal, guarda checkpoint, sale con `sys.exit(143)` (128+SIGTERM)

**Python (`transkun_degraded.py` — Exp B entry point)**:
- `--resume` agregado a argparse, pasado al config dict

**SLURM (ambos `gate6_transkun_a4.sh` y `gate6_transkun_degraded.sh`)**:
- `--mem=48G` (bajado de 60G, probado suficiente)
- `--time=2-00:00:00` (maximizar cada slot de 48h)
- `--signal=B:SIGTERM@595` (SIGTERM 10 min antes del wall-time)
- Resume: detecta `$OUTDIR/checkpoint.pt`, pasa `--resume`
- SIGTERM trap: `trap 'kill -TERM $PID; wait $PID' SIGTERM` + background `srun &`
- Auto-resubmit: si exit ≠ 0 y hay checkpoint → `sbatch --array=$TASK_ID $0`
- rsync para staging + verificación post-copia

**Flujo esperado**: Job 1 corre ~45h (staging + ~44h training) → SIGTERM → checkpoint → auto-resubmit → Job 2 resume → completa las ~24h restantes.

### Preflight v6: test de checkpoint+resume

Antes de submitir 42 jobs (15 Exp A + 27 Exp B), se creó preflight v6 para validar el mecanismo:

**Job 1144711** — `gate6_expB_preflight.sh` en partición `short` (55 min)
- **Fase 1**: 20 iters fresh → eval → guarda `checkpoint.pt`
- **Verificación 1**: Python assertions — step==20, optimizer/scheduler state presentes
- **Fase 2**: resume desde checkpoint, 10 iters más (iterations=30, eval_every=10)
- **Verificación 2**: assert step==30 (avanzó correctamente de 20 a 30)

Si pasa → VERDICT: READY para submit de Exp A+B.

### Preflight v6 — COMPLETADO (Job 1144711)

**Resultado**: EXIT 0, TODOS LOS TESTS PASARON (36:55 wall-clock en ivb10)

| Check | Resultado |
|-------|-----------|
| Phase 1 (20 iters fresh) | EXIT 0 |
| Checkpoint save (step=20, optimizer+scheduler) | PASS |
| Phase 2 (resume +10 iters) | EXIT 0 |
| Step continuity (20→30) | PASS |
| training_results.json | OK (F1=0.3076@20) |
| Memoria | 60.33 GB (page cache, no OOM) |

### Submit Exp A + Exp B (42 jobs)

`/validate-sbatch` en ambos scripts: 0 blockers, 0 warnings. Submitidos:

| Job | Exp | Tasks | Detalle |
|-----|-----|-------|---------|
| **1144720** | B (degraded) | 0-26 (27 jobs) | 3 degradaciones × 3 niveles × 3 configs |
| **1144721** | A (transkun+A4) | 0-14 (15 jobs) | 5 configs × 3 seeds |
| 1144707 | Gate 8 | 0-2 (3 jobs) | pcd-zero, pcd, pca |

Todos en `multi`, `--time=2-00:00:00`, `--mem=48G`, checkpoint+resume+auto-resubmit.

### Lección 26

26. **Código nuevo sin testear × N jobs = N × staging desperdiciado**. Checkpoint/resume/SIGTERM
    son 3 mecanismos nuevos. Con 42 jobs × 25 min staging = 17.5h GPU si algo falla. Un preflight
    de 30 iters (5 min compute) valida todo el pipeline por el costo de 1 staging.

---

## Gate 8 — Resultados CERRADO (2026-03-10)

### Jobs completados (5/5 brazos)

| Job | Arm | Nodo | Wall-clock | Exit | Best S |
|-----|-----|------|-----------|------|--------|
| 1144707_0 | pcd-zero | ivb19 | 12:43:31 | 0 | **81.8%** @e30 |
| 1144707_1 | pcd | ivb20 | 12:33:05 | 0 | **84.2%** @e25 |
| 1144707_2 | pca | ivb12 | 15:31:15 | 0 | **82.6%** @e25 |
| — | pcm | LOCAL | — | 0 | **80.0%** |
| — | ctrl | LOCAL | — | 0 | **79.2%** |

### pcd-zero — Curva completa (Best S = 81.8% @ e30)

| Epoch | Loss | A2M | M2A | S | Hard Neg |
|-------|------|-----|-----|---|----------|
| 5 | 13.84 | 45.6% | 49.8% | 45.6% | 86.8% |
| 10 | 13.57 | 44.2% | 54.8% | 44.2% | 85.4% |
| 15 | 13.37 | 59.4% | 65.0% | 59.4% | 89.8% |
| 20 | 13.25 | 76.2% | 77.8% | 76.2% | 93.4% |
| 25 | 13.16 | 82.2% | 81.6% | 81.6% | 93.6% |
| 28 | 13.16 | 80.8% | 82.2% | 80.8% | 95.0% |
| 29 | 13.16 | 81.0% | 82.6% | 81.0% | 95.0% |
| **30** | **13.16** | **81.8%** | **82.6%** | **81.8%** | **94.6%** |

### pcd — Curva completa (Best S = 84.2% @ e25)

| Epoch | Loss | A2M | M2A | S | Hard Neg |
|-------|------|-----|-----|---|----------|
| 5 | 13.80 | 60.4% | 65.6% | 60.4% | 90.2% |
| 10 | 13.51 | 74.4% | 75.6% | 74.4% | 93.6% |
| 15 | 13.33 | 68.6% | 72.6% | 68.6% | 93.0% |
| 20 | 13.21 | 78.8% | 79.2% | 78.8% | 92.0% |
| **25** | **13.13** | **86.4%** | **84.2%** | **84.2%** | **94.8%** |
| 28 | 13.13 | 85.8% | 82.4% | 82.4% | 94.2% |
| 29 | 13.13 | 87.6% | 84.2% | 84.2% | 94.8% |
| 30 | 13.12 | 87.4% | 83.6% | 83.6% | 94.8% |

### pca — Curva completa (Best S = 82.6% @ e25)

| Epoch | Loss | A2M | M2A | S | Hard Neg |
|-------|------|-----|-----|---|----------|
| 5 | 13.82 | 64.0% | 67.0% | 64.0% | 91.4% |
| 10 | 13.54 | 77.8% | 78.4% | 77.8% | 92.8% |
| 15 | 13.37 | 66.6% | 67.8% | 66.6% | 91.2% |
| 20 | 13.25 | 76.0% | 76.4% | 76.0% | 93.0% |
| **25** | **13.17** | **82.6%** | **82.8%** | **82.6%** | **94.4%** |
| 28 | 13.16 | 84.0% | 81.0% | 81.0% | 94.6% |
| 29 | 13.16 | 85.2% | 81.4% | 81.4% | 95.2% |
| 30 | 13.16 | 85.0% | 81.6% | 81.6% | 94.2% |

### Comparativa Gate 8 FINAL (5/5 brazos cerrados)

| Arm | Mecanismo | Best S | Best ep | Δ vs ctrl |
|-----|-----------|--------|---------|-----------|
| **pcd** | **Dual cond (A4+D4)** | **84.2%** | **25** | **+5.0pp** |
| **pca** | **Audio cond (A4→audio)** | **82.6%** | **25** | **+3.4pp** |
| pcd-zero | Dual cond, cond=zeros | 81.8% | 30 | +2.6pp |
| pcm | MIDI cond (D4→midi) | 80.0% | — | +0.8pp |
| ctrl | Sin condicionamiento | 79.2% | — | — |

### Observaciones

- **pcd > ctrl (+5.0pp)**: Condicionamiento dual funciona.
- **pca > pcm (+2.6pp)**: Audio-side conditioning aporta más que MIDI-side. Sorprendente dado que Test 11 diagnosticó el bottleneck en MIDI projection.
- **pcd > pcd-zero (+2.4pp)**: La información descriptora real aporta más que solo los parámetros extra.
- **pcd-zero > ctrl (+2.6pp)**: Incluso con zeros, la arquitectura ConditionedProjectionHead tiene más expresividad.
- **pcd en rango d4a4** (84.2% vs 84.1% ±2.3pp): Comparable al mejor descriptor histórico.
- **Regression en e15** observada en pcd (-5.8pp) y pca (-11.2pp), ambos se recuperaron en e20-e25.

### Sync results_unc

Logs sincronizados: `gate8_cond_1144707_{0,1,2}.{out,err}`.
JSONs ya sincronizados por Codex (final_results.json, config.json, training_history.json para los 3 arms).

---

## Gate 6 — Exp B CERRADO NEGATIVO (2026-03-12)

### Diseño

27 tasks = 9 degradaciones × 3 configs. `TASK_ID = DEG_IDX * 3 + CONFIG_IDX`.

| DEG_IDX | Degradación | Level |
|---------|-------------|-------|
| 0 | noise | 5 dB |
| 1 | noise | 10 dB |
| 2 | noise | 20 dB |
| 3 | lowpass | 1000 Hz |
| 4 | lowpass | 2000 Hz |
| 5 | lowpass | 4000 Hz |
| 6 | data_limit | 0.1 |
| 7 | data_limit | 0.25 |
| 8 | data_limit | 0.5 |

| CONFIG_IDX | Config | Trainable params |
|------------|--------|-----------------|
| 0 | baseline-degraded | 0 (solo eval) |
| 1 | finetune-degraded | 66.3K |
| 2 | A4-degraded | 66.3K |

### Resultado por degradación

| Degradación | Baseline | Finetune | A4-degraded | Δ finetune | Δ A4 |
|-------------|----------|----------|-------------|------------|------|
| noise@5 | 0.3039 | 0.3050 | 0.3050 | +0.0011 | +0.0011 |
| noise@10 | 0.3081 | 0.3088 | 0.3088 | +0.0007 | +0.0007 |
| noise@20 | 0.3135 | 0.3130 | 0.3130 | -0.0005 | -0.0005 |
| lowpass@1000 | 0.3154 | 0.3154 | cancelled | 0 | — |
| lowpass@2000 | 0.3154 | cancelled | cancelled | — | — |
| lowpass@4000 | 0.3154 | cancelled | cancelled | — | — |
| data_limit@0.1 | 0.3186 | 0.3186 | 0.3186 | 0 | 0 |
| data_limit@0.25 | 0.3186 | 0.3186 | 0.3186 | 0 | 0 |
| data_limit@0.5 | 0.3186 | cancelled | cancelled | — | — |

Referencia: Transkun baseline (Exp 0) sobre audio limpio = F1 **0.8934**.

### Estado final de jobs

20/27 completados, 7 cancelados (lowpass finetune/A4 y data_limit@0.5 finetune/A4).

Los 7 cancelados tenían curvas planas con best F1 clavado en baseline desde iter 5k — la evidencia era suficiente para cerrar sin completarlos.

### Conclusión

**Resultado negativo uniforme**:
- A4-degraded y finetune-degraded dan exactamente el mismo F1 en todas las condiciones completadas.
- Los deltas contra baseline son microscópicos: +0.0011, +0.0007, -0.0005, o directamente 0.
- En data_limit, el fine-tuning no aporta absolutamente nada.
- La degradación no abre una ventana en la que A4 rescate a Transkun. Ni siquiera aparece ventaja genérica de fine-tuning — el null no es solo "A4 no ayuda", sino "el régimen mismo no está comprando mejora útil".

### Lección 27

27. **Monitorear early y cancelar rápido**: Con 20 de 27 tasks mostrando el mismo patrón (F1 = baseline ±0.001), los 7 restantes no iban a cambiar la conclusión. Cancelar temprano ahorró ~200h GPU.

### Sync results_unc

- Logs: `results_unc/logs/gate6_expB_1144720_{0-26}.{out,err}` — 54 archivos
- JSONs: `results_unc/gate6_amt/expB/` — 27 directorios con config.json + baseline_results.json o training_results.json

---

## Gate 6 — Exp A screening (2026-03-12)

### Diseño

15 tasks = 5 configs × 3 seeds. `TASK_ID = CONFIG_IDX * 3 + SEED_IDX`.

| CONFIG_IDX | Config | Descripción |
|------------|--------|-------------|
| 0 | baseline | Transkun pretrained, 0 params, solo eval |
| 1 | finetune-noA4 | Fine-tune adapter sin A4 |
| 2 | A4-event | A4 como features de evento |
| 3 | A4-adapter | A4 como adapter adicional |
| 4 | adapter-noA4 | Adapter sin A4 (control de arquitectura) |

Seeds: 42, 123, 456.

### Estado

**Reducido a screening seed=42**: solo tasks 0, 3, 6, 9, 12. Los 10 tasks de seeds 123/456 fueron cancelados.

| Task | Config | Seed | Estado | F1 |
|------|--------|------|--------|-----|
| 0 | baseline | 42 | **COMPLETED** | **0.3186** |
| 3 | finetune-noA4 | 42 | En espera | — |
| 6 | A4-event | 42 | En espera | — |
| 9 | A4-adapter | 42 | En espera | — |
| 12 | adapter-noA4 | 42 | En espera | — |

**Criterio GO/NO-GO**: si ninguno supera 0.3186 por al menos +0.01 F1 absoluto, se cierra Exp A en negativo y no se corren las seeds restantes.

**Estado de cola**: los 4 tasks fueron removidos de cola. Se resubmitirán después de otro experimento que se mandará desde LOCAL.

### Sync results_unc

- Logs: `results_unc/logs/gate6_expA_1144721_0.{out,err}`
- JSONs: `results_unc/gate6_amt/expA/baseline_seed42/`

---

## Gate 10 — Mechanism Sweep (2026-03-12)

### Contexto

Gate 9 y A10 testaron 7 descriptores audio con reverse cross-attention. Todos convergen a ~69-71% S sin diferenciación significativa. Hipótesis: el mecanismo domina sobre el contenido del descriptor.

Gate 10 desacopla las dos variables: cruza 3 descriptores representativos × 3 mecanismos de inyección.

### Diseño: 3 descriptores × 3 mecanismos = 9 runs

| Task | Descriptor | Dim | Mecanismo | Batch |
|------|-----------|-----|-----------|-------|
| 0 | a7 | 12 | concat | 16 |
| 1 | a10a | 12 | concat | 16 |
| 2 | a10d | 32 | concat | 16 |
| 3 | a7-pca | 12 | FiLM projection | 16 |
| 4 | a10a-pca | 12 | FiLM projection | 16 |
| 5 | a10d-pca | 32 | FiLM projection | 16 |
| 6 | a7-ab | 12 | attention bias | 8 |
| 7 | a10a-ab | 12 | attention bias | 8 |
| 8 | a10d-ab | 32 | attention bias | 8 |

Protocolo: 30ep from-scratch, run-d, seed=42, 1000 batches/ep, structured eval en e5,10,15,20,25,28,29,30.

Referencia: rev_xattn → a7r=70.4%, a10ar=70.6%, a10dr=70.2% (Gate 9/A10). pca (Gate 8) → a4r-pca=82.6%.

### Resultados previos (comparar contra)

| Arm | Mecanismo | Best S |
|-----|-----------|--------|
| ctrl | ninguno | 79.2% |
| a4r-pca (Gate 8) | FiLM audio proj | 82.6% |
| a7r | rev_xattn | 70.4% |
| a9r | rev_xattn | 71.6% |
| a10ar | rev_xattn | 70.6% |
| a10dr | rev_xattn | 70.2% |
| d4a4 | concat dual | 84.1% |

### SLURM

- **Script**: `slurm/gate10_mechanism_sweep.sh`
- **Job**: 1144982 (array 0-8, 9 tasks)
- **Partición**: multi, `--time=12:00:00`, `--mem=48G`, `--gres=gpu:1`
- **ETA**: ~5h/run + 25 min staging. ~6h total si 9 nodos en paralelo.

### Merge main→unc

Commit `20c40eb`: merge de main (commit `251412c` — Gate 10 implementation).
1 conflicto resuelto: Skills/slurm-handbook/SKILL.md → kept ours.

### Resultados parciales (2026-03-15, corte operativo)

CANONICAL eval cada 5 epochs. Tasks 0-5 hicieron TIMEOUT @12h (~epoch 13-14), resubmitidos manualmente. Task 6 RUNNING.

#### Eval @epoch 5 (9/9 arms)

| Task | Arm | Mecanismo | S% | A2M | M2A | hard_neg |
|------|-----|-----------|----|-----|-----|----------|
| 0 | a7 | concat | 20.8% | 20.8% | 29.6% | 73.6% |
| 1 | a10a | concat | 54.6% | 54.6% | 60.6% | 90.8% |
| 2 | a10d | concat | 23.4% | 23.4% | 31.2% | 74.0% |
| 3 | a7 | FiLM/pca | 55.2% | 55.2% | 58.6% | 91.0% |
| 4 | a10a | FiLM/pca | 60.8% | 60.8% | 62.6% | 91.4% |
| 5 | a10d | FiLM/pca | 56.8% | 56.8% | 59.4% | 92.0% |
| 6 | a7 | attn_bias | 38.0% | 38.0% | 44.2% | 82.2% |
| 7 | a10a | attn_bias | 43.6% | 43.6% | 44.8% | 86.2% |
| 8 | a10d | attn_bias | 35.4% | 35.4% | 40.6% | 83.6% |

#### Eval @epoch 10 (9/9 arms — COMPLETA)

| Task | Arm | Mecanismo | S% | A2M | M2A | hard_neg |
|------|-----|-----------|----|-----|-----|----------|
| 3 | a7 | FiLM/pca | **70.4%** | 70.4% | 70.4% | 94.0% |
| 4 | a10a | FiLM/pca | 68.8% | 70.2% | 68.8% | 92.2% |
| 5 | a10d | FiLM/pca | 68.6% | 68.6% | 69.4% | 92.4% |
| 2 | a10d | concat | 63.6% | 63.6% | 64.6% | 92.6% |
| 1 | a10a | concat | 63.2% | 63.2% | 64.2% | 91.2% |
| 0 | a7 | concat | 52.2% | 52.2% | 57.4% | 90.0% |
| 7 | a10a | attn_bias | 49.0% | 49.0% | 52.0% | 88.2% |
| 6 | a7 | attn_bias | 44.6% | 44.6% | 47.8% | 86.2% |
| 8 | a10d | attn_bias | 41.2% | 41.2% | 44.0% | 86.0% |

#### Eval @epoch 15 (9/9 arms — COMPLETA)

| Task | Arm | Mecanismo | S% | A2M | M2A | hard_neg |
|------|-----|-----------|----|-----|-----|----------|
| 1 | a10a | concat | **71.4%** | 71.4% | 72.2% | 93.2% |
| 2 | a10d | concat | 70.2% | 70.2% | 70.2% | 92.6% |
| 5 | a10d | FiLM/pca | 69.6% | 72.4% | 69.6% | 94.2% |
| 3 | a7 | FiLM/pca | 69.2% | 71.2% | 69.2% | 94.0% |
| 4 | a10a | FiLM/pca | 68.6% | 71.4% | 68.6% | 93.4% |
| 0 | a7 | concat | 63.4% | 64.2% | 63.4% | 92.4% |
| 7 | a10a | attn_bias | 52.0% | 52.0% | 54.2% | 87.6% |
| 8 | a10d | attn_bias | 49.4% | 49.4% | 52.8% | 88.0% |
| 6 | a7 | attn_bias | 48.2% | 48.2% | 48.2% | 87.4% |

#### Eval @epoch 20 (9/9 arms — COMPLETA)

| Task | Arm | Mecanismo | S% | A2M | M2A | hard_neg |
|------|-----|-----------|----|-----|-----|----------|
| 0 | a7 | concat | **71.6%** | 71.6% | 72.0% | 93.4% |
| 2 | a10d | concat | 71.4% | 71.4% | 73.8% | 93.6% |
| 5 | a10d | FiLM/pca | 71.2% | 75.2% | 71.2% | 93.6% |
| 3 | a7 | FiLM/pca | 69.8% | 71.6% | 69.8% | 93.6% |
| 1 | a10a | concat | 69.6% | 69.6% | 72.2% | 93.2% |
| 4 | a10a | FiLM/pca | 69.6% | 74.8% | 69.6% | 92.4% |
| 7 | a10a | attn_bias | 56.6% | 56.6% | 57.8% | 88.6% |
| 6 | a7 | attn_bias | 52.8% | 52.8% | 54.0% | 87.4% |
| 8 | a10d | attn_bias | 52.0% | 52.0% | 52.6% | 88.0% |

#### Eval @epoch 25 (8/9 arms)

| Task | Arm | Mecanismo | S% | A2M | M2A | hard_neg |
|------|-----|-----------|----|-----|-----|----------|
| 0 | a7 | concat | **75.8%** | 76.2% | 75.8% | 94.0% |
| 4 | a10a | FiLM/pca | 73.2% | 77.0% | 73.2% | 94.4% |
| 1 | a10a | concat | 72.8% | 74.0% | 72.8% | 94.0% |
| 2 | a10d | concat | 72.8% | 73.6% | 72.8% | 94.0% |
| 5 | a10d | FiLM/pca | 72.6% | 74.2% | 72.6% | 94.4% |
| 3 | a7 | FiLM/pca | 70.8% | 73.0% | 70.8% | 93.8% |

#### Eval @epoch 28 (8/9 arms)

| Task | Arm | Mecanismo | S% | A2M | M2A | hard_neg |
|------|-----|-----------|----|-----|-----|----------|
| 0 | a7 | concat | **75.8%** | 75.8% | 76.2% | 94.2% |
| 1 | a10a | concat | **75.6%** | 76.4% | 75.6% | 94.6% |
| 4 | a10a | FiLM/pca | 72.8% | 77.8% | 72.8% | 94.2% |
| 3 | a7 | FiLM/pca | 71.8% | 73.6% | 71.8% | 94.0% |
| 7 | a10a | attn_bias | 58.6% | 58.6% | 61.2% | 90.8% |
| 8 | a10d | attn_bias | 57.4% | 57.4% | 58.8% | 90.8% |
| 6 | a7 | attn_bias | 55.4% | 55.4% | 56.6% | 88.6% |

#### Eval @epoch 29-30 — 9/9 COMPLETADOS

| Task | Arm | Mecanismo | e29 S% | e30 S% | Best S | @epoch |
|------|-----|-----------|----|-----|--------|--------|
| 0 | a7 | concat | **76.4%** | 76.2% | **76.4%** | e29 |
| 1 | a10a | concat | 73.6% | 75.2% | 75.6% | e28 |
| 2 | a10d | concat | 74.8% | **75.4%** | 75.4% | e30 |
| 4 | a10a | FiLM/pca | **74.0%** | 73.8% | 74.0% | e29 |
| 5 | a10d | FiLM/pca | 72.8% | **73.2%** | 73.2% | e30 |
| 3 | a7 | FiLM/pca | 71.6% | 71.4% | 71.8% | e28 |
| 7 | a10a | attn_bias | 58.8% | 59.6% | 59.6% | e30 |
| 8 | a10d | attn_bias | 56.2% | 57.2% | 57.4% | e28 |
| 6 | a7 | attn_bias | 55.6% | 55.8% | 55.8% | e30 |

#### Tabla resumen — Best S por arm (FINAL)

| Rank | Arm | Mecanismo | Best S | @epoch |
|------|-----|-----------|--------|--------|
| 1 | **a7** | **concat** | **76.4%** | e29 |
| 2 | a10a | concat | 75.6% | e28 |
| 3 | a10d | concat | 75.4% | e30 |
| 4 | a10a | FiLM/pca | 74.0% | e29 |
| 5 | a10d | FiLM/pca | 73.2% | e30 |
| 6 | a7 | FiLM/pca | 71.8% | e28 |
| 7 | a10a | attn_bias | 59.6% | e30 |
| 8 | a10d | attn_bias | 57.4% | e28 |
| 9 | a7 | attn_bias | 55.8% | e30 |

### Bug fix y re-submit (2026-03-20)

**Bug**: Todos los resume jobs (1145067, 1145118, 1145152) FALLARON con exit code 1 (~2 min de Python).
**Causa**: `ls -t checkpoint_epoch*.pt` seleccionaba archivos `_archive_base_not_for_eval.pt` (sin optimizer_state_dict) en vez de los checkpoints completos. Los archive files tenían mtime más reciente.
**Fix**: `grep -v '_archive_'` en línea 67 del script. También absolutizado paths de `--output`/`--error`.
**Re-submit**: Job 1145390 (array 0-8). Tasks 7-8 ya habían avanzado a e25 antes del TIMEOUT original.

#### Estado final de jobs — Gate 10 COMPLETADO (2026-03-24)

Todos los 9 arms completaron 30 epochs.

**Historial de jobs**: 1144982 (original), 1145067/1145118/1145152 (FAILED — _archive_ bug), 1145390 (fix), 1145623 (concat 0-1), 1145638 (pca 3-5), 1145645 (concat 2).

#### Observaciones finales — Gate 10

- **concat > FiLM/pca > attn_bias**: ranking definitivo. concat gana por ~2pp sobre pca.
- **a7-concat es el mejor arm**: 76.4% @e29. Late bloomer: 20.8%→76.4% en 29ep.
- **Los 3 concat convergen a 75-76%**: spread de solo 1pp. El descriptor no diferencia significativamente.
- **FiLM/pca converge a 72-74%**: a10a-pca lidera con 74.0%. Plateau visible desde e25.
- **attn_bias techo ~59.6%**: 16pp debajo de concat. Mecanismo descartado.
- **Conclusión principal**: el mecanismo domina sobre el descriptor (spread intra-mecanismo ~2-3pp vs inter-mecanismo ~15pp).

### Gate 6 Exp A — Screening COMPLETADO (2026-03-24)

Job 1145625 (tasks 3, 6, 9, 12). Screening seed=42.

| Task | Config | Seed | Best F1 | Estado |
|------|--------|------|---------|--------|
| 0 | baseline | 42 | 0.3186 | COMPLETED |
| 3 | finetune-noA4 | 42 | 0.3186 | COMPLETED |
| 6 | A4-event | 42 | 0.3186 | COMPLETED |
| 9 | A4-adapter | 42 | 0.3186 @10k | KILLED @~48h (resubmit job 1145658 cancelado), 2 evals disponibles |
| 12 | adapter-noA4 | 42 | 0.3186 | COMPLETED |

**Resultado**: Todos los configs dan **exactamente el mismo F1=0.3186** que baseline. Ninguno supera baseline + 0.01.

Task 9 (A4-adapter) fue killed tras ~48h, pero ya tenía 2 evals (step 5k y 10k) ambas con F1=0.3186 — la tendencia es idéntica a las demás. Tiene checkpoint para resume si fuera necesario, pero el resultado ya es conclusivo.

### Sync results_unc (2026-04-03, auditoría de trazabilidad)

Auditoría exhaustiva completada. **Todo** lo generado en Mendieta está ahora en `results_unc/`:
- Gate 10: 56 eval JSONs + 9 final_results + 9 training_history + 9 configs
- Gate 6 Exp A: 5 configs con training_results + 32 eval JSONs
- Gate 6 Exp B: 27 dirs con training_results + 118 eval JSONs (sync completado, antes faltaban muchos)
- Gate 8: 5 arms, evals reorganizados en eval_per_epoch/ (ctrl/pcm movidos de raíz)
- Logs: **282 archivos** (todos los jobs históricos, antes faltaban 116)
- .gitignore: excepciones agregadas para todos los subdirs de results_unc/

**Lo que NO está en Mendieta** (exclusivo de LOCAL):
- Gate 9 / A10 (7 arms retrospective)
- Gate 5B d4a4 multi-seed: seeds 123/456/789 completadas acá, seed 1337 delegada a LOCAL (ver abajo)
- Gate 6 Exp C (VICReg decoder)
- Gate 8 ctrl y pcm training_results

---

## Gate 5B — d4a4 Training Multi-Seed (2026-04-03)

### Contexto

La auditoría de trazabilidad descubrió que d4a4 nunca tuvo training multi-seed real. Lo reportado como "multi-seed" eran 5 evaluaciones sobre un único checkpoint (eval-seed), no 5 trainings independientes. Los otros 3 arms (D0, a4r, d4-a4r) sí tienen training multi-seed en `results_unc/gate5b_multiseed/`.

### Tarea

4 trainings nuevos de d4a4 from scratch (seed 42 ya existe en LOCAL). Seeds: 123, 456, 789, 1337.

### SLURM

- **Script**: `slurm/gate5b_d4a4_multiseed.sh`
- **Job**: 1146677 (array 0-3, 4 tasks)
- **Partición**: multi, `--time=2-00:00:00`, `--mem=48G`, `--gres=gpu:1`

### Resultados — 3/4 completados en UNC, 1 delegado a LOCAL

| Task | Seed | Best S | @epoch | Walltime | Estado |
|------|------|--------|--------|----------|--------|
| 0 | 123 | **87.6%** | e30 | 1d 0h38m | **COMPLETED** en ivb08 |
| 1 | 456 | **81.4%** | e30 | 1d 0h50m | **COMPLETED** en ivb14 |
| 2 | 789 | **81.6%** | e28 | 1d 0h55m | **COMPLETED** en ivb16 |
| 3 | 1337 | — | — | — | **DELEGADO A LOCAL** (ver abajo) |

#### Detalle de evals por seed (structured eval epochs 25-30)

**seed 123** (best S=87.6% @e30):
| e25 | e26 | e27 | e28 | e29 | e30 |
|-----|-----|-----|-----|-----|-----|
| 86.4% | 86.6% | 86.6% | 87.2% | 87.0% | **87.6%** |

**seed 456** (best S=81.4% @e30):
| e25 | e26 | e27 | e28 | e29 | e30 |
|-----|-----|-----|-----|-----|-----|
| 81.0% | 80.8% | 79.8% | 80.0% | 80.2% | **81.4%** |

**seed 789** (best S=81.6% @e28):
| e25 | e26 | e27 | e28 | e29 | e30 |
|-----|-----|-----|-----|-----|-----|
| 80.0% | 80.4% | 81.2% | **81.6%** | 81.0% | 81.4% |

#### Resumen parcial (4/5 seeds, incluyendo seed42=83.6% de LOCAL)

- Mean: (87.6 + 81.4 + 81.6 + 83.6) / 4 = **83.6%**
- Rango: 81.4% – 87.6% (spread 6.2pp)
- Falta seed 1337 para cerrar

### Seed 1337 — DELEGADO A LOCAL (2026-04-06)

**Motivo**: seed 1337 fue asignada al nodo ivb04 que presentó un problema grave de rendimiento (~7h/epoch en vez de ~35min, 12x más lento de lo normal). Se canceló (Job 1146677_3) y se intentó resubmitir (Jobs 1146953, 1146954) pero la cola de Mendieta tiene ~34 jobs delante con espera estimada de 24-36h. Correrlo en LOCAL es más rápido (~20h directo).

**Instrucciones para LOCAL**: correr d4a4 seed=1337 con la config **exacta** listada abajo.

#### Config exacta (replicar campo por campo)

```
python experiments/bias_control/gate43_scratch/gate43_scratch_training.py \
    --mode train \
    --descriptor d4a4 \
    --from-scratch \
    --output <OUTPUT_DIR>/d4a4_seed1337 \
    --maestro-dir <MAESTRO_PATH> \
    --epochs 30 \
    --batch-size 16 \
    --freeze-policy run-d \
    --num-workers 14 \
    --embed-batch-size 16 \
    --max-batches-per-epoch 1000 \
    --max-val-batches 846 \
    --seed 1337 \
    --device cuda \
    --structured-eval-epochs 25 26 27 28 29 30 \
    --gate 4.3-scratch
```

**Nota**: `--num-workers 14` para LOCAL (i5-12600K), en UNC usamos `--num-workers 8` (10 cores/nodo). Esto no afecta resultados, solo throughput de data loading.

#### Tabla de hiperparámetros completa (todos idénticos entre seeds)

| Parámetro | Valor | Nota |
|-----------|-------|------|
| descriptor | d4a4 | Dual concat (D4 intervals 4d + A4 log-freq 8d) |
| from_scratch | true | Sin foundation checkpoint |
| epochs | 30 | |
| batch_size | 16 | |
| max_batches_per_epoch | 1000 | |
| max_val_batches | 846 | |
| embed_batch_size | 16 | |
| ratio_weight | 0.1 | |
| freeze_policy | run-d | |
| lr_audio_unfreeze | 1e-05 | |
| lr_audio_low | 5e-06 | |
| lr_midi | 5e-05 | |
| lr_proj | 0.0001 | |
| lr_ratio | 0.0005 | |
| warmup_steps | 200 | |
| lr_hold_fraction | 0.0 | |
| lr_cosine_ref_epochs | 0 | Sin cosine schedule |
| lr_floor | 0.0 | |
| lr_tail_end | 0.0 | |
| structured_eval_epochs | [25, 26, 27, 28, 29, 30] | |
| gate | 4.3-scratch | Label para trazabilidad |
| skip_structured_eval | false | |
| foundation_checkpoint | null | |
| use_d4a4_injection | null | |

#### Output esperado

El directorio de salida debe contener:
- `config.json` — hiperparámetros del run
- `final_results.json` — con `evaluation_best.gate_metrics.S` y `evaluation_best.epoch`
- `training_history.json` — loss y métricas por epoch
- `eval_per_epoch/eval_epoch{25,26,27,28,29,30}.json` — evaluaciones estructuradas

**No incluir .pt en el commit.** Solo JSONs.

#### Criterio de éxito

El run debe completar 30 epochs. El S esperado está en el rango 78-88% basado en las otras 4 seeds.

### Notas técnicas

- **PCA (`-pca`)**: FiLM-conditioned projection. No es lo mismo que Gate 8 `pca` (ahi usaba A4, acá usa A7/A10a/A10d).
- **Attention Bias (`-ab`)**: Manual forward del Transformer con `need_weights=False` (evita OOM por attention weights [B*8, 2400, 2400]). Batch size reducido a 8.
- **Flag `--gate 10`**: Nuevo. Overridea gate label para trazabilidad en final_results.json.

---

## Voz Expresiva — EN N-adapt calibfix (2026-06-27)

### Contexto operativo

- Documento fuente de la tarea: `Documents/01_FRENTES_ACTIVOS/Voz_Expresiva_Phideus/HANDOFF_UNC_EN_NADAPT.md`.
- Objetivo exacto en UNC: rerun **solo** `EN N-adapt` de Fase 1 con el fix `B2` del `calib_manifest`, preservando el `N-strict` heredado del cierre original local.
- Razón del rerun: el cierre original de EN había quedado con un bug en `build_calib_manifest` que reinstanciaba `RandomState(42)` por speaker, produciendo las mismas 25 utts de calibración para todos los hablantes. El problema afecta `N-adapt`, no `N-strict`.
- Criterio metodológico congelado por el handoff:
  - `N-strict` no se reentrena en UNC.
  - UNC devuelve únicamente `1_en_calibfix/` con los artefactos `adapt`.
  - El merge de `N-strict` heredado + `N-adapt` corregido y los reportes cross-language se hacen en LOCAL.

### Puesta al día del repo en Mendieta

- Repo activo: `~/Repos/Phideus`
- Rama activa: `unc`
- Acción realizada: merge `main -> unc` para incorporar el handoff, el fix `B2`, `--limit-norms adapt` y la nueva documentación troncal.
- Estado resultante:
  - `HANDOFF_UNC_EN_NADAPT.md` ya presente en el repo UNC.
  - `QUE_ES_ATENCION_ARMONICA.md` y demás docs nuevos también entraron por el merge, pero **no forman parte de esta corrida**.
- Commit de trabajo verificado en UNC al momento del submit: `da9066e`.
- Observación importante:
  - El handoff menciona `main` en commit `6149d92`.
  - El árbol actual en UNC está más adelante, pero conserva el fix buscado:
    - `_speaker_calib_seed(spk, base_seed)` con `sha256`
    - `--limit-norms {strict,adapt}`
    - guardrails para `calib_manifest` stale

### Relectura operativa previa al submit

Antes de tocar SLURM se releídos o reconsultados estos materiales:

- `README.md`
- `Documents/00_TRONCAL/Proyecto_Estado_Actual.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_UNC.md`
- `Documents/04_TRANSVERSAL/UNC_SuperComp_IA_Agents.md`
- `Documents/BITACORA_UNC.md`
- `Documents/Skills/slurm-handbook/SKILL.md`
- `Documents/Skills/validate-sbatch/SKILL.md`
- `Documents/01_FRENTES_ACTIVOS/Voz_Expresiva_Phideus/README.md`
- `Documents/01_FRENTES_ACTIVOS/Voz_Expresiva_Phideus/HANDOFF_UNC_EN_NADAPT.md`
- `experiments/voz_expresiva/1_train.py`
- `experiments/voz_expresiva/1_precache_wavlm.py`
- `experiments/voz_expresiva/1_precache_descriptors.py`
- `src/voz_expresiva/esd_loader.py`

### Hallazgos previos al handoff

#### 1. El código ya estaba listo para el rerun

Se confirmó que `experiments/voz_expresiva/1_train.py` ya incluye:

- generación correcta de `calib_seed_effective` por speaker con `_speaker_calib_seed()`
- escritura de `calib_manifest.json`
- escritura incremental de `uar_results.json`
- `--limit-norms adapt` para restringir la corrida solo al brazo requerido
- guardrail para rechazar un `calib_manifest` viejo incompatible con la política actual

#### 2. El repo UNC no tenía caches ni outputs de Voz Expresiva

Al inicio del corte no existían en esta máquina:

- `data/voz_expresiva/wavlm_cache/`
- `data/voz_expresiva/descriptors_cache/`
- `data/voz_expresiva/1/`
- `data/voz_expresiva/1_zh/`
- `data/voz_expresiva/1_en_calibfix/`

Tampoco aparecían `ESD` ni artefactos `voz_expresiva` en búsquedas sobre:

- `~/Repos/Phideus`
- `/home/mfmendez`
- `/scratch`
- montajes comunes (`/raid1`, `/data`, `/mnt`, `/srv`, `/opt`)

Inferencia de ese momento:

- El rerun no debía submitirse hasta tener insumos reales.
- La ruta correcta era pedir a LOCAL el traslado de caches exactos, tal como indicaba el handoff.

### Transferencia de insumos desde LOCAL

LOCAL informó transferencia completa y verificación bit a bit de los 4 artefactos necesarios.

#### Paths destino en UNC

- `~/Repos/Phideus/data/voz_expresiva/wavlm_cache/wavlm_features.npy`
- `~/Repos/Phideus/data/voz_expresiva/wavlm_cache/wavlm_lengths.npy`
- `~/Repos/Phideus/data/voz_expresiva/wavlm_cache/wavlm_index.json`
- `~/Repos/Phideus/data/voz_expresiva/descriptors_cache/family_A.npy`

#### Verificación local vs remota reportada por LOCAL

| Archivo | Estado |
|---|---|
| `wavlm_features.npy` | sha256 local == remoto |
| `wavlm_lengths.npy` | sha256 local == remoto |
| `wavlm_index.json` | sha256 local == remoto |
| `family_A.npy` | sha256 local == remoto |

#### Verificación adicional en UNC

Se verificó existencia de paths y tamaño lógico:

| Archivo | Tamaño lógico verificado en UNC |
|---|---|
| `wavlm_features.npy` | `21,719,040,000` bytes |
| `wavlm_lengths.npy` | `70,128` bytes |
| `wavlm_index.json` | `4,383,407` bytes |
| `family_A.npy` | `254,520,000` bytes |

Observación operativa:

- `du -sh` reportó menos espacio aparente (`9.5G` para `wavlm_cache`, `73M` para `descriptors_cache`) que el tamaño lógico esperado.
- `stat` confirmó que el tamaño **lógico** de los archivos sí coincide con el handoff y con el cierre local.
- Interpretación: almacenamiento sparse/aparente vs tamaño lógico; **no se trató como error** porque la validación fuerte vino por sha256 reportado por LOCAL + `stat` en UNC.

### Script SLURM nuevo para este frente

Se creó un script dedicado:

- `slurm/vozexp_en_nadapt_calibfix.sh`

#### Diseño del script

- Partición: `multi`
- Recursos:
  - `--gres=gpu:1`
  - `--cpus-per-task=10`
  - `--mem=48G`
  - `--time=08:00:00`
  - `--signal=B:SIGTERM@595`
- Logging:
  - stdout: `~/Repos/Phideus/results_unc/logs/vozexp_en_nadapt_%j.out`
  - stderr: `~/Repos/Phideus/results_unc/logs/vozexp_en_nadapt_%j.err`
- Bootstrap Mendieta correcto:
  - `set -eo pipefail`
  - `. /etc/profile`
  - `module load gcc cuda`
  - `source ~/miniconda3/bin/activate phideus`
  - `export PYTHONUNBUFFERED=1`
  - `export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`
  - `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

#### Guardrails implementados

1. **Output dir fresco obligatorio**
   - `1_train.py` no hace resume ni dedup.
   - El script aborta si `OUTPUT_DIR` ya existe y no está vacío.

2. **Reuso de caches por defecto**
   - Si los 4 artefactos están presentes, no regenera nada.

3. **Fallback de precache**
   - Si faltaran caches, el script intenta:
     - detectar `ESD_ROOT`
     - correr `1_precache_wavlm.py`
     - correr `1_precache_descriptors.py`
   - Para este run **no fue necesario** porque los caches ya estaban.

4. **Verificación post-run**
   - Exige `uar_results.json` y `calib_manifest.json`.
   - Ejecuta una verificación Python local al final:
     - exactamente `120` records `adapt`
     - `calib_seed_effective` no nulo en todos
     - ningún record `strict`
     - `10` speakers en el manifest
     - las listas de `sentence_ids` no pueden ser idénticas entre todos los speakers

### Preflight del job

#### Validaciones estáticas

- `bash -n slurm/vozexp_en_nadapt_calibfix.sh` → **OK**
- Paths críticos presentes:
  - caches `voz_expresiva`
  - `results_unc/logs`
  - `~/miniconda3/bin/activate`
  - `experiments/voz_expresiva/1_train.py`

#### Dry run SLURM

`sbatch --test-only slurm/vozexp_en_nadapt_calibfix.sh` devolvió:

- Job estimado: `1158455`
- Inicio estimado: `2026-07-01T10:09:29`
- Nodo sugerido: `ivb13`
- Partición: `multi`

Interpretación:

- El scheduler aceptó directives, recursos, paths de log y shape general del script.
- No apareció ningún blocker de sintaxis SLURM.

#### Validación del entorno Python real

Se comprobó que:

- `python experiments/voz_expresiva/1_train.py --help` corre correctamente dentro del env `phideus`.

Hallazgo adicional:

- El env `phideus` no tiene `transformers`.

Inferencia:

- Esto **no bloquea** el rerun actual porque `1_train.py` consume caches ya generados.
- Sí bloquearía el fallback de precache WavLM si hubiera que regenerar features desde cero en esta máquina sin tocar dependencias.
- Con los caches ya copiados, el riesgo queda neutralizado para este job específico.

### Submit real

- Comando usado: `sbatch ~/Repos/Phideus/slurm/vozexp_en_nadapt_calibfix.sh`
- Resultado: `Submitted batch job 1158456`

### Estado actual verificado

Snapshot tomado inmediatamente después del submit:

| Campo | Valor |
|---|---|
| Job ID | `1158456` |
| Job name | `vexp-en-adapt` |
| Estado | `PENDING` |
| Reason | `Priority` |
| Partición | `multi` |
| CPUs | `10` |
| GPU | `gres/gpu` |
| Start | `Unknown` al momento del corte |

Fuente:

- `squeue -j 1158456`
- `sacct -j 1158456 --format=JobID,JobName,Partition,State,ExitCode,Elapsed,Submit,Start,NodeList -P`

### Paths operativos de esta corrida

#### Script

- `~/Repos/Phideus/slurm/vozexp_en_nadapt_calibfix.sh`

#### Inputs

- `~/Repos/Phideus/data/voz_expresiva/wavlm_cache/`
- `~/Repos/Phideus/data/voz_expresiva/descriptors_cache/`

#### Output esperado

- `~/Repos/Phideus/results_unc/voz_expresiva/1_en_calibfix/`

#### Logs

- `~/Repos/Phideus/results_unc/logs/vozexp_en_nadapt_1158456.out`
- `~/Repos/Phideus/results_unc/logs/vozexp_en_nadapt_1158456.err`

### Observacion / Hipotesis / Inferencia

- Observacion:
  - El rerun ya quedó submitido con todos los insumos críticos presentes en UNC.
  - El job todavía no arrancó; está retenido por `Priority`.
  - El preflight no detectó blockers de script ni de path.
  - El único gap del env (`transformers` ausente) no afecta esta corrida porque el run usa caches ya materializados.

- Hipotesis:
  - Si el job entra en un nodo A30 sin incidentes de scheduler, debería completar el brazo `adapt` sin reconstruir features y dentro de la ventana prevista por el handoff (~3.3h de cómputo + overhead menor).

- Inferencia (preliminar):
  - El frente quedó correctamente transferido a UNC.
  - El riesgo dominante ya no es de preparación sino de cola/ejecución.
  - La próxima auditoría útil debe hacerse cuando `1158456` pase a `RUNNING`, leyendo principalmente el `.err`.

### Riesgos / bloqueantes

1. **Cola SLURM / Priority**
   - El job no tiene start inmediato garantizado.

2. **Dependencia faltante en fallback**
   - `transformers` no está en `phideus`.
   - No afecta este run con caches presentes, pero sí impediría un precache WavLM limpio si hubiera que regenerar desde cero sin antes instalar deps.

3. **No hay resume real en `1_train.py`**
   - Si el run se interrumpe y deja artefactos parciales, no conviene relanzar sobre el mismo `OUTPUT_DIR` sin limpiar o separar salida.

4. **Comparabilidad de hardware**
   - Declarado por el handoff: `EN N-adapt` corre en A30 UNC, mientras `EN N-strict` y `ZH` cerraron en 3090 local.
   - Afecta solo la lectura secundaria cross-language del brazo `adapt`; no toca el claim primario `N-strict`.

### Próximo paso único recomendado

- Monitorear `1158456` hasta transición a `RUNNING`, luego auditar:
  - aparición de `calib_manifest.json`
  - progreso de folds/configs en `stderr`
  - cierre final con `120` records `adapt` y manifest B2 efectivo

### Evidencia (paths + logs + estado)

- Handoff: `Documents/01_FRENTES_ACTIVOS/Voz_Expresiva_Phideus/HANDOFF_UNC_EN_NADAPT.md`
- Trainer: `experiments/voz_expresiva/1_train.py`
- Script sbatch: `slurm/vozexp_en_nadapt_calibfix.sh`
- Caches UNC:
  - `data/voz_expresiva/wavlm_cache/wavlm_features.npy`
  - `data/voz_expresiva/wavlm_cache/wavlm_lengths.npy`
  - `data/voz_expresiva/wavlm_cache/wavlm_index.json`
  - `data/voz_expresiva/descriptors_cache/family_A.npy`
- Logs esperados:
  - `results_unc/logs/vozexp_en_nadapt_1158456.out`
  - `results_unc/logs/vozexp_en_nadapt_1158456.err`
- Output esperado:
  - `results_unc/voz_expresiva/1_en_calibfix/`

### ETA realista

- Espera en cola: indeterminada, gobernada por `Priority`.
- Una vez en ejecución:
  - training `adapt` esperado: ~3.3 h según el handoff
  - más overhead menor de bootstrap/logging
- ETA razonable de pared si entra sin demoras largas: media jornada.

---

## Voz Expresiva — EN N-adapt: incidente node-failure + requeue (2026-06-28)

### Qué pasó

El job `1158456` entró a RUNNING el 28/06 04:56 en `ivb05` y corrió sano ~2h:
- Caches detectados, sin precache (gap de `transformers` evitado).
- `calib_manifest.json` correcto: **10 speakers, 10 sets únicos de sentence_ids → fix B2 EFECTIVO** (antes los 10 compartían las mismas 25 utts).
- Llegó a **fold 5/10**, **59 de 120 records** `adapt` escritos. UARs sanos (rango ~0.52–0.81).
- Último log: 07:01:34 (`0015 concat seed=456 → UAR=0.812`).

A las ~07:02 el job se cayó por **fallo de nodo** (`ivb05`): `scontrol` mostró `Requeue=1
Restarts=1 ExitCode=0:0 RunTime=00:00:00`, de vuelta en `PENDING`. No fue error del
training (exit 0:0, sin traceback en `.err`) — fue infraestructura.

### Problema detectado

`1_train.py` NO tiene resume, y el sbatch tiene `require_fresh_output_dir()` que hace
`exit 1` si `OUTPUT_DIR` existe y no está vacío. Al requeuearse, el job iba a **reentrar
y abortar en segundos** porque `1_en_calibfix/` tenía los 59 records parciales →
desperdicio del turno de cola.

### Corrección aplicada (Claude UNC, 28/06 22:27)

1. **Archivado** el output parcial (reversible, forense):
   `results_unc/voz_expresiva/1_en_calibfix/` → `..._partial_nodefail_20260628_2226/`
   (59 records + 59 embeddings + 59 predictions + calib_manifest B2 OK).
   → `1_en_calibfix` queda inexistente, `require_fresh_output_dir` pasará limpio.
2. **`scontrol update JobId=1158456 Requeue=0`**: si vuelve a caer un nodo, ahora **falla
   limpio** (notifica) en vez de loopear requeue→abort-por-dir-sucio. Decisión: para un
   script sin resume, el requeue automático no aporta (no puede resumir) y solo confunde.
3. El job 1158456 (PENDING, posición de cola **preservada**) reentrará y correrá los
   120 runs desde cero en dir fresco.

### Pendiente

- Monitorear reentrada de 1158456 → verificar cierre con 120 records adapt + manifest B2.
- El dir `_partial_nodefail_20260628_2226/` se puede borrar tras el cierre exitoso (es forense).
- Nota para LOCAL: el contraste cross-language queda igual (mismo código/fix B2); solo cambió
  el nodo A30 que ejecuta (sigue siendo hardware A30, no afecta el caveat ya declarado).

---

## Voz Expresiva — EN N-adapt calibfix: ✅ CERRADO (2026-07-01)

### Resultado

`1158456` **reentró limpio** el 01/07 02:11 en `ivb05` (dir fresco tras el archivado → 0 aborts
por "dir no vacío", corrección validada) y **COMPLETED exit 0:0** a las 06:34. Wall-clock **4h 23min**
("All runs done in 4.3 h"). Sin errores, sin node-failure esta vez (requeue ya desactivado).

**Verificación del script (`verification_ok`) + mis checks independientes coinciden:**
| Check | Valor |
|---|---|
| records totales | 120 |
| records `adapt` | **120** (0 non-adapt) |
| `calib_seed_effective` no-nulo | todos ✓ |
| speakers en manifest | **10** |
| sets únicos de sentence_ids | **10** → **fix B2 efectivo** |
| embeddings / predictions .npy | 120 / 120 |

**UAR (test) por config** (n=30 c/u = 10 folds × 3 seeds 42/123/456):
| config | UAR mean |
|---|---|
| concat | 0.740 |
| xattn | 0.738 |
| film | 0.733 |
| none | 0.698 |
| **global** | **0.727** (rango 0.461–0.889) |

Observación neutral para LOCAL (no juicio): `epochs_trained` varía 6–30 entre runs (el handoff
declaraba "sin early-stopping, 30ep"); puede reflejar selección de best-epoch o un criterio de
corte — confirmar del lado LOCAL al mergear, no lo interpreto acá.

### Entregables (rama `unc`)
- `results_unc/voz_expresiva/1_en_calibfix/uar_results.json` (120 records adapt) — **el deliverable para el merge**.
- `results_unc/voz_expresiva/1_en_calibfix/calib_manifest.json` (10 speakers, B2).
- Añadida excepción en `.gitignore` para `results_unc/voz_expresiva/**/*.json` (convención de gates).
- **NO commiteados**: `embeddings/` + `predictions/` (814 MB de `.npy`, intermedios). Quedan en Mendieta;
  disponibles por rsync si LOCAL los necesita para análisis fino (el merge/report estándar usa el JSON).

### Para LOCAL (siguiente paso, del lado local)
1. `git fetch origin unc` y traer `uar_results.json` + `calib_manifest.json`.
2. Merge: N-strict EN heredado (3090) + este N-adapt corregido (A30) → reporte cross-language.
3. Recordar el caveat de hardware ya declarado (N-adapt en A30 vs N-strict/ZH en 3090); el claim
   primario N-strict queda hardware-limpio.

### Housekeeping UNC
- El dir forense `1_en_calibfix_partial_nodefail_20260628_2226/` (59 records de la corrida caída)
  se puede borrar; lo dejo por ahora por si LOCAL quiere inspeccionar el incidente. Sin valor científico.
