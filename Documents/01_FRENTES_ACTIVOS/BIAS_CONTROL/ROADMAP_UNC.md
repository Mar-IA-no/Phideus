<div align="center">

# Roadmap Distribuido: Local + UNC CCAD
### Phideus BIAS_CONTROL — Gates 4.3F5 a 5B

![Version](https://img.shields.io/badge/Version-1.0-111827?style=for-the-badge)
![Fecha](https://img.shields.io/badge/Fecha-2026--02--21-1F6FEB?style=for-the-badge)
![Estado](https://img.shields.io/badge/Estado-Gate_4.4_CERRADO_%2B_BATCH_60EP-F59E0B?style=for-the-badge)

</div>

> [!IMPORTANT]
> **Principio operativo**: LOCAL = laboratorio de diseno iterativo. UNC = fabrica de experimentos paralelos.
> Ningun servidor espera al otro — siempre hay trabajo util en ambos lados.

> [!NOTE]
> **Avance al corte (2026-02-21)**: Gate 4.4 sigue cerrado. En extensión temporal, `a4r 60ep` ya cerró (`S=79.4%`), `D0/d4a4` 60ep y `t3-wt` 50ep hold siguen activos, y el batch `cosine-tail` 60ep quedó enviado a cola.

---

## 1. Infraestructura

### 1.1 Servidores

```
LOCAL (Inference01)                    UNC (Mendieta CCAD)
┌─────────────────────────┐            ┌─────────────────────────┐
│ GPU: 1x RTX 3090 24GB   │            │ GPU: 36x A30 24GB       │
│ CPU: i5-12600K 16 cores  │            │     (2/nodo, 18 nodos)  │
│ RAM: 64 GB               │            │ Scheduler: SLURM        │
│ Disco: M.2 1TB + RAID1   │            │ Storage: NFS 200TB free │
│ IP: 131.72.205.6         │            │ /scratch: 400GB SSD/nodo│
│ Acceso: directo (tmux)   │            │ Max job: 48h            │
│ Datos: TODO presente     │            │ GPUs reales: 4-8 simult │
│ Agente: Claude Opus 4.6  │            │ Agente: Claude Opus 4.6 │
└─────────────────────────┘            └─────────────────────────┘
```

### 1.2 Estado de setup UNC

| Componente | Estado | Notas |
|-----------|--------|-------|
| Repo (git clone) | OK | 39MB, completo |
| Miniconda + env phideus | OK | Python 3.x, PyTorch 2.5.1+cu121 |
| CUDA en compute | OK | A30, driver 535, CUDA 12.2 |
| MAESTRO dataset | OK | disponible en entorno UNC para runs Gate 4.3 Fase 5 |
| foundation_locked_e25.pt | **OK** | GitHub Release v0.1.0-foundation, MD5 verificado |
| segments_metadata.json | ~~NO NECESARIO~~ | El loader lee `maestro-v3.0.0.json` directo |
| sbatch templates | OK | scripts UNC operativos y validados en Fase 5 |

### 1.3 Diferencias SLURM Mendieta

| Parametro | Valor correcto | Nota |
|-----------|---------------|------|
| GPU request | `--gpus=1` | NO `--gres=gpu:1` |
| Particion | `multi` | GPU partition |
| Max walltime | 48h | Checkpoints obligatorios |
| Array throttle | `--array=0-N%4` | 4 concurrentes = realista sin cola larga |
| I/O pattern | Copiar a `/scratch/$SLURM_JOB_ID` | NFS lento para training |
| Signal handling | `--signal=B:SIGTERM@595` | Auto-resubmit para runs >48h |

---

## 2. Estrategia de division

### 2.1 Principios

```
LOCAL es mejor para:                   UNC es mejor para:
├── Arquitecturas nuevas (debug)       ├── N arms independientes en paralelo
├── Ciclos rapidos de iteracion        ├── Array jobs (sbatch --array)
├── Runs largos monitoreados           ├── Multi-seed replication
├── Analisis visual e interactivo      ├── Sweeps de hiperparametros
└── Implementacion de codigo nuevo     └── Ablaciones masivas
```

### 2.2 Protocolo Git: dos ramas

**Cada Claude pushea SOLO a su rama. Nunca a la del otro.**

```
                         GitHub repo
                     ┌───────────────────┐
                     │                   │
                     │   main ◄── LOCAL  │
                     │     │             │
                     │     │  merge      │
                     │     ▼             │
                     │   unc  ◄── UNC   │
                     │                   │
                     └───────────────────┘
```

| | Rama `main` | Rama `unc` |
|---|---|---|
| **Pushea** | LOCAL (este Claude) | UNC (otro Claude) |
| **Nunca toca** | UNC | LOCAL |
| **Contiene** | Código core, modelos, descriptores | Adaptaciones UNC, SLURM scripts, fixes runtime |

**Flujo de sincronización:**
```
LOCAL                                  UNC
  │                                      │
  │  implementar + pilot GPU             │
  │         │                            │
  │         ▼                            │
  │  git push main ──────────────►  git pull origin main
  │                                      │  (o merge main → unc)
  │                                      │
  │                                      ▼
  │                                  adaptar/arreglar si necesario
  │                                  git push unc
  │                                      │
  │                                      ▼
  │                                  sbatch array job
  │                                      │
  │                                      ▼
  │  ◄─── usuario comunica fix ────  "arreglé X en gate42_training.py"
  │                                      │
  │  cherry-pick fix → push main         │
  │                                      │
  ▼                                      │
  analizar resultados                    │
```

**Regla clave**: Cuando UNC encuentra y arregla un bug en código compartido (ej: `gate42_training.py`), lo pushea a `unc`. El usuario le avisa a LOCAL, que incorpora el fix a `main`. UNC luego hace `git pull origin main` para mantenerse sincronizado.

**Creación de la rama `unc`** (una sola vez, en UNC):
```bash
git checkout -b unc origin/main
git push -u origin unc
```

---

## 3. Plan por Gate

### 3.1 Gate 4.3 Fase 5 — Nuevos descriptores + reverse cross-attention

**Pregunta**: Hay descriptores/mecanismos mejores que los ganadores de Gate 4.3?

| Brazo | Descriptor | Mecanismo | Compara contra |
|-------|-----------|-----------|----------------|
| A4r | A4 log-freq (8d) | reverse cross-att | A4x (regular cross-att) |
| D4r | D4 intervals (4d) | reverse cross-att | D4x (regular cross-att) |
| A8 | onset-weighted chroma (12d) | concat | A4, A7 |
| A9 | IDF-weighted attractor (12d) | concat | A7 |

**Protocolo**: 5ep fresh desde foundation, freeze-policy run-d, pool=256/queries=500/seed=42.

| | LOCAL | UNC |
|--|-------|-----|
| **Tarea** | d4a4-scratch termina (e24->e30) | 4 arms en paralelo |
| **Razon** | Ya corriendo, no mover | 4 jobs independientes, perfecto SLURM |
| **Tiempo** | ~6h (termina solo) | ~3h (4 GPUs) + queue |
| **Bloqueante** | — | foundation + metadata transfer |

**Decision post-Fase 5**: Si A8/A9 superan A4 -> probar dual (d4a8, d4a9) antes de Gate 4.4.

```
sbatch --array=0-3 --gpus=1 --partition=multi --time=06:00:00 gate43_fase5.sh
# SLURM_ARRAY_TASK_ID: 0=a4r, 1=d4r, 2=a8, 3=a9
```

---

### 3.2 Gate 4.4 — Arquitecturas mayores

**Pregunta**: Una arquitectura radicalmente diferente cambia el juego?

| Variante | Concepto | Params nuevos est. |
|----------|----------|--------------------|
| Third Tower | Ratios como modalidad propia con encoder independiente | ~5-10M |
| FiLM (audio/midi/dual) | Modulación de capas internas condicionada por descriptor | ~0.5-1.6M |
| MoE + Ratio Expert | Mixture of Experts con experto dedicado a ratios | ~10-15M |

**Estado actual**:
- Gate 4.4 quedó cerrado en UNC con screening completo (8 brazos base + MoE v2/v3/v4).
- Tabla final 5ep consolidada en `results_unc/` y sincronizada a `main`.
- Runs largos scratch de esta familia cerrados:
  - `t3-wt` 30ep: `S=79.8%` (e30).
  - `moe-dual` 30ep: `S=72.6%` (e30).
- Nueva fase enviada a UNC:
  - batch 60ep: `D0`, `d4a4`, `a4r`, `d4-a4r`, `moe-dual`
  - `t3-wt` 50ep con `--lr-hold-fraction=0.5`.
  - batch `cosine-tail` 60ep: `D0`, `d4a4`, `a4r`, `d4-a4r`.

| | LOCAL | UNC |
|--|-------|-----|
| **Tarea** | Curaduría comparativa + sincronización documental | Ejecución paralela de bloque 60ep/hold |
| **Razon** | Mantener consistencia metodológica y trazabilidad | Aprovechar paralelismo A30 para corridas largas |
| **Tiempo** | ciclo continuo por corte | ~1-2 días efectivos según cola y requeue |
| **Dependencia** | Import de artefactos en `results_unc/` | Disponibilidad de A30 y manejo de walltime 48h |

**Flujo detallado**:
```
Diseño + implementación + pilotos Gate 4.4: COMPLETADO
Screening 5ep (24 brazos): COMPLETADO
Runs largos 30ep (t3-wt, moe-dual): COMPLETADO
Batch 60ep (cosine estándar): a4r COMPLETADO, D0/d4a4 EN CURSO, d4-a4r/moe-dual PENDIENTE
t3-wt 50ep hold: EN CURSO (S@e40=80.6%)
Batch 60ep cosine-tail: EN COLA
```

**Comparación de referencia Gate 4.4 (protocolo 5ep)**:
| Referencia | Valor |
|-----------|-------|
| `d4a4` (Gate 4.3) | `S=69.8%` |
| `D0` (Gate 4.3) | `S=60.2%` |

---

### 3.3 Gate 5A — Barrido comprehensivo

**Pregunta**: Cual es la combinacion optima descriptor x mecanismo?

**Matriz factorial**:

| | Concat | Cross-att | Reverse |
|--|--------|-----------|---------|
| **D4** (intervals) | 63.6% | 60.0% | ? |
| **A4** (log-freq) | 63.6% | 62.6% | ? |
| **A7** (attractor) | 58.8% | 62.2% | ? |
| **A8** (chroma+onset) | ? | — | — |
| **A9** (IDF attractor) | ? | — | — |
| **Nuevos (Gate 5A)** | ? | ? | ? |

Celdas con `%` = ya medidas en Gate 4.3. Celdas con `?` = pendientes. Celdas con `—` = baja prioridad.

**Nota**: Las celdas a correr dependen de resultados de Gate 4.3 Fase 5 y Gate 4.4. No todas las 80+ combinaciones son necesarias — se priorizan las prometedoras.

| | LOCAL | UNC |
|--|-------|-----|
| **Tarea** | Implementar nuevos descriptores + grid de barrido | Correr barrido como array job |
| **Razon** | Priorización de celdas con mejor señal post Gate 4.4 | 20+ arms independientes = caso de uso perfecto |
| **Tiempo** | 1-2 dias implementacion | 1-2 dias con `--array=0-N%4` |
| **Dependencia** | — | Resultados Fase 5 + 4.4 (para definir scope) |

```
# Ejemplo: 20 arms del barrido
sbatch --array=0-19%4 --gpus=1 --partition=multi --time=06:00:00 gate5a_sweep.sh
# Cada task mapea a una combinacion descriptor x mecanismo
```

**Cross-modal injection** (3 arms adicionales):
| Brazo | Audio encoder recibe | MIDI encoder recibe |
|-------|---------------------|---------------------|
| CM-a | — | Best audio descriptor |
| CM-m | Best MIDI descriptor | — |
| CM-bi | Best MIDI descriptor | Best audio descriptor |

**Nota**: d4a4cm (cross-modal) fue el peor brazo de Gate 4.3 (-7.8pp). Cross-modal injection se mantiene por completitud cientifica pero con expectativa baja.

---

### 3.4 Gate 5B — Showcase cientifico

**Pregunta**: El best model es robusto, causal, y publicable?

**Prerequisito**: Best model determinado por Gates 4.3F5 + 4.4 + 5A.

**13 tests ordenados por relevancia**:

| # | Test | Tipo | Servidor | Tiempo est. |
|---|------|------|----------|-------------|
| 1 | Causal ablation (zero-out) | eval | LOCAL | ~1h |
| 2 | Parameter-matched ablations | training | **UNC** | ~6h |
| 3 | RatioProbeDecoder + cross-decoding | training | **UNC** | ~8h |
| 4 | Invariancia transposicion MIDI | eval | LOCAL | ~4h |
| 5 | Multi-seed replication (5 seeds x 30ep) | training | **UNC** | ~15h (5 paralelo) |
| 6 | RSA/CKA entre capas | eval | LOCAL | ~8h |
| 7 | Counterfactual Decoder | training | **UNC** | ~40h |
| 8 | Ratio decoding report | eval | LOCAL | ~8h |
| 9 | Invariancia suite completa | eval | LOCAL | ~12h |
| 10 | UMAP/t-SNE visualizations | eval | LOCAL | ~2h |
| 11 | CrossModalSequenceDecoder | training | **UNC** | ~15h |
| 12 | Gate scoreboard reproducible | eval | LOCAL | ~1h |
| 13 | Retrieval demo UI | eval | LOCAL | ~4h |

| | LOCAL | UNC |
|--|-------|-----|
| **Tarea** | Tests eval-only (1,4,6,8,9,10,12,13) | Tests training-heavy (2,3,5,7,11) |
| **Razon** | Necesitan inspeccion visual e iteracion | N jobs identicos en paralelo |
| **Tiempo** | ~2 dias analisis | ~2-3 dias (paralelo) |

**Multi-seed (test #5)** — manejo de walltime 48h:
```
# Cada seed es un job independiente
# Con --signal=B:SIGTERM@595 para checkpoint antes de kill
# Auto-resubmit si no termino en 48h
sbatch --array=0-4 --gpus=1 --time=48:00:00 gate5b_multiseed.sh
# SLURM_ARRAY_TASK_ID: 0=seed42, 1=seed123, 2=seed456, 3=seed789, 4=seed1337
```

---

## 4. Timeline Gantt

```
Dia       LOCAL                              UNC
────────  ────────────────────────────       ──────────────────────────────
 0        Cierre documental Gate 4.4          COMPLETADO: screening 24 brazos
          + tabla unificada 5ep               + transferencia `results_unc/`
          |                                  |
 1-2      Diseño de bloque temporal           Envío batch 60ep (`D0/d4a4/a4r/
          (comparabilidad 30ep vs 60ep)       d4-a4r/moe-dual`) + hold 50ep
          |                                  |
 3-4      Consolidar primer corte             Recoleccion eval cada 5 epochs
          de curvas largas nuevas             + checkpoints/requeue si aplica
          |                                  |
 5+       Comparativa temporal                Cierre de corridas largas
          (30ep cerrado vs 50/60ep)           para decisión pre Gate 5A/5B
```

**Estimación operativa actual**: bloque largo 50/60ep sujeto a cola SLURM y límite de 48h (con requeue/checkpoint para runs extensos).

---

## 5. Transferencia de datos

### 5.1 Archivos a transferir (LOCAL -> UNC)

| Archivo | Tamano | Prioridad | Metodo | Estado |
|---------|--------|-----------|--------|--------|
| `foundation_locked_e25.pt` | 288 MB | **BLOQUEANTE** | GitHub Release | **DONE** |
| ~~`segments_metadata.json`~~ | ~~62 MB~~ | ~~BLOQUEANTE~~ | — | NO NECESARIO |
| Checkpoints Gate 4.3 | 41 GB | OPCIONAL | rsync bajo demanda | — |
| d4a4-scratch outputs | 25+ GB | OPCIONAL | rsync bajo demanda | — |

**MD5 foundation**: `ddb2ebf7075eec4dcec1628341ec4942`
**GitHub Release**: https://github.com/AlterMundi/Phideus/releases/tag/v0.1.0-foundation
**Descarga en UNC**: `gh release download v0.1.0-foundation -p "foundation_locked_e25.pt"`

### 5.2 Datos que UNC descarga directo

| Archivo | Tamano | Fuente | Estado |
|---------|--------|--------|--------|
| MAESTRO v3.0.0 | 120 GB | Google Storage (wget) | EN CURSO |

### 5.3 Patron de I/O en jobs SLURM

```bash
# Al inicio de cada job: copiar datos a /scratch (SSD local del nodo)
SCRATCH="/scratch/$SLURM_JOB_ID"
mkdir -p "$SCRATCH/maestro" "$SCRATCH/checkpoints"

# Copiar solo lo necesario (foundation + metadata + MAESTRO)
cp ~/Phideus/data/.../foundation_locked_e25.pt "$SCRATCH/checkpoints/"
cp ~/Phideus/data/.../segments_metadata.json "$SCRATCH/"
cp -r ~/Phideus/data/maestro_v3/maestro-v3.0.0/ "$SCRATCH/maestro/"

# Training apunta a /scratch
python gate42_training.py --maestro-dir "$SCRATCH/maestro/maestro-v3.0.0" \
  --checkpoint "$SCRATCH/checkpoints/foundation_locked_e25.pt" ...

# Al final: copiar resultados de vuelta a /home NFS
cp -r "$OUTPUT_DIR" ~/Phideus/data/results/
```

**Optimizacion para array jobs en mismo nodo**: Si 2 tasks caen en el mismo nodo (2 GPUs), evitar doble copia con lock file:
```bash
LOCKFILE="/scratch/maestro_copy.lock"
if ! mkdir "$LOCKFILE" 2>/dev/null; then
    # Otro task ya esta copiando, esperar
    while [ -d "$LOCKFILE" ]; do sleep 5; done
fi
```

---

## 6. Resultados esperados y sincronizacion

### 6.1 Donde se guardan resultados

| Gate | UNC output path | Sync a LOCAL |
|------|----------------|--------------|
| 4.3 Fase 5 | `~/Phideus/data/results/gate43_fase5/` | rsync post-job |
| 4.4 | `~/Phideus/data/results/gate44/` | rsync post-job |
| 5A | `~/Phideus/data/results/gate5a/` | rsync post-job |
| 5B | `~/Phideus/data/results/gate5b/` | rsync post-job |

### 6.2 Formato de resultados (por arm)

Cada arm produce:
```
{arm_name}/
├── training.log
├── checkpoint_epoch{N}.pt   (por cada epoch)
├── best_model.pt
└── eval_per_epoch/
    └── eval_epoch{N}.json   (structured pool evaluation)
```

El JSON de eval contiene `gate_metrics.S`, `gate_metrics.hard_neg`, etc. — compatible con el analisis que hacemos en LOCAL.

---

## 7. Riesgos y mitigaciones

| Riesgo | Impacto | Mitigacion |
|--------|---------|-----------|
| Cola SLURM larga (>4h wait) | Retraso en timeline | `--array=%4` limita concurrencia, reducir si hay mucha cola |
| Job falla por walltime 48h | Pierde training | `--signal=B:SIGTERM@595` + checkpoint recovery + auto-resubmit |
| MAESTRO I/O lento en NFS | Training 2-3x mas lento | Copiar a `/scratch` al inicio de cada job |
| Gate 4.4 cambia scripts significativamente | sbatch templates obsoletos | Flujo: LOCAL testea -> push -> UNC adapta templates |
| foundation corrupto en transfer | Resultados invalidos | Verificar MD5 post-transfer: `ddb2ebf7` |
| Nodo de computo sin internet | No puede descargar MERT/modelos | MERTEncoderLite es custom, no necesita HuggingFace |
| Doble copia MAESTRO en mismo nodo | Desperdicio de /scratch | Lock file para compartir copia entre tasks |

---

## 8. Punto de decision critico (Dia 7-8)

```
                    Resultados Gates 4.3F5 + 4.4 + 5A
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     SELECCION BEST MODEL      │
                    │                               │
                    │  Criterios:                    │
                    │  1. S = min(A2M, M2A) maximo  │
                    │  2. hard_neg >= 90%            │
                    │  3. Convergencia estable       │
                    │  4. Parsimonia (menos params   │
                    │     en caso de empate)         │
                    └───────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              Best = d4a4?    Best = Gate 4.4?   Best = Gate 5A?
              (extender a     (nueva arqui-       (nuevo desc o
               50ep)           tectura)            mecanismo)
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │    GATE 5B: SHOWCASE           │
                    │    13 tests sobre best model   │
                    │    Multi-seed + ablaciones     │
                    │    en UNC, analisis en LOCAL    │
                    └───────────────────────────────┘
```

---

## 9. Checklist pre-ejecucion

### Para empezar Gate 4.3 Fase 5 en UNC:

- [x] foundation_locked_e25.pt transferido y verificado (MD5: `ddb2ebf7`) — GitHub Release
- [x] ~~segments_metadata.json~~ — NO NECESARIO (loader lee maestro-v3.0.0.json directo)
- [ ] MAESTRO descomprimido en ~/Phideus/data/maestro_v3/maestro-v3.0.0/
- [ ] Rama `unc` creada: `git checkout -b unc origin/main && git push -u origin unc`
- [ ] sbatch template creado y testeado con 1 arm, 1 epoch, 100 batches (dry run)
- [ ] Verificar que evaluate_structured_pool.py funciona en compute node
- [ ] Verificar VRAM: A30 24GB soporta batch_size=16 + embed_batch_size=16

### Para cada gate subsiguiente:

- [ ] Codigo testeado en LOCAL (pilot GPU OK)
- [ ] Push a `main`
- [ ] UNC: `git pull origin main` (o merge main → unc)
- [ ] UNC adapta sbatch si necesario, pushea a `unc`
- [ ] Dry run 1 arm antes de lanzar array completo

---

## Apendice: Referencia rapida de comandos

### LOCAL

```bash
# Monitorear scratch
tmux attach -t d4a4scratch

# Evaluar checkpoint
python experiments/bias_control/evaluate_structured_pool.py \
  --model <checkpoint.pt> --pool-size 256 --n-queries 500 --seed 42 \
  --maestro-dir data/maestro_v3/maestro-v3.0.0 --output eval_result.json

# Push codigo testeado
git add -A && git commit -m "feat: ..." && git push
```

### UNC

```bash
# Pull codigo nuevo de main
cd ~/Phideus && git pull origin main

# (Si en rama unc) merge main
git checkout unc && git merge main

# Lanzar array job
sbatch experiments/bias_control/slurm/gate43_fase5.sh

# Monitorear jobs
squeue -u mfmendez
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS,MaxVMSize

# Ver output de un job
cat slurm-<JOBID>_<ARRAYID>.out
```
