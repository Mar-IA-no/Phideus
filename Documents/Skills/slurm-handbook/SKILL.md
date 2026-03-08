# SLURM Operations Handbook — CCAD/UNC (Mendieta)

Compendio operativo para trabajar con SLURM en el CCAD de la Universidad Nacional de Cordoba.
Construido a partir de experiencia real en el cluster Mendieta corriendo ML training jobs.

Cuando el usuario invoque `/slurm-handbook`, presentar un menu interactivo con las secciones disponibles.
Si el usuario pasa un tema especifico (ej: `/slurm-handbook memoria`), ir directo a esa seccion.

---

## Menu principal

Cuando se invoque sin argumentos, presentar:

```
=== SLURM Operations Handbook — CCAD/UNC ===

Secciones disponibles:
  1. Arquitectura HPC         — Login node vs compute nodes, reglas basicas
  2. Script sbatch             — Template completo, directivas, orden de setup
  3. Trampas conocidas         — 15+ errores reales y como evitarlos
  4. Data staging              — MAESTRO a /scratch, rsync vs cp, verificacion
  5. Checkpoint & resume       — Patron de resume, SIGTERM, auto-resubmit
  6. Array jobs                — 1D y 2D, decode de indices, logging
  7. Memoria                   — Sizing, page cache vs OOM, profiling
  8. Scheduling                — Backfill, --time tuning, particiones
  9. Monitoreo                 — squeue, scontrol, seff, logs
 10. Debugging                 — Preflight strategy, interactive sessions
 11. Wiki CCAD                 — Links de referencia organizados por tema
 12. Checklist pre-submit      — Verificacion rapida antes de sbatch
 13. Templates                 — Scripts listos para copiar y adaptar
 14. Referencia rapida         — Comandos mas usados en una tabla

Decime un numero o tema, o "todo" para el compendio completo.
```

---

## 1. Arquitectura HPC

```
SSH --> Nodo Cabecera (login node)       Nodos de Computo (via SLURM)
        - Internet: SI                    - Internet: NO
        - CPU/RAM: LIMITADOS (cgroups)    - CPU/RAM/GPU: COMPLETOS
        - PROHIBIDO computo pesado        - Todo el computo va aca
        - Claude Code funciona aca        - Claude Code NO funciona aca
        - ~4-8 GB RAM (killed sin aviso)  - 64 GB RAM + 2x A30 GPU
```

### Hardware Mendieta

| Recurso | Valor |
|---------|-------|
| Nodos computo | 18 (ivb02-ivb20, ivb01 retirado) |
| CPU/nodo | Intel Xeon E5-2680v2, 20 cores |
| RAM/nodo | 64 GB |
| GPU/nodo | 2x NVIDIA A30 (24 GB HBM2 c/u) |
| Total GPUs | 36 NVIDIA A30 |
| Red | InfiniBand QDR 40 Gbps |
| Disco local/nodo | 400 GB SSD (/scratch) |

### NVIDIA A30 specs para ML
- Ampere GA100, 3584 CUDA cores, 224 Tensor Cores (3ra gen)
- 24 GB HBM2 (~933 GB/s bandwidth)
- Soporte: FP64, FP32, TF32, FP16, BF16, INT8
- TDP: 165W (eficiente vs RTX 3090 350W)
- Habilitar TF32: `torch.backends.cuda.matmul.allow_tf32 = True`

### Particiones

| Particion | Tiempo max | Default | Uso |
|-----------|-----------|---------|-----|
| `short` | 1 hora | SI | Preflight, debug, tests rapidos |
| `multi` | 2 dias | NO | Training real |

### Almacenamiento

| Dir | Tipo | Velocidad | Persistencia | Uso |
|-----|------|-----------|-------------|-----|
| `/home/` | NFS compartido | Lenta | Permanente | Codigo, scripts, resultados |
| `/scratch/$SLURM_JOB_ID/` | XFS local (SSD) | Rapida | SE BORRA al terminar job | Datos durante training |
| `/tmp/` | tmpfs (RAM) | Muy rapida | Se borra | Solo archivos tiny |

### Otros recursos CCAD

| Recurso | Specs | Acceso | Ideal para |
|---------|-------|--------|-----------|
| Nabucodonosor | 10 cores, 64GB, 1xA30, internet | SSH directo (solicitar acceso) | Debug con GPU, sin cola |
| Serafin | 64 nodos, AMD EPYC 64-core, 256GB | SLURM (short/multi) | CPU masivo |
| Mulatona | 7 nodos, Xeon 32-core, 128GB | SLURM (short/mono) | CPU mediano |
| Jupyter | 32 cores, 125GB, Intel Arc A770 | Web (jupyterhub.ccad.unc.edu.ar) | Notebooks interactivos (NO CUDA) |

---

## 2. Script sbatch — Template completo

### Bloque de entorno (ORDEN OBLIGATORIO)

```bash
#!/bin/bash
#SBATCH --job-name=mi_job
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/path/to/logs/job_%j.out
#SBATCH --error=/path/to/logs/job_%j.err

set -eo pipefail

# === ESTE ORDEN ES OBLIGATORIO ===
. /etc/profile                                      # 1. PRIMERO (modules dependen de esto)
module load gcc cuda                                # 2. DESPUES de /etc/profile
source ~/miniconda3/bin/activate mi_env             # 3. Conda
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK         # 4. Evitar explosion de threads
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 5. Prevenir CUDA OOM por fragmentacion
export PYTHONUNBUFFERED=1                           # 6. OBLIGATORIO — sin esto no hay logs
```

### Por que este orden importa

| Paso | Que pasa si falta/esta mal |
|------|---------------------------|
| `. /etc/profile` | `module load` no encuentra nada, script falla inmediatamente |
| `module load gcc cuda` | No hay compilador ni CUDA runtime |
| `conda activate` | Python del sistema es 3.6.8, inutilizable para PyTorch |
| `OMP_NUM_THREADS` | OpenMP puede crear cientos de threads, mata performance |
| `PYTORCH_CUDA_ALLOC_CONF` | CUDA OOM por fragmentacion de memoria |
| `PYTHONUNBUFFERED=1` | Python bufferea TODO, logs aparecen vacios durante el job |

---

## 3. Trampas conocidas (aprendidas con dolor)

### CRITICA: NO usar `set -u`
- **Sintoma**: Script muere inmediatamente al ejecutar `. /etc/profile`
- **Causa**: `/etc/profile` de Mendieta referencia `LC_ALL` sin definir. Con `-u`, bash aborta.
- **Solucion**: Usar `set -eo pipefail` (sin `-u`). NUNCA `set -euo pipefail`.

### CRITICA: `--gres=gpu:N`, NO `--gpus-per-task`
- **Desde**: 2026-02-25 (cambio en validacion SLURM de Mendieta)
- **Sintoma**: SLURM rechaza el job submission
- **Solucion**: Reemplazar `--gpus-per-task=1` por `--gres=gpu:1`

### CRITICA: `--mem=0` bloquea scheduling
- **Sintoma**: Job queda PENDING con Reason=Resources indefinidamente
- **Causa**: `--mem=0` pide nodo completo, no cabe en nodos `mix` (parcialmente usados)
- **Solucion**: Usar valor explicito: `--mem=32G`, `--mem=48G`, `--mem=60G`

### CRITICA: Falta `--error=` pierde logs de Python
- **Sintoma**: Logs vacios o sin informacion de training
- **Causa**: Python usa `logging` (stderr). Sin `--error=`, stderr se mezcla con stdout o se pierde
- **Solucion**: Siempre agregar `#SBATCH --error=.../logs/nombre_%j.err`

### CRITICA: `PYTHONUNBUFFERED=1` es obligatorio
- **Sintoma**: Archivo `.err` vacio durante horas, de repente aparece todo junto
- **Causa**: Python bufferea output por defecto en modo no-interactivo
- **Solucion**: `export PYTHONUNBUFFERED=1` en el script

### IMPORTANTE: `ls checkpoint*.pt` crashea con pipefail
- **Sintoma**: Script muere silenciosamente en la primera ejecucion (sin checkpoints previos)
- **Causa**: `ls` retorna exit code 2 si no hay matches, `set -e` lo mata
- **Solucion**: `mkdir -p $OUTDIR` antes del ls, y usar `|| true`:
  ```bash
  mkdir -p $OUTDIR
  LAST_CKPT=$(ls -t $OUTDIR/checkpoint_epoch*.pt 2>/dev/null | head -1 || true)
  ```

### IMPORTANTE: `sbatch --test-only` es pesimista
- **Contexto**: Las estimaciones de tiempo de espera son worst-case
- **Realidad**: Los jobs suelen entrar MUCHO antes del estimado
- **Accion**: No alarmarse por el estimado, pero si usarlo como sanity check

### IMPORTANTE: Resubmit pierde prioridad
- **Contexto**: Cancelar y re-submitir un job lo pone al final de la cola
- **Solucion**: Si solo hay que ajustar `--time`, usar `scontrol update` en lugar de cancelar:
  ```bash
  scontrol update JobId=XXXXX TimeLimit=10:00:00
  ```

### AVISO: Page cache infla MaxRSS en `seff`
- **Sintoma**: `seff` reporta 60GB de memoria con `--mem=48G` (126% eficiencia)
- **Causa**: `cp -r 120GB` a scratch llena page cache del kernel. SLURM lo cuenta como RSS.
- **Realidad**: El proceso usa ~3-20GB reales. El page cache se libera on-demand.
- **Validacion**: Si exit code es 1 (error Python), no 137 (SIGKILL) -> no fue OOM.

### CRITICA: Usar `rsync`, NUNCA `cp -r` para staging
- **Regla**: TODO staging de datos a `/scratch` DEBE usar `rsync -a --info=progress2`, NUNCA `cp -r`.
- **Por que**: `cp -r` de 120GB tarda 22-35 min con CERO output. No hay forma de saber si está progresando o colgado. Si falla a mitad de camino, no hay forma de recuperar parcialmente.
- **Mediciones reales (MAESTRO 120GB)**:
  - 1 job solo: ~22 min
  - varios jobs paralelos: ~35-40 min (NFS compartido)
- **Patron correcto**:
  ```bash
  rsync -a --info=progress2 $MAESTRO_SRC/ $SCRATCH/maestro-v3.0.0/
  ```
- **Trailing slash importa**: `SRC/` copia contenidos. `SRC` (sin slash) copia el directorio entero.
- **Fix masivo 2026-03-08**: 31 scripts corregidos de `cp -r` a `rsync` en una sola pasada.

### AVISO: Python del sistema es 3.6.8
- **Consecuencia**: PyTorch moderno (2.x) no funciona. Miniconda es obligatorio.
- **Setup**: `conda create -n env python=3.11 pytorch pytorch-cuda=12.1 -c pytorch -c nvidia`

### AVISO: Nodos de computo no tienen internet
- **Impacto**: No se puede `pip install`, descargar modelos HuggingFace, ni hacer API calls
- **Solucion**: Instalar todo en login node, pre-descargar modelos, usar cache local

### AVISO: Claude Code crashea en sesiones SLURM interactivas
- **Bug**: v2.0.43+, subprocesos consumen stdin (issue #12507)
- **Workarounds**: npm install global, usar Nabucodonosor, downgrade a 2.0.42, modo no-interactivo

---

## 4. Data staging — Mover datos a /scratch

### Patron basico

```bash
SCRATCH=/scratch/$SLURM_JOB_ID
mkdir -p $SCRATCH

echo "Staging datos a scratch..."
COPY_START=$(date +%s)
rsync -a --info=progress2 $DATA_SRC/ $SCRATCH/dataset/
COPY_END=$(date +%s)
echo "Staging completo en $((COPY_END - COPY_START)) segundos."
```

### Por que /scratch y no /home

| Aspecto | /home (NFS) | /scratch (SSD local) |
|---------|-------------|---------------------|
| Throughput | ~100-200 MB/s (compartido) | ~500+ MB/s (exclusivo) |
| Latencia | Alta (red) | Baja (local) |
| Random I/O | Muy lenta | Rapida |
| Persistencia | Permanente | SE BORRA al terminar job |
| Capacidad | Ilimitada (cuota) | 400 GB/nodo |

Para ML training con miles de archivos pequenos (audio, imagenes), la diferencia es 2-5x en velocidad de epoch.

### Verificacion post-copia

Siempre verificar que la copia fue exitosa:

```bash
if [ ! -f "$SCRATCH/dataset/metadata.json" ]; then
    echo "ERROR: metadata no encontrada en scratch"
    exit 1
fi
echo "Verificacion OK: $(ls $SCRATCH/dataset/ | wc -l) archivos"
```

### rsync vs cp

| | `cp -r` | `rsync -a --info=progress2` |
|---|---------|--------------------------|
| Progreso visible | NO | SI (% completado) |
| Incremental | NO (copia todo) | SI (solo cambios) |
| Verificacion | NO | SI (checksums con -c) |
| Overhead | Menor | ~5% mas lento |
| **Recomendacion** | Solo si velocidad es critica | **PREFERIDO** para staging |

### Recuperar archivos de scratch antes de que mueran

```bash
# Desde login node, si el job aun esta corriendo:
sgather /scratch/archivo prefijo

# O copiar resultados dentro del script ANTES de terminar:
cp -r $SCRATCH/results /home/usuario/results/
```

---

## 5. Checkpoint & resume

### Patron de resume en bash

```bash
OUTDIR=/path/to/output
mkdir -p $OUTDIR

# Buscar ultimo checkpoint (safe con pipefail)
LAST_CKPT=$(ls -t $OUTDIR/checkpoint_epoch*.pt 2>/dev/null | head -1 || true)
RESUME_FLAG=""
if [ -n "$LAST_CKPT" ]; then
    echo "Resumiendo desde: $LAST_CKPT"
    RESUME_FLAG="--resume $LAST_CKPT"
fi

srun python train.py --output $OUTDIR $RESUME_FLAG --device cuda
```

### SIGTERM handler para checkpoint graceful

Para jobs largos (>6h), SLURM puede matar el job por timeout. Con `--signal`, le avisa antes:

```bash
#SBATCH --signal=B:SIGTERM@595    # Avisa 9m55s antes del timeout

# Handler que captura la signal y la reenvia al proceso Python:
trap 'echo "SIGTERM a las $(date), reenviando..."; kill -TERM $PID; wait $PID' SIGTERM

srun python train.py --output $OUTDIR $RESUME_FLAG &
PID=$!
wait $PID
EXIT_CODE=$?
```

**Importante**: El proceso Python debe manejar SIGTERM internamente (guardar checkpoint al recibirlo).

### Auto-resubmit si no completo

```bash
# Despues del wait:
if [ $EXIT_CODE -ne 0 ]; then
    # Verificar que hay checkpoint para resumir
    if ls $OUTDIR/checkpoint_epoch*.pt &>/dev/null; then
        echo "Training incompleto, re-submitiendo..."
        sbatch $0    # Re-envia el mismo script (resume desde checkpoint)
    fi
fi
```

### Variante con contador de intentos

```bash
ATTEMPT_FILE=$OUTDIR/.attempt_count
ATTEMPT=$(cat $ATTEMPT_FILE 2>/dev/null || echo 0)
ATTEMPT=$((ATTEMPT + 1))
echo $ATTEMPT > $ATTEMPT_FILE

if [ $ATTEMPT -le 3 ]; then
    echo "Intento $ATTEMPT de 3, re-submitiendo..."
    sbatch $0
else
    echo "Maximo de intentos alcanzado, requiere intervencion manual"
fi
```

---

## 6. Array jobs

### 1D: Lista simple de parametros

```bash
#SBATCH --array=0-2    # 3 tasks

ARMS=(arm-a arm-b arm-c)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
OUTDIR=$RESULTS_BASE/${ARM}_seed42

srun python train.py --arm $ARM --output $OUTDIR
```

### 2D: Producto cartesiano (seeds x descriptores)

```bash
#SBATCH --array=0-14    # 5 seeds x 3 descriptors = 15 tasks

SEEDS=(42 123 456 789 1337)
DESCRIPTORS=(d0 a4r d4-a4r)

N_DESC=${#DESCRIPTORS[@]}
SEED_IDX=$((SLURM_ARRAY_TASK_ID / N_DESC))
DESC_IDX=$((SLURM_ARRAY_TASK_ID % N_DESC))

SEED=${SEEDS[$SEED_IDX]}
DESC=${DESCRIPTORS[$DESC_IDX]}
OUTDIR=$RESULTS_BASE/${DESC}_seed${SEED}
```

### Logging de array jobs

Usar `%A` (array job ID) y `%a` (task ID) en los nombres de log:

```bash
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
```

Genera: `job_1144698_0.out`, `job_1144698_1.out`, `job_1144698_2.out`

### Requeue de un solo task

```bash
scontrol requeue ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
```

### Limitar concurrencia

```bash
#SBATCH --array=0-14%4    # Maximo 4 tasks simultaneos (de 15 total)
```

Util cuando todos los tasks compiten por NFS (ej: copiar MAESTRO).

---

## 7. Memoria

### Guia de sizing

| Tipo de job | `--mem` recomendado | Notas |
|-------------|-------------------|-------|
| Training ligero (gate43, gate5b) | `32G` | Suficiente para modelo + datos |
| Training con dataloaders pesados (gate6 transkun) | `48G` | num_workers=2, pickles en memoria |
| Training + staging 120GB dataset | `48G-60G` | Page cache infla, pero no es OOM real |
| Dry run / preflight | `32G` | Siempre suficiente |

### Como saber si necesitas mas memoria

1. **Post-job**: `seff <JOBID>` muestra Memory Utilized vs Memory Requested
2. **Dentro del job**: Agregar profiling:
   ```bash
   mem_usage() {
       echo "[MEM $(date +%H:%M:%S)] RSS=$(awk '/VmRSS/{print $2}' /proc/$$/status 2>/dev/null || echo '?')kB | free=$(free -g | awk '/Mem:/{print $4}')G"
   }
   ```
3. **Exit code 137 = SIGKILL = OOM real**. Exit code 1 = error Python (no OOM).

### Page cache: no es memoria real

Cuando copias 120GB a scratch, Linux llena la RAM con page cache. `seff` lo reporta como MaxRSS (~60GB) pero:
- El proceso real usa ~3-20GB
- El page cache se libera automaticamente cuando se necesita la RAM
- SLURM en Mendieta no enforce cgroups de memoria estrictamente
- **No subir `--mem` por esto** — solo hace mas dificil el scheduling

---

## 8. Scheduling

### Particiones y backfill

SLURM usa backfill scheduling: puede meter jobs cortos en huecos entre jobs largos.

**Implicacion**: Un job de `--time=8:00:00` tiene MUCHA mas probabilidad de entrar por backfill que uno de `--time=2-00:00:00`, incluso si ambos necesitan los mismos recursos.

### Estrategia de `--time`

| Enfoque | Ventaja | Riesgo |
|---------|---------|--------|
| `--time=2-00:00:00` (maximo) | Nunca timeout | Peor backfill, espera mas |
| `--time` ajustado (1.3x real) | Mejor backfill | Si subestimas, pierdes el run |
| Con `--signal` + resume | Mejor de ambos | Complejidad en script |

**Recomendacion**: Usar `--time` ajustado + `--signal=B:SIGTERM@595` + auto-resubmit.

### Calibracion empirica

1. Correr preflight en `short` (100 iters, medir throughput)
2. Extrapolar a iteraciones totales
3. Multiplicar por 1.3-1.5x de margen
4. Redondear hacia arriba

```bash
# Ejemplo: preflight dio 0.15s/iter, necesito 50000 iters
# 50000 * 0.15 = 7500s = 2.08h de training
# + 25 min staging = 2.5h
# * 1.5 margen = 3.75h -> --time=4:00:00
```

### Ajustar --time sin perder cola

```bash
# Si el job ya esta en PENDING y queres ajustar:
scontrol update JobId=XXXXX TimeLimit=10:00:00
# NO canceles y re-submitas — perdes prioridad!
```

---

## 9. Monitoreo

### Durante el job

```bash
# Estado de mis jobs
squeue -u $USER -o "%.10i %.9P %.30j %.8T %.10M %.9l %R"

# Detalle de un job especifico
scontrol show job <JOBID>

# Ver logs en tiempo real (Python va a .err)
tail -f /path/to/logs/job_12345.err

# Estado del cluster
sinfo -p multi -o "%.6D %.6t %N"
```

### Post-job

```bash
# Eficiencia (CPU%, memoria%, wall-clock)
seff <JOBID>

# Historial
sacct -u $USER --starttime=2026-03-01 --format=JobID,JobName,State,Elapsed,MaxRSS,NodeList

# Verificar post-submit que SLURM parseo bien las directivas
scontrol show job <JOBID> | grep -E "Partition|TimeLimit|NumCPUs|Gres|ArrayTaskId"
```

### Interpretar razones de PENDING

| Razon | Significado | Accion |
|-------|------------|--------|
| `(Resources)` | No hay nodo con los recursos pedidos libres | Esperar, o reducir recursos |
| `(Priority)` | Hay nodos pero otros jobs tienen prioridad | Esperar (normal) |
| `(AssocMaxJobsLimit)` | Limite de jobs simultaneos por usuario/grupo | Esperar a que termine otro |
| `(Dependency)` | Depende de otro job | Verificar dependencia |
| `(PartitionTimeLimit)` | `--time` excede limite de particion | Reducir --time |

---

## 10. Debugging

### Estrategia de preflight

Antes de submitir un job largo a `multi`, validar en `short`:

1. **Crear version preflight** del script:
   - `--partition=short`, `--time=00:55:00`
   - Reducir a 1 epoch o 100 iteraciones
   - Agregar profiling de memoria
   - Single arm (no array completo)

2. **Verificar que pasa**:
   - Staging de datos OK
   - Imports OK
   - GPU detectada
   - Training arranca y completa sin error
   - Medir throughput (s/iter)

3. **Calibrar** `--time` del job real con la medicion

### Sesion interactiva

```bash
# Pedir nodo interactivo (entra por cola como cualquier job)
srun -p short --gres=gpu:1 --cpus-per-task=10 --mem=32G --time=00:30:00 --pty bash

# Ya en el nodo:
module load gcc cuda
source ~/miniconda3/bin/activate phideus
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Alternativa: salloc

```bash
# Reservar nodo sin entrar (ejecutar comandos remotos)
salloc -p short --gres=gpu:1 --time=00:30:00
srun python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Nabucodonosor (sin cola)

Para debug urgente con GPU, sin esperar cola:

```bash
ssh usuario@nabucodonosor.ccad.unc.edu.ar
# Acceso directo, internet disponible, 1xA30
# Requiere solicitar acceso: soporte@ccad.unc.edu.ar
```

---

## 11. Wiki CCAD — Links de referencia

Consultar bajo demanda, no cargar preventivamente.

### Infraestructura
| Tema | URL | Cuando consultar |
|------|-----|-----------------|
| Clusters (specs) | https://wiki.ccad.unc.edu.ar/infra/clusters.html | Specs exactos de otro cluster |
| Almacenamiento | https://wiki.ccad.unc.edu.ar/infra/almacenamiento.html | Dudas sobre /scratch, /tmp, sgather |
| Particiones | https://wiki.ccad.unc.edu.ar/infra/slurm-particiones.html | Limites de tiempo/nodos por cluster |
| Standalone | https://wiki.ccad.unc.edu.ar/infra/computadoras.html | Nabucodonosor, Jupyter, specs |

### Tutoriales
| Tema | URL | Cuando consultar |
|------|-----|-----------------|
| Script sbatch | https://wiki.ccad.unc.edu.ar/tutoriales/slurm-script.html | Directivas avanzadas, arrays, mail |
| Modulos | https://wiki.ccad.unc.edu.ar/tutoriales/modules.html | Software preinstalado, flags |
| Spack | https://wiki.ccad.unc.edu.ar/tutoriales/spack.html | Compilar software no disponible |
| Paralelizacion | https://wiki.ccad.unc.edu.ar/tutoriales/parallel.html | GNU parallel, xargs, CPU pinning |
| Eficiencia | https://wiki.ccad.unc.edu.ar/tutoriales/check-efficiency.html | seff, sacct, analisis de jobs |
| SLURM signals | https://wiki.ccad.unc.edu.ar/tutoriales/slurm-signals.html | Checkpointing, --signal |
| Dirs compartidos | https://wiki.ccad.unc.edu.ar/tutoriales/shared-dir.html | Compartir datos entre usuarios |
| Licencias | https://wiki.ccad.unc.edu.ar/tutoriales/licenses.html | Software con licencia |
| tmux | https://wiki.ccad.unc.edu.ar/tutoriales/tmux.html | Referencia rapida |
| Archivos | https://wiki.ccad.unc.edu.ar/tutoriales/archivos/index.html | scp, rsync, SSHFS, fstab |

### Primeros pasos
| Tema | URL | Cuando consultar |
|------|-----|-----------------|
| SLURM basico | https://wiki.ccad.unc.edu.ar/empezar/slurm.html | squeue, sbatch, srun |
| Acceso/SSH | https://wiki.ccad.unc.edu.ar/empezar/acceso/index.html | Problemas de conexion |
| Desde cero | https://wiki.ccad.unc.edu.ar/empezar/desde-cero.html | Usuarios nuevos |

### Programas
| Tema | URL | Cuando consultar |
|------|-----|-----------------|
| Quantum Espresso | https://wiki.ccad.unc.edu.ar/programas/qe.html | Simulaciones QE |
| NextFlow | https://wiki.ccad.unc.edu.ar/programas/nextflow.html | Workflows con containers |

### Ayuda
| Tema | URL | Cuando consultar |
|------|-----|-----------------|
| Soporte | https://wiki.ccad.unc.edu.ar/ayuda/soporte.html | Contacto, consultas (Calendly) |
| Colaborar | https://wiki.ccad.unc.edu.ar/ayuda/colaborar.html | Contribuir a la wiki |

---

## 12. Checklist pre-submit

Verificacion rapida antes de `sbatch`. Para validacion completa usar `/validate-sbatch`.

```
[ ] #!/bin/bash como primera linea
[ ] set -eo pipefail (SIN -u)
[ ] . /etc/profile ANTES de module load
[ ] module load gcc cuda
[ ] conda activate del entorno correcto
[ ] PYTHONUNBUFFERED=1 exportado
[ ] --gres=gpu:1 (NO --gpus-per-task)
[ ] --mem=32G o valor explicito (NO --mem=0)
[ ] --error= especificado (separado de --output)
[ ] --time no excede limite de particion (short=1h, multi=48h)
[ ] --signal=B:SIGTERM@595 si job >6h
[ ] rsync para staging (NO cp -r) — progreso visible, incremental
[ ] Todos los paths existen (verificar con test -d / test -f)
[ ] Directorio de logs existe
[ ] Python script existe y es legible
[ ] Dataset source existe
[ ] bash -n script.sh pasa sin errores
[ ] sbatch --test-only script.sh retorna job ID
```

---

## 13. Templates

### Template 1: Job simple con GPU

```bash
#!/bin/bash
#SBATCH --job-name=mi_training
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -eo pipefail

. /etc/profile
module load gcc cuda
source ~/miniconda3/bin/activate mi_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

SCRATCH=/scratch/$SLURM_JOB_ID
mkdir -p $SCRATCH

echo "=== Job info ==="
echo "  Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Staging datos..."

rsync -a --info=progress2 /path/to/data/ $SCRATCH/data/

echo "  Staging completo. Iniciando training..."

srun python train.py \
    --data $SCRATCH/data \
    --output /home/$USER/results/run_$(date +%Y%m%d_%H%M) \
    --epochs 30 --batch-size 16 --device cuda

echo "=== Completado ==="
```

### Template 2: Job largo con resume + auto-resubmit

```bash
#!/bin/bash
#SBATCH --job-name=long_train
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00
#SBATCH --signal=B:SIGTERM@595
#SBATCH --output=logs/long_%j.out
#SBATCH --error=logs/long_%j.err

set -eo pipefail

. /etc/profile
module load gcc cuda
source ~/miniconda3/bin/activate mi_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

OUTDIR=/home/$USER/results/long_run
mkdir -p $OUTDIR

# Resume desde ultimo checkpoint
LAST_CKPT=$(ls -t $OUTDIR/checkpoint_epoch*.pt 2>/dev/null | head -1 || true)
RESUME_FLAG=""
[ -n "$LAST_CKPT" ] && RESUME_FLAG="--resume $LAST_CKPT" && echo "Resumiendo desde: $LAST_CKPT"

# SIGTERM handler
trap 'echo "SIGTERM a las $(date), reenviando..."; kill -TERM $PID; wait $PID' SIGTERM

SCRATCH=/scratch/$SLURM_JOB_ID
mkdir -p $SCRATCH
rsync -a --info=progress2 /path/to/data/ $SCRATCH/data/

srun python train.py --data $SCRATCH/data --output $OUTDIR $RESUME_FLAG --device cuda &
PID=$!
wait $PID
EXIT_CODE=$?

echo "Exit code: $EXIT_CODE"

# Auto-resubmit si no completo
if [ ! -f "$OUTDIR/training_complete.flag" ]; then
    if ls $OUTDIR/checkpoint_epoch*.pt &>/dev/null; then
        echo "Incompleto, re-submitiendo..."
        sbatch $0
    fi
fi

exit $EXIT_CODE
```

### Template 3: Array job

```bash
#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err

set -eo pipefail

. /etc/profile
module load gcc cuda
source ~/miniconda3/bin/activate mi_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# Decode array index
CONFIGS=(config_a config_b config_c config_d)
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
OUTDIR=/home/$USER/results/${CONFIG}

echo "=== Array task $SLURM_ARRAY_TASK_ID: $CONFIG ==="

SCRATCH=/scratch/$SLURM_JOB_ID
mkdir -p $SCRATCH
rsync -a --info=progress2 /path/to/data/ $SCRATCH/data/

srun python train.py --config $CONFIG --data $SCRATCH/data --output $OUTDIR --device cuda

echo "=== Completado: $CONFIG ==="
```

### Template 4: Preflight en short

```bash
#!/bin/bash
#SBATCH --job-name=preflight
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:55:00
#SBATCH --output=logs/preflight_%j.out
#SBATCH --error=logs/preflight_%j.err

set -eo pipefail

. /etc/profile
module load gcc cuda
source ~/miniconda3/bin/activate mi_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "=== PREFLIGHT ==="
echo "  Node: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  CUDA: $CUDA_VISIBLE_DEVICES"

# Verificar imports
python -c "import torch; print(f'torch={torch.__version__}, CUDA={torch.cuda.is_available()}')" || exit 1

# Staging
SCRATCH=/scratch/$SLURM_JOB_ID
mkdir -p $SCRATCH
COPY_START=$(date +%s)
rsync -a --info=progress2 /path/to/data/ $SCRATCH/data/
COPY_END=$(date +%s)
COPY_TIME=$((COPY_END - COPY_START))
echo "Staging: ${COPY_TIME}s"

# Benchmark corto
BENCH_START=$(date +%s)
srun python train.py --data $SCRATCH/data --epochs 1 --device cuda
BENCH_END=$(date +%s)
BENCH_TIME=$((BENCH_END - BENCH_START))

echo "=== RESULTADOS ==="
echo "  Staging: ${COPY_TIME}s"
echo "  1 epoch: ${BENCH_TIME}s"
echo "  30 epochs estimado: $(echo "scale=1; $BENCH_TIME * 30 / 3600" | bc)h"
echo "  GPU VRAM: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | head -1)"
```

---

## 14. Referencia rapida

### Comandos mas usados

```bash
# --- SUBMISION ---
sbatch script.sh                        # Enviar job
sbatch --test-only script.sh            # Validar sin enviar

# --- MONITOREO ---
squeue -u $USER                         # Mis jobs
squeue -u $USER -o "%.10i %.9P %.30j %.8T %.10M %.9l %R"  # Formato detallado
scontrol show job <ID>                  # Todo sobre un job
sinfo -p multi -o "%.6D %.6t %N"        # Estado de nodos

# --- MODIFICAR JOBS ---
scontrol update JobId=<ID> TimeLimit=10:00:00   # Ajustar tiempo sin perder cola
scancel <ID>                            # Cancelar job
scancel -u $USER                        # Cancelar todos

# --- POST-JOB ---
seff <ID>                               # Eficiencia (CPU%, mem%, wall-clock)
sacct -j <ID> --format=JobID,State,Elapsed,MaxRSS  # Historial

# --- INTERACTIVO ---
srun -p short --gres=gpu:1 --mem=32G --time=00:30:00 --pty bash  # Shell en nodo

# --- VARIABLES EN JOBS ---
$SLURM_JOB_ID                  # ID del job
$SLURM_ARRAY_JOB_ID            # ID del array padre
$SLURM_ARRAY_TASK_ID           # Indice del task en array
$SLURM_CPUS_PER_TASK           # Cores asignados
$SLURM_NODELIST                # Nodo(s) asignado(s)
$CUDA_VISIBLE_DEVICES          # GPU(s) visible(s)
```

### Modulos

```bash
module av                       # Listar disponibles
module load gcc cuda            # Cargar (Mendieta: gcc/11.2.0, nvhpc/22.3)
module list                     # Ver cargados
module purge                    # Limpiar todos
module show cuda                # Ver paths/variables de un modulo
```
