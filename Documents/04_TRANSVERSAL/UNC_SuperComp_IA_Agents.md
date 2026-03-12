# CCAD UNC — Guia Exhaustiva para Agentes IA en Supercomputo

**Centro de Computacion de Alto Desempeno (CCAD)**
**Universidad Nacional de Cordoba, Argentina**
**Wiki oficial**: https://wiki.ccad.unc.edu.ar/

**Fecha de elaboracion**: 2026-02-15
**Proposito**: Documento de referencia completo para agentes IA (Claude Code u otros) operando dentro de la infraestructura CCAD. Contiene TODA la informacion necesaria para operar sin errores, entender las restricciones del entorno, y maximizar el uso de los recursos disponibles.

---

## TABLA DE CONTENIDOS

1. [Arquitectura General del CCAD](#1-arquitectura-general-del-ccad)
2. [Clusters Disponibles — Hardware Completo](#2-clusters-disponibles--hardware-completo)
3. [Maquinas Standalone (Nabucodonosor y Jupyter)](#3-maquinas-standalone)
4. [Nodos Cabecera vs Nodos de Computo — CRITICO](#4-nodos-cabecera-vs-nodos-de-computo--critico)
5. [Sistema de Almacenamiento](#5-sistema-de-almacenamiento)
6. [SLURM — El Gestor de Trabajos](#6-slurm--el-gestor-de-trabajos)
7. [Particiones SLURM por Cluster](#7-particiones-slurm-por-cluster)
8. [Jobs Interactivos (srun, salloc)](#8-jobs-interactivos-srun-salloc)
9. [Scripts de Lanzamiento (sbatch)](#9-scripts-de-lanzamiento-sbatch)
10. [Sistema de Modulos (module)](#10-sistema-de-modulos-module)
11. [Gestion de Software y Entornos Python](#11-gestion-de-software-y-entornos-python)
12. [Conectividad y Red](#12-conectividad-y-red)
13. [Transferencia de Archivos](#13-transferencia-de-archivos)
14. [tmux — Persistencia de Sesiones](#14-tmux--persistencia-de-sesiones)
15. [GPU — Uso Especifico en Mendieta y Nabucodonosor](#15-gpu--uso-especifico-en-mendieta-y-nabucodonosor)
16. [Restricciones y Politicas](#16-restricciones-y-politicas)
17. [Claude Code en HPC — Bug Conocido y Soluciones](#17-claude-code-en-hpc--bug-conocido-y-soluciones)
18. [Recetas Operativas para Agentes IA](#18-recetas-operativas-para-agentes-ia)
19. [Troubleshooting](#19-troubleshooting)
20. [Contacto y Soporte CCAD](#20-contacto-y-soporte-ccad)

---

## 1. Arquitectura General del CCAD

El CCAD opera como un conjunto de clusters HPC clasicos. La arquitectura sigue el patron estandar:

```
Usuario (SSH) ──────────> Nodo Cabecera (login/head node)
                               │
                               ├── Tiene internet
                               ├── Acceso a /home (NFS compartido)
                               ├── SLURM client (sbatch, srun, squeue)
                               └── NO para computo pesado
                               │
                          SLURM scheduler
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
              Nodo Comp 1  Nodo Comp 2  Nodo Comp N
                    │          │          │
                    ├── SIN internet
                    ├── Acceso a /home (NFS)
                    ├── /scratch local (temporal)
                    └── Toda la CPU/GPU/RAM disponible
```

**Concepto fundamental**: el usuario se conecta por SSH al nodo cabecera, prepara su trabajo, y lo envia a los nodos de computo via SLURM. El nodo cabecera es compartido por todos los usuarios y tiene restricciones de recursos. Los nodos de computo son exclusivos durante el job.

---

## 2. Clusters Disponibles — Hardware Completo

### 2.1 Serafin (cluster principal, CPU)

| Caracteristica | Valor |
|---------------|-------|
| **Nodos de computo** | 64 |
| **CPU por nodo** | AMD EPYC 7532 — 64 cores |
| **RAM por nodo** | 256 GB |
| **GPU** | Ninguna |
| **Interconexion** | InfiniBand HDR100 (100 Gbps) |
| **Disco local** | 1 TB NVMe por nodo |
| **Hostname cabecera** | `serafin.ccad.unc.edu.ar` |
| **Hostname interno** | `head.rome.ccad.unc.edu.ar` |
| **IP interna cabecera** | 10.6.10.250 |
| **Total cores** | 64 × 64 = 4,096 cores |
| **Total RAM** | 64 × 256 GB = 16 TB |

**Uso tipico**: simulaciones masivas CPU, paralelismo MPI, trabajos que requieren mucha RAM.

### 2.2 Mendieta (cluster GPU — EL MAS RELEVANTE PARA ML/DL)

| Caracteristica | Valor |
|---------------|-------|
| **Nodos de computo** | 18 |
| **CPU por nodo** | Intel Xeon E5-2680v2 — 20 cores / 20 threads |
| **RAM por nodo** | 64 GB |
| **GPU por nodo** | **2x NVIDIA A30 (24 GB HBM2 cada una)** |
| **Interconexion** | InfiniBand QDR (40 Gbps) |
| **Disco local** | 400 GB SSD por nodo |
| **Hostname cabecera** | `mendieta.ccad.unc.edu.ar` |
| **Hostname interno** | `head.ivb.ccad.unc.edu.ar` |
| **IP interna cabecera** | 10.10.10.250 |
| **Total GPUs** | 18 × 2 = 36 NVIDIA A30 |
| **Total VRAM** | 36 × 24 GB = 864 GB |

**NVIDIA A30 specs**:
- Arquitectura: Ampere (GA100)
- 24 GB HBM2 (bandwidth ~933 GB/s — significativamente mayor que RTX 3090 GDDR6X)
- 3,584 CUDA cores, 224 Tensor Cores
- TDP: 165W
- Sin salida de video (datacenter only)
- Soporte FP64, FP32, TF32, FP16, BF16, INT8
- Comparable en VRAM a RTX 3090 (24GB) pero con HBM2 (mas bandwidth, mejor para modelos grandes)
- MIG support (Multi-Instance GPU) — puede particionarse en hasta 4 instancias

**Uso tipico**: deep learning, machine learning, cualquier carga GPU.

### 2.3 Mulatona (cluster CPU mediano)

| Caracteristica | Valor |
|---------------|-------|
| **Nodos de computo** | 7 |
| **CPU por nodo** | Intel Xeon E5-2683v4 — 32 cores / 32 threads |
| **RAM por nodo** | 128 GB |
| **GPU** | Ninguna |
| **Interconexion** | InfiniBand QDR (40 Gbps) |
| **Disco local** | 95 GB SSD |
| **Hostname cabecera** | `mulatona.ccad.unc.edu.ar` |
| **Hostname interno** | `head.bdw.ccad.unc.edu.ar` |
| **IP interna cabecera** | 10.3.10.250 |

### 2.4 Eulogia (cluster Xeon Phi — legacy)

| Caracteristica | Valor |
|---------------|-------|
| **Nodos de computo** | 32 |
| **CPU por nodo** | Intel Xeon Phi 7210 — 64 cores / 256 threads |
| **RAM por nodo** | 96 GB |
| **GPU** | Ninguna |
| **Interconexion** | InfiniBand QDR (40 Gbps) |
| **Disco local** | 200 GB SSD |
| **Hostname cabecera** | `eulogia.ccad.unc.edu.ar` |
| **Hostname interno** | `head.knl.ccad.unc.edu.ar` |
| **IP interna cabecera** | 10.2.10.250 |

**Nota**: Xeon Phi es una arquitectura legacy (Knights Landing). Puede tener soporte de software limitado para frameworks modernos de ML.

### 2.5 Boogie (cluster CPU grande por nodo)

| Caracteristica | Valor |
|---------------|-------|
| **Nodos de computo** | 8 |
| **CPU por nodo** | AMD EPYC 7B12 — 128 cores / 128 threads |
| **RAM por nodo** | 512 GB |
| **GPU** | Ninguna |
| **Interconexion** | Ethernet 25 Gbps |
| **Disco local** | No especificado |

**Uso tipico**: trabajos que requieren mucha RAM (512 GB/nodo) o muchos cores por nodo.

---

## 3. Maquinas Standalone

Estas maquinas NO son clusters. Son servidores individuales accesibles directamente por SSH. No usan SLURM (o lo usan de forma simplificada). Son particularmente relevantes para trabajo interactivo.

### 3.1 Nabucodonosor (ML dedicada)

| Caracteristica | Valor |
|---------------|-------|
| **CPU** | Intel Xeon E5-2680v2 — 10 cores / 20 threads |
| **RAM** | 64 GB |
| **GPU** | **NVIDIA A30 (24 GB HBM2)** |
| **Proposito** | Machine Learning dedicada |
| **Hostname** | `nabucodonosor.ccad.unc.edu.ar` |
| **Acceso** | Requiere solicitud explicita |

**MUY RELEVANTE para agentes IA**: Es una maquina standalone con GPU, probablemente con internet directo y sin las restricciones de cgroups agresivas de los login nodes de clusters. Ideal para correr Claude Code + training en GPU simultaneamente.

### 3.2 Jupyter (interactiva)

| Caracteristica | Valor |
|---------------|-------|
| **CPU** | AMD Ryzen Threadripper PRO 3975WX — 32 cores / 64 threads |
| **RAM** | 125 GB |
| **GPU** | Intel Arc A770 (12 GB) — NO sirve para CUDA/PyTorch |
| **Proposito** | Trabajo interactivo, notebooks |
| **Hostname** | `jupyter.ccad.unc.edu.ar` |

**Nota**: La Intel Arc A770 usa drivers oneAPI, NO CUDA. No es util para PyTorch/TensorFlow estandar. Pero la maquina es muy buena para trabajo interactivo CPU (32 cores, 125 GB RAM). Claude Code probablemente funciona bien aqui.

---

## 4. Nodos Cabecera vs Nodos de Computo — CRITICO

Este es el concepto mas importante para entender por que ciertos programas fallan en HPC.

### 4.1 Nodo Cabecera (Head/Login Node)

**Que es**: El servidor al que te conectas cuando haces `ssh usuario@cluster.ccad.unc.edu.ar`. Es COMPARTIDO por TODOS los usuarios del cluster simultaneamente.

**Que SI se puede hacer**:
- Editar archivos, scripts
- Compilar codigo (moderadamente)
- Enviar jobs a SLURM (`sbatch`, `srun`, `squeue`, `scancel`)
- Transferir archivos (`scp`, `rsync`, `wget`)
- Usar `tmux`/`screen` para persistencia
- Tareas administrativas ligeras

**Que NO se puede hacer**:
- **Ejecutar programas pesados** (esta PROHIBIDO por politica del CCAD)
- Procesos que consuman mucha RAM (limite de cgroups)
- Procesos que usen mucha CPU por tiempo extendido
- Training de modelos
- Cualquier carga computacional significativa

**Restricciones tipicas de cgroups en login nodes HPC** (el CCAD no publica valores exactos, pero la industria estandar es):

| Recurso | Limite tipico |
|---------|--------------|
| CPU time por proceso | 30 min de CPU time |
| RAM por usuario (agregada) | 4-8 GB across all processes |
| Cores | 2-4 de los disponibles |
| Enforcement | Linux cgroups — OOM killer silencioso |

**IMPORTANTE**: Cuando el OOM killer mata un proceso por exceder el limite de cgroups, NO muestra ningun mensaje de error. El proceso simplemente desaparece. Esto es casi con certeza parte de por que Claude Code "se cierra" en el login node.

### 4.2 Nodo de Computo (Compute Node)

**Que es**: Un servidor dedicado exclusivamente a tu job durante el tiempo que SLURM te asigna. Tienes acceso a TODOS los recursos del nodo (CPU, RAM, GPU).

**Que SI se puede hacer**:
- Todo el computo pesado
- Training de modelos
- Uso completo de GPU
- Uso de toda la RAM del nodo

**Que NO se puede hacer**:
- **Acceder a internet** — los nodos de computo estan en red interna sin salida a internet
- Esto significa que Claude Code NO puede correr en un nodo de computo directamente (necesita llamar a api.anthropic.com)
- Workaround posible: tunel SSH a traves del nodo cabecera (ver seccion 18)

### 4.3 Tabla resumen

| Aspecto | Nodo Cabecera | Nodo de Computo | Standalone (Nabu/Jupyter) |
|---------|--------------|-----------------|--------------------------|
| Acceso | SSH directo | Solo via SLURM | SSH directo |
| Internet | SI | NO | Probablemente SI |
| CPU | Limitado (cgroups) | Completo | Completo |
| RAM | Limitada (cgroups) | Completa | Completa |
| GPU | N/A | Completa | Completa |
| Compartido | SI (todos los users) | NO (exclusivo) | Depende |
| Claude Code | Funciona con limitaciones | No funciona (sin internet) | Deberia funcionar |

---

## 5. Sistema de Almacenamiento

### 5.1 /home (persistente, compartido)

```
/home/$USER/
```

- **Tipo**: NFS (Network File System), montado en TODOS los nodos (cabecera y computo)
- **Persistencia**: Permanente — los archivos persisten entre jobs y sesiones
- **Velocidad**: LENTA para I/O intensivo (red, compartido, no optimizado para throughput)
- **Uso**: codigo fuente, scripts, resultados finales, configuraciones
- **Cuota**: variable por usuario (consultar con administradores)
- **Backup**: NO garantizado por defecto (hacer backups propios)

### 5.2 /home/shared (directorios compartidos entre grupos)

```
/home/shared/<nombre_grupo>/
```

- Directorios compartidos entre miembros de un grupo de investigacion
- Se solicitan por email a soporte CCAD
- Mismas caracteristicas que /home pero con permisos de grupo

### 5.3 /scratch (temporal, local, RAPIDO)

```
/scratch/
```

- **Tipo**: XFS sobre disco local del nodo (SSD o NVMe segun cluster)
- **Persistencia**: **SE BORRA AL TERMINAR EL JOB** — no guardar resultados permanentes aqui
- **Velocidad**: RAPIDA (disco local, sin red)
- **Uso**: datos temporales durante el job, datasets intermedios, checkpoints temporales
- **Patron tipico**:
  1. Al inicio del job: copiar datos de `/home` a `/scratch`
  2. Ejecutar el computo usando `/scratch`
  3. Al final del job: copiar resultados de `/scratch` a `/home`

### 5.4 /tmp (temporal, RAM)

```
/tmp/
```

- **Tipo**: tmpfs (montado en RAM)
- **Persistencia**: SE BORRA al terminar el job
- **Velocidad**: La mas rapida posible (es RAM)
- **PELIGRO**: Usa la RAM del nodo — evitar. Limite recomendado: no mas del 50% de la RAM
- **Uso**: solo para archivos temporales muy pequenos y efimeros

### 5.5 Patron de uso recomendado para training ML

```bash
#!/bin/bash
#SBATCH ...

# 1. Crear directorio de trabajo en scratch
WORKDIR=/scratch/$SLURM_JOB_ID
mkdir -p $WORKDIR

# 2. Copiar dataset y codigo
cp -r /home/$USER/datasets/maestro $WORKDIR/
cp -r <repo-root> $WORKDIR/phideus/

# 3. Ejecutar training
cd $WORKDIR/phideus
python train.py --data $WORKDIR/maestro --output $WORKDIR/results

# 4. Copiar resultados a home
cp -r $WORKDIR/results /home/$USER/results_$(date +%Y%m%d)

# 5. Limpiar (opcional, se borra automaticamente)
rm -rf $WORKDIR
```

---

## 6. SLURM — El Gestor de Trabajos

SLURM (Simple Linux Utility for Resource Management) es el sistema que administra los recursos de computo. Todos los clusters del CCAD usan SLURM.

### 6.1 Conceptos Fundamentales

| Concepto | Descripcion |
|----------|-------------|
| **Job** | Una unidad de trabajo enviada a SLURM |
| **Partition** | Un grupo de nodos con politicas comunes (tiempo maximo, prioridad) |
| **Node** | Un servidor fisico |
| **Task** | Un proceso dentro de un job |
| **CPU** | Un core de procesador |
| **GRES** | Generic Resource (GPUs, por ejemplo) |

### 6.2 Comandos Esenciales

```bash
# Ver estado de la cola
squeue                      # Todos los jobs
squeue -u $USER             # Mis jobs
squeue -p multi             # Jobs en particion 'multi'

# Enviar un job batch
sbatch mi_script.sh

# Job interactivo
srun --pty bash              # Shell interactivo en nodo de computo
srun --gpus=1 --pty bash     # Shell interactivo con GPU

# Reservar recursos (sin ejecutar inmediatamente)
salloc -p multi -N 1 --gres=gpu:1

# Cancelar un job
scancel <JOB_ID>
scancel -u $USER             # Cancelar TODOS mis jobs

# Info sobre particiones
sinfo                        # Estado general
sinfo -p multi               # Particion especifica

# Info detallada sobre un job
scontrol show job <JOB_ID>

# Historial de jobs
sacct -u $USER --starttime=2026-02-01

# Ver recursos disponibles en nodos
sinfo -N -l                  # Nodos con detalle
sinfo -o "%n %c %m %G %t"   # Custom: nodo, cores, RAM, GPU, estado
```

### 6.3 Variables de Entorno SLURM

Dentro de un job, SLURM define variables utiles:

| Variable | Contenido |
|----------|-----------|
| `$SLURM_JOB_ID` | ID del job |
| `$SLURM_JOB_NAME` | Nombre del job |
| `$SLURM_NODELIST` | Lista de nodos asignados |
| `$SLURM_NNODES` | Numero de nodos |
| `$SLURM_NTASKS` | Numero de tasks |
| `$SLURM_CPUS_PER_TASK` | CPUs por task |
| `$SLURM_SUBMIT_DIR` | Directorio desde donde se envio el job |
| `$SLURM_TMPDIR` | Directorio temporal en /scratch |

---

## 7. Particiones SLURM por Cluster

### 7.1 Serafin

| Particion | Tiempo maximo | Nodos max | Default | Notas |
|-----------|--------------|-----------|---------|-------|
| **short** | 1 hora | — | SI | Para pruebas rapidas |
| **multi** | 2 dias | — | NO | Para trabajos largos |

### 7.2 Mendieta (GPU)

| Particion | Tiempo maximo | Nodos max | Default | Notas |
|-----------|--------------|-----------|---------|-------|
| **short** | 1 hora | — | SI | Para pruebas rapidas con GPU |
| **multi** | 2 dias | — | NO | Para training largo con GPU |

### 7.3 Mulatona

| Particion | Tiempo maximo | Nodos max | Default | Notas |
|-----------|--------------|-----------|---------|-------|
| **short** | 1 hora | 1 | SI | Para pruebas rapidas |
| **mono** | 2 dias | 1 | NO | Para trabajos largos (1 nodo max) |

### 7.4 Eulogia

| Particion | Tiempo maximo | Nodos max | Default | Notas |
|-----------|--------------|-----------|---------|-------|
| **short** | 1 hora | — | NO | Para pruebas rapidas |
| **multi** | 4 dias | — | SI | Default: 4 dias max (la mas generosa) |

### 7.5 Implicaciones para training ML

- **Mendieta multi** da maximo **2 dias** de GPU — 48 horas continuas
- Si tu training dura mas de 48h, necesitas:
  - Checkpoint recovery (guardar estado y re-lanzar)
  - O solicitar excepciones al administrador
- **Estrategia**: checkpoints cada N epochs + script de relanzamiento automatico

---

## 8. Jobs Interactivos (srun, salloc)

### 8.1 srun basico (shell interactivo)

```bash
# Shell en un nodo de computo (sin GPU)
srun --pty bash

# Shell con GPU en Mendieta
srun --gpus=1 --pty bash

# Shell con mas recursos
srun -p multi -N 1 --cpus-per-task=10 --mem=32G --gpus=1 --time=4:00:00 --pty bash

# Shell con 2 GPUs
srun -p multi --gpus=2 --pty bash
```

### 8.2 salloc (reserva sin ejecutar)

```bash
# Reservar recursos
salloc -p multi -N 1 --gres=gpu:1 --time=8:00:00

# Una vez asignado, conectarse al nodo
srun --pty bash

# Al terminar
exit  # salir del shell interactivo
exit  # liberar la reserva (o scancel)
```

### 8.3 Advertencia sobre stdin en SLURM

**CRITICO PARA CLAUDE CODE**: Los jobs interactivos de SLURM (`srun --pty bash`) tienen un manejo especial de stdin/stdout/stderr. Hay un bug conocido en Claude Code v2.0.43+ donde los subprocesos de deteccion de shell consumen stdin, causando que el prompt reciba EOF y se cierre. Ver seccion 17 para detalles y soluciones.

---

## 9. Scripts de Lanzamiento (sbatch)

### 9.1 Estructura basica

```bash
#!/bin/bash
#SBATCH --job-name=mi_trabajo       # Nombre del job
#SBATCH --partition=multi           # Particion
#SBATCH --nodes=1                   # Numero de nodos
#SBATCH --ntasks-per-node=1         # Tasks por nodo
#SBATCH --cpus-per-task=10          # CPUs por task
#SBATCH --mem=0                     # Toda la RAM del nodo (0=todo)
#SBATCH --time=2-00:00              # Tiempo maximo (dias-horas:minutos)
#SBATCH --output=slurm_%j.out       # Archivo de salida (%j = job ID)
#SBATCH --error=slurm_%j.err        # Archivo de errores
#SBATCH --mail-type=ALL             # Notificaciones por email
#SBATCH --mail-user=tu@email.com    # Email para notificaciones

# Cargar modulos necesarios
. /etc/profile
module load gcc cuda

# Variables de entorno
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Ejecutar
srun python mi_script.py
```

### 9.2 Template para training ML con GPU en Mendieta

```bash
#!/bin/bash
#SBATCH --job-name=phideus_train
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=2-00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

. /etc/profile
module load gcc cuda

# Setup entorno Python
source /home/$USER/venv/bin/activate

# Configuracion PyTorch
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Copiar datos a scratch para I/O rapido
WORKDIR=/scratch/$SLURM_JOB_ID
mkdir -p $WORKDIR
cp -r /home/$USER/data/maestro_v3 $WORKDIR/

# Training
srun python <repo-root>/experiments/bias_control/gate42_training.py \
    --descriptor d4a4 \
    --checkpoint <repo-root>/models/foundation_locked_e25.pt \
    --output outputs/gate43_$(date +%Y%m%d_%H%M) \
    --maestro-dir $WORKDIR/maestro_v3/maestro-v3.0.0 \
    --epochs 30 --batch-size 16 --num-workers 8 \
    --freeze-policy run-d --seed 42 --device cuda

# Copiar resultados
cp -r outputs/gate43_* results/
```

### 9.3 Template con checkpoint recovery (para runs > 48h)

```bash
#!/bin/bash
#SBATCH --job-name=phideus_resume
#SBATCH --partition=multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=2-00:00

. /etc/profile
module load gcc cuda
source /home/$USER/venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Buscar ultimo checkpoint
OUTDIR=outputs/gate43_long_run
LAST_CKPT=$(ls -t $OUTDIR/checkpoint_epoch*.pt 2>/dev/null | head -1)

if [ -n "$LAST_CKPT" ]; then
    echo "Resuming from $LAST_CKPT"
    RESUME_FLAG="--resume $LAST_CKPT"
else
    echo "Starting from scratch"
    RESUME_FLAG=""
fi

srun python <repo-root>/experiments/train.py \
    --output $OUTDIR \
    $RESUME_FLAG \
    --epochs 30 --device cuda

# Auto-resubmit si no termino (opcional)
if [ ! -f "$OUTDIR/training_complete.flag" ]; then
    echo "Training incompleto, re-enviando job..."
    sbatch $0
fi
```

---

## 10. Sistema de Modulos (module)

El CCAD usa el sistema de modulos de entorno (Lmod/Environment Modules) para gestionar software preinstalado.

### 10.1 Comandos basicos

```bash
module av                    # Listar TODOS los modulos disponibles
module av gcc                # Buscar modulos con "gcc" en el nombre
module load gcc              # Cargar modulo
module load gcc cuda         # Cargar multiples modulos
module list                  # Ver modulos cargados
module unload gcc            # Descargar un modulo
module purge                 # Descargar TODOS los modulos
module show gcc              # Ver que hace un modulo (paths, variables)
module spider python         # Buscar en todos los modulos (mas exhaustivo)
```

### 10.2 Compiladores disponibles por cluster

| Cluster | Compiladores disponibles |
|---------|-------------------------|
| **Serafin** | GCC 12, AMD AOCC 4.1 |
| **Mendieta** | GCC 11, NVIDIA HPC SDK 22.3 |
| **Mulatona** | GCC 12, Intel 2021, Intel OneAPI 2023 |
| **Boogie** | GCC 13 |

### 10.3 CUDA en Mendieta

```bash
module load cuda             # Cargar CUDA toolkit
nvcc --version               # Verificar version
nvidia-smi                   # Ver GPUs disponibles (solo en nodo con GPU)
```

**IMPORTANTE**: `nvidia-smi` solo funciona en nodos de computo con GPU o en Nabucodonosor. En el nodo cabecera de Mendieta NO hay GPU, asi que nvidia-smi fallara.

### 10.4 Flags de optimizacion

Despues de cargar un modulo de compilador, verificar flags con:

```bash
env | grep FLAGS
echo $CFLAGS $CXXFLAGS $FFLAGS
```

Estos flags se configuran automaticamente para optimizar para la arquitectura del cluster.

---

## 11. Gestion de Software y Entornos Python

### 11.1 Miniconda (recomendado para Python)

```bash
# Instalar Miniconda en /home
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate

# Crear entorno
conda create -n phideus python=3.11 pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda activate phideus
```

### 11.2 venv (si Python ya esta disponible)

```bash
module load python  # si existe el modulo
python3 -m venv $HOME/venv
source $HOME/venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 11.3 Spack (gestor de paquetes HPC)

```bash
# Instalar Spack
git clone --branch releases/v0.21 https://github.com/spack/spack.git ~/spack
source ~/spack/share/spack/setup-env.sh

# Instalar software
spack install python@3.11
spack install py-torch
spack load python@3.11
```

### 11.4 Node.js (necesario para Claude Code via npm)

```bash
# Opcion A: buscar modulo
module av | grep -i node

# Opcion B: instalar con nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install --lts
nvm use --lts
node --version  # verificar
npm --version   # verificar

# Opcion C: instalar con conda
conda install -c conda-forge nodejs
```

---

## 12. Conectividad y Red

### 12.1 Acceso SSH

```bash
# Desde tu maquina local
ssh usuario@serafin.ccad.unc.edu.ar      # Serafin
ssh usuario@mendieta.ccad.unc.edu.ar      # Mendieta
ssh usuario@nabucodonosor.ccad.unc.edu.ar # Nabucodonosor
ssh usuario@jupyter.ccad.unc.edu.ar       # Jupyter

# Con clave SSH (recomendado)
ssh-keygen -t ed25519 -f ~/.ssh/ccad
ssh-copy-id -i ~/.ssh/ccad.pub usuario@serafin.ccad.unc.edu.ar
```

### 12.2 Internet desde nodos

| Tipo de nodo | Internet | Consecuencia para Claude Code |
|-------------|----------|------------------------------|
| Nodo cabecera | SI | Claude Code puede llamar a API |
| Nodo de computo | **NO** | Claude Code NO funciona directamente |
| Nabucodonosor | Probablemente SI | Claude Code deberia funcionar |
| Jupyter | Probablemente SI | Claude Code deberia funcionar |

### 12.3 Tunel SSH para internet desde nodo de computo

Si necesitas internet desde un nodo de computo (por ejemplo, para Claude Code + GPU):

```bash
# Desde el nodo de computo, tunel a traves del cabecera:
ssh -D 1080 -N head.ivb.ccad.unc.edu.ar &

# Configurar proxy SOCKS
export https_proxy=socks5://localhost:1080
export HTTPS_PROXY=socks5://localhost:1080

# O tunel directo a la API de Anthropic
ssh -L 8443:api.anthropic.com:443 head.ivb.ccad.unc.edu.ar -N &
```

**NOTA**: Esto es avanzado y puede no estar permitido por politica del CCAD. Consultar con administradores antes de implementar.

---

## 13. Transferencia de Archivos

### 13.1 Metodos disponibles

```bash
# SCP (simple)
scp archivo.tar.gz usuario@serafin.ccad.unc.edu.ar:/home/usuario/

# rsync (incremental, recomendado para directorios grandes)
rsync -avhP --update ./data/ usuario@serafin.ccad.unc.edu.ar:/home/usuario/data/

# SSHFS (montar directorio remoto como local)
sshfs usuario@serafin.ccad.unc.edu.ar:/home/usuario ~/mnt_ccad

# wget/curl (descargar desde internet, solo en cabecera)
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip
```

### 13.2 Transferencia de datasets grandes

Para datasets como MAESTRO (120 GB), descargar directamente en el nodo cabecera:

```bash
# En el nodo cabecera (tiene internet)
cd /home/$USER/data
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip
unzip maestro-v3.0.0.zip
rm maestro-v3.0.0.zip  # liberar espacio

# Verificar cuota
quota -s  # si esta configurado
du -sh /home/$USER/
```

---

## 14. tmux — Persistencia de Sesiones

tmux es esencial en HPC para mantener sesiones persistentes cuando la conexion SSH se cae.

### 14.1 Comandos basicos

```bash
# Crear sesion nueva
tmux new-session -s trabajo

# Desconectarse (detach) — la sesion sigue corriendo
# Ctrl+b, luego d

# Reconectarse
tmux attach -t trabajo

# Listar sesiones
tmux ls

# Enviar comandos a una sesion existente (util para scripting)
tmux send-keys -t trabajo "python train.py" Enter

# Matar sesion
tmux kill-session -t trabajo
```

### 14.2 Patron para training largo

```bash
# 1. Crear sesion tmux en el nodo cabecera
tmux new-session -s training

# 2. Dentro de tmux, lanzar job interactivo
srun -p multi --gpus=1 --time=48:00:00 --pty bash

# 3. Una vez en el nodo de computo, activar entorno y lanzar training
source ~/venv/bin/activate
python train.py --epochs 30 ...

# 4. Desconectarse de tmux (Ctrl+b, d)
# 5. Reconectarse despues: tmux attach -t training
```

**ADVERTENCIA**: tmux en el nodo cabecera persiste, pero el job de SLURM dentro de tmux tiene un tiempo maximo. Si el job termina (por timeout), el shell dentro de tmux tambien muere. El tmux sigue vivo pero el contenido (el job) ya no.

---

## 15. GPU — Uso Especifico en Mendieta y Nabucodonosor

### 15.1 Solicitar GPU en Mendieta

```bash
# 1 GPU
srun -p multi --gpus=1 --pty bash

# 2 GPUs (para multi-GPU training)
srun -p multi --gpus=2 --pty bash

# 1 GPU + especificar recursos
srun -p multi --gpus=1 --cpus-per-task=10 --mem=60G --time=24:00:00 --pty bash

# En script sbatch
#SBATCH --gpus-per-task=1
```

### 15.2 Verificar GPU dentro del job

```bash
nvidia-smi                           # Ver GPUs asignadas
nvidia-smi -L                        # Listar GPUs
nvidia-smi --query-gpu=memory.used,memory.total --format=csv  # VRAM
```

### 15.3 PyTorch + CUDA en Mendieta

```python
import torch
print(torch.cuda.is_available())       # True si GPU esta disponible
print(torch.cuda.device_count())       # Numero de GPUs
print(torch.cuda.get_device_name(0))   # Nombre de la GPU
print(torch.cuda.mem_get_info(0))      # VRAM libre y total

# Usar GPU
device = torch.device('cuda')
model = model.to(device)
```

### 15.4 NVIDIA A30 — Consideraciones especificas

La A30 tiene HBM2 en vez de GDDR6X (como la RTX 3090):
- **Bandwidth**: ~933 GB/s (vs ~936 GB/s en RTX 3090 — similar)
- **VRAM**: 24 GB (igual que RTX 3090)
- **Tensor Cores**: 224 (3ra gen)
- **TF32**: Soportado — acelera operaciones de matmul significativamente
- **BF16**: Soportado nativamente
- **Sin salida de video**: No tiene puertos de video (datacenter only)
- **TDP**: 165W (vs 350W de RTX 3090 — mucho mas eficiente)

Para PyTorch, habilitar TF32 para mejor performance:
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

---

## 16. Restricciones y Politicas

### 16.1 Politicas explicitas del CCAD

1. **Ejecutar programas pesados en el nodo cabecera esta PROHIBIDO**
2. Todo computo debe pasar por SLURM
3. Las particiones tienen tiempos maximos (1h short, 2-4 dias multi)
4. Los datos en /scratch se borran al terminar el job
5. No se garantiza backup de /home (hacer backups propios)

### 16.2 Politicas implicitas (practica estandar HPC)

1. **cgroups en login nodes**: limitan CPU y RAM por usuario
2. **Fair-share scheduling**: si usas muchos recursos, tu prioridad baja
3. **Cuotas de disco**: limites en /home (consultar)
4. **No minar criptomonedas** (prohibicion universal en HPC academico)
5. **No almacenar datos sensibles sin autorizacion**

### 16.3 Consecuencias de violar politicas

- Procesos pesados en login node: **killed por cgroups sin aviso**
- Exceder cuota de disco: no poder crear archivos nuevos
- Exceder tiempo de job: job cancelado por SLURM
- Uso indebido: posible suspension de cuenta

---

## 17. Claude Code en HPC — Bug Conocido y Soluciones

### 17.1 El Problema

Claude Code se cierra inmediatamente al intentar escribir en el prompt. El proceso arranca, muestra el banner, pero al presionar cualquier tecla, se cierra con codigo 0.

### 17.2 Causa Raiz: Bug #12507

**GitHub Issue**: https://github.com/anthropics/claude-code/issues/12507

**Titulo**: "[BUG] Claude Code exits immediately on HPC interactive sessions - stdin consumed by shell detection subprocesses"

**Mecanismo**: Claude Code v2.0.43+ lanza subprocesos para detectar el entorno de shell (aliases, opciones de shopt, etc.). Estos subprocesos **heredan stdin** y lo consumen. Cuando terminan, el prompt principal recibe EOF y se cierra limpiamente.

**Version afectada**: v2.0.43 y posteriores. v2.0.42 y anteriores NO tienen este bug.

**Confirmado via strace**: Los subprocesos de deteccion leen de fd 0 (stdin), drenando el buffer.

### 17.3 Soluciones (ordenadas por probabilidad de exito)

#### Solucion 1: Instalar via npm (RECOMENDADA)

Multiples usuarios confirman que la instalacion via npm funciona incluso con versiones recientes:

```bash
# Si Node.js esta disponible
npm install -g @anthropic-ai/claude-code
claude
```

Si no hay Node.js:
```bash
# Instalar nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install --lts

# Instalar Claude Code
npm install -g @anthropic-ai/claude-code
claude
```

#### Solucion 2: Downgrade a v2.0.42

```bash
curl -fsSL https://claude.ai/install.sh | bash -s 2.0.42
```

Despues, prevenir auto-update (si es posible en la configuracion de Claude Code).

#### Solucion 3: Usar maquina standalone

Conectarse a Nabucodonosor o Jupyter donde no hay SLURM intermediando stdin:

```bash
ssh usuario@nabucodonosor.ccad.unc.edu.ar
# o
ssh usuario@jupyter.ccad.unc.edu.ar

claude
```

#### Solucion 4: Modo no-interactivo

Si nada funciona interactivamente, el modo `-p` (prompt directo) SI funciona:

```bash
claude -p "analiza este codigo y sugiere mejoras" < mi_archivo.py
claude -p "escribe un script sbatch para training con 1 GPU en Mendieta"
```

#### Solucion 5: Workarounds de stdin (menor probabilidad)

Estos fueron reportados como fallidos por algunos usuarios, pero vale intentar:

```bash
# Intentar con script
script -q /dev/null -c "claude"

# Intentar redirigiendo stdin
claude < /dev/tty

# Intentar con shell limpio
bash --norc --noprofile -c "claude"

# Intentar con setsid
setsid bash -c "claude </dev/tty" </dev/tty
```

### 17.4 Escenario optimo para Claude Code + GPU

```
                 SSH
Tu maquina ──────────> Nabucodonosor (standalone)
                          │
                          ├── Internet: SI (puede llamar API)
                          ├── GPU: NVIDIA A30 24GB
                          ├── No SLURM (no bug stdin)
                          ├── No cgroups agresivos
                          └── Claude Code funciona directamente
```

Si Nabucodonosor no esta disponible:

```
                 SSH
Tu maquina ──────────> Cabecera Mendieta
                          │
                          ├── Internet: SI
                          ├── GPU: NO
                          ├── cgroups: SI (limites RAM/CPU)
                          ├── Claude Code: puede funcionar (con npm install)
                          └── Solo para tareas ligeras, NO training
                          │
                     sbatch/srun
                          │
                    Nodo de computo Mendieta
                          │
                          ├── Internet: NO (Claude Code no funciona)
                          ├── GPU: 2x NVIDIA A30
                          └── Para training via scripts sbatch
```

---

## 18. Recetas Operativas para Agentes IA

### 18.1 Workflow: Claude Code en cabecera + training en GPU via sbatch

Este es el patron mas practico si Claude Code funciona en el nodo cabecera:

1. **Claude Code corre en el nodo cabecera** (dentro de tmux para persistencia)
2. **Claude Code escribe scripts sbatch** y los envia con `sbatch`
3. **El training corre en nodo de computo** con GPU
4. **Claude Code monitorea** con `squeue`, `tail -f slurm_*.out`
5. **Resultados vuelven a /home** donde Claude Code los puede leer

```bash
# Paso 1: tmux + Claude Code en cabecera
tmux new-session -s claude
claude

# Paso 2: Dentro de Claude, pedir que escriba y envie un script
# "Escribe un sbatch para entrenar d4a4 en Mendieta con 1 GPU, 30 epochs"

# Paso 3: Claude ejecuta
sbatch scripts/train_d4a4.sh
# Output: Submitted batch job 12345

# Paso 4: Monitorear
squeue -u $USER
tail -f logs/train_12345.out
```

### 18.2 Workflow: Claude Code en Nabucodonosor (todo junto)

Si tienes acceso a Nabucodonosor, todo es mas simple:

```bash
ssh usuario@nabucodonosor.ccad.unc.edu.ar
tmux new-session -s claude
claude
# Claude tiene GPU + internet + sin restricciones de SLURM
# Puede correr training directamente como en una maquina local
```

### 18.3 Monitoreo de jobs desde Claude Code

```bash
# Estado de mis jobs
squeue -u $USER -o "%.10i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R"

# Output del job en tiempo real
tail -f slurm_12345.out

# Uso de GPU (si tienes acceso al nodo)
ssh nodo_computo nvidia-smi  # probablemente no funcione (sin SSH entre nodos)

# Alternativa: incluir nvidia-smi en el script sbatch
# El output aparecera en slurm_*.out
```

### 18.4 Variables de entorno utiles para scripts

```bash
# En .bashrc del CCAD
export MAESTRO_DIR=/home/$USER/data/maestro_v3/maestro-v3.0.0
export FOUNDATION=<repo-root>/models/foundation_locked_e25.pt
export OUTPUT_BASE=outputs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## 19. Troubleshooting

### 19.1 Claude Code se cierra inmediatamente

**Diagnostico**: ver seccion 17. Probable bug stdin (v2.0.43+) o OOM killer (cgroups).

**Verificar version**:
```bash
claude --version
```

**Verificar si fue OOM**:
```bash
dmesg | tail -20  # puede no tener permisos
journalctl --user -n 50  # alternativa
```

### 19.2 "Permission denied" al instalar con npm

```bash
# Instalar en directorio local
npm install -g --prefix=$HOME/.npm-global @anthropic-ai/claude-code
export PATH=$HOME/.npm-global/bin:$PATH
echo 'export PATH=$HOME/.npm-global/bin:$PATH' >> ~/.bashrc
```

### 19.3 CUDA not available en nodo de computo

```bash
# Verificar que cargaste el modulo
module list
module load cuda
python -c "import torch; print(torch.cuda.is_available())"

# Verificar que SLURM asigno GPU
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
```

### 19.4 Job cancelado por tiempo

```bash
# Verificar tiempo maximo de la particion
sinfo -p multi -o "%P %l %a"

# Usar particion con mas tiempo
#SBATCH --partition=multi  # 2 dias en Mendieta
#SBATCH --time=2-00:00

# Implementar checkpoint recovery (ver seccion 9.3)
```

### 19.5 Disco lleno / cuota excedida

```bash
# Verificar uso
du -sh /home/$USER/
du -sh /home/$USER/* | sort -rh | head -20

# Limpiar
rm -rf /home/$USER/.cache/pip  # cache de pip
conda clean --all              # cache de conda
find /home/$USER -name "*.pyc" -delete  # bytecode Python
find /home/$USER -name "__pycache__" -type d -exec rm -rf {} +
```

### 19.6 Job pendiente mucho tiempo (PENDING)

```bash
# Ver razon
squeue -u $USER -o "%.10i %.9P %.30j %.8T %R"
# Razones comunes:
# Priority - esperando por fair-share
# Resources - no hay nodos disponibles
# QOSMaxJobsPerUserLimit - limite de jobs simultaneos

# Ver disponibilidad
sinfo -p multi
```

---

## 20. Contacto y Soporte CCAD

- **Email general**: soporte-ccad@unc.edu.ar (verificar en wiki)
- **Wiki**: https://wiki.ccad.unc.edu.ar/
- **Solicitud de cuenta**: via formulario en la wiki
- **Solicitud de acceso a Nabucodonosor**: email explicando necesidad de ML/GPU
- **Solicitud de directorio compartido**: email con nombre del grupo y miembros

---

## APENDICE A: Resumen de Hostnames

| Recurso | Hostname externo | Hostname interno | IP interna |
|---------|-----------------|-----------------|------------|
| Serafin (cabecera) | serafin.ccad.unc.edu.ar | head.rome.ccad.unc.edu.ar | 10.6.10.250 |
| Mendieta (cabecera) | mendieta.ccad.unc.edu.ar | head.ivb.ccad.unc.edu.ar | 10.10.10.250 |
| Mulatona (cabecera) | mulatona.ccad.unc.edu.ar | head.bdw.ccad.unc.edu.ar | 10.3.10.250 |
| Eulogia (cabecera) | eulogia.ccad.unc.edu.ar | head.knl.ccad.unc.edu.ar | 10.2.10.250 |
| Nabucodonosor | nabucodonosor.ccad.unc.edu.ar | — | — |
| Jupyter | jupyter.ccad.unc.edu.ar | — | — |

## APENDICE B: Cheatsheet de Comandos

```bash
# === SLURM ===
sbatch script.sh              # Enviar job batch
srun --gpus=1 --pty bash      # Job interactivo con GPU
squeue -u $USER               # Mis jobs
scancel <JOBID>               # Cancelar job
sinfo -p multi                # Estado particion

# === MODULOS ===
module av                     # Listar disponibles
module load gcc cuda          # Cargar modulos
module list                   # Ver cargados
module purge                  # Descargar todos

# === TMUX ===
tmux new -s nombre            # Nueva sesion
tmux attach -t nombre         # Reconectar
tmux ls                       # Listar sesiones
# Ctrl+b, d                   # Desconectar

# === GPU ===
nvidia-smi                    # Estado GPU
nvidia-smi -l 5               # Monitoreo cada 5s

# === ALMACENAMIENTO ===
du -sh /home/$USER/           # Uso de disco
ls /scratch/                  # Scratch local (en nodo computo)

# === CLAUDE CODE ===
npm install -g @anthropic-ai/claude-code  # Instalar (npm)
claude                                     # Iniciar interactivo
claude -p "prompt"                         # Modo no-interactivo
```

---

> **Nota final**: Este documento fue elaborado a partir de la wiki oficial del CCAD (https://wiki.ccad.unc.edu.ar/), documentacion publica de HPC, y el issue tracker de Claude Code. Los valores exactos de cgroups y algunas politicas internas pueden variar — consultar con administradores del CCAD para confirmacion.






















  JupyterHub en CCAD/UNC — Todo lo que encontré                                                                                                             
                                                                                                                                                            
  La infraestructura del CCAD tiene 3 capas de acceso                                                                                                       

  1. Clusters HPC con SLURM (lo que usamos hoy):                                                                                                            
  - Mendieta — 22 nodos × NVIDIA A30 (24GB), particiones multi (48h) y short (1h). Acceso via SSH + sbatch.                                                 
  - Serafín, Eulogia, Mulatona — CPU clusters, sin GPU.                                                                                                     

  2. Máquinas dedicadas (sin SLURM):
  - Nabucodonosor — Para ML. Specs: Xeon E5-2680v2, 64GB RAM, 3× GTX 1080Ti (11GB cada una). SSH directo a nabucodonosor.ccad.unc.edu.ar. Requiere solicitar
   acceso explícitamente. Software viejo: CUDA 8/9, driver 390.
  - Jupyter — Para computación interactiva. AMD Threadripper PRO 3975WX (32C/64T), 125GB RAM, Intel Arc A770 (12GB). SSH a jupyter.ccad.unc.edu.ar. Esta es
  la que NO queremos confundir con JupyterHub.

  3. Servicios web:
  - jupyterhub.ccad.unc.edu.ar — Interfaz web tipo Google Colab. Login con usuario/password (no SSH keys). Si no tenés password, lo generás via reset.
  - chat.ccad.unc.edu.ar — Servicio de ML/LLM (no relevante para nosotros).

  ¿Cómo funciona JupyterHub en un cluster HPC?

  La arquitectura típica (y probablemente la de CCAD):

  Usuario (browser)
      │
      ▼
  JupyterHub (login node)          ◄── Web server + autenticación
      │
      ├── Authenticator              ◄── Verifica usuario/password
      │
      ├── Spawner (BatchSpawner)     ◄── Genera sbatch script
      │       │
      │       ▼
      │   SLURM scheduler            ◄── Encola job como cualquier sbatch
      │       │
      │       ▼
      │   Compute node (GPU)         ◄── Corre jupyter-singleuser como job
      │       │
      │       ▼
      └── Proxy                      ◄── Rutea tráfico browser → compute node

  El flujo concreto:

  1. Abrís jupyterhub.ccad.unc.edu.ar en el browser
  2. Ponés usuario + password (no SSH keys)
  3. Te aparece un formulario ("spawn page") donde elegís:
    - Partición (multi, short)
    - Recursos (cores, memoria, GPUs)
    - Tiempo máximo
  4. El Hub genera un script sbatch internamente y lo submite
  5. Tu job entra en la cola de SLURM como cualquier otro
  6. Cuando el scheduler le da un nodo, arranca un Jupyter server en ese nodo
  7. El proxy del Hub rutea tu browser directamente al compute node
  8. Tenés un notebook corriendo en el compute node con GPU, sin SSH tunneling manual

  Cuando terminás, el job de SLURM se cancela y los recursos se liberan.

  ¿Cómo se diferencia de lo que hacemos hoy?

  ┌──────────────────┬────────────────────────────────┬───────────────────────────────────┐
  │     Aspecto      │ Workflow actual (SSH + sbatch) │            JupyterHub             │
  ├──────────────────┼────────────────────────────────┼───────────────────────────────────┤
  │ Acceso           │ SSH keys + terminal            │ Browser + password                │
  ├──────────────────┼────────────────────────────────┼───────────────────────────────────┤
  │ Submit jobs      │ Escribir .sh, sbatch manual    │ Click en formulario web           │
  ├──────────────────┼────────────────────────────────┼───────────────────────────────────┤
  │ Interactividad   │ Nula (submit & wait)           │ Completa (notebook en vivo)       │
  ├──────────────────┼────────────────────────────────┼───────────────────────────────────┤
  │ GPU              │ Asignada por SLURM en el .sh   │ Asignada por SLURM via formulario │
  ├──────────────────┼────────────────────────────────┼───────────────────────────────────┤
  │ Monitoring       │ squeue, tail de logs           │ Output en celdas del notebook     │
  ├──────────────────┼────────────────────────────────┼───────────────────────────────────┤
  │ Debugging        │ srun --pty bash (interactivo)  │ Celda por celda, inmediato        │
  ├──────────────────┼────────────────────────────────┼───────────────────────────────────┤
  │ Reproducibilidad │ Script .sh + .py separados     │ Todo en un .ipynb                 │
  ├──────────────────┼────────────────────────────────┼───────────────────────────────────┤
  │ Claude Code      │ SSH → terminal → funciona      │ No funciona directo en notebook   │
  ├──────────────────┼────────────────────────────────┼───────────────────────────────────┤
  │ Long runs (>2h)  │ sbatch + resume                │ Limitado al timeout del spawn     │
  ├──────────────────┼────────────────────────────────┼───────────────────────────────────┤
  │ Cola             │ Igual                          │ Igual (usa SLURM por debajo)      │
  └──────────────────┴────────────────────────────────┴───────────────────────────────────┘

  ¿Qué implicaría para nuestro workflow?

  Lo que JupyterHub haría bien:

  1. Prototipado rápido — Probar un nuevo loader, verificar shapes, hacer mini-runs de 20 batches, todo interactivo con feedback inmediato. En vez de
  escribir script → sbatch → esperar cola → leer log → fix → repetir, hacés celda → run → veo resultado → siguiente celda.
  2. Exploración de datos — El audit de alineación de P0, inspeccionar waveforms, plotear distribuciones de F0. Todo visual e inmediato.
  3. Debugging — Cuando un job falla por un bug (como el torch.stack de Gate 6), en vez de submit → fail 13s → leer stderr → fix → resubmit → esperar cola,
  abrís notebook con GPU, cargás un batch, y debugueás celda por celda.
  4. Sin SSH tunneling — Hoy para correr un Jupyter en Mendieta hay que hacer salloc --gres=gpu:1 + SSH tunnel manual + recordar el nodo. JupyterHub lo
  automatiza todo.

  Lo que NO cambiaría o empeoraría:

  1. Training largos — Nuestros runs de 30+ epochs (3-68 horas) siguen necesitando sbatch con checkpointing y auto-resubmit. Un notebook que pierde conexión
   = job muerto. JupyterHub no reemplaza esto.
  2. Claude Code no funciona dentro — Claude UNC opera via SSH en el login node. No puede interactuar con un kernel de Jupyter. Nuestro workflow de "Claude
  escribe script → sbatch → monitorea" no cambia.
  3. La cola es la misma — JupyterHub usa SLURM por debajo. Si la cola tiene 50 jobs, tu notebook espera igual que un sbatch.
  4. Array jobs, multi-node — Para los sweeps de Gate 6 (42 jobs) o Gate 8 (3 arms en paralelo), sbatch sigue siendo la herramienta correcta. No vas a abrir
   42 notebooks.
  5. Reproducibilidad de experimentos — Nuestros scripts .py con argparse y seeds son más reproducibles que notebooks. Los notebooks tienden a acumular
  estado oculto (ejecución fuera de orden, variables de celdas anteriores).

  Modelo híbrido realista

  JupyterHub                          SSH + sbatch
  ─────────────                       ──────────────
  Exploración de datos (P0)           Training largos (P2, Gate 6/8)
  Audit de alineación                 Array jobs / sweeps
  Debug interactivo de bugs           Runs con checkpoint/resume
  Prototipar nuevos loaders           Production runs multi-seed
  Visualización rápida                Claude Code automation
  Mini-runs (20 batches)              Cualquier cosa > 2 horas

  Para empezar

  Lo que necesitaríamos:
  1. Verificar que nuestra cuenta CCAD tenga password habilitado (hoy usamos solo SSH keys)
  2. Ir a jupyterhub.ccad.unc.edu.ar, generar password si no existe
  3. Probar spawn con GPU en partición short (1h)
  4. Verificar que el environment tenga PyTorch + nuestras dependencias (o si hay que instalar)

  La pregunta abierta más importante es qué environment/kernel ofrece el JupyterHub. Si trae un kernel pre-configurado con PyTorch + CUDA, es plug-and-play.
   Si hay que armar el environment desde cero cada vez que se spawnea, la fricción es alta.

  ---
  Sources:
  - https://wiki.ccad.unc.edu.ar/empezar/acceso/index.html
  - https://wiki.ccad.unc.edu.ar/infra/computadoras.html
  - https://wiki.ccad.unc.edu.ar/empezar/acceso/connect.html
  - https://wiki.ccad.unc.edu.ar/empezar/slurm.html
  - https://wiki.ccad.unc.edu.ar/empezar/abrir-cuenta.html
  - https://supercomputo.unc.edu.ar/equipamiento/computadora-nabucodonosor/
  - https://github.com/jipphysics/jupyter-ccad
  - https://github.com/jupyterhub/batchspawner
  - https://jupyterhub.readthedocs.io/en/stable/reference/spawners.html
  - https://www.kth.se/blogs/pdc/2019/01/using-jupyter-notebooks-to-manage-slurm-jobs/
  - https://github.com/silx-kit/jupyterhub_moss
