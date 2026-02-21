# Notas de Claude LOCAL para Codex

> Fecha: 2026-02-20
> Sesión: cosine-tail LR scheduler + batch cosine-tail 60ep
> Commits: (pendiente)

---

## 1. Contexto: problema con el LR scheduler en runs de 60ep

### Resultados observados

Los runs de 30ep (cosine estándar) produjeron los mejores S:
- d4a4: 83.6% (e30)
- a4r: 82.0% (e29)

Los runs de 60ep (cosine estirado a 60K steps) no alcanzan esos niveles:
- d4a4 60ep: best S=79.0% (e25), aún corriendo
- a4r 60ep: final S=79.4% (e60) — nunca alcanzó el 82.0% del 30ep
- D0 60ep: oscila 68-72% desde e15, el control no mejora con más epochs

### Diagnóstico: el LR profile importa más que el número de epochs

Comparando las curvas de LR:

| Epoch | 30ep LR mult | 60ep LR mult |
|-------|-------------|-------------|
| 5 | 0.944 | 0.986 |
| 10 | 0.764 | 0.939 |
| 15 | 0.513 | 0.861 |
| 20 | 0.256 | 0.758 |
| 25 | 0.072 | 0.636 |
| 30 | 0.000 | 0.493 |

El 30ep fuerza una transición agresiva exploración→explotación. A e25 ya tiene LR=0.07 (modo explotación) y ambos modelos alcanzan sus mejores S. El 60ep a e25 todavía tiene LR=0.64 — sigue explorando cuando debería consolidar.

---

## 2. Nuevo scheduler: cosine-tail

### Concepto

Combinar lo mejor de ambos mundos:
1. **Replicar exactamente** la curva del 30ep (cosine agresivo) hasta que el LR llega a 0.10
2. **Cola lineal suave** de 0.10 → 0.02 hasta el final del training
3. Así el modelo nunca queda sin gradiente (como en 30ep) ni demasiado caliente (como en 60ep)

### Implementación

Se extendió `LinearWarmupCosineScheduler` en `gate43_scratch_training.py` con 3 nuevos parámetros:

```
--lr-cosine-ref-epochs 30    # Referencia para la fase cosine (simula run de 30ep)
--lr-floor 0.10              # LR mult donde cosine se detiene y arranca la cola
--lr-tail-end 0.02           # LR mult final al terminar el training
```

### Fases del schedule

```
LR mult
1.00 ─┐
      │╲  cosine (idéntico a 30ep)
      │  ╲
      │    ╲
0.10 ─┤─────╲─────────────────
      │       ╲  cola lineal
      │         ╲___________
0.02 ─┤                      ╲
      └──────────────────────────
      e0     e15    e24   e40   e60
      warm   cosine  tail (0.10→0.02)
```

### Curva LR verificada

| Epoch | Phase | LR mult |
|-------|-------|---------|
| 1 | cosine | 0.999 |
| 5 | cosine | 0.944 |
| 10 | cosine | 0.765 |
| 15 | cosine | 0.513 |
| 20 | cosine | 0.258 |
| 24 | tail | 0.100 |
| 25 | tail | 0.098 |
| 30 | tail | 0.087 |
| 35 | tail | 0.076 |
| 40 | tail | 0.064 |
| 45 | tail | 0.053 |
| 50 | tail | 0.042 |
| 55 | tail | 0.031 |
| 60 | tail | 0.020 |

### Verificación

- Fase cosine: **diff = 0.0** vs scheduler de 30ep estándar (idéntica)
- Backward compatible: sin los nuevos flags, comportamiento idéntico al original
- Transición suave: LR pasa de 0.100 (floor) a 0.020 (tail_end) linealmente
- state_dict/load_state_dict actualizados para resume

### Código modificado

El scheduler ahora tiene 3 modos mutuamente excluyentes:
1. **Estándar** (default): warmup → cosine → 0
2. **Trapezoidal** (--lr-hold-fraction): warmup → hold → cosine → 0
3. **Cosine-tail** (--lr-cosine-ref-epochs): warmup → cosine(ref) → linear tail

---

## 3. Batch cosine-tail 60ep — 4 runs para UNC

### Diseño experimental

Mismas condiciones que los runs de 30ep/60ep existentes, pero con el nuevo scheduler cosine-tail. 60 epochs totales, eval cada 5ep.

### Scripts SLURM creados

| Script | Arm | Output dir | Params | Referencia 30ep |
|--------|-----|-----------|--------|-----------------|
| `batch_60ep_ctail_d0.sh` | D0 (control) | `batch_60ep_ctail_d0/` | ~65M | 72.0% (e30) |
| `batch_60ep_ctail_d4a4.sh` | d4a4 | `batch_60ep_ctail_d4a4/` | ~66.5M | 83.6% (e30) |
| `batch_60ep_ctail_a4r.sh` | a4r | `batch_60ep_ctail_a4r/` | ~68.2M | 82.0% (e29) |
| `batch_60ep_ctail_d4-a4r.sh` | d4-a4r | `batch_60ep_ctail_d4-a4r/` | ~69.6M | 79.8% (e30) |

Todos: seed 42, batch 16, run-d, 1000 batches/ep, eval epochs 5,10,...,55,60.

### Flags clave (comunes a todos)

```
--lr-cosine-ref-epochs 30
--lr-floor 0.10
--lr-tail-end 0.02
```

### Qué buscar en los resultados

1. **S@e25 vs 30ep**: deberían ser ~iguales (misma curva LR hasta e24)
2. **S@e30-e60**: ¿la cola suave permite seguir mejorando? Es la pregunta central
3. **D0 control**: si D0 mejora mucho → el scheduler es mejor en general. Si no → los descriptores son los que aprovechan
4. **d4-a4r**: con +4.6M params, la hipótesis es que se beneficia más de la cola extendida
5. **lr_mult en training_history.json**: verificar que registra los valores del cosine-tail

### Tiempo estimado

~25-31h por run (igual que 60ep estándar). Todos caben en 48h de SLURM.

---

## 4. Estado de jobs UNC actualmente corriendo

Al momento de esta sesión (2026-02-20):

| Job | Estado | Epoch | S más reciente |
|-----|--------|-------|----------------|
| t3-wt 50ep hold | RUNNING | 42/50 | S@e40=80.6% |
| D0 60ep | RUNNING | 42/60 | S@e40=72.4% |
| d4a4 60ep | RUNNING | 39/60 | S@e35=75.6% |
| a4r 60ep | COMPLETADO | 60/60 | S@e60=79.4% |
| d4-a4r 60ep | PENDING | — | — |
| moe-dual 60ep | PENDING | — | — |

Los 4 nuevos scripts cosine-tail se suman a esta cola.

---

## 5. Archivos modificados/creados

| Archivo | Cambio |
|---------|--------|
| `experiments/bias_control/gate43_scratch/gate43_scratch_training.py` | Cosine-tail scheduler mode |
| `experiments/bias_control/slurm/batch_60ep_ctail_d0.sh` | NUEVO |
| `experiments/bias_control/slurm/batch_60ep_ctail_d4a4.sh` | NUEVO |
| `experiments/bias_control/slurm/batch_60ep_ctail_a4r.sh` | NUEVO |
| `experiments/bias_control/slurm/batch_60ep_ctail_d4-a4r.sh` | NUEVO |

---

*Fin de notas — Claude LOCAL, 2026-02-20*
