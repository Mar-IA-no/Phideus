# Gate 4.5 — LR Schedule Optimization

**Estado**: EN CURSO  
**Fecha de corte**: 2026-02-22  
**Origen**: reestructuracion post Gate 4.4 para separar optimizacion de scheduler/ventana temporal de cambios de arquitectura.

---

## Pregunta central

Con arquitectura y descriptores fijos (definidos en Gates 4.3/4.4),
que combinacion de scheduler y duracion (50-60 epocas)
maximiza `S=min(A2M, M2A)` sin perder comparabilidad?

Variable independiente:
- scheduler de LR
- duracion total del entrenamiento

Variables fijas:
- checkpoint/foundation, seed, batch size, freeze policy, protocolo de evaluacion.

---

## Schedulers bajo prueba

| Scheduler | Descripcion | Flags |
|-----------|-------------|-------|
| cosine stretched | Cosine estandar estirado a 60ep | default (`--epochs 60`) |
| trapezoidal hold | hold al pico de LR y luego cosine decay | `--lr-hold-fraction 0.5` |
| cosine-tail | replica curva 30ep hasta LR floor y cola lineal | `--lr-cosine-ref-epochs 30 --lr-floor 0.10 --lr-tail-end 0.02` |

---

## Tabla de runs Gate 4.5

| Run | Scheduler | Estado | Best S | Best ep | Delta vs 30ep |
|-----|-----------|--------|--------|---------|---------------|
| d4a4 60ep | cosine stretched | COMPLETO | 83.8% | e50 | +0.2pp (nuevo record) |
| a4r 60ep | cosine stretched | COMPLETO | 79.4% | e60 | -2.6pp |
| D0 60ep | cosine stretched | COMPLETO | 72.8% | e50 | +12.6pp |
| t3-wt 50ep | trapezoidal hold | COMPLETO | 81.2% | e50 | +1.4pp |
| d4-a4r 60ep | cosine stretched | PENDIENTE UNC | — | — | — |
| moe-dual 60ep | cosine stretched | PENDIENTE UNC | — | — | — |
| D0 60ep | cosine-tail | PENDIENTE UNC | — | — | — |
| d4a4 60ep | cosine-tail | PENDIENTE UNC | — | — | — |
| a4r 60ep | cosine-tail | PENDIENTE UNC | — | — | — |
| d4-a4r 60ep | cosine-tail | PENDIENTE UNC | — | — | — |

---

## Observaciones del corte

1. `d4a4` mejora marginalmente con cosine stretched y marca nuevo record (`83.8%`).
2. `a4r` no recupera su pico de 30ep bajo cosine stretched.
3. `t3-wt` mejora con hold trapezoidal (`+1.4pp`).
4. `D0` mejora con mas epocas, pero queda por debajo de los mejores brazos con descriptor.
5. `cosine-tail` queda como contraste principal pendiente para validar si preserva la dinamica util de 30ep y mejora la cola larga.

---

## Scripts asociados (SLURM)

- Cosine stretched:
  - `batch_60ep_d0.sh`
  - `batch_60ep_d4a4.sh`
  - `batch_60ep_a4r.sh`
  - `batch_60ep_d4-a4r.sh`
  - `batch_60ep_moe-dual.sh`
- Trapezoidal hold:
  - `gate44_t3-wt_scratch_50ep_hold.sh`
- Cosine-tail:
  - `batch_60ep_ctail_d0.sh`
  - `batch_60ep_ctail_d4a4.sh`
  - `batch_60ep_ctail_a4r.sh`
  - `batch_60ep_ctail_d4-a4r.sh`

Todos en `experiments/bias_control/slurm/`.

---

## Proxima conexion de roadmap

Gate 4.5 se mantiene abierto hasta completar el bloque pendiente
(cosine stretched restante + batch cosine-tail).

Con ese cierre, el roadmap habilita el pase operativo a:
- `10_GATE_5_LINEA_A_BARRIDO/`
- `11_GATE_5_LINEA_B_SHOWCASE/`

---

## Referencias

- `Documents/NOTAS_CLAUDE-CODEX.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/RANKING_DESCRIPTORES_UNIFICADO.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_UNC.md`
