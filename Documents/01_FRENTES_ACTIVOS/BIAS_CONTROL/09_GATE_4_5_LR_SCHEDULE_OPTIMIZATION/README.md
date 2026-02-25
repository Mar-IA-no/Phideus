# Gate 4.5 — LR Schedule Optimization

**Estado**: CIERRE OPERATIVO (bloque de soporte para Gate 5B)
**Fecha de corte**: 2026-02-25
**Origen**: reestructuración post Gate 4.4 para separar optimización de scheduler/ventana temporal de cambios de arquitectura.

---

## Pregunta central

Con arquitectura y descriptores fijos (definidos en Gates 4.3/4.4),
qué combinación de scheduler y duración (50-60 épocas)
maximiza `S=min(A2M, M2A)` sin perder comparabilidad.

Variable independiente:
- scheduler de LR
- duración total del entrenamiento

Variables fijas:
- foundation/checkpoint, seed, batch size, freeze policy, protocolo de evaluación.

---

## Schedulers bajo prueba

| Scheduler | Descripción | Flags |
|-----------|-------------|-------|
| cosine stretched | Cosine estándar estirado a 60ep | default (`--epochs 60`) |
| trapezoidal hold | hold al pico de LR y luego cosine decay | `--lr-hold-fraction 0.5` |
| cosine-tail | replica curva 30ep hasta LR floor y cola lineal | `--lr-cosine-ref-epochs 30 --lr-floor 0.10 --lr-tail-end 0.02` |

---

## Tabla de runs Gate 4.5

### Bloque cosine stretched / hold

| Run | Scheduler | Estado | Best S | Best ep | Delta vs 30ep |
|-----|-----------|--------|--------|---------|---------------|
| d4a4 60ep | cosine stretched | COMPLETO | 83.8% | e50 | +0.2pp (record global) |
| a4r 60ep | cosine stretched | COMPLETO | 79.4% | e60 | -2.6pp |
| D0 60ep | cosine stretched | COMPLETO | 72.8% | e50 | +12.6pp |
| t3-wt 50ep | trapezoidal hold | COMPLETO | 81.2% | e50 | +1.4pp |
| d4-a4r 60ep | cosine stretched | COMPLETO | 79.8% | e55 | ±0.0pp |
| moe-dual 60ep | cosine stretched | DEAD (time limit) | 73.0% | e30 | +0.4pp (no sostenido) |

### Bloque cosine-tail

| Run | Scheduler | Estado | Best S | Best ep | Delta vs 30ep |
|-----|-----------|--------|--------|---------|---------------|
| a4r 60ep | cosine-tail | COMPLETO | 80.6% | e60 | -1.4pp |
| d4a4 60ep | cosine-tail | CIERRE OPERATIVO (referencia interna) | 83.4% | e30 | -0.4pp (provisorio) |
| D0 60ep | cosine-tail | CIERRE OPERATIVO (usado en Gate 5B) | 73.4% | e50 | +13.2pp vs D0@30ep |
| d4-a4r 60ep | cosine-tail | FUERA DE RUTA CRÍTICA | — | — | — |

---

## Observaciones del corte

1. `d4a4` es el único brazo que mejoró su techo absoluto en stretched (`83.8%`).
2. `a4r` no recupera su nivel de 30ep en ningún schedule extendido (79.4 stretched, 80.6 ctail vs 82.0 en 30ep).
3. `t3-wt` sí mejora con hold trapezoidal (`+1.4pp`).
4. `d4-a4r` stretched confirma empate con 30ep (`79.8%`), sin ganancia neta.
5. `moe-dual` mostró peak temprano no sostenido y quedó muerto por time limit.
6. `cosine-tail` mejora el control D0 frente a stretched (best operativo 73.4% vs 72.8%).

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

## Próxima conexión de roadmap

Gate 4.5 queda como bloque de soporte metodológico y de trazabilidad histórica.
El frente activo del programa se ubica en:
- `11_GATE_5_LINEA_B_SHOWCASE/` (paquete local consolidado y cierre final en curso).
- `10_GATE_5_LINEA_A_BARRIDO/` (pendiente de ejecución plena).

---

## Referencias

- `Documents/NOTAS_CLAUDE-CODEX.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/RANKING_DESCRIPTORES_UNIFICADO.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_UNC.md`
