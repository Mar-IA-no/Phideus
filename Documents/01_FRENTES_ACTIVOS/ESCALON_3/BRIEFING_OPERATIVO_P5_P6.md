# Briefing Operativo P5/P6

Fecha de consolidación: `2026-03-21`

## Propósito

Este documento es la versión corta y ejecutable de `PLAN_E3_P5_P6_GEOMETRIA_NO_PLANA.md`.

No redefine la metodología. La toma como fija y la traduce en una secuencia operativa limpia para implementación, smoke tests, runs completos y entrega de artefactos.

## Regla de trabajo

Siguiendo `Documents/00_TRONCAL/PROTOCOLO_OPERATIVO_CODEX_CLAUDE.md`:

- la semántica experimental ya está cerrada en el plan metodológico;
- este briefing existe para que la implementación y la ejecución no tengan que rediseñar `P5/P6` sobre la marcha;
- cualquier cambio que afecte comparabilidad debe quedar explícito en `config.json` y volver a auditoría.

## Lo que hay que implementar

Scripts nuevos:

- `experiments/escalon3/eval_torus_escalon3.py`
- `experiments/escalon3/f5_mixed_geometry.py`
- `experiments/escalon3/f6_tvicreg.py`

## Matriz de runs

| Run | Audio encoder | Image encoder | Geometría | Output sugerido |
|-----|---------------|---------------|-----------|-----------------|
| `P5-flatgeo` | `baseline` | `baseline` | mixta | `data/escalon3/p5_mixed_flat_seed42` |
| `P5-shiftgeo` | `cqtshift` | `baseline` | mixta | `data/escalon3/p5_mixed_cqtshift_seed42` |
| `P6-flatgeo` | `baseline` | `baseline` | toroidal completa | `data/escalon3/p6_tvicreg_flat_seed42` |
| `P6-shiftgeo` | `cqtshift` | `baseline` | toroidal completa | `data/escalon3/p6_tvicreg_cqtshift_seed42` |

## Invariantes

No tocar:

- dataset de `E3-P0`
- splits `iid / scale_ood / equiv_ood / render_ood`
- atlas OOD `train + val + test`
- encoders fuera de la familia del run
- métricas de retrieval de `P2`
- métricas de activation de `P4`

## Schedule por defecto

Primera pasada completa:

- `epochs = 50`
- `batch_size = 64`
- `lr_enc = 5e-4`
- `lr_proj = 1e-3`
- `warmup_steps = 200`
- `eval_epochs = [5, 10, 20, 30, 40, 50]`
- `seed = 42`

Si hay que mover `batch_size` o usar microbatching:

- dejarlo explícito en `config.json`
- mantenerlo constante dentro de la fase

## P5: mixed geometry

### Arquitectura

- encoders según la familia del run
- projector mixto:
  - rama euclídea `128d`
  - rama toroidal `64` dimensiones angulares derivadas de `128` outputs crudos organizados en pares

### Loss

- VICReg euclídeo:
  - `lambda_inv = 25`
  - `lambda_var = 25`
  - `lambda_cov = 1`
- T-VICReg toroidal:
  - `lambda_t_inv = 10`
  - `lambda_t_var = 10`
  - `lambda_t_cov = 1`

Si el primer smoke de `P5-flatgeo` deja la rama toroidal subactiva:

- mini-sweep corta `lambda_t_inv in {5, 10, 25}`
- mantener `lambda_t_var = 10`
- mantener `lambda_t_cov = 1`

### Retrieval obligatorio

Reportar:

- `euclidean-only`
- `torus-only`
- `mixed`

Score mixed:

- `d_euc_norm = (1 - cosine) / 2`
- `d_torus_norm = mean(wrap_dist^2) / pi^2`
- `d_mix = 0.5 * d_euc_norm + 0.5 * d_torus_norm`
- `score_mix = -d_mix`

### Checkpoint

Seleccionar `best_model.pt` por:

- `val_S_mixed`

Y loguear también:

- `val_S_euclidean`
- `val_S_torus`
- `val_S_mixed`

## P6: toroidal completo

### Arquitectura

- encoders según la familia del run
- projector toroidal completo:
  - salida cruda `256`
  - reshape `128 x 2`
  - pair-normalize
  - `128` dimensiones angulares finales

### Loss

- `lambda_t_inv = 10`
- `lambda_t_var = 10`
- `lambda_t_cov = 1`

Si el primer smoke de `P6-flatgeo` deja la rama toroidal subactiva:

- mini-sweep corta `lambda_t_inv in {5, 10, 25}`
- mantener `lambda_t_var = 10`
- mantener `lambda_t_cov = 1`

### Retrieval obligatorio

Reportar:

- `geodesic retrieval`
- `probe retrieval`

Baseline de lectura:

- `geodesic_nn`

### Checkpoint

Seleccionar `best_model.pt` por:

- `val_S_geodesic`

Los probes no entran en la selección de checkpoint.

## Evaluación compartida

Splits:

- `iid`
- `scale_ood`
- `equiv_ood`
- `render_ood`

Direcciones:

- `a2i`
- `i2a`

Retrieval:

- `scene_hit@10` en `iid`
- `ratio_hit@10` en `iid` y `scale_ood`
- `equiv_hit@10` en `equiv_ood`
- `S = min(a2i@10, i2a@10)` cuando corresponda

Activation:

- `activation_gain`
- `locking_selectivity`
- `coverage_uniformity`
- `relocking_depth`
- `basin_exposure`

Geométricas nuevas:

- `torus_silhouette_ratio`
- `torus_silhouette_equiv`
- `mean_geodesic_pos`
- `mean_geodesic_neg`
- `angular_uniformity`

## Comparaciones obligatorias

- `P5-flatgeo` contra `P2-flat` y `P4-flat`
- `P5-shiftgeo` contra `P2-cqtshift` y `P4-cqtshift`
- `P6-flatgeo` contra `P2-flat` y `P5-flatgeo`
- `P6-shiftgeo` contra `P2-cqtshift` y `P5-shiftgeo`

## Entregables mínimos por run

- `config.json`
- `history.json`
- `best_model.pt`
- `final_results.json`
- `torus_eval.json`
- `probe_results.json` si el régimen de probes se integra en la misma fase
- log monitoreable

Además, antes de dar un run por cerrado:

- param count
- tiempo por epoch
- memoria efectiva
- criterio de selección de checkpoint
- cobertura OOD

## Orden operativo

1. Implementar `eval_torus_escalon3.py`
2. Implementar `P5`
3. Smoke corto de `P5-flatgeo`
4. Run completo de `P5-flatgeo`
5. Run completo de `P5-shiftgeo`
6. Pasar diff + artefactos + lectura operativa a auditoría Codex
7. Implementar `P6`
8. Smoke corto de `P6-flatgeo`
9. Run completo de `P6-flatgeo`
10. Run completo de `P6-shiftgeo`
11. Pasar diff + artefactos + lectura operativa a auditoría Codex

## Regla de claims

No mezclar:

- encoder
- geometría
- probe

Lectura correcta:

- si la mejora aparece solo en `shiftgeo`, hablar de interacción `encoder × geometría`
- si aparece en `flatgeo` y `shiftgeo`, la lectura puede acercarse a ventaja de la geometría misma

## Cierre

Este briefing no reemplaza el plan metodológico largo. Lo acompaña.

La regla práctica es simple:

- usar este documento para implementar y correr;
- usar `PLAN_E3_P5_P6_GEOMETRIA_NO_PLANA.md` para cualquier duda de semántica experimental o criterio de lectura.
