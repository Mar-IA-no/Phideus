# Plan E3-P5 / E3-P6 - Geometría No Plana

Fecha de consolidación: `2026-03-21`

## Propósito

Este documento fija la especificación metodológica de `E3-P5` y `E3-P6` después de `P4`.

Nota posterior al primer ciclo completo:

- la ejecución inicial de `P5/P6` ya fue completada;
- la lectura consolidada vive ahora en `Resultados_E3_P5_P6.md`;
- este documento conserva su función original de especificación metodológica y trazabilidad del diseño.

La decisión vigente del frente es esta:

- `P4` ya fue corrido y queda documentado como resultado informativo sobre lectura en latente plano;
- ese resultado no alcanza para clausurar la hipótesis geométrica fuerte;
- por lo tanto, `P5` y `P6` se ejecutan **completos**, no como “rescate” ni como experimento accesorio.

## Regla de trabajo

Siguiendo `Documents/00_TRONCAL/PROTOCOLO_OPERATIVO_CODEX_CLAUDE.md`:

- **Codex** define protocolo, comparabilidad, métricas y criterios de lectura.
- **Claude** implementa scripts, tuning operativo, ejecución, monitoreo y runs largos.

Este documento está escrito como handoff metodológico para implementación y ejecución por Claude.

---

## 1. Invariantes globales

Nada de esto se toca en `P5/P6`:

- dataset canónico de `E3-P0`;
- splits IID / `scale_ood` / `equiv_ood`;
- atlas OOD `train + val + test` reducidos;
- métricas de retrieval de `P2`;
- métricas de activation introducidas en `P4`;
- lectura dual de baselines:
  - `P2-flat` = referencia canónica de retrieval general;
  - `P2-cqtshift` = referencia comparativa ratio-aware del lado audio.

Regla de comparabilidad:

- no cambiar a la vez dataset, encoder family y geometría sin dejar trazabilidad explícita;
- los efectos de geometría deben compararse siempre contra un baseline `L0` correspondiente.

---

## 2. Matriz de runs

La evaluación completa queda definida por cuatro runs principales:

| ID | Fase | Audio encoder | Image encoder | Geometría |
|----|------|---------------|---------------|-----------|
| `P5-flatgeo` | `P5` | `baseline` | `baseline` | mixta |
| `P5-shiftgeo` | `P5` | `cqtshift` | `baseline` | mixta |
| `P6-flatgeo` | `P6` | `baseline` | `baseline` | toroidal completa |
| `P6-shiftgeo` | `P6` | `cqtshift` | `baseline` | toroidal completa |

Nombres sugeridos de output:

- `data/escalon3/p5_mixed_flat_seed42`
- `data/escalon3/p5_mixed_cqtshift_seed42`
- `data/escalon3/p6_tvicreg_flat_seed42`
- `data/escalon3/p6_tvicreg_cqtshift_seed42`

Si hay presupuesto para replicación:

- correr primero `seed=42` para los cuatro runs;
- luego repetir solo los runs más informativos con `seed=43` y `seed=44`.

### Schedule por defecto

La primera pasada completa de `P5/P6` debe heredar el schedule de `P2` para preservar comparabilidad:

- `epochs = 50`
- `batch_size = 64`
- `lr_enc = 5e-4`
- `lr_proj = 1e-3`
- `warmup_steps = 200`
- `eval_epochs = [5, 10, 20, 30, 40, 50]`
- `seed = 42`

Si alguna familia obliga a ajustar `batch_size` o microbatching por memoria o latencia, ese cambio debe:

- quedar explícito en `config.json`;
- mantenerse constante dentro de la misma fase;
- y no arrastrar cambios oportunistas adicionales del schedule.

---

## 3. P5 - Mixed Geometry Latent

### Objetivo

Introducir una geometría no plana de forma controlada, manteniendo una parte del espacio en régimen euclídeo.

La pregunta de `P5` no es si el toro “gana” de forma mágica.  
La pregunta es:

> si una separación explícita entre subespacio periódico y subespacio residual permite una mejor organización o una mejor lectura sin degradación trivial.

### Arquitectura

Los encoders se mantienen según la familia del run:

- `flatgeo`: audio `baseline`, imagen `baseline`
- `shiftgeo`: audio `cqtshift`, imagen `baseline`

La diferencia entra en el **projector**, no en los encoders.

#### Projector mixto

Salida total comparable con `proj_dim = 256`, repartida así:

- **rama euclídea**: `128` dims
- **rama toroidal**: `64` dimensiones angulares, representadas como `128` outputs crudos organizados en pares `(x_i, y_i)`

Esquema:

```text
encoder -> shared MLP trunk -> {
  euclidean_head: 128d
  torus_head_raw: 128d -> reshape [64, 2] -> pair-normalize -> theta[64]
}
```

### Parametrización angular

Para cada par `(x_i, y_i)`:

1. normalizar a radio unitario:
   `u_i = (x_i, y_i) / ||(x_i, y_i)||`
2. obtener ángulo:
   `theta_i = atan2(u_i_y, u_i_x)`

Esto evita depender de radios arbitrarios y mantiene una representación angular limpia.

### Loss de `P5`

#### Rama euclídea

Usar VICReg estándar sobre `z_euc`:

- `lambda_inv = 25`
- `lambda_var = 25`
- `lambda_cov = 1`

#### Rama toroidal

Usar `T-VICReg` sobre `theta`.

Componentes:

1. **toroidal invariance**
   - distancia geodésica angular entre pares positivos
2. **circular variance**
   - evitar colapso angular
3. **circular covariance**
   - evitar que las dimensiones angulares se vuelvan redundantes

Hiperparámetros iniciales:

- `lambda_t_inv = 10`
- `lambda_t_var = 10`
- `lambda_t_cov = 1`

#### Loss total

Primera versión recomendada:

```text
L_total = L_vicreg_euclidean + L_tvicreg_torus
```

Sin coupling extra entre ramas en la primera implementación.  
La razón es de atribución: conviene ver primero si la mezcla ya ayuda sin agregar otra fuente de efecto.

### Nota sobre lambdas toroidales

Los lambdas toroidales deben tratarse como iniciales, no como constantes sagradas.

Regla operativa:

- en el primer smoke de `P5-flatgeo`, Claude debe inspeccionar la magnitud relativa de `L_inv_euc` y `L_t_inv`;
- si la rama toroidal queda claramente subponderada o casi inactiva, hacer una mini-sweep corta de `lambda_t_inv in {5, 10, 25}` antes de lanzar la grilla completa;
- `lambda_t_var = 10` y `lambda_t_cov = 1` se mantienen fijos en esa primera mini-sweep.

### Retrieval en `P5`

Reportar **tres lecturas**, no una sola:

1. **euclidean-only**
   - retrieval usando solo la rama euclídea
2. **torus-only**
   - retrieval usando solo distancia geodésica en la rama toroidal
3. **mixed**
   - score combinado

Score combinado sugerido:

```text
d_mix = 0.5 * d_euc_norm + 0.5 * d_torus_norm
score_mix = -d_mix
```

donde:

- `d_euc_norm = (1 - cosine(z_e_q, z_e_c)) / 2`
- `d_torus_norm = mean_i wrap_dist(theta_q_i, theta_c_i)^2 / pi^2`

Así ambas ramas quedan en `[0, 1]` antes de combinarse.

No tunear pesos antes de tener la primera lectura.

### Probe regime en `P5`

Para no mezclar lectura y entrenamiento:

- el régimen de probes debe actuar sobre la **rama toroidal**;
- la rama euclídea del query queda fija;
- el score del probe mixed se calcula como:

```text
score_probe_mix(c) =
  - [0.5 * d_euc_norm_fixed + 0.5 * min_{m,t} d_torus_norm(probe_t, theta_c)]
```

Familias de probe:

- `geodesic_nn` como baseline geométrico directo
- `rational_simple`
- `rational_complex`
- `phi`
- `noble`
- `irrational_random_1/2/3`

---

## 4. P6 - Full T-VICReg

### Objetivo

Testear la hipótesis geométrica fuerte sin dejar una rama euclídea residual.

La pregunta ya no es si una mezcla mejora algo, sino:

> si la organización periódica/armónica se representa mejor cuando el espacio latente completo se entrena en geometría toroidal.

### Arquitectura

Encoders iguales a la familia del run:

- `flatgeo`: audio `baseline`, imagen `baseline`
- `shiftgeo`: audio `cqtshift`, imagen `baseline`

Projector toroidal completo:

- salida cruda `256`
- reshape a `128 x 2`
- pair-normalize
- `128` dimensiones angulares finales

No hay rama euclídea residual en `P6`.

### Loss de `P6`

`T-VICReg` completo:

- toroidal invariance
- circular variance
- circular covariance

Hiperparámetros iniciales:

- `lambda_t_inv = 10`
- `lambda_t_var = 10`
- `lambda_t_cov = 1`

Regla operativa inicial:

- en el primer smoke de `P6-flatgeo`, Claude debe inspeccionar la magnitud relativa de la pérdida toroidal de invariancia y su efecto sobre el retrieval geodésico;
- si la rama toroidal queda claramente subactiva o muy por debajo de lo esperado, correr una mini-sweep corta de `lambda_t_inv in {5, 10, 25}` antes de lanzar la grilla completa de `P6`;
- `lambda_t_var = 10` y `lambda_t_cov = 1` se mantienen fijos durante esa mini-sweep.

### Retrieval en `P6`

Reportar dos lecturas principales:

1. **geodesic retrieval**
   - nearest neighbor por distancia geodésica media
2. **probe retrieval**
   - families racionales / `phi` / noble / irracionales sobre el toro

### Probe regime en `P6`

Acá sí el probe actúa sobre el espacio completo, porque el espacio completo ya es toroidal.

Probes:

- `rational_simple`
- `rational_complex`
- `phi`
- `noble`
- `irrational_random_1/2/3`

Baseline:

- `geodesic_nn`

No hace falta arrastrar `cosine` como métrica principal en `P6`.  
Si se reporta, que sea solo como control de coordenadas sobre representación `sin/cos`, no como lectura canónica.

### Selección de checkpoint

#### `P5`

El mejor checkpoint debe seleccionarse por:

- `val_S_mixed`

Y además deben loguearse en cada `eval_epoch`:

- `val_S_euclidean`
- `val_S_torus`
- `val_S_mixed`

#### `P6`

El mejor checkpoint debe seleccionarse por:

- `val_S_geodesic`

Los probes quedan como evaluación posterior y no entran en la selección de `best_model.pt`.

---

## 5. Evaluación compartida de `P5/P6`

### Splits obligatorios

- `iid`
- `scale_ood`
- `equiv_ood`
- `render_ood`

### Direcciones

- `a2i`
- `i2a`

### Métricas de retrieval

- `scene_hit@10` en `iid`
- `ratio_hit@10` en `iid` y `scale_ood`
- `equiv_hit@10` en `equiv_ood`
- `S = min(a2i@10, i2a@10)` cuando corresponda

### Métricas de activation

Las mismas de `P4`:

- `activation_gain`
- `locking_selectivity`
- `coverage_uniformity`
- `relocking_depth`
- `basin_exposure`

### Métricas geométricas nuevas

Agregar:

- `torus_silhouette_ratio`
- `torus_silhouette_equiv`
- `mean_geodesic_pos`
- `mean_geodesic_neg`
- `angular_uniformity`

Estas métricas no reemplazan las de retrieval; las complementan.

---

## 6. Comparaciones obligatorias

### Para `P5-flatgeo`

Comparar contra:

- `P2-flat`
- `P4-flat`

### Para `P5-shiftgeo`

Comparar contra:

- `P2-cqtshift`
- `P4-cqtshift`

### Para `P6-flatgeo`

Comparar contra:

- `P2-flat`
- `P5-flatgeo`

### Para `P6-shiftgeo`

Comparar contra:

- `P2-cqtshift`
- `P5-shiftgeo`

Esto evita claims sueltos tipo “mejoró respecto a algo”, sin decir respecto a qué familia o geometría.

---

## 7. Criterios de lectura

### `P5` se considera positivo si aparece al menos una mejora defendible en:

- estructura latente por ratio;
- retrieval mixed o torus-only sin colapso de `IID`;
- `activation_gain`;
- menor relocking;
- mejor `coverage_uniformity`.

Y además:

- training estable;
- sin degradación trivial del baseline correspondiente.

### `P6` se considera positivo si supera a `P5` o aporta una señal cualitativamente nueva en:

- estructura latente;
- `activation_gain`;
- relocking;
- coverage;
- o transferencia al tier dinámico.

No pedir solo un `R@10` más alto.

---

## 8. Regla de claims

No mezclar estos tres niveles:

1. **mejor encoder**
2. **mejor geometría**
3. **mejor probe**

Si una mejora aparece solo en `shiftgeo`, la lectura correcta es:

- interacción entre encoder ratio-aware y geometría no plana

Si aparece en `flatgeo` y `shiftgeo`, la lectura ya puede acercarse más a:

- ventaja de la geometría misma

---

## 9. Entregables que Claude debería dejar

### Scripts

- `experiments/escalon3/f5_mixed_geometry.py`
- `experiments/escalon3/f6_tvicreg.py`
- `experiments/escalon3/eval_torus_escalon3.py`

### Artefactos por run

- `history.json`
- `best_model.pt`
- `final_results.json`
- `torus_eval.json`
- `probe_results.json` si se integra el régimen de probes en la misma fase
- logs monitoreables en `tmux`

### Auditoría mínima esperada

Antes de considerar un run “cerrado”, Claude debería dejar:

- param count
- tiempo por epoch
- memoria efectiva
- criterio de selección de checkpoint
- cobertura OOD
- artefactos finales completos

---

## 10. Orden operativo recomendado

1. Implementar `eval_torus_escalon3.py`
2. Implementar `P5`
3. Smoke test corto de `P5-flatgeo`
4. Run completo de los dos brazos `P5`
5. Auditoría Codex
6. Implementar `P6`
7. Smoke test corto de `P6-flatgeo`
8. Run completo de los dos brazos `P6`
9. Auditoría Codex
10. Lectura conjunta `P4 + P5 + P6`

## 11. Decisión metodológica consolidada

La línea geométrica de Escalón 3 ya tiene una primera pasada completa y documentada.

`P4` ya no se interpreta como permiso o veto total sobre esa línea.  
Su papel queda fijado como comparación de métodos de lectura en `L0`.

Las preguntas más fuertes del frente pasaron efectivamente a `P5` y `P6`, donde la hipótesis de Harmonic Information Theory dejó de ponerse a prueba solo como lectura post-hoc y pasó a medirse sobre espacios entrenados con geometría explícitamente no plana. La lectura de ese primer ciclo ya no vive en este plan, sino en el documento de resultados correspondiente.
