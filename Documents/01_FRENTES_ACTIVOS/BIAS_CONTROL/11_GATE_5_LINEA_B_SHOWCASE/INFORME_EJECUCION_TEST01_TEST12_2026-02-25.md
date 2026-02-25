# Informe de Ejecucion Gate 5B
## Test12 Scoreboard + Test01 Causal Ablation

Fecha de cierre de corrida: 2026-02-24/2026-02-25

## 1) Alcance del informe

Este informe consolida la evidencia operativa de Gate 5B para:
- validacion del loader universal de checkpoints;
- cierre de Test12 (scoreboard canonico);
- cierre de Test01 (causal ablation) en los 4 arms;
- lectura metodologica de la duda central D4 vs A4/A4r.

## 2) Artefactos fuente (evidencia primaria)

- `data/gate5b_results/scoreboard.json`
- `data/gate5b_results/D0/test12_scoreboard.json`
- `data/gate5b_results/d4a4/test12_scoreboard.json`
- `data/gate5b_results/a4r/test12_scoreboard.json`
- `data/gate5b_results/d4-a4r/test12_scoreboard.json`
- `data/gate5b_results/D0/test01_causal_ablation.json`
- `data/gate5b_results/d4a4/test01_causal_ablation.json`
- `data/gate5b_results/a4r/test01_causal_ablation.json`
- `data/gate5b_results/d4-a4r/test01_causal_ablation.json`
- `/tmp/gate5b_test01_v2.log`

## 3) Test12 Scoreboard (config canonica)

Configuracion fija usada:

```python
CANONICAL_EVAL_CONFIG = {
    'pool_size': 256,
    'n_queries': 500,
    'n_hard_negatives': 64,
    'n_semi_hard_negatives': 32,
    'seed': 42,
}
```

Metrica principal:

- `S = min(a2m.mean_recall@10, m2a.mean_recall@10)`

Resultados observados:

| Arm | S | A2M R@10 | M2A R@10 | Validacion |
|---|---:|---:|---:|---|
| D0 | 73.4% | 74.8% | 73.4% | PASS |
| d4a4 | 83.8% | 84.4% | 83.8% | PASS |
| a4r | 82.0% | 82.6% | 82.0% | PASS |
| d4-a4r | 79.8% | 81.4% | 79.8% | PASS |

Conclusión:
- El loader reconstruye correctamente wrappers y pesos de los 4 checkpoints Gate 5B.

## 4) Test01 Causal Ablation: que demuestra cada modo

Modos de intervención:

- `zero_*`: reemplaza descriptor por ceros.
  - Mide dependencia causal "bruta" de la señal.
- `noise_*`: reemplaza descriptor por ruido gaussiano con igual `mean/std` del descriptor real.
  - Mide si importa contenido semántico vs estadística global.
- `shuffle_*`: permuta descriptor entre muestras del batch.
  - Mide dependencia de la correspondencia muestra-descriptor.

Interpretación estándar:

- Si `delta` grande en `zero`: el modelo usa esa señal.
- Si `noise` cae menos que `zero`: la estadística global preserva parte de la utilidad.
- Si `shuffle` cae: el modelo usa correspondencia específica por muestra.

## 5) Test01 resultados completos

Metrica:
- `S = min(A2M R@10, M2A R@10)`
- `delta = S_normal - S_ablated`

### 5.1 D0 (control negativo)

| Arm | S_normal | Ablaciones |
|---|---:|---|
| D0 | 73.4% | No aplica (sin descriptores) |

### 5.2 d4a4

`S_normal = 83.8%`

| Modo | S | delta |
|---|---:|---:|
| zero_audio | 7.8% | +76.0 pp |
| zero_midi | 84.4% | -0.6 pp |
| zero_both | 7.6% | +76.2 pp |
| noise_audio | 22.0% | +61.8 pp |
| noise_midi | 84.4% | -0.6 pp |
| noise_both | 19.6% | +64.2 pp |
| shuffle_audio | 46.6% | +37.2 pp |
| shuffle_midi | 83.8% | +0.0 pp |
| shuffle_both | 48.4% | +35.4 pp |

### 5.3 a4r

`S_normal = 82.0%`

| Modo | S | delta |
|---|---:|---:|
| zero_audio | 4.4% | +77.6 pp |
| noise_audio | 29.0% | +53.0 pp |
| shuffle_audio | 49.8% | +32.2 pp |

### 5.4 d4-a4r

`S_normal = 79.8%`

| Modo | S | delta |
|---|---:|---:|
| zero_audio | 4.4% | +75.4 pp |
| zero_midi | 79.4% | +0.4 pp |
| zero_both | 4.4% | +75.4 pp |
| noise_audio | 26.8% | +53.0 pp |
| noise_midi | 79.8% | +0.0 pp |
| noise_both | 25.6% | +54.2 pp |
| shuffle_audio | 47.4% | +32.4 pp |
| shuffle_midi | 79.8% | +0.0 pp |
| shuffle_both | 47.6% | +32.2 pp |

## 6) Lectura metodologica (Observacion / Hipotesis / Inferencia)

### Observaciones

1. En `d4a4` y `d4-a4r`, ablacionar audio descriptor derrumba fuertemente `S`.
2. En `d4a4` y `d4-a4r`, ablacionar D4 (`zero/noise/shuffle_midi`) tiene impacto casi nulo.
3. Historicamente, `d4` solo mejora sobre `d0`, pero marginal:
   - Gate42 screening: `60.4% -> 64.2%` (+3.8 pp)
   - Gate43 screening: `60.2% -> 63.6%` (+3.4 pp)

### Hipotesis

- D4 contiene señal útil, pero de baja magnitud; en modelos duales con A4/A4r su aporte queda mayormente redundante o dominado por la rama de audio.

### Inferencia operativa (con evidencia actual)

- Para los checkpoints top Gate 5B, el driver causal dominante en inferencia es la rama de audio descriptor.
- D4 no queda invalidado en general: su aporte parece dependiente de arquitectura/régimen y marginal en esta familia dual.

## 7) Incidente técnico y fix aplicado durante corrida

Incidente:
- `collect_descriptor_stats` fallaba por concatenación de tensores D4 con longitud temporal variable entre batches (`[B, N, 4]` con `N` variable).

Fix:
- flatten per-batch antes de concatenar:

```python
flat_midi = torch.cat([v.reshape(-1, v.size(-1)) for v in midi_vals], dim=0)
```

Impacto:
- Test01 completado para los 4 arms sin bloqueo adicional.

## 8) Estado de cierre y siguiente paso

Cerrado:
- Loader universal.
- Fix `evaluate_structured_pool`.
- Harness Gate 5B.
- Test12 Scoreboard.
- Test01 Causal Ablation.

Siguiente paso recomendado:
1. Cerrar Test04 (transposition invariance) usando cache de embeddings.
2. Consolidar tabla de invariancia por arm.
3. Continuar con Test06/08/09.

