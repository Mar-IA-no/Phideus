# Gate 8 -- Descriptor-Conditioned Projection Heads

**Estado**: ACTIVO, con `a4r-ctrl` y `a4r-pcm` ya cerrados en local; `a4r-pcd-zero` y `a4r-pcd` ya cerrados en UNC; `a4r-pca` sigue abierto en el último corte sincronizado de `unc`
**Origen**: Promocion operativa de Gate 5A C1. Trazabilidad: Gate 5A/C1 sigue documentado en `10_GATE_5_LINEA_A_BARRIDO/README.md`.

## Proposito

Gate 8 ataca el cuello de botella diagnosticado en las projection heads.

**Evidencia convergente**:
- Test 11 Pre-Proj: MIDI projection 512->256 destruye ~88% de info condicionante
- Gate 7.1a: un encoder de audio 5x mas fuerte (MERT-330M frozen) no mejora retrieval (S=75.0% vs D0_lite 75.2%)
- Ambas pistas apuntan al lado MIDI/projection como limite operativo

**Precision**: esto no prueba "el cuello son las projections". Es la intervencion downstream mas barata y mas alineada con la evidencia actual.

## Mecanismo

Reemplazar `ProjectionHead` (MLP 3 capas) por `ConditionedProjectionHead` (misma estructura + FiLM modulation):

```
h' = (1 + gamma) * h + beta
```

- gamma/beta generados por MLP pequeno: `cond_dim -> 64 -> 2*hidden_dim`
- Zero-init en ultima capa -> identidad al inicio, crece desde ahi
- NO toca encoders. Solo modifica projection heads.
- Overhead: ~265K params extra (~0.3% del total)

### Conditioning signals

| Projection | Signal | Dim | Origen |
|------------|--------|-----|--------|
| Audio | `compute_audio_band_energy()` | 8 | Mean log-magnitude per A4 band (envolvente espectral) |
| MIDI | `compute_local_interval_features()` mean | 4 | D4 local intervals (prev/next semitone + log ratio) |

**Nota**: el audio conditioning usa band energy (no-degenerado), NO el A4 z-scored (que tiene mean=0 por construccion y daria conditioning ~vacio).

## Brazos

| Arm | Audio proj | MIDI proj | Condicion | Pregunta |
|-----|-----------|-----------|-----------|----------|
| a4r-ctrl | standard | standard | -- | Reproducibilidad del baseline |
| a4r-pcm | standard | **conditioned** | D4->midi | Cuello en MIDI proj? (hipotesis mas fuerte) |
| a4r-pcd-zero | conditioned | conditioned | zeros fijos | Control de overhead parametrico |
| a4r-pcd | conditioned | conditioned | band_energy + D4 | Brazo principal |
| a4r-pca | **conditioned** | standard | band_energy->audio | Cuello en audio proj? |

## Orden de ejecucion

1. `a4r-ctrl` (30ep) -- cerrado localmente como baseline de referencia
2. `a4r-pcm` (30ep) -- cerrado localmente; primera hipótesis sobre cuello MIDI-side
3. `a4r-pcd-zero` (30ep) -- control overhead, migrado a UNC
4. `a4r-pcd` (30ep) -- ambos condicionados, migrado a UNC
5. `a4r-pca` (30ep) -- audio only, migrado a UNC

## Resultados locales ya cerrados

| Brazo | Best S | Best epoch | hard_neg | Lectura mínima |
|-------|--------|------------|----------|----------------|
| `a4r-ctrl` | `79.2%` | `ep30` | `94.2%` | baseline reproducido para la familia conditioned projections |
| `a4r-pcm` | `80.0%` | `ep29` | `95.2%` | mejora marginal de `+0.8pp` al condicionar solo la proyección MIDI |

Lectura prudente:
- observación: `pcm` sí supera a `ctrl`, pero por margen chico y en una sola seed;
- hipótesis compatible: la projection MIDI puede ser parte del cuello, pero no parece un desbloqueo masivo;
- inferencia válida hoy: Gate 8 sigue siendo línea oportunista útil, no nueva ruta crítica del programa.

## Resultados UNC ya sincronizados

| Brazo | Best S | Best epoch | hard_neg | Lectura mínima |
|-------|--------|------------|----------|----------------|
| `a4r-pcd-zero` | `81.8%` | `ep30` | `94.6%` | la arquitectura conditioned projection agrega expresividad aun con conditioning nulo |
| `a4r-pcd` | `84.2%` | `ep25` | `94.8%` | el conditioning dual real (`A4 + D4`) supera a `ctrl` y a `pcd-zero` |

Lectura prudente del corte UNC:
- observación: `pcd` supera a `ctrl` por `+5.0pp` y a `pcd-zero` por `+2.4pp`;
- hipótesis compatible: el conditioning dual sí preserva o reorganiza información útil en las projection heads;
- inferencia válida hoy: Gate 8 sigue sin volverse ruta crítica del programa, pero deja una de las señales positivas más claras de la línea `Gate 5A/C1`.

## Estado UNC

- `a4r-pcd-zero`, `a4r-pcd` y `a4r-pca` salieron de la GPU local y quedaron en el array `1144707` tras el resubmit con memoria ampliada.
- `pcd-zero` y `pcd` ya cerraron correctamente en UNC.
- `pca` sigue abierto en el último corte sincronizado.
- La práctica de `resume` ya quedó integrada en el script experimental para soportar requeue/autoresubmit.

## Lecturas esperadas

- `a4r-pcm > a4r-ctrl` -> confirmaria cuello en MIDI projection
- `a4r-pcd > a4r-pcd-zero` -> mejora causal del conditioning, no overhead
- `a4r-pcm > a4r-pca` -> cuello es MIDI-side (consistente con Test 11 + Gate 7.1a)

## Archivos

```
experiments/bias_control/gate5a_proj_cond.py  # Script principal (nombre preservado por trazabilidad)
src/bias_control/encoders/projection.py       # ConditionedProjectionHead
src/bias_control/audio_descriptors.py         # compute_audio_band_energy (NUEVO)
```

## Sanity checks

El script loguea stats de conditioning vectors (mean, std, min, max) en el primer batch.
Verificar que:
- cond_audio: std >> 0 (si ~0, el conditioning es degenerado)
- cond_midi: std >> 0

## Comando

```bash
python experiments/bias_control/gate5a_proj_cond.py \
    --arm a4r-pcm \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/gate8_results/a4r-pcm \
    --epochs 30 --batch-size 16 --seed 42
```

## Conexion con otros Gates

| Gate | Conexion |
|------|----------|
| Gate 5A | Gate 8 = promocion de 5A/C1. Codigo y trazabilidad preservados. |
| Gate 5B Test 11 | Diagnostico original: MIDI proj destruye 88% info. |
| Gate 7.1a | Refuerzo: más encoder no ayuda por sí solo, lo que vuelve más plausible el lado projection/MIDI. |
| Gate 6 | Independiente (AMT downstream). |
