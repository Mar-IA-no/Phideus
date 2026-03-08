# Gate 8 -- Descriptor-Conditioned Projection Heads

**Estado**: IMPLEMENTADO, listo para correr (2026-03-06)
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

1. `a4r-ctrl` (30ep) -- verificar reproducibilidad baseline a4r
2. `a4r-pcm` (30ep) -- **hipotesis mas fuerte**: MIDI projection es el cuello
3. `a4r-pcd-zero` (30ep) -- control overhead
4. `a4r-pcd` (30ep) -- ambos condicionados
5. `a4r-pca` (30ep) -- audio only

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
| Gate 7.1a | Refuerzo: mas encoder no ayuda, cuello en projection/MIDI. |
| Gate 6 | Independiente (AMT downstream). |
