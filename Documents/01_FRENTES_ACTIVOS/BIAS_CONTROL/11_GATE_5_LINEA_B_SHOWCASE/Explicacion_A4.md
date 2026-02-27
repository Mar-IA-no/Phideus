# A4 — Qué es realmente (corte 2026-02-25)

## Definición operativa

A4 no calcula ratios entre picos espectrales. A4 calcula **dinámica temporal de energía espectral** por bandas de octava.

Pipeline conceptual:

1. Audio (4 s a 24 kHz)
2. STFT (`n_fft=2048`, `hop=512`)
3. Agrupación en 8 bandas log-spaced
4. Energía por banda a lo largo del tiempo
5. Delta temporal por banda (`t - (t-1)`)
6. Normalización

Resultado: 8 dimensiones por frame.

## Dimensiones A4 (nombres canónicos)

- `band0_47Hz`
- `band1_94Hz`
- `band2_188Hz`
- `band3_375Hz`
- `band4_750Hz`
- `band5_1500Hz`
- `band6_3000Hz`
- `band7_6000Hz`

Estos nombres son los que usa el JSON de Test08 (`descriptor_dims.audio.features`).

## Qué significa científicamente

A4 captura "cómo cambia" la energía por regiones de frecuencia, no el valor espectral estático.

Eso permite:
- sensibilidad a estructura temporal-musical,
- mayor tolerancia a cambios globales de ganancia,
- una señal informativa que en Gate 5B mostró causalidad fuerte (Test01).

## Relación con Test01/Test08

- Test01: cuando se sabotea A4 (`zero/noise/shuffle_audio`), `S` cae fuerte en modelos top.
- Test08: las bandas de mayor frecuencia (en especial desde ~`750 Hz`) aparecen con mayor sensibilidad en los modelos augmentados.

## Fuente técnica

Implementación de referencia:
- `src/bias_control/audio_descriptors.py` (`compute_audio_descriptor_a4`)
