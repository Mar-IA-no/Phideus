# ROADMAP - Escalon 4: Audio XY ↔ Figuras de Lissajous

> Fecha de creacion: 2026-03-12
> Estado: CONCEPTUAL, con diseño experimental base ya definido

## 1. Pregunta del frente

Escalón 4 pregunta si Phideus puede aprender, recuperar y generar relaciones armónicas cuando esas relaciones estan codificadas a la vez:

- como señal estéreo `XY`,
- como trayectoria geometrica visible,
- y como parametros latentes exactos (`ratio`, fase, amplitud).

No es un frente de "imagenes bonitas". Es un banco de pruebas donde la relacion armónica puede estudiarse sin la ambigüedad propia de datasets acusticos naturales.

## 2. Objeto canonico

La unidad de trabajo del escalón es una **scene**.

Cada scene deberia incluir:

- `xy_audio.wav`
- `xy_trace.npy`
- `figure_clean.png`
- `figure_style_*.png`
- `meta.json`

Campos minimos de `meta.json`:

- `p`, `q`, `ratio_reduced`
- `ratio_float`, `equiv_id`, `ratio_id`
- `base_frequency`, `fx`, `fy`
- `phase_rad`
- `amp_x`, `amp_y`, `amp_ratio`
- `duration`, `sample_rate`, `closure_period`
- `render_style`, `noise_snr`, `line_width`, `blur`
- `scene_id`, `split`

## 3. Fases del frente

### F0 - Generador canónico

Objetivo:
- producir escenas sintéticas reproducibles con senoidales puras;
- fijar ratios racionales reducidos como base;
- guardar audio, traza, imagen y metadatos.

Criterio de cierre:
- generador determinista;
- metadatos completos;
- renders consistentes;
- primer set pequeño de sanity check.

### F1 - Clasificación / parameter recovery

Objetivo:
- predecir `p:q`, fase y `amp_ratio` desde audio o imagen.

Criterio de cierre:
- baseline trivial funcionando;
- accuracy alta en ratios train;
- error interpretable en fase / amplitud.

### F2 - Retrieval cross-modal

Objetivo:
- alinear `audio XY ↔ figura` en un embedding compartido.

Criterio de cierre:
- benchmark con dual encoders;
- split IID y `ratio-OOD`;
- lectura separada entre scene-level y structure-level retrieval.

### F3 - Ablación descriptorial

Objetivo:
- comparar familias de descriptores:
  - `R1 natural`: `p:q`, complejidad armónica, distancia a vecinos
  - `R2 perceptual-control`: log2, cents, buckets temperados
  - `R3 geométrico`: lóbulos, cruces, area, simetrías
  - `R4 dinámico`: drift, cierre, phase velocity

Criterio de cierre:
- tabla descriptor × tarea;
- lectura explícita de qué es descriptor, qué es control y qué es mecanismo.

### F4 - Capa física / captura real

Objetivo:
- reproducir escenas sintéticas en setup XY físico y capturarlas por cámara u osciloscopio.

Criterio de cierre:
- primer test synthetic → real;
- estabilidad temporal suficiente;
- documentación del gap dominio sintético / físico.

## 4. Diseño de splits

No alcanza con `train/val/test` random.

Splits mínimos recomendados:

1. `IID` - escenas nuevas con ratios vistos
2. `ratio-OOD` - ratios reducidos no vistos en train
3. `scale-OOD` - frecuencias base no vistas
4. `render-OOD` - estilos visuales o ruido no vistos

La prueba fuerte del escalón no es memorizar escenas, sino generalizar estructura de ratio.

## 5. Primer dataset viable

Piloto recomendado:

- `16` ratios de train
- `4` ratios OOD
- `6` frecuencias base
- `8` fases
- `4` amp ratios
- `2` repeticiones
- `3` renders por scene

Eso deja un dataset suficientemente serio para baselines sin volver el frente inmanejable desde el inicio.

## 6. Criterios GO / NO-GO

### GO

- el generador produce escenas estables y correctamente etiquetadas;
- retrieval y parameter recovery muestran señal clara;
- `ratio-OOD` deja una lectura interpretable;
- los descriptores geométricos agregan algo que no sea puro artefacto de render.

### NO-GO / pausa

- si el dataset no separa bien ratio de fase / estilo / escala;
- si el benchmark queda trivial por leakage entre splits;
- si el frente deriva demasiado rápido a estilización visual sin banco científico limpio.

## 7. Dependencias conceptuales ya fijadas

Fuentes base del frente, ya relevadas en el repo:

- `Plan_Claude.md`
- `Plan_inaugural_construccion_dataset_Codex.md`
- `Documents/90_ARCHIVO_GLOBAL/Legacy/Rosetta/PROPUESTA_ROSETA_2_AUDIO_CINEMATICA.md`

## 8. Relación con la triplescaloneta

- Escalón 1: validación descriptor-guided en música
- Escalón 2: prueba fuerte de armonía natural en voz
- Escalón 3: expansión fisiológica ECG ↔ PPG
- Escalón 4: banco sintético de ratios visibles y generables

Escalón 4 no reemplaza a los anteriores. Su valor es distinto: ofrece un dominio donde la relación armónica puede estudiarse con control total de los factores latentes.
