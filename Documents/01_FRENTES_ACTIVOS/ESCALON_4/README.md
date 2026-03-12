<div align="center">

# Escalón 4
### Audio XY ↔ Figuras de Lissajous

![Status](https://img.shields.io/badge/Status-Conceptual_Front-0A7E3B?style=for-the-badge)
![Focus](https://img.shields.io/badge/Focus-Lissajous-1F6FEB?style=for-the-badge)
![Updated](https://img.shields.io/badge/Updated-2026--03--12-F59E0B?style=for-the-badge)

</div>

> [!IMPORTANT]
> **Estado actual**: este frente existe como linea conceptual nueva y su nombre correcto es **`ESCALON_4`**. No reemplaza al Escalón 3 fisiologico de la triplescaloneta; lo expande. Todavia no hay codigo ni dataset versionado, pero ya hay una logica experimental suficientemente definida para tratarlo como frente real del programa.
> **Hipótesis de trabajo**: si una relacion armónica puede hacerse visible como figura de Lissajous y audible como señal XY estéreo, entonces Phideus gana un banco sintético donde el ratio deja de ser solo descriptor implícito y pasa a ser objeto visible, generable y medible con ground truth determinista.

## Qué es este frente

Escalón 4 propone un banco de pruebas nuevo para Harmonic Information Theory: pares `audio XY ↔ figura de Lissajous` donde la relacion entre ambas modalidades no sea ambigua ni inferida post-hoc, sino **determinista por construccion**.

La idea central es simple:

- el audio canonico no es mono acustico, sino señal estéreo `XY`;
- la figura canonica no es una imagen arbitraria, sino la trayectoria `X(t), Y(t)`;
- el ratio `p:q` pasa a organizar simultaneamente la topologia visual y la estructura temporal de la señal.

Eso hace de Escalón 4 un banco especialmente limpio para explorar:

1. retrieval audio ↔ imagen;
2. recuperacion de parametros generativos (`p:q`, fase, amplitud);
3. descriptores de ratios visibles y geometricos;
4. generalizacion OOD por familias de ratios;
5. transferencia sintético → captura física.

## Por qué importa para Phideus

Escalón 1 validó la mecanica descriptor-guided en musica. Escalón 2 abrió la prueba fuerte de armonia natural en voz. Escalón 4 agrega otra cosa: un dominio donde la relacion armónica no solo puede oirse sino tambien **verse** y **generarse** con control total.

Eso ofrece ventajas metodologicas concretas:

- ground truth exacto sobre los parametros latentes;
- ausencia de ambigüedad entre modalidad y etiqueta;
- posibilidad de partir por un benchmark sintético serio antes de pasar a hardware o captura real;
- espacio natural para estudiar descriptores geometricos y de ratio bajo splits IID y OOD.

## Decisión conceptual ya fijada

La modalidad canonica del frente es:

- `audio canónico = señal XY estéreo`
- `figura canónica = trazado X contra Y`
- `audio perceptual/mic = modalidad secundaria`

Eso evita un error de diseño importante: una figura de Lissajous no sale de un audio mono cualquiera, sino de la relacion entre dos ejes.

## Estado del arte operativo

No existe hoy un dataset público canónico y adoptado de `audio ↔ figuras de Lissajous`. La recomendacion convergente de la investigacion hecha en el repo es:

- arrancar con un dataset **synthetic-first**;
- guardar los parametros latentes exactos de cada scene;
- separar desde el diseño ratio, fase, amplitud, render y frecuencia base;
- dejar la capa física para una fase posterior.

Documentos base ya disponibles:

- `Plan_Claude.md` - relevamiento inicial, herramientas y analogias de dominio
- `Plan_inaugural_construccion_dataset_Codex.md` - diseño sintético-first, taxonomia de tiers y propuesta de metadatos

## Unidad canonica del dataset

La unidad recomendada no es solo "wav + png", sino una **scene** con sus factores latentes:

- `xy_audio.wav`
- `xy_trace.npy`
- `figure_clean.png`
- `figure_style_*.png`
- `meta.json`

Metadatos minimos:

- `p`, `q`, `ratio_reduced`
- `fx`, `fy`, `base_frequency`
- `phase_rad`
- `amp_x`, `amp_y`, `amp_ratio`
- `closure_period`, `duration`, `sample_rate`
- `render_style`, `line_width`, `blur`, `noise_snr`
- `scene_id`, `ratio_id`, `equiv_id`, `split`

## Tareas prioritarias

Orden recomendado del frente:

1. **parameter recovery** - imagen o audio → `p:q`, fase y amplitud
2. **cross-modal retrieval** - audio ↔ figura con dual encoders
3. **structure-aware retrieval** - misma scene vs mismo `ratio_id` / `equiv_id`
4. **conditional generation** - parametros o figura objetivo → audio XY controlado

## Próximo paso único

El siguiente paso correcto no es correr un modelo grande, sino fijar una **v0.1 del dataset**:

1. taxonomia de ratios;
2. estructura de carpetas;
3. manifiesto de metadatos;
4. splits IID / ratio-OOD / scale-OOD / render-OOD;
5. primer generador sintético reproducible.

## Relación con el resto del programa

- Escalón 1 sigue siendo el cierre fuerte de la mecanica descriptor-guided en musica.
- Escalón 2 sigue siendo el foco principal para la tesis fuerte de armonia natural.
- Escalón 3 permanece reservado al frente fisiologico ECG ↔ PPG.
- Escalón 4 abre una arena nueva: ratios visibles, control total y benchmark sintético con ground truth exacto.
