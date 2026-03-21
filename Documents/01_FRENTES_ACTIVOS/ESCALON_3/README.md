<div align="center">

# Escalón 3
### Audio XY ↔ Figuras de Lissajous

![Status](https://img.shields.io/badge/Status-E3--P0_Materialized-0A7E3B?style=for-the-badge)
![Focus](https://img.shields.io/badge/Focus-Lissajous-1F6FEB?style=for-the-badge)
![Updated](https://img.shields.io/badge/Updated-2026--03--21-F59E0B?style=for-the-badge)

</div>

> [!IMPORTANT]
> **Estado actual**: este frente ya no es solo línea conceptual. `E3-P0` ya dejó un generador reproducible en `experiments/escalon3/generate_lissajous_dataset.py` y un banco canónico de scenes en `data/escalon3/scenes/`. Lo que todavía no está cerrado como capa canónica son `E3-P1` y `E3-P2`.
> **Corte operativo**: el banco actual tiene `6,016` scenes (`train/val/test=2144/448/480`, `ratio_ood=768`, `scale_ood=1024`, `equiv_ood=1152`), con `train/val/test` puro sobre ratios reducidos y no reducidos reservados para `equivalence-OOD`.
> **Hipótesis de trabajo**: si una relacion armónica puede hacerse visible como figura de Lissajous y audible como señal XY estéreo, entonces Phideus gana un banco sintético donde el ratio deja de ser solo descriptor implícito y pasa a ser objeto visible, generable, medible y también **activable** mediante probes racionales o no-locking.

## Estado operativo del frente

Escalón 3 ya tiene por fin un objeto experimental real. El frente no vive solo en `README`, `roadmap` o planes: hoy ya existe un generador canónico y un primer dataset materializado.

Lo que ya quedó fijado:

- `experiments/escalon3/generate_lissajous_dataset.py` genera scenes deterministas con audio XY, traza canónica, render limpio y variantes de estilo.
- `data/escalon3/scenes/` ya contiene el banco `E3-P0`.
- el banco ya separa `IID`, `ratio-OOD`, `scale-OOD` y `equivalence-OOD`.
- `train/val/test` quedaron sobre ratios reducidos solamente, para que la figura canónica no quede ambigua durante `P1/P2`.
- los no reducidos (`6:4`, `4:2`, `6:3`, etc.) quedaron fuera de training y sirven para testear equivalencias racionales.

Lo que todavía no debe sobreafirmarse:

- no hay todavía `P1` cerrado de parameter recovery;
- no hay todavía `P2` cerrado de retrieval cross-modal;
- `phi` sigue fuera de training y queda reservado para `E3-P4`.

## Qué es este frente

Escalón 3 propone un banco de pruebas nuevo para Harmonic Information Theory: pares `audio XY ↔ figura de Lissajous` donde la relacion entre ambas modalidades no sea ambigua ni inferida post-hoc, sino **determinista por construccion**.

La idea central es simple:

- el audio canonico no es mono acustico, sino señal estéreo `XY`;
- la figura canonica no es una imagen arbitraria, sino la trayectoria `X(t), Y(t)`;
- el ratio `p:q` pasa a organizar simultaneamente la topologia visual y la estructura temporal de la señal.

Eso hace de Escalón 3 un banco especialmente limpio para explorar:

1. retrieval audio ↔ imagen;
2. recuperacion de parametros generativos (`p:q`, fase, amplitud);
3. descriptores de ratios visibles y geometricos;
4. generalizacion OOD por familias de ratios;
5. diferencias entre probes que lockean y probes que recorren el espacio sin relocking inmediato;
6. transferencia sintético → captura física.

## Por qué importa para Phideus

Escalón 1 validó la mecanica descriptor-guided en musica. Escalón 2 abrió la prueba fuerte de armonia natural en voz. Escalón 3 agrega otra cosa: un dominio donde la relacion armónica no solo puede oirse sino tambien **verse** y **generarse** con control total.

Eso ofrece ventajas metodologicas concretas:

- ground truth exacto sobre los parametros latentes;
- ausencia de ambigüedad entre modalidad y etiqueta;
- posibilidad de partir por un benchmark sintético serio antes de pasar a hardware o captura real;
- espacio natural para estudiar descriptores geometricos y de ratio bajo splits IID y OOD;
- y, ahora, una arena especialmente fértil para volver experimental la distinción del libro entre `storage` y `retrieval`.

## Qué cambió en el encuadre del frente

El libro HIT cambió la lectura de Escalón 3. La pregunta ya no es solo si Phideus puede recuperar o generar figuras de Lissajous. La pregunta nueva es:

> qué parte del frente corresponde a organización armónica almacenada y qué parte corresponde a organización armónica activada por un probe.

Por eso el roadmap ya quedó dividido conceptualmente en dos arenas:

- **Storage Arena**: ratios racionales reducidos, closure, recurrencia, equivalencias.
- **Activation Arena**: drift, near-rational regimes, traversal del campo latente y probes `phi` / noble-number.

Y también en tres niveles geométricos:

- `L0` baseline plano;
- `L1` lectura angular post-hoc sobre embeddings planos;
- `L2` latente toroidal explícito (`T-VICReg`).

## Decisión conceptual ya fijada

La modalidad canonica del frente es:

- `audio canónico = señal XY estéreo`
- `figura canónica = trazado X contra Y`
- `audio perceptual/mic = modalidad secundaria`

Eso evita un error de diseño importante: una figura de Lissajous no sale de un audio mono cualquiera, sino de la relacion entre dos ejes.

## Estado del arte operativo

Sigue siendo cierto que no hay un dataset público canónico y adoptado de `audio ↔ figuras de Lissajous`. Pero esa observación ya no alcanza para describir el estado del repo, porque Phideus ya generó su propio banco canónico.

La lectura correcta ahora es:

- el diseño `synthetic-first` ya no es solo recomendación, sino artefacto generado;
- los factores latentes (`ratio`, `fase`, `amplitud`, `frecuencia base`, `split`) ya viven en `meta.json`;
- la capa física sigue para una fase posterior;
- `Plan_Claude.md` y `Legacy/Plan_inaugural_construccion_dataset_Codex.md` quedan como documentos de diseño y trazabilidad, no como sustitutos del estado actual.

## Unidad canonica del dataset

La unidad canónica ya materializada no es solo "wav + png", sino una **scene** con sus factores latentes:

- `xy_audio.wav`
- `xy_trace.npy`
- `figure_clean.png`
- `figure_style_*.png`
- `meta.json`

Metadatos mínimos ya presentes:

- `p`, `q`, `p_raw`, `q_raw`
- `ratio_reduced`, `ratio_float`, `ratio_id`, `equiv_id`
- `fx`, `fy`, `base_frequency`
- `phase_rad`
- `amp_x`, `amp_y`, `amp_ratio`
- `closure_period_s`, `duration`, `sample_rate`
- `renders`, `line_width`, `noise_snr`
- `scene_id`, `split`, `tier`

## Tareas prioritarias

Orden recomendado del frente:

1. **parameter recovery** - imagen o audio → `p:q`, fase y amplitud
2. **cross-modal retrieval** - audio ↔ figura con dual encoders
3. **structure-aware retrieval** - misma scene vs mismo `ratio_id` / `equiv_id`
4. **probe-dependent retrieval** - coseno, probe racional, probe `phi`, noble-number traversal
5. **activation mapping** - medir cobertura, locking y relocking del campo latente
6. **conditional generation** - parametros o figura objetivo → audio XY controlado

## Próximo paso único

El siguiente paso correcto ya no es “fijar una `v0.1` del dataset”, porque ese banco ya existe. Lo correcto ahora es:

1. congelar y auditar el banco `E3-P0` como objeto canónico;
2. bundlearlo para entrenamiento/evaluación sin romper la separación clean vs render-OOD;
3. correr `E3-P1` y `E3-P2`;
4. recién después abrir el gate `E3-P4`, donde se compararán probes racionales y no-locking sobre el mismo espacio latente.

## Relación con el resto del programa

- Escalón 1 sigue siendo el cierre fuerte de la mecanica descriptor-guided en musica.
- Escalón 2 sigue siendo el foco principal para la tesis fuerte de armonia natural.
- Escalón 3 abre una arena nueva: ratios visibles, control total, benchmark sintético con ground truth exacto y convergencia experimental con Beacon, ahora ya formulada también como banco de `storage / retrieval / activation`.
- Escalón 4 queda como expansión fisiologica ECG ↔ PPG.
