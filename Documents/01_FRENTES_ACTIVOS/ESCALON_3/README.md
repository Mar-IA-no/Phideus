<div align="center">

# Escalón 3
### Audio XY ↔ Figuras de Lissajous

![Status](https://img.shields.io/badge/Status-P5--P6_Consolidated-0A7E3B?style=for-the-badge)
![Focus](https://img.shields.io/badge/Focus-Lissajous-1F6FEB?style=for-the-badge)
![Updated](https://img.shields.io/badge/Updated-2026--03--21-F59E0B?style=for-the-badge)

</div>

> [!IMPORTANT]
> **Estado actual**: este frente ya no es solo línea conceptual. `E3-P0` ya dejó un generador reproducible en `experiments/escalon3/generate_lissajous_dataset.py`, `E3-P1` ya validó aprendibilidad por `ratio`, `E3-P2` ya dejó un baseline dual, `E3-P4` ya fue corrido sobre ambos `L0`, y `E3-P5/P6` ya completaron la primera pasada geométrica del frente.
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
- `E3-P1` ya quedó cerrado como baseline de aprendibilidad por `ratio`: `ratio_acc = 1.000` en ambas modalidades.
- `E3-P2` ya dejó dos referencias reales:
  - `P2-flat` como baseline general de retrieval (`IID S = 0.583`, `silhouette = 0.960`);
  - `P2-cqtshift` como baseline alternativo para invariancia audio→image (`scale-OOD a2i = 0.476`, `equiv-OOD a2i = 0.458`).
- `E3-P4` ya dejó un corte interpretable:
  - sobre `P2-flat`, algunos traversals mejoran marginalmente a `cosine` en `scale-OOD a2i`, pero `phi` no queda claramente separado de otros irracionales;
  - sobre `P2-cqtshift`, las métricas primarias saturan en `1.0` y dejan de discriminar familias de probe;
  - el resultado completo quedó resumido en `Resultados_E3_P4.md`.
- `E3-P5` y `E3-P6` ya quedaron corridos y leídos con checkpoints estructuralmente correctos:
  - `P5-flat` no desplaza a `P2-flat`, pero deja evidencia de contribución causal de la rama toroidal;
  - `P5-cqtshift` queda como mejor brazo geométrico/OOD actual (`scale-OOD S = 0.508`, `equiv-OOD S = 0.472`);
  - `P6-flat` sale negativo;
  - `P6-cqtshift` organiza muy bien el toro, pero no supera a `P5-cqtshift` en las métricas OOD primarias.
- el resultado completo de la línea geométrica quedó resumido en `Resultados_E3_P5_P6.md`.

Lo que todavía no debe sobreafirmarse:

- `P2-flat` y `P2-cqtshift` no son intercambiables ni deben promediarse;
- `P2-cqtshift` no reemplaza al baseline canónico porque pierde retrieval IID y robustez de render frente a `P2-flat`;
- `P5-cqtshift` no debe venderse como ganador universal del frente: `P2-flat` sigue siendo mejor baseline general de `IID`;
- el resultado de `P6` no debe venderse como refutación definitiva de toda geometría toroidal posible: lo que hoy queda documentado es que **la receta toroidal pura actual no gana**;
- la lectura `GO / NO-GO` del frente ya no debe descansar en thresholds sueltos heredados de scripts o planes; la referencia operativa pasa a ser `CRITERIOS_GO_NO_GO_ESCALON_3.md`.

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

## Decisión operativa vigente

Escalón 3 ya no necesita seguir discutiendo si `P2` produjo o no un solo ganador. El frente dejó dos referencias útiles, pero con roles distintos:

- `L0-Flat Canonical` = `P2-flat`.
  Es la referencia principal para retrieval general, robustez visual y lectura canónica de `IID`.
- `L0-Shift Ratio-Aware` = `P2-cqtshift`.
  Es la referencia comparativa cuando la pregunta principal pasa a ser invariancia de ratio del lado audio.

La regla metodológica nueva ya no termina en `P4`. Hoy debe leerse así:

- `P4` se formula primero sobre `L0-Flat Canonical`;
- y se replica después sobre `L0-Shift Ratio-Aware`;
- sin mezclar embeddings, sin promediar scores y sin vender una mejora de probe como si fuera automáticamente independiente del encoder que la sostiene.
- pero un resultado negativo o ambiguo en `P4` no alcanza por sí solo para clausurar la hipótesis de geometría no plana, porque `P4` solo interroga métodos de lectura sobre embeddings entrenados en `L0`.
- después de esa comparación, la primera pasada geométrica ya deja otra lectura consolidada:
  - `P5-flat` no reemplaza al baseline general;
  - `P5-cqtshift` es el mejor brazo geométrico/OOD del corte;
  - `P6` no desplazó a `P5` bajo la receta actual.

Y, desde este punto del frente, también conviene fijar otra regla práctica: **Codex no debería ser el ejecutor por defecto de `P4` ni de los runs largos del escalón**. El diseño metodológico, la auditoría de scripts, la lectura de resultados y la documentación canónica quedan mejor del lado de Codex; la implementación operativa, el tuning técnico, el monitoreo en `tmux` y la ejecución sostenida quedan mejor del lado de Claude. La versión corta de ese reparto quedó consolidada en `Documents/00_TRONCAL/PROTOCOLO_OPERATIVO_CODEX_CLAUDE.md`.

## Tareas prioritarias

Orden recomendado del frente:

1. **conservar `P2-flat` como baseline canónico general** - referencia principal para `IID` y robustez visual
2. **conservar `P2-cqtshift` como baseline alternativo ratio-aware** - referencia comparativa para invariancia audio-side
3. **tomar `P5-cqtshift` como mejor brazo geométrico/OOD actual** - sin convertirlo en ganador universal del frente
4. **documentar `P6` como hipótesis pura no ganadora en esta receta** - estructura toroidal fuerte sin mejora suficiente de retrieval
5. **si Escalón 3 se reabre experimentalmente** - partir desde el mejor brazo actual o desde replicación, no desde reruns ciegos del toro puro
6. **descriptor × mechanism (`P3`) y activation arena (`P7`)** - quedan como líneas abiertas de segundo tiempo, no como deuda metodológica básica de `P5/P6`

La especificación operativa detallada de `P5/P6` ya no queda implícita en el roadmap: ahora vive en `PLAN_E3_P5_P6_GEOMETRIA_NO_PLANA.md`.

## Próximo paso único

El siguiente paso correcto ya no es “correr `P5/P6`”, porque esa primera pasada completa ya existe. Tampoco es seguir insistiendo con el toro puro como si todavía no hubiera resultado. Lo correcto ahora es:

1. congelar `P2-flat` como baseline canónico general;
2. congelar `P5-cqtshift` como mejor brazo geométrico/OOD actual;
3. registrar `P6` como hipótesis pura interesante, pero no ganadora bajo la receta actual;
4. si Escalón 3 vuelve a abrirse, hacerlo desde replicación, `activation` o transferencia, no desde una repetición ciega del mismo `P6`.

## Relación con el resto del programa

- Escalón 1 sigue siendo el cierre fuerte de la mecanica descriptor-guided en musica.
- Escalón 2 sigue siendo el foco principal para la tesis fuerte de armonia natural.
- Escalón 3 abre una arena nueva: ratios visibles, control total, benchmark sintético con ground truth exacto y convergencia experimental con Beacon, ahora ya formulada también como banco de `storage / retrieval / activation`.
- Escalón 4 queda como expansión fisiologica ECG ↔ PPG.
