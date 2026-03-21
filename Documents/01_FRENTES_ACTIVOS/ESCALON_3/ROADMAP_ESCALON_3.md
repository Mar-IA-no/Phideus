# ROADMAP - Escalon 3: Audio XY ↔ Figuras de Lissajous

> Fecha de creacion: 2026-03-12
> Ultima actualizacion: 2026-03-21
> Estado: `E3-P0` ya materializado; `E3-P1/P2` pendientes; `E3-P4` sigue siendo el gate decisivo del frente

## Addendum operativo del corte

Escalón 3 ya no debe leerse como roadmap puramente prospectivo. El banco canónico de `E3-P0` ya existe en el árbol:

- generador reproducible en `experiments/escalon3/generate_lissajous_dataset.py`;
- scenes materializadas en `data/escalon3/scenes/`;
- `6,016` scenes totales;
- `train/val/test=2144/448/480`;
- `ratio_ood=768`, `scale_ood=1024`, `equiv_ood=1152`.

La parte del roadmap que sigue plenamente abierta ya no es “cómo sería el dataset”, sino cómo convertir este banco en `P1` y `P2` sin contaminar la lectura de `P4`:

- `train/val/test` puro sobre ratios reducidos;
- no reducidos fuera de training como `equivalence-OOD`;
- `render-OOD` separado por diseño;
- `phi` todavía fuera del training y reservado para lectura/probing posterior.

## 0. Tesis del frente

Escalón 3 ya no debe leerse solo como un banco de `audio XY ↔ figuras de Lissajous`. Ese es el soporte material del frente, pero no su tesis más fuerte.

La formulación correcta, después del libro HIT, es esta:

> Escalón 3 es el banco sintético donde Phideus puede intentar separar experimentalmente tres cosas que en los otros dominios todavía aparecen mezcladas: **storage**, **retrieval** y **activation**.

La razón de fondo viene del nuevo arco teórico del libro:

- `Chapter 8`: storage por recurrencia y organización estable;
- `Chapter 9`: consonancia, orientación y regulación;
- `Chapter 10`: `activation problem`, donde la pregunta ya no es cómo se almacena una organización, sino cómo se la lee sin relocking inmediato.

Lissajous es el primer frente del programa donde esa distinción puede hacerse operativa con **ground truth total**:

- la relación armónica es visible;
- la señal es generable;
- la estructura paramétrica es exacta;
- el cierre o no-cierre de la curva es controlable;
- y el comportamiento del probe puede medirse sin la ambigüedad propia de datasets naturales.

La directiva nueva del frente es:

> Escalón 3 no debe diseñarse solo para clasificar figuras o alinear modalidades. Debe diseñarse para estudiar cuándo una organización armónica se almacena, cuándo se recupera y cuándo puede ser activada por un probe que no se relockea inmediatamente con ella.

---

## 1. Preguntas científicas del frente

### 1.1 Storage

¿Puede un modelo cross-modal aprender una organización latente estable de escenas armónicas cuando esas escenas están definidas por ratios racionales reducidos, cierre de curva y periodicidad explícita?

### 1.2 Structure recovery

¿El espacio latente representa solo identidad de escena o también estructura de ratio, equivalencia reducida, fase, amplitud y complejidad armónica?

### 1.3 Retrieval regime

¿La forma de leer el espacio importa tanto como la forma de entrenarlo?

### 1.4 Activation hypothesis

¿Un probe `phi` o noble-number produce una lectura del espacio latente diferente a la de probes racionales, en el sentido preciso de:

- menor relocking;
- mayor cobertura del campo latente;
- mejor legibilidad estructural;
- o acceso a organización que un vecino coseno local no recupera?

### 1.5 Transfer

Si aparece una diferencia robusta entre probes racionales y no-locking en el banco sintético, ¿esa diferencia sobrevive cuando el frente pasa a captura física y converge con Beacon?

---

## 2. Reencuadre conceptual del frente

## 2.1 Dos arenas experimentales

### Arena S - Storage Arena

Dominada por:

- ratios racionales reducidos `p:q`;
- curvas cerradas;
- periodicidad;
- closure;
- recurrencia;
- equivalence classes.

Pregunta:

> cómo se organiza la estructura armónica en el espacio latente.

### Arena A - Activation Arena

Dominada por:

- near-rational regimes;
- drift;
- phase slip;
- no closure;
- probes irracionales;
- `phi` y noble numbers;
- traversal del espacio;
- no-locking readout.

Pregunta:

> cómo se lee o se activa una organización ya almacenada sin reducirla de nuevo a lockeo local inmediato.

Esta distinción es metodológicamente obligatoria. `phi` no debe entrar primero como una clase más del dataset. Debe entrar como **operador**, **trayectoria de lectura** y **probe**.

## 2.2 Tres niveles geométricos

### L0 - Baseline plano

- embeddings euclídeos estándar;
- retrieval por coseno o distancia local;
- benchmark base del frente.

### L1 - Angular post-hoc

- modelo todavía entrenado en espacio plano;
- embeddings leídos con coordenadas angulares;
- retrieval por distancia circular y traversal `phi` o noble-number;
- diagnóstico de locking / no-locking sin reescribir todavía la loss.

### L2 - Latente toroidal explícito

- embeddings viviendo en `T^N`;
- training con `T-VICReg`;
- retrieval por probe sequence en el toro.

La secuencia recomendada del frente es **L0 -> L1 -> L2**, no salto directo a `T-VICReg`.

---

## 3. Objeto canónico del escalón

La unidad de trabajo sigue siendo una **scene**.

Cada scene debe incluir:

- `xy_audio.wav`
- `xy_trace.npy`
- `figure_clean.png`
- `figure_style_*.png`
- `meta.json`

La modalidad canónica del frente queda fijada así:

- `audio canonico = señal estéreo XY`
- `figura canonica = trazado X contra Y`
- `audio perceptual/mic = modalidad secundaria`

Esto evita un error de diseño básico: una figura de Lissajous no sale de un audio mono cualquiera, sino de la relación entre dos ejes.

## 3.1 Metadatos mínimos obligatorios

- `scene_id`
- `split`
- `p`, `q`
- `ratio_reduced`
- `ratio_float`
- `ratio_id`
- `equiv_id`
- `base_frequency`
- `fx`, `fy`
- `phase_rad`
- `amp_x`, `amp_y`, `amp_ratio`
- `duration`, `sample_rate`
- `closure_period`
- `render_style`, `noise_snr`, `line_width`, `blur`

## 3.2 Metadatos nuevos para lectura estructural

### Complejidad racional

- `harmonic_order = p + q`
- `max_component = max(p, q)`
- `coprime`
- `distance_to_nearest_simple_ratio`

### Geometría visible

- `lobes_x`, `lobes_y`
- `self_intersections`
- `eccentricity`
- `bounding_area`
- `symmetry_x`, `symmetry_y`
- `curvature_stats`
- `fourier_shape_descriptor`

### Dinámica

- `closed_curve`
- `closure_error`
- `phase_velocity`
- `drift_rate`
- `beat_period` si aplica
- `quasi_periodic`

### Anotaciones de activation

- `activation_regime`
  - `locked`
  - `near_locked`
  - `non_locking`
- `probe_family_target`
- `expected_relocking_depth`
- `noble_probe_relevance`

Estas últimas pueden empezar como anotaciones derivadas del generador, no como verdad “psicológica” del sistema.

---

## 4. Diseño del dataset

## 4.1 Tierización recomendada

### Tier 0 - Canonical Closed Rational

Condiciones:

- senoides puras;
- ratios racionales reducidos;
- cero drift;
- cero modulación;
- closure exacta.

Rol:

- storage puro;
- sanity benchmark;
- parameter recovery limpio;
- retrieval scene-level y structure-level.

### Tier 1 - Nuisance-Controlled Rational

Mismos ratios racionales, pero con variación controlada de:

- fase;
- amplitud;
- frecuencia base;
- render style;
- blur;
- noise;
- line width;
- persistence.

Rol:

- robustez;
- separación entre ratio y estilo;
- prevención de leakage visual trivial.

### Tier 2 - Near-Rational Dynamic

Regímenes:

- `p:q + epsilon`;
- drift lento;
- AM/FM suave;
- phase walk;
- closure incompleta.

Rol:

- abrir la Activation Arena;
- distinguir locking vs no-locking;
- estudiar cuándo una figura casi cerrada pasa a requerir otro tipo de lectura.

### Tier 3 - Noble / phi-driven Traversal

No debe pensarse como un “dataset phi” paralelo al racional, sino como:

- escenas preparadas para testear probes de lectura;
- comparaciones entre traversal racional, traversal `phi`, traversal noble-number y traversal aleatorio control;
- campos de activación.

Rol:

- test directo de las predicciones del `Chapter 10`.

### Tier 4 - Physical Capture

- reproducción real en setup XY;
- captura por cámara, vectorscope u osciloscopio;
- no idealidad instrumental;
- blur real, jitter, latencia, ruido.

Rol:

- synthetic -> real;
- convergencia con Beacon;
- prueba de si el fenómeno no era solo un artefacto del renderer.

---

## 5. Diseño de splits

No alcanza con `train/val/test` random.

## 5.1 Splits mínimos

1. `IID` - escenas nuevas con ratios vistos
2. `ratio-OOD` - ratios reducidos no vistos en train
3. `scale-OOD` - frecuencias base no vistas
4. `render-OOD` - estilos visuales o ruido no vistos

## 5.2 Splits nuevos recomendados

5. `equivalence-OOD` - generalización entre equivalencias racionales (`6:4` vs `3:2`)
6. `complexity-OOD` - train en ratios simples, test en ratios de mayor complejidad armónica
7. `closure-OOD` - train en regímenes cerrados, test en near-rational / quasi-periodic
8. `activation-OOD` - train sin probes irracionales explícitos, test con lectura `phi` / noble-number

La prueba fuerte del escalón no es memorizar escenas. Es generalizar estructura de ratio y después verificar si el método de lectura cambia la estructura accesible.

---

## 6. Primer dataset viable

El piloto `v0.1` ya quedó materializado, con una lectura un poco más austera que la recomendación original:

- `16` ratios reducidos en `train/val/test`;
- `4` ratios `ratio-OOD`;
- `6` equivalencias no reducidas en `equivalence-OOD`;
- `6` frecuencias base para el régimen principal;
- `2` frecuencias nuevas para `scale-OOD`;
- `8` fases;
- `4` amp ratios;
- `3` renders por scene (`clean`, `noisy`, `thick`);
- sin repeticiones en `P0-P2`.

Eso dejó un banco lo bastante serio para arrancar `P1/P2` sin inflar el frente antes de tiempo:

- `3,072` scenes reducidas para `train/val/test`;
- `768` scenes `ratio-OOD`;
- `1,024` scenes `scale-OOD`;
- `1,152` scenes `equivalence-OOD`;
- `6,016` scenes totales.

La recomendación operativa ya no es “crear el piloto”, sino no crecer de tamaño antes de fijar:

- taxonomía de ratios;
- classes de equivalencia;
- reglas de closure;
- metadatos geométricos;
- y diagnóstico de leakage entre ratio, fase, estilo y frecuencia base.

---

## 7. Tareas del frente

## 7.1 Tarea A - Parameter Recovery

Desde audio o imagen, recuperar:

- `ratio_id`
- `p:q`
- `phase_rad`
- `amp_ratio`
- `closure class`

Función:

- sanity check;
- verificar que el banco es aprendible;
- estimar qué factores son triviales y cuáles no.

## 7.2 Tarea B - Scene Retrieval

Audio ↔ figura de la misma scene.

Función:

- benchmark básico multimodal del escalón.

## 7.3 Tarea C - Structure Retrieval

No solo misma scene, sino:

- mismo `ratio_id`;
- mismo `equiv_id`;
- misma familia de complejidad;
- mismo régimen dinámico.

Función:

- separar identidad puntual de estructura armónica compartida.

## 7.4 Tarea D - Probe-dependent Retrieval

Comparar retrieval usando:

- coseno plano;
- distancia geodésica angular;
- probe racional;
- probe `phi`;
- probe noble-number generalizado;
- probe aleatorio control.

Pregunta:

> qué clase de estructura recupera cada probe y cómo cambia la lectura del campo latente según el tipo de query.

## 7.5 Tarea E - Activation Mapping

Dado un query point, medir:

- cobertura;
- densidad de visitas;
- clusters tocados;
- profundidad de relocking;
- sensibilidad a racionalidad del probe.

Esta es la tarea nueva fuerte del escalón y la más directamente conectada con el libro.

## 7.6 Tarea F - Conditional Generation

Solo después de fijar el banco limpio.

No conviene abrir generación black-box antes de saber:

- si el modelo distingue ratio de estilo;
- si el retrieval estructura el espacio correctamente;
- y si el método de lectura ya dejó señal diferencial.

---

## 8. Métricas del escalón

## 8.1 Métricas clásicas

- `A2I R@10`
- `I2A R@10`
- `S = min(A2I, I2A)`
- MRR
- `R@1`, `R@5`, `R@20`

## 8.2 Métricas estructurales

- `ratio-class accuracy`
- `equiv-class retrieval`
- `complexity-level accuracy`
- `closure classification`

## 8.3 Métricas geométricas del latente

- silhouette por `ratio_id`
- silhouette por `equiv_id`
- cluster purity
- neighborhood consistency
- RSA / CKA con matrices generativas de ratio, equivalencia y complejidad

## 8.4 Métricas nuevas de activation

### Activation Gain

Mejora o diferencia de retrieval estructural con probe `phi` o noble-number frente a coseno estándar.

### Locking Selectivity

Cuánto concentra un probe racional sus visitas en clusters conmensurables específicos.

### Coverage Uniformity

Qué tan uniformemente cubre el campo latente un probe en `K` pasos.

### Relocking Depth

Número de pasos hasta captura estable por un basin racional.

### Probe Sensitivity Spectrum

Curva de desempeño según familia de probe:

- racional simple
- racional complejo
- irracional aleatorio
- `phi`
- noble-number

### Basin Exposure

Cuántos clusters distintos se vuelven legibles bajo cada probe.

Estas métricas no son ornamentales. Son la forma de volver experimental la diferencia entre `storage` y `activation`.

---

## 9. Arquitecturas a comparar

## 9.1 Línea base obligatoria

Dual encoder estándar:

- audio XY encoder
- image encoder
- projection heads
- VICReg estándar

## 9.2 Línea descriptorial

Comparar familias:

- `R1 natural`
- `R2 perceptual-control`
- `R3 geometric`
- `R4 dynamic`

por mecanismos:

- concat
- conditioned projection
- FiLM
- cross-attention liviano si realmente agrega algo

## 9.3 Línea geométrica

### G0

Flat latent + cosine.

### G1

Flat latent + angular readout.

### G2

Mixed geometry:

- subespacio toroidal para componentes periódicas;
- subespacio euclídeo para residuo no periódico.

### G3

Full toroidal latent + `T-VICReg`.

La recomendación del frente es clara:

> no empezar por `G3`. Empezar por `G0 -> G1 -> G2`.

## 9.4 Línea de probes

Separada de la arquitectura:

- coseno local
- kNN angular
- rational shift
- phi shift
- noble-number shift
- random shift

Es importante factorizar **probe** y **training geometry** para no mezclar dos hipótesis distintas.

---

## 10. Roadmap por fases

## E3-P0 - Canonical Generator

### Estado actual

Ya materializado en `experiments/escalon3/generate_lissajous_dataset.py` + `data/escalon3/scenes/`.

### Objetivo

Fijar el generador determinista y el manifiesto de metadatos.

### Entregables

- generador reproducible;
- manifiesto de scene;
- set de sanity scenes;
- validación de closure;
- validación de equivalence classes.

### Criterio de cierre

- reproducibilidad exacta;
- metadatos completos;
- renders coherentes;
- cero leakage obvio entre ratio y estilo.

## E3-P1 - Parameter Recovery

### Estado actual

Pendiente. El banco ya existe; falta convertirlo en baseline de aprendibilidad.

### Objetivo

Probar que el banco es aprendible.

### Experimentos

- imagen -> `ratio_id`
- audio -> `ratio_id`
- imagen -> fase / amp_ratio
- audio -> fase / amp_ratio

### Criterio de cierre

- accuracy alta en IID;
- errores interpretables en OOD;
- baseline fuerte antes de retrieval.

## E3-P2 - Flat Cross-Modal Retrieval

### Estado actual

Pendiente. El banco y sus splits ya están disponibles, pero el benchmark multimodal todavía no quedó corrido ni estabilizado.

### Objetivo

Construir el benchmark multimodal base.

### Experimentos

- dual encoder `audio XY ↔ image`
- scene retrieval
- structure retrieval
- splits IID / ratio-OOD / equivalence-OOD / render-OOD

### Criterio de cierre

- señal clara de retrieval;
- latente ya organizado por ratio y equivalencia;
- benchmark canónico del frente.

## E3-P3 - Descriptor × Mechanism Matrix

### Objetivo

Separar descriptor, control y mecanismo desde el inicio.

### Experimentos

Cruce entre:

- `R1 natural`
- `R2 perceptual-control`
- `R3 geometric`
- `R4 dynamic`

por mecanismos:

- concat
- conditioned projection
- FiLM
- algún cross-attention liviano solo si no ensucia la lectura

### Criterio de cierre

- tabla descriptor × mecanismo;
- lectura explícita de qué aporta cada familia;
- no repetir la confusión histórica entre descriptor y arm.

## E3-P4 - Probe Regime on Flat Latent

### Objetivo

Testear el `activation problem` sin reescribir todavía el training.

### Experimentos

Sobre embeddings ya entrenados:

- coseno estándar;
- distancia angular;
- rational probe traversal;
- phi traversal;
- noble-number traversal;
- random irrational control.

### Criterio de cierre

- evidencia de que el método de lectura cambia la estructura recuperada;
- o evidencia clara de que no la cambia.

Este es el gate central del frente. Decide si vale la pena avanzar a `T-VICReg`.

## E3-P5 - Mixed Geometry Latent

### Objetivo

Introducir geometría no plana con riesgo moderado.

### Diseño

- subespacio toroidal para periodicidad;
- subespacio euclídeo para residuo;
- comparación contra el baseline plano.

### Criterio de cierre

- training estable;
- mejor estructura o mejor lectura;
- sin colapso raro ni degradación trivial.

## E3-P6 - Full T-VICReg

### Objetivo

Testear la hipótesis geométrica fuerte.

### Diseño

- projector toroidal;
- geodesic invariance;
- circular variance;
- circular covariance;
- retrieval por probe sequence en el toro.

### Criterio de cierre

No pedir solo mejor `R@10`. Pedir al menos una mejora defendible en:

- estructura latente por ratio;
- `activation gain`;
- menor relocking;
- mejor coverage uniformity;
- o transferencia al Tier dinámico.

## E3-P7 - Dynamic Activation Arena

### Objetivo

Mover el frente desde “ratios visibles” a “activación visible”.

### Experimentos

- near-rational regimes;
- drift;
- phase slip;
- no closure;
- modulation;
- probes de distintas familias.

### Preguntas

- cuándo aparece locking;
- cuándo aparece no-locking;
- cómo cambia el campo latente;
- qué probe revela mejor la organización.

## E3-P8 - Physical Transfer / Beacon Convergence

### Objetivo

Llevar el hallazgo hacia captura física y convergencia con Beacon.

### Experimentos

- synthetic scene -> reproducción XY física;
- captura por cámara / osciloscopio;
- comparación de comportamiento latente;
- condiciones enteras vs offsets de activation.

### Criterio de cierre

No hace falta cerrar “producto”. Hace falta cerrar “puente experimental”.

---

## 11. Criterios GO / NO-GO

### GO

- el generador produce escenas estables y correctamente etiquetadas;
- parameter recovery y retrieval muestran señal clara;
- los splits `ratio-OOD` y `equivalence-OOD` dejan lecturas interpretables;
- el banco no colapsa ratio con estilo;
- aparece alguna señal consistente de diferencia entre probes racionales y no-locking.

### NO-GO / pausa

- si el dataset no separa bien ratio de fase / estilo / escala;
- si el benchmark queda trivial por leakage entre splits;
- si `E3-P4` no produce ninguna diferencia interpretable y `G1`/`G2` tampoco justifican complejidad extra;
- si el frente deriva demasiado rápido a estilización visual o generación antes de fijar el banco científico.

---

## 12. Dependencias conceptuales ya fijadas

Fuentes base del frente:

- `Plan_Claude.md`
- `Legacy/Plan_inaugural_construccion_dataset_Codex.md`
- `Documents/90_ARCHIVO_GLOBAL/Legacy/Rosetta/PROPUESTA_ROSETA_2_AUDIO_CINEMATICA.md`
- `Biblioteca/Toroidal_Latent_Fields/00_INVESTIGACION_CAMPO_LATENTE_TOROIDAL.md`
- `Biblioteca/Toroidal_Latent_Fields/01_PAPERS_CLAVE.md`
- `Biblioteca/Toroidal_Latent_Fields/02_HALLAZGOS_ADICIONALES.md`

Lectura adicional ya incorporada por contexto:

- libro HIT en `manifiesto_HIT_Beancon_Phideus/`
- nuevo `Chapter 10` (`The Activation Problem`)

---

## 13. Relación con la triplescaloneta

- Escalón 1: validación descriptor-guided en música
- Escalón 2: prueba fuerte de armonía natural en voz
- Escalón 3: banco sintético donde storage, retrieval y activation pueden separarse con control total
- Escalón 4: expansión fisiológica ECG ↔ PPG

Escalón 3 no reemplaza a los anteriores. Su valor es distinto:

- ofrece un dominio donde la relación armónica puede estudiarse con control total de los factores latentes;
- permite volver visible la diferencia entre organización almacenada y organización activada;
- y funciona como laboratorio formal de convergencia con Beacon.
