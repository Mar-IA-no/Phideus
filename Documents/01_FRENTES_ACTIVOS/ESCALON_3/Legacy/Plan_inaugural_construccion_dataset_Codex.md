> Nota de estado (2026-03-21): este documento debe leerse como **diseño originario** del dataset de Escalón 3. El estado canónico actual del frente ya vive en `README.md` y `ROADMAP_ESCALON_3.md`, y `E3-P0` ya quedó materializado en `data/escalon3/scenes/` a través del generador `experiments/escalon3/generate_lissajous_dataset.py`.

  No encontré un dataset público canónico, abierto y claramente adoptado para audio ↔ figuras de Lissajous. Lo que sí existe es un ecosistema técnico
  bastante útil: herramientas para generar audio XY/oscilloscope, trabajos que usan figuras de Lissajous como representación analítica, y precedentes
  sólidos de datasets multimodales sintéticos y finamente alineados. Mi recomendación académica es clara: para Phideus, conviene construir un dataset
  sintético-first, parameter-grounded y splitteado por ratios, y recién después agregar una capa física/hardware.

  Además, esta recomendación converge bastante con el borrador legacy interno PROPUESTA_ROSETA_2_AUDIO_CINEMATICA.md.

  Hallazgos que importan

  - Una figura de Lissajous está determinada por x = A sin(at + φ) y y = B sin(bt). La topología depende del ratio a:b; 6:4 y 3:2 producen la misma figura
    reducida. Fuente: Academo Lissajous Curves.
  - Un vectorscope grafica canal izquierdo contra canal derecho. O sea: la figura no “sale” de un audio mono cualquiera; sale de una señal estéreo XY.
    Fuente: Academo Vectorscope.
  - Para sintetizar estas curvas por audio, conviene generar buffers sobre un rango de t que contenga un número entero de períodos para evitar
    discontinuidades. Fuente: Academo Vectorscope.
  - La comunidad de oscilloscope music trabaja justamente así: el mismo audio se usa como señal audible y como entrada X/Y del osciloscopio. Fuente:
    Oscilloscope Music.
  - Hay herramientas maduras para generar y renderizar este tipo de señales: XYscope, osci-render, corrscope. No son datasets, pero sí infraestructura
    excelente para generarlos.
  - Ya hay trabajos recientes que usan métricas geométricas de Lissajous como centroid, bounding area, eccentricity y similarity index para clasificación.
    Eso valida que vale la pena guardar descriptores geométricos explícitos, no solo imágenes. Fuente: Scientific Reports 2025.
  - En multimodal music ML, un dataset sintético y finamente alineado es perfectamente defendible académicamente. El precedente más claro es MSMD. Fuente:
    Zenodo MSMD.

  La decisión conceptual clave

  Si quieren estudiar “qué sonido corresponde a qué figura”, primero tienen que fijar qué significa “sonido”.

  La formulación correcta para Fase 1 no es “audio acústico mono grabado por micrófono”, sino:

  - audio canónico = señal estéreo XY
  - figura canónica = trazado de X contra Y
  - audio perceptual/mic = modalidad secundaria

  Si empiezan por audio mono, meten una ambigüedad innecesaria: la figura depende de la relación entre dos ejes, no de una mezcla monofónica colapsada.

  Cómo empezaría el dataset

  La unidad canónica no debería ser “un wav y un png”, sino una scene con parámetros latentes conocidos.

  Cada scene debería tener:

  - xy_audio.wav: estéreo, donde L=x(t) y R=y(t)
  - xy_trace.npy: matriz T x 2 en float32 con la trayectoria exacta
  - figure_clean.png: raster limpio, sin glow ni blur
  - figure_style_*.png: vistas estilizadas con persistencia, ruido, blur, grosor, etc.
  - meta.json: parámetros generativos y descriptores derivados

  Metadatos mínimos:

  - p, q, ratio_float, ratio_reduced
  - base_frequency, fx, fy
  - phase_rad
  - amp_x, amp_y, amp_ratio
  - duration, sample_rate, closure_period
  - render_style, line_width, blur, noise_snr
  - lobe_h, lobe_v, crossings, eccentricity, area, symmetry_x, symmetry_y
  - scene_id, ratio_id, equiv_id, split

  La lógica experimental correcta

  Primero separaría el dataset en capas:

  1. Tier 0: Closed canonical
     Solo ratios racionales reducidos, senoides puras, sin drift.
     Objetivo: aprender la correspondencia ratio ↔ geometría sin confounds.
  2. Tier 1: Nuisance-controlled
     Misma física, pero con variación de fase, amplitud, base frequency, resolución, grosor, blur, ruido, persistencia.
     Objetivo: que el modelo aprenda ratio y no artefactos de render.
  3. Tier 2: Near-rational / dynamic
     Ratios casi racionales, phase drift, AM/FM suave.
     Objetivo: explorar descriptores dinámicos y hard negatives. Acá la observación de XYscope sobre que pequeñas diferencias de frecuencia animan la figura
     es muy útil.
  4. Tier 3: Real capture
     Misma scene sintética, pero reproducida en un setup real XY/oscilloscopio/láser y capturada por cámara, y opcionalmente por micrófono.
     Objetivo: transferencia sintético → real.

  Cómo lo splittearía

  No haría solo train/val/test random.

  Haría al menos cuatro tests:

  - IID: escenas nuevas con ratios ya vistos
  - ratio-OOD: ratios reducidos no vistos en train
  - scale-OOD: frecuencias base no vistas
  - render-OOD: estilos visuales o ruido no vistos

  Si no hacen ratio-OOD, el escalón no testea generalización de ratios; solo memoriza familias vistas.

  Qué tareas pondría primero

  El orden correcto sería:

  1. parameter recovery
     Desde audio o imagen, predecir p:q, fase y amp_ratio.
     Esto les da un sanity check muy fuerte.
  2. cross-modal retrieval
     Audio → figura y figura → audio, con dual encoders contrastivos.
     Esto es el primer benchmark serio.
  3. structure-aware retrieval
     No solo misma scene, sino misma ratio_id o equiv_id.
     Esto sirve mucho para Phideus porque separa estructura de escala.
  4. conditional generation
     Recién después.
     Y no iría a una black-box diffusion de entrada. Iría a:
     prompt/imagen/audio -> latent params -> renderer/synth determinista
     Porque acá la física ya la conocen.

  Qué descriptores de ratios conviene explorar en Phideus

  Acá conviene ser consistente con la rectificación vigente del repo: ratio lineal primero, log/perceptual como control.

  Yo definiría cuatro familias:

  - R1 natural
    p:q reducido, fx/fy, complejidad armónica (p+q, max(p,q)), distancia a ratios armónicos vecinos.
  - R2 perceptual-control
    log2(fx/fy), cents, buckets temperados.
    Útiles solo como brazo comparativo.
  - R3 geométrico
    lóbulos H/V, self-intersections, eccentricity, area, simetrías, Fourier descriptors de contorno.
  - R4 dinámico
    cierre temporal, drift, phase velocity, curvature-over-time.

  Tamaño con el que sí arrancaría

  Para un piloto serio, no enorme:

  - 16 ratios de train
  - 4 ratios OOD
  - 6 frecuencias base
  - 8 fases
  - 4 amp ratios
  - 2 semillas o repeticiones
  - 3 renders por scene

  Eso da 16 x 6 x 8 x 4 x 2 = 6,144 scenes de train, y con 3 renders 18,432 pares observables. Ya alcanza para retrieval y baselines. Después lo pueden
  escalar a 50k+ sin cambiar diseño.

  Mi recomendación fuerte

  No empiecen por “imágenes bonitas de Lissajous y wavs cualquiera”. Empiecen por un benchmark científico con latent factors explícitos. Si lo hacen bien,
  este escalón puede servir a la vez para:

  - retrieval audio↔figura
  - validación de descriptores de ratio
  - generalización OOD por ratios
  - generación controlable basada en física
  - transferencia sintético→real

  Eso sería un escalón muy limpio para Phideus.

  Fuentes

  - Academo, Lissajous Curves: https://academo.org/demos/lissajous-curves/
  - Academo, Vectorscope: https://academo.org/demos/vectorscope/
  - Oscilloscope Music: https://www.oscilloscopemusic.com/info/
  - XYscope: https://github.com/ffd8/xyscope
  - osci-render: https://github.com/jameshball/osci-render
  - Corrscope: https://github.com/corrscope/corrscope
  - Scientific Reports 2025, Dynamic Lissajous patterns: https://www.nature.com/articles/s41598-025-10218-4
  - MSMD (Zenodo): https://zenodo.org/records/2597505




  • No encontré un dataset público estándar y adoptado para audio ↔ figuras de Lissajous. Para ESCALON_3, lo útil hoy no es “bajar un benchmark ya hecho”,
  sino apoyarse en un conjunto de fuentes que sirven para: definir bien el objeto, montar tooling de generación/captura, elegir baselines de retrieval y
  diseñar un dataset sintético serio.

  Fuentes que sí valen la pena

  1. Academo: Lissajous Curves (https://academo.org/demos/lissajous-curves/)
     Pertinencia: es la referencia más limpia para fijar la física mínima del escalón. Deja explícito que la forma depende del ratio a:b, que 6:4 y 3:2
     colapsan a la misma figura reducida, y que amplitud/fase modulan la apariencia sin cambiar esa estructura base. Sirve para definir el espacio latente
     canónico del dataset.
  2. Academo: Vectorscope (https://academo.org/demos/vectorscope/)
     Pertinencia: probablemente la fuente más importante a nivel ingeniería. Explica que la figura sale de graficar L contra R, no de un audio mono
     cualquiera, y describe cómo sonificar ecuaciones paramétricas llenando buffers estéreo. Para ESCALON_3, esto empuja a definir la modalidad canónica
     como audio XY estéreo, no como mezcla acústica colapsada.
  3. Oscilloscope Music – INFO (https://oscilloscopemusic.com/info/)
     Pertinencia: no es una fuente académica, pero sí una referencia de dominio muy valiosa. Fija el principio operativo real de la escena oscilloscope/
     vector synthesis: el mismo audio que se oye es el que dibuja la imagen en X/Y. Es útil para pensar la futura fase física del escalón y para no diseñar
     un dataset desconectado de prácticas reales del dominio.
  4. XYscope (GitHub) (https://github.com/ffd8/xyscope)
     Pertinencia: tooling muy útil para una fase de captura o generación “hardware-aware”. Convierte gráficos vectoriales en audio para osciloscopio/láser,
     documenta sample rates, DACs, displays, modo XY, y además remarca que el ratio entre frecuencias es crucial. Para ESCALON_3 lo veo como herramienta
     fuerte para la fase synthetic-to-physical, no necesariamente para la primera versión del dataset.
  5. osci-render (GitHub) (https://github.com/jameshball/osci-render)
     Pertinencia: probablemente el tooling creativo más potente de los que vi. Permite generar audio de osciloscopio a partir de .svg, .obj, texto y escenas
     Blender, con scripting y control paramétrico. No lo usaría para el núcleo canónico del dataset, pero sí para:

  - generar variantes estilizadas,
  - producir datos “no ideales”,
  - explorar generación controlable más adelante.

  6. Corrscope (GitHub) (https://github.com/corrscope/corrscope)
     Pertinencia: lo pondría como herramienta secundaria de inspección y QA. Renderiza vistas de osciloscopio a partir de WAVs y tiene triggering por
     correlación. No lo veo como generador central de figuras de Lissajous, pero sí como ayuda para depurar estabilidad temporal, periodicidad y
     consistencia visual de las señales generadas.
  7. Dynamic Lissajous patterns for real time identification and localization of power quality disturbance (Scientific Reports, 2025)
     (https://www.nature.com/articles/s41598-025-10218-4)
     Pertinencia: es la mejor fuente que encontré para justificar features geométricas explícitas. Aunque el dominio es otro, usa patrones de Lissajous como
     firma visual y extrae métricas como área, skewness, kurtosis y centroid deviation. Para ESCALON_3, esto respalda guardar no solo imagen y audio, sino
     también un paquete de descriptores geométricos derivados.
  8. MSMD – Multimodal Sheet Music Dataset (Zenodo) (https://zenodo.org/records/2597505)
     Pertinencia: no tiene nada que ver con Lissajous en contenido, pero sí en diseño experimental. Es el mejor precedente que encontré para defender un
     dataset multimodal sintético, finamente alineado y pensado para retrieval. El punto fuerte para ustedes es metodológico: muestra que un benchmark
     sintético bien alineado puede ser científicamente válido y generalizar más allá del dominio de entrenamiento.
  9. Learning Audio–Sheet Music Correspondences for Cross-Modal Retrieval and Piece Identification (TISMIR, 2018)
     (https://transactions.ismir.net/articles/10.5334/tismir.12)
     Pertinencia: esta es la referencia académica más directa para la tarea de retrieval. Usa embeddings compartidos entre modalidades y retrieval cruzado.
     Para ESCALON_3, sirve como plantilla de evaluación y como argumento para arrancar con dual encoders contrastivos antes de pensar en generación fuerte.
  10. ImageBind (GitHub oficial) (https://github.com/facebookresearch/ImageBind)
     Pertinencia: baseline moderno para embedding multimodal. Une audio y visión en un espacio compartido y ya viene con pesos. No está hecho para
     Lissajous, así que no lo tomaría como solución final, pero sí como benchmark o inicialización para probar retrieval rápido. Precaución: licencia CC-BY-
     NC 4.0 en código/pesos.
  11. AudioCLIP (GitHub oficial) (https://github.com/AndreyGuzhov/AudioCLIP)
     Pertinencia: baseline más práctico que ImageBind para audio + visual. Tiene código y pesos, licencia MIT, y está pensado justo para alinear audio con
     el mundo CLIP. Puede servir como baseline inicial de retrieval si el lado visual se adapta a figuras rasterizadas.
  12. Wav2CLIP (GitHub oficial) (https://github.com/descriptinc/lyrebird-wav2clip)
     Pertinencia: útil si quieren un encoder de audio liviano que proyecte al espacio CLIP sin entrenar todo desde cero. Lo veo como baseline de arranque o
     como control de “audio-side only”, no como arquitectura final del escalón.

  Qué usaría de verdad para arrancar

  - Base conceptual del dataset: Academo Lissajous + Academo Vectorscope.
  - Tooling de generación: renderer propio simple primero; XYscope y osci-render después, como ramas de expansión.
  - Features explícitas: ideas geométricas del paper de Scientific Reports.
  - Diseño de benchmark y evaluación: MSMD + TISMIR.
  - Baselines de retrieval: AudioCLIP primero, ImageBind como benchmark adicional.

  Qué no haría

  - No usaría osci-render o material de oscilloscope music como base canónica del dataset.
    Razón: meten demasiada estilización y complejidad antes de fijar el núcleo físico del problema.
  - No empezaría por audio mono grabado por micrófono.
    Razón: para Lissajous, la modalidad natural es X/Y estéreo.

  Lectura estratégica para ESCALON_3
  La pila más coherente sería:

  - capa 1: dataset sintético canónico con parámetros exactos,
  - capa 2: retrieval audio-figura con dual encoders,
  - capa 3: descriptores de ratios y descriptores geométricos como brazos comparativos,
  - capa 4: transferencia a captura física con XYscope / osciloscopio / láser.

  Además del internet, ya vi que el repo tiene un antecedente interno muy alineado con esto: PROPUESTA_ROSETA_2_AUDIO_CINEMATICA.md. No lo tomaría como
  canónico sin revisión, pero sí como material base para no arrancar desde cero.

  Si querés, el siguiente paso útil es que te arme un documento de trabajo para ESCALON_3 con:

  - objetivos científicos,
  - taxonomía de modalidades,
  - estructura de carpetas,
  - metadatos,
  - splits,
  - y una primera lista de descriptores candidateables.
