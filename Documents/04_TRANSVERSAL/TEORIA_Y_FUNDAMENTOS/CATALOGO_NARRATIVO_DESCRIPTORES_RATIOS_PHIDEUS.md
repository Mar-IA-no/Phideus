# Catalogo Narrativo de Descriptores de Ratios en Phideus

Fecha: 2026-02-26
Base de referencia: `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/07_GATE_4_3_RATIO_RE_CENTRICO/INFORME_GATE_4_3_RATIO_RE_CENTRICO.md`, `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/07_GATE_4_3_RATIO_RE_CENTRICO/plan_gate_4.3.md`, `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/08_GATE_4_4_ARQUITECTURAS_MAYORES/README.md`, `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/09_GATE_4_5_LR_SCHEDULE_OPTIMIZATION/README.md`  
Estilo narrativo de referencia: `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/06_GATE_4_2_RATIO_CENTRICO/Explicacion_gate4.2_claude.md`

---

## Addendum Operativo (2026-02-25, Gate 5B activo con validación científica)

Estado del frente al corte:
1. Gate 4.3 cerró con 13 brazos 5ep y ranking estable.
2. `d4a4` (D4+A4 same-mod concat) quedó como mejor brazo corto (`S_best=69.8%`).
3. `A4r` (reverse cross-att audio) quedó como mejor brazo single-descriptor (`S_best=68.6%`).
4. Run largo `d4a4-scratch` cerró en 30ep (`S=83.6%`, multi-seed e30 `84.1% +/- 2.3pp`).
5. Gate 4.4 cerró screening completo de 24 brazos (incluye `moe-a4-v2/v3/v4`):
   - Third Tower: `t3-wt=67.6%`, `t3-tri=65.0%`, `t3-anc=42.2%`.
   - FiLM: `film-a4=59.2%`, `film-d4=58.6%`, `film-dual=59.4%`.
   - MoE: `moe-a4=58.2%`, `moe-dual=59.2%`, `v2=60.2%`, `v3=59.2%`, `v4=59.4%`.
6. Bloque de runs largos 30ep quedó completamente cerrado:
   - `d4a4=83.6%`, `a4r=82.0%`, `d4-a4r=79.8%`, `t3-wt=79.8%`, `d4a4r=74.4%`, `moe-dual=72.6%`.
7. Gate 4.5 formalizado (scheduler/LR), con stretched/hold ya cerrados:
   - `d4a4 60ep` completado (`S=83.8%`, nuevo record).
   - `a4r 60ep` completado (`S=79.4%`).
   - `D0 60ep` completado (`S=72.8%`).
   - `d4-a4r 60ep` completado (`S=79.8%`, empate con 30ep).
   - `t3-wt` 50ep hold completado (`S=81.2%`).
   - `moe-dual 60ep` dead por time limit (best `S=73.0%` en e30, no sostenido).
8. Bloque de contraste de scheduler (`cosine-tail`) en cierre operativo:
   - `a4r ctail` completado (`S=80.6%`),
   - `D0 ctail` y `d4a4 ctail` disponibles como referencia operativa,
   - `d4-a4r ctail` fuera de ruta crítica para Gate 5B,
   - parámetros: `--lr-cosine-ref-epochs 30 --lr-floor 0.10 --lr-tail-end 0.02`.
9. Gate 5B en ejecución:
   - Test12 (scoreboard) cerrado con checkpoints canónicos (`D0`, `d4a4`, `a4r`, `d4-a4r`).
   - Test01 (causal ablation) cerrado en 5 arms (`D0`, `d4`, `d4a4`, `a4r`, `d4-a4r`).
   - hallazgo descriptorial del corte:
     - A4/A4r = señal causal dominante en inferencia;
     - D4 = aporte marginal en duales.
   - Test04 (transposición) cerrado.
   - Test03 (ratio probe), Test06 (RSA/CKA), Test08 (ratio decoding) y Test10 (visualizaciones) cerrados.
   - Test09 cerrado (`D0`, `d4a4`, `a4r`, `d4-a4r`) con patrón consolidado:
     - temporal robusto;
     - velocity/octava frágiles;
     - ruido bimodal (D0 más robusto en 40-20 dB, reverse xatt mejor en 5 dB).
   - pendientes UNC: Test02 y Test05.
   - paquete visual Gate 5B consolidado: `24 PNG` + `6 GIF`.
10. Test11 perceptual (pipeline de eventos MIDI) entra en fase de cierre operativo:
   - barridos `midi2events` cerrados en `D0` y `a4r`,
   - barridos finos GPU cerrados en `D0` (config humana líder: `t104_k44_p099`),
   - entrenamiento `audio2events` activo en GPU (`D0` en curso, `a4r` en cola), con reuso de caches de embeddings.

Este catálogo mantiene el inventario de descriptores; el estado experimental canónico vive en:
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/00_TRONCAL/Proyecto_Estado_Actual.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/07_GATE_4_3_RATIO_RE_CENTRICO/INFORME_GATE_4_3_RATIO_RE_CENTRICO.md`

---

## Introduccion

Este documento enumera, uno por uno, los descriptores de ratios que fueron apareciendo en Phideus desde las etapas de Roseta/UOEMD hasta el catalogo operativo vigente (Gate 4.3 + Gate 4.4 + Gate 4.5/5A).

La idea no es solo listar nombres tecnicos, sino dejar una historia corta por descriptor:

1. De donde salio.
2. Que intenta capturar.
3. Como se representa.
4. Que aprendimos de su uso.

Este catalogo se mantiene como documento vivo: cada vez que se incorpore un descriptor nuevo en BIAS_CONTROL, se actualiza aqui y en `INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md`.

---

## Parte I - Descriptores ya usados (historico + vigente)

### 1. H0 - Histograma de ratios STFT v2.2

Este descriptor nace en el revisionismo UOEMD, cuando se corrige la extraccion inicial para evitar la explosion combinatoria de pares no informativos. Su rol fue reestablecer una base densa y estable.

Representa la distribucion de ratios como un vector de bins continuos. Conceptualmente es un "resumen estadistico global": no guarda secuencia, no guarda eventos individuales, pero guarda forma de distribucion.

Leccion: fue util como punto de apoyo. Mostro mejor separacion que los enfoques sparse en UOEMD y termino inspirando la rama D1 de Gate 4.

### 2. H1 - Enriched v4.1 (prop, energia, entropia)

Sale de la etapa en que el proyecto deja de usar solo conteo de ratios y empieza a pedir semantica por bin. El histograma ya no dice solo "cuanto aparece", tambien dice "con que energia" y "con que dispersion local".

Forma tipica: tensor `[512, 3]` (o equivalente segun version). Es una representacion mas rica que H0, pero aun global en el tiempo.

Leccion: agrego expresividad, pero la falta de temporalidad global seguia siendo un limite para tareas de alineacion fina.

### 3. H2 - Temporal v5.0 / Roseta v2.2

Este descriptor nace cuando se hace el cambio de paradigma: de un histograma por archivo a una secuencia de histogramas por frame/ventana temporal.

Forma tipica: `[T, 256, 3]`. Mantiene la idea enriquecida, pero ahora en el eje tiempo. Esto permite capturar evolucion, no solo promedio.

Leccion: pre-red mejoro claramente, pero no alcanzo por si solo para garantizar que la red capitalice toda la mejora en UOEMD. Igual dejo una direccion clave: temporalidad importa.

### 4. H3 - Gate 4 baseline (pairwise pitch histogram)

Es el primer gran intento en BIAS_CONTROL de inyectar ratios de forma explicita durante entrenamiento cross-modal audio/MIDI.

Se calcula sobre pares de pitches MIDI (`f_i/f_j`), con binning suave en rango controlado. Es un descriptor global de intervalo de pitch, no de ritmo.

Leccion: en Gate 4 aporto senal moderada en algunos cortes, pero no resolvio por si mismo el problema grande de esa etapa. Sirvio como base directa para D1.

### 5. H4 - Gate 4 enriched (3 canales: velocity, duration, unweighted)

Aparece como extension de H3 para preguntar si "como se toca" (dinamica y duracion) ayuda a que los ratios carguen mas informacion.

Estructura: histograma multicanal, luego flatten para encoder. Es la misma intuicion que H3, pero con contexto expresivo agregado.

Leccion: mostro mejoras puntuales, y quedo como semilla directa de D2 en Gate 4.2.

### 6. H5 - Gate 4.2 D1 (pitch ratio histogram)

Es la continuidad moderna de H3 dentro del framework robusto de Gate 4.2 (foundation lock, preflight, drift sentinel, validaciones mas duras).

Descripcion: histograma de ratios de pitch pairwise, branch auxiliar con VICReg auxiliar.

Leccion hasta ahora: no colapsa, se mantiene competitivo con el control en la ventana temprana. Queda pendiente su comportamiento de mayor profundidad temporal.

### 7. H6 - Gate 4.2 D2 (enriched 3-channel)

Es la continuidad moderna de H4 en el mismo marco de Gate 4.2.

Descripcion: version de D1 con canales ponderados por velocity/duration + canal unweighted.

Leccion: su valor esperado es capturar expresividad que D1 no ve. En el momento de este catalogo, su corrida depende de la progresion de stage correspondiente.

### 8. H7 - Gate 4.2 D3 (temporal-rhythmic)

D3 nace para romper la dependencia exclusiva en pitch. Introduce ratios temporales: IOI ratios, duration ratios y un histograma de intervalos de pitch consecutivos.

Es el puente entre la escuela "histograma de pitch global" y la escuela "estructura ritmica local/temporal".

Leccion esperada: distinguir si la senal de ratios util para retrieval esta mas en pitch o en ritmo.

### 9. H8 - Gate 4.2 D4 (local interval input augmentation)

D4 cambia de mecanismo, no solo de descriptor. En vez de agregar una loss auxiliar, inyecta features locales de intervalo directamente en la entrada del encoder MIDI.

Descripcion: 4 features por nota (`semitone_prev`, `semitone_next`, `log_ratio_prev`, `log_ratio_next`), luego proyeccion al espacio interno del encoder.

Leccion: es un test clave para separar "fallo de informacion" vs "fallo de mecanismo de inyeccion".

### 10. S0 - Constellation tokens UOEMD (sparse)

Nace desde la inspiracion Shazam: en vez de histogramas densos, usar pares anchor-target como tokens compactos (`[log_ratio, delta_t, weight, anchor_band, target_band]`).

Descripcion: representacion sparse por eventos relevantes.

Leccion: en UOEMD, C1-C6 no mostraron rendimiento convincente para alineacion cross-modal aprendida. Fue una alerta fuerte sobre perdida de informacion al sparsear en exceso.

### 11. S1 - Constellation tokens MAESTRO (pipeline implementado)

Es la adaptacion del enfoque constellation al frente MAESTRO. El pipeline de extraccion existe y esta conectado a scripts de entrenamiento.

Descripcion: tensor de tokens por frame para audio y MIDI, con mascara.

Leccion: el pipeline es valioso como infraestructura, pero no quedo como base principal de Gate 4.2. Su estado es mas de capacidad instalada que de resultado central consolidado.

### 12. K0 - Hash voting base (pre-red v2)

Es la version "fingerprinting exacto" de los ratios: discretizar y votar por coincidencia de hashes.

Descripcion: matching exacto, rapido, interpretable, pero fragil a colisiones y cuantizacion.

Leccion: diagnostico de colision generica. Habia coincidencias, pero poco discriminativas entre pares correctos vs aleatorios.

### 13. K1 - Route A (Event-Based)

Nace para corregir K0 con lenguaje de eventos musicales (onsets, intervalos, tipos de token) y hashing mas controlado.

Descripcion: tokens event-based + puntuacion por overlap ponderado.

Leccion: en pilotos pequenos brillo mucho; en auditorias grandes bajo, pero siguio arriba de random con eficiencia de tokens favorable frente a Route B.

### 14. K2 - Route B (Improved TF)

Nace como alternativa mas pesada y espectral a Route A: harmonic folding, onset anchoring, IDF mas agresivo, mayor volumen de tokens.

Descripcion: enfoque TF refinado con hashing mas especifico.

Leccion: tuvo pilotos muy altos, pero en escalas mas grandes su ventaja se redujo y el costo de tokens fue alto.

## Parte II - Catalogo operativo vigente (Gate 4.2 -> 4.4)

A diferencia de la version anterior, Gate 4.3 ya no se define como barrido D0-D10.

Desde este corte, el programa se organiza en tres lineas:

1. Linea MIDI (temperada): descriptores sobre eventos MIDI discretos (12-TET).
2. Linea Audio (armonia natural): descriptores sobre estructura espectral continua/no temperada.
3. Linea Dual: combinaciones MIDI+Audio para medir sinergia.

### 15. D0 - Control VICReg puro

Origen: baseline canonico de BIAS_CONTROL.

Descripcion: no agrega descriptor extra. Es el control transversal para comparar cualquier brazo MIDI, Audio o Dual.

Leccion: sin D0 del mismo bloque no hay inferencia causal limpia.

### 16. D1 - Pitch pairwise histogram (MIDI)

Origen: heredero directo de Gate 4 baseline (H3) y su version Gate 4.2 (H5).

Descripcion: histograma global de relaciones de pitch entre pares de notas MIDI.

Leccion: ya fue probado en Gate 4.2; queda como referencia historica y no es prioridad inmediata de rerun en Gate 5A.

### 17. D2 - Enriched 3-channel histogram (MIDI)

Origen: heredero de H4/H6.

Descripcion: extension de D1 con canales de velocity y duration.

Leccion esperada: medir si el contexto expresivo hace mas util el canal de ratios.

### 18. D3 - Temporal-rhythmic descriptor (MIDI)

Origen: evolucion de la linea temporal H2.

Descripcion: mezcla IOI ratios, duration ratios e intervalos locales de pitch.

Leccion esperada: distinguir si la senal proporcional util vive mas en ritmo que en altura.

### 19. D4 - Local interval input augmentation (MIDI)

Origen: mecanismo de Gate 4.2.

Descripcion: inyeccion local por nota en el input del encoder MIDI (en lugar de branch auxiliar).

Leccion: descriptor bisagra para separar fallo de informacion vs fallo de mecanismo de inyeccion. En Gate 4.2 cerró con `S_best=64.2%` (3ep y 8ep) y `hard_neg_best=91.6%`, confirmando estabilidad y techo en este regimen.

### 20. A4 - Audio local log-frequency deltas (Audio)

Origen: espejo conceptual de D4 en el encoder de audio.

Descripcion: features locales de cambio espectral frame a frame, inyectadas del lado audio.

Leccion esperada: medir si el patron "input augmentation" funciona tambien en el dominio continuo del audio.

### 21. A7 - Rational-Attractor descriptor (Audio)

Origen: propuesta nueva para testear directamente la hipotesis fundacional Phideus.

Descripcion: distribucion de cercania de ratios observados en audio a razones simples (2/1, 3/2, 4/3, 5/4, 7/4, etc.) con asignacion suave.

Leccion esperada: validar si la estructura de armonia natural aporta senal alineable en retrieval cross-modal.

### 22. D4+A4 - Dual temperado + audio local

Origen: combinacion factorial de Gate 4.3.

Descripcion: inyeccion simultanea de descriptor MIDI local (D4) y audio local (A4).

Leccion esperada: detectar sinergia entre informacion intervalica discreta y continua.

### 23. D4+A7 - Dual temperado + rational attractor

Origen: combinacion factorial de Gate 4.3.

Descripcion: conserva D4 en MIDI y agrega attractores racionales del lado audio.

Leccion esperada: testear si el dual side produce mejora superior al caso MIDI-only.

---

## Parte III - Barrido ampliado Gate 5A

Tras la decisión de arquitectura, Gate 4.4 concentra **Third Tower + FiLM + MoE** y el barrido descriptorial pasa a Gate 5A.

### 24. D5 - Event-Language Dense (MIDI)

Origen: recuperacion densificada de Route A (event-based), sin hash exacto.

Descripcion: codifica eventos en representacion continua y entrenable.

### 25. D6 - Soft-Constellation Dense (MIDI)

Origen: recuperacion densificada de Route B, migrada a descriptor continuo.

Descripcion: constellations suavizadas en espacio de `log_ratio` y `delta_t`.

### 26. D7 - Soft-Hash Sketch (MIDI)

Origen: reconversion entrenable de la familia hash K0-K2.

Descripcion: sketch continuo tipo count-min en lugar de voting exacto.

### 27. D8 - Ratio Motif n-gram (MIDI)

Origen: extension sintactica de ratios consecutivos.

Descripcion: n-gramas (bi/tri) de motivos de ratio con binning suave.

### 28. D9 - Ratio Graph Topology (MIDI)

Origen: expansion a estructura relacional de alto nivel.

Descripcion: grafo de notas por relaciones de ratio y estadisticos topologicos.

### 29. D10 - Multi-Scale Temporal Pyramid (MIDI)

Origen: consolidacion temporal multi-escala de la linea H2.

Descripcion: histogramas por ventanas de distintas escalas fusionados en un descriptor unico.

### 30. A1 - Histograma global de ratios espectrales (Audio)

Origen: espejo audio de la logica D1.

Descripcion: resumen global de relaciones espectrales en dominio continuo.

### 31. A2 - Histograma enriquecido por energia/estabilidad (Audio)

Origen: espejo audio de la logica D2.

Descripcion: agrega contexto energetico y de estabilidad al histograma espectral.

### 32. A3 - Ratios temporales de audio (Audio)

Origen: espejo audio de la logica D3.

Descripcion: ratios sobre onsets y duraciones espectrales en eje temporal.

### 33. A5 - Soft-Constellation denso (Audio)

Origen: recuperacion de la escuela constellation en version diferenciable para audio.

Descripcion: tokens espectrales anclados, convertidos a representacion continua.

### 34. A6 - Soft-Hash sketch espectral (Audio)

Origen: recuperacion de hash/fingerprint en formato continuo para entrenamiento.

Descripcion: sketch espectral suave, sin matching exacto.

---

## Estado vivo al corte (2026-02-22)

| Descriptor | Rama | Estado operativo |
|---|---|---|
| D0 | Control | Referencia de comparación (S=60.2% en Gate 4.3) |
| D1 | MIDI | Ya evaluado en Gate 4.2 |
| D2 | MIDI | Planificado Gate 5A |
| D3 | MIDI | Prioridad Gate 5A |
| D4 | MIDI | Cerrado en Gate 4.2 (`S_best=64.2%`) |
| D4x / D4r | MIDI | Probados en Gate 4.3 (reverse > regular) |
| D5-D10 | MIDI | Planificados Gate 5A (barrido) |
| A4 | Audio | Brazo robusto (`S=63.6%`) |
| A4x / A4r | Audio | Probados en Gate 4.3 (`A4r` mejor single-descriptor) |
| A7 | Audio | Probado (`S=58.8%`, no supera baseline) |
| A8 / A9 | Audio | Probados en Gate 4.3 Fase 5 (no superan baseline) |
| A1/A2/A3/A5/A6 | Audio | Planificados Gate 5A |
| D4+A4 | Dual | Mejor brazo 5ep (`S=69.8%`) |
| d4a4-scratch | Dual | Completado 30ep (`S=83.6%`, multi-seed `84.1% +/- 2.3pp`) |
| d4a4-60ep | Dual | Gate 4.5 completado (`S=83.8%`) |
| a4r-60ep | Audio | Gate 4.5 completado (`S=79.4%`) |
| t3-wt-50ep-hold | Third Tower | Gate 4.5 completado (`S=81.2%`) |
| d4a4cm | Dual | Probado, degradante (`S=52.4%`) |

---

## Cierre

La lectura actual del catalogo es simple:

1. El núcleo descriptorial robusto de la etapa fue `D4 + A4`.
2. Reverse cross-attention abrió una alternativa fuerte de mecanismo (`A4r`, `D4r`).
3. El mejor resultado global llegó con dual same-modality en training largo (`d4a4-scratch`).
4. Gate 4.5 abre la capa de preguntas de scheduler antes de Gate 5A/5B.

Con esta estructura, el catálogo preserva trazabilidad histórica sin perder sincronía con el estado operativo real del frente.
