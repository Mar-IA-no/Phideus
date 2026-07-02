# Roadmap general — Voz Expresiva Phideus

> Documento estructural del frente. No contiene receta de pipeline ni umbrales numéricos: las decisiones de implementación se congelan en el plan mode de cada fase.

## §1 Identidad del frente

**Qué es.** Un programa de pruebas piloto que aplica el patrón Phideus (extracción de descriptores ratio-based + mecanismos de inyección sobre encoder preentrenado + medición de cambio geométrico/funcional) al dominio de la expresión vocal y los correlatos paralingüísticos/afectivos del habla.

**Qué no es.** No es un nuevo sistema SER competitivo. No es una teoría de la emoción. No es una continuación directa del Escalón 2 (Speech↔EGG cerró NULL en su pregunta original; este frente plantea preguntas distintas). No es un compromiso con la nomenclatura EIR/EMR — esa hipótesis quedó bajada a sub-pregunta interna, no a marca maestra.

**Hipótesis de trabajo (calibradas, ninguna declarada cierta).**
- **H1** — Los descriptores ratio-based (voice quality + formant ratios + dinámica F0) tienen señal discriminativa en tareas paralingüísticas controladas (ESD).
- **H2** — Inyectar esos descriptores en un encoder SSL preentrenado vía concat/FiLM/xattn produce alguno de: (a) mejora de performance, (b) reorganización geométrica del latente, (c) handles interpretables que SSL solo no da.
- **H3** — El efecto sostiene en datos naturalistas (MSP-Podcast).
- **H4** — En datos donde voz y EGG están sincronizados (Lombard), parte de la señal expresiva correlaciona mejor con descriptores derivados de EGG que con descriptores derivados de audio.

**Calibración retórica del mecanismo.** El patrón "descriptor + inyección" mostró valor en otro dominio oscilatorio (música, Escalón 1, paper arXiv:2604.10283). Por eso merece ser probado en voz con bajo costo. La transferencia es plausible, **no garantizada**. Un negativo en este frente debe leerse como información sobre el alcance del mecanismo, no como fracaso del programa Phideus.

## §2 Origen documental

- **Investigación bibliográfica de junio 2026** — referencia editorial externa al repo Phideus, en `editorial-altermundi/Biblioteca/analisis-carga-emocional-del-habla/`. Cubre el campo SER moderno (SSL backbones, OSS stack, datasets, métricas, problemas metodológicos), el substrato físico de la voz (source-filter, voice quality ratios, EGG, fisiología cross-modal) y la representación/generación expresiva (disentanglement, voice conversion, expressive TTS, speech LLMs).
- **Antecedente Phideus** — paper arXiv:2604.10283 (Escalón 1, Audio↔MIDI MAESTRO).
- **Antecedente reciente** — `../EIR-EMR/` (preservado como exploratorio, no activo).

## §3 Estado del frente

**Frente ya propagado a la capa troncal mínima.** La apertura conceptual ya no está sólo en su carpeta local: tras el cierre conjunto de `Fase 0A` y `Fase 0B`, el frente ya quedó reflejado también en `00_TRONCAL/bitacora_desarrollo.md`, `INDICE_DOCUMENTACION.md` y `Proyecto_Estado_Actual.md`.

**Estado empírico al corte 2026-07-02.**
- `Fase 0A` cerró con señal descriptorial específica frente al control no-ratio.
- `Fase 0B` cerró con lectura dual:
  - `N-strict`: sin validación fuerte todavía en speaker-independent estricto;
  - `N-adapt`: especificidad ratio convincente frente al control `C`, con mejora pequeña sobre `eGeMAPS`.
- `Fase 1` sobre `ESD` English ya también cerró:
  - `WavLM-only` levantó con claridad el techo de `N-strict` (`UAR=0.698`);
  - `concat` aportó sobre baseline bajo generalización honesta (`+0.039`, `CI95=[+0.019,+0.060]`);
  - `FiLM` y `xattn` quedaron positivos pero no robustos en `N-strict`;
  - en `N-adapt`, los tres mecanismos mejoraron de forma robusta;
  - la métrica `CKA` dejó una disociación útil entre mejora funcional con reorganización fuerte (`concat`, `xattn`) y mejora funcional con geometría cercana al baseline (`FiLM`).
- la réplica `ZH` ya no es deuda metodológica sino resultado consolidado dentro del cierre cross-language:
  - en **`N-adapt`**, `concat` y `FiLM` replican limpio entre `EN` y `ZH`, con shifts centrados en `0`;
  - en **`N-strict`**, el lift inglés no transfiere: `concat` queda cerca de nulo y `film/xattn` se vuelven negativos en `ZH`;
  - el caveat `0A ZH` (`A/C=0.69` vs `2.88` en EN) queda absorbido como parte de la interpretación, no como deuda abierta.

La pregunta viva del Carril A ya no es si esa lectura **sobrevive al cierre analítico EN↔ZH**: eso ya fue respondido. La pregunta siguiente pasa a ser qué hacer con esa disociación: profundizar `ESD` con una `Fase 1.2`, o mover el frente al régimen naturalístico (`MSP-Podcast`) sabiendo que el positivo cross-language existe, pero solo bajo `N-adapt`.

## §4 Dos carriles del frente

El frente se organiza en dos líneas programáticas distintas, con función, ritmo y riesgo distintos.

| Carril | Función programática | Ritmo | Riesgo | Fases |
|---|---|---|---|---|
| **A — Entrada** | Aterrizar piloto rápido y barato; dar primer dato concreto sobre si el patrón Phideus tiene tracción en voz | Semanas | Bajo | 0A → 0B → 1 → 3 |
| **B — Diferenciación** | Donde Phideus se distingue del resto del campo SER: correlatos físicos voz↔EGG, posible reapertura cross-modal con descriptores validados | Mediano | Medio | 2 (paralela a A) → 4 (deferred) |

**Carril A** responde "¿el patrón Phideus tiene tracción en voz?". **Carril B** responde "¿la expresión vocal admite una lectura ratio-based clínicamente sustentada que el campo SER no está aprovechando?". Pueden correr en paralelo. Cada uno tiene sus propios criterios de éxito.

## §5 Stack de referencia (orientativo, no congelado)

Los siguientes elementos se proponen como punto de partida. La receta específica de cada fase se cierra en el plan mode correspondiente.

**Datasets candidatos.**
- ESD (Emotional Speech Dataset): mismas frases × mismo hablante × 5 emociones × 10 EN + 10 ZH. Sandbox primario para Carril A.
- French Lombard v1.1: ya en el repo Phideus, voz + EGG sincronizado. Sandbox primario para Carril B.
- MSP-Podcast: 400h+ naturalístico con anotaciones VAD continuas (valence/arousal/dominance) y Common Licenses redistribuibles. Sandbox de generalización para Fase 3.

**Backbones y herramientas candidatos.**
- WavLM (Microsoft, paralingüístico) como encoder SSL preentrenado.
- openSMILE con feature set eGeMAPS para descriptores prosódicos clínicos.
- parselmouth (wrapper Python de Praat) para CPP y voice quality complementaria.
- SpeechBrain para pipeline de fine-tuning.
- pyannote.audio para diarización si Fase 3 entra en escenarios conversacionales.

**Familia de descriptores propuesta.** Vector ratio-based con tres sub-familias:
- *Tilt espectral / voice quality*: H1-H2, H1-A1, H1-A3, HNR, CPP, jitter, shimmer.
- *Formantes*: ratios F2/F1, F3/F1 (VTL-invariantes).
- *Dinámica F0*: z-scores intra-frase, ΔF0, alpha-ratio, energy delta.

Dimensión, orden exacto y composición final se cierran en plan mode Fase 0A.

**Métricas.**
- *Primarias*: UAR para clasificación categórica, CCC para regresión VAD continua.
- *Secundarias*: silhouette, CKA (reorganización geométrica), attribution gap respecto a baselines descriptor-only.

CKA es métrica secundaria, no condición de entrada del frente.

## §6 Fases del roadmap

Cada fase decide su propio GO/NO-GO al cerrarse. Los criterios listados son **de lectura esperados**, sin umbrales numéricos congelados — el GO/NO-GO formal lo decide el usuario al cierre de cada fase, con plan mode propio antes de arrancarla.

### Fase 0A — Descriptor extraction + visualización (Carril A)

**Pregunta del frente**: ¿Los descriptores ratio-based muestran separabilidad sobre ESD por sí solos, antes de meter SSL o clasificador?

**Inputs**: ESD raw, sin training.

**Outputs**: descriptores ratio-based + eGeMAPS extraídos por frame y por utterance; plots de PCA/UMAP coloreados por emoción y por hablante; boxplots por descriptor; métricas exploratorias (silhouette por emoción, descomposición de varianza intra-hablante vs inter-hablante).

**Criterios de lectura esperados**: separabilidad visual interpretable entre emociones en algún subconjunto de descriptores; señal mayor intra-hablante (entre emociones) que inter-hablante (entre personas) en al menos parte del vector. Ausencia de separabilidad es información válida también.

**Costo aprox**: CPU solamente, 1-2 días.

**Dependencias**: ninguna; es la fase de arranque.

### Fase 0B — Baselines clásicos descriptor-only (Carril A)

**Pregunta del frente**: Antes de meter SSL, ¿los descriptores tienen señal en un clasificador clásico chico (SVM, LogReg)?

**Inputs**: descriptores extraídos en 0A + splits speaker-independent.

**Outputs**: UAR de dos baselines clásicos: eGeMAPS → clasificador chico; Phideus-ratios → clasificador chico. Curvas de aprendizaje, intervalos de confianza, análisis de error.

**Criterios de lectura esperados**: ambos baselines deberían superar majority class; si Phideus-ratios supera o iguala a eGeMAPS, la familia ratio propuesta tiene señal complementaria; si queda muy por debajo, conviene revisar composición antes de Fase 1.

**Costo aprox**: CPU solamente, 1-2 días.

**Dependencias**: requiere Fase 0A cerrada.

### Fase 1 — SSL injection (Carril A)

**Pregunta del frente**: Inyectar los descriptores en WavLM vía concat/FiLM/xattn, ¿aporta algo (perf y/o geometría) sobre baselines?

**Inputs**: WavLM frozen + ESD + descriptores Phideus + (opcional) emotion2vec embeddings como segundo descriptor de comparación.

**Outputs**: baseline `WavLM-only` y tres mecanismos homogéneos de inyección de la familia `A` (`concat`, `FiLM`, `xattn`) con la misma receta multi-seed. UAR primaria, CKA y análisis comparativo por mecanismo como lectura secundaria.

**Criterios de lectura esperados**: separar el efecto del descriptor (vs `WavLM-only`) y el efecto del mecanismo (comparación directa `concat/FiLM/xattn` bajo la misma plantilla). Resultado positivo, negativo o mixto son todos información válida con interpretación distinta.

**Estado al corte**: el bloque `ESD` ya quedó empíricamente cerrado también en su lectura translingüística mínima. `concat` había sido el único mecanismo que pasó robustamente el contraste primario en `N-strict` sobre `EN`, pero ese positive no replica limpiamente en `ZH`; en cambio, `concat` y `FiLM` sí replican con claridad en `N-adapt`, mientras `xattn` queda más débil en mandarín.

**Costo aprox**: GPU, primera pasada ya ejecutada en ~`6.9 h` wall-clock sobre RTX 3090 gracias al precache de `WavLM` y de la familia `A`.

**Dependencias**: requiere Fase 0B cerrada con señal mínima.

**Siguiente cierre dentro del mismo bloque**: ya no faltan artifacts para leer `EN ↔ ZH`. Lo que sigue es una decisión de programa: cerrar Fase 1 con esta lectura acotada, abrir una `Fase 1.2` para atacar el cuello de `N-strict`, o saltar a `MSP-Podcast`/Carril B`.

### Fase 2 — Correlatos físicos voz↔EGG en Lombard (Carril B, paralela a Fase 1)

**Pregunta del frente**: ¿Qué parte de la señal expresiva correlaciona mejor con descriptores derivados de EGG que con descriptores derivados de audio?

**Inputs**: French Lombard v1.1 (ya en el repo, voz + EGG sincronizado).

**Outputs**: vector de descriptores audio (ratio-based del lado filtrado) + vector de descriptores EGG (CQ, OQ, SQ + ratios voice quality computados sobre el contacto glotal). Correlaciones por hablante, por condición de Lombard, con controles estadísticos.

**Criterios de lectura esperados**: detectar componentes de la señal expresiva con correlato más limpio del lado fuente (EGG); detectar componentes que se mantienen del lado audio; o detectar resultado mixto. La pregunta admite respuesta parcial.

**Costo aprox**: CPU + GPU light, 3-5 días.

**Dependencias**: ninguna respecto a Carril A; puede correr en paralelo a Fase 1. Plan mode propio antes de arrancarla.

### Fase 3 — Generalización naturalística MSP-Podcast (Carril A)

**Pregunta del frente**: ¿El efecto observado en ESD sostiene en habla naturalística con anotaciones VAD continuas?

**Inputs**: MSP-Podcast (acceso via paper de UT Dallas).

**Outputs**: regresión VAD con descriptores Phideus inyectados sobre WavLM; cross-corpus (train ESD eval MSP, train MSP eval ESD); métrica CCC.

**Criterios de lectura esperados**: si el efecto se sostiene, valida transferencia entre actuado y naturalístico; si se queda en ESD, acota el alcance al régimen actuado.

**Costo aprox**: GPU, 2-3 semanas.

**Dependencias**: requiere Fase 1 cerrada con GO; plan mode propio.

### Fase 4 — Reapertura Speech↔EGG retrieval con descriptores validados (Carril B, deferred)

**Pregunta del frente**: Con descriptores validados en Fases 1+2+3, ¿se puede reabrir el experimento cross-modal Speech↔EGG que cerró NULL en Escalón 2 con resultado distinto?

**Estado**: deferred. La decisión de abrirla espera resultados consolidados de las fases 1, 2 y 3. Antes de arrancarla, plan mode propio que justifique qué hizo cambiar el contexto respecto al cierre original de Escalón 2.

## §7 Lo que NO se congela en este roadmap

Para evitar que supuestos de conveniencia se conviertan en doctrina del frente, este roadmap **deja explícitamente abiertas**:

- **Política de normalización** (per-speaker, per-utterance, global, z-score, robusto) — se cierra en plan mode Fase 0A.
- **Política de alineación temporal** entre frames openSMILE/Praat y frames WavLM — se cierra en plan mode Fase 0A o 0B según donde se necesite primero.
- **Número mínimo de seeds** — se cierra en plan mode Fase 1.
- **Composición exacta del vector descriptor** (qué dimensión final, qué orden, qué sub-familias incluir) — se propone direccionalmente en §5 pero se cierra en plan mode Fase 0A.
- **Umbrales numéricos de GO/NO-GO** — se fijan en el plan mode de cada fase, no a priori.
- **Clasificador específico para Fase 0B** (SVM RBF, LogReg, MLP chico, gradient boosting) — se cierra en plan mode Fase 0B.

## §8 Riesgos epistemológicos

1. **Sobreleer un negativo como fracaso del mecanismo Phideus**. El mecanismo se validó en otro dominio (música). Un negativo en voz es información sobre el alcance del mecanismo en este dominio específico, no sobre el mecanismo en general. Mitigación: nombrar siempre dominio + tarea testeada en los reportes.
2. **Techo SSL**. WavLM-large + head simple puede ya capturar implícitamente lo que H1-H2 y CPP aportan. Si pasa eso, los ratios no agregan sobre SSL en performance — pero pueden seguir aportando en interpretabilidad o reorganización geométrica. Mitigación: incluir métricas secundarias además de UAR/CCC desde el día 1.
3. **Speaker leakage**. Splits flojos hacen que el modelo aprenda identidad en lugar de afecto. Mitigación: política de split explícita en plan mode de cada fase, siempre speaker-independent.
4. **Acted vs naturalistic**. ESD es actuado; si los ratios discriminan en ESD pero no en MSP-Podcast, el frente cae en el mismo problema metodológico del campo entero. Mitigación: Fase 3 explícita para chequear este punto.

## §9 Lo que el frente NO va a ser

- **No es claim sobre emoción auténtica**. Ni sobre "comprensión" emocional.
- **No es competidor de Hume/Ellipsis/Kintsugi**. No vamos por mercado de voice emotional AI ni por biomarcadores clínicos.
- **No es teoría de la afectividad**. No promete una taxonomía de afectos.
- **No es continuación directa de Escalón 2** (Lombard cerró NULL). Las preguntas del Carril B son distintas; Fase 4 es deferred y requiere su propia justificación.
- **No es EIR-EMR**. Esa nomenclatura quedó como antecedente exploratorio. Si una distinción tipo "invariante vs modulado" emerge empíricamente más adelante, ahí se decide si vale rebautizarla.

## §10 Próximo paso

La pregunta operativa correcta del siguiente corte ya no es abrir `Fase 1` ni cerrar `EN ↔ ZH`. Ese bloque ya quedó resuelto. El siguiente conjunto de preguntas es otro:

1. ¿Conviene abrir una `Fase 1.2` que apunte específicamente al régimen `N-strict`, dado que el positivo cross-language quedó acotado a `N-adapt`?
2. ¿El frente gana más moviéndose a `MSP-Podcast` para chequear naturalización del efecto, o afinando primero el régimen actuado donde apareció la disociación?
3. ¿La lectura correcta del descriptor en voz debe seguir centrada en “speaker-independent emotion SER”, o el lugar donde el patrón muestra hoy más tracción es precisamente el régimen con anclaje mínimo por hablante?

Las fases `0A`, `0B`, `1 EN`, la réplica `ZH` y el cierre cross-language ya ocurrieron. Lo que sigue ya no es sumar training ciego ni tratar `ZH` como deuda, sino decidir si esta transferencia parcial alcanza como cierre de `Fase 1` o si merece una iteración adicional antes del salto naturalístico.
