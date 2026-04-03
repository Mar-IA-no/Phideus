│ Plan: Revisión Epistemológica y Rectificación del Roadmap de Escalón 2                                                                                   │
│                                                                                                                                                          │
│ [!IMPORTANT]                                                                                                                                             │
│ **Addendum operativo (2026-03-15):** este documento conserva la revisión epistemológica que ordenó `S2-P2.5`, pero ya no debe leerse como estado vivo   │
│ del frente. `P2.5` y `P2.5b` quedaron cerrados como **null mecanístico inicial** (`12/12` condiciones ≈ `D0` o peor), y el paso siguiente ya quedó      │
│ **implementado/en ejecución** como `S2-P3`, con encoder speech frozen tipo `WavLM-Large` y diagnóstico comparativo `P2 vs P3` todavía pendiente. El     │
│ estado canónico actual vive en                                                                                                                            │
│ `README.md` y `ROADMAP_ESCALON_2.md`.                                                                                                                     │
│                                                                                                                                                          │
│ Context                                                                                                                                                  │
│                                                                                                                                                          │
│ Escalón 2 (Speech ↔ EGG) existe para responder la pregunta que Escalón 1 (Audio ↔ MIDI) no respondió limpiamente: si la armonía natural — razones        │
│ lineales de frecuencia, serie armónica física, regularidades del oscilador — constituye una estructura informacional privilegiada para la alineación     │
│ cross-modal, distinta de descriptores espectrales genéricos y de codificaciones perceptuales/logarítmicas.                                               │
│                                                                                                                                                          │
│ Escalón 1 estableció que la intervención descriptor-guided funciona (d4a4=84.1% como referencia eval-seed, +9.4pp causal, +82% CKA). Pero sus descriptores │
│ envolvente espectral; D4: intervalos MIDI en log2/semitonos) no son de armonía natural en sentido fuerte. A4 es espectral-genérico (Familia C). D4 opera │
│  sobre representaciones perceptualmente mediadas (Familia D). Escalón 1 validó la mecánica. No validó la ontología.                                      │
│                                                                                                                                                          │
│ Este plan es una auditoría de segundo orden del frente entero de Escalón 2. No invalida ni bloquea lo que está corriendo (S2-P2.5). Impone claridad      │
│ epistemológica sobre lo que cada resultado significa.                                                                                                    │
│                                                                                                                                                          │
│ Estado experimental al momento de esta revisión:                                                                                                         │
│ - S2-P0: COMPLETO. French Lombard v1.1, protocolo congelado.                                                                                             │
│ - S2-P1: COMPLETO. CCA S=64.4%.                                                                                                                          │
│ - S2-P2-control (D0): COMPLETO. S=77.8% @ ep25.                                                                                                          │
│ - S2-P2-main (concatenación): COMPLETO. V4-lin=-10pp, H-series=-18pp (colapso ep8), A4-16k=+0pp.                                                         │
│ - S2-P2.5 / P2.5b (atención + conditioned projection): CERRADO. Null mecanístico inicial: 12/12 condiciones ≈ D0 o peor.                                │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ 1. Qué queda vigente del Escalón 2 actual                                                                                                                │
│                                                                                                                                                          │
│ 1.1 El dominio (Speech ↔ EGG) es correcto. Dos sensores físicos del mismo oscilador, F0 continuo, registro simultáneo, sin mediación simbólica. La arena │
│  correcta para la tesis fuerte.                                                                                                                          │
│                                                                                                                                                          │
│ 1.2 El protocolo (S2-P0) está correctamente congelado. sr=16kHz, seg=2s, hop=0.5s, pool=128, S=min(S2E,E2S)@10, CI agrupado por hablante, split por      │
│ hablante, seed=42.                                                                                                                                       │
│                                                                                                                                                          │
│ 1.3 La progresión de baselines (CCA → D0 → descriptores) es sólida. Cada fase bloquea la siguiente.                                                      │
│                                                                                                                                                          │
│ 1.4 D0 como baseline simétrico es epistemológicamente correcto para Fase 1. Encoders simples, simétricos, from-scratch. La simplicidad es un requisito   │
│ para interpretabilidad.                                                                                                                                  │
│                                                                                                                                                          │
│ 1.5 La extracción F0 per-modalidad es correcta. PYIN para speech, autocorrelación para EGG, sin leakage cross-modal.                                     │
│                                                                                                                                                          │
│ 1.6 El fracaso de la concatenación es un resultado negativo informativo sobre mecanismo. Invalida la concatenación como mecanismo, no los descriptores   │
│ como contenido. La misma evidencia en Escalón 1 (a4r +5.5pp con cross-attention) soporta esta lectura.                                                   │
│                                                                                                                                                          │
│ 1.7 La transición a inyección atencional (S2-P2.5) está bien motivada. Descriptores como principios organizacionales (modulación de atención), no como   │
│ contenido (augmentación de features).                                                                                                                    │
│                                                                                                                                                          │
│ 1.8 El matching asimétrico descriptor-mecanismo en P2.5 es correcto. V4-lin (temporal, inter-frame) → attention bias. H-series (intra-frame, armónico) → │
│  cross-attention. No es arbitrario: cada descriptor recibe el mecanismo que corresponde a su naturaleza informacional.                                   │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ 2. Qué debe rectificarse (razones epistemológicas)                                                                                                       │
│                                                                                                                                                          │
│ 2.1 V4-lin ha sido tácitamente elevado a "descriptor de armonía natural" cuando no testea la tesis fuerte                                                │
│                                                                                                                                                          │
│ V4-lin mide F0[t]/F0[t-1] — dinámica temporal del oscilador. Es "natural" en el sentido de que usa ratios lineales (no log2) y captura una magnitud      │
│ física. Pero NO es la serie armónica. No mide H2/H1, H3/H1. No mide la estructura armónica interna de la señal.                                          │
│                                                                                                                                                          │
│ La tesis fuerte de HIT dice: "la estructura de la serie armónica física es un organizador privilegiado." V4-lin testea una tesis más débil y distinta:   │
│ "la dinámica del oscilador contiene invariantes cross-modales privilegiados."                                                                            │
│                                                                                                                                                          │
│ Rectificación: Toda la documentación y narrativa debe posicionar V4-lin como Familia A (dinámica del oscilador), nunca como proxy de "armonía natural"   │
│ en sentido fuerte. Si V4-lin tiene éxito, la conclusión correcta es sobre dinámica del oscilador, no sobre la serie armónica.                            │
│                                                                                                                                                          │
│ 2.2 H-series ha sido sub-priorizado respecto a su importancia epistemológica                                                                             │
│                                                                                                                                                          │
│ H-series (H2/H1..H6/H1, concentración armónica, desviación armónica) es el descriptor más directamente alineado con la tesis fuerte de HIT. Mide la      │
│ estructura de la serie armónica física — el objeto central de la Harmonic Information Theory. Sin embargo:                                               │
│ - En S2-P2-main, el colapso de H-series se trató como falla del descriptor, no del mecanismo.                                                            │
│ - En S2-P2.5, V4-lin corre primero; H-series todavía no corrió.                                                                                          │
│ - La matriz experimental tiene 3 arms pero H-series, el más alineado con la tesis fuerte, está detrás de V4-lin en la cola.                              │
│                                                                                                                                                          │
│ Rectificación: H-series debe ser explícitamente designado como el test más directamente alineado con la tesis fuerte de HIT disponible dentro de Escalón │
│  2. Sus resultados tienen más peso epistemológico que los de V4-lin para la pregunta central del programa. Sin embargo, un resultado negativo de         │
│ H-series no falsifica automáticamente HIT — puede estar fallando la familia concreta de descriptores (H2/H1..H6/H1), el mecanismo de inyección           │
│ (content-only cross-attention), la asimetría speech/EGG no manejada, o la adecuación del encoder simétrico de Fase 1. H-series es el mejor probe         │
│ disponible, no un tribunal supremo.                                                                                                                      │
│                                                                                                                                                          │
│ 2.3 La resolución temporal de 10ms para descriptores F0 es una asunción no justificada                                                                   │
│                                                                                                                                                          │
│ hop_length=160 (10ms @ 16kHz) → 201 frames por 2s. Adoptado del default de PYIN/STFT sin justificación explícita de que 10ms sea suficiente para         │
│ capturar la dinámica de ratios que V4-lin y H-series necesitan.                                                                                          │
│                                                                                                                                                          │
│ Rectificación: Documentar como asunción explícita: "10ms es suficiente para dinámica macro-prosódica. Dinámica sub-ciclo requeriría diferente resolución │
│  y representa otra hipótesis." Si H-series o V4-lin fallan, la resolución temporal debe listarse como confound a investigar.                             │
│                                                                                                                                                          │
│ 2.4 La distinción "natural vs perceptual" no es tan limpia para V4-lin como se presenta                                                                  │
│                                                                                                                                                          │
│ V4-lin usa F0[t]/F0[t-1] (lineal). V4-log usa log2(F0[t]/F0[t-1]). La relación es monótona — una red con suficiente capacidad puede transformar una en   │
│ la otra. La diferencia no es de contenido informacional sino de sesgo representacional: linear trata un salto 2:1 como el doble de un salto 3:2; log2    │
│ los trata más simétricamente. Este sesgo, aunque no testea la ontología fuerte de la serie armónica, puede ser sustantivo para una red con capacidad     │
│ limitada — la parametrización determina qué relaciones son linealmente accesibles y cuáles requieren aproximación no-lineal.                             │
│                                                                                                                                                          │
│ Rectificación: V4-lin vs V4-log no testea "armonía natural vs armonía perceptual" en sentido fuerte. Testea un sesgo representacional relevante sobre    │
│ cómo parametrizar ratios temporales de F0 — secundario respecto a HIT fuerte, pero no trivial. El test fuerte natural-vs-perceptual pertenece a H-series │
│  (ratios de amplitudes armónicas físicas) vs A4-16k (dinámica espectral sin ratios).                                                                     │
│                                                                                                                                                          │
│ 2.5 Falta una definición operativa falsificable de "armonía natural"                                                                                     │
│                                                                                                                                                          │
│ ¿Qué resultados específicos constituirían evidencia A FAVOR de la tesis fuerte, y cuáles evidencia EN CONTRA? Sin una matriz de predicciones             │
│ pre-registrada, existe riesgo de racionalización post-hoc.                                                                                               │
│                                                                                                                                                          │
│ Rectificación: Crear una matriz de predicciones explícita para S2-P2.5 ANTES de que H-series-xattn y A4-16k-xattn produzcan resultados.                  │
│                                                                                                                                                          │
│ Regla operativa para comparaciones bajo incertidumbre:                                                                                                   │
│ El protocolo canónico de Escalón 2 usa S = min(S2E, E2S) con CI grouped bootstrap por hablante. Para que la matriz de predicciones sea falsificable, se  │
│ fija la siguiente regla:                                                                                                                                 │
│                                                                                                                                                          │
│ Δ = S_A - S_B se computa con grouped bootstrap pareado sobre la diferencia: en cada iteración bootstrap se computa S para ambos modelos sobre el mismo   │
│ resample de hablantes, y se registra la diferencia. Esto produce un CI directamente sobre Δ.                                                             │
│                                                                                                                                                          │
│ A > B se declara cuando: (1) Δ_point >= 2pp, Y (2) CI_Δ excluye 0. Ambas condiciones deben cumplirse.                                                    │
│                                                                                                                                                          │
│ A ≈ B (indistinguibles bajo esta configuración) se declara cuando CI_Δ contiene 0, o Δ_point < 2pp.                                                      │
│                                                                                                                                                          │
│ Esta regla previene que múltiples celdas de la matriz sean compatibles con el mismo resultado empírico. Comparar CIs individuales de A y B no es         │
│ equivalente — el bootstrap pareado captura la correlación entre modelos evaluados sobre los mismos datos.                                                │
│                                                                                                                                                          │
│ ┌───────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────┐                   │
│ │           Patrón de resultados            │                             Interpretación epistemológica                              │                   │
│ ├───────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤                   │
│ │ H-series-xattn > D0 > A4-16k-xattn        │ Evidencia fuerte para HIT: la estructura armónica es específicamente privilegiada      │                   │
│ ├───────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤                   │
│ │ H-series-xattn > A4-16k-xattn > D0        │ Evidencia para HIT, pero el mecanismo atencional también ayuda genéricamente           │                   │
│ ├───────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤                   │
│ │ A4-16k-xattn >= H-series-xattn > D0       │ El mecanismo atencional ayuda, pero la estructura armónica NO es privilegiada          │                   │
│ ├───────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤                   │
│ │ D0 >= todos                               │ Ni el mecanismo ni los descriptores ayudan en esta configuración                       │                   │
│ ├───────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤                   │
│ │ V4-lin-attnbias > D0, H-series-xattn ≈ D0 │ La dinámica del oscilador es útil, pero la serie armónica no (bajo esta configuración) │                   │
│ ├───────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤                   │
│ │ H-series-xattn > D0, V4-lin-attnbias ≈ D0 │ La serie armónica funciona pero la dinámica temporal no (inesperado pero informativo)  │                   │
│ └───────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────┘                   │
│                                                                                                                                                          │
│ Nota: "≈ D0" incluye tanto empate como derrota — lo relevante es la ausencia de mejora significativa bajo la regla operativa.                            │
│                                                                                                                                                          │
│ Alcance de la matriz: Esta es una matriz mínima de patrones ancla, no exhaustiva. Cubre los patrones epistemológicamente más informativos. Combinaciones │
│  reales pueden no encajar limpiamente en una sola celda (e.g., H-series > D0 en S2E pero no en E2S). En esos casos, se reporta el resultado observado y  │
│ se indica qué celdas son parcialmente compatibles, sin forzar una interpretación. La matriz previene racionalización post-hoc para los patrones          │
│ principales; no pretende cubrir toda la superficie de resultados posibles.                                                                               │
│                                                                                                                                                          │
│ 2.6 La simetría de encoders es una asunción no explicitada con consecuencias epistemológicas                                                             │
│                                                                                                                                                          │
│ Speech y EGG son señales físicamente diferentes:                                                                                                         │
│ - Speech: broadband, formantes, 50Hz-8kHz, armónicos modulados por tracto vocal.                                                                         │
│ - EGG: narrowband, cuasi-periódica, dominada por fundamental y armónicos bajos, sin filtro de tracto vocal.                                              │
│                                                                                                                                                          │
│ Encoders simétricos asumen que la misma extracción es igualmente apropiada para ambas modalidades. Si H-series falla, una explicación podría ser que las │
│  CNN simétricas extraen armónicos pobremente de EGG.                                                                                                     │
│                                                                                                                                                          │
│ Rectificación: Declarar encoders simétricos como asunción explícita de Fase 1 (simple, controlable). El roadmap debe identificar Fase 2 (asimetría       │
│ controlada) como paso siguiente si los resultados de Fase 1 son ambiguos.                                                                                │
│                                                                                                                                                          │
│ 2.7 La asimetría de H-series entre speech y EGG está reconocida pero no operacionalmente manejada                                                        │
│                                                                                                                                                          │
│ H2/H1 en speech = fuente glotal + filtro del tracto vocal. H2/H1 en EGG = solo fuente glotal. Son la MISMA computación pero miden cantidades físicas     │
│ DIFERENTES. Si H-series mejora la alineación cross-modal, esto significaría que el componente compartido (la fuente glotal) es suficiente para organizar │
│  la alineación a pesar del confound del tracto vocal en speech. Esto es un resultado MÁS fuerte que simple descriptor matching.                          │
│                                                                                                                                                          │
│ Rectificación: Documentar explícitamente: "Si H-series organiza la alineación Speech↔EGG, demuestra que el componente compartido de la fuente glotal es  │
│ suficiente pese al confound del tracto vocal. Esto fortalece la tesis de que la estructura armónica del oscilador es un invariante cross-modal."         │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ 3. Qué era control metodológico válido pero no test de la tesis fuerte                                                                                   │
│                                                                                                                                                          │
│ ┌────────────────────────────┬───────────────────────────────┬───────────────────────────────────────────────────────────────────────────┐               │
│ │         Componente         │              Rol              │                         Test de la tesis fuerte?                          │               │
│ ├────────────────────────────┼───────────────────────────────┼───────────────────────────────────────────────────────────────────────────┤               │
│ │ A4-16k                     │ Control no-ratio espectral    │ NO. Es el adversario: si iguala o supera a H-series, la tesis se debilita │               │
│ ├────────────────────────────┼───────────────────────────────┼───────────────────────────────────────────────────────────────────────────┤               │
│ │ V4-log                     │ Control paramétrico de V4-lin │ NO. Testea sesgo inductivo, no armonía natural                            │               │
│ ├────────────────────────────┼───────────────────────────────┼───────────────────────────────────────────────────────────────────────────┤               │
│ │ D0                         │ Control neural sin descriptor │ NO. Referencia para todas las comparaciones                               │               │
│ ├────────────────────────────┼───────────────────────────────┼───────────────────────────────────────────────────────────────────────────┤               │
│ │ S2-P2-main (concatenación) │ Control de mecanismo          │ NO. Testea "descriptores como features" (rechazado)                       │               │
│ ├────────────────────────────┼───────────────────────────────┼───────────────────────────────────────────────────────────────────────────┤               │
│ │ Escalón 1 (A4, D4)         │ Validación de la mecánica     │ NO. Descriptores no-naturales en sentido fuerte                           │               │
│ └────────────────────────────┴───────────────────────────────┴───────────────────────────────────────────────────────────────────────────┘               │
│                                                                                                                                                          │
│ H-series es el descriptor más directamente alineado con la tesis fuerte de HIT en Escalón 2, pero su resultado no es juicio final: un null de H-series   │
│ puede deberse al descriptor específico, al mecanismo de inyección, o a confounds de la configuración de Fase 1 (encoders simétricos, asimetría           │
│ speech/EGG). V4-lin testea una tesis adyacente valiosa pero distinta (dinámica del oscilador).                                                           │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ 4. Taxonomía estricta de descriptores                                                                                                                    │
│                                                                                                                                                          │
│ Familia A: Dinámica temporal del oscilador                                                                                                               │
│                                                                                                                                                          │
│ - Qué mide: Cambios en F0 entre frames consecutivos. Ratios lineales frame-to-frame, regularidad del periodo, fortaleza de voicing.                      │
│ - Descriptores: V4-lin (4d). V4-log (4d, control paramétrico).                                                                                           │
│ - Por qué entra: Captura estructura relacional físicamente natural del oscilador vocal. Las transiciones de F0 codifican información prosódica y         │
│ fonológica compartida entre speech y EGG.                                                                                                                │
│ - Qué hipótesis testea: "La dinámica temporal del oscilador, expresada como ratios lineales de frecuencia, contiene invariantes cross-modales que        │
│ mejoran la alineación cuando se inyectan como bias de atención."                                                                                         │
│ - Qué NO testea: NO testea la serie armónica (estructura intra-frame). NO testea si la estructura de sobretonos físicos es un organizador privilegiado.  │
│ Un resultado positivo de V4-lin dice algo sobre dinámica del oscilador, no sobre la serie armónica.                                                      │
│ - Mecanismo P2.5: Attention bias en self-attention del Transformer (bilineal factorizado asimétrico, phi/psi/W).                                         │
│                                                                                                                                                          │
│ Familia B: Estructura armónica natural intra-frame                                                                                                       │
│                                                                                                                                                          │
│ - Qué mide: Relaciones de amplitud entre armónicos de F0 dentro de un frame. H2/H1..H6/H1, concentración armónica, desviación armónica.                  │
│ - Descriptores: H-series (8d). Normalización congelada per-modalidad.                                                                                    │
│ - Por qué entra: Es el descriptor más directamente alineado con la tesis central de HIT. La serie armónica física (múltiplos enteros de F0, con          │
│ amplitudes gobernadas por la forma del pulso glotal) es el objeto central de la Harmonic Information Theory.                                             │
│ - Qué hipótesis testea: "La estructura de la serie armónica física, expresada como ratios de amplitud entre armónicos, es un organizador privilegiado de │
│  información cross-modal." Esta es la tesis fuerte de HIT en su forma experimental más directa dentro de Escalón 2.                                      │
│ - Qué NO testea: NO testea dinámica temporal (Familia A). NO testea si cualquier descriptor auxiliar ayuda (para eso está D0 vs todo lo demás). Testea   │
│ específicamente si la estructura armónica interna — la firma de la serie armónica física — porta información privilegiada.                               │
│ - Hipótesis abierta: H-series captura perfiles armónicos diferentes en speech (fuente + tracto vocal) vs EGG (solo fuente). Si la alineación mejora,     │
│ demuestra que el componente compartido (fuente glotal) es suficiente pese al confound del tracto vocal.                                                  │
│ - Mecanismo P2.5: Cross-attention post-CNN. Q=descriptor (proyectado), K/V=features CNN crudas (sin pos_emb). Permutación-equivariante por diseño: la    │
│ estructura temporal entra después via pos_emb + Transformer.                                                                                             │
│                                                                                                                                                          │
│ Familia C: Controles espectrales no-ratio                                                                                                                │
│                                                                                                                                                          │
│ - Qué mide: Dinámica de energía espectral por bandas, sin referencia a F0 ni ratios armónicos.                                                           │
│ - Descriptores: A4-16k (8d). Deltas temporales de log-magnitud en 8 bandas de octava, z-score por segmento.                                              │
│ - Por qué entra: Como control adversario. Si A4-16k mejora sobre D0, información espectral genérica ayuda. Si H-series o V4-lin mejoran sobre A4-16k, la │
│  mejora es específica de descriptores basados en ratios. Si A4-16k iguala o supera a los descriptores de armonía natural, la tesis "la armonía natural   │
│ es especial" se debilita.                                                                                                                                │
│ - Qué hipótesis testea: "Cualquier descriptor espectral auxiliar mejora la alineación cross-modal" (la hipótesis nula contra la tesis fuerte).           │
│ - Qué NO testea: No testea estructura de ratios. No testea la serie armónica. No testea dinámica del oscilador. Testea si dinámica espectral genérica es │
│  suficiente.                                                                                                                                             │
│ - Caveat: A4-16k es UN control no-ratio específico, no "todos los posibles controles genéricos." Si H-series > A4-16k, la conclusión correcta es "la     │
│ estructura armónica supera este control espectral particular."                                                                                           │
│                                                                                                                                                          │
│ Familia D: Variantes perceptuales/logarítmicas como comparación                                                                                          │
│                                                                                                                                                          │
│ - Qué mide: Las mismas magnitudes físicas que Familia A, pero en coordenadas perceptuales (logarítmicas).                                                │
│ - Descriptores: V4-log (4d). log2(F0[t]/F0[t-1]).                                                                                                        │
│ - Por qué entra: Para testear si la parametrización lineal de ratios F0 da un sesgo inductivo diferente (esperadamente mejor) que la logarítmica.        │
│ - Qué hipótesis testea: "El sistema de coordenadas (lineal vs log2) importa para el sesgo inductivo de descriptores temporales de F0." Hipótesis         │
│ secundaria, metodológica.                                                                                                                                │
│ - Qué NO testea: No testea la serie armónica (Familia B). No testea dinámica espectral genérica (Familia C).                                             │
│ - Prioridad: Secundaria. Solo corre si V4-lin muestra señal en P2.5.                                                                                     │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ 5. Impacto en S2-P2-main/P2.5 (Nivel 1: rectificación local inmediata)                                                                                   │
│                                                                                                                                                          │
│ 5.1 Sin cambios de arquitectura ni de modelos. Las implementaciones de encoders, descriptores e inyección son técnicamente correctas y no se modifican.  │
│                                                                                                                                                          │
│ Sí se requieren cambios reales en evaluación y tooling:                                                                                                  │
│ - Código de evaluación: eval_escalon2.py necesita implementar grouped bootstrap pareado sobre Δ = S_A - S_B para generar CI_Δ directamente. El bootstrap │
│  actual computa CI de S para un solo modelo; el pareado requiere que ambos modelos se evalúen sobre el mismo resample de hablantes en cada iteración.    │
│ - Criterios de lectura: La regla operativa (Δ >= 2pp Y CI_Δ excluye 0) reemplaza la comparación visual de point estimates.                               │
│ - Preregistro: Creación de PREDICCIONES_EPISTEMOLOGICAS_P25.md como artefacto nuevo.                                                                     │
│                                                                                                                                                          │
│ Esto no es solo narrativa — es un cambio de protocolo estadístico con impacto en código.                                                                 │
│                                                                                                                                                          │
│ 5.2 V4-lin-attnbias (corriendo) debe completarse. Sus resultados se interpretarán como test de Familia A (dinámica del oscilador), no como test de la    │
│ tesis fuerte de HIT.                                                                                                                                     │
│                                                                                                                                                          │
│ 5.3 H-series-xattn debe correr siguiente, con framing explícito como test primario de la tesis fuerte. Sea cual sea el resultado de V4-lin, H-series es  │
│ el experimento epistemológicamente prioritario.                                                                                                          │
│                                                                                                                                                          │
│ 5.4 A4-16k-xattn sigue siendo el control. Testea si el mecanismo atencional per se (no el contenido descriptor) explica cualquier mejora.                │
│                                                                                                                                                          │
│ 5.5 Crear la matriz de predicciones (sección 2.5) ANTES de que H-series-xattn y A4-16k-xattn produzcan resultados. Pre-registro interpretativo.          │
│                                                                                                                                                          │
│ 5.6 El reporte de resultados debe usar la taxonomía de familias consistentemente. "V4-lin muestra +Xpp" debe reportarse como "Familia A (dinámica del    │
│ oscilador) muestra +Xpp", no como "armonía natural muestra +Xpp."                                                                                        │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ 6. Impacto en la metodología de Escalón 2 (Nivel 2)                                                                                                      │
│                                                                                                                                                          │
│ 6.1 La narrativa de Escalón 2 requiere reescritura estructural. El ROADMAP y README actuales presentan V4-lin, H-series y A4-16k como lista plana. La    │
│ narrativa corregida los organiza jerárquicamente:                                                                                                        │
│ - H-series como test primario de la tesis fuerte (Familia B)                                                                                             │
│ - V4-lin como test de la tesis sobre dinámica del oscilador (Familia A)                                                                                  │
│ - A4-16k como control adversario (Familia C)                                                                                                             │
│ - V4-log como control de parametrización (Familia D)                                                                                                     │
│                                                                                                                                                          │
│ 6.2 Los resultados de concatenación deben reenmarcarse. De "la concatenación falló" a "la concatenación testeó la hipótesis de que los descriptores      │
│ funcionan como features adicionales. Esa hipótesis fue rechazada. La inyección atencional testea la hipótesis de que funcionan como principios           │
│ organizacionales."                                                                                                                                       │
│                                                                                                                                                          │
│ 6.3 Asunciones implícitas a hacer explícitas:                                                                                                            │
│                                                                                                                                                          │
│ ┌───────────────────────────────────────────────────────────┬───────────────────────────┬─────────────────────────────────────────────────────────────┐  │
│ │                         Asunción                          │       Estado actual       │                           Acción                            │  │
│ ├───────────────────────────────────────────────────────────┼───────────────────────────┼─────────────────────────────────────────────────────────────┤  │
│ │ 10ms hop suficiente para dinámica de ratios               │ Implícita                 │ Documentar como asunción con justificación                  │  │
│ ├───────────────────────────────────────────────────────────┼───────────────────────────┼─────────────────────────────────────────────────────────────┤  │
│ │ PYIN y autocorrelación son estimadores comparables        │ Documentado como confound │ Suficiente, listar como follow-up                           │  │
│ ├───────────────────────────────────────────────────────────┼───────────────────────────┼─────────────────────────────────────────────────────────────┤  │
│ │ F0 per-modalidad previene leakage                         │ Documentado               │ Suficiente                                                  │  │
│ ├───────────────────────────────────────────────────────────┼───────────────────────────┼─────────────────────────────────────────────────────────────┤  │
│ │ Pool=128, k=10 son decisiones de protocolo                │ Congelados sin            │ Documentar como decisiones, no como óptimos                 │  │
│ │                                                           │ justificación             │                                                             │  │
│ ├───────────────────────────────────────────────────────────┼───────────────────────────┼─────────────────────────────────────────────────────────────┤  │
│ │ Encoders simétricos adecuados para ambas modalidades      │ Implícita                 │ Hacer explícita como asunción de Fase 1                     │  │
│ ├───────────────────────────────────────────────────────────┼───────────────────────────┼─────────────────────────────────────────────────────────────┤  │
│ │ H-series captura cantidades físicas distintas en speech   │ Documentado como          │ Fortalecer: documentar como confound con consecuencias      │  │
│ │ vs EGG                                                    │ hipótesis                 │ interpretativas                                             │  │
│ ├───────────────────────────────────────────────────────────┼───────────────────────────┼─────────────────────────────────────────────────────────────┤  │
│ │ n_fft=2048 con búsqueda ±2 bins suficiente para           │ Implícita                 │ Documentar como decisión de ingeniería                      │  │
│ │ extracción armónica                                       │                           │                                                             │  │
│ └───────────────────────────────────────────────────────────┴───────────────────────────┴─────────────────────────────────────────────────────────────┘  │
│                                                                                                                                                          │
│ 6.4 La frase "armonía natural" en documentación de Escalón 2 debe siempre especificar qué familia. "Descriptor de armonía natural" es ambiguo: puede     │
│ referir a V4-lin (Familia A) o H-series (Familia B). Ambos son "naturales" en el sentido de "físicos, no perceptuales", pero testean hipótesis           │
│ distintas.                                                                                                                                               │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ 7. Impacto en la estrategia arquitectónica (Nivel 3)                                                                                                     │
│                                                                                                                                                          │
│ Fase 1 (actual): Encoders simples y simétricos. CNN+Transformer, d=512, from scratch, idénticos para ambas modalidades. Epistemológicamente correcto     │
│ para esta etapa: maximiza interpretabilidad, el descriptor es la única variable.                                                                         │
│                                                                                                                                                          │
│ Fase 2 (siguiente, condicional a señal en Fase 1): Asimetría controlada.                                                                                 │
│ Si Fase 1 muestra señal pero más débil que lo esperado, una explicación es que encoders simétricos son subóptimos para el par asimétrico speech/EGG.     │
│ Paso siguiente:                                                                                                                                          │
│ - CNNs asimétricas (speech necesita más resolución alta para formantes; EGG necesita mejor resolución baja para el pulso glotal)                         │
│ - Puntos de inyección diferentes por modalidad (H-series podría funcionar distinto en speech vs EGG encoder por el confound del tracto vocal)            │
│ - Ablación: H-series solo en speech encoder vs solo en EGG encoder vs ambos                                                                              │
│                                                                                                                                                          │
│ Trigger para Fase 2: Cualquiera de:                                                                                                                      │
│ - Resultados ambiguos: H-series mejora pero dentro del CI de D0 (≈ D0 bajo la regla operativa), o mejora en una dirección S2E pero no E2S.               │
│ - Null limpio de H-series bajo training sano Y evidencia de uso real del mecanismo: H-series ≈ D0 o H-series < D0, pero se verifican TODAS las           │
│ siguientes condiciones:                                                                                                                                  │
│   a. Training sano: convergencia sin colapso, loss estable, no degeneración de covarianza.                                                               │
│   b. Uso real del mecanismo: xattn_scale no degenerado (no colapsado a 0 ni saturado), contribución de la rama cross-attention no trivial (medida como   │
│ norma relativa del residuo xattn vs features crudas).                                                                                                    │
│   c. Sensibilidad al descriptor: Si es factible, ablación rápida que substituya H-series por ruido gaussiano de misma estadística — si el resultado no   │
│ cambia, el mecanismo nunca se enganchó y el null no es informativo sobre el descriptor.                                                                  │
│ Solo si (1), (2) y (3) se cumplen, la simetría del encoder es una explicación plausible que merece testeo en Fase 2. Sin estas verificaciones, un null   │
│ puede ser simplemente "el mecanismo no se enganchó" y no justifica atribuirlo a confounds arquitectónicos.                                               │
│                                                                                                                                                          │
│ Fase 3 (condicional a Fase 2): Benchmark con encoder fuerte. WavLM/HuBERT frozen como speech encoder, encoder trainable pequeño para EGG. Testea si el   │
│ efecto descriptor persiste con representaciones de nivel foundation model. Análogo a Gate 7.1a en Escalón 1.                                             │
│                                                                                                                                                          │
│ Trigger para Fase 3: Fase 1 o 2 muestra señal positiva clara para descriptores de armonía natural, y la pregunta pasa a ser si el efecto persiste con    │
│ encoders más fuertes.                                                                                                                                    │
│                                                                                                                                                          │
│ Principio rector: Los cambios de arquitectura son eventos epistemológicos, no optimizaciones de ingeniería. Pasar de simétrico a asimétrico cambia qué   │
│ testea el experimento. Pasar a foundation models cambia qué testea el experimento. No son "upgrades" — son experimentos diferentes que testean preguntas │
│  diferentes.                                                                                                                                             │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ 8. Documentos que deben reescribirse                                                                                                                     │
│                                                                                                                                                          │
│ Prioridad: 1                                                                                                                                             │
│ Documento: Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md                                                                                              │
│ Alcance: Reorganizar familias con taxonomía estricta, agregar matriz de predicciones, reencuadrar concatenación, posicionar H-series como test primario, │
│                                                                                                                                                          │
│   explicitar asunciones                                                                                                                                  │
│ ────────────────────────────────────────                                                                                                                 │
│ Prioridad: 2                                                                                                                                             │
│ Documento: Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/plan_rectificacion_armonia_natural.md                                                            │
│ Alcance: Disciplina terminológica (siempre especificar familia), agregar matriz de predicciones, fortalecer prioridad epistemológica de H-series, caveat │
│                                                                                                                                                          │
│   V4-lin parametrización                                                                                                                                 │
│ ────────────────────────────────────────                                                                                                                 │
│ Prioridad: 3                                                                                                                                             │
│ Documento: Documents/01_FRENTES_ACTIVOS/ESCALON_2/ROADMAP_ESCALON_2.md                                                                                   │
│ Alcance: Actualizar sección S2-P2-main con interpretación de mecanismo, agregar terminología Familia A/B/C/D al glosario                                 │
│ ────────────────────────────────────────                                                                                                                 │
│ Prioridad: 4                                                                                                                                             │
│ Documento: MARCO_EPISTEMOLOGICO_PHIDEUS.md                                                                                                               │
│ Alcance: Agregar subsección sobre "predicciones operativas" (qué predice HIT, qué patrones debilitarían la tesis). Propagar la taxonomía de 4 familias a │
│  la                                                                                                                                                      │
│   capa epistemológica — la rectificación no debe quedar confinada solo a Escalón 2 porque ya dialoga con la posición transversal del programa.           │
│ ────────────────────────────────────────                                                                                                                 │
│ Prioridad: 5                                                                                                                                             │
│ Documento: Documents/04_TRANSVERSAL/.../CATALOGO_NARRATIVO_DESCRIPTORES_RATIOS_PHIDEUS.md                                                                │
│ Alcance: Actualizar sección Escalón 2 con narrativa rectificada, agregar distinción mecanismo vs contenido                                               │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ 9. Qué NO tocar todavía                                                                                                                                  │
│                                                                                                                                                          │
│ - Código: No se necesitan cambios. Las implementaciones son técnicamente correctas.                                                                      │
│ - El experimento corriendo (V4-lin-attnbias): Que termine. Se re-categoriza (Familia A), no se invalida.                                                 │
│ - El protocolo congelado: sr, pool, split, segment index, CI — intocables.                                                                               │
│ - Multi-seed: Viene después de que el marco interpretativo esté limpio.                                                                                  │
│ - Condiciones de ruido: noise1-3 es S2-P2.5+ o posterior.                                                                                                │
│ - Gate 6, Gate 8, Gate 7: Pertenecen al cierre de Escalón 1.                                                                                             │
│ - Escalón 4 (ECG ↔ PPG): Solo concepto. No diseñar hasta que Escalón 2 produzca resultados limpios.                                                      │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ 10. Orden de implementación recomendado                                                                                                                  │
│                                                                                                                                                          │
│ 1. Inmediato (antes de que H-series-xattn corra): Crear la matriz de predicciones (sección 2.5). Pre-registro interpretativo.                            │
│ 2. Inmediato: Reescribir secciones clave de README.md de Escalón 2 con la taxonomía corregida y la prioridad de H-series.                                │
│ 3. Al completar V4-lin-attnbias (30ep): Reportar resultados con framing de Familia A ("test de dinámica del oscilador"), no como "test de armonía        │
│ natural."                                                                                                                                                │
│ 4. Correr H-series-xattn (30ep): El experimento primario. Framing como test de la tesis fuerte de HIT desde el inicio.                                   │
│ 5. Correr A4-16k-xattn (30ep, comparable a los demás arms): El control adversario. Si entra en la matriz de predicciones pre-registrada, debe correr al  │
│ mismo schedule que H-series-xattn y V4-lin-attnbias. Un run corto (10ep) solo serviría como filtro de colapso temprano; cualquier inferencia comparativa │
│  basada en un arm de 10ep queda marcada explícitamente como provisional y no se usa para declarar A > B ni A ≈ B en la preregistración.                  │
│ 6. Al tener todos los resultados P2.5 Phase 1: Aplicar la matriz de predicciones. Reportar observaciones. El usuario decide interpretación.              │
│ 7. Si hay señal (algún descriptor > D0 con atención): Correr V4-log como control paramétrico. Cross-variants (V4-lin+xattn, H-series+attnbias) solo si   │
│ la hipótesis de matching de mecanismo necesita testeo.                                                                                                   │
│ 8. Actualizar plan_rectificacion y ROADMAP con resultados y narrativa corregida.                                                                         │
│ 9. Actualizar MARCO y CATALOGO solo después de que los resultados estén interpretados y la narrativa estabilizada.                                       │
│ 10. Considerar cambios arquitectónicos Fase 2 (encoders asimétricos) si resultados Fase 1 son ambiguos, o si H-series produce un null limpio bajo        │
│ training sano (la simetría de encoders es confound reconocido que merece testeo directo antes de interpretar el null como evidencia contra HIT).         │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ 11. Impacto estructural en el árbol documental                                                                                                           │
│                                                                                                                                                          │
│ La rectificación epistemológica no puede quedar solo como cambio de narrativa. Tiene que traducirse en una arquitectura documental clara, para que el    │
│ frente no quede repartido entre documentos que hablan desde marcos distintos sin jerarquía explícita.                                                    │
│                                                                                                                                                          │
│ 11.1 Problema actual                                                                                                                                     │
│                                                                                                                                                          │
│ El árbol documental de Escalón 2 es semánticamente híbrido:                                                                                              │
│ - Parte sigue escrita desde el diseño original de P2-main (concatenación).                                                                               │
│ - Parte ya está en modo P2.5 (attention).                                                                                                                │
│ - Parte está en modo "rectificación epistemológica" (taxonomía de familias, predicciones).                                                               │
│ - No queda claro qué documento manda sobre qué capa.                                                                                                     │
│                                                                                                                                                          │
│ 11.2 Jerarquía documental explícita                                                                                                                      │
│                                                                                                                                                          │
│ Nivel: 1. Estado canónico                                                                                                                                │
│ Documento: Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md                                                                                              │
│ Función: Estado actual del frente, taxonomía vigente, resultados acumulados                                                                              │
│ Manda sobre: Todo lo demás dentro de ESCALON_2/                                                                                                          │
│ ────────────────────────────────────────                                                                                                                 │
│ Nivel: 2. Secuencia operativa                                                                                                                            │
│ Documento: Documents/01_FRENTES_ACTIVOS/ESCALON_2/ROADMAP_ESCALON_2.md                                                                                   │
│ Función: Qué fase está activa, bifurcaciones abiertas, secuencia de ejecución                                                                            │
│ Manda sobre: Decisiones operativas                                                                                                                       │
│ ────────────────────────────────────────                                                                                                                 │
│ Nivel: 3. Marco vivo de rectificación                                                                                                                    │
│ Documento: Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/plan_rectificacion_armonia_natural.md                                                            │
│ Función: Documento rector de la rectificación epistemológica de S2-P2. Contiene taxonomía de familias, criterios de lectura, asunciones explícitas       │
│ Manda sobre: Interpretación de resultados de P2-main y P2.5                                                                                              │
│ ────────────────────────────────────────                                                                                                                 │
│ Nivel: 4. Preregistro interpretativo                                                                                                                     │
│ Documento: Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/PREDICCIONES_EPISTEMOLOGICAS_P25.md                                                              │
│ Función: Artefacto nuevo. Matriz de predicciones, regla operativa de CI pareado, guardrails para null                                                    │
│ Manda sobre: Lectura falsificable de resultados P2.5                                                                                                     │
│ ────────────────────────────────────────                                                                                                                 │
│ Nivel: 5. Discusión técnica                                                                                                                              │
│ Documento: Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/Discusion_Inyeccion_descriptores.md                                                              │
│ Función: Discusión técnica de soporte sobre mecanismos de inyección. NO es documento canónico                                                            │
│ Manda sobre: Diseño de implementación                                                                                                                    │
│ ────────────────────────────────────────                                                                                                                 │
│ Nivel: 6. Historia de diseño                                                                                                                             │
│ Documento: Documents/01_FRENTES_ACTIVOS/ESCALON_2/PLAN_IMPLEMENTACION_ESCALON2.md                                                                        │
│ Función: Plan original de implementación (histórico)                                                                                                     │
│ Manda sobre: Nada — referencia                                                                                                                           │
│                                                                                                                                                          │
│ Autoridad por capa (no jerarquía lineal — cada documento manda en su dominio):                                                                           │
│                                                                                                                                                          │
│ ┌──────────────────────────┬───────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────┐  │
│ │           Capa           │          Documento que manda          │                               Alcance de autoridad                               │  │
│ ├──────────────────────────┼───────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤  │
│ │ Estado canónico del      │ README.md                             │ Qué descriptores hay, qué resultados existen, qué fase está activa               │  │
│ │ frente                   │                                       │                                                                                  │  │
│ ├──────────────────────────┼───────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤  │
│ │ Secuencia operativa      │ ROADMAP_ESCALON_2.md                  │ Qué corre siguiente, qué bifurcaciones están abiertas, qué es histórico          │  │
│ ├──────────────────────────┼───────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤  │
│ │ Marco interpretativo /   │ plan_rectificacion_armonia_natural.md │ Taxonomía de familias, asunciones explícitas, criterios de lectura               │  │
│ │ taxonomía                │                                       │                                                                                  │  │
│ ├──────────────────────────┼───────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤  │
│ │ Lectura falsificable de  │ PREDICCIONES_EPISTEMOLOGICAS_P25.md   │ Preregistro que manda sobre cómo se leen los resultados de P2.5. Regla de CI     │  │
│ │ P2.5                     │                                       │ pareado, matriz de predicciones, guardrails para nulls                           │  │
│ ├──────────────────────────┼───────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤  │
│ │ Diseño técnico de        │ Discusion_Inyeccion_descriptores.md   │ Cómo funciona attn bias vs xattn, por qué se eligió cada mecanismo               │  │
│ │ mecanismos               │                                       │                                                                                  │  │
│ ├──────────────────────────┼───────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤  │
│ │ Historia de diseño       │ PLAN_IMPLEMENTACION_ESCALON2.md       │ Referencia histórica. Documento superseded — no tiene autoridad sobre decisiones │  │
│ │                          │                                       │  actuales                                                                        │  │
│ └──────────────────────────┴───────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                                                                          │
│ Convención de lectura: Para saber el estado actual → README. Para saber qué hacer siguiente → ROADMAP. Para saber cómo interpretar resultados →          │
│ PREDICCIONES (manda sobre la lectura de P2.5) + plan_rectificacion (marco general). Para entender decisiones técnicas → Discusion_Inyeccion.             │
│                                                                                                                                                          │
│ Resolución de conflictos: No es cadena lineal. Si README y PREDICCIONES dicen cosas distintas sobre cómo leer un resultado, PREDICCIONES manda (es el    │
│ preregistro). Si README y ROADMAP discrepan sobre qué fase está activa, README manda (es el estado canónico). Cada documento es autoridad en su capa.    │
│                                                                                                                                                          │
│ 11.3 Separación semántica dentro de S2_P2/                                                                                                               │
│                                                                                                                                                          │
│ Sin reorganizar directorios todavía, se declara la siguiente pertenencia:                                                                                │
│                                                                                                                                                          │
│ ┌───────────────────────────┬────────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────┐  │
│ │         Concepto          │                               Documentos                               │                     Estado                     │  │
│ ├───────────────────────────┼────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤  │
│ │ S2-P2-control (D0)        │ Resultados en README.md y ROADMAP. Sin documento propio necesario      │ COMPLETO, referencia                           │  │
│ ├───────────────────────────┼────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤  │
│ │ S2-P2-main                │ Resultados en ROADMAP (sección histórica). Interpretación en           │ COMPLETO, resultado negativo informativo sobre │  │
│ │ (concatenación)           │ plan_rectificacion                                                     │  mecanismo                                     │  │
│ ├───────────────────────────┼────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤  │
│ │ S2-P2.5 (attention)       │ Discusion_Inyeccion (diseño), PREDICCIONES (preregistro), ROADMAP      │ FASE ACTIVA                                    │  │
│ │                           │ (secuencia activa)                                                     │                                                │  │
│ ├───────────────────────────┼────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤  │
│ │ Rectificación             │ plan_rectificacion (marco), PREDICCIONES (falsificabilidad)            │ Marco transversal de lectura de P2-main y P2.5 │  │
│ │ epistemológica            │                                                                        │                                                │  │
│ └───────────────────────────┴────────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────┘  │
│                                                                                                                                                          │
│ Documento superseded: PLAN_IMPLEMENTACION_ESCALON2.md debe marcarse explícitamente en su propio encabezado como [HISTÓRICO / SUPERSEDED] para evitar que │
│  parezca un documento vivo. Agregar una nota al inicio: "Este documento es el plan original de implementación de Escalón 2. La taxonomía vigente, los    │
│ criterios de lectura y el preregistro interpretativo están en plan_rectificacion_armonia_natural.md y PREDICCIONES_EPISTEMOLOGICAS_P25.md                │
│ respectivamente."                                                                                                                                        │
│                                                                                                                                                          │
│ Criterio para reorganización de directorios: Si P2.5 se consolida como fase larga con sub-ramas propias (e.g., V4-lin-attnbias, H-series-xattn,          │
│ A4-16k-xattn producen resultados que requieren análisis detallados individuales), entonces se crea S2_P2/attention/ y se mueve S2_P2/concat/ como        │
│ histórico. Hasta entonces, la separación semántica declarada aquí es suficiente.                                                                         │
│                                                                                                                                                          │
│ 11.4 Decisión sobre la matriz de predicciones                                                                                                            │
│                                                                                                                                                          │
│ La matriz de predicciones de la sección 2.5 se extrae a un artefacto propio:                                                                             │
│                                                                                                                                                          │
│ Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/PREDICCIONES_EPISTEMOLOGICAS_P25.md                                                                         │
│                                                                                                                                                          │
│ Contenido:                                                                                                                                               │
│ - Regla operativa de comparación (bootstrap pareado sobre Δ, CI_Δ, umbral 2pp)                                                                           │
│ - Matriz de predicciones completa (6 patrones → interpretaciones)                                                                                        │
│ - Guardrails para interpretar nulls (evidencia de uso real del mecanismo)                                                                                │
│ - Fecha de creación (pre-registro: antes de que H-series-xattn y A4-16k-xattn corran)                                                                    │
│ - Referencia cruzada a plan_rectificacion para contexto                                                                                                  │
│                                                                                                                                                          │
│ Justificación: Separa preregistro interpretativo de explicación general. Permite que la matriz sea citada y verificada independientemente. Deja más      │
│ limpia la distinción entre diseño experimental, rectificación epistemológica y lectura falsificable.                                                     │
│                                                                                                                                                          │
│ plan_rectificacion_armonia_natural.md referencia a PREDICCIONES pero no lo duplica.                                                                      │
│                                                                                                                                                          │
│ 11.5 Sync transversal obligatorio                                                                                                                        │
│                                                                                                                                                          │
│ La rectificación no puede quedarse encapsulada solo dentro de ESCALON_2/. Al estabilizarse, requiere propagación a la capa transversal.                  │
│                                                                                                                                                          │
│ Documento: Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md                                                                                              │
│ Qué se actualiza: Taxonomía, H-series como test primario, resultados                                                                                     │
│ Cuándo: Inmediato (paso 2 de sección 10)                                                                                                                 │
│ ────────────────────────────────────────                                                                                                                 │
│ Documento: Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/plan_rectificacion_armonia_natural.md                                                            │
│ Qué se actualiza: Terminología de familias, link a PREDICCIONES                                                                                          │
│ Cuándo: Inmediato                                                                                                                                        │
│ ────────────────────────────────────────                                                                                                                 │
│ Documento: Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/PREDICCIONES_EPISTEMOLOGICAS_P25.md                                                              │
│ Qué se actualiza: Creación del artefacto                                                                                                                 │
│ Cuándo: Inmediato (antes de que H-series-xattn corra)                                                                                                    │
│ ────────────────────────────────────────                                                                                                                 │
│ Documento: Documents/01_FRENTES_ACTIVOS/ESCALON_2/ROADMAP_ESCALON_2.md                                                                                   │
│ Qué se actualiza: Glosario Familia A/B/C/D, separación concat/attention                                                                                  │
│ Cuándo: Inmediato                                                                                                                                        │
│ ────────────────────────────────────────                                                                                                                 │
│ Documento: MARCO_EPISTEMOLOGICO_PHIDEUS.md                                                                                                               │
│ Qué se actualiza: Taxonomía de 4 familias, predicciones operativas                                                                                       │
│ Cuándo: Después de P2.5 Fase 1 (cuando la taxonomía esté estabilizada por resultados)                                                                    │
│ ────────────────────────────────────────                                                                                                                 │
│ Documento: Documents/00_TRONCAL/INDICE_DOCUMENTACION.md                                                                                                  │
│ Qué se actualiza: Registrar PREDICCIONES_EPISTEMOLOGICAS_P25.md                                                                                          │
│ Cuándo: Inmediato (al crear el artefacto)                                                                                                                │
│ ────────────────────────────────────────                                                                                                                 │
│ Documento: Documents/00_TRONCAL/Proyecto_Estado_Actual.md                                                                                                │
│ Qué se actualiza: Actualizar sección Escalón 2 con framing rectificado                                                                                   │
│ Cuándo: Después de P2.5                                                                                                                                  │
│ ────────────────────────────────────────                                                                                                                 │
│ Documento: Documents/00_TRONCAL/bitacora_desarrollo.md                                                                                                   │
│ Qué se actualiza: Entrada de rectificación epistemológica                                                                                                │
│ Cuándo: Inmediato (al ejecutar la rectificación)                                                                                                         │
│ ────────────────────────────────────────                                                                                                                 │
│ Documento: Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/CATALOGO_NARRATIVO_DESCRIPTORES_RATIOS_PHIDEUS.md                                               │
│ Qué se actualiza: Sección Escalón 2, distinción mecanismo vs contenido, taxonomía                                                                        │
│ Cuándo: Después de P2.5 (narrativa estabilizada)                                                                                                         │
│ ────────────────────────────────────────                                                                                                                 │
│ Documento: Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md                                                    │
│ Qué se actualiza: Agregar la transición concat→attention como hito representacional                                                                      │
│ Cuándo: Después de P2.5 (narrativa estabilizada)                                                                                                         │
│                                                                                                                                                          │
│ 11.6 Criterio de intervención estructural                                                                                                                │
│                                                                                                                                                          │
│ La rectificación actual requiere:                                                                                                                        │
│ - Reescritura de documentos existentes: Sí (README, ROADMAP, plan_rectificacion).                                                                        │
│ - Creación de documento nuevo: Sí (PREDICCIONES_EPISTEMOLOGICAS_P25.md).                                                                                 │
│ - Reorganización del árbol: No todavía. La separación semántica declarada en 11.3 es suficiente.                                                         │
│                                                                                                                                                          │
│ Trigger para reorganización del árbol: P2.5 se consolida con 3+ sub-ramas que producen resultados detallados → se crean S2_P2/attention/ y               │
│ S2_P2/concat/.                                                                                                                                           │
│                                                                                                                                                          │
│ 11.7 Test de verificación documental                                                                                                                     │
│                                                                                                                                                          │
│ Después de ejecutar la rectificación, cualquier agente o lector debe poder responder sin ambigüedad:                                                     │
│                                                                                                                                                          │
│ ┌─────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────┐                            │
│ │                    Pregunta                     │                                Respuesta                                │                            │
│ ├─────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤                            │
│ │ ¿Cuál es el documento canónico de Escalón 2?    │ README.md                                                               │                            │
│ ├─────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤                            │
│ │ ¿Cuál es el plan vivo?                          │ plan_rectificacion_armonia_natural.md                                   │                            │
│ ├─────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤                            │
│ │ ¿Dónde está el preregistro interpretativo?      │ PREDICCIONES_EPISTEMOLOGICAS_P25.md                                     │                            │
│ ├─────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤                            │
│ │ ¿Qué documento guarda la historia del rediseño? │ PLAN_IMPLEMENTACION_ESCALON2.md + sección histórica de ROADMAP          │                            │
│ ├─────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤                            │
│ │ ¿Qué parte corresponde a concatenación?         │ Resultados históricos en ROADMAP, interpretación en plan_rectificacion  │                            │
│ ├─────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤                            │
│ │ ¿Qué parte corresponde a attention?             │ Discusion_Inyeccion + PREDICCIONES + ROADMAP (fase activa)              │                            │
│ ├─────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤                            │
│ │ ¿Qué docs transversales hay que resincronizar?  │ Tabla de 11.5                                                           │                            │
│ ├─────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤                            │
│ │ ¿Quién manda sobre cómo leer resultados P2.5?   │ PREDICCIONES_EPISTEMOLOGICAS_P25.md (preregistro)                       │                            │
│ ├─────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤                            │
│ │ ¿Quién manda cuando hay contradicción?          │ Autoridad por capa: cada documento manda en su dominio (ver tabla 11.2) │                            │
│ └─────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────┘                            │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ Verificación                                                                                                                                             │
│                                                                                                                                                          │
│ La rectificación se verifica por:                                                                                                                        │
│ 1. Cada documento reescrito usa la taxonomía de 4 familias consistentemente                                                                              │
│ 2. H-series está posicionado como el test más directamente alineado con la tesis fuerte, sin convertirlo en árbitro absoluto                             │
│ 3. V4-lin nunca se presenta como proxy de "armonía natural" en sentido fuerte                                                                            │
│ 4. La matriz de predicciones existe ANTES de que H-series-xattn y A4-16k-xattn produzcan resultados                                                      │
│ 5. La regla operativa de comparación usa bootstrap pareado sobre Δ (no comparación de CIs individuales)                                                  │
│ 6. A4-16k-xattn corre a 30ep comparable, o toda inferencia basada en él queda marcada como provisional                                                   │
│ 7. Las asunciones listadas en 6.3 están explicitadas en la documentación relevante                                                                       │
│ 8. El eval script implementa grouped bootstrap pareado sobre Δ para generar CI_Δ                                                                         │
│ 9. El guardrail de uso real del mecanismo (xattn_scale, contribución, ablación de ruido) está documentado como prerrequisito para interpretar nulls como │
│  confound de arquitectura                                                                                                                                │
│ 10. PREDICCIONES_EPISTEMOLOGICAS_P25.md existe como artefacto propio, separado del plan_rectificacion                                                    │
│ 11. La jerarquía documental de 11.2 está declarada y el test de 11.7 se responde sin ambigüedad                                                          │
│ 12. INDICE_DOCUMENTACION.md registra el nuevo artefacto                                                                                                  │
│                                                                                                                                                          │
│ Archivos críticos                                                                                                                                        │
│                                                                                                                                                          │
│ - Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md — Estado canónico, prioridad 1                                                                        │
│ - Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/plan_rectificacion_armonia_natural.md — Plan vivo de rectificación, prioridad 2                           │
│ - Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/PREDICCIONES_EPISTEMOLOGICAS_P25.md — Artefacto nuevo: preregistro interpretativo, prioridad 2            │
│ - Documents/01_FRENTES_ACTIVOS/ESCALON_2/ROADMAP_ESCALON_2.md — Roadmap técnico, prioridad 3                                                             │
│ - MARCO_EPISTEMOLOGICO_PHIDEUS.md — Posición epistemológica, prioridad 4 (taxonomía después de P2.5)                                                     │
│ - Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/CATALOGO_NARRATIVO_DESCRIPTORES_RATIOS_PHIDEUS.md — Capa transversal, prioridad 5                        │
│ - Documents/00_TRONCAL/INDICE_DOCUMENTACION.md — Registrar nuevo artefacto, inmediato                                                                    │
│ - src/bias_control/vocal_descriptors.py — Implementación de referencia (solo lectura, no se modifica)
