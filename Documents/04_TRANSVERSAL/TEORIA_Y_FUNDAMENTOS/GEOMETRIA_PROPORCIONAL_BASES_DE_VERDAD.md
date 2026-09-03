# Geometría proporcional y bases de verdad

## El problema que antecede a la arquitectura

La hipótesis de una Proportional Processing Unit enfrenta una dificultad distinta de la que suele presentarse cuando se diseña una red para una tarea ya estabilizada: no existe todavía un archivo reconocido de “estructuras proporcionales resueltas” que determine qué debe aprender la arquitectura. AlphaFold no recibió una ontología completa de las proteínas, pero sí pudo trabajar sobre décadas de estructuras experimentales, secuencias evolutivamente relacionadas, regularidades geométricas y evaluaciones ciegas. Phideus no dispone de un equivalente directo para las proporciones.

Esta carencia no puede resolverse eligiendo por anticipado la serie armónica, los números racionales o una geometría favorita y tratándolos como si ya fueran la forma general de lo real. La primera campaña de investigación sobre bases de verdad proporcional produjo una conclusión más austera: **no aparece una única geometría de la proporción; aparecen familias de problemas proporcionales definidas por equivalencias, leyes de composición y regímenes dinámicos diferentes**.

Una razón aislada todavía no constituye una geometría. Empieza a hacerlo cuando sabemos qué transformaciones preservan el fenómeno, cómo se compone con otras razones y qué observaciones pertenecen al mismo estado. En datos composicionales, la escala global puede ser irrelevante. En un grafo de log-ratios, los ciclos imponen identidades. En geometría proyectiva, el cross-ratio permanece mientras las distancias ordinarias cambian. En una red de osciladores, en cambio, la cercanía a una razón racional no establece por sí sola un estado de locking: la relación debe verificarse en la trayectoria.

Una tercera ola amplió esa distinción. El cambio de unidad conserva una cantidad, pero no constituye por sí solo una simetría de la dinámica; la similitud física exige preservar ecuaciones, geometría, condiciones iniciales/de borde y todos los controles adimensionales relevantes; una power law puede ser exacta, asintótica o meramente empírica. En biología ocurre una separación análoga: correlación alométrica, mecanismo morfogenético y geometría de forma o linaje son objetos distintos.

Una cuarta ola encontró una recurrencia algebraica entre química, materiales, circuitos y control. Balances estequiométricos y leyes de Kirchhoff aportan restricciones exactas sobre núcleos y ciclos, pero no determinan cinética, termodinámica ni leyes de componente. Reducción de Kron, movimientos `Y-Delta`, bases alternativas de un subespacio y representaciones cristalinas equivalentes muestran además que una respuesta puede estar identificada aunque su realización interna no lo esté. Esta recurrencia no unifica los dominios; ofrece operaciones comunes que pueden probarse sin confundir estructura, constitución y observación.

Una quinta ola examinó complejos de cadenas, Hodge/DEC y sheaves. Su resultado vuelve más estricta la separación anterior: `B^2=0`, una sección global o una cohomología son exactas respecto del complejo o diagrama elegido, pero no prueban que sus celdas, stalks, restricciones o métricas pertenezcan al fenómeno. Su valor posible es de interfaz: tipar canales y gauges, distinguir existencia de unicidad, separar ruido de obstrucción y someter la estructura a controles verdaderos, sham y mal especificados.

Una sexta ola llevó ese mismo criterio a distribuciones y medidas. Fisher–Rao describe distinguibilidad dentro de una familia y un canal de observación; Aitchison describe información relativa entre partes; Wasserstein resuelve un acoplamiento bajo un costo declarado; Gromov–Wasserstein compara estructuras internas cuando no existe un costo cruzado. La coincidencia de soporte numérico no vuelve equivalentes esos objetos. Tampoco un coupling óptimo identifica por sí solo el mecanismo que produjo dos marginales, ni natural gradient convierte una mejora del optimizador en evidencia sobre la representación.

La séptima ola introdujo dos separaciones que afectan directamente el diseño de red. En dinámica, una ley variacional, una forma simpléctica, un vector field y un integrador son objetos diferentes: preservar estructura no identifica energía ni mecanismo, y un solver especializado puede producir el lift que se atribuye al prior neuronal. En sistemas abiertos, wiring, semántica, constitución local, implementación numérica y equivalencia observable tampoco son intercambiables. Una composición puede ser sintácticamente exacta y físicamente falsa; un lazo de feedback necesita además una autoridad de delay, fixed point o DAE que asegure que está bien planteado.

La octava ola agregó una pregunta anterior y otra posterior a ese pipeline. Antes de operar sobre un estado latente hay que establecer qué parte del mecanismo es identificable desde las observaciones e intervenciones disponibles, y bajo qué gauge. Después de proponer una relación hay que distinguir si el solver concluyó, si el artefacto pasó un checker, si la proposición fue probada o refutada y si sus premisas tienen autoridad física. Estas dimensiones no forman una sola escala ni caben en una etiqueta `válido/inválido`: una representación puede predecir sin identificar, un certificado puede ser exacto sobre premisas físicamente falsas y un artefacto rechazado no refuta por sí mismo la proposición.

La novena ola desplazó la pregunta desde la estructura hacia la autoridad de la evidencia que la sostiene. Una proporción medida no es un número desprendido del aparato: depende de un mensurando, indicaciones, una referencia, una función de calibración, una incertidumbre conjunta y, cuando numeradores, denominadores o sesiones comparten fuentes, una covarianza. Del mismo modo, una observación elegida activamente no es equivalente a una muestra dada: depende de la historia visible, el espacio legal de acciones, la utilidad, el costo, la seguridad y el régimen de acceso. Metrología y diseño experimental no agregan otra geometría; impiden atribuir a la arquitectura lo que proviene de una calibración retrospectiva o una política de adquisición privilegiada.

La décima ola volvió explícita una geometría que faltaba: el flujo entre resoluciones. Renormalizar no es sólo expresar una escala en otras unidades ni acumular features multirresolución. Es declarar un kernel que elimina información, una base de operadores, observables que deben transportarse y un error de truncación; cuando los mapas son compatibles, también puede estudiarse su composición y el subespacio de perturbaciones que permanece relevante. Esta autoridad sigue siendo condicional al modelo y al kernel. Por eso el programa separa ejecutar un coarse-graining conocido de descubrir uno desde acceso microscópico igualado. En paralelo, la psicofísica de magnitud aporta curvas conductuales y tuning neural como evidencia externa, pero no permite inferir una división explícita ni una métrica física universal: representación, ruido y decisión deben adjudicarse por separado.

La undécima ola preguntó cuándo una razón y una macrovariable dicen algo más que lo que ya fue puesto en su definición. La teoría representacional de la medición obliga a separar comparaciones finitas, existencia de una representación, unicidad bajo un grupo admisible, meaningfulness, verdad empírica y autoridad. La abstracción causal agrega una exigencia distinta: una macrovariable no se vuelve causal porque prediga bien, sino cuando conserva estados e intervenciones dentro de una jurisdicción declarada y sus realizadores no funcionan como atajos. `P2l` y `P2m` operan así como firewalls contra dos falsos ground truths: el cociente calculable que se presenta como magnitud y el mapa plantado que se presenta como única abstracción correcta.

La duodécima ola incorporó dos pruebas algebraicas adicionales. La consistencia proyectiva pregunta si una familia de representaciones sigue siendo coherente al restringir cardinalidad o exposición; equivariancia a permutaciones dentro de cada tamaño no alcanza. Graphons densos y graphexes dispersos requieren samplers, índices de escala y clases de equivalencia distintos, y una ley finita simétrica puede no ser extendible. La geometría tropical, por otra parte, sólo tiene autoridad exacta cuando el dominio compone alternativas por `min/max` y trayectorias por suma. Un límite de baja temperatura y una función piecewise-linear pueden parecerse al mismo operador sin compartir su ley. Por eso `P2n` separa equivariancia, conmutatividad paired, projectivity en ley y extendibilidad; `P2o` separa tropicalidad exacta, dequantización y ajuste piecewise-linear.

La decimotercera ola llevó esa precaución a objetos cuya estructura puede ser exacta aun cuando una coordinatización falle o sea insuficiente. Un chirotope no entrega puntos: entrega signos orientados sujetos a alternancia y Grassmann–Plücker. Sus circuitos, cocircuitos, duales y menores forman una geometría combinatoria módulo gauges y relabelings; el teorema de representación por pseudoesferas no implica que el arreglo pueda enderezarse a hiperplanos. Esta diferencia vuelve a los matroides orientados un banco especialmente útil: permite pedir consistencia global, traducción entre vistas y conmutación con borrado o contracción sin convertir coordenadas plantadas en la única verdad disponible. `A13/P2p` registra esa posibilidad, con factores verdaderos comparados contra shams de igual incidencia y costo, y con loss, proyección, dualidad, menores y solver separados como fuentes de efecto.

La misma ola mostró que “información” tampoco define una sola región geométrica. Las desigualdades de Shannon delimitan un cono polimatroidal necesario, pero no caracterizan la región entrópica; Ingleton caracteriza una jurisdicción lineal más estrecha y puede ser violada por distribuciones genuinamente entrópicas. Una tabla de entropías, una PMF conjunta y una fuente lineal son, por tanto, outputs distintos. `A14` queda como interfaz de consultas sobre subconjuntos, no como arquitectura ómnibus: `A14a/SLIB-DSF`, `A14b/SLIB-PMF` y `A14c/SLIB-LINEAR` deben tener baselines, privilegios y efectos independientes. También cambia el estatuto de la inferencia finita: una PMF empírica conjunta siempre induce un vector entrópico; una aparente violación universal allí denuncia error numérico, mientras otros estimadores pueden ser incoherentes sin refutar la distribución poblacional.

La decimocuarta ola introdujo una distinción entre reconocer una jurisdicción y ejecutar una operación cerrada dentro de ella. Estabilidad real, hiperbolicidad y Lorentzianidad no son tres nombres de una misma log-concavidad; tampoco submodularidad, `M/M^natural`-convexidad, gross substitutes y assignment valuations forman una única clase. En vez de pedir una cabeza ómnibus, `A15/P2r` aísla la contracción direccional de un polinomio cuya tabla completa debe inferirse antes de conocer la dirección, mientras `A16/P2s` aísla el scoring assignment aditivo sobre una topología pública conocida. Clase, cono, pesos verdaderos, certificados y solvers permanecen fuera del input deployable. El puente formal entre ambas ramas es deliberadamente estrecho: el soporte de un polinomio Lorentziano homogéneo es M-convexo. No se deduce de allí una tropicalización común, ni que coeficientes, raíces, valuaciones y orientación sean recuperables unos de otros.

La decimoquinta ola volvió esa prudencia todavía más operativa. En completación de distancias, la existencia de una relación proporcional entre constraints incidentes no implica que la distancia faltante sea identificable: `A17/P2t` restringe el estimando al valor de un bottleneck de log-ratios dentro de una arquitectura de cuñas fija y agrega un detector de aplicabilidad que debe abstenerse fuera de la fibra identificable. En exchange local, una fórmula correcta no acredita discovery si la columna que la gobierna no puede distinguirse desde el contexto observado: `A18/P2u` exige enumerar exactamente el version space y admite sólo componentes que determinan una única órbita de columna. Así, inferencia y ejecución quedan separadas, y los casos ambiguos no se convierten en ruido de entrenamiento.

La decimosexta ola agregó una diferencia entre **cocientar**, **realizar** y **adjudicar correspondencia**. Los espacios de forma de Kendall ofrecen una geometría exacta para configuraciones etiquetadas módulo traslación, escala y `SO(m)`, pero no deciden por sí mismos que dos landmarks sean homólogos; pasar a distancias normalizadas puede ampliar silenciosamente el cociente a `O(m)` y borrar quiralidad. `P2v` conserva este aporte como diagnóstico con espejos, relabeling y degeneraciones, sin inventar una arquitectura donde Procrustes ya agota la operación. La conformalidad discreta mostró el límite complementario: cross-ratios y leyes locales pueden ser correctos sin satisfacer las desigualdades globales de existencia. El primer generador neuronal de circle patterns quedó bloqueado porque filtraba la clase por una identidad local perfecta. La fuente matemática sigue aportando checker y ground truth; no existen todavía `A19/P2w`.

La decimoséptima ola hizo explícita una geometría de **transportes, gauges y referencias externas**. En sincronización, las relaciones `h_ij=g_i^{-1}g_j` pueden determinar una órbita global aunque ninguna coordenada absoluta sea identificable; el cierre de ciclos certifica integrabilidad, no verdad física. En interferometría, las closures cancelan gains bajo un modelo preciso, pero no vuelven inyectivo el mapa desde una imagen hacia las visibilidades que el aparato alcanzó a medir. `P2a-G` conserva el primer aporte como extensión auditada de P2a/A3 y separa residual exacto, mixer group-aware y sincronizador downstream. El segundo permanece como preflight de objeto, aparato, nullspace y covariance. Esta ola tampoco agrega una candidata `A*`: agrega un contrato para no confundir la geometría de una órbita con la autoridad de una reconstrucción.

La decimoctava ola introdujo una distinción adicional dentro de la propia idea de representación. La teoría de invariantes muestra que una función puede permanecer constante a lo largo de una acción y aun colapsar órbitas distintas; cuando los generadores separan el cociente, todavía pueden aparecer syzygies, valores no realizables o mal condicionamiento cerca de estabilizadores. Una moving frame fija una gauge sólo dentro de su carta regular. Persistent homology presenta el problema complementario: el barcode pertenece a una cadena que incluye observación, métrica, complejo, filtración y coeficientes; su estabilidad no autoriza esa cadena ni garantiza que un summary conserve cualquier query. `ORBIT-REPRESENTATION-AUDIT-v0` y `FILTRATION-AUTHORITY-PREFLIGHT-v0` convierten estas reservas en contratos de P0. No agregan `A19` ni un nuevo `P2*`: antes de atribuir una geometría a la red exigen separar invariante, separador, realizabilidad, gauge, operador, reader y autoridad.

La decimonovena ola extendió esa separación hacia dos lenguajes especialmente cercanos al horizonte de una PPU. El análisis armónico ofrece expansiones completas sobre grupos compactos, pero esa completitud no migra automáticamente desde el continuo hacia una truncación o un lattice: el espectro de potencia puede ser invariante sin separar órbitas, y el bispectrum sólo puede funcionar como referencia de completitud bajo una acción, una clase funcional, condiciones de rango y un régimen de muestreo declarados. La teoría de estados predictivos introduce una cautela paralela. El estado causal mínimo pertenece a la ley del proceso y a la equivalencia entre futuros, no al nombre de una variable oculta plantada; una aproximación por tests finitos, un rango de Hankel o una realización lineal no garantizan por sí solos una realización probabilística no negativa ni suficiencia para control bajo otra política. `HARMONIC-ORBIT-AUTHORITY-AUDIT-v0` y `PREDICTIVE-STATE-AUTHORITY-AUDIT-v0` convierten estas diferencias en suites P0 con evidencia igualada, acceso de solver tipado, preimágenes puntuadas cuando no hay identificabilidad y una ley de proceso íntegramente declarada. Tampoco agregan `A19` ni un nuevo `P2*`: obligan a que una futura arquitectura diga qué objeto representa antes de atribuirle una geometría armónica o un estado predictivo.

La vigésima ola desplazó esa precaución desde el estado representado hacia el operador que organiza sus transformaciones. En geometría espectral, el espectro del Laplaciano, la diagonal del heat kernel, el kernel completo y el semigrupo contienen cantidades distintas de información; una firma estable puede ser suficiente para una query sin identificar una variedad, y dos interiores no isométricos pueden compartir parte de la evidencia espectral. En dinámica, un predictor de observables puede aproximar bien trayectorias sin cerrar un subespacio de Koopman, identificar el operador de transferencia, recuperar su generador ni sostener una intervención. `SPATIAL-OPERATOR-AUTHORITY-AUDIT-v0` y `DYNAMICAL-OPERATOR-AUTHORITY-AUDIT-v0` organizan esas diferencias como contratos P0. La convergencia arquitectónica es deliberadamente condicional: un atlas que separa contrato de objeto/equivalencia, encoder relacional, propuesta de operador tipada, solver, reader y competencia/abstención, con autoridad externa registrada. No constituye `A19`, un nuevo `P2*` ni una operación común ya aprendida.

## Una base estratificada

La respuesta operativa no es un dataset total, sino una base estratificada con cuatro fuentes de evidencia y dos funciones transversales de validación.

| Tipo | Estrato o función | Qué fija | Recursos iniciales | Límite |
|---|---|---|---|---|
| fuente | verdad analítica | equivalencias, invariantes y composición exacta | Aitchison, ciclos, complejos, variación y wiring | puede ser exacta y no describir un fenómeno natural |
| fuente | simulación generativa | estado completo, mecanismo e intervención dentro del generador | Kuramoto, REBOUND, BioModels, dinámica geométrica y sistemas abiertos | hereda el mundo definido por el simulador; no adquiere autoridad física |
| fuente | cámara física | distancia entre estado y observación instrumental | Causal Chambers, ETH/Polimi | valida un aparato y un régimen, no universalidad |
| fuente | evidencia externa | recurrencia en materiales, percepción o sistemas naturales | fonones, HPatches, observación orbital, conducta auditiva y psicofísica de magnitud | suele ofrecer targets parciales e incertidumbre |
| validación | falsación adversarial | atajos, atribución y alcance | shuffles, no-ratio, OOD, controles param-matched | delimita; no aporta ontología positiva |
| validación | adjudicación ciega | resistencia al ajuste retrospectivo | benchmark prospectivo con modelos congelados | evalúa claims declarados, no universalidad |

Una serie de contratos atraviesa esa base sin convertirse en nuevas fuentes positivas. Medición, adquisición, mapa entre escalas, tarea perceptual, escala de medición, abstracción causal, familia proyectiva, operador tropical y las autoridades operatoriales espaciales y dinámicas declaran qué objeto, acceso, gauge, sampling, semigrupo, generador, solver y query sostienen cada claim. Ninguno reemplaza a una fuente de autoridad: impiden atribuir a la red lo que proviene del aparato, la política, el mapa, el readout, el sampler, el solver o una aproximación operatorial mal tipada.

Entre los bancos nuevos, Buckingham/SI ofrece una acción exacta de unidades; PDEBench, Sedov-Taylor, NACA y NIST-FDS permiten pasar de consistencia dimensional a similitud física; Ising/JHTDB fuerzan a tratar régimen, crossover y abstención. En biología, auxina/phyllotaxis, pescoids y organoides aportan intervención, mientras linajes, alas y mallas celulares aportan geometrías externas con gauges propios.

La exactitud y la externalidad no se reemplazan entre sí. Un oráculo analítico puede decir con precisión si una red respetó una ley de composición, pero no si esa ley organiza un sistema físico. Un corpus natural puede mostrar recurrencia, pero rara vez expone el estado causal completo. La base se vuelve fuerte cuando esas autoridades se encadenan sin confundirse.

## Separar estado, observación y juicio

El schema experimental propuesto abandona el campo genérico `label` como contenedor indiferenciado. Cada sistema debería distinguir:

- el **estado privilegiado**, conocido por el generador o el protocolo pero oculto al modelo;
- la **observación**, que incluye sensor, muestreo y degradación;
- las **relaciones locales** disponibles o inferibles;
- la **estructura global**, como órbita, potencial, partición, modo o régimen;
- la **equivalencia** bajo la cual se evalúa una respuesta;
- la **incertidumbre**, ya sea numérica, instrumental, posterior o humana;
- la **procedencia**, con versión, licencia y una jerarquía explícita entre unidad independiente, grupo atómico de split y unidades parentales;
- el **estatuto de cada resultado**, separando artefacto, claim formal, solver, identificabilidad causal, autoridad física y decisión del sistema.
- el **contrato de medición**, con función y versión de calibración congeladas prospectivamente, incertidumbre conjunta y grafo de referencias compartidas;
- el **contrato de adquisición**, con constructor de candidatos, inputs permitidos, filtros, política, costo, seguridad y unidad de campaña trazables.
- el **contrato de escala**, con kernel, resoluciones, base de operadores, observables, equivalencias y error de truncación;
- el **contrato de tarea perceptual**, con estudio, aparato, especie/cohorte, individuo, estímulo, ruido y decisión jerarquizados.
- el **contrato de familia proyectiva**, con índice de escala, leyes, kernels de restricción, sampler autorizado, coupling, equivalencias y horizonte de extendibilidad;
- el **contrato de operador tropical**, con estatuto exacto/asintótico/PWL, semiring, gauge, soporte, temperatura, active sets y residuos o fases observables.
- el **contrato de completación identificable**, con constraints públicas, fibra de observación, target oculto, detector de aplicabilidad, solver y regla de abstención;
- el **contrato de exchange identificable**, con probes, catálogo de órbitas, version space, query retenida y checker racional antes de cualquier entrenamiento.
- el **contrato de operador espacial**, con dominio, medida, frontera, discretización, Laplaciano, escala temporal, observable espectral, equivalencia y query declarados;
- el **contrato de operador dinámico**, con ley, observables, excitación, horizonte, política, subespacio, mapa/kernel/generador, solver y alcance de control separados.

Esta distinción impide equiparar harmonicidad con consonancia, resonancia con proximidad racional o una fórmula exacta con una preferencia humana. También obliga a que splits y bootstrap operen sobre sistemas, participantes, especímenes o escenas, no sobre ventanas y pares correlacionados.

## Consecuencia arquitectónica

La investigación no respalda una mega-arquitectura inmediata. Respalda mecanismos con jurisdicciones definidas.

Para cocientes de escala, una transformación `clr/ilr` seguida de un MLP es un baseline fuerte: si resuelve la tarea, una red que “aprende invariancia” sólo agrega valor si generaliza mejor o recupera una estructura que la transformación cerrada no entrega. Para composición, un solver exacto, una red de pares sin mezcla, una mezcla local param-matched y un triangle-shuffle son controles necesarios antes de atribuir capacidad a una actualización triangular. Para dinámica, un encoder temporal simple precede a una red hamiltoniana o a un neural operator. Para partición, el reader que estima cardinalidad debe separarse del encoder que produce relaciones.

La singularidad posible de una PPU se vuelve así más precisa. No consistiría inicialmente en poseer una métrica universal, sino en poder:

1. representar acciones y clases de equivalencia;
2. componer relaciones locales;
3. integrar evidencia temporal parcial;
4. recuperar estructura global y cardinalidad;
5. expresar incertidumbre y abstención;
6. transferir la operación sin copiar el contenido del dominio.
7. operar bajo una geometría cuyo tipo y autoridad fueron declarados externamente, o inferir una rama calibrada sólo desde evidencia deployable.
8. separar ley y operador dinámico del integrador que los ejecuta.
9. separar wiring, semántica y constitución, y leer sistemas abiertos por su equivalencia observable.
10. declarar qué consultas son observacionales, predictivas, interventionales o contrafactuales y qué estructura permanece sólo módulo gauge.
11. proponer relaciones en una IR tipada y mantener separados solver, checker, claim formal y autoridad física.
12. adjudicar medición y adquisición sin confundir referencia, calibración, valor informativo y mérito arquitectónico.
13. ejecutar o descubrir mapas entre escalas sin confundir kernel dado, truncación y error aprendido.
14. contrastar sensibilidad perceptual a razones sin confundir representación, ruido, decisión y magnitudes continuas.
15. distinguir un cociente numérico de una razón empíricamente significativa, separando factibilidad finita, teorema de representación, meaningfulness, verdad y autoridad.
16. aceptar una macrovariable como causal sólo dentro de una jurisdicción donde estados e intervenciones conmutan, sus realizadores son intercambiables y la cobertura no se obtiene por colapso.
17. distinguir equivariancia dentro de una cardinalidad de coherencia proyectiva entre cardinalidades, densidades o exposiciones.
18. operar en un semiring tropical sólo cuando el dominio lo autoriza, conservando active sets, gauge y abstención ante cancelaciones o anti-dominios.
19. representar orientación, dualidad y menores módulo gauge sin convertir realizabilidad lineal en condición de existencia del objeto.
20. tipar los outputs de información y sus autoridades, separando vector de subconjuntos, PMF conjunta y witness de rango lineal.
21. detectar cuándo una completación está identificada por la observación y abstenerse cuando la misma vista admite targets incompatibles.
22. separar inferencia de una ley local de su ejecución, exigiendo un version space exacto antes de adjudicar discovery.
23. separar cocientes `SO/O`, correspondencia, tamaño y quiralidad antes de atribuir una geometría de forma.
24. exigir realizabilidad global y gates de feature-triviality antes de convertir invariantes conformes locales en un banco neuronal.
25. derivar el tipo de salida desde la respuesta pública: punto, clase, región alcanzable, conjunto compatible o abstención, sin elevar el representante plantado a verdad única cuando pertenece a una fibra observacional.
26. tratar el experimento y la query como parte de la geometría: conservar por separado cociente observacional, orden informacional/deficiencia y visibilidad/estabilidad de componentes.
27. representar respuestas set-valued sin fusionar identificación poblacional, witnesses internos, aproximaciones exteriores, autoridad numérica e inferencia muestral.
28. declarar objeto, observación, equivalencia, geometría y query antes de medir espacios intrínsecos, cocientes o estratos, sin sustituir correspondences generales por biyecciones por comodidad computacional.
29. tratar las razones operatoriales mediante espectros generalizados, gauges y soportes tipados, manteniendo interior SPD, frontera PSD y regularización como objetos distintos.
30. distinguir relación observada, estructura compatible, estructura identificada y estructura autorizada en ultramétricas, tree metrics y jerarquías, sin atribuir al scorer aprendido la autoridad del executor o del checker.
31. distinguir reutilización de método, identidad paramétrica y transferencia operacional entre autoridades, conservando núcleo congelado, adapters, destinos, controles, protocolos separados de replay local y cross-authority, atribución causal y autoridad externa como partes explícitas del contrato.

La Ola 20 vuelve más precisa la factorización que sigue: `objeto/equivalencia -> encoder relacional -> propuesta de operador tipada -> solver/aplicación -> reader de query`, más una salida de competencia/abstención y un ledger externo de autoridad. Es un atlas de interfaces candidatas, no una operación universal ni una arquitectura promovida.

La Ola 21 agrega el recorrido inverso: `contrato de respuesta -> encoder de respuestas/probes -> objeto cociente identificado -> punto | representante certificado | conjunto compatible | UNKNOWN`. La respuesta puede haber realizado ya el cociente; por eso una operación aprendida sólo recibe crédito cuando subsiste variación observable bajo una acción pública. `Response-Quotient Atlas` conserva dos sistemas distintos —inversión tipada y compatibilidad de candidatos— y mantiene generator, checker y ledger de autoridad como factores trazables. Es una candidata aceptada documentalmente, no una arquitectura promovida.

La Ola 22 incorpora el canal de observación a la propia definición geométrica. `Experiment-Relative Geometry Atlas` organiza la interfaz `contrato experimental -> ley/operador observable -> cociente + orden informacional + campo de visibilidad -> recovery autorizado -> salida tipada`. Un mismo cociente puede corresponder a experimentos con riesgos distintos, y una unicidad global puede coexistir con direcciones inestables. Completion por prior queda fuera del estimando de recovery. La candidata fue aceptada documentalmente después de una auditoría independiente, pero no fue implementada ni promovida.

La Ola 23 vuelve explícita la geometría de una respuesta no puntual. `Identified-Set Authority Stack` organiza dos circuitos que no se autorizan mutuamente. El primero parte de propuestas inner/outer, pasa por verificación exacta o intervalar y produce un ledger poblacional; el segundo parte de un estimador del conjunto, calibra una región muestral y verifica qué objeto cubre y bajo qué cuantificador. El posible aporte aprendido queda restringido a un campo de compatibilidad condicionado por contrato. Checkers, solvers y ledgers son autoridades externas: pueden volver más honesto al sistema sin demostrar por ello una singularidad neuronal. La candidata fue aceptada documentalmente tras dos reauditorías, pero no fue implementada ni promovida.

La Ola 24 pregunta cómo conservar esa autoridad cuando el conjunto atraviesa una transformación. La inclusión exterior antecede a cualquier medida de precisión: `top` puede contener todo y no responder ninguna query, mientras una envolvente pequeña pierde toda validez si excluye un estado compatible. `Guaranteed Set Transformer Stack` organiza esta tensión como una cadena entre semántica concreta, dominio, propuesta, checker, composición, ledger de pérdidas y reader. Su primer protocolo no mezcla las cuatro posibilidades de aprendizaje: mantiene dominio, reducción e implícito fijos y cambia sólo el transformer. La propuesta aprendida, la clásica y una sham pasan por checker y repair comunes; si el wrapper reconstruye la salida, el mérito posible se reduce a amortización. La candidata fue aceptada documentalmente tras cerrar ocho findings, pero no tiene checker implementado, ejecución ni promoción.

La Ola 25 retrocede un paso necesario: antes de preguntar cómo transformar un conjunto hay que decidir qué significa que dos conjuntos estén cerca, converjan o preserven una cantidad. Esa autoridad no viene dada por almacenar ambos como nubes de puntos. Un cuerpo convexo full-dimensional, un conjunto cerrado no acotado, el epígrafo de una función y el grafo de un operador viven bajo objetos, gauges y nociones de convergencia diferentes. `Set Geometry Authority Contract` vuelve explícitos ambiente, dimensión afín, regularidad, observación, vacío/infinito, operación, topología o métrica, reader y checker. Su arquitectura no empieza por una capa neuronal: empieza por una fase de schema y autoridad sin modelos. Sólo después compara representaciones sobre inputs idénticos, con un executor clásico común y un ledger que separa validez, informativeness y costo. La aceptación fue documental; el registry de bancos existe como especificación, pero ninguna instancia puede materializarse hasta congelar implementación, hash, tolerancia, hardware y costo.

La Ola 26 muestra que esa jurisdicción comienza todavía antes: el objeto mismo debe quedar tipado antes de elegir su geometría. Un espacio métrico-medido, un cociente por una acción y un objeto estratificado no son tres codificaciones de un mismo tensor. `Intrinsic Object Authority Contract` declara observación, equivalencia, query y autoridad, y usa la API común como dispatch hacia procedimientos distintos. Cuatro anclas checker-only materializaron esta diferencia: GH exacto por correspondences no coincide en general con bottleneck biyectivo; leyes de distancias de bajo orden pueden no separar grafos no isométricos; y los cocientes espejo y ortante exigen declarar la acción que identifica estados. Las corridas byte-identical adjudican esos contratos finitos. No adjudican las diecisiete entradas todavía propuestas ni una representación neuronal común.

La Ola 27 encuentra proporciones operatoriales precisas sin convertirlas en cocientes escalares. En conos positivos, Hilbert compara rayos y elimina escala, mientras Thompson conserva componente radial; en operadores SPD, la relación primaria entre un par ordenado es el multiset de autovalores generalizados o una función simétrica de su log-espectro. `Positive/Operator Authority Contract` mantiene separados objeto, soporte, gauge, acción, reader y operación. La frontera PSD no hereda automáticamente los readers del interior, y sumar `epsilon I` modifica el objeto. El posible aprendizaje queda restringido a readers, selectors, proposers o políticas de frontera; eigensolver, cálculo funcional, solver y checker retienen autoridad externa. Sus veintidós bancos permanecen diseñados y no ejecutados.

La Ola 28 aplica esa misma disciplina a jerarquías. Una relación ultramétrica observada, una familia de completaciones compatibles, un split identificado y una estructura autorizada ocupan niveles distintos. `Hierarchy/Tree Authority Contract` codifica veintisiete variantes completas en vez de combinar exactitud, missingness, ruido, actividad y estadística mediante flags libres. Cinco suites materializaron veintidós suite-bank IDs. La primera ejecutó trece contratos sobre ties, missingness, root, nodos de grado dos, aristas cero, compatibilidad global, rango y política numérica. La segunda agregó dos falsadores: una misma jerarquía ordinal etiquetada puede provenir de presentaciones p-ádicas con primes, alturas y valuaciones distintos, y una familia de métricas de cuatro puntos puede aproximarse a un límite arbóreo mientras su tight span cambia de dimensión. La tercera separó observación y repair: la ultramétrica subdominante de una tripleta no coincide necesariamente con su proyección óptima en norma infinito, y el defecto nulo de cuatro puntos no reemplaza al margen de resolución. La cuarta incorporó tres fronteras algebraicas: una presentación finita de building depende de una base elegida, un gauge puede conservar jerarquía ordinal mientras cambia realización y un input no métrico debe ser rechazado antes de toda inferencia estructural.

La quinta suite vuelve explícita la autoridad de una observación parcial. Una máscara puede identificar de forma única los pesos de arista **dada** una topología pública y, sin embargo, ser compatible globalmente con dos topologías positivas inequivalentes. Del mismo modo, dos máscaras con igual número de cords pueden tener autoridad opuesta: un minimum triplet cover identifica topología y pesos, mientras una máscara diseñada de igual cardinalidad deja pesos y topologías múltiples. Cardinalidad observada, identificabilidad condicional e identificabilidad global son propiedades diferentes. El ladder resultante conserva seis estados entre compatibilidad, topología y pesos; impide que un score aprendido o una topología propuesta se presente como certificado. Las corridas canónicas cerraron `2/2` baselines, `22/22` mutaciones y `8/8` guards con resultados byte-identical, y la reauditoría independiente final no dejó findings sustantivos.

La contabilidad exige dos escalas. Las cinco suites ejecutaron veintidós contratos propios, mientras el atlas raw heredado contiene treinta y tres bancos R57/R58. Como las suites comprimen, reutilizan y cruzan esos bancos, `22/33` sólo conserva valor como índice histórico no teórico-conjuntista: no define once IDs restantes. La cobertura raw canónica es `9 fully adjudicated / 20 partially covered / 4 not adjudicated`; el edge-weight lasso quedó adjudicado para un caso exacto de seis etiquetas, mientras el contraste de triplet covers cubre sólo el mínimo `n=5` y no la familia aleatoria mayor. Los módulos aprendibles pueden ordenar una búsqueda, puntuar una resolución local, elegir una consulta o anticipar un riesgo acotado; no emiten por sí mismos el certificado ni el estado terminal. La consecuencia programática se vuelve más concreta: cuando el método clásico agota el estimando, la singularidad neuronal debe buscarse en propuesta, adquisición, lectura o abstención, no en reemplazar ornamentalmente al solver. Estas suites no ejecutan modelos, no demuestran una geometría universal y no deciden promoción ni GO/NO-GO.

La Ola 29 muestra que esa autoridad también depende de convenciones que suelen parecer detalles de representación. En una tree metric semi-labeled, una etiqueta observada puede ocupar un nodo interno y no debe ser convertida silenciosamente en hoja por medio de una arista pendant de peso cero; por eso los roles del output bruto se validan antes de toda contracción o comparación canónica. En el régimen unweighted, una longitud unitaria y una realización por subdivisión no son la misma afirmación si la unidad no fue fijada de antemano. Y una reescala positiva o un relabeling pueden preservar topología y exactitud arbórea sin constituir igualdad literal ni preservar el subtipo de arista unitaria. B-03, B-06 y B-16 quedan así especificados como guards checker-only. La auditoría R87 corrigió el orden de canonicalización y la frontera entre especificación y adjudicación; R88 cerró esa reauditoría sin findings sustantivos. Los checkers aún no fueron implementados, la cobertura raw no cambia y ninguna de estas convenciones justifica por sí sola una primitive neuronal.

La Ola 31 incorpora una geometría algebraico-estadística que vincula, sin confundir, dos usos de una misma matriz entera `A`. Esa matriz puede parametrizar un modelo log-lineal o tórico y, bajo una estadística suficiente observada, definir fibras discretas y movimientos en su kernel. La continuidad algebraica no vuelve equivalentes la pertenencia o clausura del modelo, la igualdad del estadístico, la conectividad de una fibra, la irreducibilidad de una cadena, su estacionariedad o su tiempo de mezcla. El contrato resultante tipa por separado `A`, el esquema de muestreo, el objeto de modelo, `FiberSpec`, el certificado de finitud, los movimientos algebraicos, las transiciones factibles, la masa de propuesta y el kernel Metropolis-Hastings efectivamente ejecutado. `Toric Constraint Router`, `Certified Fiber Explorer` y `Algebraic Proposal Machine` quedan preservadas como arquitecturas candidatas: un scorer puede ordenar movimientos o elegir una ruta de solver, pero el checker, la aceptación MH y el ledger de autoridad permanecen externos. La auditoría independiente R94 y las reauditorías R95-R96 cerraron los defectos de atribución causal, fallback, finitud, trazabilidad y novedad. No se ejecutaron bancos ni modelos y no hubo promoción arquitectónica ni decisión GO/NO-GO.

La Ola 32 introduce una geometría de redes de reacción que hace visible una estratificación especialmente fértil para el problema proporcional. Una realización de reacción determina matrices de complejos e incidencia, un subespacio estequiométrico y hojas compatibles; la cinética agrega un campo sobre esas hojas; los balances autorizan clases distintas de equilibrio; y la semántica estocástica agrega un generador, una lattice de estados, clases comunicantes, una composición local de saltos o canales y un reloj físico. Ninguna de esas capas reemplaza a las demás. El mismo diagrama puede sostener claims deterministas y estocásticos diferentes, y una observación parcial puede identificar sólo una clase de realizaciones compatibles. Por eso el contrato separa `p_jump`, identificable desde el generador agregado, de `p_channel`, que requiere un manifiesto de canales, y exige convención de propensión, volumen, conversión de constantes, scaling, horizonte y error antes de vincular ODE y CTMC. También conserva un límite de autoridad decisivo: la Global Attractor Conjecture general queda como propuesta de prueba y no se usa como teorema asentado ni como fuente de labels; sólo pueden emplearse casos especiales publicados cuando sus hipótesis estén codificadas y verificadas. La auditoría independiente corrigió cinco fallas de autoridad, leakage y cobertura; la reauditoría focal confirmó las correcciones sin findings nuevos sustantivos. La suite checker-only permanece diseñada, no materializada ni ejecutada. `Stoichiometric Authority Router`, `Compatibility-Class Dynamics Network`, `Equivalence-Aware Reaction Proposal Network`, `Propensity-Simplex Generator` y `Stochastic Reaction Explorer` quedan preservadas como alternativas, no promovidas. La singularidad posible ya no reside en “usar mass-action”, sino en mantener autoridad, equivalencia, abstención y el puente determinista-estocástico como tipos operativos cuya utilidad deba demostrarse frente a compiladores y baselines equivalentes.

La Ola 33 no agregó otra familia por acumulación. Sometió el corpus a una búsqueda adversarial de vacíos y separó cuatro preguntas que antes podían confundirse: si un dominio aporta una autoridad nueva, si existe una primitive aprendible no redundante, si otro carril independiente la corrobora y si el contrato completo de evidencia está cerrado. Termodinámica estocástica de trayectorias y gauge curvo ampliaron la cobertura de autoridad; reducción de bases reticulares y composición `boxplus` bajo paridad XOR aportaron contratos operacionales sostenidos por un solo carril. Ninguna candidata obtuvo corroboración afirmativa independiente de ambos. Por eso el cierre conserva las cuatro alternativas, pero no selecciona dominio, no registra `A19`, no abre un nuevo `P2*` y no promueve una arquitectura. La falta de convergencia cruzada es una propiedad de la evidencia disponible, no una prueba de inexistencia.

La Ola 34 completó el examen que faltaba y volvió más precisa esa ausencia de convergencia. Termodinámica de trayectorias y gauge curvo retienen autoridades físicas que el atlas no absorbía, pero no una primitive aprendible nueva en las consultas auditadas: cuando el estado relevante está completamente observado, el cálculo exacto agota la operación; cuando falta información, el residuo cae en estimación genérica, inferencia condicional, prior art o no identificación. Reducción reticular y `boxplus` retienen otro tipo de valor. La primera puede alojar una policy que proponga transformaciones legales o priorice un solver, pero lo hace dentro de una órbita unimodular, una métrica y un executor ya autorizados. La segunda ofrece una identidad local exacta y un banco controlado para estudiar loops, damping, schedule y estabilidad numérica, pero especializa sum-product sobre factores XOR. Ninguna de las cuatro combina todavía una jurisdicción nueva con una operación aprendible nueva.

Este cierre introduce una disciplina conceptual que afecta al diseño de la PPU. Un contrato puede estar completamente especificado y, aun así, no cumplir sus condiciones de admisión; la corroboración pertenece a cada claim y no a la candidata considerada como bloque. Por eso una ciencia puede ampliar autoridad sin producir una arquitectura, y una policy puede tener utilidad computacional sin fundar una ontología proporcional nueva. Las cuatro alternativas se preservan con discriminantes separados de autoridad/no redundancia y de efecto aprendido, pero no se registra `A19`, no se abre suite ni se selecciona dominio.

La Ola 35 desplazó el problema desde la novedad aislada hacia la transferencia. Reusar una receta en otro dominio no demuestra que una misma primitive haya cruzado de autoridad; conservar parámetros tampoco basta si cambia el evaluator, la interfaz o el acceso a evidencia. Los casos estudiados se repartieron en dos mitades. Algunos sistemas trabajan frente a autoridades diversas, pero no conservan un núcleo identificable cuyo comportamiento pueda reproducirse causalmente entre destinos. Otros conservan identidad paramétrica fuerte, pero permanecen dentro de una autoridad estrecha o común. Ninguno reunió el contrato completo de transferencia operacional.

La forma arquitectónica que sobrevive a ese resultado es más modesta y, por eso mismo, más precisa. Un proposer produce candidatos en una representación tipada; un evaluator y un checker externos conservan la autoridad del dominio; la abstención delimita dónde la policy no tiene licencia. Este loop queda registrado como patrón recuperable, no como arquitectura promovida. Su valor actual es contractual: permite preguntar qué permanece idéntico, qué cambia con el dominio y qué ventaja no queda agotada por cálculo exacto, estimación genérica o prior art. `EVIDENCE-CHANGING-ACTION` conserva otro estatuto: es una hipótesis contractual de A8 sobre acciones que modifican la evidencia disponible y no forma parte del loop ni de su primer test.

La Ola 36 auditó esa hipótesis separada. Ningún caso revisado reunió en un mismo sistema el contrato completo de una acción que cambia evidencia y luego sostiene una inferencia transferible entre autoridades. La literatura ofrece, otra vez, capacidades parciales: CAD²RL transfiere un núcleo instrumental/perceptual desde render a cámara y vehículo reales, pero no documenta una adquisición epistémica completa ni replay causal entre autoridades; A-Lab y CPBE ejecutan ciclos de acción, medición externa y actualización, aunque sin transferir un núcleo congelado; DAD, Step-DAD y Pang et al. amortizan o adaptan la selección experimental, pero no cruzan autoridad material bajo un holdout de autoridad. El resultado no invalida A8: impide presentarla como primitive establecida.

Lo que sí queda establecido es el contrato que una prueba futura debería satisfacer: `estado público de evidencia -> acción legal -> indicación externa -> resultado de medición -> actualización -> claim o abstención`. La autoridad conserva instrumento, calibración, ley de observación, modelo de medición, updater, reader, costos y gates. Replay factual, evaluación off-policy y simulación contrafactual son regímenes distintos; ninguno puede sustituir silenciosamente a los otros. `EVIDENCE-CHANGING-ACTION` permanece así como alternativa arquitectónica recuperable, en estado `UNRESOLVED`, sin dominio, suite, modelo ni promoción.

La Ola 37 examinó una operación todavía anterior: inducir la relación o la acción
que organiza las transformaciones. Su cierre separa Track E —relación,
partición o identified set— de Track A —flechas, aplicación, dominio y
composición— porque el mapa de una acción a sus órbitas e invariantes pierde
información. El positivo Track E exige transiciones indexadas por acciones
observadas, observación inyectiva hasta equivalencia de interacción, acciones
disponibles puras respecto de un único factor, una condición de composición
acotada, un mundo finito con todas las transiciones y un mínimo global. Sólo bajo
ese régimen la partición de las acciones disponibles por factor es identificable,
y sólo hasta permutación de etiquetas de factor, equivalencia de interacción e
isomorfismo latente. La acción completa no enumerada permanece `UNRESOLVED`: el
positivo Track E no autoriza flechas Track A.

El ledger empírico normalizó quince records por régimen, target y gauge. Ninguno
reunió todos los requisitos aplicables en el corpus acotado, ni apareció
action-family OOD o authority holdout. Este negativo no establece imposibilidad.
Delimita una alternativa recuperable: un inductor tipado con estado relacional,
cabezas Track E y Track A, scope, gauge, dominio y abstención, sometido a
checkers algebraico, relacional y fenomenológico externos. Su estatuto es
`NO-A19`; no constituye suite, modelo ni arquitectura promovida.

La Ola 38 preguntó cómo componer claims entre autoridades sin confundir
compatibilidad formal con verdad, ni confianza con jurisdicción. La búsqueda
recuperó formalismos parciales para gluing, conflicto, provenance, lineage,
dependencia, retractación, incertidumbre, pooling y transporte. Sin embargo, los
`24` casos fuente y sus `264` estados MC no materializaron las claves canónicas
completas `9/7/10` del contrato v1. El cierre no distribuye esos casos entre gaps
y soluciones: deja `0` celdas canónicas y `0` clases adjudicables.

Este resultado establece una disciplina anterior al diseño neuronal. La unidad
de claim, la instancia de composición y la celda de adjudicación deben existir de
forma canónica antes de preguntar por reducción, residuo o aprendizaje. Un
formalismo que resuelve un subproblema conserva valor local, pero no puede recibir
crédito por una composición cuyo target, scope, dependencia o autoridad no están
materializados. R151 verificó esa frontera mediante un fixture positivo, `42`
tests negativos y un join exacto entre casos estructurados y ledger. La
composición tipada queda investigada pero no materializada: no crea `A19`, suite,
modelo, nuevo `P2*` ni decisión experimental.

La Ola 39 preguntó por una mediación todavía anterior: qué autoriza a tratar
dos observaciones como términos de una proporción. Los tres carriles separaron
quantity records metrológicos, correspondencia incierta e identificabilidad
módulo equivalencias. Su convergencia introduce cinco actos de autoridad que no
deben fusionarse: detectar candidatos, asociarlos, tipar cantidades, calibrarlas
y formar una ratio. Cada acto puede resolverse sin autorizar el siguiente. Una
correspondencia puede ser set-valued; un target puede existir sólo como clase de
equivalencia; dos valores adimensionales pueden seguir refiriendo a quantities,
frames o dependencias incompatibles.

El corpus conservó `19` casos fuente y `27` registros bibliográficos, pero
ningún caso materializó la unidad contractual completa. El cierre tiene `0`
celdas canónicas y `0` adjudicaciones `TG`. Esta ausencia no demuestra que el
grounding proporcional sea imposible: impide convertir una plantilla
bibliográfica en ground truth. La consecuencia de diseño es una hipótesis
estratificada —nodos tipados, aristas probabilísticas o set-valued y espacios
cocientados por la observation law— que todavía no selecciona arquitectura,
suite o dominio.

La Ola 40 preguntó qué ocurre después de que un claim ya fue grounded: cómo
revisarlo bajo evidencia nueva sin perder la identidad del objeto, la autoridad
de la decisión ni la posibilidad de reconstruir qué cambió. La revisión formal
de creencias, la crítica de modelos, el provenance versionado y la adaptación
aportan piezas compatibles con ese problema, pero no son intercambiables. Una
actualización bayesiana puede conservar coherencia sin registrar autoridad; un
cambio de schema puede volver incomparable el estado anterior; un fine-tuning
puede modificar conducta sin localizar qué claim fue revisado; y un diff puede
describir versiones sin probar que la revisión fue epistemológicamente válida.

El inventario auditado reunió `47` fuentes, `37` casos tipados —`25` formales,
`3` materiales y `9` contrastes independientes— y `12` relaciones. Ningún
episodio material cerró la cadena completa: los tres conservan `C2M=NO`, los
seis enlaces de stack carecen de crédito documental y los cuatro ledgers de
decisión, replay y materialización permanecen vacíos. La inferencia recuperable
es un **ledger versionado de revisión contractual** que preserve contrato
anterior, falsificador autorizado, decisión pre-outcome, contrato posterior,
diff localizado y replay de los claims retenidos. Su estatuto es conceptual:
`NO-ARCHITECTURE / NOT-EXECUTED`, sin suite, dominio, modelo ni decisión
experimental.

La Ola 41 examinó el álgebra geométrica/Clifford como candidata para expresar
de manera nativa una geometría proporcional. El formalismo ofrece una unidad
real: magnitudes, direcciones, áreas orientadas y transformaciones pueden
participar de un mismo sistema de operaciones. Pero esa unidad depende de un
contrato explícito. Firma, degeneración, base, orientación, dualidad, grades y
gauge determinan qué producto se está ejecutando; no son detalles que puedan
elegirse después de observar el rendimiento.

El resultado decisivo es una reducción exacta. Fijado el contrato, el producto
geométrico se escribe como `z_C = sum_{A,B} C_AB^C x_A y_B`. El tensor de
estructura contiene los signos, ceros y acoplamientos del álgebra, de modo que
una contracción tensorial ordinaria puede ejecutar la misma aplicación. La
hipótesis arquitectónica cambia entonces de estatuto: Clifford no promete una
función inaccesible a otros modelos; podría aportar una factorización tipada,
sparse y compartida que mejore eficiencia muestral, estabilidad u OOD cuando la
firma corresponde al fenómeno.

Las `47` URLs auditadas acreditan autoridad matemática, ejecución material y
arquitecturas equivariantes, pero no una ventaja aprendida aislada. Falta un
gemelo tensorial exacto con la misma interfaz, acción, gauge, capacidad, cómputo
y evaluación. Por eso la contribución aprendida permanece `UNVERIFIABLE` y la
transferencia entre autoridades, `TRANSFER-UNVERIFIED`. El paquete recuperable
conserva un ejecutor Clifford tipado, el control tensorial obligatorio y
comparadores steerable/G-equivariant, attention y tensor products. No selecciona
firma, dominio, suite, modelo ni arquitectura.

La oportunidad arquitectónica más concreta es factorizar una **capa de grounding de objetos, cantidades, unidades y entidades**, un **operador estructural** —incidencia, estequiometría, simetría o restricción—, **canales y gauges tipados**, **módulos constitutivos o autoridades geométricas locales**, diagnósticos de **existencia, ambigüedad e identificabilidad**, un **proposer con IR tipada**, un **solver/compilador instrumentado**, un **checker independiente**, un adjudicador causal/físico, un **ledger versionado de claims y revisiones** y un **reader de equivalencia, coupling, partición o respuesta terminal**. Para dinámica aparece un atlas candidato `observación -> estado -> generador -> operador geométrico -> solver`; para sistemas abiertos, un compositor `wiring -> semántica -> constitución -> compilador -> frontera`; para relaciones formales, una cadena `proposer -> IR -> solver -> checker` cuyos estados permanecen separados. La Ola 37 agrega una escalera anterior a esa cadena: una cabeza Track E sólo puede proponer relaciones o particiones; una cabeza Track A sólo recibe crédito cuando materializa flechas, aplicación y composición bajo scope y gauge declarados. La novena ola agrega `A8`, un adjudicador metrológico y controlador de evidencia que primero debe existir como protocolo externo GUM/OED/system-ID. La décima registra `A9`, un operador tipado de escala con dos jurisdicciones: `KNOWN-KERNEL-TRANSPORT` adjudica ejecución y composición bajo un mapa común; `KERNEL-DISCOVERY` exige producir el mapa desde acceso microscópico idéntico al de los controles. La undécima no agrega una mega-arquitectura: `P2l` funciona como firewall de significatividad y `P2m` como firewall de abstracción causal. `A10` queda como especialización causal experimental de A7/A9. La duodécima conserva esa prudencia: `A11` es por ahora una **especialización proyectiva de A9**, cuyo único componente atribuible es una constraint de conmutación sobre un forward compartido; `A12` es un **bloque max-plus proyectivo estrecho**, no una arquitectura tropical general. La decimotercera registra `A13` como bloque relacional candidato sobre factores de Grassmann–Plücker y `A14` sólo como interfaz: sus heads DSF, PMF y lineal permanecen candidatos independientes porque no existe un lift común que pueda adjudicarse sin mezclar targets y privilegios. Una ReLU compilada que ejecuta el mismo máximo es control de equivalencia funcional, y las variedades tropicales completas permanecen como generadores, oráculos o checkers. Los oráculos quedan fuera del estimando y fixed points, subespacios o mapas plantados sólo reciben estatuto fuerte después de demostrar convergencia o identificabilidad dentro de su alcance. `A17/P2t` agrega un bottleneck de log-ratios sobre constraints incidentes, pero limita su claim a esa transformación dentro de una arquitectura de cuñas fija y obliga a estimar aplicabilidad. `A18/P2u` agrega un executor local de exchange sólo después de que un checker exacto demuestre que los probes identifican una órbita de columna. Todos necesitan una rama residual o abstención cuando falta autoridad. Son hipótesis registradas, no modificaciones aprobadas.

Natural Harmonic Geometry designa la hipótesis posterior: que algunas de estas operaciones reaparezcan con estabilidad suficiente entre dominios físicos como para hablar de una organización transversal. Esa recurrencia todavía debe demostrarse.

## Programa experimental derivado

El programa inicial queda escalonado para que cada fallo tenga una localización interpretable.

### 1. Acciones, ciclos y conservación exactos

El primer prototipo contiene tracks separados. Un track usa matrices dimensionales y cambios de unidad para evaluar subespacios Buckingham, equivalencia entre bases `Pi` y covariancia. Otro usa composiciones positivas y grafos de log-ratios para evaluar órbitas de escala, potenciales hasta gauge y ciclos corrompidos. Un tercero usa estequiometría y redes resistivas para separar balance, ley local y respuesta global. Un cuarto usa complejos exactos y sampling sheaves para distinguir compatibilidad, constitución, existencia, ambigüedad, obstrucción y ruido. Un quinto compara geometrías estadísticas y de medidas tipadas. Un sexto cruza prior dinámico con solver común/nativo. Un séptimo separa wiring, semántica y constitución. Un octavo cruza consulta, identificabilidad y anclaje físico. Un noveno cruza `PPU±solver` y `control±solver` sobre la misma IR y checker. Un décimo (`P2i`) adjudica metrología; un undécimo (`P2j`), adquisición activa. Un duodécimo (`P2k`) separa transporte con kernel conocido de discovery de kernel, con oráculos fuera del estimando y convergencia prospectiva. Un decimotercero (`P5e`) cruza representación, observación/ruido y decisión/readout sobre psicofísica externa. `P2l` separa factibilidad finita, witness, aplicabilidad teoremática, meaningfulness, verdad y autoridad. `P2m` separa chequeo, ejecución y discovery causal, con privilegios tipados y controles contra identidad, singleton, reescala y exclusión de acciones difíciles. `P2n` cruza modelo, familia generadora y sampler para separar equivariancia, conmutatividad paired, projectivity en ley y extendibilidad. `P2o` mantiene protocolos independientes para tropicalidad exacta, dequantización y ajuste PWL, con hard negatives de cancelación, gauge y semiring equivocado. `P2p` separa validez orientada, realizabilidad y conmutación de menores bajo ablaciones de una sola operación. `P2q` mantiene independientes los tracks vectorial, probabilístico y lineal, con estados de inferencia finita que no convierten error del estimador en propiedad de la población. `P2r/P2s` aíslan ejecución polinómica y scoring assignment bajo observación parcial. `P2t` compara ratio y sham sobre la misma arquitectura de cuñas y puntúa error sólo donde un detector común declara completación aplicable. `P2u` exige un manifest racional de componentes identificables antes de comparar inferencia de columna y ejecución de exchange. `P2v` diagnostica conformidad a cocientes de forma sin promover una arquitectura; el carril de factibilidad conforme permanece bloqueado antes de `A19/P2w`. `P2a-G` cruza evidencia raw/residual con mixer genérico/group-typed bajo DAG y whitelist comunes; phase/closure conserva un preflight separado de objeto y aparato. Las suites de Olas 18 y 19 auditan representación orbital, autoridad de filtración, autoridad armónica y estado predictivo antes de abrir cualquier nuevo `P2*`. En el último caso, el contrato incluye además acción, muestreo, política, inicialización, estacionariedad, soporte de historias y pesos de mezcla: sin una ley de proceso cerrada, dos modelos no están prediciendo necesariamente el mismo futuro. Las métricas permanecen por objeto y por track, y cada brazo declara evidencia, capacidad, compute, firma dimensional, normalización, acceso, solver y reader.

La Ola 20 extiende ese preflight a la autoridad operatorial espacial y dinámica. Antes de abrir otro `P2*`, exige declarar observable, semigrupo o generador, discretización, excitación, solver y query, y separar predicción de identificación o control. La Ola 21 exige además que el target permanezca constante sobre cada clase observacional, que los candidates provengan de un generator público congelado y que una reconstrucción compatible no sea confundida con el mecanismo físico plantado. La Ola 22 agrega un `PROTOCOL-LOCK` anterior a masters, pilot y fit, authority types para exactitud algebraica, certificación numérica y near-null, y una unidad raíz que mantiene juntos aparatos, masks, gauges, queries y derivados. La Ola 23 agrega bancos exactos y muestrales separados, gates no compensables para inner/outer/exact, una genealogía explícita de coverage y controles contra regiones universo-total que cubren sin informar. La Ola 24 agrega `CHECKER-AUTHORITY-v0`, un factorial proposal×postprocess, licencias locales/universales de composición y una raíz que mantiene los brazos pareados dentro de la misma instancia concreta. La Ola 25 agrega `SCHEMA-AND-CHECKER-AUTHORITY-v0`, contrato externo byte-identical entre brazos y un registry clásico obligatorio antes de preguntar si un reader aprendido aporta algo. La Ola 26 agrega una suite mínima checker-only ya ejecutada y un firewall entre dispatch de objeto y forward aprendido. La Ola 27 exige output regimes y separación interior/frontera antes de comparar N1–N4. La Ola 28 exige variantes atómicas, genealogía por raíz observacional y separación estricta entre propuesta aprendida, executor y checker antes de materializar cualquiera de sus treinta y tres bancos. La Ola 29 agrega validación de roles crudos, políticas de arista cero tipadas, unidad fija y schemas cerrados antes de toda canonicalización o lectura de equivariancia. La Ola 31 agrega `FiberSpec`, certificado de finitud y una separación entre movimiento algebraico, transición factible, propuesta y kernel MH ejecutado. Su primer discriminante debe ser checker-only; cualquier scorer posterior se compara con baselines deterministas y localmente balanceados bajo splits por matriz, familia y especificación completa de fibra, sin usar features oraculares y con presupuesto multidimensional. La Ola 32 agrega `ReactionSystemSpec`, una jurisdicción separada para queries deterministas, estocásticas y de puente, y vistas distintas para conformance y despliegue. Su primer discriminante posible también es checker-only: antes de cualquier router aprendido debe materializar certificados, equivalencias y abstenciones, congelar el manifest anti-leakage y comparar contra compiladores deterministas y portafolios de reglas. La Ola 33 agrega un preflight anterior a toda nueva familia: declarar por separado novedad de autoridad, novedad de primitive, corroboración independiente y completitud contractual; un `NOT-ASSESSED` no puede contarse como fallo.

La Ola 34 agrega el cruce obligatorio de esos ejes y dos ledgers separados: especificación contractual frente a admisión, y corroboración por claim. Para ser admitida como una familia nueva que reclama simultáneamente novedad de jurisdicción y de primitive, una candidata debe superar controles independientes sobre ambos ejes y luego demostrar efecto aprendido bajo evidencia igualada. Una autoridad, policy o testbed puede conservar un programa más estrecho, siempre sujeto a una decisión y un plan separados.

La Ola 35 agrega un firewall de transferencia anterior a cualquier suite cross-domain. La authority registry, los destinos, splits, interfaces, adapters, controles, protocolos de replay y políticas de preservación se congelan antes de elegir el destino o entrenar. Después de ejecutar se preservan propuestas, fallos, scores y logs. El test debe separar atribución causal, replay local y replay entre autoridades, y medir por separado residuo computacional y novedad de primitive. Este contrato no selecciona dominio ni modelo: evita que una adaptación posterior o una definición móvil de la tarea fabrique retrospectivamente la transferencia que se quería medir.

La Ola 36 agrega un contrato prospectivo para acciones que cambian evidencia. Antes de entrenar se congelan y hashean autoridades, splits, canonicalizers, adapters, checkpoints, construcción de conjuntos candidatos, costos y reglas de preservación. Durante la ejecución se registran propensiones, soporte realizado, acciones legales, indicaciones crudas, resultados de medición, actualizaciones y abstenciones; la positividad se diagnostica y, si falla, bloquea el claim off-policy en lugar de ser tratada como un objeto configurable. La prueba debe distinguir replay factual, evaluación off-policy y simulación contrafactual, incluir preservación local y entre autoridades, estresar misspecification, medir calibración y cobertura del claim y comprobar funcionalmente que los adapters no absorben la decisión del núcleo congelado. Este protocolo no afirma que la policy exista ni que sea transferible: vuelve falsable esa posibilidad sin conceder al modelo la autoridad de su propia medición.

La Ola 37 agrega una escalera conceptual que todavía no abre experimento. `E0`
pregunta por relaciones finitas bajo evidencia parcial y `unknown`; `A0`, por
acciones finitas desde transiciones; `A1`, por generadores o acciones locales
con dominio y composición verificables; `T`, por transferencia a familias de
acciones retenidas. Cada escalón conserva el máximo objeto identificable y debe
compararse con controles que distingan score, partición, generadores y acción.
La promoción entre tracks depende de evidencia que informe flechas, no del éxito
predictivo ni de una clausura ciega. No se asigna identificador `P2*`, umbral ni
autorización de implementación.

La Ola 39 agrega un preflight de grounding anterior a esos tracks. Toda unidad
experimental futura debe declarar la indicación observada, el objeto o target,
el quantity kind, unidad/frame, calibración, provenance, relación de
correspondencia, incertidumbre y equivalencia admitida por la ley de observación.
Los pares o ratios derivados de un mismo master permanecen anidados; matching
MAP, regularización o reconstrucción no sustituyen un witness de identidad o
identificabilidad. Este contrato todavía no abre suite ni entrenamiento.

La Ola 40 agrega un contrato de revisión posterior al grounding. Cada cambio
debe preservar la identidad del claim, distinguir evidencia de decisión,
registrar quién tiene autoridad para revisarlo, localizar el diff entre versiones
y ejecutar replay sobre aquello que se declara retenido. Un cambio de posterior,
schema, weights o policy no satisface por sí solo esa cadena. El ledger versionado
queda como alternativa recuperable, no como nuevo `P2*` ni como autorización de
entrenamiento.

La Ola 41 agrega un discriminante para geometrías algebraicas aprendibles. Toda
capa Clifford debe compararse con la contracción tensorial exacta del mismo
producto y con comparadores equivariantes que reciban idénticos objetos, acción,
gauge y canonicalización. Los tests adversariales deben cambiar firma, grades,
orientación, degeneración, base y acciones OOD sin alterar el target. Sólo una
ventaja que sobreviva esos controles con capacidad, cómputo y búsqueda igualados
puede atribuirse a la factorización aprendible; usar correctamente el formalismo
o resolver un caso físico no basta.

La Ola 42 agrega una condición anterior a ese discriminante. La observación, la
métrica admisible, el gauge, el target de identificabilidad y el tipo de output
deben recibir autoridad externa antes de que una red proponga una geometría. La
existencia de una realización no fija coordenadas; la identificación de una
clase no elige representante; un cono no determina por sí solo firma y escala;
y un campo SPD construido por una red no hereda autoridad material. El output
máximo puede ser `point`, `class`, `identified set` o `UNKNOWN` según el régimen.
La arquitectura recuperable separa contrato, proposer, solver, canonicalizer,
checker y reader, y conserva estado pre-checker y post-checker. Todavía no abre
suite, dominio, `A19` ni entrenamiento.

La Ola 43 introduce la mediación que vuelve operativa esa arquitectura. Una
geometría no precede a la relación que la autoriza: observación y ley, bajo un
régimen declarado, permiten construir un símbolo o una relación característica;
su factorización, hiperbolicidad y compatibilidad deben ser examinadas por
witnesses externos antes de habilitar un ejecutor. El resultado puede ser una
clase conforme o proyectiva, un representante sólo cuando clocks, volumen o
medida autorizan la escala, un cono único, varios conos, una variedad
higher-order, una forma degenerada o un conjunto identificado. Cuando ninguna
de esas salidas queda justificada, el estado correcto es `NONE_AUTHORIZED`, no
la geometría que mejor optimice una métrica downstream.

La evidencia formal y material sostiene esa pluralidad, pero el carril
experimental todavía no materializa el circuito completo. De `15` obligaciones,
`10` quedaron cubiertas y `5` como `CONTRACTUAL-GAP`; cuatro de estas últimas
pertenecen al benchmark: faltan artefactos característicos auditables, un
contraste Clifford/tensor completamente matched, abstención pre/post-checker y
OOD de lineage independiente. Por eso proposer/router, scale/gauge head,
identified-set head, ejecutores multigeometría y harness externo de autoridad se
preservan como alternativas separadas, no como módulos ya promovidos de una PPU.

La Ola 44 convierte esos faltantes en condiciones de adjudicación. Un benchmark
no es independiente porque su test tenga otro nombre, otro sitio o parámetros
nuevos: necesita un grafo de lineage que declare generadores, aparatos,
curaciones, autoridades y ancestros compartidos. Tampoco basta con que el
pipeline produzca una salida válida. La propuesta anterior al checker debe
persistirse con su hash y su vista de entrada; repair, canonicalización y solver
deben emitir artefactos sucesores y deltas; cada componente debe declarar qué
targets o sidecars pudo consultar. Sólo así puede medirse cuánto de la relación
estaba en la representación y cuánto fue reconstruido por una autoridad
posterior.

La misma separación alcanza a la incertidumbre. `IDENTIFIED_SET`,
`NON_IDENTIFIABLE`, `OUT_OF_JURISDICTION` y `UNRESOLVED` describen estados del
problema; `PREDICT`, `RETURN_SET` y `ABSTAIN` describen decisiones del sistema.
Riesgo selectivo, cobertura y eficiencia no autorizan por sí solos una ontología
puntual ni una garantía de transferencia. El contrato resultante encadena
lineage graph, split atómico, proposer target-withheld, artefacto pre-checker,
repair delta, access graph, status estructural, objeto point/set, policy e
inferencia por unidad independiente.

El relevamiento de `26` casos, `21` familias y `33` URLs no encontró una
materialización pública que reuniera lineage OOD independiente, contribución
pre/post-checker y abstención o sets bajo shift. Las `12` celdas empíricas
quedaron en `0 COVERED / 3 PARTIAL / 9 CONTRACTUAL-GAP`. El aporte de la ola es,
por tanto, el contrato falsable del benchmark, no un ground truth ya disponible.
Compilador de lineage, proposer con firewall de crédito, cabezas separadas de
status/set/policy, ejecutores con repair observable, estimador de jurisdicción y
harness ciego externo permanecen alternativas diferenciadas y no seleccionadas.

La Ola 45 pregunta si ese contrato existe ya, aunque sea disperso, dentro de
prácticas científicas públicas. La respuesta exige distinguir piezas de un
bundle. Cristalografía, acústica, metrología, astronomía y tomografía aportan
relaciones, aparatos o targets con autoridad material diversa. CAMEO–wwPDB,
SecureDNA y NIST aportan ventanas, retención o testbeds externos. PXRD, SMDP,
DAFx y PANN hacen visibles cadenas entre proposer, solver, checker y
refinamiento. Ninguna de estas propiedades puede prestarse a otra fuente sólo
porque comparta dominio, institución o forma de evaluación.

El corpus normalizó `40` source-cases, `26` familias efectivas y `48` URLs
únicas. Sus `19` contratos adjudicados produjeron `497` predicados y `285`
coordenadas de hard negatives, pero ningún bundle reunió autoridad material,
payload y unidad independientes, sellado actual, lineage, access graph,
artefactos pre/post y controles matched. CSP7 quedó como compatibilidad
histórica parcial y no como join; BIPM, DAFx y PXRD/CCDC muestran anti-joins o
coincidencias de un solo carril.

Este resultado precisa la ontología experimental del ground truth. No es sólo
el valor esperado de una salida: es una composición gobernada de autoridades
sobre observación, relación, target, acceso, propuesta, transformación y
verificación. La cadena necesita poder negar crédito cuando el target estaba
en el payload, el solver reconstruyó la estructura, el split compartió lineage
o el checker recibió información privilegiada. H44-A–F quedan más materiales,
pero no promovidas. La ausencia de un bundle listo define el contrato que debe
construirse; no selecciona una arquitectura ni decide GO/NO-GO.

La Ola 46 hizo explícita la forma de esa construcción sin convertirla en una
elección. Tres carriles relevaron interfaces de identidad y lineage,
evaluación sellada model-to-data y atribución proposer–solver–checker–repair.
Los `34` component cases y `49` URLs únicas produjeron `52` claims integrables,
`36/36` joins materiales y `408` coordenadas de hard negatives. La
adjudicación no encontró una celda constitutiva completa ni un join tricarril:
lo que existe son piezas compatibles bajo obligaciones diferentes.

La singularidad que emerge es un harness tipado de autoridad, organizado en
cinco blueprints cuyas fronteras no son intercambiables. El primero compila en
un mismo manifest identidad, versión, digest, lineage, split y acceso. El
segundo conserva propuesta pre-checker, checker, repair y decisión como
artefactos atribuibles. El tercero gobierna evaluación sellada, ventanas,
queries y revelación entre modelo y datos. El cuarto mantiene separados status
estructural, objeto, jurisdicción y policy: la infraestructura puede persistir
y enrutar esos outputs, pero no define el status verdadero de una relación,
cuya autoridad debe provenir del contrato material del dominio. El quinto
reconstruye crédito causal por brazo y por etapa. Su composición tampoco
autoriza retrospectivamente la observación o el target. Firma no equivale a
independencia, provenance no equivale a lineage OOD, certificado no equivale a
autoridad material y reporting no equivale a crédito causal.

Esta diferencia determina el estatuto de los cinco blueprints de la ola. Son
alternativas prospectivas para construir un aparato de adjudicación, no una
arquitectura neuronal ni un benchmark material ya disponible. La investigación
vuelve más preciso el lugar de una futura PPU: no bastaría con aprender
relaciones; tendría que operar dentro de un contrato que preserve qué objetos
y autoridades vuelven interpretables esas relaciones. Ningún blueprint fue
seleccionado y no hubo decisión GO/NO-GO.

La Ola 47 examinó entonces la autoridad que faltaba en el centro de ese
contrato: la del target material. Materiales certificados, comparaciones,
desafíos prospectivos, cámaras y round robins mostraron que un target no se
reduce a su valor. Su estatuto depende del mensurando, la raíz material, el
método, la incertidumbre, el owner, la jurisdicción, la versión, el lineage y
la política de revisión. La integración auditada reunió `12` casos, `48`
fuentes, `135` claims, `180` predicados, `144` hard negatives y `445` joins
internos; no produjo una autoridad universal ni un join que trasladara
evidencia entre casos.

El resultado introduce una separación necesaria para una PPU. La autoridad
material puede sostener que cierto objeto o valor funciona como target dentro
de un protocolo, pero no acredita por sí sola la relación proporcional que una
arquitectura atribuye al fenómeno. Ninguno de los doce casos cerró esa segunda
operación. Por eso una geometría proporcional necesitaría dos contratos
enlazados: uno para la producción, medición, acceso y revisión del target; otro
para adjudicar invariancia, gauge y alcance de la relación aprendida. Cinco
alternativas conservan esa posibilidad —registro de autoridad, escrow
prospectivo, bus tipado de evidencia, registro de bridges y gate no
compensable—, pero ninguna fue seleccionada ni promovida.

La Ola 48 avanzó sobre el contrato complementario: no preguntó si dos números
permiten formar un cociente, sino qué autoriza a tratar ese cociente como una
relación proporcional del fenómeno. Tres carriles independientes integraron
`27` casos, `31` URLs usadas o contextuales, `40` claims, `135` celdas y `378`
hard negatives. La relación quedó condicionada por cuatro mediaciones que no se
reemplazan entre sí: autoridad material del target; meaningfulness bajo cambios
de escala, unidad, origen, gauge o base; régimen físico o constitutivo; y
validación discriminante y de transporte frente a familias rivales.

Ningún caso observado reunió los cuatro planos. El alcance de ese resultado es
deliberadamente acotado: describe una ausencia en el corpus relevado y no una
imposibilidad universal. Para la arquitectura, sin embargo, la consecuencia es
fuerte. Objeto y relación, clase de equivalencia y representante, lineage de
base, régimen, alternativas y owner de la evidencia deben conservarse como
estados distintos. Un proposer puede sugerir una familia relacional, pero no
adjudicarse la autoridad que sólo pueden aportar el contrato del dominio, los
controles discriminantes y el transporte. Las cinco alternativas C48
materializan distintos modos de organizar esa separación; ninguna fue
seleccionada ni promovida.

La Ola 49 desplazó una de esas alternativas desde el contrato hacia la
ejecución. Su banco sintético no preguntó sólo cuál curva ajustaba mejor, sino
qué conjunto de familias seguía siendo compatible bajo error en ambas
variables y cuándo correspondía abstenerse fuera de catálogo. EIV aumentó con
claridad la cobertura de la familia verdadera, pero a costa de conjuntos más
anchos; la abstención conformal redujo incompatibilidades y detectó cerca de la
mitad del OOD con muy baja falsa abstención dentro de catálogo. La
proporcionalidad forzada fracasó frente al catálogo plural. Así, el primer
resultado experimental de este tramo no descubre una geometría: muestra que la
adjudicación relacional se beneficia de modelar incertidumbre y de una frontera
explícita de competencia. Si una salida set-valued supera a una decisión cerrada
matched sigue siendo una pregunta experimental, no un resultado de este banco.

La integridad del resultado descansa en un lockbox, attestation semántica,
checker, mutation suite y replay exacto, no en la autoridad física de la verdad
sintética. Esto abre una prueba neuronal estrecha: comparar una decisión cerrada
con compatibilidad set-valued bajo información y presupuesto iguales. No
autoriza aún una primitive aprendida ni convierte el selector clásico en teoría
de la naturaleza.

La Ola 50 ejecutó esa prueba neuronal con un encoder DeepSets compartido. El
brazo `sigmoid_set` conservó más del conjunto compatible que
`softmax_partial`: en `NEAR_RIVAL`, la diferencia de recall fue `+0.1148`, con
IC 97,5% `[+0.0693,+0.1582]`. La segunda condición preregistrada no se cumplió:
el top-1 casi empató en el punto, pero el límite inferior `-0.03125` quedó por
debajo del margen `-0.03`. El sigmoid también excedió por `0.0065` el límite de
transferencia de ancho, y el selector clásico EIV mantuvo mejor equilibrio
entre recall, incompatibilidad y amplitud de la respuesta. Por eso la evidencia
no selecciona una arquitectura, aunque sí vuelve operativo un principio: el
conjunto de relaciones todavía identificables y la acción que el sistema toma
dentro de él son objetos diferentes. Una futura PPU puede representarlos en dos
etapas, sin convertir la política de decisión en verdad retrospectiva sobre la
geometría.

La adjudicación requirió recuperación técnica sobre el mismo lockbox luego de
dos errores de alineación y de un preview observado. La equivalencia de
checkpoints, logits, normalizador, thresholds y predicciones quedó comprobada
antes de reabrir el oracle, y un replay posterior reprodujo los artefactos bajo
exclusiones runtime explícitas. Esto sostiene la trazabilidad del contraste,
pero no lo vuelve una réplica por generador independiente.

La Ola 51 sometió a prueba la continuación arquitectónica más inmediata de ese
resultado: un encoder común con una cabeza para el conjunto compatible y otra
para la elección, entrenada después de congelar la primera etapa. La cabeza de
elección retuvo señal frente a un control con targets barajados, pero esa señal
no produjo una mejora del sistema. El brazo factorizado empató el top-1 gated
del sigmoid de referencia, perdió recall frente al sigmoid con el mismo
presupuesto total y no mostró una contribución material atribuible al staging o
al congelamiento. La reproducción exacta de arrays y estados vuelve poco
plausible explicar el negativo por inestabilidad de ejecución.

El resultado no elimina la diferencia entre identificar un conjunto y decidir
dentro de él; impide resolverla mediante una separación puramente mecánica de
cabezas. Una política necesita una fuente de autoridad que no estaba presente
en el target parcial del banco, como utilidad, costo o contexto externo, o una
formulación multiobjetivo que vuelva explícito el compromiso entre cobertura,
ancho y elección. La factorización conserva así valor como interfaz conceptual,
pero su versión two-stage simple deja de ser una continuación privilegiada.

La Ola 52 introdujo la autoridad ausente como un orden de utilidad contractual y
preguntó si una misma representación set-valued podía sostener decisiones bajo
políticas ordinales nuevas. La utilidad resultó operativamente activa dentro
del banco: ocultarla degradó con fuerza la elección, mientras el reader explícito aumentó la tasa de
acciones compatibles y redujo regret frente a un selector contextual directo.
La ventaja no se trasladó a una mejora concluyente de exactitud y el control
contrafactual quedó por debajo del criterio fijado. La factorización adquiere
así una función operacional sin convertirse todavía en arquitectura elegida.

El límite observado permite precisar el objeto que falta. Una región
identificada no debería llegar a la política como un conjunto binario que
finge fronteras exactas. Si el estimador omite una alternativa compatible, el
reader posterior no puede recuperarla; si admite una alternativa espuria, la
utilidad puede volverla acción. Una arquitectura proporcional orientada a
decisión necesita transportar conjuntamente compatibilidad, incertidumbre y
costo, y reservar la abstención para los casos en que ninguna acción resulte
estable bajo esa incertidumbre. El problema ya no consiste sólo en separar
representación y policy, sino en preservar el estatuto epistémico del conjunto
durante su transformación en una acción.

La Ola 53 probó la primera resolución de ese problema conservando cuatro
probabilidades marginales. El producto Bernoulli condicionado a conjunto no
vacío permitió calcular regret esperado y elevó la compatibilidad de las
acciones, pero perdió exactitud frente al conjunto duro y no redujo regret de
manera concluyente. La abstención basada en riesgo previsto sí separó una región
de menor regret. El negativo principal apareció en otro nivel: calibrar cada
pertenencia por separado no reconstruyó la distribución de cardinalidad ni las
dependencias entre familias.

La Ola 54 volvió explícita esa geometría probabilística. Una región compatible
dejó de representarse como yuxtaposición de cuatro eventos independientes y pasó
a ocupar uno de los quince estados no vacíos de un retículo finito. El posterior
regularizado separó pendientes de pertenencia, sesgos de cardinalidad y
contrastes heterogéneos de interacción. Bajo encoder y logits congelados, redujo
la NLL exacta frente al mejor independiente, corrigió parte de la distribución
de cardinalidad y mostró una contribución adicional de las interacciones. Esto
establece que el objeto conjunto contiene estructura predictiva que las
marginales no preservaban.

La misma prueba también delimitó esa ganancia. La política inducida elevó la
compatibilidad, pero perdió exactitud frente al conjunto duro y no redujo regret
con la magnitud e incertidumbre exigidas. El patrón predeclarado quedó falso. La
geometría probabilística y su reader decisional no son, por tanto, una sola
capacidad: mejorar la distribución sobre regiones no determina automáticamente
la acción más adecuada bajo una familia de utilidades. El próximo discriminante
debe operar sobre esa interfaz y sobre el soporte incompleto de conjuntos, sin
atribuir el límite al encoder ni convertir una mejora de likelihood en autoridad
física.

### 2. Dinámica, partición y cardinalidad

Redes de Kuramoto con sistemas completos separados entre splits permiten distinguir proximidad de frecuencia, locking, partición y número de grupos. Un encoder temporal común alimenta lectores diferentes, de manera que un cambio de clustering no sea atribuido al representation learner.

### 3. Aparato y transferencia

La capacidad que sobreviva a los bancos anteriores pasa por una cámara física intervenible y luego por un dominio externo. El modelo ve observaciones; el estado causal queda reservado para adjudicación.

### 4. Evaluación prospectiva

Un `Critical Assessment of Proportional structure` mínimo congelaría protocolo, modelos y hashes antes de generar o medir el test final. La publicación posterior incluiría predicciones crudas, incertidumbre, manifests e incidentes. Su función sería análoga a CASP en la independencia del juicio, no en escala ni madurez disciplinar.

## Alcance de la campaña

Las cincuenta y cuatro olas y sus ciento ocho investigaciones independientes con procedencia explícita, más el benchmark clásico y los experimentos neuronales ejecutados en las Olas 49–54, aportan una base para diseñar experimentos menos ciegos y una definición falsable de capacidad proporcional. No demuestran que toda proporción sea informacionalmente privilegiada, que triangle, Hodge o sheaves sean operadores canónicos, que Fisher/OT sean geometrías intrínsecas del mundo, que una forma simpléctica identifique la ley, que una categoría valide la física de sus componentes, que renormalización seleccione un coarse-graining natural universal, que un efecto de razón revele una métrica neural única, que una macrovariable predictiva sea causal, que equivariancia implique projectivity, que toda función piecewise-linear sea tropical, que validez orientada implique coordenadas lineales, que una lista finita de desigualdades caracterice entropicidad, que soporte M-convexo identifique coeficientes, raíces o valuaciones, que rigidez genérica vuelva identificable toda distancia faltante, que una exchange relation conocida demuestre discovery de su seed, que una distancia de forma adjudique correspondencia, que invariantes conformes locales garanticen realizabilidad global, que cycle consistency pruebe verdad física, que closures completas identifiquen una imagen, que invariancia implique separación orbital, que estabilidad persistente implique suficiencia o autoridad de filtración, que el power spectrum separe órbitas, que una unicidad bispectral continua sobreviva a cualquier discretización, que el estado oculto plantado coincida con el estado causal mínimo, que rango de Hankel implique realizabilidad probabilística, que un espectro determine una geometría, que predicción de observables identifique un generador o autorice control, que una respuesta identifique el representante interior plantado, que un mismo cociente ordene la información de los canales, que unicidad implique visibilidad estable, que una aproximación exterior o una región de confianza equivalgan al conjunto identificado poblacional, que un transformer aprendido sea sound por pasar ejemplos, que una API de objetos intrínsecos sea un solver universal, que una razón operatorial seleccione por sí sola su reader, que un score jerárquico autorice la partición final, que una convención de representación deba convertirse en una cabeza aprendida, que separar y congelar cabezas otorgue autoridad a la política de decisión, que una base de Markov garantice mixing rápido, que una red de reacción publicada identifique el mecanismo biológico único ni que una ODE determine la cadena estocástica finita o su estacionaria. Tampoco demuestran que la armonía musical constituya la geometría general de la naturaleza, que una autoridad física nueva constituya por sí sola una primitive aprendible, que una policy útil abra una jurisdicción proporcional nueva, que reutilizar un método equivalga a transferir una primitive entre autoridades, que una acción aprendida produzca por sí sola evidencia autorizada, que una partición orbital identifique sus flechas o composición, que un formalismo local componga automáticamente claims entre autoridades, que detectar un candidato identifique la entidad, que un matching MAP pruebe homología, que una ratio calculable autorice sus operands, que cambiar posterior, schema o weights materialice una revisión contractual completa y reproducible, que usar correctamente Clifford demuestre una ventaja aprendida frente a su expansión tensorial exacta, ni que una geometría propuesta por una red quede autorizada por el mero hecho de mejorar una métrica, que toda relación característica se reduzca a un cono cuadrático único, que una geometría efectiva ascienda sin bridge a geometría del soporte, que un benchmark con labels útiles materialice por sí solo una cadena auditable de factorización, abstención y transferencia OOD, que piezas correctas de autoridad, sellado y crédito provenientes de contratos distintos formen por suma un ground truth común, que interfaces compatibles materialicen por sí solas un harness científico con autoridad, que un certificado equivalga a verdad universal, que un reveal tardío pruebe producción fresca, que una réplica pruebe independencia material, que un cambio de aparato autorice OOD, que la autoridad material de un target valide por sí sola una relación proporcional, que una salida sigmoid constituya por sí sola un conjunto identificado físicamente autorizado, que meaningfulness formal, régimen físico y validación discriminante puedan sustituirse entre sí, que marginales calibradas identifiquen una distribución conjunta sobre conjuntos compatibles, ni que una mejor likelihood conjunta determine por sí sola una mejor política de acción. Tampoco declaran GO/NO-GO.

Lo que sí cambia es la forma de formular el frente arquitectónico. La pregunta ya no es “¿qué red se parece al AlphaFold de las proporciones?”, sino “¿qué operación proporcional, bajo qué equivalencia y en qué estrato de evidencia, justifica cada mecanismo de la red?”. Esa reformulación convierte una intuición amplia en un programa acumulativo.

## Fuentes

La campaña conserva internamente informes crudos separados, matrices comparativas, bibliografía y un programa experimental detallado. Este documento público sintetiza sus resultados sin convertir el archivo de agentes en una dependencia de lectura. Entre las fuentes primarias de entrada:

- Aitchison: https://doi.org/10.1111/j.2517-6161.1982.tb01195.x
- Dörfler, Chertkov y Bullo: https://doi.org/10.1073/pnas.1212134110
- Causal Chambers: https://causalchambers.org/
- REBOUND: https://github.com/hannorein/rebound
- Dataset ETH/Polimi: https://zenodo.org/records/15516419
- Marjieh et al.: https://www.nature.com/articles/s41467-024-45812-z
- AlphaFold: https://www.nature.com/articles/s41586-021-03819-2
- CASP: https://predictioncenter.org/
- wwPDB: https://www.wwpdb.org/
- CAMEO: https://cameo3d.org/about
- Wilson, RG y fenómenos críticos: https://doi.org/10.1103/PhysRevB.4.3174
- Koch-Janusz y Ringel, RSMI: https://doi.org/10.1038/s41567-018-0081-4
- DeWind et al., número/tamaño/espaciado: https://doi.org/10.1016/j.cognition.2015.05.016
- Vallentin y Nieder, proporciones espaciales: https://doi.org/10.1016/j.cub.2008.08.042
- Buckingham: https://doi.org/10.1103/PhysRev.4.345
- SI Brochure: https://www.bipm.org/en/publications/si-brochure
- PDEBench: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-2986
- Auxina y phyllotaxis: https://doi.org/10.7554/eLife.55832
- Alas de Drosophila: https://doi.org/10.7554/eLife.66750
- Rhea: https://www.rhea-db.org/
- spglib: https://spglib.readthedocs.io/en/stable/
- Reducción de Kron: https://doi.org/10.1109/TCSI.2012.2215780
- Port-Hamiltonian systems on graphs: https://doi.org/10.1137/110840091
- Discrete Exterior Calculus: https://arxiv.org/abs/math/0508341
- Cellular sheaves: https://doi.org/10.1007/s41468-019-00038-7
- Sampling sheaves: https://arxiv.org/abs/1405.0324
- Fisher, información y suficiencia: https://doi.org/10.1098/rsta.1922.0009
- Information Geometry and Its Applications: https://doi.org/10.1007/978-4-431-55978-8
- Aitchison frente a simplex composicional: https://doi.org/10.1111/j.2517-6161.1982.tb01195.x
- Computational Optimal Transport: https://optimaltransport.github.io/pdf/ComputationalOT.pdf
- Gromov-Wasserstein y matching: https://doi.org/10.1007/s10208-011-9093-5
- Transporte no balanceado: https://doi.org/10.1016/j.jfa.2018.03.008
- Mecánica discreta e integradores variacionales: https://doi.org/10.1017/S096249290100006X
- Hamiltonian Neural Networks: https://proceedings.neurips.cc/paper/2019/hash/26cd8ecadce0d4efd6cc8a8725cbd1f8-Abstract.html
- Sistemas port-Hamiltonian: https://doi.org/10.4171/022-3/65
- Sistemas abiertos y cospans estructurados: https://arxiv.org/abs/1911.04630
- Operad de wiring diagrams: https://arxiv.org/abs/1305.0297
- Black-boxing composicional: https://arxiv.org/abs/1812.03601
- Identificabilidad causal desde secuencias intervenidas (CITRIS): https://proceedings.mlr.press/v162/lippe22a/lippe22a.pdf
- Identificabilidad estructural y práctica por profile likelihood: https://www.jeti.uni-freiburg.de/papers/Raue_Bioinformatics_printed_1923.pdf
- Causal Chambers como testbed físico: https://www.nature.com/articles/s42256-024-00964-x
- Hilbert Nullstellensatz, Stacks Project: https://stacks.math.columbia.edu/tag/00FV
- SMT-LIB, lógicas formales: https://smt-lib.org/logics.shtml
- Certificados SOS racionales: https://www.mit.edu/~parrilo/pubs/files/PeyrlParrilo-ComputingSumOfSquaresDecompositionsWithRationalCoefficients.pdf
- Vocabulario Internacional de Metrología: https://www.bipm.org/en/doi/10.59161/jcgm200-2012
- GUM y propagación de incertidumbre: https://www.bipm.org/en/doi/10.59161/JCGM100-2008E
- Propagación multivariada y covarianza: https://www.bipm.org/en/doi/10.59161/jcgm102-2011
- BIPM KCDB: https://www.bipm.org/kcdb/
- NIST Standard Reference Data: https://www.nist.gov/srd
- Bayesian experimental design: https://doi.org/10.1214/ss/1177009939
- Robust expected information gain: https://proceedings.mlr.press/v180/go22a.html
- Fisher information y no-identificabilidad: https://arxiv.org/abs/2003.07315
- Graph limits and graphons: https://doi.org/10.1090/S0273-0979-06-01126-7
- Projectivity in exponential random graph models: https://doi.org/10.1214/11-AOS902
- Sparse exchangeable graphs and graphexes: https://doi.org/10.1214/16-AOS1518
- Introduction to Tropical Geometry: https://bookstore.ams.org/gsm-161
- Dequantization of real algebraic geometry: https://arxiv.org/abs/math/0011041
- Tropical varieties and tropical bases: https://doi.org/10.1214/009117905000000044
- Lorentzian polynomials: https://annals.math.princeton.edu/2020/192-3/p04
- Negative dependence and the geometry of polynomials: https://arxiv.org/abs/0707.2340
- Discrete Convex Analysis: https://epubs.siam.org/doi/book/10.1137/1.9780898718508
- Valuated matroids: https://doi.org/10.1016/0001-8708%2892%2990028-J
- Generic global rigidity: https://www.eecs.harvard.edu/~sjg/papers/ggr.pdf
- Bearing rigidity and localization: https://arxiv.org/abs/1608.08559
- Cluster algebras I: https://arxiv.org/abs/math/0104151
- Positivity for cluster algebras: https://arxiv.org/abs/1306.2415
- Kendall, shape manifolds y densidades de difusión: https://doi.org/10.1112/blms/16.2.81
- Generalized Procrustes Analysis: https://doi.org/10.1111/j.2517-6161.1991.tb01825.x
- Discrete conformal maps y vertex scaling: https://arxiv.org/abs/1005.2698
- Circle patterns y condiciones variacionales globales: https://arxiv.org/abs/math/0203250
- Deficiency zero y one theorems: https://web.mit.edu/~jadbabai/www/ESE680/Fei87a.pdf
- Product-form stationary distributions para CRNT estocástica: https://doi.org/10.1007/s11538-010-9517-4
- Semántica y formatos CRNT/SBML: https://sbml.org/software/sbml-test-suite/
- BioModels: https://www.ebi.ac.uk/biomodels/
- Formalización de álgebra geométrica en Lean: https://arxiv.org/abs/2110.03551
- Projective Geometric Algebra: https://arxiv.org/abs/1901.05873
- Geometric Algebra Transformer: https://arxiv.org/html/2305.18415
- Clifford Group Equivariant Neural Networks: https://arxiv.org/html/2305.11141
- Tensor products equivariantes en e3nn: https://docs.e3nn.org/en/stable/api/o3/o3_tp.html
