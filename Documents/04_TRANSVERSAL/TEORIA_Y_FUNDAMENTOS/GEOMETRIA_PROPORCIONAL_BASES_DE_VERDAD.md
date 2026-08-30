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

La Ola 20 vuelve más precisa la factorización que sigue: `objeto/equivalencia -> encoder relacional -> propuesta de operador tipada -> solver/aplicación -> reader de query`, más una salida de competencia/abstención y un ledger externo de autoridad. Es un atlas de interfaces candidatas, no una operación universal ni una arquitectura promovida.

La Ola 21 agrega el recorrido inverso: `contrato de respuesta -> encoder de respuestas/probes -> objeto cociente identificado -> punto | representante certificado | conjunto compatible | UNKNOWN`. La respuesta puede haber realizado ya el cociente; por eso una operación aprendida sólo recibe crédito cuando subsiste variación observable bajo una acción pública. `Response-Quotient Atlas` conserva dos sistemas distintos —inversión tipada y compatibilidad de candidatos— y mantiene generator, checker y ledger de autoridad como factores trazables. Es una candidata aceptada documentalmente, no una arquitectura promovida.

La oportunidad arquitectónica más concreta es factorizar una **capa de cantidades, unidades y entidades**, un **operador estructural** —incidencia, estequiometría, simetría o restricción—, **canales y gauges tipados**, **módulos constitutivos o autoridades geométricas locales**, diagnósticos de **existencia, ambigüedad e identificabilidad**, un **proposer con IR tipada**, un **solver/compilador instrumentado**, un **checker independiente**, un adjudicador causal/físico y un **reader de equivalencia, coupling, partición o respuesta terminal**. Para dinámica aparece un atlas candidato `observación -> estado -> generador -> operador geométrico -> solver`; para sistemas abiertos, un compositor `wiring -> semántica -> constitución -> compilador -> frontera`; para relaciones formales, una cadena `proposer -> IR -> solver -> checker` cuyos estados permanecen separados. La novena ola agrega `A8`, un adjudicador metrológico y controlador de evidencia que primero debe existir como protocolo externo GUM/OED/system-ID. La décima registra `A9`, un operador tipado de escala con dos jurisdicciones: `KNOWN-KERNEL-TRANSPORT` adjudica ejecución y composición bajo un mapa común; `KERNEL-DISCOVERY` exige producir el mapa desde acceso microscópico idéntico al de los controles. La undécima no agrega una mega-arquitectura: `P2l` funciona como firewall de significatividad y `P2m` como firewall de abstracción causal. `A10` queda como especialización causal experimental de A7/A9. La duodécima conserva esa prudencia: `A11` es por ahora una **especialización proyectiva de A9**, cuyo único componente atribuible es una constraint de conmutación sobre un forward compartido; `A12` es un **bloque max-plus proyectivo estrecho**, no una arquitectura tropical general. La decimotercera registra `A13` como bloque relacional candidato sobre factores de Grassmann–Plücker y `A14` sólo como interfaz: sus heads DSF, PMF y lineal permanecen candidatos independientes porque no existe un lift común que pueda adjudicarse sin mezclar targets y privilegios. Una ReLU compilada que ejecuta el mismo máximo es control de equivalencia funcional, y las variedades tropicales completas permanecen como generadores, oráculos o checkers. Los oráculos quedan fuera del estimando y fixed points, subespacios o mapas plantados sólo reciben estatuto fuerte después de demostrar convergencia o identificabilidad dentro de su alcance. `A17/P2t` agrega un bottleneck de log-ratios sobre constraints incidentes, pero limita su claim a esa transformación dentro de una arquitectura de cuñas fija y obliga a estimar aplicabilidad. `A18/P2u` agrega un executor local de exchange sólo después de que un checker exacto demuestre que los probes identifican una órbita de columna. Todos necesitan una rama residual o abstención cuando falta autoridad. Son hipótesis registradas, no modificaciones aprobadas.

Natural Harmonic Geometry designa la hipótesis posterior: que algunas de estas operaciones reaparezcan con estabilidad suficiente entre dominios físicos como para hablar de una organización transversal. Esa recurrencia todavía debe demostrarse.

## Programa experimental derivado

El programa inicial queda escalonado para que cada fallo tenga una localización interpretable.

### 1. Acciones, ciclos y conservación exactos

El primer prototipo contiene tracks separados. Un track usa matrices dimensionales y cambios de unidad para evaluar subespacios Buckingham, equivalencia entre bases `Pi` y covariancia. Otro usa composiciones positivas y grafos de log-ratios para evaluar órbitas de escala, potenciales hasta gauge y ciclos corrompidos. Un tercero usa estequiometría y redes resistivas para separar balance, ley local y respuesta global. Un cuarto usa complejos exactos y sampling sheaves para distinguir compatibilidad, constitución, existencia, ambigüedad, obstrucción y ruido. Un quinto compara geometrías estadísticas y de medidas tipadas. Un sexto cruza prior dinámico con solver común/nativo. Un séptimo separa wiring, semántica y constitución. Un octavo cruza consulta, identificabilidad y anclaje físico. Un noveno cruza `PPU±solver` y `control±solver` sobre la misma IR y checker. Un décimo (`P2i`) adjudica metrología; un undécimo (`P2j`), adquisición activa. Un duodécimo (`P2k`) separa transporte con kernel conocido de discovery de kernel, con oráculos fuera del estimando y convergencia prospectiva. Un decimotercero (`P5e`) cruza representación, observación/ruido y decisión/readout sobre psicofísica externa. `P2l` separa factibilidad finita, witness, aplicabilidad teoremática, meaningfulness, verdad y autoridad. `P2m` separa chequeo, ejecución y discovery causal, con privilegios tipados y controles contra identidad, singleton, reescala y exclusión de acciones difíciles. `P2n` cruza modelo, familia generadora y sampler para separar equivariancia, conmutatividad paired, projectivity en ley y extendibilidad. `P2o` mantiene protocolos independientes para tropicalidad exacta, dequantización y ajuste PWL, con hard negatives de cancelación, gauge y semiring equivocado. `P2p` separa validez orientada, realizabilidad y conmutación de menores bajo ablaciones de una sola operación. `P2q` mantiene independientes los tracks vectorial, probabilístico y lineal, con estados de inferencia finita que no convierten error del estimador en propiedad de la población. `P2r/P2s` aíslan ejecución polinómica y scoring assignment bajo observación parcial. `P2t` compara ratio y sham sobre la misma arquitectura de cuñas y puntúa error sólo donde un detector común declara completación aplicable. `P2u` exige un manifest racional de componentes identificables antes de comparar inferencia de columna y ejecución de exchange. `P2v` diagnostica conformidad a cocientes de forma sin promover una arquitectura; el carril de factibilidad conforme permanece bloqueado antes de `A19/P2w`. `P2a-G` cruza evidencia raw/residual con mixer genérico/group-typed bajo DAG y whitelist comunes; phase/closure conserva un preflight separado de objeto y aparato. Las suites de Olas 18 y 19 auditan representación orbital, autoridad de filtración, autoridad armónica y estado predictivo antes de abrir cualquier nuevo `P2*`. En el último caso, el contrato incluye además acción, muestreo, política, inicialización, estacionariedad, soporte de historias y pesos de mezcla: sin una ley de proceso cerrada, dos modelos no están prediciendo necesariamente el mismo futuro. Las métricas permanecen por objeto y por track, y cada brazo declara evidencia, capacidad, compute, firma dimensional, normalización, acceso, solver y reader.

La Ola 20 extiende ese preflight a la autoridad operatorial espacial y dinámica. Antes de abrir otro `P2*`, exige declarar observable, semigrupo o generador, discretización, excitación, solver y query, y separar predicción de identificación o control. La Ola 21 exige además que el target permanezca constante sobre cada clase observacional, que los candidates provengan de un generator público congelado y que una reconstrucción compatible no sea confundida con el mecanismo físico plantado.

### 2. Dinámica, partición y cardinalidad

Redes de Kuramoto con sistemas completos separados entre splits permiten distinguir proximidad de frecuencia, locking, partición y número de grupos. Un encoder temporal común alimenta lectores diferentes, de manera que un cambio de clustering no sea atribuido al representation learner.

### 3. Aparato y transferencia

La capacidad que sobreviva a los bancos anteriores pasa por una cámara física intervenible y luego por un dominio externo. El modelo ve observaciones; el estado causal queda reservado para adjudicación.

### 4. Evaluación prospectiva

Un `Critical Assessment of Proportional structure` mínimo congelaría protocolo, modelos y hashes antes de generar o medir el test final. La publicación posterior incluiría predicciones crudas, incertidumbre, manifests e incidentes. Su función sería análoga a CASP en la independencia del juicio, no en escala ni madurez disciplinar.

## Alcance de la campaña

Las veintiuna olas —cuarenta y dos investigaciones independientes y dos carriles reconstruidos por el coordinador con procedencia explícita— aportan una base para diseñar experimentos menos ciegos y una definición falsable de capacidad proporcional. No demuestran que toda proporción sea informacionalmente privilegiada, que triangle, Hodge o sheaves sean operadores canónicos, que Fisher/OT sean geometrías intrínsecas del mundo, que una forma simpléctica identifique la ley, que una categoría valide la física de sus componentes, que renormalización seleccione un coarse-graining natural universal, que un efecto de razón revele una métrica neural única, que una macrovariable predictiva sea causal, que equivariancia implique projectivity, que toda función piecewise-linear sea tropical, que validez orientada implique coordenadas lineales, que una lista finita de desigualdades caracterice entropicidad, que soporte M-convexo identifique coeficientes, raíces o valuaciones, que rigidez genérica vuelva identificable toda distancia faltante, que una exchange relation conocida demuestre discovery de su seed, que una distancia de forma adjudique correspondencia, que invariantes conformes locales garanticen realizabilidad global, que cycle consistency pruebe verdad física, que closures completas identifiquen una imagen, que invariancia implique separación orbital, que estabilidad persistente implique suficiencia o autoridad de filtración, que el power spectrum separe órbitas, que una unicidad bispectral continua sobreviva a cualquier discretización, que el estado oculto plantado coincida con el estado causal mínimo, que rango de Hankel implique realizabilidad probabilística, que un espectro determine una geometría, que predicción de observables identifique un generador o autorice control, ni que una respuesta identifique el representante interior plantado. Tampoco demuestran que la armonía musical constituya la geometría general de la naturaleza ni declaran GO/NO-GO.

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
