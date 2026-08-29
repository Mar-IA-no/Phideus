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

## Una base estratificada

La respuesta operativa no es un dataset total, sino una base estratificada con cuatro fuentes de evidencia y dos funciones transversales de validación.

| Tipo | Estrato o función | Qué fija | Recursos iniciales | Límite |
|---|---|---|---|---|
| fuente | verdad analítica | equivalencias, invariantes y composición exacta | Aitchison, ciclos, complejos, variación y wiring | puede ser exacta y no describir un fenómeno natural |
| fuente | simulación generativa | estado completo, mecanismo e intervención dentro del generador | Kuramoto, REBOUND, BioModels, dinámica geométrica y sistemas abiertos | hereda el mundo definido por el simulador; no adquiere autoridad física |
| fuente | cámara física | distancia entre estado y observación instrumental | Causal Chambers, ETH/Polimi | valida un aparato y un régimen, no universalidad |
| fuente | evidencia externa | recurrencia en materiales, percepción o sistemas naturales | fonones, HPatches, observación orbital, conducta auditiva | suele ofrecer targets parciales e incertidumbre |
| validación | falsación adversarial | atajos, atribución y alcance | shuffles, no-ratio, OOD, controles param-matched | delimita; no aporta ontología positiva |
| validación | adjudicación ciega | resistencia al ajuste retrospectivo | benchmark prospectivo con modelos congelados | evalúa claims declarados, no universalidad |

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

La oportunidad arquitectónica más concreta es factorizar una **capa de cantidades, unidades y entidades**, un **operador estructural** —incidencia, estequiometría, simetría o restricción—, **canales y gauges tipados**, **módulos constitutivos o autoridades geométricas locales**, diagnósticos de **existencia, ambigüedad e identificabilidad**, un **proposer con IR tipada**, un **solver/compilador instrumentado**, un **checker independiente**, un adjudicador causal/físico y un **reader de equivalencia, coupling, partición o respuesta terminal**. Para dinámica aparece un atlas candidato `observación -> estado -> generador -> operador geométrico -> solver`; para sistemas abiertos, un compositor `wiring -> semántica -> constitución -> compilador -> frontera`; para relaciones formales, una cadena `proposer -> IR -> solver -> checker` cuyos estados permanecen separados. Todos necesitan una rama residual y abstención cuando falta autoridad. Esta separación evita que la red trate como sinónimos “adimensional”, “físicamente similar” y “autosimilar”, que confunda balance con ley material, forma conservada con mecanismo o composición legal con adecuación física. Son arquitecturas candidatas registradas, no modificaciones aprobadas.

Natural Harmonic Geometry designa la hipótesis posterior: que algunas de estas operaciones reaparezcan con estabilidad suficiente entre dominios físicos como para hablar de una organización transversal. Esa recurrencia todavía debe demostrarse.

## Programa experimental derivado

El programa inicial queda escalonado para que cada fallo tenga una localización interpretable.

### 1. Acciones, ciclos y conservación exactos

El primer prototipo contiene tracks separados. Un track usa matrices dimensionales y cambios de unidad para evaluar subespacios Buckingham, equivalencia entre bases `Pi` y covariancia. Otro usa composiciones positivas y grafos de log-ratios para evaluar órbitas de escala, potenciales hasta gauge y ciclos corrompidos. Un tercero usa estequiometría y redes resistivas para separar balance, ley local y respuesta global. Un cuarto usa complejos exactos y sampling sheaves para distinguir compatibilidad, constitución, existencia, ambigüedad, obstrucción y ruido. Un quinto compara geometrías estadísticas y de medidas tipadas: Fisher frente a Aitchison con observaciones pareadas, y costos OT verdaderos, aprendidos, sham o falsos bajo solvers y controles no-OT separados. Un sexto cruza prior dinámico `TRUE/SHAM/WRONG/NONE` con solver común/nativo y ejecuta además la ley exacta con cada solver para medir discretización; energía y defecto de pullback se reportan por separado, distinguiendo coordenadas canónicas, formas dependientes del estado y balances abiertos. Un séptimo separa `LOCAL-TRUE/LEARNED`, `MPNN-TRUE/LEARNED`, monolito, sham, semántica equivocada y mal tipado. Un octavo cruza consulta, identificabilidad práctica/estructural y anclaje físico bajo regímenes de acceso congelados. Un noveno cruza `PPU±solver` y `control±solver` sobre la misma IR y checker para medir propuesta, certificación e interacción arquitectura×solver. Las métricas permanecen por objeto y por track, y cada brazo declara evidencia, capacidad, compute, firma dimensional y normalización.

### 2. Dinámica, partición y cardinalidad

Redes de Kuramoto con sistemas completos separados entre splits permiten distinguir proximidad de frecuencia, locking, partición y número de grupos. Un encoder temporal común alimenta lectores diferentes, de manera que un cambio de clustering no sea atribuido al representation learner.

### 3. Aparato y transferencia

La capacidad que sobreviva a los bancos anteriores pasa por una cámara física intervenible y luego por un dominio externo. El modelo ve observaciones; el estado causal queda reservado para adjudicación.

### 4. Evaluación prospectiva

Un `Critical Assessment of Proportional structure` mínimo congelaría protocolo, modelos y hashes antes de generar o medir el test final. La publicación posterior incluiría predicciones crudas, incertidumbre, manifests e incidentes. Su función sería análoga a CASP en la independencia del juicio, no en escala ni madurez disciplinar.

## Alcance de la campaña

Las ocho olas y dieciocho investigaciones aportan una base para diseñar experimentos menos ciegos y una definición falsable de capacidad proporcional. No demuestran que toda proporción sea informacionalmente privilegiada, que triangle, Hodge o sheaves sean operadores canónicos, que Fisher/OT sean geometrías intrínsecas del mundo, que una forma simpléctica identifique la ley ni que una categoría valide la física de sus componentes. Tampoco demuestran que la armonía musical constituya la geometría general de la naturaleza ni declaran GO/NO-GO.

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
