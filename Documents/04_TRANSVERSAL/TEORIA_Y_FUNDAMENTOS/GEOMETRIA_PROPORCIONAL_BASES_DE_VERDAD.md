# Geometría proporcional y bases de verdad

## El problema que antecede a la arquitectura

La hipótesis de una Proportional Processing Unit enfrenta una dificultad distinta de la que suele presentarse cuando se diseña una red para una tarea ya estabilizada: no existe todavía un archivo reconocido de “estructuras proporcionales resueltas” que determine qué debe aprender la arquitectura. AlphaFold no recibió una ontología completa de las proteínas, pero sí pudo trabajar sobre décadas de estructuras experimentales, secuencias evolutivamente relacionadas, regularidades geométricas y evaluaciones ciegas. Phideus no dispone de un equivalente directo para las proporciones.

Esta carencia no puede resolverse eligiendo por anticipado la serie armónica, los números racionales o una geometría favorita y tratándolos como si ya fueran la forma general de lo real. La primera campaña de investigación sobre bases de verdad proporcional produjo una conclusión más austera: **no aparece una única geometría de la proporción; aparecen familias de problemas proporcionales definidas por equivalencias, leyes de composición y regímenes dinámicos diferentes**.

Una razón aislada todavía no constituye una geometría. Empieza a hacerlo cuando sabemos qué transformaciones preservan el fenómeno, cómo se compone con otras razones y qué observaciones pertenecen al mismo estado. En datos composicionales, la escala global puede ser irrelevante. En un grafo de log-ratios, los ciclos imponen identidades. En geometría proyectiva, el cross-ratio permanece mientras las distancias ordinarias cambian. En una red de osciladores, en cambio, la cercanía a una razón racional no establece por sí sola un estado de locking: la relación debe verificarse en la trayectoria.

Una tercera ola amplió esa distinción. El cambio de unidad conserva una cantidad, pero no constituye por sí solo una simetría de la dinámica; la similitud física exige preservar ecuaciones, geometría, condiciones iniciales/de borde y todos los controles adimensionales relevantes; una power law puede ser exacta, asintótica o meramente empírica. En biología ocurre una separación análoga: correlación alométrica, mecanismo morfogenético y geometría de forma o linaje son objetos distintos.

## Una base estratificada

La respuesta operativa no es un dataset total, sino una base estratificada con cuatro fuentes de evidencia y dos funciones transversales de validación.

| Tipo | Estrato o función | Qué fija | Recursos iniciales | Límite |
|---|---|---|---|---|
| fuente | verdad analítica | equivalencias, invariantes y composición exacta | Aitchison, ciclos log-ratio, cross-ratio | puede ser exacta y no describir un fenómeno natural |
| fuente | simulación causal | estado completo, intervención y fronteras dinámicas | Kuramoto, circle maps, modos, REBOUND | hereda el mundo definido por el simulador |
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
- la **procedencia**, con versión, licencia y unidad independiente.

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

La oportunidad arquitectónica más concreta es factorizar una **capa de cantidades y unidades**, una **capa de controles físicos adimensionales** y un **operador de estado, solución o coarse-graining**. Un reader de forma, linaje o partición puede agregarse cuando el dominio lo exige. Esta separación evita que la red trate como sinónimos “adimensional”, “físicamente similar” y “autosimilar”. Es una arquitectura candidata registrada, no una modificación aprobada.

Natural Harmonic Geometry designa la hipótesis posterior: que algunas de estas operaciones reaparezcan con estabilidad suficiente entre dominios físicos como para hablar de una organización transversal. Esa recurrencia todavía debe demostrarse.

## Programa experimental derivado

El programa inicial queda escalonado para que cada fallo tenga una localización interpretable.

### 1. Acciones y ciclos exactos

El primer prototipo contiene dos tracks separados. Uno usa matrices dimensionales y cambios de unidad para evaluar subespacios Buckingham, equivalencia entre bases `Pi` y covariancia. El otro usa composiciones positivas y grafos de log-ratios para evaluar órbitas de escala, aristas retenidas, potenciales hasta gauge y ciclos corrompidos. Las comparaciones incluyen fórmulas exactas, MLPs, MPNNs, pair-state sin mezcla, mezcla local y actualización triangular.

### 2. Dinámica, partición y cardinalidad

Redes de Kuramoto con sistemas completos separados entre splits permiten distinguir proximidad de frecuencia, locking, partición y número de grupos. Un encoder temporal común alimenta lectores diferentes, de manera que un cambio de clustering no sea atribuido al representation learner.

### 3. Aparato y transferencia

La capacidad que sobreviva a los bancos anteriores pasa por una cámara física intervenible y luego por un dominio externo. El modelo ve observaciones; el estado causal queda reservado para adjudicación.

### 4. Evaluación prospectiva

Un `Critical Assessment of Proportional structure` mínimo congelaría protocolo, modelos y hashes antes de generar o medir el test final. La publicación posterior incluiría predicciones crudas, incertidumbre, manifests e incidentes. Su función sería análoga a CASP en la independencia del juicio, no en escala ni madurez disciplinar.

## Alcance de la campaña

La campaña aporta una base para diseñar experimentos menos ciegos y una definición falsable de capacidad proporcional. No demuestra que toda proporción sea informacionalmente privilegiada, que el triangle sea el operador canónico ni que la armonía musical constituya la geometría general de la naturaleza. Tampoco declara GO/NO-GO.

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
