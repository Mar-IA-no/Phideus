---
schema_version: 1
id: ground-truth-geometria-proporcional
kind: concept
page_status: current
front_status: transversal
updated: 2026-08-31
verified_at: 2026-08-31
valid_at: 2026-08-31
recorded_at: 2026-08-31
evidence_commit: 19b6e503d5bb8663ddd3c71b0ca22e7e4e985c26
source_paths:
  - Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/GEOMETRIA_PROPORCIONAL_BASES_DE_VERDAD.md
  - Documents/01_FRENTES_ACTIVOS/ESCALON_3/README.md
  - Documents/01_FRENTES_ACTIVOS/Atencion_Armonica/README.md
depends_on: [ppu-natural-harmonic-geometry, front-escalon-3, front-atencion-armonica]
tangents: [phideus-evidence-regime]
---

# Ground truth para una geometría proporcional

## Estado de la hipótesis

La investigación inicial no encontró un corpus único ni una métrica universal.
Encontró cuatro fuentes de evidencia capaces de orientar experimentos: verdad
analítica, simulación generativa, cámara física y medición externa. Falsación
adversarial y adjudicación prospectiva son funciones de validación
transversales, no fuentes positivas equivalentes.

## Familias geométricas

| Familia | Equivalencia o estado | Ejemplo |
|---|---|---|
| dimensional | cambio de unidad / subespacio `Pi` | Buckingham, SI, GUM |
| composicional | cociente por escala | Aitchison, `clr/ilr` |
| relacional | composición y ciclos | grafos de log-ratios |
| proyectiva | acción de `PGL` | cross-ratio |
| dinámica | estabilidad temporal | locking, Arnold tongues |
| particional | membresía y cardinalidad | clusters de osciladores |
| alométrica | covariación condicionada por clado, estadio y ambiente | AVONET y cohortes externas |
| morfogenética | intervención, intermediarios y trayectoria | auxina, pescoids, organoides |
| forma/linaje | mallas, árboles, deformaciones y gauges biológicos | alas, MorphoGraphX, linajes celulares |
| estequiométrica | rayos enteros, balances y subespacios conservados | Rhea, BioModels |
| periódica/material | celda, simetría, hull y fase | COD, spglib, OQMD, XRD |
| red conservativa | ciclos, puertos y equivalencia terminal | Kron, `Y-Delta`, port-Hamiltonian |
| local-global tipada | incidencias, stalks, restricciones, gauges y gluing | DEC/FEEC, sampling sheaves |
| estadística tipada | distinguibilidad dentro de una familia y canal | Fisher–Rao, KL/Bregman |
| medida/costo | acoplamiento bajo costo, masa y soporte declarados | Wasserstein, UOT |
| relacional entre espacios | estructura interna módulo isomorfismos | GW/FGW |
| dinámica geométrica | ley, forma, gauge e integrador separados | Hamilton/Lagrange, contacto, port-H, GENERIC |
| composicional abierta | wiring, semántica, constitución y black-box | operads, cospans, corelations, FMI |
| causal identificable | consulta, intervención y estructura módulo gauge | system ID, CITRIS, Causal Chambers |
| algebraica certificada | ideal, radical, dominio, witness y checker | Gröbner, QE/SMT, SOS |
| algebraico-estadística | modelo positivo o clausura, estadístico suficiente, fibra y movimientos tipados | ideales tóricos, polytopes marginales, bases de Markov y Graver, MH |
| reacción determinista-estocástica | realización, hoja compatible, cinética, balances, lattice, composición local, reloj y cociente observacional | CRNT, mass-action, CME/CTMC, product-form |
| medición representacional | primitivos, grupo admisible, meaningfulness, verdad y autoridad | teoría de medición, conjoint measurement |
| abstracción causal | conmutatividad de estados/intervenciones dentro de una jurisdicción | causal abstraction, interchange interventions |
| proyectiva entre índices | coherencia bajo restricción, marginalización o exposición | exchangeability, graphons, graphexes, extendibilidad |
| tropical / max-plus | semimódulo, gauge aditivo, active sets y dominancia | shortest paths, Viterbi, dinámica max-plus |
| polinómica tipada | raíces, dirección/cone, coeficientes, soporte y operación cerrada | estabilidad, hiperbolicidad, Lorentzianidad, derivación |
| convexa discreta | exchange, valuación, potenciales y optimización entera | `M/L`-convexidad, matroides valuados, assignment |
| rigidez y completación | configuración módulo gauge, anclas, constraints observadas e identificabilidad de una entrada | distance geometry, lateración, stress y completion |
| atlas de exchange local | seed/columna módulo órbita, probes, mutación y consistencia de chart | cluster algebras, exchange relations, Laurent |
| cociente de forma | configuración etiquetada módulo traslación, escala y `SO/O`, con correspondencia separada | Kendall, Procrustes, Bookstein, semilandmarks |
| realizabilidad conforme | vertex/edge/face/cycle, gauge, cross-ratios, holonomía y desigualdades globales | vertex scaling, circle patterns, uniformización discreta |
| operador espacial | dominio, medida, frontera, discretización, Laplaciano, heat kernel, semigrupo, equivalencia y query | geometría espectral, difusión, HKS y operadores de calor |
| operador dinámico | ley, observables, excitación, horizonte, subespacio, mapa/kernel/generador, solver y control | Koopman, Perron–Frobenius, DMD y system identification |
Estas familias no deben colapsarse bajo una sola distancia. La recurrencia de un
mecanismo entre varias constituye una pregunta empírica.

## Interfaz de evidencia

Toda tarea candidata separa `privileged_state`, `observation`,
`intervention`, `local_relations`, `global_structure`,
`equivalence_group`, `uncertainty`, `provenance` y `system_id`. También declara
`authority`, `authority_scope`, `protocol_version`, `method`, una jerarquía de
`independent_unit_id`, `split_atomic_group_id` y `parent_unit_ids`, además de
`batch_specimen_id` y `curation_status`: una identidad exacta, un simulador, un
espécimen y un consenso no son labels intercambiables. Cada brazo declara
además vista de entrada, acceso a targets/diagrama, profundidad/campo receptivo,
búsqueda de hiperparámetros, intervenciones, llamadas al solver, evaluaciones de
función, memoria y tiempo. También fija firma dimensional, sistema de unidades,
conversiones, nondimensionalización, estadísticas de normalización,
transformación inversa y unidades de scoring. Sólo así
`evidence/capacity/compute matched` es auditable. La salida conserva por
separado estado del artefacto, claim formal, solver, identificabilidad causal,
autoridad física y decisión; un checker que rechaza un artefacto no refuta la
proposición.

La novena ola añade dos contratos transversales. `measurement_contract`
congela mensurando, calibración, drift, referencia, incertidumbre y covarianza
antes del bloque retenido. `acquisition_contract` congela constructor de
candidatos, inputs permitidos, política, costo, safety y stopping. La campaña o
episodio completo es la réplica independiente; sus decisiones están anidadas y
no se bootstrapean como muestras autónomas.

| Contrato transversal | Qué gobierna | Autoridades o herramientas |
|---|---|---|
| metrológico proporcional | mensurando, referencia, calibración y covarianza | VIM/GUM, KCDB, SRM/SRD |
| adquisición activa | historia, acción, costo, acceso y campaña | OED, system ID, Causal Chambers |
| mapa entre escalas | kernel, resolución, observables, composición y truncación | Ising, RSMI, TRG/TNR |
| tarea de magnitud | estímulo, organismo, aparato, ruido y decisión | Panamath, OpenNeuro, psicofísica controlada |
| escala de medición | primitivos, muestra, sistema formal, gauge, claims y autoridad | representación, unicidad, meaningfulness |
| abstracción causal | SCM, mapas, intervenciones, realizadores, queries, gauge y cobertura | conmutatividad, pérdida, identificabilidad |
| familia proyectiva | leyes, kernels, sampler, coupling, índice de escala y extendibilidad | projectivity paired/en ley, LP, graphon/graphex |
| operador tropical | estatuto exacto/asintótico/PWL, semiring, gauge, soporte y residuos | max-plus, dequantización, tropical bases |
| orientación combinatoria | chirotope, circuitos/cocircuitos, dualidad, menores, gauge y realizabilidad | OMLIB, SAT, TOPCOM, pseudoesferas |
| región de información | función de subconjuntos, PMF/fuente, jurisdicción Shannon/entrópica/lineal y witness | ITIP/PSITIP, LP/QP, Sage, PMFs exactas |
| operación polinómica | tabla observada, máscara, gauge, dirección, clase y checker separados | `P2r-mask/gauge/direction`, contracción exacta |
| fibra algebraico-estadística | `A`, muestreo, objeto de modelo, `FiberSpec`, finitud, movimiento, transición, propuesta y kernel ejecutado | invariantes tóricos, checker de fibra, bases de Markov/Graver y MH |
| sistema de reacción tipado | realización, `Y/I/N`, cinética, semántica ODE/CTMC, canales, volumen, scaling, observación, intervención y query | `ReactionSystemSpec`, registry de teoremas/solvers y estados de abstención |
| exchange discreto | topología pública, pesos ocultos, scorer y executor separados | assignment, matching, `M^natural`-concavidad |
| completación identificable | constraints públicas, fibra de observación, target oculto, detector y abstención | lateración 2D exacta, solver y witness |
| exchange identificable | probes, catálogo de órbitas, version space, query retenida y decoder exacto | mutación de seed, columnas skew-symmetric, checker racional |
| cociente de shape | correspondencia, grupo `SO/O`, quiralidad, tamaño, template y degeneración | `P2v`, Procrustes exacto, espejos y relabeling |
| factibilidad conforme global | incidencia, ángulos/pesos, subconjuntos, frontera de positividad y checker | enum/min-cut; protocolo neuronal bloqueado antes de `A19/P2w` |
| sincronización de grupo | relaciones relativas, gauge, familia topológica, autoridad de outliers y solver | `P2a-G`, factorial residual×mixer, outputs pre-solver y dos sincronizadores |
| phase/closure | objeto, aparato, visibilidades, gains, invariantes, nullspace y covariance | preflight R36; hard pairs y factorial objeto×aparato, sin red todavía |
| representación orbital | acción, separación de órbitas, Hilbert image, estabilizador, estrato y dominio de gauge | `ORBIT-REPRESENTATION-AUDIT-v0`; embedding raw, invariantes exactos y canonicalización local separados |
| autoridad de filtración | objeto, observación, métrica, complejo, filtración, módulo, summary, reader y query | `FILTRATION-AUTHORITY-PREFLIGHT-v0`; exactitud, estabilidad, suficiencia y autoridad separadas |
| autoridad armónica | grupo, acción, clase funcional, rango, muestreo, truncación, máscara de validez y acceso del solver | `HARMONIC-ORBIT-AUTHORITY-AUDIT-v0`; power, bispectrum y scattering sin equivalencias adelantadas |
| estado predictivo | ley del proceso, historia, tests, política, equivalencia futura, realizabilidad y consulta | `PREDICTIVE-STATE-AUTHORITY-AUDIT-v0`; estado causal, aproximación finita, rango y control separados |
| autoridad de operador espacial | dominio, medida, frontera, discretización, Laplaciano, escala temporal, observable espectral, equivalencia y query | `SPATIAL-OPERATOR-AUTHORITY-AUDIT-v0`; espectro, diagonal, kernel y semigrupo separados |
| autoridad de operador dinámico | ley, observables, excitación, horizonte, política, subespacio, mapa/kernel/generador, solver y alcance de control | `DYNAMICAL-OPERATOR-AUTHORITY-AUDIT-v0`; predicción, cierre, identificación y control separados |

El split y el bootstrap usan la unidad causal independiente, no ventanas, pares
o sensores correlacionados.

## Secuencia experimental

1. Cambios de unidad, acciones de escala y ciclos exactos en tracks separados.
2. Estequiometría y redes resistivas para separar conservación, ley local y respuesta global.
3. Complejos y sheaves para separar compatibilidad, constitución, ambigüedad, obstrucción y ruido.
4. Fisher/Aitchison pareados y transporte bajo costos/solvers controlados.
5. Dinámica geométrica con ley exacta, solver-only y arquitectura × solver.
6. Sistemas abiertos con diagrama dado/aprendido y controles semánticos.
7. Identificabilidad causal: consulta × identificabilidad × anclaje físico.
8. Certificación algebraica con `PPU±solver`, IR y checker comunes.
9. Adjudicación metrológica `P2i`: protocolo externo, A8 aprendido y sham/ablación.
10. Adquisición activa `P2j`: arquitectura × motor × política bajo acceso pareado.
11. Operadores de escala `P2k`: transporte con kernel conocido separado de discovery.
12. Psicofísica `P5e`: representación × ruido × decisión bajo componentes/rangos retenidos.
13. Medición `P2l`: factibilidad finita, witness, teorema, meaningfulness, verdad y autoridad separados.
14. Abstracción causal `P2m`: chequeo, ejecución y discovery bajo privilegios tipados y gates anti-colapso.
15. Consistencia proyectiva `P2n`: equivariancia, conmutatividad paired, projectivity en ley y extendibilidad como tracks separados.
16. Operadores tropicales `P2o`: tropicalidad exacta, dequantización y ajuste PWL bajo protocolos no fusionables.
17. Orientación combinatoria `P2p`: validez, realizabilidad, traducciones, dualidad y menores sin coordenadas privilegiadas.
18. Regiones de información `P2q`: vector, PMF y fuente lineal en tracks independientes, con hard negatives tipados.
19. Operaciones polinómicas `P2r`: ejecución conocida separada de inferencia bajo máscara/gauge/direcciones congeladas.
20. Exchange discreto `P2s`: scorer aditivo separado de encoder y executor sobre topología pública común.
21. Completación `P2t`: ratio frente a sham dentro de una cuña fija, con detector de aplicabilidad común y abstención.
22. Exchange local `P2u`: version space racional, órbita de columna identificable y query retenida antes del executor.
23. Shape quotient `P2v`: `SO/O`, quiralidad, tamaño, relabeling y degeneración como diagnóstico sin nueva arquitectura.
24. Factibilidad conforme global: reabrir sólo si un calibration pool demuestra que ningún probe local trivial resuelve la clase; no existen todavía `A19/P2w`.
25. Sincronización `P2a-G`: separar residual analítico, mixer group-aware y solver bajo gauge y familias OOD.
26. Phase/closure: cerrar suficiencia de invariantes, nullspace y aparato antes de cualquier aprendizaje.
27. Representación orbital: distinguir invariancia, separación, realizabilidad, condicionamiento y gauge local bajo ACL por brazo.
28. Autoridad de filtración: manifestar la cadena operador-summary-reader y demostrar suficiencia para la query sin elevar estabilidad a reconstrucción.
29. Autoridad armónica: separar completitud continua, invariancia, separación orbital, discretización, estabilidad y solver bajo una acción declarada.
30. Estado predictivo: separar equivalencia finita, causal state teórico, rango de Hankel, realizabilidad probabilística y suficiencia de control bajo una ley completa.
31. Autoridad espacial: separar espectro, diagonal del heat kernel, kernel completo, semigrupo, equivalencia y suficiencia para la query.
32. Autoridad dinámica: separar predicción de observables, cierre invariante, mapa/kernel, generador, intervención y control.
33. Fibras algebraico-estadísticas: empezar por finitud, pertenencia, conectividad y ejecución MH checker-only; abrir un scorer sólo contra propuestas deterministas y localmente balanceadas bajo splits por `A`, familia y `FiberSpec`.
34. Redes de reacción: materializar primero autoridad, equivalencia, absorción, canales duplicados y puente ODE–CTMC en una suite checker-only; abrir un router sólo frente a compilador determinista, portafolio de reglas y abstención con la misma vista deployable.
35. Dinámica, partición y cardinalidad variable.
36. Cámara física intervenible.
37. Transferencia a otro dominio.
38. Ronda ciega con modelos congelados.

Cada etapa conserva baselines cerrados, controles param-matched, shuffles y
estados crudos de evaluación. No existe promoción automática: el usuario
conserva GO/NO-GO.

## Implicancia para PPU

La PPU queda definida provisionalmente por capacidades, no por una ontología
adelantada: representar equivalencias, componer relaciones, integrar evidencia
temporal, recuperar estructura global y transferir esas operaciones con
incertidumbre.

Una arquitectura candidata factoriza cantidades/unidades, operador estructural,
canales y gauges tipados, leyes constitutivas o autoridades geométricas,
diagnósticos local-global, proposer, IR tipada, solver/compilador instrumentado,
checker independiente y reader de equivalencia o identificabilidad. El atlas dinámico opera bajo una ley
adjudicada o una rama inferida desde evidencia deployable; el compositor abierto
mantiene separados wiring, semántica y constitución. La
separación impide confundir consistencia dimensional, similitud física y
autosimilitud, balance con comprensión material, residual con obstrucción,
Fisher con Aitchison, coupling óptimo con mecanismo causal, energía con
simplécticidad, composición legal con adecuación física o certificado formal
con autoridad sobre el mundo. `A7/P2g/P2h` registran la ampliación identificable
y certificada. `A8/P2i/P2j` registran medición y adquisición, pero exigen validar
primero protocolos externos GUM/OED/system-ID y sólo después atribuir un módulo
aprendido frente a sham/ablación. `A9/P2k` separa ejecutar un kernel común de
descubrirlo desde acceso microscópico igualado; oráculos, truncación y convergencia
quedan explícitos. `P5e` usa conducta y señales neurales como adjudicación externa
condicionada, no como ontología física. `P2l` impide tratar cualquier cociente como
razón meaningful sin primitivos y grupo admisible; `P2m` exige preservar
intervenciones y realizadores, no sólo distribuciones. `A10` queda como
especialización causal experimental de A7/A9: no es arquitectura independiente
hasta que un bloque de conmutatividad supere su ablación exacta. `A11` tampoco es
por ahora una arquitectura autónoma: especializa A9 con una constraint de
conmutación entre inferencia y restricción, y sólo podría ascender si un forward
propio supera la variante constraint-only y sus controles. `A12` registra un
bloque max-plus proyectivo estrecho con active sets y gauge autorizado; no recibe
crédito por ser meramente piecewise-linear y se compara contra soft-LSE, suma y
una ReLU compilada funcionalmente equivalente. Ninguna candidata está promovida
ni implementada. `A13/P2p` agrega factores relacionales orientados, pero sólo
recibe crédito frente a un sham de idéntica incidencia y costo, con enforcement,
menores, dualidad y solver como ejes separados. `A14/P2q` no agrega una cabeza
universal de información: organiza tres candidatos independientes para vector
de entropías, PMF conjunta y fuente lineal. Su separación evita que una
desigualdad válida en un cono se exporte como autoridad sobre los otros.
`A15/P2r` agrega una contracción direccional estrecha: el encoder materializa
una única `C_hat` antes de conocer `v`, y clase, cono, certificado y checker no
entran al forward. `A16/P2s` agrega un scorer assignment aditivo condicionado a
topología conocida; encoder, score y executor forman factores separados. El
soporte M-convexo de un polinomio Lorentziano no autoriza a identificar su capa
analítica con una valuación tropical o una geometría discreta completa.
`A17/P2t` agrega un bottleneck de log-ratios sobre constraints incidentes, pero
su claim queda restringido a esa transformación dentro de una arquitectura de
cuñas fija; un detector común decide si la completación es aplicable. `A18/P2u`
agrega un executor local de exchange sólo después de que una enumeración exacta
demuestre que los probes identifican una única órbita de columna. Ambos son
protocolos auditados y no ejecutados.
`P2v` agrega únicamente una suite diagnóstica de cocientes de forma: un solver
Procrustes exacto que agota la tarea no acredita una arquitectura. El carril de
circle patterns conserva una ley global exacta y un checker min-cut, pero su
primer generador fue rechazado porque una identidad local separaba perfectamente
las clases. Por eso no se registraron `A19/P2w`: la fuente matemática fue
integrada y el protocolo neuronal permanece bloqueado por feature-triviality.

Las suites de Ola 19 tampoco agregan una arquitectura. Delimitan dos interfaces
que una PPU podría necesitar: un encoder armónico que represente órbitas sin
confundir invariancia con completitud, y un encoder de estado predictivo que
comprima historias según futuros observables sin heredar como verdad el estado
oculto del generador. En ambos casos el módulo aprendido, el solver y el reader
quedan separados. La convergencia entre ambos lenguajes es una hipótesis de
diseño posterior, no un resultado ya obtenido.

Las suites de Ola 20 desplazan la misma exigencia hacia la operación. Un
espectro o una firma de difusión pueden ser útiles sin determinar un interior;
un predictor de observables puede funcionar sin identificar el mapa, el kernel
o el generador que gobierna la dinámica. La arquitectura que asoma de esa
comparación no es una cabeza universal, sino un atlas de operadores tipados:
contrato de objeto y equivalencia, encoder relacional, propuesta operatorial,
solver, reader de query y salida de competencia o abstención, con la autoridad
registrada fuera del módulo aprendido. La ola no agrega `A19`, un nuevo `P2*`
ni una promoción arquitectónica.

La Ola 21 vuelve sobre esa cadena desde el extremo opuesto. Si la única vista
pública es una respuesta de frontera o input-output, el target no puede fijarse
automáticamente como el interior plantado: puede ser un punto, una clase módulo
gauge, una región alcanzable, un conjunto compatible o `UNKNOWN`. La candidata
`Response-Quotient Atlas` separa la inversión hacia ese objeto identificado del
scoring de candidatos generados públicamente y de la autoridad física externa.
Su cierre es documental; no hubo implementación ni promoción.

La Ola 22 agrega que el cociente tampoco agota la geometría del experimento.
Dos canales pueden inducir las mismas clases de indistinguibilidad y conservar
distinta capacidad decisional. Una adquisición puede, además, transmitir unas
direcciones covectoriales de manera estable y ocultar otras en un kernel o una
continuación inestable. `Experiment-Relative Geometry Atlas` conserva cociente,
orden informacional y campo de visibilidad como estructuras distintas antes de
producir un punto, una clase, una componente visible, un conjunto identificado o
`UNKNOWN`. El cierre es documental y no ejecutado.

La Ola 23 muestra que tampoco toda respuesta autorizada es puntual. En un
problema parcialmente identificado, los datos pueden determinar una fibra de
parámetros o medidas compatibles sin seleccionar uno de sus elementos. El
representante plantado pertenece entonces al conjunto, pero no recibe por ello
estatuto de verdad única. `Identified-Set Authority Stack` mantiene separados
el conjunto identificado poblacional, los witnesses internos, las
aproximaciones exteriores, la autoridad numérica y la región de inferencia
muestral. El campo de compatibilidad es la operación potencialmente aprendible;
solvers, checkers y ledgers permanecen fuera del módulo y autorizan cada claim
según su jurisdicción. La candidata fue aceptada documentalmente tras dos
reauditorías, pero no fue implementada, promovida ni convertida en GO/NO-GO.

La Ola 24 desplaza el problema desde la autoridad de un conjunto hacia la
autoridad de su transformación. Una outer approximation puede ser válida y tan
amplia que no permita decidir nada; una salida más estrecha puede parecer mejor
y ser inválida si excluye un estado compatible. `Guaranteed Set Transformer
Stack` mantiene semántica concreta, dominio abstracto, propuesta, checker,
composición, pérdidas y reader como mediaciones distintas. Su primer protocolo
cambia sólo el transformer y cruza propuestas learned, classical y sham con
checker y repair comunes. La candidata fue aceptada documentalmente después de
cerrar F01–F08, pero no tiene implementación, ejecución ni promoción.

La Ola 25 muestra que tampoco existe una geometría única para todas las salidas
set-valued. Hausdorff puede ser natural para compactos; Wijsman o Attouch–Wets
pueden conservar sentido cuando hay no acotación; epi/Mosco y convergencia
gráfica responden a objetos y readers diferentes. En convexidad, support
functions, support measures, suma de Minkowski y volúmenes mixtos requieren
dimensión, regularidad y normal declaradas. `Set Geometry Authority Contract`
convierte esas condiciones en una interfaz externa: primero adjudica schema y
checker sin modelos; después compara readers y representaciones bajo el mismo
input, solver y autoridad. El cierre fue documental y reauditado. Los bancos
siguen bloqueados hasta congelar implementaciones, hashes, tolerancias, hardware
y costos; no hubo ejecución ni promoción.

La Ola 26 extiende el contrato desde la geometría de conjuntos hacia el estatuto
intrínseco del objeto. `Intrinsic Object Authority Contract` tipa observación,
equivalencia, geometría, query y autoridad para espacios métricos-medidos,
cocientes y estratos antes de cualquier encoder. La ola materializó cuatro
anclas checker-only: GH exacto frente a bottleneck biyectivo, separación
Rook/Shrikhande por leyes de distancia, cociente espejo y cociente ortante. Dos
corridas fueron byte-identical y pasaron dentro del presupuesto congelado. El
resultado adjudica esos cuatro bancos; otras diecisiete entradas continúan
propuestas y ningún modelo fue ejecutado.

La Ola 27 estudia conos positivos y operadores SPD/PSD sin asumir una razón
universal. `Positive/Operator Authority Contract` distingue rayos, escala,
soporte, gauge, acción, reader y operación. El espectro generalizado ordenado y
sus funciones simétricas ocupan el lugar de relación primaria entre operadores;
eigensolvers, cálculo funcional, solvers y checkers permanecen externos. Cuatro
primitives aprendibles estrechas quedaron formuladas como hipótesis. Los
veintidós bancos no fueron implementados ni ejecutados.

La Ola 28 estudia ultramétricas, dendrogramas, tree metrics, splits y tight
spans. `Hierarchy/Tree Authority Contract` define veintisiete variantes atómicas
y separa relación observada, estructura compatible, estructura identificada y
estructura autorizada. Los outputs aprendibles se limitan a prioridades, scores,
adquisición, riesgo estrecho o lectura; executor y checker retienen soluciones,
certificados y estado terminal. La quinta verificación independiente cerró la
trazabilidad del contrato documental. Una suite smoke posterior materializó
trece bancos exactos y checker-only: dos corridas fueron byte-identical, con
`13/13 PASS`, y una verificación adversarial independiente confirmó que
alteraciones en el orden de completación o en la topología reconstruida son
rechazadas. Una extensión exacta añadió dos casos discriminantes. Dos primes
distintos realizaron la misma jerarquía ordinal con métricas y valuaciones
diferentes, y una familia racional de cuatro puntos distinguió una celda 2D de
su límite tree-like. La extensión cerró `2/2 PASS`, byte-identical y con cuatro
mutaciones rechazadas. Una tercera suite agregó una proyección ultramétrica en
norma infinito con una familia exacta de óptimos y un banco de cuartetos que
separa defecto de cuatro puntos, margen de resolución y escala. Volvió a cerrar
`2/2 PASS`, byte-identical y con seis mutaciones rechazadas. Una cuarta suite
materializó el radio dos de una presentación del building de Bruhat–Tits sobre
`Q2`, un gauge que conserva la jerarquía ordinal mientras cambia matrices y
alturas, y un preflight que rechaza datos no métricos antes de inferir una
jerarquía. Cerró `3/3 PASS`, rechazó diecisiete mutaciones y reprodujo veinte
fixtures desde su manifest serializado; la reauditoría independiente no dejó
findings vigentes. Veinte bancos estaban ejecutados y trece seguían no
adjudicados en esa fotografía de cuatro suites. Una quinta suite añadió dos
contratos de autoridad bajo observación parcial. En el primero, la misma máscara
identifica pesos de arista de forma única cuando se publica cada topología, pero
dos topologías positivas e inequivalentes explican el mismo vector observado:
identificación condicional de pesos no equivale a identificación global del
árbol. En el segundo, dos máscaras de siete cords sobre cinco hojas tienen igual
cardinalidad, pero sólo el minimum triplet cover identifica topología y pesos;
la máscara diseñada deja ambos múltiples. La cantidad de observaciones no
determina su autoridad.

El nuevo corte cerró `2/2` baselines, `22/22` mutaciones y `8/8` guards con
resultados byte-identical; la reauditoría final no dejó findings sustantivos.
Ola 28 acumula veintidós suite-bank IDs ejecutados en cinco suites. El antiguo
índice `22/33` cruza esos contratos con los treinta y tres bancos raw R57/R58 y
no define once IDs restantes. La cobertura raw canónica es `9 fully adjudicated /
20 partially covered / 4 not adjudicated`: el edge-weight lasso está adjudicado
para su caso exacto de seis etiquetas, mientras el contraste de triplet covers
cubre sólo el mínimo `n=5` y no su familia aleatoria mayor. No hubo modelos.

Una investigación posterior comparó los cuatro bancos raw todavía no cubiertos
por su poder discriminante y dejó a R58-B-08 como **candidato de próximo corte**,
sin promoverlo. B-08 estudia Neighbor Joining bajo ruido completo alrededor de
la garantía suficiente de Atteson. La frontera `r_inf < 0.5` no se convierte en
label de fallo: por encima sólo desaparece ese certificado y el resultado debe
clasificarse como éxito observado no certificado o fallo observado. El corte
propuesto es primero checker-only, con familias de ruido, convención de arista
mínima, cuartetos, splits, pesos y residual preservados por separado.

La Ola 29 completó la especificación convencional de los otros tres bancos raw
no cubiertos sin alterar esa prioridad. B-03 exige distinguir árboles
leaf-labeled, semi-labeled y all-node: una etiqueta observada situada en un nodo
interno debe preservarse, y los roles del output bruto se validan antes de
cualquier contracción de aristas de peso cero. B-06 separa tree metrics weighted,
series-reduced unweighted y realizabilidad por subdivisión bajo una unidad fija.
B-16 trata relabeling y escala positiva como transformaciones tipadas: pueden
preservar topología y exactitud arbórea sin preservar igualdad literal ni el
subtipo de arista unitaria. R87 corrigió orden de validación y estatuto
documental; R88 reauditó esas correcciones sin findings sustantivos. Los tres
bancos siguen sin checker implementado y no están adjudicados.

La Ola 30 cerró la auditoría del plan B-08 sin convertirlo en ejecución. Fijó
trusted base, familias de ruido, convención de ties, separación entre executor
y verifier y dominio de la distorsión multiplicativa. El banco y `N2-Q`
continúan sin implementar; el plan auditado sólo reduce ambigüedades si el
usuario decide abrir ese corte.

La Ola 31 incorporó estadística algebraica y fibras condicionales. Una misma
matriz entera puede organizar una parametrización tórica y una estadística
suficiente sin volver equivalentes pertenencia al modelo, igualdad de fibra,
conectividad, irreducibilidad, estacionariedad y mixing. `Toric Constraint
Router`, `Certified Fiber Explorer` y `Algebraic Proposal Machine` quedaron
preservadas como candidatas con checker y aceptación MH externos. La ola cerró
documentalmente después de tres auditorías; no materializó bancos ni modelos.

La Ola 32 incorporó CRNT como geometría estratificada. Realización, hoja
estequiométrica, cinética, balances, lattice estocástica, clases comunicantes,
composición local, reloj e identificabilidad observacional conservan autoridad
propia. `p_jump` puede ser identificable desde `Q` cuando `p_channel` no lo es;
una ley product-form no prueba detailed balance ni mixing; y el puente ODE–CTMC
requiere convención, volumen, conversión de tasas, scaling, horizonte y error.
R100 encontró cinco defectos de autoridad, leakage y cobertura; R101 confirmó
sus correcciones sin findings nuevos sustantivos. La suite checker-only sigue
`DESIGN-ONLY / NOT-MATERIALIZED / NOT-EXECUTED`.

La Global Attractor Conjecture general no se usa como teorema asentado ni como
fuente de labels. Sólo tienen autoridad los casos especiales publicados cuyas
hipótesis hayan sido codificadas y verificadas por el contrato.

Las alternativas `Stoichiometric Authority Router`, `Compatibility-Class
Dynamics Network`, `Equivalence-Aware Reaction Proposal Network`,
`Propensity-Simplex Generator` y `Stochastic Reaction Explorer` permanecen
recuperables. Su singularidad posible no está en usar mass-action o neural ODE,
sino en sostener autoridad, equivalencia, abstención y el cruce
determinista-estocástico como tipos operativos frente a baselines equivalentes.

La Ola 33 no abrió otra familia temática. Una búsqueda orientada a autoridades
de dominio encontró termodinámica de trayectorias y gauge curvo; otra búsqueda,
orientada a primitives aprendibles y contrastes causales, encontró reducción
reticular y composición `boxplus`. Como ambos carriles no corroboraron
afirmativamente una misma vía, el cierre distingue cuatro ejes: novedad de
autoridad, novedad de primitive, corroboración independiente y completitud del
contrato. `NOT-ASSESSED` queda separado de `NO`.

La Ola 34 completó esa cross-validación. Termodinámica y gauge curvo conservan
autoridad de dominio, pero no establecieron una primitive aprendible nueva en
las consultas auditadas: cálculo exacto, estimación genérica, prior art o no
identificación agotan el residuo observado. Reducción reticular conserva una
policy condicionada dentro de una órbita, métrica y solver ya autorizados;
`boxplus` conserva una identidad exacta y un banco ingenieril, pero especializa
sum-product. Ninguna vía combina aún novedad de jurisdicción y de primitive.

El cierre obliga a distinguir `contract_specification` de
`admission_conditions`: tipar bien objeto, checker y discriminantes no equivale
a satisfacer el contrato. También vuelve claim-specific la corroboración y
separa el futuro test de autoridad/no redundancia del test de efecto aprendido.
No se seleccionó dominio, no se registró `A19`, no se abrió un nuevo `P2*` y no
se promovió arquitectura.

La Ola 35 preguntó qué cuenta como transferencia de una primitive entre
autoridades. La revisión separó tres pruebas que no pueden sustituirse: reutilizar
un método, conservar la identidad paramétrica de un núcleo y transferir
operacionalmente ese núcleo bajo otra autoridad. Los casos estudiados ofrecieron
las dos mitades del problema, pero no su conjunción: diversidad de autoridades sin
identidad, atribución causal y replay cross-authority suficientes, o identidad paramétrica dentro de una
autoridad estrecha o compartida.

El cierre conserva `TYPED-PROPOSER-EVALUATOR-LOOP` como patrón de diseño
recuperable. El proposer aprendido opera sobre una interfaz tipada; evaluator,
checker y autoridad permanecen externos; la abstención limita el alcance. La
adquisición que modifica evidencia conserva un contrato distinto como hipótesis de
A8 y no integra este loop. No es una primitive
establecida ni una arquitectura ejecutada. Cualquier prueba futura deberá congelar
autoridades, destinos, splits, contratos, adapters y controles antes de seleccionar
el destino o entrenar, y separar atribución causal, replay local y replay entre autoridades.

La Ola 36 examinó la hipótesis separada `EVIDENCE-CHANGING-ACTION`: una policy que no sólo propone dentro de evidencia fija, sino que elige acciones capaces de producir evidencia nueva. Ningún caso revisado satisfizo conjuntamente el contrato de ocho requisitos. CAD²RL aporta transferencia de un núcleo instrumental desde simulación hacia cámara y vehículo reales; A-Lab y CPBE aportan ciclos de acción, medición externa y actualización; DAD, Step-DAD y Pang et al. aportan selección amortizada, adaptación o generalización. Esas tres capacidades aparecen por separado, no como una primitive única bajo cambio de autoridad y replay completo.

El resultado conserva A8 en estado `UNRESOLVED`, no descartado ni promovido. Su contrato durable es `estado público de evidencia -> acción legal -> indicación externa -> resultado de medición -> actualización -> claim o abstención`. Instrumento, calibración, ley de observación, modelo de medición, updater, reader, costos y gates permanecen bajo autoridad externa. Replay factual, evaluación off-policy y simulación contrafactual se registran como regímenes distintos. Una prueba futura deberá congelar y hashear autoridades, splits, canonicalizers, adapters, checkpoints, construcción de candidatos, costos y reglas de preservación; luego registrará propensiones y soporte realizado, diagnosticará positividad, separará indicación de medición, estresará misspecification, medirá calibración/cobertura y verificará que los adapters no absorban la decisión.

La Ola 37 retrocedió un paso en la cadena y preguntó qué significa inducir el
contrato relacional u operatorio. La respuesta separa Track E y Track A. Una
relación o partición puede quedar identificada sin que queden identificadas las
flechas, la aplicación y la composición que la producen. El positivo Track E
requiere transiciones indexadas por acciones observadas, observación inyectiva
hasta equivalencia de interacción, acciones disponibles puras respecto de un
único factor, condición de composición acotada, mundo finito con todas las
transiciones y mínimo global. Sólo entonces la partición de esas acciones por
factor es identificable, hasta permutación de etiquetas de factor, equivalencia
de interacción e isomorfismo latente. La acción completa no enumerada conserva
una clave aparte y estado `UNRESOLVED`.

El ledger empírico normalizó quince records por régimen, target y gauge. Ninguno
cerró todos los requisitos aplicables en el corpus acotado; no apareció
action-family OOD ni authority holdout. La consecuencia arquitectónica
recuperable es un inductor tipado que mantenga estado relacional y separe una
cabeza Track E de una Track A, junto con scope, gauge, dominio y abstención. Los
checkers algebraico, relacional y fenomenológico permanecen externos. La escalera
`E0/A0/A1/T` sólo organiza un programa conceptual: no abre suite, modelo, `A19`,
nuevo `P2*` ni GO/NO-GO.

La Ola 38 desplazó la pregunta desde inducir una acción hacia componer claims
entre autoridades. La búsqueda encontró formalismos locales para compatibilidad,
conflicto, provenance, dependencia, retractación, incertidumbre, pooling y
transporte, pero sus `24` casos fuente no materializaron las claves canónicas
`9/7/10` del schema v1. Por eso el cierre conserva `264` estados MC sin
repartirlos entre soluciones y gaps: existen `0` celdas canónicas y `0` clases de
cierre adjudicables.

La lección no es que la composición sea imposible. Es que la identidad del
claim, la instancia de composición y la celda evaluada deben existir antes de
contar reducción, incertidumbre o residuo aprendible. R151 verificó esta frontera
con un fixture positivo, `42/42` negativos y un join exacto JSON↔ledger. La vía
queda investigada pero no materializada, sin `A19`, suite, modelo, nuevo `P2*` ni
GO/NO-GO.

Esta selección también ordenó las alternativas arquitectónicas. `N2-Q`, un
scorer de resoluciones de cuartetos con executor y checker externos, es la
primitive que B-08 podría discriminar después de materializar el banco clásico.
Una cabeza de riesgo para matrices completas queda registrada como variante
nueva todavía no autorizada. `N3-T`, en cambio, pertenece a observación parcial:
un minimum triplet cover de `2n-3` pares es una referencia privilegiada que usa
la topología verdadera, no una política deployable de adquisición. Por eso la
adquisición activa conserva una suite missing separada, con query model, costo,
solver, transcript y autoridad propios. Una auditoría independiente corrigió
cuatro problemas de trazabilidad sin encontrar una mezcla central de regímenes.

El corte acumulado de treinta y ocho olas favorece una familia de contratos y un
dispatch tipado entre geometrías, no una geometría universal ni una
mega-arquitectura promovida. Las decisiones de promoción y GO/NO-GO permanecen
abiertas al usuario.

## Fuente

- [Informe transversal](../../04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/GEOMETRIA_PROPORCIONAL_BASES_DE_VERDAD.md)
- [PPU / Natural Harmonic Geometry](ppu-geometria-armonica-natural.md)
- [Régimen de evidencia](regimen-de-evidencia.md)
