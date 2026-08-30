---
schema_version: 1
id: ground-truth-geometria-proporcional
kind: concept
page_status: current
front_status: transversal
updated: 2026-08-30
verified_at: 2026-08-30
valid_at: 2026-08-30
recorded_at: 2026-08-30
evidence_commit: 172f0a5ef0304b8da87ccb6b69de9a6fa9c21b4a
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
31. Dinámica, partición y cardinalidad variable.
32. Cámara física intervenible.
33. Transferencia a otro dominio.
34. Ronda ciega con modelos congelados.

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

## Fuente

- [Informe transversal](../../04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/GEOMETRIA_PROPORCIONAL_BASES_DE_VERDAD.md)
- [PPU / Natural Harmonic Geometry](ppu-geometria-armonica-natural.md)
- [Régimen de evidencia](regimen-de-evidencia.md)
