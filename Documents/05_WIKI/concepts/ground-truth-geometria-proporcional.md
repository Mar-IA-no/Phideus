---
schema_version: 1
id: ground-truth-geometria-proporcional
kind: concept
page_status: current
front_status: transversal
updated: 2026-08-29
verified_at: 2026-08-29
valid_at: 2026-08-29
recorded_at: 2026-08-29
evidence_commit: 0b4524a40a39b850e06303049026e8b1473f00fd
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
11. Dinámica, partición y cardinalidad variable.
12. Cámara física intervenible.
13. Transferencia a otro dominio.
14. Ronda ciega con modelos congelados.

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
aprendido frente a sham/ablación. Ninguna candidata está promovida ni implementada.

## Fuente

- [Informe transversal](../../04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/GEOMETRIA_PROPORCIONAL_BASES_DE_VERDAD.md)
- [PPU / Natural Harmonic Geometry](ppu-geometria-armonica-natural.md)
- [Régimen de evidencia](regimen-de-evidencia.md)
