# Ola 57 — cierre prospectivo del proposer con guard de daño

> **Estado:** `COMPLETE / EXACT-REPLAY / PATTERN-INDETERMINATE / PASS-CON-RIESGOS / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-04
> **Plan congelado:** `WAVE_57_CONTEXTUAL_TAIL_GUARD_PLAN.md`
> **Auditoría de resultados:** `../agent_reports/405_wave57_contextual_tail_guard_execution_results_audit.md`

## Qué se puso a prueba

La Ola 57 separó dos operaciones que la compuerta contextual de la Ola 56 había
reunido: proponer la acción bayesiana porque promete gain medio y autorizarla
sólo cuando una segunda cabeza estima bajo riesgo de daño. Ambas cabezas fueron
lineales, usaron las mismas diecisiete features inference-safe y mantuvieron
congelados encoder, posterior, catálogo de 24 políticas y utilidad.

El contraste se ejecutó sobre un draw fresco de la misma ley sintética. Train
ajustó un Ridge de gain y una Logistic de daño; validation seleccionó primero
el proposer y congeló su máscara, y después seleccionó el guard; lockbox
adjudicó una sola vez. Cinco guards de capacidad igualada recibieron labels de
daño permutados condicionalmente. Ninguna fase reabrió selección.

## Recuperación del draw único

El primer intento produjo y comprometió correctamente el benchmark, pero se
detuvo antes de inferencia porque el preparador heredado comparó los `768`
tokens elegibles esperados contra los `1152` tokens totales del nuevo protocolo.
El intento quedó preservado como
`wave57_contextual_tail_guard_fresh_v1.failed_20260904T215941566314Z`; no fue
redibujado.

Una recuperación pre-oráculo versionada corrigió exclusivamente la semántica de
ese conteo. La cadena de autoridad quedó congelada en los commits
`d21f3c9 -> 9e9a714 -> 5b335cb -> db3b1a3 -> 72eb98b -> fad81bb`. La
recuperación reutilizó el escrow SHA `d489f443...`, el manifest SHA
`3d444db8...` y el amendment SHA `8cfcb045...`. R404 emitió `PASS` antes de
abrir contenido semántico del origen.

## Ejecución y replay

El primario y el replay completaron:

```text
PREPARED -> FIT_COMPLETE -> SELECT_COMPLETE -> COMPLETE
```

Ambos declararon Python `3.13.5`, NumPy `2.3.5`, SciPy `1.17.0`, scikit-learn
`1.8.0` y `device="cpu"`. No se usó ni consultó GPU. Los workers analíticos
corrieron como UID/GID `65534`, sin recibir el root del benchmark, con
capabilities nulas y probes sellados denegados.

El replay cerró `23/23` compromisos de preparación y `13/13` objetos analíticos
exactos. Una verificación adicional comparó `519/519` arrays entre primario y
replay sin discrepancias. El reporte final y `analysis_core.json` son
byte-idénticos, SHA `f3a49287...`.

Los JSON congelados de preparación retienen por herencia los nombres de fase
`wave56-preparation-exact-replay` y `wave56-stage1-preparation-complete`. La
inconsistencia es nominal y no cambia hashes, checks ni resultados. No se
reescribieron esos artefactos: el emisor quedó corregido, después del draw,
para que sólo los futuros replays Wave 57 usen la etiqueta correspondiente y
para que Wave 56 conserve su comportamiento histórico.

## Selección

En validation, el proposer eligió `q=0.8`, threshold `0.3719541471`, con `210`
filas y `53` tokens propuestos. El guard eligió `q=0.4`, threshold
`0.2799548876`, con `84` filas y `25` tokens autorizados. Los dos shards
seleccionaron proposer y guard no triviales y conservaron los signos del
selector completo.

El control advantage-only eligió `hard_only`; el control escalar eligió
`gamma=0.4`. Entre los cinco shams, sólo `0` y `4` pasaron los mínimos. Los
otros tres conservaron Hamming global por encima de `0.25`, pero su Hamming
ponderado fue `0.2260`, `0.2418` y `0.2482`, por debajo del mínimo congelado.

## Resultado en monitor

La población primaria contiene `306` pair tokens, `1183` filas de desacuerdo y
`183` tokens con algún desacuerdo.

| Brazo | Accuracy | Compatible | Regret | Worst regret |
|---|---:|---:|---:|---:|
| hard-set | 0.841231 | 0.940359 | 0.122617 | 0.385349 |
| proposer solo | 0.839325 | 0.942947 | 0.118657 | 0.404139 |
| proposer + guard | 0.846269 | 0.937908 | 0.118782 | 0.375817 |
| pure joint | 0.772467 | 0.969227 | 0.120983 | 0.439270 |
| oracle positive-gain | 0.881264 | 0.975490 | 0.072576 | 0.291394 |

Respecto del proposer fijo, el guard mejora accuracy `+0.006944`, IC95
`[+0.001362,+0.012527]`, y worst regret `-0.028322`, IC95
`[-0.041394,-0.016340]`, mientras regret cambia sólo `+0.000125`. La separación
entre propuesta y autorización tiene, por tanto, una contribución incremental
observable dentro del draw.

Respecto del hard-set, el cuadro es más débil: accuracy mejora `+0.005038`, pero
compatibilidad cae `-0.002451`; regret mejora `-0.003835` y worst regret
`-0.009532`, ambos con intervalos que cruzan cero. El guard produjo `115`
overrides: `69` beneficiosos, `44` perjudiciales y `2` neutrales.

## Patrón predeclarado

| Condición | Estado |
|---|---|
| regret vs hard con magnitud e IC | `false` |
| no inferioridad de accuracy y compatibilidad vs hard | `false` |
| worst regret vs hard con IC | `false` |
| mejora incremental del guard vs proposer | `true` |
| regret y worst regret vs promedio de cinco shams | `NOT_EVALUABLE` |
| estabilidad de shards y replay exacto | `true` |

El campo terminal es `prospective_pattern_observed:null`, no `true` ni `false`,
porque el contrato no permite promediar menos de cinco shams. Las tres
condiciones falsas impiden de todos modos sostener la conjunción positiva del
draw actual. La formulación correcta es evidencia mixta con control causal
incompleto, no validación y no refutación general de toda arquitectura de
riesgo.

## Lectura y siguiente discriminante

La Ola 57 conserva una idea arquitectónica recuperable: separar propuesta de
valor y autorización de daño. Esa separación repara parte del comportamiento
del proposer, pero todavía no desplaza al hard-set en el patrón conjunto.

Una nueva realización debería construir las permutaciones condicionales de modo
que satisfagan por diseño los mínimos global y ponderado, sin relajar umbrales
después del draw. Reparar el sham no basta por sí solo, porque las condiciones
1–3 ya fallan: el próximo diseño debe atacar también la magnitud de regret y la
cesión de compatibilidad frente al hard, posiblemente mediante un estimando de
riesgo conjunto o de cola explícitamente preregistrado. Esta alternativa queda
preservada como candidata experimental, no promovida.

Los cinco conjuntos ausentes `0,4,8,10,12` vuelven a tener soporte `0/30` y
permanecen fuera del alcance evaluable. No hay autoridad física, transporte
externo, arquitectura promovida ni decisión `GO/NO-GO`.
