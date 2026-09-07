# Cierre de programa: factibilidad del mapeo y portfolio experimental

> **Estado:** `RESEARCH-BASE-CLOSED / COMMON-FACTORIAL-NOT-FEASIBLE / TWO-NATIVE-CONTRASTS-READY-FOR-DESIGN / ROUTER-DEFERRED / NO-GO-NOGO`
> **Fecha:** 2026-09-06
> **Corte de evidencia:** Olas 1–60, gate `MAPPING-FEASIBILITY` y auditoría R544
> **Autoridad de promoción y GO/NO-GO:** usuario

## Resultado de cierre

El goal acumulativo queda terminado como base de investigación y diseño
trazable. No queda terminado Phideus ni resuelta la hipótesis de una geometría
proporcional natural. El cierre consiste en haber convertido una exploración
abierta en una decisión experimental finita y verificable.

El gate CPU `MAPPING-FEASIBILITY` comprobó que las tres líneas consideradas no
pueden tratarse como celdas de un mismo factorial bajo sus contratos actuales.
Aunque la query verbal pudo mantenerse, no existe un namespace común de unidad,
los brazos reciben observaciones materialmente distintas, sus targets tienen
schemas diferentes y EIV, el núcleo relacional y el posterior set-valued no
comparten semántica de score, executor ni reader. Forzar un adapter común
cambiaría el objeto o prestaría autoridad de una rama a otra.

La salida técnica es `BIFURCATE_NATIVE_CONTRASTS`. No es una selección
científica entre arquitecturas: conserva dos experimentos nativos comparables
dentro de cada jurisdicción y descarta sólo el factorial común vigente.

## Evidencia material

La ejecución canónica produjo dos runs deterministas y un paquete de `313`
archivos (`26.186.051` bytes). Los `148/148` archivos core y los `153/153`
archivos científicos fueron byte-exactos entre runs. La corrida completa tomó
`220,457 s`, con RSS máximo de `391.495.680` bytes y sin usar ni consultar GPU.

El checker cerró `13 PASS / 4 FAIL`:

| Hoja | Resultado | Lectura |
|---|---|---|
| Común M1–M5 | M1–M4 `FAIL`; M5 `PASS` | Faltan unidad, observación, target y stack decisional comunes; la separación de autoridad y fases sí es válida |
| Relacional R1–R6 | `6/6 PASS` | El contraste nativo puede formularse sin filtrar mecanismo ni target y con estimando/controles internos coherentes |
| Set-valued S1–S6 | `6/6 PASS` | El contraste nativo conserva posterior, utilidad externa, soporte y estimando sin fusionar reader y representación |

La suite de mutación v4 recorrió `67` pares únicos sobre los `17` predicados;
`44` ramas relacionales/set-valued se produjeron mediante cambios materiales de
inputs o contratos y las seis corrupciones integrales fueron rechazadas. R544
recalculó los bytes, manifests, predicados, replay, límites y álgebra de cierre
y emitió `PASS` con `0 HIGH / 0 MEDIUM / 0 LOW`.

## Portfolio experimental resultante

| Línea | Estado después del gate | Evidencia favorable | Evidencia adversa o faltante | Próximo discriminante |
|---|---|---|---|---|
| Núcleo relacional tipado con adaptación por executor | contraste nativo listo para diseño; no promovido | señal pre-solver y beneficio WLS localizado por arista; hoja R `6/6 PASS` | dos seeds, dominio sintético y dependencia del executor; IRLS no preservó la ventaja | `GENERIC/TYPED × WLS/IRLS` sobre la IR nativa de relación y peso |
| Posterior de conjuntos con política y guard separados | contraste nativo listo para diseño; no promovido | mejor NLL, cardinalidad e interacciones; hoja S `6/6 PASS` | la mejora representacional no produjo decisión superior y HGB/HGB no separó controles matched | `marginal/joint × hard/contextual` bajo scores, utilidad y soporte comunes |
| Router tipado con IR, executors, checkers y reader externos | integración diferida | contrato de separación de jurisdicciones y `BudgetPath` mecánicamente válido | ninguna primitive estrecha tiene aún evidencia afirmativa suficiente | sólo se reabre si uno de los dos contrastes nativos sobrevive a sus controles |

EIV con calibración conformal permanece como referencia clásica externa. No es
una tercera celda representacional: su observación, target y reader no son los
del grafo ni los del posterior de conjuntos.

## Forma del goal sucesor

El siguiente goal, si se abre, debe diseñar y congelar dos protocolos
coordinados pero separados. La coordinación se limita a reglas comunes de
lineage, splits, fases, replay, ledger, reporting y auditoría; no iguala objetos
que no son equivalentes.

1. **Contraste relacional.** Misma entrada grafo, mismo target módulo gauge y
   mismos budgets; representación `generic/typed` cruzada con executor
   `WLS/IRLS`. Los outputs pre-solver y post-solver se adjudican por separado.
2. **Contraste set-valued.** Mismos logits, target booleano, soporte y utilidad;
   representación `marginal/joint` cruzada con decisión `hard/contextual`. El
   proposer y el guard conservan parámetros y crédito separados.
3. **Arbitraje posterior.** Comparar qué primitive sobrevive dentro de su
   jurisdicción, no sus métricas crudas entre jurisdicciones. Sólo entonces se
   decide si hay base para diseñar el router.

El design freeze debe preceder cualquier nuevo draw o entrenamiento. Train,
calibration, selection, auditoría del freeze y monitor conservarán roles
separados. Los controles matched, shuffles, estado crudo por unidad y replay
serán obligatorios.

## Frontera CPU/GPU

El gate de compatibilidad fue proporcionado para CPU y ya quedó resuelto. Su
costo no permite extrapolar honestamente el costo de los dos contrastes
neuronales: todavía faltan tamaños, seeds, epochs y roster congelados. El
trabajo de diseño, reutilización de artefactos, preflights, checkers y estimación
de costo puede continuar por CPU. No se sustituirá una etapa de entrenamiento
materialmente más eficiente en GPU por una corrida CPU de muchas horas.

Mientras la suspensión de GPU siga vigente, cualquier etapa que realmente la
necesite queda en cola. Antes de ejecutarla se registrará el bloqueo durable y
se informarán objetivo, duración y VRAM estimadas; no se cargará CUDA sin una
habilitación explícita.

## Condición de reapertura bibliográfica

La investigación bibliográfica expansiva queda cerrada. Una búsqueda futura
sólo se justifica si el design freeze identifica una dependencia puntual que no
puede resolverse con el corpus o los artefactos locales. Esa recuperación debe
nombrar el componente y la garantía que desbloquea y archivarse sin convertirse
en otra ola abierta.

## Fuentes de cierre

- `PROGRAM_TERMINAL_ARCHITECTURE_SYNTHESIS_AND_HANDOFF.md`
- `experiments/geometria_proporcional/PLAN_PROPORTIONAL_MAPPING_FEASIBILITY_CPU.md`
- `data/geometria_proporcional/proportional_mapping_feasibility_v1/run_a/adjudication.json`
- `data/geometria_proporcional/proportional_mapping_feasibility_v1/runtime.json`
- `agent_reports/543_proportional_mapping_feasibility_material_mutation_reaudit.md`
- `agent_reports/544_proportional_mapping_feasibility_canonical_artifact_audit.md`
