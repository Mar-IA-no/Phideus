# R570 — Reauditoría independiente del plan del paquete físico set-valued

**Fecha:** 2026-09-07  
**Commit auditado:** `2cd2f86e76ded8c4ae0ac1f200125c1a1f41f9f9`  
**Parent:** `bc73ea8942605254574a4de3840cd390a1e64eff`  
**Objeto:** `experiments/geometria_proporcional/PLAN_PROPORTIONAL_SET_VALUED_PHYSICAL_PACKAGE_CPU.md`  
**SHA-256 del objeto:** `6703c48c6cf323ed5401a54d0185ed17899d5208637aaf98ff69d5fde1f91b7f`  
**Régimen:** lectura completa y probes read-only CPU; sin consultar, inicializar ni usar GPU/CUDA  
**Antecedente contrastado:** R569 y la referencia ejecutable cerrada en R564

## Veredicto

**REVISE — 1 HIGH / 1 MEDIUM / 1 LOW.**

La revisión `2cd2f86` resuelve sustancialmente los ocho findings de R569: cierra
la ruta fresca antes de cualquier path, convierte los invariantes cross-array
en predicados, separa masks/thresholds del canal outcome-aware, liga la
referencia R564, especifica el commit protocol y el layout input/output,
materializa receipts de campañas, amplía la contención del runtime y asigna
budgets a las campañas auxiliares. No obstante, la lista exacta del runtime no
es importable con el código que el propio plan ordena congelar, y el handoff de
evaluación no garantiza que `EVALUATION_TRUTH` reciba la metadata pública
estrecha necesaria para reproducir las sensitivities sin reabrir logits. Queda
además una ambigüedad menor en el alcance del límite RSS.

El preflight continúa siendo evidencia histórica abierta y no prospectiva.
Este dictamen es técnico: no promueve una arquitectura ni constituye una
decisión científica `GO/NO-GO`.

## Contraste completo con los ocho findings de R569

| Finding R569 | Estado en `2cd2f86` | Evidencia principal |
|---|---|---|
| H1 — autoridad fresh incompleta | **Resuelto** | `FRESH_PROSPECTIVE` es reservado y terminalmente rechazado con `FRESH_PROSPECTIVE_NOT_AUTHORIZED_V1` antes de crear input, output, stage o worker; no existe flag ni fallback (`PLAN...:145-164`). La mutación exige el mismo rechazo previo a cualquier path (`:752-753`). |
| H2 — invariantes semánticos cross-array | **Resuelto** | El plan fija endian, dtypes, shapes y anchos (`:166-188`), `cardinality == target.sum`, rango, unicidad, identidad token/cluster y vocabulario (`:190-194`), consistencia public/truth y disjunción (`:196-225`), y roster `[17,29,43]` más media ensemble bit-exacta y doble validación (`:227-242`). |
| M1 — `authorized_rows` y thresholds outcome-aware | **Resuelto** | El proposer separa `selection_key_metadata` de `apply_metadata` (`:299-313`); el evaluator recibe masks pero no thresholds ni logits/state, computa `authorized_rows` desde `override.sum()` y devuelve una decisión mínima (`:315-335`); el freeze recupera thresholds sólo del freeze target-blind y reaplica bit a bit (`:337-350`). |
| M2 — identidad y comparadores R564 | **Resuelto** | El source freeze nomina commit/informe R564 y ambos manifests por SHA-256 (`:476-505`); la paridad usa comparadores exactos por clase y no `allclose` (`:687-703`). Los cuatro digests declarados coinciden con Git y los artefactos actuales. |
| M3 — journal, recovery y layout | **Resuelto** | Input inmutable queda fuera de primaria/replay (`:518-551`); states, truth enum, preparación durable, pending sibling, `fsync`/`os.replace`, tabla state×artifact y recuperación quedan especificados (`:572-663`). |
| M4 — receipts de campañas | **Resuelto** | Paths exactos para unit, primary/replay checker, mutaciones, recovery y evidence manifest, con schema y hashes, aparecen en `:800-828`; el checker final valida cobertura mediante `--evidence`. |
| M5 — runtime staged/env/cap sets | **Resuelto en diseño, pero bloqueado por H1 de esta reauditoría** | El plan ahora fija runtime, ownership/modos, `python -s -P`, env desde cero, módulos y las cinco capability sets (`:405-474`). La allowlist concreta, sin embargo, omite una dependencia obligatoria de `__init__.py`. |
| L1 — budgets incompletos | **Resuelto en wall/disco; residual LOW en RSS** | Hay límites para primaria/replay, checker+mutaciones, unit/permissions, recovery, scratch, evidencia y archivo (`:830-851`). El límite RSS sigue nombrando sólo coordinador/worker. |

## Findings nuevos o residuales

### H1 — La allowlist exacta del runtime staged no puede importar el core

**Evidencia.** El plan ordena copiar exactamente cinco blobs y prohíbe copiar
otros módulos (`PLAN_PROPORTIONAL_SET_VALUED_PHYSICAL_PACKAGE_CPU.md:450-465`):

```text
_proportional_set_valued_phase_worker.py
geometria_proporcional/__init__.py
geometria_proporcional/proportional_set_valued_native.py
geometria_proporcional/wave53_uncertainty.py
geometria_proporcional/wave54_joint_set.py
```

Pero el `__init__.py` vigente que el plan exige copiar importa
`geometria_proporcional.wave49_schema` (`src/geometria_proporcional/__init__.py:1-4`),
y `wave49_schema.py` no está en esa allowlist. La importación de
`geometria_proporcional.proportional_set_valued_native` ejecuta primero el
initializer del package.

Un probe CPU en un directorio temporal, copiando exactamente los cuatro blobs
del package ya existentes y ejecutando Python con `-s -P`, environment vacío,
`PYTHONPATH` único, un thread y `CUDA_VISIBLE_DEVICES=''`, terminó con:

```text
ModuleNotFoundError: No module named 'geometria_proporcional.wave49_schema'
```

El worker futuro aún no existe y por eso no participó del probe; no cambia la
falla previa del import del package.

**Impacto.** Ninguna de las siete fases puede cargar el core bajo el runtime
exactamente prescripto. El defecto bloquea ejecución, replay y receipts, antes
de que puedan probarse los límites físicos.

**Corrección requerida.** Incluir
`geometria_proporcional/wave49_schema.py` en la allowlist y en el source/runtime
freeze, actualizar el conteo de blobs y probar un import fresco bajo el env
exacto; o stagear un initializer mínimo e inerte, con bytes y hash propios, que
no importe dependencias fuera de la allowlist. No conviene omitir simplemente
`__init__.py` sin congelar y probar la semántica de package resultante.

### M1 — `EVALUATION_TRUTH` no tiene un contrato explícito para `design_stratum`

**Evidencia.** `evaluate_public.npz` contiene `pair_token`, `design_stratum`,
`cardinality` y logits (`PLAN...:196-206`), mientras `evaluate_truth.npz`
contiene sólo `pair_token` y `target` (`:211-223`). Esto es una buena separación
de truth y logits. `EVALUATION_APPLY` recibe la vista pública y publica actions,
masses y sensitivities target-blind (`:353-365`), pero `EVALUATION_TRUTH`
recibe únicamente truth, utilidades, config y esos arrays (`:367-372`). La
matriz de stages reduce el handoff a `evaluation actions/masses/action freeze`
y tampoco enumera metadata (`:390-398`).

Así, `EVALUATION_TRUTH` conserva `pair_token` y puede recomponer
`cardinality = target.sum(axis=1)` gracias al invariante del schema, pero no
puede recuperar `design_stratum` sin que el plan autorice implícitamente una
key no enumerada o vuelva a stagear `evaluate_public`, que también contiene
logits. La referencia R564 usa explícitamente la cardinalidad pública para
construir las filas de sensibilidad por checkpoint
(`run_proportional_set_valued_native_preflight.py:1090-1151`). Aunque esa
sensibilidad vigente no slicea por `design_stratum`, el plan promete patterns y
sensitivities y exige paridad/recomposición independiente; el contrato de
handoff no preserva toda la metadata pública canónica necesaria para auditar
esas agrupaciones sin ampliar el canal de forma ad hoc.

**Impacto.** Dos implementaciones pueden discrepar entre derivar cardinalidad,
copiar metadata no declarada o reintroducir el bundle público con logits. Esa
ambigüedad impide que stage allowlists, checker y mutaciones prueben el mismo
contrato causal, aunque no abre por sí sola un canal de truth hacia el applier.

**Corrección requerida.** Hacer que `EVALUATION_APPLY` publique un bundle
estrecho y de keys exactas —por ejemplo `evaluation_metadata.npz` con
`pair_token <U64[N]`, `design_stratum <U16[N]` y `cardinality <i8[N]`, sin
logits—, ligado por hash tanto a `evaluate_public` como al action freeze.
Autorizarlo explícitamente como input de `EVALUATION_TRUTH`; allí comprobar
identidad/orden de tokens y `cardinality == target.sum(axis=1)` antes de usarlo.
Agregar una mutación que altere cada campo y espere un reason code específico.

### L1 — El límite RSS no cubre nominalmente todos los harnesses

**Evidencia.** El plan ahora fija wall time para las cuatro campañas y límites
de disco/scratch/evidencia (`PLAN...:830-851`). También dice que los harnesses
miden wall/RSS y que una excedencia invalida el preflight (`:849-852`). Sin
embargo, la única cota numérica se denomina `RSS máximo por
coordinador/worker`, 1.5 GiB (`:840`); checker, mutation harness, fixtures y
recovery harness no son nominalmente coordinador ni worker.

**Impacto.** El riesgo observado es bajo y la medición queda preservada, pero
`P15_COST` no tiene un threshold inequívoco para todos los procesos que debe
aceptar o rechazar.

**Corrección requerida.** Declarar que 1.5 GiB es el RSS máximo de **cada
proceso/harness** de la campaña —o fijar cotas separadas— y hacer que P15 y los
receipts fallen explícitamente ante cualquier excedencia.

## Aspectos verificados favorablemente

- `2cd2f86` modifica sólo el plan auditado respecto de su parent; no mezcla
  implementación ni resultados.
- La ruta fresca queda rechazada antes de materializar cualquier superficie y
  el estado máximo permanece abierto, no prospectivo (`PLAN...:31-56`,
  `:145-164`, `:904-925`).
- Los schemas ya no descansan sólo en hashes: describen y vuelven a comprobar
  relaciones internas que sí cambian folds, features y sensitivities
  (`:166-242`). Probes CPU sobre los artefactos históricos W54/W59 confirmaron
  igualdad token/cluster, cardinalidad/target, targets no vacíos y media
  ensemble/per-seed bit-exacta; esto valida el fixture observado, no una
  realización fresca.
- El canal outcome-aware de selección recibe actions y masks ya congelados,
  calcula `authorized_rows` sin logits y no puede devolver thresholds; el
  freeze target-blind recompone la política (`:299-350`).
- Los identificadores R564 del plan coinciden con la historia y los archivos
  reales: commit del informe
  `f7ad9227868f83f381ebbc0a8995fefa5a1a272f`, SHA-256 del informe
  `2221b3938b03728e28133ac0ac5b05918c56b5a62845966293ac113aea4479cb`,
  manifest primario
  `a0401834c3958680ef687ad264b8b56a017a8996eaade904873b342319528a39` y
  replay `2752c103f80ae8655747cf92709fe9462ef753dd282c20f449694a90b6e44039`.
- La preparación y cada fase tienen un orden de persistencia recuperable y el
  input inmutable nunca se archiva junto con un output fallido (`:518-663`).
- Las campañas dejan receipts con argv, versiones, hashes, wall/RSS, bytes y
  resultados por caso; el manifest final los cubre sin autorreferencia circular
  (`:800-828`).
- El checker conserva independencia explícita y comparadores exactos por clase
  (`:687-735`).

## Comprobaciones realizadas

1. Lectura lineal completa del archivo vigente. El objeto tiene **925 líneas**,
   no 921; la diferencia fue comprobada con `wc -l` y no se omitieron las cuatro
   líneas finales.
2. Verificación del commit, parent, diff plan-only y SHA-256 del objeto.
3. Contraste uno por uno con los ocho findings verbatim de R569.
4. Contraste focal con R564: informe, manifests, runner y core NumPy vigentes.
5. Probes CPU read-only de invariantes históricos y reproducción aislada del
   fallo de import del runtime staged.

No se ejecutó el futuro runner —todavía no existe—, no se editó el plan ni la
implementación y no se consultó ni utilizó GPU/CUDA.

## Condición para una siguiente reauditoría

Corregir primero la allowlist importable del runtime y explicitar el bundle de
metadata pública estrecha hacia `EVALUATION_TRUTH`. Cerrar además la cobertura
RSS nominal en la misma pasada. Después, una reauditoría puede verificar que
la versión del plan determina un único runtime y un único canal de evaluación
sin ampliar acceso a logits ni a truth.
