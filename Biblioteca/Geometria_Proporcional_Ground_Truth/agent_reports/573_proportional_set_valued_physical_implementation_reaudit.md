# R573 — Reauditoría independiente de implementación del paquete físico set-valued CPU

**Fecha:** 2026-09-07  
**Objeto:** estado posterior a las correcciones de R572 del plan `PLAN_PROPORTIONAL_SET_VALUED_PHYSICAL_PACKAGE_CPU.md`  
**Régimen:** auditoría independiente de trabajo ajeno; CPU solamente; sin uso ni consulta de GPU/CUDA; sin edición de implementación ni artefactos canónicos  
**Veredicto técnico:** **REVISE — 1 HIGH y 5 MEDIUM abiertos; 3 de los 6 findings de R572 quedan cerrados, 2 quedan parcialmente resueltos y 1 continúa abierto.**

Este dictamen no es un `GO/NO-GO` científico, no promueve arquitectura y no convierte el preflight sobre datos históricos abiertos en evidencia prospectiva. La decisión científica continúa perteneciendo al usuario.

## 1. Alcance, freeze y estado observado

Se leyeron completos `AGENTS.md`, `CODEX.md`, el plan, R569–R572, R564 y los 13 archivos restantes del source freeze: **8.241 líneas** de reglas y fuente, de las cuales **7.846** pertenecen al freeze vigente. Se recorrieron íntegramente los cuatro árboles canónicos:

| Raíz | Archivos | Directorios incl. raíz | Bytes |
|---|---:|---:|---:|
| input | 12 | 5 | 244.516 |
| primaria | 60 | 9 | 6.650.531 |
| replay | 60 | 9 | 6.650.958 |
| evidence | 6 | 1 | 46.503 |
| **Total** | **138** | **24** | **13.592.508** |

Se parsearon los **95 JSON**, se cargaron con `allow_pickle=false` los **40 NPZ** y el único NPY, y se leyeron los dos reportes Markdown. No se encontraron JSON no canónicos, object arrays, symlinks, objetos especiales ni hardlinks múltiples. Todos los objetos son `root:root`; input presenta los 12/12 modos esperados y primaria/replay presentan cada una 50 archivos `0444`, 10 `0400`, siete fases `0500` y dos directorios `0700`.

Referencias verificadas:

- `HEAD=e362d54f526afde54b23bfd67e2cf8a3042b66a9`;
- último commit del source freeze: el mismo `e362d54…`, padre `022b1496fafe168b5181de8a560ac98ff0e1f181`, con un único path modificado;
- source freeze SHA-256: `286e5973b02fef248cbb78dc92b2865bf4309fafc34ca8f52ffcea3bf9546b5f`;
- **19/19** archivos ligados coinciden con sus SHA-256 (`proportional_set_valued_physical_source_freeze_v1.json:5-24`);
- las **12/12** entradas R564 de paridad coinciden actualmente con los digests congelados (`:31-43`);
- manifest física primaria: `4e51fd81448de6f09c45d70c2207a8bcc9380ad1d5a025c9e345c03c42ae3a64`, **67/67** entradas correctas;
- manifest física replay: `9d31210291c68005b13377f013073608be3056e0aec405ddac5109396659d5a5`, **67/67** entradas correctas;
- evidence manifest: `cfd05ec2e23b041ebc33d81b8d3c7e601b58bd80622f0fd3c44b9f6daeafac45`.

El worktree estaba limpio antes de crear exclusivamente este informe.

## 2. Checks CPU reejecutados y evidencia positiva

Todos los comandos se ejecutaron con `CUDA_VISIBLE_DEVICES=''`, los cuatro límites de threads en 1 y `PYTHONDONTWRITEBYTECODE=1`:

- checker físico primaria: **15/15 PASS**, 2,14 s, RSS pico 1.325.760.512 bytes;
- checker físico replay contra primaria: **15/15 PASS**, 2,38 s, RSS pico 1.325.092.864 bytes;
- checker de evidence: **PASS**, con `10/10`, `15/15`, `15/15`, `108/108` y `14/14`;
- suite unitaria: **10/10 PASS**;
- checker independiente R564 sobre primaria y replay históricos: **14/14 PASS** en ambos;
- receipt de paridad física: **36/36 exactos**, SHA-256 `1ca406cd37960e277051a78129ace3acf3c46750bed37181f264ab969abbc933` en primaria y replay;
- comparación replay: **57 archivos**, incluidos journals y 14 worker receipts mediante **39 JSON normalizados**; no sólo los 40 archivos que cubría R572.

Los 14 worker receipts observados tienen uid/gid 65534, grupos vacíos, `NoNewPrivs=1`, los cinco capability sets nulos, environment exacto de 14 claves, cinco módulos y seis runtime files ligados al freeze. Los probes por fase son 3/3/4/3/3/3/4 y todos registran denegación. Las cuatro poblaciones mantienen 192/768/768/192 tokens, seis intersecciones nulas, schemas estrechos y cardinalidad consistente.

Los budgets canónicos observados quedan dentro de config: primaria+replay 46,20 s de runtime y 13.301.489 bytes; mutaciones 194,05 s y 6.912.057 bytes temporales; recovery 425,63 s y 120.762.854 bytes temporales/preservados. Los claims siguen acotados a CPU, datos abiertos, `prospective_evidence=false`, `architecture_promoted=false` y `scientific_decision=null`.

Estas son observaciones sobre los bytes presentes. No prueban por sí solas que todos los validadores sean fail-closed frente a los casos mínimos del plan.

## 3. Revisión uno por uno de R572

| Finding R572 | Estado actual | Fundamento |
|---|---|---|
| H1 — recomposición científica incompleta | **PARCIAL; HIGH residual** | P8 y P11 ahora se recomponen completos, pero P5/P6 sustituyen marginal, joint, OOF/refit, Ridge y logísticas por igualdad con un root R564 mutable. |
| H2 — preparación/frontera física incompleta | **PARCIAL; MEDIUM residual** | P2/P3 y el reuso del input fueron endurecidos de forma sustancial, pero el runner previo a ejecución todavía acepta keysets ampliados en roles full y no valida `split_role`, finitud ni rango completo. |
| M1 — exclusiones amplias de replay | **ABIERTO; MEDIUM** | Journals y receipts entraron a la comparación normalizada, pero `runtime.json` y `recovery_origin.json` siguen excluidos enteros sin equivalencia semántica completa. |
| M2 — conteo de mutaciones sin catálogo contractual | **ABIERTO; MEDIUM** | El catálogo v2 fija 108 IDs y prueba que cada caso cambia un objeto, pero aún omite varias clases mínimas explícitas del plan. |
| M3 — restart/recovery incompleto | **RESUELTO en su núcleo** | Prefix total, journals exclusivos y 14 combinaciones `after_promotion`/`after_journal` por las siete fases quedan cubiertos. Las mutaciones de divergencia faltantes permanecen contabilizadas en M2. |
| L1 — budgets desalineados | **RESUELTO** | P15 y el checker de evidence consumen las claves específicas de config para wall, RSS, agregado de raíces, scratch, evidence y archivo individual. |

## 4. Findings abiertos

### HIGH-1 — P5/P6 aún confían en el path mutable de R564 en lugar de recomponer o usar la identidad congelada

**Observación.** El plan exige que el checker no confíe en el root R564 por nombre y para eso incorpora los digests concretos de paridad (`PLAN...md:510-512`); además define P5 como recomposición de marginal, joint, OOF, refit y shuffle, y P6 como recomposición de features, Ridge, logísticas y controles (`:728-729`). El freeze contiene 12 digests explícitos (`proportional_set_valued_physical_source_freeze_v1.json:31-43`).

El checker actual valida la forma declarativa del receipt 36/36, compara cinco archivos de posterior contra `data/.../proportional_set_valued_native_preflight_v1/posterior_fit` y sólo recompone el shuffle (`check_proportional_set_valued_physical_preflight.py:416-427`). P6 comprueba nombres de features, cantidad/orden de controles y portabilidad, pero los seis archivos decisivos se aceptan por igualdad byte a byte con el mismo root mutable (`:429-438`). Ni P1 ni P5/P6 comparan esos archivos de referencia con `historical_reference.parity_entries`; P1 sólo valida los dos manifests R564 como archivos ligados por config (`:239-254`).

La observación positiva es fuerte: en esta auditoría los 12 digests R564 sí coinciden, el checker R564 pasa 14/14 dos veces y los bytes físicos coinciden. El defecto no describe corrupción actual.

**Inferencia.** La corrección de R572 cerró la mayor parte de H1 —P8/P11 ya son genuinamente independientes—, pero dejó la autoridad de P5/P6 apoyada en ubicación presente, precisamente el caso que el freeze pretendía eliminar. Por tratarse de los fits que alimentan toda la selección posterior, el contrato fail-closed sigue incompleto en una frontera de alta severidad.

**Cierre verificable.** Comparar cada archivo R564 usado con el digest de `parity_entries` antes de usarlo y ejecutar la recomposición independiente de posterior/policy —directamente o mediante una API pura congelada del checker R564—. P5/P6 deben fallar si diverge el root histórico aunque un receipt declare 36/36.

### MEDIUM-1 — La validación pre-ejecución del input no es todavía idéntica a P2

**Observación.** `validate_input_package` ahora exige inventario cerrado, tipos, owners, modos, nlink=1, hashes, package ID, freeze, escrow, manifest, receipt, journal y disjunción (`run_proportional_set_valued_physical_preflight.py:272-335`), y `prepare_input` la aplica tanto al reuso como después de publicar (`:338-391`). Esto corrige la parte principal de H2.

Sin embargo, `validate_role` no exige el keyset exacto de los dos bundles full, no inspecciona dtype/contenido de `cluster_id` y `split_role`, no comprueba finitud de logits ni rango 1–4/no vacío (`:260-269`). En cambio P2 sí exige keys, dtypes, shapes, finitud, identidad y rango (`check_proportional_set_valued_physical_preflight.py:262-323`). Por tanto, un input coherentemente republicado con metadata ampliada o valores no cubiertos puede ser consumido por las fases antes de que el checker posterior lo rechace.

**Inferencia.** La aceptación final sí cierra ante esos bytes, pero el runner no cumple todavía la promesa más fuerte de rechazarlos antes de ejecutar. La severidad baja de HIGH a MEDIUM porque el checker independiente ya evita que el resultado se acepte y el régimen es de datos históricos abiertos.

**Cierre verificable.** Compartir un schema puro o duplicar deliberadamente en el runner las invariantes exactas de `_validate_full_role`, incluyendo keyset, `split_role`, finitud y rango, y agregar mutaciones de cada una contra el runner pre-fase.

### MEDIUM-2 — Replay conserva exclusiones de archivo completo sin semántica total

**Observación.** La normalización nueva es una mejora real: sólo elimina wall/RSS, paths staged, digest del request y hashes dependientes del worker dentro de receipts/journals (`check_proportional_set_valued_physical_preflight.py:144-159`), y compara 57 archivos (`:162-179`). Pero `replay_files` todavía excluye enteros `artifact_manifest.json`, `runtime.json`, `replay_receipt.json` y `recovery_origin.json` (`:162-164`); el runner replica la misma regla (`run_proportional_set_valued_physical_preflight.py:547-572`).

P12 valida el receipt y delega los otros archivos a P13/P14/P15 (`check...py:641-648`). P14/P15 no exigen schema/keyset exactos de runtime, `execution_class`, secuencia y cardinalidad de `phases_executed/phases_reused`, ni equivalencia primaria/replay de esos estados (`:666-684`). `recovery_origin.json` no recibe validación semántica. El plan permite excluir campos operativos, pero exige por cada exclusión igualdad exacta de keys, estados, conteos y límites y prohíbe exclusiones por directorio/archivo grueso (`PLAN...md:682-692`).

**Inferencia.** M1 no está cerrado: se corrigieron las exclusiones más grandes de R572, pero aún puede cambiar semántica operacional relevante sin que P12 la compare.

**Cierre verificable.** Normalizar campos concretos dentro de runtime/recovery, exigir sus schemas y keysets, y comparar exactamente clases, listas de fases, estados y límites. Agregar mutaciones de `execution_class`, `phases_executed`, `phases_reused`, key extra y contenido de recovery con reason code P12.

### MEDIUM-3 — 108/108 sigue sin demostrar el mínimo adversarial enumerado por el plan

**Observación.** El catálogo v2 y el evidence checker fijan versión, cantidad, conjunto único y SHA-256 de los 108 case IDs (`proportional_set_valued_physical_preflight_v1.json:70-71`; `check_proportional_set_valued_physical_preflight.py:713-725`). El harness toma snapshots antes/después y aborta salvo que cambie exactamente un objeto (`tests/run_proportional_set_valued_physical_mutations.py:251-268`). Esta parte de M2 sí fue corregida.

El contenido sigue por debajo del mínimo del plan (`PLAN...md:744-807`). Entre las ausencias verificables están:

- autoridad: commit padre divergente y variantes fresh con firma/commitment;
- preparación: archivo/key extra, symlink, hardlink externo, FIFO, directorio vacío, truth key pública, generation escrow, dtype, shape y checkpoint axis;
- frontera: cwd alterado, stage faltante, truth/state privado entregado a fase incorrecta, import de torch y threadpool excedido;
- causalidad: candidata omitida/reordenada, métrica privada en proposer, selector con modelos, evaluator con modelos o regeneración, utility/penalty/precedencia alterados y `NOT_EVALUABLE` reinterpretado;
- recovery/inventario: output futuro, output sin journal, resume con config/source freeze/preparación distintos, owner/tipo/visibilidad del manifest y disco no contabilizado.

Los casos implementados ocupan `tests/run_proportional_set_valued_physical_mutations.py:137-247`; por ejemplo P2 tiene 13 casos, P3 18, P4 6 y P12 6, pero ninguno representa varias clases anteriores. El hash del catálogo acredita estabilidad del subconjunto implementado, no equivalencia con el catálogo normativo.

**Inferencia.** `108/108 PASS` es una observación correcta y valiosa; no es todavía evidencia suficiente de la cobertura mínima aprobada.

**Cierre verificable.** Materializar una tabla versionada que mapee cada bullet normativo a uno o más case IDs y hacer que el checker de evidence valide esa cobertura, además del hash y conteo.

### MEDIUM-4 — P13 no valida parte de la semántica que el manifest declara autoritativa

**Observación.** `build_manifest` publica por entrada `path`, `type`, `class`, `bytes`, `sha256`, `mode`, `uid`, `gid` y `phase` (`run_proportional_set_valued_physical_preflight.py:527-544`), en línea con el plan, que exige fase de creación y visibilidad además de metadata física (`PLAN...md:575-577`).

P13 verifica inventario, mode/uid/gid, un `type` derivado con `Path.is_dir()` y bytes/hash de archivos (`check_proportional_set_valued_physical_preflight.py:650-664`). No valida `schema_version`, `self_excluded`, `class`, `phase`, bytes cero de directorios, conjunto exacto de keys por fila ni nlink. Además, `Path.is_dir()`/`is_file()` sigue symlinks, de modo que no equivale a clasificar el tipo con `lstat`. El harness sólo muta hash, modo, omisión y serialización (`tests/run_proportional_set_valued_physical_mutations.py:233-238`).

Los manifests canónicos actuales son correctos 67/67 y no hay links especiales: ésa es la observación. La inferencia adversarial es que una mutación aislada de `class`, `phase` o schema dentro del manifest puede conservar P13 PASS, por lo que la autoridad de visibilidad/fase no está cerrada.

**Cierre verificable.** Exigir schemas/keysets exactos, recomponer `class` y `phase`, clasificar con `lstat`, rechazar symlinks/objetos especiales y exigir nlink=1 para archivos; agregar una mutación por campo y tipo.

### MEDIUM-5 — El modo evidence valida resultados resumidos, no el contrato íntegro de receipts

**Observación.** El plan exige que cada receipt fije schema, source freeze, argv, inputs/outputs por hash, versiones, exit, wall, RSS y bytes temporales/preservados (`PLAN...md:823-829`). El checker de evidence liga los cinco archivos por manifest y valida status/exit, conteos, subset de resultados, catálogos, freeze parcial y budgets (`check_proportional_set_valued_physical_preflight.py:690-752`). No valida schema/suite de los receipts unit/primary/replay; tampoco argv, versiones, mapas completos de input/output o que las 15 filas tengan IDs únicos, set exacto y reason codes nulos.

Además, los receipts canónicos `unit_test_receipt.json`, `primary_check_receipt.json` y `replay_check_receipt.json` no contienen `peak_temporary_bytes` ni `preserved_bytes_before_receipt`, aunque el contrato dice “cada receipt”. Los de checker guardan `passed/total` sólo anidados en `stdout_last_json`, no como campos de receipt; esto no es necesariamente incorrecto, pero debe quedar schema-validado.

**Inferencia.** El evidence manifest acredita integridad de los bytes presentes, no autenticidad ni completitud semántica: el manifest es regenerable y no reemplaza la validación de cada receipt. La campaña observada parece genuina, pero el modo final puede aceptar receipts fabricados coherentemente con su manifest.

**Cierre verificable.** Definir schemas exactos por suite; validar argv permitido, versiones, todos los hashes de inputs/outputs, las 15 filas exactas y las métricas de recursos requeridas. Agregar mutaciones de schema, suite, argv, versión, input/output digest, ID duplicado/faltante y bytes temporales.

## 5. Cierre

Las correcciones posteriores a R572 son sustantivas: H2 quedó mayormente endurecido, P8/P11 ahora recomponen la matemática final, journals y worker receipts entraron al replay, recovery cubre 14 intervalos, los catálogos quedaron ligados y los budgets usan config. M3 y L1 quedan cerrados; los bytes canónicos actuales pasan todas las verificaciones reejecutadas y no muestran corrupción.

El paquete todavía no satisface, sin embargo, la condición de aceptación del plan que exige cero findings altos o medios. La brecha principal es H1 residual: P5/P6 no usan la identidad congelada ni recomponen sus fits. Se suman cinco brechas medias en validación pre-ejecución, replay, cobertura adversarial, semántica del manifest y receipts de evidencia.

**Resultado técnico final: REVISE — 1 HIGH, 5 MEDIUM, 0 LOW abiertos.** Este resultado no decide `GO/NO-GO`, no promueve arquitectura y no extiende los claims más allá del preflight físico CPU sobre datos históricos abiertos.
