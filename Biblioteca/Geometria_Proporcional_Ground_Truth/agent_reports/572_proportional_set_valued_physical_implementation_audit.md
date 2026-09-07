# R572 — Auditoría independiente de implementación del paquete físico set-valued CPU

**Fecha:** 2026-09-07  
**Objeto:** implementación y evidencia canónica del plan `PLAN_PROPORTIONAL_SET_VALUED_PHYSICAL_PACKAGE_CPU.md`  
**Régimen:** auditoría de trabajo ajeno; CPU solamente; sin uso ni consulta de GPU/CUDA; sin edición de implementación ni artefactos canónicos  
**Veredicto técnico:** **REVISE — la campaña canónica observada es internamente coherente, pero el paquete no satisface todavía el contrato de aceptación fail-closed aprobado.**

Este veredicto no es un `GO/NO-GO` científico, no promueve arquitectura y no convierte el preflight abierto en evidencia prospectiva. La decisión científica continúa perteneciendo al usuario.

## 1. Alcance y lectura

Se leyeron completos:

- plan (934 líneas), R569 (143), R570 (190), R571 (167) y R564 (97): **1.531 líneas**;
- runner, worker, checker físico, checker R564 y cinco fuentes matemáticas/runtime congeladas: **5.176 líneas**;
- config y source freeze: **129 líneas**;
- test unitario, harness de mutaciones y harness de recovery: **394 líneas**.

Total de fuente textual revisada: **7.230 líneas**. Además se recorrieron íntegramente los cuatro árboles canónicos: input (**12 archivos; 244.516 bytes**), primaria (**60; 6.618.180**), replay (**60; 6.618.487**) y evidence (**6; 25.797**): **138 archivos y 13.506.980 bytes**. Se parsearon los **95 JSON** y se cargaron todos los NPZ/NPY con `allow_pickle=false`, verificando inventarios, shapes, dtypes, finitud donde corresponde, owners, modos, tipos y hashes.

Referencias congeladas verificadas:

- source freeze: `experiments/geometria_proporcional/configs/proportional_set_valued_physical_source_freeze_v1.json`, SHA-256 `329f42b6d4281ebf2e8ebf737a2383194f47a4a8c8a9c80333e71d560ae11421`;
- **18/18** archivos del freeze coinciden con sus hashes;
- `HEAD=f14dec8cdfd2a43539a69cfa440059926f9ba24f`, padre `836c06c6a9750fa62b38d91f7f5686e683430c2c`; el commit de freeze modifica sólo el source freeze;
- manifest primaria: SHA-256 `b5def9d0298774057f5089264502170d8f81b6c2791067020b5fde6d052b6d3f`, **67/67** entradas correctas;
- manifest replay: SHA-256 `5835e60abfa95cf149b83f3a27e98e16db77d8175a81da04406dbdce455220d0`, **67/67** entradas correctas;
- evidence manifest: `data/geometria_proporcional/proportional_set_valued_physical_evidence_v1/evidence_manifest.json`, SHA-256 observado `df5c93eda3055a5b7a396d693d4be9d12f2280ca0da14c4fd75c919121371666`.

## 2. Evidencia positiva observada

Las siguientes observaciones son válidas para los bytes canónicos actuales, aunque no corrigen los defectos de cobertura de aceptación que se detallan después:

- checker físico primario: **15/15 PASS**; replay: **15/15 PASS**; evidence: **PASS** con `10/10`, `15/15`, `15/15`, `63/63`, `7/7`;
- suite unitaria ejecutada nuevamente: **10/10 PASS**;
- checker independiente R564 ejecutado nuevamente sobre primaria y replay históricos: **14/14 PASS** en ambos;
- el receipt de paridad física registra **36/36 exactos** (`data/geometria_proporcional/proportional_set_valued_physical_preflight_v1/r564_parity_receipt.json`, SHA-256 `1ca406cd37960e277051a78129ace3acf3c46750bed37181f264ab969abbc933`);
- cuatro poblaciones de 192/768/768/192 tokens, seis intersecciones pairwise iguales a cero y views public/truth con schemas estrechos; input sin symlinks, FIFOs ni hardlinks externos;
- primaria y replay tienen root `0700`, fases `0500`, archivos públicos `0444`, privados `0400`, owner `root:root`; no se observaron objetos especiales ni hardlinks múltiples;
- los 14 receipts de worker observados registran uid/gid 65534, grupos vacíos, NNP=1, cinco capability sets en cero, threadpools en 1, entorno cerrado, paths bajo runtime staged y hashes de los cinco módulos importados iguales al freeze; los conteos de probes son 3/3/4/3/3/3/4 y todos devolvieron `PermissionError`;
- las 40 salidas incluidas por el comparador son byte-exactas entre primaria y replay;
- los límites observados quedan holgadamente bajo config: 13,24 MiB combinados para primaria+replay, archivo máximo 2,07 MiB, wall 65,17 s y RSS máximos declarados por debajo de 1,5 GiB por proceso;
- los claims canónicos permanecen acotados: `prospective_evidence=false`, `architecture_promoted=false`, `scientific_decision=null`, CPU y sin torch.

## 3. Hallazgos

### HIGH-1 — El checker no recompone una parte sustantiva de la matemática que declara comprobar

El plan exige recomponer marginal/joint, OOF/refit, Ridge/logísticas/controles y, en evaluación, bootstrap común, ocho familias de estimandos, precedencia y patterns (`PLAN...md:720-742`). El checker físico hace algo materialmente menor:

- P5 confía en `r564_parity_receipt.json` y sólo recompone el shuffle (`check_proportional_set_valued_physical_preflight.py:221-229`); no contrasta el receipt contra los bytes R564 ni recompone posterior marginal, joint, OOF o refit;
- P6 comprueba nombres/count de features, número/orden de controles y ausencia de object arrays (`:231-237`); no vuelve a ajustar ni verificar Ridge, logísticas, scores, estados o controles;
- P8 recompone arrays de métricas y el índice ganador, pero no compara los valores escalares ni los tres digests declarados en la decisión (`:259-273`);
- P11 recompone raw y el bootstrap global, pero luego reduce la tabla a un set de IDs y tres claims (`:308-335`). No verifica las **13 filas concretas**, unicidad `(id, instance)`, orientación, medias, intervalos, status/precedencia, bootstraps de soporte común, summaries de sensibilidad, duplicaciones ni `patterns`.

La propia fuente independiente R564 contiene la recomposición que falta: bootstrap común y tokens (`check_proportional_set_valued_native_preflight.py:1313-1332`), inventario y valores de estimandos (`:1333-1378`), sensibilidades (`:1379-1423`), duplicaciones (`:1424-1445`) y patterns (`:1446-1464`). Los bytes actuales parecen correctos y el R564 histórico pasa, pero eso no vuelve fail-closed al checker físico: una tabla final numéricamente falsa o un receipt de paridad fabricado puede conservar `15/15 PASS`.

**Condición verificable de cierre:** portar al checker físico la recomposición independiente completa de P5/P6/P8/P9/P10/P11, comparar directamente contra fuentes R564 congeladas y exigir inventarios/keys exactos. Agregar mutaciones aisladas de coeficientes/refit, cada métrica/digest de decisión, bootstrap común, cada campo de estimando, duplicaciones, sensibilidad, precedencia y patterns; cada una debe fallar con el reason code de su predicado.

### HIGH-2 — El trust boundary de preparación y ejecución física no queda validado de forma durable y fail-closed

La ejecución observada sí construye stages y lanza `setpriv` con reducción real de privilegios (`run_proportional_set_valued_physical_preflight.py:378-414`), y el worker valida allowlist/hashes (`_proportional_set_valued_phase_worker.py:131-152`) y runtime (`:155-187`). El defecto está en la revalidación independiente y en la reutilización del input:

- si el input ya existe, el runner sólo comprueba `package_id` no nulo y SHA del source freeze, y lo reutiliza (`run...py:259-263`); no contrasta los siete hashes del preparation freeze/public manifest, owners, modos, tipos ni package ID derivado;
- P2 verifica keys, algunas identidades, disjunción, conteos, ensemble y vocabulario, pero no hashes del input, dtypes/shapes completos, unicidad, finitud/rangos, `cluster_id==pair_token`, checkpoint axis integral, symlink/hardlink/type/mode ni correspondencia completa escrow-manifests (`check...py:172-195`);
- P3 valida root/fases, identidad/caps, tres flags, set de nombres de módulos, outcomes de probes y threads (`:197-209`), pero no compara hashes/paths de módulos con el freeze, no cubre el worker ejecutable, no verifica `sys.path`, cwd, set/valores exactos del environment, `stage_files`, allowlist por fase, hashes/modos del stage ni identidad y cardinalidad exactas de probes.

Esto contradice expresamente el contrato del plan (`PLAN...md:411-481`, especialmente `:470-481`) y permite que evidencia física fabricada o un input previamente alterado pase el checker. Que los receipts canónicos actuales sean consistentes no resuelve la ausencia de autenticación/recomposición.

**Condición verificable de cierre:** revalidar siempre el input completo contra preparation freeze, public manifest y escrow antes de usarlo; derivar y comparar package ID; comprobar tipos/owners/modos/links y schemas/dtypes/shapes/rangos. En P3, contrastar el receipt íntegro con constantes independientes: seis blobs runtime, hashes y paths staged, `sys.path`, cwd, environment exacto, inputs/hashes/modes y probes exactos por fase. El checker debe rechazar campos extra, faltantes o receipts fabricados.

### MEDIUM-1 — El replay excluye directorios y receipts completos sin comparación semántica

El plan permite excluir sólo campos operativos y exige una comparación semántica exacta por exclusión; prohíbe excluir un directorio completo (`PLAN...md:682-692`). Runner y checker excluyen por nombre todos los `worker_receipt.json`, además de config/bindings y seis top-level, y descartan todo `journals/` con `startswith("journals/")` (`run...py:459-468`; `check...py:337-346`). Luego declaran `semantic_exclusions_valid=true` sin ejecutar comparación semántica alguna. El test que pretende proteger este contrato sólo busca las cadenas `excluded_fields`/`excluded_directories`, por lo que no detecta la exclusión efectiva por `startswith` (`tests/test_proportional_set_valued_physical.py:106-109`). En el run actual sólo se comparan **40/60 archivos**.

**Condición verificable de cierre:** comparar journals, bindings, config y receipts completos tras normalizar exclusivamente los campos enumerados; documentar un schema de normalización por path y prohibir exclusiones por basename/directorio. Agregar una mutación en cada campo semántico excluido y otra que ensanche la exclusión; ambas deben fallar P12.

### MEDIUM-2 — `63/63` mide los casos implementados, no la cobertura mínima aprobada

El plan enumera mutaciones obligatorias de autoridad/preparación, frontera física, ciencia/causalidad, recovery/replay e inventario (`PLAN...md:744-807`). El harness actual concentra P3 en cinco campos de identidad, CUDA y un outcome de probe (`tests/run_proportional_set_valued_physical_mutations.py:113-117`); P5/P6 no mutan modelos ni fits (`:122-129`); P11 sólo muta un raw, bootstrap global y dos claims (`:147-150`); P12 sólo cambia tres booleanos/campos del receipt (`:151-153`). El evidence checker exige únicamente total 63, igualdad expected/observed y booleano `passed` (`check...py:409-416`), no un catálogo mínimo de case IDs/cobertura.

Por tanto, **63/63 es verdadero como conteo pero falso como demostración de la cobertura contractual**. Faltan, entre otros, runtime hash/path/sys.path/env/stage/cwd/receipt fabricado; dtypes/shapes/links; posterior y policy recomputados; orientación/CI/status/pattern; journal reescrito/futuro; y exclusión amplia.

**Condición verificable de cierre:** convertir el catálogo obligatorio del plan en una lista versionada de case IDs y hacer que el evidence checker exija exactamente ese conjunto, no sólo `total==63`. Cada caso debe modificar un elemento, ejecutar el predicado pertinente y conservar reason code observado.

### MEDIUM-3 — Restart/recovery no prueba todas las invariantes de la máquina de estados

`validate_completed_prefix` deja de recorrer en el primer par journal/fase ausente y después sólo rechaza directorios de fases futuras (`run_proportional_set_valued_physical_preflight.py:425-436`); un journal futuro huérfano no entra en esa segunda comprobación y puede ser reemplazado después, pese a que el plan prohíbe reescribir journals y exige rechazar artefactos futuros (`PLAN...md:603-607`). Tampoco revalida allí `previous_state`, `phase`, `package_id`, `maximum_truth_materialized`, inputs ni digest del receipt.

El harness recovery sí inyecta siete crashes después de promoción, pero marca `scientific_byte_exact=true` porque el runner devolvió `replay=true`; no recalcula los hashes y hereda la exclusión amplia de P12 (`tests/run_proportional_set_valued_physical_recovery.py:52-65`). No cubre de forma separada todos los intervalos e invariantes anunciados en `PLAN...md:655-670`.

**Condición verificable de cierre:** validar el inventario total de journals y fases antes de decidir el prefix; rechazar todo journal/directorio futuro, journal huérfano, duplicado o divergente; validar todos sus campos y hashes; publicar journals con creación exclusiva. En recovery, recomputar directamente igualdad funcional/científica y comparación semántica normalizada, y añadir casos explícitos para ambos lados del intervalo promoción/journal y para cada divergencia de resume.

### LOW-1 — La aplicación de budgets no coincide exactamente con la config aprobada

P15 usa un literal de 512 MiB sobre un único root y no comprueba el agregado primaria+replay ni el límite individual de archivo (`check...py:372-380`). El evidence checker compara wall primaria+replay contra `checker_plus_mutations_seconds`, no contra el budget específico de esa categoría (`:423-428`), y usa literales de scratch/evidence en lugar de todos los valores de config (`:432-435`). No hay excedencia observada, por lo que el defecto es de enforcement, no un exceso actual.

**Condición verificable de cierre:** consumir exclusivamente las claves de config, comprobar agregado primario+replay, máximo por archivo, scratch y evidence, y agregar mutaciones independientes exactamente alrededor de cada umbral.

## 4. Cierre

La implementación ejecutada ofrece evidencia positiva real: separación en siete subprocesses, permisos y receipts plausibles, paridad observada con R564, determinismo de 40 artefactos científicos y claims correctamente acotados. Sin embargo, el criterio aprobado no era sólo producir bytes correctos una vez, sino demostrar con un checker independiente y adversarial que bytes o proveniencia incorrectos no pueden ser aceptados. HIGH-1 y HIGH-2 impiden esa conclusión; MEDIUM-1 a MEDIUM-3 muestran que replay, cobertura adversarial y restart tampoco cumplen aún su contrato.

**Resultado técnico final: REVISE.** No corresponde etiquetar el paquete como plan-compliant hasta satisfacer y volver a auditar las condiciones anteriores. Este cierre no decide `GO/NO-GO`, no promueve arquitectura y no amplía la evidencia más allá del preflight físico CPU sobre datos históricos abiertos.
