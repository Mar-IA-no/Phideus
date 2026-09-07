# Cierre del runner set-valued nativo antes del draw

Fecha: 2026-09-07  
Estado técnico: `RUNNER_PREFLIGHT_VALID`  
Régimen: CPU, datos históricos ya abiertos, sin draw fresco, monitor ni lockbox

## Qué quedó construido

El relevo abierto por el dual native freeze ya tiene una implementación
ejecutable para su única rama habilitada. El runner materializa de punta a punta
el factorial `MARGINAL/JOINT × HARD/CONTEXTUAL`: prepara poblaciones disjuntas,
ajusta el posterior marginal y el conjunto joint, construye readers ligados a
cada posterior, selecciona acciones sin abrir truth durante la aplicación,
transporta cinco controles matched por posterior y preserva estimandos,
bootstrap, sensibilidad, estados y arrays raw suficientes para reanálisis.

La implementación canónica está en:

- `src/geometria_proporcional/proportional_set_valued_native.py`;
- `experiments/geometria_proporcional/run_proportional_set_valued_native_preflight.py`;
- `experiments/geometria_proporcional/check_proportional_set_valued_native_preflight.py`;
- `experiments/geometria_proporcional/configs/proportional_set_valued_native_preflight_v1.json`;
- `tests/test_proportional_set_valued_native.py`;
- `tests/run_proportional_set_valued_mutations.py`.

Los artefactos locales preservados son
`data/geometria_proporcional/proportional_set_valued_native_preflight_v1/` y su
replay `proportional_set_valued_native_preflight_replay_v1/`. Cada árbol contiene
35 archivos, 17 NPZ, 17 JSON, 585 arrays y 35.701.634 valores. Los 32 archivos
comparables son byte-exactos.

## Evidencia de cierre

La corrida primaria necesitó `28,27180427312851 s` y alcanzó `986.238.976`
bytes de RSS; el replay necesitó `28,148873522877693 s` y `985.055.232` bytes.
La suma, `56,4206777960062 s`, queda por debajo del límite duro de 1.800 s y
ambos procesos mantuvieron CUDA invisible, un thread por backend y torch sin
importar.

El checker independiente recompone fuentes, fases, fits, readers, selección,
controles, estimandos, sensibilidad y representación canónica. Primario y replay
pasaron `14/14` predicados. Las pruebas unitarias pasaron `9/9`. La suite
adversarial v3 rechazó `54/54` alteraciones y también exigió que los dos artefactos
válidos pasaran; su ejecución principal consumió `141,19438233226538 s` y
`642.932.736` bytes de RSS pico hijo, dentro de `900 s` y `1,5 GiB`.

R562 abrió `1 HIGH / 4 MEDIUM / 1 LOW`. Las correcciones hicieron que el checker
reextrajera los cuatro bundles, recompusiera el estado OOF joint, sensibilidad,
duplicaciones y controles, renderizara el reporte desde datos estructurados y
midiera la campaña completa. R563 confirmó resueltos los seis hallazgos y dejó
una deuda baja sobre canonicalidad JSON. Esa deuda también se corrigió: R564
verificó los 17 JSON de cada artefacto, rechazó constantes no finitas y la
reescritura pretty-printed aun con manifest refrescado, y cerró `PASS — 0 HIGH /
0 MEDIUM / 0 LOW`.

Fuentes de auditoría:

- `agent_reports/562_proportional_set_valued_runner_final_audit.md`;
- `agent_reports/563_proportional_set_valued_runner_final_reaudit.md`;
- `agent_reports/564_proportional_set_valued_runner_final_reaudit.md`;
- `experiment_reports/proportional_set_valued_native_mutation_suite_v3.json`.

## Qué muestran y qué no muestran los datos abiertos

El diagnóstico de implementación seleccionó el candidato 48 en ambos
posteriors. `JOINT` mejoró el exact-set NLL frente a `MARGINAL` (`-0,09250258`,
IC95 `[-0,11763691,-0,06799824]`), pero el Brier marginal no se resolvió
(`+0,00118569`, IC95 `[-0,00051026,+0,00294318]`). Los dos posteriors separaron
sus target-shuffles. Los readers contextuales mejoraron regret medio e
incompatibilidad frente a hard, pero empeoraron worst regret. La comparación con
controles matched no fue evaluable: el soporte común cubrió sólo `67/215` tokens
en marginal y `74/235` en joint. Por eso los patrones conjuntos permanecen
`false`.

Estas cifras ejercitan el contrato sobre material abierto. No estiman transporte
a una realización nueva, variabilidad de entrenamiento ni atribución
prospectiva; tampoco autorizan promover `JOINT`, `CONTEXTUAL` o su combinación.

## Relevo finito

El próximo objetivo no abre todavía el draw. Debe diseñar y auditar el paquete
de ejecución prospectiva físicamente separada: schemas finales, source freeze,
workers por fase, permisos, escrow, receipts, recovery, replay y presupuesto.
Sólo después de que ese paquete quede cerrado tendrá sentido materializar una
realización fresca. La GPU permanece suspendida y este relevo es CPU-native.

No hubo decisión `GO/NO-GO`, promoción arquitectónica ni reapertura de la
campaña bibliográfica.
