# R398 — Reauditoría independiente de implementación de Ola 57

## Dictamen

**PASS**

El estado terminado en
`a559684681ef5f91b641a5ae4869bb4b5903f058` resuelve los cinco findings de
R397. No encontré findings materiales nuevos en el delta completo contra
`d29d1aa52256f112bc6fa2e1569dc9a7e20e77f7`.

Este `PASS` se limita a la correspondencia entre el paquete implementado, el
plan congelado en `b271b8885cd5bcf08421fecf9e81a9df884bea2a` y las correcciones
pedidas por R397. No adjudica el patrón prospectivo, no promueve arquitectura y
no constituye una decisión `GO/NO-GO`.

No ejecuté preparación oficial, extracción de claves, draw, oracle fresco,
materialización oficial de labels ni GPU/CUDA.

## Identidad y alcance

- HEAD auditado: `a559684681ef5f91b641a5ae4869bb4b5903f058`.
- Base de comparación: `d29d1aa52256f112bc6fa2e1569dc9a7e20e77f7`.
- Plan vigente: `b271b8885cd5bcf08421fecf9e81a9df884bea2a`.
- Auditoría base: R397, dictamen `REVISE`.
- Delta leído completo: seis archivos, `778 insertions`, `63 deletions`.
- Estado Git previo a este informe: limpio.
- Ejecución: CPU, con `CUDA_VISIBLE_DEVICES=''` y threads BLAS/OMP/MKL fijados
  a uno.

## Resolución de findings R397

### F1 — CERRADO — Freeze completo y rechazo del worker incorrecto

`validate_wave57_frozen_config()` autentica ahora el objeto de configuración
completo mediante una serialización JSON canónica y un SHA-256 congelado
(`wave57_tail_guard.py:32-52`). El hash observado del config canónico fue
exactamente:

```text
a986b642cd75cb66120232de0df628b82a21c734f189c96e9ac1146bcdedbdda
```

Repetí adversarialmente las diez familias señaladas por R397 y agregué las dos
familias nuevas que el delta incorpora: grilla escalar y cierre del runtime. En
cada caso, tanto `validate_wave57_frozen_config()` como
`validate_prospective_config()` rechazaron la mutación:

```text
hard_set_tau                 REJECTED / REJECTED
incompatible_regret_penalty  REJECTED / REJECTED
selection margin             REJECTED / REJECTED
diagnostic criterion value   REJECTED / REJECTED
quantile_method              REJECTED / REJECTED
harm_epsilon                 REJECTED / REJECTED
bootstrap interval           REJECTED / REJECTED
shard salt                   REJECTED / REJECTED
boundary semantics           REJECTED / REJECTED
Wave 56 phase worker         REJECTED / REJECTED
scalar gamma grid            REJECTED / REJECTED
phase runtime sources        REJECTED / REJECTED
accepted_mutations           []
```

La prueba parametrizada versionada reproduce las doce familias
(`tests/test_wave57_prospective.py:87-118`). En particular, sustituir el
entrypoint por `_wave56_phase_worker.py` ya no llega a
`build_phase_runtime()`: el config falla primero contra el hash completo.

### F2 — CERRADO — Freezes y config visibles y adversariales en replay

`compare_reference()` incorpora ahora tres checks nombrados que faltaban:

- `preparation/prospective_config`;
- `fit/fit_freeze.json`;
- `select/selection_freeze.json`.

Los freezes se comparan byte por byte y el config queda autenticado a la vez por
el SHA del archivo y por igualdad del objeto embebido
(`run_wave56_contextual_gate.py:939-970`). El receipt sintético exacto expuso
trece checks, incluidos los tres anteriores.

Ejecuté tres mutaciones independientes sobre paquetes sintéticos completos:

```text
prospective config mutation  REJECTED
fit_freeze.json mutation     REJECTED
selection_freeze.json        REJECTED
```

La comparación del caso no mutado devolvió `all_exact=True`. Los tests
versionados prueban la presencia de los tres checks y rechazan por separado los
dos freezes (`tests/test_wave57_prospective.py:413-450`).

La comparación de config dentro de `compare_reference()` es condicional porque
el coordinador sigue siendo compartido con fixtures legacy. Esto no abre un
hueco en la ruta oficial Wave 57: `current_state()` exige
`preparation_freeze.json` (`run_wave56_contextual_gate.py:324-326`) y
`validate_replay_reference()` exige igualdad completa entre el freeze de replay
y el primario antes de ADJUDICATE (`:1002-1040`). Por tanto, el receipt oficial
sí contiene el check nombrado y no puede obtener exactitud sin config.

### F3 — CERRADO — Granularidad `NOT_EVALUABLE`

La cobertura nueva verifica los cinco límites pedidos por R397:

1. proposer `hard_only` produce secuencia terminal `NOT_EVALUABLE`
   (`tests/test_wave57_prospective.py:157-179`);
2. celdas guard con poco soporte quedan no evaluables individualmente mientras
   otras celdas siguen evaluables (`:182-201`);
3. un sham que falla Hamming/permutabilidad no termina FIT, SELECT ni el brazo
   principal (`:204-283`);
4. un shard con `39 < 40` tokens queda `NOT_EVALUABLE` sin terminar SELECT y el
   otro shard continúa (`:286-339`);
5. el mismo test registra mínimos `(40,25)` para full y `(20,12)` para shard.

El código mantiene la misma granularidad: SELECT completo usa los mínimos full,
cada shard comprueba primero sus mínimos locales y sólo los shards habilitados
llaman la secuencia con los mínimos reducidos
(`_wave57_phase_worker.py:417-454`).

### F4 — CERRADO — Support set evaluable con summary y estados sham

Cada support set evaluable preserva ahora:

- índices bootstrap locales;
- summaries de todos los brazos, incluido
  `mean_plus_shuffled_harm_guard`;
- estado por brazo;
- estados de las cinco réplicas;
- contrastes locales.

La implementación está en `_wave57_phase_worker.py:796-847`. El fixture
sintético fuerza un set con `40 >= 30` tokens y verifica summary sham no nula,
estado `PASS`, cinco estados de réplica y contraste
`main_minus_shuffled` (`tests/test_wave57_prospective.py:399-407`). El probe
adversarial adicional de sham inválido confirmó que ese mismo support set pasa
a summary `None`, estado `NOT_EVALUABLE` y contraste no numérico.

### F5 — CERRADO — Gate escalar y slices observacionales

La grilla de Ola 55 está declarada y congelada como
`scalar_gamma_grid=[0.0,0.01,0.02,0.05,0.1,0.2,0.4,"hard_only"]`. SELECT ejecuta
la selección escalar, conserva su grilla completa, congela el operating point y
guarda score, threshold, acciones, overrides y métricas
(`_wave57_phase_worker.py:428-432,557-600`). ADJUDICATE transporta ese operating
point sin reselección y publica summary y diagnósticos (`:653-659,719,876-885`).

Los slices observacionales y las 24 políticas individuales se reportan mediante
`system_sensitivities()`: la salida contiene `primary_by_policy` y
`all_in_catalog_by_observational_slice`, con estrato y cardinalidad como ejes de
reporte, no como features. El smoke sintético verifica ambos diagnósticos
(`tests/test_wave57_prospective.py:408-411`).

## Control adicional — sham `NOT_EVALUABLE`

Forcé una réplica sham a `NOT_EVALUABLE`, mantuve FIT/SELECT/ADJUDICATE vivos y
forcé además un support set evaluable. El resultado fue:

```text
global sham summary          None
global main_minus_shuffled   NOT_EVALUABLE
diagnostic_condition_5       NOT_EVALUABLE
support-set sham summary     None
support-set sham status      NOT_EVALUABLE
support-set contrast         NOT_EVALUABLE
sham average arrays          []
replicate evaluability       [False, True, True, True, True]
```

La implementación sólo construye el promedio si las cinco réplicas son
evaluables; si falla una, no publica arrays de promedio y sustituye tanto el
contraste global como el local por estados explícitos
(`_wave57_phase_worker.py:671-752,791-793,815-838,855-858`). No encontré un
promedio o contraste numérico falso.

## Tests y comprobaciones

```text
git diff --check d29d1aa... a559684...
# limpio

CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 venv/bin/python -m pytest -q \
  tests/test_wave57_tail_guard.py \
  tests/test_wave57_prospective.py \
  tests/test_wave56_prospective.py \
  tests/test_wave56_preoracle_recovery.py
# 144 passed in 218.27s

CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 venv/bin/python -m pytest -q
# 487 passed in 267.44s
```

También ejecuté dos probes CPU aislados en directorios temporales: el primero
mutó las doce familias de config y los tres componentes de replay; el segundo
transportó un sham inválido hasta un monitor con support set evaluable. Ninguno
escribió dentro del paquete oficial.

## SHA-256 de inputs principales

```text
db0d69ba197bfa0dcf853c9cecc1f848487ccc25910424036ebcde745a1ccedd  WAVE_57_CONTEXTUAL_TAIL_GUARD_PLAN.md
36bd2814799f40e1085762e5085aeaa1d9206624f8f8c1a2ff33bd444fd77616  397_wave57_contextual_tail_guard_implementation_audit.md
35698118492bf412c5cf68e86c3e72e02dc02c597cc2dc95bfb0e701a5f47465  _wave57_phase_worker.py
fb21a43cb4e356a7f10c293e5cb0c0fc037a8ecc206883e99122115b18812a83  wave57_contextual_tail_guard_fresh.json
f37468415b8eabc822546d1c4d937336d794e301fc9b12713f61586cbbf0e112  run_wave56_contextual_gate.py
059572890c71de4723b35d44d9c7c91d1698e5d515ba5d2551e971180b479a99  wave57_tail_guard.py
5238778afb0008c200529b9883430d321793023006f8b57a5d9c1ec12022e52b  test_wave57_prospective.py
```

## Cierre

Los cinco findings de R397 quedan cerrados en código y por evidencia
adversarial. La suite integral CPU pasa `487/487`, el delta no presenta errores
de whitespace y no quedan findings materiales que impidan `PASS` de
implementación antes del draw.
