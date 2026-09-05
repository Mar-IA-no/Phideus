# Ola 59 — auditoría independiente de implementación

## Dictamen: REVISE

El commit estable implementa la mayor parte de la mecánica científica prevista:
los workers físicos corren sin privilegios, las vistas de calibración y monitor
son inference-safe, los mínimos y soportes usan `T_primary`, los controles de
desplazamiento máximo alcanzan el máximo por construcción, los shards no
intervienen en el monitor y la suite relevante completa pasa. Sin embargo, la
implementación todavía no preserva la frontera prospectiva que esos artefactos
declaran.

Hay cinco bloqueantes P1 reproducidos. Los dos más graves son directos:

1. VALIDATE y MONITOR-EVALUATE aceptan arrays de acciones distintos de los
   hashes contenidos en sus freezes y abren truth igualmente;
2. la reanudación denominada hash-identical acepta un `analysis.json` alterado
   después del fallo y conserva incluso un `scientific_decision` no nulo.

También falta hacer vinculante la auditoría de implementación antes del draw,
el contrato de fallos no se aplica al preparador y el comparador replay ignora
`recovery_amendment.json`. Por tanto, **no debe crearse el triplete de secretos
ni iniciarse el fresh draw con `05c582f`**. No hay P0 observado porque los dos
outputs canónicos Wave 59 están ausentes y la config sigue
`IMPLEMENTATION_PRE_DRAW`; los findings describen capacidad de invalidación,
no leakage consumado.

## Identidad y estado auditado

- Commit corto solicitado: `05c582f`.
- Commit completo observado:
  `05c582fa42e66fe0aed1c09185fa6a706e592eef`.
- Parent: `cbdcbe0ec920862161ff44c6bdf200b73d104801`.
- Subject: `Complete Wave 59 prospective runtime boundaries`.
- Plan final R420:
  `Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_59_FRESH_HGB_GUARD_BRACKET_PLAN.md`.
- SHA-256 del plan observado:
  `7e74f892bf27c4c51fa5f44e4e04b4564f7d63d986d8cc69c7316663bc5eabfb`.
- Auditoría de plan aceptada R420: SHA-256
  `c7d1ed28554bb3b1174bfb787ae9469ee20392062f0c36deda7d0d10dcab0134`.
- Worktree observado: `HEAD=05c582f`, sin cambios tracked o untracked
  visibles por `git status --short --untracked-files=all` antes de escribir
  este informe.
- Outputs canónicos primario y replay: ambos ausentes.
- Config: `IMPLEMENTATION_PRE_DRAW`, con `implementation_binding` y
  `source_sha256` todavía nulos (`config:24-29,335`).

La auditoría fue CPU-only, con `CUDA_VISIBLE_DEVICES=''` y cuatro hilos. No se
consultó GPU, web ni Mendieta. Se leyeron completos el plan, config, módulo
estadístico, worker, coordinador y tests Wave 59; el preparador compartido se
contrastó en las ramas modificadas y en sus fronteras heredadas Wave 56/57.

## Findings bloqueantes

### P1 — los freezes no autorizan realmente los inputs que llegan a truth

El plan exige que VALIDATE verifique el freeze de calibración antes de
materializar truth y que MONITOR-EVALUATE sólo reciba las acciones promovidas
por `monitor_action_freeze.json` (`plan:276-305`). El código invierte esa
autoridad:

- `validate_stage` sólo verifica que `phase_request.json` describa los archivos
  presentes y sus hashes actuales; no contrasta esos hashes con el freeze de la
  fase anterior (`_wave59_phase_worker.py:135-150`);
- el coordinador genera ese request a partir de cualquier copia que tenga en el
  stage (`run_wave59_hgb_guard_bracket.py:248-259`);
- `run_validate` abre `truth_bundle.npz` en su primera instrucción, carga
  policy arrays y scores, pero nunca comprueba los campos `files` de
  `calibration_freeze.json` (`_wave59_phase_worker.py:387-400`);
- `run_monitor_evaluate` abre truth y evalúa las acciones antes incluso de
  cargar `monitor_action_freeze.json`; después sólo usa ese JSON para soportes y
  estado de controles (`_wave59_phase_worker.py:733-758`);
- el coordinador copia truth y freeze juntos al sandbox sin una verificación
  cruzada previa (`run_wave59_hgb_guard_bracket.py:1030-1043,1065-1094`).

La cadena también queda incompleta nominalmente: MONITOR-APPLY no recibe
`validation_freeze.json` y su action freeze sólo menciona FIT y calibración
(`_wave59_phase_worker.py:537-550`; runner `1048-1060`), aunque el plan exige
vincular preparación, FIT, calibración y validación (`plan:290-293`).

**Reproducción ejecutada.** Sobre el output temporal creado por la propia suite:

1. se copió `validation_policy_arrays.npz`, se cambió una acción y se mantuvo el
   `calibration_freeze.json` original;
2. hash congelado:
   `28014bffbccf3b71e6d07a37a31997cbe5a5c343ce9dbc471d4e47e28210724b`;
3. hash alterado:
   `a22e6eab84081f202d7b2e493293b1744cda4edb547c2ca0308579924d11fc3a`;
4. `_run_phase(..., "validate", ...)` terminó
   `VALIDATION_COMPLETE` y publicó `validation_freeze.json`.

El mismo probe sobre monitor produjo:

- hash congelado de acciones:
  `1b87a2d69ac9843fd6353714d7d55fdfdbc1353c6a2b0b1644c7f6b95b2182d5`;
- hash alterado:
  `5dcccfb43bb0baab14c0f5270489c74bdcb0cd731b395958baee496dd403eb3e`;
- resultado: `worker_status="COMPLETE"` y `analysis.json` publicado.

Esto permite una reselección o sustitución post-freeze antes de abrir truth sin
que el runtime la detecte. Los permisos `0444` reducen accidentes, pero no
sustituyen la verificación criptográfica que define el protocolo.

**Corrección mínima.** Implementar un validador común de freeze que compare
schema, fase y SHA-256 de cada input congelado. El coordinador debe ejecutarlo
antes de copiar cualquier truth al sandbox; el worker debe repetirlo antes de
leer truth. Añadir `validation_freeze.json` como input de MONITOR-APPLY y enlazar
en `monitor_action_freeze.json` config, source bindings, preparation, FIT,
calibración, validación, states, scores y acciones. Añadir tests negativos que
alteren cada artefacto congelado sin regenerar su freeze.

### P1 — el pre-draw no exige commit, auditoría ni hashes de implementación

Los campos nulos son correctos mientras la config siga en estado de borrador.
El problema es que basta cambiar sólo `status` para atravesar el preflight:

- `validate_pre_draw_config` no examina `implementation_binding`,
  `source_sha256`, el hash/path de R421 ni la versión efectiva de sklearn
  (`wave59_hgb_guard_bracket.py:86-196`);
- la rama Wave 59 del preparador sólo exige el string de estado, una lista de
  sources y que `source_binding` sea algún diccionario
  (`prepare_wave56_fresh.py:459-496`);
- `require_sources_at_head` captura dinámicamente el HEAD vigente, pero no lo
  compara con el commit auditado (`prepare_wave56_fresh.py:427-446,751-812`);
- cada fase vuelve a copiar módulos desde el worktree actual sin contrastarlos
  con los hashes congelados en PREPARE (`run_wave59_hgb_guard_bracket.py:235-245`);
- el runner acepta cualquier policy manifest estructuralmente válido y tampoco
  verifica los bundles contra `preparation_freeze.json`
  (`run_wave59_hgb_guard_bracket.py:221-232,978-1008`).

**Reproducción ejecutada.** Se cargó la config actual, se cambió solamente:

```python
cfg["status"] = "FROZEN_PROSPECTIVE_PROTOCOL_PRE_KEY_DRAW"
```

y se ejecutaron consecutivamente `validate_pre_draw_config(cfg)` y
`prepare_wave56_fresh.validate_prospective_config(cfg)`. Ambas aceptaron:

```json
{
  "accepted": true,
  "implementation_binding": {
    "audit_path": null,
    "audit_sha256": null,
    "commit": null,
    "status": "PENDING_IMPLEMENTATION_AUDIT"
  },
  "source_sha256": null
}
```

El entorno observado sí tiene sklearn `1.8.0`; falta el control, no la
dependencia actual.

**Corrección mínima.** Antes de aceptar el estado frozen, exigir audit path/hash
válidos, commit de implementación auditado y mapa completo de SHA-256 por cada
`required_execution_source`. Verificar esos valores —y la versión efectiva de
dependencias— en PREPARE y antes de cada worker. El policy manifest y los cinco
bundles deben contrastarse contra los bindings/preparation freeze, no sólo
contra hashes calculados en el mismo stage.

### P1 — la reanudación “hash-identical” acepta estado científico alterado

`restore_identical_hash_attempt` verifica schema del fallo, path original y
config, pero ignora por completo `failure_inventory.json`, sus hashes y los
outputs de journals previos (`run_wave59_hgb_guard_bracket.py:433-461`). Para
una evaluación ya promovida, el camino de resume comprueba sólo los hashes de
inputs y la mera existencia de tres outputs; no verifica esos outputs contra
`monitor_evaluate.json` (`run_wave59_hgb_guard_bracket.py:1072-1086`).

**Reproducción ejecutada.** Se archivó una corrida completa temporal, se cambió
`analysis.json["scientific_decision"]` por
`"TAMPERED_AFTER_FAILURE"`, se restauró y se llamó de nuevo a `execute`. La
reanudación terminó sin error y preservó exactamente:

```json
{
  "resume_accepted": true,
  "tampered_scientific_decision_after_resume": "TAMPERED_AFTER_FAILURE"
}
```

Esto contradice tanto la reanudación hash-idéntica como
`scientific_decision=null` (`plan:330-347`).

**Corrección mínima.** Verificar íntegramente el failure inventory antes de
copiar, rechazar todo hash faltante o distinto, revalidar cada journal y sus
outputs en el destino restaurado, y enlazar explícitamente los tres outputs
promovidos de MONITOR-EVALUATE con su journal. La recuperación también debe
revalidar el binding de código/config/dependencias del finding anterior.

### P1 — el schema de fallos es declarativo, no una matriz cerrada efectiva

Hay dos implementaciones de fallo incompatibles:

- el runner escribe `wave59-failed-attempt-v1`, pero clasifica sólo archivos
  existentes; no calcula obligatorios por último journal y rellena siempre
  `missing_required_through_last_journal=[]`, `extra=[]`, `overlap=[]` y
  `unclassified=[]` (`run_wave59_hgb_guard_bracket.py:861-946`);
- el preparador compartido sigue archivando fallos Wave 59 con el formato
  legacy: `FAILURE.json` sin `schema_version`, con el mensaje crudo y sin
  `failure_inventory.json` (`prepare_wave56_fresh.py:2732-2765`).

**Reproducciones ejecutadas.** Un árbol con `journals/prepare.json`, pero sin
`pre_generation_freeze.json` ni `preparation_freeze.json`, fue aceptado por
`archive_failed_attempt`; su inventario declaró las tres listas vacías. En un
fallo inyectado `before_escrow` del preparador Wave 59 se observó:

```json
{
  "schema_version": null,
  "failure_inventory_exists": false,
  "raw_message_present": true
}
```

Los terminales `NOT_EVALUABLE` tienen un hueco análogo: FIT, calibración y
monitor publican `*_not_evaluable.json` (`_wave59_phase_worker.py:259-261,
366-368,522-524`), pero esos paths no pertenecen a ninguna clase y `execute`
retorna antes de crear manifest (`runner:1011-1013,1027-1029,1062-1064`). Esto
deja sin closed-world precisamente los cierres que no deben habilitar redraw.

**Corrección mínima.** Unificar preparación y análisis bajo un único escritor
Wave 59 de failure schema; derivar el conjunto requerido/prohibido de paths a
partir de `run_role`, recovery context y último journal, y calcular realmente
missing/extra/overlap/unclassified. Añadir clases y manifests terminales para
cada salida `NOT_EVALUABLE`. Los tests deben borrar un archivo previo, añadir
uno de una fase futura y exigir rechazo en ambos casos.

### P1 — replay puede declarar exactos amendments distintos

R420 ubicó `recovery_amendment.json` en `scientific_exact` y exige igualdad
byte-exacta entre primary y replay (`plan:501-545`). `_artifact_classes` sí lo
añade cuando detecta el archivo, pero `_scientific_paths` y `compare_runs` no lo
comparan ni exigen que ambos lados compartan recovery context
(`run_wave59_hgb_guard_bracket.py:529-574,642-730,748-804`). Además, el contexto
se infiere circularmente por presencia del archivo (`runner:807-811`), por lo
que su ausencia no demuestra modo normal.

**Reproducción ejecutada.** Se añadieron amendments de bytes diferentes a dos
outputs temporalmente exactos y se llamó `compare_runs(replay, primary)`:

```json
{
  "amendments_differ": true,
  "amendment_in_scientific_exact": false,
  "all_exact": true
}
```

**Corrección mínima.** Derivar recovery context de metadata previamente
autenticada, exigir igualdad del contexto entre ambos lados y añadir el
amendment al conjunto exacto condicional. El comparador debe fallar por archivo
ausente, extra o hash diferente antes de declarar `all_exact=true`.

## Findings importantes no prospectivamente invalidantes

### P2 — falta el contraste monitor contra proposer-only

El plan predeclara cada política principal menos `HGB-PROPOSER-ONLY` como
contraste de atribución (`plan:415-427`). `run_monitor_evaluate` sólo publica
`mean_vs_hard`, `tail_vs_hard`, `head_to_head` y los 36 factoriales
(`_wave59_phase_worker.py:743-758`). En el artefacto histórico del test,
`jq '.contrasts|keys' analysis.json` confirmó exactamente esas cuatro claves.
Los shards sí calculan el diagnóstico local, pero no sustituyen el estimando
global de monitor.

**Corrección mínima.** Añadir los dos contrastes principales contra
`HGB-PROPOSER-ONLY`, con las métricas predeclaradas y los mismos índices
bootstrap. Incluirlos en tests y replay exacto.

### P2 — los presupuestos se registran, pero no se aplican

La config fija `1800 s` por corrida, `3600 s` combinados, `8 GiB` RSS y GPU
prohibida (`config:235-240`; `plan:596-613`). Los workers analíticos sí reciben
`CUDA_VISIBLE_DEVICES=''` y límites de cuatro hilos (`runner:291-303`), pero
`subprocess.run` no tiene timeout ni límite de memoria; duración y RSS sólo se
registran después de terminar (`runner:304-327,949-967`). Tampoco se suma el
presupuesto primary+replay. El worker de inferencia heredado fija hilos de
Torch, pero su entorno de preparación no fija `CUDA_VISIBLE_DEVICES`
(`prepare_wave56_fresh.py:2265-2287`).

**Corrección mínima.** Aplicar un deadline cancelable por fase/corrida y un
límite observable de memoria antes del subprocess; abortar y preservar al
superar el presupuesto. Mantener un receipt acumulado primary+replay y fijar
CUDA invisible también en el preparador y el worker de inferencia.

## Áreas verificadas sin finding material

- **Mínimos y unidad:** FIT, calibración, shards y monitor cuentan filas y pair
  tokens sobre primary; soporte conserva `authorized_rows` y
  `authorized_pair_tokens` (`_wave59_phase_worker.py:179-201,217-261,331-368,
  404-432,507-524`; módulo `259-282`).
- **Controles:** las dos familias usan strata `(policy_index,
  disagreement_count)`, PCG64, máximo Hamming, hashes distintos y thresholds
  `.25/.02` (`wave59_hgb_guard_bracket.py:289-429,565-630`).
- **Shards:** la asignación es determinista, recalibra sólo dentro de validation,
  publica siete deltas y no modifica acciones de monitor
  (`_wave59_phase_worker.py:402-491`).
- **Modelos y scorer:** hay 2 proposers, 4 guards y 10 controles; el scorer HGB
  portable se contrasta con `rtol=0`, `atol=2e-15`, y los joblib son copias
  funcionales (`wave59_hgb_guard_bracket.py:432-562`; worker `262-323`).
- **Separación de procesos:** la corrida histórica usada por tests confirmó UID
  y GID `65534`, capabilities efectivas cero, `NoNewPrivs=1`, grupos vacíos y
  probes de truth denegados. Esto es evidencia real de aislamiento DAC, aunque
  no repara el desacople criptográfico de los freezes.
- **Compatibilidad Wave 56/57:** las modificaciones del preparador están
  condicionadas por `WAVE59_CONFIG_SCHEMA`; la suite combinada de preparación,
  recovery y protocolos heredados pasó completa. No se observó deriva
  byte-semántica de Wave 56/57 en el alcance ejercitado.

## Comandos y resultados

```text
git rev-parse HEAD
# 05c582fa42e66fe0aed1c09185fa6a706e592eef

sha256sum Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_59_FRESH_HGB_GUARD_BRACKET_PLAN.md
# 7e74f892bf27c4c51fa5f44e4e04b4564f7d63d986d8cc69c7316663bc5eabfb

git diff --check 05c582f^ 05c582f
# exit 0

venv/bin/python -m py_compile \
  src/geometria_proporcional/wave59_hgb_guard_bracket.py \
  experiments/geometria_proporcional/_wave59_phase_worker.py \
  experiments/geometria_proporcional/run_wave59_hgb_guard_bracket.py \
  experiments/geometria_proporcional/prepare_wave56_fresh.py \
  tests/test_wave59_hgb_guard_bracket.py tests/test_wave59_prospective.py
# exit 0

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 venv/bin/python -m pytest -q \
  tests/test_wave59_hgb_guard_bracket.py tests/test_wave59_prospective.py \
  tests/test_wave56_preoracle_recovery.py tests/test_wave56_prospective.py \
  tests/test_wave57_prospective.py
# 188 passed in 250.85s
```

Los probes quirúrgicos se ejecutaron con `venv/bin/python` sobre copias en
`/tmp`: mutación NPZ seguida de `_run_phase`, config en memoria seguida de ambos
validadores, archive→mutación→restore→execute, y comparación de amendments
distintos. No modificaron el repositorio ni los outputs canónicos.

## Hashes del corpus de implementación

| Archivo | SHA-256 |
|---|---|
| `experiments/geometria_proporcional/configs/wave59_fresh_hgb_guard_bracket.json` | `dc6d10141307b5d4dd1456b9082a558e0aa7f133438dd5eb70995b37b2113e18` |
| `src/geometria_proporcional/wave59_hgb_guard_bracket.py` | `b174a8807ae051abd6d1b63e7b63b12e0e901181239e82377be9242b53939c9d` |
| `experiments/geometria_proporcional/_wave59_phase_worker.py` | `8ece758363ff07f52648f922e9e2aaf1e1a448dc394850e9a4045483a16a9c1e` |
| `experiments/geometria_proporcional/run_wave59_hgb_guard_bracket.py` | `01e3aa855707352e5c1ba21624a2180138a188b810b75f72f0343d3cf8b9a2f0` |
| `experiments/geometria_proporcional/prepare_wave56_fresh.py` | `68826ccb7bf879e92f35eada5d7995094d2b8c4a7d66022a5da93564f53198d8` |
| `tests/test_wave59_hgb_guard_bracket.py` | `d837a8ae4948eec6a59c3a62c751935d2d90521a7d7178f7411006d002898f51` |
| `tests/test_wave59_prospective.py` | `2da9ffcb7f7675b22ba2a54a1ecde8c63b600a35fe5ea0ff71f9e33ba81eb3a5` |

## Condición para PASS

Resolver los cinco P1, añadir los tests de mutación/fallo que hoy faltan y
reauditar el commit corregido antes de cambiar la config a frozen o crear el
triplete. Los dos P2 deben resolverse en la misma pasada porque corresponden a
un estimando predeclarado y a límites operativos explícitos, no a cosmética.

Este dictamen no juzga resultados científicos: todavía no existen. No promueve
arquitectura, no declara techo y deja cualquier `GO/NO-GO` al usuario.
