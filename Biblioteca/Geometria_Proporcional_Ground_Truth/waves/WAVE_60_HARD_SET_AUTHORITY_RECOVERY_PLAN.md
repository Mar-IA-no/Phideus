# Ola 60 — recuperación de la autoridad `hard_set_tau` tras el terminal v3

> **Estado:** `PRE-IMPLEMENTATION / TERMINAL-V3-SEALED / NO-SCORING / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-06
> **Intento terminal:** `wave60_frozen_policy_transport_attempt_v3`
> **Config v3:** commit `0cc349c10b91cffe0eb53ae6669fcffbfaf8c756`, auditoría R500 en `b1370608d8fa1a80267f34253be77fa11a19a0d3`
> **Pregunta científica inalterada:** ¿la ley HGB/HGB congelada de Ola 59 transporta a una realización independiente sin refit, recalibración ni selección?

## 1. Observación física

La config v3 y su preflight fueron aceptados por R500. El namespace v3 se
inicializó atómicamente y la preparación primaria comenzó en modo recovery a
partir de `attempt_v2/primary`, sin redraw ni `--force`. La ejecución se detuvo
en `materialize_prepared_bundles()` con `KeyError: 'hard_set_tau'`. El error
ocurrió después de regenerar por copia el draw y ejecutar la inferencia ciega,
pero antes de publicar `preparation_freeze.json`, `preparation_receipt.json` o
una autoridad firmada de preparación.

El preparador conservó el estado parcial bajo
`primary/failed_preparation`. El runner cerró después el par con su semántica
vigente. La autoridad firmada resultante es:

```text
pair terminal       PAIR_ABORTED_PRE_TRUTH
primary terminal    INVALID_PREPARATION
replay terminal     INVALID_PREPARATION
any_truth_accessed  false
recovery_allowed    true
```

Bindings físicos principales:

```text
pair/FAILURE.json                         560ab42f250e90b0b52f54e0db9fe4aced78c2df623d21b717ba56ff51889660
pair/pair_status.json                     45be1a374e62d2030aa10bfd18393a96fb6745b43d1e7c03195f8c6f874b9b1b
pair/failure_inventory.json               5c36bc5da16248c5e29073a44672172e95777835b94ffbb22c0b64b1ee7ac0cb
pair/failure_attestation.json             9470265c78bf066e3abe453dd91dd2a4c0d47115206f69a62f937ff21741cc0e
pair/artifact_manifest.json               5e23d90ec1316686cb5d83d56f20b5dbda8af19127eeb788babadf58365baffe
primary/failure_attestation.json          dd108fce38bebf6cf0161fa355371b1e25e194d313f8affb234ceceb85bccced
replay/failure_attestation.json           17ac5d9da65dbd73d4dbe7d625ef81255d0b00492a01a35a7740b29797133174
failed_preparation/preparation_error.json  7dee40f8c6ffbf9d6d317d4992c90514b2f3248a2274ce7f63a49e9ffa0d1547
```

El inventario sellado contiene 44 registros en primaria y 8 en replay. El hash
del mensaje de error coincide con `str(KeyError("hard_set_tau"))`. El recibo de
inferencia declara `fit_operations=false`,
`oracle_or_labels_available=false` y un probe de verdad denegado con
`PermissionError`.

## 2. Alcance exacto de “pre-truth”

El terminal firmado usa `truth_accessed=false` en el sentido operacional de la
Ola 60: ninguna fase de scoring o evaluación abrió los bundles de verdad y no
se congelaron acciones sobre el lockbox. No hubo refit, recalibración, selección
ni inspección de resultados científicos.

La preparación root-only sí alcanzó la llamada normal a
`compute_oracle_splits()` dentro del materializador antes de consultar la clave
ausente. Esa llamada creó labels temporales de `train` y comenzó a construir el
bundle protegido; el bloque de error retiró el directorio `prepared` completo y
no dejó esos labels como artefactos. Por tanto, no se describirá el incidente
como ausencia absoluta de cómputo de oracle. La distinción durable es:

```text
oracle de preparación root-only: alcanzado transitoriamente
truth científico de scoring/evaluación: no accedido
outputs científicos observados: ninguno
adaptación de modelos o política: ninguna
```

Esta distinción mantiene la comparabilidad del ensayo sin ampliar el significado
del flag firmado más allá de lo que su implementación prueba.

## 3. Causa raíz

La config científica de Ola 60 omite deliberadamente `hard_set_tau`. Ese valor
proviene de la config snapshot de Ola 59, transitivamente autenticada por la
source law v2. El recovery de preparación v2 resolvió esa separación mediante
un `hard_set_contract` cerrado que liga la fuente y fija `hard_set_tau=0.5`; el
preparador inyecta una copia efímera de la config con ese valor únicamente
cuando el amendment tiene schema
`wave60-invalid-preparation-recovery-amendment-v1`.

El amendment estático v3 autenticó correctamente el nuevo guard de identidad,
pero no heredó `hard_set_contract`. La condición del preparador tampoco incluyó
el schema estático. En consecuencia, la config sin `hard_set_tau` llegó al
materializador real.

El hueco escapó a R498–R500 porque:

1. `preparation_preflight()` verifica sources, histórico, amendment, terminal y
   presupuesto, pero no resuelve la config efectiva del materializador;
2. el E2E sintético v2→v3 reemplaza `materialize_prepared_bundles()` por
   `fake_bundles`, de modo que valida lineage, copia y ledger, pero no la interfaz
   real `config["hard_set_tau"]`;
3. el test real del materializador cubre sólo el schema de recovery v2.

La afirmación anterior de que el E2E acreditaba toda la preparación v3 queda,
por ello, restringida a la superficie simulada que efectivamente ejercitó.

## 4. Recovery propuesto: intento v4

No se reabre, elimina ni sobrescribe v3. La recuperación usará un namespace
nuevo:

```text
data/geometria_proporcional/wave60_frozen_policy_transport_attempt_v4
```

El draw se reutilizará desde
`attempt_v3/primary/failed_preparation`. Sus archivos deberán coincidir por bytes
con el mapa ya preservado y copiarse a inodos nuevos. El amendment v4 tendrá un
schema propio y ligará simultáneamente:

- el terminal firmado y los inventarios completos de v3;
- la config v3 y R500;
- el amendment estático v3 y R499;
- el terminal y ledger firmado de v2;
- el draw nested de v3, closed-world y sin `prepared` persistente;
- el recibo de inferencia ciega y el error exacto;
- el `hard_set_contract` ya autenticado por la source law v2;
- el plan presente, su auditoría, la implementación correctiva y su auditoría;
- la partición exacta de fuentes cambiadas e invariantes.

La autoridad `hard_set_tau` no se copiará como número suelto. El validador deberá
recorrer manifest de source law → request → alias único → snapshot físico de Ola
59 → SHA-256 → valor finito exacto `0.5`. Recién entonces construirá una copia
efímera de la config para el materializador. La config canónica de Ola 60 seguirá
sin esa clave.

## 5. Presupuesto acumulativo

V2 dejó una autoridad firmada de `215.36700256168842 s`. V3 falló antes de
publicar su ledger de preparación. `/usr/bin/time -v` observó 36,47 s para la
preparación fallida y 2,29 s para el cierre firmado; ambos registros viven sólo
en el transcript operacional y no se promoverán a medición durable exacta.

El amendment v4 aplicará una única vez un débito conservador no firmado de
60 segundos para v3, siguiendo el régimen ya usado en el recovery v1→v2. El
punto de partida de la primaria v4 será entonces:

```text
215.36700256168842 + 60.0 = 275.3670025616884 s
```

El validador deberá probar que:

- los 215,367 s provienen nuevamente del ledger firmado v2;
- v3 carece de `preparation_receipt` y `preparation_attestation` en ambas raíces;
- el débito de 60 s aparece una sola vez y está ligado al terminal v3;
- replay v4 comienza en el acumulado firmado de la primaria v4;
- el par completo permanece por debajo de 900 s.

No se atribuyen 0 swaps a artefactos que no los registran. Las dos invocaciones
externas de v3 observaron 0 swaps de proceso, pero esa evidencia permanece como
contexto operacional.

## 6. Superficie de implementación

La corrección puede modificar solamente:

```text
experiments/geometria_proporcional/prepare_wave56_fresh.py
tests/test_wave60_frozen_policy_transport.py
```

El preparador incorporará:

1. schema y validador tipado para v4;
2. autenticación del origen nested v3 y de sus dos terminales
   `INVALID_PREPARATION`;
3. continuidad `ledger v2 + debit v3 + primaria v4 + replay v4`;
4. resolución común y previa de la config efectiva del materializador;
5. inyección del `hard_set_tau` autenticado para recovery v2 y v4;
6. fallo antes de materializar draw/inferencia o modificar la raíz inicializada
   si esa autoridad no puede resolverse.

Permanecerán byte-exactos:

```text
src/geometria_proporcional/wave60_frozen_policy_transport.py
experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py
experiments/geometria_proporcional/_wave60_phase_worker.py
```

No cambian modelos, trece estados HGB, 1.300 tree keys, features, thresholds,
policies, scorer, estimandos, bootstrap, penalty, seeds ni acceso de inferencia.

## 7. Pruebas obligatorias

1. reproducción read-only del terminal v3, firmas, inventarios y error exacto;
2. rechazo si v3 contiene `prepared`, labels persistentes, recibos de preparación
   o cualquier archivo fuera del inventario ligado;
3. rechazo si el access receipt deja de declarar inferencia ciega sin labels;
4. rechazo de alteración en el terminal v3, pair failure, config/R500 o lineage;
5. autenticación positiva y ataques del `hard_set_contract`: manifest, request,
   alias, snapshot, SHA y valor;
6. test del seam real: la ruta v4 debe invocar el materializador real con
   `hard_set_tau=0.5`, mientras la config canónica sigue sin esa clave;
7. prueba negativa que demuestre que omitir la resolución previa falla antes de
   materializar draw/inferencia en la raíz inicializada;
8. E2E v3→v4 con mismo draw por bytes, inodos nuevos y ledger acumulativo;
9. rechazo de doble débito, fuente equivocada, redraw, `--force`, inventario
   abierto o delta de sources no autorizado;
10. suite Wave 60 completa y regresión Waves 56–59, CPU-only, con RSS y swaps.

El test crítico no podrá reemplazar el materializador por un fake para acreditar
la interfaz `hard_set_tau`. Los dobles seguirán siendo válidos para tests de
lineage o fallos específicos, pero deberán describirse como tales.

## 8. Cadena de autoridad prevista

```text
este plan
  -> R501 auditoría independiente del plan
  -> implementación mínima
  -> R502 auditoría independiente de implementación
  -> amendment v4
  -> R503 auditoría independiente del amendment
  -> config v4
  -> R504 auditoría independiente de config y preflight de ejecución
  -> preparación y ejecución v4
  -> R505 auditoría independiente de resultados o terminal
```

Cada eslabón documental se incorporará en un commit exclusivo y con parent
directo. Ningún `PASS` promueve por sí mismo la arquitectura ni decide
`GO/NO-GO`.

## 9. Criterio de continuidad

Sólo se implementará si R501 confirma que el terminal v3 permite recovery, que
el débito conservador no duplica tiempo y que la autoridad `hard_set_tau` queda
cerrada antes de escribir output. Si el v4 abre truth de scoring/evaluación y
luego falla, no habrá otra corrida. Si falla nuevamente antes de esa barrera, se
conservará el terminal y se reevaluará sin ampliar automáticamente el número de
intentos. Si completa, su resultado seguirá representando el único draw
originado en v1, no una realización independiente adicional.
