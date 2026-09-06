# Ola 60 — corrección y recuperación del guard de identidad del protocolo estático

> **Estado:** `REVISED-AFTER-R496 / PRE-IMPLEMENTATION / PRE-TRUTH / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-06
> **Intento preparado:** `wave60_frozen_policy_transport_attempt_v2`
> **Config auditada:** commit `ef4a620ae79f5eb4502eea9cca1547da86216f93`, auditoría R495 en commit `da9a9bf1f7c03ef2e062d874dd9f9f1a6f8d67bd`
> **Auditoría inicial del plan:** R496 `REVISE / 0 HIGH + 0 MEDIUM + 1 LOW`, commit `eb674e000f6a94392214fdcd26d1635e420baa78`
> **Pregunta científica inalterada:** ¿la ley HGB/HGB congelada de Ola 59 transporta a una realización independiente sin refit, recalibración ni selección?

## 1. Hecho observado antes de truth

El intento v2 fue inicializado y sus dos raíces se prepararon con la config y la
amendment auditadas. La primaria reutilizó el draw preservado del intento v1 y
el replay reutilizó la primaria v2. Ambas raíces pasan por separado
`validate_prepared_root()`, declaran `truth_accessed=false` y
`fit_operations=false`, y conservan paquetes iguales por bytes con inodos
distintos.

El ledger firmado de preparación es:

| tramo | prior (s) | duración (s) | acumulado (s) | RSS máximo | swap del proceso |
|---|---:|---:|---:|---:|---:|
| débito conservador v1 | — | 60,000000 | 60,000000 | no atribuido | no atribuido |
| primaria v2 | 60,000000 | 77,441179 | 137,441179 | 951.365.632 B | no preservado |
| replay v2 | 137,441179 | 77,925824 | 215,367003 | 956.383.232 B | no preservado |

Antes de invocar `execute-prepared-pair`, una llamada read-only al mismo guard
produjo `INVALID_NEW_DRAW_IDENTITY`. La descomposición exhaustiva mostró:

- los cinco commitment groups requeridos son iguales entre primaria y replay;
- los dominios físicos contienen exactamente 22 entradas en cada raíz;
- todas las entradas primaria/replay son byte-exactas y ninguna comparte
  `(st_dev, st_ino)`;
- frente a cada una de las cinco raíces antecedente de Ola 59, la única colisión
  es `bytes:benchmark/protocol_config.json`;
- no hay colisión de escrow, claves, semantic root, commitment map, bundles,
  visible/sealed data ni inodos.

No se abrió truth, no se ejecutó scoring y no se materializó una inferencia
científica. Según el transcript operacional no archivado, el primer comando de
preparación usó por error la raíz agregada
`wave55_policy_bridge_results_v1` y falló dentro del preflight antes de escribir
el draw; fue repetido con la raíz física correcta
`wave55_policy_bridge_fresh_v1`, cuyo `decision_select.npz` coincide con el SHA
congelado. Esta observación se conserva sólo como contexto operacional y no
integra la autoridad de recovery ni el ledger.

## 2. Diagnóstico

`opaque_draw_fingerprint()` valida todos los miembros declarados por el
manifest y los incorpora al mapa físico. `validate_new_draw_pair()` compara
después cada miembro contra los antecedentes y exige que ninguno tenga bytes
iguales. La regla es incorrecta para
`benchmark/protocol_config.json`: Wave 60 exige explícitamente el mismo
protocolo `wave49-relational-benchmark-v2` y liga su SHA-256
`c45a7fb245950521ceac4c6de75b51e746152f506522c52697d05bdc30673468`
como fuente upstream. Por construcción, ese archivo debe coincidir con Ola 59.

El plan original exige novedad para escrow, tres key commitments, semantic
root, mapa de commitments y bundles; no exige cambiar el protocolo compartido.
El guard implementó una condición más fuerte e incompatible con el propio
contrato. El test sintético no lo detectó porque `_fake_draw()` no incluía
`protocol_config.json` en el manifest.

## 3. Corrección mínima del guard

La implementación deberá introducir una allowlist cerrada de archivos estáticos
compartidos con una única entrada:

```text
benchmark/protocol_config.json
```

El archivo continuará sujeto a todas estas comprobaciones:

1. regular file, path canónico, bytes y tamaño consistentes con el manifest;
2. SHA idéntico entre primaria y replay;
3. inodos distintos entre primaria y replay;
4. ningún hardlink/inodo compartido con un antecedente;
5. validación previa de cada raíz contra la config y el SHA upstream congelado.

La única relajación será no interpretar la igualdad **de bytes** de ese archivo
con un antecedente como identidad del draw. Todo otro archivo y commitment
seguirá exigiendo novedad. No se cambiarán scorer, worker, features, modelos,
thresholds, policies, estimandos, bootstrap, seeds, penalty ni acceso a truth.

## 4. Terminal v2 y recuperación v3

Después de la reauditoría independiente focal R497, el runner
congelado de v2 se invocará una vez para que publique el terminal que su propia
semántica determina. El estado esperado es:

```text
primary = INVALID_NEW_DRAW_IDENTITY
replay  = PEER_ABORTED_PRE_TRUTH
pair    = PAIR_ABORTED_PRE_TRUTH
any_truth_accessed = false
recovery_allowed = true
```

El resultado físico, no esta expectativa, será la autoridad. Si aparece otro
terminal o truth pasa a `true`, no se implementará una nueva corrida.

La corrección usará un namespace nuevo
`wave60_frozen_policy_transport_attempt_v3`. Su amendment tendrá schema propio
para este recovery y ligará:

- terminal y manifest firmados del intento v2;
- config v2 y auditoría R495;
- este plan y su auditoría;
- implementación y auditoría de la corrección;
- inventario closed-world del draw preservado;
- ledger firmado acumulado de primaria y replay v2;
- partición exacta de fuentes cambiadas e invariantes.

La preparación v3 reutilizará el escrow de v2 sin redibujar y comenzará desde el
tiempo durable heredado. Primaria y replay v3 volverán a copiar el mismo draw
por bytes con inodos nuevos. No se usa `--force` ni se altera el intento v2.

## 5. Superficie de implementación

La corrección puede modificar solamente:

```text
experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py
experiments/geometria_proporcional/prepare_wave56_fresh.py
tests/test_wave60_frozen_policy_transport.py
```

El runner cambia únicamente en la clasificación de la colisión estática. El
preparador cambia sólo para autenticar el schema de recovery v3, su lineage y
el delta de fuentes. El test materializa la prueba positiva y los ataques. Se
mantienen byte-exactos:

```text
src/geometria_proporcional/wave60_frozen_policy_transport.py
experiments/geometria_proporcional/_wave60_phase_worker.py
```

La source law HGB, sus trece modelos, 1.300 tree keys, thresholds y scorer
permanecen sin cambios. La config v3 sólo podrá diferir de v2 en namespace,
autoridad de recovery, auditoría final, self-binding y hashes de las tres
fuentes autorizadas.

## 6. Pruebas obligatorias

1. caso positivo con protocolo byte-idéntico al antecedente y resto del draw
   nuevo;
2. rechazo si cualquier archivo no estático coincide por bytes con un
   antecedente;
3. rechazo si el protocolo comparte inodo/hardlink con antecedente o replay;
4. rechazo si protocolo y manifest discrepan o si primaria/replay no coinciden;
5. reproducción read-only del falso positivo v2 y aceptación del mismo par bajo
   el guard corregido, antes de materializar v3;
6. validación del terminal firmado v2 y de su ausencia de truth;
7. prueba positiva de recovery v2→v3 con ledger continuo y mismo draw por bytes,
   sin inodos compartidos;
8. negativos de amendment, config, lineage, inventario, firmas, source delta,
   doble débito y presupuesto;
9. suite Wave 60 completa y regresión explícita Waves 56–59, CPU-only, con RSS
   y swaps registrados.

## 7. Cadena de autoridad

La numeración prevista es:

```text
este plan
  -> R496 auditoría inicial del plan: REVISE por trazabilidad documental
  -> esta revisión documental
  -> R497 reauditoría focal del plan
  -> terminal firmado v2 con el runner congelado
  -> implementación mínima
  -> R498 auditoría independiente de implementación
  -> amendment v3
  -> R499 auditoría independiente de amendment
  -> config v3
  -> R500 auditoría independiente de config y preflight
  -> preparación y ejecución v3
  -> R501 auditoría independiente de resultados
```

Cada artefacto documental se incorporará mediante un commit exclusivo y parent
directo. Una auditoría `PASS` autoriza sólo el eslabón siguiente; no promueve la
arquitectura ni decide `GO/NO-GO`.

## 8. Criterio de continuidad

Si v2 no sella un terminal pre-truth recuperable, se detiene esta vía. Si v3
falla antes de truth, se conserva el terminal y se reevalúa sólo dentro del
presupuesto restante. Si cualquier raíz v3 accede a truth y luego falla, no hay
nueva corrida. Si completa, el resultado se interpreta como transporte sobre
el único draw originado en v1 y no como una realización adicional.
