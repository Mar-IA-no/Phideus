# R483 — Reauditoría independiente de implementación de recuperación `INVALID_PREPARATION`

**Dictamen técnico: `REVISE` — 0 HIGH / 1 MEDIUM / 0 LOW.**

El commit sucesor cierra el finding central de R481: existe ahora una prueba que atraviesa `run_preparation_transaction()`, llega al ramal productivo que autentica `hard_set_tau=0.5` y ejecuta el materializador real, y verifica la salida firmada. No encontré un defecto funcional en ese recorrido ni en el nuevo cierre físico `closed-world`. El `REVISE` se debe a que la matriz negativa y de ledger implementada es sólo parcial frente a los casos independientes exigidos por el plan R481; por tanto, la implementación todavía no satisface por completo su contrato de evidencia.

## Identidad y alcance

El target auditado es exactamente:

- commit: `5aee5fb6064cdd232620a5eebe9a9e856dae9f0f`;
- parent directo R482: `f844526bc42ee8ae366adee64bc63a9f780d8093`;
- árbol Git: `362baef0316b41fbddb9270c9597d23a68692b79`;
- diff exclusivo: `1033` inserciones y `55` eliminaciones en:
  - `experiments/geometria_proporcional/prepare_wave56_fresh.py`;
  - `tests/test_wave60_frozen_policy_transport.py`.

Leí completos `WAVE_60_INVALID_PREPARATION_RECOVERY_R481_RESOLUTION_PLAN.md`, el informe R481 y la auditoría R482, además del diff íntegro de ambos paths. El plan y R482 coinciden entre filesystem y sus blobs Git:

- plan: `471fc998e21e357c262b22e9edfe27c3c795fb0a1fdb522e1744e7b350297823`;
- R482: `170d9fff63de32a2ef3030ddec2704f92fbb9592dd952837f6804366cbabd1f6`.

Los blobs modificados coinciden físicamente con el target:

| Path | SHA-256 parent | SHA-256 target/físico |
|---|---|---|
| `prepare_wave56_fresh.py` | `05571d22f2f406b07e89132482cb39715128e4b9c3a5abda1c85aac0b7143dcb` | `fd3a3809fc98c825cf1c3159b6af8602db50c59a529d7a32b12f2148f687cfb5` |
| `test_wave60_frozen_policy_transport.py` | `369f1970c19b9fab2f80fd744bebadc6d0822d23b2593ab70d7f26f3e9dc5b17` | `d8e5c78554f5eb5ae1be5472f42ba15761e7034f305591a8f2c7931f63f7ce62` |

Los tres sources científicos continúan byte-exactos respecto de R475 `9f1a229d9c0ccb5e46b921e6c92281becc317139` y del filesystem actual:

- módulo: `46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65`;
- runner: `1c778c3e60c1bbcebeb5c83430601a7c0b148e447528195f1dec4296322825aa`;
- worker: `c6c5c832633d9ed8a99079dd4fc4f5677073f7e1883a204fa6bb41f6d646bac7`.

## Cierre efectivo del finding R481

La nueva positiva ya no separa artificialmente validador y materializador. `test_invalid_preparation_transaction_wires_tau_and_signed_provenance()` prepara el recovery context, conserva la config sin `hard_set_tau`, sustituye únicamente `stage_and_infer()` por una copia de los logits pre-truth inventariados y envuelve el materializador sólo para observar su argumento, delegando inmediatamente en la función real (`tests/test_wave60_frozen_policy_transport.py:1897-2101`).

La llamada atraviesa `run_preparation_transaction()` y `execute_preparation()` reales (`prepare_wave56_fresh.py:6068-6463`). En el ramal bajo prueba:

1. se revalida el origen antes de extraer las claves;
2. se republican escrow y freeze con hashes exactos;
3. se copia el benchmark preservado sin llamar al generador;
4. tras la única sustitución permitida de inferencia, se llama al hard validator real;
5. se crea `deepcopy(config)`, se inyecta allí el `tau` autenticado y se llama al materializador real (`prepare_wave56_fresh.py:6241-6261`).

La positiva observa una sola invocación con `hard_set_tau=0.5`, prueba que el objeto y el archivo de config siguen sin esa clave, compara bytes e inodes de escrow, freeze y todos los archivos del benchmark, verifica ausencia de oracle/labels, y confirma que el origen no cambió (`tests/test_wave60_frozen_policy_transport.py:2121-2164`). Si se elimina el ramal productivo, la aserción de una invocación y los bundles dejan de satisfacerse. El defecto probatorio específico de R481 queda, por tanto, cerrado.

## Provenance, attestation y frontera pre-truth

`recovery_provenance()` conserva el objeto exacto de extensiones —`INVALID_PREPARATION`, `0.5` y `60.0`— (`prepare_wave56_fresh.py:5736-5764`). La transacción lo publica en `generation_receipt.json`, `preparation_freeze.json` y `preparation_receipt.json`; la prueba exige igualdad de los tres y del objeto esperado (`tests/test_wave60_frozen_policy_transport.py:2125-2132`).

La finalización añade el presupuesto al receipt antes de publicar la attestation (`prepare_wave56_fresh.py:6677-6704`). La prueba valida la firma Ed25519 y comprueba que el record firmado de `preparation_receipt.json` contiene su SHA-256 físico (`tests/test_wave60_frozen_policy_transport.py:2142-2154`). Así la provenance queda transitivamente firmada. No aparece oracle, label autorizado ni redraw, y el draw original permanece inventarialmente idéntico.

## Lineage R481→R483 y separación R475/recuperación

`validate_wave60_invalid_preparation_implementation_suffix()` autentica la implementación rechazada, R481 `REVISE 0/1/0`, la resolución, R482 PASS, la implementación sucesora exclusiva y la autoridad futura R483 (`prepare_wave56_fresh.py:510-679`). El validador de amendment incorpora las cuatro autoridades históricas nuevas, conserva el keyset de 23 claves, y renumera amendment/config a R484/R485 (`prepare_wave56_fresh.py:4479-4905`). Los deltas de preparer/test siguen calculándose desde R475, mientras módulo/runner/worker permanecen bajo R475.

La fixture Git sintética construye el sufijo completo y rechaza R481 presentado como PASS, un cruce de hash y cada parent saltado en resolución, R482, implementación final y R483 (`tests/test_wave60_frozen_policy_transport.py:4250-4505`). La lógica productiva es estricta y no observé circularidad ni cruce efectivo de autoridades. La insuficiencia residual de negativos específicos se incluye en el único finding siguiente.

## Closed world físico y débito one-shot

El nuevo cierre material exige que el draw contenga exactamente los archivos derivados del manifest más `generation_receipt.json`, `preparation_error.json`, `inference/access_receipt.json` y los nueve logits seed×split, y exactamente sus directorios padres (`prepare_wave56_fresh.py:4412-4449`). Esto se suma al inventario firmado completo del primary, la ausencia de hardlinks, los modos/owners sensibles, el mapa preservado cerrado y las firmas del par/origen (`prepare_wave56_fresh.py:4295-4476`). No encontré una vía para autorizar un archivo auxiliar adicional mediante la propia amendment.

El débito unsigned continúa limitado al container v1, config v2, amendment canónica, objeto exacto `60.0/applied_once=true` y ausencia de cualquier receipt o attestation de preparación en primary o replay (`prepare_wave56_fresh.py:6559-6619`). Una vez existe autoridad firmada, `wave60_prior_preparation_elapsed()` usa el ledger firmado y no vuelve a entrar al débito unsigned (`prepare_wave56_fresh.py:6622-6674`). La positiva integrada firma `prior=60`, `cumulative=61` y comprueba que el prior del replay es `61` (`tests/test_wave60_frozen_policy_transport.py:2102-2161`).

## Finding MEDIUM — matriz negativa contractual incompleta

El plan ordena negativos independientes para cada familia física y de ledger, no sólo una muestra representativa (`WAVE_60_INVALID_PREPARATION_RECOVERY_R481_RESOLUTION_PLAN.md:191-212`). La implementación no cubre toda esa matriz:

- la parametrización física prueba `escrow_bytes`, `draw_mode`, `sensitive_owner`, symlink, hardlink, nodo especial, archivo adicional, firma primary, firma del par y mapa preservado extra (`tests/test_wave60_frozen_policy_transport.py:2397-2469`), pero no altera por separado bytes de freeze, manifest y un archivo de benchmark; modo de un archivo sensible; owner del draw; ni firma de replay;
- la parametrización del débito prueba los tres valores inválidos de seconds, `applied_once`, régimen, wall, autoridad externa, versión, source y presencia de `preparation_receipt.json` en primary/replay (`tests/test_wave60_frozen_policy_transport.py:2472-2528`), pero no prueba attestation-only en ninguno de los roles ni otro `prior_attempt_container`;
- la positiva une el débito inicial con el prior del replay, pero no ejecuta/firma ese replay ni una recuperación posterior desde el par firmado; por eso no demuestra sobre el schema `INVALID_PREPARATION` que los 60 segundos no se suman de nuevo después del par completo. El test legacy v2→v3 cubre la mecánica general de ledgers firmados, pero no parte de este débito unsigned (`tests/test_wave60_frozen_policy_transport.py:3049-3983`);
- la fixture de sufijo no contiene los rechazos específicos prometidos para findings R481 distintos, path adicional en cada commit exclusivo, scope/target/audit-id/hash físico/blob Git alterados y cruce de un source científico R475. Los helpers genéricos tienen cobertura previa, pero el nuevo ensamblaje del sufijo no prueba esas entradas (`tests/test_wave60_frozen_policy_transport.py:4250-4505`).

### Reproducción

La ausencia se reproduce sin alterar el repositorio inspeccionando los parámetros y aserciones de las tres pruebas nuevas:

```text
nl -ba tests/test_wave60_frozen_policy_transport.py | sed -n \
  '2397,2528p;4250,4505p'
```

Esos rangos enumeran exhaustivamente sus mutations/cases y no contienen los casos anteriores. La suite pasa porque no intenta esas variantes; esto no implica que las defensas productivas hayan fallado, sino que los rechazos independientes exigidos como evidencia no fueron implementados.

### Corrección requerida

No hace falta modificar producción salvo que aparezca una falla al completar los tests. Se debe:

1. ampliar la matriz física con las variantes omitidas de bytes, metadata y firma replay;
2. añadir attestation-only primary/replay y container drift al ledger;
3. completar una secuencia positiva `60 → primary cumulative → replay cumulative → recuperación desde par firmado`, afirmando que `60` no vuelve a debitarse;
4. completar la matriz específica del sufijo Git con los campos y commits exclusivos omitidos.

## Pruebas y entorno

Se compiló en memoria cada blob Python modificado y `git diff --check f844526 5aee5fb` quedó limpio. La suite focal completa se ejecutó con CUDA invisible y basetemp propio sobre `/mnt/m2-1TB`:

```text
CUDA_VISIBLE_DEVICES='' venv/bin/pytest -q \
  tests/test_wave60_frozen_policy_transport.py \
  --disable-warnings \
  --basetemp=/mnt/m2-1TB/.wave60-r483-audit.d9Y7uH/pytest

131 passed in 247.25s
```

El RSS observado del proceso fue `830132 KiB`; había más de `21 GiB` de RAM disponible. El temporal propio ocupó `2.7 GiB` y fue eliminado por su path exacto al terminar. No se usó ni consultó GPU. No se modificó código, config ni otro documento; este informe es el único archivo creado y no forma parte del target auditado.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R483",
  "scope": "INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
  "target": {
    "implementation_commit": "5aee5fb6064cdd232620a5eebe9a9e856dae9f0f"
  },
  "technical_verdict": "REVISE",
  "findings": {
    "high": 0,
    "medium": 1,
    "low": 0
  },
  "files_modified": false,
  "gpu_used_or_queried": false
}
```
