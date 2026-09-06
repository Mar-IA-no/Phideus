# R481 — Auditoría independiente de implementación de la recuperación `INVALID_PREPARATION`

**Dictamen técnico: `REVISE` — 0 HIGH / 1 MEDIUM / 0 LOW.**

La implementación productiva inspeccionada es coherente con el diseño aprobado y no encontré un defecto funcional concreto en sus fronteras de seguridad, autoridad o compatibilidad legacy. El `REVISE` responde a una brecha de prueba integrada: la prueba positiva añadida valida la cadena dura y luego invoca directamente el materializador, por lo que no recorre el cableado real de `execute_preparation()` que debía corregir el `KeyError: 'hard_set_tau'` original. En consecuencia, el código parece correcto por inspección y la suite focal pasa, pero la evidencia de prueba todavía no autoriza cerrar la recuperación.

## Identidad, alcance y contraste Git/físico

El target auditado es exactamente:

- commit de implementación: `e617e15be290a62e5b0027c3748f1f5e85abd083`;
- parent directo R480: `abb8fd2e8c9119e46fabee2aa15405ceb146d4b2`;
- árbol Git: `3ff5cc84a16e88dccae22d9df83a9d4303a21546`;
- diff exclusivo: `1836` inserciones y `14` eliminaciones en dos paths:
  - `experiments/geometria_proporcional/prepare_wave56_fresh.py`;
  - `tests/test_wave60_frozen_policy_transport.py`.

Se leyeron completos los tres planes —base, resolución R478 y resolución R479—, las auditorías R478, R479 y R480, y el diff íntegro de ambos paths. El informe R480 físico coincide con su blob Git, SHA-256 `d4860501a98acfe71659f49b0fbeb89c1c8fec7d3fcdd1fa4444664c7e6aaad3`.

Los dos archivos modificados también coinciden físicamente con el target Git:

| Path | SHA-256 R480 | SHA-256 R481 físico/Git |
|---|---|---|
| `prepare_wave56_fresh.py` | `7d7ead44f6d0e64802dafa585a59a20ae78f43f5e975e03c60e6bd8a1de33d66` | `05571d22f2f406b07e89132482cb39715128e4b9c3a5abda1c85aac0b7143dcb` |
| `test_wave60_frozen_policy_transport.py` | `328c934c63f2cb402633b72b52699e2d48433ff7e94966169a5b1428e6f63519` | `369f1970c19b9fab2f80fd744bebadc6d0822d23b2593ab70d7f26f3e9dc5b17` |

Los tres blobs científicos ligados a R475 permanecen byte-exactos, tanto físicamente como en Git:

- módulo: `46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65`;
- runner: `1c778c3e60c1bbcebeb5c83430601a7c0b148e447528195f1dec4296322825aa`;
- worker: `c6c5c832633d9ed8a99079dd4fc4f5677073f7e1883a204fa6bb41f6d646bac`.

## Resultado de la auditoría técnica

### Seguridad, closed world y ausencia de nuevo truth/redraw

La recuperación conserva el límite pre-truth. `_validate_wave60_invalid_preparation_recovery_origin()` autentica el paquete firmado del par, exige `PAIR_ABORTED_PRE_TRUTH`, ambos terminales `INVALID_PREPARATION`, `truth_accessed=false` y `recovery_allowed=true`; después compara el inventario físico completo de `primary`, rechaza hardlinks y comprueba owner, group y modos del draw anidado (`prepare_wave56_fresh.py:4121-4272`). El conjunto permitido del draw se deriva de los tres archivos de raíz y del manifest del benchmark, y se compara como closed world antes de reutilizarlo.

El flujo productivo toma `recovery_context["reuse_source"]`, que para este schema es el draw `primary/failed_preparation`; copia escrow y benchmark desde allí y no llama al generador (`prepare_wave56_fresh.py:5900-5914`). La validación dura ocurre antes del materializador y no hay acceso a oracle/truth introducido por este cambio (`prepare_wave56_fresh.py:5993-6007`). No observé una vía de redraw ni una ampliación silenciosa de autoridad.

### Cadena dura `manifest → request → alias → snapshot → tau`

`validate_wave60_invalid_preparation_hard_set_contract()` aplica el objeto exacto de nueve claves y recorre los eslabones en orden (`prepare_wave56_fresh.py:728-826`):

1. resuelve canónicamente el authority root y exige el manifest ligado por la config;
2. llama al `validate_source_authority()` real del runner R475;
3. compara el registro físico del request —bytes, owner, group, modo y hash— con el manifest;
4. liga de forma unívoca el alias del request con path y SHA-256 del snapshot;
5. rechaza traversal, symlink o archivo no regular;
6. verifica el SHA-256 físico del snapshot y extrae un `hard_set_tau` numérico, finito, no booleano e igual a `0.5`.

En el ramal productivo, el valor se inserta sólo en un `deepcopy(config)` efímero antes de llamar al materializador (`prepare_wave56_fresh.py:5993-6007`); no se muta la config canónica y no existe default alternativo.

### Lineage, autoridad R475/R481 y compatibilidad legacy

El nuevo schema tiene validador separado y keysets exactos. `_validate_wave60_invalid_preparation_recovery_amendment()` liga de manera directa y exclusiva la historia documental R477→R478→R479→R480, la implementación futura R481, la amendment descendiente y la auditoría futura R482; además particiona los dos sources modificados bajo R481 y los tres sources científicos inmutables bajo R475 (`prepare_wave56_fresh.py:4275-4762`). La validación final de config conserva esa partición y exige el informe canónico R481 con scope, target, verdict y conteos exactos.

La ampliación está despachada sólo para `wave60-invalid-preparation-recovery-amendment-v1`; los schemas anteriores mantienen sus validadores y reglas previas (`prepare_wave56_fresh.py:4763-5107`). `_validate_wave60_contract_delta()` permite exactamente los cambios del preparador y del test para el schema nuevo, sin relajar la regla legacy. No encontré cruce de autoridad R475↔R481 ni regresión de los schemas existentes.

### Escrow anidado, provenance y débito unsigned one-shot

El origen anidado queda autenticado por inventario físico, ausencia de hardlinks, metadata root-only, hashes preservados, contrato del escrow y manifest cerrado (`prepare_wave56_fresh.py:4121-4272`). La procedencia se extiende con `recovery_kind=INVALID_PREPARATION`, `hard_set_tau=0.5` y `unledgered_preparation_debit_seconds=60.0` (`prepare_wave56_fresh.py:5487-5515`) y el flujo la inserta en los receipts/freezes de recuperación.

`_wave60_invalid_preparation_unsigned_debit()` acepta exclusivamente la config v2, la amendment canónica ligada por path y hash, el objeto exacto de débito con `applied_once=true`, y el source primario previamente autenticado (`prepare_wave56_fresh.py:6310-6370`). Rechaza mezclar el débito con cualquier receipt o attestation de preparación firmado en `primary` o `replay`. Los caminos firmados heredados continúan usando su ledger previo. No encontré una segunda aplicación silenciosa ni una ruta que transforme el registro externo unsigned en autoridad firmada.

## Finding MEDIUM — la prueba positiva evita el cableado productivo que debe demostrar

El plan base identifica como causa del incidente que el lifecycle había sustituido el materializador y exige una preparación positiva con el materializador real (`WAVE_60_INVALID_PREPARATION_RECOVERY_PLAN.md:20-29,55-59,160-163`). La resolución R478 exige una preparación Wave 60 real con extensión trazada y una cadena completa hasta R481/R482 (`WAVE_60_INVALID_PREPARATION_RECOVERY_R478_RESOLUTION_PLAN.md:255-277,310-328,349-377`). R479 vuelve a exigir validación antes de construir `materializer_config` y una prueba positiva con el materializador real sin mockear sus fronteras (`WAVE_60_INVALID_PREPARATION_RECOVERY_R479_RESOLUTION_PLAN.md:72-96,101-116`).

Sin embargo, `test_invalid_preparation_hard_set_real_chain_and_materializer()` llama primero al validador duro y después importa e invoca directamente `materialize_prepared_bundles()`, construyendo manualmente `{**config, "hard_set_tau": tau}` (`tests/test_wave60_frozen_policy_transport.py:1784-1829`). No llama a `execute_preparation()` ni a la transacción. Por ello, ese test seguiría pasando si el ramal real de `prepare_wave56_fresh.py:5993-6007` se eliminara, no recibiera el `recovery_context`, mutara la config equivocada o entregara otra config al materializador. Esa es precisamente la clase de desconexión que originó el fallo recuperado.

La misma brecha deja sin ejercicio integrado otras obligaciones del nuevo schema:

- ningún test nuevo invoca `validate_recovery_amendment()` con una amendment completa `wave60-invalid-preparation-recovery-amendment-v1`; la prueba de lineage termina en R480 y comprueba sólo un salto rechazado (`tests/test_wave60_frozen_policy_transport.py:1832-1956`);
- la prueba de escrow/débito construye una amendment parcial y llama a helpers estrechos, no al validador completo ni al flujo de preparación (`tests/test_wave60_frozen_policy_transport.py:1959-2045`);
- la prueba de partición R475/R481 construye una amendment mínima para `validate_wave60_final_config_authority()`, pero no prueba end-to-end la cadena R481→amendment→R482 (`tests/test_wave60_frozen_policy_transport.py:3781-3905`);
- no se verifica en una salida real que la copia del escrow/benchmark conserve bytes pero cree inodes nuevos, ni que `generation_receipt`, `preparation_freeze`, `preparation_receipt` y la attestation repitan exactamente la provenance ampliada;
- faltan negativos integrados para cada parent saltado; deriva de bytes/modo/owner/symlink/hardlink/firma/inventario del origen anidado; y débito cero, negativo, distinto, `applied_once=false`, segunda aplicación o mezcla parcial con autoridad firmada.

### Corrección requerida

No corresponde cambiar el código productivo salvo que las pruebas nuevas revelen un defecto. Hay que completar la suite con una fixture sintética de la cadena futura y, como mínimo:

1. ejecutar `validate_recovery_amendment()` sobre una amendment completa con ancestry directo R481→amendment→R482 y negativos independientes de parents saltados y cruces R475/R481;
2. recorrer `execute_preparation()` —o la transacción productiva equivalente— con el recovery context real, sin sustituir el validador duro ni el materializador, y demostrar que `tau=0.5` llega desde la cadena autenticada sin mutar la config canónica;
3. comprobar bytes e inodes del escrow/benchmark copiados y la provenance exacta en todos los artefactos de preparación;
4. añadir la matriz negativa del origen anidado y del débito one-shot enumerada arriba.

## Pruebas y entorno

La suite focal se ejecutó de forma aislada y CPU-only:

```text
CUDA_VISIBLE_DEVICES='' venv/bin/pytest -q \
  tests/test_wave60_frozen_policy_transport.py \
  --disable-warnings --basetemp=/mnt/m2-1TB/.wave60-r481-audit.Jqyvgt

108 passed in 173.07s
```

Una corrida amplia previa quedó invalidada por `ENOSPC` en `/tmp`; se abortaron únicamente los procesos y temporales propios, se recuperó el espacio y ese resultado no se utilizó como evidencia sobre el código. No se usó ni consultó GPU. No se modificó código, configuración ni artefactos: el único archivo creado por esta auditoría es este informe, que no forma parte del target evaluado.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R481",
  "scope": "INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
  "target": {
    "implementation_commit": "e617e15be290a62e5b0027c3748f1f5e85abd083"
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
