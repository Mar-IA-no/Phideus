# R491 — Auditoría independiente de aceptación de la implementación `INVALID_PREPARATION`

**Dictamen técnico: `REVISE` — 0 HIGH / 1 MEDIUM / 0 LOW.**

El sucesor cierra el finding probatorio de R489: materializa once auditorías y
55 negativos semánticos, y cada caso afirma explícitamente
`blob Git == archivo físico == binding` antes de esperar el rechazo. También
amplía correctamente el lineage, las cinco implementaciones, las 22 etapas y
el keyset de 39 claves; conserva R489 como `REVISE`, reserva R491 para la
aceptación futura y mantiene la frontera científica byte-exacta. Sin embargo,
el guard productivo de la auditoría de amendment conserva el ID anterior R490
en vez del R492 exigido por el plan. La futura amendment canónica quedaría
rechazada pese a una auditoría R492 válida, por lo que la implementación no
puede recibir autoridad PASS.

## Identidad y alcance

El target auditado es exactamente:

- commit: `4e50425bc8c33ed668da6de21dcd2a22882268fa`;
- parent directo R490: `9503a697efe6c9c766fcae6fee4b4d7982ba6221`;
- árbol Git: `c4ce5ba853b9a207653310fdf205a698fe5d75c3`;
- diff exclusivo: `469` inserciones y `46` eliminaciones en:
  - `experiments/geometria_proporcional/prepare_wave56_fresh.py`;
  - `tests/test_wave60_frozen_policy_transport.py`.

El parent introduce exclusivamente
`490_wave60_invalid_preparation_recovery_r489_resolution_plan_audit.md`, es hijo
directo del plan R489 y su blob/archivo coincide en
`34a90af6491ddd7d479b66f02e5c5645c05a72dc3718a339ade191beafe815f0`.
El plan R489 fue introducido exclusivamente por
`d04806bb88482637635feb3a41337fdfdaa637b7`, con SHA-256 físico/Git
`8743082b69c6988b8587afbc1368fd7f4c5efa14179a30be16f4a1e81304de61`.

Leí completos el plan aprobado R489, R490, el diff y ambos archivos target
vigentes, y contrasté las defensas productivas alcanzadas. Los hashes físicos
coinciden con los blobs del target:

| Path | SHA-256 target/físico |
|---|---|
| `prepare_wave56_fresh.py` | `83525c3edd0e5b77584d1341a9b65d0d4e45996ad3b9a831f7fc09274df109b4` |
| `test_wave60_frozen_policy_transport.py` | `11d05738e96537a0774a0a90514a615cb3c426e06c2d341fbc6585ae759de11a` |

`git diff --check` quedó limpio. No se modificó código, config, amendment ni
otro artefacto durante esta auditoría.

## Lineage, matrices y frontera científica

La positiva sintética contiene once auditorías consecutivas R481–R491. R481,
R483, R485, R486, R487 y R489 conservan sus verdicts `REVISE`; R482, R484,
R488, R490 y la autoridad futura R491 son `PASS`. El preparador autentica R489
como historia rechazada, R490 como auditoría del plan y sólo acepta R491 como
`recovery_implementation` final
(`prepare_wave56_fresh.py:510-1084`).

`audit_specs` contiene exactamente once entradas. El loop ejecuta los cinco
campos `audit_id`, `target`, `scope`, `technical_verdict` y `findings` para cada
una, cuenta exactamente 55 casos y, antes del `pytest.raises`, afirma:

```text
blob_sha256 == physical_sha256 == case[binding][sha_field]
```

Además comprueba diferencia semántica única, path exclusivo y parent directo
(`tests/test_wave60_frozen_policy_transport.py:5450-5533`). El finding de R489
queda, por tanto, cerrado en el lugar exacto requerido.

Las cinco implementaciones —cuatro rechazadas y la final— reciben negativos
independientes de `old_sha256`, `new_sha256`, path y todos los cruces de hashes
para preparer y test. Las tres ramas científicas reconstruyen las 22 etapas del
sufijo y alteran materialmente, una por vez, módulo, runner y worker. Las
matrices de parent saltado y path extra usan también 22 etapas y comprueban los
deltas exactos antes del rechazo (`tests/test_wave60_frozen_policy_transport.py:5195-5228,5968-6295`).

El keyset productivo de amendment contiene las 35 claves anteriores más:

```text
r487_resolution_implementation
r487_resolution_implementation_audit
r489_resolution_plan
r489_resolution_plan_audit
```

para un total exacto de 39. Sus bindings preservan `fb0248f`/R489 como
implementación rechazada, ligan el plan R489 y R490, y reservan
`recovery_implementation` al sucesor auditado por R491
(`prepare_wave56_fresh.py:4882-5569`).

Los sources científicos permanecen byte-exactos respecto de R475
`9f1a229d9c0ccb5e46b921e6c92281becc317139`, del target y del filesystem:

- módulo: `46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65`;
- runner: `1c778c3e60c1bbcebeb5c83430601a7c0b148e447528195f1dec4296322825aa`;
- worker: `c6c5c832633d9ed8a99079dd4fc4f5677073f7e1883a204fa6bb41f6d646bac7`.

La positiva de transaction/provenance usa R491, no R489, exige igualdad del
objeto completo en generation receipt, preparation freeze y preparation
receipt, verifica la firma Ed25519 y liga el receipt por path, bytes y SHA-256
físicos exactos (`tests/test_wave60_frozen_policy_transport.py:2034-2040,2132-2171`).
El schema firmado no se amplía.

## Finding MEDIUM — el validador de amendment exige R490 en lugar de R492

El lineage aprobado es inequívoco:

```text
R489 REVISE -> plan R489 -> R490 PASS -> implementación -> R491 PASS
-> amendment -> R492 PASS -> config v2 -> R493 PASS
```

La implementación renumera correctamente la aceptación final a R491 y la
auditoría de config a R493. La fixture de config futura también crea una
auditoría de amendment R492 (`tests/test_wave60_frozen_policy_transport.py:6384-6391`).
Pero `_validate_wave60_invalid_preparation_recovery_amendment()` todavía llama:

```python
validate_wave60_audit_commit(
    ...,
    scope="INVALID_PREPARATION_RECOVERY_AMENDMENT",
    ...,
    expected_audit_id="R490",
)
```

en `prepare_wave56_fresh.py:5600-5610`. Éste es el ramal productivo que deberá
autenticar la amendment v2. Una auditoría válida R492 será rechazada por ID. R490
no puede suplirla: ya audita el plan R489 y su scope/target tampoco coinciden con
`INVALID_PREPARATION_RECOVERY_AMENDMENT` y `amendment_sha256`.

La suite queda verde porque el test de partición futura usa un archivo textual
R492 como antecedente del commit de config, pero no atraviesa el validador
completo de amendment. La corrección requerida es cambiar ese
`expected_audit_id` productivo a `R492` y añadir un test que alcance este ramal
con la numeración futura. No corresponde modificar otros guards ni ciencia.

## Pruebas, recursos y limpieza

Todo se ejecutó CPU-only con `CUDA_VISIBLE_DEVICES=''`, sin usar ni consultar
GPU, bajo un único root temporal dedicado en `/mnt/m2-1TB`:

```text
Wave 60 completa:
147 passed in 346.01s
wall 5:47.29; max RSS 921092 KiB; process swaps 0

Regresión explícita Waves 56–59 (nueve archivos test_wave56..test_wave59):
336 passed, 1 skipped in 395.64s
wall 6:36.90; max RSS 1065776 KiB; process swaps 0
```

El root `/mnt/m2-1TB/.wave60-r491-audit.RybIRi` fue inventariado antes de
borrarlo: modo `0700`, owner `0:0`, `5.9 GiB` y `14,824` archivos. Se eliminó
por su path exacto y se verificó su ausencia. El swap global del host permaneció
en aproximadamente `22 GiB`; `/usr/bin/time` registró cero swaps para ambas
corridas. Este informe es el único archivo creado y no forma parte del target.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R491",
  "scope": "INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
  "target": {
    "implementation_commit": "4e50425bc8c33ed668da6de21dcd2a22882268fa"
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
