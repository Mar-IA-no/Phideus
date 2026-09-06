# R485 — Reauditoría final independiente de la implementación de recuperación `INVALID_PREPARATION`

**Dictamen técnico: `REVISE` — 0 HIGH / 1 MEDIUM / 0 LOW.**

La implementación sucesora completa la matriz física, la matriz del débito y la
secuencia positiva `unsigned → primary firmado → replay firmado → par → recovery`
en lo sustantivo. Tampoco encontré una vía productiva que promueva R483: el
validador real liga la provenance al objeto final `recovery_implementation`,
reservado para R485. Sin embargo, la matriz contractual del sufijo Git sigue
siendo parcial frente a las obligaciones exhaustivas y explícitas del plan
R483. Además, una positiva de transaction/provenance construye y firma
artificialmente `implementation_audit: {audit_id: R483}`. Por eso no corresponde
emitir la autoridad positiva `0/0/0` requerida para publicar la amendment.

## Identidad, alcance y frontera congelada

El target auditado es exactamente:

- commit: `f32ba2bb5a6f38b4a3e9e9afbafb9393430f6719`;
- parent directo R484: `f26404044cd87cc14deea22cb0d14fa54b6134ae`;
- árbol Git: `99f80a4c5e34c6ab285c9c42cfe34ea02e05bc34`;
- diff exclusivo: `761` inserciones y `40` eliminaciones en:
  - `experiments/geometria_proporcional/prepare_wave56_fresh.py` (`198/8`);
  - `tests/test_wave60_frozen_policy_transport.py` (`563/32`).

Leí completos el plan R483, el informe R483 y la auditoría R484; inspeccioné el
diff íntegro de ambos sources modificados y las defensas productivas relacionadas.
Sus bindings físicos/Git son exactos:

| Artefacto | Commit | SHA-256 físico/Git |
|---|---|---|
| plan R483 | `a034a9aa36e673f0376c2af0a26a54dbf463eb43` | `0c22469829176132efb9f039ab261f8cc93821f440117f54669f20d3623bd352` |
| informe R483 | `b944cac9fbcbb3f28e2d7753161c58c978868226` | `7aa50f0fca72b3f796ddd4319f99cf2b9d60c005cbca651ac04c113d3d6be905` |
| auditoría R484 | `f26404044cd87cc14deea22cb0d14fa54b6134ae` | `2b8453e1ce541a7c6ce24d9ecc9b85d6b19c8bf870775ebf55e18cc6f9f09b21` |

Los blobs modificados coinciden con el target y el filesystem:

| Path | SHA-256 parent | SHA-256 target/físico |
|---|---|---|
| `prepare_wave56_fresh.py` | `fd3a3809fc98c825cf1c3159b6af8602db50c59a529d7a32b12f2148f687cfb5` | `b21d89af10021904347563997b3ec1e13292558cfc30519bd2c5f4d8cc8d7f32` |
| `test_wave60_frozen_policy_transport.py` | `d8e5c78554f5eb5ae1be5472f42ba15761e7034f305591a8f2c7931f63f7ce62` | `979054582ca1a61e6834f19d65e8d39954b431e1482f2c72a14af830c0363fdd` |

Módulo, runner y worker permanecen byte-exactos respecto de R475
`9f1a229d9c0ccb5e46b921e6c92281becc317139`, del target y del filesystem:

- módulo: `46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65`;
- runner: `1c778c3e60c1bbcebeb5c83430601a7c0b148e447528195f1dec4296322825aa`;
- worker: `c6c5c832633d9ed8a99079dd4fc4f5677073f7e1883a204fa6bb41f6d646bac7`.

`git diff --check` quedó limpio. No se modificaron módulo, runner, worker,
config, amendment ni artefactos científicos.

## Evidencia que sí quedó cerrada

La matriz física ahora prueba de forma independiente bytes de escrow, freeze,
manifest y benchmark; modos y owners; symlink, hardlink, FIFO, archivo auxiliar
autorizado por inventario, inventarios y attestations de primary/replay/par, y
drift del mapa preservado (`tests/test_wave60_frozen_policy_transport.py:2408-2518`).
Cada caso llega a la defensa productiva de origen, que además exige closed world,
metadata root, `nlink=1`, inventario exacto y autoridades firmadas
(`prepare_wave56_fresh.py:4403-4584`).

La matriz del débito cubre `0`, negativo, `59`, el resto del objeto exacto,
versión/container/source, path/hash de amendment, boundaries parciales y presencia
de ambas piezas en primary o replay
(`tests/test_wave60_frozen_policy_transport.py:2521-2617`). La secuencia positiva
parte realmente del v1 `INVALID_PREPARATION`, obtiene `60`, firma primary con
acumulado `61`, firma replay con acumulado `63`, publica el par pre-truth con
`0.5` segundos adicionales y recupera v3 con prior `63.5`, mientras un guard
demuestra que el ramal unsigned no vuelve a ejecutarse
(`tests/test_wave60_frozen_policy_transport.py:2620-2820`).

En producción, `_wave60_invalid_preparation_unsigned_debit()` autentica la
amendment y el origen físico, exige ausencia completa de ambos boundaries y
devuelve exactamente `60.0`; `wave60_prior_preparation_elapsed()` cambia al
ledger firmado apenas existe un boundary completo y valida la duración durable
del par (`prepare_wave56_fresh.py:6749-6864`). No encontré doble débito ni
aceptación de authority parcial.

El preparador amplía el keyset de amendment a las 27 claves requeridas, preserva
ambos `REVISE`, añade plan R483/R484, reserva la autoridad final para R485 y
renumera amendment/config/resultados a R486/R487/R488. La implementación final
desciende directamente de R484 y sólo cambia los dos paths autorizados.

## Finding MEDIUM — el sufijo Git no satisface su matriz exhaustiva y una positiva conserva R483

El plan exige rechazos independientes de `scope`, `target`, `audit ID`,
`verdict` y conteos **en cada auditoría**, blob Git distinto del hash físico
declarado, cruce de blobs R475/recuperación y mutación real de cualquiera de los
tres sources científicos (`WAVE_60_INVALID_PREPARATION_RECOVERY_R483_RESOLUTION_PLAN.md:199-216`).
La fixture positiva sí arma toda la cadena, pero sus negativos no cubren ese
contrato completo:

- la tabla `audit_field_drifts` omite `target` en R481 y R483; omite
  `scope`, `target` y `verdict` en R482 y R484; y omite `target`, `verdict` y
  `findings` en R485 (`tests/test_wave60_frozen_policy_transport.py:4846-4869`);
- esas mutaciones cambian diccionarios de binding, pero ninguna construye un
  reporte Git cuyo blob haya cambiado manteniendo un hash físico declarado
  discrepante;
- `scientific_cross` sólo acorta la lista
  `unchanged_source_law_sources`; no altera ni commitea uno de los tres sources
  científicos, por lo que no prueba la defensa de igualdad de blobs frente a
  una mutación real (`tests/test_wave60_frozen_policy_transport.py:4904-4911`);
- los casos de path adicional reutilizan como árbol una revisión posterior
  completa más `unexpected-exclusive-path.txt`, en vez de añadir aisladamente un
  único path al árbol propio de cada clase de commit. El rechazo existe, pero no
  demuestra de manera independiente la exclusividad de cada transición
  (`tests/test_wave60_frozen_policy_transport.py:4913-4955`).

El punto señalado en la positiva de transaction/provenance es real pero no es
un bypass productivo. La fixture crea manualmente el contexto con
`implementation_commit = "8" * 40` e
`implementation_audit = {"audit_id": "R483"}`
(`tests/test_wave60_frozen_policy_transport.py:2034-2040`), ejecuta la
transacción, firma el receipt y luego exige igualdad de esa provenance. En el
flujo real, `_validate_wave60_invalid_preparation_recovery_amendment()` devuelve
como `implementation_audit` el objeto final `recovery_implementation`, ya
validado como R485 (`prepare_wave56_fresh.py:5181-5199`). Por tanto R483 no puede
autorizar producción, pero sí queda presentado como autoridad dentro de una
prueba positiva que debía representar la historia final. Esto contradice la
regla del plan de que `5aee5fb`/R483 no aparezca como autoridad positiva y puede
ocultar una regresión de provenance.

La corrección requerida es exclusivamente probatoria salvo que uno de los casos
nuevos reproduzca un defecto: usar R485 en la positiva de provenance y ampliar
la fixture Git con mutaciones aisladas de cada campo ausente, del blob Git frente
al binding físico y de un source científico real. Después debe repetirse la
suite focal y la regresión.

## Pruebas, recursos y limpieza

Todo se ejecutó CPU-only con `CUDA_VISIBLE_DEVICES=''`, sin usar ni consultar
GPU, bajo un único root temporal dedicado en `/mnt/m2-1TB`:

```text
Wave 60 completa:
147 passed in 259.19s
wall 4:20.42; max RSS 913640 KiB; process swaps 0

Regresión explícita Waves 56–59 (nueve archivos test_wave56..test_wave59):
336 passed, 1 skipped in 388.28s
wall 6:29.52; max RSS 1065508 KiB; process swaps 0
```

El root `/mnt/m2-1TB/.wave60-r485-audit.UWRp26` fue inspeccionado antes de
borrarlo: modo `0700`, owner `0:0`, `5.9 GiB` y `13,995` archivos. Se eliminó
por su path exacto y se verificó su ausencia. El swap global del host permaneció
en aproximadamente `22 GiB`; `/usr/bin/time` registró cero swaps para ambas
corridas. Este informe es el único archivo creado y no forma parte del target.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R485",
  "scope": "INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
  "target": {
    "implementation_commit": "f32ba2bb5a6f38b4a3e9e9afbafb9393430f6719"
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
