# R511 — Auditoría del plan de resolución de normalización de replay de Ola 60

## Dictamen técnico: REVISE

El plan diagnostica correctamente el falso mismatch de R509 y propone una
resolución científicamente conservadora: una vista derivada fuera del intento
sellado, sin recovery, re-forward, retraining, cambio de métricas ni decisión
`GO/NO-GO`. La regla local-versus-cross-root es correcta y el caso físico es
viable por CPU. Sin embargo, el contrato futuro todavía no cierra dos
superficies materiales: la autoridad Git/auditoría de la cadena R510–R515 y la
identidad física sin aliases del intento que promete declarar inalterado.

Encontré **0 HIGH, 2 MEDIUM y 1 LOW**. Los findings no cuestionan los números ni
los patrones `false/false` recompuestos por R509; impiden autorizar todavía la
implementación a partir de este plan.

## Identidad y alcance auditados

El target es el commit
`fa6ee25b359e06c9bef2ce4ec08768d8f3a46ff8`, hijo directo de R509
`92305f4e54e72ee78924ca4b51ae5889369d805b`. El commit introduce un único path,
el plan auditado. Su blob Git y el archivo físico coinciden en SHA-256
`31dc9a49e5b11df9d369537878651da788c0607e80c9d0e8fffd8093a584acee`.
R509 también conserva su identidad declarada: el informe físico y su blob Git
tienen SHA-256
`006a43e9257a340b162583bcf1190cf34a27085186643a1ea37d987b4fa45e28`.

Esta auditoría evalúa únicamente el plan. No implementa el checker, no publica
la corrección JSON, no modifica ni reejecuta el intento v4 y no decide
promoción arquitectónica ni `GO/NO-GO`.

## Findings

### R511-01 — MEDIUM — La cadena futura no queda autenticada de manera fail-closed

El plan enumera R511, implementación, R513, artefacto y R515, y afirma que un
finding se resolverá en otro eslabón (`WAVE_60_R509_REPLAY_NORMALIZATION_RESOLUTION_PLAN.md:171-187`).
También pide que la salida ligue plan/auditoría e implementación/auditoría, pero
lo formula como un mínimo sin definir campos exactos, parentage, exclusividad
de commits ni semántica parseable de PASS
(`WAVE_60_R509_REPLAY_NORMALIZATION_RESOLUTION_PLAN.md:130-149`). La matriz de
ataques tampoco incluye parent incorrecto, commit mixto, blob distinto, bloque
JSON ausente/duplicado o autoridad contradictoria
(`WAVE_60_R509_REPLAY_NORMALIZATION_RESOLUTION_PLAN.md:151-169`).

Ese vacío permite una implementación literalmente compatible con el plan que
confíe en prosa, en un hash físico suelto o en un informe no exclusivo. También
permite que el JSON derivado preserve de R509 sólo `REVISE` y los conteos
`0/1/1`, tal como dice el mínimo (`WAVE_60_R509_REPLAY_NORMALIZATION_RESOLUTION_PLAN.md:135-140`),
sin autenticar necesariamente su commit, path, SHA-256 y único bloque JSON. No
es el régimen ya establecido en el código vigente: el parser de auditorías
exige un único bloque JSON, keyset exacto, scope/target exactos, PASS y
`0/0/0` (`prepare_wave56_fresh.py:431-460`), y los validadores de documentos
verifican introducción, commit exclusivo, blob y parent directo
(`prepare_wave56_fresh.py:504-546`).

Corrección requerida:

1. Definir bindings exactos `commit/path/sha256` para R509, plan/R511 e
   implementación/R513, con scopes, targets, veredictos y conteos exactos.
2. Exigir una única autoridad JSON fail-closed en cada auditoría; la prosa no
   concede autoridad.
3. Fijar la genealogía lineal: R511 hijo directo y exclusivo del plan;
   implementación hija directa de R511 y limitada exactamente a checker/test;
   R513 hija directa y exclusiva de la implementación; artefacto R514 hijo
   directo y exclusivo de R513; R515 hijo directo y exclusivo de R514.
4. Agregar ataques para parent incorrecto, commit no exclusivo, path/blob/hash
   divergente, binding faltante/extra y auditoría con JSON ausente, duplicado o
   contradictorio. R515 debe autenticar el commit y SHA del JSON ya publicado;
   el JSON R514 no necesita autorreferenciar un commit futuro.

### R511-02 — MEDIUM — Los hashes no prueban por sí solos ausencia de mutación física

El plan promete que la auditoría final comprobará ausencia de mutaciones en el
intento (`WAVE_60_R509_REPLAY_NORMALIZATION_RESOLUTION_PLAN.md:146-149`), pero
la única prueba física explícita fija los seis hashes target y los ocho hashes
de sources (`WAVE_60_R509_REPLAY_NORMALIZATION_RESOLUTION_PLAN.md:166-169`).
Eso no conserva la propiedad que R509 sí midió: 141 archivos, 141 inodos
únicos, `st_nlink=1`, sin symlinks ni nodos especiales
(`509_wave60_frozen_policy_transport_v4_result_or_terminal_audit.md:112-120`).

El validador histórico rechaza symlinks y nodos especiales, pero
`inventory_metadata()` sólo registra bytes, owner, group y mode; no registra
`st_dev`, `st_ino` ni `st_nlink`
(`run_wave60_frozen_policy_transport.py:943-970`).
`validate_evaluated_root()` compara precisamente ese inventario, el roster y
los hashes públicos (`run_wave60_frozen_policy_transport.py:3175-3233`). Por
ello un reemplazo por hardlink byte-idéntico puede conservar los manifests, los
seis hashes target, las attestations y la recomposición, aunque el paquete ya no
tenga la identidad física auditada. El ataque es concreto porque los dos
`evaluation/analysis.json` son byte-exactos y comparten SHA
`f1378cb22f45e10580c18d3f0d5d12d8b6a4ed19d39cea6fd612bf0121bdb5e0`
(`509_wave60_frozen_policy_transport_v4_result_or_terminal_audit.md:198-206`).

Corrección requerida: congelar como precondición del checker el roster físico
completo de 141 paths, todos regulares, `st_nlink=1` y pares `(st_dev, st_ino)`
únicos, además de los modos/owners/groups/tamaños/hashes ya ligados; validar por
separado que attempt, roots y pair son directorios físicos canónicos. Agregar
negativos para hardlink intra-root/cross-root, symlink, nodo especial, path
extra/faltante y deriva de metadata. Los secretos continúan opacos: para ellos
se compara metadata y hash comprometido, no contenido semántico.

### R511-03 — LOW — La nota normalizada atribuye la adjudicación al eslabón equivocado

La vista derivada reemplazaría la limitación pendiente por una nota según la
cual el replay fue adjudicado “mediante la resolución R509”
(`WAVE_60_R509_REPLAY_NORMALIZATION_RESOLUTION_PLAN.md:120-124`). R509 no es la
resolución: emitió `REVISE` con `0/1/1`
(`509_wave60_frozen_policy_transport_v4_result_or_terminal_audit.md:1-26`). La
adjudicación sólo adquiriría autoridad tras R514 y R515, según la propia cadena.

Corrección requerida: nombrar “resolución de los findings R509” y ligar la nota
al artefacto R514 auditado por R515. Mantener R509 como observación original
`REVISE`, sin reescribirlo como adjudicación.

## Superficies correctas del plan

- El intento v4 permanece post-truth, `COMPLETE` e inmutable; la salida vive
  fuera del paquete y se crea sin overwrite
  (`WAVE_60_R509_REPLAY_NORMALIZATION_RESOLUTION_PLAN.md:10-22,47-74`).
- La normalización distingue correctamente igualdad semántica cross-root de
  integridad referencial local: valida cada
  `generation_receipt_sha256` contra su propio root y permite únicamente
  `execution_mode=recovery/replay`
  (`WAVE_60_R509_REPLAY_NORMALIZATION_RESOLUTION_PLAN.md:76-102`).
- Rechaza mismatches adicionales científicos, funcionales, secretos u
  operacionales y no adopta una regla genérica de ignorar diferencias
  (`WAVE_60_R509_REPLAY_NORMALIZATION_RESOLUTION_PLAN.md:85-102,151-168`).
- La derivación sólo vuelve a agregar condiciones booleanas selladas; no
  recalcula logits, acciones, métricas ni bootstrap, y conserva los patrones
  `false/false`, las tres limitaciones científicas, `scientific_decision=null` y
  la autoridad del usuario
  (`WAVE_60_R509_REPLAY_NORMALIZATION_RESOLUTION_PLAN.md:104-128`).
- El costo es viable por CPU. La recomposición read-only ejecutada para R511
  igualó exactamente el JSON original en 1,63 s, con RSS máximo 828.264 KiB y
  cero swaps; no se usó ni consultó GPU.

## Comprobaciones ejecutadas

1. Verificación Git de commit, parent directo, path exclusivo y SHA-256 físico/
   blob del plan; `git diff --check` pasó.
2. Verificación equivalente del commit y SHA-256 del informe R509.
3. Verificación física de los seis hashes target: pair manifest, pair status,
   final analysis, replay comparison y ambos root manifests; todos coinciden
   con R509.
4. Recomposición read-only de `compare_evaluated_roots()` con el runner vigente:
   igualdad exacta con el JSON publicado, 36 checks, 35 verdaderos y un único
   falso `operational:preparation_receipt.json`.
5. Comprobación independiente de ambos enlaces locales de generation receipt y
   de que los generation receipts sólo difieren estructuralmente en
   `execution_mode=recovery/replay`.
6. Inventario read-only actual: 141 archivos, cero `st_nlink != 1` y cero pares
   `(st_dev, st_ino)` duplicados. Esta observación confirma el estado presente;
   el finding R511-02 señala que el plan aún no obliga a preservarlo.
7. Inspección focal de `compare_evaluated_roots()`, `finalize_pair()`,
   `finalize_patterns()`, manifests, validators y fixtures. No se ejecutó el
   experimento ni se abrió semánticamente material secreto.

El primer intento de invocar el chequeo con el nombre genérico `python` falló
porque ese binario no existe en el entorno; se repitió correctamente con
`venv/bin/python`. Todas las comprobaciones efectivas fijaron
`CUDA_VISIBLE_DEVICES=''` y `PYTHONDONTWRITEBYTECODE=1`. No se usó ni consultó
GPU.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R511",
  "scope": "R509_REPLAY_NORMALIZATION_RESOLUTION_PLAN",
  "target": {
    "plan_commit": "fa6ee25b359e06c9bef2ce4ec08768d8f3a46ff8",
    "plan_sha256": "31dc9a49e5b11df9d369537878651da788c0607e80c9d0e8fffd8093a584acee"
  },
  "technical_verdict": "REVISE",
  "findings": {
    "high": 0,
    "medium": 2,
    "low": 1
  },
  "files_modified": false,
  "gpu_used_or_queried": false
}
```
