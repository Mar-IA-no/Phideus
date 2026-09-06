# R515 — Auditoría del plan final de resolución de normalización de replay de Ola 60

## Dictamen técnico: PASS

El plan R514 cierra los tres findings de R513 y deja una especificación
implementable de manera fail-closed. La cadena futura R514–R519, sus paths,
parents, pathsets, bindings, targets y autoridades JSON quedan definidos antes
de implementar; los tres manifests autorreferenciales reciben una autoridad de
metadata exacta; y la identidad física requerida cubre los 141 archivos del
attempt. La futura corrección R518 queda representada honestamente como
`CANDIDATE_PENDING_R519_AUDIT`: no afirma que su auditoría futura ya ocurrió ni
activa por sí sola la vista corregida.

No encontré findings materiales: **0 HIGH, 0 MEDIUM y 0 LOW**. La corrección
prevista cambia únicamente la adjudicación operacional de replay. No modifica
el intento sellado, la config, sources, logits, acciones, métricas, bootstrap,
condiciones científicas distintas de `replay_exact`, patrones, autoridad
decisional ni estado de promoción.

## Identidad y alcance auditados

El target es el commit
`574f79810e5283dec0478c8c1c53e3496c808e97`, hijo directo de R513
`b522debe93ee55b7abba9ec04e5de67669c6231b`. El commit introduce exclusivamente
`Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_R513_REPLAY_NORMALIZATION_FINAL_RESOLUTION_PLAN.md`.
El archivo físico y su blob Git son idénticos y tienen SHA-256
`9f96084160627a4c9102949ff413d98b6c28a22a20bf2b6c4e115e44cc888b78`;
`git diff --check` pasa.

Leí completos R509, R510, R511, R512 y R513. Sus seis commits hasta R514
forman una genealogía directa; cada commit introduce exactamente un path y
cada archivo físico coincide con su blob. Los hashes históricos declarados por
R514 coinciden `5/5`, y los únicos bloques JSON de R509, R511 y R513 coinciden
`3/3` con sus scopes, targets, veredictos y conteos normativos.

Esta auditoría revisa el plan y el estado read-only que debe autenticar su
implementación. No implementa el checker, no publica R518, no modifica ni
reejecuta el attempt y no emite una decisión `GO/NO-GO`.

## Cierre de R513-01 — autoridades futuras exactas

R514 reemplaza la cadena tentativa anterior por una genealogía inequívoca:
R515 audita el plan; R516 introduce exactamente checker y test; R517 audita
esa implementación; R518 introduce exclusivamente el JSON candidato; y R519
audita ese artefacto. Los cinco paths están fijados en el plan y cada commit
debe ser exclusivo e hijo directo del anterior
(`WAVE_60_R513_REPLAY_NORMALIZATION_FINAL_RESOLUTION_PLAN.md:22-45`).

Los informes R515, R517 y R519 tienen un único bloque JSON, keyset superior
común, scope y target específicos y findings `0/0/0` para conceder PASS. R517
liga sin ambigüedad la implementación de dos archivos mediante un mapa
`files` de dos claves exactas; R519 liga commit, path y SHA del artefacto ya
publicado (`WAVE_60_R513_REPLAY_NORMALIZATION_FINAL_RESOLUTION_PLAN.md:47-116`).

R518 contiene nueve autoridades con nombres exactos. Los planes se ligan por
`commit/path/sha256`, las auditorías además preservan el bloque de autoridad
completo y la implementación usa `commit/files`; los valores históricos
R509–R513 están congelados en el propio plan
(`WAVE_60_R513_REPLAY_NORMALIZATION_FINAL_RESOLUTION_PLAN.md:118-159`). El
payload R518 también tiene keyset superior y keysets anidados normativos,
bindings exactos para config, sources, intento, métricas y limitaciones, y debe
rechazar claves ausentes o agregadas en cada nivel
(`WAVE_60_R513_REPLAY_NORMALIZATION_FINAL_RESOLUTION_PLAN.md:187-289`).

Con ello, parent incorrecto, commit mixto, introducción distinta, path/blob/hash
divergente, binding faltante o extra, autoridad ausente/duplicada/contradictoria
y mutación de keysets tienen un único valor normativo contra el cual fallar.
R513-01 queda cerrado.

## Cierre de R513-02 — identidad física y self-manifests

El filesystem confirma exactamente la topología que R514 congela:

| Superficie | Observación | Contrato R514 |
|---|---:|---:|
| primary | 65 archivos | 65 |
| replay | 66 archivos | 66 |
| pair | 10 archivos | 10 |
| total | 141 regulares | 141 |
| `st_nlink=1` | 141 | 141 |
| pares `(st_dev, st_ino)` únicos | 141 | 141 |
| symlinks / nodos especiales | 0 / 0 | 0 / 0 |

Los manifests internos comprometen correctamente 64, 65 y 9 paths: 138 en
total. Los tres paths omitidos son únicamente sus self-references. La nueva
tabla normativa coincide `3/3` con el estado físico:

| Self-manifest | bytes | uid:gid | mode | SHA-256 |
|---|---:|---:|---:|---|
| primary | 17.792 | `0:0` | `0444` | `a96497d5a06e8ec23b7844aa13a2ef7455ef0a8bf6b410980a96ef8ebf7ed982` |
| replay | 18.047 | `0:0` | `0444` | `4ffabc2bd54de623820cd373b84ad4f95e49ff45b7ddc48da8eaa894f7dc7eb0` |
| pair | 2.448 | `0:0` | `0444` | `4a51993c420a96f9f8283a3ced5e9dee279097ce37453fbd2b7480580805e686` |

R514 exige para los otros 138 paths roster, bytes, SHA, UID, GID y mode, y
para los tres self-manifests la tabla anterior, además de directorios físicos,
ausencia de aliases, roster cerrado y unicidad de inodos
(`WAVE_60_R513_REPLAY_NORMALIZATION_FINAL_RESOLUTION_PLAN.md:161-184`). También
exige negativos específicos sobre mode y owner/group de un self-manifest. Ya
no existe la superficie de `chmod/chown` no autenticada observada por R513-02.

Los seis hashes target coinciden `6/6`; los ocho source bindings coinciden
`8/8`; y la config coincide tanto en SHA físico como en self-binding canónico
`2/2`. Los secretos sólo fueron hasheados como bytes opacos, sin interpretación
semántica.

## Cierre de R513-03 — candidato antes de autoridad

R518 declara `artifact_status=CANDIDATE_PENDING_R519_AUDIT` y contiene una
condición de activación exacta que exige R519, scope correcto, PASS y findings
`0/0/0`. El artefacto sólo presenta la vista normalizada como derivación
condicional; la documentación no puede adoptarla antes de R519
(`WAVE_60_R513_REPLAY_NORMALIZATION_FINAL_RESOLUTION_PLAN.md:187-227`). R519
repite además la derivación y autentica commit, exclusividad y SHA de R518
(`WAVE_60_R513_REPLAY_NORMALIZATION_FINAL_RESOLUTION_PLAN.md:291-308`). Así,
R518 no atribuye a una auditoría futura un cierre ya ocurrido. R513-03 queda
cerrado.

## Contraste con código y artefactos

La recomposición read-only de `compare_evaluated_roots()` coincide exactamente
con `pair/replay_comparison.json`: 36 checks, 35 verdaderos y un único mismatch
`operational:preparation_receipt.json`. El código compara primero los campos
semánticos del generation receipt y luego reintroduce el SHA local dentro de
la comparación de preparation receipt
(`run_wave60_frozen_policy_transport.py:3364-3387,3399-3442`). La regla del
plan —integridad referencial local y normalización semántica cross-root— ataca
exactamente ese defecto y no abre una allowlist genérica.

La reagregación con `finalize_patterns()` reproduce la vista histórica
`false/false`; al cambiar sólo `replay_exact` a `true`, ambos patrones continúan
`false/false` (`wave60_frozen_policy_transport.py:933-946`). El final sellado
conserva `scientific_decision=null`, `decision_authority=user` y las tres
limitaciones científicas que R518 debe mantener. Por tanto, la normalización no
altera la ciencia ni convierte el resultado en promoción arquitectónica.

### Precaución de implementación, no finding

El helper congelado `_validate_worker_phase()` construye hoy el payload
esperado de attestations con `git_commit()` del `HEAD` corriente
(`run_wave60_frozen_policy_transport.py:2932-2943`). Una invocación ingenua de
`validate_evaluated_root()` desde R516, cuando `HEAD` ya sea posterior a la
corrida, rechaza las attestations legítimas firmadas en
`789c4ea2298fcaba97c9bdecdd1db4360186012c`. Al validarlas contra esa autoridad
histórica firmada y configurada, los dos roots pasan íntegramente y la
recomposición vuelve a dar 36/35 con el mismatch esperado.

El workaround correcto para R516 es validar los roots contra el commit
histórico ligado por config/attestation, sin modificar el runner congelado y
sin hacer depender la autoridad del `HEAD` actual. R514 aporta exactamente ese
binding mediante `config_binding.audit` y preserva el SHA del runner; por eso
la cautela no abre un finding del plan. R517 debe comprobar esta propiedad y
un negativo de commit histórico divergente.

## Viabilidad y matriz adversarial

La implementación puede separar de forma segura: autenticación histórica,
inventario físico, recomposición read-only, construcción determinista del
payload, validación cerrada y publicación exclusiva. Los ataques heredados de
R512 cubren parent/path/blob/hash/JSON, hardlinks, symlinks, nodos especiales,
metadata, roster, target/source drift, enlaces locales cruzados, mismatches
adicionales, receipts con roles o campos divergentes, transición de patrones y
publicación no canónica. R514 añade keyset mutation por nivel y el ataque
específico a metadata de self-manifests.

Todo el trabajo requerido es hashing, verificación criptográfica, lectura
estructural y agregación booleana. Es nativo de CPU; no exige ni justifica GPU,
re-forward o training. Esta auditoría no usó ni consultó GPU.

## Comprobaciones ejecutadas

1. Git, parent directo, introducción exclusiva, pathset y blob/SHA para
   R509–R514: `6/6 PASS`.
2. Única autoridad JSON y valores exactos de R509/R511/R513: `3/3 PASS`.
3. Identidad física: `141/141` archivos regulares, `141/141` con
   `st_nlink=1`, `141/141` pares device/inode únicos y self-manifests `3/3`.
4. Seis hashes target, ocho source bindings y config físico/self-binding:
   `6/6`, `8/8` y `2/2 PASS`.
5. Validación integral de ambos roots contra la autoridad histórica firmada:
   `2/2 PASS`.
6. Recomposición exacta de replay: `36` checks, `35` verdaderos, único mismatch
   esperado.
7. Reagregación histórica y normalizada: `false/false` en ambos casos; sólo
   cambia `replay_exact`.
8. `git diff --check` del commit objetivo: PASS.

Todas las invocaciones efectivas fijaron `CUDA_VISIBLE_DEVICES=''` y
`PYTHONDONTWRITEBYTECODE=1`. No se crearon fixtures, no se escribió data y no
se abrió semánticamente material secreto.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R515",
  "scope": "R513_REPLAY_NORMALIZATION_RESOLUTION_PLAN",
  "target": {
    "plan_commit": "574f79810e5283dec0478c8c1c53e3496c808e97",
    "plan_sha256": "9f96084160627a4c9102949ff413d98b6c28a22a20bf2b6c4e115e44cc888b78"
  },
  "technical_verdict": "PASS",
  "findings": {
    "high": 0,
    "medium": 0,
    "low": 0
  },
  "files_modified": false,
  "gpu_used_or_queried": false
}
```
