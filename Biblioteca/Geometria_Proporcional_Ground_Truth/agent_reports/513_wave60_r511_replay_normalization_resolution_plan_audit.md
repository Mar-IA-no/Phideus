# R513 — Auditoría del plan de resolución de los findings R511 de Ola 60

## Dictamen técnico: REVISE

El plan R512 conserva correctamente la ciencia y resuelve una parte sustantiva
de R511: fija la genealogía lineal, autentica las autoridades históricas,
congela la topología física de 141 archivos y amplía la matriz de ataques. La
regla de normalización continúa siendo correcta y viable por CPU, sin tocar ni
reejecutar el intento. Sin embargo, el contrato todavía deja dos superficies
materiales abiertas: no define de forma exacta las autoridades futuras que el
checker deberá aceptar, y no dispone de una autoridad de metadata para los tres
manifests autorreferenciales que quedan fuera de sus propios inventarios.

Encontré **0 HIGH, 2 MEDIUM y 1 LOW**. Los findings no cuestionan la integridad
actual del intento, el diagnóstico de R509 ni los patrones `false/false`; sí
impiden autorizar todavía R514 a partir de este plan.

## Identidad y alcance auditados

El target es el commit
`bb300df7bfb47021a30072a209054fe4c5be4efb`, hijo directo de R511
`86405020425f7c2310a68d66a215e3a35a00e982`. Introduce exclusivamente
`Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_R511_REPLAY_NORMALIZATION_RESOLUTION_PLAN.md`.
El blob Git y el archivo físico coinciden en SHA-256
`faa752fa22fead8609feef293d9c791262fe59bf6a6c51b8fcd6ff173755b57e`;
`git diff --check` pasa.

Leí completos R509, R510 y R511. Sus commits forman la cadena directa y
exclusiva declarada; sus archivos físicos y blobs coinciden respectivamente en
`006a43e9257a340b162583bcf1190cf34a27085186643a1ea37d987b4fa45e28`,
`31dc9a49e5b11df9d369537878651da788c0607e80c9d0e8fffd8093a584acee` y
`bb80e4efd3f8dc896ac20b83611ed7c46d25e2b3a229a2c358a1c694b70b789c`.
R509 y R511 contienen cada uno un único bloque de autoridad JSON con keyset,
scope, target, veredicto y conteos esperados.

Esta auditoría evalúa sólo el plan. No implementa el checker, no publica la
corrección, no modifica ni reejecuta el intento v4 y no emite `GO/NO-GO`.

## Findings

### R513-01 — MEDIUM — La autoridad futura sigue sin bindings y targets normativos completos

R512 fija correctamente commit, parent, path y SHA de R509–R511, y exige una
genealogía futura exclusiva. Para R513, R515 y R517, en cambio, sólo prescribe
scope y que el target ligue «commit y SHA-256 exactos del objeto auditado»
(`WAVE_60_R511_REPLAY_NORMALIZATION_RESOLUTION_PLAN.md:56-65`). No fija los
paths exactos de esos informes, los nombres y keysets exactos de sus bindings ni
los mapas `target` exactos. La ambigüedad es material en R515: su objeto son dos
paths de implementación, pero el plan no define si el target liga sólo el
commit, dos hashes de archivo, un manifest ordenado u otro digest compuesto
(`WAVE_60_R511_REPLAY_NORMALIZATION_RESOLUTION_PLAN.md:48-61,133-140`).

La salida R516 debe «ligar R509–R515» y tener schema/keysets exactos, pero el
plan tampoco enumera el keyset de esas autoridades ni los campos obligatorios
`commit/path/sha256` para cada una
(`WAVE_60_R511_REPLAY_NORMALIZATION_RESOLUTION_PLAN.md:63-65,148-158`). Por
ello los ataques de binding faltante/extra, target divergente o path distinto
que la matriz promete rechazar no tienen todavía un valor normativo único
contra el cual fallar (`WAVE_60_R511_REPLAY_NORMALIZATION_RESOLUTION_PLAN.md:160-166`).

Esto no queda suplido por exigir un único bloque JSON. El validador precedente
distingue ambas capas: compara el mapa `target` exacto y el keyset de autoridad
(`experiments/geometria_proporcional/prepare_wave56_fresh.py:431-460`), y
valida por separado `audit_commit`, `audit_path`, `audit_sha256`, introducción,
blob, exclusividad y parent directo
(`experiments/geometria_proporcional/prepare_wave56_fresh.py:524-551`). R511
pidió precisamente bindings exactos y scopes/targets exactos para cada eslabón
(`agent_reports/511_wave60_r509_replay_normalization_resolution_plan_audit.md:57-70`).

Corrección requerida: definir los paths y bindings exactos de R513/R515/R517,
los mapas target completos y sus keysets, y una regla inequívoca para ligar la
implementación de dos archivos. Definir también el keyset exacto de autoridades
en R516. Los valores que sólo existen después de un commit pueden suministrarse
al publicar, pero sus nombres, cardinalidad, relación con el pathset y reglas de
validación deben quedar fijados antes de implementar.

### R513-02 — MEDIUM — La metadata de los manifests autorreferenciales no está congelada

R512 exige que tamaño, owner, group, mode y SHA-256 de «cada archivo» coincidan
con el root/pair manifest que lo compromete, y promete rechazar toda deriva de
metadata (`WAVE_60_R511_REPLAY_NORMALIZATION_RESOLUTION_PLAN.md:67-80,167-169`).
Esa condición no puede aplicarse literalmente a los tres
`artifact_manifest.json`: cada manifest declara una self-reference con hashes
omitidos y el validador elimina ese path antes de comparar inventario y metadata
(`experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:3175-3223`).

La inspección física confirmó 65 archivos en primary, 66 en replay y 10 en
pair, pero los manifests contienen sólo 64, 65 y 9 registros. El único path
físico ausente en cada inventario es su propio `artifact_manifest.json`. Los
seis hashes target fijan los bytes de los tres manifests, no su owner, group o
mode. Un `chmod` o `chown` de uno de ellos conserva SHA, roster, inode,
`st_nlink=1`, firmas y todos los registros de manifest; por lo tanto puede pasar
una implementación literal de las guardas restantes aunque contradiga la
identidad física que R511 pidió preservar. El problema no requiere abrir
semánticamente secretos.

La metadata observada hoy es: primary, 17.792 bytes; replay, 18.047 bytes; pair,
2.448 bytes. Los tres tienen UID/GID `0:0`, mode `0444` y `st_nlink=1`. Estos
valores permiten construir el binding faltante, pero la observación de R513 no
los vuelve por sí sola autoridad normativa para el checker futuro.

Corrección requerida: añadir una autoridad exacta y autenticada para metadata
de los tres manifests autorreferenciales —o un snapshot cerrado de los 141
paths que incluya sus cinco campos físicos— y hacer que los tests alteren
específicamente owner/group/mode de un self-manifest. Mantener los hashes target
como binding de bytes, no presentarlos como binding de metadata.

### R513-03 — LOW — R516 atribuiría cierre a R517 antes de su existencia

La nueva etiqueta ya no convierte a R509 en la resolución, lo cual corrige el
error nominal detectado por R511. Sin embargo, el plan dice que la nota
publicada por R516 significa que «la cadena R512–R517 resolvió» los findings
(`WAVE_60_R511_REPLAY_NORMALIZATION_RESOLUTION_PLAN.md:127-131`), mientras
R516 sólo puede ligar autoridades hasta R515 y R517 aún no existe
(`WAVE_60_R511_REPLAY_NORMALIZATION_RESOLUTION_PLAN.md:48-65,148-158`). La
autoridad auditada recién queda disponible tras R517.

Corrección requerida: hacer que R516 describa una adjudicación candidata
derivada por la cadena disponible R512–R515 y que R517 conceda la autoridad de
cierre, o formular la nota sin afirmar dentro de R516 que su auditoría futura
ya ocurrió. La documentación puede adoptar la vista corregida después del PASS
de R517.

## Estado físico y semántico comprobado

Las comprobaciones read-only confirman el estado que el plan pretende
preservar:

- attempt, primary, replay y pair son directorios físicos canónicos;
- hay exactamente 65/66/10 archivos, 141 en total, 141 pares
  `(st_dev, st_ino)` únicos, todos regulares, todos con `st_nlink=1`, sin
  symlinks ni nodos especiales;
- los 138 paths no autorreferenciales coinciden exactamente con sus manifests
  en roster, bytes, SHA-256, owner, group y mode;
- los seis hashes target de R509 y los ocho source bindings de la config R508
  coinciden; el self-binding de la config se verificó con su normalización
  canónica;
- las nueve attestations verifican criptográficamente y `pair_status.json`
  conserva `COMPLETE`, dos roots `EVALUATED_IMMUTABLE`,
  `any_truth_accessed=true`, `recovery_allowed=false`;
- `compare_evaluated_roots()` recompone exactamente el JSON publicado: 36
  checks, 35 verdaderos y el único falso
  `operational:preparation_receipt.json`;
- cada preparation receipt liga su generation receipt local, los freezes son
  byte-exactos, los generation receipts tienen keyset idéntico y difieren sólo
  en `execution_mode=recovery/replay`;
- con `normalized_replay_exact=true`, `finalize_patterns()` conserva
  `incompatibility=false` y `harm=false`; decisión `null`, autoridad `user`.

La recomposición focal tardó 1,63 s, alcanzó 827.684 KiB de RSS y registró cero
swaps. Todas las invocaciones efectivas fijaron `CUDA_VISIBLE_DEVICES=''` y
`PYTHONDONTWRITEBYTECODE=1`. No se usó ni consultó GPU.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R513",
  "scope": "R511_REPLAY_NORMALIZATION_RESOLUTION_PLAN",
  "target": {
    "plan_commit": "bb300df7bfb47021a30072a209054fe4c5be4efb",
    "plan_sha256": "faa752fa22fead8609feef293d9c791262fe59bf6a6c51b8fcd6ff173755b57e"
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
