# Ola 60 — resolución de los findings R511 sobre la adjudicación de replay

> **Estado:** `PRE-IMPLEMENTATION / R511-REVISE / PAIR-COMPLETE-IMMUTABLE / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-06
> **Plan base R510:** commit `fa6ee25b359e06c9bef2ce4ec08768d8f3a46ff8`, SHA-256 `31dc9a49e5b11df9d369537878651da788c0607e80c9d0e8fffd8093a584acee`
> **Auditoría R511:** commit `86405020425f7c2310a68d66a215e3a35a00e982`, SHA-256 `bb80e4efd3f8dc896ac20b83611ed7c46d25e2b3a229a2c358a1c694b70b789c`, `REVISE 0/2/1`

## 1. Decisión de diseño

Se conserva el diagnóstico y la regla semántica del plan R510, pero ese plan
no autoriza implementación. Esta resolución añade las precondiciones que R511
encontró ausentes: autoridad Git fail-closed, genealogía lineal exclusiva e
identidad física sin aliases del intento.

La corrección seguirá siendo una adjudicación externa. No modificará ni
volverá a ejecutar
`data/geometria_proporcional/wave60_frozen_policy_transport_attempt_v4`, la
config R508 ni ninguno de sus ocho sources. No recalculará logits, acciones,
métricas o bootstrap.

## 2. Autoridades históricas exactas

El checker debe autenticar estas autoridades mediante `commit`, `path`,
SHA-256 físico y blob Git, parent directo, pathset exclusivo y un único bloque
JSON parseable:

| Autoridad | Commit | Parent | Path exclusivo | Veredicto/findings |
|---|---|---|---|---|
| R509 | `92305f4e54e72ee78924ca4b51ae5889369d805b` | `789c4ea2298fcaba97c9bdecdd1db4360186012c` | `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/509_wave60_frozen_policy_transport_v4_result_or_terminal_audit.md` | `REVISE 0/1/1` |
| R510 | `fa6ee25b359e06c9bef2ce4ec08768d8f3a46ff8` | R509 | `Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_R509_REPLAY_NORMALIZATION_RESOLUTION_PLAN.md` | plan SHA `31dc9a49...` |
| R511 | `86405020425f7c2310a68d66a215e3a35a00e982` | R510 | `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/511_wave60_r509_replay_normalization_resolution_plan_audit.md` | `REVISE 0/2/1` |

Los hashes completos de R509 y R511 son, respectivamente,
`006a43e9257a340b162583bcf1190cf34a27085186643a1ea37d987b4fa45e28` y
`bb80e4efd3f8dc896ac20b83611ed7c46d25e2b3a229a2c358a1c694b70b789c`.
El parser rechazará cero o más de un bloque JSON, JSON inválido, keyset
faltante o extra, `audit_id`, `scope`, `target`, verdict, conteos,
`files_modified` o `gpu_used_or_queried` distintos de los esperados. La prosa
no concede autoridad.

## 3. Genealogía correctiva normativa

La cadena que reemplaza la secuencia tentativa de R510 es:

```text
R511 REVISE
  -> R512 este plan, commit exclusivo hijo directo de R511
  -> R513 auditoría de este plan, PASS 0/0/0, exclusiva e hija directa de R512
  -> R514 implementación, exactamente checker + test, hija directa de R513
  -> R515 auditoría de implementación, PASS 0/0/0, exclusiva e hija directa de R514
  -> R516 JSON de corrección, exclusivo e hijo directo de R515
  -> R517 auditoría del JSON, PASS 0/0/0, exclusiva e hija directa de R516
  -> documentación y wiki
```

Cada eslabón futuro debe validar el parent y pathset exactos de todos los
anteriores. R513 usa scope
`R511_REPLAY_NORMALIZATION_RESOLUTION_PLAN`; R515,
`R509_REPLAY_NORMALIZATION_RESOLUTION_IMPLEMENTATION`; R517,
`WAVE60_V4_REPLAY_NORMALIZATION_CORRECTION`. Sus targets deben ligar el commit
y SHA-256 exactos del objeto auditado.

El JSON R516 ligará todas las autoridades hasta R515. No intentará
autorreferenciar su commit futuro. R517 será quien autentique commit, parent,
exclusividad, path y SHA-256 de R516.

## 4. Identidad física congelada

Antes de normalizar, el checker debe verificar la topología física observada
por R509:

- el attempt y sus directorios `primary`, `replay` y `pair` son directorios
  físicos canónicos, no symlinks;
- existen exactamente 141 archivos: 65 en primary, 66 en replay y 10 en pair;
- cada path del roster es un archivo regular, no symlink ni nodo especial;
- los 141 pares `(st_dev, st_ino)` son únicos y cada `st_nlink` es 1;
- no hay paths extra o faltantes;
- tamaño, owner, group, mode y SHA-256 de cada archivo coinciden con el
  root/pair manifest que lo compromete;
- los manifests conservan sus hashes target R509.

Los secretos siguen opacos: el checker verifica bytes mediante el digest ya
comprometido y metadata, pero no interpreta su contenido. La recomposición
semántica sólo abre artefactos públicos u operacionales autorizados.

También debe fijar los seis hashes target de R509 y los ocho source bindings
de la config R508. Para el self-binding de la config se aplica su normalización
canónica; para los otros siete sources se exige hash físico exacto.

## 5. Regla semántica preservada

Sólo se admite la normalización si, además de la identidad anterior:

1. el terminal es `COMPLETE` y ambos roots son `EVALUATED_IMMUTABLE`;
2. pair/root manifests, firmas, receipts y enlaces locales son válidos;
3. la recomposición histórica produce exactamente 36 checks: 35 verdaderos y
   el único mismatch
   `operational:preparation_receipt.json`;
4. cada preparation receipt liga el SHA de su propio generation receipt;
5. los preparation freezes son byte-exactos y ambos receipts comparten
   `preparation_freeze_sha256` y `next_state=PREPARED`;
6. los generation receipts tienen keyset idéntico y, después de retirar
   `execution_mode`, son exactamente iguales; los únicos valores retirados son
   `recovery` para primary y `replay` para replay;
7. los ocho artefactos científicos y los analysis completos son exactos;
8. R509 permanece ligado como la recomposición numérica independiente que
   validó 14 acciones, 56 arrays, 301 pair-tokens y 5.000 bootstraps.

Cualquier otra diferencia aborta. No hay allowlist extensible ni opción para
ignorar checks.

## 6. Vista corregida

El checker derivará `normalized_replay_exact=true`, llamará a la agregación
con las condiciones root-level selladas y comprobará:

- patrones originales `false/false`;
- patrones normalizados `false/false`;
- incompatibility continúa fallando el contraste de regret contra controles
  de desplazamiento máximo, con límite superior `+0.001548...`;
- harm continúa fallando worst regret contra esos controles, con límite
  superior `+0.009468...`;
- `scientific_decision=null` y `decision_authority=user`;
- los hashes de ambos analysis y todos los valores científicos permanecen
  ligados al original.

La limitación `replay_exact_pending_pair_finalize` se sustituirá por
`replay_exact_adjudicated_by_r509_findings_resolution_chain`. La nota significa
que la cadena R512–R517 resolvió los findings abiertos por R509; no presenta
R509 como resolución. Se preservan exactamente las otras tres limitaciones
científicas.

## 7. Implementación y publicación

R514 añadirá solamente:

```text
experiments/geometria_proporcional/adjudicate_wave60_v4_result.py
tests/test_wave60_v4_result_adjudication.py
```

El checker tendrá dos operaciones separadas:

- validación read-only y construcción determinista en memoria;
- publicación canónica por creación exclusiva fuera del attempt, sin
  overwrite, seguida de validación de su propia salida.

R516 añadirá solamente:

```text
Biblioteca/Geometria_Proporcional_Ground_Truth/waves/
WAVE_60_V4_REPLAY_NORMALIZATION_CORRECTION.json
```

El schema y todos sus keysets serán exactos. Ligará R509–R515, config R508,
seis hashes target, ocho sources, inventario físico, observación original,
evidencia normalizada, condiciones/patrones antes y después, limitaciones y
autoridad decisional.

## 8. Matriz mínima de ataques

Los tests deben rechazar:

1. parent incorrecto o commit no exclusivo en cualquier eslabón;
2. path, blob, SHA, target, scope, verdict o conteos divergentes;
3. auditoría sin JSON, con dos JSON, JSON inválido o autoridad contradictoria;
4. hardlink intra-root, cross-root o hacia fuera del attempt;
5. symlink, nodo especial, path extra/faltante o deriva de metadata;
6. hash target o source binding alterado;
7. enlace local a generation receipt roto o cruzado;
8. mismatch científico, funcional, secreto u operacional adicional;
9. generation receipts con otra diferencia, campo extra o roles distintos;
10. transición incorrecta de condiciones, patrones o limitaciones;
11. publicación dentro del attempt, no canónica o sobre un path existente.

El caso real positivo debe pasar read-only. Los tests de mutación operan sólo
sobre fixtures o copias temporales en `/mnt/m2-1TB`, nunca sobre el intento.

## 9. Presupuesto y autoridad científica

Toda validación es CPU-only, con `CUDA_VISIBLE_DEVICES=''`, medición de wall,
RSS y swaps. No se sustituye con CPU una operación propia de GPU: no hay
training ni forward y la tarea es hashing, validación estructural y
reagregación booleana.

Resolver la etiqueta falsa de replay no cambia ninguna métrica ni convierte
los patrones negativos en positivos. No se promueve arquitectura y no se
declara `GO/NO-GO`; la decisión permanece en Mariano.
