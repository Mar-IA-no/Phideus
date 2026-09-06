# R509 — Auditoría final de resultado/terminal Wave 60 v4

## Dictamen técnico: REVISE

La ejecución y sus resultados científicos son íntegros, pero el agregado
pair-level contiene un defecto de normalización operacional. Ambos roots están
cerrados como `EVALUATED_IMMUTABLE`, el pair es un terminal físico y firmado
`COMPLETE`, los artefactos científicos primary/replay son exactos y la
recomposición independiente de métricas, intervalos, soportes y condiciones
coincide con `analysis.json`. Sin embargo, `replay_comparison.json` registra
`MISMATCH` porque compara entre roots el SHA bruto de dos generation receipts
que ya había comparado correctamente por campos semánticos. Esa falsa
discrepancia se propaga como `replay_exact=false` a `final_analysis.json`.

Resultado por régimen de evidencia:

- **integridad de ejecución:** PASS;
- **validez numérica:** PASS;
- **validez del agregado final:** REVISE;
- **decisión científica:** permanece `null`; la autoridad sigue siendo el
  usuario.

Encontré **0 HIGH, 1 MEDIUM y 1 LOW**. El finding medio no invalida el draw, la
reproducibilidad científica observada, los patrones core ni el terminal de
ejecución. Sí impide usar sin corrección la etiqueta pair-level
`replay_exact=false` como descripción de la reproducibilidad.

## Findings

### R509-01 — MEDIUM — La normalización operacional reintroduce una identidad local no portable

`compare_evaluated_roots()` compara primero `generation_receipt.json` por seis
campos semánticos allowlisted; esa comparación da `true` en este resultado
(`run_wave60_frozen_policy_transport.py:3364-3387`). Luego compara
`preparation_receipt.json` incluyendo `generation_receipt_sha256`
(`run_wave60_frozen_policy_transport.py:3399-3404`). Este último valor es un
enlace bruto local: primary y replay validan cada uno su propio enlace, pero no
tienen por qué compartirlo.

La evidencia física es concluyente:

- `preparation_freeze.json` es byte-exacto entre roots, SHA-256
  `ed47e512452ee13689427bca97166c323233c2a8825d89d18cb9c91b73efe6f9`;
- los generation receipts coinciden en todos los campos allowlisted, pero
  declaran correctamente `execution_mode=recovery` y `execution_mode=replay`;
- por ello sus SHA-256 locales son, respectivamente,
  `847ec250397a459eeb25e4ca6bb10774551e40dbadceaeeac8cb27527c60be44` y
  `9088275ca41be5f82d14bb7c227eed36b443e080062786bc77dd8e38483e3e13`;
- cada `preparation_receipt.json` liga exactamente el receipt de su propio
  root; no hay enlace roto ni sustitución;
- las otras 35 comprobaciones publicadas son `true`: 3 JSON científicos, 5
  NPZ científicos, 6 estados funcionales, 12 secretos/inventario opacos y 9
  comprobaciones operacionales.

El comparador convierte cualquier check falso en `status=MISMATCH`
(`run_wave60_frozen_policy_transport.py:3453-3463`) y el finalizador convierte
ese status global en `replay_exact`
(`run_wave60_frozen_policy_transport.py:4795-4809`). Así, una diferencia de
identidad operacional deliberada queda
representada como falta de exactitud del replay completo. Esto contradice el
contrato del plan: validar los enlaces locales brutos dentro de cada root y
comparar entre roots los artefactos operacionales mediante igualdad semántica
allowlisted.

No es un fallo de reproducibilidad científica: los ocho hashes científicos,
los arrays tipados, los estados funcionales, los hashes secretos opacos y los
análisis completos son exactos. Tampoco explica por sí solo el resultado
negativo. Si se normaliza correctamente este único enlace y se evalúa
`replay_exact=true`, ambos patrones finales continúan `false` porque ya fallan
sus condiciones core contra los controles de desplazamiento máximo.

La corrección mínima futura debe conservar la validación local de cada
`generation_receipt_sha256`, pero comparar cross-root el resultado de la
normalización semántica ya calculada, no los dos SHA locales. La cobertura
también debe construir receipts de generación role-specific: el helper
end-to-end vigente copia el mismo `generation_receipt.json` a ambos roots, por
lo que su expectativa `EXACT` no reproduce este caso real. Dado que el pair ya
es post-truth e inmutable, cualquier republicación debe seguir una recuperación
pair-finalize auditada; no corresponde parchear estos artefactos in situ.

### R509-02 — LOW — `final_analysis.json` conserva una limitación de fase ya resuelta

El análisis root-level declara
`replay_exact_pending_pair_finalize`, correctamente antes de finalizar el
pair (`wave60_frozen_policy_transport.py:900-905`). El finalizador copia la
lista sin transición de estado (`run_wave60_frozen_policy_transport.py:4809`),
por lo que `final_analysis.json` sigue diciendo que el replay está pendiente
después de haber publicado comparación, freeze, receipt, attestation y
terminal `COMPLETE`. En este resultado la descripción correcta sería que la
comparación fue realizada y quedó afectada por R509-01. Es una inconsistencia
de representación, sin efecto sobre métricas ni decisión.

## Identidad, closed world y terminal

La autoridad Git auditada es `789c4ea2298fcaba97c9bdecdd1db4360186012c`,
hija directa de `b156f6857eaa36edc8bda9e7687de7b8e1ea9721`. La config física
de 11.599 bytes tiene SHA-256
`191483d2909c3a95a1e82488e1834b55c849763f55c72549f07f2c4cf81d6416`;
el validador integral de autoridad la aceptó contra ese HEAD.

Los bindings solicitados coinciden con el filesystem:

| Artefacto | SHA-256 |
|---|---|
| pair manifest | `4a51993c420a96f9f8283a3ced5e9dee279097ce37453fbd2b7480580805e686` |
| pair status | `98bcfcadf9476127cff6bcdb18dcf7f8ec36dd4897721f42c441bc51874ccffd` |
| final analysis | `9292c64e9d03a45fd59fdab39c58355abb08fbb97d7402e696cb50e0c580b21c` |
| replay comparison | `d5ac8ff07170d30f54f1ce440ae9f60aede6d8e315cd4d08674d393c655a2f29` |
| primary root manifest | `a96497d5a06e8ec23b7844aa13a2ef7455ef0a8bf6b410980a96ef8ebf7ed982` |
| replay root manifest | `4ffabc2bd54de623820cd373b84ad4f95e49ff45b7ddc48da8eaa894f7dc7eb0` |

El inventario contiene 65 archivos en primary, 66 en replay y 10 en pair: 141
archivos, 141 inodos únicos, `st_nlink=1` en todos, sin symlinks ni nodos
especiales. Los tres manifests son closed-world y reproducen exactamente
tamaño, SHA-256, owner, group, modo y clase de cada archivo. Los secretos y
truth preparados mantienen `0600`; los outputs científicos, journals y
terminales sellados están en `0444`; los archivos preparatorios públicos o de
inferencia conservan los modos declarados. Ningún secreto fue interpretado en
esta auditoría: su integridad y equivalencia se comprobaron por hashes opacos
y manifests.

Las attestations Ed25519 de preparación, source law, score, evaluación y pair
final verifican con fingerprint
`e5b8fcb229908503343f04c44dfed04646960de82454b411fbdc16b3c9525350`.
El freeze y el receipt pair-level ligan exactamente ambos root manifests,
`pair_status`, `replay_comparison`, `final_analysis`, journal y runtime. El
status declara dos roots `EVALUATED_IMMUTABLE`, `any_truth_accessed=true`,
`recovery_allowed=false` y terminal `COMPLETE`, coherente con los terminales
físicos. `COMPLETE` acredita cierre de la ejecución; no convierte una
comparación negativa ni una adjudicación en éxito científico.

## Barreras, procesos y source law

En ambos roots la secuencia firmada es:

```text
source_bind  truth_accessed=false
score_apply truth_accessed=false
evaluate    truth_accessed=true
```

La inferencia visible se ejecutó como UID/GID `65534:65534`, sin grupos
suplementarios, capabilities `0000000000000000` y `no_new_privileges=1`; el
probe de truth sellado fue denegado con `PermissionError`. Score y evaluate
también registran UID/GID `65534:65534`, capabilities cero, `no_new_privs=1` y
tres probes prohibidos denegados. El coordinador root aparece sólo en
preparación/finalización y el receipt pair-level; el journal final declara
`truth_accessed=false`. Los runtimes fijan `cuda_visible_devices=""` y cuatro
threads CPU.

La autoridad source-law v2, su manifest y su attestation verifican contra la
config. El estado transportado contiene 13 modelos usados, 1.300 árboles y
3.900 arrays con roster cerrado; frozen policy, feature schema y seis estados
funcionales son byte-exactos entre primary y replay. Score produce 13 arrays
raw, 14 acciones cerradas y la misma evaluación tipada en ambos roots.

## Provenance R506–R508 y draw preservado

La config y los receipts conservan las capas sin colapsarlas:

- implementación correctiva R506:
  `464ceb59b8b634e34d625fe9458201075cdc8e3f`, auditada en
  `fb5dfa64ebe1f8a2b73b5123c3453695678c962e`;
- amendment suplementario: `fd1270b084d4ebcbd6b4b675169665ce5f9c207b`,
  auditado por R507 en `1bfc0de3ebed7d66a8378e4ec24c38bed52aa37a`;
- config v4: `b156f6857eaa36edc8bda9e7687de7b8e1ea9721`, auditada por
  R508 en `789c4ea2298fcaba97c9bdecdd1db4360186012c`.

Los 17 bindings `preserved_draw_sha256` coinciden simultáneamente con el
escrow pre-truth de `attempt_v3`, primary v4 y replay v4. Incluyen
`pre_generation_freeze`, `generation_escrow`, manifest, commitments, visibles
y sellados. La falla v3 ligada por la config es
`560ab42f250e90b0b52f54e0db9fe4aced78c2df623d21b717ba56ff51889660`,
terminal `PAIR_ABORTED_PRE_TRUTH`, `any_truth_accessed=false` y
`recovery_allowed=true`. No hubo redraw.

## Presupuesto y journals

El ledger acumulativo recompone sin huecos:

| Componente | Segundos |
|---|---:|
| preparación acumulada al terminar primary | 355.472399 |
| preparación acumulada al terminar replay | 435.003199 |
| seis fases worker firmadas | 9.366660 |
| máximo coordinador observado | 10.810634 |
| durable (`preparación + max(worker, coordinador)`) | 445.813833 |
| antes de finalize | 446.138732 |
| finalize, 1 invocación | 0.361945 |
| total observado | 446.500676 / 900.000000 |

El máximo RSS registrado por la ejecución fue 960.319.488 bytes durante
preparación, por debajo de 1.610.612.736 bytes. Los journals de score/evaluate
registran máximos entre 731.070.464 y 740.466.688 bytes. Los artefactos no
preservan contador de swaps del experimento, por lo que no se infiere uno; las
dos comprobaciones de auditoría medidas abajo registraron cero swaps.

## Recomposición numérica independiente

Sin re-forward ni retraining, recompuse desde `analysis_arrays.npz`,
`bootstrap_indices.npz`, `evaluation_index.npz` y
`monitor_policy_arrays.npz` las 14 acciones, 56 arrays métricos, 301 tokens
primarios, los 5.000 índices PCG64 seed 6007, medias, percentiles 2,5/97,5,
soportes y 14 condiciones core. Todo coincide exactamente con los dos
`analysis.json`; ambos análisis también son byte-exactos, SHA-256
`f1378cb22f45e10580c18d3f0d5d12d8b6a4ed19d39cea6fd612bf0121bdb5e0`.

| Contraste contra HARD-SET | mean diff | IC95 |
|---|---:|---:|
| mean / accuracy | 0.008998 | [0.000138, 0.017996] |
| mean / compatible | 0.014396 | [0.007198, 0.022425] |
| mean / regret | -0.018411 | [-0.028344, -0.009159] |
| mean / worst regret | -0.022702 | [-0.044581, -0.002769] |
| tail / accuracy | 0.008721 | [0.001246, 0.016334] |
| tail / compatible | 0.010520 | [0.003876, 0.017996] |
| tail / regret | -0.014904 | [-0.023844, -0.006344] |
| tail / worst regret | -0.017996 | [-0.031838, -0.005260] |

Los soportes principales son 46 pair-tokens autorizados para incompatibility
y 35 para harm. Cada patrón falla una condición distinta de replay:

- incompatibility vs media de controles máximamente desplazados, regret:
  `mean=-0.000623`, IC95 `[-0.003128, 0.001548]`; el extremo superior no es
  menor que cero;
- harm vs media de controles máximamente desplazados, worst regret:
  `mean=-0.001661`, IC95 `[-0.014064, 0.009468]`; el extremo superior no es
  menor que cero.

Por ello los patrones core ya son `false/false`. El final registrado es también
`false/false`; al sustituir únicamente el falso negativo operacional por
`replay_exact=true`, continúa `false/false`. Esta observación se limita al draw
sintético nuevo y a la ley Wave 59 congelada, con intervalos condicionales sin
corrección de multiplicidad; no identifica por sí sola un efecto target ni
autoriza una decisión científica.

## Comprobaciones CPU-only

Todas las órdenes fijaron `CUDA_VISIBLE_DEVICES=''` y no usaron ni consultaron
GPU. Pytest deshabilitó plugins externos y cache, y usó un basetemp propio bajo
`/mnt/m2-1TB`, retirado después de comprobar ownership.

| Check | Resultado | wall | max RSS | swaps |
|---|---:|---:|---:|---:|
| checker independiente de autoridad, manifests, 141 hashes/inodos/modos, firmas, barriers, source law, draw, budget y recomposición numérica | PASS | 3.41 s | 830.856 KiB | 0 |
| pytest focal: patrones/finalize idempotente/presupuesto/status/NPZ tipado | 6 PASS, 168 deselected | 2.31 s | 841.652 KiB | 0 |

No se reejecutó el experimento, no se abrió semánticamente material secreto,
no se modificó código, config ni data y no se usaron recursos de otros
procesos.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R509",
  "scope": "RESULT_OR_TERMINAL",
  "target": {
    "attempt_path": "data/geometria_proporcional/wave60_frozen_policy_transport_attempt_v4",
    "pair_manifest_sha256": "4a51993c420a96f9f8283a3ced5e9dee279097ce37453fbd2b7480580805e686"
  },
  "technical_verdict": "REVISE",
  "findings": {
    "high": 0,
    "medium": 1,
    "low": 1
  },
  "files_modified": false,
  "gpu_used_or_queried": false
}
```
