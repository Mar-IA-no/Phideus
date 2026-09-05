# Ola 59 — auditoría independiente final del freeze prospectivo

## Dictamen: REVISE

El commit `ebb71970fa35247306de3366aa86a37b227c0ece` congela correctamente
la config y pasa el preflight productivo completo: el status es
`FROZEN_PROSPECTIVE_PROTOCOL_PRE_KEY_DRAW`, el implementation binding apunta
exactamente a `ff9c6ed4d42119520169ba3922b036093afe4428` y R426, las 33 fuentes
requeridas están limpias en HEAD, los 32 hashes físicos no recursivos y el
autobinding canónico de la config coinciden, el linaje Git es directo, y los
bindings históricos/upstream son exactos. El preflight CPU terminó PASS con
tres seeds y 384 tokens históricamente exactos, CUDA vacío y cuatro hilos. Los
outputs primary/replay Wave 59 y su escrow siguen ausentes.

No obstante, el freeze no alcanza PASS porque una de esas fuentes requeridas,
`tests/test_wave59_prospective.py`, no fue rebasada al cambio de estado. La
ejecución del archivo congelado terminó `3 failed, 23 passed, 20 errors`. Dos
tests siguen suponiendo que la config canónica está pre-freeze; además, el
fixture físico y el restore canónico fabrican autoridad
`TEST_ONLY_PREPARED_BUNDLES`, que el runner ahora rechaza correctamente cuando
la config está frozen. Por tanto, la suite ligada por hash no valida el estado
que el freeze declara. Es un P1 prospectivo y bloquea escrow/draw hasta que la
suite se adapte, se actualicen su source hash y el autobinding de la config, y
la regresión vuelva a pasar en un nuevo commit auditable.

No se modificó ningún archivo de implementación o configuración, no se ejecutó
el preparador completo y no se creó escrow, draw, clave ni output canónico.

## Identidad y transición

- HEAD auditado: `ebb71970fa35247306de3366aa86a37b227c0ece`.
- Padre directo: `d5a45d05ff6b71290848210dadc04eb881bf8851`.
- Subject: `Freeze Wave 59 prospective protocol before draw`.
- Timestamp: `2026-09-05T03:32:36-03:00`.
- `d5a45d05ff6b71290848210dadc04eb881bf8851` desciende directamente de
  `ff9c6ed4d42119520169ba3922b036093afe4428` y sólo introduce R426.
- `ebb7197` desciende directamente de `d5a45d0` y sólo modifica la config
  prospectiva: `41` inserciones y `6` eliminaciones.
- `git diff --check d5a45d0..ebb7197` terminó con exit 0.
- R426 fue introducido exactamente en `d5a45d0`; contiene
  `## Dictamen: PASS` y nombra como auditado a `ff9c6ed...`.

El linaje observado es:

```text
ff9c6ed4d42119520169ba3922b036093afe4428
  -> d5a45d05ff6b71290848210dadc04eb881bf8851
  -> ebb71970fa35247306de3366aa86a37b227c0ece
```

## Contrato frozen y bindings

La config vigente declara el status frozen en la línea 3 y el binding aceptado
en las líneas 24–28. Los valores observados son:

```text
implementation_binding.status=ACCEPTED_IMPLEMENTATION_AUDIT
implementation_binding.commit=ff9c6ed4d42119520169ba3922b036093afe4428
implementation_binding.audit_path=Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/426_wave59_implementation_fifth_reaudit.md
implementation_binding.audit_sha256=387358ab81b536f5eb97cbb318946d559da60328f7b25efe3047d13c0f1c4426
```

El path de R426 aparece tanto en `required_execution_sources` como en
`source_sha256`; su hash físico coincide con el binding. El validator core
exige binding aceptado, commit/audit hash bien formados, cobertura exacta del
mapa de fuentes, presencia del audit entre las fuentes y autobinding canónico
de la config (`wave59_hgb_guard_bracket.py:271-302`). El preparador añade
fuentes tracked y clean-at-HEAD, igualdad del mapa, ancestralidad, hash físico
del audit y contenido PASS para el commit ligado
(`prepare_wave56_fresh.py:441-460,745-795`).

La verificación independiente produjo:

| Control | Resultado |
|---|---|
| `required_execution_sources` | 33 paths, 33 únicos |
| keys de `source_sha256` | igualdad exacta con los 33 paths |
| archivos físicos vs blobs HEAD | 33/33 exactos |
| hashes declarados no-config | 32/32 exactos |
| SHA físico de config | `e25750973e0dc7c5e68d2fc3f8779b41779770affc6796c2c7f246c7d9f056de` |
| autobinding canónico esperado/observado | `e63a9b65189083c190fe2b8fe619995add9febde2c1876aa14e992a3e906bdef` |
| R426 físico/binding | `387358ab81b536f5eb97cbb318946d559da60328f7b25efe3047d13c0f1c4426` |

La diferencia entre el SHA físico de la config y su entrada de source map es
intencional: `config_self_binding_sha256` normaliza únicamente su digest
recursivo a 64 ceros antes del hash (`wave59_hgb_guard_bracket.py:305-323`).

## Policy manifest, upstreams y preflight CPU

Se verificaron 20 hashes físicos ligados: public key; Wave 50 visible val,
authorized val y protocol; Wave 51 normalizer, split manifest y tres
checkpoints; Wave 52 policy manifest y tres arrays val-monitor; tres artefactos
Wave 54; dos bundles Wave 55; y dos artefactos Stage 0 Wave 56. No hubo
mismatches. El policy manifest observado fue:

```text
f8a608d396ad48ba3b0336df2dc1955940515be0f51fa08328cd5ddb1e9a21e1
```

El `generation_key_commitment` del manifest histórico Wave 50 también coincide
con el binding declarativo. El preparador comprueba nueve upstreams directamente
y el re-forward histórico cubre los restantes hashes de Wave 50/51/52
(`prepare_wave56_fresh.py:681-742,795-855`).

Se llamó solamente a `preparation_preflight`, no a la transacción de
preparación. Resultado:

```text
status=PASS
elapsed_seconds=1.3876
git_commit=ebb71970fa35247306de3366aa86a37b227c0ece
source_count=33
upstream_count=9
historical_status=PASS
historical_seed_count=3
historical_tokens=384
historical_all_exact=true
CUDA_VISIBLE_DEVICES=""
torch_threads=4
```

## Negativos quirúrgicos en memoria

Se clonó la config sólo en memoria; no se escribieron variantes. Los seis
casos fueron rechazados antes del re-forward histórico:

| Mutación | Rechazo observado |
|---|---|
| status vuelve a `IMPLEMENTATION_PRE_DRAW` | `config is not frozen for a pre-key draw` |
| binding status vuelve a pending | `accepted implementation audit is not bound` |
| commit ligado cambia a 40 ceros, con autobinding recalculado | `Git provenance is not ancestral` |
| audit SHA cambia a 64 ceros, con autobinding recalculado | `implementation audit hash drifted` |
| hash de `prepare_wave56_fresh.py` cambia a 64 ceros, con autobinding recalculado | `execution source binding drifted` |
| entrada de autobinding de config cambia a 64 ceros | `config self-binding drifted` |

Estos resultados confirman que el finding de tests no proviene de una debilidad
del freeze validator o del preflight.

## Finding bloqueante

### P1 — la suite requerida no representa ni pasa el estado frozen

`test_shared_preparer_dispatches_wave59_and_blocks_unfrozen_config` carga la
config canónica ya congelada y todavía espera que el validator la rechace como
“not frozen” (`tests/test_wave59_prospective.py:44-48`).
`test_frozen_status_alone_cannot_bypass_implementation_binding` vuelve a asignar
el mismo status frozen, pero deja intacto el binding aceptado y espera un
rechazo de implementation audit (`tests/test_wave59_prospective.py:51-57`). Los
dos fallan porque el sistema acepta correctamente la config vigente.

El fixture físico construye cinco bundles sueltos y llama directamente al
runner con la config frozen, sin crear una `preparation_freeze.json` autorizada
(`tests/test_wave59_prospective.py:93-118`). El runner lo rechaza con
`frozen Wave 59 execution lacks preparation_freeze.json`, como exige su frontera
prospectiva (`run_wave59_hgb_guard_bracket.py:2355-2398`). Ese único setup causa
20 errores dependientes.

Finalmente, el test de restore canónico crea expresamente un freeze
`TEST_ONLY_PREPARED_BUNDLES` (`tests/test_wave59_prospective.py:1120-1160`) y lo
usa contra la config frozen; el restore lo rechaza con
`preparation authority differs from frozen execution`. Éste es el tercer fallo
independiente.

Resultado completo del archivo requerido:

```text
FF.EEEEEEE.....EEEEEEEEEE...EEE...........F... [100%]
3 failed, 23 passed, 20 errors in 5.06s
```

No interpreto estos rechazos como defectos del runner: el runner está aplicando
la autoridad frozen que R426 aprobó. El defecto está en congelar una suite que
aún codifica el régimen pre-freeze. Como el propio test es fuente de ejecución
requerida y su hash está fijado, no corresponde iniciar escrow/draw con esta
revisión.

## Corrección mínima requerida

1. En los dos primeros tests, mutar copias en memoria a los estados inválidos
   antes de esperar rechazo; la config canónica frozen debe pasar.
2. Adaptar el fixture físico y el restore canónico para presentar una
   `preparation_freeze.json` sintética pero estructuralmente auténtica para la
   config frozen —incluidos config SHA, prospective config, 33 source hashes,
   source bindings y hashes de bundles— o aislar explícitamente el modo de test
   con una config temporal no-frozen. La primera opción tiene más poder
   diagnóstico sobre el camino que se va a ejecutar.
3. Repetir al menos `tests/test_wave59_hgb_guard_bracket.py` y
   `tests/test_wave59_prospective.py`, idealmente la regresión de cinco archivos
   de R426.
4. Actualizar el hash de `tests/test_wave59_prospective.py`, recalcular el
   autobinding canónico de la config y publicar un nuevo commit pre-draw.
5. Reauditar el nuevo HEAD antes de cualquier escrow o draw. El binding de
   implementación puede seguir apuntando a `ff9c6ed` si no cambia la
   implementación auditada y éste continúa siendo ancestro; el nuevo source map
   debe capturar exactamente la adaptación de tests.

## Comandos y resultados principales

```text
git show --no-patch --format=... HEAD
# ebb71970fa35247306de3366aa86a37b227c0ece
# parent d5a45d05ff6b71290848210dadc04eb881bf8851

git diff --check d5a45d05ff6b71290848210dadc04eb881bf8851..ebb71970fa35247306de3366aa86a37b227c0ece
# exit 0

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
BLIS_NUM_THREADS=4 venv/bin/python [validator/source-binding probe]
# validators PASS; 33/33 disk=HEAD; 32/32 non-config hashes; self-binding exact

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
BLIS_NUM_THREADS=4 venv/bin/python [preparation_preflight only]
# PASS; 3 seeds exact; 384 tokens; 4 torch threads; CUDA empty

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
BLIS_NUM_THREADS=4 venv/bin/python -m pytest -q \
  tests/test_wave59_prospective.py
# 3 failed, 23 passed, 20 errors in 5.06s
```

La selección inicial de tres tests hizo visible el problema con `2 failed,
1 passed`; luego se ejecutó el archivo completo para medir el alcance. El
directorio temporal propio retenido por pytest (`21M`) se retiró de forma
explícita y se confirmó nuevamente la ausencia de ambos outputs Wave 59.

## Hashes del corpus auditado

| Archivo | SHA-256 |
|---|---|
| Config frozen Wave 59, bytes físicos | `e25750973e0dc7c5e68d2fc3f8779b41779770affc6796c2c7f246c7d9f056de` |
| Config frozen Wave 59, autobinding canónico | `e63a9b65189083c190fe2b8fe619995add9febde2c1876aa14e992a3e906bdef` |
| Core/validator Wave 59 | `8f4daf8c545407b5b0e1c7a50af7350e00c16cf0817f7f81b66f9a6a046cd8a8` |
| Preparer/preflight compartido | `fb5345dd978a3bd2d658a8709f874e6ff3944f6f88e62f0330c9b7d9946c9778` |
| `test_wave59_hgb_guard_bracket.py` | `d837a8ae4948eec6a59c3a62c751935d2d90521a7d7178f7411006d002898f51` |
| `test_wave59_prospective.py` | `b34ac5648f49e99b728825bb5deb471cf24c1e66813ad1561c16379fabf7dfd0` |
| R426 | `387358ab81b536f5eb97cbb318946d559da60328f7b25efe3047d13c0f1c4426` |
| Plan final Wave 59 | `7e74f892bf27c4c51fa5f44e4e04b4564f7d63d986d8cc69c7316663bc5eabfb` |
| R420, auditoría de plan aceptada | `c7d1ed28554bb3b1174bfb787ae9469ee20392062f0c36deda7d0d10dcab0134` |

## Estado de no-draw al cierre

Los paths
`data/geometria_proporcional/wave59_fresh_hgb_guard_bracket_v1` y
`data/geometria_proporcional/wave59_fresh_hgb_guard_bracket_v1_replay`
permanecen ausentes. Al no existir esos árboles tampoco existe escrow Wave 59.
No se ejecutó generación ni se materializó truth nueva. La condición para
reanudar es un nuevo HEAD pre-draw con la suite rebasada, hashes/autobinding
actualizados y regresión limpia.
