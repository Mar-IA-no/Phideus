# Ola 59 — reauditoría independiente del protocolo congelado

## Dictamen: PASS

El HEAD `46aff20f7450b2059687b235a3de28d65cf47bb5` resuelve el P1 de R427 sin
relajar el runner aprobado. Los dos primeros tests representan ahora la config
canónica frozen; el fixture físico y el restore canónico materializan una
autoridad de preparación completa para esa misma config; y la regresión ligada
terminó exactamente `222 passed` en CPU. No encontré findings P0, P1 ni P2.

La config conserva el binding de implementación aceptado a
`ff9c6ed4d42119520169ba3922b036093afe4428` y R426. Sus 33 fuentes requeridas
son blobs limpios de HEAD, los 32 hashes físicos no recursivos y el autobinding
canónico son exactos, y los 20 hashes upstream más el commitment histórico
coinciden. El preflight productivo, ejecutado sin iniciar la preparación, pasó
con tres seeds y 384 tokens históricamente exactos, CUDA vacío y cuatro hilos.

Los outputs primary/replay y cualquier escrow Wave 59 canónico permanecieron
ausentes antes y después de la auditoría. No se ejecutó preparación, generación,
escrow ni draw. Este PASS cierra el defecto técnico de R427; no constituye por sí
mismo una decisión científica GO/NO-GO ni ejecuta el siguiente paso operativo.

## Identidad, linaje y alcance de la corrección

- HEAD auditado: `46aff20f7450b2059687b235a3de28d65cf47bb5`.
- Padre directo: `580fcec90cd9f8e1d11460605cb01d93cf29e6d0`.
- Subject: `Represent canonical Wave 59 preparation in tests`.
- Timestamp: `2026-09-05T03:50:07-03:00`.
- R427 quedó archivado en `bf49908ae810854f717653aeb815d4cc457ae360`.
- `git diff --check bf49908..HEAD` terminó con exit 0.
- El worktree estaba limpio al inicio y antes de escribir este informe.

El linaje directo observado es:

```text
ff9c6ed4d42119520169ba3922b036093afe4428  implementación aceptada por R426
  -> d5a45d05ff6b71290848210dadc04eb881bf8851  incorpora R426
  -> ebb71970fa35247306de3366aa86a37b227c0ece  freeze inicial
  -> bf49908ae810854f717653aeb815d4cc457ae360  incorpora R427
  -> cf4c9679e3c5b96ade2479e308a30c3d6ff23d2c
  -> 2fe4b2e285319e4774978f188e951f6678bdd19b
  -> 580fcec90cd9f8e1d11460605cb01d93cf29e6d0
  -> 46aff20f7450b2059687b235a3de28d65cf47bb5
```

Los cuatro commits posteriores a R427 modifican exclusivamente la config y
`tests/test_wave59_prospective.py`; no cambian preparer, core, worker ni runner.
Cada commit actualiza los dos hashes recursivamente acoplados. El diff agregado
desde R427 es de 102 inserciones y 71 eliminaciones en esos dos archivos.

## Resolución del P1 de R427

### Los dos tests de entrada representan el estado frozen

`test_shared_preparer_dispatches_wave59_and_accepts_frozen_config` valida
primero la config canónica frozen y recién después muta una copia en memoria a
`IMPLEMENTATION_PRE_DRAW` para comprobar el rechazo (`test_wave59_prospective.py:44-50`).
El segundo test reemplaza explícitamente el implementation binding por el
estado pending antes de exigir el rechazo de ambos validators
(`test_wave59_prospective.py:53-64`). De este modo, ninguno depende ya del estado
pre-freeze que causó los dos primeros fallos de R427.

### El fixture físico presenta autoridad frozen completa

El helper `write_frozen_test_preparation_authority` (`test_wave59_prospective.py:100-173`)
construye, a partir de la config canónica vigente:

- SHA físico de config y copia íntegra de `prospective_config`;
- hashes físicos de las 33 fuentes requeridas;
- `source_bindings` completo;
- hashes de los cinco bundles bajo `prepared/`;
- journal `prepare` con estado `PREPARED`, modo `fresh` o `replay`, hash del
  freeze y los mismos cinco hashes de bundles;
- inventario completo de artefactos de fase 0 requerido por el closed world.

El fixture físico ubica los cinco bundles bajo `root/prepared`, escribe esa
autoridad y llama al runner sobre el root canónico
(`test_wave59_prospective.py:176-202`). El replay conserva la misma topología y
declara `execution_mode=replay`. La inspección del fixture realmente
materializado durante pytest confirmó `33/33` hashes físicos, `5/5` hashes de
bundles, config completa y source bindings exactos, journal `PREPARED/fresh` y
hash exacto de `preparation_freeze.json`.

### El restore canónico usa la misma autoridad

`test_canonical_complete_restore_rebuilds_closed_manifest` invoca el mismo
helper antes de construir los journals de fases (`test_wave59_prospective.py:1197-1217`).
El árbol materializado conservó schema/config frozen, 33 fuentes y cinco
bundles. El test de restore pasó dentro de la regresión completa.

El runner no fue modificado por estos commits. Continúa rechazando una ejecución
frozen sin `preparation_freeze.json` y cotejando bundle hashes, SHA físico de
config, config completa, las 33 fuentes físicas y todos los source bindings
(`run_wave59_hgb_guard_bracket.py:2355-2398`). `_execute_once` sigue validando la
config y sus execution bindings antes de aceptar la autoridad
(`run_wave59_hgb_guard_bracket.py:2421-2489`). Por ello la corrección prueba la
frontera existente, no la elude.

## Freeze, implementation binding y fuentes

La config declara `FROZEN_PROSPECTIVE_PROTOCOL_PRE_KEY_DRAW` y mantiene:

```text
implementation_binding.status=ACCEPTED_IMPLEMENTATION_AUDIT
implementation_binding.commit=ff9c6ed4d42119520169ba3922b036093afe4428
implementation_binding.audit_path=Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/426_wave59_implementation_fifth_reaudit.md
implementation_binding.audit_sha256=387358ab81b536f5eb97cbb318946d559da60328f7b25efe3047d13c0f1c4426
```

R426 conserva `## Dictamen: PASS`, nombra exactamente el commit ligado, su hash
físico coincide y `ff9c6ed` es ancestro de HEAD. La validación independiente dio:

| Control | Resultado |
|---|---|
| `required_execution_sources` | 33 paths, 33 únicos |
| keys de `source_sha256` | igualdad exacta con los 33 paths |
| archivo físico vs blob HEAD | 33/33 exactos |
| hashes declarados no-config | 32/32 exactos |
| SHA físico de config | `3c2dd85541210b41c0a4cfa5846b88ba982a2df5b43a03da19b757250c9610ff` |
| autobinding canónico esperado/observado | `4334e7947b85f8c4a2bd95a854e4a659ef053f6db2a62dc34e2bc73802ec60de` |
| R426 físico/binding | `387358ab81b536f5eb97cbb318946d559da60328f7b25efe3047d13c0f1c4426` |

La diferencia entre el SHA físico y el autobinding de la config es deliberada:
el digest recursivo propio se normaliza a 64 ceros para calcular el segundo.

## Upstreams y preflight CPU

Se recalcularon los 20 hashes upstream ligados por la config: public key; tres
artefactos Wave 50; normalizer, split manifest y tres checkpoints Wave 51;
policy manifest y tres arrays val-monitor Wave 52; tres artefactos Wave 54; dos
bundles Wave 55; y dos artefactos Stage 0 Wave 56. Los 20/20 coincidieron. El
policy manifest observado fue
`f8a608d396ad48ba3b0336df2dc1955940515be0f51fa08328cd5ddb1e9a21e1`.
El `generation_key_commitment` del manifest Wave 50 coincidió con
`967ca4d8d6ecc93f019ee06efa65660f69466b32f38e2606d49c40ada0324014`.

Se llamó sólo a `preparation_preflight`, nunca a la transacción de preparación:

```text
status=PASS
elapsed_seconds=1.2931
git_commit=46aff20f7450b2059687b235a3de28d65cf47bb5
source_count=33
upstream_count=9
historical_seed_count=3
historical_tokens=384
historical_all_exact=true
CUDA_VISIBLE_DEVICES=""
torch_threads=4
output_exists_after_preflight=false
```

## Regresión congelada

Con CUDA oculto y todos los límites de thread configurados en 4 se ejecutó:

```text
venv/bin/python -m pytest -q \
  tests/test_wave59_hgb_guard_bracket.py \
  tests/test_wave59_prospective.py \
  tests/test_wave56_preoracle_recovery.py \
  tests/test_wave56_prospective.py \
  tests/test_wave57_prospective.py

222 passed in 278.47s (0:04:38)
```

El árbol temporal exclusivo de la auditoría ocupó 1.4 GB. Se inspeccionaron allí
la autoridad física y la restaurada; después se retiró el path exacto
`/mnt/m2-1TB/r428-pytest` y se confirmó su ausencia.

## Estado pre-draw

Se comprobaron explícitamente como ausentes:

```text
data/geometria_proporcional/wave59_fresh_hgb_guard_bracket_v1
data/geometria_proporcional/wave59_fresh_hgb_guard_bracket_v1_replay
```

Una búsqueda acotada bajo `data/geometria_proporcional` tampoco encontró
escrow, secretos ni pre-generation freeze asociados a Wave 59. Los archivos
efímeros de esos tipos que exige el closed-world test sólo existieron dentro del
árbol pytest propio ya eliminado.

## Comandos principales

```text
git log --format='%H %P %s' -6 HEAD
git diff --check bf49908ae810854f717653aeb815d4cc457ae360..HEAD

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
BLIS_NUM_THREADS=4 venv/bin/python [validators + 33-source + upstream probe]
# PASS; 33/33 disk=HEAD; 32/32 non-config; self-binding; 20/20 upstream

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
BLIS_NUM_THREADS=4 venv/bin/python [preparation_preflight only]
# PASS; 3 seeds; 384 tokens; CUDA empty; torch threads 4; no output

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
BLIS_NUM_THREADS=4 venv/bin/python -m pytest -q [five frozen files]
# 222 passed in 278.47s
```

## Hashes del corpus auditado

| Archivo | SHA-256 |
|---|---|
| Config Wave 59, bytes físicos | `3c2dd85541210b41c0a4cfa5846b88ba982a2df5b43a03da19b757250c9610ff` |
| Config Wave 59, autobinding canónico | `4334e7947b85f8c4a2bd95a854e4a659ef053f6db2a62dc34e2bc73802ec60de` |
| Core/validator Wave 59 | `8f4daf8c545407b5b0e1c7a50af7350e00c16cf0817f7f81b66f9a6a046cd8a8` |
| Preparer/preflight compartido | `fb5345dd978a3bd2d658a8709f874e6ff3944f6f88e62f0330c9b7d9946c9778` |
| Runner Wave 59 | `d78a414de79eea4ae51b98913de37b0c83a579b509574386f71d5060e5b9b816` |
| `test_wave59_hgb_guard_bracket.py` | `d837a8ae4948eec6a59c3a62c751935d2d90521a7d7178f7411006d002898f51` |
| `test_wave59_prospective.py` | `8121b00a8e2e87785ab9fb17859e902f6f336f32a7179bb3517509b00bff45e4` |
| R420 | `c7d1ed28554bb3b1174bfb787ae9469ee20392062f0c36deda7d0d10dcab0134` |
| R426 | `387358ab81b536f5eb97cbb318946d559da60328f7b25efe3047d13c0f1c4426` |
| R427 | `be9adaab4e2e07999e3c093ff6bbf27f706e718136ef3bdeef05c391d85acc80` |
| Plan final Wave 59 | `7e74f892bf27c4c51fa5f44e4e04b4564f7d63d986d8cc69c7316663bc5eabfb` |

## Findings

No hay findings P0, P1 ni P2. El P1 de R427 queda corregido y verificado en el
HEAD auditado.
