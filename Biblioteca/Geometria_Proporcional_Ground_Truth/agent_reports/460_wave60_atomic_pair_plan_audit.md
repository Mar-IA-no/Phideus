```yaml
audit_id: R460
target_commit: 8a428d19ea3ef2b0d65fcbf62adb4a5a8447293f
expected_parent: aecc04901533513c1c6770cbf934f17051c77b7d
plan_sha256: f60bd73372f543f9ea7a8d0703ab495cca8d3491fd7d75a9638423cc6a4d8a20
technical_verdict: REVISE
findings:
  high: 1
  medium: 1
  low: 0
implementation_authorized: false
draw_authorized: false
gpu_used_or_queried: false
mendieta_used: false
web_used: false
secrets_or_truth_semantically_opened: false
files_modified: false
tests:
  command: CUDA_VISIBLE_DEVICES='' ... pytest -q tests/test_wave59_hgb_guard_bracket.py
  result: 11_passed
  duration_seconds: 1.72
```

## Dictamen

`REVISE`.

La tercera revisión cierra correctamente F10 y F12 de R459, y resuelve el núcleo atómico de F11 mediante una única publicación pair-level. También preserva F01–F05 de R457 y el DAG acíclico de F06–F08 de R458. Sin embargo, el cierre de abortos pre-truth sigue siendo incompleto: una falla asimétrica deja a la root peer exitosa sin ningún terminal ni inventario closed-world permitido. Esto contradice un invariante explícito del plan y obliga al implementador a inventar estado, binding y schema. La recuperación posterior a ese aborto tampoco tiene namespace de publicación declarado.

## Findings

### F13 — HIGH — Un aborto pre-truth asimétrico deja una root sin terminal ni inventario closed-world

El plan exige que toda root tenga manifest o inventario de fallo closed-world ([plan:464–471](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)) y define una barrera por la cual ambas roots deben completar `SOURCE` y `SCORE` antes de abrir truth ([plan:655–657](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)).

La matriz root-level sólo admite:

- fallos propios de preparación, identidad, score o evaluación;
- `EVALUATED_IMMUTABLE` con root seal.

No admite una root peer que haya completado correctamente `COMMON+SOURCE`, o `COMMON+SOURCE+SCORE`, cuando la otra falla antes de truth ([plan:659–665](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)).

Caso concreto:

1. primary completa y promueve `SCORE`;
2. replay falla en `score_apply`;
3. replay puede terminar `SCORE_APPLY_FAILED_PRE_TRUTH`;
4. primary queda en `COMMON+SOURCE+SCORE`, sin `artifact_manifest.json`, sin `failure_inventory.json` y sin terminal enumerado;
5. el par publica `PAIR_ABORTED_PRE_TRUTH`, pero el plan prohíbe reescribir las roots para reflejar el estado del par ([plan:667–682](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)).

Además, `pair_status.json` exige `primary_terminal`, `replay_terminal` y sus bindings ([plan:558–563](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)), pero para la peer anterior no existe terminal ni artefacto de binding definido. El manifest pair-level no satisface la afirmación independiente de que cada root tenga manifest o inventario propio.

Corrección mínima: incorporar un terminal root-level `PEER_ABORTED_PRE_TRUTH`, con presencia exacta según la última fase promovida —al menos después de `COMMON`, `SOURCE` y `SCORE`— y permitir que el coordinador añada únicamente su triple de fallo/inventario antes de construir el paquete pair-level. Alternativamente, retirar el invariante por-root y definir un snapshot pair-level exacto de roots parciales, con schema y binding físicos propios. Deben probarse fallos asimétricos en ambos órdenes y en preparación/score.

### F14 — MEDIUM — La recuperación pre-truth no tiene namespace de intento realizable

El plan permite recuperar un fallo pre-truth mediante amendment auditado sin redibujar ([plan:464–471](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)), pero publica el aborto atómicamente en la única root canónica pair-level ([plan:667–674](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)). Las raíces canónicas sólo enumeran un path pair-level ([plan:720–731](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)), mientras los workers rechazan outputs preexistentes ([plan:248–254](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)).

Después de publicar `PAIR_ABORTED_PRE_TRUTH`, una recuperación no puede:

- reemplazar ese paquete inmutable;
- publicar `COMPLETE` en el mismo path;
- elegir otro path sin inventar naming, lineage y binding.

Corrección mínima: elegir una política explícita. O bien cada intento/amendment usa roots versionadas e inmutables y un binding canónico aparte, o bien un aborto pre-truth termina definitivamente esos paths y la recuperación requiere una nueva versión de protocolo/config/output, aunque conserve el mismo draw. Debe congelarse también si los timestamps de `pair_status` y receipt son operacionales normalizables o parte de la igualdad exigida al reintento.

## Cierre de findings anteriores

- **F01 R457 — PASS:** `score_mask=disagreement` y `decision_mask=primary AND disagreement` permanecen separados. Probe físico: `2.339` posiciones de score, `1.055` de decisión y `1.284` disagreements no primarios.
- **F02 R457 — PASS:** guard opaco contra cinco roots físicas Wave 59, con rechazo de alias, symlink y hardlink antes de semántica.
- **F03 R457 — PASS:** worker Wave 60 dedicado, tres fases, UID/GID `65534`, capabilities vacías, `NoNewPrivs` y allowlists separadas.
- **F04 R457 — PASS:** aplicador transport-only cerrado de trece modelos; el aplicador completo Wave 59 queda sólo para equivalencia retrospectiva.
- **F05 R457 — PASS:** la inferencia está acotada al transporte de pipelines congeladas completas.
- **F06 R458 — PASS:** `outputs → freeze → receipt → journal → attestation` es acíclico; freezes no hashean receipts propios.
- **F07 R458 — PASS:** `analysis.json` permanece inmutable; `replay_exact` vive sólo en `final_analysis.json`.
- **F08 R458 — PASS:** `verify_source_law` es una autoridad única pre-draw, ligada a request, implementación y auditoría.
- **F09 R458 — REVISE parcial:** filenames, schemas de fallo y matrices fueron agregados, pero el hueco de peer abort de F13 mantiene incompleto el closed world.
- **F10 R459 — PASS:** existe `replay_finalize_receipt.json`, con keyset separada, y seis outputs pair-level acíclicos.
- **F11 R459 — REVISE parcial:** la doble promoción imposible fue sustituida correctamente por staging pair-level y un solo `rename`; persiste el aborto asimétrico de F13.
- **F12 R459 — PASS:** autoridad única antes del draw, allowlist exacta de `12` inputs y aliases inequívocos. Los nueve hashes exigidos coinciden físicamente `9/9`.

## Verificaciones positivas

- `HEAD`, parent y SHA-256 coinciden exactamente; el commit modifica sólo el plan y el worktree está limpio.
- Las cinco roots Wave 59 declaradas existen y tienen inodos distintos.
- Las cuatro roots Wave 60 aún no existen.
- Manifest fuente: `16` modelos exactos, `13` transportados y sólo `3` no usados.
- Estados HGB: `1.300/1.300` tree keys únicas; `1.300` arrays de nodos y `2.600` auxiliares categóricas vacías. Las otras `12` arrays corresponden únicamente a los tres modelos lineales no transportados.
- Los nueve hashes físicos, thresholds y artefactos fuente coinciden.
- El manifest Wave 59 liga físicamente también el inference bundle y el action freeze usados por la allowlist.
- Los cinco paths autorizados son suficientes: cuatro archivos nuevos y una rama acotada del preparador compartido.
- Presupuesto plausible: Wave 59 consumió `219,539 s` primaria+replay y menos de `0,76 GiB` RSS observado, frente a `900 s` y `1,5 GiB` por proceso en Wave 60.
- Test focal CPU con CUDA invisible: `11 passed in 1.72s`.

No corresponde comenzar implementación ni generar la autoridad source law/draw hasta cerrar F13 y F14.
