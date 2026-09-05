```yaml
audit_id: R459
target_commit: eb41df23b55f086b0e2732b3ecb3bfa240925602
expected_parent: 1b2442545737724c382847caa130b6c9820e11a9
plan_sha256: 0796bb348f5d072f0ab47c1f549e39e2043d464265e89f233c2017ce34ba6050
technical_verdict: REVISE
findings:
  high: 2
  medium: 1
  low: 0
implementation_authorized: false
draw_authorized: false
gpu_used_or_queried: false
mendieta_used: false
web_used: false
secrets_or_truth_semantically_opened: false
files_modified: false
```

## Dictamen

`REVISE`.

La revisión resolvió el núcleo de F06 y F07: los freezes de las tres fases worker ya no hashean sus propios receipts y `analysis.json` permanece inmutable antes del replay. También avanzó sustantivamente sobre F08 y F09. No obstante, la nueva clausura presenta dos bloqueantes de realizabilidad: la attestation de `replay_finalize` exige un receipt inexistente y la operación sobre dos raíces carece de una transacción o terminal capaz de representar fallos asimétricos. Además, `SOURCE_LAW_VERIFIED` sigue contradiciendo su carácter pre-draw y no tiene acceso físico directo a tres de las nueve fuentes que afirma verificar.

## Resolución de R458

- **F06, parcialmente resuelto:** el DAG `outputs → freeze → receipt → attestation` es acíclico para `verify_source_law`, `score_apply` y `evaluate` ([plan:250–256](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)). Persiste un contrato imposible en `replay_finalize`, detallado como F10.
- **F07, núcleo resuelto:** `analysis.json` contiene sólo 14 condiciones intradraw, queda congelado y `replay_exact` aparece únicamente en `final_analysis.json` ([plan:283–287, 438–457, 704–713](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)). La publicación conjunta entre raíces sigue abierta, detallada como F11.
- **F08, parcialmente resuelto:** ahora existe un productor durable, `verify_source_law`, con freeze, receipt y attestation. Su orden y cobertura física aún son contradictorios, detallados como F12.
- **F09, parcialmente resuelto:** se agregaron schemas de failure, siete terminales y una matriz de presencia. Falta cerrar estados asimétricos entre primaria/replay; forma parte de F11.

## Preservación de R457

- **F01 preservado:** `score_mask=disagreement` y `decision_mask=primary AND disagreement` siguen separados ([plan:291–309](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)).
- **F02 preservado:** continúa el guard opaco contra cinco raíces Wave 59, con rechazo de identidad, symlink, hardlink y alias antes de acceso semántico ([plan:171–206](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)).
- **F03 preservado:** worker Wave 60 autorizado, tres invocaciones UID/GID `65534`, capabilities vacías, `NoNewPrivs` y allowlists separadas ([plan:208–248](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)).
- **F04 preservado:** aplicador transport-only cerrado de trece modelos; el aplicador completo Wave 59 queda limitado a regresión retrospectiva ([plan:311–315](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)).
- **F05 preservado:** la inferencia continúa acotada al transporte comparativo de pipelines completas, sin atribución causal específica al target ([plan:59–75, 390–402](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)).

## Findings

### F10 — HIGH — `replay_finalize_attestation` depende de un receipt inexistente

La tabla congela exactamente cuatro outputs de replay:

```text
replay_comparison.json
final_analysis.json
replay_finalize_freeze.json
replay_finalize_attestation.json
```

No existe `replay_finalize_receipt.json` ([plan:438–457](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)). El propio plan confirma que hay sólo tres receipts, correspondientes a las fases worker ([plan:502–504](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)).

Sin embargo, inmediatamente dispone que **cada** attestation de fase liga hashes de `freeze`, `receipt` y `journal`, y luego particulariza esa misma attestation de replay finalize ([plan:512–516](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)). No hay valor válido para su `receipt` sin inventar un quinto output o violar la keyset/conteo congelado.

Corrección mínima, eligiendo una sola variante:

1. Declarar que el contrato común aplica sólo a las tres attestations worker y congelar un payload distinto para replay finalize, sin receipt; o
2. agregar `replay_finalize_receipt.json`, actualizar a cinco outputs, definir su keyset y preservar el orden `comparison → final_analysis → freeze → receipt → journal → attestation`.

### F11 — HIGH — No existe commit atómico ni terminal closed-world para el par primaria/replay

`replay_finalize` debe escribir sus cuatro outputs una sola vez en **cada** paquete y sólo entonces habilitar `COMPLETE` ([plan:452–457](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)). Los outputs parciales deben permanecer únicamente en temporales y los cuatro outputs REPLAY deben estar todos presentes o todos ausentes ([plan:561–566, 594–597](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)).

Dos raíces ya existentes no pueden promoverse mediante un único `rename` atómico. Un crash después de promover el directorio replay de la primera raíz y antes de promover el de la segunda deja un estado que ninguna fila de la matriz admite. El problema no puede corregirse reintentando bajo el contrato actual porque todo fallo post-truth exige protocolo y draw nuevos ([plan:599–604](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)).

La matriz tampoco cubre el caso anterior al truth: si una raíz termina `SOURCE_LAW_INVALID` o `SCORE_APPLY_FAILED_PRE_TRUTH`, su par puede haber quedado en `PREPARED`, `SOURCE_LAW_VERIFIED` o `LOCKBOX_ACTIONS_FROZEN`; ninguno es terminal y no existe `PEER_ABORTED_PRE_TRUTH`. La regla especial sólo cubre al par que ya abrió truth ([plan:603–605](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)).

Corrección mínima:

- introducir un artefacto pair-level con staging y commit marker único, dejando las raíces como consumidores de ese compromiso; o
- declarar una publicación recuperable e idempotente, bajo el mismo commit y exclusivamente sobre inputs inmutables, con estado explícito de finalize incompleto;
- agregar terminal `PEER_ABORTED_PRE_TRUTH` —o una regla determinista equivalente— con matriz exacta;
- probar crash después de la primera promoción y fallo pre-truth de cualquiera de las dos raíces.

### F12 — MEDIUM — `SOURCE_LAW_VERIFIED` aún no cumple su propio contrato pre-draw y físico

La sección 4 exige demostrar portabilidad **antes de admitir un draw nuevo** ([plan:132–154](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)), y la batería “antes del draw” exige que `verify_source_law` produzca freeze, receipt y attestation durables ([plan:641–659](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)). La máquina de estados, en cambio, exige:

```text
PREPARED -> SOURCE_LAW_VERIFIED
```

([plan:266–277](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)).

Además, la allowlist física de `verify_source_law` proporciona directamente sólo seis de las nueve fuentes declaradas: fit freeze, manifest de estados, arrays, calibration freeze, monitor scores y monitor policy arrays. Omite:

- `artifact_manifest.json` de Wave 59;
- `config.snapshot.json` de Wave 59 —el nombre `config.snapshot.json` de la fase corresponde naturalmente a la config Wave 60 y es ambiguo—;
- auditoría R454.

Por ello, el worker puede reproducir la ley, pero no verificar físicamente por sí mismo los nueve hashes que su freeze/attestation afirma ligar ([plan:99–111, 221–228, 438–440](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)).

Corrección mínima:

- mover `SOURCE_LAW_VERIFIED` antes de `PREPARED`;
- producir una única autoridad pre-draw reutilizada por ambas raíces;
- añadir aliases inequívocos para las tres fuentes faltantes, o declarar y congelar un preflight del coordinador que las verifica físicamente antes de firmar y que el worker liga sin afirmar haberlas abierto.

## Verificaciones que pasan

- `HEAD`, parent y SHA-256 coinciden exactamente con el encargo.
- El commit modifica únicamente el plan; worktree limpio.
- Los nueve hashes fuente coinciden físicamente.
- Conteo estructural CPU:
  - 13 modelos transportados;
  - 1.300 árboles;
  - 3.900 arrays físicas de estado, de las cuales 2.600 son bitsets auxiliares vacíos; el scorer portable necesita las 1.300 arrays de nodos, por lo que la proyección declarada es realizable;
  - selección fuente exacta: 26 arrays derivadas = 1 proposal + 12 authorizations + 13 actions, más hard;
  - output nuevo coherente: 14 actions = hard + proposer-only + 2 principales + 10 controles.
- Los cinco paths autorizados bastan para la implementación propuesta: módulo, runner, worker, rama del preparer y tests. La config queda correctamente como artefacto posterior, fuera del commit de implementación.
- La topología sandbox puede reutilizar el patrón existente que copia runtime e inputs allowlisted antes de bajar a UID/GID `65534`.
- Prueba focal CPU con CUDA oculta: `11 passed in 1.35s`.
- Presupuesto de `900 s`, cuatro threads y `1,5 GiB` por proceso continúa siendo plausible.

No corresponde implementar ni generar el draw con esta versión.
