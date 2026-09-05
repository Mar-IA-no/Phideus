# R457 — auditoría independiente del plan Wave 60

```yaml
audit_id: R457
target_commit: a51b7fa590ff99400152ab6b5dcc47bdf83ebac5
expected_parent: 807b5059bf1c20dab1c4526330a7564d9309c4ef
plan_sha256: 88bcc9f103fa2adc447a0f51969d9a752c147f83876aa0e9147b1ac84e369ccf
technical_verdict: REVISE
findings:
  high: 3
  medium: 2
  low: 0
implementation_authorized: false
draw_authorized: false
gpu_used_or_queried: false
mendieta_used: false
files_modified: false
```

## Dictamen

`REVISE`.

La pregunta científica es válida y Wave 60 sí cambia el estimando respecto de Wave 59: transporta modelos y umbrales congelados a otra realización, sin refit ni recalibración. Los thresholds, patrones, bootstrap y presupuesto son coherentes con Wave 59. Sin embargo, el plan todavía no permite una implementación inequívoca y fail-closed: contiene una contradicción verificable sobre la máscara de scores, no exige probar la independencia del nuevo draw y deja indeterminada la topología física que protege el lockbox.

## Findings

### F01 — HIGH — Contradicción entre reproducción exacta y máscara de scores

El plan exige simultáneamente:

- reproducir bit a bit los scores preservados de Wave 59, admitiendo NaN sólo fuera de la máscara activa ([plan:124–138](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md));
- definir la máscara activa como `primary AND disagreement` y exigir NaN fuera de ella ([plan:181–192](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)).

Pero el scorer congelado de Wave 59 llena scores sobre todo `disagreement`, sin intersectarlo con `primary` ([wave58_open_diagnostic.py:392](/mnt/m2-1TB/Phideus/src/geometria_proporcional/wave58_open_diagnostic.py:392)). En el monitor físico hay, por cada modelo, `2.339` scores finitos: `1.055` en `primary AND disagreement` y otros `1.284` en disagreements no primarios. Los 13 modelos reproducen exactamente ese patrón.

No se pueden satisfacer ambos contratos como están escritos.

Corrección mínima: distinguir explícitamente:

```text
score_mask = disagreement
decision_mask = primary AND disagreement
```

Los scores raw/source-law deben reproducir Wave 59 sobre `score_mask`. Proposals, authorizations, actions, soporte y métricas usan `decision_mask`. Si se desea publicar un score masked adicional, debe tener otro nombre y derivación hasheada.

### F02 — HIGH — La independencia del draw nuevo se afirma, pero no se verifica

La independencia entre realizaciones es constitutiva del estimando ([plan:23–26](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md), [plan:143–156](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)). Sin embargo, la batería obligatoria no exige que escrow, compromisos, semantic root o contenido del benchmark Wave 60 sean distintos de Wave 59. Preservar los hashes anteriores, como dispone [plan:377–385](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md), no detecta que una rama del preparer reutilice o copie accidentalmente el draw fuente.

Ese fallo podría pasar conteos, replay, patrones y hashes internos y ser presentado incorrectamente como transporte.

Corrección mínima: antes de `PREPARED`, verificar mediante hashes opacos, sin abrir secretos, que:

- las tres key commitments y el escrow Wave 60 difieren de las raíces válidas y fallidas Wave 59;
- semantic root, commitments del benchmark y bundles lockbox difieren de Wave 59;
- los paths resueltos no son aliases, symlinks ni hardlinks de raíces anteriores;
- primaria y replay Wave 60 sí comparten exactamente el nuevo escrow, como corresponde al replay.

Una colisión o identidad debe abortar antes de acceso semántico con terminal de intento inválido, no producir `NOT_EVALUABLE`.

### F03 — HIGH — La frontera física del lockbox carece de worker/topología autorizada

El plan requiere worker allowlisted, ejecución como UID/GID restringido y congelar acciones antes de autorizar truth ([plan:158–179](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md), [plan:359–366](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)). Pero la implementación autorizada sólo enumera módulo, runner, preparer y test ([plan:306–321](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)); no dice cuál de esos procesos es el worker sin privilegios ni cuáles son sus inventarios de entrada exactos.

El worker existente de Wave 59 no es reutilizable directamente: sólo reconoce sus `PHASE_FILES`, rechaza fases desconocidas y valida la config Wave 59 ([`_wave59_phase_worker.py:96`](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/_wave59_phase_worker.py:96), [`_wave59_phase_worker.py:851`](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/_wave59_phase_worker.py:851)). Tampoco está autorizado editarlo.

Corrección mínima: declarar una de estas dos topologías antes de implementar:

1. Autorizar un `_wave60_phase_worker.py`; o
2. especificar que el runner Wave 60 es dual-mode y se reinvoca como worker UID/GID `65534`.

Congelar dos allowlists cerradas:

- `score/apply`: config, bindings, 13 estados, arrays y lockbox inference-safe; sin truth ni train/validation nuevos;
- `evaluate`: truth lockbox, action freeze y actions hasheadas; sin capacidad de alterar/recalcular scores o acciones.

Cada worker debe probar denegación física de los paths prohibidos y emitir receipt con UID/GID, capabilities, `NoNewPrivs`, inventario y hashes.

### F04 — MEDIUM — El aplicador Wave 59 no admite el roster reducido de 13 modelos

El plan declara exactamente 13 modelos transportados y tres estados no usados ([plan:103–109](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md), [plan:124–134](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)), exige rechazar políticas que usen modelos adicionales ([plan:340–348](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)) y propone reutilizar la aplicación Wave 59 ([plan:317–321](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)).

Pero `apply_calibrated_policies()` itera todos los proposers y policies presentes y consume incondicionalmente `legacy`, Ridge y guards logistic ([wave59_hgb_guard_bracket.py:935](/mnt/m2-1TB/Phideus/src/geometria_proporcional/wave59_hgb_guard_bracket.py:935), [wave59_hgb_guard_bracket.py:945](/mnt/m2-1TB/Phideus/src/geometria_proporcional/wave59_hgb_guard_bracket.py:945), [wave59_hgb_guard_bracket.py:954](/mnt/m2-1TB/Phideus/src/geometria_proporcional/wave59_hgb_guard_bracket.py:954)). Una calibración reducida a los 13 modelos falla con `KeyError: legacy`.

Corrección mínima: usar el aplicador completo únicamente para verificar retrospectivamente Wave 59 con los 16 estados. Para Wave 60, definir en el módulo nuevo un aplicador allowlisted que acepte exactamente:

- un proposer HGB;
- dos políticas principales;
- diez controles;
- `HARD-SET` y `HGB-PROPOSER-ONLY`.

Debe rechazar claves extra y demostrar equivalencia exacta con las 26 arrays seleccionadas del monitor Wave 59.

### F05 — MEDIUM — El alcance causal está formulado más fuerte que el contraste

El plan llama “estimando causal de control” a main menos promedio de cinco controles ([plan:45–65](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)). Pero main y controles difieren conjuntamente en target de entrenamiento, estado ajustado y threshold calibrado individualmente ([plan:111–120](Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md)); además se prohíbe igualar soporte en el draw nuevo.

Por tanto, el contraste identifica transporte diferencial de la pipeline congelada completa, no separa causalmente “target aprendido/localización contextual” de cobertura o calibración transportada. Los diagnósticos de soporte y overlap describen esa mediación, pero no la identifican.

Corrección mínima: renombrar el estimando como `comparative frozen-pipeline transport contrast` y acotar la inferencia. Si se quiere adjudicar específicamente target/localización, hace falta predeclarar antes del draw un factorial adicional de score/threshold o una intervención de cobertura; no debe agregarse post hoc.

## Aspectos que pasan

- Commit exacto `a51b7fa`, parent directo `807b505`, un único path agregado y worktree limpio.
- SHA del plan exacto: `88bcc9f...e369ccf`.
- Los nueve hashes fuente declarados coinciden físicamente.
- Los tres thresholds principales y los diez thresholds individuales coinciden con `calibration_freeze.json`.
- Manifest: `16` estados exactos; `13` requeridos y únicamente los tres extras declarados.
- `1.300/1.300` tree keys requeridas únicas y presentes; cero nodos categóricos y cero bitsets categóricos no vacíos.
- Reproducción CPU inference-safe:
  - scores: `13/13` exactos;
  - proposal/authorization/action seleccionadas: `26/26` exactas.
- Los thresholds y operadores `>`, `<`, `>=`, `<=` de ambos patrones coinciden con Wave 59.
- Bootstrap `PCG64`, orden lexicográfico, `5.000` réplicas y unidad `pair_token` son adecuados.
- La salida ternaria para falta de soporte es conceptualmente correcta.
- El presupuesto CPU de `900 s` y `1,5 GiB` por proceso es plausible frente a R454: `219,540 s` combinados y pico observado de preparación cercano a `1,112 GiB`.
- Suite focal vigente: `11 passed in 1.18s`, CUDA invisible.
- Wave 60 no contamina por diseño los artefactos Wave 56–59 si se conservan las regresiones y hashes exigidos; el riesgo restante está en la frontera compartida del preparer, cubierto una vez resueltos F02–F04.

## Corrección mínima consolidada

1. Separar `score_mask` de `decision_mask`.
2. Añadir guards explícitos de no identidad del draw.
3. Congelar la topología física y los allowlists de los dos workers.
4. Definir un aplicador transport-only estricto de 13 modelos.
5. Enumerar antes de implementar los outputs por fase, schemas, clases del manifest y terminales de failure/replay.
6. Acotar la atribución al transporte de la pipeline congelada completa.

Tras esos cambios corresponde una reauditoría focal del plan. No debe comenzar implementación ni generarse escrow Wave 60 con la versión actual.
