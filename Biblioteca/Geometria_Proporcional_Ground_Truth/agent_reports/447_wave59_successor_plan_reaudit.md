---
report_schema: wave59-successor-plan-reaudit-v1
audited_commit: f00860d3c34e16234f16b6d3a8c46bd6151d1e65
audited_sha256: 0c3378581db0a86263ade0b68b43e30398c0a6412085975536e94c40bd055baf
expected_parent: 8c41e3cfeda640e68ef7c8aaa5d91ded8894f2b0
parent_report_sha256: 874fc5e300202ea83f9e51b6e096ee8562d5544bc27ad678f5027bb6bd5d6dfb
dictamen: REVISE
severity_counts:
  BLOCKER: 1
  HIGH: 0
  MEDIUM: 0
  LOW: 0
worktree_clean: true
gpu_used_or_queried: false
web_used: false
mendieta_used_or_queried: false
recovery_or_draw_executed: false
---

# Reauditoría independiente — plan sucesor Wave 59

## Dictamen: REVISE

La revisión cierra cinco de los seis puntos materiales solicitados, pero el delta mecánico entre la config original y la sucesora conserva una contradicción determinista. El implementation commit debe modificar cinco archivos; cuatro de ellos ya tienen entradas en `source_sha256`. Esos cuatro hashes necesariamente cambiarán, pero la allowlist del plan no autoriza su reemplazo y luego exige igualdad canónica de todo campo no enumerado.

Por tanto, una config con hashes actualizados violaría el validator de delta; una config que retuviera los hashes originales fallaría el preflight de execution sources. El protocolo no puede congelarse de forma simultáneamente ejecutable y conforme al delta declarado.

No corresponde ejecutar implementación ni draw hasta corregir esta especificación. Este dictamen no constituye una decisión `GO/NO-GO`.

## Finding

### BLOCKER — El delta cerrado omite cuatro reemplazos obligatorios en `source_sha256`

El plan fija un implementation commit de exactamente cinco paths:

1. `src/geometria_proporcional/wave59_hgb_guard_bracket.py`;
2. `experiments/geometria_proporcional/prepare_wave56_fresh.py`;
3. `experiments/geometria_proporcional/run_wave59_hgb_guard_bracket.py`;
4. `tests/test_wave59_prospective.py`;
5. `tests/test_wave59_preoracle_recovery.py`.

Referencia: `WAVE_59_REPLAY_NORMALIZATION_SUCCESSOR_PLAN.md:205-211`.

En la config original, los primeros cuatro ya pertenecen tanto a `required_execution_sources` como a `source_sha256`:

- preparador: config original `:266,341`;
- runner: `:271,346`;
- módulo: `:289,364`;
- test prospectivo: `:294,369`.

El quinto test no estaba incorporado y su alta sí está autorizada expresamente por el plan.

Sin embargo, la allowlist del delta sólo autoriza:

- reemplazar el self-source de config;
- reemplazar R426;
- agregar plan, auditoría del plan y test de recovery;
- reemplazar `implementation_binding`;
- agregar `successor_authority`;
- agregar `pair_token_count_basis`.

Referencias: plan `:230-245`.

No autoriza cambiar los valores preexistentes de `source_sha256` para módulo, preparador, runner y test prospectivo. Inmediatamente después, el plan exige que todo campo no incluido sea canónicamente igual a la config original: plan `:247-251`.

Esto no queda cubierto por “`implementation_binding` completo”: `implementation_binding` y `source_sha256` son objetos top-level distintos. Tampoco puede inferirse como excepción implícita, porque el propósito declarado es precisamente un delta reproducible y fail-closed.

El fallo sería material:

- el preparador compara el mapa observado completo contra `source_sha256`: `prepare_wave56_fresh.py:784-809`;
- el runner repite esa igualdad exacta: `run_wave59_hgb_guard_bracket.py:360-377`.

Corrección mínima requerida:

- autorizar explícitamente el reemplazo de las cuatro entradas preexistentes de `source_sha256`;
- exigir para cada una:
  - old hash igual al de la config original;
  - new hash igual al blob del path en el implementation commit;
  - igualdad del mismo blob en HEAD, dado que los commits posteriores sólo pueden introducir auditoría, config y auditoría final;
- conservar como alta separada la entrada de `tests/test_wave59_preoracle_recovery.py`;
- agregar un test negativo que altere cualquier otro valor de `source_sha256` y exija rechazo.

## Verificación de los seis cierres

| Cierre | Estado | Evidencia |
|---|---|---|
| Conteo elegible tipado y reachability del fresh primary | CERRADO EN DISEÑO | Campo exacto `pair_token_count_basis=eligible_unique_pair_tokens`, restringido a la identidad sucesora; test `1152 total / 768 eligible`; rechazo del total como sustituto. Plan `:52-70,164-165`. El preparador está incluido en el alcance. |
| Unión exclusiva de attestations fresh/recovery | CERRADO EN DISEÑO | Dos schemas/formas exclusivas; fresh `primary/replay` sin provenance/amendment y recovery con ambos; reconstrucción desde bytes físicos y firma individual. Plan `:119-140,166-167`. |
| Auditoría final material fail-closed | CERRADO EN DISEÑO | Config introducida por commit exclusivo; auditoría final como HEAD exclusivo, hijo directo; bloque con commit, SHA bruto y PASS único; worktree global limpio y ausencia de commits posteriores. Plan `:253-268,286-291`. |
| Delta mecánico respecto de config original | ABIERTO / BLOCKER | Omite los cuatro reemplazos inevitables de hashes de fuentes ya existentes. Plan `:230-251`; config original `:266,271,289,294,341,346,364,369`. |
| Test real primary↔replay y regresión recovery | CERRADO EN DISEÑO | Exige modos frescos reales, hashes brutos distintos, firmas reales sin monkeypatch y regresión recovery separada. Plan `:144-179`. |
| Allowlist exacta, self-source y sentinels | CERRADO EN DISEÑO | Dos paths exactos; rechazo por sufijo, doble self-source o self-source distinto del ejecutado. Sentinels públicos explícitos. Plan `:168-172,213-219,250-251,293-319`. |

## Comprobaciones independientes

La cadena Git coincide exactamente:

- commit auditado: `f00860d3c34e16234f16b6d3a8c46bd6151d1e65`;
- parent único: `8c41e3cfeda640e68ef7c8aaa5d91ded8894f2b0`;
- el parent introduce únicamente R446;
- `f00860d` modifica únicamente el plan;
- `git diff --check f00860d^ f00860d`: exit 0;
- SHA-256 del plan congelado: `0c3378581db0a86263ade0b68b43e30398c0a6412085975536e94c40bd055baf`;
- worktree global: limpio.

Los sentinels públicos observados coinciden exactamente con el plan:

- `analysis.json`: `79a4c1eca78497d9fd6ea49177508ac9f879cabe0824b566b437d8b9f8e7eb96`;
- `artifact_manifest.json`: `ce30675744b27d656ff7eb177145ae5fac77d9f092ccc483d329425864185770`;
- `FAILURE.json`: `4967c6108ffd07e1b8cf6c0f301e702722be592f0d678594dd89e659053aa3c4`;
- `failure_inventory.json`: `353afe6f61c476a82d5c061fadcdfa13bad5c27787629189cb80094d4769b9a9`;
- `failure_attestation.json`: `8cb19ce7d741a0caaa73c88d44c96fe10557dbba6283290f90e98bbec0cb93e3`.

El análisis público conserva `scientific_decision=null`; ambos patrones mantienen `replay_exact=PENDING` y `aggregate_with_replay=null`. El failure record público conserva `run_role=replay` y `last_state=COMPLETE`.

## Alcance operativo

Se leyeron completos el plan congelado y R446 y se contrastaron con el módulo, preparador, runner, tests y config vigentes. No se editó ningún archivo, no se abrió semánticamente escrow, secretos ni truth, no se ejecutaron recovery o draw y no se usaron ni consultaron GPU, web o Mendieta.
