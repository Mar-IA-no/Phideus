# R299 — reauditoría independiente de la implementación de Ola 53

> **Rol:** auditor independiente con contexto mínimo
> **Fecha:** 2026-09-03
> **Resultado:** `REVISE`

## Informe verbatim

**Findings**
1. **Alta, bloqueante: el runner no puede consumir los artefactos reales de Wave52.** `load_metadata()` agrupa el JSONL bruto únicamente por `pair_token`, pero éste contiene tanto vistas `canonical_preserving` como `origin_translation_break`. En `val_threshold`, 93/384 tokens tienen múltiples pares `{design_stratum, cardinality}`; en `val_monitor`, 99/384. La corrida real aborta en el primer caso con `ValueError: inconsistent metadata`. Wave52 filtra explícitamente `canonical_preserving` antes de agrupar; Wave53 omite ese filtro. [runner:177](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave53_uncertainty_policy.py#L177), [runner:575](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave53_uncertainty_policy.py#L575), [wave50_neural.py:130](/mnt/m2-1TB/Phideus/src/geometria_proporcional/wave50_neural.py#L130).

2. **Alta: la cronología freeze/monitor sigue parcialmente incumplida.** `execution_sources()` fue correctamente adelantado y la disjunción triple sí se verifica. Sin embargo, `validate_sources()` hashéa el JSONL de labels y todos los NPZ de `val_monitor` antes de `analysis_freeze.json`; además, `load_metadata()` parsea el JSONL global completo antes de filtrar tokens. La afirmación `written-before-val_monitor-target-load` no demuestra el contrato del plan. [runner:163](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave53_uncertainty_policy.py#L163), [runner:170](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave53_uncertainty_policy.py#L170), [runner:562](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave53_uncertainty_policy.py#L562), [runner:610](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave53_uncertainty_policy.py#L610).

3. **Alta operativa: el preflight sigue sin poder pasar.** Runner, módulo y tests están sin trackear; plan y config también están sin trackear pero ocultos por `.gitignore`. Los cuatro artefactos exigidos por `execution_sources()` fallan actualmente `git ls-files`. [runner:108](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave53_uncertainty_policy.py#L108), [.gitignore:26](/mnt/m2-1TB/Phideus/.gitignore#L26), [.gitignore:49](/mnt/m2-1TB/Phideus/.gitignore#L49).

4. **Baja, estabilidad/serialización:** `calibration_metrics()` puede producir correlaciones y máximo `NaN` cuando alguna columna residual tiene varianza cero. `write_json()` permite entonces el literal no estándar `NaN`. No apareció con la población real corregida, pero falta fail-closed o representación explícita. [runner:320](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave53_uncertainty_policy.py#L320), [runner:337](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave53_uncertainty_policy.py#L337), [wave49_schema.py:149](/mnt/m2-1TB/Phideus/src/geometria_proporcional/wave49_schema.py#L149).

**R298**
| Finding | Estado |
|---|---|
| 1. Freeze/monitor y disjunción | **PARCIAL/FAIL**: disjunción resuelta; cronología no |
| 2. Replay externo | **RESUELTO** |
| 3. Preflight Git | **FAIL** |
| 4. Platt sobre población primaria | **RESUELTO** |
| 5. Máscara selectiva común | **RESUELTO** |
| 6. Provenance/software/per-seed/risk curve | **RESUELTO** |
| 7. Nearest-integer | **RESUELTO**: `33/66`, `50/66`, `59/66` |

**Verificación**
Los 11 tests Wave53 y los 24 tests Wave52+53 pasan. Los extremos `0/1/±inf` producen masas y tensores finitos con shapes correctos. La comparación de replay rechazó mutaciones de arrays y JSONL.

Inyectando únicamente el filtro canónico en memoria, dos procesos independientes completaron y coincidieron exactamente en 5 NPZ, 13 artefactos textuales y el hash semántico. Esto confirma que el replay dejó de ser tautológico, pero el runner vigente no alcanza esa etapa.

**Veredicto: `REVISE`.** No está listo para commit ni corrida CPU. No modifiqué archivos.
