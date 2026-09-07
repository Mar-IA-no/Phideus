# R564 — Auditoría focalizada de cierre del runner set-valued proporcional

**Fecha:** 2026-09-07  
**HEAD auditado:** `53ea61ce41acda82affe2c83310bf0713c86dbc0`  
**Régimen:** CPU exclusivamente; entorno de threads fijado; no se consultó ni utilizó GPU.  
**Alcance:** corrección de R563-F01, artefactos primario/replay regenerados y mutation suite v3. No se editó implementación ni documentación del frente.

## Dictamen

**PASS — 0 HIGH, 0 MEDIUM, 0 LOW.**

R563-F01 quedó corregido. Todo JSON del artefacto, incluido `artifact_manifest.json`, está sujeto a representación byte-canónica y a rechazo estricto de constantes no finitas. La mutación semánticamente equivalente pero pretty-printed se rechaza por la ruta requerida después de refrescar el manifest. Los artefactos válidos primario y replay conservan 14/14 checks PASS y equivalencia byte-exacta. La campaña v3 completa 54/54 casos dentro de sus presupuestos declarados. No apareció ninguna regresión ni finding nuevo en el alcance auditado.

Este dictamen es técnico; no constituye una decisión científica de GO/NO-GO.

## 1. Corrección de canonicalidad JSON

El checker vigente implementa las tres capas necesarias:

- `reject_json_constant` convierte cualquier `NaN`, `Infinity` o `-Infinity` en error, y `read_json` lo instala como `parse_constant` (`check_proportional_set_valued_native_preflight.py:107-114`).
- `canonical_json_bytes` serializa con claves ordenadas, UTF-8 no escapado, separadores compactos, `allow_nan=False` y newline terminal (`check_proportional_set_valued_native_preflight.py:117-127`).
- P12 recorre `self.root.rglob("*.json")` y compara los bytes de cada JSON con esa serialización canónica antes de validar inventario y hashes (`check_proportional_set_valued_native_preflight.py:1466-1473`). Esa enumeración incluye expresamente `artifact_manifest.json`; el manifest ya no queda fuera del control byte-canónico.

La inspección íntegra de los artefactos actuales confirmó, para primario y replay por separado:

- 35 archivos totales;
- 17 JSON;
- los 17/17 JSON byte-canónicos, incluido `artifact_manifest.json`;
- 585 arrays NPZ inspeccionados, con 35.701.634 valores por artefacto.

## 2. Probes negativos independientes

Los probes se ejecutaron sobre copias temporales, refrescando el manifest cuando correspondía para aislar canonicalidad de integridad hash.

| Probe | Resultado observado | Clasificación |
|---|---|---|
| `source_bindings.json` pretty-printed, semánticamente equivalente, manifest refrescado | exit 1; `CheckFailure: JSON encoding is noncanonical: source_bindings.json` | `RAW_OR_REPLAY_INVALID` |
| `source_bindings.json` con `NaN`, manifest refrescado | exit 1; `ValueError: non-finite JSON constant forbidden: NaN` | `SOURCE_OR_SCOPE_INVALID`, porque el archivo se consume durante la inicialización estricta |
| `artifact_manifest.json` con `Infinity` | exit 1; `ValueError: non-finite JSON constant forbidden: Infinity` | `RAW_OR_REPLAY_INVALID` |

Por tanto, las constantes no finitas se rechazan tanto en JSON ordinario como en el propio manifest. El caso específicamente requerido —JSON pretty-printed semánticamente equivalente con manifest refrescado— llega a `RAW_OR_REPLAY_INVALID`.

## 3. Artefactos válidos y replay

Reejecutados bajo:

```text
CUDA_VISIBLE_DEVICES=''
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
BLIS_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
```

Resultados:

| Artefacto | Checks | Exit | Wall | Torch | GPU |
|---|---:|---:|---:|---|---|
| Primario | 14/14 PASS | 0 | 3.818607278 s | no importado | no usada |
| Replay contra primario | 14/14 PASS | 0 | 3.860367015 s | no importado | no usada |

La comparación directa encontró 32 archivos comparables y 0 diferencias: `byte_exact=true`. El receipt de replay declara los mismos 32 archivos y referencia el manifest primario con SHA-256 `a0401834c3958680ef687ad264b8b56a017a8996eaade904873b342319528a39`.

Los costes archivados también son internamente consistentes:

- primario: wall `28.27180427312851 s`, CPU `28.268283517 s`, peak RSS `986238976 B`;
- replay: wall `28.148873522877693 s`, CPU `28.139536221 s`, peak RSS `985055232 B`;
- suma wall declarada: `56.4206777960062 s`, exactamente la suma de ambos valores dentro de precisión decimal;
- `within_hard_budget=true`, sin importación de torch ni uso GPU.

## 4. Mutation suite v3 y veracidad de receipts

El harness incorpora `rewrite_json_noncanonical`, que reserializa con indentación y luego refresca el manifest (`run_proportional_set_valued_mutations.py:129-139`). El caso `json_noncanonical` exige `RAW_OR_REPLAY_INVALID` (`run_proportional_set_valued_mutations.py:193`). El flujo de ejecución registra los checkers válidos, los casos, wall, `RUSAGE_CHILDREN` peak RSS y presupuestos, y sólo retorna éxito si coinciden todas las rutas, pasan ambos válidos y se respetan los costes (`run_proportional_set_valued_mutations.py:201-260`).

Reejecución independiente completa:

- schema: `proportional-mutation-suite-v3`;
- 54 casos únicos;
- 54/54 PASS;
- 54/54 rutas observadas iguales a las esperadas;
- validación primaria: PASS, exit 0;
- validación replay: PASS, exit 0;
- wall observado: `134.92942690104246 s` frente a presupuesto `900.0 s`;
- peak child RSS observado: `642617344 B` frente a presupuesto `1610612736 B`;
- `within_budget=true` y exit global 0.

Las dos desigualdades presupuestarias son verdaderas (`134.92942690104246 < 900.0`; `642617344 < 1610612736`) y el agregado 54/54 coincide con los 54 registros únicos del receipt. SHA-256 del receipt independiente: `51563639cca0dffa31ba092b4b87105cf379edf09af0d6f288bdb63de7ccb657`.

## 5. Revisión de regresiones

El commit auditado modifica únicamente el checker y el harness de mutaciones. El runner y las primitivas numéricas permanecen sin cambios respecto de R563. Sobre los artefactos regenerados, ambos checkers conservan 14/14 PASS, el replay sigue siendo byte-exacto y la campaña negativa amplía cobertura sin perder ninguno de sus 54 casos. No se detectó regresión funcional, de reproducibilidad, de integridad o de coste en el alcance de cierre.

## Findings

Ninguno.
