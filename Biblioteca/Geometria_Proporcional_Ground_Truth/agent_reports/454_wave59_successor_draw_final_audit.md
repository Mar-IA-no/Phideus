# R454 — auditoría final independiente del draw sucesor de Ola 59

**Dictamen técnico:** `PASS`

**Findings:** bloqueantes `0`, altos `0`, medios `0`, bajos `0`.

## Observaciones verificadas

- HEAD exacto y worktree limpio: `8edc23d1120a91981e01aeb8c385c344da42b3fb`.
- La autoridad final, los bindings de ejecución y los cinco sentinels antecedentes validaron sin divergencias.
- Firmas Ed25519 verificadas individualmente:
  - primario → `primary`
  - replay → `replay`
  - clave pública confiable SHA-256: `e5b8fcb229908503343f04c44dfed04646960de82454b411fbdc16b3c9525350`
- Preparación replay: `26/26`, `all_exact=true`.
- Comparación recomputada, idéntica al recibo almacenado:
  - scientific exact: `21/21`
  - arrays: `21/21`
  - secretos por hash opaco: `11/11`
  - estado funcional: `16/16` en cada raíz
  - semántica operacional: `11/11`
  - `all_exact=true`
- Manifests cerrados:
  - primario: `82/82` archivos
  - replay: `84/84`
  - sin faltantes, extras, solapamientos ni no clasificados; hashes y tamaños públicos exactos.
- Contrato fresco:
  - sin `recovery_amendment.json`
  - sin `recovery_provenance`
  - `recovery_context=false`
  - `oracle_materialized=false`
- Población por cada split y raíz: `4,992` filas, `1,152` pair tokens totales, `768` elegibles, `192` no canónicos, `384` out-of-catalog.
- Estados completos en ambas raíces: `PREPARED → FIT_COMPLETE → CALIBRATION_FROZEN → VALIDATION_COMPLETE → MONITOR_ACTIONS_FROZEN → COMPLETE`.
- CPU exclusiva, `CUDA_VISIBLE_DEVICES=""`, presupuestos activos:
  - primario: `109.136 s`, RSS máximo `746,455,040 B`
  - replay: `110.404 s`, RSS máximo `751,722,496 B`
  - combinado: `219.540 s < 3,600 s`
- Derivaciones públicas recomputadas:
  - `192/192` arrays analíticos
  - bootstrap exacto, forma `5000 × 303`
  - `32/32` summaries
  - seis grupos de contrastes, incluidos `36/36` factoriales
  - condiciones prospectivas y `REPORT.md` exactos.

## Resultado observado

- Patrón incompatibility: `7/8` condiciones verdaderas; falla el contraste contra el control de desplazamiento máximo, cuyo IC95 superior es `+0.0008435`. Agregado: `false`.
- Patrón harm: `6/8`; fallan compatibilidad frente a hard —IC95 inferior `-0.0015127`— y control de desplazamiento máximo —IC95 superior `+0.0034653`—. Agregado: `false`.
- Soporte autorizado: `37` tokens para mean y `28` para tail, ambos sobre el mínimo `25`.
- `scientific_decision=null` y autoridad de decisión reservada al usuario.

## Inferencia de auditoría

Los artefactos satisfacen integridad, trazabilidad, clausura, autenticidad, reproducibilidad exacta y límites operativos. Los dos patrones prospectivos negativos son resultados científicos correctamente representados, no defectos de ejecución. Esta auditoría no declara promoción arquitectónica ni GO/NO-GO.

Anclas públicas principales:

- Config: `f6edfd2106fe87c8150562d096469e29b64a108a73de2dae0d371bd689a4a9b6`
- Analysis/Report: `25a262c22bfe1d924578031d3d8a6faf7cc6e2632af4e8bc31e90f3ad8fca9b8` / `0bc83f6cff43ee4881c0f4b59dd29346a8159ec22f3a4317a7de560b67fb1f6d`
- Manifest primario/replay: `909361b45e9fb51977229063afcdf9587144bfb80280fd3c70dda86de19a6371` / `6b8bf391c3f51be0c862fc3af266e95717836e79bdf4072d13179f4b4e6fb13a`
- Replay comparison: `4b0d2a420bf62966a451f8fd00ad6bc8e871c43e6c55cd836c73123242da156a`
- Preparation replay: `5ffa4ffbaf871f8a597b66bb0d9d6300da40e65270d105d3d1f1aa43a635e341`

Los secretos y escrows sólo se trataron mediante hashes y validadores autenticados; no se abrieron semánticamente.
