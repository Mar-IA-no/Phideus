# Ola 60 — resolución de la falsa atribución detectada por R517

> **Estado:** `PRE-IMPLEMENTATION / R517-REVISE / R514-R515-REINSTATED / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-06
> **R516 no autorizante:** commit `d564d1c078248cc0083ecda206db81b0c80752da`, SHA-256 `c2b8fec2e6426c1456403bb2ed714d91ff6fe2611530c656f0ea13b976aade6b`
> **R517:** commit `dfb1b86938685899194f11449de481b9e14e45ee`, SHA-256 `dd67ef3f771013d9950f3138a2a5d3d61a2b238c5acdc93f680b5d21c8d3a14b`, `REVISE 1/0/0`

## 1. Resolución factual

R517 demuestra que R516 atribuyó a R514 una divergencia inexistente. El valor
normativo de R514, la config física/blob R508 y el source físico/blob es en los
cuatro casos:

```text
46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65
```

El valor `...fc4f...` provenía del snapshot conversacional y fue introducido
únicamente en un borrador no commiteado del checker. El test de igualdad con
`config.source_sha256` lo rechazó correctamente. Por tanto:

- R514 y R515 recuperan su condición de autoridades sustantivas de la
  implementación;
- R516 se conserva como plan de premisa falsa y no concede autoridad;
- R517 se conserva como auditoría que refuta esa premisa;
- el borrador sólo puede integrarse después de cambiar su constante a
  `...fc4e...` y pasar nuevamente la igualdad literal `8/8`.

No se modifica ningún documento histórico ni el intento.

## 2. Cadena de cierre revisada

Como R516/R517 ya forman parte de la genealogía, el cierre continúa de manera
lineal y explícita:

```text
R517 REVISE
  -> R518 este plan, exclusivo, parent R517
  -> R519 auditoría de este plan, exclusiva, parent R518
  -> R520 implementación checker+test, exclusiva, parent R519
  -> R521 auditoría de implementación, exclusiva, parent R520
  -> R522 corrección candidata JSON, exclusiva, parent R521
  -> R523 auditoría del artefacto, exclusiva, parent R522
  -> documentación y wiki
```

Paths exactos:

```text
R519: Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/519_wave60_r517_false_attribution_resolution_plan_audit.md
R520: experiments/geometria_proporcional/adjudicate_wave60_v4_result.py
      tests/test_wave60_v4_result_adjudication.py
R521: Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/521_wave60_r509_replay_normalization_implementation_audit.md
R522: Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_V4_REPLAY_NORMALIZATION_CORRECTION.json
R523: Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/523_wave60_v4_replay_normalization_correction_audit.md
```

R519 usa scope `R517_FALSE_ATTRIBUTION_RESOLUTION_PLAN` y target exacto
`{plan_commit,plan_sha256}`. R521 conserva scope
`R509_REPLAY_NORMALIZATION_RESOLUTION_IMPLEMENTATION` y target exacto
`{implementation_commit,files}` con los dos paths R520. R523 conserva scope
`WAVE60_V4_REPLAY_NORMALIZATION_CORRECTION` y target exacto
`{artifact_commit,artifact_path,artifact_sha256}`.

Cada informe contiene un único JSON con el keyset fijado por R514, `PASS
0/0/0`, `files_modified=false` y `gpu_used_or_queried=false`; commits, parents,
pathsets, blobs y hashes son exactos y fail-closed.

## 3. Autoridad acumulada de R522

R522 conserva todas las autoridades R509–R515 definidas por R514 y añade
exactamente:

```text
r516_false_source_hash_plan
r517_false_source_hash_plan_audit
r518_false_attribution_resolution_plan
r519_false_attribution_resolution_plan_audit
r520_implementation
r521_implementation_audit
```

Los planes usan `{commit,path,sha256}`; las auditorías
`{commit,path,sha256,authority_json}`; la implementación
`{commit,files}`. R516 y R517 se rotulan históricos/no autorizantes en sus
propios contenidos y verdicts; no se los transforma en PASS.

Los valores ya conocidos son:

- R516: commit y SHA del encabezado, path
  `Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_R515_SOURCE_HASH_AUTHORITY_CORRECTION_PLAN.md`;
- R517: commit y SHA del encabezado, path
  `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/517_wave60_r515_source_hash_authority_correction_plan_audit.md`,
  autoridad exacta `REVISE 1/0/0`, scope
  `R515_SOURCE_HASH_AUTHORITY_CORRECTION_PLAN` y target R516.

R518/R519 se completan con este commit y su auditoría. R520/R521 se completan
al existir la implementación auditada. R522 no se autorreferencia.

## 4. Cambios obligatorios del borrador

Antes de R520 se recupera el borrador no commiteado y se exige:

1. constante source principal exactamente `...fc4e...`;
2. `SOURCE_BINDINGS == config.source_sha256` como mapa completo;
3. hashes físicos de siete sources y self-binding de config `8/8`;
4. paths/números/authority chain actualizados a R518–R523;
5. `artifact_status=CANDIDATE_PENDING_R523_AUDIT`;
6. `activation_condition` ligada al path, scope, PASS y `0/0/0` de R523;
7. todas las guardas, keysets, self-manifest metadata, identidad 141/141,
   normalización semántica y ataques de R514 preservados.

El regression test que descubrió el typo debe pasar sin relajar la comparación.
El primer resultado observado (`20 passed, 1 failed`) se conserva como
evidencia de desarrollo, no como validación final.

## 5. Ciencia y cómputo

Esta resolución sólo corrige nuestra descripción del linaje y un borrador. El
intento sigue `COMPLETE`, sus métricas y patrones no cambian y la corrección de
replay continúa siendo candidata hasta R523. Todo es CPU-only, sin uso ni
consulta de GPU. No hay promoción ni decisión `GO/NO-GO`.
