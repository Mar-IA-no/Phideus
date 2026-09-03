# Auditoría independiente de Ola 51

> Informes individuales preservados verbatim. Auditor read-only, esfuerzo high.

## Primera pasada

**Findings**
1. **Bloqueante: `staged_unfrozen` no aísla el efecto de congelar.** `factored_frozen` cambia a `choice_partial` y congela encoder/set path, mientras `staged_unfrozen` mantiene todo entrenable y usa `joint_equal`. Cambian simultáneamente trainabilidad y pérdida; una ventaja no puede atribuirse específicamente al congelamiento como declara el plan. [plan:45](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_51_FACTORED_SET_POLICY_SMOKE_PLAN.md#L45), [runner:537](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave51_factored_smoke.py#L537), [runner:569](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave51_factored_smoke.py#L569)

2. **Bloqueante: la etiqueta causal no implementa el criterio predeclarado.** El plan dice que `temporal_freeze_mechanism_specific` requiere las condiciones 1–4 y además la 5. El código la define únicamente mediante los dos contrastes de mecanismo; puede quedar `true` aunque `factored_candidate_promising` sea `false`. [plan:96](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_51_FACTORED_SET_POLICY_SMOKE_PLAN.md#L96), [runner:789](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave51_factored_smoke.py#L789)

3. **Alta: la garantía de no acceso al lockbox no es fail-closed.** La config se lee antes del guard; los manifiestos se leen durante el binding sin pasar por `_assert_no_lockbox_inputs`; después `access_receipt.json` sólo inventaría las lecturas devueltas por `load_labeled_records` y fija `lockbox_accessed: false`. Una config o manifiesto redirigido puede leerse antes de ser rechazado y no aparecer en `files_read`. [runner:117](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave51_factored_smoke.py#L117), [runner:369](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave51_factored_smoke.py#L369), [runner:799](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave51_factored_smoke.py#L799)

4. **Alta: el source binding nominal es correcto, pero no suficiente como invariante general.** Verifica los hashes antes del parseo, y los hashes actuales coinciden, pero acepta cualquier conjunto de cinco claves en vez del conjunto autorizado exacto. Además existe una ventana TOCTOU entre hashing y carga sin comparación posterior contra el binding. [runner:134](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave51_factored_smoke.py#L134), [runner:150](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave51_factored_smoke.py#L150), [runner:395](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave51_factored_smoke.py#L395)

5. **Media: `--force` puede borrar artefactos antes de validar las fuentes ejecutables.** El output se elimina y recién después `_execution_sources` comprueba que código, plan y config coincidan con `HEAD`. En el checkout auditado esa comprobación falla, por lo que `--force` destruiría una corrida previa y abortaría sin producir reemplazo. [runner:382](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave51_factored_smoke.py#L382), [runner:388](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave51_factored_smoke.py#L388)

6. **Media: el manifiesto no cubre el paquete final.** `artifact_manifest.json` excluye explícitamente `summary.json` y el reporte y se cierra antes de escribirlos. El summary enlaza el manifiesto, pero ningún root hash enlaza o permite verificar summary y reporte finales. [runner:812](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave51_factored_smoke.py#L812), [runner:878](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave51_factored_smoke.py#L878)

**Verificado**
True-vs-shuffled está correctamente matched entre controles y entra al criterio sólo mediante true-minus-shuffled; no se compara con main. Los cinco brazos usan la misma arquitectura de 13.384 parámetros y los primarios igualan batches/backprops. Las métricas agregadas y los JSONL por token respetan canonical-view averaging. Los hashes configurados coinciden con `wave50_prospective_v1`.

`9 passed in 1.49s`. No ejecuté el smoke completo: las fuentes difieren de `HEAD` y el guard lo rechaza; tampoco existen outputs Wave 51 para auditar post-run. No edité archivos.

## Segunda pasada

**PASS**
No quedan bloqueantes ni findings altos de los seis anteriores.

Las correcciones están presentes: contraste `staged_unfrozen` matched, etiqueta causal condicionada por el core, guards antes de toda lectura, allowlist exacta y revalidación post-load, validación de fuentes antes de `--force`, y `package_manifest.json` como raíz final. [runner:127](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave51_factored_smoke.py#L127), [runner:379](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave51_factored_smoke.py#L379), [runner:421](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave51_factored_smoke.py#L421), [runner:581](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave51_factored_smoke.py#L581), [runner:805](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave51_factored_smoke.py#L805), [runner:899](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave51_factored_smoke.py#L899).

Riesgos residuales: no existe un test end-to-end del manifiesto final y el guard anti-lockbox depende de rutas resueltas más binding, no de aislamiento del filesystem. El smoke completo no se ejecutó porque las fuentes actuales difieren de `HEAD`, como exige el guard. Suite focal: `9 passed in 1.33s`. No edité archivos.
