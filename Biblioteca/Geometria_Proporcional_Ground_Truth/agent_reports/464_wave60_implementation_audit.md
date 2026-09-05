## Auditoría técnica independiente — Wave 60

La implementación no satisface todavía el contrato auditado. Encontré cuatro defectos altos reproducibles y tres medios.

### Hallazgos

1. **HIGH — El sello de una root evaluada no protege la verdad científica interna.**

   [`validate_evaluated_root()`](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:1633) compara el inventario con `artifact_manifest.json` y verifica que las attestations liguen freeze, receipt y journal, pero nunca valida el contenido de esos freezes contra los archivos científicos físicos. Tampoco exige el conjunto cerrado completo de artefactos COMMON, SOURCE, SCORE y EVALUATION.

   Reproducción en una root temporal:

   - La root original fue aceptada.
   - Se reemplazó `evaluation/analysis.json` por una versión adulterada.
   - Se regeneró únicamente el manifest raíz, que no está firmado ni ligado externamente.
   - El freeze siguió ligando el hash original.
   - `validate_evaluated_root()` volvió a aceptar la root adulterada.

   Resultado observado:

   ```text
   accepted_before=True
   freeze_still_binds_original=True
   analysis_hash_changed=True
   accepted_after_tamper=True
   common_source_and_full_scientific_files_present=False
   ```

   Esto alcanza directamente a [`finalize_pair()`](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:2521), que confía en esa validación y después consume `analysis.json` para producir el resultado final.

   Corrección mínima: un validador closed-world por fase que exija paths exactos y recalcule todos los vínculos freeze → outputs, receipts, journals, attestations y manifest antes de permitir finalización o recovery. El manifest raíz debe estar firmado o ligado por una autoridad externa inmutable.

2. **HIGH — El guard de nueva realización no inspecciona individualmente los archivos del benchmark.**

   [`opaque_draw_fingerprint()`](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:878) incorpora `benchmark/manifest.json`, semantic root, bundles y commitments, pero no inventaría físicamente los archivos `benchmark/visible/**`, sealed ni protocol. El mapa `manifest["files"]` se trata como un único commitment agregado en [líneas 903–915](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:903).

   Reproducción:

   - `benchmark/visible/train.jsonl` fue el mismo hardlink en antecedente, primary y replay.
   - Otro registro del mapa `files` se hizo distinto, evitando igualdad total del diccionario.
   - Los tres archivos tuvieron el mismo inode.
   - [`validate_new_draw_pair()`](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:929) devolvió `PASS`.

   Esto contradice el control por archivo protegido del plan en [líneas 202–210](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md:202).

   Corrección mínima: expandir `manifest.files` a identidades físicas por path y comparar, para cada elemento, hash, path resuelto y `(st_dev, st_ino)` entre primary, replay y todos los antecedentes.

3. **HIGH — La matriz de presencia terminal no se hace cumplir.**

   [`seal_root_failure()`](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:1955) rechaza algunos directorios futuros, pero no exige que estén presentes todos los artefactos de las fases declaradas como completas. Luego publica incondicionalmente:

   ```json
   "missing_expected": [],
   "forbidden_present": []
   ```

   en [líneas 2053–2070](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:2053).

   La propia prueba [`test_failure_bindings_are_directional...`](/mnt/m2-1TB/Phideus/tests/test_wave60_frozen_policy_transport.py:549) sella exitosamente:

   - `SCORE_APPLY_FAILED_PRE_TRUTH / SOURCE_LAW_BOUND` con sólo config y un journal;
   - `PEER_ABORTED_PRE_TRUTH / PREPARED` con sólo config.

   Ambas roots carecen de los conjuntos COMMON/SOURCE o PREPARED que sus estados declaran completos. El plan exige presencia exacta por `last_complete_phase` en [líneas 747–752](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md:747).

   Corrección mínima: tablas declarativas de artefactos requeridos y prohibidos para cada combinación terminal/fase/rol; negarse a sellar si los conjuntos no son exactos y poblar `missing_expected`/`forbidden_present` desde la comprobación real.

4. **HIGH — La suite focal no cubre el protocolo de aceptación obligatorio.**

   Hay 22 pruebas focales, pero faltan, entre otras:

   - ciclo real completo de `execute_prepared_pair`;
   - sellado de roots completas y finalización real;
   - adulteración posterior al freeze de analysis, actions, index, receipts, attestations y manifest;
   - matrices terminales completas y fallos asimétricos por fase;
   - recovery/amendment v2;
   - presupuesto combinado;
   - recomputación independiente de métricas, deltas, CIs y patterns;
   - casos completos true/false/`NOT_EVALUABLE`;
   - preservación hash-exacta de antecedentes;
   - comprobación del invariante de no creación de outputs canónicos.

   Además, la prueba de reanudación de finalización reemplaza [`validate_evaluated_root()` por un mock](/mnt/m2-1TB/Phideus/tests/test_wave60_frozen_policy_transport.py:524), por lo que no prueba la unión crítica entre recovery e integridad.

5. **MEDIUM — El límite de 900 segundos no es combinado.**

   La preparación inicia un presupuesto independiente en [`wave59_coordinator_budget()`](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/prepare_wave56_fresh.py:4471) para cada invocación primary/replay. Después, [`execute_prepared_pair()`](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:2183) reinicia otro contador de 900 segundos.

   Por tanto, el protocolo puede consumir aproximadamente 900 + 900 + 900 segundos, aunque el plan fija un máximo combinado de 900 segundos en [línea 1008](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md:1008).

   Corrección mínima: ledger durable del presupuesto del intento, compartido por ambas preparaciones y la ejecución posterior.

6. **MEDIUM — Las autoridades de auditoría futura se validan por forma y hash, no por veredicto ni cadena exclusiva.**

   [`validate_pre_draw_config()`](/mnt/m2-1TB/Phideus/src/geometria_proporcional/wave60_frozen_policy_transport.py:932) acepta cualquier archivo ligado por SHA-256 acompañado por el string `ACCEPTED_IMPLEMENTATION_AUDIT`; no parsea su veredicto, commit objetivo ni contenido. El preflight sólo comprueba que los commits sean ancestros y que coincidan los hashes en [líneas 960–970](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/prepare_wave56_fresh.py:960).

   Corrección mínima: parser canónico de auditoría con keyset cerrado, único veredicto `PASS`, commit exacto auditado, path esperado y verificación de parent directo/cadena exclusiva para implementación y autoridad source-law.

7. **MEDIUM — Un error transitorio durante finalización destruye staging recuperable.**

   [`execute_prepared_pair()`](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:2500) captura cualquier excepción de `finalize_pair`, elimina recursivamente `pair.initializing` y publica un aborto permanente post-truth. Esto contradice la recuperación idempotente del staging coincidente descrita por el plan.

   Corrección mínima: preservar staging ante errores operativos recuperables; publicar aborto permanente sólo ante drift o inconsistencia probada de bindings.

### Verificaciones favorables

- El ref solicitado `77d6b89a` no existe; el commit presente y auditado fue `77d6b89e563a37b8a912d642dd2556141cecdb5c`, con parent directo `a28a077db78fe68ff98f1c35e68333dd967ddcf5`.
- Worktree limpio y exactamente cinco paths cambiados.
- No existen config final, autoridad source-law ni attempt canónico.
- Los nueve hashes fuente, proyección 13+3, 1.300 keys, 3.900 arrays y reconstrucción retrospectiva son consistentes.
- No observé fitting ni recalibración; los monkeypatches de prohibición pasan.
- Workers separados mediante `bwrap`, usuario `nobody`, capacidades vacías, `NoNewPrivs`, CUDA oculta y cuatro threads.
- Tests focales: `22 passed`.
- Regresión Wave 56–60: `357 passed, 1 skipped`; el único fallo inicial fue el uso de hardlinks entre filesystems distintos por el `tmp_path` de pytest. Repetido con `--basetemp` dentro del filesystem del repositorio: `1 passed`. Evidencia acumulada: 358 tests ejecutables pasan y 1 queda skipped.
- Temporales propios eliminados; worktree final limpio.

**Decisión final: REVISE**
