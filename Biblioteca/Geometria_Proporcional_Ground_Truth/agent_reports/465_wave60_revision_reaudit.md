## Veredicto: REVISE — 2 HIGH, 3 MEDIUM, 0 LOW

El commit auditado es exactamente `eabc487f2ff641110f214720ba45350b32882502`, con parent directo `71649f3265e38f72b2b6932d11e9bb2ddb01385a`. Modifica exclusivamente los cuatro paths autorizados: `167/9`, `645/108`, `68/8` y `627/44` líneas añadidas/eliminadas respectivamente; total `1507/169`.

### Findings abiertos

1. **HIGH — `replay_finalize` vuelve a abrir físicamente truth y secretos sellados.**

   `finalize_pair()` llama a `validate_evaluated_root()` en [run_wave60_frozen_policy_transport.py:3047](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:3047). Esa ruta entra en `validate_completed_root_phases()` y `_validate_worker_phase(evaluate)`, que recalcula directamente el SHA-256 de `prepared/sealed_monitor_truth_bundle.npz` en [run_wave60_frozen_policy_transport.py:1970](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:1970).

   Además, `compare_evaluated_roots()` vuelve a leer:

   - `gate_select_truth_bundle.npz`;
   - `sealed_monitor_truth_bundle.npz`;
   - todos los archivos bajo `benchmark/sealed/**`;

   en [run_wave60_frozen_policy_transport.py:2241](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:2241).

   Son lecturas opacas para hashing, no una carga semántica con `np.load`, pero siguen siendo reaperturas físicas de truth durante finalización. El contrato exige que `replay_finalize` consuma hashes, manifests, freezes, attestations y outputs científicos ya inmutables, sin reabrir truth ni estados ([plan:516](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md:516)).

   Corrección: validar las cadenas científicas contra los hashes ya congelados y firmados; para presencia closed-world usar nombres, `lstat` y el manifest, sin volver a leer contenidos secretos. La comparación de secretos debe consumir los commitments previamente sellados.

2. **HIGH — El recovery Wave 60 v2 se valida superficialmente, pero no puede ejecutarse.**

   `validate_recovery_amendment()` no tiene rama para `WAVE60_CONFIG_SCHEMA`; una config Wave 60 cae en `_validate_wave56_recovery_amendment()` en [prepare_wave56_fresh.py:3454](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/prepare_wave56_fresh.py:3454).

   Probe ejecutado:

   ```text
   {'dispatched': 'wave56'}
   ```

   Hay un segundo bloqueo: `validate_invocation()` sólo admite que el escrow provenga de la primary actual o de archives dentro del mismo contenedor ([prepare_wave56_fresh.py:848](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/prepare_wave56_fresh.py:848)). Por ello rechaza la primary terminal de `attempt_v1` como fuente para el contenedor nuevo `attempt_v2`, que es precisamente la topología exigida por el plan.

   Probe:

   ```text
   ValueError recovery escrow must come from this primary or one of its archives
   ```

   La única prueba v2 actual, [test_wave60_frozen_policy_transport.py:1132](/mnt/m2-1TB/Phideus/tests/test_wave60_frozen_policy_transport.py:1132), verifica solamente que una estructura ficticia pasa `validate_pre_draw_config()`; no recorre amendment, failure binding, copia sin hardlinks, preservación del draw ni preparación real.

3. **MEDIUM — La matriz terminal aún permite firmar estados internamente contradictorios.**

   `seal_root_failure()` fija `last_complete_phase`, pero no liga cada terminal a su `phase`, `truth_accessed` y `recovery_allowed` exactos ([run_wave60_frozen_policy_transport.py:2503](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:2503)).

   Probe sobre una root mínima válida:

   ```text
   {'terminal': 'INVALID_PREPARATION',
    'phase': 'WRONG_PHASE',
    'truth_accessed': True,
    'recovery_allowed': False}
   ```

   Esa autoridad fue sellada y firmada aunque `INVALID_PREPARATION` es necesariamente pre-truth. Debe existir una tabla declarativa exacta `terminal → phase, last_complete_phase, truth_accessed, recovery_allowed, peer fields`, aplicada antes de firmar. La suite tampoco recorre todavía todas las fallas asimétricas reales de preparación, source, score y evaluación.

4. **MEDIUM — La cadena Git sigue sin autoridad final de config/HEAD.**

   La revisión agrega un parser canónico correcto para auditorías de implementación y source law. Sin embargo, el schema de config no contiene binding de la auditoría final de config ([wave60_frozen_policy_transport.py:856](/mnt/m2-1TB/Phideus/src/geometria_proporcional/wave60_frozen_policy_transport.py:856)), y el preflight sólo requiere que las auditorías de implementación y source law sean ancestros.

   No se exige:

   - commit exclusivo de config;
   - auditoría de config como hijo directo;
   - veredicto normativo `PASS` de esa auditoría;
   - que esa auditoría sea `HEAD`;
   - que los blobs ejecutados sigan siendo los auditados por el commit de implementación.

   Esto deja sin implementar los pasos 19–20 y la prohibición de ejecutar antes de que la auditoría final sea HEAD ([plan:989](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md:989)).

5. **MEDIUM — El presupuesto combinado no es todavía un ledger durable completo.**

   Primary y replay de preparación sí encadenan ahora duraciones firmadas. Durante `execute_prepared_pair()`, en cambio, el consumo posterior se mantiene sólo en `time.monotonic()` del proceso actual ([run_wave60_frozen_policy_transport.py:2712](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:2712)).

   Problemas restantes:

   - no hay acumulador durable del total preparación + score + evaluate + finalize;
   - `finalize_pair()` directo no valida presupuesto;
   - la finalización no está encerrada en timer;
   - `runtime.json` final no registra presupuesto observado, sólo CPU/CUDA ([run_wave60_frozen_policy_transport.py:3210](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:3210)).

### Revisión de los siete findings R464

| R464 | Estado |
|---|---|
| 1. Cadena científica no protegida | **Corregido en integridad**, pero la solución reabre truth durante finalize: nuevo HIGH |
| 2. Guard físico por manifest file | **Corregido**; compara cada miembro por hash, path resuelto e inode |
| 3. Matriz terminal | **Parcial**; presencia de fases mejorada, pero acepta `phase/truth/recovery` contradictorios |
| 4. Protocolo de aceptación | **Parcial**; hay ciclo real y tamper tests, pero recovery v2 no funciona y faltan matrices/fallos asimétricos completos |
| 5. Presupuesto combinado | **Parcial**; preparación durable, ejecución/finalize no |
| 6. Parser y cadena Git | **Parcial**; implementación/source law corregidos, auditoría final de config ausente |
| 7. Staging transitorio | **Corregido**; errores operativos no clasificados como drift preservan staging |

### Pruebas ejecutadas

- Focal Wave 60:

  ```text
  29 passed in 37.66s
  ```

- Regresión Wave 56–60, diez archivos:

  ```text
  364 passed, 1 skipped, 1 failed in 497.38s
  ```

  El único fallo fue ambiental: `os.link` desde `/mnt/m2-1TB` hacia el `tmp_path` alojado en `/tmp` devolvió `EXDEV`. Repetido con `--basetemp` en el mismo filesystem:

  ```text
  1 passed in 2.30s
  ```

  Resultado efectivo único: **365 passed, 1 skipped**.

- `git diff --check`: PASS.
- `py_compile` de los cuatro paths: PASS.
- Probes específicos: dispatch recovery Wave 60→Wave 56 confirmado; rechazo de escrow v1→v2 confirmado; autoridad terminal contradictoria confirmada.
- Focal completo verificó ciclo feliz real, recomputación independiente, 14 actions, cuatro métricas, 5.000 bootstraps, closed-world, ausencia de fitting/recalibration y varios tamper cases.

### Higiene y límites

- Worktree final limpio.
- Config Wave 60 canónica: ausente.
- Source authority canónica: ausente.
- Attempt canónico: ausente.
- Temporales propios eliminados.
- No se usó ni consultó GPU, web o Mendieta.
- Todas las ejecuciones usaron `CUDA_VISIBLE_DEVICES=''` y cuatro threads.
- No se modificaron archivos versionados ni commits.

No incluyo bloque normativo de autoridad porque el veredicto es `REVISE`.
