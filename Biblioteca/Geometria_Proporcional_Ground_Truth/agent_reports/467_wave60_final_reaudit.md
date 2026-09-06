## R467 — REVISE

Resultado: **1 HIGH, 1 MEDIUM, 0 LOW**. No emito bloque normativo `wave60-audit-authority-v1`.

### HIGH — La recuperación v2 no puede conservar el ledger acumulativo de 900 s

La cadena Git de recuperación quedó estructuralmente corregida, pero el intento recuperado no es ejecutable con el ledger que produce el flujo real.

- [`wave60_prior_preparation_elapsed()`](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/prepare_wave56_fresh.py:5117) exige que cualquier primary usado como antecedente tenga `prior_elapsed_seconds == 0` y `cumulative_duration_seconds == duration_seconds` (líneas 5163–5172).
- Sin embargo, [`main()`](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/prepare_wave56_fresh.py:5259) carga el tiempo del primary v1 y lo pasa al presupuesto del primary recuperado v2 (líneas 5267–5273). Por diseño, ese primary v2 queda firmado con `prior_elapsed_seconds > 0`.
- Después, la preparación del replay v2 intenta leer ese receipt y falla por la exigencia anterior.
- Independientemente, [`preparation_budget_record()`](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:805) rechaza todo primary cuyo `prior_elapsed_seconds != 0` (línea 835), por lo que el executor tampoco acepta el primary recuperado.
- [`pair_preparation_elapsed()`](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:841) compara el prior del replay contra `primary.duration_seconds` (línea 844), no contra `primary.cumulative_duration_seconds`.

Reproducción CPU-only con receipt recuperado y firmado:

```text
SIGNED_RECOVERED_PRIMARY_REJECTED RuntimeError Wave 60 prior preparation budget ledger is invalid
RECOVERY_PAIR_LEDGER_REJECTED IntegrityDriftError Wave 60 preparation budget values drifted
```

El e2e actual oculta el defecto: reescribe manualmente el receipt del primary recuperado a `prior=0, cumulative=1` y el del replay a `prior=1, cumulative=2`, y vuelve a firmarlos en [`test_wave60_frozen_policy_transport.py:2031–2115`](/mnt/m2-1TB/Phideus/tests/test_wave60_frozen_policy_transport.py:2031). No prueba el ledger producido por `main()`.

Corrección necesaria:

- Preservar el acumulado v1 en el primary v2.
- Devolver el `cumulative_duration_seconds` firmado, no sólo `duration_seconds`.
- Permitir un primary con prior positivo cuando exista una recuperación v2 autorizada y enlazada.
- Exigir que el prior del replay sea el acumulado del primary.
- Añadir una prueba que ejecute el camino real de preparación v2 sin reescribir receipts.

### MEDIUM — La finalización exitosa acepta un contenedor `attempt` que es symlink

El endurecimiento físico añadido al cierre fallido no alcanza el camino exitoso:

- [`validate_root_terminal()`](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:2922) sí aplica `require_physical_directory`.
- [`validate_pair_status_against_roots()`](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:3106) también verifica físicamente el contenedor.
- Pero [`finalize_pair()`](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:3777) llama directamente a `validate_evaluated_root()` sin validar físicamente `attempt`, `primary` ni `replay`.
- [`validate_evaluated_root()`](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:2352) valida contenido, pero no la identidad física del root.

Reproducción con dos roots reales `EVALUATED_IMMUTABLE` y `attempt-alias -> physical-attempt`:

```text
ACTUAL_ROOT_VALIDATOR_ON_SYMLINK 66f3fb408592fbb1
FINALIZE_VIA_SYMLINK_PUBLISHED True .../attempt-alias/pair
PHYSICAL_PAIR_EXISTS True
```

Es decir, `finalize_pair(attempt-alias)` publica correctamente a través de un namespace alternativo, contrariando la topología física/no-alias del contrato.

Corrección necesaria: validar `attempt` como directorio físico antes de derivar sus roots y aplicar la misma garantía a `primary` y `replay` en la finalización exitosa.

### Estado de los cuatro findings R466

1. **Cadena Git:** corregida estructuralmente, pero la recuperación sigue bloqueada operacionalmente por el finding HIGH.
2. **Cierre fallido de ambos roots:** corregido; hay validación física, inventario cerrado y bindings direccionales.
3. **Traversal/symlink del intento previo:** corregido en recuperación; queda la brecha independiente del camino exitoso descrita arriba.
4. **Finalize acumulativo 899+2:** corregido para reanudaciones directas y firmado antes del rename; no corrige la continuidad temporal entre v1 y v2.

### Verificación ejecutada

```text
CUDA_VISIBLE_DEVICES=''
OMP_NUM_THREADS=4
pytest -q tests/test_wave60_frozen_policy_transport.py --basetemp=<mktemp dentro del workspace>
45 passed in 70.99s
maximum RSS: 873216 KiB
swap: 0
```

También pasaron:

- `git diff --check`
- `py_compile` de los cinco paths del contrato

No ejecuté la regresión amplia Wave56–60 porque estaba condicionada a que la revisión focal quedara limpia.

### Identidad auditada

- Commit: `b1f61a92f12633738882bb369a314fc243a5649d`
- Parent: `4c28438da90831056de95b043fae1d58a02c8b02`
- Tree: `46693c55582709e7cbc5908a2173ebbdec52d4a8`
- Diff del candidato: 2 archivos, 22 inserciones y 1 eliminación.
- Worktree final: limpio.
- Temporales `.r467-*`: ninguno restante.

SHA-256 físicos:

```text
40688554e7d97de9d065b34930303324fdbad6f2f55e4745536751ebac588da0  wave60_frozen_policy_transport.py
db31827fa03a97164e5efea27673527c39a6a71166900228492ee92d56235a27  run_wave60_frozen_policy_transport.py
c2ffafbda6234e2d7c92cc829b8f5634c9f5085a7b965f8e280243c31ef5591b  _wave60_phase_worker.py
6bdc63de19286a79704ec5eed9c3b381a2c7064f56fe14877c21a6106e3091fe  prepare_wave56_fresh.py
2f4c7f58e8287e529eab4b19154d9d70f07ec4c8bf68e0d152153c1fcbdcb746  test_wave60_frozen_policy_transport.py
```

No modifiqué archivos ni generé artefactos de auditoría.
