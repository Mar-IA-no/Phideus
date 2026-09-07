# R557 — Reauditoría independiente final del dual native freeze

```text
PASS — 0 HIGH / 0 MEDIUM / 0 LOW.

Hallazgos R553/R555 cerrados:

- Recomputé independientemente el NPZ: 64 grafos, 192 estados, 191 convergencias, p99 RMSE `1.7497836e-3`, máximo `0.1281575`, gradientes conformes y resultado K192 `FAIL`. Coincide campo por campo con el reporte. El checker ahora deriva summary, R11, conteos y `design_state` desde raw ([checker:1056](</mnt/m2-1TB/Phideus/experiments/geometria_proporcional/check_proportional_dual_native_freeze.py:1056>), [checker:2013](</mnt/m2-1TB/Phideus/experiments/geometria_proporcional/check_proportional_dual_native_freeze.py:2013>), [checker:2042](</mnt/m2-1TB/Phideus/experiments/geometria_proporcional/check_proportional_dual_native_freeze.py:2042>)).
- R11 falla efectivamente; los otros 29 predicados pasan y la adjudicación resultante es exclusivamente `SET_VALUED_FREEZE_ONLY_VALID` ([checker:1861](</mnt/m2-1TB/Phideus/experiments/geometria_proporcional/check_proportional_dual_native_freeze.py:1861>), [scientific_report.json:1](</mnt/m2-1TB/Phideus/data/geometria_proporcional/proportional_dual_native_freeze_v1/run_a/scientific_report.json:1>)).
- Las siete variaciones de R555 son rechazadas por sus predicados correspondientes y están congeladas en la suite ([checker:1705](</mnt/m2-1TB/Phideus/experiments/geometria_proporcional/check_proportional_dual_native_freeze.py:1705>), [checker:1748](</mnt/m2-1TB/Phideus/experiments/geometria_proporcional/check_proportional_dual_native_freeze.py:1748>), [checker:1765](</mnt/m2-1TB/Phideus/experiments/geometria_proporcional/check_proportional_dual_native_freeze.py:1765>), [checker:1833](</mnt/m2-1TB/Phideus/experiments/geometria_proporcional/check_proportional_dual_native_freeze.py:1833>), [tests:26](</mnt/m2-1TB/Phideus/tests/test_proportional_dual_native_freeze.py:26>)).
- Los bindings literales requeridos incluyen config/compute/runtime del smoke y W49/W59; C1 comprueba presencia, hashes y contenido crítico contra el commit ligado ([checker:111](</mnt/m2-1TB/Phideus/experiments/geometria_proporcional/check_proportional_dual_native_freeze.py:111>), [checker:223](</mnt/m2-1TB/Phideus/experiments/geometria_proporcional/check_proportional_dual_native_freeze.py:223>)).
- El roster raw contiene exactamente 64 IDs de grafo con patrones `{w0,w1,w2}`, 0 estados solapados con la calibración K64 y run_a/run_b son idénticos array por array. Manifests y replay exigen rosters exactos y hashes completos ([checker:1957](</mnt/m2-1TB/Phideus/experiments/geometria_proporcional/check_proportional_dual_native_freeze.py:1957>), [checker:2051](</mnt/m2-1TB/Phideus/experiments/geometria_proporcional/check_proportional_dual_native_freeze.py:2051>)).
- Todos los claims permanecen en CPU, sin promoción ni GO/NO-GO ([coordinator config:14](</mnt/m2-1TB/Phideus/experiments/geometria_proporcional/configs/proportional_dual_native_freeze_v1.json:14>), [set config:89](</mnt/m2-1TB/Phideus/experiments/geometria_proporcional/configs/proportional_set_valued_native_freeze_v1.json:89>)).

Criterio de reejecución: Git y manifests prueban identidad e inmutabilidad de bytes, pero por sí solos no demuestran que el NPZ fue producido por el código ligado. Para cierre científico corresponde al menos un cotejo determinista independiente; no necesita formar parte de cada chequeo ligero posterior. Lo ejecuté en memoria contra HEAD: roster idéntico y las 15 arrays coincidieron exactamente con el NPZ canónico.

Checks ejecutados:

- Recomputación NumPy independiente desde `fixed_depth_raw.npz`: coincidencia total con el summary.
- Reejecución CPU de `run_fixed_depth_conformance`: `all_arrays_equal=true`.
- Siete mutaciones focales: las siete rechazadas.
- `--check-root .../proportional_dual_native_freeze_v1`: `PASS`.
- `pytest` focal dual-freeze + surrogate fidelity: `17 passed in 46.85s`.
- SHA-256 y comparación run_a/run_b: byte-exactos.
- `git status --short`: limpio antes y después.

No edité archivos ni usé GPU/CUDA.
```
