# R550 — Diagnóstico CPU de profundidad IRLS con pesos base aprendidos

**Fecha:** 2026-09-07  
**Régimen:** diagnóstico numérico CPU-only; sin draw experimental, training, forward, monitor ni lockbox  
**Código del preflight K64:** `03144284fc1d9c9236641983016f9393a0fc1841`  
**Código del depth scan:** `d48b6db`  
**Artefactos:**

- `data/geometria_proporcional/proportional_dual_native_freeze_v1/depth_calibration_k64/`
- `data/geometria_proporcional/proportional_dual_native_freeze_v1/depth_scan_v1/`

## Pregunta

R354 había certificado el surrogate Huber-IRLS de profundidad fija `K=64` sólo
con pesos base unitarios. El nuevo freeze relacional requiere que WLS e IRLS
reciban el mismo `raw_reliability` aprendido. R550 pregunta si la extensión
base-weighted reproduce primero una referencia NumPy independiente, conserva
gradientes correctos respecto de relación y peso y, además, aproxima al executor
canónico convergido dentro de los umbrales históricos.

## Resultado del preflight K64

La implementación Torch y la referencia NumPy fixed-K coincidieron en `96`
estados (`32` grafos × tres patrones no unitarios) con error máximo
`1,1369e-13`. Los grafos cubrieron mecanismos IID/grouped y todos los tamaños
`8..16`. El executor canónico convergió en `96/96` estados.

Los gradientes también fueron conformes:

| Familia | Probes | Coordenadas | Excluidas con motivo | Coseno mediano | p95 error relativo | Inversiones de signo |
|---|---:|---:|---:|---:|---:|---:|
| `corrected_log_ratio` | 16 | 127 | 1 | 1,0 | `7,84e-10` | 0 |
| `raw_reliability` | 14 | 112 | 16 | 1,0 | `4,06e-7` | 0 |

No obstante, `K=64` falló la aproximación al executor convergido: p99 RMSE
`3,7607e-3` y máximo `7,5185e-2`, frente a límites `1e-4/1e-3`. El máximo
provino de un único estado grouped; su executor canónico necesitó `136`
iteraciones. En consecuencia, el preflight adjudicó `29 PASS / 1 FAIL`, con
`R11_BASE_WEIGHTED_K64_CONFORMANCE=FAIL`, y el estado técnico fue
`SET_VALUED_FREEZE_ONLY_VALID`. La suite de mutaciones atrapó `30/30` cambios y
ambas fixtures set-valued pasaron.

La corrida y su replay fueron byte-exactos. Juntas consumieron `5,19 s` y un
pico observado de `1.033.834.496` bytes RSS. No se usó ni consultó GPU.

## Scan de profundidad

Sobre los mismos `96` estados se compararon seis profundidades contra el
executor canónico. Este scan calibra valor numérico; no certifica todavía
gradientes del nuevo K ni habilita training.

| K | p99 RMSE | máximo RMSE | Umbral de valor |
|---:|---:|---:|:---:|
| 64 | `3,7607e-3` | `7,5185e-2` | FAIL |
| 96 | `1,4835e-3` | `2,9642e-2` | FAIL |
| 128 | `1,2184e-4` | `2,4095e-3` | FAIL |
| 160 | `1,3968e-6` | `1,4342e-6` | PASS |
| 192 | `1,3968e-6` | `1,4342e-6` | PASS |
| 256 | `1,3968e-6` | `1,4342e-6` | PASS |

Las iteraciones canónicas tuvieron mínimo `6`, mediana `21`, p99 `71,4` y
máximo `136`. `K=160` fue la primera profundidad ensayada que pasó; `K=192` se
propone como profundidad confirmatoria conservadora, sin afirmar que sea mínima
en los enteros no ensayados. La propuesta debe congelarse antes de un draw
numérico nuevo, validar nuevamente valor y ambos gradientes y quedar rechazada
sin retuning si falla esa confirmación.

El scan y replay fueron byte-exactos; tardaron `2,81 s` y `2,73 s`, con pico
`1.033.850.880` bytes RSS. No se usó ni consultó GPU.

## Alcance de la inferencia

**Observación.** La extensión base-weighted está correctamente implementada
como trayectoria fixed-K y es diferenciable respecto de sus dos inputs, pero
`K=64` no alcanza convergencia uniforme en el rango sintético congelado.

**Hipótesis de diseño.** El fallo es de profundidad, no una discrepancia
Torch↔NumPy ni un gradiente roto: desde `K=160` todos los estados observados
entran holgadamente en los límites.

**Siguiente contraste.** Congelar `K=192` y ejecutar una confirmación nueva,
independiente de este conjunto de calibración, con los mismos límites y sin
reselección. Hasta que eso ocurra, el freeze relacional permanece inválido.
Esto no promueve arquitectura ni constituye GO/NO-GO.

## Trazabilidad

- Preflight K64: `12` archivos, `29.529` bytes; replay `PASS`.
- Depth scan: `7` archivos, `18.412` bytes; replay `PASS`.
- `depth_calibration_k64/run_a/scientific_report.json`:
  `3d98d642665151e481c7f6076baa8a341ea514f8cc1840048e985a2b8c823961`.
- `depth_scan_v1/run_a/summary.json`:
  `7ef49c3562cbb65db7e365f3a55fda7c384d186f9cea63dba399881bc8d6e12b`.
- `depth_scan_v1/run_a/raw.npz`:
  `aab7192dde4b05934560f8297efaaf062dcb05782020b412c3d4b4a67933fb4d`.
