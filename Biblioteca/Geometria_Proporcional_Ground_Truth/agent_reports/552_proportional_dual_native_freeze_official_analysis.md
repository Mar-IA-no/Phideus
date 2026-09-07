# R552 — Análisis oficial de los dos design freezes nativos

**Fecha:** 2026-09-07  
**Régimen:** design freeze y preflight CPU-only; sin training, forward, draw experimental, monitor ni lockbox  
**Código del preflight final:** `ae627b2`  
**Artefacto canónico:** `data/geometria_proporcional/proportional_dual_native_freeze_v1/`

## Resultado técnico

El coordinador adjudicó:

```text
SET_VALUED_FREEZE_ONLY_VALID
```

Esto significa que el protocolo `MARGINAL/JOINT × HARD/CONTEXTUAL` queda listo
para implementar su runner. No significa que sus primitives hayan sido
ejecutadas ni que exista evidencia a favor de alguna celda. El freeze
`GENERIC/TYPED × WLS/IRLS` queda inválido bajo su objetivo compuesto K192. El
estado no promueve arquitectura ni constituye GO/NO-GO.

## Integridad del contrato

- `29/30` predicados pasaron; el único FAIL fue
  `R11_BASE_WEIGHTED_K192_CONFORMANCE`.
- `32/32` mutaciones negativas fueron atrapadas, incluidas reutilización del
  seed de calibración, K distinto de 192, retuning posterior, campos privados en
  path-shuffle, `cluster_id` en target-shuffle y unión de controles matched.
- `target_derangement_v1` reprodujo el digest congelado
  `d7aa2f128b6d42dbe7448415dcd8d4d69ca0ad8311394a5b1209ca9579e03904`.
- La fixture matched reconstruyó `U_common=4/5`, cobertura exacta `0,8`, mediante
  intersección de cinco máscaras.
- Run y replay fueron byte-exactos en los cinco archivos científicos por raíz;
  ambos manifests también coincidieron.

## Confirmación fresca K192

La confirmación usó seed `2026090731`, distinta de la calibración R550,
`64` grafos, `192` estados y tres patrones de peso base por grafo. Cubrió IID,
grouped y todos los tamaños `8..16`.

La extensión sigue siendo internamente correcta:

- error máximo Torch↔NumPy fixed-K: `1,1369e-13`;
- gradiente de relación: 24 probes, 191 coordenadas, una excluida, coseno
  mediano `1,0`, p95 relativo `7,04e-10`, cero inversiones;
- gradiente de peso: 23 probes, 184 coordenadas, ocho excluidas, coseno mediano
  `1,0`, p95 relativo `4,62e-8`, cero inversiones.

Sin embargo, el executor canónico convergió en `191/192` estados; uno agotó
`7500` iteraciones. Aun restringiendo la aproximación a los estados canónicos
convergidos, K192 obtuvo p99 RMSE `1,7498e-3` y máximo `1,2816e-1`, por encima
de los límites congelados `1e-4/1e-3`. El contrato auditado prohibía probar otro
K después de esta confirmación. Por eso la rama relacional se rechaza sin
training y sin atribuir el fallo a Torch, a NumPy o a un gradiente roto.

## Costo y replay

La ejecución oficial y replay tardaron juntos `21,77 s`, con pico RSS observado
de `1.034.473.472` bytes. El artefacto set-valued conserva una proyección
CPU-proporcionada de `120/420/1800 s` para rango bajo/central/alto de runner más
replay; es una estimación previa al runner, no tiempo medido.

No se usó ni consultó GPU. Tampoco se sustituyó una etapa GPU por una corrida
CPU larga: todo el goal permaneció por debajo del techo de 900 segundos por
preflight.

## Incidente de cobertura corregido

Una primera ejecución confirmatoria cubrió 64 grafos pero omitió `n=12` por el
orden de selección. Se la retiró del lugar canónico y se preservó, sin
sobrescribir, en `superseded_missing_n12/`. El checker fue corregido para
selección round-robin por cardinalidad y para exigir `8..16` como condición
efectiva. El artefacto canónico posterior cubre los nueve tamaños. Esta
corrección no cambió el veredicto K192.

## Lectura acotada

**Observación.** La rama set-valued satisface sus contratos declarativos,
fixtures, fuentes, fases y controles y queda lista únicamente para implementar
runner. La rama relacional no satisface su confirmación numérica fresca.

**Inferencia.** No conviene invertir ahora en el entrenamiento relacional dual
solver: su surrogate fijo no representa de forma suficientemente uniforme al
executor canónico con pesos aprendidos dentro del universo confirmado. Una vía
futura tendría que cambiar la estrategia —por ejemplo, diferenciación implícita
o un objetivo robusto distinto— y entrar como alternativa nueva, no retuning de
este freeze.

**Siguiente discriminante CPU.** Implementar y auditar el runner set-valued,
manteniendo cerrados posterior_fit, policy_fit, decision_select y monitor. No se
abre aún el draw fresco ni el monitor durante esa implementación.

## Trazabilidad

- `run_a/scientific_report.json`:
  `518c1037b616daf31b09153da3ba628167f012c80f890a40c28dcbd6cc2951c5`.
- `run_a/fixed_depth_raw.npz`:
  `0632663e032c6e71f328c6651c49c303cd372633ee2b039328271970e35b3c08`.
- `replay_comparison.json`:
  `3025015bf37599a009aaa579d5572a79f851f7787d513239755c192cbd657286`.
- `manifest.json`:
  `f8856f839c4ee1b588f5b8220832d6431fdd31b8b89ebddfd73dbbcb49efb04b`.
