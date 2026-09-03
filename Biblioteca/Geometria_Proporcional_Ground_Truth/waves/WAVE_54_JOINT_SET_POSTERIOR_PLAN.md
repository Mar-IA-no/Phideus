# Ola 54 — plan de posterior conjunto regularizado

> **Estado:** `AUDITED-FROZEN / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-03
> **Antecedente:** `WAVE_53_UNCERTAINTY_AWARE_POLICY_CLOSED.md`

## Pregunta

¿La pérdida decisional observada en la Ola 53 proviene de representar una región
compatible como cuatro pertenencias Bernoulli independientes? La Ola 54 mantiene
congelados encoder, logits, splits, utilidades, pérdida y población, y sustituye
solamente esa factorización por una distribución conjunta regularizada sobre los
quince subconjuntos no vacíos.

El estimando sigue siendo de desarrollo sobre datos históricamente abiertos. No
prueba una utilidad natural, una geometría física ni superioridad causal del
encoder. No lee lockbox, no usa GPU y no promueve arquitectura.

## Población y cronología

Se reutilizan los logits crudos separados por split de la Ola 52, los parámetros
Platt y el split de la Ola 53, todos ligados por hash. Antes del análisis, un
preparador ciego y sin lógica estadística materializa dos artefactos físicamente
separados: `fit_select_bundle.npz` y `sealed_monitor_bundle.npz`, más un manifest
de hashes. El preparador copia por lista cerrada logits, targets y metadatos;
no ajusta, selecciona, evalúa ni produce métricas. Se versiona antes de ejecutarse.
Las unidades siguen siendo `pair_token`; las 24 políticas de un token permanecen
juntas.

El orden operativo es: versionar preparador, runner, primitives, tests y config
con hashes upstream; ejecutar el preparador; incorporar a la config los hashes
del manifest y de ambos bundles; volver a versionar; y sólo entonces permitir la
corrida analítica. El runner falla cerrado si Git está sucio o si cualquiera de
esos hashes difiere.

- `calibration_fit` (`192` tokens): ajusta parámetros para cada estructura y
  cada regularización candidata.
- `decision_select` (`192` tokens): elige una regularización por estructura
  minimizando NLL de conjunto exacto en su población primaria `NEAR_RIVAL` con
  cardinalidad mayor o igual que dos; no se refitea con este split. Una selección
  sobre los `192` tokens completos se conserva como sensibilidad secundaria.
- `val_monitor` (`384` tokens): se lee una sola vez después de escribir el
  freeze de selección. La población primaria conserva `NEAR_RIVAL` con
  cardinalidad compatible mayor o igual que dos (`148` tokens observados en la
  Ola 53); `all_in_catalog` queda como sensibilidad.

El runner de análisis sólo recibe `fit_select_bundle.npz` al ajustar. Escribe
`selection_freeze.json` con estructuras, grilla, parámetros, hashes seleccionados
y el hash esperado del bundle sellado. Recién entonces abre
`sealed_monitor_bundle.npz`. `access_receipt.json` registra archivos, hashes y
orden de acceso. La separación es física, no una convención de acceso por clave.

## Familia de posterior

Sea `z in R^4` el vector de logits ensemble y `S` uno de los quince conjuntos
no vacíos. El modelo usa una energía condicional

```text
score(S,z) = sum_i a_i S_i z_i
           + sum_{i<j} J_tilde_ij S_i S_j
           + c_|S|
p(S|z) = softmax_S score(S,z)
```

`c_1=0` y `sum_{i<j} J_tilde_ij=0` fijan los dos gauges. Cinco coordenadas libres
determinan las primeras cinco interacciones y la sexta es su suma negativa. Así,
la componente uniforme de seis interacciones, indistinguible de una función de
cardinalidad, queda absorbida en `c_k`; sólo se interpretan contrastes heterogéneos
entre pares. Se aprenden `c_2,c_3,c_4`. El punto de referencia de la
regularización es el producto Bernoulli raw condicionado a no vacío:
`a_i=1`, `J_tilde_ij=0`, `c_k=0`. No se agregan interceptos por familia, para evitar
redundancia entre bias unary y cardinalidad. La dimensión máxima es `12`: cuatro
slopes unary, cinco contrastes de interacción y tres sesgos de cardinalidad, es
decir `12` parámetros identificados.

El ajuste minimiza NLL media de conjunto exacto más
`0.5 * lambda * ||theta-theta_ref||^2`, en `float64`, mediante L-BFGS-B con
gradiente analítico. La grilla cerrada es
`lambda in {1e-4, 1e-3, 1e-2, 1e-1, 1, 10}`. Cada fit parte de `theta_ref`, debe
converger y producir parámetros, probabilidades y gradientes finitos; cualquier
fallo aborta, sin fallback silencioso.

## Brazos y controles

1. `hard_set_policy`: baseline binario de la Ola 53.
2. `independent_raw`: producto Bernoulli de sigmoid raw condicionado a no vacío.
3. `independent_platt`: producto Bernoulli Platt de la Ola 53.
4. `joint_unary`: aprende sólo los cuatro slopes `a_i`.
5. `joint_unary_cardinality`: agrega `c_2,c_3,c_4`.
6. `joint_full`: agrega cinco contrastes de interacción `J_tilde_ij` con suma
   cero; es la candidata primaria.
7. `empirical_set_prior`: frecuencia de los quince conjuntos en
   `calibration_fit` con suavizado Dirichlet simétrico `alpha=1`; no usa logits y
   controla cuánto compra la prevalencia global.
8. `joint_full_target_shuffled`: misma receta que `joint_full`, pero ajustada con
   targets rotados sin puntos fijos dentro de `design_stratum × cardinalidad` en
   `calibration_fit`; preserva prevalencia y cardinalidad, rompe la asociación
   condicional logits-target y nunca participa en selección del brazo real.
9. `oracle_set_then_utility`: conjunto verdadero más utilidad; referencia
   privilegiada, no deployable ni techo universal.

Cada posterior induce acciones mediante la misma pérdida ordinal y la misma
penalización incompatible `1.25` de las Olas 52–53. No se retoca utilidad,
desempate, composición ni target.

## Selección y métricas

Cada estructura aprendida elige su propio `lambda` sólo por NLL media de conjunto
exacto en la población primaria de `decision_select`; empate: mayor `lambda`,
para favorecer el modelo más regularizado. Se calcula además cuál habría sido la
selección sobre todo `decision_select`; si difiere y cambia el signo de un
contraste primario en monitor, el resultado queda marcado como selector-sensitive
y no habilita una conclusión fuerte. El brazo `joint_full_target_shuffled` usa el
`lambda` primario seleccionado por `joint_full`, de modo que el control no recibe
búsqueda adicional.

El comparador `best_independent` también se congela en `decision_select`: entre
`independent_raw` e `independent_platt` se elige el de menor NLL de conjunto
exacto en la población primaria, con empate hacia raw. Monitor no elige baseline.

Se reportan por split de evaluación y por población:

- NLL de conjunto exacto y top-1 de conjunto;
- Brier marginal y NLL marginal derivados de `p(S|z)`;
- distribución de cardinalidad predicha, error L1 y MAE de cardinalidad esperada;
- accuracy de acción, tasa compatible, regret medio y promedio del peor regret
  por token bajo las 24 políticas;
- parámetros, norma L2 de `J_tilde`, estabilidad del optimizador y masa asignada
  a clases no observadas en fit.

Los contrastes usan bootstrap pareado por `pair_token`, con los mismos `2.000`
índices y seed `5407` para todas las métricas. Las políticas son mediciones fijas,
no réplicas. NLL de conjunto y métricas decisionales se agregan primero por token.

## Criterio diagnóstico conjunto

`joint_full` queda marcado como prometedor para una réplica posterior, sin
promoción, sólo si en la población primaria de monitor:

1. reduce NLL de conjunto exacto frente a `best_independent`, seleccionado antes
   en `decision_select`, por al menos `0.02`, con IC95 del delta debajo de cero;
2. reduce el error L1 de cardinalidad frente al mejor independiente por al menos
   `0.05`;
3. supera a `joint_unary_cardinality` en NLL por al menos `0.01`, con IC95 debajo
   de cero, para atribuir valor a contrastes heterogéneos de interacción y no
   sólo a cardinalidad;
4. reduce regret frente a `hard_set_policy` por al menos `0.02`, con IC95 debajo
   de cero;
5. no pierde más de `0.01` de accuracy ni reduce tasa compatible frente al
   baseline duro;
6. supera a `empirical_set_prior` y a `joint_full_target_shuffled` en accuracy de
   acción por al menos `0.05` cada uno;
7. en `decision_select`, la masa media que `joint_full` asigna a conjuntos no
   observados en `calibration_fit` no supera la del baseline `independent_raw` en
   más de `0.02`; el mismo diagnóstico se reporta en monitor, sin reinterpretar
   las clases ausentes como evidencia positiva;
8. una ejecución externa en proceso y directorio independientes reconstruye fits,
   selección, controles, bootstrap y outputs y coincide exactamente.

El soporte exacto es escaso: `calibration_fit` observa `10/15` conjuntos,
`decision_select` `9/15`, y hay clases con una sola observación. Por eso, incluso
si pasa el criterio anterior, el claim queda restringido al soporte histórico
observado y no se extiende a los cinco conjuntos ausentes de fit. El patrón es
deliberadamente exigente y no equivale a GO/NO-GO. Si mejora NLL y
cardinalidad pero no regret, la representación conjunta habrá ganado como modelo
probabilístico sin resolver la policy. Si cardinalidad explica todo y las
interacciones no agregan, el siguiente objeto será una cabeza de cardinalidad,
no una arquitectura relacional más compleja. Si el prior sin logits iguala al
modelo, la señal aparente será distribución global, no relación condicional.

## Abstención

Como sensibilidad secundaria, `joint_full` usa el máximo sobre políticas de su
riesgo mínimo esperado y aplica las fronteras `50/75/90%` ajustadas sólo en
`decision_select`. Se reportan cobertura efectiva, regret selectivo y AURC. No
entra al patrón principal y no se interpreta como garantía conformal.

## Artefactos

- `analysis_freeze.json` antes de labels/predicciones autorizadas;
- `input_bundle_manifest.json` con separación física fit/select vs monitor;
- `selection_freeze.json` antes de claves de monitor;
- manifest de inputs y acceso por clave;
- parámetros y diagnósticos por estructura y `lambda`;
- masas conjuntas, marginales, acciones, riesgos y scores selectivos;
- métricas por token y token×política;
- índices completos del bootstrap pareado;
- controles y mappings;
- runtime, package manifest, artifact manifest y replay exacto;
- JSON estricto: ningún `NaN` o infinito no estándar.

## Archivos previstos

- `src/geometria_proporcional/wave54_joint_set.py`
- `experiments/geometria_proporcional/prepare_wave54_inputs.py`
- `experiments/geometria_proporcional/run_wave54_joint_set.py`
- `experiments/geometria_proporcional/configs/wave54_joint_set.json`
- `tests/test_wave54_joint_set.py`

## Límite operativo

CPU-only. La implementación y la corrida requieren primero auditoría
independiente del plan y resolución de findings sustantivos. Un resultado que
sugiera modificar o reentrenar el encoder se diseña en otra ola; antes de usar
GPU se informa al usuario objetivo, duración y VRAM estimados.

## Resolución de auditoría independiente

R302 emitió `REVISE` por cuatro problemas materiales. La revisión: (1) reemplaza
seis interacciones crudas por cinco contrastes de suma cero, separando el gauge
uniforme de cardinalidad; (2) restaura una barrera física con bundles separados y
un preparador ciego versionado; (3) vuelve vinculante la masa sobre clases no
observadas y restringe el claim al soporte histórico; y (4) selecciona `lambda`
en la misma población primaria que motiva el experimento, dejando la selección
global como sensibilidad. Replay acredita esta nueva capa sobre logits congelados;
no revalida el entrenamiento upstream que los produjo.

R303 reauditoró focalmente las cuatro resoluciones y dio `PASS`: confirmó los
doce parámetros identificados, la separación física y el orden de binding, el
criterio vinculante de masa no observada y la selección primaria de `lambda`.
No abrió findings nuevos ni emitió GO/NO-GO.
