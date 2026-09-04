# Plan CPU — interfaz condicional de riesgo para el gate topológico

**Fecha:** 2026-09-04
**Estado:** diseño inicial; implementación pendiente
**Régimen:** predictor de beneficio congelado, cuatro realizaciones frescas y disjuntas
**Autoridad:** discrimina si incertidumbre condicional convierte señal topológica en política selectiva; no promueve arquitectura ni decide GO/NO-GO

## Pregunta

R360 estableció una separación que no conviene volver a mezclar. Las features de
localización topológica superaron al gate base y al sham matched en grouped y
balanceado, pero la regresión de media más un threshold global no acreditó
seguridad IID. El próximo contraste conserva exactamente el predictor topology
ya abierto y cambia sólo la interfaz de riesgo.

Un techo aditivo constante `mu_alpha(x)+q` sería algebraicamente equivalente a
exigir que la ventaja predicha supere un threshold global. Esa receta ya fue
adjudicada. Para producir un contraste nuevo, el límite debe depender de la
incertidumbre pública de cada vista y alpha:

```text
upper_alpha(x) = mu_alpha(x) + q * sigma_alpha(x).
```

`mu_alpha` queda congelado en R360. Sólo `sigma_alpha` se aprende y calibra en
datos nuevos.

## Fuente y fases

La fuente inmutable es
`data/geometria_proporcional/proportional_graph_topology_localization_gate_v1/`,
manifest SHA-256
`5d7659fe74cfe18ec79609935568d63bdcb90da19f96224b2c182f673d4cbaed`.
Se reutilizan su gate `topology_augmented`, las veinticuatro features, los
checkpoints post-IRLS, cuatro brazos, dos seeds neuronales y cinco alphas.

Cuatro realizaciones disjuntas cumplen roles no intercambiables:

1. **risk fit**, seed `2026090809`: ajusta la escala residual sólo sobre las
   vistas IID;
2. **risk calibration**, seed `2026090817`: con modelos ya congelados, calcula
   el cuantíl superior simultáneo IID;
3. **policy selection**, seed `2026090829`: decide si cada política condicional
   se despliega o cae a identidad bajo una banda de media IID;
4. **adjudication**, seed `2026090837`: no se materializa hasta congelar los
   tres pasos anteriores y todos sus hashes.

Cada fase exige al menos `220` masters IID/grouped completos. Tests y preflight
usan otras seeds. No se cambia una seed por conteo u outcome.

## Predictor de beneficio congelado

Para cada `arm × alpha`, el gate topology de R360 produce
`mu_alpha(x)`, la predicción de delta RMSE frente a identidad. Sus coeficientes,
normalización y alpha grid no se reajustan. El modelo recibe las mismas quince
features base y nueve topológicas verdaderas de R360.

Los targets de risk fit son errores absolutos honestos del predictor congelado:

```text
z_alpha = log(|delta_real_alpha - mu_alpha| + 1e-6).
```

Una ridge lineal de cuatro salidas predice `z_alpha`; luego
`sigma_alpha=exp(z_hat_alpha)`, clipped a `[1e-6,1]` por estabilidad. Lambdas
`{0,.01,.1,1,10,100}` se eligen con cinco folds por master dentro de risk fit.
Grouped se preserva para diagnóstico, pero no ajusta una escala que reclama
jurisdicción IID.

## Familias de riesgo

Todas comparten el mismo `mu_alpha` topology. Sólo cambia cómo estiman
`sigma_alpha`:

1. `constant_scale`: `sigma=1`; muestra la equivalencia con un margen aditivo
   global;
2. `public_base_scale`: ridge sobre las quince features heredadas;
3. `topology_scale`: ridge sobre las veinticuatro features;
4. `topology_permuted_scale`: dieciséis ridges de veinticuatro inputs que
   conservan features base y multiconjunto de corrección, pero destruyen la
   asignación topológica como en R360.

Cada réplica permutada tiene outputs, targets, folds, lambdas y presupuesto
iguales a `topology_scale`. Sus métricas se promedian por vista; nunca se elige
la mejor réplica. Esta comparación aísla si la localización ayuda a estimar
riesgo, no si vuelve a ayudar al predictor de media.

## Calibración del límite superior

Risk calibration contiene nuevos pares, pero sólo las vistas IID construyen el
score. Para cada familia o réplica:

```text
s_i = max_alpha ((delta_real_i,alpha - mu_i,alpha) / sigma_i,alpha).
```

Con miscoverage fijo `0.10`, se usa el orden estadístico split-conformal de
nivel `ceil((n+1)*0.90)`. El mismo `q` cubre simultáneamente los cuatro alphas
de una vista. No se barre ni selecciona el nivel de cobertura.

Para una futura vista IID exchangeable, el evento conjunto tiene cobertura
marginal finita bajo el contrato split-conformal. No es cobertura condicional
entre las vistas donde la política actúa, no cubre grouped, no corrige cambio de
ley y no garantiza la media de una realización finita.

## Acción y policy selection

Cada familia calcula cuatro `upper_alpha`. Se elige el alpha no identidad con
menor upper sólo si ese mínimo es estrictamente negativo; de lo contrario se
copia identidad. No hay threshold adicional.

Policy selection aplica las políticas ya calibradas a una tercera realización.
Con `2.000` bootstraps pareados por master, una política se despliega sólo si su
límite superior 95% para el delta medio IID frente a identidad es no positivo.
Si no, queda congelada como identidad. Para los dieciséis shams la regla se
aplica por réplica y luego se promedian métricas; no se escoge la réplica más
favorable.

Este segundo firewall adjudica media IID bajo la realización de selección; no
se presenta como consecuencia automática de la cobertura marginal por vista.

## Adjudicación

Sobre una cuarta realización se reporta por brazo, IID, grouped y balanceado:

- política calibrada antes y después del firewall de policy selection;
- delta frente a identidad;
- `topology_scale - constant_scale`;
- `topology_scale - public_base_scale`;
- `topology_scale - topology_permuted_scale`;
- frecuencia de acción y distribución de alphas;
- daño entre vistas actuadas, sólo como diagnóstico condicional sin garantía;
- cobertura empírica simultánea del límite por vista;
- interaction grouped menos IID;
- oracle privado por vista.

Los IC de adjudicación son pointwise. El master es la unidad bootstrap; se
promedian primero seeds y luego el par IID/grouped para balanceado. Los mismos
`2.000` índices sirven a todas las políticas. Una no convergencia invalida el
master afectado sin complete-case silencioso.

## Lecturas admisibles

- si topology risk supera identidad en grouped con límite negativo, conserva
  límite IID no positivo y supera public-base/permuted, la escala condicional
  aporta una interfaz selectiva dentro de esta ley;
- si constant y topology son equivalentes, la heteroscedasticidad no agrega
  valor sobre el threshold global ya estudiado;
- si topology no supera permuted, no se atribuye valor de riesgo a la
  localización aunque el predictor de media siga siendo útil;
- si policy selection elige identidad, no se relaja el firewall sobre
  adjudication;
- cobertura empírica o conformal marginal no equivale a cero daño condicional,
  autoridad física, transferencia, arquitectura promovida o GO/NO-GO.

## Artefactos y recursos

Se preservan inputs públicos, features reales y sham, `mu`, targets residuales,
folds, modelos de escala, `sigma`, scores, cuantiles conformales, acciones,
bandas, alpha metrics, estados forward/solver, bootstraps, receipts, manifests y
replay. Cada fase verifica el freeze anterior y la ausencia de fases futuras.

La ejecución usa `CUDA_VISIBLE_DEVICES=''`, un thread, máximo `12 min` y
`4 GiB` por fase. Todo trabajo GPU continúa en cola hasta que Mariano revoque
explícitamente la suspensión.
