# Plan CPU — gate prospectivo de cola firmada

**Fecha:** 2026-09-04  
**Estado:** diseño congelado; implementación y ejecución pendientes  
**Régimen:** predictor de media congelado, comparación pareada de interfaces y cuatro realizaciones frescas  
**Autoridad:** discrimina transporte prospectivo de la cola firmada; no promueve arquitectura ni decide GO/NO-GO

## Pregunta

R361 mostró que la escala del residuo absoluto preserva seguridad mediante
abstención casi total. R362, sobre esos mismos artefactos ya abiertos, mostró
que estimar la cola superior del residuo firmado mejora el orden OOF y duplica
la acción, aunque sigue siendo extremadamente selectiva y no supera al control
topológico permutado en adjudicación. El próximo corte no vuelve a analizar ese
test. Repite la comparación completa sobre cuatro realizaciones nuevas y
disjuntas.

La pregunta principal es si una interfaz unilateral congelada antes del test
conserva no-daño IID y beneficio grouped frente a identidad, y si su diferencia
con la escala absoluta y los controles matched transporta. Una mayor tasa de
acción por sí sola no cuenta como mejora.

## Fuente inmutable y aislamiento

El predictor de media, los checkpoints, las veinticuatro features topology, las
quince features base, los cuatro brazos, los dos seeds neuronales y los cinco
alphas provienen de
`data/geometria_proporcional/proportional_graph_topology_localization_gate_v1/`,
manifest SHA-256
`5d7659fe74cfe18ec79609935568d63bdcb90da19f96224b2c182f673d4cbaed`.
No se reajusta `mu_alpha(x)`.

Las seeds nuevas, verificadas como ausentes del corpus al congelar este plan,
tienen roles no intercambiables:

1. **risk fit**, seed `2026091109`: genera IID/grouped pareados y ajusta ambos
   modelos de riesgo sólo con IID;
2. **risk calibration**, seed `2026091117`: calcula las correcciones conformales
   con los modelos ya congelados;
3. **policy selection**, seed `2026091129`: aplica el firewall de media IID sin
   acceso a adjudication;
4. **adjudication**, seed `2026091137`: se materializa únicamente después de
   congelar y verificar los tres roles anteriores.

Cada rol exige al menos `220` masters completos. Preflight y pilotos usan seeds
distintas. Ninguna seed se reemplaza por conteo, convergencia marginal u
outcome.

## Comparación pareada de interfaces

Para cada `arm × alpha`, el target observado es el mismo residuo firmado:

```text
r_alpha(x) = delta_real_alpha(x) - mu_alpha(x).
```

La interfaz **signed-tail** ajusta el cuantil condicional `0,90` de `r_alpha`
con regresión lineal de pinball loss. Usa `QuantileRegressor(solver="highs")`,
cinco folds por master, grilla L1 `{0, 0.0001, 0.001, 0.01, 0.1, 1}` y una
regularización compartida entre los cuatro outputs. Empates dentro de `1e-12`
eligen la regularización mayor.

La interfaz **absolute-scale** reproduce R361 sobre el mismo risk fit:

```text
z_alpha(x) = log(|r_alpha(x)| + 1e-6).
```

Ajusta ridge multisalida con lambdas `{0, 0.01, 0.1, 1, 10, 100}`, cinco folds
por master y `sigma=clip(exp(z_hat), 1e-6, 1)`. Ambas interfaces reciben los
mismos masters, outputs, folds y familias de features; sus pérdidas y
regularizadores son propios porque representan targets distintos. No se
trasladan coeficientes ni cuantiles desde R361/R362.

## Familias y controles matched

Cada interfaz contiene:

1. `constant`: intercepto o escala constante;
2. `public_base`: quince features públicas;
3. `topology`: las veinticuatro features verdaderas;
4. `topology_permuted`: dieciséis réplicas de igual dimensión que preservan las
   features base y el multiconjunto firmado/absoluto de corrección, pero
   destruyen su asignación topológica.

La seed de controles es `2026091141`. La misma permutación por réplica se usa en
signed-tail y absolute-scale. Cada réplica conserva targets, outputs, folds y
presupuesto matched. Nunca se elige la réplica más favorable: primero se aplica
su propia política y firewall, luego se promedian métricas por vista.

## Calibración conformal

Sólo IID de risk calibration construye scores simultáneos sobre los cuatro
alphas no identidad.

Para signed-tail:

```text
s_i = max_alpha(r_i,alpha - u_hat_i,alpha)
upper_alpha = mu_alpha + u_hat_alpha + q_signed
```

Para absolute-scale:

```text
s_i = max_alpha(r_i,alpha / sigma_i,alpha)
upper_alpha = mu_alpha + q_absolute * sigma_alpha
```

Ambas usan miscoverage fijo `0,10` y el orden estadístico finito
`ceil((n+1)*0,90)`. No se selecciona nivel de cobertura. La acción elige el
alpha no identidad con menor upper sólo cuando es estrictamente negativo; en
otro caso copia identidad.

La garantía es marginal IID por vista y simultánea sobre cuatro alphas bajo
exchangeability. No es cobertura condicional entre vistas actuadas, no cubre
grouped ni autoriza transferencia física.

## Firewall y freeze

Policy selection usa `2.000` bootstraps pareados por master, seed
`2026091147`, y despliega una política sólo si el percentil superior `95%` del
delta medio IID frente a identidad es no positivo. La regla se aplica por
`interfaz × familia × brazo`; cada réplica permutada pasa por su propio
firewall. Una política rechazada queda congelada como identidad. No se relaja
ninguna regla después de observar adjudication.

Antes de materializar la cuarta seed se congelan config resuelta, commit,
source hashes, inputs, folds, permutaciones, modelos, scores, cuantiles,
acciones de calibración, bootstraps y decisiones del firewall. Cada transición
verifica el conjunto exacto de archivos anteriores y la ausencia de fases
futuras.

## Adjudicación predeclarada

Con `2.000` bootstraps por master y seed `2026091153`, se reporta por brazo para
IID, grouped y el promedio balanceado:

- signed topology desplegado frente a identidad;
- signed topology menos absolute topology, ambos después de su firewall;
- signed topology menos signed constant y signed public-base;
- signed topology menos el promedio de las dieciséis políticas
  topology-permuted;
- las comparaciones análogas de absolute-scale como control de reproducción;
- frecuencia de acción, distribución de alphas y solapamiento de acciones entre
  interfaces;
- cobertura empírica simultánea, daño entre vistas actuadas e interacción
  grouped menos IID;
- oracle por vista, sólo diagnóstico y nunca desplegable.

El master es la unidad bootstrap. Primero se promedian los dos seeds neuronales
y después IID/grouped para el estimando balanceado. Los mismos índices se usan
en todos los contrastes. Los intervalos de adjudicación son pointwise; no se
presentan como corrección familiar ni como umbral de promoción.

## Patrones informativos y límites

- Beneficio grouped o balanceado frente a identidad sólo es evidencia de
  transporte si aparece en la seed fresca; la selección previa no lo garantiza.
- No-daño IID se describe por el punto y el intervalo de adjudicación, separado
  del firewall que operó sobre otra realización.
- Una diferencia favorable frente a absolute-scale atribuye valor a la
  dirección del residuo; una diferencia frente al promedio permutado atribuye
  valor adicional a la localización. Ninguna sustituye a la otra.
- Si signed-tail actúa más pero no mejora efecto o controles, la mayor cobertura
  operativa no acredita la interfaz.
- Si el firewall elige identidad, se conserva ese resultado y no se reabre el
  threshold sobre adjudication.
- El experimento permanece dentro del generador y executor actuales. No prueba
  autoridad material, transferencia externa, arquitectura promovida ni
  GO/NO-GO.

## Artefactos y recursos

El paquete preservará vistas y estados solver por rol, `mu`, residuos, folds,
permutaciones, modelos de ambas interfaces, predicciones OOF, candidatos de
regularización, scores, cuantiles, uppers, acciones calibradas y desplegadas,
bootstraps, efectos por master, freeze receipts, manifests, entorno y replay.
El output canónico será
`data/geometria_proporcional/proportional_graph_signed_tail_gate_v1/`.

La ejecución es estrictamente CPU: `CUDA_VISIBLE_DEVICES=''`, un thread,
máximo `15 min` y `4 GiB` por fase. Se estiman aproximadamente `3 min` para risk
fit por la generación más ambos ajustes, y menos de `2 min` para cada fase
restante. La GPU continúa suspendida por orden de Mariano; este protocolo no la
consulta, reserva ni consume.
