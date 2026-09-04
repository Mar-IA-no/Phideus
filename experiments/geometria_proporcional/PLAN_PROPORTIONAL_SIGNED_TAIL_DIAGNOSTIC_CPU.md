# Plan CPU — diagnóstico unilateral de la cola de daño

**Fecha:** 2026-09-04
**Estado:** implementación CPU auditada; ejecución oficial pendiente
**Régimen:** post hoc sobre cuatro realizaciones ya abiertas de R361
**Autoridad:** diagnostica una interfaz candidata y puede justificar otro protocolo fresco; no adjudica una política, no promueve arquitectura ni decide GO/NO-GO

## Problema localizado

R361 congeló el predictor topológico de beneficio y aprendió
`log(|delta_real-mu|+1e-6)`. El límite resultante conservó seguridad mediante
acción casi nula: topology intervino en `13/2.032` decisiones brazo-vista y no
resolvió ventaja frente a identidad o controles. Una explicación posible es
que el residuo absoluto trate simétricamente dos desviaciones con consecuencias
opuestas. Si `delta_real-mu` es muy negativo, el predictor fue demasiado
pesimista y la intervención resultó mejor de lo esperado; si es positivo, el
predictor subestimó daño. Sólo la segunda cola debe elevar un upper bound.

El diagnóstico no abre otra adjudicación. Reutiliza los artefactos preservados
de R361 para preguntar si una estimación unilateral ordena el riesgo de manera
distinta. Todo resultado será explícitamente opened/post hoc.

## Fuentes inmutables

La fuente es
`data/geometria_proporcional/proportional_graph_conditional_risk_gate_v1/`,
manifest SHA-256
`81a977de9c187d687a4f9c5543eb641d351d6957cea9ebdbe4c04204d8c2dffe`.
Se verifican manifest, configuración, modelos de media y escala, features,
topología, deltas, cuantiles, firewall, decisiones y bootstraps de sus cuatro
roles. No hay re-forward, re-solve ni generación de views.

## Modelo unilateral

Se mantiene congelado `mu_alpha(x)` de R360. Sobre risk fit IID se define el
residuo firmado

```text
r_alpha(x) = delta_real_alpha(x) - mu_alpha(x).
```

Para cada alpha no identidad se ajusta el cuantil condicional superior `0,90`
de `r_alpha` mediante regresión lineal de pinball loss. Las features se
estandarizan dentro de cada fit; una columna constante o degenerada conserva
escala uno. Los cuatro outputs se ajustan por separado, pero una misma fuerza
L1 se selecciona por familia y brazo con la media OOF de los cuatro pinball
losses.

La grilla es `{0, 0.0001, 0.001, 0.01, 0.1, 1}` y usa cinco folds por master.
Ante empate numérico dentro de `1e-12`, se conserva la regularización mayor. El
solver es `QuantileRegressor(solver="highs")`, un thread y tolerancias default
versionadas por el entorno. Toda no convergencia invalida la corrida.

## Familias matched

Las familias reutilizan exactamente los inputs de R361:

1. `constant_signed_tail`: intercepto empírico por output, sin features;
2. `public_base_signed_tail`: quince features públicas;
3. `topology_signed_tail`: las mismas veinticuatro features del predictor de
   media;
4. `topology_permuted_signed_tail`: dieciséis réplicas de veinticuatro inputs,
   con localización permutada y multiconjunto de corrección preservado.

No se elige la mejor réplica. Predicciones, acciones y métricas de shams se
promedian por vista después de aplicar cada política.

## Corrección conformal y acción

Risk calibration permanece separada del fit. Sólo sus vistas IID construyen

```text
s_i = max_alpha(r_i,alpha - u_hat_i,alpha).
```

Con miscoverage `0,10`, `q` es el orden estadístico
`ceil((n+1)*0,90)`. El límite final es

```text
upper_alpha = mu_alpha + u_hat_alpha + q.
```

La acción elige el alpha con menor upper sólo si ese valor es estrictamente
negativo; de lo contrario copia identidad. Este límite conserva una garantía
marginal IID por vista y simultánea sobre cuatro alphas bajo exchangeability.
No promete cobertura condicional entre vistas actuadas, grouped ni dominio
externo.

## Uso de policy selection y adjudication abiertos

El diagnóstico reaplica el bootstrap y el firewall de R361 sobre policy
selection, sin cambiar el nivel `95%` ni sus índices. Luego aplica las políticas
congeladas a adjudication sólo para describir:

- pinball OOF y cobertura OOF por familia;
- asimetría de residuos positivos/negativos;
- cuantiles y correcciones conformales;
- tasa de acción y distribución de alphas;
- delta IID, grouped y balanceado frente a identidad;
- contraste signed-tail menos absolute-scale de R361;
- signed topology menos constant, public base y promedio permutado;
- cobertura simultánea y daño entre vistas actuadas;
- interaction grouped menos IID y oracle preservado.

Todos los intervalos sobre adjudication son descriptivos post hoc. El
diagnóstico no permite escoger un hiperparámetro, familia o umbral para reclamar
éxito sobre esos mismos datos. Una eventual prueba prospectiva necesita código
y protocolo nuevos, cuatro seeds frescas y freeze anterior a su adjudicación.

## Lecturas admisibles

- una tasa de acción mayor con efecto IID compatible y mejor orden frente a
  absolute-scale sostiene la hipótesis unilateral, pero no la confirma;
- si topology no supera public-base o permuted, la localización no recibe
  crédito como modelo de cola aunque la interfaz unilateral resulte más activa;
- si el firewall vuelve a identidad, la dirección del target no era el único
  cuello;
- si la cobertura empírica fluctúa alrededor de `0,90`, no se la convierte en
  cobertura condicional ni en garantía de una realización finita;
- ningún resultado modifica la evidencia de representación de R360 ni la
  adjudicación prospectiva de R361.

## Artefactos y recursos

Se preservan source hashes, residuos firmados, folds, scalers, coeficientes,
predicciones OOF, candidatos L1, pinball losses, cuantiles, scores conformales,
uppers, acciones, firewall reaplicado, métricas por master, índices bootstrap,
manifest y replay. El output canónico será
`data/geometria_proporcional/proportional_graph_signed_tail_diagnostic_v1/`.

La ejecución usa `CUDA_VISIBLE_DEVICES=''`, un thread, máximo `12 min` y
`4 GiB`. La GPU continúa suspendida; este diagnóstico no crea ni consume cola
CUDA.

## Validación informática previa

El runner, la configuración y las pruebas materializan el contrato sin abrir
otra seed. Un primer piloto detectó y corrigió antes del freeze un desacople de
nombres entre el firewall heredado y las familias `*_signed_tail`. El piloto
completo posterior terminó en `74,07 s / 0,692 GiB`; verificó `11/11` archivos
deterministas, cuatro paquetes NPZ sin valores no finitos, separación
calibrated/deployed, descripción de ambas colas y entorno versionado. Ningún
outcome del piloto cambió features, target, cuantíl, grilla, controles o regla
de acción.
