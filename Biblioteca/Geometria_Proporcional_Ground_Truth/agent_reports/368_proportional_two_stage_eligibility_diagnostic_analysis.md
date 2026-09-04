# R368 — Diagnóstico two-stage de ranking y elegibilidad

**Fecha:** 2026-09-04  
**Estado:** oficial y replay exacto completados  
**Régimen:** post hoc, dos cohortes abiertas, CPU-only  
**Autoridad:** contrasta una interfaz de decisión; no valida no-daño prospectivo, arquitectura ni GO/NO-GO

## Pregunta

R367 mostró que sumar la cola al score perjudicaba el ranking. R368 separó las
funciones sin cambiar la propuesta public-base de alpha ni los seis
presupuestos:

```text
alpha_common = argmin(mu + u_public_base)
rank_score   = mu_alpha_common
eligible_f   = mu_alpha_common + u_f,alpha_common + q_f < 0
```

`mu` fija un único orden. Constant, public-base, topology y dieciséis
topology-permuted sólo deciden elegibilidad; ninguna cola puede reordenar las
vistas. Cada `q_f` se calibra al `90%` sobre el residuo selected-action IID y el
firewall de policy-selection conserva `upper95 <= 0`. Se reportan las políticas
calibradas y las desplegadas después de sustituir por identidad cada rechazo.

## Integridad

Diseño `9cfff8f`, implementación `524b9e4`. Oficial y replay terminaron en
`18,064/17,962 s`, con `0,691/0,691 GiB`, un thread y CUDA invisible. No hubo
refit, vistas ni solves nuevos; `environment.json` registra
`gpu_queried: false`. El manifest byte-idéntico es:

```text
bffbf9fefa915b6dfd00df63e35bacf9084ec68d71831471b5c0537b11b2536e
```

Coinciden `17/17` archivos deterministas, los `3.925` arrays canónicos son
finitos y la regresión proporcional cerró `174/174`. También se verificó que
cada acción respeta el presupuesto, conserva exactamente el orden por `mu` y
que deployed sólo puede copiar calibrated o volver a identidad.

## Resultado

El filtro topology resulta extremadamente abstencionista. Declara elegibles
`31/2.032` decisiones brazo×vista en A (`1,53%`) y `7/2.040` en B (`0,34%`). En
A alcanza su soporte completo desde el presupuesto `2%`; en B ya lo agota en
`1%`. Por eso los puntos posteriores no añaden acciones: el promedio entre
presupuestos es descriptivo y contiene repeticiones de la misma política.

En las `72` celdas brazo×presupuesto×slice del stage calibrado:

- topology-filter menos mean-only es adverso en ambas cohortes en `57/72`,
  inestable en `14/72` y cero en `1/72`; mean-only gana los `12/12` promedios
  brazo×slice, y `32/72` intervalos son adversos en las dos cohortes;
- topology-filter menos topology-permuted, el control incremental primario, es
  adverso en `24/72`, favorable en `10/72`, inestable en `32/72` y cero en
  `6/72`; ningún intervalo pointwise resuelve una dirección en ambas cohortes;
- frente a public-base, topology es adverso en `30/72`, inestable en `6/72`,
  cero en `36/72` y nunca favorable en ambas;
- frente a identidad, topology es favorable en ambas en `41/72`, inestable en
  `13/72` y cero en `18/72`, sin puntos adversos compartidos. Sin embargo,
  ningún intervalo queda favorable en las dos cohortes: A resuelve once y B
  ninguno.

El firewall no rescata la atribución. Despliega `8/24` políticas topology en A
y `18/24` en B, conteos inflados por presupuestos que ya seleccionan las mismas
vistas. Después del firewall, topology menos permuted queda adverso en `29/72`,
favorable en `1/72`, inestable en `30/72` y cero en `12/72`; entre los doce
promedios no queda ningún caso favorable compartido.

La cobertura selected-action observada en adjudicación IID va de `88,19%` a
`92,91%` en A y de `87,45%` a `93,33%` en B. Es compatible con una garantía
marginal calibrada en otra muestra, no con cobertura condicional entre las
vistas elegibles. La distinción se vuelve empírica: el daño medio entre
actuadas es positivo en raw-typed grouped de A (`+0,001321`, `n=3`) y en
closure-typed IID de B (`+0,004075`, `n=1`). El filtro no autoriza a llamar
“segura” a cada intervención.

## Lectura

**Observación.** Separar ranking y cola evita el reordenamiento perjudicial de
R367, pero la cola topology reduce tanto el soporte que pierde frente al
ranking mean-only y no supera a sus controles de escala o permutación.

**Hipótesis.** La media ya concentra la señal de beneficio disponible, mientras
la calibración marginal de cola paga un costo de abstención que no compra
localización topológica atribuible. La ventaja frente a identidad describe las
pocas oportunidades seleccionadas; no demuestra que topology sea quien las
identifica.

**Inferencia acotada.** No está justificado un freeze confirmatorio de esta
interfaz ni una cabeza de cola más grande. La alternativa arquitectónica más
simple es preservar `mu` como ranker y tratar el presupuesto/firewall agregado
como módulo de decisión, dejando la cola aprendida fuera del camino primario
hasta que muestre valor incremental. Esto es una candidata experimental, no
una promoción.

## Próximo paso

Antes de diseñar otro filtro conviene atribuir la señal de `mu`: comparar el
ranking de media topology con medias public-base/reduced y controles de
localización permutada sobre el mismo alpha, presupuesto y firewall. Ese
diagnóstico CPU decide si el activo recuperable es realmente topológico o sólo
un predictor público general. Cualquier contraste GPU permanece en cola hasta
que el usuario restituya el dispositivo.

Artefactos: plan
`experiments/geometria_proporcional/PLAN_PROPORTIONAL_TWO_STAGE_ELIGIBILITY_DIAGNOSTIC_CPU.md`,
runner
`experiments/geometria_proporcional/run_proportional_graph_two_stage_eligibility_diagnostic.py`,
oficial
`data/geometria_proporcional/proportional_graph_two_stage_eligibility_diagnostic_v1/`
y replay con sufijo `_replay`.
