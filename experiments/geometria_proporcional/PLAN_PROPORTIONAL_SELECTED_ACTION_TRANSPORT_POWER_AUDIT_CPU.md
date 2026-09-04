# Plan CPU — transporte y potencia de selected-action

**Fecha:** 2026-09-04
**Estado:** diseño congelado; implementación y ejecución pendientes
**Régimen:** auditoría post hoc de dos cohortes independientes ya abiertas
**Autoridad:** decide si otra realización fresca tiene poder diagnóstico plausible; no confirma política, no declara techo, no promueve arquitectura ni decide GO/NO-GO

## Pregunta

R364 mostró que proponer antes de calibrar reduce el margen simultáneo y
recupera acción, pero no aisló topology frente a public-base o
topology-permuted. Abrir de inmediato otra realización de aproximadamente 250
masters supondría que el efecto incremental observado es estable y que ese
tamaño puede resolverlo. Ambas suposiciones pueden comprobarse primero con los
dos pipelines frescos ya preservados.

La auditoría aplica exactamente la misma regla selected-action a la cohorte
R361/R362 y la compara con R363/R364. No combina tests para fabricar una
confirmación. Pregunta si signo, orden contra controles, acción y proyección de
muestra transportan entre dos realizaciones independientes del mismo generador.

## Fuentes inmutables

### Cohorte A

- roles frescos y deltas:
  `data/geometria_proporcional/proportional_graph_conditional_risk_gate_v1/`,
  manifest
  `81a977de9c187d687a4f9c5543eb641d351d6957cea9ebdbe4c04204d8c2dffe`;
- modelos signed-tail y predicciones:
  `data/geometria_proporcional/proportional_graph_signed_tail_diagnostic_v1/`,
  manifest
  `a73ee47da9f1b8af5cb35531f8d0c8109a6ca783101d9245cb5c2cf4db5ba1a2`.

### Cohorte B

- pipeline fresco paired:
  `data/geometria_proporcional/proportional_graph_signed_tail_gate_v1/`,
  manifest
  `9beef0a4942101a876334c1280e9bc42ed1604ebea67f7e52fc40eb7f2d425b1`;
- interfaz selected-action ya materializada:
  `data/geometria_proporcional/proportional_graph_selected_action_calibration_diagnostic_v1/`,
  manifest
  `36a56ad48d9eee5f62c84a3ed8b6fa8749d7a009607e91af22d35de5afb07ad2`.

Se verifican todos los archivos deterministas de las cuatro fuentes. No hay
vistas, forwards, solves, fit ni selección de hiperparámetros nuevos.

## Reconstrucción idéntica por cohorte

En cada cohorte y familia se conserva el modelo signed-tail ajustado en su
propio risk fit. Sobre risk calibration:

```text
alpha_star = argmin_alpha(mu_alpha + u_hat_alpha)
score = delta_real_alpha_star - mu_alpha_star - u_hat_alpha_star
q_selected = conformal_order_0.90(score_IID)
```

Policy selection aplica su bootstrap original y el mismo firewall superior
`95%`. Adjudication usa sus índices originales. La cohorte B se recomputa desde
los crudos y debe igualar byte a byte las decisiones R364; ese chequeo evita que
dos implementaciones nominalmente iguales entren en la comparación.

Las familias son constant, public-base, topology y dieciséis
topology-permuted. Los controles se promedian por vista sólo después de su
propia calibración y firewall.

## Estimandos

Por cohorte, brazo y slice IID/grouped/balanceado se preservan los efectos por
master de:

1. topology selected-action menos identidad;
2. topology menos public-base selected-action;
3. topology menos el promedio topology-permuted selected-action.

El tercer contraste es el estimando incremental principal; public-base es un
control secundario necesario. El balanceado conserva igual peso para IID y
grouped. Los intervalos son pointwise, con el bootstrap original de cada
cohorte y sin pooling entre tests.

Se reportan además acción, distribución de alphas, daño entre vistas actuadas,
cobertura selected-action y estabilidad del firewall. Un cero debido a identidad
exacta o un residuo de máquina menor que `1e-12` se trata como igualdad, no como
signo resuelto.

## Transporte

Para cada estimando se clasifica:

- `FAVORABLE_BOTH`: media negativa en ambas cohortes;
- `ADVERSE_BOTH`: media positiva en ambas;
- `SIGN_UNSTABLE`: signos opuestos;
- `IDENTITY_OR_NUMERICAL_ZERO`: al menos una media tiene magnitud `<=1e-12`.

Los intervalos se informan junto con el signo, pero no se exige que dos tests
post hoc produzcan una etiqueta binaria. La clasificación describe transporte
observado bajo el generador, no generalización externa.

## Proyección de potencia bajo efecto fijo

Para una media favorable `m<0`, tamaño actual `n` y radio bootstrap superior
`r = upper95 - m`, se calcula

```text
n_projected = ceil(n * (r / |m|)^2).
```

Si el intervalo ya es favorable, el tamaño actual basta bajo esta aproximación.
Si `m>=0`, no existe proyección favorable finita bajo el efecto observado. La
proyección transport-aware sólo existe cuando ambas cohortes tienen signo
favorable y usa el máximo de sus dos tamaños proyectados.

Este cálculo supone efecto y varianza fijos, no garantiza transporte y no debe
leerse como techo ni tamaño definitivo. Se contrasta además con el tamaño
realizable histórico de unas 250 unidades por seed y con el número esperado de
acciones al mantener la tasa observada.

## Lecturas admisibles

- Signo favorable estable y proyección cercana al tamaño realizable justifica
  diseñar otra prueba fresca con topology−control como estimando principal.
- Signo inestable invalida aumentar muestra bajo un único efecto fijo; el
  siguiente trabajo debe cambiar representación, proposer o unidad de
  calibración.
- Proyecciones grandes con muy pocas acciones indican que el cuello es de
  soporte efectivo, aunque la media sea favorable.
- Una diferencia frente a identidad sin diferencia frente a controles no
  acredita localización topológica.
- Dos cohortes internas no prueban autoridad física, transferencia externa,
  arquitectura promovida ni GO/NO-GO.

## Artefactos y recursos

Se preservan source hashes, reconstrucción de cohorte A, chequeo exacto de
cohorte B, efectos por master, bootstraps, tablas de signo, proyecciones,
conteos de acción, manifest, entorno y replay. El output canónico será
`data/geometria_proporcional/proportional_graph_selected_action_transport_power_audit_v1/`.

La ejecución usa `CUDA_VISIBLE_DEVICES=''`, un thread, máximo `5 min` y `4 GiB`.
No materializa nuevas realizaciones ni ejecuta IRLS. La GPU continúa suspendida
y esta auditoría no la consulta, reserva ni consume.
