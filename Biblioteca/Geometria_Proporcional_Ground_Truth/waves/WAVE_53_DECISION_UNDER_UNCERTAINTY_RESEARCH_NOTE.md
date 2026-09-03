# Ola 53 — nota de investigación sobre decisión bajo incertidumbre de conjunto

> **Estado:** `PRIMARY-SOURCES-REVIEWED / DESIGN-INPUT / NO-GUARANTEE-TRANSFER`
> **Fecha:** 2026-09-03

## Problema heredado

La Ola 52 mostró que una política ordinal explícita puede reducir regret y
acciones incompatibles, pero no reparar errores del conjunto binarizado que
recibe. El paso siguiente necesita decidir sobre incertidumbre de pertenencia,
no sólo sobre una lista de miembros tratados como ciertos.

## Convergencia de las fuentes

La literatura revisada converge en una distinción útil. Una predicción
set-valued puede equilibrar cobertura e informativeness, pero su utilidad
depende de la función de decisión que opera después. Mortier et al. formulan la
elección de conjuntos como maximización de utilidad esperada bajo
probabilidades condicionales y muestran que umbralizar o fijar top-k son
baselines, no soluciones universales. Kiyani et al. conectan conjuntos de
predicción con decisiones adversas al riesgo y caracterizan una política
max-min para una clase particular de agentes. El-Yaniv y Wiener formalizan la
abstención como un trade-off entre riesgo y cobertura. Johnstone y Cox enlazan
regiones conformales con optimización robusta, mientras Angelopoulos et al.
extienden el control conformal desde miscoverage hacia pérdidas acotadas.

Estas fuentes no autorizan a llamar conformal al próximo smoke. Los logits
históricos de Phideus no fueron generados bajo un protocolo conformal y la
partición de desarrollo disponible no debe reutilizarse para prometer cobertura
finita. Lo transferible es la arquitectura de la pregunta: preservar una región
de incertidumbre, declarar una pérdida y separar decisión a cobertura completa
de selección con abstención.

## Consecuencia de diseño

Con cuatro familias hay quince conjuntos compatibles no vacíos. Bajo una
aproximación explícita de Bernoulli marginal independiente, los logits
calibrados inducen una masa sobre esos quince conjuntos. Para cada acción y
política puede calcularse exactamente el regret esperado, incluido el costo de
elegir una familia incompatible. La acción es la que minimiza ese regret; la
menor pérdida esperada y el margen respecto de la segunda acción ofrecen scores
de selección para construir curvas riesgo-cobertura.

La independencia entre pertenencias es una hipótesis de cálculo, no un hecho.
Debe acompañarse con diagnósticos de dependencia residual y un control que
baraje probabilidades entre tokens. Tampoco se fija un costo natural de
abstención: el smoke reporta curvas y puntos de cobertura predeclarados.

## Fuentes primarias

- https://proceedings.mlr.press/v267/kiyani25a.html
- https://arxiv.org/abs/1906.08129
- https://www.jmlr.org/papers/v11/el-yaniv10a.html
- https://proceedings.mlr.press/v152/johnstone21a.html
- https://arxiv.org/abs/2101.02703
