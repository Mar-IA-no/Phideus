# Ola 56 — nota de investigación para una compuerta residual contextual

> **Estado:** `DESIGN-RESEARCH / PRE-IMPLEMENTATION / CPU-CANDIDATE / NO-GO-NOGO`
> **Fecha:** 2026-09-03
> **Antecedente empírico:** `WAVE_55_CONSERVATIVE_POLICY_BRIDGE_CLOSED.md`

## El problema que dejó abierto la Ola 55

La Ola 55 no mostró que el posterior conjunto carezca de información útil. La
decisión bayesiana pura redujo regret y aumentó compatibilidad, pero sacrificó
accuracy. La compuerta escalar conservadora preservó la acción dura porque el
selector primario no encontró un umbral global que conciliara las tres
métricas. La sensibilidad, sin embargo, mostró que `gamma=0.2` producía
beneficio agregado: sobre `353` reemplazos token×política, `134` eran
beneficiosos y `219` perjudiciales. La magnitud de los primeros alcanzaba para
reducir regret, aunque la precisión del selector fuera sólo `37.96%`.

El límite observado es, por lo tanto, más específico que una falta de señal.
La ventaja posterior escalar correlaciona débilmente con la mejora realizada
(`r≈0.17`): ordena algo del valor, pero no identifica de modo suficiente en qué
estado y bajo qué política conviene abandonar el baseline duro. La alternativa
con mayor poder diagnóstico es aprender esa heterogeneidad sin modificar la
representación que se intenta evaluar.

## Tres marcos externos que orientan, sin autorizar por sí solos

### Selección con rechazo

La clasificación selectiva separa el predictor de una función de selección y
evalúa el compromiso entre cobertura y riesgo. Ese marco sugiere leer una
compuerta no sólo por accuracy global, sino también por cuántos casos acepta y
qué riesgo concentra en los casos seleccionados. En Phideus, la acción dura
ocupa el lugar del fallback y el reemplazo posterior el de la predicción que se
acepta selectivamente.

Fuente primaria:

https://www.jmlr.org/papers/v11/el-yaniv10a.html

### Aprender a diferir

Learning-to-defer formula el desempeño del sistema compuesto, no sólo la
confianza aislada de uno de sus componentes. La literatura muestra además que
comparar dos scores de confianza entrenados por separado puede fallar cuando el
selector no aprende las fortalezas y debilidades relativas de las alternativas.
La traducción pertinente aquí no es “diferir a una persona”, sino decidir entre
dos políticas ya disponibles: mantener la acción dura o delegar en la acción
bayesiana posterior.

Fuentes primarias:

https://proceedings.mlr.press/v119/mozannar20b.html

https://proceedings.mlr.press/v162/verma22c.html

### Mejora segura respecto de un baseline

Safe policy improvement aporta una intuición complementaria: una política
nueva puede operar donde la evidencia es suficientemente confiable y volver al
baseline donde no lo es. Su teoría pertenece a MDPs con supuestos de
incertidumbre que este banco no satisface; no transfiere garantías a la Ola 56.
Sí ofrece una forma rigurosa de plantear la pregunta: la mejora debe medirse
como regret relativo al baseline, no como mérito absoluto del candidato.

Fuente primaria:

https://proceedings.neurips.cc/paper/2016/hash/9a3d458322d70046f63dfd8b0153ece4-Abstract.html

## Traducción arquitectónica

La candidata es una **compuerta residual contextual de valor**. No vuelve a
aprender el conjunto compatible, no altera el encoder y no sustituye las
acciones disponibles. Recibe propiedades calculables en inferencia y estima el
valor de reemplazar `a_hard` por `a_posterior` para cada token y política:

```text
representación congelada ──► posterior conjunto ──► a_posterior
             │                        │
             └────► conjunto duro ───► a_hard

[estado posterior, desacuerdo, dispersión, política]
                         │
                         ▼
          gate de valor residual de baja capacidad
                         │
          mantener a_hard / usar a_posterior
```

El target de desarrollo es la mejora realizada de regret:

```text
gain_realizado = regret(a_hard) - regret(a_posterior)
```

Un valor positivo significa que el reemplazo fue beneficioso. Predecir su
magnitud conserva información que una etiqueta binaria descartaría: en la Ola
55 los reemplazos beneficiosos fueron minoritarios, pero su magnitud compensó
parte de los daños. Por eso una regresión robusta o regularizada es un primer
contraste más informativo que un clasificador de “acierto/error”.

## Variables permitidas

Todas deben existir antes de observar el target del caso evaluado:

- ventaja posterior estimada entre la acción dura y la bayesiana;
- riesgo posterior estimado de ambas acciones y margen entre las dos mejores;
- entropía, masa máxima y margen top-1/top-2 del posterior sobre conjuntos;
- cardinalidad del conjunto duro, cardinalidad posterior esperada y varianza;
- masa posterior asignada exactamente al conjunto duro;
- indicador de desacuerdo entre las dos acciones;
- dispersión entre los tres seeds del encoder;
- vector ordinal de utilidad normalizado, sin embedding de identidad de política.

La identidad discreta de la política queda excluida del brazo primario, pero el
vector de utilidades también identifica biyectivamente cada una de las
veinticuatro permutaciones. No hay leakage para el estimando dentro de ese
catálogo, aunque tampoco hay transporte demostrado hacia políticas nuevas. El
claim se restringe a las veinticuatro políticas observadas y una sensibilidad
leave-policy-group-out mide, sin convertirlo en criterio principal, cuánto se
degrada la regla frente a configuraciones ordinales no vistas en el fit.

## Qué puede y qué no puede demostrar

Una mejora prospectiva mostraría que el posterior actual contiene una señal de
valor utilizable mediante una interfaz residual condicionada. No demostraría
una garantía de mejora segura, utilidad natural, transporte a otro generador ni
una PPU. Un resultado negativo bajo una familia lineal/robusta de baja capacidad
tampoco agotaría todos los gates posibles, pero sí descartaría la explicación
más económica: que unas pocas estadísticas de incertidumbre y política bastan
para localizar los reemplazos beneficiosos.

La nota orienta el diseño; el protocolo prospectivo y sus criterios se fijan en
`WAVE_56_CONTEXTUAL_RESIDUAL_GATE_PLAN.md` antes de generar evidencia fresca.
