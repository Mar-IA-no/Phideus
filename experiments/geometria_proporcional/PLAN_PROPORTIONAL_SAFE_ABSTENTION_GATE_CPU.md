# Plan CPU — abstención residual bajo restricción de no-daño IID

**Fecha:** 2026-09-04
**Estado:** diseño e implementación auditados; realizaciones canónicas sin materializar
**Régimen:** seguimiento prospectivo de una política ya aprendida
**Autoridad:** discrimina seguridad de routing dentro del banco; no promueve arquitectura ni decide GO/NO-GO

## Pregunta

La calibración mixta reprodujo una mejora grouped en los cuatro brazos y una
mejora balanceada en los dos brazos raw. No resolvió beneficio IID y tampoco
demostró que el gate de quince features superara al control reducido de dos
medidas de escala. A la vez, el oracle por vista conserva bastante margen. El
problema siguiente no es ampliar capacidad por reflejo, sino preguntar si una
regla explícita de abstención puede retener parte del beneficio grouped sin
aceptar una política cuya estimación IID de calibración sea adversa.

No se reentrena ninguna head ni se reajusta ridge. Se reutilizan los modelos
congelados por
`data/geometria_proporcional/proportional_graph_fresh_mixed_gate_v1/`, cuyo
manifest tiene SHA-256
`d13c0f5d4a249a43bdd70eefe5b7607a15b96a99a919b7c1d8d07ea1db750175`.
El cambio se limita a una segunda decisión:

```text
alpha_base(x) = argmin({0, delta_hat_0.25(x), ..., delta_hat_1(x)})
advantage_hat(x) = max(0, -min_alpha_nonzero delta_hat_alpha(x))
alpha_tau(x) = alpha_base(x) si advantage_hat(x) > tau; 0 en otro caso.
```

Identidad sigue siendo un fallback exacto. Mecanismo y autoridad privada no
entran en `alpha_tau`.

## Frontera de datos

La fuente histórica aporta modelos full, reduced y dieciséis shuffles, además
de sus features de calibración. De esas features se deriva, antes de abrir
datos nuevos, una grilla de thresholds por `arm × family`: `tau=0`, cuantiles
`{0.25, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99}` de la ventaja
predicha positiva y `tau=+inf` como identidad. Duplicados se eliminan y el
orden exacto se congela.

Luego se usan dos realizaciones nuevas y disjuntas bajo la misma ley:

- **policy selection**, seed `2026090529`: pares IID/grouped que sólo eligen
  `tau` para cada modelo ya congelado;
- **adjudication**, seed `2026090537`: pares IID/grouped que permanecen sin
  materializar hasta escribir y hashear el freeze de thresholds.

Tests y preflights usan seeds no canónicas. El master completo es la unidad de
elegibilidad, bootstrap, bandas y contraste. Se exige nuevamente un mínimo de
`220` masters elegibles por realización, sin ajustar seeds por conteo u outcome.

## Restricción de no-daño

Para cada threshold no trivial se calcula sobre policy selection el delta IID
por master frente a identidad, después de promediar las dos seeds neuronales.
Con `2.000` remuestras por master se construye una banda superior simultánea
no studentizada dentro de cada `arm × family`:

```text
q95 = percentil_95(max_tau(mean_boot_tau - mean_observed_tau))
upper_tau = mean_observed_tau + q95.
```

Un threshold no trivial es admisible sólo si `upper_tau <= 0`. Identidad es
admisible por construcción exacta y no se introduce en el máximo, porque su
delta es algebraicamente cero y una banda conjunta con políticas aleatorias le
asignaría artificialmente un límite positivo. Entre thresholds admisibles se
elige el menor delta balanceado observado; los empates favorecen mayor
abstención y finalmente identidad.

La banda es una regla de selección interna, no una garantía para la realización
futura ni un umbral GO/NO-GO. La adjudicación informa por separado si el delta
IID fresco y su intervalo cruzan o no cero.

## Brazos

1. **identity**: `alpha=0` siempre;
2. **unconstrained full**: gate completo congelado con `tau=0`;
3. **safe full**: threshold full elegido por la restricción;
4. **unconstrained reduced**: gate de correction RMS/max con `tau=0`;
5. **safe reduced**: threshold reducido elegido por la misma restricción;
6. **safe shuffled**: dieciséis gates shuffled, cada uno con su propia grilla,
   banda y threshold seleccionados por el mismo procedimiento; sus métricas se
   promedian por vista, sin elegir la réplica más favorable;
7. **oracle per-view**: cota privada no desplegable.

No se agrega un brazo de quince features nuevo ni se usa el test ya abierto
para seleccionar familia. Full y reduced conservan coeficientes, input scale,
alphas y frontera pública. La única diferencia causal entre unconstrained y
safe es el threshold de abstención.

## Estimandos

En adjudicación se reporta por brazo y slice IID, grouped y balanceado:

- RMSE absoluto y delta pareado frente a identity;
- `safe - unconstrained` dentro de la misma familia;
- `safe full - safe reduced`;
- `safe full - safe shuffled`;
- fracción de identidad y distribución de alphas;
- correlación de ventaja predicha con beneficio realizado;
- regret frente al oracle por vista;
- interacción `efecto grouped - efecto IID`.

El bootstrap usa los mismos `2.000` índices para ambos mecanismos y todas las
políticas. Primero se promedian seeds, luego vistas dentro del master cuando el
estimando es balanceado. Cualquier no convergencia vuelve no evaluable el
master afectado; no se promedian supervivientes.

## Lecturas admisibles

- si safe mantiene un efecto IID no positivo y conserva mejora grouped frente
  a identidad y shuffles en adjudicación, la abstención aporta seguridad
  transportable bajo esta ley;
- si selecciona identidad, la restricción es informativa: los scores actuales
  no permiten intervenir con la evidencia de calibración exigida;
- si pasa la restricción en policy selection y daña IID fresco, la regla no
  transporta; no se relaja `tau` sobre el test;
- si safe full no supera safe reduced, no se atribuye valor a las trece
  features adicionales;
- un promedio balanceado favorable no sustituye un resultado IID favorable;
- ningún patrón autoriza geometría física, interfaz universal, promoción o
  decisión GO/NO-GO.

## Fases, artefactos y recursos

El comando `select` verifica el manifest fuente, deriva las grillas sólo de la
calibración histórica, materializa policy selection, ejecuta los modelos
congelados, construye bandas, elige thresholds y escribe `policy_freeze.json`.
El comando `evaluate` verifica hashes, commit, config, fuentes, selección y
ausencia previa del test; recién entonces materializa adjudicación y aplica las
políticas sin reajuste.

Se preservan features, relaciones corregidas, métricas y estados por alpha,
predicciones, grillas, draws o estadísticas suficientes para reconstruir las
bandas, thresholds, decisiones, índices de bootstrap, efectos, manifest,
replay y muestras de RAM. La corrida usa `CUDA_VISIBLE_DEVICES=''`, un thread,
techo de `12 min` y `4 GiB` por fase. Todo trabajo GPU continúa en cola hasta
que el usuario revoque explícitamente la suspensión.
