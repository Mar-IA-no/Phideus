# Plan CPU — gate residual con calibración mixta y test fresco

**Fecha:** 2026-09-04
**Estado:** corrida oficial y replay exacto completados
**Régimen:** seguimiento prospectivo sintético en dos realizaciones disjuntas
**Autoridad:** discrimina el régimen de calibración; no promueve arquitectura ni decide GO/NO-GO

## Pregunta

El gate residual público ajustado sólo con validation IID mejoró frente a
identidad y frente al control shuffled en los cuatro slices grouped, pero no
en IID. Ese resultado admite al menos dos explicaciones distintas: los
observables públicos contienen una señal de routing parcial que el régimen IID
no permite calibrar bien, o la ganancia grouped es una particularidad del test
histórico ya abierto.

La prueba nueva cambia una sola frontera metodológica: el ajuste ve una
realización fresca con pares IID/grouped, mientras que la adjudicación ocurre
en otra realización fresca que no se materializa hasta congelar la política.
No se reentrena el modelo neuronal. Se reutilizan las heads post-IRLS y la
familia de acciones

```text
y(alpha) = observed + alpha * (post_irls - observed),
alpha in {0, 0.25, 0.5, 0.75, 1}.
```

`alpha=0` sigue siendo identidad exacta. El mecanismo participa en la
construcción balanceada del conjunto de calibración y en el reporte por slice,
pero nunca entra como feature ni como dato disponible para decidir una acción.

## Dos realizaciones y unidad analítica

Se fijan dos seeds nuevas del generador bajo la misma ley, tamaños, ruido,
corrupción y topología del smoke original:

- **realización de calibración:** se toman únicamente los masters de su
  partición generativa `test`, porque son los que el contrato produce con dos
  vistas pareadas IID/grouped;
- **realización de adjudicación:** se toman del mismo modo los pares de otra
  seed, pero el comando de calibración tiene prohibido generarlos, abrirlos o
  escribirlos.

Los identificadores incluyen la seed generativa; el runner exige disjunción de
masters y vistas entre realizaciones. Cada master aporta exactamente una vista
IID y una grouped, conserva topología, señal limpia, ruido base y magnitudes de
corrupción, y difiere sólo en la localización de las aristas corruptas. Los
folds, shuffles, bootstrap y contrastes usan master como unidad atómica: nunca
se separa el par ni se trata sus dos vistas como réplicas independientes.

La elegibilidad estructural se calcula con el mismo control path-shuffle del
smoke y se aplica al master completo. Se exige un mínimo predeclarado de
masters elegibles por realización; el conteo exacto se registra, no se ajusta
después de observar resultados.

Durante la verificación de implementación se ejecutó de punta a punta un
piloto con las seeds `2026090471/2026090479`. Como esa segunda realización ya
fue abierta, ambas quedan retiradas del protocolo prospectivo y preservadas
únicamente como prueba técnica no canónica. Las seeds canónicas
`2026090509/2026090517` se fijaron después del piloto y no se materializarán en
tests ni preflights. El piloto no motivó cambios en features, familias de
política, lambdas, alphas, controles o estimandos.

## Frontera prospectiva

El runner tiene dos fases explícitas:

1. `calibrate` verifica fuentes y hashes, materializa sólo la realización de
   calibración, ejecuta las heads congeladas, resuelve todos los alpha, ajusta
   las políticas y escribe un `policy_freeze.json` junto con su hash;
2. `evaluate` verifica que código, config, fuentes y freeze sigan intactos,
   carga las políticas sin reajustarlas y recién entonces genera la realización
   de adjudicación, decide por observables públicos y abre la autoridad privada
   para puntuar.

Una prueba espía debe demostrar que `calibrate` no invoca el generador con la
seed de adjudicación. Otra debe demostrar que mutar targets, mecanismo o
métricas de adjudicación no cambia modelos ni acciones. La separación es
operativa y trazable, no un secreto criptográfico: el config contiene la seed
futura, pero ninguna estadística del test interviene en el freeze.

## Modelos congelados y features

Las ocho heads post-IRLS (`4` brazos × `2` seeds) se cargan desde
`proportional_graph_irls_loss_contrast_v1`. El input scale se reconstruye del
train original y se contrasta contra el estado del checkpoint. No hay
optimización neuronal, forward sobre GPU ni modificación de pesos.

Se conserva el vector público de quince features del gate histórico:

- nodos, aristas válidas, densidad y caminos válidos;
- resumen de varianza y escala de la observación;
- escala de la corrección post-IRLS;
- cierres de observación y corrección.

Quedan prohibidos mecanismo, split lógico, IDs, seed, señal limpia, máscara
causal y cualquier métrica u output posterior al solver. Features y targets se
promedian entre las dos seeds neuronales antes de ajustar una acción compartida.

## Políticas y controles

1. **identity:** alpha cero en toda vista;
2. **always-post:** alpha uno en toda vista;
3. **constant-balanced:** un alpha global por brazo que minimiza el promedio
   exactamente balanceado de IID y grouped en calibración;
4. **correction-scale ridge:** gate reducido con RMS y máximo de corrección;
5. **public mixed ridge gate:** gate de quince features calibrado con ambas
   vistas, sin mecanismo como predictor;
6. **historical IID ridge gate:** modelo ya congelado en el gate anterior,
   transportado sin refit como contraste directo de régimen de calibración;
7. **shuffled-target mixed ridge:** dieciséis réplicas que permutan entre
   masters del mismo `n_nodes` el bloque conjunto de targets IID+grouped, sin
   separar el par ni seleccionar la réplica más favorable;
8. **oracle per-view:** mejor alpha con autoridad privada, sólo como cota
   diagnóstica no desplegable.

Los gates ridge mantienen lambdas, cinco folds agrupados por master, regla de
empate hacia mayor regularización y elección de acción con identidad incluida
como delta cero. Las predicciones OOF de calibración son diagnóstico de tuning;
la estimación principal proviene exclusivamente de la realización fresca de
adjudicación.

## Estimandos

Por brazo y mecanismo, la unidad es el master y primero se promedian las dos
seeds neuronales. Se reportan RMSE absoluto, acciones y delta pareado del gate
mixto frente a:

- identity;
- always-post;
- constant-balanced;
- gate reducido;
- gate histórico IID;
- media por vista de las réplicas shuffled.

El contraste central es `public mixed ridge - historical IID ridge`: pregunta
si cambiar el régimen de calibración, con arquitectura y frontera pública
iguales, mejora el transporte conjunto. El bootstrap por master usa `2.000`
réplicas y los mismos índices para IID y grouped, lo que también permite
reportar el promedio balanceado y la diferencia de efecto `grouped - IID`.
Cualquier no convergencia de una seed invalida el estimando afectado; no se
promedian supervivientes.

## Lecturas admisibles

- una mejora frente a identidad y shuffled en ambos mecanismos sostendría que
  hay señal pública de routing aprovechable bajo calibración mixta;
- mejorar grouped y dañar IID repetiría la disociación, aunque el promedio
  balanceado fuese favorable;
- superar al gate histórico pero no a identidad atribuiría una mitigación al
  régimen de calibración, no utilidad neta;
- no superar shuffles debilitaría la atribución a correspondencia entre
  observables y beneficio;
- diferencias entre realizaciones se reportan como transporte bajo esta ley
  sintética, nunca como transferencia física, techo o decisión arquitectónica.

El test fresco se abre una sola vez. Después de hacerlo, cualquier extensión de
features, cambio de lambda o nueva regla de acción requiere otra realización;
no se itera sobre el mismo test.

## Artefactos, recursos y replay

La fase de calibración preserva observaciones identificadas, relaciones
post-IRLS, features, métricas por alpha y seed, estados de solver, folds,
shuffles, modelos, predicciones OOF, decisiones in-sample y freeze. La fase de
adjudicación agrega el universo fresco, forward crudo, métricas por alpha,
decisiones congeladas, bootstrap, efectos y diagnósticos. Manifest y replay
verifican fuentes, inputs, transición de fase y determinismo byte a byte.

Todo el protocolo corre en CPU con `CUDA_VISIBLE_DEVICES=''`, un thread para
PyTorch y BLAS, techo total de `12 min` por fase y `4 GiB` de RSS, con muestras
periódicas de memoria. Cualquier extensión que requiera GPU queda anotada en la
cola y no se ejecuta hasta una nueva indicación del usuario.
