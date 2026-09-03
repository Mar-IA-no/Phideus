# Ola 53 — plan de política sensible a incertidumbre

> **Estado:** `AUDITED-FROZEN / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-03
> **Antecedente:** `WAVE_52_UTILITY_CONDITIONED_POLICY_TRANSPORT_CLOSED.md`

## Pregunta

¿Puede una regla de decisión que conserva incertidumbre de compatibilidad
reducir regret y errores incompatibles frente a la política sobre conjunto
binarizado, sin perder exactitud, y ofrecer una abstención ordenada por riesgo?

El estimando es sistémico y de desarrollo. No prueba una utilidad natural, una
geometría física ni superioridad causal del encoder. No lee lockbox, no usa GPU
y no reentrena la representación.

## Datos y particiones

Se reutilizan los logits crudos por seed de Ola 52. `val_threshold` se divide
por `pair_token`, antes de ajustar nada, en dos mitades deterministas,
estratificadas por régimen y cardinalidad compatible:

- `calibration_fit`: ajusta un calibrador Platt compartido por las cuatro
  familias;
- `decision_select`: selecciona el peso del baseline de composición y los
  cortes de aceptación para coberturas nominales.

`val_monitor` permanece como evaluación. Ningún target de monitor selecciona
calibrador, hiperparámetro o threshold. La población primaria es `NEAR_RIVAL`
con cardinalidad compatible mayor o igual que dos; se reporta además el total
in-catalog como sensibilidad.

## Reglas

1. `hard_set_policy`: baseline de Ola 52, máximo de utilidad dentro del conjunto
   predicho a `tau` congelado.
2. `score_composition`: baseline continuo de Ola 52; `lambda` se vuelve a
   seleccionar sólo en `decision_select`.
3. `marginal_expected_regret`: aplica Platt compartido a los logits ensemble,
   enumera los quince conjuntos no vacíos bajo Bernoulli marginal independiente,
   condiciona la masa a no vacío y elige la acción de menor regret esperado.
4. `raw_expected_regret`: misma regla con sigmoid sin calibración; separa el
   efecto del decision rule del efecto de Platt.
5. `oracle_set_then_utility`: conjunto verdadero y utilidad; referencia
   privilegiada, no arquitectura deployable ni techo universal.

La pérdida queda congelada por continuidad con Ola 52. Para acción `a`, conjunto
no vacío `S` y utilidad `u`, `L(a,S,u)=1.25` si `a` no pertenece a `S`; si
pertenece, `L=(max_{j en S} u_j-u_a)/(max(u)-min(u))`. Las utilidades son las
24 permutaciones uniformemente ponderadas de `[1.0, 0.6, 0.2, -0.2]`. La masa
de cada `S` es el producto Bernoulli marginal, renormalizado sobre los quince
conjuntos no vacíos. `argmin` desempata por el menor índice de familia. Ésta es
una utilidad contractual y cardinalizada para el smoke, no una utilidad natural.

Platt es un único calibrador compartido `(a,b)`: se ajusta por NLL sobre una
fila por `pair_token×familia`, usando el promedio de logits crudos de los tres
seeds y el target binario. No se duplican observaciones por política ni se
tratan seeds como réplicas. Receta fija: `LogisticRegression`, L2, `C=1.0`,
`solver=lbfgs`, `max_iter=1000`, sin class weights; si el split no contiene
ambas clases, falla cerrado en vez de sustituir silenciosamente el modelo.

Controles:

- `probability_shuffled`: permuta el vector de probabilidades entre tokens
  dentro del estrato antes de decidir;
- `utility_masked`: reemplaza cada política por un orden fijo conservando el
  target contextual;
- diagnóstico de dependencia: Brier/NLL marginal, error de distribución de
  cardinalidad y correlaciones residuales de pertenencia por pares de familias.

## Abstención

No se inventa un costo natural de rechazo. Las dos reglas de regret esperado
producen por política la menor pérdida esperada y el margen entre la mejor y la
segunda acción. El score selectivo primario agrega las 24 políticas de un token
mediante el **máximo** de su menor pérdida esperada: menor es mejor y representa
el peor contexto ordinal previsto para ese token. El mínimo margen se conserva
como sensibilidad secundaria, no selecciona el veredicto.

En `decision_select` se fijan fronteras lexicográficas `(score, hash(pair_token))`
que retienen el entero más cercano a `50%`, `75%` y `90%` de sus tokens, con hash estable
independiente de labels para desempatar. Las fronteras se congelan y se aplican
a `val_monitor`, donde la cobertura efectiva puede diferir. Todas las reglas se
evalúan sobre la misma máscara de tokens aceptados por la regla de incertidumbre
que corresponda; nunca se separan las 24 políticas de un token.

Se reportan cobertura efectiva, exactitud selectiva, tasa compatible, regret
selectivo, peor regret y área bajo la curva riesgo-cobertura. El riesgo observado
por token es el regret restringido medio sobre sus 24 políticas. `AURC` es la
media discreta del riesgo acumulado a coberturas `1/N,...,N/N`, ordenando por el
score primario ascendente y desempatando por `pair_token`; no se interpreta como
garantía conformal. Las 24 políticas permanecen juntas.

## Criterio diagnóstico

`marginal_expected_regret` queda marcado como prometedor para una siguiente
réplica, sin promoción, sólo si sobre la población primaria de monitor:

1. reduce regret full-coverage frente a `hard_set_policy` por al menos `0.02`,
   con IC95 del delta debajo de cero;
2. no pierde más de `0.01` de exactitud y no reduce la tasa compatible;
3. supera a `raw_expected_regret` en Brier o NLL y no empeora su regret por más
   de `0.005`;
4. a cobertura nominal `75%`, reduce regret frente a su propia cobertura
   completa y mantiene cobertura efectiva en `[0.70,0.80]`;
5. supera al control `probability_shuffled` en exactitud por al menos `0.05`;
6. el control `utility_masked` pierde al menos `0.05` de exactitud;
7. una segunda ejecución en proceso y directorio independientes reconstruye
   calibrador, splits, controles, bootstrap y métricas desde los inputs ligados,
   y coincide exactamente en arrays, decisiones y métricas. La primera corrida
   queda `PENDING_EXTERNAL_REPLAY`; sólo la segunda puede completar el patrón.

El patrón es conjunto. Los umbrales organizan el smoke y no son GO/NO-GO. Si
la calibración marginal mejora scores pero no decisión, el resultado separa
calibración de policy. Si la dependencia residual es grande, toda lectura del
modelo Bernoulli queda condicionada y el siguiente paso deberá modelar la
distribución conjunta de conjuntos.

## Artefactos

- manifest de inputs, split y grillas antes de leer labels de monitor;
- parámetros y métricas del calibrador;
- probabilidades por seed y ensemble;
- masa sobre los quince conjuntos, pérdidas esperadas y acciones por
  token/política;
- curvas riesgo-cobertura y decisiones de abstención;
- controles shuffled/masked y mappings;
- métricas por token, política y seed;
- runtime, package manifest y replay exacto.
- índices completos de los `n_boot=2000` resamples pareados, seed `5307`, y
  resultados de los IC. Los mismos índices se usan para todos los contrastes;
  los IC son condicionales al calibrador, hiperparámetros y cortes ya ajustados.
  Policies y seeds son mediciones fijas, no unidades independientes.

## Archivos previstos

- `src/geometria_proporcional/wave53_uncertainty.py`
- `experiments/geometria_proporcional/run_wave53_uncertainty_policy.py`
- `experiments/geometria_proporcional/configs/wave53_uncertainty_policy.json`
- `tests/test_wave53_uncertainty.py`

## Límite operativo

CPU-only. Si este smoke identifica una ventaja que justifique reentrenar el
encoder o una cabeza conjunta, esa etapa posterior se diseña aparte y todo uso
de GPU se informa antes con objetivo, duración y VRAM estimados.

## Resolución de auditoría independiente

R297 marcó cuatro ambigüedades materiales: pérdida esperada, estimando de
abstención, unidad de ajuste Platt y bootstrap. Las cuatro quedaron resueltas
arriba antes de implementar. La auditoría confirmó además que el diseño no
importa garantías conformales ni selecciona sobre `val_monitor`.

R298 auditó la primera implementación y exigió: freeze cronológico antes de
cargar monitor, replay externo real, criterio Platt sobre la población primaria,
máscaras selectivas comunes, procedencia y artefactos completos y convención
entera explícita para cobertura. Ninguna corrida se ejecuta antes de resolver
estos puntos y congelar el código corregido en Git.

R299 encontró que el intake debía reproducir el filtro `canonical_preserving`
de Ola 52 y que el freeze aún se escribía demasiado tarde. La resolución fija
la receta y los hashes esperados antes de cargar cualquier label autorizado o
predicción de monitor; luego procesa sólo filas canónicas cuyos tokens están en
el split permitido. También prohíbe `NaN` no estándar en los diagnósticos.

R300 reauditó focalmente esas correcciones y dio `PASS` para la implementación:
el intake recupera `384/384` tokens canónicos en ambos splits, el freeze antecede
labels y predicciones, el split derivado antecede monitor y todo JSON falla
cerrado ante valores no finitos. El único bloqueo restante es el preflight
deliberado de Git, que se resuelve versionando estas fuentes antes de ejecutar.
