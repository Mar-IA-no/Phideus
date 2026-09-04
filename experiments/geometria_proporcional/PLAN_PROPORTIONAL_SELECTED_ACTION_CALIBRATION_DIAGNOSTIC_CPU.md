# Plan CPU — diagnóstico de calibración posterior a la propuesta

**Fecha:** 2026-09-04
**Estado:** implementación CPU auditada; ejecución oficial pendiente
**Régimen:** post hoc sobre los cuatro roles ya abiertos de R363
**Autoridad:** localiza conservatividad de interfaz y puede justificar otro protocolo fresco; no adjudica una política, no promueve arquitectura ni decide GO/NO-GO

## Problema localizado

R363 transportó la ventaja OOF de topology para modelar la cola firmada, pero
la política actuó sólo en `5/2.040` decisiones. Su corrección conformal tomó el
máximo del residuo sobre cuatro alphas para garantizar cobertura simultánea,
aunque cada vista ejecuta a lo sumo uno. El diagnóstico pregunta si ese
desacople entre jurisdicción del límite y decisión explica parte de la
abstención.

No se generan realizaciones nuevas. La prueba reutiliza el fit, calibración,
selección y adjudicación preservados de R363, por lo que cualquier resultado es
opened/post hoc y sólo puede orientar otro diseño.

## Fuente inmutable

La fuente es
`data/geometria_proporcional/proportional_graph_signed_tail_gate_v1/`, manifest
SHA-256
`9beef0a4942101a876334c1280e9bc42ed1604ebea67f7e52fc40eb7f2d425b1`.
Se verifican sus `102` archivos deterministas, config, modelos signed-tail,
features, deltas, bootstraps, firewall y decisiones de ambas interfaces. No hay
re-forward, re-solve ni refit.

## Proponer primero

Para cada familia, brazo y vista, los modelos signed-tail de risk fit ya
congelados producen cuatro límites sin corrección conformal:

```text
b_alpha(x) = mu_alpha(x) + u_hat_alpha(x).
```

El proposer elige siempre un único alpha no identidad antes de consultar
calibración:

```text
alpha_star(x) = argmin_alpha b_alpha(x).
```

Los empates conservan el primer alpha en el orden congelado
`{0,25; 0,50; 0,75; 1,00}`. No se abstiene en esta etapa: la abstención depende
del límite calibrado posterior. El proposer usa sólo observables públicos y
modelos de risk fit; no lee el target de calibración, selección o adjudicación.

## Calibrar después

Sobre las vistas IID de risk calibration se calcula un único score por master:

```text
s_i = delta_real_i,alpha_star - b_i,alpha_star.
```

Con miscoverage fijo `0,10`, el cuantíl `q_selected` usa el orden estadístico
`ceil((n+1)*0,90)`. En otra vista, la política evalúa

```text
upper_selected(x) = b_alpha_star(x) + q_selected
```

y ejecuta `alpha_star` sólo si `upper_selected < 0`; de lo contrario copia
identidad.

Si el proposer y el modelo permanecen congelados antes de calibration, este
esquema apunta a cobertura marginal IID del daño de la acción seleccionada. No
entrega cobertura simultánea para los cuatro alphas, cobertura condicional
entre vistas actuadas, garantía grouped ni autoridad externa. La garantía es
más estrecha que la de R363 y está alineada con la única acción ejecutada.

## Familias y controles

Se conservan sin cambios:

1. `constant_selected_action`;
2. `public_base_selected_action`;
3. `topology_selected_action`;
4. dieciséis `topology_permuted_selected_action` matched.

Cada familia usa su propio modelo signed-tail de R363. Las réplicas permutadas
conservan targets, outputs, folds y presupuesto; nunca se elige la mejor. Sus
acciones y efectos se promedian sólo después de aplicar su propia calibración y
firewall.

## Firewall abierto y adjudicación descriptiva

Policy selection reaplica los mismos `2.000` índices bootstrap y percentil
superior `95%` de R363. Una política se despliega sólo si su límite para delta
medio IID frente a identidad es no positivo; una réplica rechazada cae a
identidad. No se cambia la regla usando adjudication.

Sobre adjudication ya abierto se reporta por brazo, IID, grouped y balanceado:

- acción calibrada y desplegada frente a identidad;
- selected-action topology menos simultaneous topology de R363;
- topology menos constant, public-base y promedio topology-permuted;
- `q_selected` frente al `q_simultaneous` de la misma familia;
- tasa de acción y distribución de alphas;
- cobertura empírica de la acción propuesta;
- daño entre vistas actuadas e interacción grouped menos IID;
- solapamiento con la política simultánea preservada.

El master sigue siendo la unidad bootstrap. Primero se promedian los dos seeds
neuronales y luego el par IID/grouped para balanceado. Los intervalos de
adjudication son descriptivos post hoc y pointwise.

## Lecturas admisibles

- Más acción con efecto compatible y mejor orden frente al límite simultáneo
  sostiene que la jurisdicción conjunta era conservadora; no confirma la nueva
  política.
- Más acción sin beneficio o sin ventaja frente al sham indica que reducir el
  margen sólo expone error, no señal útil.
- Si topology no supera public-base o permutado, no recibe crédito adicional
  por localización aunque el cambio de interfaz sea informativo.
- Si el firewall elige identidad, no se relaja sobre adjudication.
- Ningún resultado reescribe R363 ni autoriza promoción, transferencia material
  o GO/NO-GO.

## Artefactos y recursos

Se preservan source hashes, propuestas, scores escalares, cuantiles, uppers,
acciones, firewall, efectos por master, bootstraps, manifest, entorno y replay.
El output canónico será
`data/geometria_proporcional/proportional_graph_selected_action_calibration_diagnostic_v1/`.

La ejecución usa `CUDA_VISIBLE_DEVICES=''`, un thread, máximo `3 min` y `4 GiB`.
No genera vistas ni ejecuta IRLS, por lo que se espera menos de un minuto y de
`1 GiB`. La GPU continúa suspendida y este diagnóstico no la consulta, reserva
ni consume.

## Validación informática previa

El runner verifica los `102` archivos deterministas de R363 y reconstruye el
diagnóstico sólo desde predicciones, deltas y bootstraps ya preservados. Un
piloto completo terminó en `3,32 s / 0,692 GiB`, produjo `9` archivos
deterministas y `258` arrays NPZ sin valores no finitos. En todas las familias y
réplicas se verificó `q_selected <= q_simultaneous`, consecuencia esperada de
seleccionar un componente antes de tomar el orden estadístico. La suite
proporcional acumulada queda en `158 passed`. Esto valida mecánica, no outcomes
ni autoridad científica.
