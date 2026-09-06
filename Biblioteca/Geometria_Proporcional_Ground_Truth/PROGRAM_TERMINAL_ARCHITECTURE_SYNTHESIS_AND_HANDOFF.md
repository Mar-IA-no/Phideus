# Síntesis terminal de arquitectura y relevo experimental

> **Estado:** `RESEARCH-BASE-CLOSURE-CANDIDATE / THREE-LINES-PRESERVED / ONE-FINITE-DISCRIMINANT / CPU-FIRST / NO-GO-NOGO`
> **Fecha:** 2026-09-06
> **Corte de evidencia:** Olas 1–60, incluida la auditoría R531
> **Autoridad de promoción:** usuario

## Qué termina y qué permanece abierto

La campaña acumulativa ya cumplió una función que no mejora por sumar otra ola
bibliográfica general. Estableció qué puede contar como evidencia proporcional,
qué gauges y equivalencias deben declararse, por qué una salida puede ser una
clase o un conjunto, cómo se separan proposer, solver, checker y reader, y qué
controles impiden atribuir a una red una operación exacta o una decisión que
proviene de otra capa. También materializó benchmarks y experimentos CPU que
permitieron descartar atajos concretos.

Ese cierre no resuelve la tesis científica de Phideus. Transforma una pregunta
abierta e indefinida —encontrar «la» geometría proporcional— en una cartera
finita de arquitecturas y en un contraste capaz de decidir qué componente
merece una prueba de escala. El corpus queda como base trazable; las búsquedas
externas futuras sólo se justifican cuando una dependencia específica del
experimento no pueda resolverse localmente.

## Cartera reducida

Las tres líneas no son tres caballos equivalentes. Las dos primeras arriesgan
primitives distintas; la tercera organiza su eventual composición y sólo cobra
sentido si al menos una de ellas obtiene evidencia afirmativa.

| Línea | Estatuto | Operación que arriesga | Evidencia favorable | Evidencia adversa o faltante |
|---|---|---|---|---|
| Núcleo relacional tipado con adaptación por executor | candidata inmediata | producir un estado de relaciones orientadas, gauge-equivariantes y composicionales que un executor externo pueda usar | el tipado y la mezcla de caminos redujeron error relacional; WLS aprovechó pesos localizados en los ocho slices del desentrelazado | decoder directo e IRLS no conservaron la ventaja; dos seeds y dominio sintético; interfaz no solver-agnóstica |
| Posterior de conjuntos con política y guard separados | candidata recuperable | preservar incertidumbre conjunta y postergar la acción hasta una política explícita | el posterior conjunto mejoró NLL, cardinalidad e interacciones; varias políticas contextuales mejoraron regret o compatibilidad frente a `hard` | Wave 54 no mejoró la decisión; Olas 59–60 no separaron controles matched; el bracket HGB/HGB dejó de ser continuación prioritaria |
| Router tipado con IR, executors, checkers y reader externos | arquitectura de integración condicionada | despachar sólo relaciones autorizadas y conservar punto, clase, conjunto, certificado o abstención sin fusionar jurisdicciones | el corpus, `BudgetPath` y su puerto de utilidad demostraron que estructura, incertidumbre y decisión pueden permanecer separadas y checker-valid | ninguna primitive estrecha tiene aún evidencia suficiente para justificar el ensamblaje; construirlo ahora diluiría la atribución |

### Núcleo relacional tipado con adaptación por executor

Esta línea representa cantidades positivas mediante diferencias en log-espacio,
trata cada relación orientada como estado de primera clase y mezcla caminos bajo
convenciones explícitas de signo, relabeling y gauge. Su salida no es una
solución final: produce una relación corregida y una interfaz de confianza que
un solver externo convierte en potenciales módulo escala.

El smoke neuronal mostró que el mixer tipado mejora la relación frente al
genérico tanto en IID como bajo corrupción agrupada. El desentrelazado precisó
la frontera: el peso aprendido sobre la relación observada favorece WLS en los
ocho slices primarios, mientras una única pareja `relación corregida + peso`
degrada IRLS en los seis slices evaluables. La oportunidad arquitectónica no es
una cabeza universal más grande, sino un estado relacional común con adaptación
de salida declarada por executor.

La evidencia en contra es material. La ventaja procede de un smoke de dos seeds,
la interacción con grouped se debilita y algunos controles reproducen la
inversión WLS/IRLS. La línea merece un contraste nuevo porque contiene señal
pre-solver; todavía no merece promoción.

### Posterior de conjuntos con política y guard separados

Esta línea conserva varias estructuras compatibles en lugar de forzar una
clase puntual. Un posterior sobre conjuntos expresa cardinalidad e
interacciones; una política posterior introduce utilidad; un guard distinto
autoriza o rechaza la acción según riesgo. La separación permite que mejorar la
representación no sea confundido con mejorar la decisión.

La Ola 54 confirmó que el posterior conjunto captura dependencias que las
marginales independientes omiten. Las Olas 56–60 mostraron además que existe
señal contextual transportable frente a `hard`. El límite es igualmente claro:
una mejor distribución no produjo por sí sola una política mejor, y dos draws
prospectivos no atribuyeron el efecto de las políticas HGB/HGB más allá de
controles de acción matched. La arquitectura permanece recuperable como
interfaz de incertidumbre, mientras su implementación HGB/guard actual queda
cerrada como prioridad.

### Router tipado con autoridades externas

El router es la forma de integración que emerge del corpus: contrato del
objeto, IR relacional, selección con abstención, executor autorizado, checker,
ledger y reader. Su valor reside en impedir que una red invente simultáneamente
la geometría, el criterio de validez y la decisión con la que será evaluada.

`BudgetPath` y el puerto de utilidad prueban la mecánica de esa distribución de
responsabilidades: una ruta set-valued puede permanecer inmutable, recibir una
preferencia posterior, devolver todos los empates y abstenerse cuando ninguna
política es factible. Pero ese resultado usó fixtures sintéticos y cero rutas
históricas con utilidad real. El router no es el próximo modelo; es el contrato
que sólo debe ensamblarse si una primitive estrecha sobrevive al contraste.

## Experimento discriminante finito

El próximo experimento debe distinguir si el cuello está en la representación,
en la política o en su interacción. La evidencia vigente todavía no autoriza a
tratar esos tres módulos como celdas de un mismo factorial: EIV adjudica familias
desde mediciones con error, el núcleo grafo produce relaciones y pesos para
WLS/IRLS, y el posterior conjunto opera sobre logits de compatibilidad. El
primer resultado del nuevo goal será por eso un gate de factibilidad del mapeo,
anterior a cualquier comparación neuronal conjunta.

### Gate previo: factibilidad del mapeo

El gate `MAPPING-FEASIBILITY` debe fijar una única query, objeto científico,
unidad inferencial, schema público y target lógico. Para cada brazo especificará
dos funciones tipadas: observación → estado y estado → scores de
compatibilidad. EIV se descompondrá en score, calibración conformal, reader y
abstención; no se presentará como un encoder ni como una celda causal si esa
factorización no conserva su significado. El núcleo grafo sólo podrá ingresar
si un adapter explícito transforma su relación y peso sin retirar orientación,
gauge o composición ni añadir información que los otros brazos no reciben.

El gate también debe declarar qué executor y checker son realmente comunes,
marcar la utilidad como sintética y externa, y probar mediante mutaciones que
ningún adapter cambia target, autoridad o acceso. «Mismos bytes» significa
igualdad del input autorizado, no una unión de campos de la que cada brazo pueda
elegir privilegios diferentes.

Si estas condiciones pasan, se abre el factorial conjunto descrito abajo. Si
no pasan, el protocolo se bifurca sin prolongar el goal:

1. un contraste relacional `GENERIC/TYPED × WLS/IRLS` bajo la IR nativa de
   relación y peso, con EIV como referencia externa cuando corresponda;
2. un contraste set-valued `marginal/joint × hard/contextual` bajo scores de
   familia comunes.

Los dos contrastes conservarán draws, controles y ledger coordinados, pero no
se combinarán en una interacción que sus objetos no permitan interpretar. El
router seguirá diferido.

### Factor 1: representación

1. referencia clásica `EIV + conformal`, sin crédito neuronal;
2. encoder genérico capacity-matched;
3. núcleo relacional tipado con orientación, gauge y composición local.

Sólo después de aprobar `MAPPING-FEASIBILITY` los tres brazos emitirán la misma
IR de compatibilidad e incertidumbre. Las operaciones analíticas exactas se
computarán fuera de la red y se entregarán a todos o a ninguno. El control
genérico igualará parámetros, shapes, updates y espacio de tuning; FLOPs y
latencia se reportarán sin fingir igualdad cuando no exista. Si EIV no puede
separarse del reader conformal bajo el contrato común, permanecerá como
referencia externa y el efecto representacional se estimará únicamente entre
los dos brazos neuronales comparables.

### Factor 2: decisión

1. reader duro y congelado;
2. política contextual con proposer y guard separados, entrenada únicamente en
   train/calibration y congelada antes del monitor.

Cada celda recibirá la misma utilidad contractual. La comparación incluirá
controles target-shuffled y controles matched por soporte, magnitud y
localización de acción. El objetivo es evitar que una política obtenga crédito
por actuar más o por concentrarse en casos distintos.

### Diseño de transporte

El protocolo separará cuatro roles físicos y abrirá cada uno una sola vez, en
este orden: `train → calibration → selection → freeze auditado → monitor`.
Train ajusta parámetros. Calibration puede elegir hiperparámetros, calibradores,
thresholds y márgenes dentro de reglas ya declaradas. Selection sólo aplica un
orden total congelado para elegir entre candidatos admitidos; no redefine
targets, métricas, márgenes ni controles. Un receipt posterior liga por hash la
elección completa y una auditoría independiente debe aceptarlo antes de abrir
monitor. Monitor únicamente adjudica la pipeline congelada.

| Campo | Autoridad de escritura | Evidencia de freeze |
|---|---|---|
| query, objeto, schema, target, controles y familia confirmatoria | plan previo a los draws | hash del design freeze |
| parámetros entrenables | train | manifest de checkpoints |
| hiperparámetros, conformal, thresholds, márgenes y potencia proyectada | calibration bajo reglas del plan | calibration freeze |
| identidad del candidato | selection mediante orden total predeclarado | selection receipt |
| métricas finales y patrones | monitor, sólo lectura de freezes | analysis y replay manifestados |

Ningún campo se elegirá después de abrir el monitor. El master será la unidad
independiente; las vistas relacionadas no se bootstraperán como réplicas
autónomas. IID, corrupción agrupada y al menos una familia topológica retenida
se reportarán por separado.

La lectura seguirá este orden:

1. proper scoring, cobertura y ancho del conjunto, antes de cualquier decisión;
2. efecto representación tipada menos genérica bajo el mismo reader;
3. efecto de política contextual menos hard dentro de cada representación;
4. interacción representación × política;
5. separación frente a controles matched;
6. replay exacto, fallos del solver y abstenciones.

La familia confirmatoria quedará fijada en el design freeze. Los márgenes
numéricos se derivarán únicamente bajo las reglas autorizadas para calibration,
junto con una estimación de potencia explícita, y quedarán congelados antes de
abrir selection. No se inventan en esta síntesis. `GO/NO-GO` y promoción
continúan siendo decisiones del usuario.

## Escalera operativa y condición de detención

La ejecución se divide en tres hitos finitos.

1. **Preflight CPU.** Ejecutar `MAPPING-FEASIBILITY`: auditar query, adapters,
   accesos y autoridad; decidir factorial común o dos contrastes coordinados;
   verificar los clásicos, shuffles y mutaciones; y medir el costo de un smoke
   pequeño.
2. **Freeze del contraste.** Congelar design y roles; ejecutar train,
   calibration y selection en ese orden; ligar checkpoints, calibradores,
   márgenes, candidato y controles en un receipt. Una auditoría independiente
   debe resolver findings antes de abrir monitor.
3. **Ejecución a escala proporcionada.** Si el costo medido es razonable en CPU,
   ejecutar allí. Si CPU exige muchas horas y GPU vuelve materialmente más
   eficiente la comparación neuronal, detenerse antes de cargar CUDA e informar
   objetivo, duración y VRAM estimados. La prueba queda en cola hasta indicación
   del usuario.

El experimento termina con uno de cinco estados informativos:

- no existe un mapeo semántico común y las dos líneas deben evaluarse por
  separado;
- la representación tipada no supera al genérico ni a la referencia clásica;
- la representación mejora, pero la política no añade valor frente a `hard` y
  controles matched;
- la política añade valor independiente de la representación;
- la interacción es necesaria y justifica estudiar el router compuesto.

Ninguno de esos estados declara por sí mismo una geometría física natural. Su
función es seleccionar el próximo objeto de investigación y cerrar las ramas
que el nuevo estimando ya no sostenga.

## Relevo de goal

La condición de cierre del goal acumulativo queda satisfecha cuando este corte,
el cierre de Ola 60 y la propagación documental hayan recibido auditoría
independiente sin findings materiales. El goal sucesor es deliberadamente más
estrecho: diseñar, implementar y ejecutar el preflight CPU del factorial
representación × decisión; congelar su protocolo; y detenerse con una
estimación verificable si la etapa discriminante requiere GPU.

La bibliografía deja de ser una corriente de trabajo autónoma. Una consulta
externa futura deberá nombrar qué componente, implementación o garantía del
experimento no puede resolverse con el corpus local y archivar únicamente esa
recuperación quirúrgica.

## Fuentes principales

- `waves/WAVE_49_CLASSICAL_BENCHMARK_CLOSED.md`
- `waves/WAVE_54_JOINT_SET_POSTERIOR_CLOSED.md`
- `waves/WAVE_59_FRESH_HGB_GUARD_BRACKET_CLOSED.md`
- `waves/WAVE_60_FROZEN_POLICY_TRANSPORT_CLOSED.md`
- `agent_reports/342_proportional_graph_neural_smoke_official_analysis.md`
- `agent_reports/349_proportional_solver_disentanglement_official_analysis.md`
- `agent_reports/373_proportional_budget_path_typed_interface_analysis.md`
- `agent_reports/374_proportional_budget_path_external_utility_port_analysis.md`
- `agent_reports/531_wave60_v4_replay_normalization_correction_audit.md`
