---
schema_version: 1
id: phideus-proportional-architecture-experiments
kind: roadmap
page_status: current
front_status: focus_active
architecture_status: candidate
experiment_status: neural_smoke_executed
evidence_status: CPU contract, classical baselines and two-seed neural factorial executed with byte-exact replay; solver-dependent signal and no architecture promoted
decision_status: pending_user
updated: 2026-09-03
verified_at: 2026-09-03
valid_at: 2026-09-03
recorded_at: 2026-09-03
evidence_commit: 3a683ac9a7ef444e344b746cdd83062ff03ff30a
source_paths:
  - Documents/05_WIKI/concepts/ppu-geometria-armonica-natural.md
  - Documents/05_WIKI/fronts/atencion-armonica.md
  - Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/GEOMETRIA_PROPORCIONAL_BASES_DE_VERDAD.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_17_AUDIT_RESOLUTION.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_17_PROTOCOL_DRAFT.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/332_proportional_architecture_protocol_independent_audit.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/333_proportional_architecture_protocol_reaudit.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/334_proportional_graph_preflight_implementation_audit.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/335_proportional_graph_preflight_implementation_reaudit.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/336_proportional_graph_preflight_config_focal_audit.md
  - experiments/geometria_proporcional/run_proportional_graph_preflight.py
  - experiments/geometria_proporcional/configs/proportional_graph_preflight_v1.json
  - src/geometria_proporcional/proportional_graph_contract.py
  - data/geometria_proporcional/proportional_graph_preflight_v1/PREFLIGHT_REPORT.md
  - experiments/geometria_proporcional/configs/proportional_graph_neural_smoke_v1.json
  - experiments/geometria_proporcional/run_proportional_graph_neural_smoke.py
  - src/geometria_proporcional/proportional_graph_neural.py
  - data/geometria_proporcional/proportional_graph_neural_smoke_v1/SMOKE_REPORT.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/338_proportional_graph_neural_smoke_reaudit.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/339_proportional_graph_neural_smoke_reaudit.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/340_proportional_graph_neural_smoke_final_reaudit.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/341_proportional_graph_neural_smoke_closure_audit.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/342_proportional_graph_neural_smoke_official_analysis.md
depends_on: [ppu-natural-harmonic-geometry, front-atencion-armonica]
tangents: [phideus-evidence-regime, phideus-three-routes]
---

# Programa de arquitecturas y experimentos proporcionales

## Cambio de régimen

La campaña de investigación expansiva queda detenida. El corpus local acumulado
se usa ahora como un cuerpo cerrado de diseño: su función inmediata no es abrir
otra taxonomía, sino obligar a formular módulos neuronales concretos, compararlos
con controles fuertes y producir evidencia experimental.

Esto no prohíbe recuperar un paper, descargar una fuente ya identificada o
verificar un detalle de implementación. Esas consultas quedan subordinadas a
una dependencia explícita de un experimento. Las preguntas que no bloquean la
ejecución se registran como deuda y no abren una nueva ola por sí solas.

## Lo que el corpus ya permite afirmar

La evidencia disponible no sostiene una geometría universal de las
proporciones. Sí converge en restricciones de diseño suficientemente concretas:

1. las relaciones deben ser estados de primera clase, no features incidentales;
2. gauge, unidades, dominio y equivalencia deben formar parte del contrato;
3. composición relacional y decisión puntual son operaciones distintas;
4. un output puede ser una clase o un conjunto, no necesariamente un punto;
5. executor y checker exactos deben permanecer fuera del crédito neuronal;
6. el valor de una primitive sólo se atribuye frente a baselines clásicos,
   genéricos y barajados, declarando por separado qué dimensiones de capacidad
   y cómputo quedaron efectivamente igualadas;
7. la evaluación decisiva debe retener mecanismo, cardinalidad, topología o
   régimen, no limitarse a IID aleatorio.

Atención Armónica aporta evidencia previa a favor del estado explícito de pares
y, en aquel banco, de la actualización triangular combinada con un clusterer
global bajo OOD de polifonía. No aísla una ley general de composición ni
autoriza transferir el resultado fuera de esa jurisdicción. Las Olas 49–56
aportan otra separación: representar compatibilidades, modelar el posterior
sobre conjuntos y decidir bajo utilidad son problemas distintos. El posterior
conjunto mejoró NLL y cardinalidad, pero ni una decisión bayesiana pura ni una
compuerta escalar convirtieron de manera estable esa mejora en una política
superior.

## Cartera arquitectónica

| Línea | Estado | Primitive puesta en riesgo | Experimento discriminante |
|---|---|---|---|
| Núcleo local de coherencia proporcional sobre grafos | candidata inmediata | evidencia exacta de cierre frente a mixer tipado | factorial causal sobre grafos contaminados, CPU-first |
| Posterior de conjuntos y política contextual | experimental existente | incertidumbre conjunta y decisión desacoplada | terminar la prueba prospectiva fresca de la Ola 56 |
| Lector de espectro relativo | candidata matemática | lectura de una relación SPD orientada completa | requiere fijar una query con autoridad externa |
| Router tipado con executors | arquitectura de integración | selección de relación/solver y abstención | sólo después de validar al menos una primitive estrecha |

Las cuatro líneas se preservan, pero no tienen el mismo rango. La primera es el
próximo diseño de arquitectura. La segunda es una deuda experimental ya
implementada y separada. Las dos últimas no deben convertirse todavía en una
mega-arquitectura.

## Arquitectura 1: núcleo local de coherencia proporcional sobre grafos

### Objeto, orientación y simetría

El banco parte de cantidades positivas latentes `q_i`. En coordenadas
logarítmicas, `x_i = log(q_i)`, una razón observada sobre una arista orientada es

```text
y_ij = x_j - x_i + ruido + corrupción.
```

El objeto identificable no es el vector absoluto `x`, sino su clase bajo
`x -> x + c`. Esa traslación global en log-espacio es la reescala común de las
cantidades originales. El modelo debe respetarla por construcción y ser
equivariante a relabeling de nodos.

Se fija una convención orientada única. Para cada arista canónica `e=(i,j)`, la
fila `e` de la matriz de incidencia `B` contiene `-1` en `i` y `+1` en `j`, de
modo que `(Bx)_e = x_j-x_i`. Cuando un camino recorre la arista al revés se usa
`y_ji=-y_ij`; no se duplica la observación como si fuera otra muestra.

### Forward mínimo

```text
batch público de relaciones observadas + máscara + covarianza autorizada
        |
        +--> track RAW: operandos observados
        |
        +--> track RAW+EXACT-CLOSURE: mismos operandos + r_ijk exacto
        |
        v
encoder compartido de aristas
        |
        +--> mixer GENERIC
        |
        +--> mixer COMPOSITION-TYPED
        |
        v
relación corregida y_tilde + peso/confianza por arista
        |
        +--> métricas pre-solver de atribución
        |
        +--> WLS congelado / Huber-IRLS congelado
                    |
                    v
             potenciales módulo gauge
                    |
                    v
             checker exacto externo
```

El experimento no atribuye automáticamente al mixer el cálculo analítico de un
residual. Separa dos factores: qué evidencia recibe y cómo la mezcla. La red
aprende una corrección de relación `y_tilde_e` y un peso positivo `w_e`; no
aprende a resolver el sistema lineal. Cada solver externo aplica la misma
interfaz a todos los brazos. Para WLS:

```text
x_hat = argmin_{mean(x)=0} ||W^(1/2) (B x - y_tilde)||².
```

`w_e = epsilon + (1-epsilon)*sigmoid(s_e)`, con `epsilon` fijado antes de test y
normalización `mean_e(w_e)=1`. El grafo público es conexo; el checker exige
`rank(B)=n-1`, registra la condición del Laplaciano ponderado y rechaza como
fallo numérico cualquier solve que no satisfaga el contrato. El segundo solver
usa Huber/IRLS con receta y tuning congelados en validación. Las salidas
pre-solver se leen antes de los resultados downstream, y la interacción
`brazo x solver` se reporta explícitamente.

### Factorial causal y bloque candidato

El contraste principal cruza dos niveles de evidencia y dos mixers:

| Factor | Nivel | Contenido |
|---|---|---|
| evidencia | `RAW` | `y_ij`, `y_ik`, `y_kj`, roles, máscara y covarianza pública; ninguna operación exacta multioperando |
| evidencia | `RAW+EXACT-CLOSURE` | lo anterior más `r_ijk = y_ij-(y_ik+y_kj)` computado externamente |
| mixer | `GENERIC` | MLP/atención libre con roles identificados |
| mixer | `COMPOSITION-TYPED` | weight sharing, orientación y antisimetría tipadas; en `RAW` no puede calcular por fórmula suma de caminos ni cierre |

Esto produce `RAW-GENERIC`, `RAW-TYPED`, `CLOSURE-GENERIC` y
`CLOSURE-TYPED`, más `EXACT-CLOSURE-ONLY` sin parámetros. Si una versión tipada
necesita recibir la suma `y_ik+y_kj` o la discrepancia exacta, pertenece al
track `CLOSURE`; el comparador genérico recibe exactamente el mismo tensor. El
efecto del residual, el efecto del mixer y su interacción se estiman por
separado. Sólo la diagonal `RAW-GENERIC` frente a `CLOSURE-TYPED` acredita el
paquete completo.

En `RAW`, el bloque tipado sólo puede imponer roles, signo bajo inversión,
weight sharing y equivariancia. Una forma mínima es:

```text
p_ijk = MLP_path([z_ij, z_ik, z_kj, orientation_roles])
a_ijk = softmax_k(score(p_ijk))
m_ij  = sum_k a_ijk * p_ijk
z'_ij = z_ij + Gate(z_ij, m_ij)
y_tilde_ij, s_ij = Heads(z'_ij)
```

En `RAW+EXACT-CLOSURE`, `r_ijk` se concatena al mismo `p_ijk` para ambos mixers.
Los shuffles de paths reasignan conjuntamente los operandos `(ik,kj)` dentro de
estratos congelados y recomputan `r_ijk`; nunca barajan sólo el residual dejando
operandos contradictorios. Este v0 sólo reclama integración **local de caminos
de dos saltos**. Grafos sin triángulos o con girth alto son controles de
cobertura; los ciclos largos requieren un mecanismo recurrente propio y otra
ablación.

### Salidas y tracks de aprendizaje

El primer corte no incluye router multigeometría, política de abstención ni
calibrador aprendido. Produce:

- relación corregida `y_tilde_e` y peso `w_e` por arista;
- probabilidad de procedencia causal de la alteración sólo en el track
  supervisado que la autoriza;
- potenciales reconstruidos módulo media cero, producidos por cada solver
  externo;
- diagnósticos de cierre, condición numérica y fallo del solve.

Hay dos tracks de entrenamiento que no comparten interpretación:

1. `RECONSTRUCTION-ONLY`: `L_quotient` a través del solver diferenciable más
   `lambda_rel*L_relation` y `lambda_cycle*||C y_tilde||_1`; no ve la máscara
   causal.
2. `CAUSAL-PROVENANCE-SUPERVISED`: añade BCE sobre la máscara que el simulador
   alteró. AP/Brier significan recuperación y calibración de esa procedencia,
   no detección ontológica de una única “arista falsa”.

`C` es una base de ciclos del grafo observado y el término de cierre se calcula
sobre `y_tilde`; calcularlo sobre `Bx_hat` sería idénticamente cero y hacerlo
sobre `y` crudo no dependería de la red. `L_relation` compara `y_tilde` con la
relación limpia sólo en el banco sintético y se elimina en una transferencia
sin esa autoridad. Los `lambda` se fijan en validación. Brier, ECE y
riesgo-cobertura se reportan, pero no se añade una `L_calibration` indefinida.

Los mundos con varias descomposiciones observacionalmente equivalentes forman
un slice `NONIDENTIFIABLE`: no reciben accuracy binaria. Se conserva su clase o
posterior certificado y se puntúa cobertura o proper scoring; los casos cuyo
checker no resuelve quedan `COMPUTATIONALLY-UNRESOLVED` y fuera del claim de
localización.

## Experimento 1: benchmark de coherencia local y gauge

### Pregunta

¿Un mixer con orientación y antisimetría tipadas corrige relaciones mejor que
un mixer genérico cuando ambos reciben la misma evidencia? ¿Qué parte de la
mejora proviene de entregar un cierre analítico exacto? ¿La salida conserva
utilidad bajo dos solvers externos y bajo un cambio de mecanismo de corrupción?

### Generación y frontera anti-leakage

Cada master contiene un grafo conexo, potenciales latentes, razones exactas y
una observación parcial contaminada. El generador conserva el estado limpio en
un sidecar privado. El batch público tiene una whitelist cerrada: índices
locales reindexados, `y_e` observado, `edge_exists`, `edge_valid`,
`path_incidence_valid` y covarianza generada independientemente de la máscara
causal. Quedan excluidos `q`, `x_true`, razones limpias, máscara causal,
mecanismo, seed, lineage, split, IDs persistentes y salidas oracle. Un test de
schema compara todos los brazos y persiste hashes del constructor y la
whitelist.

Todas las vistas, corrupciones y controles derivados de un mismo mundo
conservan una lineage única y nunca cruzan train, calibration, validation o
test. El master, no la arista ni el camino, es la unidad inferencial.

| Split | Cambio retenido |
|---|---|
| ID | tamaños, familias topológicas y mecanismo vistos |
| OOD-size | más nodos, misma familia, densidad y mecanismo |
| OOD-topology | familias generadoras completas retenidas; tamaño y corrupción igualados |
| OOD-corruption primario | dependencia de outliers IID hacia agrupada, con tasa, amplitud, tamaño y topología igualados |
| Gauge/permutation sanity | reescala latente común y relabeling pareados; no son nuevas muestras ni splits OOD |

El benchmark es sintético y sólo puede validar la mecánica. No acredita una
geometría física natural.

### Brazos y controles obligatorios

| Brazo | Función causal |
|---|---|
| mínimos cuadrados sin pesos | piso exacto no robusto |
| Huber/IRLS | baseline clásico robusto |
| factorial `RAW/CLOSURE x GENERIC/TYPED` | separa evidencia analítica, mixer e interacción |
| `EXACT-CLOSURE-ONLY` | mide cuánto agota el target la primitive exacta sin red |
| MLP por arista | evidencia local sin composición |
| pair-state sin mezcla | efecto de representar relaciones explícitamente |
| message passing genérico, igualado en parámetros/shapes/inicialización pero no en FLOPs | capacidad global sin tipado proporcional; no aísla tipado por sí solo |
| path-incidence shuffle balanceado | falsación causal de la incidencia de caminos |
| orientación coherentemente invertida | sanity de convención; debe transformar, no degradar |
| direct decoder centrado | diagnóstico pre/post-executor; no entra al contraste del mixer |
| pesos oracle | referencia privilegiada, nunca baseline deployable |

Los cuatro brazos factoriales comparten encoder, reader, ancho, profundidad,
datos y presupuesto entrenable. El manifiesto congela parámetros, FLOPs
entrenables, operaciones exactas, latencia, memoria y espacio de tuning. Los
solvers reciben la misma interfaz y se cruzan con cada brazo; ningún lift
post-solver se atribuye por reflejo al núcleo.

### Métricas y unidad inferencial

- superficie primaria de atribución: error de relación corregida y score de
  confianza, por master y autoridad del target;
- superficie primaria de utilidad: RMSE de potenciales en el cociente por
  master, reportada factorialmente por solver;
- secundarias: AP/Brier de procedencia causal, residual por longitud, error de
  ratios reconstruidos, condición numérica y failure rate;
- sanities: invariancia ante reescala latente, equivariancia ante permutación y
  orientación, y exactitud de los solvers en datos limpios;
- bootstrap pareado por master y resultados por seed antes del ensemble.

La inferencia se formula por split. `OOD-corruption` es el contraste
confirmatorio único del primer corte; ID, size y topology son secundarios y se
ajustan como familia cuando corresponda. Una mejora allí no se generaliza a
otros mecanismos ni a dominios físicos. El bootstrap es jerárquico por master;
las familias topológicas sólo sostienen un claim poblacional si su número lo
permite, y de otro modo se informan una por una.

### Escalera de ejecución y presupuesto

1. **Contrato y clásicos, CPU — ejecutado.** `256` masters de preflight, `n=8..16`:
   generador, target-authority table, checker, WLS e IRLS; verificar orientación,
   gauge, permutación, rank, condición y anti-leakage.
2. **Smoke neuronal, CPU — ejecutado.** `512/128/256` masters train/val/test, dos seeds,
   `10` épocas, ancho `64`, dos bloques y batch `64`; correr los cuatro brazos
   factoriales y controles mínimos bajo techo de `2 h` y `8 GiB` de RAM. Este
   corte sirve para depurar, no para claims.
3. **Desentrelazado de solver, CPU.** Reusar los estados raw para cruzar relación
   cruda/corregida y peso unidad/aprendido bajo WLS e IRLS, sin re-forward. El
   objetivo es localizar si la pérdida aparece en la corrección, en el peso o
   en su interacción con el solver robusto.
4. **Freeze confirmatorio.** Congelar generador, primary split, manifests,
   hiperparámetros y hashes; estimar tiempo real. Ejecutar tres seeds y reportar
   cada seed más ensemble. Si la proyección supera `12 h` CPU, avisar antes de
   usar GPU con duración y VRAM estimadas.
5. **Transferencia de primitive.** Sólo si la composición aporta, probar el
   mismo bloque sobre agrupamiento armónico render-then-detect o sobre otro
   banco relacional ya autorizado. No redefinir el operador después de ver el
   destino.

### Resultado del primer escalón

El preflight clásico produjo `320` vistas desde `256` masters: `128` de train,
`32` de calibración, `32` de validación y `64` masters de test con vistas
pareadas `iid/grouped`. Los `320/320` solves Huber-IRLS convergieron; el máximo
residual de ciclo limpio fue `3.96e-15`, mientras la mediana observada fue
`0.394`, de modo que el checker distingue cierre exacto de observación
contaminada sin leer autoridad privada.

En RMSE de cociente, IRLS mejoró a WLS en todos los slices. En test IID pasó de
`0.202` a `0.109`; bajo corrupción agrupada pasó de `0.204` a `0.178`. La
degradación pareada IID→agrupada de IRLS muestra que el cambio de dependencia
es un desafío real para el baseline robusto, aunque estas cifras son todavía
diagnósticas y no comparan mixers neuronales. La referencia con pesos oracle,
que usa la máscara causal privada, quedó en `0.045` IID y `0.077` agrupado: mide
solvabilidad, no constituye un baseline deployable.

El artefacto oficial conserva `1.927` archivos manifestados, hashes de las
cuatro fuentes ejecutables, inputs públicos, sidecars privados, salidas crudas
de tres solvers, índices de bootstrap y replay byte-exacto. La suite focal dio
`10 passed`; la regresión del frente completo dio `173 passed`. Las auditorías
independientes R334–R336 cerraron convergencia, máscaras, replay, trazabilidad y
versionado de la configuración canónica. El estado pasa por ello de protocolo
auditado a **preflight clásico ejecutado**. Ese resultado habilitó, sin
decidir, el smoke neuronal CPU.

No se declara por adelantado un GO ni se inventa un efecto mínimo. El informe
estima el contraste primario `TYPED-GENERIC` dentro de cada nivel de evidencia,
su interacción con `RAW/CLOSURE` y la interacción `brazo x solver` en
`OOD-corruption`, con CI pareado por master. Se congelan antes de test la
dirección del contraste, el orden de lectura y la corrección de multiplicidad
para la familia secundaria. Si el path shuffle no está balanceado o no degrada,
si `EXACT-CLOSURE-ONLY` agota la tarea, o si el lift aparece sólo con un solver,
la atribución al mixer queda rechazada aunque el paquete completo funcione. La
promoción arquitectónica pertenece al usuario.

### Resultado del segundo escalón

El smoke neuronal oficial corrió los ocho brazos y dos seeds sobre un universo
común: de `1.280` vistas generadas excluyó `26` sin shuffle balanceado factible
y conservó `496` train, `127` validation y `504` test, estos últimos como `252`
masters `iid/grouped` pareados. Los `16` trainings de `10` épocas terminaron en
`1.320,29 s` y `1,047 GiB` de RSS máximo. La repetición independiente terminó en
`1.425,12 s`; los dos paquetes tienen los mismos `48` archivos, el manifest y
los `46/46` artefactos deterministas son byte-exactos, y sólo la observación de
runtime difiere como estaba predeclarado.

El efecto tipado aparece, pero no es monolítico. En RAW reduce el RMSE de
relación frente al mixer genérico en `-0,0058` IID y `-0,0040` grouped, y reduce
WLS en `-0,0056` y `-0,0015`; con CLOSURE la reducción de relación es
`-0,0065/-0,0023`, mientras WLS queda en `-0,0080` IID y `-0,0003` grouped, con
el intervalo grouped cruzando cero. El decoder directo se mueve en dirección
contraria: el tipado aumenta su error. La interacción factorial también cambia
por slice; entregar cierre exacto amplía levemente la ventaja tipada en IID,
pero la reduce o invierte bajo corrupción grouped.

Los controles localizan mejor la capacidad. Frente al path shuffle,
`CLOSURE-TYPED` reduce el error de relación en `-0,0754` IID y `-0,0733`
grouped. Frente a pair-state sin mezcla, `RAW-GENERIC` lo reduce en
`-0,0700/-0,0723`. `EXACT-CLOSURE-ONLY` no agota la reconstrucción: su WLS es
`0,2060` IID y `0,2259` grouped. Sin embargo, la mejora pre-solver no se
transporta de modo uniforme: IRLS sobre observación cruda alcanza
`0,1142/0,1825`, mientras RAW-TYPED queda en `0,1768/0,1971`. Hubo `8` fallos
IRLS sobre `11.989` evaluaciones; el estimando conservador dejó no evaluables
las comparaciones afectadas en lugar de descartarlas silenciosamente.

La lectura es por ello doble. El mixing de caminos y el tipado contienen señal
para corregir relaciones y mejorar WLS en parte del factorial, pero una única
salida de relación y confiabilidad no sirve igual a WLS, decoder directo e
IRLS. La brecha hasta pesos oracle —WLS `0,0360` IID y `0,0976` grouped— sigue
siendo amplia. Antes de un freeze confirmatorio corresponde un contraste CPU
que desacople relación cruda/corregida y peso unidad/aprendido usando los
estados ya preservados. Esto registra una alternativa solver-específica; no
promueve arquitectura ni constituye GO/NO-GO.

### Artefactos obligatorios

Cada ejecución conserva checkpoints `last_epoch`, config resuelta, seeds,
manifiesto de lineages y splits, hash del schema público, commit y entorno. Por
master y arista guarda inputs públicos, target privado, `y_tilde`, pesos,
scores pre-solver, potenciales post-solver, diagnósticos de rank/condición,
fallos y pertenencia a slice. Se preservan índices de bootstrap y un comando de
replay. Los crudos permiten cambiar métricas, solver o bootstrap sin re-forward.

## Arquitectura 2: posterior de conjuntos y política contextual

Esta línea ya tiene componentes ejecutados:

```text
encoder DeepSets
 -> cuatro logits de compatibilidad
 -> posterior sobre 15 conjuntos no vacíos
    (unary + cardinalidad + interacciones)
 -> riesgo esperado bajo utilidad
 -> acción dura o bayesiana
 -> compuerta contextual residual
```

Su singularidad potencial no es geométrica todavía. Consiste en conservar una
región compatible y postergar la decisión hasta recibir utilidad y contexto. La
Ola 54 mostró que el posterior conjunto modela mejor dependencias y
cardinalidad; las Olas 55–56 mostraron que traducir esa mejora a una acción
estable sigue abierto.

El experimento inmediato de esta línea es terminar la prueba prospectiva fresca
ya diseñada para la compuerta contextual. Es CPU-only y debe conservarse como
un contraste sobre decisión, no presentarse como prueba del núcleo proporcional.
Su implementación de recuperación está preservada; antes de abrir inferencia
oficial resta completar su auditoría independiente y la cadena de cierre.

## Arquitectura 3: lector de espectro relativo

Para pares SPD, un executor exacto puede construir el espectro relativo
orientado `ell(B|A) = {log lambda_i(B|A)}`. Un reader permutation-invariant
podría responder queries sobre esa relación sin reducirla de antemano a
determinante, extremos o espectros ordinarios separados.

La arquitectura es concreta, pero el experimento todavía no: falta una query
externa cuya respuesta no sea una función clásica ya entregada por el executor.
Esa carencia se registra como deuda de diseño y no justifica volver ahora a una
campaña bibliográfica.

## Arquitectura 4: router tipado con executors

El horizonte de integración es un sistema modular:

```text
contrato de objeto
 -> constructor de IR relacional
 -> router aprendido con abstención
 -> primitive estrecha
 -> executor tipado común
 -> checker y ledger externos
 -> reader de punto/clase/conjunto/certificado
```

El router sólo podría elegir entre relaciones y executors autorizados por el
contrato. No tendría permiso para inventar una geometría y validarla con su
propia salida. Esta arquitectura no se implementa hasta que una primitive
estrecha demuestre valor: construirla antes impediría atribuir cualquier
resultado.

## Orden de trabajo propuesto

1. cerrar administrativamente la recuperación de la prueba prospectiva de la
   Ola 56 sin extender su investigación;
2. congelar y auditar el protocolo factorial de coherencia local;
3. implementar contrato, clásicos y smoke neuronal en CPU — completado;
4. ejecutar el desentrelazado CPU de relación, peso y solver desde los crudos;
5. mantener cualquier contraste GPU en cola mientras rige la suspensión del
   dispositivo y, después, decidir si un freeze confirmatorio está justificado;
6. sólo después estudiar integración con el posterior set-valued o transferencia
   a Atención Armónica.

## Deudas registradas, no abiertas

- query externa no agotada por el eigensolver para el lector SPD;
- mecanismo multi-hop para ciclos largos, con ablación propia;
- transferencia desde log-razones escalares hacia grupos no conmutativos;
- criterio físico externo para pasar de coherencia interna a geometría natural;
- integración eventual entre posterior set-valued, abstención y router tipado.

Estas deudas no bloquean el primer benchmark y no autorizan nuevas olas de
investigación hasta que un resultado experimental las vuelva necesarias.
