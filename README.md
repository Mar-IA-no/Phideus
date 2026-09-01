<div align="center">

# Phideus

### Harmonic Information Theory — Research Program

![Status](https://img.shields.io/badge/Status-Active_Research-0A7E3B?style=for-the-badge)
![Focus](https://img.shields.io/badge/Focus-Escalon_2-1F6FEB?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-111827?style=for-the-badge)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/AlterMundi/Phideus)

*Do frequency ratios constitute a universal informational language?*

</div>

---

## Phideus en una pagina

**Phideus** explora la **Harmonic Information Theory**: la hipotesis de que la armonia natural — razones lineales de frecuencia, serie armonica fisica, regularidades del oscilador — constituye un lenguaje informacional privilegiado para organizar, comprimir y alinear informacion entre modalidades distintas de un mismo fenomeno fisico.

El programa usa arquitecturas aprendidas como instrumentos experimentales. Si las relaciones armonicas naturales mejoran de forma causal, robusta y transferible la alineacion cross-modal entre sensores distintos — y lo hacen por encima de controles espectrales genericos y de codificaciones perceptuales —, eso constituye evidencia de que la armonia natural captura algo real de la organizacion informacional del fenomeno. La posicion epistemologica completa esta en [MARCO_EPISTEMOLOGICO_PHIDEUS.md](MARCO_EPISTEMOLOGICO_PHIDEUS.md).

**Escalon 1** (Audio <-> MIDI) establecio la mecanica: la inyeccion de descriptores reorganiza geometricamente el espacio latente y mejora retrieval de manera causal y robusta. Su resultado ya mas fuerte y metodologicamente homogéneo es `d4a4=84.0% +/-2.7pp` sobre **5 training seeds independientes**, contra `D0=75.2% +/-2.3pp`, con una separacion que sigue sosteniendo la lectura de ventaja geométrica descriptor-guided sin convertir por si sola a Escalon 1 en prueba cerrada de la tesis fuerte. **Escalon 2** (Speech <-> EGG) lleva esa mecanica al test directo de la hipotesis central: descriptores derivados de la **armonia natural** del oscilador glotal (ratios lineales de F0, estructura armonica intra-frame) contra controles espectrales y perceptuales. Al corte actual, ese frente ya cerró su primer null mecanistico: `concat`, `attn_bias`, `xattn` y `pca` dieron `12/12` condiciones `≈ D0` o peores, con `V4-lin + attn_bias` claramente por debajo. Eso no clausura la tesis fuerte, pero sí cierra el contraste sobre encoders from-scratch de este escalón. `S2-P3` ya no es fase futura: su primera pasada con encoder frozen (`WavLM-Large`) ya fue completada, y la tarea viva del frente pasa a ser el diagnostico comparativo **`P2 vs P3`**. En paralelo, Gate 9 / `A10` ya entregaron datos retrospectivos en musica y **Gate 10** ya cerró su barrido causal completo: `concat > FiLM/pca >> attn_bias`, con `a7-concat=76.4%` como mejor arm del gate y una lectura más fuerte de dominio del mecanismo sobre el descriptor. Gate 6 también se volvió más nítido: `Exp A` y `Exp B` ya cerraron negativamente en la rama `Transkun+A4`, mientras `Exp C` sigue como única línea downstream todavía abierta. **Escalon 3** ya dejó atrás la fase de apertura: `P1`, `P2`, `P4`, `P5` y `P6` ya fueron corridos en una primera pasada completa. La lectura vigente del frente es más precisa: `P2-flat` sigue como baseline general de `IID`, `P5-cqtshift` emerge como mejor brazo geométrico/OOD, y `P6` no supera a `P5` bajo la receta actual. En paralelo, el programa ya abrió dos frentes laterales con roles distintos: **Voz Expresiva Phideus**, que ya cerró su contraste `EN ↔ ZH` con una lectura más fina que un simple “replica / no replica”: en `N-adapt`, `concat` y `FiLM` replican limpio cross-language, mientras en `N-strict` el lift inglés no transfiere y `film/xattn` incluso se vuelven negativos en `ZH`; y **Atención Armónica**, que ya cerró `Fase 0`, `0.5` y `0.6`: el pair-state aparece como el salto grande, el `triangle` mejora la generalización `OOD-poly`, `connected-components` quedó falsado como lector suficiente de esa representación, y los clusterers globales deployables (`spectral`/`agglo` con `k` estimado) ya recuperan una ventaja real de `B` sobre `B-local` en `OOD-poly`. El caveat que queda es más preciso: no falta calibrar `τ`, falta resolver mejor la estimación de partición y de `k`.

Ese arco experimental ya tiene tambien una formulacion larga y teoricamente integrada en el repositorio publico del libro HIT, [AlterMundi/harmonic-information-theory](https://github.com/AlterMundi/harmonic-information-theory), con edicion web en [hit.altermundi.net](https://hit.altermundi.net/). Ahi el programa ya quedó articulado como libro de 191 páginas, incluyendo el nuevo problema de `storage / retrieval`, el `activation problem` y la convergencia con `Beacon` como parte del cierre teórico más largo del programa.

El cierre metodológico de Escalon 1 fue publicado como preprint arXiv: **[arXiv:2604.10283](https://arxiv.org/abs/2604.10283)** — *Descriptor-Injected Cross-Modal Learning: A Systematic Exploration of Audio–MIDI Alignment via Spectral and Melodic Features* (CC BY 4.0, `cs.SD` primaria, `cs.LG` cross-list).

---

## Programa actual

| Frente | Dominio | Funcion | Estado |
|---|---|---|---|
| **Escalon 1** | Audio <-> MIDI | Validacion descriptor-guided y geometria cross-modal | **Cerrado** — cierre training-seed `d4a4=84.0% +/- 2.7pp` |
| **Gate 8** | Audio <-> MIDI | Conditioned projections: donde se preserva la informacion descriptorial | **Cerrado (5/5)** — `pcd=84.2%`, `pca=82.6%` |
| **Gate 6 AMT** | Audio -> transcripcion | Validacion downstream de la senal descriptor-guided | **Activo** — `Exp A` y `Exp B` ya cerraron negativamente; `Exp C` queda como línea abierta |
| **Escalon 2** | Speech <-> EGG | Test directo de HIT: armonia natural del oscilador glotal como organizador cross-modal | **Foco principal** — null mecanistico inicial cerrado; `P3` primera pasada completa, sigue `P2 vs P3` |
| **Voz Expresiva Phideus** | Voz expresiva | Test de transferencia descriptor-guided sobre `SSL` vocal y estabilidad translingüística | **Activo** — cierre `EN ↔ ZH` ya consolidado: positivo acotado a `N-adapt`, null/negativo en `N-strict` |
| **Atencion Armonica** | Agrupamiento armónico polifónico | Incubacion arquitectonica para testear `Harmonic Pairformer`: pair-state, transitividad y `triangle` bajo evidencia per-par ambigua | **Fase 0, 0.5 y 0.6 cerradas** — `B` gana `OOD-poly` con clusterers globales deployables; queda subestimacion de `k` y Stage B |
| **Escalon 3** | Audio XY <-> Lissajous | Banco de pruebas sintetico con ground truth determinista para ratios visibles | **Activo** — baseline dual consolidado; primera linea geometrica ya corrida (`P5-cqtshift` mejor brazo OOD actual) |
| **Escalon 4** | ECG <-> PPG | Expansion a dominio fisiologico | **Proyeccion** |

En paralelo, el programa abrió una investigación transversal sobre el déficit de
ground truth para una PPU/Natural Harmonic Geometry. Cuarenta olas —ochenta y tres
investigaciones independientes y tres reconstrucciones del coordinador con
procedencia explícita— no encontraron una geometría universal de las
proporciones: organizaron una base estratificada de oráculos analíticos,
simulación generativa, cámaras físicas y evidencia empírica externa, atravesada
por falsación adversarial y adjudicación ciega. Las ampliaciones más recientes
incorporan identificabilidad/certificación, contratos de medición/adquisición y
una geometría efectiva entre escalas. Esta última separa ejecutar un
coarse-graining conocido de descubrir su kernel, exige convergencia y registra
`A9/P2k` como candidata no promovida. La psicofísica de magnitud añade un banco
externo `P5e`: separa representación, ruido y decisión para preguntar si una
regla de razón transfiere a componentes, rangos, generadores o modalidades
nuevos, sin tratar la conducta como física. La undécima ola agregó dos barreras
previas: `P2l` separa un cociente calculable de una razón empíricamente
significativa, y `P2m` exige que una variable macro preserve intervenciones y no
sólo predicción. Una muestra finita no certifica por sí sola un teorema de
representación; un mapa plantado tampoco es ground truth único si no es
identificable.
La duodécima ola agrega dos exigencias nuevas. `P2n` distingue
permutation-equivariance de coherencia proyectiva entre cardinalidades,
densidades o exposiciones; `P2o` distingue tropicalidad exacta, límite de
dequantización y ajuste piecewise-linear. `A11` queda como especialización
proyectiva de A9, todavía reducida a una constraint atribuible, y `A12` como
bloque max-plus proyectivo estrecho con controles funcionalmente equivalentes.
Ninguna fue promovida.
La decimotercera ola pregunta qué geometría subsiste cuando no hay una
coordinatización única o cuando el objeto es una función de información sobre
todos los subconjuntos. `A13/P2p` usa chirotopes, circuitos, cocircuitos, dualidad
y menores como estructura relacional exacta, pero separa validez combinatoria de
realizabilidad lineal. `A14` no designa una mega-arquitectura: es una interfaz
protocolaria que mantiene independientes `A14a/SLIB-DSF`, `A14b/SLIB-PMF` y
`A14c/SLIB-LINEAR`, porque vector entrópico, distribución conjunta y fuente
lineal no comparten output, privilegio ni estimando. `P2q` separa además el cono
Shannon, la región entrópica y el rango lineal; una PMF empírica conjunta no
puede funcionar como hard negative no entrópico.
La decimocuarta ola desplazó el foco desde reconocer una clase hacia ejecutar
una operación cerrada bajo información parcial. `A15/P2r` estudia si una tabla
parcial de coeficientes permite materializar un polinomio antes de aplicar una
contracción direccional; máscara, gauge, direcciones ID/OOD y métrica quedaron
congeladas para impedir que clase, cono o coeficientes ocultos entren como
oráculos. `A16/P2s` separa encoder, scorer aditivo y executor sobre una topología
assignment pública conocida. El único puente formal entre ambas ramas es más
estrecho que una geometría común: el soporte de un polinomio Lorentziano
homogéneo es M-convexo. No se infiere por ello una tropicalización de
polinomios Lorentzianos ni una autoridad física general.
La decimoquinta ola agregó dos bancos donde una relación local sólo adquiere
sentido por su compatibilidad con una estructura mayor. `A17/P2t` pregunta si
un bottleneck de log-ratios entre constraints incidentes ayuda a completar una
distancia faltante dentro de una arquitectura de cuñas fija, con aplicabilidad
identificable y abstención explícita. `A18/P2u` pregunta si una columna de
exchange puede inferirse desde probes locales y ejecutarse sobre una query no
observada, pero sólo después de demostrar por enumeración racional que el
contexto identifica una única órbita admisible. Ambos contratos quedaron
auditados y programáticos; todavía no fueron ejecutados ni promovidos.
La decimosexta ola examinó qué ocurre cuando la geometría vive en un cociente y
cuando invariantes locales deben satisfacer una condición global de existencia.
`P2v` quedó como suite diagnóstica de shape spaces: separa cocientes por `SO/O`,
quiralidad, tamaño, relabeling y degeneración, pero no atribuye una arquitectura
nueva por compilar Procrustes. El carril de circle patterns quedó bloqueado antes
de recibir `A19/P2w`: la auditoría mostró que su primer generador filtraba la
clase mediante una identidad local perfecta. La fuente matemática permanece
como ground truth y checker; el protocolo neuronal sólo podrá reabrirse después
de superar un gate explícito de feature-triviality.
La decimoséptima ola incorporó otra forma de geometría relacional: estados que
sólo son identificables módulo gauge a partir de cocientes de grupo. `P2a-G`
quedó auditado como extensión estrecha de P2a/A3 y separa en un factorial el
residual analítico exacto del mixer group-aware; las salidas pre-solver se
adjudican antes de cruzarlas con sincronizadores clásicos. El carril de
phase/closure permanece como preflight: cancelar gains instrumentales no basta
para identificar una imagen si el aparato conserva un nullspace. No se agregó
`A19`, no hubo ejecución y no se declaró GO/NO-GO.
La decimoctava ola hizo más exigente el significado de representar un cociente.
Una función invariante puede borrar la acción y, sin embargo, confundir órbitas;
una moving frame puede fijar una carta local sin ofrecer una gauge global. Del
mismo modo, un barcode persistente puede ser exacto y estable para una
filtración sin reconstruir el objeto ni autorizar físicamente la métrica que la
generó. `ORBIT-REPRESENTATION-AUDIT-v0` y
`FILTRATION-AUTHORITY-PREFLIGHT-v0` separan esas propiedades y extienden el
contrato P0 con jerarquía de unidades, acceso por brazo, estratos singulares,
cadena de operadores y readers comparables. No se agregó `A19`, no se asignó un
nuevo `P2*`, no hubo ejecución y no se declaró GO/NO-GO.
La decimonovena ola llevó ese cuidado a dos representaciones que podrían parecer
naturalmente aptas para una PPU. En análisis armónico, power, bispectrum y
scattering tienen propiedades distintas: una invariancia puede perder fase, una
referencia completa puede depender de rango y sampling, y ninguna expansión
autoriza por sí sola el grupo físico. En estados predictivos, equivalencia sobre
tests finitos, causal state teórico, rango de Hankel, realización probabilística y
suficiencia para control tampoco son intercambiables. Las suites
`HARMONIC-ORBIT-AUTHORITY-AUDIT-v0` y
`PREDICTIVE-STATE-AUTHORITY-AUDIT-v0` quedaron metodológicamente cerradas tras
revisión mayor y reauditoría independiente. Extienden P0 con inputs igualados,
validity masks, scoring de preimágenes, regímenes de solver separados y un
`process_law_contract` hasheado. No agregan `A19`, nuevo `P2*`, ejecución ni GO/NO-GO.
La vigésima ola trasladó ese mismo criterio desde las representaciones hacia los
operadores. Un espectro laplaciano, la diagonal de un heat kernel, el kernel
completo y el semigrupo no son objetos intercambiables; tampoco lo son predecir
observables, cerrar un subespacio de Koopman, identificar un operador, recuperar
un generador y autorizar control. Las suites de autoridad espacial y dinámica
quedaron cerradas como auditorías P0, no como una arquitectura nueva. Su síntesis
de diseño es un atlas todavía emergente: `objeto/equivalencia -> encoder
relacional -> propuesta de operador tipada -> solver/aplicación -> reader de
query`, acompañado por competencia, abstención y un ledger externo de autoridad.
No hay `A19`, nuevo `P2*`, ejecución ni promoción arquitectónica.
La vigesimoprimera ola invirtió el problema: partió de respuestas de frontera o
input-output y preguntó qué interior autorizan a recuperar. El resultado no es
siempre un estado puntual. Según el aparato, la clase y la query, puede ser una
clase módulo gauge, una región alcanzable, un conjunto compatible o `UNKNOWN`.
`Response-Quotient Atlas` conserva esa bifurcación en dos operaciones candidatas:
recuperar el objeto cociente autorizado por la respuesta y puntuar candidatos
públicos contra ella. El interior plantado, los witnesses y la autoridad física
permanecen fuera del forward. La candidata quedó aceptada documentalmente, no
promovida ni ejecutada; no recibe `A19` ni declara GO/NO-GO.
La vigesimosegunda ola incorporó al experimento como parte del objeto geométrico.
Un mismo cociente de indistinguibilidad no implica igual cantidad de información,
y una reconstrucción globalmente única puede conservar direcciones inestables o
invisibles. `Experiment-Relative Geometry Atlas` mantiene separados cociente
observacional, orden informacional/deficiencia y campo de visibilidad. Su salida
queda tipada como punto, clase, componente visible, conjunto identificado o
`UNKNOWN`; recovery autorizado y completion por prior no se agregan. La candidata
quedó aceptada documentalmente, no promovida ni ejecutada.
La vigesimotercera ola precisó el estatuto de las respuestas no puntuales. Cuando
una observación parcial identifica una fibra de medidas o parámetros compatibles,
el representante usado por el generador no puede funcionar como verdad única. La
candidata `Identified-Set Authority Stack` separa el conjunto identificado en la
población, los witnesses internos, las aproximaciones exteriores, la autoridad
numérica y la inferencia muestral. También separa el campo de compatibilidad que
podría aprender una red de los checkers y ledgers externos que autorizan cada claim.
Su cierre fue documental, después de auditoría y reauditoría independientes; no fue
implementada, promovida ni convertida en una decisión GO/NO-GO.
La vigesimocuarta ola preguntó qué ocurre cuando ese conjunto autorizado deja de
ser una salida estática y debe atravesar una transformación. Interpretación
abstracta, set-membership y reachability convergen en una condición precisa:
semántica concreta, dominio, transformer, checker, composición, pérdida y reader
no pueden compartir una autoridad implícita. `Guaranteed Set Transformer Stack`
queda como candidata no numerada y condicional. Su primer protocolo cambia sólo
el transformer dentro de un dominio fijo y cruza propuestas aprendida, clásica y
sham con checker y repair comunes. Una falsa exclusión bloquea el claim; `top`
recuerda que contener todo puede ser correcto y no informar nada. La auditoría
independiente abrió ocho findings y la reauditoría los cerró documentalmente. No
existe todavía checker implementado, baseline registry, ejecución, promoción ni
decisión GO/NO-GO.
La vigesimoquinta ola preguntó qué geometría puede atribuirse a los propios
conjuntos antes de transformarlos. Geometría convexa y convergencia de
hyperspaces mostraron que no existe una distancia set-valued autorizada por el
solo formato: dimensión ambiente y afín, regularidad, observación, política de
vacío e infinito, operación y reader determinan la jurisdicción. `Set Geometry
Authority Contract` queda como interfaz externa anterior al encoder. Su primer
producto separa una fase de schema/checker sin modelos de una comparación
posterior de representaciones, y obliga a que learned, classical y sham reciban
exactamente el mismo input. La reauditoría independiente cerró los findings
documentales, pero los once bancos finitos y el banco Hilbert simbólico siguen
bloqueados hasta congelar implementaciones, hashes, hardware y costos. No hubo
ejecución, promoción ni decisión GO/NO-GO.
La vigesimosexta ola desplazó la pregunta desde la geometría elegida hacia el
estatuto del objeto que puede legítimamente recibirla. Espacios métricos medidos,
cocientes y objetos estratificados exigen declarar equivalencia, observación,
query y autoridad antes del encoder. `Intrinsic Object Authority Contract` fija
esa interfaz y deja una primera diferencia material respecto de las olas
anteriores: cuatro bancos checker-only fueron implementados y ejecutados. GH
exacto frente a bottleneck biyectivo, la separación Rook/Shrikhande, el cociente
espejo y el cociente ortante reprodujeron resultados byte-identical dentro del
presupuesto congelado. Esa evidencia adjudica cuatro anclas finitas, no las
diecisiete entradas restantes ni una representación neuronal común.
La vigesimoséptima ola localizó proporciones matemáticamente precisas en conos
positivos y operadores SPD/PSD. `Positive/Operator Authority Contract` separa
rayos, escala radial, soporte, gauge, acción, reader y operación; impide tratar
`AB^{-1}` como un cociente escalar y mantiene eigensolvers, cálculo funcional,
solvers y checkers fuera del núcleo aprendido. De allí surgen cuatro hipótesis
estrechas —reader de log-espectro relativo, selector contextual, proposer de
eigenray y política de frontera—, todavía sin bancos ejecutados ni promoción.
La vigesimoctava ola extendió el mismo criterio a ultramétricas, dendrogramas,
tree metrics, splits y tight spans. `Hierarchy/Tree Authority Contract` tipa
veintisiete variantes completas y separa relación observada, estructura
compatible, estructura identificada y estructura autorizada. Una red puede
priorizar búsqueda, puntuar cuartetos o splits, elegir adquisiciones o estimar
un riesgo acotado; executor y checker conservan la partición final, los
certificados y la abstención. Sobre el atlas raw de treinta y tres bancos diseñado
por R57/R58, una
primera suite smoke materializó trece distinciones contractuales:
ultrametricidad y completación, cuartetos y estrellas, invariancias de raíz,
grado dos y arista cero, compatibilidad de splits, rango bajo missingness y
verdad racional frente a desempates float. Dos corridas fueron byte-identical,
con `13/13 PASS`, bajo un firewall explícito entre input público y witnesses
oracle. Una extensión exacta añadió después dos hard negatives que el primer
corte había dejado abiertos. El primero muestra que dos aritméticas p-ádicas
distintas pueden inducir la misma jerarquía ordinal finita; el segundo distingue
un tight span arbóreo de una celda bidimensional mediante la condición exacta
de cuatro puntos. Esa extensión volvió a ser byte-identical, con `2/2 PASS` y
cuatro mutaciones adversariales rechazadas. Una tercera suite materializó dos
fronteras que todavía permanecían sólo en el contrato. En una tripleta métrica,
la ultramétrica subdominante no coincide con la proyección óptima en norma
infinito: el óptimo tiene error `1/2`, admite una familia completa de soluciones
y obliga a separar dato observado, repair clásico y representante canónico. En
cuartetos tree-like, el defecto de cuatro puntos permanece en cero tanto para
casos resueltos como para una estrella, mientras el margen de resolución pasa
de `2` a `0` y puede conservarse aunque cambie la escala absoluta de las distancias. La suite
cerró `2/2 PASS`, byte-identical entre seeds y rechazó seis mutaciones. Una cuarta
suite materializó tres fronteras algebraicas —presentación finita de un building,
gauge de realización y preflight no métrico— y una quinta separó identificación
condicional de pesos, identificación global de topología y autoridad de máscaras
equipotentes bajo observación parcial. En total, las cinco suites ejecutaron `22`
suite-bank IDs. Ese número no forma un subconjunto literal de los treinta y tres
bancos raw: las suites comprimen y cruzan casos. La cobertura canónica del atlas
es `9 fully adjudicated / 20 partially covered / 4 not adjudicated`; no hubo modelos.
Las Olas 29–32 reforzaron ese dispatch con guards de convención, un plan
checker-only para Neighbor Joining, fibras algebraico-estadísticas y una
geometría estratificada de redes de reacción. La Ola 33 sometió el corpus a una
búsqueda adversarial de vacíos y preservó cuatro alternativas porque sus dos
carriles habían examinado propiedades distintas. La Ola 34 completó ese cruce.
Termodinámica de trayectorias y gauge curvo conservaron autoridad de dominio,
pero no una primitive aprendible nueva en las consultas auditadas: con
información completa domina el cálculo exacto y, con observación parcial,
aparecen estimación genérica, prior art o no identificación. Reducción reticular
y `boxplus` conservaron valor como policy condicionada o banco ingenieril, pero
especializan autoridades ya cubiertas. Ninguna de las cuatro vías reúne todavía
novedad de jurisdicción y novedad de primitive. Un contrato puede quedar bien
especificado aunque no satisfaga las condiciones de admisión, y la corroboración
se adjudica por claim, no por candidata en bloque. No se seleccionó dominio, no
se abrió suite, no se registró `A19` ni se promovió una arquitectura.
La Ola 35 examinó entonces una pregunta más exigente: qué tendría que conservarse
para que una operación aprendida pudiera transferirse entre autoridades científicas
distintas sin apropiarse de la autoridad del dominio de destino. La evidencia
revisada no produjo una primitive transferible completa. Mostró dos mitades todavía
separadas: sistemas que operan frente a autoridades diversas sin demostrar identidad,
atribución causal y replay cross-authority de un núcleo congelado, y sistemas con identidad paramétrica fuerte
cuya autoridad permanece estrecha o compartida. El resultado preserva como patrón
recuperable un loop de proposer tipado, evaluator externo, checker y abstención,
pero no lo convierte en `A19`, suite ni arquitectura
promovida. También obliga a separar residuo computacional de novedad: que quede algo
por aprender después del método exacto no prueba que haya aparecido una operación
proporcional nueva. La adquisición que modifica la evidencia permanece como la
hipótesis contractual de A8 y no debe probarse junto con el loop inicial de
proposer–evaluator.
La Ola 36 aisló esa hipótesis. No encontró una policy aprendida que, en un mismo
caso, cruzara autoridades materiales con núcleo frozen, produjera observación
externa nueva y superara atribución causal, costos igualados y replay completo.
Sí encontró las tres capacidades por separado: CAD²RL transfiere control
perceptivo de rendering a cámara física; A-Lab y CPBE cierran lazos de
acción–medición–update; DAD y trabajos afines amortizan o generalizan selección.
El resultado no las ensambla retrospectivamente. `EVIDENCE-CHANGING-ACTION`
permanece como arquitectura candidata recuperable y `UNRESOLVED`, mientras el
aporte durable es un contrato que separa score, acción, indicación, medición,
update y claim. La autoridad de instrumento, calibración, modelo de medición y
reader permanece externa; soporte realizado y propensiones se registran para
diagnosticar positividad y bloquear claims off-policy cuando no están sostenidos.
Un test futuro también debe estresar misspecification, medir calibración y cobertura
del claim y comprobar que los adapters no absorban la decisión del núcleo congelado.
La Ola 37 avanzó un nivel más atrás en la cadena: preguntó si un sistema puede
inducir la relación de equivalencia o la acción que después organiza esas propuestas
y mediciones. El cierre separó dos objetos que no deben heredar crédito entre sí.
El positivo teórico exige el régimen completo: transiciones indexadas por acciones
observadas; observación inyectiva hasta equivalencia de interacción; acciones
disponibles puras respecto de un único factor; condición de composición acotada;
mundo finito con todas las transiciones; y mínimo global. Sólo bajo esas condiciones
la partición de las acciones disponibles por factor es identificable como Track E,
hasta permutación de etiquetas de factor, equivalencia de interacción e isomorfismo
latente. Eso no identifica una acción completa no enumerada, sus flechas ni su
composición, que permanecen Track A `UNRESOLVED`. En quince records empíricos atómicos no apareció un caso que cerrara
todos los requisitos aplicables, ni action-family OOD ni authority holdout genuinos.
La consecuencia recuperable es un inductor tipado de relaciones y acciones que
conserve scope, gauge, abstención y checkers algebraico, relacional y fenomenológico.
Su escalera `E0 → A0 → A1 → T` es todavía un programa conceptual `NO-A19`: no abre
suite, modelo ni decisión de implementación.
La Ola 38 examinó la composición de claims entre autoridades y encontró un límite
anterior a cualquier mecanismo neuronal. Los formalismos relevados cubren
subproblemas reales de gluing, conflicto, provenance, dependencia, retractación,
incertidumbre, pooling y transporte, pero sus `24` casos fuente no materializan
sin inferencia las claves canónicas `9/7/10` del contrato predeclarado. El cierre
auditado conserva `264` estados MC, retira los conteos preliminares y deja `0`
celdas canónicas y `0` clases de cierre. No es evidencia de novedad ni de una
arquitectura faltante: es evidencia de que primero debe existir una unidad
contractual válida antes de contar reducciones, gaps o residuos aprendibles.
La Ola 39 retrocedió todavía más: antes de componer claims hay que fundar sus
términos. Sus `19` casos fuente separan cantidad, unidad, calibración,
correspondencia, target, equivalencia e incertidumbre, y cierran con `0` celdas
canónicas y `0` adjudicaciones `TG`. La salida no es una arquitectura elegida,
sino un preflight que impide tratar una detección como entidad, un matching como
identidad o una ratio calculable como relación autorizada.
La Ola 40 incorpora la dimensión temporal de ese mismo problema. Revisión de
creencias, crítica de modelos, provenance/versionado y adaptación ofrecen piezas
reales para corregir un claim bajo evidencia nueva, pero ninguna fuente ni stack
documentado reúne bajo una misma jurisdicción contrato before, falsación
autorizada, decisión pre-outcome, estado after, diff localizado y replay de los
claims retenidos. El inventario auditado conserva `47` fuentes, `37` casos
tipados y `12` relaciones; los tres episodios materiales tienen `C2M=NO`, los
seis enlaces de stack reciben crédito nulo y los cuatro ledgers decisivos quedan
vacíos. La alternativa recuperable es un ledger versionado de revisión
contractual, todavía `NO-ARCHITECTURE / NOT-EXECUTED`: una futura PPU tendría
que aprender relaciones y, además, declarar cómo cambia su contrato sin perder
identidad, autoridad ni trazabilidad.
El resultado acumulado es un dispatch de geometrías y
autoridades, no una mega-arquitectura universal.
Esto permite distinguir qué estructura puede recuperarse, qué claim puede
certificarse, qué autoridad tiene una medida y por qué una observación fue elegida.
Las candidatas `A7–A9/A12–A13/A15–A18`, la interfaz `A14*`, los tracks `P2g–P2v/P2a-G/P5e` y las suites diagnósticas de Olas 18–28 separan proposer, solver, checker,
calibración, incertidumbre, covarianza, política y abstención. Siguen sin ejecución
ni promoción: A8 conserva un contrato externo GUM/OED/metrológico auditado, pero
su policy transferible sigue sin demostración y exige un experimento posterior separado; A9 debe
probar ejecución y discovery en tracks distintos, con oráculos fuera del estimando.
La `A10` del catálogo PPU queda apenas como especialización causal experimental de A7/A9: sólo un
bloque de conmutatividad que supere su ablación exacta podría volverla candidata
arquitectónica independiente.
Esta base orienta
prototipos futuros
sin declarar GO/NO-GO. La síntesis está en
[Geometría proporcional y bases de verdad](Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/GEOMETRIA_PROPORCIONAL_BASES_DE_VERDAD.md).

Cada frente cumple un papel distinto en la exploracion de HIT. Escalon 1 fija la evidencia de referencia y la mecanica de inyeccion, y hoy ya puede leerse con su resultado flagship cerrado en training multi-seed real. Gate 8 pregunta donde se preserva la informacion armonica en las proyecciones. Gate 6 pregunta si la ventaja sobrevive fuera del retrieval y, por ahora, ya dejó una lectura más dura: la rama `Transkun+A4` no mostró ganancia útil ni en régimen base ni bajo degradación, mientras `Exp C` conserva abierta la pregunta solo desde el decoder serio sobre features congeladas. Escalon 2 es donde la hipotesis central — la armonia natural como organizador informacional privilegiado — se enfrenta directamente con descriptores derivados de la fisica del oscilador, contra controles espectrales y perceptuales. Ese primer contraste mecanistico ya quedó cerrado; `P3` ya fue corrido en una primera pasada, y la tarea que sigue es decidir si la comparación `P2 vs P3` cambia la lectura representacional del frente o confirma que el null descriptorial ya es estable bajo ambos regímenes de encoder. Voz Expresiva cumple otra función: preguntar si ese patrón descriptor-guided sobrevive cuando el backbone pasa a ser un encoder vocal foundation y cuando la comparación deja de ser solo intra-idioma. Ese frente ya no está esperando una réplica, sino leyendo una disociación concreta: el descriptor transfiere de forma reproducible entre `EN` y `ZH` cuando existe anclaje per-speaker en test (`N-adapt`), pero no sostiene una ventaja robusta en el régimen speaker-independent estricto (`N-strict`). Atención Armónica abre todavía otra clase de problema: no reitera la pregunta descriptorial de Escalón 2, sino que ensaya una hipótesis arquitectónica más fuerte sobre cómo incorporar estructura armónica cuando la ambigüedad local ya no puede resolverse con evidencia per-par cerrada. `Fase 0` construyó un problema válido; `Fase 0.5` mostró que el cuello no era `τ` sino `connected-components`; y `Fase 0.6` ya agregó algo más preciso: con clusterers globales deployables, `B` recupera una ventaja real sobre `B-local` en `OOD-poly`, aunque siga quedando lejos de una partición plenamente resuelta por la subestimación de `k`. Gate 9 / `A10` releen retrospectivamente parte de esa deuda dentro de musica, mientras Gate 10 ya dejó de ser un barrido parcial y pasó a ser evidencia cerrada de otra cosa: en esa rama retrospectiva el mecanismo pesa más que el descriptor, con `concat` arriba, `FiLM/pca` en segundo plano y `attn_bias` claramente abajo. Escalon 3, por su parte, ya no vive en `E3-P0`: hoy tiene un baseline dual consolidado, un régimen de probes ya auditado y una primera linea geométrica completa donde `P5-cqtshift` queda como mejor brazo OOD y `P6` no se vuelve el ganador del frente. Escalon 4 conserva la expansion fisiologica fuera de acústica.

---

## Resultados de referencia

### Escalon 1 — Audio <-> MIDI

Referencia canonica sobre MAESTRO. La mejora opera como ventaja geometrica del espacio latente (+82% CKA), no como enriquecimiento de decodificabilidad local.

| Brazo | `S` (canonical reference) | Lectura |
|---|---:|---|
| `D0` | `75.2% +/- 2.3pp` | Baseline sin descriptor |
| `a4r` | `80.7% +/- 1.9pp` | Reverse cross-attention con descriptor audio |
| `d4-a4r` | `81.2% +/- 2.5pp` | Variante mixta |
| `d4a4` | `84.0% +/- 2.7pp` | Mejor referencia del frente. Cierre sobre 5 training seeds independientes |

Los cuatro brazos canónicos de Escalon 1 ya tienen lectura homogénea en training-seed: `D0=75.2% +/- 2.3pp`, `a4r=80.7% +/- 1.9pp`, `d4-a4r=81.2% +/- 2.5pp` y `d4a4=84.0% +/- 2.7pp`.

### Gate 8 — Conditioned Projections

La informacion descriptorial es util incluso inyectada en la projection head (FiLM), no solo en el encoder.

| Brazo | Best `S` | Delta vs ctrl |
|---|---:|---:|
| `ctrl` (sin condicionamiento) | `79.2%` | — |
| `pcm` (MIDI cond) | `80.0%` | `+0.8pp` |
| `pcd-zero` (dual cond, zeros) | `81.8%` | `+2.6pp` |
| `pca` (audio cond) | `82.6%` | `+3.4pp` |
| `pcd` (dual cond A4+D4) | `84.2%` | `+5.0pp` |

`pcd > pca > pcd-zero > pcm > ctrl`: el cierre completo ya deja una lectura mas fuerte. La arquitectura conditioned aporta expresividad (`pcd-zero > ctrl`), el conditioning real aporta senal adicional (`pcd > pcd-zero`), y el lado audio responde mejor que el MIDI-side cuando se lo condiciona de forma aislada (`pca > pcm`).

### Escalon 2 — Speech <-> EGG

| Capa | Resultado | Significado |
|---|---:|---|
| Baseline lineal `CCA` | `S=64.4%` | La senal cross-modal existe antes del primer encoder neural |
| Baseline neural `D0` | `S=77.8%`, `CI=[72.0%, 80.8%]` | Piso solido para comparar descriptores |
| Concatenacion (`S2-P2-main`) | `V4-lin=67.8%`, `H-series=59.8%`, `A4-16k=77.8%` | La concatenacion trata descriptores como features — mecanismo inadecuado |
| Atencion (`S2-P2.5`) | Interpretado | `V4-lin-xattn=77.0%`, `H-series-attnbias=78.0%`, `A4-16k-attnbias=77.8%`, `A4-16k-xattn=78.0%`; ningun brazo mejora a `D0` de forma defendible |
| Proj. condicionada (`S2-P2.5b`) | Completa | `V4-lin-pca=74.6%`, `H-series-pca=77.4%`, `A4-16k-pca=77.2%`; ningun brazo superó a `D0` |
| Regimen foundation (`S2-P3`) | Primera pasada completa | `P3-D0=78.8%`, `P3-A4-16k-pca=78.2%`, `P3-V4-lin-pca=76.8%`, `P3-H-series-pca=75.6%`; siguiente tarea = `P2 vs P3` |

`S2-P2.5` testea la hipotesis central de HIT a nivel de mecanismo: la armonia natural debe guiar la atencion del modelo (organizar la computacion), no aumentar su contenido. `V4-lin` (dinamica del oscilador) entra como Familia A, `H-series` (estructura armonica intra-frame) como Familia B y probe mas directamente alineado con la tesis fuerte, y `A4-16k` queda como control no-ratio de Familia C. Esa fase ya fue leida con el preregistro [PREDICCIONES_EPISTEMOLOGICAS_P25.md](Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/PREDICCIONES_EPISTEMOLOGICAS_P25.md): la conclusion valida hoy es operativa, no grandilocuente. Los mecanismos `concat`, `attn_bias`, `xattn` y `pca` no dieron lift de retrieval sobre `D0` en Speech↔EGG y dejaron un primer null mecanistico formalmente cerrado. `S2-P3` ya cumplió su primera pasada con encoder frozen (`WavLM-Large`) y no desplazó a `P3-D0`; la tarea viva ahora es cerrar `P2 vs P3` con `CKA`, probes lineales y lectura representacional.

---

## Como entrar al repo

| Si queres... | Empezar por... |
|---|---|
| Entender que tipo de conocimiento produce Phideus | [MARCO_EPISTEMOLOGICO_PHIDEUS.md](MARCO_EPISTEMOLOGICO_PHIDEUS.md) |
| Ver el estado canonico del proyecto | [Proyecto_Estado_Actual.md](Documents/00_TRONCAL/Proyecto_Estado_Actual.md) |
| Ver el mapa visual de frentes y dependencias | [MAPA_VISUAL_DEL_PROGRAMA.md](Documents/05_WIKI/MAPA_VISUAL_DEL_PROGRAMA.md) |
| Dar contexto integral del programa a un agente | [LLM_CONTEXT.md](Documents/05_WIKI/LLM_CONTEXT.md) |
| Ver la estructura global de documentacion | [INDICE_DOCUMENTACION.md](Documents/00_TRONCAL/INDICE_DOCUMENTACION.md) |
| Entrar por la formulacion larga del programa | [AlterMundi/harmonic-information-theory](https://github.com/AlterMundi/harmonic-information-theory) |
| Leer la edición web pública del libro HIT | [hit.altermundi.net](https://hit.altermundi.net/) |
| Leer el paper de Escalon 1 | [arXiv:2604.10283](https://arxiv.org/abs/2604.10283) |
| Ir al frente musical consolidado | [ROADMAP_BIAS_CONTROL.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md) |
| Ir al frente vocal actual | [ESCALON_2/README.md](Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md) |
| Ver el frente de voz expresiva | [Voz_Expresiva_Phideus/README.md](Documents/01_FRENTES_ACTIVOS/Voz_Expresiva_Phideus/README.md) |
| Ver la incubación Atención Armónica | [Atencion_Armonica/README.md](Documents/01_FRENTES_ACTIVOS/Atencion_Armonica/README.md) |
| Ver el preregistro interpretativo de Escalón 2 | [PREDICCIONES_EPISTEMOLOGICAS_P25.md](Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/PREDICCIONES_EPISTEMOLOGICAS_P25.md) |
| Ver el nuevo frente Lissajous | [ESCALON_3/README.md](Documents/01_FRENTES_ACTIVOS/ESCALON_3/README.md) |
| Entender la historia de los descriptores | [CATALOGO_NARRATIVO_DESCRIPTORES_RATIOS_PHIDEUS.md](Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/CATALOGO_NARRATIVO_DESCRIPTORES_RATIOS_PHIDEUS.md) |
| Ver la historia larga del proyecto | [INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md](Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md) |
| Ver skills compartidas | [Documents/Skills/README.md](Documents/Skills/README.md) |

---

## Visualizaciones y documentacion viva

### Visualizaciones 3D interactivas

**[altermundi.github.io/Phideus](https://altermundi.github.io/Phideus/)**

Exploraciones de arquitecturas y lineas principales del programa: baseline cross-modal, reverse cross-attention, configuraciones duales de Escalon 1.

### Skills compartidas

**[Documents/Skills/README.md](Documents/Skills/README.md)**

Skills reutilizables concentradas en operacion HPC/SLURM, validacion pre-submit y lecciones operativas.

### Estructura de documentacion

- `Documents/00_TRONCAL/` — estado ejecutivo, indices, documentos troncales
- `Documents/01_FRENTES_ACTIVOS/` — documentacion operativa de cada frente vivo
- `Documents/04_TRANSVERSAL/` — teoria, fundamentos, historia
- `Documents/05_WIKI/` — síntesis viva de frentes, roadmaps, relaciones y alternativas para humanos y agentes

---

## Infraestructura computacional

Parte del programa utiliza recursos de **UNC Supercomputo (CCAD)** de la **Universidad Nacional de Cordoba**, integrados al **Sistema Nacional de Computacion de Alto Desempeno (SNCAD)** de la Republica Argentina.

Para publicaciones derivadas de corridas en esa infraestructura, el proyecto adopta la formulacion institucional recomendada:

**[supercomputo.unc.edu.ar/equipamiento/citar-recursos](https://supercomputo.unc.edu.ar/equipamiento/citar-recursos/)**

---

## Reproduccion minima

### Setup del entorno

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Pipeline base de BIAS_CONTROL

```bash
python experiments/bias_control/run_all_gates.py \
  --maestro-dir data/maestro_v3/maestro-v3.0.0 \
  --output data/bias_control_medium
```

### Ejemplo: Gate 4.3 `d4a4`

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python experiments/bias_control/gate42_training.py \
  --descriptor d4a4 \
  --checkpoint data/bias_control_medium/training_outputs/foundation_locked_e25.pt \
  --output data/bias_control_medium/training_outputs/gate43/d4a4 \
  --maestro-dir data/maestro_v3/maestro-v3.0.0 \
  --epochs 5 --batch-size 16 --num-workers 8 \
  --freeze-policy run-d --seed 42 --device cuda
```

### Evaluacion estructurada

```bash
python experiments/bias_control/evaluate_structured_pool.py \
  --model <checkpoint.pt> \
  --output <output.json> \
  --pool-size 256 --n-queries 500 --seed 42 \
  --maestro-dir data/maestro_v3/maestro-v3.0.0
```

Protocolo canonico: `pool=256`, `queries=500`, `seed=42`.

---

<!-- BELOW THE FOLD -->

<details>
<summary><strong>Roadmap del programa</strong></summary>

### TripleScaloneta

| Escalon | Dominio | Rol | Estado |
|---|---|---|---|
| Escalon 1 | MAESTRO Audio <-> MIDI | Validacion descriptor-guided y cierre cientifico del primer banco de pruebas | **Cerrado** |
| Escalon 2 | Speech <-> EGG | Test directo de HIT: armonia natural del oscilador como organizador cross-modal | **Activo (null mecanistico inicial cerrado; `S2-P3` primera pasada completa)** |
| Escalon 3 | Audio XY <-> Lissajous | Banco sintetico con ratio visible y control total de parametros | **Activo** (`P2/P4/P5/P6` ya corridos en primera pasada) |
| Escalon 4 | ECG <-> PPG | Expansion fisiologica | **Proyeccion** |

### Frentes activos

| Frente | Funcion | Documento |
|---|---|---|
| Gate 6 AMT | Validacion downstream | [12_GATE_6_AMT/README.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/12_GATE_6_AMT/README.md) |
| Gate 8 | Conditioned projections | [15_GATE_8_CONDITIONED_PROJECTIONS/README.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/15_GATE_8_CONDITIONED_PROJECTIONS/README.md) |
| Gate 10 | Mechanism sweep audio-only | [17_GATE_10_MECHANISM_SWEEP/README.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/17_GATE_10_MECHANISM_SWEEP/README.md) |
| Escalon 2 | Frente principal (null mecanistico inicial cerrado; sigue diagnostico `P2 vs P3`) | [ESCALON_2/README.md](Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md) |
| Escalon 3 | Banco Lissajous con baseline dual y primera linea geometrica ya consolidada | [ESCALON_3/README.md](Documents/01_FRENTES_ACTIVOS/ESCALON_3/README.md) |

### Roadmaps canonicos

- [ROADMAP_BIAS_CONTROL.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md)
- [ROADMAP_ESCALON_2.md](Documents/01_FRENTES_ACTIVOS/ESCALON_2/ROADMAP_ESCALON_2.md)
- [ROADMAP_ESCALON_3.md](Documents/01_FRENTES_ACTIVOS/ESCALON_3/ROADMAP_ESCALON_3.md)

</details>

<details>
<summary><strong>Arquitectura y familias descriptoriales</strong></summary>

### Arquitectura general

Phideus trabaja con configuraciones cross-modales contrastivas donde la armonia natural se inyecta como senal organizadora:

```text
modalidad A -> encoder -> projection -> embedding
                  ^
            armonia natural
                  v
modalidad B -> encoder -> projection -> embedding
                    \      VICReg      /
```

La investigacion no se limita a que encoder usar. La pregunta central es como entra la armonia natural (como augmentation, atencion o modulacion), que geometria induce, y si esa geometria es especifica de relaciones armonicas o aparece con cualquier descriptor auxiliar.

### Escalon 1: familias

| Familia | Ejemplos | Rol |
|---|---|---|
| Control | `D0` | Baseline sin descriptor |
| MIDI local | `D4` | Relaciones locales del lado MIDI |
| Audio espectral | `A4`, `A4r` | Dinamica espectral del lado audio |
| Dual | `d4a4`, `d4-a4r` | Combinaciones de mayor rendimiento |

### Escalon 2: taxonomia armonica

| Familia | Descriptor | Rol en la exploracion de HIT |
|---|---|---|
| **Armonia natural temporal** | `V4-lin` | Dinamica lineal del oscilador — testea si ratios naturales de F0 organizan atencion inter-frame |
| **Armonia natural intra-frame** | `H-series` | Estructura armonica (H2/H1..H6/H1) — testea si la serie armonica fisica organiza features |
| Control perceptual | `V4-log` | Misma info que V4-lin en escala logaritmica — testea si la escala importa |
| Control espectral | `A4-16k` | Dinamica espectral generica no-ratio — testea si cualquier descriptor auxiliar ayuda |

Ver: [MARCO_EPISTEMOLOGICO_PHIDEUS.md](MARCO_EPISTEMOLOGICO_PHIDEUS.md) y [plan_rectificacion_armonia_natural.md](Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/plan_rectificacion_armonia_natural.md)

</details>

<details>
<summary><strong>Linea experimental consolidada</strong></summary>

### Escalon 1

| Brazo | `S` (canonical reference) |
|---|---:|
| `D0` | `75.2% +/- 2.3pp` |
| `a4r` | `80.7% +/- 1.9pp` |
| `d4-a4r` | `81.2% +/- 2.5pp` |
| `d4a4` | `84.0% +/- 2.7pp` *(5 training seeds independientes)* |

### Gate 8

| Brazo | Best `S` | Delta vs ctrl |
|---|---:|---:|
| `ctrl` | `79.2%` | — |
| `pcm` | `80.0%` | `+0.8pp` |
| `pcd-zero` | `81.8%` | `+2.6pp` |
| `pcd` | `84.2%` | `+5.0pp` |
| `pca` | `82.6%` | `+3.4pp` |

### Gate 6 / Gate 7.1

| Frente | Corte |
|---|---|
| Gate 6 AMT | `Exp C` local cerro (`F1=0.157`); `Exp A` y `Exp B` ya cerraron negativamente en la rama `Transkun+A4` |
| Gate 7.1a | `D0_mert330m_frozen=75.0%`, sin mejora sobre `D0_lite=75.2%` |

### Escalon 2

| Capa | Resultado |
|---|---:|
| CCA baseline | `S=64.4%` |
| D0 neural | `S=77.8%` |
| Concatenacion | `V4-lin=67.8%`, `H-series=59.8%`, `A4-16k=77.8%` |
| Atencion (`S2-P2.5`) | Interpretado bajo preregistro |

</details>

<details>
<summary><strong>Documentacion clave y estructura del repo</strong></summary>

### Documentos principales

| Documento | Funcion |
|---|---|
| [Proyecto_Estado_Actual.md](Documents/00_TRONCAL/Proyecto_Estado_Actual.md) | Estado ejecutivo |
| [INDICE_DOCUMENTACION.md](Documents/00_TRONCAL/INDICE_DOCUMENTACION.md) | Mapa global |
| [ROADMAP_BIAS_CONTROL.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md) | Roadmap musical |
| [ESCALON_2/README.md](Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md) | Frente vocal |
| [Documents/Skills/README.md](Documents/Skills/README.md) | Skills compartidas |

### Estructura

```text
Phideus/
├── src/                         # Modulos del proyecto
├── experiments/                 # Training, evaluacion y utilidades experimentales
├── Documents/
│   ├── 00_TRONCAL/              # Estado ejecutivo, indices, documentos troncales
│   ├── 01_FRENTES_ACTIVOS/      # Frentes vivos
│   ├── 02_FRENTES_PAUSADOS/     # Frentes pausados
│   ├── 03_FRENTES_CERRADOS/     # Frentes cerrados
│   └── 04_TRANSVERSAL/          # Teoria, fundamentos, historia
├── viz/                         # Visualizaciones interactivas
├── data/                        # Datasets y outputs (no versionados)
└── config/                      # Configuraciones
```

</details>

---

> *"El bosque ya canta. Nuestra tarea es entender su afinacion."*

**Licencia**: MIT — ver [LICENSE.md](LICENSE.md)
