# Fuentes rivales bajo observación de frecuencias

Estado: siguiente goal formulado desde el contraste completo del lector
aprendido, 2026-09-08. Diseño inicial; requiere protocolo acotado y auditoría
antes de ejecutar la búsqueda. No implementado ni promovido.

## Pregunta y motivo

¿En qué casos del banco actual aparecen particiones distintas que permiten
explicar de manera competitiva la misma observación `q32`, y cómo se relaciona
esa competencia con los errores de los lectores?

El [contraste aprendido](RESULTS_LEARNED_PARTITION_READER.md) dejó un primario
mixto y daño bajo familia deformada. El pool contiene particiones mejores
según etiquetas, pero eso no establece que puedan reconocerse desde los
inputs. Además, el witness actual ajusta `gamma=0` incluso cuando la fuente
genera `gamma≠0`. Antes de entrenar otra cabeza conviene separar competencia
entre explicaciones, cobertura de búsqueda y desajuste de familia.

La unidad geométrica es una partición del multiconjunto de log-frecuencias
centradas, módulo permutación de eventos y nombres de fuente. El centrado
elimina escala absoluta de la entrada, pero los límites generativos de f0
siguen restringiendo qué levantamientos de ese cociente son admisibles.
No se identifica fuente por un residual pequeño.

## Experimento finito

El diagnóstico reutiliza 96 escenas de los cuatro tests ya abiertos: 24 por
escenario, seleccionadas por una regla determinista sin usar el resultado
del buscador. El protocolo fijará IDs y semillas antes de ejecutar fits.
Se conservan las predicciones de los siete lectores y los pools existentes;
no hay reentrenamiento, nuevos forwards ni ajuste sobre esos tests.

Para cada escena se evalúan particiones del pool y una vecindad finita de
intercambios de miembros predeclarada. Se incluye la partición plantada como
referencia privilegiada separada, nunca como entrada del buscador observable.
El roster de candidatos, la vecindad y el número de inicializaciones se
fijan antes del análisis; no se expanden para rescatar un caso.

Se contrastan dos familias de ajuste comunes a todas las escenas: la familia
actual con gamma cero y una extensión acotada que admite deformación. No se
entrega el nombre verdadero del régimen a un supuesto lector desplegable.
Cada test de admisibilidad respeta cardinalidad, índices sin reemplazo,
rangos de parámetros y existencia de una escala global compatible; ajustar
grupos por separado no prueba que toda la escena sea realizable.

El protocolo debe especificar función de ajuste en coordenadas observadas,
tratamiento del ruido gaussiano centrado y cuantización float32, optimización,
presupuesto fijo y tolerancias numéricas justificadas por fixtures. Reporta
residuales y márgenes entre rivales, costo de búsqueda, cobertura y fallos de
optimización, sin inventar un umbral científico de éxito. Una mejora del fit
por ampliar familia no acredita por sí sola mejor identidad de fuente.

Se relacionan esos márgenes con errores de partición de los lectores,
separando escenarios y casos con/sin soporte generativo. Los resultados son
diagnósticos retrospectivos de esta muestra, no confirmación independiente
ni estimación automática de prevalencia en audio real.

## Qué puede y qué no puede concluir

Una colisión exacta de observaciones bajo latentes distintos, si se construye
y verifica, tiene autoridad distinta de dos fits cercanos dentro del ruido.
Con ruido gaussiano de soporte pleno, encontrar dos explicaciones posibles
de una muestra no prueba no-identificabilidad estadística de sus
distribuciones. No encontrar un rival significa sólo que no se encontró
bajo el buscador y presupuesto ejecutados; no demuestra unicidad global.

La extensión de familia, las alternativas de partición y los errores del
lector deben permanecer diferenciados. La evaluación no elegirá un ganador
por similitud visual del latente ni convertirá el oracle en un método usable.

## Acción, recursos y cierre

1. Rebasar ley, puerto observable, fitter y artefactos actuales. Recuperar
   del corpus sólo conceptos que cambien una decisión concreta; consultar
   fuentes originales de forma dirigida si hay una dependencia no resuelta.
2. Fijar protocolo completo y auditar el diseño con una instancia independiente.
   Implementar en archivos nuevos; probar fixtures y controles de permutación,
   escala, parámetros fuera de soporte y separaciones de autoridad.
3. Perfilar un corte mecánico antes de la búsqueda. CPU si es proporcionada;
   GPU local habilitada si la evaluación por lotes resulta materialmente más
   eficiente. Estimar recursos y respetar ownership, sin pedir ventanas por
   corrida. No enviar jobs remotos por rutina.
4. Ejecutar el roster entero con estados recuperables, rivales y witnesses,
   fallos explícitos, configs y replay. El objetivo no termina en otro
   preflight o en una colección de contratos sin diagnóstico realizado.
5. Auditar evidencia y alineación, documentar y publicar el cierre con
   commit/push. Elegir después el siguiente experimento, sin promoción
   unilateral ni decisión GO/NO-GO.

Si aparecen rivales competitivos bajo soporte válido, se compara el valor de
una observación adicional o una salida que conserve alternativas. Si el daño
se concentra en desajuste de familia, se diseña un contraste de representación
y operación robustas a ese cambio. Si la búsqueda no explica errores
persistentes, queda justificado contrastar operación/loss o fuentes latentes,
sin afirmar identificabilidad por ausencia de rivales. Resultados mixtos o
limitaciones de cobertura deben orientar una sola pregunta siguiente, no una
cola indefinida de infraestructura.

Tiempo, fase, medición/CQT, descriptores explícitos y fuentes latentes se
conservan como alternativas. La rama set-valued histórica continúa pausada;
considerar una salida ambigua aquí no la reactiva automáticamente.
