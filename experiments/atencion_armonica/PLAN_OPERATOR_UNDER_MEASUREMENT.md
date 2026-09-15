# Operador geométrico bajo medición

Estado: revisión independiente de diseño incorporada; cores y adaptador observable
implementados con fixtures CPU; ejecutor y tests prospectivos aún cerrados.
El [estado de implementación](STATUS_OPERATOR_UNDER_MEASUREMENT.md) separa las
verificaciones mecánicas de la campaña pendiente. El protocolo define el contraste;
el perfil y la auditoría de implementación preceden a su congelación ejecutable.

## Pregunta y alcance

La [campaña anterior](RESULTS_GEOMETRIC_DECISION_ENERGY.md) favorece la ruta
geométrica frente a Inyección en deformación, pero no frente al clásico inicial.
Ahora se contrasta si esa utilidad sobre una lista de frecuencias sobrevive
a obtener la lista desde audio. No se agrega otra cabeza ni se reentrena.

La geometría sigue siendo el ajuste externo y la partición relacional.
La nueva intervención cambia observación, no simultáneamente operador y loss.
Conservar los lectores entrenados con Decisión permite evaluar su transferencia,
no afirmar que esa loss es óptima para las nuevas métricas de extremo a extremo.

El dato emitido sigue siendo sintético. El objetivo es localizar pérdida
observacional, falta de candidatos y error de elección; no identificar física
real ni validar HIT por construcción. No se requiere ganador.

## Diseño

El [protocolo](PROTOCOL_OPERATOR_UNDER_MEASUREMENT.md) especifica512escenas
nuevas, cuatro escenarios de128, cada una con interfaz canónica y tres
mediciones de audio pareadas. Se mantienen los tres backbones congelados y
las tres semillas de los cuatro lectores Decisión, más referencias clásicas.
Cambian duración o ruido por separado. Desarrollo32 y calibración64 se
mantienen fuera del test; sólo se seleccionan umbrales del detector.

Las funciones actuales de geometría, normalización, candidatos y lectores
se reutilizan sin editar sus fuentes congeladas. Su dominio es explícito:
8..32eventos y particiones de2..4grupos con4..8miembros. No se reparan mediante
truth los eventos perdidos para entrar en ese dominio. Las abstenciones y
universos vacíos son resultados de la interfaz y permanecen en el análisis.

## Entregables y secuencia finita

1. Revisar diseño y protocolo mediante instancia independiente. Resolver
   hallazgos que alteren validez, evidencia o coste; no bucles por cosmética.
2. Implementar módulos nuevos de render/detección, correspondencia/score y
   adaptador de campaña. Reutilizar kernels puros, no el supervisor histórico
   cuyo roster y contrato de144estados pertenecen a otra pregunta.
3. Pruebas mecánicas: lectura completa de algoritmo, casos de cardinalidad y
   ambigüedad, no-leakage, correspondencia de orden/rangos, descomposición de
   error, replay y fallos de publicación. Prueba independiente de los cálculos.
4. Perfil acotado en desarrollo; decidir CPU/GPU y confirmar presupuesto
   completo, incluyendo replay/auditoría. Seleccionar detector en calibración.
   No optimizar lector, fitter o métrica con esa selección.
5. Publicar protocolo/config/código/roster/exclusiones y hashes de referencias.
   Congelar antes de generar tests; producir todas las observaciones e
   inferencias sin consumir etiquetas de evaluación, sellarlas y luego evaluar.
6. Ejecutar replay, informe, auditorías técnica y de horizonte. Publicar
   resultados y mantener candidatas; responder qué siguiente cambio tiene
   poder discriminante. Cierre sólo tras campaña íntegra y documentación.

## Presupuesto y recuperación

Sin entrenamiento. Perfil CPU máximo600s para render/detector/matching y
evaluación; perfil CUDA máximo600s para forward, fitter y36lectores, medidos
por separado. El fitter histórico ya usa CUDA float64: se conserva ese backend.
Usar las16observaciones de desarrollo fijadas en el protocolo. Antes de CUDA,
informar objetivo/VRAM/duración y verificar ownership. No son costes medidos aún.

Techo inicial de campaña+replay+auditoría:28800segundos de trabajo acumulado,
21600segundos en etapas que ocupanGPU y16GiB de artefactos nuevos;
VRAM6GiB y RAM8GiB por proceso. Reservar
al menos25% del tiempo para replay/auditoría. Antes de abrir tests, proyección
desde el perfil de todo el roster más margen×1.5 debe entrar en ese saldo;
si no, revisar explícitamente el diseño/presupuesto antes de congelar, nunca
reducir roster después de ver respuestas. No son umbrales de GO/NO-GO.

Manifest append-only por etapa/escena y estado terminal recuperable; no nueva
generación/forward/fit cuando el output autenticado ya existe. Fallos de recurso
no se convierten en errores científicos puntuados como cero. Preservar parciales
y reparar causa, sin reinicio silencioso de costes ni mezclas de configuraciones.

## Criterio de cierre y bifurcaciones

El informe debe mostrar valores por condición/escenario y límites de atribución,
sin convertir nueve celdas condicionadas en nueve experimentos independientes.
La descomposición es contable respecto del score elegido, no causalidad física.

Si domina la pérdida de correspondencia, revisar observación antes de modelar
otra corrección. Si domina el universo, investigar operador con soporte incompleto
o likelihood de medición. Si el universo permite buenas decisiones pero no se
eligen, examinar aprendizaje/operación. El ajuste amortizado permanece alternativa
por coste/robustez, no siguiente paso automático. GO/NO-GO y promoción pertenecen
al usuario. El siguiente goal debe justificarse por resultados y cerrar éste
sin extenderlo indefinidamente para conseguir una victoria.
