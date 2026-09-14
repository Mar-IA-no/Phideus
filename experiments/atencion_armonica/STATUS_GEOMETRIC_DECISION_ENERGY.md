# Energía geométrica para la decisión — estado de implementación

2026-09-14. Diseño concreto e implementación parcial; campaña no iniciada.

El [protocolo](PROTOCOL_GEOMETRIC_DECISION_ENERGY.md), auditado antes de la
campaña, fija cuatro rutas por dos pérdidas: Inyección, Geométrica,
Desacoplada y Local, cada una con MSE o surrogate de decisión. Son 72
entrenamientos previstos, no ejecutados. La comparación separa la ruta
geométrica en la decisión del objetivo de aprendizaje.

La [cabeza común](../../src/atencion_armonica/geometric_decision_model.py)
implementa 2258 parámetros, salida firmada y suma float64. Los tres brazos
no Local reciben los mismos ocho canales float32; el bypass usa exactamente
la columna entregada, sin precisión ni parámetros extra. La corrección
inicial nula reproduce el bypass por construcción: no es aprendizaje.

El [núcleo observable](../../src/atencion_armonica/geometric_decision_core.py)
implementa mínimo de ramas disponibles, escala RMS por escena, donantes,
selector de empates por firma y roundtrip de cuantización/recentrado. Las
[pruebas del núcleo](test_geometric_decision_core.py) incluyen coóptimos,
aritmética del target, gradientes, padding, donantes singleton, universo vacío
y transporte. Su auditoría independiente cerró sin findings abiertos en ese
alcance; no constituye
admisión del runner futuro ni evidencia de rendimiento experimental.

El [adaptador OPEN](../../src/atencion_armonica/geometric_decision_open.py)
lee los agregados ya autenticados sin modificar sus stores. Sus
[pruebas de frontera](test_geometric_decision_open.py) cubren recibos,
rutas, formatos y exclusión de roles test. La auditoría del adaptador también
comprobó el roster completo mediante fixtures, el peso igual por escena de la
escala y la conservación del diagnóstico desacoplado. Una lectura sobre
un shard TRAIN y los tres backbones pasó sin abrir targets. Esto no equivale
a adaptar todo el corpus ni a congelar la nueva escala.

El [runner de una celda](../../src/atencion_armonica/geometric_decision_cell.py),
su kernel y su store ya fueron auditados. Conservan estados de modelo,
optimizador y generadores aleatorios, checkpoints recuperables y salidas
firmadas float64 de calibración al inicio y cada cinco épocas. Las
[pruebas de recuperación](test_geometric_decision_cell.py) completan 50 épocas
con fixtures de soporte elegible pequeño y comprueban igualdad exacta frente
a una ejecución interrumpida. No son entrenamientos científicos. La prueba
positiva de recuperación CUDA sigue pendiente del perfil del backend elegido.

La [preparación durable](../../src/atencion_armonica/geometric_decision_corpus.py)
y el [selector de época](../../src/atencion_armonica/geometric_decision_selection.py)
están implementados y auditados con pruebas mecánicas de sus puertos.
El selector exige las nueve celdas por brazo y las diez épocas elegibles;
no elige una semilla o backbone ganador.

Permanecen pendientes ejecutar la adaptación completa, completar el supervisor,
perfil CPU/GPU, admisión de recursos, entrenamiento, selección calibrada real,
freeze prospectivo, tests nuevos, probes, replay y auditorías finales.
La GPU está disponible, pero no se usó en este corte de implementación.

El protocolo conserva la corrección anterior al freeze de una seed IID
abierta durante una comprobación de diseño: queda retirada y excluida,
sin presentarla como test prospectivo. El reemplazo no se eligió por
resultados de modelos. No hubo entrenamiento ni comparación de arquitecturas
en ese incidente.

No se promueve una arquitectura. El operador geométrico sigue siendo externo
y usa una ley sintética conocida; el experimento deberá mostrar si conservar
esa operación en la decisión aporta algo más que entregarla como descriptor.
