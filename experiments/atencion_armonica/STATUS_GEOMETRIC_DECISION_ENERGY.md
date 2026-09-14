# Energía geométrica para la decisión — estado de implementación

2026-09-14. Preparación OPEN y perfiles CPU/CUDA completos; entrenamiento científico no iniciado.

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
a adaptar todo el corpus ni a congelar la nueva escala. La ejecución posterior
del [operador de preparación](prepare_geometric_decision_open.py), ya auditado,
completó las 27 entradas sobre las 4096 escenas TRAIN y 512 de calibración en
50,708750 s. El cierre local autenticado tiene SHA256
`70f5996408403ade1e83f67fd84441cd983001dd9ba789ff3f3ab29047adcb6f`.
La escala TRAIN es `7,5314913344363035`, con 4036 escenas elegibles y 60 sin
candidatos; estas últimas permanecen en el roster sin recibir peso artificial.
El recibo de escala tiene SHA256
`25da9bd505efb0b594e88755f0e20c0018919646a4149e8d1b6faae19e26bd8e`.
La ejecución no abrió targets ni repitió ajustes o forwards. El consumo OPEN
acumulado, incluido el smoke anterior, es 54,502751 s de su tope de 1800 s.

El [runner de una celda](../../src/atencion_armonica/geometric_decision_cell.py),
su kernel y su store ya fueron auditados. Conservan estados de modelo,
optimizador y generadores aleatorios, checkpoints recuperables y salidas
firmadas float64 de calibración al inicio y cada cinco épocas. Las
[pruebas de recuperación](test_geometric_decision_cell.py) completan 50 épocas
con fixtures de soporte elegible pequeño y comprueban igualdad exacta frente
a una ejecución interrumpida. No son entrenamientos científicos. Los
[perfiles CPU/CUDA](RESULTS_GEOMETRIC_DECISION_PROFILE.md) ya completaron la
prueba positiva de recuperación exacta CUDA en cuatro casos mecánicos,
además de medir cabeza, fitter y carga completa. Su consumo acumulado es
71,594984 s de los 600 s de perfil; no se repitieron los ajustes de todo OPEN.

La [preparación durable](../../src/atencion_armonica/geometric_decision_corpus.py)
y el [selector de época](../../src/atencion_armonica/geometric_decision_selection.py)
están implementados y auditados con pruebas mecánicas de sus puertos.
El selector exige las nueve celdas por brazo y las diez épocas elegibles;
no elige una semilla o backbone ganador.

Permanecen pendientes completar el supervisor de las 72 celdas,
admisión de recursos, entrenamiento, selección calibrada real,
freeze prospectivo, tests nuevos, probes, replay y auditorías finales.
La GPU se usó sólo en los perfiles; no hay resultados científicos nuevos.

El protocolo conserva la corrección anterior al freeze de una seed IID
abierta durante una comprobación de diseño: queda retirada y excluida,
sin presentarla como test prospectivo. El reemplazo no se eligió por
resultados de modelos. No hubo entrenamiento ni comparación de arquitecturas
en ese incidente.

No se promueve una arquitectura. El operador geométrico sigue siendo externo
y usa una ley sintética conocida; el experimento deberá mostrar si conservar
esa operación en la decisión aporta algo más que entregarla como descriptor.
