# Operación geométrica y objetivo: estado de implementación

2026-09-14. Goal activo e incompleto. El
[protocolo](PROTOCOL_OPERATOR_OBJECTIVE_ALIGNMENT.md) pasó auditoría
independiente después de precisar la autoridad de inputs, el orden canónico,
el canal realmente entregado, las reducciones numéricas y la agregación.
Su SHA256 auditado es
`bbdc52f010523059f95d87664670f82dd1579a7e4d43492b44393907fc50dc0b`.
El encabezado conserva el estado previo de redacción; esta página registra
la auditoría posterior sin alterar los bytes de ese contrato.

El [núcleo NumPy](../../src/atencion_armonica/operator_objective_core.py)
ya implementa entropías/targets, ARI, órdenes y empates, Kendall tau-b,
errores por componente y de suma float32, oracles separados, estratos y
diferencias pareadas por celda antes de agregar por escena. No abre archivos,
entrena, ajusta modelos ni inicia CUDA.

Las [18 pruebas del núcleo](test_operator_objective_core.py) pasaron en CPU.
Incluyen comparación con el contrato de entropías heredado y con la referencia
local de tau-b, constantes y vacíos, discrepancias de oracles, pérdida de
precisión de suma float32 y soportes disjuntos. La primera ejecución detectó
rechazo de índices vacíos, aceptación de strings como métricas y una diferencia
de redondeo en la fórmula de ARI. Se corrigieron: índices vacíos explícitos,
tipos numéricos estrictos y fórmula de ARI con conteos enteros de pares.
La ejecución posterior terminó con salida 0. Estas pruebas no certifican
componentes todavía no construidos ni producen evidencia de la campaña.

Ya se añadieron los [puertos de lectura y extracción](../../src/atencion_armonica/operator_objective_sources.py),
el [ensamblado por escena](../../src/atencion_armonica/operator_objective_scene.py)
y la [agregación por escenario](../../src/atencion_armonica/operator_objective_aggregate.py).
Las 44 pruebas conjuntas pasan en CPU: 18 del núcleo, 16 de fuentes/canales,
6 de escena y 4 de agregación. Son fixtures matemáticos y archivos temporales
pequeños, no ejecuciones sobre escenas de la campaña.

La lectura verifica referencias, rutas sin symlinks, tamaños, hashes y
compresión; las cabeceras de arrays se comprueban antes de permitir su
asignación de memoria. La extracción conserva las seis cotas, sus ramas y
el canal float32; verifica los donantes guardados sin sortear un nuevo sham.
El ensamblado calcula las cuatro referencias y las 27 celdas, y la agregación
mantiene unidad escena, soportes por métrica y slices pool/neighbor/absent.

El [adaptador del corpus cerrado](../../src/atencion_armonica/operator_objective_corpus.py)
ya liga cierre, freeze, sellos, predicciones y normalizador TRAIN; implementa
inventario, extracción y cotejo de targets/decisiones. La autenticación de
headers reales de los cuatro tests verificó 45 predicciones y 27 originales
por test, sin extraer factores ni abrir sidecars. Una comprobación inicial
detectó que orden de freeze y orden de inferencia no coinciden; la conexión
usa ahora la permutación explícita `freeze_index`, sin modificar fuentes.

La auditoría independiente de núcleo, fuentes, escena, agregación y corpus
cerró sin defectos materiales abiertos en ese corte: 55 pruebas pasadas,
headers de cuatro tests autenticados y 2958 comparaciones deterministas de
ARI contra la referencia local. No revisó ni ejecutó el inventario completo,
la extracción real, el perfil ni los componentes operativos añadidos después.

También existen [bundles/publicación](../../src/atencion_armonica/operator_objective_artifacts.py)
y [presupuesto acumulado](../../src/atencion_armonica/operator_objective_budget.py),
con 11 pruebas propias pasadas. La suite conjunta tiene 66 pruebas pasadas.
Estos dos módulos todavía requieren auditoría e integración en un ejecutor:
lock de proceso, etapas, pausas, prefijos y replay completo no están cerrados.
La proyección conserva la reserva de auditoría y el margen ×2; fallos e
intentos sin cierre no reciben devolución silenciosa de tiempo.

Falta completar ese ejecutor y auditar su frontera operativa. Después deben
ejecutarse inventario de 2048 recibos, cuatro perfiles fijados, recibo de
presupuesto, barrido y replay, seguidos de auditorías de evidencia y alineación.
No se ejecutó ninguna de esas etapas ni se atribuye al PASS parcial autoridad
para saltarlas.

El rebase de recibos observa unos 893 MB comprimidos y 9.28 GB decodificados
de factores; no es una medición de costo ni una validación de sus blobs.
El protocolo fija un operador CPU, un hilo, 6 GiB RSS y 30 minutos acumulados
de operadores, con reserva de auditoría y guardas de almacenamiento. No se
iniciará el roster si el perfil no justifica ese presupuesto.

El diagnóstico sigue siendo retrospectivo. No cambia el pool, los targets,
la época seleccionada ni las redes; tampoco convierte mínimo UB en toda la
información recibida por la cabeza. La GPU no es necesaria para este hito.
No hay resultado nuevo de alineación, promoción arquitectónica ni GO/NO-GO.
