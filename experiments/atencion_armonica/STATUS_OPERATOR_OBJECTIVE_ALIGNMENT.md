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
y [presupuesto acumulado](../../src/atencion_armonica/operator_objective_budget.py).
El [ejecutor](../../src/atencion_armonica/operator_objective_runner.py) incorpora
lock de proceso, revisión ligada a código y runtime, etapas separadas,
pausas, prefijos autenticados y replay exacto desde los compactos. Sus
[pruebas](test_operator_objective_runner.py) pasan; la suite completa
alcanza 83 pruebas CPU pasadas. El ciclo se probó con fixtures, incluidas
interrupción, corrupción y negativa a adoptar archivos huérfanos.
La proyección conserva la reserva de auditoría y el margen ×2; fallos e
intentos sin cierre no reciben devolución silenciosa de tiempo.

La auditoría operativa detectó cuatro defectos de publicación y estados.
La corrección publica cada unidad mediante una transacción de directorio,
terminaliza los fallos de recursos del arranque y liga cada cierre global
al hash de un intento exitoso, después del último control. Las pruebas
reproducen las interrupciones antes y después de publicar. Un candidato de
cierre guardado y sellado por un intento exitoso permite terminar esa
publicación tras una caída, sin recalcular el roster ni abrir otro presupuesto.
Los candidatos de intentos pausados o fallidos no reciben autoridad. La
reauditoría independiente cerró esas correcciones sin defectos materiales
abiertos en el ejecutor revisado.

El [inventario y perfil reales](RESULTS_OPERATOR_OBJECTIVE_PROFILE.md) ya
pasaron: 2048 recibos autenticados y las cuatro escenas 0 extraídas y cotejadas,
con 13.815751 s acumulados. La proyección conservadora da 10943.721359 s,
superior al límite de 1800 s. El barrido no se inició. Sigue una revisión
acotada de costo sobre los compactos preservados, sin repetir fits, reducir
el roster ni ampliar tiempo en silencio. Después siguen barrido y replay
cuando el costo lo permita, y las auditorías de evidencia y alineación.

La [primera revisión cacheada](RESULTS_OPERATOR_OBJECTIVE_CACHE_PROFILE.md)
ya fue implementada, auditada y medida sobre esos cuatro compactos: conserva
todos sus bytes y lleva la suite conjunta a 92 pruebas CPU pasadas. Acumula
15.821776 s entre inventario y perfiles, sin reiniciar presupuesto. La
proyección baja a 8023.686073 s pero sigue excediendo 1800 s; no habilita
barrido. Sigue revisar serialización y costo restante, con una revisión de
runtime explícita antes de producción. La versión congelada permanece intacta.

El diagnóstico sigue siendo retrospectivo. No cambia el pool, los targets,
la época seleccionada ni las redes; tampoco convierte mínimo UB en toda la
información recibida por la cabeza. La GPU no es necesaria para este hito.
El perfil no aporta un resultado de alineación del roster, promoción
arquitectónica ni GO/NO-GO.
