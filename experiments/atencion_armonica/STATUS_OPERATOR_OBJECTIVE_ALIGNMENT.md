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

Faltan el adaptador de lectura autenticada/extracción compacta, el ejecutor,
la agregación completa de escenarios, pruebas de integración y auditoría de
implementación. Después corresponden perfiles de las cuatro escenas fijadas,
recibo de presupuesto, barrido de las 2048 escenas y replay, seguidos de
auditorías de evidencia y alineación. No se ejecutó ninguna de esas etapas.

El rebase de recibos observa unos 893 MB comprimidos y 9.28 GB decodificados
de factores; no es una medición de costo ni una validación de sus blobs.
El protocolo fija un operador CPU, un hilo, 6 GiB RSS y 30 minutos acumulados
de operadores, con reserva de auditoría y guardas de almacenamiento. No se
iniciará el roster si el perfil no justifica ese presupuesto.

El diagnóstico sigue siendo retrospectivo. No cambia el pool, los targets,
la época seleccionada ni las redes; tampoco convierte mínimo UB en toda la
información recibida por la cabeza. La GPU no es necesaria para este hito.
No hay resultado nuevo de alineación, promoción arquitectónica ni GO/NO-GO.
