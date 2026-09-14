# Revisión de costo sin cambiar el diagnóstico

2026-09-14. Diseño de implementación dentro del goal vigente. No reemplaza
el [protocolo congelado](PROTOCOL_OPERATOR_OBJECTIVE_ALIGNMENT.md), no modifica
el manifiesto v1 y no autoriza todavía un barrido con otro ejecutor.

## Evidencia y pregunta

El [perfil fijo](RESULTS_OPERATOR_OBJECTIVE_PROFILE.md) completó inventario2048
y cuatro escenas; su proyección de 10943.721359 s supera el límite de1800 s.
El código instrumentado sobre los cuatro compactos preservados reproduce sus
bytes, pero repite oracles/órdenes de targets entre31métodos y entre estratos
con idénticos candidatos. El empaquetado también recorre recursivamente muchas
estructuras pequeñas. Esa medición identifica candidatos de optimización, no
promete un factor de mejora ni una razón para ocupar GPU.

La pregunta acotada es si se puede ejecutar exactamente el mismo diagnóstico
con menos trabajo repetido. No cambia ninguna hipótesis, target, celda, estrato,
métrica, reducción o regla de desempate. No se recalculan fits ni forwards.

## Primera implementación y prueba

Construir `src/atencion_armonica/operator_diagnostic_cached.py`, con pruebas
en `experiments/atencion_armonica/test_operator_diagnostic_cached.py`. Estos
paths quedan fuera del glob `operator_objective_*.py`: v1 exige exactamente
ocho módulos en ese glob y sus bytes no se modifican. Verificar que su
`source_snapshot()` y review sigan coincidiendo después de añadir el prototipo.

La implementación nueva debe:

1. Calcular targets y estratos con las funciones v1 intactas.
2. Compartir el oracle de target64/target32/ARI por tupla de candidatos dentro
   de cada escena. Compartir también la descripción de un mismo score sobre
   idéntica tupla cuando distintos esquemas repitan ese universo.
3. Reutilizar sin cambios las funciones de regresión, agregación, diferencias
   pareadas y esquema de salida. No mutar arrays, oracles ni resultados que
   compartan referencias en memoria. El formato durable sigue siendo el mismo.
4. Mantener inicialmente el codec v1, para medir por separado el aporte del
   cache. Sólo si el costo lo justifica se diseñará una serialización más
   rápida; no se reduce el contenido preservado.
5. Comparar bytes completos contra v1 en fixtures deterministas: vacíos,
   singleton, ties/constantes, k2/3/4, estratos coincidentes y distintos,
   predicciones distintas por celda y soportes indefinidos. Conservar rechazo
   de rosters/dtypes/targets inválidos. No sustituir esa prueba por igualdad
   de unas métricas agregadas.

Esta primera fase sólo habilita implementación y tests de desarrollo CPU.
Antes de medir sobre compactos reales, auditar el módulo y su prueba
diferencial. La medición posterior usa exclusivamente las cuatro escenas0
ya abiertas, dentro del cap de perfil y presupuesto acumulados originales;
exige igualdad exacta de cada bundle. No selecciona escenas por resultados.

## Continuidad de versión: condición antes de producción

Los archivos revisados v1, sus recibos y su manifiesto permanecen intactos.
Una revisión de runtime debe ser un registro nuevo, autenticado e inmutable
que enlace el manifiesto original, los nuevos módulos, pruebas y auditoría.
Cada intento nuevo identifica esa revisión; no basta agregar código no ligado
al freeze. La raíz original conserva toda la secuencia de AttemptBudget:
fallos, perfiles y revisiones consumen el mismo total, sin reinicio ni expansión.

Los cuatro compactos v1 pueden reutilizarse sólo como prefijos autenticados,
después de demostrar igualdad científica. El nuevo runtime debe conservar
lock, publication no-replacing, terminalización, recovery y reserva de auditoría.
Antes de implementarlo se fijará el detalle del registro de revisión y de sus
puertos; este diseño no permite monkeypatching silencioso del ejecutor v1.

La primera evaluación de costo conserva la fórmula original, incluido máximo82,
2048escenas,27celdas,cuatro esquemas,replay,margen2 y600s de auditoría. Si sigue
sin caber, no se lanza el barrido. Cualquier modelo de costo distinto exige una
enmienda operativa explícita y auditada, motivada por cantidades observables de
trabajo; no se acepta quitar el margen, sustituir por un promedio favorable
de cuatro tiempos ni descartar métricas para obtener permiso de ejecución.

## Cierre de esta revisión dentro del mismo goal

Resultado positivo: bytes preservados, costo admisible, runtime revisado y
presupuesto continuo verificable; ejecutar entonces roster y replay completos.
Resultado insuficiente: identificar el término de costo que sigue impidiendo
el barrido y diseñar la siguiente corrección concreta, sin presentarlo como
cierre del goal ni de la hipótesis geométrica. No GO/NO-GO ni promoción.
