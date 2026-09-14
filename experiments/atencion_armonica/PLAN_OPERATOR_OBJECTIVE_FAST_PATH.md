# Reducir serialización y reducciones repetidas del diagnóstico

2026-09-14. Implementación adicional dentro del goal operación–objetivo.
El turno anterior produjo evidencia de costo y equivalencia, no un bloqueo.
La pregunta científica y los 2048 casos siguen pendientes de ejecución completa.

## Motivo y alcance

El [perfil cacheado](RESULTS_OPERATOR_OBJECTIVE_CACHE_PROFILE.md) conserva
los cuatro bundles, pero proyecta 8023.686073 s frente a 1800 s. Sus cuatro
mediciones separan 49–61 ms de cálculo, 31–48 ms de codec y 12–13 ms de
publicación/guardas, aproximadamente. Optimizar únicamente codec no basta
para garantizar que pase esa fórmula: el costo de cálculo también importa.

Una inspección cProfile sobre el fixture determinista de 35 particiones 4+4
(no otra escena de campaña) identificó 41578 llamadas recursivas a `_pack`
y 6504 llamadas a `np.mean`. Pairing consume 0.033 s de 0.063 s de cálculo
instrumentado. Es evidencia de trabajo repetido; los tiempos con profiler
no sustituyen el perfil de campaña ni permiten extrapolar una aceleración.

Implementar dos optimizaciones puras, sin alterar los ocho módulos v1 ni los
cinco archivos ligados al perfil cacheado. Archivos nuevos, fuera del glob
congelado: `src/atencion_armonica/operator_diagnostic_fast.py` y
`experiments/atencion_armonica/test_operator_diagnostic_fast.py`.

## Semántica que debe permanecer exacta

1. Cache de medias local a cada llamada de escena. Validar valores antes de
   consultar el cache y usar claves con bits float64, distinguiendo −0/+0,
   orden, None y extensión. En cada miss calcular por la función v1 original;
   no sustituir `np.mean` por otra suma, no redondear, reordenar ni imputar.
   Usarlo en resúmenes, agregación entre celdas y pairing, manteniendo las
   mismas secuencias de valores, validaciones y soportes. Las funciones
   copiadas deben contrastarse línea por línea con las referencias.
2. Codec con validación completa pero sin reconstruir recursivamente cada
   mapping/lista antes de JSON. Validar tipos, claves string, tag reservado,
   arrays numéricos finitos y ausencia de ciclos. Memoizar objetos repetidos
   sólo dentro de la llamada, sin omitir validación de una rama distinta.
   Usar JSON estándar con los mismos parámetros de v1 y conversión de arrays
   y escalares NumPy mediante el puerto puro `_pack` v1. Mantener newline,
   orden de claves, gzip nivel3/mtime0, hashes y recibo exactos. El decoder
   v1 permanece intacto. No convertir la salida durable en un grafo referencial.
3. Nunca mutar inputs ni mappings compartidos, ni instalar monkeypatches en
   módulos v1/cacheados. En desarrollo los fixtures pueden instrumentar llamadas
   para probar el cache; eso no se convierte en un puerto de producción.

## Verificación y frontera de ejecución

Comparar bundles completos y recibos con v1 y con el cache R745 en fixtures
vacíos/singleton, ties, predicciones diferentes, estratos repetidos/distintos,
k2/3/4, 35 particiones, soportes parciales, ceros con signo y llamadas
sucesivas con datos mutados. Para codec: dtypes/endian/layout/shape, arrays
vacíos, alias compartidos, claves Unicode, escapes, escalares NumPy,
inválidos, tags reservados, no finitos y ciclos. No basta comparar métricas.

La auditoría independiente Sol/high revisa plan, implementación y tests como
un corte de optimización, contra fuentes actuales. Se conserva su crudo en
Biblioteca. Antes de una medición real hace falta un operador nuevo, ligado
por review a todos sus archivos y a esa auditoría; debe conservar la primera
medición exitosa por snapshot, los cuatro compactos0 y AttemptBudget original.
Este documento autoriza construir y probar las funciones puras; no autoriza
todavía otro perfil real ni el roster. El detalle del operador y su auditoría
son precondiciones de cualquier ejecución posterior.

La medición posterior debe desglosar cálculo, codec, cotejo, publicación y
guardas, conservar igualdad completa de bytes y aplicar inicialmente la misma
fórmula v1. Si sigue sin caber, revisar el modelo de costo explícitamente por
operaciones observables; no aplicar un promedio favorable ni retirar margen,
reserva, escenas o celdas. No ampliar el presupuesto en silencio.

## Condición de avance hacia el goal real

No seguir acumulando optimizaciones sin medir el efecto conjunto. Tras este
corte se requiere un perfil revisado que indique si corresponde construir
el runtime completo o corregir la proyección por términos de trabajo. Faltan
roster, replay, auditorías técnica/alineación e informe de mecanismo. Ganar
velocidad no responde la pregunta geométrica; sólo remueve su costo de ejecución.
