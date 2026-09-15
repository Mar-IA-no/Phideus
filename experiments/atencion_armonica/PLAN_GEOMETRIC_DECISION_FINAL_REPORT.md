# Cierre del contraste de energía geométrica

Preparar la lectura de resultados mientras corre el operador congelado, sin
editar sus fuentes ni consultar respuestas. Esta capa deriva tablas y
diagnósticos de los artefactos ya previstos; no cambia métricas, selección ni
contrastes. Sólo se ejecuta sobre datos reales tras evaluation y replay COMPLETE.

El informe final debe separar:

1. Cuatro escenarios, con soporte original de512escenas por escenario y
   exclusión analítica explícita de universos vacíos, nunca eliminación de filas.
2. Cuatro contrastes primarios ya fijados, IC98,75% e índices bootstrap
   preservados. No recalcular un primario diferente ni inventar umbral decisorio.
3. Ocho variantes, initial/selected, referencias clásicas, efectos de loss,
   celdas condicionadas y slices de presencia/estratos. MSE de componentes
   bajo la loss de decisión no implica identificación de esos componentes.
4. Transporte: tolerancia numérica, elecciones, márgenes y singleton frente
   a batch, a partir de los registros ya recuperados. No llamarlo invariancia
   física aprendida.
5. Roundtrip: soporte y candidatos por identidad de evento; comparar canales
   y energías sólo en candidatos correspondientes, sin forzar igualdad de
   universos. Registrar elecciones por partición de eventos, colisiones de
   coordenadas y cambios de relaciones por conversión. Es el efecto conjunto
   del roundtrip, no una atribución separada a centrado o cuantización.
6. Correspondencia desacoplada efectiva: masks/donors preservados; denominadores
   por escena/celda, incluyendo estratos singleton sin cambio.

Una función pura de correspondencia puede probarse con particiones abstractas
y permutaciones, sin generar q32 ni nuevas escenas. Ningún helper de informe
adquiere autoridad para abrir tests por sí solo. Su operador deberá exigir la
cadena COMPLETE de evaluación/replay y guardar origen de cada tabla; su coste
se contabiliza en la reserva de auditoría, sin reiniciar el ledger.

El cierre incluye auditoría técnica independiente y auditoría de horizonte:
qué respondió el contraste, qué geometría fue impuesta o aprendida, límites
del generador y por qué la evidencia justifica el siguiente goal. Los checks
mecánicos no sustituyen esa lectura. No se exige ganador ni se declara GO/NO-GO.
