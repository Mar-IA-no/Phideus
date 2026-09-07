# Compatibilidad entre parciales: preflight CPU

Fecha: 2026-09-07. Implementación congelada: `d088802`.
Estado: **preflight mecánico completo; entrenamiento pendiente**.

El [contraste de pérdidas](PLAN_SHARED_PARTIAL_COMPATIBILITY.md) ya tiene
productor frequency-only, descriptor observable, control de pesos desacoplados
(sham), pérdidas tensoriales y lector sin cardinalidad verdadera. Las suites
separadas pasaron 11, 7 y 3 pruebas; implementación y ejecutor fueron auditados
independientemente antes de esta corrida.

## Qué mostró

Las 64 escenas de desarrollo contienen 34.864 triples. Sus estadísticas
agrupadas son diagnósticas: los triples no son unidades independientes.

| Triple generativo | Conteo | Mediana de residual (cents) | Soporte medio |
|---|---:|---:|---:|
| Una fuente | 4.186 | 1,195 | 0,972 |
| Fuentes distintas | 30.678 | 48,718 | 0,181 |

El descriptor no es constante, pero hay solapamiento: algunos triples de
fuentes distintas tienen residual casi nulo. Compatibilidad local no equivale
a identificación global. El sham fue evaluable en las 64 escenas y cambió
todos sus pesos; su correlación con los pesos originales varió entre
−0,179 y 0,257, con media por escena 0,003. No es una garantía de independencia.

En las escenas de desarrollo 0 y 1, con inicialización compartida de la red
de pares, la norma del gradiente físico ponderado por lambda fue 0,235 veces
la norma de BCE; para sham, 0,234. No hubo gradiente nulo ni dominancia de esas
penalizaciones. Transitividad estuvo inactiva en esta inicialización; el
fixture de violación predeclarado sí verifica su activación. No hubo pasos de
optimización: los estados guardados son iniciales, no modelos entrenados.

## Costo y artefactos

La corrida completa duró 1,631 s. Geometría/descriptores/reader y escritura
tomaron 0,303 s, con RSS máximo de 92.401.664 bytes; el cómputo de gradientes
tomó 0,087 s después de importar torch, con RSS máximo de 600.408.064 bytes.
El tiempo total incluye los dos procesos secuenciales y sus importaciones.
Los fixtures máximos N=24/N=32 y la deformación sintética fueron finitos.
No se consultó ni usó GPU; no se abrieron splits de test.

El costo medio observado de features/sham/reader fue 0,002369 s por escena
de desarrollo. Extrapolarlo a train+validation da 21,83 s; escalar el fixture
N=24 da 78,02 s. Para las 14.336 escenas previstas fuera de desarrollo, usar
13.312 de cardinalidad train y 1.024 al costo del fixture N=32 da 52,83 s;
escalar ambos fixtures máximos da 133,98 s. Son proyecciones puntuales, no
límites superiores: excluyen generación, serialización, búsqueda del lector
y entrenamiento. La preparación conservará un cache por escena y presupuestos
medidos por bloque; no recalculará geometría en cada época. Esta proyección no
autoriza abrir test antes de congelar implementación y selección de validation.

Fuente canónica: [manifest](../../data/atencion_armonica/shared_partial_preflight_v1/manifest.json),
[geometría](../../data/atencion_armonica/shared_partial_preflight_v1/geometry_report.json)
y [gradientes](../../data/atencion_armonica/shared_partial_preflight_v1/gradient_report.json).
El manifest verifica nueve artefactos, incluidas observaciones/sidecars
separados, cache NPZ y estados/logits iniciales, y liga ocho archivos fuente.
SHA-256 del manifest: `c3d297a0261a3ff121c3ce46fbbe878adfe8b99b2ce8b211da3d2e5e6afe71ed`.

## Qué sigue

Implementar y verificar cache, entrenamiento, checkpoints y evaluación
conservando el test cerrado para selección. Después, smoke GPU autorizado de
hasta diez minutos para medir backward N=24 y evaluación N=32, batch 128,
memoria y proyección de las quince corridas contra el presupuesto del plan.
Todavía no hay tiempo GPU ni VRAM medidos. La receta sigue sin cambios.

Este resultado verifica que la relación física elegida produce un descriptor
no constante y una pérdida diferenciable en el diagnóstico acotado. No muestra
mejora aprendida, no acredita una geometría armónica universal y no completa
el goal experimental.
