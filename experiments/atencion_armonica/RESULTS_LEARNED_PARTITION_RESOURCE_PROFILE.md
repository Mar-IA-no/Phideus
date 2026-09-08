# Lector aprendido: primer perfil de recursos

Fecha: 2026-09-08. Evidencia mecánica CPU, no entrenamiento ni resultado de
generalización. El protocolo científico permanece sin cambios.

El primer perfil geométrico terminó en 4,211 s y preservó sus mediciones.
La proyección de validación por celda fue de 4.397,012 s, superior al límite
operativo de 600 s aun sin añadir entrenamiento. Por eso no se ejecutaron los
perfiles de entrenamiento CPU/GPU ni se autorizaron datos prospectivos.

| Cantidad | Primer perfil |
|---|---:|
| RSS máximo del perfil | 386.740.224 bytes |
| Preparación de pools por shard, proyectada | 1.065,344 s |
| Validación por celda, proyectada | 4.397,012 s |
| Artefactos preservados, proyección de disco | 38.102.749.184 bytes |

El término dominante fue la lectura y comprobación de pools/filas. Una de
sus repeticiones tardó 22,763 ms; las otras dos del mismo caso, 2,412 y
2,151 ms. La fórmula extrapolaba ese máximo aislado, escalado por tamaño, a
cada lectura. No se estableció la causa de la pausa ni se descartó la medición.

La corrección operativa en revisión agrupa comprobaciones numéricas conservando
su semántica y mide tres bloques de 32 lecturas. El máximo por bloque dividido
por su denominador estima throughput de fixtures cacheadas, no latencia máxima
por archivo. Conserva el margen y los límites del supervisor. El perfil original
queda como evidencia del corte anterior; no se reescribe ni se elige el mejor
de corridas repetidas. Todavía no hay un perfil del código corregido.

Fuente preservada: `data/atencion_armonica/learned_partition_reader_v1/profiles/geometry_01/report.json`.
Su manifest tiene SHA-256 `039ffa5f4bb7c3fa89acee6c6cee7ac0eb4ee14913e16360d99b223e8f640cb4`.
Este resultado identifica una dificultad operativa, no refuta ni confirma la
hipótesis del lector aprendido.
