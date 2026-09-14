# Medias y codec: resultados exactos, proyección aún fuera del límite

2026-09-14. El [fast path](../../src/atencion_armonica/operator_diagnostic_fast.py)
reprodujo los bytes y recibos de las cuatro escenas 0 preservadas. La
implementación y su operador pasaron auditoría independiente; la suite
completa pasó 107 pruebas. No se ejecutaron otras escenas, extracción,
training, forward, fit o GPU. El diagnóstico científico sigue incompleto.

| Escenario | Cálculo (s) | Codec (s) | Cotejo (s) | Publicación/guardas (s) | Total diagnóstico (s) |
|---|---:|---:|---:|---:|---:|
| IID | 0.045755 | 0.020637 | 0.002706 | 0.016811 | 0.085908 |
| Mayor beta | 0.058597 | 0.027523 | 0.005078 | 0.015852 | 0.107050 |
| Polifonía | 0.039355 | 0.025059 | 0.005082 | 0.016572 | 0.086068 |
| Familia deformada | 0.053706 | 0.025682 | 0.003992 | 0.003279 | 0.086659 |

La primera medición válida consumió 0.578567 s, con 16.400343 s acumulados y
pico RSS de 470908928 bytes. La fórmula original proyecta 6251.429694 s frente
a 1800 s; el almacenamiento proyectado continúa dentro de 4 GiB. Son cuatro
mediciones fijas, no réplicas para atribuir una aceleración causal al código.
La mejora observada en fixtures no se extrapola al roster.

El siguiente paso cambia de nivel: revisar explícitamente el modelo de costo
y la continuidad del runtime, no añadir otro cache por inercia. La fórmula
escala toda la pierna por pares y supone 82 candidatos en cada escena. Esto
amplifica también publicación y guardas. Como comprobación algebraica, usando
sólo los 0.013082740 s de publicación/guardas IID del [perfil anterior](RESULTS_OPERATOR_OBJECTIVE_CACHE_PROFILE.md)
y poniendo a cero el resto del diagnóstico, esa fórmula todavía proyecta
1990.069337 s. No es una corrida ni un límite inferior universal de runtime:
expone por qué optimizar únicamente cálculo/codec no resuelve necesariamente
el criterio de admisión actual.

La revisión debe considerar la carga auténtica del roster, el costo fijo,
las lecturas/verificaciones y el crecimiento de las guardas con archivos
publicados. También debe evaluar si separar una extracción reutilizable
permite medir y acotar el costo restante sin repetir fuentes. Todo cambio
requiere enmienda explícita y auditoría antes de ejecutarse; se conservan
roster, margen, reserva y presupuesto acumulado, sin sustituirlos por un
promedio favorable. No hay autorización de barrido, replay, promoción ni GO/NO-GO.

## Recibos

Raíz: `data/atencion_armonica/operator_objective_alignment_v1/`.

- `attempts/0004/fast_profile.json`: SHA256 `7d6e32f6aad85d2eb0306d713068a3850964068a3b0b7b115a6b2f0d7f029265`.
- `attempts/0004/finish.json`: SHA256 `68f75f969143a778c8935ac4e9976cbacf999391672fc2254ff4237f76340f53`.
- `attempts/0004/runtime_revision.json`: SHA256 `f6cc003f4ed6976b5912a37c5b6264d3fc33cf3e93c43a3269725af4339f2a82`.

El finish COMPLETE sella informe y revisión; el informe candidato por sí solo
no acredita cierre. No existe marcador global de roster ni replay.
