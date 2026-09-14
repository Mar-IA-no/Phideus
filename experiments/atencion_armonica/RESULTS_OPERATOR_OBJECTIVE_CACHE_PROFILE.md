# Cache del diagnóstico: equivalencia preservada, costo aún no admisible

2026-09-14. La primera medición de la implementación cacheada reprodujo
exactamente los cuatro bundles de escena 0 del [perfil original](RESULTS_OPERATOR_OBJECTIVE_PROFILE.md).
El barrido de 2048 escenas y su replay siguen sin ejecutar. Este resultado
informa costo computacional, no alineación geométrica ni ventaja neuronal.

El [cache](../../src/atencion_armonica/operator_diagnostic_cached.py) comparte
oracles y descripciones de subconjuntos idénticos dentro de una escena, sin
cruzar scores o escenas. Su diseño, implementación y operador de medición
pasaron auditoría independiente; la suite conjunta pasó 92 pruebas CPU.
La [enmienda de medición](AMENDMENT_OPERATOR_OBJECTIVE_CACHE_PROFILE.md)
conserva presupuesto, fórmula, fuentes y código v1. Sólo la primera medición
exitosa del mismo snapshot es elegible; una segunda llamada autenticó y
reutilizó ese informe sin abrir un nuevo intento.

## Medición fija

| Escenario, escena 0 | Candidatos | Cálculo (s) | Cotejo (s) | Codec (s) | Publicación y guardas (s) | Total diagnóstico (s) |
|---|---:|---:|---:|---:|---:|---:|
| IID | 31 | 0.052067 | 0.002588 | 0.048472 | 0.013083 | 0.116210 |
| Mayor beta | 65 | 0.060738 | 0.005300 | 0.034831 | 0.012719 | 0.113588 |
| Polifonía | 65 | 0.048541 | 0.005111 | 0.031820 | 0.013104 | 0.098575 |
| Familia deformada | 48 | 0.060106 | 0.003802 | 0.030711 | 0.011595 | 0.106214 |

Los cuatro bundles fueron byte-idénticos a v1. El operador consumió
0.596397 s, incluido cargar y autenticar los compactos, con pico RSS de
445734912 bytes. El total acumulado de los cuatro intentos es 15.821776 s;
no se reinició el presupuesto. No hubo nueva extracción, entrenamiento,
forward, fit, muestreo ni GPU.

La fórmula original proyecta 8023.686073 s frente al límite de 1800 s;
almacenamiento sigue dentro del límite, con 1388136778 bytes proyectados
frente a 4 GiB. Esa cifra temporal es una proyección conservadora, no tiempo
efectivamente ejecutado. El perfil incluye cálculo, codec e IO: la reducción
frente a v1 no es un factor de aceleración causal atribuible sólo al cache.

## Próximo paso y límites

El cache por sí solo no habilita el barrido. El desglose justifica revisar
serialización y trabajo repetido restante. También hace explícita una
limitación del modelo de costo: aplica crecimiento por pares a toda la
pierna, incluidas operaciones cuyo costo no depende sólo de pares.
Cualquier revisión de esa proyección requiere diseño y auditoría propios,
basados en trabajo observable; no basta sustituirla por el promedio favorable
de cuatro escenas, retirar el margen o ampliar el presupuesto en silencio.

El runtime completo optimizado todavía no existe. El siguiente corte debe
resolver esa continuidad preservando recibos, presupuesto y bytes científicos;
no añade métricas ni altera la pregunta geométrica para conseguir una corrida.
El goal permanece incompleto y no hay promoción ni GO/NO-GO.

## Evidencia

Raíz: `data/atencion_armonica/operator_objective_alignment_v1/`.

- `attempts/0003/cache_profile.json`: SHA256 `dcbc26992c52e14b19955d6e9fbc52fe9745f0634937c4b9183f8df7af690bb5`.
- `attempts/0003/finish.json`: SHA256 `f3cbede248f578d3480f97f2208c4adb5588da24b310a4ac6bc4db904824cc5a`.
- `attempts/0003/runtime_revision.json`: SHA256 `2a7e4eea678a73e8ee0225c870b2e1231e95efa1dabc4a8e81737c378260cf26`.

El finish exitoso sella informe y revisión. El estado `CACHE_PROFILE_CANDIDATE`
del informe no acredita por sí solo una ejecución completa: su autoridad
depende de ese sello. No hay marcador global de barrido ni replay.
