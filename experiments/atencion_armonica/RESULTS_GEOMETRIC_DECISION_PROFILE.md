# Perfiles de energía geométrica para la decisión

2026-09-14. Perfiles de primitivas y del recorrido observable completos;
los tests prospectivos siguen sin abrir. El entrenamiento posterior a los
primeros perfiles está documentado en el [estado](STATUS_GEOMETRIC_DECISION_ENERGY.md).

## Perfil inicial de primitivas

Código de ejecución: `50ac5b39c63ac960536891538da514721f307031`.

El [operador](profile_geometric_decision.py), auditado antes de ejecutar,
reutilizó OPEN y la escala congelada. Midió una carga completa del primer
backbone y extrajo su primer batch TRAIN programado. Los 32 inputs/targets
coinciden en dtype, forma y bytes con la carga completa y con su reapertura.
La carga tomó 29,255889 s; el operador completo, 55,945913 s. La calibración
OPEN se cargó, pero no se evaluó ni se usó para seleccionar modelos.

Cada backend completó los cuatro casos de cabeza —dos objetivos, batch real
y envolvente máxima—, con 25 updates mecánicos y replay exacto desde el décimo.
La recuperación CUDA quedó comprobada en ese runtime; no implica igualdad
de trayectorias entre CPU y GPU ni reemplaza una campaña completa.

| Medición | CPU | RTX 3090 |
|---|---:|---:|
| MSE, envolvente: update medio después de los primeros cinco | 12,304 ms | 7,990 ms |
| MSE, primer batch real: media de los 25 updates | 6,607 ms | 6,915 ms |
| Ajuste de los grupos medidos, suma de tiempos de fitting | 1,448735 s | 0,371480 s |

El arranque CUDA fue más costoso; no corresponde extrapolarlo como costo
recurrente de cada batch. La [proyección](../../src/atencion_armonica/geometric_decision_work.py)
cobra los primeros cinco updates por celda por separado, usa el mayor tiempo
medio restante entre casos y agrega snapshots, calibración y cargas completas.
Para 72 celdas, 50 épocas y 4036 escenas elegibles cuenta 457200 updates,
17856 snapshots y hasta 12672 batches de calibración.

La proyección diagnóstica es 6848,962501 s en CPU y 4836,115976 s en CUDA;
con reserva de 25%, 8561,203127 s y 6045,144970 s. Favorece CUDA para la
cabeza bajo este cálculo, pero no es tiempo observado de entrenamiento ni
admisión final del supervisor. El perfil del fitter mide grupos, no la
pipeline de tests: aún faltan su composición, proposer, backbone e I/O.

## Evidencia y límites

Raíz de artefactos: `data/atencion_armonica/geometric_decision_energy_v1/`.
Los resultados y sus bindings están enlazados por `control/outputs/profile-*.json`;
los intentos `0001–0005` terminaron COMPLETE. Sus tiempos acumulados suman
71,594984 s de los 600 s disponibles para perfiles, sin reiniciar el costo OPEN.

El último recibo `control/attempts/0005/finish.json` tiene SHA256
`a42a63d111ae9db668b1e5ce94d9cb5270f2dcb987836145ae6e6b4db1a9c61e`.
La preparación se cierra en `0001/finish.json`, SHA256
`d7bae9d65d4a0d4c9b28f05af45efdb879d7f18703660f1743732d6f184adc78`.
Cada caso conserva tiempos individuales, snapshots y outputs; el fitter
conserva workloads y factores completos. No hubo sampler ni nuevos tests.

Estos perfiles precedieron al supervisor y a las 72 corridas, ya completadas.
No aportan por sí solos evidencia a favor de una arquitectura.

## Recorrido observable y recuperación

El [operador del recorrido completo](profile_geometric_decision_observed.py),
auditado antes de ejecutar, terminó el intento `0010` en **113,436136 s**.
Usó las primeras 16 escenas TRAIN ya conocidas y los roundtrips de las cuatro
primeras elegibles. Completó features, tres backbones, candidatos, ajustes
compartidos, cuatro referencias clásicas y los 144 estados iniciales/seleccionados.
Conservó transportes de representación y una segunda pipeline sobre cada probe.
La recuperación devolvió exactamente el mismo recibo sin ejecutar modelos ni
ajustes. No se generaron escenas de test ni se abrieron sus respuestas.

| Medición | Segundos |
|---|---:|
| Observables originales: features, backbones, candidatos y ajustes | 43,116771 |
| Referencias clásicas originales | 8,848362 |
| Readouts originales y transportes | 11,122933 |
| Observables de los cuatro roundtrips | 8,866914 |
| Referencias clásicas de los roundtrips | 1,333576 |
| Readouts de los roundtrips | 1,695074 |
| Recuperación observable completa | 34,097760 |

El perfil registra 522190848 bytes de pico reservado CUDA, 1628557312 bytes
de RSS y 56703389 bytes de artefactos antes del informe final. Estos picos
no son cotas para todas las escenas OOD. La proyección parcial con margen
del 25% es **10170,548045 s** de recorrido observable, **5470,924331 s** de
recuperación y **9072542240 bytes**. Sobrecuenta los probes y transportes al
extrapolar desde bloques pequeños. No incluye todavía producción/serialización
de escenas nuevas, sello global ni métricas/bootstrap. El tramo final de
publicación y verificación añade 0,958818 s medidos al perfil y debe incorporarse
a la admisión total. Esta proyección no autoriza por sí sola los tests.

Fuentes bajo la raíz experimental:

- `profiles/observed-cuda-0/result.json`, SHA256
  `28178c62c4526f74b110d1db17edb809376841829c2c2ae1bdcc4eb11cf06b64`.
- `profiles/observed-cuda-0/observed/observable-complete.json`, original y
  recovery, SHA256 `024a71330754271d167df0f6ba2b23990176f7fd99e99f7bfa580e40c41ff933`.
- `control/attempts/0010/finish.json`, SHA256
  `e05314b43451bb3993b9e01124c0d970684c974152dfe3537366c171cffd30d3`.

El coste acumulado de perfiles, incluido el selector, es **189,940288 / 600 s**.
Sigue completar el presupuesto restante, extender las exclusiones con las
coordenadas del perfil y congelar la ejecución prospectiva. El goal permanece
abierto hasta completar tests, evaluación, replay y auditorías finales.

## Evaluación y cierre del perfil CPU

El [operador CPU](profile_geometric_decision_closing.py), corregido y auditado
antes de ejecutarse, completó el intento `0011` en **25,693755 s**. Reutilizó
las mismas 16 escenas TRAIN y cuatro probes; reconstruyó sólo las tuplas ya
conocidas y comprobó igualdad exacta de observaciones y respuestas. No creó
tests. Las métricas de los 144 estados y sus estratos se reprodujeron exactamente
desde archivos, tanto para originales como para probes.

| Fase | Segundos |
|---|---:|
| Admisión y setup | 0,375389 |
| Lectura/reconstrucción de verdad conocida | 0,025640 |
| Productor e IO de tuplas conocidas | 0,205225 |
| Métricas originales / replay | 7,359000 / 6,829589 |
| Métricas probes / replay | 2,846908 / 2,550117 |
| Bootstrap sobre fixture repetido, no evidencia científica | 1,519616 |
| Inventario del árbol observable | 0,381669 |

El pico RSS fue 910499840 bytes y el volumen medido antes del informe final,
30193718 bytes. El ledger acumula **215,634043 / 600 s** de perfiles. La
proyección completa original, antes de añadir el tail de 1,107130 s, es
15857,815986 s fresh y 8100,458485 s evaluación/replay, con margen del 25%.
La segunda excede los 7090,675836 s restantes de su etapa. Por ello no habilita
los tests: se prepara una revisión explícita de contabilidad de recuperación
observable, manteniendo el trabajo y los límites totales. El supervisor y esa
revisión todavía requieren validación; no hay resultados prospectivos.

Fuentes bajo la misma raíz experimental:

- `profiles/closing-cpu-0/result.json`, SHA256
  `dc1e3b20723b91303c1af7916c20497e8a36ec3679aca2d9f731ba8bcbc968b9`.
- `control/attempts/0011/finish.json`, SHA256
  `4f08e8fbd6252c087acac654928f2e7aefdf4075a6e132e3735ec1df5fb8e615`.
