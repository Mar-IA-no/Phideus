# Perfiles de energía geométrica para la decisión

2026-09-14. Preparación del perfil y cuatro combinaciones CPU/CUDA completas;
ningún entrenamiento científico iniciado. Código de ejecución: `50ac5b39c63ac960536891538da514721f307031`.

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

Sigue completar y auditar el supervisor, congelar la admisión de entrenamiento
y ejecutar las 72 corridas. Estos perfiles no aportan evidencia a favor de una
arquitectura y no cierran el goal.
