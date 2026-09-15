# Operador bajo medición — implementación, no resultado experimental

2026-09-15. El [plan](PLAN_OPERATOR_UNDER_MEASUREMENT.md) y
[protocolo](PROTOCOL_OPERATOR_UNDER_MEASUREMENT.md) incorporan la revisión
independiente de diseño. No se generó el test prospectivo, no se entrenó ni se
ejecutó inferencia de modelos en esta etapa.

Implementado:

- Render aditivo, condiciones pareadas, detector de picos y puerta de cardinalidad
  en `src/atencion_armonica/measurement_sensor.py`.
- Correspondencia conservadora, coste de detección, F/U/C y ARI/VI sobre soporte
  común en `measurement_metrics.py`, dentro del mismo directorio.
- Identidades por escena/condición, calibración global y traducción de rangos
  a eventos en `measurement_contract.py`.
- Adaptador observable a los kernels congelados en `measurement_operator.py`.
- Cuatro contrastes primarios y bootstrap pareado en `measurement_reporting.py`.
- Publicación atómica por etapa, recuperación de payloads y recibos de intentos
  en `measurement_store.py` y `measurement_stage.py`; no constituyen todavía
  el supervisor de admisión y presupuesto de campaña.
- Separación entre emisión observable y sidecar de evaluación en
  `measurement_emission.py`, y carga autenticada de las referencias congeladas
  en `measurement_reuse.py`, sin deserializar checkpoints de backbone.
- Control de presupuesto acumulado y recuperación en `measurement_control.py`;
  admisión por identidad de fuentes, congelación y sello como fases medidas en
  `measurement_admission.py`. Sólo un cierre completo habilita la fase siguiente.
- Persistencia de entradas y outputs de inferencia en `measurement_payload.py`
  y `measurement_prediction.py`; servicio CUDA explícito en `measurement_cuda.py`.
  Cada output GPU conserva su ejecución productora, incluso tras recuperación.
- Recorridos OPEN de emisión/detección, calibración y perfil en
  `measurement_open.py`, `measurement_calibration.py` y `measurement_profile.py`;
  snapshot de fuentes y guards en `measurement_snapshot.py` y
  `measurement_resources.py`. Su conexión global todavía está en implementación.

Los tests `test_measurement_primitives.py`, `test_measurement_contract.py`,
`test_measurement_operator.py` y `test_measurement_reporting.py` usan fixtures
mecánicos, no escenas prospectivas. Una revisión independiente posterior a la
implementación examinó íntegramente sensor, métricas y sus fixtures sin findings
materiales. Su alcance no incluye el ejecutor ni el sello global. Una revisión
separada del almacenamiento detectó cuatro defectos de identidad/encapsulación
y serialización; se corrigieron con regresiones específicas y la revalidación
independiente confirmó los cuatro cierres. Esto no sustituye la futura auditoría
del supervisor completo.

La revisión del control encontró cuatro defectos de contabilización, deadline,
identidad de fuentes y recuperación tras reinicio. Se corrigieron y una revisión
incremental independiente confirmó los cuatro cierres, con15fixtures oficiales
y2adicionales. El recorrido OPEN y la conexión de inferencia siguen en desarrollo;
esa revisión no acredita su integración ni constituye un perfil de recursos real.

La revisión independiente de inferencia encontró dos defectos de procedencia
CUDA y ciclo de vida del dispositivo. La revisión incremental confirmó sus
correcciones, incluyendo recuperación sin nueva ejecución y cleanup dentro de la
fase tras timeout o fallo parcial de inicialización. La suite conjunta alcanza
92pruebas CPU: escenas manuales, outputs neuronales simulados y ajustes5×5
pequeños. Ninguna de esas pruebas usa los modelos reales ni la grilla experimental
CUDA257×65. El ejecutable sólo expone snapshot y perfil CPU; aún no se ejecutaron.

Secuencia restante:

1. Conectar perfiles y calibración al ejecutor; completar exclusiones, proyección
   de coste acumulado, observación/predicción de test y evaluación/replay global.
2. Auditar esa integración antes de perfilar desarrollo y calibrar el detector.
3. Medir CPU/GPU y publicar presupuesto/manifest congelado antes del test.
4. Ejecutar las512escenas pareadas, sellar predicciones y evaluar después.
5. Completar replay, informe, auditorías técnica y de horizonte, documentación
   y elección justificada del siguiente goal.

El cierre sigue siendo el experimento completo. Estos checks no acreditan
transferencia a audio ni promueven una arquitectura. Los resultados históricos
y fuentes congeladas permanecen intactos.
