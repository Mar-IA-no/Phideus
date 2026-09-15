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

Los tests `test_measurement_primitives.py`, `test_measurement_contract.py`,
`test_measurement_operator.py` y `test_measurement_reporting.py` usan fixtures
mecánicos, no escenas prospectivas. Una revisión independiente posterior a la
implementación examinó íntegramente sensor, métricas y sus fixtures sin findings
materiales. Su alcance no incluye el ejecutor, la persistencia ni el sello global.

Secuencia restante:

1. Completar ejecutor recuperable, autenticación de referencias, separación de
   puertos y preservación de todos los artefactos. Probar interrupciones y replay.
2. Auditar esa integración antes de perfilar desarrollo y calibrar el detector.
3. Medir CPU/GPU y publicar presupuesto/manifest congelado antes del test.
4. Ejecutar las512escenas pareadas, sellar predicciones y evaluar después.
5. Completar replay, informe, auditorías técnica y de horizonte, documentación
   y elección justificada del siguiente goal.

El cierre sigue siendo el experimento completo. Estos checks no acreditan
transferencia a audio ni promueven una arquitectura. Los resultados históricos
y fuentes congeladas permanecen intactos.
