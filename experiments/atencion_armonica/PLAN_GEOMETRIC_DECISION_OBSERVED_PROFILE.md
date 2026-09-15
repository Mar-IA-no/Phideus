# Perfil del recorrido observable completo

2026-09-14. Refinamiento operativo del plan prospectivo, sin cambiar el
contraste, las pérdidas, el roster ni los límites científicos. La implementación
y este encuadre se auditan juntos antes de ejecutar el operador real.

## Entrada y recorrido

Usar las primeras 16 observaciones TRAIN del shard 0 del corpus OPEN importado,
autenticadas por su autorización y manifests. No seleccionar por resultados,
regenerar observaciones, abrir targets ni cambiar el normalizador. Conservar
sus IDs y seed originales. Leer los estados iniciales/seleccionados desde el
archivo COMPLETE ya producido; no repetir training, calibración ni exportación.

Recorrer la misma integración prevista para los tests: features, tres backbones,
pool común, ajustes compartidos con grilla257×65/stride4 y lotes de8 grupos,
entradas, referencias clásicas raw UB,144predicciones y transportes para las
primeras cuatro escenas elegibles. Repetir el recorrido observable completo
sólo sobre el roundtrip declarado de esas cuatro, conservando144predicciones
derivadas. No ejecutar un tercer recorrido sobre qcenter32.

Las nuevas coordenadas de roundtrip de TRAIN son fixtures OPEN de perfil,
no tests. Antes del freeze real deben incorporarse explícitamente a las
exclusiones prospectivas, enlazando este perfil y las coordenadas preservadas.
No modificar el inventario de archivo ya cerrado: extenderlo con procedencia.

Reabrir después el mismo recorrido con callbacks que prohíban modelo, forward
y fitting. Su recibo final debe coincidir exactamente. El perfil no puede
marcar COMPLETE si esa recuperación o cualquiera de los144estados falta.
Preservar costes por fase, unidades observadas, memoria y bytes producidos.

## Recursos y proyección

RTX3090 local con propiedad verificada antes de inicializar CUDA; una hebra CPU,
sin TF32/AMP, algoritmos deterministas y workspace CUBLAS fijado. Cabezas y
backbones CUDA; fitter CUDA float64. Reutilizar los callbacks científicos
vigentes y verificar sus checkpoints/runtime. No cambiar silenciosamente al CPU.

Reserva del operador:480s, dentro de los523,495848s de perfil todavía disponibles
al diseñarlo. Cada fase medida conserva un límite120s, además del límite global.
El ledger común cuenta desde el lanzamiento, incluidos importaciones/admisión.
Guards RSS/VRAM8GiB, disco nuevo100GiB y libre30GiB. No reintentar un perfil
parcial por simple relanzamiento: conservar recibos y reconciliar primero.

Proyectar original observable/clásico/readout por2048/16 y roundtrip por4
escenarios. La proyección del readout original sobrecuenta los transportes
(cuatro por cada bloque de16, cuando el test exige cuatro por512); conservar
esa reserva, no venderla como tiempo esperado exacto. Contar admisión del
archivo por escenario. Aplicar25% de margen al recorrido prospectivo y a la
recuperación observable; extrapolar bytes de todo el perfil por2048/16 también
sobrecuenta probes. Publicar unidades/costos, no afirmar cota computacional
para todo OOD desde16escenas TRAIN. Los guards rigen durante la campaña.

Cada recibo de fase explicita cantidad y tipo de unidad, además de cabezas,
backbones o probes cuando corresponda. El tiempo desde lanzamiento que no
pertenece a esas fases se registra como overhead observado y se agrega cuatro
veces tanto a fresh como a recuperación antes del25%, conservadoramente. El
tramo posterior al snapshot —publicar informe, verificar fuentes y finish—
queda declarado como no medido en esa proyección parcial; el finish lo cobra
y debe incorporarse al revisar la admisión total, no desaparecer del ledger.

El perfil informa sólo recorrido y recuperación observable. La futura admisión
completa debe agregar producción única/serialización de draws, exclusiones,
sello y evaluación de métricas/bootstrap al presupuesto correspondiente; no
presentar esta proyección parcial como autorización automática de tests.

## Cierre del hito

Operador COMPLETE con outputs y recuperación comprobados, o intento incompleto
con evidencia del límite/fallo. Ningún resultado de modelos se usa para elegir
variantes o modificar el protocolo. Sigue freeze completo y ejecución2048,
sellado antes de truth, evaluación/replay y auditorías del goal original.
