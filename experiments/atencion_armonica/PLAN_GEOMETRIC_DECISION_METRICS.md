# Métricas de la decisión geométrica

2026-09-14. Implementación del apartado «Estimandos y evaluación» del protocolo
vigente y de PLAN_GEOMETRIC_DECISION_FRESH; no modifica su diseño científico.

Reutilizar el kernel de entropías/ARI ya verificado del diagnóstico anterior,
pero no sus sumas float32 ni su supuesto de predicciones no negativas. El
target matemático tM suma componentes normalizados float64; tD convierte cada
componente a float32 y suma esos valores en float64. Las predicciones h/E
permanecen firmadas float64. Verificar E=sum64(h) antes de medir. Calcular
targets una sola vez por universo de candidatos, compartidos por las144cabezas.

El núcleo puro no lee archivos ni concede autoridad para abrir labels. La
integración posterior sólo entrega labels después del sello global de los
cuatro tests y probes. La identidad por evento y la validación de sidecars
pertenecen a ese adaptador, no al kernel. Mantener separados diagnóstico por
candidato/escena, agregación por escenario y diagnóstico de transformaciones.

El núcleo implementará decisiones exactas por firma canónica, regret tM/tD,
ARI, pertenencia al conjunto óptimo, tau-b y errores de componentes. Reutilizar
la lógica de empates exactos: ninguna tolerancia numérica modifica elecciones.
Vacío produce NA y singleton conserva tau indefinido, no cero.

Para el primario, recibir el tensor de regret de todas las512escenas,
8brazos×3backbones×3readers y máscara explícita C no vacío. En pruebas puras
pueden usarse extensiones menores; el operador real exige512. Ninguna celda
puede faltar en una escena elegible. Promediar primero9celdas, formar los
cuatro contrastes fijados y después promediar escenas. Bootstrap pareado de
10000remuestras PCG64/2026091494; guardar índices, valores por escena y
distribuciones. Reutilizar los mismos índices en los cuatro contrastes.
Percentiles0.625 y99.375; no prueba GO/NO-GO. Evaluar remuestras por bloques
para limitar memoria y permitir checks de presupuesto.

Pruebas CPU de aritmética entregada frente a matemática, salidas negativas,
coóptimos, vacío/singleton, orden canónico, integridad de E, celda faltante,
peso de escena e interacción; verificar el bootstrap contra cálculo directo.
No nuevas observaciones ni truth de test. Auditar núcleo e integración de
evaluación antes de métricas reales y medir su coste con datos OPEN autorizados.
