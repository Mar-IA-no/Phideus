# Evaluación prospectiva y probes de energía geométrica

2026-09-14. Plan de implementación del protocolo congelado; ejecución pendiente
de selección COMPLETE, admisión de recursos y auditoría del mecanismo.
No habilita todavía draws, tests, forward ni fitting nuevos.

## Puertos y orden

1. Reabrir la selección y sus referencias de campaña/calibración. Conservar
   las 72 celdas initial y selected, sin elegir un backbone ni reader ganador.
   Sellar los estados CPU autenticados, normalizadores TRAIN y escala RMS;
   no usar estados nuevos generados desde una inicialización supuestamente
   equivalente cuando ya existe el checkpoint initial de cada celda.
2. Extender el inventario previo de exclusiones, preservando fuentes y aliases,
   con los cuatro tests históricos del lector generativo, sus observaciones
   transformadas ya usadas y los fixtures observacionales de este goal.
   Incluir la reconstrucción R754 de iid/0/2026091482 como reconstrucción,
   no como autenticación de los originales perdidos. La seed completa1482
   queda retirada; IID usa1582. Arrays de logits, scores o targets aritméticos
   no son nuevas observaciones y no se inventaría para ellos un q32.
3. Congelar protocolo, código importado, roster2048, seleccionados, exclusiones,
   backend y presupuesto. Dibujar cada tupla una sola vez con el productor
   vigente `_draw_scene` de learned_partition_data, sin modificar sus tablas
   globales ni los codecs históricos que fijan seeds anteriores. Conservar
   intent antes del draw, observación y sidecar separados y receipt después.
   Un par incompleto o colisión detiene; no rellenar con otro draw.
4. Puerto observable nuevo con seeds explícitas y arrays ragged. Reusar las
   funciones científicas puras: feature_record, checkpoint_forward, build_pool,
   candidate_inventory, GroupFitter, observable_rows, model_inputs y
   delivered_interface. No llamar wrappers históricos que adoptan las seeds,
   raíces o autoridad de otro experimento. No monkeypatch de constantes.
5. Preservar por escena features, logits de tres backbones, pools y unión de
   candidatos, ajuste compartido por grupo/rama, cotas y disponibilidad raw,
   inputs comunes normalizados, z32/d32 y donors. No repetir fits por backbone
   cuando usan la misma observación y universo. Los logits y estados crudos
   permiten reanálisis sin volver a entrenar ni ejecutar redes.
6. Obtener energías float64 initial/selected de las72celdas con los inputs
   float32 congelados. Elegir por energía y firma canónica con empate exacto;
   no imponer positividad a salidas firmadas. Conservar también la referencia
   geométrica clásica y distinguirla de z32 y de su precisión raw64.

## Probes y evaluación

- Para las primeras cuatro escenas elegibles por test, transporte consistente
  de candidatos, grupos, incidencia, canales y bypass. El cambio de base de
  canales exige transportar las columnas de pesos correspondientes; no basta
  permutar features contra una red que sigue interpretando sus viejas columnas.
  Guardar mapas y outputs sin invertir, invertir sólo para la comparación.
- Para esas escenas, ejecutar una segunda pipeline completa sobre qprobe32
  con el roundtrip ya fijado. qcenter32 queda sólo como diagnóstico. Los
  eventos conservan identidad antes de ordenar; ties de frecuencias no
  habilitan a inventar una correspondencia unívoca entre rangos.
- Sellar TODOS los outputs, incluidos probes, antes de parsear sidecars.
  Métricas originales validan la observación original; las transformadas
  reutilizan labels por identidad de evento, no afirman que el sidecar
  reconstruya qprobe32. Preservar las escenas sin candidatos como tales.
- Implementar las métricas y bootstrap del protocolo sobre outputs sellados;
  separar target matemático tM y target entregado tD. Replay de elecciones y
  estimandos sin training/forward/fitting. No inferir física desde el generador.

## Recursos y cortes

El perfil de primitivas del fitter ya existe, pero no admite automáticamente
la pipeline completa. Medir el trabajo adicional con observaciones TRAIN
ya autorizadas o fixtures explícitos, sin draws prospectivos. Estimar todo el
roster más25%, contando backbone, pool, grupos/ramas, serialización, lecturas,
readouts y probes; registrar supuestos de extrapolación. Perfil600s acumulados,
fresh21600s, evaluación/replay7200s y auditoría3600s mantienen sus límites.
La implementación nueva se audita antes de activar los puertos reales.

Auditorías de validez técnica y alineación después del contraste completo.
El cierre exige informe, wiki/documentación y commit/push. Los resultados,
no esta secuencia de implementación, decidirán el siguiente goal.

## Correcciones de interfaz de la auditoría R764

Antes de ejecución, el store exige schemas cerrados de factores y sus
asignaciones, y recompone los fits desde esos factores con igualdad canónica;
no repite el barrido de grilla. Ese costo de replay entra en el perfil completo.
La fuente lleva derivación tipada: original enlaza bytes observacionales
autenticados; roundtrip enlaza una fuente original autenticada, recalcula el
linaje y conserva las coordenadas diagnósticas. La autoridad de que un original
provenga de un draw admitido pertenece al freeze/productor futuro, no a una
etiqueta proporcionada por un caller arbitrario. El source conserva una única
observación JSON canónica para el hash y el retorno. No cambia la hipótesis,
el roster, las pérdidas ni los presupuestos del protocolo.
