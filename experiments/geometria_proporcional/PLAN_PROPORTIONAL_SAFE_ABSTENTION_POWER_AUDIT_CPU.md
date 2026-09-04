# Plan CPU — potencia y transporte del threshold de abstención

**Fecha:** 2026-09-04
**Estado:** implementación, corrida oficial y replay exacto completados
**Régimen:** diagnóstico post hoc de planificación sobre estados ya abiertos
**Autoridad:** orienta el próximo experimento; no confirma seguridad, no promueve arquitectura y no decide GO/NO-GO

## Pregunta

El gate de abstención segura seleccionó identidad en todos los brazos porque
ningún threshold no trivial satisfizo la banda superior simultánea IID. Cuatro
candidatos full tuvieron medias IID negativas y límites superiores próximos a
cero. Ese resultado admite dos explicaciones que deben separarse antes de
generar otra realización: falta de potencia bajo un efecto estable, o falta de
transporte del propio efecto IID.

La auditoría usa solamente tres universos ya materializados:

1. policy selection de `proportional_graph_safe_abstention_gate_v1`;
2. adjudicación abierta de `proportional_graph_fresh_mixed_gate_v1`;
3. adjudicación abierta de `proportional_graph_safe_abstention_gate_v1`.

No reentrena, no reajusta thresholds, no materializa seeds y no abre un nuevo
test. La calibración histórica conserva su único rol: haber fijado la grilla de
thresholds y los modelos antes de estas realizaciones.

## Cálculo de potencia

Para cada `arm × family × threshold` no identidad se recuperan el efecto IID
por master, el efecto balanceado, las acciones y la cota simultánea de
selección. Si la media IID es estrictamente negativa y la política interviene
al menos una vez sobre IID, se calcula una proyección de tamaño bajo efecto y
forma de incertidumbre fijos:

```text
n_projected = n_selection * (simultaneous_q95 / -mean_iid)^2.
```

La fórmula sólo responde cuántos masters harían falta si el punto estimado y
la contracción `1/sqrt(n)` se mantuvieran. No es una garantía, un cálculo de
potencia paramétrico exacto ni una autorización para abrir esa muestra. Los
candidatos con media positiva, media nula o intervención IID nula quedan sin
proyección finita.

Dentro de cada familia se identifica, como resumen de planificación, el
candidato finito con menor `n_projected`. No se lo convierte en política
seleccionada: ya fue rechazado por el freeze confirmatorio.

## Transporte sobre adjudicaciones abiertas

Cada threshold congelado se reaplica sin ajuste a las dos adjudicaciones. Por
realización se preservan:

- delta IID, grouped y balanceado frente a identidad;
- intervalo bootstrap pareado por master;
- fracción de intervención total, IID y grouped;
- signo de los tres efectos;
- presencia de una política degenerada que actúa cero veces.

El bootstrap sólo describe los universos abiertos. No vuelve confirmatoria la
comparación ni habilita seleccionar el threshold que mejor se vea en ellos.
Los dos datasets se reportan separados; no se los concatena ni se usa uno para
corregir al otro.

## Lecturas admisibles

- una proyección moderada junto con signo IID estable en ambas adjudicaciones
  justificaría diseñar una selección fresca más grande;
- una proyección moderada con inversión del signo IID indica que aumentar `n`
  bajo un efecto fijo no resuelve por sí solo el problema;
- mejora grouped estable y signo IID inestable favorecen investigar qué
  observable público separa localización topológica de magnitud;
- thresholds que parecen seguros porque no intervienen se clasifican como
  identidad empírica, no como evidencia de routing;
- ninguna lectura se extrapola fuera de la ley sintética ni adjudica autoridad
  física, interfaz universal, arquitectura promovida o GO/NO-GO.

## Artefactos y recursos

La implementación escribirá un JSON canónico con hashes de entradas, tabla
completa de candidatos y resumen, más un manifest y un replay. Un test sintético
verificará la fórmula, la exclusión de políticas sin acción IID, la aplicación
sin targets y el bootstrap pareado. El análisis debe terminar en menos de dos
minutos y 2 GiB, con `CUDA_VISIBLE_DEVICES=''` y un único thread.

Todo trabajo GPU continúa en cola hasta que Mariano revoque explícitamente la
suspensión.
