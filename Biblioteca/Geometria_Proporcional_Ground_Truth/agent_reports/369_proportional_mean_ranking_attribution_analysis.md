# R369 — Atribución de la señal del ranking de media

**Fecha:** 2026-09-04  
**Estado:** oficial y replay exacto completados  
**Régimen:** post hoc, dos cohortes abiertas, CPU-only  
**Autoridad:** atribuye rankings congelados; no valida deployment prospectivo, arquitectura ni GO/NO-GO

## Pregunta

R367 identificó a `mu` como mejor ranker que `mu + u_topology`; R368 mostró
que usar la cola sólo como filtro sobreabstiene. R369 vuelve a los modelos de
media congelados de R360 para preguntar qué información sostiene el ranking:
dos features reduced, quince public-base, base más nueve topology, dieciséis
localizaciones topology-permuted y dieciséis modelos con target-shuffled.

El contraste primario fija `alpha=0,25` para todos. Una sensibilidad conserva
el proposer común de R368, que ya elegía ese alpha en `93,9–99,6%` de las
vistas. Cada familia ordena el mismo universo y actúa exactamente sobre
`1/2/5/10/20/40%`; el stage `ranked` compara ranking puro. Policy-selection
aplica luego `upper95 <= 0` a cada política y el stage `deployed` vuelve a
identidad cuando falla.

## Integridad

Diseño `d06307f`, implementación `18c4f2a`. Oficial y replay terminaron en
`22,145/22,153 s`, con `0,778/0,774 GiB`, un thread y CUDA invisible. No hubo
refit, vistas ni solves nuevos; `environment.json` registra
`gpu_queried: false`. El manifest byte-idéntico es:

```text
bc898c97ed3730e8719e344dc51d102a61637324eb7c3d79cc3c7e677941f994
```

Coinciden `16/16` deterministas y los `5.015` arrays canónicos son finitos. La
regresión proporcional cerró `177/177`. La reconstrucción de
`topology_augmented` igualó el `mu` preservado: diferencia máxima `0` en A y
`2,78e-17` en B. También se verificó que todos los rankers y controles usan el
mismo alpha y presupuesto antes del firewall, y que deployed sólo copia la
acción ranked o identidad.

## Ranking con alpha fijo

En las `72` celdas brazo×presupuesto×slice del stage `ranked`, topology-mean:

- mejora identidad en ambas cohortes en `58/72`, cambia signo en `12/72` y
  queda en cero en `2/72`; `33/72` intervalos son favorables en ambas;
- supera reduced-mean en `43/72`, sin ninguna celda adversa compartida;
  `13/72` intervalos son favorables en ambas y los `12/12` promedios
  brazo×slice tienen punto favorable;
- supera public-base-mean en `38/72`, es adverso en `2/72`, inestable en
  `28/72` y cero en `4/72`; `9/12` promedios son favorables, pero ningún
  intervalo se resuelve en ambas cohortes;
- supera target-shuffled-topology en `45/72`, es adverso en `1/72` e inestable
  en `26/72`; `13/72` intervalos y los `12/12` promedios son favorables;
- frente al control incremental topology-permuted es favorable en `37/72`,
  adverso en `11/72` e inestable en `24/72`. Ocho de doce promedios son
  favorables y cuatro inestables; ningún intervalo pointwise se resuelve en
  ambas cohortes.

La última comparación localiza una señal de punto, no una atribución cerrada.
Por slice, topology−permuted es favorable en `16/24` grouped y `14/24`
balanced, pero sólo `7/24` IID, donde hay `8/24` adversas y `9/24` inestables.
La ventaja crece con presupuesto: hay `2/12` celdas favorables al `1%`, `6/12`
al `5–10%`, `10/12` al `20%` y `8/12` al `40%`.

La sensibilidad con el proposer de R368 sostiene la dirección: topology menos
permuted queda favorable en `39/72`, adverso en `7/72`, inestable en `24/72` y
cero en `2/72`; ocho de doce promedios son favorables. Tampoco aparece un
intervalo primario resuelto en ambas cohortes.

## El firewall no transporta el ranking

Con alpha fijo, topology pasa `15/24` políticas en A y `12/24` en B. Los
controles permutados pasan `287/384` y `254/384` réplicas respectivamente. Esta
no linealidad cambia el estimando: deployed topology−permuted queda favorable
en `11/72`, adverso en `20/72`, inestable en `29/72` y cero en `12/72`; hay
seis intervalos adversos compartidos y ninguno favorable. En los promedios no
queda ningún punto favorable, frente a tres adversos y nueve inestables.

Con el proposer de R368, topology pasa `13/24` en ambas cohortes, pero deployed
topology−permuted sólo queda `19/72` favorable, `19/72` adverso, `20/72`
inestable y `14/72` cero. El ranking transporta mejor que su política de
selección. Promediar controles después de seleccionar cada réplica describe el
control congelado, pero no transfiere su estabilidad al único ranker topology.

## Lectura

**Observación.** La media topology supera de manera amplia a reduced y
target-shuffled, mejora a public-base por punto y conserva una ventaja más
estrecha contra localización permutada, sobre todo grouped/balanceado. El
firewall por política revierte la comparación incremental.

**Hipótesis.** Las features topology contienen información útil para ordenar
beneficio más allá de escala y target espurio. Parte de esa información también
está en la base pública o sobrevive a permutar localización; el incremento
topológico puro es pequeño respecto de su variabilidad. El cuello de policy
selection es distinto del cuello de ranking.

**Inferencia acotada.** La primitive recuperable no es la cola: es un ranker de
media con señal topológica parcial. R369 contradice una lectura demasiado
fuerte de R368 según la cual topology carecería de todo valor arquitectónico;
carece de valor incremental como cola/filtro, pero aporta al estimador de media.
La ausencia de intervalos compartidos frente al control permutado y el fracaso
del deployment impiden promoverlo o justificar todavía un freeze.

## Próximo paso

Antes de ajustar otra cabeza conviene auditar potencia y selección del
estimando `topology_mean - topology_permuted_mean`: cuantificar qué celdas
transportan, qué tamaño exigirían bajo los efectos observados y cuánto de la
reversión deployed proviene del firewall no lineal. Ese diagnóstico puede
reusar los efectos y bootstraps R369 por CPU. Sólo si deja un contraste
realizable corresponde diseñar una confirmación fresca; cualquier opción GPU
permanece en cola.

Artefactos: plan
`experiments/geometria_proporcional/PLAN_PROPORTIONAL_MEAN_RANKING_ATTRIBUTION_CPU.md`,
runner
`experiments/geometria_proporcional/run_proportional_graph_mean_ranking_attribution.py`,
oficial
`data/geometria_proporcional/proportional_graph_mean_ranking_attribution_v1/`
y replay con sufijo `_replay`.
