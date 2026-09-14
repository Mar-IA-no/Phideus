# Carga observada del diagnóstico completo

2026-09-14. El inventario autenticó las 2048 escenas y la concordancia de
offsets y particiones entre los tres checkpoints. No extrajo factores,
reconstruyó etiquetas, calculó el diagnóstico ni usó GPU. Su implementación
pasó revisión independiente y siete pruebas específicas.

| Escenario | Candidatos | Máximo por escena | Escenas vacías | Pares de candidatos | Pares con piso C=31 |
|---|---:|---:|---:|---:|---:|
| IID | 27891 | 68 | 7 | 802937 | 811904 |
| Mayor beta | 27981 | 69 | 15 | 820509 | 833343 |
| Polifonía | 29620 | 69 | 61 | 957987 | 986352 |
| Familia deformada | 27675 | 70 | 28 | 821862 | 840034 |
| Total | 113167 | 70 | 111 | 3403295 | 3471633 |

Cada escenario conserva sus 512 escenas, incluidas las vacías. Suponer 82
candidatos en cada escena, como hacía la proyección inicial, representa
6801408 pares: aproximadamente el doble de la carga observada. El piso de 31
se fijó antes del inventario y sólo describe una carga conservadora para
escenas pequeñas; no constituye una nueva fórmula de admisión. Tampoco el
conteo de pares explica por sí solo lectura, serialización, agregación o
controles sobre el árbol creciente de archivos.

El intento consumió 7.419831 s, con pico RSS de 490070016 bytes. El consumo
acumulado es 23.820175 s y el subtotal de perfiles, 17.644244 s. Estos tiempos
corresponden a los operadores registrados, no al tiempo humano o de agentes
empleado en diseñarlos y auditarlos.

Sigue una enmienda explícita de admisión y ejecución que cuente costos fijos,
carga variable, validación y replay. Si se revisa el límite operativo,
deberá conservar el consumo anterior y pasar auditoría; no se modificará en
silencio el contrato congelado. Este inventario no habilita por sí mismo el
barrido ni satisface el cierre científico del goal.

## Evidencia

Raíz de artefactos: `data/atencion_armonica/operator_objective_alignment_v1/`.
El estado `WORKLOAD_COMPLETE` sólo tiene autoridad mediante el sello de un
intento `COMPLETE`; no equivale al cierre del diagnóstico.

- `attempts/0005/workload.json`: SHA256 `6e8311af85610faa45d81cd60d5a54e4cd87455c502b206a48c6e993ac85fe94`.
- `attempts/0005/finish.json`: SHA256 `d8187418c5111a157cea1c6cd3f6b75aa0a4382ec95e715188ff6acd48819a27`.
- `attempts/0005/runtime_revision.json`: SHA256 `8c700d3c30254d929feff4971865a5468873f83de841c71de59a6da67859f27f`.
- [Plan del inventario](PLAN_OPERATOR_OBJECTIVE_WORKLOAD.md),
  [operador](inventory_diagnostic_workload.py) y
  [pruebas](test_inventory_diagnostic_workload.py).
