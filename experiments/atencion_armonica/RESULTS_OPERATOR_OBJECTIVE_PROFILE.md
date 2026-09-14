# Perfil del diagnóstico operación–objetivo

2026-09-14. El inventario y el perfil fijado se ejecutaron en CPU, después de
cerrar la auditoría de implementación. El barrido completo **no se inició**:
la proyección conservadora supera el presupuesto. Esto informa costo de
implementación, no alineación geométrica ni ventaja de una arquitectura.

## Observaciones

Se autenticaron los 2048 recibos: 893130717 bytes comprimidos y 9280977105
decodificados declarados. El inventario no descomprimió factores; el perfil
extrajo únicamente la escena 0 de cada escenario y cotejó sus targets,
canales y decisiones con el contraste cerrado.

| Escenario, escena 0 | Candidatos | Bytes decodificados | Extracción (s) | Diagnóstico y bundle (s) | Preparación del split (s) |
|---|---:|---:|---:|---:|---:|
| IID | 31 | 4891037 | 0.142675 | 0.166153 | 1.577279 |
| Mayor beta | 65 | 7085256 | 0.187315 | 0.170105 | 1.477919 |
| Polifonía | 65 | 1824750 | 0.057431 | 0.188210 | 1.652850 |
| Familia deformada | 48 | 5191490 | 0.128031 | 0.316313 | 1.445656 |

El inventario consumió 6.175930 s y el perfil 7.639820 s: 13.815751 s
acumulados. El pico RSS observado fue 461774848 bytes. Los cuatro bundles
quedan preservados para análisis y replay sin repetir la extracción.

## Proyección y siguiente acción

La fórmula del [protocolo](PROTOCOL_OPERATOR_OBJECTIVE_ALIGNMENT.md) toma el
máximo costo por byte para extracción y por par/celda/esquema para diagnóstico;
proyecta 2048 escenas y hasta 82 candidatos, añade replay, margen ×2 y 600 s
de auditoría. Produce 10943.721359 s frente a 1800 s disponibles como límite
total, y 1388136778 bytes nuevos frente a 4 GiB. La guarda temporal no pasa;
la de almacenamiento sí. No son tres horas medidas de ejecución: son una
estimación conservadora, que tampoco autoriza suprimir el margen.

Corresponde perfilar cálculo y serialización sobre los cuatro compactos ya
guardados, revisar el trabajo repetido y preservar las fórmulas y los bytes
científicos. No se reduce el roster, se amplía el presupuesto en silencio ni
se inicia una corrida CPU larga. El goal continúa incompleto.

## Recibos fuente

Raíz de artefactos: `data/atencion_armonica/operator_objective_alignment_v1/`.

- `inventory/inventory.json`: SHA256 `0731c42734589b77a582f35794a50d6377f34f7541f8d13af71ae4da320018a5`.
- `profile.complete.json`: SHA256 `c65e3759a1397f941614964c3c4888812debd3514018efcddb03ae8cb7e92553`.
- `attempts/0000/finish.json`: SHA256 `4d32569070dcbf68284da1eb3b38815276c5d3c500464daaac2740036a39b6fc`.
- `attempts/0001/finish.json`: SHA256 `70776d81fc280397c1af3d03089fa6dd6c39a2c85c219dfa0999b968eda24818`.

Estos recibos distinguen perfil completo de roster completo. No hay marcador
global `COMPLETE` ni `REPLAYED`, promoción arquitectónica o GO/NO-GO.
