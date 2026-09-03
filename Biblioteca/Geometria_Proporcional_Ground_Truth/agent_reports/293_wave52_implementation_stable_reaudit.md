# Reauditoria independiente de implementacion - Ola 52

> Estado: `REVISE`
> Fecha: 2026-09-03
> Instancia independiente: Huygens (`01a06637-f6d4-70a1-b289-95967532332b`)
> Alcance: snapshot estable posterior a los fixes del informe 291.

## Findings verbatim

1. **HIGH: `tau` se selecciona sobre una representacion distinta de la utilizada por el reader primario.** La evaluacion promedia logits por token y luego entre seeds, pero `tau` se calibra con logits por fixture/vista y despues se aplica al ensemble por token. Esto puede cambiar conjuntos, fallbacks y metricas primarias.

2. **HIGH: el finding de artefactos/replay quedo parcialmente abierto.** Los shuffles se generan durante entrenamiento pero sus mappings no se persisten. El replay reconstruye logits/scores desde checkpoints, pero las metricas se reproducen usando las acciones ya guardadas, sin comprobar checkpoint -> acciones -> metricas de extremo a extremo. El contrato de completitud solo verifica conteos y flags.

3. **MEDIUM: el reporting per-seed/per-policy sigue incompleto.** Las filas por politica no contienen seed; los resultados por seed se reducen a agregados. Ademas, los IC omiten `worst_restricted_regret`.

## Resolucion del audit 291

| Finding | Estado |
|---|---|
| Scope/targets vacios | PASS |
| Unidad generativa y bootstrap | PASS |
| Mismo ensemble entre readers | PASS |
| Contrafactual por checkpoint | PASS |
| Artefactos y replay | REVISE |
| Worst regret | PASS |
| Metricas/costos/per-seed/per-policy | REVISE |
| Barrera anti-leakage | PASS |

## Veredicto

`REVISE`
