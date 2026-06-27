# Resultados de E3-P2

Fecha de consolidacion: `2026-03-21`

## Decision vigente

`E3-P2` ya no debe resumirse como “un solo baseline”. El frente dejo dos referencias `L0` con roles distintos:

- `P2-flat` queda como baseline canonico para retrieval general.
- `P2-cqtshift` queda como baseline alternativo ratio-aware para el lado audio.

## Tabla corta

| Brazo | Rol | IID `S` | Silhouette | `scale-OOD a2i` | `equiv-OOD a2i` | `render-OOD thick` |
|------|-----|---------|------------|-----------------|-----------------|--------------------|
| `P2-flat` | baseline canonico `L0` | `0.583` | `0.960` | `0.096` | `0.240` | `0.506` |
| `P2-cqtshift` | baseline alternativo ratio-aware | `0.515` | `1.000` | `0.476` | `0.458` | `0.344` |

## Lectura

- `P2-flat` sigue siendo la mejor referencia general del frente porque conserva el mejor retrieval `IID` y la mejor robustez visual.
- `P2-cqtshift` no reemplaza a `P2-flat`, pero abre una señal nueva y fuerte sobre invariancia de ratio del lado audio.
- La diferencia no es menor ni cosmetica: `cqtshift` sacrifica parte de la identidad de escena y de la robustez `render-OOD` para ganar mucha mas consistencia en `scale-OOD` y `equivalence-OOD` cuando la consulta sale desde audio.

## Decision operativa

- `P2-flat` queda fijado como `L0-Flat Canonical`.
- `P2-cqtshift` queda fijado como `L0-Shift Ratio-Aware`.
- `E3-P4` debe correrse primero sobre `L0-Flat Canonical` y luego replicarse sobre `L0-Shift Ratio-Aware`.

## Validacion posterior de esta decision

La línea siguiente ya fue corrida y, al menos en esta receta, confirmó que esta separación era metodológicamente correcta:

- `P2-flat` siguió siendo el mejor baseline general de `IID`;
- `P5-cqtshift` terminó emergiendo como mejor brazo geométrico/OOD;
- y `P6` no reemplazó esa combinación.

Eso refuerza la lectura central de `P2`: no había que forzar un ganador único, sino congelar dos referencias con roles distintos.

## Lo que no corresponde decir

- no corresponde decir que `cqtshift` “gano” `P2` en general;
- no corresponde decir que el baseline flat ya no sirve;
- no corresponde promediar ambos y hablar de un unico numero de `P2`.

La lectura correcta es mas precisa: Escalon 3 ahora tiene un baseline canonico para retrieval general y un baseline alternativo para stress-testear la hipotesis de invariancia armónica del lado audio.
