# Lectura critica de E3-P2: del baseline unico al baseline dual

Este documento ya no debe leerse como auditoria de una corrida aislada. `E3-P2` paso por varias correcciones metodologicas reales: seleccion de checkpoint por `val_S`, atlas OOD con cobertura completa, `render-OOD` explicito y una segunda ola de encoders orientados a invariancia. El resultado final del frente ya no es “quien gano `P2`”, sino otra cosa: **una separacion limpia entre baseline general de retrieval y baseline alternativo de invariancia audio-side**.

## Observaciones

### 1. `P2-flat` sigue siendo el mejor baseline general

En `IID`, [p2_baseline_seed42/final_results.json](../../../data/escalon3/p2_baseline_seed42/final_results.json) mantiene la mejor referencia general del frente:

- `S = 0.583`
- `silhouette_combined = 0.960`
- `render-OOD thick = 0.506`

La conclusion valida de ese bloque es sencilla: el espacio latente plano sigue siendo la mejor referencia canónica para retrieval general y para robustez visual.

### 2. `P2-cqtshift` abre una señal nueva del lado audio

En [p2_cqtshift_seed42/final_results.json](../../../data/escalon3/p2_cqtshift_seed42/final_results.json), el brazo `cqtshift` no gana en `IID`, pero cambia mucho la lectura OOD desde audio:

- `scale-OOD a2i = 0.476`
- `equiv-OOD a2i = 0.458`

Eso ya no es un delta marginal respecto del baseline canónico. Es una reorganizacion real del lado audio cuando el encoder recibe un sesgo explicitamente ratio-aware.

### 3. El costo de esa mejora es visible

`cqtshift` no viene gratis:

- `IID S = 0.515`
- `render-OOD thick = 0.344`

O sea: gana invariancia audio-side, pero pierde retrieval general y robustez visual frente al baseline plano.

## Hipotesis

La hipotesis de trabajo mas razonable es esta:

- `P2-flat` conserva mejor informacion de identidad de escena y de matching cross-modal general;
- `P2-cqtshift` fuerza al encoder audio a organizarse mas por relaciones de ratio que por detalle absoluto o identidad fina;
- por eso el brazo alternativo mejora mucho `scale-OOD` y `equivalence-OOD` desde audio, pero paga esa mejora con una degradacion clara en `IID` y `render-OOD`.

Esta hipotesis es fuerte, pero sigue siendo una hipotesis. No debe venderse como prueba final sobre la naturaleza del espacio latente hasta que `P4` compare probes sobre ambos regímenes.

## Inferencias

De esas observaciones no sale “dos ganadores”. Sale algo mas util:

1. `P2-flat` debe seguir siendo el baseline canónico `L0`.
2. `P2-cqtshift` merece quedar fijado como baseline alternativo serio, no como experimento descartado.
3. `E3-P4` ya no conviene correrse sobre un unico embedding family, porque ahora el frente tiene una tension experimental real:
   - retrieval general versus
   - invariancia audio-side orientada por ratio.

## Decision operativa

La decision correcta para la documentación y para el roadmap es esta:

- `L0-Flat Canonical` = `P2-flat`
- `L0-Shift Ratio-Aware` = `P2-cqtshift`

Regla de uso:

- `P4` se decide primero sobre `L0-Flat Canonical`.
- Luego se replica exactamente sobre `L0-Shift Ratio-Aware`.
- Si una senal aparece solo en `cqtshift`, la lectura ya no sera “phi funciona en general”, sino “phi interactua con un encoder mas ratio-aware”.

## Lo que no conviene hacer

- no promediar ambos baselines;
- no mezclar sus resultados en una unica tabla final como si midieran la misma virtud;
- no reemplazar el baseline canonico solo porque `cqtshift` mejora una familia de OOD;
- no descartar `cqtshift` solo porque pierde `IID`.

## Lo que paso despues

La línea siguiente ya fue corrida y valida, en lo esencial, esta lectura crítica:

1. `P4` efectivamente tuvo que correrse sobre ambos `L0` para evitar una lectura falsa de “ganador único”.
2. `P5/P6` terminaron confirmando que la tensión `IID general` versus `OOD audio-side` era real y no un artefacto narrativo.
3. `P2-flat` siguió como baseline canónico general.
4. `P5-cqtshift` terminó emergiendo como mejor brazo geométrico/OOD.

La decisión dual de `P2`, por lo tanto, ya no es solo una prudencia metodológica retrospectiva. Pasó a ser una pieza validada por la línea completa `P2 -> P4 -> P5 -> P6`.
