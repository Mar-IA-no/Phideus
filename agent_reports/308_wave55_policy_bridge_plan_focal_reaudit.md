# R308 — Reauditoría focal independiente del plan Wave 55

> **Dictamen:** `REVISE`
> **Fecha:** 2026-09-03
> **Alcance:** resolución de R307; sin edición de código ni GO/NO-GO.

## Findings

1. **Alta: el preflight histórico aún no es fail-closed.** El plan exige reproducir los NPZ de Wave 52, pero no fija sus hashes ni los inputs históricos usados para regenerarlos. Una referencia reemplazada podría validar una transformación equivocada. Los tres hashes autoritativos ya existen en el manifest de Wave 52. Deben incorporarse, junto con los bindings de observaciones históricas necesarios para el re-forward.

2. **Media-alta: `selector-sensitive` sigue sin definición ejecutable.** “Cambia el signo de un contraste primario” no especifica qué contrastes, métricas, orientación ni tratamiento de cero/tolerancia, aunque luego es condición vinculante. Implementaciones conformes podrían adjudicar distinto.

3. **Media: los IC95 no están completamente protocolizados.** Se fija unidad, reducción previa y pairing, pero faltan `n_boot`, seed y construcción del intervalo. La convención vigente usa índices PCG64 y percentiles 2,5/97,5; el plan debe congelarla o definir otra antes de generar claves.

4. **Media: las métricas de override quedan indefinidas con denominador cero.** `"hard_only"` tiene exactamente cero overrides, pero se exigen regret condicionado y `override_precision` sin definir su representación cuando no hay overrides ni el denominador exacto de las tres tasas. Esto entra en conflicto con JSON sin `NaN`.

Cronología/redraw, soporte ausente, hashes declarados, theta `primary`, copias inference-only, márgenes y dirección de los contrastes quedaron corregidos en lo sustantivo. El auditor no editó archivos.
