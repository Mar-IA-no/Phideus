# R307 — Auditoría independiente del plan Wave 55

> **Dictamen:** `REVISE`
> **Fecha:** 2026-09-03
> **Alcance:** plan del puente conservador posterior-decisión; sin edición de código ni GO/NO-GO.

## Findings

1. **Alta: la cronología prospectiva no queda cerrada y el control de solapamiento por `pair_token` no prueba frescura.** El plan no exige congelar config, conteos, código, hashes upstream y política de fallos antes de extraer las claves, ni prohíbe descartar y regenerar un paquete tras observar inventarios o resultados. Además, `pair_token` es un HMAC del secreto de identidad y de metadatos factoriales, mientras que los draws dependen de otra clave: cambiar sólo la identidad garantiza tokens distintos aunque no demuestra nuevas realizaciones. `_sealed_key` incluso reutiliza secretos si encuentra archivos existentes. Debe imponerse freeze previo a claves, un único paquete primario, cuarentena de intentos fallidos sin redraw y comparación de compromisos de la clave de generación, no sólo tokens.

2. **Alta: el criterio sobre conjuntos no vistos probablemente carecerá de denominador.** Ola 54 identifica como ausentes del fit los índices `[0,4,8,10,12]`. El recuento directo del paquete fresco de Ola 50 encontró exactamente cero ocurrencias verdaderas de esos cinco conjuntos en `train`, `val` y `lockbox`: `0/2.304` tokens canónicos in-catalog. La Ola 55 conserva la misma ley, por lo que el criterio puede quedar indefinido. Debe fijarse un `n_min`, declarar que `n<n_min` impide esa lectura, y separar cualquier benchmark enriquecido en soporte porque ya no sería la misma ley primaria.

3. **Alta: la genealogía de checkpoints, logits y parámetros no es todavía fail-closed; el replay interno no detectaría una transformación upstream equivocada.** El plan nombra checkpoints y parámetros seleccionados, pero no fija artefacto autoritativo, hashes ni que deben usarse los modelos `primary` y no los de sensibilidad global. Esa distinción es material: Ola 54 conserva thetas primary/global diferentes. Los hashes canónicos de normalizador y checkpoints sí existen, al igual que los hashes de `selection_freeze.json` y `selection_state.npz`. Deben incorporarse literalmente al binding y exigirse un preflight que reproduzca logits históricos Wave 52 array-exact antes de tocar el benchmark fresco, como ya hacía Ola 52. Además, los checkpoints Ola 51 contienen optimizer states; el worker debería recibir copias inference-only, no los checkpoints completos.

4. **Media-alta: varios criterios vinculantes dependen sólo del signo o del punto estimado.** Sólo el contraste bridge-versus-hard en regret exige IC; la no inferioridad de accuracy/compatibilidad, la comparación con `pure_joint_full` y las ventajas frente a Platt/shuffled no exigen intervalos. Así, una diferencia microscópica puede marcar “supera” y un delta frente a controles puede pasar aunque su IC cruce cero. Deben fijarse márgenes e IC pareados por token para todos los contrastes vinculantes, aclarando que las 24 políticas se reducen antes del bootstrap.

5. **Media: la política residual y algunas métricas no están completamente determinadas.** Con `advantage >= gamma`, `gamma=0` reproduce siempre `a_p`, también en empates, contradiciendo “salvo empates”. Debe fijarse desempate por índice como en la implementación vigente, tolerancia numérica y assertions de identidad para `gamma=0/+inf`. `+inf` necesita un sentinel serializable porque el plan prohíbe infinitos JSON. También deben definirse `override_precision` como override beneficioso, neutro o perjudicial respecto del hard, y “peor regret por token” como la media entre tokens del máximo sobre 24 políticas, siguiendo la convención vigente.

## Factibilidad

La ejecución CPU es factible: Ola 49 completa usó aproximadamente `622 MiB` y `5:14`, mientras Ola 52, que además entrenó múltiples modelos, tomó unos `99 s`. El auditor no editó archivos ni emitió GO/NO-GO del proyecto.
