# Plan — Atención Armónica Fase 0.5

> Estado: **ejecutada**. Este documento preserva el diseño de la auditoría tal como se corrió. La lectura actual del resultado ya no vive acá sino en `README.md`, `ROADMAP_ATENCION_ARMONICA.md` y `Explicacion_fase_0_5_calibracion_codex.md`.

> Post-audit de calibración y decisión de clustering. No reemplaza `Fase 0 v2.1`: la complementa. La `Fase 0` ya cerró el resultado de representación; `Fase 0.5` pregunta si ese ranking de pares puede convertirse en agrupamiento estable sin usar información privilegiada de test.

## Resultado que dejó

La auditoría terminó corrigiendo su propia hipótesis inicial. El cuello no estaba principalmente en la transferencia de `τ`: para `B`, `oracle_tau_global_test` no mejora sobre `baseline` en `OOD-poly` (`gap_dist≈0`). El problema real quedó localizado en la regla `connected-components`, que no sabe extraer bien la partición desde una matriz de pares donde `B` sí mejora threshold-free y también bajo `agglo_true_k`.

Eso vuelve este documento un registro de diseño útil, pero ya no la mejor puerta de entrada conceptual al hallazgo.

## Contexto

`Fase 0` mostró un resultado dual. El pair-state explícito fue el salto principal, y el `triangle` aportó en `OOD-poly` bajo métricas threshold-free (`AUC/AP`) frente a `B-local`. El caveat apareció al convertir esa matriz de scores en clusters: `ARI@τ_val` puede colapsar aunque el ranking de pares mejore. Eso separa dos problemas distintos:

- representación: ordenar correctamente qué pares deberían ir juntos;
- decisión de clustering: transformar esa matriz en una partición robusta.

`Fase 0.5` se concentra solo en el segundo problema. No cambia dataset, modelos, arquitectura ni gate. Audita la decisión de operación.

## Motivo del re-run

El run original guardó `test_pairs/` y `test_ari/`, pero no guardó logits de validación ni checkpoints. Eso alcanza para parte del análisis de test, pero no para ajustar reglas deployables en `val`: `Platt`, `isotonic`, `τ_val_ari`, `τ_val_f1` y la selección de regla necesitan predicciones sobre validación. Sin esos artefactos, la auditoría completa requiere re-correr los `54` trainings de `Fase 0` con guardado ampliado.

El re-run no busca nuevos resultados de arquitectura. Debe reproducir el run anterior dentro de tolerancia y agregar artefactos reutilizables.

## Artefactos que debe guardar

Por cada `(run, model, seed)`:

- `val_mats/`: matrices por mezcla con `logit_mat`, `token_mask`, `pair_valid`, `target_mat`, `polyphony`, `regime`, `mixture_id`, `n_peaks`.
- `test_mats/`: mismas matrices para test.
- `checkpoints/`: `last_epoch` siempre; opcionalmente `best_by_val_auc` y `best_by_val_ap`.

No se usa `best_by_val_ari` como checkpoint primario, porque `ARI@τ_val` es justamente el criterio que se está auditando.

La reproducción esperada es determinista. Si aparece ruido numérico leve, se acepta una tolerancia de `1e-4` en métricas por celda; diferencias mayores frenan el análisis.

## Sistema primario de evaluación

La unidad primaria es el ensemble de `3` seeds:

```text
logits crudos alineados por mixture_id/par
        ↓
promedio de logits
        ↓
calibrador único entrenado en val
        ↓
clustering
```

Esto mantiene continuidad con `REPORTE_0`. La evaluación por seed queda como sensibilidad secundaria.

## Calibradores

Los calibradores transforman logits de pares en probabilidades:

- `none`: `sigmoid(raw_logit)`;
- `platt`: `LogisticRegression` 1-D sobre logits de val;
- `isotonic`: `IsotonicRegression` sobre logits de val.

La receta es `fit pair-pooled`: entrenar sobre todos los pares válidos de `val`. La incertidumbre se reporta por bootstrap de mezclas, no por bootstrap de pares, para respetar la dependencia intra-mezcla.

Cada calibrador elige su propio `τ_val`. Los umbrales no se comparan numéricamente entre calibradores.

## Clusterer deployable

El clusterer central sigue siendo connected-components con umbral `τ`, elegido solo en validación:

- `cc@τ_val_ari`: τ que maximiza ARI en `val`; baseline predeclarado, reproduce el caveat de `REPORTE_0`.
- `cc@τ_val_f1`: τ que maximiza F1 pairwise en `val`; lectura secundaria.

La regla `best_val_deployable` se selecciona por `(run, model)` sobre todo `val poly≥2`, nunca por celda de test. Después queda congelada para todas las celdas de test de ese run.

El reporte debe distinguir:

- `baseline_deployable`: `ensemble + none + cc@τ_val_ari`;
- `best_val_deployable`: mejor combinación `calibrador × τ-objetivo` según `val-ARI`.

## Diagnósticos privilegiados

Los diagnósticos privilegiados no son métricas principales. Sirven para medir brechas:

- `oracle_tau_global_test`: un τ por `(celda, modelo)` elegido en test; upper-bound de calibración por distribución dentro de la familia CC.
- `oracle_tau_per_mixture_test`: mejor τ por mezcla; upper-bound extremo.
- `agglo_true_k`: agglomerative-average con `k` verdadero, usando calibrador fijo `none`.

Los gaps se reportan separados:

- `gap_dist = oracle_tau_global_test - best_deployable`;
- `gap_extreme = oracle_tau_per_mixture_test - best_deployable`;
- `gap_k = agglo_true_k - best_deployable`.

No se colapsan en un único número, porque cada oracle responde una pregunta distinta.

## Fuera del veredicto

`cc_robust` y `spectral_eigengap` quedan fuera del veredicto salvo que se congele una receta completa antes de mirar resultados. Si se exploran, deben marcarse como exploratorios.

## Pregunta central

Bajo una regla deployable seleccionada solo con `val`:

```text
¿B ≥ B-local en ARI OOD-poly?
¿B ≫ B-shuffle en ARI OOD-poly?
```

Si la respuesta es sí, la ventaja representacional del `triangle` se convierte en sistema de agrupamiento. Si la respuesta es no, el frente conserva el resultado como ranker relacional, pero deja la partición estable como cuello abierto.

## Verificación

1. `tau_val` del audit reproduce `ARI@τ_val` de `REPORTE_0` dentro de tolerancia.
2. El cross-check por mezcla valida orden y reconstrucción: `target` reconstruido coincide con flat guardado y el número de pares válidos coincide.
3. Los oracles cumplen las desigualdades solo dentro de su familia comparable: `oracle_per_mixture ≥ oracle_global ≥ cc@τ_val` para el mismo calibrador y CC.
4. Test nunca participa en selección de regla deployable.
5. `agglo_true_k` usa `k` verdadero solo para clusterizar, nunca para elegir una regla deployable.

## Lotes

Lote A, código AA:

- `experiments/atencion_armonica/1_train_grouping.py`
- `experiments/atencion_armonica/2_calibration_audit.py`

Lote B, política documental transversal:

- `AGENTS.md` o documento operativo troncal como hogar público de la directiva de preservación de artefactos.
- `CLAUDE.md` solo como reflejo privado de Claude, editado por Claude o por el usuario.

## Siguiente fase

Si `Fase 0.5` muestra que la calibración convierte el ranking en clusters robustos, el paso siguiente es `Fase 1a`: picos detectados por CQT sobre mezclas renderizadas, con ground truth sintético todavía exacto. Audio real/stems queda para una fase posterior.
