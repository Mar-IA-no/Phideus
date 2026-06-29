# Atención Armónica

> Frente nuevo en incubación local que prueba si una representación explícita de pares con actualización triangular puede capturar estructura armónica global de una mezcla polifónica mejor que un backbone token-only con features armónicas inyectadas.

## Convención de nombre

El **frente** sigue llamándose **Atención Armónica**. La **arquitectura** principal que el frente pone a prueba queda fijada con dos nombres complementarios:

- **nombre técnico**: `Harmonic Pairformer`
- **nombre explicativo**: `atención por geometría armónica`

La precisión importa porque evita leer mal el resultado. El salto de `Fase 0` no vino de “atender sobre ratios” en abstracto, sino de sostener una **representación explícita de pares** y propagar consistencia sobre ella. Por eso también queda descartado encuadrarla como “ratio-based attention transformer”: ese nombre sobrerrepresenta el ingrediente menor y borra la pieza que realmente produjo el salto, que fue el plano relacional.

## Estado actual: Fase 0, 0.5 y 0.6 cerradas, resultado dual y GO acotado (2026-06-29)

Este frente **todavía no debe leerse como frente canónico del programa**, pero ya no está en estado de training abierto ni de auditoría pendiente sobre `τ`. La `Fase 0` cerró sobre el pool sintético `v2.1`: el sweep pasó, el `final_pool` quedó congelado con gate `PASS`, el smoke supervisado confirmó aprendibilidad sin saturación y el training decisivo completó `54/54` corridas. La lectura resultante fue dual: el pair-state es el salto grande, y el `triangle` aporta específicamente como sesgo de generalización a polifonía nueva.

La razón de esa cautela es metodológica. La pregunta del frente no es si una red cualquiera puede agrupar parciales. La pregunta es más precisa: **si la maquinaria pair-state + transitividad + triangle update aporta algo por encima de un baseline con las mismas features armónicas cuando la evidencia per-par es genuinamente ambigua**. Si el dataset deja que una feature cerrada resuelva la tarea sola, el contraste `B vs A-rich` queda anulado por construcción.

La formulación geométrica vigente también quedó más precisa. Atención Armónica no presupone todavía una geometría métrica cerrada de la armonía, al estilo de un espacio 3D. Lo que prueba es una **geometría relacional**: los picos son nodos, las relaciones `same-source` son aristas aprendidas, y la estructura válida es una partición global en fuentes generativas armónicas. El `triangle update` opera sobre esa matriz de pares para propagar consistencia de pertenencia, no para imponer una identidad trivial en `log f`.

La `Fase 0.5` agregó una precisión decisiva sobre esa lectura. El cuello de `B` en `OOD-poly` no estaba en una mala calibración de `τ`: el post-audit mostró `gap_dist≈0`, es decir, ni siquiera con `oracle_tau_global_test` `connected-components` recupera bien la partición. El problema real está en la regla de clustering. La representación de `B` sí mejora fuera de distribución, y bajo `agglo_true_k` pasa a ser la mejor en la celda más dura; lo que falla es la lectura por conectividad de esa geometría.

La `Fase 0.6` ya cerró el siguiente paso lógico. La pregunta dejó de ser “si existe una representación mejor que el clusterer no sabe leer” y pasó a ser “qué familia deployable sí sabe leerla”. La respuesta quedó acotada pero positiva: `connected-components` sigue siendo demasiado frágil incluso con poda de puentes, pero clusterers globales con `k` estimado (`spectral_eigengap` y `agglo_estimated_k`) ya recuperan una ventaja real de `B` sobre `B-local` en `OOD-poly`. El caveat que queda no es `τ`; es la subestimación sistemática de `k`, que todavía deja un gap visible respecto de la referencia privilegiada con `k` conocido.

## Qué pasó hasta acá

### v1: parciales exactos armónicos, artefacto de ratios

La primera formulación de `Fase 0` trabajaba con parciales exactos y armónicos enteros (`beta=0`). La auditoría del pool encontró enseguida que `common_f0_residual` y `ratio_residual` separaban `same-source` con `AUC≈1.0`. Eso volvía trivial la tarea para `A-rich`: el baseline decisivo recibía en sus pair features una respuesta casi cerrada.

### v2: inarmonicidad, pero leak por amplitud

La segunda formulación introdujo `beta>0` per-source y dropout de parciales. Eso rompió el oráculo de ratios enteros, pero apareció otro canal cerrado: la envolvente determinística `amp = 1/n` filtraba el índice armónico. Para `same-source`, `log_amp_diff` quedaba casi igual a `dlogf`, y un `PairMLP` chico seguía separando la tarea casi al techo.

### v2.1: inarmonicidad + amplitud randomizada + gate fuerte

La versión vigente rompe ambos canales cerrados a la vez:

- `beta>0` per-source para desarmar la armonicidad exacta.
- amplitud randomizada per-source `amp_n = (1 / n^alpha) * exp(epsilon_n)` para romper el leak `log_amp_diff ≈ dlogf`.
- dropout de parciales con `min_partials=4` y restauración determinística por amplitud.
- gate obligatorio de **feature-triviality** antes de cualquier training GPU.

### Sweep y final_pool: la etapa de dataset ya quedó resuelta

El sweep `v2.1` sobre `calibration_pool` encontró `16/16` combos elegibles bajo la regla congelada. La combo elegida por desempate determinístico fue:

- `beta-center = 1e-3`
- `alpha-range = [0.5, 1.5]`
- `sigma_amp = 0.5`
- `p_drop = 0.3`

La lectura útil de esa calibración fue doble:

- **headroom real**: `PairMLP` quedó en la banda `0.79-0.83`, lejos del techo trivial;
- **solvabilidad upper-bound**: `oracle_priv = 1.0` en todas las celdas decisivas.

El caveat importante quedó explícito desde el sweep: `oracle_unpriv_f0only` colapsa a `~0.07`. Eso no bloquea el frente, pero sí obliga a decir con precisión qué demostró el gate. El dataset ya no es feature-trivial; no quedó probado todavía que cualquier aproximación simple pueda recuperar la estructura sin supervisión.

Después de congelar la combo, el `final_pool` se regeneró con seed distinta y volvió a pasar el gate. Eso clausura la discusión “¿el dataset deja headroom real?” para esta fase.

## Tesis y contraste

La tesis fuerte del frente no es “inyectar armonía en un backbone genérico”, sino probar si `Harmonic Pairformer`, una arquitectura con estado de pares y actualización triangular, puede operar dentro de una geometría armónica donde la consistencia global importa. El contraste decisivo sigue siendo el mismo:

- `A-naive`: token attention + bias relativo, sin pair features explícitas.
- `A-rich`: mismo backbone token-only, pero con las mismas pair features que `B`.
- `B`: Harmonic Pairformer completo.
- `B-local`: control param-matched que aísla la suma sobre `k`.
- `B-minus`: ablación sin triangle.
- `B-shuffle`: control negativo parcial.

Si `A-rich` ya resuelve la tarea al techo, el frente no puede contestar su propia pregunta. Por eso el gate del dataset es parte constitutiva del experimento, no un extra operativo.

## Protocolo vigente

Ningún pool pasa a GPU sin cumplir dos condiciones sobre las celdas decisivas `poly2/3 × easy/hard`:

1. **Headroom real**: single features, `LogReg` y `PairMLP` sobre TODO lo que recibe `A-rich` deben quedar por debajo del umbral de feature-triviality.
2. **Solvabilidad real**: un `oracle_privileged_upper_bound` debe mostrar que la estructura todavía es recuperable globalmente.

La calibración actual se hace sobre un `calibration_pool` separado del `final_pool`. En `v2.1`, ese paso ya quedó cumplido: el `final_pool` vigente pasó el gate y es el único pool habilitado para el training de `Fase 0`.

## Resultado de Fase 0

### Diagnostic smoke de A-rich

Antes de comprometer GPU, el frente corrió el smoke que faltaba: `A-rich` sobre la combo elegida, en CPU y con protocolo acotado. El resultado importante no fue una F1 alta, sino otra cosa:

- `A-rich` aprende por encima de chance en todas las celdas decisivas;
- no satura;
- `poly3_hard` sigue siendo aprendible.

Eso confirmó que el problema ya no es ni trivial ni imposible desde el punto de vista supervisado. Era la última condición para habilitar el training real.

### Training decisivo cerrado

El run completo de `Fase 0` cerró sobre el `final_pool`, con los 6 modelos, `3` seeds y `3` runs (`ID`, `OOD-poly`, `OOD-regime`).

La lectura ya no depende de un parcial:

- `B-minus ≫ A-rich`: representar pares explícitamente es el salto grande.
- `B ≫ B-shuffle`: la estructura del triángulo importa; no es solo capacidad.
- `B vs B-local`: el efecto del `triangle` es split-dependiente. En `IID` y `OOD-regime`, `B-local` iguala o supera levemente a `B`; en `OOD-poly`, `B` supera a `B-local` en `AUC/AP` threshold-free (`ΔAUC +0.053`, `ΔAP +0.093`, CI excluye 0).

El caveat central quedó primero formulado como un problema de `ARI@τ_val`, pero la `Fase 0.5` corrigió esa interpretación. No era un problema de transferencia de `τ`. Era un problema de `connected-components`: con `oracle_tau_global_test`, `B` no mejora; con `agglo_true_k`, sí. `Fase 0.6` agregó la pieza que faltaba: la representación de `B` ya no necesita un `k` verdadero para volverse útil, pero sí un clusterer global. Bajo `spectral` y `agglo` deployables, `B` pasa a ganar en `OOD-poly`; bajo `cc_bridge_prune`, no.

Lectura local: **GO acotado** hacia una siguiente fase que ya no debe centrarse en más tuning de `τ`, porque ese frente ya quedó cerrado. Lo que sigue es decidir cómo cerrar el gap de partición que todavía deja la subestimación de `k`: un Stage B con cabeza explícita de `k/partición`, o el salto a CQT/picos detectados una vez fijada esa lectura global. No se declara que el triángulo gana siempre; se declara algo más preciso: su representación generaliza mejor en `OOD-poly`, y ya puede extraerse con una familia deployable de clusterers globales, aunque todavía no de manera plenamente resuelta.

## Documentación local de incubación

- Roadmap del frente: `./ROADMAP_ATENCION_ARMONICA.md`
- Explicación conceptual de la arquitectura: `./Explicacion_arq_RNA_codex.md`
- Explicación conceptual de `Fase 0.5`: `./Explicacion_fase_0_5_calibracion_codex.md`
- Explicación conceptual de `Fase 0.6`: `./Explicacion_fase_0_6_clusterer_deployable_codex.md`
- Plan vigente de `Fase 0 v2.1`: `./PLAN_FASE_0_v2_1.md`
- Plan de post-audit `Fase 0.5`: `./PLAN_FASE_0_5_CALIBRACION.md`
- Plan anterior preservado por trazabilidad: `./PLAN_FASE_0_v1_superseded.md`

## Regla de propagación

Este frente ya puede figurar en `Documents/00_TRONCAL/` como incubación con `Fase 0` cerrada y resultado interpretable, pero no como tesis canónica sin reservas. La propagación correcta debe preservar tres distinciones:

- pair-state como cimiento fuerte;
- `triangle` como sesgo positivo en `OOD-poly`, no como ganador universal;
- cuello del sistema ya no en `τ`, sino en cómo cerrar la estimación de `k` y la partición global a partir de la matriz de pares.

La capa documental correcta sigue siendo esta carpeta local para el detalle técnico y el troncal para la lectura sintética.
