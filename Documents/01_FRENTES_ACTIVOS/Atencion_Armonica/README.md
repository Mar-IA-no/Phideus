# Atención Armónica

> Frente nuevo en incubación local que prueba si una representación explícita de pares con actualización triangular puede capturar estructura armónica global de una mezcla polifónica mejor que un backbone token-only con features armónicas inyectadas.

## Estado actual: incubación metodológica local, sin propagación al troncal (2026-06-27)

Este frente **todavía no debe leerse como frente canónico del programa**. Su estado real hoy es otro: código de `Fase 0` escrito y auditado por capas, dos artefactos fatales del generador ya detectados antes de tocar GPU, y una tercera iteración del diseño (`v2.1`) preparada para calibración de dataset con gate explícito de feature-triviality.

La razón de esa cautela es metodológica. La pregunta del frente no es si una red cualquiera puede agrupar parciales. La pregunta es más precisa: **si la maquinaria pair-state + transitividad + triangle update aporta algo por encima de un baseline con las mismas features armónicas cuando la evidencia per-par es genuinamente ambigua**. Si el dataset deja que una feature cerrada resuelva la tarea sola, el contraste `B vs A-rich` queda anulado por construcción.

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

## Tesis y contraste

La tesis fuerte del frente no es “inyectar armonía en un backbone genérico”, sino probar si una arquitectura con estado de pares y actualización triangular puede operar dentro de una geometría armónica donde la consistencia global importa. El contraste decisivo sigue siendo el mismo:

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

La calibración actual se hace sobre un `calibration_pool` separado del `final_pool`. Si ninguna combo del sweep cae en la ventana de headroom + solvabilidad, el frente vuelve a rediseño antes de tocar GPU.

## Documentación local de incubación

- Roadmap del frente: `./ROADMAP_ATENCION_ARMONICA.md`
- Plan vigente de `Fase 0 v2.1`: `./PLAN_FASE_0_v2_1.md`
- Plan anterior preservado por trazabilidad: `./PLAN_FASE_0_v1_superseded.md`

## Regla de propagación

Este frente **no se propaga a `Documents/00_TRONCAL/`** hasta que exista al menos un primer resultado real de `Fase 0`: gate PASS sobre `final_pool` y training ya ejecutado. Antes de eso, su capa documental correcta es esta carpeta local de incubación.
