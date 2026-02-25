# Test 08 — Ratio Decoding (Perturbation Sensitivity)

## Qué mide

Test08 mide cuánto cambia el embedding final cuando se perturba una dimensión individual del descriptor.

- Si la perturbación de una dimensión produce cambio grande -> dimensión influyente.
- Si produce cambio pequeño -> dimensión secundaria.

## Por qué es perturbation-based (y no gradiente)

En este stack, los descriptores se computan fuera del camino estándar de gradiente para el análisis (y en ciertos wrappers se usan con detach/no-grad en la parte descriptorial). Para evitar interpretaciones frágiles, el test usa perturbación directa y mide efecto en salida.

## Descriptor A4 correcto en este test

A4 está definido por 8 bandas de octava:

`band0_47Hz`, `band1_94Hz`, `band2_188Hz`, `band3_375Hz`, `band4_750Hz`, `band5_1500Hz`, `band6_3000Hz`, `band7_6000Hz`

No son `ratio_1_2`, `spec_centroid`, etc. Esos labels quedaron obsoletos.

## Resumen canónico (top-3 A4 por arm)

Fuente: `data/gate5b_results/*/test08_ratio_decoding.json`.

| Arm | Top-1 | Top-2 | Top-3 |
|---|---|---|---|
| `d4a4` | `band4_750Hz` (0.664) | `band5_1500Hz` (0.662) | `band3_375Hz` (0.546) |
| `a4r` | `band7_6000Hz` (0.933) | `band6_3000Hz` (0.875) | `band4_750Hz` (0.478) |
| `d4-a4r` | `band6_3000Hz` (1.092) | `band4_750Hz` (0.773) | `band5_1500Hz` (0.599) |

Lectura:
- La sensibilidad dominante cae en bandas medias/altas (desde ~`750 Hz`).
- `a4r` y `d4-a4r` muestran picos más altos en alta frecuencia que `d4a4`.

## D4 en el mismo test

En modelos duales, la sensibilidad máxima D4 es baja comparada con A4:

- `d4a4`: top D4 `duration_ratio = 0.077`
- `d4-a4r`: top D4 `duration_ratio = 0.124`

Esto es consistente con Test01: en duales top, la rama audio domina causalidad en inferencia.

## Cómo usar este resultado

- Para diseño de descriptor: priorizar ingeniería fina en bandas medias/altas de A4.
- Para interpretación científica: no concluir causalidad solo con sensibilidad; combinar con Test01 (causal ablation) y Test06 (alineación representacional).
