# Gate 5B — Explicación de tests (corte 2026-02-25)

Este documento resume qué mide cada test, cómo se interpreta y cuál es su estado real.

## Estado rápido

- Cerrados: `Test12`, `Test01`, `Test04`, `Test03`, `Test06`, `Test08`, `Test10`, `Test09`.
- Pendientes UNC: `Test02` y `Test05`.

## Test 12 — Scoreboard (infraestructura)

Pregunta: "¿el loader reconstruye bien cada arquitectura?"

- Reevalúa checkpoints canónicos con config fija (`pool=256`, `queries=500`, `seed=42`).
- Si los valores coinciden con históricos, la infraestructura de evaluación es confiable.

Resultado: PASS (`D0=73.4%`, `d4a4=83.8%`, `a4r=82.0%`, `d4-a4r=79.8%`).

## Test 01 — Causal Ablation (mecanismo)

Pregunta: "¿la mejora viene de la señal del descriptor o el modelo la ignora?"

Modos:
- `zero_*`: descriptor a cero.
- `noise_*`: descriptor reemplazado por ruido gaussiano con la misma media y desviación estándar del descriptor real.
- `shuffle_*`: descriptor correcto pero asignado al sample equivocado.

Lectura:
- Caída grande en `S` -> dependencia causal fuerte.
- Caída pequeña o nula -> aporte marginal o nulo.

Resultado global:
- A4/A4r: causal dominante en modelos top.
- D4: marginal en `d4a4` y `d4-a4r`; efecto pequeño en `d4` puro.

## Test 04 — Transposition Invariance

Pregunta: "¿el modelo aprendió estructura relativa o pitch absoluto?"

- Se transpone MIDI en `[-6,-3,-1,+1,+3,+6]` semitonos y se reevalúa retrieval.
- Mayor retención de `S` bajo transposición implica mejor invariancia relativa.

Resultado: cerrado para `D0`, `d4a4`, `a4r`, `d4-a4r`.

## Test 03 — Ratio Probe

Pregunta: "¿el embedding de un dominio contiene información linealmente decodificable del otro?"

- Probes lineales para `z_audio -> features MIDI` y `z_midi -> features audio`.
- Métrica: `R²`.

Lectura del corte:
- No aparece "smoking gun" lineal fuerte a favor de augmented.
- La mejora de retrieval parece vivir más en geometría no lineal que en decodificación lineal simple.

## Test 06 — RSA/CKA

Pregunta: "¿qué tanto se alinean internamente los encoders audio y MIDI?"

- Hooks por capa, matriz de similitud representacional entre capas.
- Métricas: RSA y CKA.

Resultado del corte:
- Fuerte aumento de alineación cross-encoder con A4/A4r respecto a `D0`.

## Test 08 — Ratio Decoding (sensitivity)

Pregunta: "¿qué dimensiones del descriptor pesan más en el embedding final?"

- Método por perturbación (no gradientes), midiendo cambio en embedding por dimensión.
- A4 modelado como 8 bandas de octava (`band0_47Hz` ... `band7_6000Hz`).

Resultado del corte:
- Mayor sensibilidad en bandas medias/altas (aprox. `750+ Hz`) en modelos augmented.

## Test 09 — Invariance Suite

Pregunta: "¿qué tan robusto es el modelo ante perturbaciones realistas?"

Perturbaciones:
- temporal shift,
- velocity scaling,
- octave transposition,
- audio noise (SNR).

Estado actual:
- cerrado en `D0`, `d4a4`, `a4r` y `d4-a4r`.

Lectura global:
- temporal shift: robustez aceptable en los 4 arms (peor caso entre `-3.6pp` y `-7.2pp`);
- velocity scaling y octave transposition: fragilidad alta en todos los modelos;
- audio noise: patrón bimodal (`D0` domina en 40-20 dB; `a4r/d4-a4r` retienen mejor en 5 dB).

## Test 02 y Test 05 (UNC)

- `Test02` (parameter-matched ablations): controla confound de cantidad de parámetros.
- `Test05` (multi-seed): robustece inferencia estadística.

Sin estos dos tests, el cierre científico final sigue abierto.
