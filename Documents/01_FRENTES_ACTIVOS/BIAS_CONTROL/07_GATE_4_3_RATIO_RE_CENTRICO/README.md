# Gate 4.3 - Ratio Re-Centrico

Estado: CERRADO (fase experimental principal completa)  
Fecha de cierre de resultados: 2026-02-16  
Corte documental: 2026-02-17

Gate 4.3 respondió cinco preguntas operativas:

1. Si los descriptores de ratios aportan señal en MIDI, en audio o en ambos.
2. Si `concat` o `cross-attention` integra mejor esa señal.
3. Si el acoplamiento dual (MIDI+Audio) es aditivo o superaditivo.
4. Si la inyección cross-modal (descriptor de A en encoder de B) ayuda o degrada.
5. Si el mejor brazo escala en training largo desde scratch.

## Resultado global

- Mejor brazo 5ep: `d4a4` (dual same-modality concat) con `S=69.8%` (`+9.6pp` vs `D0`).
- Mejor brazo single-descriptor: `A4r` (reverse cross-attention) con `S=68.6%`.
- Mejor run largo: `d4a4-scratch` 30ep con `S=83.6%` (record absoluto del proyecto).

## Tabla final — 13 brazos (5ep)

| Rank | Brazo | Descriptor / Familia | Mecanismo | Best S | vs D0 |
|------|-------|----------------------|-----------|--------|-------|
| 1 | d4a4 | D4 + A4 (dual same-mod) | concat | 69.8% | +9.6pp |
| 2 | A4r | A4 (audio) | reverse cross-att | 68.6% | +8.4pp |
| 3 | D4r | D4 (midi) | reverse cross-att | 64.2% | +4.0pp |
| 4 | D4 | D4 (midi) | concat | 63.6% | +3.4pp |
| 4 | A4 | A4 (audio) | concat | 63.6% | +3.4pp |
| 6 | A4x | A4 (audio) | cross-att | 62.6% | +2.4pp |
| 7 | A7x | A7 (audio) | cross-att | 62.2% | +2.0pp |
| 8 | D0 | control | baseline | 60.2% | — |
| 9 | D4x | D4 (midi) | cross-att | 60.0% | -0.2pp |
| 10 | A7 | A7 (audio) | concat | 58.8% | -1.4pp |
| 10 | A9 | A9 (audio) | concat | 58.8% | -1.4pp |
| 12 | A8 | A8 (audio) | concat | 57.4% | -2.8pp |
| 13 | d4a4cm | D4->audio + A4->midi | cross-modal concat | 52.4% | -7.8pp |

## d4a4-scratch (30ep)

Output: `data/bias_control_medium/training_outputs/gate43/gate43_d4a4_scratch_30ep/`

| Epoch | S | hard_neg | MRR_avg |
|------:|---:|---------:|--------:|
| 10 | 74.6% | 93.0% | 0.336 |
| 15 | 65.8% | 91.0% | 0.316 |
| 20 | 75.6% | 93.6% | 0.370 |
| 25 | 82.2% | 95.4% | 0.430 |
| 28 | 82.8% | 94.8% | 0.444 |
| 29 | 82.6% | 95.2% | 0.443 |
| 30 | 83.6% | 95.2% | 0.444 |

Multi-seed e30 (5 seeds): `S=84.1% +/- 2.3pp`.

## Hallazgos de diseño

1. `concat` gana sobre cross-attention regular para descriptores fuertes (`D4`, `A4`).
2. Reverse cross-attention (`Q=descriptor`, `K/V=features`) supera a cross-attention regular en audio y midi.
3. `d4a4` muestra sinergia superaditiva (dual > suma de mejoras individuales).
4. Cross-modal injection en esta forma (`d4a4cm`) degrada fuertemente la señal.
5. Los descriptores más robustos del ciclo fueron `D4` (midi intervals) y `A4` (audio log-freq deltas).

## Estado posterior a Gate 4.3

- Gate 4.4: screening cerrado (24 brazos) + runs largos `t3-wt`/`moe-dual` cerrados.
- Bloque largo 30ep cerrado para comparación de mecanismos: `d4a4`, `a4r`, `d4-a4r`, `t3-wt`, `d4a4r`, `moe-dual`.
- Nueva etapa activa formalizada como Gate 4.5: bloque 50ep/60ep + contraste de scheduler.
- Gate 5A/Gate 5B: siguen como siguiente bloque estructural, condicionado al cierre de Gate 4.5.

## Documento eje

- `INFORME_GATE_4_3_RATIO_RE_CENTRICO.md`
- `plan_gate_4.3.md`
