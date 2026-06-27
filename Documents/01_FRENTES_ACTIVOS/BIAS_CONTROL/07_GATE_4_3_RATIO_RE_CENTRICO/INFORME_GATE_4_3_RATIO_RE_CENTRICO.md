# Informe Gate 4.3 — Ratio Re-Centrico

Fecha de cierre: 2026-02-16  
Corte documental: 2026-02-17  
Estado: CERRADO (13 brazos de 5 epocas + 1 run largo scratch completado)

---

## 1) Marco experimental

Gate 4.3 se ejecutó como bloque causal controlado desde `foundation_locked_e25.pt`, con
protocolo canónico de evaluación (`pool=256`, `queries=500`, `seed=42`) y comparación
entre familias de descriptor+mecanismo.

Objetivos:

1. Medir aporte de descriptores ratio en MIDI y Audio.
2. Comparar mecanismos de inyección (`concat`, `cross-att`, `reverse cross-att`).
3. Validar interacción dual same-modality vs cross-modal.
4. Decidir candidato para entrenamiento largo (scratch).

---

## 2) Tabla final (13 brazos, 5ep)

| Rank | Brazo | Best ep | Best S | hard_neg | Lectura resumida |
|------|-------|---------|--------|----------|------------------|
| 1 | d4a4 | e5 | 69.8% | 91.6% | Mejor 5ep; dual same-mod superaditivo |
| 2 | A4r | e5 | 68.6% | 91.6% | Mejor single-descriptor; reverse audio |
| 3 | D4r | e5 | 64.2% | 93.2% | Reverse midi supera D4/D4x en hard_neg |
| 4 | D4 | e5 | 63.6% | 91.2% | Descriptor MIDI robusto |
| 4 | A4 | e5 | 63.6% | 92.4% | Descriptor Audio robusto |
| 6 | A4x | e5 | 62.6% | 92.4% | Cross-att audio queda debajo de concat |
| 7 | A7x | e5 | 62.2% | 92.0% | Cross-att rescata A7, pero no lidera |
| 8 | D0 | e3 | 60.2% | 90.0% | Baseline canónico |
| 9 | D4x | e5 | 60.0% | 91.4% | Cross-att midi no mejora baseline |
| 10 | A7 | e5 | 58.8% | 90.2% | Atractor concat debajo de baseline |
| 10 | A9 | e5 | 58.8% | 90.4% | IDF-attractor no supera D0 |
| 12 | A8 | e5 | 57.4% | 90.6% | Onset-chroma no supera D0 |
| 13 | d4a4cm | e5 | 52.4% | 89.6% | Cross-modal concat destructivo |

---

## 3) Hallazgos causales

### 3.1 Same-modality dual: efecto superaditivo

- D4 solo: `+3.4pp` vs D0.
- A4 solo: `+3.4pp` vs D0.
- d4a4 dual: `+9.6pp` vs D0.

La mejora dual excede la suma simple de mejoras individuales en este régimen.

### 3.2 Mecanismo

- Para descriptores fuertes (`D4`, `A4`), `concat` supera a cross-attention regular.
- Reverse cross-attention supera consistentemente a cross-attention regular:
  - `A4r (68.6%) > A4x (62.6%)`
  - `D4r (64.2%) > D4x (60.0%)`

Interpretación operativa: en esta familia de modelos, usar el descriptor como consulta
(`Q=descriptor`) organiza mejor las features que usarlo solo como memoria (`K/V=descriptor`).

### 3.3 Cross-modal injection

`d4a4cm` (`D4->audio`, `A4->midi`) cierra en `52.4%` (`-7.8pp` vs D0).

Conclusión práctica: en este diseño, el intercambio temprano de descriptores entre modalidades
rompe más señal de la que aporta.

---

## 4) Run largo: d4a4-scratch 30ep

Run desde `MERT pretrained + MIDI random` (sin foundation fine-tune previo), mismo régimen `run-d`.

| Epoch | Loss | S | A2M | M2A | hard_neg | MRR_avg |
|------:|-----:|---:|----:|----:|---------:|--------:|
| 10 | 13.60 | 74.6% | 74.6% | 75.0% | 93.0% | 0.336 |
| 15 | 13.38 | 65.8% | 65.8% | 68.6% | 91.0% | 0.316 |
| 20 | 13.26 | 75.6% | 75.6% | 76.8% | 93.6% | 0.370 |
| 25 | 13.21 | 82.2% | 82.8% | 82.2% | 95.4% | 0.430 |
| 28 | 13.19 | 82.8% | 82.8% | 83.6% | 94.8% | 0.444 |
| 29 | 13.19 | 82.6% | 82.6% | 83.8% | 95.2% | 0.443 |
| 30 | 13.20 | 83.6% | 84.0% | 83.6% | 95.2% | 0.444 |

Resultado final: `S=83.6%` (record del proyecto), `+21.8pp` sobre D-02 best (`61.8%`).

Referencia eval-seed e30 (5 eval-seeds, 1 checkpoint): `S=84.1% +/- 2.3pp`.

---

## 5) Síntesis técnica

1. La hipótesis de utilidad ratio queda validada en inyección same-modality (`D4`, `A4`, `d4a4`).
2. En Gate 4.3, el descriptor audio más robusto fue `A4`; `A7/A8/A9` no superaron baseline en concat.
3. Reverse cross-attention emerge como mecanismo fuerte para single-descriptor.
4. El mejor candidato de continuidad larga quedó siendo d4a4-scratch (ya completado con record).
5. Quedan abiertos los tests estratégicos de `a4r-scratch` y `d4a4r-scratch` como comparadores de simplicidad/eficiencia y de continuidad dual reverse.

---

## 6) Estado de transición

- Gate 4.3: cerrado formalmente.
- Próximo bloque arquitectural: Gate 4.4 (third tower + FiLM + MoE).
- Validación científica extendida: Gate 4.5 (scheduler) y luego Gate 5A/5B según roadmap vigente.

Referencias directas:
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_UNC.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/07_GATE_4_3_RATIO_RE_CENTRICO/INFORME COMPLETO: d4a4-scratch 30 epochs.md`
