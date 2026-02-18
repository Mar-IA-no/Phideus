<div align="center">

# Proyecto Estado Actual
### Phideus v5.0

![Program](https://img.shields.io/badge/Program-Research_Active-0A7E3B?style=for-the-badge)
![Current Focus](https://img.shields.io/badge/Focus-Escalon_1--C-1F6FEB?style=for-the-badge)
![Bias Control](https://img.shields.io/badge/BIAS_CONTROL-Gate_4.4_Corte_Parcial_UNC-F59E0B?style=for-the-badge)

</div>

> [!IMPORTANT]
> **Actualizado**: 2026-02-18  
> **Estado**: Gate 4.3 cerrado con 13 brazos (5ep) + `d4a4-scratch` 30ep completo. Record del proyecto en `S=83.6%` (e30), multi-seed `S=84.1% +/- 2.3pp`.  
> **Decisión operativa vigente**: Gate 4.4 en corte parcial avanzado de screening UNC (6 brazos con e5 cerrado, 2 con e3 provisional y e5 pendiente), manteniendo protocolo foundation + `run-d`.  
> **Infraestructura**: estrategia distribuida LOCAL+UNC activa; release de foundation publicado (`v0.1.0-foundation`).

## Navegación rápida

- [Resumen Ejecutivo](#resumen-ejecutivo)
- [Estado por Gate](#estado-por-gate)
- [Hallazgos Causales del Corte](#hallazgos-causales-del-corte)
- [Plan Operativo Vigente](#plan-operativo-vigente)
- [Frentes y Documentos](#frentes-y-documentos)

---

## Resumen Ejecutivo

El cierre de Gate 4.3 cambió el baseline práctico del programa: el mejor brazo dual (`d4a4`) no sólo superó al control, sino que en training largo desde scratch empujó el sistema a un rango nuevo de rendimiento (`S>80%`).

No fue una curva lineal. Hubo dips, recuperaciones y cambios de mecanismo que obligaron a sostener la disciplina metodológica: comparar con protocolo fijo, separar observación de inferencia, y no declarar techos tempranos.

### Baseline oficial de comparación (histórico)

`Gate 2 - checkpoint_epoch45`

| Métrica | Valor |
|--------|-------|
| A2M R@10 (pool 256/500/seed42) | 34.4% |
| M2A R@10 (pool 256/500/seed42) | 37.6% |
| Hard negative accuracy | 80.4% |
| S=min(A2M,M2A) | 34.4% |

### Cierre Gate 4.3 (13 brazos, 5ep)

| Rank | Brazo | Mecanismo | Best S | vs D0 |
|------|-------|-----------|--------|-------|
| 1 | d4a4 | dual same-mod concat | 69.8% | +9.6pp |
| 2 | A4r | reverse cross-att audio | 68.6% | +8.4pp |
| 3 | D4r | reverse cross-att midi | 64.2% | +4.0pp |
| 4 | D4 | concat midi | 63.6% | +3.4pp |
| 4 | A4 | concat audio | 63.6% | +3.4pp |
| 6 | A4x | cross-att audio | 62.6% | +2.4pp |
| 7 | A7x | cross-att audio attractor | 62.2% | +2.0pp |
| 8 | D0 | control | 60.2% | — |
| 9 | D4x | cross-att midi | 60.0% | -0.2pp |
| 10 | A7 | concat attractor | 58.8% | -1.4pp |
| 10 | A9 | concat IDF attractor | 58.8% | -1.4pp |
| 12 | A8 | concat onset-chroma | 57.4% | -2.8pp |
| 13 | d4a4cm | dual cross-modal concat | 52.4% | -7.8pp |

### Gate 4.4 (corte parcial UNC, 5ep)

| Brazo | Familia | S@e3 | S@e5 | Estado |
|-------|---------|------|------|--------|
| t3-wt | Third Tower | 47.6% | 67.6% | cerrado |
| t3-tri | Third Tower | 47.4% | 65.0% | cerrado |
| t3-anc | Third Tower | 40.2% | 42.2% | cerrado |
| moe-a4 | MoE | 58.8% | 58.2% | cerrado (best en e3) |
| film-a4 | FiLM | 59.2% | 59.2% | cerrado |
| film-d4 | FiLM | 58.8% | 58.6% | cerrado |
| film-dual | FiLM | 58.2% | pendiente | e5 en training |
| moe-dual | MoE | 59.2% | pendiente | e4/e5 en training (provisional e3) |

Referencia corta canónica: `d4a4=69.8%` (`D0=60.2%`), mismo protocolo 5ep + foundation + `run-d`.

### Run largo d4a4-scratch (30ep, completo)

| Epoch | S | hard_neg | MRR_avg |
|------:|---:|---------:|--------:|
| 10 | 74.6% | 93.0% | 0.336 |
| 15 | 65.8% | 91.0% | 0.316 |
| 20 | 75.6% | 93.6% | 0.370 |
| 25 | 82.2% | 95.4% | 0.430 |
| 28 | 82.8% | 94.8% | 0.444 |
| 29 | 82.6% | 95.2% | 0.443 |
| 30 | 83.6% | 95.2% | 0.444 |

Multi-seed e30 (5 seeds): `84.1% +/- 2.3pp`.

### Runs largos adicionales (30ep, scratch)

| Descriptor | Best S | Best Ep | A2M | M2A | hard_neg |
|-----------|--------|---------|-----|-----|----------|
| a4r | 82.0% | 29 | 82.6% | 82.0% | 94.4% |
| d4a4r | 74.4% | 30 | 74.4% | 74.8% | 92.0% |
| d4-a4r | en curso | — | — | — | — |
| t3-wt | en curso | — | — | — | — |

---

## Estado por Gate

| Gate / Etapa | Estado | Resultado |
|--------------|--------|-----------|
| Gate 0 | Completado | GO |
| Gate 1 | Completado | GO (sanity intra-modal) |
| Gate 2 | Completado | GO (baseline canónico) |
| Gate 2.5 | Completado | Diagnóstico de separabilidad |
| Gate 3 (DANN) | Cerrado | NO-GO |
| Gate 4.1 | Cerrado | NO-GO (`R1-rescue` insuficiente) |
| Gate 6 (diagnóstico) | Completado | Causa raíz confirmada |
| Bloque A v1.1 | Cerrado | `D-02 e25` como foundation lock |
| Gate 4.2 ratio-céntrico | Cerrado | `D4 8ep` (`S=64.2%`) |
| Gate 4.3 ratio re-céntrico | **Cerrado** | 13 brazos + scratch; record `S=83.6%` |
| Gate 4.4 arquitecturas mayores | Corte parcial UNC | 6 brazos cerrados en e5, 2 con e3 provisional y e5 pendiente |
| Gate 5A barrido | Pendiente | Barrido descriptor x mecanismo + cross-modal injection |
| Gate 5B showcase científico | Pendiente | 13 tests de validación |

---

## Hallazgos Causales del Corte

1. **Dual same-modality es superaditivo**  
D4 y A4 por separado dan `+3.4pp`; juntos (`d4a4`) dan `+9.6pp`.

2. **Reverse cross-attention supera al cross-attention regular**  
Se observó en audio y MIDI (`A4r>A4x`, `D4r>D4x`).

3. **Cross-modal injection temprana no fue mecanismo ganador**  
`d4a4cm` quedó por debajo del baseline (`-7.8pp` vs D0).

4. **El mejor espacio no apareció por accidente**  
`d4a4-scratch` superó a `D-02` por más de 20pp en el mismo marco de evaluación.

5. **El descriptor A4 (log-freq deltas) y D4 (intervalos MIDI) siguen siendo la pareja más robusta**  
A7/A8/A9 no desplazaron ese núcleo en este gate.

6. **Third Tower weighted (`t3-wt`) mostró recuperación tardía en ventana corta**  
Pasó de `S=47.6%` (e3) a `S=67.6%` (e5), quedando como mejor brazo Gate 4.4 en el corte parcial.

7. **FiLM/MoE siguen compactados en banda 58-59% en la ventana corta**  
`film-a4`, `film-d4`, `moe-a4` y `moe-dual@e3` se mantienen en zona cercana entre sí, todavía por debajo de `D0=60.2%` en el corte 5ep.

---

## Plan Operativo Vigente

Secuencia inmediata:

1. Cerrar los 2 brazos aún pendientes de e5 (`film-dual`, `moe-dual`) sin mezclar resultados provisionales con ranking final.
2. Consolidar tabla completa `S/A2M/M2A/hard_neg` para los 8 brazos en e3/e5.
3. Registrar resultados de runs largos en curso (`d4-a4r`, `t3-wt`) en la tabla larga comparativa.
4. Con tabla completa, dejar el frente listo para decisión de continuidad hacia Fase 2 (30ep) y Gate 5A/5B.

---

## Frentes y Documentos

| Documento | Rol |
|-----------|-----|
| `README.md` | Entrada principal del repositorio |
| `Documents/00_TRONCAL/HANDOFF.md` | Continuidad operativa |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md` | Plan maestro vigente |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/INDEX_BIAS_CONTROL.md` | Navegación del frente |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/07_GATE_4_3_RATIO_RE_CENTRICO/INFORME_GATE_4_3_RATIO_RE_CENTRICO.md` | Cierre técnico de Gate 4.3 |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_UNC.md` | Estrategia distribuida LOCAL+UNC |
| `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md` | Evolución histórica de representaciones |
| `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/CATALOGO_NARRATIVO_DESCRIPTORES_RATIOS_PHIDEUS.md` | Catálogo vivo de descriptores |

Nota operativa:
- Foundation lock publicado en GitHub Release: `v0.1.0-foundation` (`foundation_locked_e25.pt`, MD5 `ddb2ebf7075eec4dcec1628341ec4942`).

---

*Documento actualizado al corte parcial avanzado de Gate 4.4 en UNC (2026-02-18).* 
