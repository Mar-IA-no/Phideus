<div align="center">

# Proyecto Estado Actual
### Phideus v5.0

![Program](https://img.shields.io/badge/Program-Research_Active-0A7E3B?style=for-the-badge)
![Current Focus](https://img.shields.io/badge/Focus-Escalon_2-1F6FEB?style=for-the-badge)
![Bias Control](https://img.shields.io/badge/BIAS_CONTROL-Gate_6_AMT_ACTIVO-1F6FEB?style=for-the-badge)

</div>

> [!IMPORTANT]
> **Actualizado**: 2026-03-05
> **Estado**: **Gate 5B quedó completamente cerrado**. `Test05` ya estaba consolidado en `results_unc` (`15/15`) y `Test02` pasó a leerse como **4/4 completo**: `real=83.0%`, `zero=75.0%`, `random=73.6%`, `shuffled=73.6%*`. La lectura multi-seed vigente queda en `d4a4=84.1%±2.3pp`, `d4-a4r=81.2%±2.5pp`, `a4r=80.7%±1.9pp`, `D0=75.2%±2.3pp`. **Test 11 A/B pre-projection** ya cerró `4/4` y dejó el ranking mecanístico completo: `d4a4=0.770 > d4-a4r=0.748 > a4r=0.712 > D0=0.597`. **Test 13G-B** también quedó cerrado `4/4`: `D0(pool-188)=0.1089`, `d4a4=0.1037`, `a4r=0.1024`, `d4-a4r=0.1021`, sin ventaja descriptor-guided en decodificabilidad pre-pooling.
> **Gate 6 AMT**: la línea downstream sigue abierta, pero ya con un corte local completo. `Exp 0` se completó localmente con baseline `Transkun` sobre segmentos de `4s` y `16s`; `Exp C` falló primero en UNC por path absoluto de MAESTRO, fue corregido en `3` scripts SLURM y reenviado como `job 1144560`; además, la corrida local `a4r` del decoder grande cerró `80` épocas con `best_F1=0.1570 @ ep50`, por encima del techo de `13G-B`. `transkun` ya está instalado en UNC, de modo que `Exp A` queda listo para submitir y `Exp B` sigue bloqueado por `Exp A`.
> **Gate 7**: `Exp 7.0` ya quedó completo y redujo la ambigüedad del lado audio (`MERT-330M=0.850`, `MERTLite=0.734`, `MERT-95M=0.659`), mientras `Gate 7.1` ya no está en estado difuso: quedó formalizado como plan v2 bifásico, con `7.1a` (`D0` pilot sobre `MERT-330M` congelado) como próximo experimento de decisión si el programa decide volver a invertir en Escalón 1.
> **Decisión operativa vigente**: (1) tratar `Test02` como cierre causal del argumento de capacidad, (2) leer `13G-B` como resultado negativo/generativo genérico y usar `Test11` para sostener el hallazgo mecanístico del cuello de proyección, (3) mantener Escalón 2 como foco principal y Gate 6 como validación downstream real, y (4) tratar Gate 7.1 como piloto decisional acotado, no como campaña obligatoria.
> **Encuadre estrategico**: Gate 5A deja de ser barrido bloqueante. Conditioned projections queda implementado como línea oportunista; Escalón 2 pasa a foco principal tras el cierre efectivo de Gate 5B; Gate 6 AMT abre una línea de validación musical concreta sin reabrir el cierre canónico del gate anterior; Gate 7.1 queda como la forma más corta de reabrir Escalón 1 si hiciera falta resolver la ambigüedad residual encoder vs geometría.
> **Infraestructura**: estrategia distribuida LOCAL+UNC activa; foundation lock publicado (`v0.1.0-foundation`).

\* `shuffled` se tomó como cierre operativo por convergencia clara en `e20`.

## Navegación rápida

- [Resumen Ejecutivo](#resumen-ejecutivo)
- [Estado por Gate](#estado-por-gate)
- [Hallazgos Causales del Corte](#hallazgos-causales-del-corte)
- [Plan Operativo Vigente](#plan-operativo-vigente)
- [Frentes y Documentos](#frentes-y-documentos)

---

## Resumen Ejecutivo

Gate 4.3 dejó una base fuerte (`d4a4=69.8%` a 5ep; `d4a4=83.6%` a 30ep), y Gate 4.4 completó el filtro arquitectural con evidencia comparable en toda la grilla corta. El bloque largo confirmó techo competitivo en variantes ratio-céntricas y habilitó selección robusta de checkpoints para validación científica.

Ese bloque de dinámica temporal queda encapsulado como **Gate 4.5 (LR Schedule Optimization)** y opera ahora como soporte de **Gate 5B**: checkpoints consolidados para evaluación científica (`D0`, `d4`, `d4a4`, `a4r`, `d4-a4r`) con loader universal y protocolo canónico fijo.

### Baseline oficial de comparación (histórico)

`Gate 2 - checkpoint_epoch45`

| Métrica | Valor |
|--------|-------|
| A2M R@10 (pool 256/500/seed42) | 34.4% |
| M2A R@10 (pool 256/500/seed42) | 37.6% |
| Hard negative accuracy | 80.4% |
| S=min(A2M,M2A) | 34.4% |

### Screening @5ep (ranking unificado, top del frente)

| Rank | Brazo | Gate/Familia | Best S | A2M | M2A | hard_neg | vs D0 |
|------|-------|--------------|--------|-----|-----|----------|-------|
| 1 | d4a4 | 4.3 Dual concat | 69.8% | 69.8% | 70.6% | 91.6% | +9.6pp |
| 2 | a4r | 4.3-F5 reverse | 68.6% | 68.6% | 69.0% | 91.6% | +8.4pp |
| 3 | t3-wt | 4.4 Third Tower | 67.6% | 71.4% | 67.6% | 91.2% | +7.4pp |
| 4 | t3-tri | 4.4 Third Tower | 65.0% | 65.4% | 65.0% | 90.6% | +4.8pp |
| 10 | D0 | 4.3 baseline | 60.2% | 60.4% | 60.2% | 90.0% | — |
| 11 | moe-a4-v2 | 4.4-MoE | 60.2% | 60.4% | 60.2% | 90.8% | 0.0pp |

Notas de cierre 4.4:
- `film-dual` cerró en `59.4%` (e5), `moe-dual` en `59.2%` (e5).
- `moe-a4-v2/v3/v4` no superan D0 (v2 empata D0).

### Runs largos (30ep, scratch) — todos cerrados

| Descriptor | Best S | Best Ep | A2M | M2A | hard_neg |
|-----------|--------|---------|-----|-----|----------|
| d4a4 | 83.6% | 30 | 83.6% | 84.2% | 95.2% |
| a4r | 82.0% | 29 | 82.6% | 82.0% | 94.4% |
| d4-a4r | 79.8% | 30 | 81.4% | 79.8% | 94.2% |
| t3-wt | 79.8% | 30 | 82.4% | 79.8% | 94.8% |
| d4a4r | 74.4% | 30 | 74.4% | 74.8% | 92.0% |
| moe-dual | 72.6% | 30 | 72.8% | 72.6% | 93.4% |

Multi-seed e30 (5 seeds): `d4a4 = 84.1% +/- 2.3pp`.

### Gate 4.5 + Gate 5B (corte operativo 2026-03-01)

| Bloque | Corridas | Estado |
|--------|----------|--------|
| Batch 60ep (cosine estándar) | `a4r` | **completado** (`S=79.4%` en e60) |
| Batch 60ep (cosine estándar) | `D0`, `d4a4` | **completados** (`D0=72.8%`, `d4a4=83.8%`) |
| Batch 60ep (cosine estándar) | `d4-a4r` | **completado** (`S=79.8%` en e55) |
| Batch 60ep (cosine estándar) | `moe-dual` | **dead por time limit** (`best S=73.0%` en e30, no sostenido) |
| Hold scheduler 50ep | `t3-wt` (`--lr-hold-fraction=0.5`) | **completado** (`S=81.2%` en e50) |
| Batch 60ep (cosine-tail) | `a4r` | **completado** (`S=80.6%` en e60) |
| Gate 5B Test12 (scoreboard) | `D0`, `d4a4`, `a4r`, `d4-a4r` | **cerrado** (`73.4%`, `83.8%`, `82.0%`, `79.8%`) |
| Gate 5B Test01 (causal ablation) | `D0`, `d4`, `d4a4`, `a4r`, `d4-a4r` | **cerrado** (A4/A4r causal fuerte; D4 marginal en duales) |
| Gate 5B Test04 (transposition) | `D0`, `d4a4`, `a4r`, `d4-a4r` | **cerrado** (robustez relativa mayor en modelos con A4/A4r) |
| Gate 5B Test03 (ratio probe) | `D0`, `d4a4`, `a4r`, `d4-a4r` | **cerrado** (sin “smoking gun” lineal; mejora vive en geometría de retrieval) |
| Gate 5B Test06 (RSA/CKA) | `D0`, `d4a4`, `a4r`, `d4-a4r` | **cerrado** (alineación cross-encoder aumenta fuerte con A4/A4r) |
| Gate 5B Test08 (ratio decoding) | `d4a4`, `a4r`, `d4-a4r` | **cerrado** (bandas 750+ Hz dominan sensibilidad) |
| Gate 5B Test10 (visualizaciones) | `D0`, `d4a4`, `a4r`, `d4-a4r` | **cerrado** (paquete visual v2: 24 PNG + 6 GIF) |
| Gate 5B Test09 (invariance suite) | `D0`, `d4a4`, `a4r`, `d4-a4r` | **cerrado** (temporal robusto; alta fragilidad a velocity/octava; robustez a ruido con patrón bimodal) |
| Gate 5B Test05 (multi-seed, UNC) | `D0`, `a4r`, `d4-a4r` x `5` seeds | **cerrado** (`15/15` en `results_unc`) |
| Gate 5B Test02 (parameter-matched, UNC) | `real`, `random`, `shuffled`, `zero` | **cerrado (4/4)** (`83.0%`, `75.0%`, `73.6%`, `73.6%*`; misma arquitectura, misma receta) |
| Gate 5B Test13G-A (generative encoder) | `D0`, `λ={0.03,0.1,0.3}` | **cerrada** (`best_S≈64.4-64.6%`, `PR F1≈0.11`; ruta `z=256 -> piano-roll` descartada) |
| Gate 5B Test11 Pre-Proj A/B | `D0`, `a4r`, `d4a4`, `d4-a4r` | **cerrado** (`retention ratio`: `0.597`, `0.712`, `0.770`, `0.748`) |
| Gate 5B Test13G-B (post-hoc pre-pooling) | `D0`, `d4a4`, `a4r`, `d4-a4r` (+ control `D0 pool-to-188`) | **cerrado** (`F1≈0.10` en todos; `D0 pool-188` levemente superior, sin ventaja descriptor-guided) |

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
| Gate 6 (diagnóstico histórico) | Completado | Causa raíz confirmada |
| Bloque A v1.1 | Cerrado | `D-02 e25` como foundation lock |
| Gate 4.2 ratio-céntrico | Cerrado | `D4 8ep` (`S=64.2%`) |
| Gate 4.3 ratio re-céntrico | Cerrado | 13 brazos + scratch; record 30ep `S=83.6%` |
| Gate 4.4 arquitecturas mayores | **Cerrado** | Screening 24 brazos + 30ep (`t3-wt`, `moe-dual`) |
| Gate 4.5 LR schedule optimization | **Cierre operativo** | resultados consolidados y usados en selección de checkpoints |
| Gate 5A | Replanteado | conditioned projections (implementado) + combinatorios `t3-wt` + dos slots TBD; ejecucion oportunista en paralelo con Escalon 2 |
| Gate 5B showcase científico | **Cerrado** | `Test02` 4/4, `Test13G-B` completo y cierre formal de la Línea B de Escalón 1-C |
| Gate 6 AMT | Activo | `Exp 0` completo; `Exp C` con brazo local `a4r` ya completo (`best_F1=0.1570 @ ep50`) y resubmisión UNC `1144560`; `Exp A` listo; `Exp B` bloqueado |
| Gate 7 | Acotado / en decisión | `Exp 7.0` completo; `Exp 7.0b` opcional; `Gate 7.1` ya tiene plan v2 bifásico (`7.1a` D0 pilot, `7.1b` `a4r-mert` condicional), implementación pendiente |

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

5. **Third Tower weighted (`t3-wt`) mostró convergencia tardía real**  
Pasó de `S=40.0%` (e5 en 30ep scratch) a `S=79.8%` (e30).

6. **MoE mejoró transitoriamente, pero no sostuvo el pico en extendido**  
`moe-dual` llegó a `73.0%` (e30, 60ep stretched) y luego cayó a banda 69-72; terminó dead por time limit.

7. **En 5ep, FiLM/MoE quedaron en banda 58-60%**  
La familia 4.4 no desplazó a los ganadores de Gate 4.3 en screening corto.

8. **Gate 5B valida causalidad de la rama audio**
En Test01, ablacionar A4/A4r produce caídas masivas de `S` (32-78pp según modo/modelo), mientras que ablaciones D4 en modelos duales muestran efecto marginal o nulo.

9. **Gate 5B confirma invariancia relativa en modelos con A4**
En Test04 (cerrado), los modelos con A4/A4r retienen más `S` bajo transposición MIDI que `D0`, consistente con uso de señal relativa (ratios) y no solo pitch absoluto.

10. **Test09 cierra la lectura de robustez con un patrón no trivial**
Todos los arms son robustos a shifts temporales moderados, frágiles a escalado de velocity y transposición de octava; en ruido aparece patrón bimodal: `D0` domina en SNR 40-20 dB y `d4-a4r/a4r` retienen más `S` en 5 dB.

---

## Plan Operativo Vigente

Secuencia inmediata:

1. **Tratar Gate 5B como bloque cerrado**: `Test02` ya cerró el control de capacidad y `13G-B` ya cerró la línea generativa post-hoc sin ventaja descriptor-guided.
2. **Sostener el hallazgo Test 11 A/B pre-projection**: `D0` retiene `59.7%` de la informacion MIDI al cruzar modalidad y `a4r` retiene `71.2%`, con destruccion de `81-88%` en la proyeccion MIDI 512→256.
3. **Leer Gate 6 como frente ya activo**: `Exp C` ya dejó una referencia local completa (`a4r`, `best_F1=0.1570 @ ep50`) y el array UNC `1144560` sigue siendo la comparación pendiente para `D0/d4a4/d4-a4r`.
4. **Mantener Gate 7 como línea acotada de decisión**: `Exp 7.0` ya resolvió la pregunta barata del lado audio y `Gate 7.1` quedó rediseñado como plan v2 bifásico; si se reabre Escalón 1, el primer movimiento correcto es `7.1a` (`D0` pilot sobre `MERT-330M` congelado), no un `7.1` grande.
5. **Abrir `Exp A` cuando haya slot**: `transkun` ya está instalado en UNC; el bloqueo ya no es de entorno sino de prioridad/recursos.
6. **Mantener `Exp B` condicionado por `Exp A`**: no conviene abrir degradaciones antes de validar el pipeline `Transkun+A4`.
7. **Abrir Escalón 2 como foco principal**: la transición estratégica ya no depende de Gate 5A ni de Gate 6.
8. **Mantener Gate 5A como línea paralela y oportunista**: conditioned projections y combinatorios `t3-wt` absorben solo recursos libres.
9. **Usar `13G-B` como resultado negativo útil**: confirma que la ventaja de descriptores vive en la geometría de retrieval, no en una mayor decodificabilidad de piano-roll.

Marco estrategico inmediato:

1. Gate 5B ya quedó cerrado como cierre principal de Escalón 1-C.
2. Escalón 2 (Speech <-> EGG) pasa a ser el foco principal del programa.
3. Gate 5A continúa como línea paralela y oportunista: no bloquea Escalón 2 y solo absorbe recursos libres para conditioned projections, combinatorios `t3-wt` y futuras hipótesis acotadas.
4. Gate 6 AMT abre una validación downstream concreta: no reemplaza Escalón 2, pero sí prueba si la ventaja descriptor-guided sobrevive fuera del retrieval.
5. Gate 7 queda en modo de resolución acotada: `Exp 7.0` ya está cerrado y `Gate 7.1` solo se justifica como piloto corto de decisión, no como nueva campaña principal.

---

## Frentes y Documentos

| Documento | Rol |
|-----------|-----|
| `README.md` | Entrada principal del repositorio |
| `Documents/00_TRONCAL/HANDOFF.md` | Continuidad operativa |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md` | Plan maestro vigente |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/INDEX_BIAS_CONTROL.md` | Navegación del frente |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/RANKING_DESCRIPTORES_UNIFICADO.md` | Tabla canónica corta+larga |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/09_GATE_4_5_LR_SCHEDULE_OPTIMIZATION/README.md` | Gate 4.5 (scheduler/LR) |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/12_GATE_6_AMT/README.md` | Gate 6 AMT (validación downstream) |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/13_GATE_7_MERT_PROBE/README.md` | Gate 7 (probe lineal MERT-large) |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/14_GATE_7.1/README.md` | Gate 7.1 (plan v2 bifásico, implementación pendiente) |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_UNC.md` | Estrategia distribuida LOCAL+UNC |
| `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md` | Evolución histórica de representaciones |
| `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/CATALOGO_NARRATIVO_DESCRIPTORES_RATIOS_PHIDEUS.md` | Catálogo vivo de descriptores |

Nota operativa:
- Foundation lock publicado en GitHub Release: `v0.1.0-foundation` (`foundation_locked_e25.pt`, MD5 `ddb2ebf7075eec4dcec1628341ec4942`).

---

*Documento actualizado al corte operativo 2026-03-05 (Gate 5B completamente cerrado; Test11 y Test13G-B en 4/4; Gate 6 activo con `a4r` local completo y resubmisión UNC `1144560`; Gate 7 Exp 7.0 completo y Gate 7.1 con plan v2 formalizado).*
