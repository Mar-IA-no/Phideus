<div align="center">

# Proyecto Estado Actual
### Phideus v5.0

![Program](https://img.shields.io/badge/Program-Research_Active-0A7E3B?style=for-the-badge)
![Current Focus](https://img.shields.io/badge/Focus-Escalon_2-1F6FEB?style=for-the-badge)
![Bias Control](https://img.shields.io/badge/BIAS_CONTROL-Gate_6_AMT_ACTIVO-1F6FEB?style=for-the-badge)

</div>

> [!IMPORTANT]
> **Actualizado**: 2026-09-03 (Olas 26–55 de ground truth proporcional integradas; la Ola 55 probó sobre evidencia fresca una compuerta conservadora posterior–decisión y mostró que un umbral global no convierte de forma estable la mejora representacional en una ventaja conjunta de accuracy, compatibilidad y regret; conserva el corte experimental consolidado al 2026-07-02)
> **Estado**: **Gate 5B quedó completamente cerrado**. `Test05` consolidó `20/20` trainings (`15/15` originales en `results_unc` + `5/5` de `d4a4`) y `Test02` pasó a leerse como **4/4 completo**: `real=83.0%`, `zero=75.0%`, `random=73.6%`, `shuffled=73.6%*`. La lectura canónica de Escalón 1 ya no depende de una referencia eval-seed: `d4a4` completó sus **5 training seeds independientes** con cierre `84.0%±2.7pp`, frente a `d4-a4r=81.2%±2.5pp`, `a4r=80.7%±1.9pp` y `D0=75.2%±2.3pp`. **Test 11 A/B pre-projection** ya cerró `4/4` y dejó el ranking mecanístico completo: `d4a4=0.770 > d4-a4r=0.748 > a4r=0.712 > D0=0.597`. **Test 13G-B** también quedó cerrado `4/4`: `D0(pool-188)=0.1089`, `d4a4=0.1037`, `a4r=0.1024`, `d4-a4r=0.1021`, sin ventaja descriptor-guided en decodificabilidad pre-pooling.
> **Gate 6 AMT**: la línea downstream ya dejó atrás la fase de screening abierto. `Exp 0` sigue fijando el baseline `Transkun`; `Exp C` local `a4r` mantiene `best_F1=0.1570 @ ep50`; y la rama `Transkun+A4` ya quedó metodológicamente mucho más cerrada: **`Exp B` ya puede leerse como cierre negativo** porque `20/27` tareas bastaron para mostrar que fine-tuning y `A4-degraded` convergieron a la misma banda del baseline degradado, con deltas entre `+0.0011` y `-0.0005`; y **`Exp A` también ya cerró negativamente** porque `baseline`, `finetune-noA4`, `A4-event`, `A4-adapter` y `adapter-noA4` dieron el mismo `F1=0.3186`, sin superar el criterio mínimo de `+0.01`. La consecuencia útil ya no es "seguir probando A4 en Transkun", sino conservar `Exp C` como única rama downstream todavía abierta.
> **Gate 7**: `Exp 7.0` ya quedó completo y redujo la ambigüedad del lado audio (`MERT-330M=0.850`, `MERTLite=0.734`, `MERT-95M=0.659`), mientras `Gate 7.1a` ya cerró su pilot `D0` sobre `MERT-330M` congelado con `S=75.0%`, esencialmente igual a `D0_lite=75.2%`. La lectura útil del corte es austera: un backbone de audio mucho más fuerte, pero congelado, no mejoró el retrieval bajo el mismo régimen VICReg.
> **Gate 8 / Gate 5A**: la línea de conditioned projections ya quedó cerrada en sus `5/5` brazos. Localmente `a4r-ctrl` cerró con `S=79.2%` y `a4r-pcm` con `S=80.0%`; en UNC `a4r-pcd-zero` cerró con `S=81.8% @ e30`, `a4r-pca` con `S=82.6% @ e25` y `a4r-pcd` con `S=84.2% @ e25`. La lectura final ya no es la de una línea “casi cerrada”: `pcd > pca > pcd-zero > pcm > ctrl`. La arquitectura conditioned agrega expresividad por sí misma (`pcd-zero > ctrl`), el conditioning real agrega señal adicional (`pcd > pcd-zero`) y el lado audio responde más que el MIDI-side cuando se lo condiciona de manera aislada (`pca > pcm`).
> **Escalón 2**: el frente ya no debe describirse como "factorial corriendo". `S2-P0`, `S2-P1`, `S2-P2-control`, `S2-P2-main`, `S2-P2.5` y `S2-P2.5b` ya quedaron absorbidos en una misma lectura mecanística: `concat`, `attn_bias`, `xattn` y `pca` dieron `12/12` condiciones `≈ D0` o peores, con `V4-lin + attn_bias` y `V4-lin-pca` claramente por debajo. Ese null mecanístico inicial ya puede darse por cerrado con CI. `S2-P3` ya también quedó corrido y consolidado en su primera pasada con `WavLM-Large` frozen: `P3-D0` cerró con `S=78.8% @ ep15`, `P3-A4-16k-pca` con `S=78.2% @ ep25`, `P3-V4-lin-pca` con `S=76.8% @ ep28` y `P3-H-series-pca` con `S=75.6% @ ep25`. Ningún brazo descriptor-guided superó a `P3-D0`; el cambio de régimen elevó levemente el baseline respecto de `P2-D0=77.8%`, pero no desplazó el null descriptorial. El trabajo siguiente ya no es “terminar P3”, sino cerrar la comparación `P2 vs P3` sobre representaciones y decidir si el frente merece una nueva iteración o un cierre provisional.
> **Voz Expresiva Phideus**: el frente ya no está en “ZH corrido, cierre pendiente”. Ese contraste ya quedó cerrado. `Fase 0A`, `0B` y `1 EN` habían dejado señal descriptorial específica, baseline `WavLM-large` fuerte y un caso positivo robusto para `concat` en `N-strict`; el cierre `EN ↔ ZH` agrega ahora una lectura más matizada y más útil: en **`N-adapt`** `concat` y `FiLM` replican limpio cross-language, mientras en **`N-strict`** el lift inglés no transfiere y `film/xattn` se vuelven negativos en `ZH`. La lectura vigente del frente ya no es “replicó” o “no replicó”, sino otra: el patrón descriptor-guided transfiere entre lenguas bajo anclaje per-speaker en test, pero no sostiene una ventaja robusta en speaker-independent estricto. La decisión abierta ya no es metodológica sino estratégica: cerrar Fase 1 con este matiz, abrir una `Fase 1.2` o saltar a `MSP-Podcast`.
> **Atención Armónica**: el frente nuevo ya cerró `Fase 0`, `0.5` y `0.6`. `Fase 0` dejó un resultado dual: el **pair-state** es el salto grande (`B-minus ≫ A-rich`), y la estructura del **triangle** no domina `IID` ni `OOD-regime` frente a `B-local`, pero sí gana en `OOD-poly` bajo métricas threshold-free. `Fase 0.5` corrigió el primer caveat: el problema de `B` en `OOD-poly` **no** era la transferencia de `τ` (`gap_dist≈0`), sino la fragilidad de `connected-components`; con `agglo_true_k`, `B` pasaba a ser el mejor modelo en esa celda (`ARI 0.605` vs `0.482` en `B-local`). `Fase 0.6` cerró el paso que faltaba: con clusterers globales deployables, `B` ya recupera esa ventaja de forma extraíble en `OOD-poly` (`spectral` y `agglo_estimated_k` comunes, CI excluye `0`), mientras `cc_bridge_prune` confirma el diagnóstico de puentes pero no alcanza. La implicancia vigente es más precisa que hace un día: la representación relacional de `B` no solo generaliza mejor, sino que ya puede leerse con una familia deployable concreta; el caveat que queda es la **subestimación de `k`**, no la calibración de `τ`.
> **Gate 9 / revisión A10 / Gate 10**: la rama retrospectiva ya dejó de ser solo plan. `A7r` cerró con `70.4% @ e29` y `A9r` con `71.6% @ e30`. `A10a-e` ya quedaron formalmente cerrados en una banda muy estrecha: `a10ar=70.6% @ e28`, `a10br=70.0% @ e29`, `a10cr=69.2% @ e29`, `a10dr=70.2% @ e30` y `a10er` con **mejor observado `71.8% @ e27`** y cierre final `70.2% @ e30`. Esa compresión retrospectiva ya recibió su contraste causal completo: **Gate 10 quedó terminado `9/9`** y dejó un ranking estable `concat > FiLM/pca >> attn_bias`. El mejor arm fue `a7-concat=76.4% @ e29`; los tres brazos `concat` convergieron en banda `75-76%`, `FiLM/pca` quedó en `72-74%` y `attn_bias` cerró mucho más abajo (`55.8-59.6%`). La inferencia útil del corte es más dura que la lectura provisional: en esta rama el mecanismo domina sobre el descriptor, y aun el mejor `concat` queda por debajo de `ctrl=79.2%` y por debajo también de `d4a4=84.0%±2.7pp`.
> **Escalón 3**: ya no está en “dataset listo, falta correr todo”. `E3-P0` ya materializó el banco canónico con `6,016` scenes, `E3-P1` ya cerró aprendibilidad por `ratio`, `E3-P2` ya dejó dos referencias `L0` con roles distintos y `E3-P4` ya dejó una lectura informativa sobre probes en latente plano. La línea geométrica ya también tuvo su primera pasada completa: `P5-flat` no desplazó a `P2-flat`, pero sí mostró contribución causal de la rama toroidal; `P5-cqtshift` pasa a ser el mejor brazo OOD actual (`scale-OOD S=0.508`, `equiv-OOD S=0.472`); `P6-flat` sale negativo; y `P6-cqtshift`, aun con estructura toroidal casi perfecta, no supera a `P5-cqtshift` en las métricas OOD primarias. La lectura vigente del frente queda así: `P2-flat` como baseline general, `P5-cqtshift` como mejor brazo geométrico/OOD y `P6` como hipótesis pura no ganadora bajo la receta actual.
> **Escalón 4**: `ECG <-> PPG` queda como expansión fisiológica fuera de acústica. No desaparece, pero ya no es el frente siguiente inmediato del programa.
> **Ground truth proporcional / PPU**: una campaña de cincuenta y cinco olas y ciento ocho investigaciones independientes no encontró una geometría universal ni un corpus único. Organizó fuentes analíticas, simulación generativa, cámaras físicas y evidencia externa bajo falsación adversarial y adjudicación ciega. Las Olas 20–48 construyeron contratos para objeto, unidades, equivalencia, identificabilidad, transformación, autoridad material y autoridad de la relación. La Ola 49 materializó un benchmark clásico sellado; las Olas 50–52 separaron conjunto, elección y utilidad contractual; la Ola 53 transportó incertidumbre marginal; y la Ola 54 mostró que un posterior conjunto mejora NLL, cardinalidad e interacciones sin garantizar una mejor decisión. La Ola 55 probó la interfaz conservadora más simple sobre una realización fresca: el selector primario eligió `hard_only`, el puente quedó idéntico al baseline y sólo `4/9` condiciones pasaron. Una selección global alternativa redujo regret (`-0.0109`) y elevó compatibilidad (`+0.0184`), pero perdió accuracy (`-0.0117`), cambiando `5/7` signos predeclarados; el resultado es selector-sensitive. La deuda ya no pide otro umbral ni un encoder mayor, sino una compuerta condicional de baja capacidad que estime cuándo confiar en el posterior, además de un brazo separado para los cinco conjuntos aún ausentes. El patrón acumulado separa `contrato material del target -> autoridad de relación/equivalencia -> manifest/attestation -> proposer o selector -> oracle/checker -> conjunto identificado con incertidumbre -> política de decisión -> abstención -> audit/replay -> reader`. No hay `A19`, nuevo `P2*`, arquitectura promovida ni decisión GO/NO-GO.
> **Libro HIT**: la consolidación teórica larga del programa ya no vive dentro de este repo. Ahora se mantiene en el repositorio público [AlterMundi/harmonic-information-theory](https://github.com/AlterMundi/harmonic-information-theory), con edición web en [hit.altermundi.net](https://hit.altermundi.net/). Allí el manuscrito ya quedó consolidado en 191 páginas, con build limpio y cierre editorial pre-ISBN.
> **Skills compartidas**: el repo ya expone un índice público en `Documents/Skills/README.md` para skills reutilizables de operación HPC/SLURM.
> **Política documental vigente**: la capa canónica viva del repo debe reflejar este estado actual; los planes viejos, bitácoras fechadas y frentes pausados/cerrados se preservan como registro histórico y no deben reescribirse retrospectivamente para simular que siempre dijimos lo que hoy sabemos.
> **Decisión operativa vigente**: (1) tratar `Test02` como cierre causal del argumento de capacidad, (2) leer `13G-B` como resultado negativo/generativo genérico y usar `Test11` para sostener el hallazgo mecanístico del cuello de proyección, (3) tratar la rama `Transkun+A4` de Gate 6 como cierre negativo útil (`Exp A` y `Exp B`) y conservar `Exp C` como única línea downstream abierta, (4) leer Gate 8 ya como línea cerrada con una señal positiva real sobre proyección (`pcd > pca > pcd-zero > pcm > ctrl`), (5) sostener Escalón 1 con cierre training-seed homogéneo real en `d4a4=84.0%±2.7pp`, (6) tratar Escalón 2 como frente principal ya con null mecanístico inicial cerrado y `S2-P3` **ya completado en su primera pasada**, (7) usar Gate 9 / `A10` como evidencia retrospectiva ya disponible pero subordinada, y (8) leer Gate 10 ya como contraste causal cerrado donde el mecanismo domina sobre el descriptor (`concat > FiLM/pca >> attn_bias`).
> **Encuadre estrategico**: Gate 5A deja de ser barrido bloqueante y queda absorbido operacionalmente por Gate 8; Gate 6 AMT conserva su rol downstream, pero ya con una poda metodológica más fuerte tras el negativo combinado de `Exp A` y `Exp B`; Gate 7.1 ya no es campaña pendiente sino evidencia para acotar hipótesis; Escalón 2 pasa del cierre mecanístico inicial a `S2-P3` ya implementado sobre `WavLM-Large` frozen; Gate 10 deja de nacer y pasa a cerrar como mecanismo de desconfusión dentro de Escalón 1 retrospectivo, con dominio claro del mecanismo sobre el descriptor; Escalón 3 pasa a ser Lissajous como banco sintético ya abierto y ya corrido en su primera línea geométrica, con `P2-flat` fijado como baseline general, `P5-cqtshift` como mejor brazo OOD y `P6` como hipótesis pura no ganadora en esta receta; y Escalón 4 queda como expansión fisiológica posterior.
> **Infraestructura**: estrategia distribuida LOCAL+UNC activa; foundation lock publicado (`v0.1.0-foundation`).

\* `shuffled` se tomó como cierre operativo por convergencia clara en `e20`.

> [!TIP]
> El mapa transversal de frentes, dependencias y bifurcaciones se mantiene en la
> [Wiki viva de Phideus](../05_WIKI/README.md). La vista humana está en
> [MAPA_VISUAL_DEL_PROGRAMA.md](../05_WIKI/MAPA_VISUAL_DEL_PROGRAMA.md) y el
> contexto denso para agentes en [LLM_CONTEXT.md](../05_WIKI/LLM_CONTEXT.md).

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

Cierre canónico posterior (5 training seeds): `d4a4 = 84.0% +/- 2.7pp`.

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
| Gate 5B Test05 (multi-seed, UNC+LOCAL) | `D0`, `a4r`, `d4-a4r`, `d4a4` × `5` seeds | **cerrado** (`20/20` total) |
| Gate 5B Test02 (parameter-matched, UNC) | `real`, `random`, `shuffled`, `zero` | **cerrado (4/4)** (`83.0%`, `75.0%`, `73.6%`, `73.6%*`; misma arquitectura, misma receta) |
| Gate 5B Test13G-A (generative encoder) | `D0`, `λ={0.03,0.1,0.3}` | **cerrada** (`best_S≈64.4-64.6%`, `PR F1≈0.11`; ruta `z=256 -> piano-roll` descartada) |
| Gate 5B Test11 Pre-Proj A/B | `D0`, `a4r`, `d4a4`, `d4-a4r` | **cerrado** (`retention ratio`: `0.597`, `0.712`, `0.770`, `0.748`) |
| Gate 5B Test13G-B (post-hoc pre-pooling) | `D0`, `d4a4`, `a4r`, `d4-a4r` (+ control `D0 pool-to-188`) | **cerrado** (`F1≈0.10` en todos; `D0 pool-188` levemente superior, sin ventaja descriptor-guided) |

---

### Escalón 2 (Speech ↔ EGG) — apertura efectiva

| Artefacto / decisión | Estado verificado |
|----------------------|-------------------|
| Dataset | French Lombard `v1.1` inspeccionado localmente (`38` speakers, `9,120` clips, ~`20h`) |
| Split canónico | `28/5/5` speakers (`train/val/test`), balanceado por género |
| Manifest | `data/lombard/manifest.json` (`9,120` clips) |
| Segment index | `data/lombard/segment_index.json` (`108,536` segmentos; `noise0`: `19,910/3,624/3,629`) |
| Audit de alineación | `data/lombard/alignment_audit.json` (`lag_correction_samples=0`, `voiced_threshold=0.1494`, `0` clipping) |
| Baseline lineal | `data/lombard/p1_results/p1_results_noise0.json` (`CCA S=64.4%`, `raw cosine S=46.8%`, random `7.8%`) |
| Control neural `D0` | `data/lombard/d0_control/` cerrado (`best S=77.8% @ ep25`, `CI [72.0%, 80.8%]`) |
| Protocolo piloto | `16 kHz`, ventanas de `2s`, `hop=0.5s`, positivo = misma ventana temporal del mismo clip |
| Próximo paso | cerrar el diagnóstico comparativo `P2 vs P3` (`CKA`, probes lineales, lectura representacional) y decidir si el frente amerita una nueva iteración o un cierre provisional |

Escalón 2 todavía no tiene claim positivo de descriptor, pero ya tampoco está en “posibilidad abstracta” ni en simple baseline. Tiene población congelada, split por speaker, segmentación canónica, auditoría de sincronía, baseline lineal, baseline neural, una fase concat cerrada, un factorial attention-based ya interpretado, una fase `pca` ya cerrada `3/3` y una primera pasada `P3` ya consolidada. La lectura prudente del corte es que, bajo `concat`, `attn_bias`, `xattn`, `pca` y ahora también bajo el régimen foundation-encoder `WavLM-Large` frozen, los descriptores testeados no mejoraron retrieval sobre el brazo control correspondiente. La ambigüedad ya no pasa por “falta correr P3”, sino por si el contraste `P2 vs P3` reordena la interpretación representacional del null o confirma su estabilidad.

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
| Gate 5A / Gate 8 | Mixto: Gate 8 cerrado, backlog Gate 5A oportunista | `ctrl=79.2%`, `pcm=80.0%`, `pcd-zero=81.8%`, `pca=82.6%`, `pcd=84.2%`; Gate 8 ya cerró `5/5`, mientras combinatorios `t3-wt` y backlog legado siguen fuera de ruta crítica |
| Gate 5B showcase científico | **Cerrado** | `Test02` 4/4, `Test13G-B` completo y cierre formal de la Línea B de Escalón 1-C |
| Gate 6 AMT | Activo | `Exp 0` completo; `Exp C` con brazo local `a4r` ya completo (`best_F1=0.1570 @ ep50`); `Exp A` y `Exp B` ya cierran negativamente la rama `Transkun+A4` |
| Gate 7 | Acotado / en decisión | `Exp 7.0` completo; `7.1a` también cerrado (`D0_mert=75.0% ≈ D0_lite=75.2%`); `7.1b` queda condicional y ya no bloquea ninguna decisión de programa |
| Gate 9 natural harmony | Datos disponibles / oportunista | `a7r=70.4%`, `a9r=71.6%`; `A10a-e` ya cerraron en banda `69.2–71.8%`, con `a10er` mejor observado en `71.8% @ e27`; la rama ya motivó Gate 10 |
| Gate 10 mechanism sweep | Completado | Piloto `3x3` (`a7`, `a10a`, `a10d` × `concat/pca/attn_bias`) ya cerrado `9/9`; ranking final `concat > pca >> attn_bias`, con `a7-concat=76.4%` como mejor arm |
| Escalón 2 | Activo | `S2-P0` y `S2-P1` completos; `S2-P2-control`, `S2-P2-main`, `S2-P2.5` y `S2-P2.5b` ya integrados en un null mecanístico inicial cerrado; `S2-P3` ya quedó corrido con `P3-D0=78.8%`, `P3-A4-16k-pca=78.2%`, `P3-V4-lin-pca=76.8%`, `P3-H-series-pca=75.6%`, sin lift descriptorial sobre el brazo control del régimen |
| Voz Expresiva Phideus | Activo exploratorio | `Fase 0A`, `0B`, `1 EN` y cierre `EN↔ZH` ya consolidados; positivo cross-language acotado a `N-adapt`, con lectura `null/negativa` en `N-strict` |
| Atención Armónica | Fase 0, 0.5 y 0.6 cerradas | `v2.1` pasó gate final y `Fase 0` cerró `54/54`: pair-state aporta fuerte; `triangle` gana `OOD-poly`; `Fase 0.5` mostró que el cuello era `connected-components`; `Fase 0.6` mostró que `spectral/agglo` deployables recuperan a `B` en `OOD-poly`, con caveat de subestimación de `k` |
| Escalón 3 | Activo con línea geométrica ya corrida | Audio XY <-> figuras de Lissajous como banco sintético ya materializado (`6,016` scenes), con baseline dual consolidado y primera comparación `P2/P5/P6` ya leída |
| Escalón 4 | Conceptual | ECG <-> PPG como expansión fisiológica fuera de acústica; sin frente activo todavía |

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
3. **Leer Gate 6 con poda metodológica real**: `Exp C` ya dejó una referencia local completa (`a4r`, `best_F1=0.1570 @ ep50`), y la rama `Transkun+A4` ya quedó cerrada negativamente por `Exp A` + `Exp B`.
4. **Mantener Gate 7 como línea acotada de decisión**: `Exp 7.0` ya resolvió la pregunta barata del lado audio y `7.1a` ya devolvió su resultado útil; si Escalón 1 vuelve a absorber recursos, la discusión ya no parte de cero.
5. **Sostener `Exp C` como única línea downstream abierta**: la pregunta ya no es si falta correr `Transkun+A4`, sino si el decoder serio sobre features congeladas devuelve algo que retrieval no capturó.
6. **Usar el negativo combinado de `Exp A` y `Exp B` como acotación real**: A4 no mostró ventaja útil en `Transkun` ni en régimen base ni bajo degradación en esta receta.
7. **Sostener Escalón 2 como frente principal ya reencuadrado**: `S2-P2-control`, concat, `P2.5`, `P2.5b` y la primera pasada `P3` ya absorbieron el contraste mecanístico y el contraste de encoder del frente; lo que sigue ya no es correr nuevos brazos ciegamente, sino consolidar la comparación `P2 vs P3` y decidir si corresponde una nueva iteración o un cierre provisional.
8. **Leer Gate 8 como línea paralela ya cerrada**: el frente ya no está a mitad de camino. La evidencia completa `pcd > pca > pcd-zero > pcm > ctrl` refuerza que el bottleneck de proyección era una intervención downstream bien elegida, aunque siga sin desplazar a Escalón 2 como foco principal.
9. **Usar `13G-B`, `7.1a` y `Exp B` como resultados negativos útiles**: los tres acotan dónde no está automáticamente la ventaja descriptor-guided y desplazan la atención hacia proyección, mecanismo e interpretación causal más fina.
10. **Mantener Gate 9 como reapertura retrospectiva ya informativa y leer Gate 10 como contraste causal ya cerrado**: `A7r/A9r` y `A10` ya dejaron datos suficientes para sospechar que el mecanismo domina; Gate 10 confirmó esa sospecha con `concat > pca >> attn_bias` y spread intra-mecanismo mucho menor que el inter-mecanismo.
11. **Tratar Escalón 3 como frente ya abierto también en su línea geométrica, pero con una primera lectura ya cerrada**: el banco canónico ya existe, `P2` ya dejó dos referencias `L0`, `P4` ya quedó documentado y `P5/P6` ya fueron corridos. La lectura vigente no es “seguir empujando el toro puro”, sino conservar `P2-flat` como baseline general, `P5-cqtshift` como mejor brazo geométrico/OOD actual y `P6` como hipótesis pura no ganadora bajo la receta actual.

Marco estrategico inmediato:

1. Gate 5B ya quedó cerrado como cierre principal de Escalón 1-C.
2. Escalón 2 (Speech <-> EGG) ya está abierto operativamente y pasa a ser el foco principal del programa.
3. Gate 5A continúa como línea paralela y oportunista, ya reexpresada operativamente como Gate 8 para conditioned projections: no bloquea Escalón 2 y solo absorbe recursos libres.
4. Gate 6 AMT abre una validación downstream concreta: no reemplaza Escalón 2, pero sí prueba si la ventaja descriptor-guided sobrevive fuera del retrieval.
5. Gate 7 queda en modo de resolución acotada: `Exp 7.0` ya está cerrado y `Gate 7.1` solo se justifica como piloto corto de decisión, no como nueva campaña principal.
6. Gate 9 reabre la pregunta por armonía natural dentro de música solo como rama retrospectiva y secundaria, acompañada por la revisión `A10` para separar mejor hipótesis dirigidas, controles genéricos y variantes continuas ontology-free.

---

## Frentes y Documentos

| Documento | Rol |
|-----------|-----|
| `README.md` | Entrada principal del repositorio |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md` | Plan maestro vigente |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/INDEX_BIAS_CONTROL.md` | Navegación del frente |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/RANKING_DESCRIPTORES_UNIFICADO.md` | Tabla canónica corta+larga |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/09_GATE_4_5_LR_SCHEDULE_OPTIMIZATION/README.md` | Gate 4.5 (scheduler/LR) |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/12_GATE_6_AMT/README.md` | Gate 6 AMT (validación downstream) |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/13_GATE_7_MERT_PROBE/README.md` | Gate 7 (probe lineal MERT-large) |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/14_GATE_7.1/README.md` | Gate 7.1 (`7.1a` cerrado, `7.1b` condicional) |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/15_GATE_8_CONDITIONED_PROJECTIONS/README.md` | Gate 8 (conditioned projections, ya cerrado `5/5` con `pcd > pca > pcd-zero > pcm > ctrl`) |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/16_GATE_9_NAT_HARM_DESCRIPTOR/PLAN_GATE9.md` | Gate 9 como piloto retrospectivo sobre armonía natural en música (`A7r/A9r`) |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/16_GATE_9_NAT_HARM_DESCRIPTOR/PLAN_GATE9_DESCRIPTOR_REVISION.md` | Revisión `A10`: taxonomía continua ontology-free y extensión futura para música / voz |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/17_GATE_10_MECHANISM_SWEEP/README.md` | Gate 10: barrido causal descriptor × mecanismo para audio-only |
| `Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md` | Estado canónico de Escalón 2 (`S2-P0/P1` completos, null mecanístico inicial cerrado y `S2-P3` ya completado en su primera pasada) |
| `Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/PREDICCIONES_EPISTEMOLOGICAS_P25.md` | Preregistro interpretativo de `S2-P2.5`: bootstrap pareado sobre `Delta`, matriz de predicciones y guardrails para nulls |
| `Documents/01_FRENTES_ACTIVOS/Voz_Expresiva_Phideus/README.md` | Estado canónico del frente de voz: cierre `EN↔ZH` ya consolidado, con positivo acotado a `N-adapt` y lectura `null/negativa` en `N-strict` |
| `Documents/01_FRENTES_ACTIVOS/Voz_Expresiva_Phideus/ROADMAP_VOZ_EXPRESIVA_PHIDEUS.md` | Roadmap del frente de voz: Carril A ya con `0A/0B/1 EN/ZH` y cierre translingüístico mínimo cerrado; siguiente decisión = `Fase 1.2` vs `MSP-Podcast` |
| `Documents/01_FRENTES_ACTIVOS/Voz_Expresiva_Phideus/PLAN_FASE_1_ZH.md` | Plan archivado de la réplica `ZH`, preservado ya como fase ejecutada y absorbida en el cierre cross-language |
| `Documents/01_FRENTES_ACTIVOS/Voz_Expresiva_Phideus/EXPLICACION_PIPELINE_FASE_1.md` | Explicación pedagógica del pipeline `WavLM` + descriptor `A` + mecanismos `concat/FiLM/xattn` |
| `data/voz_expresiva/1/REPORTE_1.md` | Cierre empírico de `Fase 1` sobre `ESD` English: baseline `WavLM-only`, comparación mecánica y lectura `CKA` |
| `Documents/01_FRENTES_ACTIVOS/Atencion_Armonica/README.md` | Estado canónico local del frente: `Fase 0`, `0.5` y `0.6` cerradas, resultado dual pair-state/triangle y lectura deployable actual del clusterer |
| `Documents/01_FRENTES_ACTIVOS/Atencion_Armonica/Explicacion_arq_RNA_codex.md` | Explicación conceptual de `Harmonic Pairformer`: plano token/par, `triangle update`, geometría relacional y caminos derivados |
| `Documents/01_FRENTES_ACTIVOS/Atencion_Armonica/Explicacion_fase_0_5_calibracion_codex.md` | Explicación conceptual del último hallazgo: el cuello de `OOD-poly` está en `connected-components`, no en `τ` |
| `Documents/01_FRENTES_ACTIVOS/Atencion_Armonica/Explicacion_fase_0_6_clusterer_deployable_codex.md` | Explicación conceptual de `Fase 0.6`: por qué la ventaja de `B` ya es extraíble con clusterers globales deployables y qué caveat queda en `k` |
| `Documents/01_FRENTES_ACTIVOS/Atencion_Armonica/ROADMAP_ATENCION_ARMONICA.md` | Roadmap del frente incubado: `Fase 0.6` ya cerrada, clusterers globales deployables ya probados y foco siguiente desplazado hacia Stage B / detección real |
| `Documents/01_FRENTES_ACTIVOS/Atencion_Armonica/PLAN_FASE_0_5_CALIBRACION.md` | Plan ejecutado del post-audit: preservación de artefactos, re-run determinista y auditoría calibrador×clusterer |
| `Documents/01_FRENTES_ACTIVOS/Atencion_Armonica/PLAN_FASE_0_v2_1.md` | Plan operativo de `Fase 0` ya ejecutado: diseño `v2.1`, combo congelada, gate `PASS`, training completo y resultado threshold-free |
| `Documents/01_FRENTES_ACTIVOS/ESCALON_3/README.md` | Estado canónico de Escalón 3: baseline dual consolidado y línea geométrica `P5/P6` ya leída |
| `Documents/01_FRENTES_ACTIVOS/ESCALON_3/Resultados_E3_P4.md` | Lectura consolidada de `P4`: resultado informativo sobre lectura en latente plano, sin veto suficiente sobre `P5/P6` |
| `Documents/01_FRENTES_ACTIVOS/ESCALON_3/Resultados_E3_P5_P6.md` | Lectura consolidada de la primera pasada geométrica completa: `P5-cqtshift` mejor brazo OOD actual, `P6` no ganador |
| `Documents/01_FRENTES_ACTIVOS/ESCALON_3/PLAN_E3_P5_P6_GEOMETRIA_NO_PLANA.md` | Especificación metodológica de la línea geométrica completa de Escalón 3 (`P5` mixto y `P6` toroidal completo) |
| `experiments/escalon3/generate_lissajous_dataset.py` | Generador reproducible del banco canónico de scenes Lissajous |
| `data/escalon3/scenes/` | Dataset materializado de Escalón 3 (`6,016` scenes; splits IID + OOD ya generados) |
| [AlterMundi/harmonic-information-theory](https://github.com/AlterMundi/harmonic-information-theory) | Repositorio público del libro HIT: formulación larga del programa, arquitectura editorial, fuente LaTeX y edición web en `hit.altermundi.net` |
| `Documents/Skills/README.md` | Índice público de skills compartidas del proyecto |
| `MARCO_EPISTEMOLOGICO_PHIDEUS.md` | Posición metodológica estable sobre Phideus como programa de investigación |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_UNC.md` | Estrategia distribuida LOCAL+UNC |
| `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md` | Evolución histórica de representaciones |
| `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/CATALOGO_NARRATIVO_DESCRIPTORES_RATIOS_PHIDEUS.md` | Catálogo vivo de descriptores |
| `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/GEOMETRIA_PROPORCIONAL_BASES_DE_VERDAD.md` | Síntesis durable de ground truth estratificado y programa PPU/NHG |

Nota operativa:
- Foundation lock publicado en GitHub Release: `v0.1.0-foundation` (`foundation_locked_e25.pt`, MD5 `ddb2ebf7075eec4dcec1628341ec4942`).

---

*Documento revisado el 2026-09-03. El corte experimental general sigue siendo el consolidado al 2026-07-02; la campaña proporcional llegó a cincuenta y cinco olas. La Ola 55 mostró que una compuerta global conservadora no convierte de manera estable la ventaja representacional del posterior conjunto en una mejora conjunta de decisión. El siguiente contraste debe condicionar la confianza en el posterior sin ampliar el encoder. No constituye una PPU validada ni una decisión GO/NO-GO.*
