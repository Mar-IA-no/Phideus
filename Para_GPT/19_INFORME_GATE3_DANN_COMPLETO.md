# Gate 3 DANN — Informe Completo

**Fecha**: 2026-02-07
**Estado**: ✅ CERRADO — 4 Runs completados y evaluados
**Resultado**: **DANN no mejora sobre Gate 2 en ningún régimen probado**. Gate 2 (sin DANN) es el mejor checkpoint.

> [!NOTE]
> Addendum de vigencia (2026-02-14): documento de cierre histórico de Gate 3.
> Estado operativo actual: Gate 4.3 en ejecución (run causal de 6 brazos; `D0` y `D4` ya completados).
> Seguimiento vigente en `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md` y `Documents/00_TRONCAL/Proyecto_Estado_Actual.md`.

---

## 1. Resumen Ejecutivo

Se ejecutaron 4 runs de DANN (Domain-Adversarial Neural Network) sobre el checkpoint Gate 2 (epoch 45) para reducir la separabilidad modal (92.7%) detectada en Gate 2.5.

**Conclusión definitiva**: DANN no mejora el retrieval cross-modal. El mejor checkpoint DANN (Run C ep4, λ~0.3 transitorio) apenas empata con Gate 2. Run D (λ=0.3 sostenido) es **peor** que Gate 2 (-7pp R@10 a2m). La separabilidad modal no es el factor limitante del retrieval.

**Decisión**: Gate 3 cerrado. Avanzar a Gate 4 (Ratio Auxiliary View) usando Gate 2 como checkpoint base.

---

## 2. Configuración de los 4 Runs

| Parámetro | Run A | Run B | Run C | Run D |
|-----------|-------|-------|-------|-------|
| F.normalize pre-DANN | No | **Sí** | **Sí** | **Sí** |
| Lambda schedule | linear 0→1 | linear 0→1 | warmup_ramp_cap | warmup_ramp_cap |
| Lambda max | 1.0 (teórico) | 1.0 (teórico) | **0.8** | **0.3** |
| Warmup steps | - | - | 2000 | **1000** |
| Ramp steps | - | - | 6000 | **3000** |
| LR MIDI encoder | 5e-5 | 5e-5 | **1e-4** | **1e-4** |
| LR projection | 5e-4 | 5e-4 | 5e-4 | 5e-4 |
| LR domain head | =projection | =projection | **2e-4** | **2e-4** |
| Weight decay | 1e-4 | 1e-4 | **1e-3** | **1e-3** |
| Domain dropout | 0.1 | 0.1 | **0.3** | **0.3** |
| Val batches | 200 | 200 | **846 (todas)** | **846 (todas)** |
| Epochs | 10 | 10 | 27 (de 30) | **15** |
| Estado | Detenido | Completado | Detenido | ✅ Completado |

Todos parten del mismo checkpoint: `gate2/checkpoint_epoch45.pt`

---

## 3. Evaluación Comparativa (Structured Pool)

**Protocolo**: 256 candidatos (64 hard + 32 semi-hard + 159 random + 1 positivo), 500 queries, seed 42.

### 3.1 Retrieval Metrics

| Checkpoint | R@1 a2m | R@5 a2m | R@10 a2m | R@1 m2a | R@5 m2a | R@10 m2a |
|-----------|---------|---------|----------|---------|---------|----------|
| **gate2_ep45** | 4.4% | 20.8% | **34.4%** | 5.2% | 24.6% | 37.6% |
| runA_best_ep7 | 6.0% | 17.2% | 27.8% | 6.0% | 22.6% | 35.4% |
| runB_ep5 | 3.6% | 16.2% | 24.6% | 4.4% | 20.2% | 32.0% |
| runB_ep10 | 5.4% | 18.2% | 29.8% | 4.2% | 23.2% | 34.6% |
| **runC_best_ep4** | 5.8% | **21.8%** | **34.6%** | 6.0% | 22.6% | **39.2%** |
| runC_ep13 | **6.2%** | 20.0% | 32.2% | 6.0% | **24.6%** | 38.0% |
| runD_best_ep12 | **6.4%** | 17.0% | 27.4% | 6.2% | 23.6% | 36.4% |

### 3.2 MRR y Ranking

| Checkpoint | MRR a2m | MRR m2a | Mean Rank a2m | Median Rank a2m |
|-----------|---------|---------|---------------|-----------------|
| gate2_ep45 | 0.138 | 0.158 | 37.4 | 18.0 |
| runA_best_ep7 | 0.132 | 0.148 | 55.7 | 30.5 |
| runB_ep5 | 0.112 | 0.132 | 61.8 | 33.0 |
| runB_ep10 | 0.130 | 0.140 | 55.1 | 28.0 |
| **runC_best_ep4** | **0.148** | **0.159** | **39.6** | **19.0** |
| runC_ep13 | 0.144 | 0.163 | 48.2 | 22.0 |
| runD_best_ep12 | 0.134 | 0.158 | 52.7 | 26.0 |

### 3.3 Hard Negative Analysis

| Checkpoint | vs Same-Piece-Diff-Time | vs Random | Decision |
|-----------|------------------------|-----------|----------|
| gate2_ep45 | **80.4%** | **87.0%** | **GO** |
| runA_best_ep7 | 74.8% | 80.6% | GO |
| runB_ep5 | 70.4% | 72.4% | WEAK-GO |
| runB_ep10 | 73.6% | 78.2% | GO |
| **runC_best_ep4** | **81.2%** | 86.2% | **GO** |
| runC_ep13 | 76.6% | 80.8% | GO |
| runD_best_ep12 | 73.2% | 78.6% | GO (pero peor) |

### 3.4 Improvement over Random

| Checkpoint | R@1 a2m (×) | R@10 a2m (×) | R@1 m2a (×) | R@10 m2a (×) |
|-----------|-------------|-------------|-------------|-------------|
| gate2_ep45 | 11.3× | 8.8× | 13.3× | 9.6× |
| runA_best_ep7 | 15.4× | 7.1× | 15.4× | 9.1× |
| runB_ep5 | 9.2× | 6.3× | 11.3× | 8.2× |
| runB_ep10 | 13.8× | 7.6× | 10.8× | 8.9× |
| **runC_best_ep4** | **14.8×** | **8.9×** | **15.4×** | **10.0×** |
| runC_ep13 | 15.9× | 8.2× | 15.4× | 9.7× |
| runD_best_ep12 | 16.4× | 7.0× | 15.9× | 9.3× |

---

## 4. Análisis por Run

### 4.1 Run A (sin normalización)

- **Problema**: Domain classifier usaba la magnitud del embedding como shortcut trivial
- **Resultado**: Domain acc oscila 62-77% (no baja de 62%), retrieval empeora vs Gate 2
- **En structured pool**: R@10 a2m cae de 34.4% a 27.8% (-6.6pp), hard neg acc cae de 80.4% a 74.8%

### 4.2 Run B (con F.normalize)

- **Fix**: F.normalize elimina shortcut de magnitud
- **Resultado**: Métricas de training parecían mejores (R@10 9.4%* en pool de 3.2K), pero en structured pool pierde vs Gate 2
- **En structured pool**: Run B ep5 es el peor (24.6%), ep10 mejora a 29.8% pero sigue por debajo de Gate 2
- **Lección**: Los R@10 de training con pool chico (200 batches) eran engañosos

### 4.3 Run C (hiperparámetros optimizados)

- **Config**: 3 LR groups, warmup_ramp_cap, λ_max=0.8, dropout=0.3
- **Mejor epoch**: 4 (λ~0.3, antes del cap) — **único checkpoint que iguala/supera marginalmente Gate 2**
- **Después del cap (λ=0.8)**: Retrieval degrada progresivamente (ep13 ya pierde vs ep4)
- **Diagnóstico**: λ=0.8 es excesivo — sobre-regularización adversarial

### 4.4 Run D (λ=0.3 sostenido)

- **Config**: Misma que Run C pero λ_max=0.3, warmup=1000, ramp=3000, 15 epochs
- **Hipótesis**: Si Run C ep4 empata con Gate 2 en λ~0.3 transitorio, ¿λ=0.3 sostenido mejoraría?
- **Resultado**: **NO** — Run D es peor que Gate 2 en todas las métricas principales
- **En structured pool**: R@10 a2m cae de 34.4% a 27.4% (-7pp), hard neg acc cae de 80.4% a 73.2% (-7.2pp)
- **Training metrics**: gap bajó de 0.478 (Gate 2) a 0.352, domain acc oscila 57-73% (nunca llega a 50%)
- **Diagnóstico**: Incluso λ=0.3 sostenido degrada el retrieval. Run C ep4 funcionaba no *por* λ=0.3, sino *a pesar* de él — era suficientemente temprano en el training para no haber destruido señal todavía

---

## 5. Conclusiones

### 5.1 DANN no resuelve el problema de separabilidad modal

A pesar de que Gate 2.5 detectó 92.7% de separabilidad modal, forzar invariancia con DANN no mejora el retrieval. Esto sugiere que:

1. La información modal que el probe detecta **no es el factor limitante** del retrieval
2. O bien DANN destruye información útil al intentar removerla
3. O la separabilidad es un artefacto inofensivo (los embeddings son diferentes pero igualmente útiles para matching)

### 5.2 El mejor checkpoint DANN es marginal

Run C ep4 (R@10 a2m 34.6%, hard neg 81.2%) apenas supera Gate 2 (34.4%, 80.4%). La diferencia está dentro del ruido estadístico (±2pp con 500 queries).

### 5.3 Lambda siempre destruye

| Lambda | Run | R@10 a2m | Efecto en invariancia |
|--------|-----|---------|----------------------|
| 0.0 (Gate 2) | - | **34.4%** | 92.7% separable |
| ~0.3 transitorio | Run C ep4 | 34.6% | 69.6% (parcial) |
| **~0.3 sostenido** | **Run D** | **27.4%** | **57-73% (no estable)** |
| ~0.8 sostenido | Run C ep13 | 32.2% | 53-72% (no estable) |

**Conclusión definitiva**: Cualquier nivel de DANN sostenido degrada retrieval. Run C ep4 aparentaba funcionar porque el DANN aún no había tenido tiempo de destruir la señal.

### 5.4 Métricas de training vs structured pool

Las métricas R@10 del training loop (pool variable según val batches) son engañosas:
- Run B parecía mejor que Gate 2 (9.4% vs 2.6%) pero era por pool 4× más chico
- En structured pool idéntico, Run B pierde vs Gate 2

**Lección**: Siempre evaluar con structured pool para decisiones GO/NO-GO.

---

## 6. Decisión Final

### Gate 3: DANN CERRADO ❌

Run D confirmó que λ=0.3 sostenido es **peor** que Gate 2 (-7pp R@10 a2m). DANN en cualquier régimen probado degrada o, en el mejor caso, empata con el baseline.

### Checkpoint seleccionado para Gate 4

**Gate 2 `checkpoint_epoch45.pt`** (sin DANN) — el mejor checkpoint disponible:
- R@10 a2m: 34.4%, R@10 m2a: 37.6%
- Hard neg accuracy: 80.4%
- Gap: 0.478, MRR a2m: 0.138

### Siguiente paso: Gate 4 (Ratio Auxiliary View)

Reinyectar el "ratio insight" de Phideus como vista auxiliar sobre el embedding Gate 2, sin forzar invariancia modal.

---

## 7. Archivos en este directorio

| Archivo | Descripción |
|---------|-------------|
| `INFORME_GATE3_COMPLETO.md` | Este informe |
| `COMPARISON_GATE3.md` | Tabla comparativa generada por script |
| `INFORME_GATE3_DANN_SIN_NORM.md` | Informe detallado Runs A/B |
| `comparison_summary.json` | Resultados completos (JSON) |
| `gate2_ep45.json` | Evaluación Gate 2 baseline |
| `runA_best_ep7.json` | Evaluación Run A best |
| `runB_ep5.json` | Evaluación Run B epoch 5 |
| `runB_ep10.json` | Evaluación Run B epoch 10 |
| `runC_best_ep4.json` | Evaluación Run C best |
| `runC_ep13.json` | Evaluación Run C epoch 13 |
| `runC_training.log` | Log completo del training Run C |
| `runD_best_ep12.json` | Evaluación Run D best (structured pool) |
