# Gate 3 DANN — Run A: Sin Normalización de Embeddings

**Fecha**: 2026-02-05/06
**Estado**: DETENIDO en epoch 10/30 (para comparación A/B)
**Output**: `data/bias_control_medium/training_outputs/gate3/`
**Checkpoints**: `checkpoint_epoch5.pt`, `checkpoint_epoch10.pt`, `best_model.pt` (epoch 7)

---

## Configuración

| Parámetro | Valor |
|-----------|-------|
| Base checkpoint | Gate 2 `checkpoint_epoch45.pt` |
| Epochs | 30 (detenido en 10) |
| Batch size | 16 |
| Segment length | 4.0s |
| Hop | 1.0s |
| Max batches/epoch | 1000 |
| Max val batches | 200 |
| DANN weight | 0.01 |
| Lambda schedule | Linear 0→1 sobre total steps |
| LR projection | 5e-4 |
| LR MIDI encoder | 5e-5 |
| Gate 2 recall baseline | 0.026 |
| **Normalización pre-domain head** | **NO** |

---

## Métricas Epoch-by-Epoch

| Epoch | Loss | Domain Acc | R@10 (a2m) | Gap | Lambda | Notas |
|-------|------|-----------|------------|-----|--------|-------|
| 1 | 14.108 | 67.6% | 6.2% | 0.387 | 0.03 | |
| 2 | 14.082 | 74.0% | 5.5% | 0.335 | 0.07 | |
| 3 | 14.069 | 77.4% | 6.6% | 0.398 | 0.10 | Pico domain acc |
| 4 | 14.048 | 65.0% | 5.2% | 0.378 | 0.13 | |
| **5** | **14.031** | **65.2%** | **6.1%** | **0.367** | **0.17** | **Checkpoint** |
| 6 | 14.025 | 65.8% | 6.8% | 0.386 | 0.20 | |
| **7** | **13.992** | **62.7%** | **6.3%** | **0.364** | **0.23** | **★ Best model** |
| 8 | 13.981 | 63.1% | 4.6% | 0.336 | 0.27 | |
| 9 | 13.975 | 75.2% | 6.5% | 0.418 | 0.30 | Rebote domain acc |
| **10** | **13.953** | **65.9%** | **5.7%** | **0.376** | **0.33** | **Checkpoint** |

---

## Cuadro Comparativo Rápido (para copiar a Run B)

| Métrica | Gate 2 | Run A ep5 | Run A ep7 (best) | Run A ep10 | Run B ep5 | Run B ep10 |
|---------|--------|-----------|------------------|------------|-----------|------------|
| Domain Acc | 92.7% | 65.2% | **62.7%** | 65.9% | — | — |
| R@10 (a2m) | 2.6% | 6.1% | **6.3%** | 5.7% | — | — |
| Gap | 0.478 | 0.367 | **0.364** | 0.376 | — | — |
| Loss | 14.09 | 14.031 | **13.992** | 13.953 | — | — |
| Lambda | — | 0.17 | 0.23 | 0.33 | — | — |

> Rellenar columnas Run B a medida que avance el training con normalización.

---

## Análisis

### Domain Accuracy

```
92.7% ─── Gate 2 baseline (sin DANN)
       ↓
77.4% ─── Pico ep3 (clasificador se adapta antes que GRL)
       ↓
62.7% ─── Mínimo ep7 (★ best model)
       ↑
75.2% ─── Rebote ep9 (clasificador encuentra nuevo shortcut?)
       ↓
65.9% ─── ep10 (oscilando)
```

**Patrón**: No monotónico. El domain classifier oscila entre 62-77%, sugiriendo que encuentra y pierde shortcuts alternativamente. La hipótesis de GPT es que la **magnitud del embedding** actúa como discriminador trivial — si audio y MIDI tienen normas sistemáticamente diferentes, el clasificador puede re-descubrir la separación por magnitud aun cuando el GRL elimina features direccionales.

### Recall@10

```
Gate 2 baseline: 2.6%
Run A rango:     4.6% — 6.8%  (1.8× — 2.6× Gate 2)
Run A media:     5.9%         (2.3× Gate 2)
```

El DANN **no degrada retrieval** — de hecho lo mejora. Posible explicación: eliminar features superficiales de modalidad obliga al modelo a usar features semánticas para VICReg.

### Loss

Convergencia suave: 14.108 → 13.953 (−1.1% en 10 epochs). Sin explosiones ni inestabilidad.

### Gap

Oscila entre 0.335-0.418. Más bajo que Gate 2 (0.478) pero esperado — el DANN reduce la separación global al mezclar representaciones modales.

---

## Criterios GO/NO-GO (parcial, epoch 10)

| Métrica | Umbral | Run A ep10 | Status |
|---------|--------|-----------|--------|
| Domain accuracy | 50% ± 5% | 65.9% | ⏳ No alcanzado (pero bajó de 92.7%) |
| Recall@10 (global) | >= 2.6% | 5.7% | ✅ PASS (2.2×) |
| Recall@10 (pool 256) | >= 34.4% | Pending | Post-training |
| Hard neg accuracy | >= 80.4% | Pending | Post-training |

---

## Razón de Detención

Se detuvo en epoch 10 para implementar normalización L2 de embeddings antes del domain head (`F.normalize(embeddings, dim=1)`) y lanzar Run B como comparación A/B.

**Hipótesis**: La norma del embedding es un discriminador trivial de dominio. Sin normalización, el domain classifier puede usar magnitud para distinguir audio vs MIDI, limitando la eficacia del GRL.

**Run B**: `data/bias_control_medium/training_outputs/gate3_norm/`
- Mismo checkpoint base (Gate 2 epoch 45)
- Mismos hiperparámetros
- Única diferencia: `F.normalize` antes del domain head

El training puede resumirse desde `checkpoint_epoch10.pt` si se necesita continuar Run A.

---

## Timestamps

| Evento | Hora (UTC) |
|--------|------------|
| Inicio training | 2026-02-05 20:22 |
| Epoch 5 checkpoint | 2026-02-05 22:32 |
| Epoch 7 best model | 2026-02-05 23:24 |
| Epoch 10 checkpoint | 2026-02-06 00:42 |
| Ctrl-C (detención) | 2026-02-06 00:43 |
| Run B lanzado | 2026-02-06 00:47 |
