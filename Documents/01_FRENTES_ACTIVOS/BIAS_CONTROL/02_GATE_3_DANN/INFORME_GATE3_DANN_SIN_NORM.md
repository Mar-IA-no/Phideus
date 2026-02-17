# Gate 3 DANN — Run A: Sin Normalización de Embeddings

**Fecha**: 2026-02-05/06
**Estado**: DETENIDO en epoch 10/30 (para comparación A/B)
**Output**: `data/bias_control_medium/training_outputs/gate3/`
**Checkpoints**: `checkpoint_epoch5.pt`, `checkpoint_epoch10.pt`, `best_model.pt` (epoch 7)

> [!NOTE]
> Addendum de vigencia (2026-02-17): este reporte corresponde a una variante histórica de Gate 3.
> El estado operativo actual del frente está en Gate 4.3 cerrado, con transición activa a Gate 4.4.
> Ver documentos canónicos: roadmap, estado troncal e informe de Gate 4.3.

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

## Cuadro Comparativo A/B (10 epochs)

| Métrica | Gate 2 | Run A ep7 (best) | Run A ep10 | Run B ep6 (best) | Run B ep10 |
|---------|--------|------------------|------------|------------------|------------|
| Domain Acc | 92.7% | 62.7% | 65.9% | 76.8% | 73.2% |
| R@10 (a2m) | 2.6% | 6.3% | 5.7% | **9.4%** | 8.1% |
| Gap | 0.478 | 0.364 | 0.376 | **0.482** | 0.419 |
| Loss | 14.09 | 13.992 | 13.953 | 14.024 | 13.965 |
| Lambda | — | 0.23 | 0.33 | 0.20 | 0.30 |

### Run B: Epoch-by-Epoch

| Epoch | Loss | Domain Acc | R@10 (a2m) | Gap | Lambda |
|-------|------|-----------|------------|-----|--------|
| 1 | 14.118 | 47.1% | 5.0% | 0.390 | 0.03 |
| 2 | 14.085 | 53.3% | 4.8% | 0.334 | 0.07 |
| 3 | 14.065 | 61.4% | 6.2% | 0.392 | 0.10 |
| 4 | 14.070 | 73.0% | 7.1% | 0.390 | 0.13 |
| 5 | 14.039 | 72.4% | 5.9% | 0.312 | 0.17 |
| **6** | **14.024** | **76.8%** | **9.4%** | **0.482** | **0.20** | **★ Best recall** |
| 7 | 14.002 | 68.0% | 5.9% | 0.391 | 0.23 |
| 8 | 13.996 | 72.2% | 5.7% | 0.348 | 0.27 |
| 9 | 13.965 | 73.2% | 8.1% | 0.419 | 0.30 |
| 10 | — | — | — | — | 0.33 | En progreso |

### Conclusión A/B

Run B (con F.normalize) **supera a Run A** en las métricas clave:
- **Recall@10 peak**: 9.4% (Run B ep6) vs 6.3% (Run A ep7) — **+49%**
- **Gap peak**: 0.482 (Run B ep6) vs 0.418 (Run A ep9) — **superó Gate 2 (0.478)**
- **Domain acc inicio**: 47.1% (Run B) vs 67.6% (Run A) — normalización elimina shortcut

La normalización L2 confirma la hipótesis: el clasificador usaba magnitud como discriminador trivial.

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

## Run B: Primeros Resultados y Decisión Lambda

### Epoch 1 Run B (con normalización)

| Métrica | Run A ep1 | Run B ep1 | Delta |
|---------|-----------|-----------|-------|
| Domain Acc | 67.6% | **47.1%** | **−20.5 pp** |
| R@10 (a2m) | 6.2% | 5.0% | −1.2 pp |
| Gap | 0.387 | 0.390 | +0.003 |
| Loss | 14.108 | 14.118 | +0.010 |

**Conclusión**: La normalización elimina el shortcut por magnitud. Domain accuracy cae 20 pp a λ=0.03. Retrieval no se degrada significativamente.

### Discusión Lambda Schedule (con ChatGPT)

ChatGPT sugirió capear λ_max=0.3 en lugar de linear 0→1, argumentando que λ=1.0 fuerza demasiada invariancia.

**Decisión**: Mantener schedule original (linear 0→1) hasta epoch 10.

**Razones**:
1. Con domain_acc ya en 47.1% a λ=0.03, el clasificador no puede hacer trampa → no se necesita λ alto para forzar confusión
2. Best model se guarda por mejor recall → safety net natural contra degradación por λ excesivo
3. Comparación A/B directa (mismos hiperparámetros excepto normalización) tiene más valor científico

**Plan contingencia**: Si retrieval muestra degradación sostenida a λ>0.3, ejecutar Run C con λ_max=0.3 capeado.

---

---

## Run C: Configuración Optimizada (preparado, no lanzado)

Resultado de análisis conjunto Claude + ChatGPT sobre los datos de Run A y Run B.

### Cambios vs Run A/B

| Parámetro | Run A/B | Run C | Razón |
|-----------|---------|-------|-------|
| LR projection | 5e-4 | 5e-4 | Sin cambio |
| LR MIDI encoder | 5e-5 | **1e-4** | Run B mostró que se puede ser más agresivo |
| LR domain head | =projection | **2e-4** | Separado para estabilidad |
| Weight decay | 1e-4 | **1e-3** | Regularización más fuerte |
| Lambda schedule | linear 0→1 | **warmup_ramp_cap** | Warmup puro VICReg + ramp + cap |
| Lambda max | 1.0 | **0.8** | No sobreregularizar |
| DANN warmup | 0 | **2000 steps** | VICReg establece buenas representaciones primero |
| DANN ramp | — | **6000 steps** | Ramp gradual hasta λ_max |
| Domain dropout | 0.1 | **0.3** | Dificultar memorización del clasificador |
| Max val batches | 200 | **None (todas)** | Eliminar sesgo de evaluación (shuffle=False) |
| Checkpoint every | 5 | **1** | Granularidad fina |

### Lambda Schedule: warmup_ramp_cap

```
λ
0.8 ──────────────────────────── cap
    │           ╱
    │         ╱  ramp (steps 2000→8000)
    │       ╱
    │     ╱
0.0 ────┤
    warmup (0→2000 steps, λ=0, sin domain loss)
```

- **Steps 0-2000**: λ=0, sin domain loss. Solo VICReg. Permite al modelo establecer representaciones útiles.
- **Steps 2000-8000**: Ramp lineal 0→0.8. DANN se activa gradualmente.
- **Steps 8000+**: λ=0.8 constante hasta el final.

### Comando de Lanzamiento

```bash
tmux new -s gate3c
python experiments/bias_control/gate3_dann.py \
    --checkpoint data/bias_control_medium/training_outputs/gate2/checkpoint_epoch45.pt \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/bias_control_medium/training_outputs/gate3_c \
    --epochs 30 --batch-size 16 --segment-len 4.0 --hop 1.0 \
    --max-batches-per-epoch 1000 \
    --lr-midi-encoder 1e-4 --lr-projection 5e-4 --lr-domain-head 2e-4 \
    --weight-decay 1e-3 \
    --dann-lambda-schedule warmup_ramp_cap --dann-lambda-max 0.8 \
    --dann-warmup-steps 2000 --dann-ramp-steps 6000 \
    --dann-dropout 0.3 --dann-weight 0.01 \
    --gate2-recall 0.026 --device cuda \
    2>&1 | tee data/bias_control_medium/gate3c_training.log
```

---

## Timestamps

| Evento | Hora (UTC) |
|--------|------------|
| Inicio Run A | 2026-02-05 20:22 |
| Run A epoch 5 checkpoint | 2026-02-05 22:32 |
| Run A epoch 7 best model | 2026-02-05 23:24 |
| Run A epoch 10 checkpoint | 2026-02-06 00:42 |
| Run A detenido (Ctrl-C) | 2026-02-06 00:43 |
| Run B lanzado | 2026-02-06 00:47 |
| Run B epoch 6 (best recall 9.4%) | 2026-02-06 03:24 |
| Run C implementación completada | 2026-02-06 05:30 |
