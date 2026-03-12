# Gate 7 — MERT-large Linear Probe

**Estado**: EXP 7.0 COMPLETO (LOCAL, 2026-03-05)
**Fecha apertura**: 2026-03-05
**Motivación**: Gate 6 Exp C plateau (~F1=0.157) compatible con techo del encoder. Gate 7 estrecha la ambigüedad sobre el lado audio.

## Resultados Exp 7.0 — Probe Segment-Level (Ridge, 5 splits)

| Encoder | R²_global | ±std | hidden_size |
|---------|-----------|------|-------------|
| **MERT-v1-330M** | **0.850** | 0.126 | 1024 |
| MERTLite-D0 | 0.734 | 0.229 | 1024 |
| MERT-v1-95M | 0.659 | 0.178 | 768 |
| Null (shuffled) | -1.568 | — | — |
| Null (dummy) | -0.038 | — | — |

**Interpretación**: A4 (envolvente espectral por bandas) es linealmente accesible en todos los encoders. MERT-330M más accesible (+11.6pp sobre MERTLite-D0). Señal muy por encima de nulls. Ambigüedad reducida: el encoder era una limitación relevante, aunque no exclusiva.

**Nota sobre la target A4 usada**: Se usó *media de log-magnitud STFT por banda A4* (envolvente espectral), no el descriptor A4 z-scored interno (que tiene media ≈0 por construcción y sería degenerado para probe segment-level).

---

## Pregunta Central

¿Cuánto de la información del descriptor A4 está codificada de forma linealmente accesible en cada encoder de audio?

Encoders comparados:

| Encoder | Params | Origen |
|---------|--------|--------|
| `MERTLite (D0)` | ~60M | Entrenado con VICReg Gate 5B sobre MAESTRO |
| `MERT-v1-95M` | ~95M | HuggingFace, audio foundation model sin régimen cross-modal |
| `MERT-v1-330M` | ~330M | HuggingFace, **test principal** |

---

## Interpretación Correcta de Resultados

**Lo que Gate 7 reduce**:
La ambigüedad sobre si A4 es accesible linealmente desde encoders audio más fuertes.

**Lo que NO resuelve**:
- Un R² alto no prueba que "el cuello era exclusivamente el encoder"
- Un R² bajo no prueba "complementariedad genuina" — puede reflejar target/probe insuficiente
- Gate 7 es el lado audio; la ambigüedad cross-modal completa requiere Exp 7.1

**Nota asimétrica importante**:
MERTLite fue entrenado en régimen cross-modal (VICReg sobre MAESTRO). MERT-95M/330M son foundation models audio sin ese régimen. La diferencia de R² mezcla tamaño, datos de pretraining, y objetivo. No es una comparación simétrica.

---

## Experimentos

### Exp 7.0 — Probe A4 vs Encoders (central)

**Endpoint primario**: `LinearProbe segment-level` (Ridge cerrado, 5 group splits por pieza)
**Endpoint secundario**: `LinearProbe frame-level` (solo para comparación within-encoder)
**Exploratorio (post resultado)**: `MLPProbe` + Exp 7.0b per-layer

**Protocolo estadístico**:
- Split 80/20 por pieza → todos los segmentos de una pieza caen en el mismo fold
- 5 repeated group splits → CIs sobre varianza de split
- Ridge regression (solución cerrada, λ=1e-3) → sin varianza de optimización
- Normalización: z-score fit en train, applied to test (no leakage)

**Métricas reportadas**:
```
R²_seg ± std  [5 splits × Ridge cerrado]
  ── por banda A4 (8) + promedio global ──
  ── para: MERTLite-D0, MERT-v1-95M, MERT-v1-330M ──

Nulls:
  R²_shuffled_between ± std   (null global, expected ≈ 0, bug si > 0.05)
  R²_dummy                    (baseline trivial, expected ≈ 0)
```

### Exp 7.0b — Per-Layer MERT-330M

¿En qué capa del transformer emerge la información tipo A4?

Activa con `--per-layer`. Genera curva R² vs layer depth (25 capas de MERT-330M).

Valor científico: si A4 emerge en capas tempranas → feature de bajo nivel. Si en capas tardías o no emerge → información de nivel alto o genuinamente complementaria.

### Exp 7.1 — Mini Test02 con MERT-large (COMPLETAMENTE DIFERIDA)

No se diseña hasta ver resultados de Exp 7.0. La decisión depende del patrón:
- ¿Hay señal claramente sobre los nulls?
- ¿Hay diferencia o no entre MERTLite y MERT-large?
- ¿Queda ambigüedad residual relevante?

---

## Archivos

```
experiments/bias_control/gate7/
├── __init__.py
├── mert_large_feature_extractor.py   # Wrapper HF MERT → features por capa
└── mert_large_probe.py               # Script principal: probe + nulls + plots

experiments/bias_control/slurm/
└── gate7_mert_probe.sh               # SLURM para UNC (Exp 7.0)

data/gate7_results/                   # (gitignored)
├── features/
│   ├── MERTLite_features.npz
│   ├── MERT_v1_95M_features.npz
│   └── MERT_v1_330M_features.npz
└── probe_results/
    ├── probe_results.json
    ├── comparison_r2_seg.png
    └── per_band_breakdown.png
```

---

## Ejecución

### Local (RTX 3090):
```bash
cd <repo-root>
source venv/bin/activate

# Exp 7.0 básico (segment-level, ~1-2h por descarga HF + extracción)
python experiments/bias_control/gate7/mert_large_probe.py \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --d0-checkpoint models/gate5b/D0/best_model.pt \
    --output data/gate7_results \
    --encoders MERTLite MERT-v1-95M MERT-v1-330M \
    --n-splits 5 --seed 42

# Con frame-level (secondary, más lento):
python ... --frame-level

# Con per-layer (Exp 7.0b, requiere más VRAM/RAM):
python ... --per-layer
```

### UNC (SLURM):
```bash
# Desde UNC, después de git pull origin main:
sbatch experiments/bias_control/slurm/gate7_mert_probe.sh
```

---

## Verificaciones de Sanidad

1. Features shape correcta:
   - MERTLite: `[N, 1024]` pooled
   - MERT-95M: `[N, 768]` pooled
   - MERT-330M: `[N, 1024]` pooled
2. `Null (shuffled_between) R² ≤ 0.05` — si falla, bug en protocolo
3. `Dummy R² ≈ 0` — baseline trivial
4. Ridge cerrado: R² reproducible entre runs con mismo split seed

---

## Conexión con Otros Gates

| Gate | Conexión |
|------|----------|
| Gate 5B | MERTLite-D0 checkpoint y A4 descriptor usados aquí |
| Gate 6 | Plateau AMT motivó Gate 7. Si 330M no ayuda → techo no es encoder |
| Gate 7.1 | Se diseña post Exp 7.0 según patrón de resultados |
