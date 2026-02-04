# Rosetta1 2.0 - Plan de Implementación

**Fecha de creación**: 2026-01-23
**Estado**: En progreso (Implementación completada, pendiente ejecución)
**Basado en**: Diagnóstico GPT5.2Pro (Enero 2026)

---

## Objetivo

Demostrar **cross-modality real** (no solo alineamiento) entre Audio y Vibración, siguiendo las recomendaciones del diagnóstico de consistencia.

## Criterios de Éxito Mínimos

| Criterio | Métrica | Umbral |
|----------|---------|--------|
| No hay leakage | Split por archivo con test set | 0 overlap |
| z_private funciona | var(z_private) | > 0.1 |
| z_private diferenciado | \|z_priv_audio - z_priv_vib\| | > 0.5 |
| Cross-reconstruction | Pearson corr (test) | > 0.75 |
| Cross-recon vs baseline | Δcorr vs mean_hist | > +0.10 |
| Retrieval | Top-1 accuracy | > 15% |
| Controles negativos | Shuffled retrieval | ~random |

---

## Work Packages (WPs)

### WP1: Congelar Baseline + Trazabilidad ✅

**Objetivo**: Establecer punto de referencia reproducible.

**Archivos creados**:
- `config/rosetta1_baseline.yaml` - Configuración exacta del modelo original
- `config/rosetta1_fix_private.yaml` - Configuración para fix de z_private
- `experiments/freeze_baseline.py` - Script para congelar artefactos

**Entregables**:
```
artifacts/baseline/
├── checkpoint.pt      # Pesos del modelo
├── latents.npz        # Representaciones latentes extraídas
├── metrics.json       # Métricas del baseline
└── README.md          # Documentación de reproducibilidad
```

**Comando de ejecución**:
```bash
python experiments/freeze_baseline.py \
    --checkpoint models/roseta_vae_best.pt \
    --data data/datasets/roseta_full.npz \
    --output artifacts/baseline
```

---

### WP2: Integridad Metodológica ✅

**Objetivo**: Eliminar leakage y establecer controles negativos.

**Cambio 1: Split 3-way por archivo**

Modificado `src/datasets/roseta_dataset.py`:
- Función `create_roseta_dataloaders()` ahora soporta train/val/test
- Split por ARCHIVO (no por frame) para evitar leakage
- Assertions automáticos verifican 0 overlap entre splits
- Backwards compatible con llamadas legacy (2 loaders)

```python
# Nuevo uso
train_loader, val_loader, test_loader = create_roseta_dataloaders(
    npz_path='data/datasets/roseta_full.npz',
    train_split=0.70,
    val_split=0.15,
    test_split=0.15,
    return_test=True,  # Nuevo parámetro
)
```

**Cambio 2: Controles negativos**

Modificado `experiments/evaluate_cross_reconstruction.py`:
- `--pairing aligned|shuffled` - Romper pareo audio↔vib
- `--baseline none|mean_hist` - Comparar con histograma promedio
- `--zshared real|random` - Usar z_shared de N(0,1)
- `--run-all-controls` - Ejecutar todas las condiciones

**Validación Go/No-Go**:
- Con shuffled: retrieval debe caer a ~random (~0.8% para 128 muestras)
- Con shuffled: cross-recon debe empeorar ≥0.15 en correlación
- Modelo debe superar mean_hist baseline

---

### WP3: Fix z_private Collapse ✅ (CRÍTICO)

**Problema identificado**: z_private colapsa a prior N(0,1), no codifica información.

**Solución implementada** en `src/RNA/roseta_vae.py`:

1. **KL selectivo**:
   - `beta_kl_shared = 1.0` (normal)
   - `beta_kl_private = 0.01` (muy bajo, permite varianza)

2. **Dropout en z_shared durante decoding**:
   - Fuerza al decoder a usar z_private
   - `dropout_shared = 0.5` durante training

3. **Loss de diferenciación**:
   - Penaliza si z_private_audio ≈ z_private_vib
   - Hinge loss con margin = 1.0
   - `lambda_diff = 0.1`

**CLI actualizado** en `run_roseta_experiment.py`:
```bash
python experiments/run_roseta_experiment.py \
    --phase full \
    --data data/datasets/roseta_full.npz \
    --beta-kl-private 0.01 \
    --dropout-shared 0.5 \
    --lambda-diff 0.1 \
    --all-data
```

**Métricas de validación** (ahora reportadas durante training):
- `z_private_audio_var` - Debe ser > 0.1
- `z_private_vib_var` - Debe ser > 0.1
- `z_private_diff` - Debe ser > 0.5

---

### WP4: Métricas de Traducción Robustas ✅

**Archivo creado**: `experiments/evaluate_retrieval.py`

**Modos de evaluación**:
1. **Global**: Todos vs todos
2. **Intra-condition**: Solo candidatos de misma condición
3. **Cross-condition**: Candidatos de otras condiciones

**Métricas**:
- Top-k accuracy (k=1, 5, 10)
- Mean Reciprocal Rank (MRR)
- Mean Rank

**Baselines incluidos**:
- Shuffled (pares rotos)
- Random embeddings

**Comando**:
```bash
python experiments/evaluate_retrieval.py \
    --model models/roseta_vae_best.pt \
    --data data/datasets/roseta_full.npz \
    --output data/evaluations/retrieval
```

---

### WP5: Separación de Regímenes ✅

**Archivo creado**: `experiments/evaluate_regime_separation.py`

**Métricas implementadas**:
1. **Silhouette Score** - En z_shared RAW (no en UMAP)
2. **Linear Probe AUC** - Regresión logística con CV
3. **Centroid Distance normalizada** - Distancia entre centroides / std pooled
4. **Fisher Ratio** - Scatter between / scatter within

**Interpretación**:
- Silhouette > 0.5: Separación fuerte
- Silhouette 0.25-0.5: Separación moderada
- Silhouette < 0.25: Separación débil

**Comando**:
```bash
python experiments/evaluate_regime_separation.py \
    --model models/roseta_vae_best.pt \
    --data data/datasets/roseta_full.npz \
    --output data/evaluations/regime_separation
```

---

### WP6: Ablations ✅

**Archivo creado**: `experiments/run_ablations.py`

**Condiciones**:

| ID | Descripción | InfoNCE | Ratio-hist | Aux channels |
|----|-------------|---------|------------|--------------|
| A | Propuesto completo | ✓ | ✓ | ✓ |
| B | Sin InfoNCE | ✗ | ✓ | ✓ |
| C | Raw PSD | ✓ | ✗ | - |
| D | Sin canales auxiliares | ✓ | ✓ | ✗ |

**Preguntas que responde**:
- Si C ≈ A → ratio-hist no aporta (problema para el claim)
- Si B << A → InfoNCE es crítico (esperado)
- Si D < A → canales auxiliares importan

**Comando**:
```bash
python experiments/run_ablations.py \
    --data data/datasets/roseta_full.npz \
    --output data/ablations \
    --epochs 50
```

---

## Orden de Ejecución Recomendado

```
Fase 1 (Preparación):
├── [1] freeze_baseline.py          # Congelar estado actual
└── [2] Verificar dataset existe    # data/datasets/roseta_full.npz

Fase 2 (Re-entrenamiento con fix):
└── [3] run_roseta_experiment.py    # Con --beta-kl-private 0.01

Fase 3 (Validación):
├── [4] Verificar z_private fix     # var > 0.1, diff > 0.5
├── [5] evaluate_cross_reconstruction.py --run-all-controls
├── [6] evaluate_retrieval.py
└── [7] evaluate_regime_separation.py

Fase 4 (Ablations - si hay tiempo):
└── [8] run_ablations.py            # Identificar fuente de novedad
```

---

## Estado Actual

### Completado ✅
- Todos los scripts implementados y verificados sintácticamente
- Configuraciones YAML creadas
- Modificaciones a roseta_dataset.py (3-way split)
- Modificaciones a roseta_vae.py (KL selectivo, dropout, diff loss)
- Modificaciones a run_roseta_experiment.py (nuevos CLI args)
- Nuevos scripts de evaluación (retrieval, regime separation)
- Script de ablaciones

### Pendiente ⏳
1. **Ejecutar freeze_baseline.py** - Requiere modelo entrenado
2. **Re-entrenar con fix z_private** - ~100 epochs
3. **Verificar criterios Go/No-Go** después del re-entrenamiento
4. **Ejecutar evaluaciones completas** en modelo corregido
5. **Ejecutar ablations** si el tiempo lo permite
6. **Documentar resultados finales**

---

## Archivos Creados/Modificados

### Nuevos
| Archivo | Descripción |
|---------|-------------|
| `config/rosetta1_baseline.yaml` | Config baseline congelada |
| `config/rosetta1_fix_private.yaml` | Config para fix z_private |
| `experiments/freeze_baseline.py` | Congela artefactos baseline |
| `experiments/evaluate_retrieval.py` | Evaluación retrieval extendida |
| `experiments/evaluate_regime_separation.py` | Evaluación separación regímenes |
| `experiments/run_ablations.py` | Estudios de ablación |

### Modificados
| Archivo | Cambios |
|---------|---------|
| `src/datasets/roseta_dataset.py` | Split 3-way, anti-leakage |
| `src/RNA/roseta_vae.py` | KL selectivo, dropout, diff loss |
| `experiments/run_roseta_experiment.py` | Nuevos CLI args |
| `experiments/evaluate_cross_reconstruction.py` | Controles negativos |

---

## Notas Técnicas

### Sobre el fix de z_private

El colapso de z_private ocurre porque el KL divergence penaliza igualmente z_shared y z_private, pero z_shared recibe señal adicional del InfoNCE. El decoder aprende a ignorar z_private.

La solución:
1. Reducir KL en z_private (β=0.01 vs β=1.0)
2. Dropout en z_shared fuerza al decoder a usar z_private
3. Loss de diferenciación asegura que z_private captura info modality-specific

### Sobre los controles negativos

Los controles shuffled y random son críticos para validar que:
1. El modelo realmente aprende correspondencia audio↔vib
2. No hay leakage ni shortcuts espurios
3. Las métricas reflejan capacidad real de traducción

Si shuffled ≈ aligned, hay un problema fundamental.

---

## Referencias

- Diagnóstico: `Documents/Roseta/Rosetta1_consistence_evaluation_GPT5.2Pro.md`
- Roadmap original: `Documents/Roseta/Rosetta1_2.0_-_Roadmap_GTP5.2Pro.md`
- Paper principal: `Documents/PHIDEUS_RESEARCH_PROGRAM_2026.md`
