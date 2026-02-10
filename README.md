# Phideus v5.0

Programa de investigacion sobre representaciones de ratios armonicos y aprendizaje cross-modal.

**Estado actual (2026-02-10)**: Escalon 1-A/B completado; Escalon 1-C en curso (Gate 4 + Gate 6).

---

## Que es Phideus

Phideus investiga si las relaciones armonicas (ratios de frecuencia) pueden funcionar como una representacion transferible entre modalidades (audio, MIDI y otras).

Hipotesis activas:

- **H1 (estructura)**: validada.
- **H2 (aprendibilidad)**: validada.
- **H3 (cross-modalidad)**: prometedora, en validacion continua.

---

## Estado de BIAS_CONTROL

`BIAS_CONTROL` es la linea principal para Escalon 1 (MAESTRO Audio<->MIDI).

Resultados consolidados:

- **Gate 2 (baseline operativo)**: GO.
  - Gap: `0.478`
  - Recall@10 (structured pool): `34.4%` a2m / `37.6%` m2a
  - Hard negative accuracy: `80.4%`
- **Gate 3 (DANN)**: cerrado; no mejora robusta sobre Gate 2.
- **Gate 4 (ratio auxiliary)**: en ejecucion causal A/B.
  - Run A: `ratio_weight=0.1` (en curso)
  - Run B: `ratio_weight=0.0` (control, pendiente)

---

## Quick Start

### 1) Entorno

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2) Run A (Gate 4, ratio activo)

```bash
python experiments/bias_control/gate4_ratio_auxiliary.py \
  --maestro-dir data/maestro_v3/maestro-v3.0.0 \
  --checkpoint data/bias_control_medium/training_outputs/gate2/checkpoint_epoch45.pt \
  --output data/bias_control_medium/training_outputs/gate4_runA \
  --epochs 30 --ratio-weight 0.1 \
  --batch-size 16 --segment-len 4.0 --hop 1.0 --num-workers 8 \
  --max-batches-per-epoch 1000 --max-val-batches 846 \
  --seed 42 --device cuda
```

### 3) Run B (Gate 4, control)

```bash
python experiments/bias_control/gate4_ratio_auxiliary.py \
  --maestro-dir data/maestro_v3/maestro-v3.0.0 \
  --checkpoint data/bias_control_medium/training_outputs/gate2/checkpoint_epoch45.pt \
  --output data/bias_control_medium/training_outputs/gate4_runB \
  --epochs 30 --ratio-weight 0.0 \
  --batch-size 16 --segment-len 4.0 --hop 1.0 --num-workers 8 \
  --max-batches-per-epoch 1000 --max-val-batches 846 \
  --seed 42 --device cuda
```

### 4) Evaluacion structured pool

```bash
python experiments/bias_control/evaluate_structured_pool.py \
  --model data/bias_control_medium/training_outputs/gate4_runA/best_model_base.pt \
  --maestro-dir data/maestro_v3/maestro-v3.0.0 \
  --pool-size 256 --n-queries 500 --seed 42 --device cuda
```

---

## Estructura del repositorio

```text
Phideus/
├── src/
│   ├── analizador/
│   ├── bias_control/
│   │   ├── architectures/
│   │   ├── datasets/
│   │   ├── encoders/
│   │   └── losses/
│   ├── extractors/
│   ├── RNA/
│   └── hrm/
├── experiments/
│   └── bias_control/
├── Documents/
│   ├── BIAS_CONTROL/
│   ├── ESCALON_1/
│   ├── UOEMD/
│   ├── INDICE_DOCUMENTACION.md
│   └── Proyecto_Estado_Actual.md
├── config/
└── README.md
```

Notas:

- `data/`, `models/`, `train/`, `test/` y artefactos pesados no se versionan.
- Varias rutas de `Documents/` son de trabajo interno e historial experimental.

---

## Documentacion recomendada

- `Documents/INDICE_DOCUMENTACION.md`
- `Documents/Proyecto_Estado_Actual.md`
- `Documents/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/BIAS_CONTROL/INFORME_GATE2_COMPLETO.md`
- `Documents/BIAS_CONTROL/Gate3_DANN_Results/INFORME_GATE3_COMPLETO.md`
- `Documents/BIAS_CONTROL/AUDITORIA_BIAS_CONTROL_CODEX.md`

---

## Licencia

MIT. Ver `LICENSE.md`.
