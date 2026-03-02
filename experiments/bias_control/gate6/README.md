# Gate 6 — AMT with Descriptor Conditioning

Gate 6 abre una validación downstream después del cierre de Gate 5B.

Pregunta central:
**¿la ventaja descriptor-guided se traduce a AMT o vive sobre todo en la geometría de retrieval?**

> Nota histórica: el antiguo `Gate 6` diagnóstico quedó absorbido por `Gate 5B / Test06`.

## Estado al corte

| Bloque | Estado | Nota |
|--------|--------|------|
| `Exp 0` | **COMPLETO (LOCAL)** | baseline `Transkun` validado en `4s` y `16s` |
| `Exp C` | **SUBMITTED (UNC)** | decoder AMT sobre VICReg, job `1144325` |
| `Exp A` | **PENDIENTE** | `Transkun + A4`, falta entorno `transkun` en UNC |
| `Exp B` | **PENDIENTE** | degradación, bloqueado por `Exp A` |

## Experimentos

| Exp | Pregunta | Método | Régimen |
|-----|----------|--------|---------|
| `0` | ¿Transkun transcribe nuestros segmentos? | inference pretrained | `44.1kHz`, `4s + 16s` |
| `A` | ¿A4 aporta info que un SOTA no tiene? | inyección A4 en `Transkun` con controles param-matched | `44.1kHz`, `16s` |
| `B` | ¿A4 ayuda más bajo degradación? | `Transkun + A4` con ruido / low-pass / data limit | `44.1kHz`, `16s` |
| `C` | ¿Nuestras features VICReg son más transcribibles? | decoder AMT serio sobre encoders congelados | `24kHz`, `4s` |

## Exp 0 — baseline

| Régimen | note_onset_F1 | note+offset_F1 | note+offset+velocity_F1 | frame_F1 |
|---------|---------------|----------------|--------------------------|----------|
| `4s` | `0.938` | `0.667` | `0.607` | `0.784` |
| `16s` | `0.972` | `0.729` | `0.718` | `0.814` |

## Archivos

```text
experiments/bias_control/gate6/
├── README.md
├── __init__.py
├── evaluation.py
├── test_transkun_baseline.py
├── a4_descriptor_standalone.py
├── transkun_a4_finetune.py
├── transkun_degraded.py
├── amt_decoder_model.py
└── vicreg_amt_decoder.py

experiments/bias_control/slurm/
├── gate6_vicreg_decoder.sh
├── gate6_transkun_a4.sh
└── gate6_transkun_degraded.sh
```

## Quick start

```bash
cd /mnt/m2-1TB/Phideus
source venv/bin/activate

python experiments/bias_control/gate6/test_transkun_baseline.py \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/gate6_results/transkun_baseline \
    --device cuda
```

## Convenciones de evaluación

| Parámetro | Valor |
|-----------|-------|
| Onset tolerance | `50ms` |
| Offset tolerance | `50ms` o `20%` |
| Pedal extension | `No Ext` |
| Note clipping | en bordes de segmento |
| Velocity bins | `128` |
