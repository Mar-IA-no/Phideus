# Briefing para Claude UNC — Gate 6 AMT

> **Corte 2026-03-24**: este briefing ya no debe leerse como orden activa para abrir `Exp A/B`. La rama `Transkun+A4` quedó cerrada negativamente por `Exp A` + `Exp B`; `Exp C` queda como única línea downstream todavía abierta. El documento se conserva como referencia operativa del frente y de la infraestructura UNC.

## Contexto

Gate 5B cerró con dos hechos complementarios:

1. `Test02` confirmó causalidad fuerte: la mejora descriptor-guided no se explica por capacidad extra.
2. `Test13G-B` no mostró una ventaja descriptor-guided clara en decodificabilidad pre-pooling bajo un decoder moderado.

Eso deja abierta una pregunta precisa:

**¿la ventaja de los descriptores sobrevive en una tarea musical concreta o vive principalmente en la geometría de retrieval?**

Gate 6 abre esa validación vía **AMT (Automatic Music Transcription)**.

> **Nota de nomenclatura**: el antiguo `Gate 6` diagnóstico de RSA/CKA fue absorbido por `Gate 5B / Test06`. En este frente, `Gate 6` ya refiere solo a la línea AMT.

## Resumen de experimentos

| Exp | Pregunta | Método | Régimen | Servidor | Estado |
|-----|----------|--------|---------|----------|--------|
| `0` | ¿Transkun transcribe nuestros segmentos? | inference pretrained | `44.1kHz`, `4s + 16s` | LOCAL | **DONE** |
| `A` | ¿A4 aporta info que un SOTA no tiene? | `Transkun + A4` con controles param-matched | `44.1kHz`, `16s` | UNC | **CERRADO NEGATIVO** |
| `B` | ¿A4 ayuda más bajo degradación? | `Transkun + A4` degradado | `44.1kHz`, `16s` | UNC | **CERRADO NEGATIVO** |
| `C` | ¿Las features VICReg permiten mejor AMT? | decoder serio sobre encoders Gate 5B congelados | `24kHz`, `4s` | LOCAL + UNC | **ACTIVO ACOTADO** |

## Exp 0 — Ya completado en local

`Transkun v2` ya se verificó sobre `100` segmentos MAESTRO (`50x4s + 50x16s`).

| Régimen | note_onset_F1 | note+offset_F1 | note+offset+velocity_F1 | frame_F1 | onset_F1 |
|---------|---------------|----------------|--------------------------|----------|----------|
| `4s` | `0.938` | `0.667` | `0.607` | `0.784` | `0.576` |
| `16s` | `0.972` | `0.729` | `0.718` | `0.814` | `0.572` |

Lectura: baseline sano. Se puede confiar en `Transkun` como herramienta para `Exp A/B`.

## Hallazgo arquitectónico fijado

Transkun **no** usa “event tracks” como tokens discretos independientes. El backbone real:

1. CNN frontend sobre mel-spectrograma,
2. embeddings posicionales para `88` notas + `2` pedales en la dimensión de frecuencia,
3. axial transformer,
4. scoring `Semi-CRF`.

Por eso la inyección A4 quedó adaptada a la arquitectura real:
- tracks adicionales en la dimensión de frecuencia, o
- FiLM/adapters después de `BasicBlock`.

## Exp A — Transkun + A4

### Scripts

- `experiments/bias_control/gate6/transkun_a4_finetune.py`
- `experiments/bias_control/slurm/gate6_transkun_a4.sh`

### Configs

| Config | Inyección | Params nuevos | Rol |
|--------|-----------|---------------|-----|
| `baseline` | ninguna | `0` | techo pretrained |
| `finetune-noA4` | tracks constantes `=0` | `~66K` | control param-matched |
| `A4-event` | tracks con `A4` | `~66K` | test principal |
| `adapter-noA4` | FiLM con input `0` | `~803K` | control param-matched |
| `A4-adapter` | FiLM con `A4` | `~803K` | test principal |

### Regla metodológica

Comparación primaria:
- `A4-event` vs `finetune-noA4`
- `A4-adapter` vs `adapter-noA4`

No comparar el efecto de A4 contra `baseline` congelado como si fuera la evidencia principal; ese contraste mezcla adaptación y descriptor.

### Estado

- código y SLURM ya validados;
- `transkun` y dependencias ya instalados en UNC;
- screening `seed=42` ya completado;
- `baseline`, `finetune-noA4`, `A4-event`, `A4-adapter` y `adapter-noA4` cerraron en `F1=0.3186`;
- no apareció ningún brazo por encima del criterio GO/NO-GO de `+0.01`, así que la rama `Transkun+A4` queda cerrada negativamente en esta receta.

## Exp B — Degraded conditions

### Scripts

- `experiments/bias_control/gate6/transkun_degraded.py`
- `experiments/bias_control/slurm/gate6_transkun_degraded.sh`

### Regla crítica

`A4` se computa siempre desde el **audio degradado**, nunca desde el limpio.

### Degradaciones previstas

- ruido gaussiano: `SNR 5/10/20 dB`
- low-pass: `4/2/1 kHz`
- data limit: `10/25/50%`

### Estado

- el pipeline quedó implementado y corrido;
- la degradación no abrió una ventana donde `A4` rescatara a `Transkun`;
- fine-tuning y `A4-degraded` convergieron a la misma banda del baseline degradado;
- `20/27` tasks bastaron para cerrar el frente y `7` se cancelaron por curvas planas;
- la lectura correcta ya es **cierre negativo útil**, no bloqueo pendiente.

## Exp C — AMT decoder sobre VICReg

### Scripts

- `experiments/bias_control/gate6/amt_decoder_model.py`
- `experiments/bias_control/gate6/vicreg_amt_decoder.py`
- `experiments/bias_control/slurm/gate6_vicreg_decoder.sh`

### Arms

| Arm | Pre-pooling shape | Checkpoint |
|-----|-------------------|------------|
| `D0` | `[B, 2400, 1024]` | `models/gate5b/D0/best_model.pt` |
| `d4a4` | `[B, 2400, 1024]` | `models/gate5b/d4a4/best_model.pt` |
| `a4r` | `[B, 188, 1024]` | `models/gate5b/a4r/best_model.pt` |
| `d4-a4r` | `[B, 188, 1024]` | `models/gate5b/d4-a4r/best_model.pt` |

### Estado actual

- primer envío `1144325` falló por path absoluto de MAESTRO;
- fix aplicado en los `3` scripts Gate 6 (`MAESTRO_SRC=$REPO/data/maestro_v3/maestro-v3.0.0`);
- `main` ya incluye además el fix `1da73fb` para `build_pr_targets()`; UNC debe asegurar `git pull origin main` antes de que el job salga de la cola;
- array reenviado: `1144560`
  - `1144560_0 = D0`
  - `1144560_1 = d4a4`
  - `1144560_2 = a4r`
  - `1144560_3 = d4-a4r`

## Setup de referencia en UNC

```bash
cd $REPO
git pull origin main

# Dependencias para Exp A/B (ya instaladas en `phideus`; repetir solo si el env fue recreado)
pip install transkun pretty_midi midi2audio

# Checkpoints Gate 5B
ls -lh models/gate5b/{D0,d4a4,a4r,d4-a4r}/best_model.pt
```

## Fixes SLURM ya incorporados

Los scripts Gate 6 ya incluyen los fixes específicos para Mendieta:

- stderr separado,
- `set -eo pipefail`,
- `source /etc/profile`,
- path de MAESTRO corregido.

## Orden recomendado

1. Tratar `Exp A` y `Exp B` como frentes ya cerrados negativamente, no como arrays pendientes.
2. Monitorear `Exp C` solo si se decide seguir explotando la línea downstream abierta con decoder serio.
3. Preservar scripts, fixes y setup UNC como referencia operativa reproducible.

## Riesgos concretos

1. No mezclar el `Gate 6` histórico con `Gate 6 AMT`.
2. No volver a describir `Exp A` o `Exp B` como pendientes: en este corte ambos ya cierran negativamente la rama `Transkun+A4`.
3. Verificar que `models/gate5b/d4-a4r/best_model.pt` esté presente antes de confiar en cualquier reapertura de `Exp C`.
