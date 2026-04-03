# Gate 6 — AMT with Descriptor Conditioning

**Fecha inicio**: 2026-03-02  
**Estado**: `Exp 0` completo en local, `Exp C` con brazo local `a4r` ya completo, `Exp B` ya cerrado como negativo útil en UNC (`20/27` tareas completadas, `7` canceladas por curva plana) y `Exp A` ya también cerrado negativamente: `baseline`, `finetune-noA4`, `A4-event`, `A4-adapter` y `adapter-noA4` dieron el mismo `F1=0.3186`.

La lectura pública del gate sí cambió respecto del corte anterior: ya no es correcto describirlo como "arrays submitidos esperando drenar". La rama degradada respondió y no abrió una ventana de rescate descriptor-guided; lo que queda vivo ahora es una pregunta más acotada sobre complementariedad con `Transkun`.

## Motivación

Gate 5B dejó una lectura fuerte pero incompleta.

1. `Test02` confirmó que la mejora descriptor-guided es causal y no un efecto de capacidad.
2. `Test11` mostró que la proyección `512 -> 256` destruye una fracción enorme de la información condicionante.
3. `Test13G-B` cerró sin ventaja descriptor-guided en decodificabilidad pre-pooling: `F1≈0.10` para todos los arms.

La pregunta que queda abierta es otra: ¿esa ventaja vive solo en la geometría de retrieval o sobrevive cuando se formula como una tarea musical concreta?

Gate 6 abre esa validación downstream usando **Automatic Music Transcription (AMT)**.

> **Nota histórica**: el antiguo `Gate 6` diagnóstico de RSA/CKA quedó absorbido por `Gate 5B / Test06`. En este directorio, `Gate 6` ya no nombra esa fase histórica sino la línea AMT.

## Pregunta central

**¿La ventaja descriptor-guided se traduce a una tarea musical concreta o permanece confinada al espacio de retrieval?**

## Experimentos

| Exp | Pregunta | Método | Régimen | Estado |
|-----|----------|--------|---------|--------|
| `0` | ¿Transkun transcribe bien nuestros segmentos? | inference pretrained | `44.1kHz`, `4s + 16s` | **COMPLETO (LOCAL)** |
| `A` | ¿A4 aporta info que un SOTA no tiene? | `Transkun + A4` con controles param-matched | `44.1kHz`, `16s` | **CERRADO NEGATIVO (UNC)** |
| `B` | ¿A4 ayuda más bajo degradación? | `Transkun + A4` con ruido / low-pass / data limit | `44.1kHz`, `16s` | **CERRADO NEGATIVO (UNC)** |
| `C` | ¿Nuestras features VICReg decodifican música mejor? | decoder AMT serio sobre features congeladas | `24kHz`, `4s` | **ACTIVO (LOCAL + UNC)** |

## Hallazgo arquitectónico clave: Transkun v2

La inspección real del backbone corrigió una suposición inicial.

Transkun **no** usa “event tracks” como tokens discretos concatenados al estilo imaginado al principio. El backbone real:

1. procesa mel-spectrogramas con CNN,
2. concatena embeddings posicionales para `88` notas y `2` pedales en la dimensión de frecuencia,
3. aplica `6` capas de axial transformer,
4. produce scoring `Semi-CRF` para la salida AMT.

Por eso la inyección de `A4` quedó redefinida de dos maneras plausibles:

- tracks adicionales en la dimensión de frecuencia,
- o FiLM/adapters después de cada `BasicBlock`.

## Exp 0 — Transkun baseline verificado

`Transkun v2` ya fue corrido localmente sobre `100` segmentos MAESTRO de validación (`50x4s + 50x16s`).

| Régimen | note_onset_F1 | note+offset_F1 | note+offset+velocity_F1 | frame_F1 | onset_F1 |
|---------|---------------|----------------|--------------------------|----------|----------|
| `4s` | `0.938` | `0.667` | `0.607` | `0.784` | `0.576` |
| `16s` | `0.972` | `0.729` | `0.718` | `0.814` | `0.572` |

Lectura operativa:
- el baseline es suficientemente sano para usar `Transkun` como banco de prueba real;
- los segmentos de `4s` degradan métricas de nota/offset respecto del régimen largo, pero no invalidan la línea;
- la herramienta ya quedó calibrada antes de gastar GPU-days en UNC.

Artefactos locales asociados:
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/resultados_compartir/07_gate6_amt/exp0_transkun_baseline/`
- `experiments/bias_control/gate6/test_transkun_baseline.py`

## Exp A — Transkun + A4

Objetivo: medir si `A4` aporta información complementaria a un transcriptor SOTA ya muy fuerte.

Configs definidas:

| Config | Inyección | Rol |
|--------|-----------|-----|
| `baseline` | ninguna | techo pretrained |
| `finetune-noA4` | tracks constantes `=0` | control param-matched |
| `A4-event` | tracks con valores `A4` | test principal |
| `adapter-noA4` | FiLM con input `0` | control param-matched |
| `A4-adapter` | FiLM con `A4` | test de mayor capacidad |

Estado actual:
- código implementado;
- script SLURM validado por `preflight v6`;
- `transkun` ya instalado en UNC;
- screening `seed=42` ya completado;
- baseline `task 0` ya cerrado con `F1=0.3186`;
- `finetune-noA4`, `A4-event`, `A4-adapter` y `adapter-noA4` también cerraron en `F1=0.3186`;
- no apareció ningún brazo por encima del criterio GO/NO-GO de `+0.01` F1 absoluto, por lo que la rama `Transkun+A4` queda cerrada negativamente en esta receta.

Artefactos UNC asociados:
- `results_unc/gate6_amt/expA/`

## Exp B — Condiciones degradadas

Objetivo: probar si `A4` se vuelve más útil cuando la señal acústica se deteriora.

Degradaciones previstas:
- ruido gaussiano (`SNR 5/10/20 dB`),
- low-pass (`4/2/1 kHz`),
- data limit (`10/25/50%`).

Regla metodológica fija:
- `A4` siempre se computa desde el **audio degradado**, no desde el limpio.

Lectura vigente:
- `20/27` tasks alcanzaron evidencia suficiente y `7` se cancelaron temprano por curvas planas;
- la degradación no abrió una ventana donde `A4` rescatara a `Transkun`;
- fine-tuning y `A4-degraded` convergieron a la misma banda del baseline degradado;
- los deltas observados contra baseline quedaron entre `+0.0011` y `-0.0005`, o directamente en `0`;
- por eso `Exp B` ya se documenta como **negativo útil**, no como frente abierto.

Consecuencia metodológica:
- no tiene sentido seguir presentándolo como array simplemente "corriendo";
- la conclusión correcta es que, bajo este régimen y estas degradaciones, el descriptor no compró mejora práctica.

Complemento del corte:
- `Exp A` endurece todavía más esa lectura, porque tampoco en régimen base apareció lift sobre `Transkun`;
- por lo tanto, la rama `Transkun+A4` ya no debe contarse como línea abierta ni como screening pendiente.

Artefactos UNC asociados:
- `results_unc/gate6_amt/expB/`

## Exp C — Decoder AMT sobre VICReg features

Objetivo: reabrir la pregunta de decodificabilidad de nuestras representaciones, pero con un decoder mucho más serio que el de `Test13G-B`.

Arms:
- `D0`
- `d4a4`
- `a4r`
- `d4-a4r`

Características:
- decoder grande con cross-attention,
- features pre-pooling congeladas,
- régimen Phideus (`24kHz`, `4s`, `188` frames),
- comparación directa contra el cierre negativo de `13G-B`.

Estado actual:
- el primer array UNC `1144325` falló por path absoluto de MAESTRO;
- los `3` scripts Gate 6 ya quedaron corregidos para usar `$REPO/data/maestro_v3/maestro-v3.0.0`;
- `main` también incorporó el fix `1da73fb` para evitar targets en CPU dentro de `build_pr_targets()`;
- el array UNC fue reenviado como `1144560`;
- en local `a4r` ya cerró con `best_F1=0.1570 @ ep50`, muy por encima del decoder de `13G-B`;
- en UNC el preflight siguió iterando hasta `v6`, después de corregir mixed sample rates, el truncado previo a `torch.stack` y el bug de `torch.utils.checkpoint` sobre inputs sin gradiente;
- `preflight v6` (`job 1144711`) ya validó `checkpoint + resume + SIGTERM + auto-resubmit`;
- el throughput real quedó en `4.9 s/iter` y proyecta ~`68h` para `50k` iteraciones, por lo que toda la línea larga debe pensarse como corrida chunked y no como job único “limpio”.

## Estado operativo al corte

| Bloque | Estado | Nota |
|--------|--------|------|
| `Exp 0` | **COMPLETO** | baseline local ya fijado |
| `Exp C` | **ACTIVO** | `a4r` local completo; sirve como referencia downstream y mantiene abierta la pregunta sobre decodificabilidad con decoder serio |
| `Exp A` | **CERRADO NEGATIVO** | screening `seed=42` completado; todos los configs quedaron en `F1=0.3186`, sin superar `+0.01` |
| `Exp B` | **CERRADO NEGATIVO** | `20/27` tasks bastaron para cerrar que degradación no produjo ventaja descriptor-guided útil sobre `Transkun` |

## Scripts relevantes

- `experiments/bias_control/gate6/README.md`
- `experiments/bias_control/gate6/test_transkun_baseline.py`
- `experiments/bias_control/gate6/transkun_a4_finetune.py`
- `experiments/bias_control/gate6/transkun_degraded.py`
- `experiments/bias_control/gate6/vicreg_amt_decoder.py`
- `experiments/bias_control/slurm/gate6_vicreg_decoder.sh`
- `experiments/bias_control/slurm/gate6_transkun_a4.sh`
- `experiments/bias_control/slurm/gate6_transkun_degraded.sh`

## Lectura estratégica

Gate 6 no reabre Gate 5B. Lo que hace es aprovechar su cierre.

- Gate 5B dejó causalidad, bottleneck y un límite generativo claro.
- Gate 6 pregunta si, aun con ese límite, la ventaja descriptor-guided sobrevive cuando la tarea ya no es retrieval sino transcripción.
- Escalón 2 sigue siendo el foco principal del programa; Gate 6 AMT funciona como validación downstream paralela.
- El hallazgo útil del corte no es "Gate 6 falló", sino algo más específico: **la señal descriptor-guided no se tradujo automáticamente a una mejora útil en `Transkun`, ni en régimen base ni bajo degradación**.
- La pregunta downstream no desaparece, pero cambia de lugar: ya no vive en insistir con `Transkun+A4`, sino en lo que todavía pueda decir `Exp C` sobre features congeladas y decoder serio.
