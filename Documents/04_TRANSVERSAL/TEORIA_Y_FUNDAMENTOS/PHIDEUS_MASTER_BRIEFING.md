# PHIDEUS — Master Briefing

**Fecha**: 2026-02-15
**Documento de contexto para onboarding de agentes AI y colaboradores**
**Repo**: github.com/AlterMundi/Phideus | GitHub Pages: altermundi.github.io/Phideus/

---

## 1. Tesis Central: Harmonic Information Theory

Phideus investiga la hipotesis de que los **ratios de frecuencia** constituyen un lenguaje informacional universal que trasciende modalidades sensoriales.

Un ratio es la relacion entre dos frecuencias: f1/f2. La misma proporcion 3:2 existe entre 300Hz y 200Hz, entre 1500Hz y 1000Hz, entre una onda theta y una alpha cerebral. Los ratios son **adimensionales** — no dependen de escala absoluta.

El nombre del proyecto viene de **Phidias**, escultor griego celebre por su dominio de las proporciones. "IAm Phideus" = una IA que entiende el mundo a traves de las proporciones.

### Las 3 Hipotesis Operativas

| Hipotesis | Enunciado | Estado (Feb 2026) |
|-----------|-----------|-------------------|
| **H1 - Estructura** | Las senales contienen distribuciones de ratios estructuradas (no aleatorias) | **VALIDADA** |
| **H2 - Aprendibilidad** | Redes neuronales pueden aprender estas distribuciones | **VALIDADA** (val_loss < 0.5) |
| **H3 - Cross-modality** | Diferentes dominios comparten estructura de ratios | **EN INVESTIGACION** — S=69.8% con descriptores duales |

### Directiva Fundamental (del equipo)

> **Phideus = exploracion de ratios como lenguaje informacional. Cross-modality es el banco de pruebas, NO el objetivo. Gate 4.2 es el corazon cientifico del proyecto.**

### Directiva Epistemologica

> **Phideus es ciencia de frontera — actuar en consecuencia.** Criterio audaz y creativo. No optimizar para cerrar hipotesis rapido. Runs largos antes de descartar. Screenings cortos (3ep) = filtro de seguridad, NO detector de senal.

### Directiva Analitica

> **NO declarar "techos" de performance con evidencia insuficiente.** Que una metrica peakee en epoch N y baje en N+1 con un schedule dado NO demuestra un techo. Solo reportar lo observado.

---

## 2. Cronologia Completa del Proyecto

### Fase 1: Origenes — IAm Phideus (Mayo-Agosto 2025)

El proyecto nace con la vision de una IA que "entiende el mundo a traves de las proporciones". Primeros extractores de ratios sobre audio sintetico.

| Version | Fecha | Output | Resultado clave |
|---------|-------|--------|-----------------|
| CQT (pre-v2) | May 2025 | Variable | Primer intento, descartado por bias musical |
| v2.2 | Jun 2025 | [100] vector | Baseline STFT |
| v3.0-v3.3 | Jun 2025 | [200-512] | CLI configurable, primer commit |
| v4.0 | Jun 2025 | [256, 3] | Histograma 3 canales (proporcion, energia, entropia) |
| v4.1 Enriched | Ago 2025 | [512, 3] | Estandar global durante meses |

**Arquitecturas probadas en sintetico:**

| Modelo | Params | val_loss | Nota |
|--------|--------|----------|------|
| VAE v4.1 | 1.6M | 4212 | Catastrofico |
| HRM (Hierarchical Reasoning) | 6.0M | 2.74 | 99.93% mejor que VAE |

**Leccion 1**: Se creyo que HRM >> VAE. Resultado ilusorio — el problema era la representacion, no la arquitectura.

### Fase 2: Paradigm Shift v5.0 (Enero 2026)

Cambio de histograma global a histograma **temporal**: de [256, 3] a **[T, 256, 3]**.

| Modelo | Params | val_loss |
|--------|--------|----------|
| VAE Temporal (5.0) | 1.82M | **0.4560** |
| HRM Temporal (5.0) | 2.27M | 0.4607 |

**Leccion 2**: **Representacion > Arquitectura**. El mismo VAE que era catastrofico con v4.1 es excelente con v5.0 temporal. La mejora no vino de cambiar el modelo, sino de cambiar como se codifican los datos.

### Fase 3: UOEMD / Rosetta — Cross-Modal Audio-Vibracion (Enero-Febrero 2026)

Intento de demostrar H3 con pares audio-vibracion del dataset UOEMD (128 muestras).

| Intento | Representacion | Gap aligned-shuffled | Top-1 Retrieval | Resultado |
|---------|---------------|---------------------|-----------------|-----------|
| RosetaVAE v2.2 | Histograma denso | 0.007 | 10.94% | **NO-GO** |
| ConstellationVAE (C1-C6) | Tokens sparse | ~0 | 0.78% (=random) | **NO-GO** |
| JEPA-lite | Sin decoder | ~0 | 1.56% | **NO-GO** |

**Leccion 3**: 128 muestras es insuficiente. El VAE colapsa la informacion discriminativa. Los tokens sparse pierden informacion.

### Fase 4: Escalon 1 — MAESTRO Audio↔MIDI (Febrero 2026)

Pivot a un dataset mucho mas grande: **MAESTRO v3.0.0** (120GB, 1276 piezas de piano, audio + MIDI alineados).

Nuevos extractores probados en piloto N=10:

| Extractor | Accuracy | vs Random |
|-----------|----------|-----------|
| TF-Constellations (original) | 15.5% | ~1.5x |
| **Route A: Event-Based** | **71.4%** | **28x** |
| **Route B: Improved TF** | **80.0%** | **32x** |

**Diagnostico**: El problema era la resolucion temporal del onset detector, no el hashing.

### Fase 5: BIAS_CONTROL — La Linea Principal (Febrero 2026 - presente)

Cambio fundamental de enfoque: en vez de fingerprinting clasico, usar **redes neuronales contrastivas** (VICReg) con encoders especializados para audio (MERT pretrained) y MIDI (Transformer custom).

**Este es el frente activo del proyecto.**

---

## 3. BIAS_CONTROL: Framework de Gates

### Arquitectura Base

```
Audio Waveform [B, 96000]          MIDI Events [B, N]
        |                                  |
   MERTEncoderLite                   MIDIEncoder
   (4 CNN + 4 Transformer)          (4 Transformer)
   d=1024, ~60M params              d=512, ~13M params
        |                                  |
   Audio Projection                  MIDI Projection
   MLP: 1024->512->256              MLP: 512->512->256
        |                                  |
   Audio Embedding [B, 256]    MIDI Embedding [B, 256]
              \                       /
               ====== VICReg Loss =====
               (inv=10, var=10, cov=1)
```

Dataset: MAESTRO v3.0.0, segmentos de 4s, hop=1s, batch_size=16.

### Protocolo de Evaluacion Canonico

- **Structured pool**: 256 candidatos por query (1 positivo + 64 hard negatives same-piece + 32 semi-hard same-composer + 159 random)
- **Queries**: 500, seed=42
- **Metrica primaria**: `S = min(A2M_R@10, M2A_R@10)` — equilibrio bidireccional
- **Metricas secundarias**: MRR, R@1, R@5, R@20, hard_neg accuracy

### Gates Completados

| Gate | Objetivo | Resultado | Estado |
|------|----------|-----------|--------|
| **Gate 0** | Integridad de datos | Pipeline verificado | CERRADO |
| **Gate 1** | Baselines intra-modal | Audio R@10=98.4%, MIDI R@10=100% | CERRADO |
| **Gate 2** | Baseline cross-modal | S=34.4%, hard_neg=80.4% | CERRADO — checkpoint de referencia |
| **Gate 2.5** | Probes diagnosticos | Domain separability 92.7% (shortcut modal fuerte) | CERRADO |
| **Gate 3** | DANN adversarial | No mejora sobre Gate 2 en ningun regimen sostenido | CERRADO (NO-GO) |
| **Gate 4/4.1** | Ratio auxiliary (loss auxiliar) | Senal marginal, no supero umbral de promocion | CERRADO |

### Diagnostico Post Gate 4.1

**Hallazgo critico**: El fine-tuning movia solo MIDI encoder + projections, dejando audio encoder congelado. Esto creaba una asimetria fatal — los puentes cross-modal se alargaban y la separacion entre correctos e incorrectos caia.

**Solucion**: Bloque A — probar distintas estrategias de descongelamiento del audio encoder.

### Bloque A — Foundation Training (5 Runs)

| Run | Estrategia | Best S | Best epoch | hard_neg |
|-----|-----------|--------|------------|----------|
| S0 | Eval-only (control) | 34.4% | - | 80.4% |
| A | Adapters only | 30.0% | 5 | 76.8% |
| B | Unfreeze layers 2-3 | 43.2% | 3 | 85.2% |
| C | Hybrid (adapters 0-1 + unfreeze 2-3) | 49.4% | 5 | 88.4% |
| D | Full unfreeze (split-LR) | 51.0% | 5 | 89.2% |
| **D-02** | **Same as D, 30 epochs** | **61.8%** | **25** | **90.4%** |

**Foundation Lock**: El checkpoint epoch 25 de D-02 se bloqueo como `foundation_locked_e25.pt` (MD5: ddb2ebf7, chmod 444). Todos los experimentos Gate 4.2+ parten de este checkpoint.

### Gate 4.2 — Ratio-Centrico (CERRADO)

**Pregunta**: Los ratio descriptors aportan informacion que el modelo end-to-end no captura solo?

Se probaron 5 descriptores (D0-D4):

| Descriptor | Tipo | Mecanismo | Resultado | vs Baseline |
|------------|------|-----------|-----------|-------------|
| D0 | Sin descriptor (control) | — | S=60.4% | — |
| D1 | Pitch ratio histogram | Loss auxiliar | S=60.6% | +0.2pp (NO) |
| **D4** | **Local interval features** | **Input augmentation (concat)** | **S=64.2% (8ep)** | **+3.8pp (YES)** |

**Resultado**: D4 (inyeccion de intervalos locales en la entrada del encoder MIDI) mejora significativamente. D1 (histograma de ratios) no aporta. El **mecanismo** de inyeccion importa tanto como el descriptor.

### Gate 4.3 — Ratio Re-Centrico (CERRADO, 13 brazos + scratch)

**Pregunta**: Que descriptor Y que mecanismo de inyeccion maximiza la senal de ratios?

Se exploraron 3 ejes:
1. **Lado**: MIDI vs Audio vs Ambos
2. **Descriptor**: D4 intervals vs A4 log-freq deltas vs A7 rational attractor
3. **Mecanismo**: Concat vs Cross-attention vs Cross-modal

**TABLA Fases 0-3 (historica, 9 brazos; sorted by S):**

| Rank | Arm | Mecanismo | Best S | Best ep | hard_neg | vs D0 |
|------|-----|-----------|--------|---------|----------|-------|
| **1** | **d4a4** | **Dual same-mod concat** | **69.8%** | **e5** | **91.6%** | **+9.6pp** |
| 2 | D4 | MIDI intervals concat | 63.6% | e5 | 91.2% | +3.4pp |
| 2 | A4 | Audio desc concat | 63.6% | e5 | 92.4% | +3.4pp |
| 4 | A4x | Audio desc cross-att | 62.6% | e5 | 92.4% | +2.4pp |
| 5 | A7x | Audio attractor cross-att | 62.2% | e5 | 92.0% | +2.0pp |
| 6 | D0 | Baseline | 60.2% | e3 | 90.0% | — |
| 7 | D4x | MIDI intervals cross-att | 60.0% | e5 | 91.4% | -0.2pp |
| 8 | A7 | Audio attractor concat | 58.8% | e5 | 90.2% | -1.4pp |
| 9 | d4a4cm | Dual cross-modal | 52.4% | e5 | 89.6% | -7.8pp |

**Cierre completo Gate 4.3 (13 brazos, 5ep):**
- Mejor brazo: `d4a4` con `S=69.8%` (`+9.6pp` vs D0).
- Mejor mecanismo individual: `A4r` con `S=68.6%`.
- `d4a4-scratch` (30ep) completo en `S=83.6%` (record), multi-seed e30 `84.1% +/- 2.3pp`.

### Hallazgos Clave Gate 4.3

1. **Concat > Cross-attention** consistentemente: D4>D4x, A4>A4x
2. **Same-modality > Cross-modal**: d4a4(69.8%) >> d4a4cm(52.4%), gap de 17.4pp
3. **Dual SUPERADITIVO**: D4 solo = +3.4pp, A4 solo = +3.4pp, juntos = **+9.6pp** (no +6.8pp)
4. **d4a4 seguia subiendo fuerte a e5** (+5.0pp en ultima epoch) — necesita mas epochs
5. **Log-freq (A4) > Attractor (A7)** en todos los mecanismos
6. **Cross-attention rescata descriptores debiles** (A7x=62.2% vs A7=58.8%) pero no supera concat con descriptores fuertes

### d4a4 From Scratch — Completado (30 epochs)

**Pregunta**: Puede d4a4 entrenar desde cero (MERT pretrained + MIDI random, sin foundation) y alcanzar/superar D-02 (61.8%)?

| Ep | S | hard_neg | Nota |
|----|---|----------|------|
| 10 | 74.6% | 93.0% | hito intermedio fuerte |
| 20 | 80.4% | 94.0% | consolidacion |
| 30 | **83.6%** | **95.2%** | **record del proyecto** |

Resultado: `+21.8pp` vs D-02 best. Multi-seed e30: `84.1% +/- 2.3pp`.

---

## 4. Roadmap Actual (Feb 2026)

| Fase | Estado | Descripcion |
|------|--------|-------------|
| Gate 4.3 (13 brazos + scratch) | **COMPLETE** | d4a4 ganador (69.8%) + d4a4-scratch 30ep (83.6%) |
| A4r scratch (30ep) | **EN COLA (UNC)** | corrida larga de mecanismo reverse cross-att |
| Gate 4.4 (Arquitecturas mayores) | PENDING | Third Tower + MoE Ratio Expert |
| Gate 5A (Barrido comprehensivo) | PENDING | Descriptores x mecanismos x cross-modal injection |
| Gate 5B (Showcase cientifico) | PENDING | train largo + bateria de validaciones |

---

## 5. Descriptores de Ratios — Catalogo Resumido

### MIDI-side

| ID | Nombre | Dims | Que computa | Resultado |
|----|--------|------|------------|-----------|
| D0 | Baseline | 0 | Sin descriptor | S=60.2% |
| D1 | Pitch ratio histogram | [B, 128] | Histograma de f_i/f_j | S=60.6% (NO) |
| D2 | Enriched 3-channel | [B, 384] | D1 + velocity + duration | No evaluado |
| D3 | Temporal-rhythmic | [B, 153] | IOI ratios + duration ratios + pitch intervals | No evaluado |
| **D4** | **Local intervals** | **[B, N, 4]** | **semitone_prev, semitone_next, log_ratio_prev, log_ratio_next** | **S=63.6% (YES)** |

### Audio-side

| ID | Nombre | Dims | Que computa | Resultado |
|----|--------|------|------------|-----------|
| **A4** | **Log-freq deltas** | **[B, T, 8]** | **STFT -> 8 bandas log-freq -> delta temporal -> normalizar** | **S=63.6% (YES)** |
| A7 | Rational attractor | [B, T, 12] | STFT -> peaks -> pairwise log2 ratios -> soft assignment a 12 atractores JI | S=58.8% (debajo baseline) |

### Mecanismos de inyeccion

| Mecanismo | Clave | Params tipicos |
|-----------|-------|---------------|
| **Concat** | Concatenar descriptor a features, proyectar back | ~267K (MIDI), ~1M (audio) |
| Cross-attention | Q=features, K/V=descriptor. Atencion dinamica | ~1M (MIDI), ~4.2M (audio) |
| Cross-modal | Descriptor de otro dominio al encoder | ~1.3M |

---

## 6. Hallazgos Cientificos Consolidados

### Confirmados

1. **Representacion > Arquitectura**: Consistente desde v4.1→v5.0 hasta D4 vs D1. Como se codifican los datos importa mas que que modelo se usa.

2. **Input augmentation > Loss auxiliar**: D4 (concat en entrada) supera consistentemente a D1-D3 (loss auxiliar). Enriquecer la entrada es mas efectivo que agregar una senal de training.

3. **Concat > Cross-attention para descriptores fuertes**: La mezcla lineal simple funciona mejor que la atencion dinamica cuando el descriptor ya es informativo.

4. **Same-modality > Cross-modal**: Cada encoder se beneficia de los ratios de su propia senal (+9.6pp), pero se perjudica al recibir ratios de la otra modalidad (-7.8pp).

5. **Superaditividad dual**: D4+A4 juntos aportan mas que la suma de sus contribuciones individuales. Sugiere complementariedad informacional.

6. **Log-freq > Attractor JI**: El descriptor simple (A4: deltas temporales en bandas de frecuencia) supera al teoricamente elegante (A7: atractores de afinacion justa). La naturaleza temperada de MAESTRO puede explicar esto.

### Resultado Negativo Importante

- **DANN destruye informacion util**: Gradient Reversal Layer para domain invariance no mejora en ningun regimen sostenido (Gate 3).
- **Cross-modal injection es destructiva**: Dar a un encoder los ratios del otro dominio empeora el rendimiento (d4a4cm, -7.8pp).

---

## 7. Evolucion de Performance

```
Gate 2 baseline:     S = 34.4%   (audio congelado)
Run D-02:            S = 61.8%   (full unfreeze, 25 epochs)
Gate 4.3 D0:         S = 60.2%   (5ep desde foundation)
Gate 4.3 D4:         S = 63.6%   (MIDI intervals concat)
Gate 4.3 A4:         S = 63.6%   (audio log-freq concat)
Gate 4.3 d4a4:       S = 69.8%   (dual same-mod, still climbing)
```

Mejora total desde Gate 2: **+35.4pp** (34.4% -> 69.8%)

---

## 8. Estructura del Repositorio

```
/mnt/m2-1TB/Phideus/
├── src/
│   ├── bias_control/              # Codigo principal actual
│   │   ├── architectures/         # CrossModalModel (MERT + MIDI + VICReg)
│   │   ├── encoders/              # MERTEncoder, MERTEncoderLite, MIDIEncoder
│   │   ├── datasets/              # MaestroSegmentDataset
│   │   ├── losses/                # VICReg, DANN
│   │   ├── adapters/              # Adapter modules (Bloque A)
│   │   ├── audio_descriptors.py   # A4 (log-freq), A7 (attractor)
│   │   └── ratio_descriptors.py   # D3 (temporal-rhythmic), D4 (local intervals)
│   ├── RNA/                       # Modelos historicos
│   │   ├── vicreg.py              # VICReg loss
│   │   ├── roseta_vae.py          # Dual-domain VAE (UOEMD)
│   │   ├── constellation_vae.py   # Sparse token VAE
│   │   └── jepa_lite.py           # JEPA sin decoder
│   ├── hrm/                       # Hierarchical Reasoning Model
│   ├── analizador/                # Extractores de ratios (v5.0, roseta, maestro)
│   ├── extractors/                # Route A/B extractors
│   └── datasets/                  # Dataset loaders
│
├── experiments/
│   ├── bias_control/
│   │   ├── gate42_training.py     # Training Gate 4.2/4.3 (TODOS los modelos)
│   │   ├── gate43_scratch/        # From-scratch experiment
│   │   ├── bloqueA_training.py    # Foundation training (Runs A-D)
│   │   ├── evaluate_structured_pool.py  # Evaluacion canonica
│   │   └── run_gate43.sh          # Script wrapper
│   └── maestro/                   # Escalon 1 (gate0-gate5)
│
├── viz/                           # 8 visualizaciones WebGL2 interactivas
│   └── src/{phideus,crossatt,dann,hrm,constellation,jepa,bloquea,roseta}/
│
├── Documents/
│   ├── 00_TRONCAL/                # Proyecto_Estado_Actual, bitacora
│   ├── 01_FRENTES_ACTIVOS/        # BIAS_CONTROL (activo), ESCALON_1
│   ├── 02_FRENTES_PAUSADOS/       # VibeTensor (pausado)
│   ├── 03_FRENTES_CERRADOS/       # UOEMD (NO-GO)
│   └── 04_TRANSVERSAL/            # Teoria, analisis externos
│
├── data/                          # Datasets + outputs (NO en git)
│   ├── maestro_v3/maestro-v3.0.0/ # 121GB, 1276 WAV + 1276 MIDI
│   └── bias_control_medium/training_outputs/  # Checkpoints y resultados
│
├── CLAUDE.md                      # Instrucciones para Claude Code
├── NOTAS_CLAUDE_PARA_CODEX.md     # Bitacora detallada Gate 4.3
└── README.md                      # Overview publico
```

---

## 9. Equipo y Herramientas

- **GPU**: NVIDIA RTX 3090 (24GB VRAM)
- **CPU**: Intel i5-12600K (16 cores)
- **Storage**: NVMe 1TB (315GB used, 555GB free) + RAID1 backup (5TB)
- **Framework**: PyTorch + torchaudio
- **Agentes AI**: Claude Code (implementacion + ejecucion) + Codex (documentacion + auditoria)
- **Visualizacion**: Next.js + WebGL2 custom renderer → GitHub Pages

---

## 10. Glosario

| Termino | Significado |
|---------|-------------|
| **S** | Score = min(A2M_R@10, M2A_R@10). Metrica canonica de equilibrio bidireccional |
| **A2M** | Audio-to-MIDI retrieval |
| **M2A** | MIDI-to-Audio retrieval |
| **R@K** | Recall at K: positivo correcto en top-K |
| **hard_neg** | Accuracy en hard negatives (misma pieza, distinto tiempo) |
| **MRR** | Mean Reciprocal Rank |
| **VICReg** | Variance-Invariance-Covariance Regularization (loss contrastiva) |
| **DANN** | Domain Adversarial Neural Network (gradient reversal) |
| **MERT** | Music Understanding Model with Large-Scale Self-supervised Training |
| **Foundation** | Checkpoint `foundation_locked_e25.pt` (D-02 epoch 25, S=61.8%) |
| **concat** | Mecanismo: concatenar descriptor a features + Linear projection |
| **cross-att** | Mecanismo: cross-attention (Q=features, K/V=descriptor) |
| **d4a4** | Dual same-modality concat: D4→MIDI + A4→Audio |
| **d4a4cm** | Dual cross-modal: D4→Audio + A4→MIDI (resultado negativo) |
| **pp** | Percentage points (diferencia absoluta entre porcentajes) |

---

## 11. Documentos de Referencia

| Documento | Contenido |
|-----------|-----------|
| `PHIDEUS_NEURAL_ARCHITECTURES.md` | Detalle tecnico de cada red, hiperparametros, freeze policies |
| `ROADMAP_BIAS_CONTROL.md` | Framework completo de Gates 0-6 |
| `NOTAS_CLAUDE_PARA_CODEX.md` | Bitacora operativa detallada Gate 4.3 |
| `INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md` | Evolucion de todas las representaciones |
| `CATALOGO_NARRATIVO_DESCRIPTORES_RATIOS_PHIDEUS.md` | Cada descriptor con historia y lecciones |
| `BACKPROPAGANDO_PHIDEUS.md` | Redefinicion epistemologica |
| `plan_gate_4.2.md` | Diseno experimental Gate 4.2 (57K, muy detallado) |
| `Explicacion_gate4.2_claude.md` | Narrativa accesible de Gate 4.2 |
