<div align="center">

# Phideus

### Harmonic Information Theory — Research Program

![Status](https://img.shields.io/badge/Status-Active_Research-0A7E3B?style=for-the-badge)
![Gate](https://img.shields.io/badge/Gate_4.4-Cerrado_24_brazos_%2B_30ep-F59E0B?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-111827?style=for-the-badge)

*Do frequency ratios constitute a universal informational language?*

</div>

---

**Phideus** investiga si los ratios armonicos de frecuencia (3:2, 5:4, 7:4...) funcionan como unidades fisicas de informacion transferibles entre modalidades. El banco de pruebas actual es **Audio <-> MIDI** cross-modal retrieval sobre MAESTRO, con entrenamiento contrastivo (VICReg) y evaluacion estructurada.

> **Foco actual**: corrida larga comparativa en UNC con **batch 60ep** (`D0`, `d4a4`, `a4r`, `d4-a4r`, `moe-dual`) + `t3-wt` scratch 50ep con scheduler trapezoidal (`--lr-hold-fraction=0.5`).
> **Hallazgo clave**: Gate 4.4 cerró screening de 24 brazos (21 originales + MoE v2/v3/v4). En 30ep scratch ya cerrados, el ranking es `d4a4=83.6%`, `a4r=82.0%`, `d4-a4r=79.8%`, `t3-wt=79.8%`, `d4a4r=74.4%`, `moe-dual=72.6%`.
> **Arquitecturas**: explora las redes del proyecto en visualizaciones 3D interactivas → **[altermundi.github.io/Phideus](https://altermundi.github.io/Phideus/)**

---

## Glosario

| Termino | Definicion |
|---------|-----------|
| **S** | `min(A2M R@10, M2A R@10)` — metrica canonica de retrieval (mayor = mejor) |
| **R@10** | Recall at 10: fraccion de matches correctos en el top 10 de un pool de 256 |
| **A2M / M2A** | Audio-to-MIDI / MIDI-to-Audio — direccion de retrieval |
| **hard_neg** | Precision distinguiendo un segmento de otra seccion de la *misma pieza* |
| **D0** | Modelo baseline (sin inyeccion de descriptores) |
| **pp** | Percentage points (diferencia absoluta) |
| **VICReg** | Variance-Invariance-Covariance Regularization (loss contrastiva) |

---

## Hipotesis de Investigacion

| Hipotesis | Estado | Evidencia |
|-----------|--------|-----------|
| **H1 — Estructura** | **Validada** | Distribuciones de ratios no aleatorias en multiples tipos de senal |
| **H2 — Aprendibilidad** | **Validada** | Redes neuronales aprenden representaciones compactas de ratios (val_loss < 0.5) |
| **H3 — Cross-modality** | **Prometedor** | Gate 4.3 consolidó `d4a4=69.8%` (+9.6pp) y `d4a4-scratch e30=83.6%`. Gate 4.4 cerró screening de 24 brazos; en runs largos, `t3-wt` y `d4-a4r` empatan en `79.8%` y `moe-dual` llega a `72.6%`. |

---

## Resultados Clave — Gate 4.3: Inyeccion Ratio-Centrica

13 brazos, 5 epochs cada uno desde `foundation_locked_e25.pt`, freeze-policy run-d.
Evaluacion estructurada: pool=256, queries=500, seed=42.

| Rank | Brazo | Descriptor | Mecanismo | Best S | hard_neg | vs D0 |
|------|-------|-----------|-----------|--------|----------|-------|
| **1** | **d4a4** | **MIDI intervals + Audio log-freq** | **Dual same-mod concat** | **69.8%** | **91.6%** | **+9.6pp** |
| 2 | A4r | Audio log-freq deltas (8d) | Reverse cross-att | 68.6% | 91.6% | +8.4pp |
| 3 | D4r | MIDI intervals (4d) | Reverse cross-att | 64.2% | 93.2% | +4.0pp |
| 4 | D4 | MIDI intervals (4d) | Concat | 63.6% | 91.2% | +3.4pp |
| 4 | A4 | Audio log-freq deltas (8d) | Concat | 63.6% | 92.4% | +3.4pp |
| 6 | A4x | Audio log-freq deltas (8d) | Cross-attention | 62.6% | 92.4% | +2.4pp |
| 7 | A7x | Audio rational attractor (12d) | Cross-attention | 62.2% | 92.0% | +2.0pp |
| 8 | D0 | — | Baseline | 60.2% | 90.0% | — |
| 9 | D4x | MIDI intervals (4d) | Cross-attention | 60.0% | 91.4% | -0.2pp |
| 10 | A9 | IDF-weighted attractor (12d) | Concat | 58.8% | 90.4% | -1.4pp |
| 10 | A7 | Audio rational attractor (12d) | Concat | 58.8% | 90.2% | -1.4pp |
| 12 | A8 | Onset-weighted chroma (12d) | Concat | 57.4% | 90.6% | -2.8pp |
| 13 | d4a4cm | MIDI intervals + Audio log-freq | Dual cross-modal | 52.4% | 89.6% | -7.8pp |

**Observaciones** (sin extrapolacion — [directiva analitica](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md)):
- **d4a4 es superaditivo**: D4 solo = +3.4pp, A4 solo = +3.4pp, d4a4 = +9.6pp (no +6.8pp).
- **Reverse cross-att es el mejor mecanismo individual**: A4r = 68.6% con un solo descriptor, casi iguala a d4a4 dual (69.8%).
- **Same-modality >> cross-modal**: d4a4cm (cross-modal injection) destruye senal (-7.8pp).
- **Reverse > forward cross-att**: A4r (+8.4pp) >> A4x (+2.4pp); D4r (+4.0pp) > D4x (-0.2pp).
- **A4/D4 son los mejores descriptores**: A8 y A9 no superan baseline. A7 tampoco.

### d4a4 from-scratch — 30 epochs (COMPLETO)

Mismo punto de partida que D-02 (MERT pretrained + random MIDI). Unica diferencia: inyeccion d4a4.
Duracion: 636 min (10.6h). Best model: epoch 30.

| Epoch | Loss | S | A2M | M2A | hard_neg | MRR | R@1 | mean_rank | vs D-02 best |
|-------|------|---|-----|-----|----------|-----|-----|-----------|--------------|
| 10 | 13.60 | 74.6% | 74.6% | 75.0% | 93.0% | 0.336 | 15.9% | 7.7 | +12.8pp |
| 15 | 13.38 | 65.8% | 65.8% | 68.6% | 91.0% | 0.316 | 16.4% | 10.0 | +4.0pp |
| 20 | 13.26 | 75.6% | 75.6% | 76.8% | 93.6% | 0.370 | 19.0% | 7.0 | +13.8pp |
| 25 | 13.21 | 82.2% | 82.8% | 82.2% | 95.4% | 0.430 | 25.2% | 5.7 | +20.4pp |
| 28 | 13.19 | 82.8% | 82.8% | 83.6% | 94.8% | 0.444 | 26.4% | 5.6 | +21.0pp |
| 29 | 13.19 | 82.6% | 82.6% | 83.8% | 95.2% | 0.443 | 26.3% | 5.4 | +20.8pp |
| **30** | **13.20** | **83.6%** | **84.0%** | **83.6%** | **95.2%** | **0.444** | **25.9%** | **5.4** | **+21.8pp** |

D-02 best = S=61.8% (epoch 25). El modelo seguia mejorando en e30 (no saturado).

---

## Visualizaciones Interactivas de Arquitectura

Exploraciones 3D de las redes neuronales del proyecto:
**[altermundi.github.io/Phideus](https://altermundi.github.io/Phideus/)**

| Visualizacion | Arquitectura | Descripcion |
|--------------|-------------|-------------|
| [Cross-Attention Injection](https://altermundi.github.io/Phideus/crossatt) | Gate 4.3 | Inyeccion de descriptores de ratios via cross-attention |
| [MERT + MIDI Transformer](https://altermundi.github.io/Phideus/phideus) | Run D Foundation | Arquitectura cross-modal Audio+MIDI (foundation model) |
| [Hybrid Adapter Fine-Tuning](https://altermundi.github.io/Phideus/bloquea) | BloqueA Run C | Adapters en capas congeladas + unfreeze parcial |
| [Domain Adversarial Network](https://altermundi.github.io/Phideus/dann) | Gate 3 DANN | Gradient reversal layer para embeddings domain-invariant |
| [Hierarchical Reasoning Model](https://altermundi.github.io/Phideus/hrm) | HRM | L-Module (rapido) + H-Module (lento) con Adaptive Computation Time |
| [ConstellationVAE](https://altermundi.github.io/Phideus/constellation) | C1-C4 | VAE con tokens sparse, encoder/decoder modular |
| [JEPA-Lite](https://altermundi.github.io/Phideus/jepa) | Sin decoder | Arquitectura predictiva bidireccional con stop-gradient e InfoNCE |
| [RosetaVAE](https://altermundi.github.io/Phideus/roseta) | Dual-Domain | Factorizacion shared/private para Audio-Vibracion |

> Adaptado de [bbycroft/llm-viz](https://github.com/bbycroft/llm-viz) (MIT).

---

<!-- BELOW THE FOLD -->

<details>
<summary><strong>Roadmap</strong></summary>

### Progresion de Gates

```mermaid
flowchart LR
  G02["Gates 0-2\nFoundation"] --> G3["Gate 3\nDANN"]
  G02 --> G4["Gates 4.0-4.1\nRatio Aux"]
  G4 --> G6["Gate 6\nRetroanalysis"]
  G6 --> BA["Bloque A\nRecovery"]
  BA --> G42["Gate 4.2\nRatio-Centrico"]
  G42 --> G43["Gate 4.3\n13 brazos + scratch"]
  G43 --> G44["Gate 4.4\nThird Tower + FiLM + MoE"]
  G44 --> G5A["Gate 5A\nBarrido"]
  G44 --> G5B["Gate 5B\nShowcase"]

  style G02 fill:#dcfce7,stroke:#16a34a
  style G3 fill:#fee2e2,stroke:#dc2626
  style G4 fill:#e5e7eb,stroke:#6b7280
  style G6 fill:#dcfce7,stroke:#16a34a
  style BA fill:#dcfce7,stroke:#16a34a
  style G42 fill:#dcfce7,stroke:#16a34a
  style G43 fill:#fef3c7,stroke:#d97706
  style G44 fill:#dbeafe,stroke:#2563eb
  style G5A fill:#dbeafe,stroke:#2563eb
  style G5B fill:#dbeafe,stroke:#2563eb
```

### Matriz de Gates

| Gate | Proposito | Estado | Decision |
|------|----------|--------|----------|
| Gate 0 | Integridad de datos | Completado | GO |
| Gate 1 | Baselines intra-modales | Completado | GO |
| Gate 2 | Foundation baseline | Completado | GO (R@10=34.4%, hard_neg=80.4%) |
| Gate 2.5 | Sonda de separabilidad | Completado | GO (diagnostico) |
| Gate 3 | DANN | **Cerrado** | **NO-GO** (sin mejora robusta) |
| Gate 4.0 | Ratio auxiliary | Completado | Senal mixta |
| Gate 4.1 | Matriz causal | **Cerrado** | **NO-GO** (dS=+0.8pp < umbral) |
| Gate 6 | Retroanalysis | Completado | Causa raiz confirmada (audio encoder congelado) |
| Gate 4.2 | Pre-red dual-domain | **Cerrado** | **NO-GO** (AUC ~0.50) |
| Bloque A | Recovery (S0/A/B/C/D) | Completado | D-02 e25 -> foundation lock |
| **Gate 4.3** | **Ratio re-centrico (13 brazos + scratch)** | **Cerrado** | **d4a4-scratch=83.6% (record)** |
| Gate 4.4 | Third tower + FiLM + MoE | **Cerrado (screening + 30ep clave)** | Screening 24 brazos cerrado; runs largos t3-wt/moe-dual cerrados |
| Gate 5A | Barrido descriptor x mecanismo + cross-modal | Pending | |
| Gate 5B | Showcase cientifico (13 tests) | Pending | |

### TripleScaloneta

| Escalon | Dominio | Estado | Criterio de avance |
|---------|---------|--------|--------------------|
| **1** | MAESTRO Audio <-> MIDI | **Activo** (batch 60ep + t3-wt 50ep hold) | Medir efecto de extensión de LR/epochs antes de Gate 5A/5B |
| 2 | Speech <-> EGG | Planificado | Cierre robusto de Escalon 1 |
| 3 | ECG <-> PPG | Proyeccion | Evidencia de generalidad en Escalon 2 |

### Roadmap Visual Interactivo

Visualizacion detallada con timeline, tablas de resultados y status en tiempo real:
`Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/roadmap_visual.html` (abrir en navegador).

</details>

<details>
<summary><strong>Arquitectura</strong></summary>

### Foundation Model (Run D)

```
Audio (waveform 96kHz x 4s) -> MERTEncoderLite (4 CNN + 4 Transformer, d=1024) -> MLP -> [B, 256]
MIDI  (event sequence)      -> 4x Transformer (d=512)                          -> MLP -> [B, 256]
                                        VICReg Loss (inv=10, var=10, cov=1)
```

| Componente | Parametros | Detalles |
|-----------|-----------|---------|
| Audio encoder | ~60M | 4 CNN stages + 4 Transformer layers, output d=1024 |
| MIDI encoder | ~13M | 4 Transformer layers, d=512 |
| Proyecciones | ~0.5M | Audio: 1024->512->256, MIDI: 512->512->256 |
| d4a4 inyeccion | ~1.3M | interval_projection (516->512+LN) + audio_descriptor_projection (1032->1024+LN) |
| Foundation | bloqueado | `foundation_locked_e25.pt` (MD5: ddb2ebf7, chmod 444) |

### Descriptores de Ratios (Gate 4.3)

**D4 — MIDI intervals** (4d por token): pitch diff, IOI, velocity diff, duration ratio.
Computado desde eventos MIDI. Inyectado post-embedding, pre-Transformer.

**A4 — Audio log-frequency deltas** (8d por frame): 8 bandas log-freq, delta temporal, normalizado.
Computado desde STFT (n_fft=2048, hop=512). Inyectado post-CNN, pre-Transformer.

**A7 — Rational attractor** (12d por frame): asignacion suave de ratios pairwise de picos a 12 atractores de afinacion justa.
Testea la hipotesis Phideus directamente: la proximidad a ratios JI (3:2, 5:4, etc.) lleva senal cross-modal?

**Mecanismos de inyeccion testeados**: concat (proyeccion lineal), cross-attention (Q=features, K/V=descriptor), reverse cross-attention (Q=descriptor, K/V=features — 12.8x menos compute), dual (ambos encoders).

</details>

<details>
<summary><strong>Foundation Training — Bloque A</strong></summary>

### Comparacion de Estrategias de Fine-Tuning

| Run | Estrategia | Best S | Best Epoch | hard_neg |
|-----|-----------|--------|------------|----------|
| S0 | Eval-only (control) | 34.4% | — | 80.4% |
| A | Adapters only | 30.0% | 5 | 76.8% |
| B | Unfreeze capas 2-3 | 43.2% | 3 | 85.2% |
| C | Hibrido (adapters 0-1 + unfreeze 2-3) | 49.4% | 5 | 88.4% |
| D | Full unfreeze (split-LR) | 51.0% | 5 | 89.2% |
| **D-02** | **Full unfreeze, 30 epochs** | **61.8%** | **25** | **90.4%** |

D-02 epoch 25 fue bloqueado como `foundation_locked_e25.pt` — punto de partida para todos los brazos de Gate 4.3.

</details>

<details>
<summary><strong>Reproduccion / Quick Start</strong></summary>

### Setup del Entorno

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Pipeline Completo

```bash
python experiments/bias_control/run_all_gates.py \
  --maestro-dir data/maestro_v3/maestro-v3.0.0 \
  --output data/bias_control_medium
```

### Gate 4.3 (ejemplo: brazo d4a4)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python experiments/bias_control/gate42_training.py \
  --descriptor d4a4 \
  --checkpoint data/bias_control_medium/training_outputs/foundation_locked_e25.pt \
  --output data/bias_control_medium/training_outputs/gate43/d4a4 \
  --maestro-dir data/maestro_v3/maestro-v3.0.0 \
  --epochs 5 --batch-size 16 --num-workers 8 \
  --freeze-policy run-d --seed 42 --device cuda
```

### Evaluacion Estructurada

```bash
python experiments/bias_control/evaluate_structured_pool.py \
  --model <checkpoint.pt> \
  --output <output.json> \
  --pool-size 256 --n-queries 500 --seed 42 \
  --maestro-dir data/maestro_v3/maestro-v3.0.0
```

**Protocolo canonico**: pool=256, queries=500, seed=42.

</details>

<details>
<summary><strong>Experimentos Anteriores</strong></summary>

### Escalon 1 — MAESTRO Hashing (pausado)

Identificacion cross-modal basada en hashing. Route B alcanzo 80% piece accuracy (N=10), pero la resolucion temporal del onset limita el escalado.

| Experimento | Route A | Route B | vs Random |
|-----------|---------|---------|-----------|
| N=10 (corregido) | 42.5% | 32.9% | 4.2x / 3.3x |
| N=20 (replicacion) | 26.6% | 21.4% | 5.3x / 4.3x |

### UOEMD Revisionismo (cerrado — NO-GO)

Dual-domain (Audio <-> Vibracion) con 128 muestras. Multiples enfoques testeados (histogramas, constellation tokens, JEPA-lite). Dataset insuficiente para validar H3.

### Resumen Experimental

| Bloque | Resultado | Lectura |
|--------|-----------|---------|
| Escalon 1 hashing | 27% piece accuracy | Senal detectada, insuficiente para cierre robusto |
| UOEMD revisionismo (F0-F3A) | NO-GO | Confirmados limites de dataset/regimen |
| BIAS_CONTROL Gate 2 | Gap 0.478, R@10=34.4% | Baseline operativo |
| BIAS_CONTROL Gate 3 (DANN) | 4 runs, sin mejora | Invariancia modal no era el cuello |
| BIAS_CONTROL Gate 4.0-4.1 | Mixto -> cerrado | Control causal insuficiente |
| BIAS_CONTROL Gate 6 + 4.2 | Diagnostico completado | Causa raiz: audio encoder congelado |
| **BIAS_CONTROL Gate 4.3** | **13 brazos; d4a4=69.8%, A4r=68.6% (5ep); scratch=83.6% (30ep)** | **Inyeccion de ratios funciona, superaditiva. Reverse cross-att mejor mecanismo individual.** |
| **BIAS_CONTROL Gate 4.4** | **24 brazos cerrados (incl. MoE v2/v3/v4) + 2 runs largos extra** | **Third Tower (t3-wt) gana familia 4.4; FiLM/MoE en banda 58-60% a 5ep.** |
| **Runs largos scratch (30ep)** | **d4a4 83.6 > a4r 82.0 > d4-a4r/t3-wt 79.8 > d4a4r 74.4 > moe-dual 72.6** | **Base comparativa cerrada para abrir batch 60ep.** |

### Hallazgos Metodologicos

1. **La metrica canonica debe ser structured pool**, no solo metricas globales de validacion.
2. **Forzar invariancia (DANN) puede destruir senal de retrieval** si la hipotesis causal no esta bien acotada.
3. **Comparabilidad de regimen** (pool, seed, batch budget) es critica para decisiones correctas.

</details>

<details>
<summary><strong>Documentacion</strong></summary>

### Documentos Principales

| Documento | Descripcion |
|----------|------------|
| [ROADMAP_BIAS_CONTROL.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md) | Plan maestro y criterios de decision |
| [INDEX_BIAS_CONTROL.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/INDEX_BIAS_CONTROL.md) | Navegacion por fases y arbol documental |
| [Proyecto_Estado_Actual.md](Documents/00_TRONCAL/Proyecto_Estado_Actual.md) | Estado ejecutivo del proyecto |
| [INDICE_DOCUMENTACION.md](Documents/00_TRONCAL/INDICE_DOCUMENTACION.md) | Mapa completo de documentacion |

### Documentacion por Gate

| Gate | Documento |
|------|----------|
| Gate 2 | [INFORME_GATE2_COMPLETO.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/01_GATES_0_2_5/GATE_2_FOUNDATION/INFORME_GATE2_COMPLETO.md) |
| Gate 3 | [INFORME_GATE3_COMPLETO.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/02_GATE_3_DANN/INFORME_GATE3_COMPLETO.md) |
| Diagnostico | [INFORME_DEC005_DIAGNOSTICO_COMPLETO.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/04_DIAGNOSTICO_GATE_6_Y_GATE_4_2/INFORME_DEC005_DIAGNOSTICO_COMPLETO.md) |
| Bloque A | [PLAN_EJECUCION_POST_DEC005_v1.1.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/05_PLAN_POST_DIAGNOSTICO_BLOQUE_A/PLAN_EJECUCION_POST_DEC005_v1.1.md) |
| Gate 4.2 | [plan_gate_4.2.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/06_GATE_4_2_RATIO_CENTRICO/plan_gate_4.2.md) |
| Gate 4.4 | [README.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/08_GATE_4_4_ARQUITECTURAS_MAYORES/README.md) |
| Gate 5A | [README.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/09_GATE_5_LINEA_A_BARRIDO/README.md) |
| Gate 5B | [README.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/10_GATE_5_LINEA_B_SHOWCASE/README.md) |

### Estructura del Repositorio

```
Phideus/
├── src/bias_control/          # Modulos core (encoders, losses, descriptors)
├── experiments/bias_control/  # Scripts de training, evaluacion, gates
├── Documents/
│   ├── 00_TRONCAL/            # Documentos ejecutivos
│   ├── 01_FRENTES_ACTIVOS/    # Frentes activos (BIAS_CONTROL, ESCALON_1)
│   ├── 02_FRENTES_PAUSADOS/   # Pausados (VibeTensor)
│   ├── 03_FRENTES_CERRADOS/   # Cerrados (UOEMD)
│   └── 04_TRANSVERSAL/        # Transversal (teoria, visualizaciones)
├── viz/                       # Visualizaciones WebGL interactivas (Next.js)
├── data/                      # Datasets y outputs (no en git)
└── config/                    # Configuraciones
```

> `data/`, `models/`, `train/` y otros artefactos pesados no se versionan.

</details>

---

> *"El bosque ya canta. Nuestra tarea es entender su afinacion."*

**Licencia**: MIT — ver [LICENSE.md](LICENSE.md)
