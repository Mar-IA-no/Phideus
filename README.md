<div align="center">

# Phideus v5.0
### Harmonic Information Theory Research

![Program](https://img.shields.io/badge/Program-Research_Active-0A7E3B?style=for-the-badge)
![Focus](https://img.shields.io/badge/Focus-BIAS_CONTROL-1F6FEB?style=for-the-badge)
![Escalon](https://img.shields.io/badge/Escalon-1--C-F59E0B?style=for-the-badge)
![Current Gates](https://img.shields.io/badge/Current-Gate_6_%2B_Gate_4.2_Diagnostic-7C3AED?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-111827?style=for-the-badge)

</div>

> [!IMPORTANT]
> **Estado**: programa de investigacion activo  
> **Ultima actualizacion**: 2026-02-11  
> **Foco actual**: `BIAS_CONTROL` (Escalon 1-C: Gate 6 + Gate 4.2 diagnostico, `DEC-005`)

---

## Navegacion Rapida

- [Resumen](#resumen)
- [Lineas Experimentales](#lineas-experimentales-del-proyecto)
- [Plan Operativo BIAS_CONTROL](#plan-operativo-bias_control-escalon-1-c)
- [Proyeccion TripleScaloneta](#proyeccion-triplescaloneta)
- [Concepto Central](#concepto-central)
- [Hallazgos Principales](#hallazgos-principales)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Quick Start](#quick-start)
- [Documentacion](#documentacion)
- [Hipotesis de Investigacion](#hipotesis-de-investigacion)
- [Arquitectura BIAS_CONTROL](#arquitectura-bias_control-resumen)

---

## Resumen

Phideus investiga si las **relaciones armonicas (ratios de frecuencia)** pueden funcionar como un lenguaje fisico transferible entre modalidades.

La linea principal hoy es **audio <-> MIDI** sobre MAESTRO, con entrenamiento contrastivo y evaluaciones estructuradas para retrieval cross-modal en escenarios dificiles (`hard negatives`).

### Estado de Hipotesis (Febrero 2026)

| Hipotesis | Estado | Evidencia actual |
|-----------|--------|------------------|
| **H1: Estructura** | VALIDADA | Distribuciones de ratios no aleatorias en multiples contextos |
| **H2: Aprendibilidad** | VALIDADA | VAE/HRM con `val_loss < 0.5` en representaciones de ratios |
| **H3: Cross-modality** | PROMETEDOR | `BIAS_CONTROL` (Gate 2) con gap robusto y buen `hard_neg_acc` |

### Experimento Actual: BIAS_CONTROL (Escalon 1)

```mermaid
flowchart LR
  G0["Gate 0<br/>Data Integrity"] --> G1["Gate 1<br/>Intra-modal Baselines"]
  G1 --> G2["Gate 2<br/>Foundation Baseline"]
  G2 --> G25["Gate 2.5<br/>Separability Probe"]
  G25 --> G3["Gate 3<br/>DANN"]
  G3 --> G4["Gate 4<br/>Ratio Auxiliary Base"]
  G4 --> G41["Gate 4.1<br/>Causal Matrix DEC-004/004-A"]
  G41 --> G6["Gate 6<br/>Retroanalysis (DEC-005)"]
  G6 --> G42["Gate 4.2<br/>Dual-domain Diagnostics"]
  G42 --> G5["Gate 5<br/>Optional"]

  style G2 fill:#dcfce7,stroke:#16a34a,color:#111827
  style G3 fill:#fee2e2,stroke:#dc2626,color:#111827
  style G4 fill:#dbeafe,stroke:#2563eb,color:#111827
  style G41 fill:#e5e7eb,stroke:#6b7280,color:#111827
  style G6 fill:#fef3c7,stroke:#d97706,color:#111827
  style G42 fill:#fce7f3,stroke:#be185d,color:#111827
```

#### Controles de navegacion por gate

- [Gate 0 - Data Integrity](#gate-0---data-integrity)
- [Gate 1 - Intra-modal Baselines](#gate-1---intra-modal-baselines)
- [Gate 2 - Foundation Baseline](#gate-2---foundation-baseline)
- [Gate 2.5 - Embedding Diagnostics](#gate-25---embedding-diagnostics)
- [Gate 3 - DANN](#gate-3---dann-cerrado)
- [Gate 4 - Ratio Auxiliary](#gate-4---ratio-auxiliary-base-completado)
- [Gate 4.1 - Matriz Causal](#gate-41---matriz-causal-dec-004004-a-cerrado)
- [Gate 4.2 - Diagnóstico Dual-Domain](#gate-42---dual-domain-ratios-diagnostico-dec-005)
- [Gate 5 - Curriculum](#gate-5---curriculum-opcional)
- [Gate 6 - Retroanalysis](#gate-6---retroanalysis-diagnostico-aprobado)

#### Matriz de control (estado actual)

| Gate | Rol en el roadmap | Estado | Decision |
|------|--------------------|--------|----------|
| Gate 0 | Integridad de datos y alineacion | Completado | GO |
| Gate 1 | Baselines intra-modales | Completado | GO |
| **Gate 2** | Baseline cross-modal principal | **Completado** | **GO (checkpoint de referencia)** |
| Gate 2.5 | Diagnostico de separabilidad | Completado | GO (diagnostico, no objetivo final) |
| **Gate 3 (DANN)** | Prueba de invariancia modal | **Cerrado** | **NO-GO (sin mejora robusta)** |
| **Gate 4 (Ratio Auxiliary base)** | Test inicial de señal de ratios | **Completado** | Señal mixta, requiere control causal |
| **Gate 4.1 (DEC-004/004-A)** | Matriz causal por fases | **Cerrado** | `R1-rescue` no supera umbral (`dS=+0.8pp`) |
| **Gate 6 (DEC-005)** | Retroanalisis representacional | **Aprobado (diagnostic-only)** | Track 1 paralelo |
| **Gate 4.2 (H4.2-6)** | Diagnostico dual-domain ratios | **Aprobado (diagnostic-only)** | Track 2 paralelo (P0/P1) |
| Gate 5 | Curriculum/extensiones | Hold | Opcional, no prioritario |

Metricas clave del baseline actual (Gate 2, `checkpoint_epoch45`):

| Metrica | Valor |
|---------|-------|
| Gap (aligned - random) | **0.478** |
| Recall@10 structured pool (a2m) | **34.4%** |
| Recall@10 structured pool (m2a) | **37.6%** |
| Hard Negative Accuracy | **80.4%** |

---

## Lineas Experimentales del Proyecto

Phideus hoy opera con dos enfoques que se complementan:

| Enfoque | Objetivo | Hallazgo principal | Estado |
|---------|----------|-------------------|--------|
| **Pre-analisis (hashing/ratios)** | Medir si la estructura de ratios contiene señal discriminativa antes de modelos grandes | Señal real detectada (`vs random 5.4x`), pero rendimiento insuficiente para cierre fuerte en setup historico | Pausado como linea principal |
| **BIAS_CONTROL (cross-modal contrastivo)** | Validar alineacion Audio<->MIDI robusta con control de sesgo | Gate 2 baseline robusto; Gate 3 cerrado; Gate 4.1 cerrado; fase actual en diagnostico controlado (`DEC-005`) | Linea principal activa |

### Experimentos ya realizados (resumen corto)

| Bloque | Resultado | Lectura |
|--------|-----------|---------|
| Escalon 1 hashing historico | Piece accuracy 27% (insuficiente) | Util como señal inicial, no como cierre |
| UOEMD revisionismo (F0-F3A) | NO-GO para escalar H3 | Confirmo limites del regimen y del tokenizado |
| BIAS_CONTROL Gate 2 | Gap 0.478, R@10 a2m 34.4%, hard neg 80.4% | Baseline operativo actual |
| BIAS_CONTROL Gate 3 (DANN) | 4 runs, sin mejora robusta sobre Gate 2 | Invariancia modal no era cuello principal |
| BIAS_CONTROL Gate 4 Run A | 30 épocas + structured pool (ep5 mejor que ep30) | Señal mixta; abre Gate 4.1 causal |
| BIAS_CONTROL Gate 4.1 cierre | `RA5` vs `RB0`, `R1-rescue` completado | Cierre por umbral (`dS=+0.8pp < +1.5pp`) |
| BIAS_CONTROL DEC-005 | Gate 6 + Gate 4.2 dual-domain pre-red | Diagnóstico paralelo aprobado, sin training automático |

---

## Plan Operativo BIAS_CONTROL (Escalon 1-C)

> [!IMPORTANT]
> En esta fase se consolida el Escalon 1-C con dos piezas de diagnóstico:  
> **Gate 6** (evidencia explicativa de embeddings) + **Gate 4.2 pre-red** (viabilidad dual-domain de ratios), bajo `DEC-005` (`diagnostic-only`).

<a id="gate-0---data-integrity"></a>
### Gate 0 - Data Integrity
- **Objetivo**: validar pares Audio<->MIDI, splits y consistencia de metadata.
- **Criterio de decision**: GO si no hay fallas sistemicas de alineacion/datos.
- **Estado**: completado.
- **Artefactos**: `experiments/bias_control/gate0_data_integrity.py`.

<a id="gate-1---intra-modal-baselines"></a>
### Gate 1 - Intra-modal Baselines
- **Objetivo**: establecer piso intra-modal para controlar regresiones.
- **Criterio de decision**: GO si baseline intra-modal es estable y util para monitoreo.
- **Estado**: completado.
- **Artefactos**: `experiments/bias_control/gate1_intra_modal.py`.

<a id="gate-2---foundation-baseline"></a>
### Gate 2 - Foundation Baseline
- **Objetivo**: baseline cross-modal con entrenamiento contrastivo/VICReg.
- **Decision actual**: **checkpoint de referencia** `checkpoint_epoch45`.
- **Resultado clave**: `Gap=0.478`, `R@10 a2m=34.4%`, `R@10 m2a=37.6%`, `hard_neg=80.4%`.
- **Artefactos**: `experiments/bias_control/gate2_foundation.py`.

<a id="gate-25---embedding-diagnostics"></a>
### Gate 2.5 - Embedding Diagnostics
- **Objetivo**: inspeccionar separabilidad modal y salud del embedding.
- **Resultado clave**: separabilidad alta (92.7%), util como diagnostico.
- **Lectura correcta**: no usar este gate como criterio unico para redisenar el pipeline.

<a id="gate-3---dann-cerrado"></a>
### Gate 3 - DANN (cerrado)
- **Objetivo**: forzar invariancia modal y medir impacto causal en retrieval.
- **Resultado**: cerrado por no superar de forma robusta al Gate 2 en structured pool.
- **Conclusión tecnica**: la separabilidad modal no era el factor limitante principal.
- **Artefactos**: `experiments/bias_control/gate3_dann.py`, `Documents/BIAS_CONTROL/Gate3_DANN_Results/INFORME_GATE3_COMPLETO.md`.

<a id="gate-4---ratio-auxiliary-base-completado"></a>
### Gate 4 - Ratio Auxiliary base (completado)
- **Hipotesis**: una vista auxiliar de ratios puede mejorar retrieval sin destruir señal principal.
- **Run ejecutado**: 30 épocas (`ratio_weight=0.1`, régimen `1000/846`, `seed=42`).
- **Structured pool**:
  - `RA5`: A2M R@10=31.4, M2A R@10=40.6, hard_neg=79.0
  - `RA30`: A2M R@10=29.2, M2A R@10=36.4, hard_neg=74.8
- **Lectura**: mejor desempeño temprano (ep5) y degradación con entrenamiento largo.
- **Conclusión**: se requiere control causal explícito antes de iterar descriptores.
- **Artefactos**: `experiments/bias_control/gate4_ratio_auxiliary.py`.

<a id="gate-41---matriz-causal-dec-004004-a-cerrado"></a>
### Gate 4.1 - Matriz causal DEC-004/004-A (cerrado)
- **Fase 0 (RB0)**:
  - `RA5`: A2M R@10=31.4, M2A R@10=40.6, hard_neg=79.0
  - `RB0`: A2M R@10=30.2, M2A R@10=38.2, hard_neg=77.6
  - `dS=+1.2pp`, `dH=+1.4pp` -> zona inconclusa.
- **Rescue único (R1-rescue)**:
  - `R1`: A2M R@10=31.0, M2A R@10=40.2, hard_neg=78.8
  - `dS=+0.8pp` vs `RB0` (no alcanza umbral `+1.5pp`).
- **Decisión**: Gate 4.1 cerrado por regla pre-registrada (sin más enmiendas).
- **Lectura**: los ratios aportan señal marginal, pero no suficiente para promoción en esta rama.

<a id="gate-42---dual-domain-ratios-diagnostico-dec-005"></a>
### Gate 4.2 - Dual-domain ratios (diagnóstico DEC-005)
- **Hipótesis principal (H4.2-6)**: el sesgo MIDI-only de la loss auxiliar contribuye a la asimetría A2M/M2A.
- **Protocolo pre-red**:
  - `P0` oracle (audio sintetizado desde MIDI),
  - `P1` audio real vs MIDI,
  - métricas: `AUC`, `delta_sim`, Wilcoxon + bootstrap CI.
- **Disciplina metodológica**:
  - una sola ronda de tuning de extractor en zona inconclusa,
  - no habilita training automático.
- **Estado**: aprobado para implementación diagnóstica (sin entrenamiento).

<a id="gate-5---curriculum-opcional"></a>
### Gate 5 - Curriculum (opcional)
- **Rol**: posible optimizacion adicional si Gate 4.1/6 no cierran hipótesis.
- **Estado**: hold.
- **Politica**: no bloquea cierre de Escalon 1-C.

<a id="gate-6---retroanalysis-diagnostico-aprobado"></a>
### Gate 6 - Retroanalysis (diagnóstico aprobado)
- **Objetivo**: explicar *que* aprendio el embedding y *donde* impacta ratios.
- **Analisis esperados**: RSA/CKA, probes, disagreement analysis, inspeccion de hard negatives.
- **Estado**: aprobado bajo `DEC-005` (`diagnostic-only`, sin entrenamiento).
- **Rol**: generar evidencia para la próxima DEC de entrenamiento.

---

## Proyeccion TripleScaloneta

```mermaid
flowchart TB
  S1["Escalon 1<br/>MAESTRO Audio<->MIDI<br/>Objetivo: demostracion robusta"] --> S2["Escalon 2<br/>Speech<->EGG<br/>Objetivo: generalidad en sensores distintos"]
  S2 --> S3["Escalon 3<br/>ECG<->PPG<br/>Objetivo: transferencia fuera de audio"]

  style S1 fill:#dbeafe,stroke:#2563eb,color:#111827
  style S2 fill:#fef3c7,stroke:#d97706,color:#111827
  style S3 fill:#fee2e2,stroke:#dc2626,color:#111827
```

| Escalon | Dominio | Estado | Criterio de avance |
|---------|---------|--------|--------------------|
| **1** | MAESTRO Audio<->MIDI | **Activo (1-C diagnóstico)** | Completar DEC-005 (Gate 6 + Gate 4.2 pre-red) y cerrar auditoría final |
| 2 | Speech<->EGG | Planificado | Iniciar solo con cierre robusto del Escalon 1 |
| 3 | ECG<->PPG | Proyeccion | Iniciar luego de evidencia de generalidad en Escalon 2 |

---

## Concepto Central

Los paisajes sonoros contienen relaciones de frecuencia significativas (por ejemplo `3:2`, `5:4` y otras proporciones estables). Phideus intenta capturar esa estructura sin depender de codificaciones musicales ad hoc.

> [!NOTE]
> **Hipotesis central**: los ratios armonicos son unidades fisicas de informacion con poder de transferencia entre modalidades.

---

## Hallazgos Principales

| Hito | Resultado | Significado |
|------|-----------|-------------|
| **BIAS_CONTROL Gate 2** | Gap 0.478 + buen hard negatives | Primera señal fuerte de alineacion cross-modal util |
| **Gate 3 (DANN)** | 4 runs sin mejora estable sobre Gate 2 | La separabilidad modal alta no era el cuello de botella principal |
| **Analizador 5.0** | VAE ~ HRM (`val_loss` similar) | La representacion pesa mas que la arquitectura en ese regimen |
| **Extractor v2.2** | Gap pre-red alto en condiciones controladas | Los histogramas de ratios contienen senal discriminativa |
| **UOEMD revisionismo** | NO-GO para escalar H3 | Dataset pequeno/regimen insuficiente para cierre fuerte |

### Descubrimientos metodologicos relevantes

1. **La metrica canonica de decision debe ser structured pool**, no solo metricas globales de validacion.
2. **Forzar invariancia (DANN) puede destruir senal de retrieval** si la hipotesis causal no esta bien acotada.
3. **Comparabilidad de regimen** (`pool`, `seed`, `batch budget`) es critica para no tomar decisiones falsas.

---

## Estructura del Repositorio

```text
Phideus/
├── src/
│   ├── analizador/
│   │   ├── analizador_5.0.py
│   │   ├── analizador_roseta.py
│   │   └── ...
│   ├── bias_control/
│   │   ├── architectures/
│   │   ├── datasets/
│   │   ├── encoders/
│   │   └── losses/
│   ├── extractors/
│   ├── RNA/
│   └── hrm/
│
├── experiments/
│   └── bias_control/
│       ├── gate0_data_integrity.py
│       ├── gate1_intra_modal.py
│       ├── gate2_foundation.py
│       ├── gate3_dann.py
│       ├── gate4_ratio_auxiliary.py
│       └── evaluate_structured_pool.py
│
├── Documents/
│   ├── INDICE_DOCUMENTACION.md
│   ├── Proyecto_Estado_Actual.md
│   ├── BIAS_CONTROL/
│   ├── ESCALON_1/
│   └── UOEMD/
│
├── config/
├── requirements.txt
└── README.md
```

> [!TIP]
> `data/`, `models/`, `train/`, `test/` y otros artefactos pesados no se versionan.

---

## Quick Start

### 1) Configurar entorno

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2) Pipeline completo (referencia)

```bash
python experiments/bias_control/run_all_gates.py \
  --maestro-dir data/maestro_v3/maestro-v3.0.0 \
  --output data/bias_control_medium
```

<details>
<summary><strong>3) DEC-005 - Comandos diagnósticos (Gate 6 + Gate 4.2 pre-red)</strong></summary>

<br>

**Gate 6 (retroanalysis) — ejemplo base**

```bash
python experiments/bias_control/visualize_embeddings_multigate.py \
  --maestro-dir data/maestro_v3/maestro-v3.0.0 \
  --output data/bias_control_medium/evaluations/gate6 \
  --device cuda
```

**Gate 4.2 pre-red (`P0/P1`) — script planificado**

```bash
python experiments/bias_control/h426_prered_test.py \
  --maestro-dir data/maestro_v3/maestro-v3.0.0 \
  --output data/bias_control_medium/evaluations/gate42 \
  --n-segments 100 --seed 42 --device cuda
```

**Nota de protocolo**

`DEC-005` es `diagnostic-only`: no habilita entrenamiento de Gate 4.2 sin una DEC posterior explícita.

</details>

---

## Documentacion

### Documentos Principales

| Documento | Descripcion |
|-----------|-------------|
| [Documents/INDICE_DOCUMENTACION.md](Documents/INDICE_DOCUMENTACION.md) | Mapa actualizado de documentacion |
| [Documents/Proyecto_Estado_Actual.md](Documents/Proyecto_Estado_Actual.md) | Estado ejecutivo del proyecto |
| [Documents/Rosetta_triplescaloneta.md](Documents/Rosetta_triplescaloneta.md) | Proyeccion por escalones y criterios de avance |
| [Documents/INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md](Documents/INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md) | Historia tecnica de representaciones |

### BIAS_CONTROL

| Documento | Descripcion |
|-----------|-------------|
| [Documents/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md](Documents/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md) | Plan maestro y criterios de decision |
| [Documents/BIAS_CONTROL/INFORME_GATE2_COMPLETO.md](Documents/BIAS_CONTROL/INFORME_GATE2_COMPLETO.md) | Informe tecnico de Gate 2 |
| [Documents/BIAS_CONTROL/Gate3_DANN_Results/INFORME_GATE3_COMPLETO.md](Documents/BIAS_CONTROL/Gate3_DANN_Results/INFORME_GATE3_COMPLETO.md) | Cierre tecnico de Gate 3 |
| [Documents/BIAS_CONTROL/AUDITORIA_BIAS_CONTROL_CODEX.md](Documents/BIAS_CONTROL/AUDITORIA_BIAS_CONTROL_CODEX.md) | Auditoria v1 + addendums Gate 4.1 y DEC-005 |
| [Documents/BIAS_CONTROL/Planes_Claude/plan_gate4.md](Documents/BIAS_CONTROL/Planes_Claude/plan_gate4.md) | Plan Gate 4 base (histórico) |
| [Documents/BIAS_CONTROL/Planes_Claude/plan_gate4_codex.md](Documents/BIAS_CONTROL/Planes_Claude/plan_gate4_codex.md) | Revision tecnica de Gate 4 |

### Otros frentes

| Frente | Documento | Estado |
|--------|-----------|--------|
| Escalon 1 (hashing historico) | [Documents/ESCALON_1/Plan_implementacion.md](Documents/ESCALON_1/Plan_implementacion.md) | Pausado |
| UOEMD revisionismo | [Documents/UOEMD/UOEMD_Revisionismo/ROADMAP.md](Documents/UOEMD/UOEMD_Revisionismo/ROADMAP.md) | No-go |

---

## Hipotesis de Investigacion

### H1: Estructura de Ratios (validada)
Las senales naturales muestran distribuciones de ratios no triviales y estables.

### H2: Aprendibilidad (validada)
Modelos neuronales pueden aprender representaciones compactas de esa estructura.

### H3: Cross-modality (en validacion)
La evidencia actual muestra alineacion util en retrieval audio<->MIDI. El cierre fuerte del escalón depende de la fase diagnóstica `DEC-005` (Gate 6 + Gate 4.2 pre-red).

---

## Arquitectura BIAS_CONTROL (resumen)

### Baseline (Gate 2)

```text
Audio (waveform) -> MERT encoder -> Projection -> Embedding (256d)
MIDI  (tokens)   -> MIDI encoder -> Projection -> Embedding (256d)
                             \    VICReg + retrieval objectives
```

### Extension actual (post Gate 4.1)

```text
Audio embedding <-> MIDI embedding  (loss principal)
       \             /
        \           /
         Ratio auxiliary branch (evaluado en Gate 4/4.1, línea cerrada)

Fase actual: diagnóstico `DEC-005` para decidir siguiente entrenamiento:
- Gate 6: análisis de embeddings y drift
- Gate 4.2 pre-red: viabilidad dual-domain ratios (P0/P1)
```

---

## Tooling de Documentación

| Componente | Ubicación | Rol |
|------------|-----------|-----|
| Skill dinámica | `tools/skills/phideus-doc-maintainer/` | Detección de frente activo + selección de docs por política |
| Runtime skill | `/root/.codex/skills/phideus-doc-maintainer/` | Uso operativo por Codex |

La skill `phideus-doc-maintainer` quedó implementada y activa para mantener consistencia documental por frente (`bias_control`, `escalon_1`, `uoemd`, etc.) con política de actualización global mínima.

---

## Licencia

MIT License. Ver [LICENSE.md](LICENSE.md).

---

> "El bosque ya canta. Nuestra tarea es entender su afinacion."
