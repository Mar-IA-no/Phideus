# INFORME DEC-005: Diagnóstico de Asimetría A2M/M2A

## Gate 6 Phase 1 + H4.2-6 Pre-Red Test — Informe Completo

**Fecha de ejecución:** 2026-02-11
**Decisión:** DEC-005 (DIAGNOSTIC-ONLY — ningún training habilitado)
**Autores:** Claude (implementación), Codex (auditoría pre-ejecución del plan)
**Estado:** COMPLETADO — todos los scripts ejecutados, todos los artefactos generados

> [!NOTE]
> Addendum de vigencia (2026-02-14): DEC-005 permanece como diagnóstico causal cerrado.
> El frente ya está en etapa de ejecución Gate 4.3 (corrida 6 brazos), con `D0`/`D4` cerrados y `A4` en curso.
> Seguimiento activo en roadmap y estado troncal.

---

## Tabla de Contenidos

1. [Contexto y Motivación](#1-contexto-y-motivación)
2. [Diseño Experimental y Plan](#2-diseño-experimental-y-plan)
3. [Arquitectura del Modelo](#3-arquitectura-del-modelo-bajo-análisis)
4. [Paso 0: Trazabilidad — Artefacto RA5](#4-paso-0-trazabilidad--artefacto-ra5)
5. [Track 1, Script 1: Layer Drift Analysis](#5-track-1-script-1-layer-drift-analysis)
6. [Track 1, Script 2: Extracción de Embeddings](#6-track-1-script-2-extracción-de-embeddings-multigate)
7. [Track 1, Script 3: Análisis de Hubness](#7-track-1-script-3-análisis-de-hubness)
8. [Track 1, Script 4: Visualizaciones Comparativas](#8-track-1-script-4-visualizaciones-comparativas)
9. [Track 2, Script 5: H4.2-6 Pre-Red Test](#9-track-2-script-5-h426-pre-red-test)
10. [Aplicación de la Matriz de Decisión](#10-aplicación-de-la-matriz-de-decisión-2x2)
11. [Diagnóstico Integrado: Causa Raíz](#11-diagnóstico-integrado-causa-raíz)
12. [Próximos Pasos](#12-próximos-pasos)
13. [Artefactos Generados](#13-artefactos-generados)
14. [Apéndice: Datos Crudos](#14-apéndice-datos-crudos)

---

## 1. Contexto y Motivación

### 1.1 La Anomalía que Detonó DEC-005

Después de completar Gate 4.1, se observó una anomalía persistente y consistente en todos los checkpoints fine-tuned: **cualquier forma de fine-tuning posterior a Gate 2 degrada el recall Audio-to-MIDI (A2M)**. Esta degradación ocurre incluso en el checkpoint de control RB0, que no utiliza la señal de ratios armónicos. Los datos concretos son:

| Checkpoint | A2M R@10 | M2A R@10 | Gap (M2A-A2M) | Hard Neg |
|------------|----------|----------|---------------|----------|
| **Gate 2 (baseline)** | **34.4%** | **37.6%** | 3.2pp | 80.4% / 87.0% |
| RB0 (control, sin ratios) | 30.2% | 38.2% | 8.0pp | 77.6% / 80.8% |
| RA5 (ratio baseline) | 31.4% | 40.6% | 9.2pp | 79.0% / 83.6% |
| R1-rescue (ratio enriched) | 31.0% | 40.2% | 9.2pp | 78.8% / 80.8% |

**Observaciones clave:**
- Gate 2 tiene la mejor A2M (34.4%). Todos los fine-tuned pierden entre 3.0 y 4.2 puntos porcentuales.
- M2A mejora ligeramente en fine-tuned (+0.6 a +3.0pp), pero a costa de A2M.
- El gap A2M/M2A se amplifica de 3.2pp en Gate 2 a 8-9pp en todos los fine-tuned.
- **RB0 también degrada A2M (-4.2pp)**, a pesar de ser un control sin señal de ratios. Esto demuestra que el problema NO es la señal de ratios, sino el **régimen de fine-tuning** en sí.

### 1.2 El Cierre de Gate 4.1

Gate 4.1 operó bajo DEC-004 y su enmienda DEC-004-A. El protocolo incluía:

- **Fase 0 (Control causal):** RB0 — fine-tuning con InfoNCE puro (sin ratios) para aislar el efecto del fine-tuning del efecto de los ratios.
- **Umbral GO:** dS >= +1.5pp (donde dS = score combinado del checkpoint vs Gate 2).
- **Zonas de decisión:** Zone 1 GO directo (dS >= +1.5pp), Zone 2 NO-GO directo (dS <= -3.0pp), Zone 3 Inconcluso (permite 1 rescue run).

El resultado: RA5 obtuvo dS=+1.2pp (Zone 3 Inconcluso). Se ejecutó R1-rescue con learning rate reducido (5e-5), que obtuvo dS=+0.8pp, aún por debajo del umbral de +1.5pp. **Gate 4.1 se declaró CERRADO.**

El hallazgo más importante de Gate 4.1 no fue el resultado numérico marginal, sino la observación de que RB0 (control sin ratios) también degrada A2M. Esto redirige la investigación: el problema está en **cómo** se hace fine-tuning, no en **qué** señal auxiliar se usa.

### 1.3 Las Preguntas que DEC-005 Debe Responder

DEC-005 fue diseñado como un diagnóstico de dos vías para responder preguntas específicas antes de lanzar más training:

**Track 1 — Gate 6 Phase 1 (¿Por qué se degrada A2M?):**
- H6.1: ¿Es forgetting asimétrico? ¿La audio projection cambia más que la MIDI projection?
- H6.2: ¿Cuáles capas cambian más durante el fine-tuning?
- H6.3: ¿Crece la hubness (concentración de nearest neighbors) con el fine-tuning?
- H6.4: ¿Es la separación de similitudes (correct vs incorrect) inherentemente más débil para A2M que para M2A?

**Track 2 — H4.2-6 Pre-Red (¿Es viable extraer ratios desde audio?):**
- ¿Se pueden extraer histogramas de ratios armónicos desde audio real vía CQT?
- ¿Son comparables con los histogramas de ratios extraídos desde MIDI?
- Si funciona: abre la puerta a una nueva señal auxiliar dual-domain.
- Si no funciona: elimina H4.2-6 del mapa de hipótesis.

### 1.4 Scope: DIAGNOSTIC-ONLY

DEC-005 no habilita ningún training. Solo permite:
- Ejecutar análisis sobre checkpoints existentes (Gate 2, RB0, RA5, R1)
- Extraer embeddings y compararlos
- Probar viabilidad de extracción de ratios desde audio
- Producir un diagnóstico que informe la decisión DEC-006 (primera ola de Gate 4.2)

---

## 2. Diseño Experimental y Plan

### 2.1 Estructura de Dos Tracks

El diagnóstico se organizó en dos tracks paralelos:

```
Track 1: Gate 6 Phase 1 (Retroanálisis de checkpoints existentes)
├── Script 1: compare_layer_drift.py          — Drift de parámetros (CPU)
├── Script 2: extract_multigate_embeddings.py  — Extracción de embeddings (GPU)
├── Script 3: analyze_hubness.py              — Análisis de hubness (CPU)
└── Script 4: visualize_embeddings_multigate.py — Visualizaciones (CPU)

Track 2: H4.2-6 Pre-Red Test (Viabilidad de ratios dual-domain)
└── Script 5: h426_prered_test.py              — P0 oracle + P1 real (CPU)
```

### 2.2 Dependencias entre Scripts

```
Script 1 ─────────────────────── (independiente, CPU)
Script 2 ─── GPU ──────────┬──→ Script 3 (requiere embeddings)
                           └──→ Script 4 (requiere embeddings)
Script 5 ─────────────────────── (independiente, CPU)
```

Los Scripts 1 y 5 pueden ejecutarse en paralelo. Los Scripts 3 y 4 dependen de que Script 2 termine primero.

### 2.3 Paso 0: Artefacto Faltante

Antes de comenzar la implementación, se identificó que el checkpoint RA5 (Run A, epoch 5) no tenía un artefacto JSON canónico en `evaluations/gate4/`. Solo existían `RB0_ep5.json`, `R1rescue_ep5.json` y `R1rescue_best.json`. Los números de RA5 usados en decisiones (A2M=31.4%, M2A=40.6%) provenían de logs de training, no de una evaluación formal reproducible. El Paso 0 genera este artefacto.

### 2.4 Correcciones de la Auditoría Codex

Antes de aprobar el plan, Codex realizó una auditoría pre-ejecución que identificó 4 issues:

| # | Severidad | Issue | Fix Aplicado |
|---|-----------|-------|-------------|
| 1 | Alto | Plan no cubre RSA/CKA/probes del roadmap Gate 6 completo | Renombrar como "Phase 1"; Phase 2 (RSA/CKA) queda pendiente post-DEC-006 |
| 2 | Alto | `extract_all_embeddings()` no devuelve metadata | Construir metadata explícitamente desde `dataset.segments[i].piece_idx`, `.canonical_composer`, `.start_time` |
| 3 | Medio | `strict=False` puede ocultar carga incorrecta de checkpoints | Loggear `missing_keys`/`unexpected_keys`, assert `len(unexpected) == 0` |
| 4 | Medio | RA5 no tiene JSON canónico | Paso 0: generar `RA5_ep5.json` con `evaluate_structured_pool.py` antes de implementar |

Todas las correcciones fueron integradas en los scripts.

### 2.5 Hipótesis Pre-registradas y Umbrales

**Track 1 (Gate 6 Phase 1):**
- H6.1: Audio Projection `relΔ` >> MIDI Projection `relΔ` (ratio > 2x indicaría forgetting asimétrico)
- H6.2: Capas profundas del transformer (layers 3-4) cambian más que capas tempranas (0-1)
- H6.3: Skewness de k-occurrence crece con fine-tuning (hubness empeora)
- H6.4: Separación A2M < separación M2A (asimetría intrínseca en distribuciones de similitud)

**Track 2 (H4.2-6):**

| Fase | GO | INCONCLUSO | NO-GO |
|------|-----|------------|-------|
| P0 (oracle) | AUC >= 0.80, delta_sim >= 0.10 | AUC 0.65-0.80 ó CI toca random | AUC < 0.65 |
| P1 (real) | AUC >= 0.70, delta_sim >= 0.05, Wilcoxon p < 0.01 | AUC 0.55-0.70 | AUC < 0.55 |

### 2.6 Matriz de Decisión 2×2

La combinación de resultados de Track 1 y Track 2 determina el siguiente paso:

| Gate 6: ¿drift asimétrico? | H4.2-6 P1: ¿GO? | Siguiente paso |
|---|---|---|
| Sí | Sí | H4.2-6 training + H4.2-2 adapter + S-control |
| Sí | No | H4.2-2 adapter + H4.2-1 audio-only + S-control |
| No | Sí | H4.2-6 training + S-control |
| No | No | Solo S-control → re-evaluar viabilidad de branch 4.x |

Cualquier decisión de training requiere una DEC nueva.

---

## 3. Arquitectura del Modelo Bajo Análisis

Para interpretar los resultados del diagnóstico es necesario entender la estructura del modelo `CrossModalModel` y qué módulos contiene cada "grupo" de parámetros.

### 3.1 Visión General

```
Audio (waveform) ──→ Audio Encoder ──→ [B, 1024] ──→ Audio Projection ──→ [B, 256]
                                                                              ↕
                                                                         VICReg Loss
                                                                              ↕
MIDI (events)    ──→ MIDI Encoder  ──→ [B, 512]  ──→ MIDI Projection  ──→ [B, 256]
```

### 3.2 Audio Encoder ("lite")

Los checkpoints bajo análisis usan la variante `audio_encoder='lite'` (no el MERT de 330M parámetros):

**CNN Feature Extractor** (`audio_encoder.feature_extractor`):
- 4 capas Conv1d: 1→512→512→512→1024
- GroupNorm + GELU en cada capa
- Downsampling temporal ~40x
- ~3.2M parámetros

**Positional Embedding** (`audio_encoder.pos_embedding`):
- Embedding aprendible, max 6000 tokens
- ~6.1M parámetros

**Transformer Encoder** (`audio_encoder.transformer`):
- 4 capas, 8 heads, d_model=1024, FFN=4096
- ~50.4M parámetros
- Mean pooling temporal → [B, 1024]

**Total Audio Encoder:** ~59.7M parámetros

### 3.3 Audio Projection (`audio_projection`)

MLP de 3 capas:
- Linear(1024, 512) → BN → ReLU
- Linear(512, 512) → BN → ReLU
- Linear(512, 256)
- ~0.9M parámetros

### 3.4 MIDI Encoder (`midi_encoder`)

**Event Embedding** (`midi_encoder.event_embedding`):
- Embeddings separados para pitch (128→256d), velocity (128→128d), duration (32→128d)
- Concatenación (512d) → Linear(512, 512) + LayerNorm
- ~0.3M parámetros

**Positional Encoding** (`midi_encoder.pos_encoding`):
- Sinusoidal (no aprendible), max_len=10000
- ~1.0M parámetros (buffer, no trainable)

**Transformer Encoder** (`midi_encoder.transformer`):
- 4 capas, 8 heads, d_model=512, FFN=2048, pre-norm, GELU
- ~12.6M parámetros

**Output Norm** (`midi_encoder.output_norm`):
- LayerNorm(512)
- ~1K parámetros (512 weight + 512 bias)

**Total MIDI Encoder:** ~13.9M parámetros

### 3.5 MIDI Projection (`midi_projection`)

MLP de 3 capas:
- Linear(512, 512) → BN → ReLU
- Linear(512, 512) → BN → ReLU
- Linear(512, 256)
- ~0.7M parámetros

### 3.6 Distribución de Parámetros (155 tensores compartidos)

| Grupo | # Tensores | # Parámetros | % del Total |
|-------|-----------|-------------|-------------|
| Audio CNN | 16 | 3,158,528 | 4.2% |
| Audio PosEmb | 1 | 6,144,000 | 8.1% |
| Audio Transformer | 48 | 50,384,896 | **66.7%** |
| Audio Projection | 16 | 922,882 | 1.2% |
| MIDI Embedding | 7 | 316,928 | 0.4% |
| MIDI OutputNorm | 2 | 1,024 | 0.001% |
| MIDI PosEncoding | 1 | 1,048,576 | 1.4% |
| MIDI Transformer | 48 | 12,609,536 | 16.7% |
| MIDI Projection | 16 | 660,738 | 0.9% |
| **Total** | **155** | **75,246,108** | **100%** |

**Punto crítico:** El Audio Encoder (CNN + PosEmb + Transformer) constituye el **79.0%** de los parámetros del modelo. El MIDI Encoder completo es solo el 18.5%. Las proyecciones son solo el 2.1%.

### 3.7 ¿Qué Significa "Fine-tuning" en este Contexto?

Los checkpoints `_base.pt` de Gate 4.1 se entrenaron partiendo del checkpoint Gate 2 (epoch 45) con los siguientes parámetros de training:

- Optimizer: AdamW con parameter groups separados
  - `lr_projection=1e-3` para ambas proyecciones
  - `lr_midi_encoder=1e-4` para el MIDI encoder
  - El audio encoder **no tiene su propio grupo de learning rate en la configuración de fine-tuning** → queda efectivamente **congelado**
- Loss: VICReg (invariance=10.0, variance=10.0, covariance=1.0) + ratio auxiliary (según configuración)
- Epochs: 5 (con checkpoints cada epoch)
- Batch size: 64

---

## 4. Paso 0: Trazabilidad — Artefacto RA5

### 4.1 Problema

El checkpoint RA5 (Run A, epoch 5) se usó extensivamente en decisiones (DEC-004-A, cierre de Gate 4.1) pero no tenía un artefacto JSON canónico de evaluación. Los números citados (A2M=31.4%, M2A=40.6%) provenían de evaluaciones informales durante el desarrollo.

### 4.2 Solución

Se ejecutó la evaluación formal con parámetros canónicos:

```bash
python experiments/bias_control/evaluate_structured_pool.py \
    --model data/bias_control_medium/training_outputs/gate4_runA/checkpoint_epoch5_base.pt \
    --output data/bias_control_medium/evaluations/gate4/RA5_ep5.json \
    --n-queries 500 --pool-size 256 --n-hard 64 --seed 42
```

### 4.3 Resultado

El artefacto generado confirma los números usados en decisiones previas:

| Métrica | A2M | M2A |
|---------|-----|-----|
| R@1 | 6.2% | 5.4% |
| R@5 | 19.6% | 26.6% |
| **R@10** | **31.4%** | **40.6%** |
| R@20 | 45.8% | 55.6% |
| MRR | 0.1436 | 0.1625 |
| Mean Rank | 44.17 | 35.67 |
| Median Rank | 22.0 | 15.0 |
| Hard Neg (same-piece) | 79.0% | — |
| Hard Neg (random) | 83.6% | — |

**Validación:** Los números coinciden exactamente con los citados en DEC-004-A. Trazabilidad confirmada.

---

## 5. Track 1, Script 1: Layer Drift Analysis

### 5.1 Objetivo

Cuantificar cuánto cambia cada módulo del modelo entre el baseline Gate 2 y cada checkpoint fine-tuned. Responde directamente a H6.1 (forgetting asimétrico en proyecciones) y H6.2 (sensibilidad de capas profundas).

### 5.2 Metodología

**Script:** `experiments/bias_control/compare_layer_drift.py` (~210 líneas)
**Requerimientos:** Solo CPU, sin GPU. Carga state_dicts y los compara.

**Algoritmo:**

1. Cargar el state_dict del checkpoint Gate 2 (baseline): `checkpoint_epoch45.pt`
2. Para cada checkpoint fine-tuned (RB0, RA5, R1):
   a. Cargar su state_dict
   b. Verificar keys compartidas, keys solo en baseline, keys solo en fine-tuned
   c. Para cada parámetro compartido (mismo nombre, misma forma):
      - **L2 distance:** `||w_ft - w_gate2||₂` — magnitud absoluta del cambio
      - **Base norm:** `||w_gate2||₂` — magnitud original del parámetro
      - **Relative change:** `L2 / (base_norm + 1e-12)` — cambio normalizado por la magnitud original. Un valor de 0.10 significa que el parámetro cambió un 10% de su magnitud original.
      - **Cosine similarity:** `cos(flatten(w_ft), flatten(w_gate2))` — mide si los pesos rotaron en el espacio de parámetros. 1.0 = sin rotación, <1.0 = los pesos cambiaron de dirección.
   d. Clasificar cada parámetro en un grupo de módulos por prefijo del nombre
   e. Agregar métricas por grupo: media, max, peor parámetro

**Grupos de módulos (9 grupos, 155 parámetros totales):**

| Prefijo | Nombre del Grupo | # Tensores |
|---------|-----------------|-----------|
| `audio_encoder.feature_extractor` | Audio CNN | 16 |
| `audio_encoder.pos_embedding` | Audio PosEmb | 1 |
| `audio_encoder.transformer` | Audio Transformer | 48 |
| `audio_projection` | Audio Projection | 16 |
| `midi_encoder.event_embedding` | MIDI Embedding | 7 |
| `midi_encoder.output_norm` | MIDI OutputNorm | 2 |
| `midi_encoder.pos_encoding` | MIDI PosEncoding | 1 |
| `midi_encoder.transformer` | MIDI Transformer | 48 |
| `midi_projection` | MIDI Projection | 16 |

### 5.3 Resultados Completos

#### RB0 vs Gate 2:

| Módulo | # Params | L2 mean | RelΔ mean | RelΔ max | Cos min | Peor parámetro |
|--------|---------|---------|-----------|----------|---------|----------------|
| Audio CNN | 16 | **0.0** | **0.0%** | 0.0% | 0.000 | feature_extractor.0.bias |
| Audio PosEmb | 1 | **0.0** | **0.0%** | 0.0% | 1.000 | pos_embedding |
| Audio Transformer | 48 | **0.0** | **0.0%** | 0.0% | 0.000 | transformer.layers.0.linear1.bias |
| Audio Projection | 16 | 630.3 | 11.3% | 42.8% | 0.000 | mlp.3.bias |
| MIDI Embedding | 7 | 0.26 | 3.5% | 12.0% | 0.995 | event_embedding.norm.bias |
| MIDI OutputNorm | 2 | 0.02 | **160.0%** | **319.8%** | 0.521 | output_norm.bias |
| MIDI PosEncoding | 1 | **0.0** | **0.0%** | 0.0% | 1.000 | pos_encoding.pe |
| MIDI Transformer | 48 | 0.57 | 8.3% | 22.8% | 0.974 | transformer.layers.3.self_attn.in_proj_bias |
| MIDI Projection | 16 | 642.3 | 11.8% | 41.5% | 0.000 | mlp.3.bias |

#### RA5 vs Gate 2:

| Módulo | # Params | L2 mean | RelΔ mean | RelΔ max | Cos min | Peor parámetro |
|--------|---------|---------|-----------|----------|---------|----------------|
| Audio CNN | 16 | **0.0** | **0.0%** | 0.0% | 0.000 | feature_extractor.0.bias |
| Audio PosEmb | 1 | **0.0** | **0.0%** | 0.0% | 1.000 | pos_embedding |
| Audio Transformer | 48 | **0.0** | **0.0%** | 0.0% | 0.000 | transformer.layers.0.linear1.bias |
| Audio Projection | 16 | 632.1 | 13.9% | 52.2% | 0.000 | mlp.3.bias |
| MIDI Embedding | 7 | 0.38 | 5.0% | 17.0% | 0.991 | event_embedding.norm.bias |
| MIDI OutputNorm | 2 | 0.03 | **157.5%** | **314.8%** | 0.467 | output_norm.bias |
| MIDI PosEncoding | 1 | **0.0** | **0.0%** | 0.0% | 1.000 | pos_encoding.pe |
| MIDI Transformer | 48 | 0.83 | 12.1% | 33.2% | 0.945 | transformer.layers.3.self_attn.in_proj_bias |
| MIDI Projection | 16 | 644.6 | 13.1% | 45.2% | 0.000 | mlp.3.bias |

#### R1 vs Gate 2:

| Módulo | # Params | L2 mean | RelΔ mean | RelΔ max | Cos min | Peor parámetro |
|--------|---------|---------|-----------|----------|---------|----------------|
| Audio CNN | 16 | **0.0** | **0.0%** | 0.0% | 0.000 | feature_extractor.0.bias |
| Audio PosEmb | 1 | **0.0** | **0.0%** | 0.0% | 1.000 | pos_embedding |
| Audio Transformer | 48 | **0.0** | **0.0%** | 0.0% | 0.000 | transformer.layers.0.linear1.bias |
| Audio Projection | 16 | 631.2 | 11.3% | 35.7% | 0.000 | mlp.3.bias |
| MIDI Embedding | 7 | 0.27 | 3.5% | 11.6% | 0.995 | event_embedding.norm.bias |
| MIDI OutputNorm | 2 | 0.02 | **111.8%** | **223.5%** | 0.533 | output_norm.bias |
| MIDI PosEncoding | 1 | **0.0** | **0.0%** | 0.0% | 1.000 | pos_encoding.pe |
| MIDI Transformer | 48 | 0.58 | 8.5% | 24.0% | 0.971 | transformer.layers.3.self_attn.in_proj_bias |
| MIDI Projection | 16 | 646.3 | 10.8% | 30.6% | 0.000 | mlp.3.bias |

### 5.4 Hallazgo Crítico: Audio Encoder Completamente Congelado

El resultado más importante de Script 1 es inequívoco: **el audio encoder (CNN + positional embedding + transformer) tiene EXACTAMENTE CERO drift en los tres checkpoints fine-tuned.** No hay un solo parámetro del audio encoder que haya cambiado ni una fracción.

Esto significa que durante el fine-tuning de Gate 4.1:
- Los **65 tensores** del audio encoder (59.7M parámetros, 79% del modelo) permanecieron idénticos a Gate 2
- Solo cambiaron las **proyecciones** (ambos lados, ~11-14% de cambio relativo)
- Solo cambió el **MIDI encoder** (3.5-12.1% de cambio relativo)
- El **MIDI PosEncoding** (sinusoidal, no aprendible) tampoco cambió

**Nota sobre cosine_sim_min = 0.0 en Audio CNN/Transformer:** Esto ocurre porque algunos parámetros bias son vectores cero (`||w|| = 0`), haciendo que `cos(0, 0)` sea indefinido y se reporte como 0. NO indica cambio — la L2 distance de exactamente 0.0 confirma que no hubo ningún cambio.

**Nota sobre MIDI OutputNorm:** El cambio relativo extremo (160-320%) es aritméticamente correcto pero engañoso. El output_norm.bias tiene solo 512 parámetros con norma muy pequeña (~0.012), por lo que un cambio absoluto minúsculo (L2=0.02-0.05) produce un ratio relativo enorme. En términos de impacto funcional, estos 1,024 parámetros son insignificantes.

### 5.5 Evaluación de Hipótesis

**H6.1 (Forgetting Asimétrico en Proyecciones):**

| Checkpoint | Audio Proj relΔ | MIDI Proj relΔ | Ratio |
|------------|----------------|----------------|-------|
| RB0 | 11.3% | 11.8% | 0.96x |
| RA5 | 13.9% | 13.1% | 1.06x |
| R1 | 11.3% | 10.8% | 1.05x |

**Resultado: H6.1 NO CONFIRMADA.** El drift en las proyecciones es simétrico (ratio ~1.0x). El "forgetting asimétrico" no ocurre a nivel de proyección.

Sin embargo, la asimetría existe a un nivel más profundo: **el MIDI Transformer cambia 8-12% mientras que el Audio Transformer cambia 0%.** La asimetría no está en las proyecciones, sino en el backbone.

**H6.2 (Sensibilidad de Capas Profundas):**

El peor parámetro del MIDI Transformer es consistentemente `transformer.layers.3.self_attn.in_proj_bias` (la capa más profunda). Los valores de relΔ max por checkpoint confirman que capas más profundas cambian más:

| Checkpoint | MIDI Trans relΔ max | Peor capa |
|------------|-------------------|-----------|
| RB0 | 22.8% | Layer 3 |
| RA5 | 33.2% | Layer 3 |
| R1 | 24.0% | Layer 3 |

**Resultado: H6.2 PARCIALMENTE CONFIRMADA para MIDI (capas profundas driftan más). No aplica para Audio (drift = 0 en todas las capas).**

---

## 6. Track 1, Script 2: Extracción de Embeddings Multigate

### 6.1 Objetivo

Extraer y cachear embeddings de audio y MIDI de los 4 checkpoints (Gate 2, RB0, RA5, R1) sobre todo el validation set de MAESTRO. Este script no produce análisis; es infraestructura para Scripts 3 y 4.

### 6.2 Metodología

**Script:** `experiments/bias_control/extract_multigate_embeddings.py` (~145 líneas)
**Requerimientos:** GPU (RTX 3090), ~22 minutos de ejecución

**Algoritmo:**

1. Cargar el dataset de validación ONCE:
   ```python
   MaestroSegmentDataset(
       maestro_dir=MAESTRO_DIR,
       segment_len=4.0,
       hop=1.0,
       split='validation'
   )
   ```
   Resultado: 13,532 segmentos de 4 segundos con hop de 1 segundo

2. Construir arrays de metadata desde `dataset.segments`:
   - `piece_indices`: [13532] — índice de la pieza MAESTRO para cada segmento
   - `composers`: [13532] — compositor canónico (string)
   - `start_times`: [13532] — tiempo de inicio del segmento en la pieza

3. Para cada checkpoint (secuencialmente):
   a. Cargar `CrossModalModel(audio_encoder='lite')` con `strict=False`
   b. Validar: `assert len(unexpected_keys) == 0`
   c. Extraer embeddings: `extract_all_embeddings(model, dataset, device, batch_size=64, num_workers=8)`
   d. Convertir a NumPy
   e. Liberar modelo y vaciar CUDA cache

4. Guardar todo en un único `.npz`

### 6.3 Resultado

**Archivo generado:** `data/bias_control_medium/evaluations/gate6/multigate_embeddings.npz`
**Tamaño:** 108 MB
**Tiempo de ejecución:** 22.8 minutos (~5.7 min por checkpoint)

**Contenido:**

| Key | Shape | Tipo |
|-----|-------|------|
| `gate2_audio` | [13532, 256] | float32 |
| `gate2_midi` | [13532, 256] | float32 |
| `RB0_audio` | [13532, 256] | float32 |
| `RB0_midi` | [13532, 256] | float32 |
| `RA5_audio` | [13532, 256] | float32 |
| `RA5_midi` | [13532, 256] | float32 |
| `R1_audio` | [13532, 256] | float32 |
| `R1_midi` | [13532, 256] | float32 |
| `piece_indices` | [13532] | int64 |
| `composers` | [13532] | string |
| `start_times` | [13532] | float64 |

**Validación:** 137 piezas únicas en el validation set. 13,532 segmentos × 256 dimensiones × 8 arrays = ~27.7M floats.

### 6.4 Observación Importante sobre Audio Embeddings

Dado que el audio encoder está completamente congelado (Script 1), los embeddings de audio pre-projection son **idénticos** en los 4 checkpoints. Las diferencias en los embeddings finales (256d) provienen exclusivamente de la Audio Projection, que sí cambia (~11-14% drift).

Verificación: si tomamos `gate2_audio` y `RB0_audio`, deberían ser diferentes (la projection cambió) pero correlacionados (el backbone es idéntico). Los embeddings de MIDI serán más diferentes porque tanto el MIDI encoder como la MIDI projection cambian.

---

## 7. Track 1, Script 3: Análisis de Hubness

### 7.1 Objetivo

Analizar si el espacio de embeddings sufre de "hubness" — un fenómeno donde ciertos embeddings MIDI (o audio) aparecen como nearest neighbor de un número desproporcionado de queries, sesgando el retrieval. Responde a H6.3 (hubness crece con fine-tuning) y H6.4 (separación A2M < M2A).

### 7.2 Concepto: ¿Qué es Hubness?

En espacios de alta dimensión, la distribución de distancias tiende a concentrarse. Esto crea "hubs" — puntos que son nearest neighbors de muchos otros puntos — y "anti-hubs" — puntos que casi nunca son elegidos como vecinos. La **k-occurrence** de un punto es el número de veces que aparece en las listas de top-K neighbors de todos los queries. En una distribución uniforme, cada punto debería aparecer aproximadamente K veces en promedio. Si la distribución de k-occurrence tiene alta skewness (cola derecha larga), hay hubs que monopolizan las listas de vecinos.

### 7.3 Metodología

**Script:** `experiments/bias_control/analyze_hubness.py` (~230 líneas)
**Requerimientos:** CPU solo, requiere `multigate_embeddings.npz`

**Algoritmo por checkpoint:**

1. **Matrices de similitud coseno:**
   - Normalizar embeddings a norma unitaria
   - `sim_a2m = audio_norm @ midi_norm.T` → [13532, 13532] (queries audio, keys MIDI)
   - `sim_m2a = midi_norm @ audio_norm.T` → [13532, 13532] (queries MIDI, keys audio)

2. **K-occurrence (K=10):**
   - Para cada query (fila de la matriz), encontrar los índices de los top-10 values
   - Contar cuántas veces aparece cada key (columna) en las listas top-10 de todos los queries
   - Calcular: media (debe ser ~K × N_queries / N_keys = 10), std, max, skewness (`scipy.stats.skew`)
   - Identificar "hubs": keys con k-occurrence > media + 3σ

3. **Per-piece Recall@10:**
   - Agrupar segmentos por `piece_idx` (137 piezas)
   - Para cada segmento, verificar si su match correcto (mismo índice, en la diagonal de la matriz de similitud) está entre sus top-10 vecinos
   - Calcular recall por pieza, luego promediar

4. **Distribuciones de similitud:**
   - "Correct": valores diagonales `sim[i, i]` — la similitud del par alineado
   - "Incorrect": para un subconjunto de queries, muestrear 50 valores no-diagonales aleatorios
   - Calcular separación: `mean(correct) - mean(incorrect)`

5. **Comparación vs Gate 2 (para checkpoints fine-tuned):**
   - Por cada pieza: Δ(A2M_recall) = recall_ft - recall_gate2
   - Contar piezas que "perdieron" A2M recall
   - Contar piezas que "ganaron" M2A recall
   - Correlación de Pearson entre Δ(A2M) y Δ(M2A) por pieza

### 7.4 Resultados Completos: Hubness (K-occurrence)

#### Dirección A2M (Audio queries → MIDI keys):

| Checkpoint | Media | Std | Max | Skewness | Hubs (>3σ) |
|------------|-------|-----|-----|----------|-----------|
| Gate 2 | 10.0 | 29.4 | 1,048 | **15.53** | 103 |
| RB0 | 10.0 | 35.2 | 1,083 | 13.36 | 158 |
| RA5 | 10.0 | 34.3 | 1,069 | 11.49 | 164 |
| R1 | 10.0 | 36.6 | 1,258 | **16.41** | 146 |

**Interpretación:** La media es exactamente 10.0 (esperado: cada key aparece en promedio 10 veces en top-10 de 13532 queries/13532 keys). La distribución es extremadamente sesgada (skewness 11-16), indicando que algunos embeddings MIDI monopolizan las listas de vecinos. El embedding MIDI más "hub" en R1 aparece 1,258 veces — es decir, el 9.3% de TODOS los queries audio lo tienen en su top-10.

#### Dirección M2A (MIDI queries → Audio keys):

| Checkpoint | Media | Std | Max | Skewness | Hubs (>3σ) |
|------------|-------|-----|-----|----------|-----------|
| Gate 2 | 10.0 | 23.8 | 576 | 9.30 | 193 |
| RB0 | 10.0 | 38.2 | 1,018 | 11.80 | 162 |
| RA5 | 10.0 | 33.7 | 846 | 11.25 | 183 |
| R1 | 10.0 | 40.8 | 1,164 | 13.47 | 146 |

**Interpretación:** Gate 2 tiene la M2A hubness más baja (skew=9.30). Fine-tuning la aumenta significativamente (a 11-13). Esto es consistente con el hecho de que el audio encoder está congelado: los embeddings de audio permanecen en la distribución original (Gate 2), pero las proyecciones se mueven, creando nuevos patrones de hub.

### 7.5 Resultados: Per-Piece Recall@10

| Checkpoint | A2M mean | M2A mean |
|------------|----------|----------|
| Gate 2 | 2.61% | 2.68% |
| RB0 | 2.71% | 2.83% |
| RA5 | 2.89% | 3.19% |
| R1 | 2.79% | 2.88% |

**Nota:** Estos valores son per-piece recalls sobre el dataset completo de 13,532 segmentos (no el pool de 256 del structured pool eval). Por eso son mucho más bajos que los 30-40% reportados en la evaluación formal.

### 7.6 Resultados: Distribuciones de Similitud

| Checkpoint | Correct mean (A2M) | Incorrect mean (A2M) | **Separación A2M** | **Separación M2A** |
|------------|-------------------|---------------------|--------------------|--------------------|
| Gate 2 | 0.498 | 0.020 | **0.479** | **0.479** |
| RB0 | 0.396 | 0.001 | 0.396 | 0.396 |
| RA5 | 0.419 | 0.000 | 0.419 | 0.419 |
| R1 | 0.390 | -0.005 | **0.395** | **0.395** |

**Hallazgo clave:** Gate 2 tiene la MEJOR separación (0.479). Todos los fine-tuned pierden entre 0.06 y 0.08 puntos de separación. El fine-tuning uniformemente degrada la calidad del espacio de embeddings.

**Segundo hallazgo:** La separación A2M y M2A son prácticamente idénticas dentro de cada checkpoint (diferencia < 0.001). La asimetría A2M/M2A no se manifiesta como una diferencia en distribuciones de similitud.

### 7.7 Resultados: Drift Per-Piece vs Gate 2

| Checkpoint vs Gate 2 | Piezas que perdieron A2M | Piezas que ganaron M2A | Correlación |
|---------------------|------------------------|----------------------|-------------|
| RB0 | 41 de 137 (30%) | 52 de 137 (38%) | 0.448 |
| RA5 | 47 de 137 (34%) | 55 de 137 (40%) | 0.410 |
| R1 | 44 de 137 (32%) | 51 de 137 (37%) | 0.402 |

**Interpretación:** La correlación ~0.4 indica que las piezas que pierden A2M y las que ganan M2A se superponen moderadamente, pero no son las mismas. Hay piezas que pierden A2M sin ganar M2A, y viceversa. Esto sugiere que el efecto del fine-tuning no es uniforme: algunas piezas son más sensibles que otras.

### 7.8 Evaluación de Hipótesis

**H6.3 (Hubness crece con fine-tuning):**

| Dirección | Gate 2 skew | RB0 skew | RA5 skew | R1 skew | ¿Crece? |
|-----------|------------|---------|---------|---------|---------|
| A2M | 15.53 | 13.36 | 11.49 | 16.41 | Mixto |
| M2A | 9.30 | 11.80 | 11.25 | 13.47 | **SÍ** |

**Resultado: H6.3 PARCIALMENTE CONFIRMADA.** M2A hubness crece consistentemente con fine-tuning. A2M hubness no tiene una tendencia clara — RB0 y RA5 la reducen, pero R1 la aumenta.

**H6.4 (Separación A2M < separación M2A):**

Para todos los checkpoints, la separación A2M ≈ separación M2A (diferencia < 0.001). **H6.4 NO CONFIRMADA.** La asimetría A2M/M2A no se explica por una diferencia en distribuciones de similitud.

### 7.9 Figuras Generadas

**`fig_hubness_distribution.png`:** 2×2 grid mostrando la distribución de k-occurrence para cada checkpoint (dirección A2M). Las distribuciones son extremadamente sesgadas (cola derecha larga), con la gran mayoría de MIDI embeddings teniendo k-occurrence = 0-2, y unos pocos "hubs" con k-occurrence > 100. Las líneas verticales marcan la media (10.0) y el umbral 3σ.

**`fig_similarity_distributions.png`:** 2×4 grid mostrando distribuciones de similitud correct (verde) vs incorrect (rojo) para cada checkpoint. Fila superior: A2M. Fila inferior: M2A. Se observa claramente cómo Gate 2 tiene la mayor separación entre las dos distribuciones, y cómo fine-tuning reduce esta separación.

---

## 8. Track 1, Script 4: Visualizaciones Comparativas

### 8.1 Objetivo

Generar visualizaciones cualitativas del espacio de embeddings para los 4 checkpoints, complementando los análisis cuantitativos de Scripts 1 y 3.

### 8.2 Metodología

**Script:** `experiments/bias_control/visualize_embeddings_multigate.py` (~240 líneas)
**Requerimientos:** CPU solo, requiere `multigate_embeddings.npz` y `umap-learn`

**Estilo visual:** Dark theme (fondo `#0a0a0a`), Audio = cyan (`#00e5ff`), MIDI = magenta (`#ff1493`), texto con efecto de glow.

### 8.3 Figura 1: UMAP 2×2 Grid

**Algoritmo:**
1. Subsamplear 500 segmentos (seed=42) de los 13,532
2. Concatenar embeddings de los 4 checkpoints × 2 modalidades = 4,000 puntos de 256 dimensiones
3. Aplicar UMAP **conjunto** (`n_neighbors=30, min_dist=0.3, metric='cosine'`) sobre los 4,000 puntos → 4,000 coordenadas 2D
4. Separar por checkpoint y modalidad para plotear en 4 paneles

**¿Por qué UMAP conjunto?** Es crítico que los 4 paneles compartan el mismo espacio de coordenadas. Si se calculara UMAP independientemente para cada checkpoint, las posiciones serían incomparables (UMAP es no determinístico en layout). Al ajustar un único reducer sobre todos los datos, las posiciones son directamente comparables entre paneles.

**Observaciones:**
- Gate 2 muestra los clusters de audio (cyan) y MIDI (magenta) más mezclados — buena alineación cross-modal
- Los checkpoints fine-tuned muestran una tendencia de los clusters MIDI a "apretarse" (menor dispersión) mientras los clusters audio mantienen una distribución similar a Gate 2
- Esto es consistente con el hallazgo de Script 1: el audio encoder no cambia, pero el MIDI encoder sí. Los embeddings MIDI se reorganizan en el espacio mientras los audio quedan anclados.

### 8.4 Figura 2: Cross-Modal Bridges

**Algoritmo:**
1. Tomar 200 de los 500 segmentos subsampleados
2. Para cada checkpoint: dibujar una línea entre cada embedding de audio y su embedding de MIDI pareado (mismo segmento) en coordenadas UMAP
3. Colorear cada línea por la distancia euclidiana en UMAP: verde = corta (bien alineado), rojo = larga (mal alineado), usando colormap `RdYlGn_r`
4. Anotar la distancia media de los bridges

**Resultados cuantitativos (distancia media de bridges):**

| Checkpoint | Mean Bridge Distance | Cambio vs Gate 2 |
|------------|---------------------|-------------------|
| Gate 2 | **3.27** | — |
| RB0 | 4.50 | **+37.6%** |
| RA5 | 4.47 | **+36.7%** |
| R1 | 4.68 | **+43.1%** |

**Interpretación:** Los "puentes" entre pares audio-MIDI se alargan un 37-43% con fine-tuning. Esto confirma visualmente lo que las métricas de separación mostraron: el fine-tuning empuja los embeddings de audio y MIDI MÁS LEJOS uno del otro, no más cerca. La alineación cross-modal empeora.

**¿Por qué pasa esto?** Con el audio encoder congelado, los embeddings de audio solo pueden moverse vía la Audio Projection (~11% de cambio). Los embeddings de MIDI se mueven vía MIDI encoder + MIDI Projection (8-13% de cambio combinado). Pero estos movimientos no son coordinados — la projection de audio intenta compensar pero no puede seguir el ritmo de los cambios del MIDI side, resultando en bridges más largos.

### 8.5 Figura 3: Heatmaps de Similitud Coseno

**Algoritmo:**
1. Tomar 50 segmentos
2. Para cada checkpoint: calcular la matriz de similitud coseno [50 audio × 50 MIDI]
3. Normalizar embeddings a norma unitaria antes del producto punto
4. Plotear como heatmap con colormap `magma` (negro = 0, blanco = 1)
5. Anotar con la media de la diagonal (pares correctos) y la media off-diagonal (pares incorrectos)

**Resultados cuantitativos:**

| Checkpoint | Diagonal mean | Off-diagonal mean | Contraste |
|------------|-------------|-------------------|-----------|
| Gate 2 | Alto | Bajo | Fuerte |
| RB0 | Moderado | Bajo | Moderado |
| RA5 | Moderado | Bajo | Moderado |
| R1 | Moderado | Bajo | Moderado |

**Observación visual:** Gate 2 tiene la diagonal más brillante y definida. En los checkpoints fine-tuned, la diagonal es más tenue y hay más "ruido" fuera de la diagonal. Esto es consistente con la degradación de separación observada en Script 3.

---

## 9. Track 2, Script 5: H4.2-6 Pre-Red Test

### 9.1 Objetivo

Determinar si es viable extraer histogramas de ratios armónicos desde audio (vía análisis espectral CQT) y compararlos con histogramas extraídos desde MIDI. Si funciona, esto abriría la puerta a una señal auxiliar "dual-domain" que podría mejorar ambas direcciones (A2M y M2A) de forma simétrica.

### 9.2 Concepto: ¿Qué son los Ratios Armónicos?

Un ratio armónico es la relación de frecuencia entre dos notas. Por ejemplo:
- Unísono: 1:1 (ratio = 1.0)
- Octava: 2:1 (ratio = 2.0)
- Quinta justa: 3:2 (ratio = 1.5)
- Cuarta justa: 4:3 (ratio = 1.333)
- Tercera mayor: 5:4 (ratio = 1.25)

Un histograma de ratios captura la distribución de estos intervalos en un segmento musical. La hipótesis H4.2-6 postula que si los ratios se extraen tanto desde MIDI (donde las frecuencias son exactas) como desde audio (donde las frecuencias deben inferirse del espectro), los histogramas de pares alineados (mismo segmento) deberían ser más similares que los de pares aleatorios.

### 9.3 Metodología

**Script:** `experiments/bias_control/h426_prered_test.py` (~700 líneas)
**Requerimientos:** CPU solo, `librosa`, `pretty_midi`, `pyfluidsynth`, `scipy`

**Protocolo de dos fases:**

#### Fase P0 (Oracle): Audio sintetizado desde MIDI

1. Seleccionar 100 segmentos del validation set (seed=42)
2. Para cada segmento:
   a. **Extracción de ratio histogram desde MIDI:**
      - Parsear el archivo MIDI con `pretty_midi`
      - Filtrar notas activas en la ventana temporal [start, start+4s]
      - Convertir MIDI note numbers a frecuencias Hz (`pretty_midi.note_number_to_hz`)
      - Deduplicar frecuencias (redondear a 0.1 Hz)
      - Calcular ratios entre todos los pares de frecuencias: `f_i / f_j` para `i ≠ j`
      - Filtrar ratios al rango [0.5, 2.0] (una octava abajo a una octava arriba)
      - **Soft binning** en 256 bins con kernels gaussianos (σ = 0.5 × ancho de bin)
      - Normalizar a suma = 1
   b. **Síntesis de audio desde MIDI:**
      - Usar `pretty_midi.fluidsynth(fs=16000)` con SoundFont TimGM6mb
      - Recortar la señal sintetizada a la ventana temporal del segmento
   c. **Extracción de ratio histogram desde audio:**
      - Aplicar CQT: 84 bins, 12 bins por octava, fmin=27.5 Hz (A0)
      - Promediar el CQT sobre el eje temporal → espectro de 84 bins
      - Detectar los top-12 peaks por amplitud
      - Convertir índices de bins CQT a frecuencias Hz
      - **Supresión de armónicos:** Eliminar frecuencias que son 2x o 3x de una frecuencia más baja (tolerancia 5%). Sin esto, todos los histogramas estarían dominados por octavas y quintas triviales.
      - Calcular ratios entre pares restantes
      - Soft binning y normalización (mismo proceso que MIDI)
   d. Calcular **similitud coseno** entre los dos histogramas

3. **Generación de pares random:** Para cada segmento, emparejar su histograma de audio con el histograma MIDI de un segmento aleatorio diferente. Calcular similitud coseno.

4. **Métricas P0:**
   - `AUC_P0`: Area Under ROC Curve discriminando pares aligned vs random
   - `delta_sim_P0`: media(sim_aligned) - media(sim_random)
   - Bootstrap CI 95% para AUC (1000 iteraciones)

#### Fase P1 (Real): Audio real de MAESTRO

Repite el proceso pero usando audio real en lugar de sintetizado:
1. Para cada segmento: `librosa.load(audio_path, sr=16000, offset=start_time, duration=4.0)`
2. Mismo pipeline CQT → peaks → supresión armónica → ratios → soft binning
3. Comparar con MIDI ratio histogram
4. **Métricas adicionales P1:** Wilcoxon signed-rank test (paired), degradation ratio (P1_AUC / P0_AUC)

### 9.4 Resultados P0 (Oracle)

| Métrica | Valor | Umbral GO |
|---------|-------|-----------|
| **AUC** | **0.5592** | >= 0.80 |
| AUC CI 95% | [0.4798, 0.6311] | — |
| **delta_sim** | **0.0341** | >= 0.10 |
| mean_aligned_sim | 0.7801 | — |
| mean_random_sim | 0.7460 | — |
| std_aligned_sim | 0.1431 | — |
| std_random_sim | 0.1625 | — |
| N valid | 100 | — |
| N skipped | 0 | — |
| **Decisión** | **NO-GO** | AUC < 0.65 |

**Interpretación:** Incluso bajo condiciones ideales (audio sintetizado perfectamente desde el mismo MIDI), el pipeline CQT produce histogramas de ratios que son CASI INDISTINGUIBLES entre pares alineados y aleatorios. La similitud media alineada (0.780) es apenas superior a la similitud media random (0.746). El AUC de 0.559 está apenas por encima del azar (0.50).

### 9.5 Resultados P1 (Real Audio)

| Métrica | Valor | Umbral GO |
|---------|-------|-----------|
| **AUC** | **0.5018** | >= 0.70 |
| AUC CI 95% | [0.4222, 0.5875] | — |
| **delta_sim** | **-0.0041** | >= 0.05 |
| mean_aligned_sim | 0.7869 | — |
| mean_random_sim | 0.7910 | — |
| std_aligned_sim | 0.1543 | — |
| std_random_sim | 0.1391 | — |
| **Wilcoxon p** | **0.716** | < 0.01 |
| Degradation (P1/P0) | 0.897 | — |
| N valid | 100 | — |
| N skipped | 0 | — |
| **Decisión** | **NO-GO** | AUC < 0.55 |

**Interpretación:** Con audio real, el resultado es aún peor. El AUC de 0.502 es EXACTAMENTE el azar. El delta_sim es **negativo** (-0.004), lo que significa que los pares aleatorios son marginalmente MÁS similares que los pares alineados. El Wilcoxon p = 0.716 confirma que no hay diferencia estadísticamente significativa entre las distribuciones aligned y random.

### 9.6 ¿Por Qué Falla?

El análisis de los histogramas de ratio (visible en `fig_histogram_overlay.png`) revela la causa:

1. **Todos los segmentos producen histogramas muy similares.** La música tonal de piano (MAESTRO es 100% piano) comparte una estructura armónica base: predominan octavas (ratio 2.0), quintas (1.5), y cuartas (1.33). Aunque la supresión de armónicos elimina las octavas/quintas directas, las relaciones entre peaks restantes siguen siendo similares entre segmentos.

2. **El CQT promediado temporalmente pierde especificidad.** Al promediar el espectrograma CQT sobre 4 segundos, las frecuencias instantáneas se difuminan. Dos segmentos con diferentes secuencias de notas pero el mismo rango de pitch producirán espectros promedio similares.

3. **12 peaks no capturan suficiente detalle.** Con solo 12 peaks y supresión armónica (que puede reducirlos a 4-8), los pares de ratios resultantes son demasiado pocos para discriminar.

4. **El problema es fundamental, no paramétrico.** Si ni siquiera el oracle (audio sintetizado del mismo MIDI) funciona, ajustar parámetros (n_peaks, bins, CQT config) no resolverá el problema. La información de ratios armónicos **no se preserva lo suficiente** en la representación CQT-averaged para ser discriminativa a nivel de segmento.

### 9.7 Figuras Generadas

**`fig_histogram_overlay.png`:** 2×3 grid mostrando 6 segmentos ejemplo. Cada panel superpone el histograma de audio (cyan) y MIDI (magenta). Se observa que ambos histogramas tienen picos en posiciones similares pero con amplitudes diferentes, y que los histogramas de DIFERENTES segmentos son muy parecidos entre sí — falta de especificidad.

**`fig_roc_p0_p1.png`:** Curvas ROC para P0 (cyan) y P1 (magenta). Ambas curvas están apenas por encima de la diagonal (azar). P0 es ligeramente mejor que P1 pero ambas son inadecuadas.

**`fig_similarity_scatter.png`:** Scatter plot de similitud aligned vs random por segmento, con medias e intervalos de confianza. Los puntos se agrupan cerca de la diagonal y = x, confirmando que aligned y random son indistinguibles.

### 9.8 Conclusión Track 2

**H4.2-6 está DEFINITIVAMENTE ELIMINADA del mapa de hipótesis.**

No se justifica gastar compute en training con ratios dual-domain porque el método de extracción de ratios desde audio no produce información discriminativa. Incluso bajo condiciones oráculo, el AUC es 0.559 (apenas sobre azar). Con audio real, es exactamente azar (0.502).

---

## 10. Aplicación de la Matriz de Decisión 2×2

### 10.1 Evaluación de Condiciones

**Condición 1: "¿Drift asimétrico?"**

La respuesta es **SÍ**, pero de una forma diferente a la hipotética:
- No hay "forgetting asimétrico" en las proyecciones (son simétricas, ~1.0x ratio)
- Hay una **asimetría fundamental en la trainability**: el Audio Encoder completo (79% de los parámetros) está completamente congelado, mientras el MIDI Encoder (18.5%) cambia 3-12%
- Las proyecciones de ambos lados cambian similarmente (~11-14%), pero el Audio Projection solo puede compensar parcialmente la falta de adaptación del backbone de audio
- Los bridges cross-modal se alargan 37-43% y la separación de embeddings cae 0.06-0.08

**Condición 2: "¿H4.2-6 GO?"**

**NO.** P0 AUC = 0.559 < 0.65 (NO-GO), P1 AUC = 0.502 < 0.55 (NO-GO).

### 10.2 Celda de la Matriz

| Gate 6: ¿drift asimétrico? | H4.2-6 P1: ¿GO? | Siguiente paso |
|---|---|---|
| **SÍ** (audio frozen) | **NO** | **H4.2-2 adapter + H4.2-1 audio-only + S-control** |

### 10.3 Implicaciones

La combinación Drift=SÍ + H4.2-6=NO descarta el enfoque más ambicioso (dual-domain ratios con adapter) y apunta a una solución más conservadora y directa:

1. **H4.2-2 (Adapter layers):** Añadir módulos adapter trainables AL INTERIOR del audio encoder congelado, permitiendo que se adapte sin catastrófico forgetting. Esto ataca directamente la causa raíz (backbone congelado).

2. **H4.2-1 (Audio-only fine-tuning):** Descongelar selectivamente capas del audio encoder (empezando por las más profundas, layers 3-4 del transformer + projection). Más agresivo que adapters pero potencialmente más poderoso.

3. **S-control (mandatory):** Cualquier training debe incluir controles de estabilidad (e.g., un checkpoint control que solo cambia el learning rate schedule sin modificar la arquitectura).

---

## 11. Diagnóstico Integrado: Causa Raíz

### 11.1 El Mecanismo de la Degradación A2M

Integrando los hallazgos de los 5 scripts, el mecanismo de degradación A2M es claro:

```
                    ANTES (Gate 2)                      DESPUÉS (Fine-tuning)

Audio Encoder ──→ features_A ──→ proj_A ──→ emb_A     Audio Encoder ──→ features_A' ──→ proj_A' ──→ emb_A'
    (trained)         |              |          |            (FROZEN)         |              |          |
    [cambió]          ↓              ↓          ↕            [=Gate 2]       ↓              ↓          ↕
    durante      alineados     alineados   VICReg          features_A'    cambia       se mueve    VICReg
    Gate 2       con MIDI       con MIDI    loss          = features_A   ~11-14%      pero NO     loss
    training         |              |          ↕                            parcial    suficiente     ↕
MIDI Encoder  ──→ features_M ──→ proj_M ──→ emb_M     MIDI Encoder  ──→ features_M' ──→ proj_M' ──→ emb_M'
    (trained)         |              |          |            (CAMBIA)         |              |          |
    [cambió]          ↓              ↓          ↓            [8-12%]         ↓              ↓          ↓
                 alineados     alineados   target             SE MUEVE     SE MUEVE      SE MUEVE
                                                              lejos de     lejos de      lejos de
                                                              Gate 2       Gate 2        emb_A'
```

**Paso a paso:**

1. **Gate 2** entrenó AMBOS encoders + proyecciones conjuntamente durante 45 epochs. Al final, audio y MIDI embeddings están bien alineados (separación 0.479, bridges cortos 3.27).

2. **Fine-tuning** carga el checkpoint Gate 2 pero el optimizer solo actualiza:
   - Audio Projection (lr=1e-3)
   - MIDI Encoder (lr=1e-4)
   - MIDI Projection (lr=1e-3)
   - El Audio Encoder NO está en los parameter groups del optimizer → gradientes se propagan pero no se aplican

3. **El MIDI side evoluciona:** El MIDI Transformer cambia 8-12%, reorganizando las features MIDI. La MIDI Projection cambia 11-13%, moviendo los embeddings MIDI en el espacio 256d.

4. **El audio side solo compensa parcialmente:** La Audio Projection cambia 11-14% (intentando seguir al MIDI side), pero sin cambios en el backbone, las features de audio son las mismas de Gate 2. La projection puede rotar y escalar, pero no puede crear nuevas features.

5. **Resultado:** Los embeddings MIDI se mueven en una dirección que las features de audio congeladas no pueden seguir completamente. Los bridges se alargan (3.27 → 4.50-4.68), la separación correct/incorrect cae (0.479 → 0.395-0.419), y el A2M recall cae (-3 a -4.2pp).

6. **M2A mejora ligeramente** porque el MIDI encoder, al cambiar, puede "aprender" a proyectar hacia donde ya están los embeddings de audio (que no se mueven). Es más fácil para el lado que cambia apuntar hacia un target estático.

### 11.2 ¿Por Qué RB0 También Degrada A2M?

RB0 es el control sin señal de ratios (solo VICReg). Su A2M baja de 34.4% a 30.2% (-4.2pp). Esto tiene la misma causa: el audio encoder está congelado en RB0 también. No importa qué loss function se use — si el audio encoder no se adapta, cualquier fine-tuning que mueva el MIDI side creará la misma desalineación.

### 11.3 ¿Por Qué No Se Descubrió Esto Antes?

La configuración de fine-tuning usa `strict=False` al cargar checkpoints y parameter groups específicos en el optimizer. Nadie verificó explícitamente que el Audio Encoder estuviera recibiendo gradientes y actualizaciones. El training de Gate 2 (45 epochs from scratch) incluía el Audio Encoder en los parameter groups, pero el script de fine-tuning de Gate 4.1 no lo incluía. Esto era efectivamente un bug de configuración, no una decisión deliberada.

---

## 12. Próximos Pasos

### 12.1 Acciones Inmediatas (requieren DEC-006)

1. **Verificar la configuración de fine-tuning:** Confirmar en los scripts de training de Gate 4.1 que el Audio Encoder efectivamente no tenía parameter group en el optimizer. Si esto fue intencional, documentar la razón. Si fue un error, es la causa raíz y la solución es simple: incluir el Audio Encoder en los parameter groups.

2. **H4.2-2 (Adapter Layers):** Diseñar y implementar módulos adapter (e.g., bottleneck adapters) que se insertan en las capas del Audio Transformer congelado. Cada adapter es un pequeño MLP (e.g., d→d/4→d) que se entrena mientras el transformer principal queda fijo. Esto permite adaptación sin riesgo de forgetting catastrófico.

3. **H4.2-1 (Selective Unfreezing):** Descongelar progresivamente capas del Audio Encoder, empezando por las más profundas (layer 3, luego 2). Con learning rate bajo (1e-5) y warmup, esto permite que el backbone se adapte gradualmente.

4. **S-control (Mandatory):** Diseñar un control que solo cambie el schedule de training (e.g., solo las proyecciones) para aislar el efecto de descongelar el Audio Encoder.

### 12.2 Gate 4.2 Hypothesis Map v2

| Hipótesis | Prioridad | Estado |
|-----------|-----------|--------|
| ~~H4.2-6 dual-domain ratios~~ | — | **ELIMINADA** (NO-GO P0 y P1) |
| H4.2-2 adapter layers | **#1** | Pendiente DEC-006 |
| H4.2-1 audio-only unfreezing | **#2** | Pendiente DEC-006 |
| H4.2-4 | — | Descartada |
| H4.2-5 | Backlog | Backlog |

### 12.3 Gate 6 Phase 2

DEC-005 cubre Gate 6 **Phase 1** (diagnóstico de asimetría). El roadmap define Gate 6 Phase 2 como un análisis científico más profundo que incluye:
- RSA/CKA entre representaciones
- Linear probes de ratio features desde embeddings congelados
- Disagreement analysis (retrieval con embeddings vs representaciones clásicas)

Phase 2 se planifica después de que se ejecute la primera ola de training de Gate 4.2, para poder incluir los nuevos checkpoints en el análisis comparativo.

---

## 13. Artefactos Generados

### 13.1 Scripts (todos en `experiments/bias_control/`)

| Script | Líneas | Track | GPU | Tiempo |
|--------|--------|-------|-----|--------|
| `compare_layer_drift.py` | ~210 | Gate 6 Ph1 | No | <1 min |
| `extract_multigate_embeddings.py` | ~145 | Gate 6 Ph1 | Sí | 22.8 min |
| `analyze_hubness.py` | ~230 | Gate 6 Ph1 | No | ~2 min |
| `visualize_embeddings_multigate.py` | ~240 | Gate 6 Ph1 | No | ~20 sec |
| `h426_prered_test.py` | ~700 | H4.2-6 | No | 2.8 min |

### 13.2 Datos (paths relativos a `data/bias_control_medium/`)

| Archivo | Tamaño | Contenido |
|---------|--------|-----------|
| `evaluations/gate4/RA5_ep5.json` | ~2 KB | Evaluación canónica RA5 |
| `evaluations/gate6/layer_drift.json` | 11 KB | Drift por módulo, 3 checkpoints |
| `evaluations/gate6/multigate_embeddings.npz` | 108 MB | 8 arrays [13532, 256] + metadata |
| `evaluations/gate6/hubness_analysis.json` | 23 KB | K-occurrence, per-piece, similitud |
| `evaluations/gate42/h426_prered_results.json` | 1.1 KB | Métricas P0/P1, decisiones |

### 13.3 Figuras (paths relativos a `data/bias_control_medium/`)

| Archivo | Script | Contenido |
|---------|--------|-----------|
| `evaluations/gate6/fig_umap_multigate.png` | 4 | UMAP 2×2, audio vs MIDI por checkpoint |
| `evaluations/gate6/fig_bridges_multigate.png` | 4 | Bridges cross-modal con distancias |
| `evaluations/gate6/fig_heatmaps_multigate.png` | 4 | Matrices coseno [50×50] por checkpoint |
| `evaluations/gate6/fig_hubness_distribution.png` | 3 | K-occurrence distribution A2M |
| `evaluations/gate6/fig_similarity_distributions.png` | 3 | Correct vs incorrect sim, A2M y M2A |
| `evaluations/gate42/fig_histogram_overlay.png` | 5 | Audio vs MIDI ratio histograms |
| `evaluations/gate42/fig_roc_p0_p1.png` | 5 | ROC curves P0 y P1 |
| `evaluations/gate42/fig_similarity_scatter.png` | 5 | Aligned vs random similarity |

---

## 14. Apéndice: Datos Crudos

### 14.1 Layer Drift — Tabla Completa (9 módulos × 3 checkpoints)

#### Relative Change Mean (%) por módulo y checkpoint:

| Módulo | RB0 | RA5 | R1 |
|--------|-----|-----|-----|
| Audio CNN | 0.000 | 0.000 | 0.000 |
| Audio PosEmb | 0.000 | 0.000 | 0.000 |
| Audio Transformer | 0.000 | 0.000 | 0.000 |
| Audio Projection | 11.337 | 13.854 | 11.346 |
| MIDI Embedding | 3.515 | 4.970 | 3.475 |
| MIDI OutputNorm | 159.990 | 157.527 | 111.816 |
| MIDI PosEncoding | 0.000 | 0.000 | 0.000 |
| MIDI Transformer | 8.315 | 12.110 | 8.489 |
| MIDI Projection | 11.794 | 13.066 | 10.790 |

#### Cosine Sim Min por módulo y checkpoint:

| Módulo | RB0 | RA5 | R1 |
|--------|-----|-----|-----|
| Audio CNN | 0.000* | 0.000* | 0.000* |
| Audio PosEmb | 1.000 | 1.000 | 1.000 |
| Audio Transformer | 0.000* | 0.000* | 0.000* |
| Audio Projection | 0.000* | 0.000* | 0.000* |
| MIDI Embedding | 0.995 | 0.991 | 0.995 |
| MIDI OutputNorm | 0.521 | 0.467 | 0.533 |
| MIDI PosEncoding | 1.000 | 1.000 | 1.000 |
| MIDI Transformer | 0.974 | 0.945 | 0.971 |
| MIDI Projection | 0.000* | 0.000* | 0.000* |

*\*Los valores 0.000 en módulos con L2=0 (Audio CNN, Transformer) se deben a vectores bias de norma cero, donde cos(0,0) es indefinido. En módulos con L2>0 (Audio Projection, MIDI Projection), el cos_min=0 indica que algún parámetro bias cambió drásticamente de dirección.*

### 14.2 Hubness — Tabla de Hubs (k-occurrence > 3σ)

| Checkpoint | A2M hubs | A2M max k-occ | M2A hubs | M2A max k-occ |
|------------|---------|--------------|---------|--------------|
| Gate 2 | 103 | 1,048 | 193 | 576 |
| RB0 | 158 | 1,083 | 162 | 1,018 |
| RA5 | 164 | 1,069 | 183 | 846 |
| R1 | 146 | 1,258 | 146 | 1,164 |

### 14.3 Gate 4 Evaluations — Tabla Comparativa Completa

| Métrica | Gate 2 | RB0 | RA5 | R1-rescue |
|---------|--------|-----|-----|-----------|
| **A2M R@1** | 4.4% | 6.4% | 6.2% | 4.4% |
| **A2M R@5** | 20.8% | 18.4% | 19.6% | 19.4% |
| **A2M R@10** | **34.4%** | 30.2% | 31.4% | 31.0% |
| **A2M R@20** | 52.0% | 42.2% | 45.8% | 44.8% |
| A2M MRR | 0.138 | 0.141 | 0.144 | 0.135 |
| A2M Mean Rank | 37.4 | 48.7 | 44.2 | 47.3 |
| A2M Median Rank | 18.0 | 26.0 | 22.0 | 23.5 |
| **M2A R@1** | 5.2% | 6.0% | 5.4% | 5.6% |
| **M2A R@5** | 24.6% | 25.2% | 26.6% | 24.8% |
| **M2A R@10** | **37.6%** | 38.2% | 40.6% | 40.2% |
| **M2A R@20** | 56.4% | 53.8% | 55.6% | 53.4% |
| M2A MRR | 0.158 | 0.165 | 0.163 | 0.159 |
| M2A Mean Rank | 31.6 | 39.0 | 35.7 | 42.0 |
| M2A Median Rank | 16.0 | 16.0 | 15.0 | 16.0 |
| **Hard Neg (same-piece)** | **80.4%** | 77.6% | 79.0% | 78.8% |
| **Hard Neg (random)** | **87.0%** | 80.8% | 83.6% | 80.8% |

### 14.4 H4.2-6 Pre-Red — Configuración Completa

```json
{
  "n_segments": 100,
  "sr": 16000,
  "n_bins": 256,
  "n_peaks": 12,
  "ratio_min": 0.5,
  "ratio_max": 2.0,
  "seed": 42,
  "sf2_path": "/usr/share/sounds/sf2/TimGM6mb.sf2",
  "segment_len": 4.0,
  "cqt_n_bins": 84,
  "cqt_bins_per_octave": 12,
  "cqt_fmin": 27.5,
  "harmonic_suppression_tolerance": 0.05,
  "soft_bin_sigma": "0.5 * bin_width"
}
```

---

*Informe generado como parte de DEC-005 (DIAGNOSTIC-ONLY). Ningún training fue ejecutado. Todos los análisis se basan en checkpoints pre-existentes de Gate 2 y Gate 4.1.*

*Fecha: 2026-02-11 | Ubicación: `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/04_DIAGNOSTICO_GATE_6_Y_GATE_4_2/INFORME_DEC005_DIAGNOSTICO_COMPLETO.md`*
