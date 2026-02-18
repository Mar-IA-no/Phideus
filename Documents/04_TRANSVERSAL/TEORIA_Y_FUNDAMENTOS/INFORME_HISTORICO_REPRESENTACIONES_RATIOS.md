# Informe Histórico de Representaciones de Ratios en Phideus

**El Hilo de Ariadna: De la Visión Fundacional a BIAS_CONTROL**

**Fecha**: 2026-02-17
**Autor**: Claude Code (análisis y síntesis)
**Versión**: 1.3

---

## Addendum Operativo Vivo (2026-02-17)

Este informe historico se mantiene sincronizado con el roadmap activo de BIAS_CONTROL en conjunto con:

- `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/CATALOGO_NARRATIVO_DESCRIPTORES_RATIOS_PHIDEUS.md`

Politica de mantenimiento transversal acordada:

1. `INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md` y `CATALOGO_NARRATIVO_DESCRIPTORES_RATIOS_PHIDEUS.md` se actualizan cuando entra o cambia un descriptor.
2. `BACKPROPAGANDO_PHIDEUS.md` queda fuera de esta regla por ser documento exploratorio/ideacional.

Estado operativo de descriptores al corte:

- Gate 4.2: cerrado con `D4 8ep` (`S_best=64.2%`, `hard_neg_best=91.6%`).
- Gate 4.3: cerrado con 13 brazos de 5ep + 1 run largo scratch.
  - Mejor dual same-modality: `d4a4` (`S=69.8%`).
  - Mejor single-descriptor: `A4r` (`S=68.6%`).
  - Mejor run largo: `d4a4-scratch e30` (`S=83.6%`, multi-seed `84.1% +/- 2.3pp`).
- Balance descriptorial al cierre:
  - Núcleo robusto: `D4` (intervalos MIDI) y `A4` (log-freq deltas audio).
  - A7/A8/A9 no desplazaron a A4 en rendimiento canónico 5ep.
  - Inyección cross-modal temprana (`d4a4cm`) quedó por debajo de baseline.
- Gate 4.4: corte parcial en UNC (8 brazos x 5ep, `foundation_locked_e25.pt`, `freeze-policy=run-d`):
  - e5 cerrados: `t3-wt` (`S=67.6%`), `t3-tri` (`S=65.0%`), `t3-anc` (`S=42.2%`), `moe-a4` (`S@e5=58.2%`, best `S@e3=58.8%`).
  - e3 disponibles: `film-a4` (`S=59.2%`), `film-d4` (`S=58.8%`).
  - pendientes de structured eval consolidada: `film-dual`, `moe-dual`.
- Extensión Gate 4.3 `d4-a4r`: corrida larga scratch relanzada en UNC (en curso).
- Gate 5A/5B: definidos y pendientes (barrido descriptorial + validación científica).

---

## 0. Resumen Ejecutivo

Este documento traza la evolución completa de los sistemas de representación de ratios armónicos en el proyecto Phideus, desde su concepción como "IAm Phideus" en 2025 hasta el experimento BIAS_CONTROL en curso (febrero 2026). La tesis central es que **la representación de datos ha sido consistentemente más determinante que la arquitectura neuronal**, y que cada paso —incluso los fracasos— ha contribuido a la comprensión de cómo codificar relaciones frecuenciales para aprendizaje automático.

### Tabla 1: Evolución de Analizadores (Extractores de Ratios)

| Versión | Fecha | Output Shape | Escala | Bins | Características |
|---------|-------|--------------|--------|------|-----------------|
| CQT (pre-v2) | ~May 2025 | Variable | Musical | 7 oct × 120 bins | Primer intento, descartado por bias musical |
| v2.2 | Jun 2025 | [100] | log₂ | 100 bins | Baseline STFT multi-resolución |
| v3.0-v3.3 | Jun 2025 | [200-512] | log₂ | 1.2-2.9 cents/bin | CLI configurable, primer commit |
| v4.0 | Jun 2025 | [256, 3] | log₂ | 256 bins | Primera representación enriquecida |
| v4.1 Enriched | Ago 2025 | [512, 3] | log₂ (cents) | 6.1 cents/bin | Estándar global, 3 canales |
| v5.0 | Ene 2026 | **[T, 256, 3]** | Linear | 256 bins | **PARADIGM SHIFT**: Temporal, 170× datos |
| Roseta v2.2 | Ene 2026 | [T, 256, 3] | Linear | 256 bins | Dual-domain, estabilidad temporal |
| Constellations | Feb 2026 | [T, K, 5] | log_ratio | Tokens sparse | Estilo Shazam, anchor-target pairs |
| Route A | Feb 2026 | Tokens | Event-based | Variable | 71.4% accuracy (N=10) |
| Route B | Feb 2026 | Tokens | CQT+onset | Variable | 80.0% accuracy (N=10) |

### Tabla 2: Arquitecturas RNA

| Modelo | Fecha | Parámetros | Resultado Clave | Dataset |
|--------|-------|------------|-----------------|---------|
| VAE v4.1 | Ago 2025 | 1.6M | val_loss: 4212 (catastrófico) | Sintético v4.1 |
| HRM | Ago 2025 | 6.0M | val_loss: 2.74 (99.93% mejor) | Sintético v4.1 |
| VAE Temporal (5.0) | Ene 2026 | 1.82M | val_loss: 0.4560 | Sintético 5.0 |
| HRM Temporal (5.0) | Ene 2026 | 2.27M | val_loss: 0.4607 | Sintético 5.0 |
| RosetaVAE | Ene 2026 | 3.16M | Gap: 0.007 (NO-GO) | UOEMD |
| ConstellationVAE | Feb 2026 | 196K-523K | Top-1: 0.78% (random) | UOEMD |
| JEPA-lite | Feb 2026 | 196K-258K | Top-1: 1.56% | UOEMD |
| **BIAS_CONTROL** | **Feb 2026** | **MERT 330M + custom** | **Gap: 0.478, Hard neg 80.4%** | **MAESTRO** |
| BIAS_CONTROL+DANN | Feb 2026 | MERT 330M + DANN GRL | Domain acc 62.7% (ep7 best) | MAESTRO |

### Tabla 3: Cronología del Proyecto

| Fecha | Commit | Evento | Resultado |
|-------|--------|--------|-----------|
| 3 Jun 2025 | `ac041c4` | **Primer commit**: v3.3, checkers, generadores | Proyecto iniciado |
| 28 Jun 2025 | `54e8b6a` | v4.0: Histogramas 3 canales | +análisis armónico |
| 6 Ago 2025 | `cdb06bf` | v4.1 Enriched: 512 bins | Estándar establecido |
| 11 Ago 2025 | `c2875d0` | Arquitectura dual v4.1 | HRM >> VAE (ilusorio) |
| 13 Ene 2026 | `995cb2a` | **v5.0 PARADIGM SHIFT** | VAE = HRM |
| 30 Ene 2026 | `6dc82a9` | Extractor v2.2 (Roseta) | Gap 172× mejor |
| 1 Feb 2026 | `601280d` | Constellations | NO-GO (= random) |
| 4 Feb 2026 | `d023811` | Route A/B | GO (N=10) |
| 5 Feb 2026 | `95e80a5` | **BIAS_CONTROL Gate 2 COMPLETADO** | **GO**: Gap 0.478, R@10 34.4%, Hard neg 80.4% |
| 5 Feb 2026 | `96d9443` | Gate 3 DANN lanzado | Smoke test GO, training 30 epochs |

---

## 1. Introducción y Orígenes Conceptuales

### 1.1 El Nombre: "IAm Phideus"

El proyecto fue bautizado **IAm Phideus**, en referencia al escultor griego Phidias (c. 500-432 a.C.), célebre por su dominio de las proporciones áureas en obras como la estatua de Zeus en Olimpia. El nombre encapsula la visión fundacional: una inteligencia artificial que "entiende el mundo a través de las proporciones".

> *"IAm Phideus se concibió con el objetivo de reconocer patrones armónicos (razones de frecuencia) en los sonidos de entornos naturales sin depender de escalas musicales tradicionales, entrenando una inteligencia artificial capaz de 'entender el mundo a través de las proporciones' e identificar estructuras relacionales profundas en los paisajes sonoros."*
> — Casi Paper Phideus

### 1.2 La Visión Fundacional

El proyecto parte de una premisa audaz: que los **intervalos** (razones f₁:f₂ entre dos frecuencias) constituyen unidades fundamentales de información que diferentes sistemas —físicos, biológicos, cognitivos— pueden interpretar. Esta "Teoría de la Información Armónica" postula que las relaciones entre señales, más que las señales aisladas, son la clave para comprender la organización de la naturaleza.

> *"IAm Phideus postula que un intervalo (la razón f₁:f₂ entre dos frecuencias) puede actuar como unidad fundamental de información que distintos sistemas – físicos, biológicos o cognitivos – pueden 'leer' e interpretar."*

### 1.3 Las 10 Premisas Originales

El marco teórico se construyó sobre 10 premisas interconectadas:

1. **Estructura Natural**: La naturaleza posee estructura armónica inherente en sus paisajes sonoros
2. **Eliminación de Bias**: Es necesario eliminar sesgos musicales humanos para descubrir patrones naturales
3. **Ratios como Información**: Las relaciones de frecuencia son unidades informativas fundamentales
4. **Histogramas de Ratios**: Los histogramas de razones de frecuencia resumen la estructura acústica
5. **Dominios Complementarios**: Los dominios logarítmico (perceptual) y lineal (físico) son complementarios
6. **Enriquecimiento**: Añadir energía y entropía por intervalo robustece la representación
7. **Multi-resolución**: El análisis multirresolución captura todos los patrones
8. **Aprendibilidad**: Un modelo de IA puede aprender la estructura armónica latente
9. **Perfiles Ambientales**: Cada entorno sonoro tiene un perfil armónico identificable
10. **IA como Oyente Resonante**: La IA puede actuar como un "oyente" alineado con estructuras profundas

Estas premisas se consolidaron posteriormente en **tres hipótesis operativas**:
- **H1 - Estructura**: Las señales contienen distribuciones de ratios estructuradas
- **H2 - Aprendibilidad**: Redes neuronales pueden aprenderlas
- **H3 - Cross-modality**: Diferentes dominios (audio, vibración, MIDI) comparten estructura de ratios

### 1.4 Fundamentos Científicos

La teoría se apoya en evidencia de múltiples campos:

**Neurociencia**: Soto-León et al. (2019) demostraron que la relación 2:1 entre ondas theta (~5 Hz) y alpha (~10 Hz) facilita la sincronización neuronal.

**Ecología**: La **Hipótesis del Nicho Acústico** de Bernie Krause propone que en ecosistemas saludables, cada especie ocupa su propio rango de frecuencias, formando una "sinfonía natural".

**Universalidad Matemática**: Una proporción 3:2 produce propiedades similares ya ocurra entre 300 Hz y 200 Hz o entre 1500 Hz y 1000 Hz —los ratios son adimensionales.

### 1.5 La Metáfora del Hilo de Ariadna

Este informe sigue el "hilo de Ariadna" que conecta cada representación de ratios con la visión original. Cada paso —desde CQT hasta BIAS_CONTROL— aborda la pregunta fundamental:

> *"¿Cómo represento un ratio si solo tengo una onda?"*

La respuesta ha evolucionado: en una señal hay múltiples componentes frecuenciales simultáneos:

```
Onda → Espectro → Picos → Ratios entre pares → Representación
```

Lo que ha cambiado es la **representación final**: histogramas globales, histogramas temporales, tokens sparse, o embeddings aprendidos.

---

## 2. Los Orígenes: Del CQT al STFT (Mayo-Junio 2025)

### 2.1 Primera Versión: CQT

El primer intento usó la **Constant-Q Transform (CQT)**, una transformación tiempo-frecuencia con resolución logarítmica en frecuencia (como la percepción humana y las escalas musicales).

**Configuración inicial**:
- 7 octavas × 120 bins/octava = 840 bins frecuenciales
- Detección de picos sobre magnitud promedio
- Cálculo de ratios entre todos los pares de picos
- Almacenamiento en JSON: `sr, peak_freqs, ratios, ratio_hist`

**El problema**: Aunque la resolución era densa (120 bins/semitono), CQT trae inherentemente una "grilla musical". Esto contradecía el objetivo de Premisa 2: eliminar sesgos musicales antropocéntricos.

> *"CQT trae grilla 'musical' (aunque densa). Incompatible con objetivo de 'descentrarse de lo temperado'."*
> — Informe de proceso con ChatGPT

**Valor intrínseco**: Estableció el pipeline básico de extracción de ratios.
**Limitación estratégica**: La escala logarítmica inherente imponía un bias musical.

### 2.2 Transición a STFT (v2.2)

La solución fue cambiar a **Short-Time Fourier Transform (STFT)**, que opera en frecuencia lineal y es "agnóstica" respecto a escalas musicales.

**Cambios clave**:
- STFT multi-resolución: múltiples `n_fft` (8192, 4096, 2048, 1024)
- Interpolación a eje común y promedio
- Escala de ratios: log₂ (rango 1.02-5.0, aproximadamente dos octavas)
- Bins: 100 bins logarítmicos
- Peso de pares: `√(A_i × A_j)` (media geométrica de amplitudes)

**Bug histórico documentado**: El archivo `phi.wav` (con frecuencias en ratio áureo φ ≈ 1.618) detectaba correctamente dos picos (~220 Hz y ~356 Hz) pero el ratio no aparecía en el output. El problema no era la detección de picos sino la lógica de filtrado/construcción del histograma.

**Valor intrínseco**: Pipeline STFT reproducible y generalizable.
**Limitación**: Sin enriquecimiento semántico; bugs en casos edge.

### 2.3 Refinamiento: v3.0 a v3.3

Las versiones 3.x refinaron el extractor con CLI configurable:

| Versión | Bins | Resolución | Notas |
|---------|------|------------|-------|
| v3.0 | 200 | 2.9 cents/bin | Primera versión CLI |
| v3.2 | 400 | 1.5 cents/bin | Mayor resolución |
| v3.3 | 512 | 1.2 cents/bin | Versión del primer commit |

**Primer commit (3 Jun 2025)**: `ac041c4` incluyó:
- Analizador v3.3
- Scripts de verificación (`check_ratios_json`)
- Generador de WAVs sintéticos con ratios conocidos (3:2, 4:3, 5:4, φ, √2)

**Pruebas sintéticas**:
- ✓ Quinta perfecta (3:2): Detectada correctamente
- ✓ Cuarta (4:3): Detectada correctamente
- ✗ Phi (φ): 0 ratios detectados (bug de filtrado)
- ✗ Octava (2:1): Dispersión en histograma

**Valor intrínseco**: Primera suite de validación; CLI configurable.
**Limitación estratégica**: Representación puramente estadística sin semántica.

---

## 3. Consolidación: v4.0 y v4.1 Enriched (Junio-Agosto 2025)

### 3.1 v4.0 — Primera Representación Enriquecida

El commit `54e8b6a` (28 Jun 2025) introdujo el primer salto semántico: de conteos puros a representación enriquecida con 3 canales.

**Innovación clave**: `compute_enriched_histogram()` con:
- Canal 0: Proporción (PDF normalizada)
- Canal 1: Momento energético
- Canal 2: Entropía local

**Output shape**: `[256, 3]` — 256 bins × 3 canales por archivo.

**Valor intrínseco**: Primera captura de "estructura" más allá de frecuencia de aparición.

### 3.2 v4.1 Enriched — La Versión de Referencia

El commit `c2875d0` (11 Ago 2025) estableció v4.1 como el estándar de la era pre-temporal:

**Especificación**:
- **Shape**: `[512, 3]` — 512 bins con resolución de 6.1 cents/bin
- **Escala**: Logarítmica (log₂), centrada en relaciones musicales
- **Los 3 canales**:
  - Canal 0: `prop_b = h_b / Σh_b` (PDF normalizada)
  - Canal 1: `moment_b = h_b × (log2_center)² / Σ` (segundo momento en log₂)
  - Canal 2: `ent_b = -prop_b × log(prop_b) / Σ` (entropía local)

**Significado de la resolución**: 6.1 cents/bin está por debajo del umbral de percepción humano (JND ≈ 5-10 cents), asegurando que cualquier diferencia perceptible se capture.

**VAE v4.1**: El primer modelo entrenado con esta representación:
- Parámetros: 1.6M
- Resultado: `val_loss = 4212` (catastrófico)
- Interpretación: El VAE colapsaba, prediciendo valores conservadores

**Valor intrínseco**: Resolución sub-perceptual; primera captura de semántica más allá de frecuencia.
**Limitación estratégica**: Estático (promedio global pierde evolución temporal); escala log₂ mantiene bias musical.

---

## 4. La Era HRM (Agosto 2025)

### 4.1 HRM: Hierarchical Reasoning Model

Ante el fracaso del VAE, se desarrolló el **HRM** (Hierarchical Reasoning Model), una arquitectura diseñada específicamente para capturar relaciones temporales armónicas.

**Arquitectura**:
```
├── Enhanced CNN Encoder: 64→128→256→384 canales + BatchNorm
├── L-Module: GRU (384 hidden, 3 capas) — Computación rápida
├── H-Module: LSTM (192 hidden) + Multi-head Attention (8 heads)
├── Hierarchical Fusion: Linear layers con ReLU + Dropout
└── Enhanced Decoder: 384→768→1536 hidden layers
```

**Parámetros**: 6.0M (3.7× más que VAE)

### 4.2 Resultados: HRM Domina

| Métrica | Enhanced HRM | Enhanced VAE | Ventaja HRM |
|---------|--------------|--------------|-------------|
| Val Loss | 2.74 | 4212.58 | **99.93% mejor** |
| Loss/M Params | 0.457 | 2573.77 | **5629× más eficiente** |
| Estabilidad | 3.74e-14 | 4.98e-06 | **132,000× más estable** |

**Conclusión de agosto 2025**:
> *"El Enhanced HRM es el claro ganador para análisis de estructura armónica."*

### 4.3 El Espejismo de la Superioridad

Lo que no se entendió en agosto 2025 fue que la victoria del HRM era **ilusoria** — dependía de la representación de datos, no de la arquitectura. Esto solo se revelaría cinco meses después, con el Analizador 5.0.

**Valor intrínseco**: Arquitectura sofisticada con atención multi-head.
**Limitación estratégica**: El problema no era VAE vs HRM sino la representación v4.1.

---

## 5. Revolución del Analizador 5.0 (Enero 2026)

### 5.1 El Cambio de Paradigma

El commit `995cb2a` (13 Ene 2026) marcó el **cambio de paradigma** más importante del proyecto. El Analizador 5.0 introdujo dos cambios fundamentales:

**Cambio 1: De estático a temporal**
```
v4.1: 1 histograma por archivo     [512, 3]
v5.0: T histogramas por archivo    [T, 256, 3]
```

**Cambio 2: De escala log₂ a escala lineal**
```
v4.1: log₂(f₂/f₁) en cents → bias musical
v5.0: f₂/f₁ lineal (1.0-6.0) → proporciones físicas adimensionales
```

**Impacto en datos**: 170× más información por archivo:
```
v4.1: 1 × 512 × 3 = 1,536 valores/archivo
v5.0: 170 × 256 × 3 = 261,888 valores/archivo (promedio T=170 frames)
```

### 5.2 Justificación Científica

El cambio a escala lineal reflejó un retorno a las raíces del proyecto:

> *"Las proporciones físicas son adimensionales. Una ratio 1.5 es 1.5 independientemente de las frecuencias absolutas. La escala logarítmica fue un compromiso con la percepción humana que introdujo el bias que queríamos evitar."*

La temporalidad captura lo que se perdía con promedios globales: la **evolución** de las relaciones armónicas a lo largo del tiempo.

### 5.3 El Descubrimiento Central

Los experimentos con el dataset 5.0 revelaron algo extraordinario:

| Métrica | Analizador 4.1 | Analizador 5.0 | Cambio |
|---------|----------------|----------------|--------|
| HRM val_loss | 2.74 | 0.4607 | **-83.2%** |
| VAE val_loss | 4212.58 | 0.4560 | **-99.99%** |
| Ventaja HRM/VAE | 153,500% | -1.0% | **VAE ahora gana** |

**La revelación**:
> *"El problema no era VAE, sino la representación de datos."*

Con datos 5.0, VAE y HRM son **equivalentes** (diferencia <1%). La superioridad de HRM en agosto 2025 era un artefacto de la representación v4.1, no una propiedad inherente de la arquitectura.

### 5.4 Implicaciones

1. **La representación importa más que la arquitectura**
2. **La escala lineal es fundamental** para VAE
3. **La temporalidad beneficia a ambas arquitecturas**
4. **No existe clara superioridad arquitectónica** con datos óptimos

**Valor intrínseco**: H1 y H2 validadas definitivamente; pipeline temporal establecido.
**Valor estratégico**: Fijó el estándar para histogramas temporales; reveló que el problema de HRM >> VAE era de datos.

---

## 6. Rosetta1: El Primer Intento Cross-Modal (Enero 2026)

### 6.1 El Objetivo: Validar H3

Con H1 y H2 validadas en datos sintéticos, el siguiente paso era **H3: Cross-modality** — demostrar que audio y vibración comparten estructura de ratios.

**Dataset UOEMD**: 128 pares audio-vibración de maquinaria industrial, con condiciones de falla (Healthy, Bearing, etc.).

### 6.2 Rosetta1 v1.0 — El Primer Intento

El primer intento generó histogramas con el mismo pipeline que datos sintéticos, pero los resultados fueron decepcionantes:

**Problema**: Los histogramas de diferentes muestras eran **casi idénticos**:
- Similitud aligned vs shuffled: 0.9541 vs 0.9501 (Δ = 0.4%)
- Entropía: ~97% del máximo (casi uniforme)

### 6.3 Diagnóstico: Explosión Combinatoria

La auditoría reveló la **causa raíz**: explosión combinatoria de ratios.

| Picos detectados | Ratios generados | Resultado |
|------------------|------------------|-----------|
| 50 | 1,225 | Histograma denso |
| 100 | 4,950 | Histograma casi uniforme |
| 200 | 19,900 | Histograma uniforme |

Con señales industriales ruidosas, el detector encontraba 50-200 picos por frame, generando miles de ratios que "llenaban" todos los bins del histograma, eliminando información discriminativa.

### 6.4 RosetaVAE — La Arquitectura

A pesar del problema de representación, se desarrolló **RosetaVAE**, una arquitectura dual-encoder con:
- Encoder de audio → z_shared + z_private_audio
- Encoder de vibración → z_shared + z_private_vib
- Pérdida: Reconstrucción + KL + InfoNCE sobre z_shared

**Parámetros**: 3.16M

**Resultado**: Gap aligned-shuffled = 0.007 (vs objetivo > 0.15) → **NO-GO**

**Valor intrínseco**: Arquitectura elegante para alineación dual-domain.
**Limitación estratégica**: UOEMD demasiado pequeño (128 muestras); representación no discriminativa.

---

## 7. El Revisionismo: Fases 0-3A (Enero-Febrero 2026)

### 7.1 Principio Guía

El fracaso de Rosetta1 2.0 llevó a un "revisionismo" sistemático, guiado por el principio:

> *"No hay modelo salvador si el descriptor no es identificable."*

### 7.2 Fase 0: Tests Sintéticos (GO)

Se creó una suite de tests sintéticos (`synthetic_ratio_suite.py`) para validar que el extractor funciona con señales donde los ratios son conocidos:

- ✓ Test de series armónicas (1:2:3:4:5)
- ✓ Test de no-false-positives (ruido puro)
- ✓ Test de degradación con ruido (SNR 20/10/5 dB)
- ✓ Test de estabilidad temporal

**Resultado**: GO — El extractor funciona correctamente con señales controladas.

### 7.3 Fase 1: Extractor v2.2 (GO)

Se realizó un **sweep de 36 configuraciones** variando:
- Top-K picos: 8, 12, 16
- Prominencia mínima: 0.1, 0.2, 0.3
- Estabilidad temporal: 0.5, 0.7
- TF-IDF: on/off
- Warped bins: on/off

**Configuración óptima**: K=8, prominencia=0.1, estabilidad=0.7

**Resultado**:
| Métrica | v2.0 | v2.2 | Mejora |
|---------|------|------|--------|
| Gap pre-red | 0.004 | 0.691 | **172×** |
| Entropía | 97% | 82% | -15pp |

**Conclusión**: GO — El histograma PUEDE ser discriminativo con extracción adecuada.

### 7.4 Fase 2: Re-entrenamiento (NO-GO)

Con el extractor v2.2 mejorado, se re-entrenó RosetaVAE:

| Métrica | Pre-mejora | Post-mejora | Objetivo |
|---------|------------|-------------|----------|
| Gap pre-red | 0.004 | 0.691 | - |
| Gap post-red | 0.002 | 0.007 | > 0.15 |
| Top-1 | 0.78% | 10.94% | > 10% |

**Interpretación crítica**: El modelo **no capitaliza** la mejora del extractor:
- Gap pre-red mejoró 172×
- Gap post-red mejoró solo 3.5×

**Conclusión**: NO-GO — El problema no es solo el extractor sino algo arquitectural o de hipótesis.

### 7.5 Fase 3A: Constellation Tokens (NO-GO)

Inspirados en Shazam, se probó una representación **sparse** de tokens en lugar de histogramas densos:

```python
token = {
    'log_ratio': np.log2(target.freq / anchor.freq),
    'delta_t': target.time - anchor.time,
    'weight': np.sqrt(anchor.amp * target.amp),
    'anchor_band': get_band_id(anchor.freq),
    'target_band': get_band_id(target.freq)
}
```

**6 configuraciones probadas**:

| Config | Encoder | Decoder | Top-1 |
|--------|---------|---------|-------|
| C1 | MLP | Histogram | 0.78% |
| C2 | MLP | Token | 0.78% |
| C3 | Transformer | Histogram | 0.78% |
| C4 | Transformer | Token | 0.78% |
| C5 | MLP | JEPA-lite | 1.56% |
| C6 | Transformer | JEPA-lite | 0.78% |

**Random baseline**: 0.78% (1/128). **Todos los modelos en nivel random.**

**Diagnóstico**: Los tokens sparse pierden la información de distribución que hace discriminativos a los histogramas densos:
- Histograma v2.2: gap = 0.691
- Constellation: gap = 0.029

**Conclusión**: NO-GO — Shazam funciona para **fingerprinting** (matching exacto), no para **cross-modal alignment** (embedding similarity).

**Valor del revisionismo**: Eliminó hipótesis alternativas. Confirmó que el problema es la **representación**, no la arquitectura.

---

## 8. Escalón 1: MAESTRO Audio↔MIDI (Febrero 2026)

### 8.1 Cambio de Dataset

Ante los NO-GO repetidos con UOEMD (128 muestras), se decidió probar con un dataset **mucho más grande**: MAESTRO v3.0.0.

**MAESTRO**:
- 121 GB de piano (1,276 pares audio-MIDI)
- ~200 horas de música
- Alineación ~3ms (perfecta)
- Metadata: compositor, año, splits oficiales

**Hipótesis**: Con datos suficientes y alineación perfecta, quizás H3 sí funciona.

### 8.2 Extractores Probados

Se probaron dos nuevos extractores diseñados para compatibilidad audio-MIDI:

**Route A — Event-Based** (`event_based_extractor.py`):
- MIDI: eventos directamente del archivo
- Audio: eventos via CQT + onset detection
- Ratio language sobre intervalos semánticos
- ~1,800 tokens/pieza

**Route B — Improved TF** (`improved_tf_extractor.py`):
- Onset anchoring + Harmonic folding + IDF agresivo
- Stoplist 30% de hashes más comunes
- ~52,000 tokens/pieza

### 8.3 Resultados Iniciales (N=10 pares)

| Test | Extractor V2 | Route A | Route B |
|------|--------------|---------|---------|
| Piece Accuracy | 15.5% | **71.4%** | **80.0%** |
| Recall@5 | 50.9% | **100%** | **100%** |
| vs Random | 1.55× | 7.14× | **8.0×** |

**Diagnóstico del problema V2**: El script `diagnose_hash_collision.py` reveló **COLISIÓN GENÉRICA**:
- Overlap aligned: 66.23%
- Overlap random: 65.13%
- Gap: 1.10% → los hashes coincidían pero igual para cualquier par

### 8.4 Análisis de Errores

Con N=20 pares en Route A (26.6% accuracy), el análisis reveló:

**Por tipo de token**:
| Tipo | Descripción | Overlap |
|------|-------------|---------|
| Chord (tipo 1) | Notas simultáneas ±30ms | **72%** |
| Sequential (tipo 2) | Notas consecutivas | 15% |
| Constellation (tipo 3) | Pares lejanos (ΔT>1s) | 5% |

**Causa raíz**: El onset detector de audio tiene resolución limitada (~50-100ms) y pierde notas rápidas, generando intervalos temporales diferentes a MIDI.

### 8.5 Estado del Experimento Piloto

| Hipótesis | Estado |
|-----------|--------|
| H1: Distribuciones compatibles | ✓ Validada |
| H2: Shazam voting funciona | ✓ Validada |
| H3: Cross-modal identification | 🟡 **Resultados prometedores pero N=10 insuficiente** |

**Decisión**: Pausar Escalón 1 para priorizar BIAS_CONTROL (enfoque diferente).

**Valor intrínseco**: 71-80% accuracy en N=10 es señal positiva.
**Limitación**: Muestra piloto demasiado pequeña para conclusiones robustas.

---

## 9. BIAS_CONTROL: El Presente (Febrero 2026)

### 9.1 Cambio de Paradigma

BIAS_CONTROL representa un **cambio fundamental de enfoque**: abandonar matching exacto de hashes por **embeddings aprendidos**.

**Lo que se abandona**:
- Matching exacto de hashes estilo Shazam
- Discretización agresiva (bins de ΔT y log_ratio)
- Criterio de éxito: "tokens Audio = tokens MIDI"

**Lo que se conserva**:
- Insight de ratios: "Las relaciones proporcionales codifican estructura transferible"
- Dataset MAESTRO: Alineación ~3ms, ideal para cross-modal

**Nuevo paradigma**:
> *"Dado un segmento de audio, recuperar el segmento MIDI correspondiente usando DISTANCIA EN EMBEDDING SPACE, superando significativamente el azar con negativos duros."*

### 9.2 Arquitectura BIAS_CONTROL

```
┌─────────────────────────────────────────────────────────┐
│   AUDIO                              MIDI               │
│     │                                  │                │
│     ▼                                  ▼                │
│  ┌──────────┐                    ┌──────────┐          │
│  │   MERT   │                    │   MIDI   │          │
│  │ (frozen) │                    │ Encoder  │          │
│  │  330M    │                    │(Transf.) │          │
│  └────┬─────┘                    └────┬─────┘          │
│       │                               │                 │
│       ▼                               ▼                 │
│  ┌──────────┐                    ┌──────────┐          │
│  │Projection│                    │Projection│          │
│  │  Head    │                    │  Head    │          │
│  └────┬─────┘                    └────┬─────┘          │
│       │                               │                 │
│       └───────────┬───────────────────┘                 │
│                   │                                     │
│                   ▼                                     │
│            ┌────────────┐                               │
│            │  VICReg    │                               │
│            │   Loss     │                               │
│            └────────────┘                               │
│                   │                                     │
│                   ▼                                     │
│            ┌────────────┐                               │
│            │    DANN    │  ← Gate 3 (Gradient Reversal) │
│            │ (optional) │                               │
│            └────────────┘                               │
└─────────────────────────────────────────────────────────┘
```

**Componentes**:
- **MERT** (330M params, frozen): Audio encoder pre-entrenado para música
- **MIDI Encoder**: Transformer sobre piano-roll
- **Projection Heads**: MLP 512→256
- **Loss**: VICReg (Variance-Invariance-Covariance Regularization)

### 9.3 Resultados Gate 2 - COMPLETADO (GO)

**Checkpoint seleccionado**: `checkpoint_epoch45.pt` (de 61 epochs, 1000 batches/epoch)

#### Pool Global (N=13,532)

| Dirección | R@1 | R@10 | vs Random |
|-----------|-----|------|-----------|
| Audio→MIDI | 0.8% | 2.5% | 34× |
| MIDI→Audio | 1.0% | 2.7% | 36× |

#### Pool Estructurado (N=256, 500 queries) - TEST DEFINITIVO

| Dirección | R@1 | R@5 | R@10 | MRR |
|-----------|-----|-----|------|-----|
| Audio→MIDI | 4.4% | 20.8% | **34.4%** | 0.138 |
| MIDI→Audio | 5.2% | 24.6% | **37.6%** | 0.158 |

#### Hard Negative Analysis

| Test | Accuracy |
|------|----------|
| vs Same-Piece-Diff-Time | **80.4%** |
| vs Random | 87.0% |

**Interpretación**: 80.4% accuracy contra hard negatives (misma pieza, distinto tiempo) demuestra que el modelo aprende **identidad temporal**, no solo "firma de pieza".

#### Criterios GO

| Métrica | Umbral GO | Valor | Status |
|---------|-----------|-------|--------|
| Gap (aligned - random) | > 0.15 | **0.478** | ✅ PASS (3.2×) |
| Recall@10 (pool 256) | > 25% | **34.4%** | ✅ PASS (1.4×) |
| Hard Neg Accuracy | > 60% | **80.4%** | ✅ PASS (1.3×) |

**Decisión**: **GO** a Gate 3 (DANN)

### 9.4 Diagnóstico Gate 2.5 - Modal Shortcut

El análisis de embeddings reveló un problema:

| Probe | Resultado | Implicación |
|-------|-----------|-------------|
| Domain Probe (Linear) | **92.7%** | Modal shortcut detectado |
| Silhouette (piece) | -0.111 | Pobre clustering por pieza |
| Dead Dims | 0/256 | Sin colapso dimensional |

**Problema**: Un clasificador lineal puede distinguir si un embedding viene de audio o MIDI con 92.7% accuracy. Esto indica que los embeddings ocupan **regiones separadas** del espacio, y las métricas de retrieval podrían estar infladas por proximidad intra-modal en lugar de alineación cross-modal genuina.

**Solución**: Gate 3 — Domain Adversarial Neural Network (DANN).

### 9.5 Gate 3: DANN — En Ejecución

Para forzar embeddings verdaderamente **modal-agnostic**, se añade un **Gradient Reversal Layer (GRL)** que entrena un clasificador de dominio adversarialmente:

```
Embeddings → GRL → Domain Classifier → ¿Audio o MIDI?
                ↑
         Gradiente invertido (fuerza confusión)
```

**Objetivo**: Reducir domain accuracy de 92.7% → ~50% (chance level), sin perder recall.

#### Smoke Test (Piloto) - GO

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| Gap | **0.477** | Sin degradación vs Gate 2 (0.478) |
| R@10 a2m/m2a | 2.6%/2.5% | Mantiene nivel Gate 2 |
| Domain accuracy | 44.7% | DANN aún no activo (λ=0.00) |
| DANN loss | 0.693 | log(2), esperado al inicio |

**Resultado**: **GO** — Script validado, métricas sin degradación.

#### Training Completo - Epoch 8/30

**Configuración**: 30 epochs, batch=16, segment=4.0s, hop=1.0s, DANN weight=0.01, lambda schedule linear 0→1.

**Progreso (epochs 1-7 completados)**:

| Epoch | Loss | Domain Acc | R@10 (a2m) | Gap | Lambda |
|-------|------|-----------|------------|-----|--------|
| 1 | 14.108 | 67.6% | 6.2% | 0.387 | 0.03 |
| 2 | 14.082 | 74.0% | 5.5% | 0.335 | 0.07 |
| 3 | 14.069 | **77.4%** | 6.6% | 0.398 | 0.10 |
| 4 | 14.048 | 65.0% | 5.2% | 0.378 | 0.13 |
| 5 | 14.031 | 65.2% | 6.1% | 0.367 | 0.17 |
| 6 | 14.025 | 65.8% | 6.8% | 0.386 | 0.20 |
| **7** | **13.992** | **62.7%** | **6.3%** | **0.364** | **0.23** |

**Análisis**:
- **Domain accuracy**: Subió a 77.4% (ep3) y ahora baja consistentemente → 62.7%. La curva es no-monotónica: el DANN necesita fuerza (lambda) para superar la resistencia del clasificador. A lambda=0.23, el GRL está empezando a dominar.
- **R@10**: Estable 5-7%, **2.4× sobre Gate 2 baseline** (2.6%). El DANN no solo no destruye retrieval, sino que parece mejorarlo — posiblemente porque forzar representaciones modal-agnostic elimina features superficiales y obliga al modelo a usar features semánticas.
- **Nuevo best** guardado en epoch 7 (recall=0.073, domain_acc=62.7%).

#### Criterios GO/NO-GO Gate 3

| Métrica | Umbral | Actual (ep7) | Status |
|---------|--------|-------------|--------|
| Domain accuracy | 50% ± 5% | 62.7% | ⏳ Bajando (tendencia OK) |
| Recall@10 (global) | >= 2.6% (Gate 2) | 6.3% | ✅ PASS (2.4×) |
| Recall@10 (pool 256) | >= 34.4% (Gate 2) | Pending | Post-training |
| Hard neg accuracy | >= 80.4% (Gate 2) | Pending | Post-training |

### 9.6 Interpretación

El **momento científico clave** de Gate 2 fue positivo: el modelo SÍ distingue "este segmento de audio a t=30s" de "mismo audio a t=45s" con 80.4% accuracy. Esto confirma **identidad temporal cross-modal**.

El desafío ahora es eliminar el shortcut modal (92.7% separabilidad) sin destruir esta capacidad. Si Gate 3 tiene éxito, los embeddings serán genuinamente **modal-agnostic** y las métricas de retrieval serán irrefutables.

**Valor de BIAS_CONTROL**: Primera evidencia sólida de cross-modality (Gap 0.478, Hard neg 80.4%).
**Progreso**: Gate 2 confirmó la señal; Gate 3 eliminará el artefacto.

---

## 10. Síntesis: El Hilo de Ariadna

### 10.1 Evolución de Representaciones

```
Global → Temporal → Sparse → Embeddings

[512,3]   [T,256,3]   [T,K,5]    z ∈ ℝ²⁵⁶
 v4.1       v5.0     Constell.  BIAS_CTRL
```

Cada transición abordó una limitación:
- **Global → Temporal**: Capturar evolución, no solo estadística
- **Temporal → Sparse**: Reducir explosión combinatoria
- **Sparse → Embeddings**: Permitir matching suave, no exacto

### 10.2 Lecciones Principales

**1. Representación > Arquitectura**

El descubrimiento más importante del proyecto: VAE pasó de "catastrófico" (4212) a "equivalente a HRM" (0.456) con el mismo cambio de representación. La arquitectura importa, pero **la representación importa más**.

**2. Tamaño de Dataset Crítico para H3**

- UOEMD (128 muestras): NO-GO
- MAESTRO N=10: Resultados prometedores
- BIAS_CONTROL (127K segmentos): Primera señal positiva

Para aprender correspondencia cross-modal, se necesitan **miles** de pares, no cientos.

**3. Hard Negatives Necesarios para Validación**

Métricas "fáciles" (cosine alto, vs random) pueden ser engañosas. El test real requiere:
- Negatives de misma pieza/segmento
- Negatives de mismo compositor/estilo
- Pool estructurado, no random

**4. El Valor de los Fracasos**

Cada NO-GO eliminó una hipótesis:
- Rosetta1 v1: El histograma global no discrimina → temporal
- Rosetta1 v2.2: El VAE no capitaliza mejora → problema arquitectural
- Constellations: Tokens sparse pierden información → embeddings

### 10.3 Por Qué Abandonamos Cada Camino

| Camino | Razón de Abandono | Valor Reconocido |
|--------|-------------------|------------------|
| CQT | Bias musical inherente | Estableció pipeline básico |
| v4.1 log₂ | Colapsaba en VAE | Estableció representación 3-canal |
| HRM >> VAE | Era ilusorio (por datos) | Mostró que arquitectura importa menos |
| Rosetta UOEMD | Dataset muy pequeño | Reveló explosión combinatoria |
| Constellations | Sparse pierde info | Confirmó necesidad de representación densa |
| Escalón 1 exacto | N=10 insuficiente | Mostró que onset detection es cuello de botella |

### 10.4 La Visión de Phideus: Estado Actual

| Hipótesis | Estado | Evidencia |
|-----------|--------|-----------|
| H1: Estructura | **VALIDADA** | Distribuciones no aleatorias en todos los datasets |
| H2: Aprendibilidad | **VALIDADA** | val_loss < 0.5 con datos 5.0 |
| H3: Cross-modality | 🟢 **PROMETEDOR** | Gap 0.478, Hard neg acc 80.4%, pendiente DANN |

**La visión original** — que los ratios constituyen un "lenguaje universal" — **está más viva que nunca**. H1 y H2 están firmemente establecidas. H3 tiene evidencia sólida: 80.4% de accuracy contra hard negatives demuestra identidad temporal cross-modal. El paso restante es confirmar que esta señal persiste cuando se elimina el shortcut modal (domain probe 92.7% → ~50% via DANN).

### 10.5 El Momento de Verdad — Parcialmente Respondido

El "momento científico clave" de BIAS_CONTROL era el **hard negative suite**:

> *Si el modelo puede distinguir "este segmento de audio a t=30s" de "mismo audio a t=45s" (hard negative), entonces tenemos evidencia real de cross-modal temporal identity.*

**Resultado**: **SÍ**. 80.4% accuracy contra hard negatives (misma pieza, distinto tiempo). El modelo aprende identidad temporal, no solo "firma de pieza".

Sin embargo, queda una pregunta residual: el 92.7% de separabilidad modal sugiere que parte de la señal podría venir de un "shortcut" — el modelo quizás no necesita entender el contenido musical, sino que usa features estadísticas de la modalidad (audio vs MIDI). **Gate 3 (DANN)** es la respuesta a esta pregunta: si el modelo mantiene 80.4% hard neg accuracy con domain accuracy ~50%, la evidencia será irrefutable.

Todo el hilo de Ariadna — desde CQT hasta DANN — ha convergido en este punto: **¿puede un embedding cross-modal capturar identidad temporal sin depender de shortcut modal?**

---

## 11. Apéndices

### Apéndice A: Tabla Completa de Configuraciones Probadas

#### Extractores

| Config | K | Prom | Stab | TF-IDF | Warped | Gap |
|--------|---|------|------|--------|--------|-----|
| v2.2-01 | 8 | 0.1 | 0.5 | ✗ | ✗ | 0.512 |
| v2.2-02 | 8 | 0.1 | 0.7 | ✗ | ✗ | 0.691 |
| v2.2-03 | 8 | 0.1 | 0.7 | ✓ | ✗ | 0.634 |
| v2.2-04 | 12 | 0.2 | 0.6 | ✗ | ✗ | 0.423 |
| ... | ... | ... | ... | ... | ... | ... |
| **Óptimo** | **8** | **0.1** | **0.7** | **✗** | **✗** | **0.691** |

#### Modelos Constellation

| Config | Encoder | Decoder | Params | Top-1 |
|--------|---------|---------|--------|-------|
| C1 | MLP+Attention | Histogram | 460K | 0.78% |
| C2 | MLP+Attention | Token | 398K | 0.78% |
| C3 | Transformer | Histogram | 523K | 0.78% |
| C4 | Transformer | Token | 461K | 0.78% |
| C5 | MLP+Attention | JEPA-lite | 196K | 1.56% |
| C6 | Transformer | JEPA-lite | 258K | 0.78% |

### Apéndice B: Arquitecturas RNA con Diagramas

#### VAE v4.1 (Catastrófico)
```
Input [512,3] → Conv [64→128→256→384] → Flatten →
μ,σ → z[128] → Decoder [384→512→768→1536] → Output [512,3]
Loss: MSE + βKL (β=1.0)
```

#### HRM (Agosto 2025)
```
Input [512,3] → CNN Encoder → L-Module (GRU) → H-Module (LSTM+Attention)
→ Hierarchical Fusion → Decoder → Output [512,3]
```

#### VAE Temporal (5.0)
```
Input [T,256,3] → Conv1D → GRU → μ,σ → z → GRU → ConvT → Output [T,256,3]
```

#### RosetaVAE (Cross-Modal)
```
Audio [T,256,3] → Encoder_A → z_shared + z_private_A
Vib [T,256,3]   → Encoder_V → z_shared + z_private_V
Loss: Recon_A + Recon_V + KL(z_shared) + KL(z_private) + InfoNCE(z_shared)
```

#### BIAS_CONTROL (Gate 2: VICReg)
```
Audio [waveform] → MERT (frozen 330M) → Proj_A → z_audio
MIDI [piano-roll] → Transformer → Proj_M → z_midi
Loss: VICReg(z_audio, z_midi)
```

#### BIAS_CONTROL + DANN (Gate 3)
```
Audio [waveform] → MERT (frozen 330M) → Proj_A → z_audio ─┐
MIDI [piano-roll] → Transformer → Proj_M → z_midi ─────────┤
                                                             ├→ VICReg Loss
                                                             └→ GRL → Domain Classifier → DANN Loss
```

### Apéndice C: Criterios GO/NO-GO por Experimento

| Experimento | Criterio Principal | Umbral GO | Resultado | Estado |
|-------------|-------------------|-----------|-----------|--------|
| HRM vs VAE (v4.1) | Val Loss | < 10 | 2.74 vs 4212 | HRM "wins" |
| VAE vs HRM (v5.0) | Val Loss | < 0.5 | 0.456 vs 0.461 | Equivalentes |
| Rosetta1 v2.0 | Gap | > 0.15 | 0.007 | NO-GO |
| Extractor v2.2 | Gap pre-red | > 0.05 | 0.691 | GO |
| Rosetta v2.2 | Gap post-red | > 0.15 | 0.007 | NO-GO |
| Constellations | Top-1 | > 15% | 0.78% | NO-GO |
| Route A/B (N=10) | Accuracy | > 50% | 71-80% | GO (piloto) |
| BIAS_CONTROL Gate 2 | Gap (global) | > 0.15 | 0.478 | **GO** ✓ |
| BIAS_CONTROL Gate 2 | Recall@10 (pool 256) | > 25% | **34.4%** | **GO** ✓ |
| BIAS_CONTROL Gate 2 | Hard neg accuracy | > 60% | **80.4%** | **GO** ✓ |
| BIAS_CONTROL Gate 3 | Domain accuracy | ~50% ± 5% | 62.7% (ep7, bajando) | 🔄 ⏳ |
| BIAS_CONTROL Gate 3 | Recall >= Gate 2 | >= 2.6% | **6.3%** (2.4× Gate 2) | 🔄 ✅ |

### Apéndice D: Línea Temporal Visual

```
2025                                                                2026
  |                                                                   |
Jun     Jul     Aug     Sep     Oct     Nov     Dec     Jan     Feb
  |       |       |       |       |       |       |       |       |
  ▼       |       ▼       |       |       |       |       ▼       ▼
v3.3    v4.0    v4.1    [quiet period]            v5.0   BIAS_CTRL
commit         HRM>>VAE                          SHIFT  Gate2→Gate3
ac041c4       c2875d0                          995cb2a  95e80a5

              "HRM wins                       "VAE=HRM"  "Gap 0.478
               99.93%"                         Paradigm   HardNeg 80%
                                               shift     DANN: 62.7%"

────────────────┬────────────────────────────────┬───────────────────
                │                                │
           FOUNDATION                       CROSS-MODAL
           (H1, H2)                         (H3: PROMETEDOR)
```

---

## Referencias

### Documentos Internos

1. `Documents/90_ARCHIVO_GLOBAL/Legacy/Otros_doc_legacy/Casi Paper Phideus.docx.md` — Paper académico fundacional
2. `Documents/90_ARCHIVO_GLOBAL/Legacy/Otros_doc_legacy/Premisas_fundamentales_de_IAm_Phideus.md` — 10 premisas originales
3. `Documents/90_ARCHIVO_GLOBAL/Legacy/Informe de proceso...ratios (PhiDeus).md` — Reconstrucción proceso inicial
4. `Documents/90_ARCHIVO_GLOBAL/Experimentos/REPORTE_COMPARATIVO_4.1_vs_5.0.md` — El cambio de paradigma
5. `Documents/03_FRENTES_CERRADOS/UOEMD/UOEMD_Revisionismo/ROADMAP.md` — Plan del revisionismo
6. `Documents/01_FRENTES_ACTIVOS/ESCALON_1/Plan_implementacion.md` — MAESTRO Audio↔MIDI
7. `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md` — Arquitectura y gates actuales

### Referencias Científicas

1. Krause, B. — Hipótesis del Nicho Acústico
2. Soto-León et al. (2019) — Sincronización theta-alpha
3. VICReg: Variance-Invariance-Covariance Regularization (arXiv:2105.04906)
4. Barlow Twins (arXiv:2103.03230)
5. MERT: Acoustic Music Understanding Model (arXiv:2306.00107)
6. MAESTRO Dataset (magenta.tensorflow.org/datasets/maestro)

---

*Documento generado por Claude Code para el proyecto Phideus*
*Fecha: 2026-02-05*
*Versión: 1.1 — Actualizado con Gate 2 completado (GO) y Gate 3 DANN en ejecución*
