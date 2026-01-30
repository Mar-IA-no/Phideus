# Propuesta Doctoral: Replanteamiento del Pipeline de Extracción de Información Armónica

**Autor**: Claude Code (rol: Investigador Doctoral Senior)
**Fecha**: 2026-01-30
**Contexto**: Post-mortem Rosetta1 2.0 + Auditoría del pipeline de histogramas

---

## Resumen Ejecutivo

Tras revisar exhaustivamente el proyecto Phideus, sus objetivos, los experimentos realizados, y el diagnóstico del problema de histogramas uniformes, presento esta propuesta que responde a tres preguntas fundamentales:

1. **¿Es el enfoque del analizador y los histogramas un buen camino?**
2. **¿Existen formas alternativas de encarar esta parte del pipeline?**
3. **¿Pueden las redes neuronales capturar información de ratios sin preprocesamiento explícito?**

**Conclusión principal**: El problema no es la hipótesis (H3), sino la implementación actual del extractor de características. Propongo un replanteamiento fundamental que va desde una corrección inmediata hasta una visión a largo plazo donde las redes aprenden directamente qué ratios importan.

---

## 1. Diagnóstico: Qué Aprendimos de los Fracasos

### 1.1 El Problema No Es la Hipótesis

La Harmonic Information Theory propone que los ratios de frecuencia constituyen un lenguaje universal cross-modal. Esta es una hipótesis científica elegante con fundamento en:

- Ingeniería mecánica (order tracking)
- Neurociencia (cross-frequency coupling)
- Música (invarianza transpositiva)
- Ecoacústica (partición frecuencial)

**El fracaso de Rosetta1 2.0 no refuta la hipótesis**. Demuestra que:
1. El método de extracción actual no captura información discriminativa
2. La arquitectura VAE+reconstrucción permite shortcuts
3. Los controles negativos (aligned vs shuffled) revelan el problema

### 1.2 La Causa Raíz: Explosión Combinatoria de Ratios

El pipeline actual:

```
Señal → STFT → 50-200 picos/frame → N*(N-1)/2 ratios → Histograma 256 bins
```

Con N=100 picos: **4,950 ratios por frame**.

Distribuidos en 256 bins: ~20 ratios/bin → **distribución casi uniforme**.

El histograma "promedia" toda la información específica, quedando solo la media global del dataset.

### 1.3 Lo Que Sí Funciona

- **H1 validada**: Con datos sintéticos (pocos picos, ratios claros), los histogramas SÍ capturan estructura
- **H2 validada**: Las redes neuronales SÍ pueden aprender representaciones de estos histogramas
- **Analizador 5.0**: Demostró que la representación importa más que la arquitectura

El problema surge específicamente con **señales industriales ruidosas** que tienen demasiados picos.

---

## 2. Pregunta 1: ¿Es el Camino de Histogramas un Buen Enfoque?

### 2.1 Argumentos a Favor

**Ventajas teóricas del histograma de ratios:**

| Propiedad | Beneficio |
|-----------|-----------|
| Invarianza a transposición | Un ratio 3:2 es igual si el fundamental es 100 Hz o 1000 Hz |
| Tamaño fijo | Cualquier señal → tensor [256, 3], permite batching |
| Interpretable | Cada bin corresponde a un ratio específico |
| Domain-agnostic | Aplica a audio, vibración, EEG, etc. |

**Evidencia empírica**: Con datos sintéticos de ratios claros, el enfoque funciona perfectamente.

### 2.2 Argumentos en Contra

**Problemas fundamentales del enfoque actual:**

| Problema | Consecuencia |
|----------|--------------|
| Explosión combinatoria | N picos → N²/2 ratios → histograma uniforme |
| Pérdida de información | Qué picos ESPECÍFICOS generan cada ratio se pierde |
| Sensibilidad al ruido | Picos espurios generan ratios espurios |
| Averaging effect | La información se diluye en la media |

### 2.3 Veredicto: El Histograma Es Viable PERO Requiere Modificación Profunda

El concepto de "representar señales por sus ratios" es correcto. La implementación actual está mal calibrada para señales ruidosas.

**Solución inmediata**: Reducir drásticamente el número de picos antes de calcular ratios.

---

## 3. Propuesta de Solución Inmediata: Analizador Sparse

### 3.1 El Cambio Clave

```
ANTES: Todos los picos → Miles de ratios → Histograma uniforme

DESPUÉS: Top-K picos → K*(K-1)/2 ratios → Histograma discriminativo
```

Con K=10: solo 45 ratios por frame.
Con K=15: solo 105 ratios por frame.

### 3.2 Criterios para Selección de Picos

No basta con "top-K por amplitud". Necesitamos picos que representen la estructura armónica real:

```python
def select_harmonic_peaks(spectrum, K=10):
    """
    Selecciona los K picos más representativos de la estructura armónica.

    Criterios:
    1. Prominencia alta (destaca sobre vecinos)
    2. Separación espectral (no redundantes)
    3. Estabilidad temporal (aparecen en múltiples frames)
    """

    # 1. Detección con prominencia estricta
    peaks = find_peaks(
        spectrum,
        prominence=0.3,        # Mucho más estricto que 0.1
        distance=10,           # Separación mínima en bins
        height=np.percentile(spectrum, 90)  # Solo el top 10%
    )

    # 2. Ordenar por prominencia * amplitud
    scores = prominence[peaks] * spectrum[peaks]

    # 3. Seleccionar top-K
    top_k = np.argsort(scores)[-K:]

    return peaks[top_k]
```

### 3.3 Modificaciones al Analizador

| Parámetro | Valor Actual | Valor Propuesto | Razón |
|-----------|--------------|-----------------|-------|
| Umbral prominencia | 1.25x mediana | 2.0-3.0x mediana | Eliminar picos débiles |
| Máximo picos | Sin límite | K=10-15 | Evitar explosión combinatoria |
| Separación mínima | 0 | 5-10 bins (~50-100 Hz) | Evitar picos redundantes |
| Percentil mínimo | 0 | 80-90 | Solo picos significativos |

### 3.4 Validación Esperada

Con esta modificación, esperamos:

```
Antes:  Entropía histograma ~97%, aligned ≈ shuffled
Después: Entropía histograma ~70%, aligned >> shuffled (Δ > 0.15)
```

**Recomendación**: Implementar y probar ANTES de cambiar arquitectura.

---

## 4. Pregunta 2: ¿Existen Formas Alternativas?

### 4.1 Alternativa A: Representación de Picos Explícita (No Histograma)

En lugar de condensar ratios en histograma, mantener los picos como entidad:

```
Señal → STFT → Top-K picos → [(freq₁, amp₁), (freq₂, amp₂), ...]
```

**Ventajas:**
- No hay pérdida de información por binning
- La red puede aprender qué relaciones importan
- Compatible con arquitecturas de conjuntos (Set Transformer)

**Desventajas:**
- Tamaño variable (requiere padding o pooling)
- Menos interpretable que histograma

### 4.2 Alternativa B: Representación en Grafo

Los picos como nodos, los ratios como aristas:

```
Picos → Nodos con features (freq, amp, width)
Ratios → Aristas con peso (valor del ratio)
```

**Ventajas:**
- Preserva estructura relacional explícita
- GNNs pueden aprender patrones complejos
- Naturalmente invariante a permutación

**Desventajas:**
- Complejidad computacional
- Requiere arquitectura especializada (GNN)

### 4.3 Alternativa C: Scattering Transform

Transformada que es **inherentemente invariante a escala**:

```
Señal → Wavelet Scattering → Coeficientes estables
```

**Ventajas:**
- Invarianza a transposición built-in
- Matemáticamente bien fundamentada
- No requiere detección de picos

**Desventajas:**
- Menos interpretable
- Puede perder información de ratios específicos

### 4.4 Alternativa D: Espectrograma con Eje Log-Frecuencia

```
Señal → STFT → Espectrograma en escala log(freq)
```

En escala logarítmica, los ratios se convierten en **diferencias**:
- log(f₂/f₁) = log(f₂) - log(f₁)

Un ratio de 2:1 (octava) es siempre la misma distancia en el eje log.

**Ventajas:**
- Preserva toda la información espectral
- Ratios = traslaciones (fácil para CNNs)
- Simple de implementar

**Desventajas:**
- Alta dimensionalidad
- La red debe descubrir los ratios

### 4.5 Comparación de Alternativas

| Alternativa | Información | Interpretable | Complejidad | Escalabilidad |
|-------------|-------------|---------------|-------------|---------------|
| Histograma sparse | Media | Alta | Baja | Alta |
| Picos explícitos | Alta | Media | Media | Alta |
| Grafo de ratios | Muy alta | Alta | Alta | Media |
| Scattering | Alta | Baja | Baja | Alta |
| Log-espectrograma | Muy alta | Baja | Baja | Alta |

**Mi recomendación**: Empezar con **Histograma Sparse** (menor cambio), luego explorar **Log-espectrograma + Red** (máxima información).

---

## 5. Pregunta 3: ¿Pueden las Redes Aprender Ratios Sin Preprocesamiento?

### 5.1 Respuesta Corta: Sí, Pero Con Arquitectura Apropiada

Las redes neuronales PUEDEN aprender información de ratios directamente de señales raw o espectrogramas, **si la arquitectura tiene el inductive bias correcto**.

### 5.2 Evidencia de la Literatura

**CNNs en espectrogramas:**
- Los filtros convolucionales pueden aprender detectores de patrones diagonales
- En escala log-frecuencia, estas diagonales corresponden a relaciones armónicas
- Trabajo de Dieleman et al. (2014) mostró que las CNNs aprenden representaciones similares a pitch

**Transformers con positional encoding:**
- La atención computa relaciones entre posiciones
- Con log-frequency positional encoding, estas relaciones son ratios
- wav2vec 2.0 y HuBERT aprenden representaciones que capturan estructura armónica

**Self-Supervised Learning:**
- SimCLR, BYOL, etc. aprenden representaciones invariantes
- Con augmentaciones apropiadas (pitch shift), aprenden invarianza a fundamental frequency
- Esto implica que la red aprende a enfocarse en relaciones, no valores absolutos

### 5.3 El Argumento Teórico

Consideremos qué significa "aprender ratios":

1. **Ratio = relación entre valores**
2. **Atención = computa relaciones entre posiciones**
3. **En escala log, ratio = diferencia**

Un Transformer operando sobre espectrograma log-frequency con positional encoding relativo **naturalmente computa ratios** porque:

```
Attention(Q, K) ∝ exp(Q·K^T)

Si Q y K codifican log(freq):
Q·K^T ∝ log(freq_i) · log(freq_j) ∝ información sobre freq_i/freq_j
```

### 5.4 Propuesta: "Ratio-Aware" Network Sin Extracción Explícita

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Señal Raw                                                     │
│       │                                                         │
│       ▼                                                         │
│   ┌─────────────┐                                               │
│   │   STFT      │  → Espectrograma [T, F]                      │
│   └─────────────┘                                               │
│       │                                                         │
│       ▼                                                         │
│   ┌─────────────┐                                               │
│   │ Log-Freq    │  → Espectrograma en escala log [T, F']       │
│   │ Resample    │     (ratios = traslaciones)                  │
│   └─────────────┘                                               │
│       │                                                         │
│       ▼                                                         │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  Backbone (opciones):                                    │  │
│   │                                                          │  │
│   │  A) CNN con log-freq aware pooling                       │  │
│   │     - Filtros aprenden patrones armónicos                │  │
│   │     - Pooling respeta estructura de ratios               │  │
│   │                                                          │  │
│   │  B) Transformer con relative positional encoding         │  │
│   │     - Atención computa relaciones frecuenciales          │  │
│   │     - Positional encoding en log(freq)                   │  │
│   │                                                          │  │
│   │  C) Scattering + MLP                                     │  │
│   │     - Invarianza a escala built-in                       │  │
│   │     - MLP aprende combinaciones relevantes               │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│       │                                                         │
│       ▼                                                         │
│   ┌─────────────┐                                               │
│   │ Embedding   │  → z ∈ ℝ^d                                   │
│   │ Projection  │     (contiene info de ratios implícita)      │
│   └─────────────┘                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.5 Ventajas del Enfoque End-to-End

| Aspecto | Extracción Explícita | End-to-End |
|---------|---------------------|------------|
| Información | Pérdida en discretización | Toda la información disponible |
| Ratios aprendidos | Solo los pre-definidos | Los que el task requiera |
| Adaptabilidad | Fija | Se adapta al dominio |
| Interpretabilidad | Alta | Requiere análisis post-hoc |
| Ingeniería | Mucha (tuning de parámetros) | Poca (la red decide) |

### 5.6 ¿Cómo Verificar Que la Red Aprendió Ratios?

Técnicas de interpretabilidad:

1. **Probing**: Entrenar clasificador lineal sobre embeddings para predecir ratios conocidos
2. **Attention visualization**: Analizar qué frecuencias atiende a cuáles
3. **Synthetic validation**: Probar con señales sintéticas de ratios conocidos
4. **Ablation de ratios**: Perturbar ratios específicos y medir impacto en embedding

---

## 6. Propuesta Integrada: Roadmap de Tres Fases

### Fase 1: Corrección Inmediata (1-2 semanas)

**Objetivo**: Validar si el problema es solo la cantidad de picos.

**Acciones:**
1. Modificar analizador para usar top-K picos (K=10-15)
2. Regenerar dataset con nuevos parámetros
3. Re-ejecutar Rosetta1 con misma arquitectura
4. Evaluar: ¿aligned >> shuffled ahora?

**Criterio de éxito**:
- Δ(aligned - shuffled) > 0.10
- Entropía histograma < 80%

**Si falla**: El problema no es solo cantidad de picos → ir a Fase 2.
**Si tiene éxito**: La representación de histograma sparse es viable → optimizar y continuar.

### Fase 2: Arquitectura Sin Reconstrucción (2-4 semanas)

**Objetivo**: Eliminar el shortcut de reconstrucción.

**Acciones:**
1. Implementar dual-encoder contrastivo (sin decoder)
2. Usar pérdida InfoNCE pura (o SigLIP)
3. Evaluar por retrieval (no por reconstrucción)
4. Opcional: Agregar predicción JEPA en embedding

**Arquitectura sugerida:**

```
Audio  → Encoder_A → z_A ─┐
                          ├─ InfoNCE Loss
Vibr   → Encoder_V → z_V ─┘

Evaluación: Retrieval Top-K
```

**Criterio de éxito**:
- Retrieval Top-1 > 15% (vs random ~0.8%)
- Retrieval Top-10 > 50%

### Fase 3: Exploración End-to-End (4-8 semanas)

**Objetivo**: Determinar si la extracción explícita de ratios es necesaria.

**Acciones:**
1. Implementar backbone directamente sobre espectrogramas log-freq
2. Comparar: Histograma sparse vs Log-espectrograma raw
3. Analizar qué aprende la red (probing, attention analysis)
4. Determinar el mejor trade-off información/interpretabilidad

**Pregunta de investigación**: ¿Los histogramas de ratios añaden valor sobre espectrogramas raw con arquitectura apropiada?

---

## 7. Recomendación Final: El "Phideus-Native" Approach

### 7.1 La Visión a Largo Plazo

El proyecto Phideus aspira a que los ratios sean un "lenguaje universal". Esto sugiere que:

1. **Los ratios deben EMERGER de los datos**, no ser impuestos
2. **La representación debe ser portable** entre dominios
3. **El sistema debe poder DESCUBRIR** nuevas relaciones

### 7.2 Arquitectura Propuesta: "Ratio Emergence Network"

Un enfoque híbrido donde:
- El **front-end** es flexible (puede ser histograma sparse O espectrograma)
- El **backbone** tiene sesgo hacia relaciones (atención con log-freq encoding)
- El **latent space** tiene estructura interpretable (slots que EMERGEN como ratios)
- El **objetivo** es contrastivo (no reconstruction)

```
┌─────────────────────────────────────────────────────────────────┐
│                   RATIO EMERGENCE NETWORK                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  FRONT-END (por dominio):                                       │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Opción A: Analizador Sparse → Histograma [256, 3]      │    │
│  │ Opción B: STFT + Log-resample → Espectrograma [T, F]   │    │
│  │ Opción C: Scattering Transform → Coeficientes [T, C]   │    │
│  └────────────────────────────────────────────────────────┘    │
│                            │                                    │
│                            ▼                                    │
│  BACKBONE COMPARTIDO:                                           │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Transformer con:                                        │    │
│  │ - Relative positional encoding (log-freq aware)        │    │
│  │ - Cross-attention para integrar información temporal   │    │
│  │ - M slots aprendibles (sin canonizar a 12 ratios)      │    │
│  └────────────────────────────────────────────────────────┘    │
│                            │                                    │
│                            ▼                                    │
│  OUTPUTS:                                                       │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ z_retrieval: Embedding para matching [d]               │    │
│  │ r_slots: Activaciones de ratio-slots [M] (interpretable)│   │
│  └────────────────────────────────────────────────────────┘    │
│                            │                                    │
│                            ▼                                    │
│  OBJETIVOS:                                                     │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ L_contrastive: InfoNCE/SigLIP sobre z_retrieval        │    │
│  │ L_sparsity: Pocos slots activos (estructura simple)    │    │
│  │ L_JEPA: Predicción cross-modal en embedding (opcional) │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 Por Qué Este Enfoque Es "Phideus-Native"

1. **No impone ratios canónicos**: Los slots aprenden qué ratios importan
2. **Es multi-dominio by design**: Agregar dominio = agregar front-end
3. **Permite interpretación**: Los slots son analizables post-training
4. **Es falsificable**: Si los slots NO convergen a patrones de ratio, la hipótesis se refina

---

## 8. Conclusiones

### 8.1 Respuestas a las Preguntas Iniciales

**¿Es el enfoque del analizador y los histogramas un buen camino?**

> **Sí, pero requiere modificación**. El concepto es correcto; la implementación actual calcula demasiados ratios. Con selección de top-K picos, debería funcionar.

**¿Se te ocurre otra forma de encarar esta parte del pipeline?**

> **Sí, varias**. Representación de picos explícita, grafos de ratios, scattering transform, o directamente espectrogramas log-frequency. El trade-off es entre información preservada e interpretabilidad.

**¿Pueden las redes neuronales capturar información de ratios sin preprocesamiento explícito?**

> **Sí, con arquitectura apropiada**. Transformers con log-frequency positional encoding o CNNs en espectrogramas log-freq naturalmente computan relaciones que corresponden a ratios. La extracción explícita puede ser innecesaria si la arquitectura tiene el sesgo correcto.

### 8.2 El Camino Hacia Adelante

1. **Inmediato**: Probar analizador sparse (top-K picos)
2. **Corto plazo**: Eliminar reconstrucción, usar objetivo contrastivo
3. **Mediano plazo**: Comparar histograma vs end-to-end
4. **Largo plazo**: Arquitectura "Ratio Emergence" multi-dominio

### 8.3 Reflexión Final

El fracaso de Rosetta1 2.0 es un **resultado científico valioso**. Nos enseñó que:

- La representación importa más que la arquitectura
- Los controles negativos son esenciales
- El camino hacia H3 requiere más cuidado del esperado

La hipótesis de que "los ratios son un lenguaje universal" sigue siendo elegante y testable. Solo necesitamos encontrar la forma correcta de extraerlos o dejar que emerjan.

> *"La información vive en las relaciones. El desafío es dejar que la red las descubra, no imponérselas."*

---

## Apéndice: Checklist de Implementación

### A. Modificación del Analizador

```python
# Cambios en analizador_roseta.py y analizador_5.0.py

# ANTES
DEFAULT_PEAK_THRESHOLD_FACTOR = 1.25

# DESPUÉS
DEFAULT_PEAK_THRESHOLD_FACTOR = 2.5
DEFAULT_TOP_K_PEAKS = 10
DEFAULT_MIN_PEAK_DISTANCE = 10  # bins
DEFAULT_MIN_PROMINENCE = 0.3
```

### B. Nuevo Criterio de Evaluación

```python
def evaluate_discriminability(dataset):
    """
    Evaluar si los histogramas son discriminativos.

    Criterios:
    - Entropía < 80% del máximo
    - Δ(aligned - shuffled) > 0.10
    - Similitud inter-archivo < 0.90
    """
    # ... implementación
```

### C. Arquitectura Contrastiva (sin decoder)

```python
class DualEncoderContrastive(nn.Module):
    """
    Encoder dual para audio-vibración con objetivo contrastivo.
    Sin decoder = sin shortcut de reconstrucción.
    """

    def __init__(self, input_dim, embed_dim=128):
        self.encoder_audio = Encoder(input_dim, embed_dim)
        self.encoder_vib = Encoder(input_dim, embed_dim)
        self.temperature = 0.07

    def forward(self, audio, vib):
        z_a = F.normalize(self.encoder_audio(audio), dim=-1)
        z_v = F.normalize(self.encoder_vib(vib), dim=-1)
        return z_a, z_v

    def loss(self, z_a, z_v):
        # InfoNCE simétrico
        logits = z_a @ z_v.T / self.temperature
        labels = torch.arange(len(z_a))
        loss_a = F.cross_entropy(logits, labels)
        loss_v = F.cross_entropy(logits.T, labels)
        return (loss_a + loss_v) / 2
```

---

*Documento preparado por Claude Code*
*Rol: Investigador Doctoral Senior en Análisis de Datos y Redes Neuronales*
*Fecha: 2026-01-30*
