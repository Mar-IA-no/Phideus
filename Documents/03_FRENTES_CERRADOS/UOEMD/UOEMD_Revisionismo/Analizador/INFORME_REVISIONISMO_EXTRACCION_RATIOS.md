# Informe: Revisionismo de Extracción de Ratios

**Fase del Proyecto**: Revisionismo de Extracción de Ratios
**Fecha**: 2026-01-30
**Rol**: Investigador Doctoral Senior en Análisis de Datos y Redes Neuronales
**Objetivo**: Análisis crítico de propuestas + Roadmap para validación de H3

---

## Tabla de Contenidos

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Contexto y Estado del Arte](#2-contexto-y-estado-del-arte)
3. [Análisis Crítico de Documentos](#3-análisis-crítico-de-documentos)
4. [Síntesis: Puntos de Consenso y Divergencia](#4-síntesis-puntos-de-consenso-y-divergencia)
5. [Propuesta de Caminos - Grupo 1: Pre-análisis](#5-propuesta-de-caminos---grupo-1-pre-análisis)
6. [Propuesta de Caminos - Grupo 2: Aprendizaje Estructural](#6-propuesta-de-caminos---grupo-2-aprendizaje-estructural)
7. [Roadmap de Ejecución](#7-roadmap-de-ejecución)
8. [Especificación Rosetta v2.2](#8-especificación-rosetta-v22)
9. [Criterios GO/NO-GO](#9-criterios-gono-go)
10. [Conclusiones](#10-conclusiones)

---

## 1. Resumen Ejecutivo

### El Problema Central

Rosetta1 2.0 falló en validar H3 (cross-modality) no por la hipótesis ni por la arquitectura, sino porque **la representación de histogramas de ratios genera distribuciones casi uniformes** cuando se aplica a señales industriales ruidosas.

**Causa raíz**: La explosión combinatoria de calcular N*(N-1)/2 ratios entre todos los picos (50-200 por frame) produce ~5,000-20,000 ratios que "llenan" uniformemente los 256 bins del histograma.

**Evidencia clave**:
- Entropía: ~97% del máximo (casi uniforme)
- Similitud aligned vs shuffled: 0.9541 vs 0.9501 (**Δ = 0.4%**)
- Similitud inter-condición: >0.98 (todas las fallas indistinguibles)

### Propuesta de Solución

Se proponen dos grupos de caminos:

| Grupo | Enfoque | Filosofía |
|-------|---------|-----------|
| **Grupo 1** | Pre-análisis de ratio-info | Extraer ratios explícitamente antes de la red |
| **Grupo 2** | Aprendizaje estructural | La red aprende ratio-info con bias arquitectónico |

**Acción inmediata recomendada**: Modificar el analizador actual (Grupo 1, opción mínima) y ejecutar **Rosetta v2.2** para validar si el cambio resuelve el problema de discriminabilidad.

---

## 2. Contexto y Estado del Arte

### 2.1 Documentos Analizados

| Documento | Autor | Fecha | Enfoque Principal |
|-----------|-------|-------|-------------------|
| **SPEC_ANALIZADOR_5.0.md** | Equipo PHIDEUS | Nov 2025 | Especificación formal del pipeline |
| **INFORME_AUDITORIA_PIPELINE_HISTOGRAMAS.md** | Claude | Ene 2026 | Diagnóstico del colapso de histogramas |
| **PROPUESTA_DOCTORAL_EXTRACCION_RATIOS.md** | Claude | Ene 2026 | Alternativas y roadmap de fases |
| **Informe y propuesta GPT5.2Think.md** | GPT5.2 | Ene 2026 | Filtros avanzados y ratio constellations |

### 2.2 La Spec 5.0: Qué Define y Qué Asume

La especificación actual define:
- Histograma de ratios con bins en **escala lineal** (no log2/cents)
- 3 canales enriquecidos: proporción, momento, entropía
- Ratio lineal: r = f_j / f_i con peso w_ij = √(A_i * A_j)
- Formato de salida: `[T, 256, 3]`

**Asunciones implícitas**:
1. La detección de picos produce un número "manejable"
2. Los ratios calculados son informativos
3. El histograma preserva información discriminativa

**Fallo de las asunciones**: Con señales ruidosas (UOEMD), se violan las tres asunciones.

---

## 3. Análisis Crítico de Documentos

### 3.1 Auditoría del Pipeline (Claude)

**Fortalezas:**
- Diagnóstico cuantitativo riguroso
- Identificación precisa de la causa raíz (explosión combinatoria)
- Métricas claras (entropía, gap aligned-shuffled, similitud inter-condición)

**Debilidades:**
- No propone soluciones específicas más allá de "top-K picos"
- No considera filtros adicionales (TF-IDF, estabilidad temporal)

**Veredicto**: Excelente diagnóstico, propuestas de solución superficiales.

### 3.2 Propuesta Doctoral (Claude)

**Fortalezas:**
- Roadmap estructurado en fases
- Considera tanto histograma sparse como end-to-end
- Argumenta que las redes pueden aprender ratios directamente

**Debilidades:**
- Sobreconfianza en que "Transformers aprenden ratios automáticamente"
- Hiperparámetros hardcodeados (distance=10 bins) son frágiles
- Expectativas numéricas (entropía ~70%) son hipótesis no validadas

**Veredicto**: Buen marco conceptual, pero necesita ajustes de implementación.

### 3.3 Propuesta GPT5.2Thinking

**Fortalezas:**
- Propuestas concretas de filtrado (TF-IDF, estabilidad temporal)
- Concepto de "Ratio Constellation" (inspirado en Shazam)
- Énfasis en pares locales (no todos-contra-todos)
- Sugiere técnicas avanzadas (Spectral Kurtosis, Scattering)

**Debilidades:**
- Puede ser excesivamente ambicioso para implementación inmediata
- La representación sparse requiere cambios significativos al pipeline

**Veredicto**: Ideas más innovadoras y mejor fundamentadas, pero mayor complejidad.

### 3.4 Tabla Comparativa de Propuestas

| Aspecto | Claude-Auditoría | Claude-Doctoral | GPT5.2Think |
|---------|------------------|-----------------|-------------|
| **Diagnóstico** | ★★★★★ | ★★★★☆ | ★★★★★ |
| **Solución inmediata** | ★★☆☆☆ | ★★★☆☆ | ★★★★☆ |
| **Solución a largo plazo** | ★☆☆☆☆ | ★★★★☆ | ★★★★★ |
| **Factibilidad** | ★★★★★ | ★★★★☆ | ★★★☆☆ |
| **Innovación** | ★★☆☆☆ | ★★★☆☆ | ★★★★★ |

---

## 4. Síntesis: Puntos de Consenso y Divergencia

### 4.1 Consenso Total

Todos los documentos coinciden en:

1. **El problema es la representación, no la hipótesis H3**
2. **La explosión combinatoria (N²/2 ratios) es la causa raíz**
3. **Reducir número de picos es necesario (top-K)**
4. **El histograma sparse es el primer paso lógico**
5. **No iterar arquitectura hasta arreglar la representación**

### 4.2 Divergencias

| Tema | Claude | GPT5.2Think | Mi Posición |
|------|--------|-------------|-------------|
| ¿Abandonar histograma? | Mantener como opción | Sí, a mediano plazo | **Validar primero sparse** |
| ¿TF-IDF de ratios? | No mencionado | Esencial | **Probar como segunda iteración** |
| ¿Estabilidad temporal? | Mencionado | Esencial | **Incluir desde el inicio** |
| ¿End-to-end viable? | Muy optimista | Cauteloso | **Research track paralelo** |
| ¿Ratio constellations? | No mencionado | Propuesta principal | **Explorar en Grupo 1b** |

### 4.3 Mi Posición Crítica Unificada

1. **El histograma sparse merece una oportunidad justa** antes de abandonarlo
2. **TF-IDF y estabilidad temporal deben incluirse** en la versión sparse
3. **End-to-end es viable pero no garantizado** - requiere investigación seria
4. **Ratio constellations es la mejor idea a mediano plazo** si el histograma falla

---

## 5. Propuesta de Caminos - Grupo 1: Pre-análisis

### Filosofía del Grupo 1
> "Extraemos la información de ratios de forma explícita antes de entrenar la red"

### Opción 1A: Histograma Sparse (Cambio Mínimo)

**Descripción**: Mantener el formato `[T, 256, 3]` pero con modificaciones al proceso de extracción.

**Modificaciones requeridas**:

```python
# Cambios en analizador_roseta.py

# ANTES (valores actuales)
DEFAULT_PEAK_THRESHOLD_FACTOR = 1.25
# Sin límite de picos
# Sin filtro de prominencia
# Sin estabilidad temporal

# DESPUÉS (propuestos)
DEFAULT_PEAK_THRESHOLD_FACTOR = 2.5          # Más estricto
DEFAULT_TOP_K_PEAKS = 12                      # Límite duro
DEFAULT_MIN_PROMINENCE = 0.2                  # Prominencia mínima
DEFAULT_MIN_PEAK_DISTANCE_HZ = 50             # Separación en Hz (no bins)
DEFAULT_TEMPORAL_STABILITY_WINDOW = 5         # Ventana de estabilidad
DEFAULT_TEMPORAL_STABILITY_THRESHOLD = 0.6    # 60% de frames
```

**Ventajas**:
- Mínimos cambios al código existente
- Mantiene compatibilidad con Spec 5.0
- Fácilmente validable

**Desventajas**:
- Puede no resolver el problema de ubiquidad
- Pierde información de "quién genera qué ratio"

**Criterio de éxito**:
- Entropía < 85%
- Gap aligned-shuffled > 5%
- Similitud inter-condición < 0.95

### Opción 1B: Histograma con TF-IDF (Cambio Moderado)

**Descripción**: Agregar ponderación TF-IDF a los bins del histograma.

```python
def apply_tfidf_weighting(histograms, idf_vector):
    """
    TF: masa del bin en el frame
    IDF: log(N_archivos / df_b) donde df_b = archivos donde bin b es activo
    """
    for hist in histograms:
        tf = hist / (hist.sum() + 1e-12)
        hist_weighted = tf * idf_vector
    return hist_weighted
```

**Ventajas**:
- Ataca directamente el problema de ubiquidad
- Mantiene formato `[T, 256, 3]`
- Fundamentado teóricamente

**Desventajas**:
- Requiere pre-cómputo de IDF sobre dataset completo
- Puede penalizar señal útil si es ubicua

### Opción 1C: Ratio Constellations (Cambio Mayor)

**Descripción**: Representación sparse inspirada en audio fingerprinting (Shazam).

```python
def extract_ratio_constellation(spectrum, K=12, M=4):
    """
    Para cada pico ancla, seleccionar M picos target en zona local.
    Output: tokens (log_ratio, delta_t, weight, band_id)
    """
    peaks = select_top_k_peaks(spectrum, K)

    tokens = []
    for anchor in peaks:
        targets = select_local_targets(spectrum, anchor, M)
        for target in targets:
            token = (
                np.log2(target.freq / anchor.freq),  # ratio en log2
                target.time - anchor.time,            # delta temporal
                np.sqrt(anchor.amp * target.amp),     # peso
                get_band_id(anchor.freq)              # banda frecuencial
            )
            tokens.append(token)

    return tokens  # E ≈ K*M tokens por frame
```

**Ventajas**:
- Preserva "quién se relaciona con quién"
- Naturalmente sparse
- Excelente para retrieval

**Desventajas**:
- Formato incompatible con Spec 5.0
- Requiere nuevo dataloader
- Mayor complejidad

### Opción 1D: Multi-Banda (Cambio Estructural)

**Descripción**: Dividir espectro en sub-bandas y generar histograma por banda.

```python
BANDS = [
    (0, 500),      # Sub-bajo
    (500, 1500),   # Bajo
    (1500, 4000),  # Medio
    (4000, 10000), # Alto
]

def extract_multiband_histogram(spectrum, n_bins_per_band=64):
    """
    Output: [T, 4, 64, 3] en lugar de [T, 256, 3]
    """
    histograms = []
    for low, high in BANDS:
        band_spectrum = spectrum[low_bin:high_bin]
        hist = compute_ratio_histogram(band_spectrum, n_bins_per_band)
        histograms.append(hist)
    return np.stack(histograms)
```

**Ventajas**:
- Evita que ruido de una banda "tape" otra
- Permite análisis por frecuencia
- Mantiene interpretabilidad

**Desventajas**:
- Cambia dimensionalidad de salida
- Requiere ajuste de red

---

## 6. Propuesta de Caminos - Grupo 2: Aprendizaje Estructural

### Filosofía del Grupo 2
> "La red aprende directamente qué relaciones frecuenciales importan, con ayuda de bias arquitectónico"

### Opción 2A: Log-Spectrogram + CNN

**Descripción**: Alimentar espectrograma en escala log-frecuencia a una CNN.

```python
class LogFreqCNN(nn.Module):
    """
    En escala log, ratios = traslaciones.
    CNN puede aprender filtros "diagonales" que detectan ratios.
    """
    def __init__(self):
        self.log_resample = LogFrequencyResample(n_bins=256)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),  # Aprende patrones locales
            nn.Conv2d(32, 64, (3, 5)), # Patrones más largos en freq
            ...
        )
```

**Ventajas**:
- Preserva toda la información espectral
- Simple de implementar
- CNNs bien entendidas

**Desventajas**:
- No garantiza que aprenda ratios
- Menos interpretable
- Alta dimensionalidad

### Opción 2B: Transformer con Log-Freq Positional Encoding

**Descripción**: Usar atención con encoding posicional en log-frecuencia.

```python
class RatioAwareTransformer(nn.Module):
    """
    La atención computa relaciones entre posiciones.
    Con log-freq encoding, estas relaciones son ratios.
    """
    def __init__(self, d_model=128, n_heads=8):
        self.freq_embedding = LogFreqPositionalEncoding(d_model)
        self.transformer = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            relative_attention=True  # Clave: atención relativa
        )
```

**Ventajas**:
- Teóricamente elegante
- Atención relativa captura relaciones naturalmente
- Escalable

**Desventajas**:
- Requiere investigación
- No garantizado que aprenda "los ratios correctos"
- Necesita probing para verificar

### Opción 2C: Scattering Transform + MLP

**Descripción**: Usar Wavelet Scattering como front-end estable.

```python
from kymatio import Scattering1D

class ScatteringEncoder(nn.Module):
    """
    Scattering es inherentemente invariante a escala (transposición).
    """
    def __init__(self, J=6, Q=8):
        self.scattering = Scattering1D(J=J, Q=Q)
        self.mlp = MLP(scattering_output_dim, embed_dim)
```

**Ventajas**:
- Matemáticamente bien fundamentado
- Invarianza a escala built-in
- Robusto a ruido

**Desventajas**:
- Menos interpretable
- Puede perder información de ratios específicos
- Requiere librería adicional

### Opción 2D: PRISM-JEPA (Propuesta GPT5.2Pro)

**Descripción**: Encoder con slots aprendibles + predicción en espacio latente.

```python
class PRISM_JEPA(nn.Module):
    """
    - Peak-Set Tokens como input
    - Ratio-Slots aprendibles (no canónicos)
    - Objetivo JEPA: predecir embedding cross-modal
    - Sin decoder/reconstrucción
    """
    def __init__(self, M_slots=32, d_model=128):
        self.peak_tokenizer = PeakSetTokenizer(K=16)
        self.backbone = SharedProportionBackbone(d_model)
        self.ratio_slots = nn.Parameter(torch.randn(M_slots, d_model))
        self.retrieval_head = ProjectionHead(d_model, 128)
        self.jepa_predictor = CrossModalPredictor(d_model)
```

**Ventajas**:
- Combina lo mejor de ambos mundos
- Interpretable via slots
- Sin shortcut de reconstrucción

**Desventajas**:
- Más complejo de implementar
- Requiere más investigación
- Posible over-engineering

---

## 7. Roadmap de Ejecución

### Fase 0: Preparación (1-2 días)

```
□ Crear branch: feature/ratio-extraction-revision
□ Backup de roseta_full.npz actual
□ Implementar script de métricas de discriminabilidad
□ Definir criterios GO/NO-GO cuantitativos
```

### Fase 1: Rosetta v2.2 - Histograma Sparse (1 semana)

```
SEMANA 1
├── Día 1-2: Modificar analizador_roseta.py
│   ├── Implementar TOP_K_PEAKS = 12
│   ├── Implementar MIN_PROMINENCE = 0.2
│   ├── Implementar MIN_PEAK_DISTANCE_HZ = 50
│   └── Implementar TEMPORAL_STABILITY (opcional, fase 1b)
│
├── Día 3: Regenerar dataset
│   └── python analizador_roseta.py --all-data --output roseta_sparse.npz
│
├── Día 4: Validar discriminabilidad (SIN red)
│   ├── Calcular entropía promedio
│   ├── Calcular gap aligned vs shuffled
│   ├── Calcular similitud inter-condición
│   └── Decisión GO/NO-GO para continuar
│
├── Día 5-6: Ejecutar Rosetta v2.2 (si GO)
│   └── python run_roseta_experiment.py --data roseta_sparse.npz --all-data
│
└── Día 7: Evaluación con controles negativos
    ├── evaluate_cross_reconstruction.py --run-all-controls
    ├── evaluate_retrieval.py
    └── Documentar resultados
```

### Fase 1b: Mejoras al Histograma (si Fase 1 parcialmente exitosa)

```
SEMANA 2 (condicional)
├── Implementar TF-IDF weighting
├── Implementar estabilidad temporal estricta
├── Re-evaluar métricas de discriminabilidad
└── Re-ejecutar Rosetta v2.2b si métricas mejoran
```

### Fase 2: Exploración de Alternativas (2-4 semanas)

```
SEMANAS 3-4
├── Grupo 1: Si histograma sparse funcionó
│   ├── Optimizar hiperparámetros (K, prominencia, etc.)
│   ├── Probar arquitectura contrastiva (sin decoder)
│   └── Establecer baseline sólido
│
└── Grupo 2: Si histograma sparse NO funcionó
    ├── Opción 2A: Implementar Log-Spectrogram + CNN
    ├── Opción 2B: Implementar Transformer con log-freq encoding
    └── Comparar con baseline de histograma
```

### Fase 3: Integración y Documentación (1 semana)

```
SEMANA 5
├── Seleccionar mejor enfoque basado en resultados
├── Documentar decisiones y justificaciones
├── Actualizar Spec 6.0 si corresponde
└── Preparar para Rosetta v3 (multi-dominio)
```

---

## 8. Especificación Rosetta v2.2

### 8.1 Cambios al Analizador

```python
# Archivo: analizador_roseta.py
# Sección: PARÁMETROS POR DEFECTO

# --- ANTES (v2.0) ---
DEFAULT_PEAK_THRESHOLD_FACTOR: float = 1.25
DEFAULT_LOCAL_MEDIAN_WINDOW: int = 30
DEFAULT_REL_PEAK_TOL: float = 0.01

# --- DESPUÉS (v2.2) ---
DEFAULT_PEAK_THRESHOLD_FACTOR: float = 2.5       # Más estricto
DEFAULT_LOCAL_MEDIAN_WINDOW: int = 30            # Sin cambio
DEFAULT_REL_PEAK_TOL: float = 0.02               # Más tolerante (evita duplicados cercanos)
DEFAULT_TOP_K_PEAKS: int = 12                    # NUEVO: límite de picos
DEFAULT_MIN_PROMINENCE: float = 0.2              # NUEVO: prominencia mínima
DEFAULT_MIN_PEAK_DISTANCE_HZ: float = 50.0       # NUEVO: separación mínima
```

### 8.2 Función de Selección de Picos (Pseudocódigo)

```python
def select_top_k_peaks_v22(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    K: int = 12,
    threshold_factor: float = 2.5,
    min_prominence: float = 0.2,
    min_distance_hz: float = 50.0,
) -> List[Tuple[float, float]]:
    """
    Selecciona los K picos más representativos de la estructura armónica.

    Returns:
        Lista de (frecuencia_hz, amplitud) ordenada por score descendente
    """
    # 1. Calcular umbral local
    threshold = local_median(spectrum) * threshold_factor

    # 2. Detectar picos con prominencia
    peaks, properties = find_peaks(
        spectrum,
        height=threshold,
        prominence=min_prominence,
        distance=hz_to_bins(min_distance_hz, freqs)
    )

    # 3. Calcular score = prominencia * amplitud
    scores = properties['prominences'] * spectrum[peaks]

    # 4. Ordenar por score descendente
    sorted_indices = np.argsort(scores)[::-1]

    # 5. Seleccionar top-K
    top_k_peaks = peaks[sorted_indices[:K]]

    # 6. Retornar como (freq, amp)
    return [(freqs[p], spectrum[p]) for p in top_k_peaks]
```

### 8.3 Expectativas de Salida

| Métrica | Rosetta v2.0 | Rosetta v2.2 Esperado |
|---------|--------------|----------------------|
| Picos por frame | 50-200 | **10-15** |
| Ratios por frame | 1,225-19,900 | **45-105** |
| Entropía histograma | 97% | **< 85%** |
| Gap aligned-shuffled | 0.4% | **> 5%** |
| Similitud inter-condición | 0.98 | **< 0.92** |

### 8.4 Protocolo de Evaluación

```bash
# 1. Regenerar dataset
python src/analizador/analizador_roseta.py \
    --input-dir data/datasets/UOEMD/raw/2_CSV_Data_Files \
    --output data/datasets/roseta_v22_sparse.npz \
    --top-k-peaks 12 \
    --thr 2.5 \
    --min-prominence 0.2 \
    --min-distance-hz 50

# 2. Validar discriminabilidad (script nuevo)
python experiments/evaluate_discriminability.py \
    --data data/datasets/roseta_v22_sparse.npz

# 3. Si pasa GO: entrenar
python experiments/run_roseta_experiment.py \
    --phase full \
    --data data/datasets/roseta_v22_sparse.npz \
    --output data/training_outputs/roseta_v22 \
    --beta-kl-private 0.01 \
    --dropout-shared 0.5 \
    --lambda-diff 0.1 \
    --epochs 100 \
    --batch-size 8 \
    --all-data

# 4. Evaluación completa con controles
python experiments/evaluate_cross_reconstruction.py \
    --model data/training_outputs/roseta_v22/best_model.pt \
    --data data/datasets/roseta_v22_sparse.npz \
    --run-all-controls

python experiments/evaluate_retrieval.py \
    --model data/training_outputs/roseta_v22/best_model.pt \
    --data data/datasets/roseta_v22_sparse.npz
```

---

## 9. Criterios GO/NO-GO

### 9.1 Fase 1: Discriminabilidad del Descriptor (pre-red)

| Criterio | Umbral GO | Umbral NO-GO | Medición |
|----------|-----------|--------------|----------|
| Entropía promedio | < 85% | > 90% | `entropy / max_entropy` |
| Gap aligned-shuffled (coseno) | > 5% | < 2% | `cos_aligned - cos_shuffled` |
| Similitud inter-condición | < 0.92 | > 0.96 | Promedio matriz similitud |
| Correlación con media global | < 0.85 | > 0.92 | `corr(hist, global_mean)` |

**Decisión**: Si 3 de 4 criterios son GO, continuar a entrenamiento.

### 9.2 Fase 1: Rosetta v2.2 (post-red)

| Criterio | Umbral GO | Umbral NO-GO | Medición |
|----------|-----------|--------------|----------|
| Cross-recon aligned - shuffled | > 0.10 | < 0.03 | `Δcorr` |
| Retrieval Top-1 (N=128) | > 10% | < 3% | `correct / N` |
| Retrieval Top-10 (N=128) | > 40% | < 15% | `correct_in_10 / N` |
| Separación de regímenes | > 0.5 | < 0.2 | Silhouette score |

**Decisión**:
- Si 3 de 4 criterios son GO → **H3 validada con histograma sparse**
- Si 2-3 son NO-GO → Iterar Fase 1b o ir a Grupo 2
- Si 4 son NO-GO → **Problema no es solo cantidad de picos**

### 9.3 Árbol de Decisiones

```
                    ┌─────────────────────────┐
                    │ Fase 1: Discriminabilidad │
                    │ del descriptor (pre-red)  │
                    └───────────┬───────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                   GO                     NO-GO
                    │                       │
                    ▼                       ▼
            ┌───────────────┐       ┌───────────────┐
            │ Entrenar      │       │ Fase 1b:      │
            │ Rosetta v2.2  │       │ TF-IDF +      │
            │               │       │ estabilidad   │
            └───────┬───────┘       └───────┬───────┘
                    │                       │
            ┌───────┴───────┐       ┌───────┴───────┐
            │               │       │               │
           GO            NO-GO     GO            NO-GO
            │               │       │               │
            ▼               ▼       ▼               ▼
    ┌───────────────┐ ┌───────────────┐     ┌───────────────┐
    │ H3 VALIDADA   │ │ Optimizar     │     │ GRUPO 2:      │
    │ con histograma│ │ arquitectura  │     │ Aprendizaje   │
    │ sparse        │ │ (sin decoder) │     │ estructural   │
    └───────────────┘ └───────────────┘     └───────────────┘
```

---

## 10. Conclusiones

### 10.1 Diagnóstico Consolidado

El fracaso de Rosetta1 2.0 se debe a un **problema de representación**, no de hipótesis ni de arquitectura. La solución requiere intervenir en el extractor de características antes de iterar sobre modelos.

### 10.2 Recomendación Inmediata

**Ejecutar Rosetta v2.2 con histograma sparse** siguiendo la especificación de la Sección 8. Este es el cambio mínimo con mayor probabilidad de éxito.

### 10.3 Plan de Contingencia

Si Rosetta v2.2 no valida H3:
1. **Fase 1b**: Agregar TF-IDF + estabilidad temporal
2. **Grupo 2**: Explorar log-spectrogram + red con bias arquitectónico
3. **Último recurso**: Ratio constellations (cambio mayor al pipeline)

### 10.4 Perspectiva a Largo Plazo

Independientemente del resultado de Rosetta v2.2, el proyecto Phideus debería:
1. **Validar empíricamente** si las redes pueden aprender ratio-info directamente
2. **Comparar** histograma sparse vs end-to-end en igualdad de condiciones
3. **Documentar** qué enfoque es más escalable a múltiples dominios

### 10.5 Reflexión Final

> *"La hipótesis H3 no ha sido refutada; simplemente no ha sido testeada correctamente. El primer paso es generar una representación que permita el test."*

Este informe establece el marco para ese test.

---

## Apéndice A: Checklist de Implementación Rosetta v2.2

```
PREPARACIÓN
□ Crear branch feature/rosetta-v22
□ Backup roseta_full.npz → roseta_v20_backup.npz
□ Verificar UOEMD raw data disponible

MODIFICACIÓN ANALIZADOR
□ Agregar argumento --top-k-peaks (default=12)
□ Agregar argumento --min-prominence (default=0.2)
□ Agregar argumento --min-distance-hz (default=50)
□ Modificar process_single_channel() con nueva lógica
□ Agregar logging de picos detectados por frame
□ Test unitario con señal sintética conocida

GENERACIÓN DATASET
□ Ejecutar analizador con nuevos parámetros
□ Verificar shape de salida [T, 256, 3]
□ Verificar metadata incluye nuevos parámetros

VALIDACIÓN DISCRIMINABILIDAD
□ Crear script evaluate_discriminability.py
□ Calcular entropía promedio
□ Calcular gap aligned-shuffled
□ Calcular matriz similitud inter-condición
□ Generar reporte con criterios GO/NO-GO

ENTRENAMIENTO (si GO)
□ Ejecutar run_roseta_experiment.py con --all-data
□ Monitorear convergencia
□ Guardar checkpoints

EVALUACIÓN
□ Ejecutar evaluate_cross_reconstruction.py --run-all-controls
□ Ejecutar evaluate_retrieval.py
□ Compilar resultados en documento

DOCUMENTACIÓN
□ Actualizar ROSETTA1_2.2_RESULTADOS.md
□ Actualizar Proyecto_Estado_Actual.md
□ Commit con mensaje descriptivo
```

---

## Apéndice B: Referencias a Documentos Fuente

| Documento | Sección Relevante |
|-----------|-------------------|
| SPEC_ANALIZADOR_5.0.md | Definición formal del pipeline actual |
| INFORME_AUDITORIA_PIPELINE_HISTOGRAMAS.md | Diagnóstico cuantitativo del colapso |
| PROPUESTA_DOCTORAL_EXTRACCION_RATIOS.md | Alternativas y roadmap de fases |
| Informe y propuesta GPT5.2Think.md | TF-IDF, constellations, técnicas avanzadas |

---

*Documento preparado por Claude Code*
*Rol: Investigador Doctoral Senior en Análisis de Datos y Redes Neuronales*
*Fecha: 2026-01-30*
*Fase: Revisionismo de Extracción de Ratios*
