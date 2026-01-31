# Roadmap Final: Revisionismo de Extracción de Ratios

**Versión**: Unificada v2 (Claude + GPT5.2Think + Críticas Finales)
**Fecha**: 2026-01-30
**Proyecto**: Phideus / Rosetta
**Objetivo**: Producir evidencia reproducible de cross-modal pairing dependency (aligned ≫ shuffled) bajo protocolo P0. Si se cumple: "H3 supported under P0"

---

## Tabla de Contenidos

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Contexto: Qué Pasó y Por Qué Estamos Aquí](#2-contexto-qué-pasó-y-por-qué-estamos-aquí)
3. [Diagnóstico Consolidado](#3-diagnóstico-consolidado)
4. [Síntesis de Propuestas Analizadas](#4-síntesis-de-propuestas-analizadas)
5. [Protocolo de Evaluación P0 (Congelado)](#5-protocolo-de-evaluación-p0-congelado)
6. [Caminos Posibles - Grupo 1: Pre-análisis](#6-caminos-posibles---grupo-1-pre-análisis)
7. [Caminos Posibles - Grupo 2: Aprendizaje Estructural](#7-caminos-posibles---grupo-2-aprendizaje-estructural)
8. [Especificación Extractor v2.2](#8-especificación-extractor-v22)
9. [Plan Experimental con Sweep](#9-plan-experimental-con-sweep)
10. [Criterios GO/NO-GO Calibrados](#10-criterios-gono-go-calibrados)
11. [Roadmap de Ejecución](#11-roadmap-de-ejecución)
12. [Entregables Requeridos](#12-entregables-requeridos)
13. [Conclusiones y Reflexión Final](#13-conclusiones-y-reflexión-final)
14. [Apéndice A: Glosario](#apéndice-a-glosario)
15. [Apéndice B: Referencias a Documentos Fuente](#apéndice-b-referencias-a-documentos-fuente)
16. [Apéndice C: Críticas Incorporadas (GPT5.2Think Final Review)](#apéndice-c-críticas-incorporadas-gpt52think-final-review)

---

## 1. Resumen Ejecutivo

### El Problema

Rosetta1 2.0 falló en demostrar cross-modality (H3) no por la hipótesis ni la arquitectura, sino porque **la representación de histogramas genera distribuciones casi uniformes** que hacen imposible distinguir pares correctos de pares aleatorios.

**Evidencia cuantitativa del fallo:**
- Entropía del histograma: ~97% (casi uniforme)
- Similitud aligned vs shuffled: 0.9541 vs 0.9501 (Δ = **0.4%**)
- Similitud entre condiciones de falla: >0.98 (indistinguibles)
- Retrieval Top-1: 0.78% ≈ random

### Causa Raíz

Explosión combinatoria: N picos → N*(N-1)/2 ratios

| Picos detectados | Ratios generados | Resultado |
|------------------|------------------|-----------|
| 50 | 1,225 | Histograma denso |
| 100 | 4,950 | Histograma casi uniforme |
| 200 | 19,900 | Histograma uniforme |

Con 256 bins, estos miles de ratios "llenan" todos los bins, perdiendo información discriminativa.

### Principio Guía (Consenso)

> **"No hay modelo salvador si el descriptor no es identificable."**

Antes de iterar arquitectura, debemos **reconstruir la discriminabilidad** del descriptor y blindar la evaluación con controles rigurosos.

### Acción Inmediata

Implementar **Extractor v2.2** con:
1. Selección estricta de picos (Top-K con prominencia)
2. **Estabilidad temporal como núcleo** (no opcional)
3. TF-IDF anti-ubiquidad
4. Sweep de hiperparámetros antes de elegir defaults
5. Protocolo de evaluación P0 congelado con controles anti-shortcut

---

## 2. Contexto: Qué Pasó y Por Qué Estamos Aquí

### 2.1 Historia del Proyecto

| Hito | Fecha | Resultado |
|------|-------|-----------|
| Analizador 5.0 + VAE/HRM | Nov 2025 | ✅ H1 y H2 validadas con datos sintéticos |
| Rosetta1 1.0 | Dic 2025 | ⚠️ cos_sim = 0.766 (prometedor pero sin controles) |
| Rosetta1 2.0 | Ene 2026 | ❌ aligned ≈ shuffled (fallo crítico) |
| Auditoría del pipeline | Ene 2026 | ✅ Causa raíz identificada |

### 2.2 Estado de las Hipótesis

| Hipótesis | Descripción | Estado |
|-----------|-------------|--------|
| **H1 - Estructura** | Las señales contienen distribuciones de ratios estructuradas | ✅ VALIDADA (datos sintéticos) |
| **H2 - Aprendibilidad** | Redes neuronales pueden aprender estas estructuras | ✅ VALIDADA (datos sintéticos) |
| **H3 - Cross-modality** | Audio y vibración comparten estructura de ratios | ❓ NO TESTEADA CORRECTAMENTE |

**Nota importante**: H3 no ha sido "refutada". El test no fue válido porque la representación no permitía discriminar.

### 2.3 Documentos Analizados en Esta Fase

| Documento | Fuente | Contribución Principal |
|-----------|--------|------------------------|
| SPEC_ANALIZADOR_5.0.md | Equipo Phideus | Definición formal del pipeline actual |
| INFORME_AUDITORIA_PIPELINE_HISTOGRAMAS.md | Claude | Diagnóstico cuantitativo del colapso |
| PROPUESTA_DOCTORAL_EXTRACCION_RATIOS.md | Claude | Alternativas y roadmap de fases |
| Informe y propuesta GPT5.2Think.md | GPT5.2 | TF-IDF, constellations, técnicas avanzadas |
| INFORME v2 — Revisionismo | GPT5.2Think | Protocolo P0, calibración, anti-shortcuts |
| INFORME_REVISIONISMO_EXTRACCION_RATIOS.md | Claude | Síntesis inicial y especificación v2.2 |

---

## 3. Diagnóstico Consolidado

### 3.1 Cadena de Fallo

```
Señal Industrial Ruidosa
        │
        ▼
┌───────────────────────┐
│ Detección de Picos    │ → Umbral bajo (1.25×) detecta demasiados picos
│ (sin prominencia)     │ → 50-200 picos por frame
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Cálculo de Ratios     │ → Todos-contra-todos: N*(N-1)/2
│ (todos-contra-todos)  │ → 1,225 a 19,900 ratios por frame
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Histograma 256 bins   │ → Miles de ratios "llenan" todos los bins
│                       │ → Distribución casi uniforme
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Modelo VAE            │ → No hay señal discriminativa
│                       │ → Aprende "histograma promedio"
│                       │ → aligned ≈ shuffled
└───────────────────────┘
```

### 3.2 Evidencia Cuantitativa

| Métrica | Valor Observado | Objetivo Aproximado | Problema |
|---------|-----------------|---------------------|----------|
| Entropía / Max | 97% | < 85% | Casi uniforme |
| Similitud aligned-shuffled | 0.004 (0.4%) | > 5% | No discrimina |
| Similitud inter-condición | 0.98 | < 0.92 | Fallas indistinguibles |
| Correlación con media global | 0.94 | < 0.85 | Todos iguales |
| Retrieval Top-1 | 0.78% | > 10% | = Random |

> **NOTA**: Estos "objetivos aproximados" son útiles como sanity check para saber si estás progresando. La calibración por baselines (Sección 10) es un **control adicional**, no un reemplazo de estos valores.

### 3.3 Por Qué el VAE Parece "Funcionar" Pero No Funciona

El modelo genera outputs que se ven "razonables" pero:
- **variance test**: var(predicción) << var(real) → predice valores conservadores
- **shuffled test**: rendimiento no cae con inputs permutados
- **mean baseline**: rendimiento similar al predictor constante

Esto indica que el modelo aprendió a predecir el histograma promedio del dataset, no la correspondencia específica entre audio y vibración.

---

## 4. Síntesis de Propuestas Analizadas

### 4.1 Tabla Comparativa

| Aspecto | Claude (Auditoría) | Claude (Doctoral) | GPT5.2 (Propuesta) | GPT5.2 (v2 Crítica) |
|---------|-------------------|-------------------|-------------------|---------------------|
| **Diagnóstico** | ★★★★★ | ★★★★☆ | ★★★★★ | ★★★★★ |
| **Solución inmediata** | ★★☆☆☆ | ★★★☆☆ | ★★★★☆ | ★★★★★ |
| **Rigor metodológico** | ★★★☆☆ | ★★★☆☆ | ★★★★☆ | ★★★★★ |
| **Controles estadísticos** | ★★☆☆☆ | ★★☆☆☆ | ★★★☆☆ | ★★★★★ |
| **Factibilidad** | ★★★★★ | ★★★★☆ | ★★★☆☆ | ★★★★☆ |
| **Innovación** | ★★☆☆☆ | ★★★☆☆ | ★★★★★ | ★★★★☆ |

### 4.2 Puntos de Consenso (Todas las fuentes)

1. **El problema es la representación, no la hipótesis H3**
2. **La explosión combinatoria (N²/2 ratios) es la causa raíz**
3. **Reducir número de picos es necesario (top-K)**
4. **Estabilidad temporal es importante**
5. **No iterar arquitectura hasta arreglar la representación**
6. **Controles negativos son obligatorios (aligned vs shuffled)**

### 4.3 Mejoras Clave de GPT5.2 Crítica (Incorporadas)

| Mejora | Original Claude | GPT5.2 v2 | Decisión Final |
|--------|-----------------|-----------|----------------|
| **Lenguaje científico** | "H3 validada" | "H3 supported under protocol P" | ✅ Adoptar GPT5.2 |
| **Unidad de ejemplo** | Frame | Ventana temporal | ✅ Adoptar GPT5.2 |
| **Estabilidad temporal** | Opcional | **CORE (obligatorio)** | ✅ Adoptar GPT5.2 |
| **Umbrales** | Arbitrarios (85%, 0.92) | Calibrados por baselines | ✅ Adoptar GPT5.2 |
| **Sweep** | Defaults fijos | Barrido de 12-20 configs | ✅ Adoptar GPT5.2 |
| **Controles anti-shortcut** | Mencionados | Obligatorios con tests específicos | ✅ Adoptar GPT5.2 |
| **Encoder mínimo** | No mencionado | Antes de redes grandes | ✅ Adoptar GPT5.2 |
| **Binning warped** | No mencionado | Densidad variable 1-2 vs 10+ | ⚠️ Evaluar en sweep |

### 4.4 Posición Unificada Final

1. **Extractor v2.2 con estabilidad temporal como núcleo** (no opcional)
2. **Protocolo P0 congelado** antes de experimentar
3. **Sweep de hiperparámetros** en lugar de defaults arbitrarios
4. **Encoder mínimo para validar aprendibilidad** antes de redes grandes
5. **Criterios calibrados por baselines** (no umbrales mágicos)
6. **Controles anti-shortcut obligatorios** en cada reporte

---

## 5. Protocolo de Evaluación P0 (Congelado)

Este protocolo es **innegociable** y debe aplicarse a TODOS los experimentos futuros.

### 5.1 Definiciones Base

#### Unidad de Ejemplo (CRÍTICO)
```
Ejemplo = Ventana Temporal (NO frame aislado)
├── Duración: 0.5 - 1.5 segundos (configurable, pero FIJO para todo el experimento)
├── Solapamiento: 50% (configurable, pero FIJO)
├── Contenido:
│   ├── audio_descriptor: [n_frames, 256, 3] o descriptor agregado por ventana
│   └── vib_descriptor: [n_frames, 256, 3] o descriptor agregado por ventana
└── pair_id: identificador único (archivo + timestamp)
```

#### Split Anti-Leakage (OBLIGATORIO)
```python
# CORRECTO: GroupSplit por archivo/ejecución/motor
splitter = GroupKFold(n_splits=5)
groups = [file_id for example in dataset]

# INCORRECTO: Random split
# splitter = KFold(n_splits=5)  # ¡PROHIBIDO!
```

**Prohibido**: Que ventanas del mismo archivo aparezcan en train y test.

### 5.2 Tareas Oficiales (Solo Estas Se Reportan)

#### Tarea A: Cross-Modal Retrieval (PRINCIPAL)

```python
def evaluate_retrieval(embeddings_audio, embeddings_vib, aligned_pairs):
    """
    Query: audio → candidatos: todas las vibraciones (y viceversa)

    Métricas obligatorias:
    - Recall@1, Recall@5, Recall@10
    - MRR (Mean Reciprocal Rank)

    Variantes obligatorias:
    - Global: todos los ejemplos de test
    - Intra-condición: candidatos solo de la misma condición (HARDER)
    - Intra-archivo: candidatos solo del mismo archivo (HARDEST)
    """
```

**REGLA CRÍTICA para Intra-archivo Retrieval** (evitar inconsistencias):

El número de candidatos puede variar entre archivos (7 en uno, 100 en otro), lo que cambia el random chance y hace incomparables los resultados.

**Regla**: Usar **N fijo por subsample**, calculado de los datos:
```python
def calculate_intra_file_N(dataset):
    """
    N se calcula como la MEDIANA de candidatos por archivo.
    Esto evita hardcodear un número arbitrario.
    """
    candidates_per_file = [count_candidates(f) for f in dataset.files]
    N = int(np.median(candidates_per_file))
    return max(N, 10)  # Mínimo 10 para que tenga sentido estadístico

def evaluate_intra_file_retrieval(embeddings, file_ids, N=None):
    """
    Si N es None: calcular de los datos (mediana).
    Si un archivo tiene más de N candidatos: subsample aleatorio a N.
    Si un archivo tiene menos de N candidatos: excluir del cálculo.

    IMPORTANTE: Reportar N usado y random chance exacto (1/N).
    """
```

Esto garantiza **comparabilidad** sin hardcodear números arbitrarios.

#### Tarea B: Dependencia del Pairing (TEST DEFINITORIO)

```python
def evaluate_pairing_dependency(model, test_data):
    """
    OBLIGATORIO: Repetir retrieval con pairing SHUFFLED

    Criterio de éxito:
    - El rendimiento DEBE colapsar hacia random
    - Si aligned ≈ shuffled → NO-GO automático
    """
    # Evaluar con pairing correcto
    metrics_aligned = evaluate_retrieval(model, test_data, shuffle=False)

    # Evaluar con pairing permutado
    metrics_shuffled = evaluate_retrieval(model, test_data, shuffle=True)

    # El ratio debe ser >> 1
    ratio = metrics_aligned['recall@1'] / max(metrics_shuffled['recall@1'], 1e-6)

    return {
        'aligned': metrics_aligned,
        'shuffled': metrics_shuffled,
        'ratio': ratio,
        'gap': metrics_aligned['recall@1'] - metrics_shuffled['recall@1']
    }
```

#### Tarea C: Regime Probing (SECUNDARIA)

```python
def evaluate_regime_probe(embeddings, labels):
    """
    Linear probe sobre embedding: Healthy vs Fault

    Métricas:
    - AUC (clasificación binaria)
    - Balanced Accuracy (multiclass si aplica)
    - Silhouette Score (sobre embedding REAL, no UMAP)
    """
```

**Nota sobre reconstrucción**: Solo se reporta si existe módulo predictivo explícito. Si hay decoders, la reconstrucción es AUXILIAR y DEBE acompañarse de controles anti-shortcut.

### 5.3 Controles Anti-Shortcut (OBLIGATORIOS)

#### Para TODAS las tareas:

```python
CONTROLES_OBLIGATORIOS = {
    'aligned_vs_shuffled': {
        'descripcion': 'Comparar rendimiento con pairing correcto vs aleatorio',
        'criterio': 'aligned >> shuffled (factor ≥ 5× o diferencia estadística clara)'
    },

    'random_input_test': {
        'descripcion': 'Evaluar con inputs aleatorios en lugar de reales',
        'criterio': 'El rendimiento DEBE colapsar'
    },

    'mean_baseline': {
        'descripcion': 'Comparar con predictor que siempre devuelve la media',
        'criterio': 'Modelo debe superar significativamente'
    }
}
```

#### Para tareas con decoder/predictor:

```python
CONTROLES_PREDICTIVOS = {
    'variance_test': {
        'descripcion': 'var(predicción) vs var(real)',
        'criterio': 'var(pred) debe ser comparable a var(real)',
        'red_flag': 'var(pred) << var(real) indica predicción conservadora'
    },

    'random_z_test': {
        'descripcion': 'Decodificar con z aleatorio',
        'criterio': 'El output DEBE ser diferente/peor'
    }
}
```

**Regla de Oro**: Si el modelo rinde parecido con inputs aleatorios, el experimento es **NO-GO** aunque el número absoluto sea alto.

### 5.4 Reporte Estadístico (Mínimo)

```yaml
Reporte obligatorio:
  seeds: 5  # mínimo 3 si recursos limitados
  formato: "promedio ± desvío"
  intervalos: "bootstrap sobre ejemplos o seeds"

Información requerida:
  - Hiperparámetros exactos del extractor
  - Hiperparámetros del modelo
  - Tamaño de splits (train/val/test)
  - Número de ejemplos por condición
  - Random chance exacto según N candidatos
```

---

## 6. Caminos Posibles - Grupo 1: Pre-análisis

### Filosofía
> "Extraemos la información de ratios de forma explícita antes de entrenar la red."

### 6.1 Opción 1A: Histograma Sparse v2.2 (RECOMENDADA COMO PRIMER PASO)

**Descripción**: Mantener formato [T, 256, 3] pero con extracción radicalmente diferente.

**Cambios clave**:
- Top-K picos con prominencia (K=8-16)
- **Estabilidad temporal obligatoria**
- TF-IDF anti-ubiquidad

**Ventajas**:
- Compatible con pipeline existente
- Cambios localizados en el extractor
- Validable rápidamente

**Desventajas**:
- Puede no resolver completamente la ubiquidad
- Sigue perdiendo "quién se relaciona con quién"

### 6.2 Opción 1B: Histograma con Binning Warped

**Descripción**: Bins no uniformes con más densidad cerca de ratios 1-2.

```python
def warped_bins(n_bins=256, ratio_min=1.0, ratio_max=10.0, gamma=0.5):
    """
    Bins densos cerca de 1-2 (donde está la física útil)
    Bins anchos hacia 10+ (donde hay menos eventos)

    IMPORTANTE: Usar función SUAVE (potencia o log) en lugar de
    concatenación abrupta que puede crear artefactos en el borde.

    gamma < 1: más densidad cerca de ratio_min
    gamma = 1: lineal (uniforme)
    gamma > 1: más densidad cerca de ratio_max
    """
    # Transformación suave tipo potencia
    t = np.linspace(0, 1, n_bins + 1)  # Parámetro uniforme [0, 1]
    t_warped = t ** gamma               # Warp suave

    # Mapear a rango de ratios
    edges = ratio_min + (ratio_max - ratio_min) * t_warped
    return edges

def warped_bins_log(n_bins=256, ratio_min=1.0, ratio_max=10.0):
    """
    Alternativa: escala logarítmica suave (densidad inversamente
    proporcional al ratio).
    """
    log_min = np.log(ratio_min)
    log_max = np.log(ratio_max)
    log_edges = np.linspace(log_min, log_max, n_bins + 1)
    edges = np.exp(log_edges)
    return edges
```

**Ventajas**:
- Preserva resolución donde importa
- Reduce dispersión en zonas vacías
- **Función suave evita artefactos de borde** (a diferencia de concatenación abrupta)

### 6.3 Opción 1C: Ratio Constellations (Shazam-Style)

**Descripción**: Representación sparse de tokens (log_ratio, delta_t, weight, band_id).

```python
def extract_ratio_constellation(spectrum, K=12, M=4):
    """
    Para cada pico ancla estable, seleccionar M vecinos target.
    Output: tokens discretos, no histograma.
    """
    stable_peaks = get_temporally_stable_peaks(spectrum, K)

    tokens = []
    for anchor in stable_peaks:
        targets = get_local_targets(anchor, M)
        for target in targets:
            tokens.append({
                'log_ratio': np.log2(target.freq / anchor.freq),
                'delta_t': target.time - anchor.time,
                'weight': np.sqrt(anchor.amp * target.amp),
                'band_id': get_band_id(anchor.freq)
            })
    return tokens
```

**Ventajas**:
- Preserva estructura relacional
- Naturalmente sparse
- Excelente para retrieval con hashing

**Desventajas**:
- Requiere nuevo dataloader
- Incompatible con redes actuales

### 6.4 Opción 1D: Multi-Banda Independiente

**Descripción**: Histogramas separados por banda frecuencial.

```python
BANDS = [
    (0, 500),      # Sub-bajo
    (500, 1500),   # Bajo
    (1500, 4000),  # Medio
    (4000, 10000), # Alto
]

# Output: [T, n_bands, bins_per_band, 3]
# Ejemplo: [T, 4, 64, 3] en lugar de [T, 256, 3]
```

**Ventajas**:
- Evita que ruido de una banda "tape" otras
- Análisis por frecuencia

---

## 7. Caminos Posibles - Grupo 2: Aprendizaje Estructural

### Filosofía
> "La red aprende directamente qué relaciones frecuenciales importan, con bias arquitectónico apropiado."

### 7.1 Opción 2A: Log-Spectrogram + CNN

**Descripción**: Alimentar espectrograma en escala log-frecuencia directamente.

```python
class LogFreqCNN(nn.Module):
    """
    En escala log-frecuencia, los ratios son traslaciones.
    CNN puede aprender filtros que detectan patrones de ratio.
    """
    def __init__(self):
        self.log_resample = LogFrequencyResample(n_bins=256)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),   # Patrones locales
            nn.Conv2d(32, 64, (3, 7)),  # Patrones más largos en freq
            # ...
        )
```

**Nota**: Esta opción **no garantiza** que la red aprenda ratios. Requiere probing para verificar.

### 7.2 Opción 2B: Transformer con Log-Freq Positional Encoding

**Descripción**: Atención relativa con encoding en log-frecuencia.

```python
class RatioAwareTransformer(nn.Module):
    """
    La atención computa relaciones entre posiciones.
    Con log-freq encoding, estas relaciones capturan ratios.
    """
    def __init__(self, d_model=128, n_heads=8):
        self.freq_embedding = LogFreqPositionalEncoding(d_model)
        self.transformer = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            relative_attention=True  # Clave
        )
```

**Ventajas**:
- Teóricamente elegante
- Atención relativa captura relaciones naturalmente

### 7.3 Opción 2C: Scattering Transform + MLP

**Descripción**: Wavelet Scattering como front-end invariante a escala.

```python
from kymatio import Scattering1D

class ScatteringEncoder(nn.Module):
    def __init__(self, J=6, Q=8):
        self.scattering = Scattering1D(J=J, Q=Q)  # Invariante a transposición
        self.mlp = MLP(scattering_dim, embed_dim)
```

**Ventajas**:
- Matemáticamente bien fundamentado
- Invarianza a escala built-in
- Robusto a ruido

### 7.4 Opción 2D: PRISM-JEPA (Dual-Encoder + Predictor Latente)

**Descripción**: Encoder con peak-tokens + ratio-slots + predicción en espacio latente (sin decoder).

```python
class PRISM_JEPA(nn.Module):
    """
    - Peak-Set Tokens como input (no histograma)
    - Ratio-Slots aprendibles (M slots, d_model)
    - Objetivo JEPA: predecir embedding cross-modal
    - SIN decoder/reconstrucción (evita shortcut)
    """
    def __init__(self, M_slots=32, d_model=128):
        self.peak_tokenizer = PeakSetTokenizer(K=16)
        self.backbone = SharedProportionBackbone(d_model)
        self.ratio_slots = nn.Parameter(torch.randn(M_slots, d_model))
        self.predictor = CrossModalPredictor(d_model)
        # NO hay decoder
```

**Ventajas**:
- Combina mejor de ambos mundos
- Sin shortcut de reconstrucción
- Slots interpretables

---

## 8. Especificación Extractor v2.2

### 8.1 Parámetros por Defecto (Ajustables en Sweep)

```python
# archivo: analizador_roseta.py

# === ANTES (v2.0 - causó el problema) ===
# DEFAULT_PEAK_THRESHOLD_FACTOR = 1.25
# Sin límite de picos
# Sin prominencia mínima
# Sin estabilidad temporal

# === DESPUÉS (v2.2 - diseño corregido) ===
class ExtractorConfigV22:
    # Selección de picos
    peak_threshold_factor: float = 2.5      # Más estricto
    top_k_peaks: int = 12                   # Límite duro por frame
    min_prominence: float = 0.2             # Prominencia mínima normalizada
    min_peak_distance_hz: float = 50.0      # Separación mínima entre picos

    # Estabilidad temporal (CORE, NO OPCIONAL)
    temporal_window_frames: int = 10        # ~0.5-1s dependiendo de hop
    temporal_stability_threshold: float = 0.6  # 60% de frames
    temporal_freq_tolerance_hz: float = 20.0   # Tolerancia para matching

    # Anti-ubiquidad (recomendado)
    use_tfidf: bool = True

    # Binning (evaluar en sweep)
    use_warped_bins: bool = False           # Evaluar
    n_bins: int = 256
```

### 8.2 Pipeline de Extracción v2.2

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE EXTRACTOR v2.2                          │
└─────────────────────────────────────────────────────────────────────────┘

Señal Raw (Audio o Vibración)
        │
        ▼
┌───────────────────────────────────────┐
│ PASO 1: STFT por ventana              │
│ - n_fft = 4096                        │
│ - hop_length = 1024                   │
│ - Normalización local (z-score/banda) │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│ PASO 2: Peak Picking Robusto          │
│ - score = prominencia × amplitud      │
│ - Umbral = mediana_local × THR (2.5)  │
│ - Prominencia mínima = 0.2            │
│ - Top-K selección (K=12)              │
│ - Distancia mínima = 50 Hz            │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────────────┐
│ PASO 3: ESTABILIDAD TEMPORAL (CORE)                                │
│                                                                    │
│ Para cada pico en frame t:                                        │
│   - Buscar en ventana [t-W, t+W]                                   │
│   - Pico "sobrevive" si aparece en ≥ p% de frames                 │
│   - Tolerancia frecuencial: ±Δf Hz                                 │
│                                                                    │
│ Solo picos ESTABLES pasan al siguiente paso                        │
│                                                                    │
│ Fallback: si sobreviven < K_min picos, relajar prominencia        │
└───────────────────────────────────┬───────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────┐
│ PASO 4: Cálculo de Ratios             │
│ - Solo entre picos estables           │
│ - Con K=12: máximo 66 ratios          │
│ - Peso: w_ij = √(A_i × A_j)          │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│ PASO 5: Histograma 3-Canal            │
│ - Canal 0: Proporción (masa PDF)      │
│ - Canal 1: Momento ponderado          │
│ - Canal 2: Entropía local             │
│ - Binning: uniforme o warped          │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│ PASO 6: TF-IDF (si habilitado)        │
│ - TF: masa del bin en ventana         │
│ - IDF: log(N / df_b) sobre train      │
│ - Objetivo: penalizar bins ubicuos    │
└───────────────────┬───────────────────┘
                    │
                    ▼
Output: [n_windows, 256, 3]
```

### 8.3 Función de Estabilidad Temporal (Pseudocódigo)

```python
def filter_temporally_stable_peaks(
    peaks_per_frame: List[List[Peak]],  # [n_frames, variable]
    window_size: int = 10,
    stability_threshold: float = 0.6,
    freq_tolerance_hz: float = 20.0,
    min_peaks_fallback: int = 3
) -> List[List[Peak]]:
    """
    Filtra picos para mantener solo los temporalmente estables.

    Un pico en frame t es estable si aparece (con tolerancia) en al menos
    stability_threshold * window_size frames de la ventana [t-W/2, t+W/2].

    Returns:
        Lista de picos estables por frame
    """
    stable_peaks_per_frame = []

    for t, frame_peaks in enumerate(peaks_per_frame):
        # Definir ventana temporal
        t_start = max(0, t - window_size // 2)
        t_end = min(len(peaks_per_frame), t + window_size // 2 + 1)
        n_frames_in_window = t_end - t_start

        stable_peaks = []
        for peak in frame_peaks:
            # Contar apariciones en la ventana
            count = 0
            for t_neighbor in range(t_start, t_end):
                neighbor_peaks = peaks_per_frame[t_neighbor]
                if any(abs(p.freq - peak.freq) < freq_tolerance_hz for p in neighbor_peaks):
                    count += 1

            # Verificar estabilidad
            if count / n_frames_in_window >= stability_threshold:
                stable_peaks.append(peak)

        # Fallback si muy pocos picos estables
        if len(stable_peaks) < min_peaks_fallback:
            # Ordenar por score y tomar los top-K originales
            stable_peaks = sorted(frame_peaks, key=lambda p: p.score, reverse=True)
            stable_peaks = stable_peaks[:min_peaks_fallback]

        stable_peaks_per_frame.append(stable_peaks)

    return stable_peaks_per_frame
```

### 8.4 Expectativas de Salida

| Métrica | Rosetta v2.0 | Rosetta v2.2 Objetivo | Criterio |
|---------|--------------|----------------------|----------|
| Picos por frame (antes estabilidad) | 50-200 | 10-15 | Top-K |
| Picos estables | N/A | 5-12 | Estabilidad |
| Ratios por frame | 1,225-19,900 | **15-66** | Reducción 100×+ |
| Entropía histograma | 97% | **< 85%** | Sanity check + baseline |
| Gap aligned-shuffled | 0.4% | **> 5%** | Sanity check + baseline |

> **NOTA**: Estos objetivos son útiles para saber si vas por buen camino. La calibración por baselines es un control adicional que se suma, no reemplaza estos números.

---

## 9. Plan Experimental con Sweep

### 9.1 Sweep de Configuraciones (ANTES de elegir defaults)

**Objetivo**: Encontrar configuración óptima por frontera de Pareto.

```python
SWEEP_GRID = {
    'top_k_peaks': [8, 12, 16],
    'min_prominence': [0.1, 0.2, 0.3],
    'temporal_stability_threshold': [0.5, 0.7],
    'temporal_freq_tolerance_hz': [10, 20, 40],
    'use_warped_bins': [False, True],
    'use_tfidf': [False, True],
}

# Total: 3 × 3 × 2 × 3 × 2 × 2 = 216 combinaciones
# Reducir a 12-20 mediante sampling inteligente o Pareto progresivo
```

### 9.2 Criterios de Pareto para Selección

```python
def pareto_score(config_results):
    """
    Evaluar cada configuración en múltiples objetivos:
    """
    return {
        'max_gap_aligned_shuffled': config_results['gap'],          # Maximizar
        'min_similarity_global_mean': config_results['sim_mean'],   # Minimizar
        'min_entropy_normalized': config_results['entropy'],        # Minimizar
        'stable_peaks_coverage': config_results['coverage'],        # Mantener aceptable
    }

# Seleccionar configuración en frontera de Pareto
# Prioridad: gap_aligned_shuffled > similarity_global_mean > entropy
```

### 9.3 Encoder Mínimo para Validar Aprendibilidad

**Antes de redes grandes**, validar con encoder mínimo:

```python
class MinimalEncoder(nn.Module):
    """
    MLP pequeño o linear probe para medir "aprendibilidad básica"
    antes de invertir en arquitecturas complejas.
    """
    def __init__(self, input_dim=256*3, embed_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )

    def forward(self, x):
        # x: [batch, T, 256, 3] -> flatten -> [batch, T, 768]
        x = x.view(x.size(0), x.size(1), -1)
        return self.encoder(x.mean(dim=1))  # Promedio temporal

# Entrenar con loss contrastivo simple
# Si esto no funciona, escalar no va a ayudar
```

### 9.4 Flujo del Sweep

```
┌─────────────────────────────────────────────────────────────────┐
│                         SWEEP FLOW                               │
└─────────────────────────────────────────────────────────────────┘

1. Generar 12-20 configuraciones del extractor
        │
        ▼
2. Para cada configuración:
   ├── Generar dataset sparse
   ├── Calcular métricas de discriminabilidad (sin red)
   │   ├── Entropía normalizada
   │   ├── Similitud con media global
   │   ├── Gap aligned-shuffled (con encoder mínimo)
   │   └── Cobertura de picos estables
   └── Registrar en sweep_results.json
        │
        ▼
3. Calcular frontera de Pareto
        │
        ▼
4. Seleccionar TOP-3 configuraciones
        │
        ▼
5. Para TOP-3:
   ├── Entrenar encoder mínimo completo (no solo probe)
   ├── Evaluar con protocolo P0 completo
   └── Seleccionar mejor configuración
        │
        ▼
6. Configuración FINAL para Rosetta v2.2
```

---

## 10. Criterios GO/NO-GO Calibrados

### 10.1 Calibración por Baselines (No Umbrales Arbitrarios)

En lugar de "entropía < 85%" o "similitud < 0.92" fijos, calibrar por datos:

```python
BASELINES_OBLIGATORIOS = {
    'random_descriptor': {
        'descripcion': 'Descriptor con bins permutados o ruido con misma energía',
        'uso': 'Establecer piso de métricas'
    },

    'shuffled_pairing': {
        'descripcion': 'Audio emparejado con vibración aleatoria',
        'uso': 'Establecer qué significa "no hay señal"'
    },

    'mean_predictor': {
        'descripcion': 'Predictor que siempre devuelve media del train',
        'uso': 'Establecer qué significa "rendimiento trivial"'
    }
}
```

### 10.2 GO-1: Descriptor Aprendible (Pre-Entrenamiento)

| Criterio | Definición | Umbral |
|----------|------------|--------|
| Gap aligned-shuffled | Diferencia con encoder mínimo | ≥ 5× shuffled o significativo estadísticamente |
| Similitud con media global | Correlación descriptor-media | < P50 de baseline random |
| Cobertura de picos estables | % ventanas con ≥ K_min picos | > 80% |

**Decisión**: Si 2 de 3 criterios son GO → Continuar a entrenamiento completo.

### 10.3 GO-2: Rosetta v2.2 Funciona (Post-Entrenamiento)

| Criterio | Definición | Umbral |
|----------|------------|--------|
| Recall@1 aligned | Retrieval con pairing correcto | ≥ 10× random chance |
| Ratio aligned/shuffled | Degradación con pairing incorrecto | ≥ 5× o caída estadísticamente clara |
| Intra-condición Recall@1 | Retrieval con hard negatives | No colapsa (> 2× random) |
| Separación de regímenes | Silhouette en embedding | > 0.3 |

**Decisión**: Si 3 de 4 criterios son GO → "H3 supported under protocol P0 + extractor v2.2"

### 10.4 NO-GO Automático (Cualquiera de Estos)

```python
NO_GO_CONDITIONS = [
    'aligned ≈ shuffled (gap < 2×)',
    'random_input_test: rendimiento no cae significativamente',
    'mean_baseline: modelo no supera predictor constante',
    'variance_test: var(pred) << var(real) (factor < 0.3)',
    'leakage_detected: splits incorrectos',
]
```

### 10.5 Árbol de Decisiones Completo

```
                    ┌─────────────────────────────┐
                    │ SWEEP: 12-20 configuraciones │
                    │ del extractor v2.2          │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────┴───────────────┐
                    │ Evaluar discriminabilidad   │
                    │ (encoder mínimo + métricas) │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────┴───────────────┐
                    │                             │
                   GO-1                        NO-GO
           (Pareto óptimo)                      │
                    │                             ▼
                    ▼                   ┌─────────────────────┐
        ┌───────────────────┐           │ GRUPO 2:            │
        │ Entrenar Rosetta  │           │ Abandonar histograma│
        │ v2.2 con config   │           │ Ir a log-spec/JEPA  │
        │ seleccionada      │           └─────────────────────┘
        └─────────┬─────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
       GO-2              NO-GO (parcial)
        │                   │
        ▼                   ▼
┌───────────────┐   ┌───────────────────┐
│ H3 SUPPORTED  │   │ Iterar:           │
│ under P0+v2.2 │   │ - TF-IDF toggle   │
│               │   │ - Warped bins     │
│ Documentar    │   │ - Multi-banda     │
│ Siguiente:    │   │ - Constelaciones  │
│ Rosetta v3    │   └─────────┬─────────┘
└───────────────┘             │
                    ┌─────────┴─────────┐
                    │                   │
                   GO                NO-GO
                    │                   │
                    ▼                   ▼
            (volver a GO-2)     ┌───────────────┐
                                │ GRUPO 2:      │
                                │ End-to-end    │
                                │ learning      │
                                └───────────────┘
```

---

## 11. Roadmap de Ejecución

### Fase 0: Preparación (1-2 días)

```
TAREAS (en orden de prioridad):

🔴 PRIMERO - Dataset Sintético (CRÍTICO):
□ CREAR tests/synthetic_ratio_suite.py
  ├── Tests con señales armónicas conocidas (1:2:3:4:5)
  ├── Tests de no-false-positives (ruido puro)
  ├── Tests de degradación con ruido (SNR 20/10/5 dB)
  └── Tests de estabilidad temporal
□ Verificar que el extractor ACTUAL pasa la suite
  (Si no pasa, ya sabés que está roto ANTES de cambiar nada)

Después - Infraestructura:
□ Crear branch: feature/extractor-v22
□ Backup de roseta_full.npz → roseta_v20_backup.npz
□ Verificar UOEMD raw data disponible
□ Definir estructura de sweep_results.json
□ Crear script base: experiments/sweep_extractor.py
□ Crear script: experiments/evaluate_discriminability.py
```

### Fase 1: Sweep de Configuraciones (3-5 días)

```
SEMANA 1 (Primera mitad)
├── Día 1-2: Implementar modificaciones a analizador_roseta.py
│   ├── Agregar argumentos: --top-k-peaks, --min-prominence, etc.
│   ├── Implementar filter_temporally_stable_peaks()
│   ├── Implementar TF-IDF opcional
│   ├── Implementar binning warped opcional
│   └── Tests unitarios con señal sintética
│
├── Día 3-4: Ejecutar sweep
│   ├── Generar 12-20 datasets con configuraciones diferentes
│   ├── Calcular métricas de discriminabilidad para cada uno
│   ├── Entrenar encoder mínimo para gap aligned-shuffled
│   └── Registrar todo en sweep_results.json
│
└── Día 5: Análisis de Pareto y selección
    ├── Visualizar frontera de Pareto
    ├── Seleccionar TOP-3 configuraciones
    └── Decisión GO/NO-GO para Fase 2
```

### Fase 2: Rosetta v2.2 (1 semana)

```
SEMANA 1 (Segunda mitad) + SEMANA 2 (Primera mitad)
├── Día 1: Generar dataset final con config óptima
│   └── roseta_v22_sparse.npz
│
├── Día 2-4: Entrenamiento
│   └── python experiments/run_roseta_experiment.py \
│       --data data/datasets/roseta_v22_sparse.npz \
│       --output data/training_outputs/roseta_v22 \
│       --all-data --epochs 100
│
└── Día 5-6: Evaluación completa con P0
    ├── evaluate_retrieval.py (aligned + shuffled + intra-*)
    ├── evaluate_regime_probe.py
    ├── Controles anti-shortcut (random input, mean baseline)
    └── Compilar reporte con 5 seeds
```

### Fase 3: Decisión y Documentación (2-3 días)

```
SEMANA 2 (Segunda mitad)
├── Si GO-2:
│   ├── Documentar resultados en ROSETTA_V22_RESULTS.md
│   ├── Actualizar Proyecto_Estado_Actual.md
│   ├── Planificar Rosetta v3 (multi-dominio)
│   └── Commit con changelog detallado
│
└── Si NO-GO:
    ├── Documentar qué falló y por qué
    ├── Seleccionar siguiente opción (1B, 1C, 1D, o Grupo 2)
    ├── Actualizar roadmap para siguiente iteración
    └── Definir experimento de Grupo 2 si procede
```

### Fase 4: Exploración de Alternativas (Condicional, 2-4 semanas)

```
SEMANAS 3-4 (si Fase 2 fue NO-GO)
├── Opción A: Iterar Grupo 1
│   ├── 1B: Implementar warped bins si no probado
│   ├── 1C: Implementar ratio constellations
│   └── 1D: Implementar multi-banda
│
└── Opción B: Ir a Grupo 2
    ├── 2A: Log-spectrogram + CNN
    ├── 2B: Transformer con log-freq encoding
    ├── 2C: Scattering transform
    └── 2D: PRISM-JEPA (más ambicioso)
```

---

## 12. Entregables Requeridos

### 12.1 Scripts Nuevos/Modificados

| Script | Propósito | Prioridad | Estado |
|--------|-----------|-----------|--------|
| `tests/synthetic_ratio_suite.py` | **Unit test sintético** | 🔴 PRIMERO | Por crear |
| `src/analizador/analizador_roseta.py` | Modificar con v2.2 params | 🟡 Después | Por implementar |
| `experiments/sweep_extractor.py` | Sweep de configuraciones | 🟡 Después | Por crear |
| `experiments/evaluate_discriminability.py` | Métricas pre-red | 🟡 Después | Por crear |
| `experiments/eval_retrieval_p0.py` | Retrieval con protocolo P0 | 🟢 Fase 2 | Por crear |
| `experiments/eval_regime_probe.py` | Probing de regímenes | 🟢 Fase 2 | Por crear |

### 12.1.1 Dataset Sintético de Refutación (NUEVO - OBLIGATORIO)

**Propósito**: Validar que el extractor funciona correctamente con señales donde los ratios son conocidos. Sin esto, se puede ajustar el extractor a ruido real y romper la coherencia del "lenguaje de ratios".

```python
# tests/synthetic_ratio_suite.py

"""
Suite de tests sintéticos para validar el extractor.

CADA VEZ que se modifica el extractor, este test DEBE pasar.

Señales sintéticas con ratios CONOCIDOS:
- Señales con ratios X (que el extractor DEBE recuperar)
- Sin ratios Y (que el extractor NO DEBE inventar)
- Con niveles de ruido controlado (SNR: 20dB, 10dB, 5dB)
"""

def test_harmonic_series():
    """
    Señal con serie armónica 1:2:3:4:5
    El extractor debe recuperar ratios 2:1, 3:2, 4:3, 5:4, etc.
    """

def test_inharmonic_ratios():
    """
    Señal con ratios específicos (ej: 1.5, 2.5, 3.7)
    El extractor debe recuperarlos con precisión.
    """

def test_no_false_positives():
    """
    Señal con solo frecuencia fundamental (sin armónicos)
    El extractor NO debe inventar ratios espurios.
    """

def test_noise_degradation():
    """
    Señal con ratios conocidos + ruido creciente.
    El extractor debe degradar suavemente (no colapsar abruptamente).

    Criterio: con SNR=10dB, recuperar >80% de ratios verdaderos.
    """

def test_temporal_stability():
    """
    Señal con picos que aparecen/desaparecen aleatoriamente vs
    señal con picos estables.

    El filtro de estabilidad temporal debe distinguirlos.
    """

# Métrica de éxito:
# - Precision: % de ratios detectados que son reales
# - Recall: % de ratios reales que fueron detectados
# - Degradación suave con ruido
```

**Regla**: El extractor NO puede desplegarse si este test falla.

### 12.2 Datasets

| Dataset | Descripción | Estado |
|---------|-------------|--------|
| `roseta_full.npz` | Original v2.0 (backup) | Existe |
| `roseta_v22_sparse.npz` | Config óptima de sweep | Por generar |
| `sweep_configs/*.npz` | Datasets del sweep | Por generar |

### 12.3 Documentación

| Documento | Propósito | Estado |
|-----------|-----------|--------|
| `Documents/Analizador/SPEC_EXTRACTOR_V22.md` | Especificación formal v2.2 | Por crear |
| `Documents/Analizador/PROTOCOL_P0.md` | Protocolo de evaluación congelado | Por crear |
| `Documents/Rosetta_v3/ROSETTA_V22_RESULTS.md` | Resultados con 5 seeds | Por crear |
| `sweep_results.json` | Resultados del sweep | Por generar |

### 12.4 Checklist de Auditoría

```
🔴 PRE-EXTRACTOR (CRÍTICO - hacer PRIMERO)
□ synthetic_ratio_suite.py EXISTE y tiene cobertura completa
□ Test de precisión con señales armónicas conocidas: PASA
□ Test de no-false-positives con ruido puro: PASA
□ Test de degradación suave con SNR decreciente: PASA
□ Si algún test falla → NO continuar hasta arreglarlo

PRE-ENTRENAMIENTO
□ Sweep ejecutado con ≥12 configuraciones
□ Frontera de Pareto calculada
□ Config óptima seleccionada con justificación
□ Gap aligned-shuffled con encoder mínimo: reportado
□ Similitud con media global: reportada
□ Cobertura de picos estables: > 80%

POST-ENTRENAMIENTO
□ 5 seeds ejecutadas (o 3 mínimo con justificación)
□ Protocolo P0 aplicado completamente
□ Retrieval aligned reportado
□ Retrieval shuffled reportado
□ Ratio aligned/shuffled: calculado
□ Random input test: ejecutado
□ Mean baseline test: ejecutado
□ Variance test: ejecutado
□ Intra-condición retrieval: reportado
□ Regime probing: reportado

DOCUMENTACIÓN
□ Hiperparámetros exactos registrados
□ Tamaño de splits reportado
□ Random chance calculado para cada métrica
□ Intervalos de confianza incluidos
□ Decisión GO/NO-GO justificada
```

---

## 13. Conclusiones y Reflexión Final

### 13.1 Diagnóstico Consolidado

El fracaso de Rosetta1 2.0 se debe a un **problema de identificabilidad de la representación**, no a la hipótesis H3 ni a la arquitectura del modelo. La solución requiere:

1. **Reconstruir discriminabilidad** del descriptor
2. **Blindar evaluación** con controles rigurosos
3. **Calibrar criterios** por baselines, no por umbrales arbitrarios

### 13.2 Principio Fundamental

> **"No hay modelo salvador si el descriptor no es identificable."**

Este principio guía todo el roadmap: **primero arreglar la representación, después optimizar arquitectura**.

### 13.3 Lenguaje Científico Correcto

- ❌ "H3 validada" (claim excesivo)
- ✅ "H3 supported under protocol P0 + extractor v2.2" (claim apropiado)

### 13.4 Próximo Paso Inmediato

Implementar **Extractor v2.2** con:
1. Top-K picos con prominencia
2. **Estabilidad temporal obligatoria**
3. Sweep de 12-20 configuraciones
4. Encoder mínimo para validar aprendibilidad
5. Protocolo P0 con controles anti-shortcut

### 13.5 Perspectiva a Largo Plazo

Independientemente del resultado de Rosetta v2.2:
1. **Documentar qué funciona y qué no** con evidencia cuantitativa
2. **Comparar Grupo 1 vs Grupo 2** si el histograma falla
3. **Escalar a multi-dominio** solo después de demostrar cross-modality en 2 dominios

### 13.6 Reflexión Final

> *"La hipótesis H3 no ha sido refutada; simplemente no ha sido testeada correctamente. El primer paso es generar una representación que permita el test, y blindar ese test con controles que impidan declarar victoria prematura."*

Este roadmap establece el marco riguroso para ese test.

---

## Apéndice A: Glosario

| Término | Definición |
|---------|------------|
| **Aligned** | Pares audio-vibración del mismo instante y archivo |
| **Shuffled** | Pares audio-vibración de diferentes archivos (aleatorios) |
| **Identificabilidad** | Capacidad de distinguir descriptores de diferentes fuentes |
| **Estabilidad temporal** | Pico que aparece consistentemente en múltiples frames |
| **Prominencia** | Altura relativa de un pico respecto a sus vecinos |
| **TF-IDF** | Term Frequency × Inverse Document Frequency (penaliza bins ubicuos) |
| **Warped bins** | Bins no uniformes con más densidad en ciertas regiones |
| **Frontera de Pareto** | Conjunto de soluciones óptimas en múltiples objetivos |
| **Encoder mínimo** | Red pequeña para validar aprendibilidad antes de escalar |
| **Protocolo P0** | Especificación congelada de cómo evaluar experimentos |

---

## Apéndice B: Referencias a Documentos Fuente

| Documento | Ubicación | Contribución |
|-----------|-----------|--------------|
| SPEC_ANALIZADOR_5.0.md | Documents/Analizador/Recursos/ | Pipeline actual formal |
| INFORME_AUDITORIA_PIPELINE_HISTOGRAMAS.md | Documents/Analizador/Recursos/ | Diagnóstico cuantitativo |
| PROPUESTA_DOCTORAL_EXTRACCION_RATIOS.md | Documents/Analizador/Recursos/ | Alternativas y fases |
| Informe y propuesta GPT5.2Think.md | Documents/Analizador/Recursos/ | TF-IDF, constellations |
| INFORME v2 — Revisionismo GPT5.2Think | Documents/Analizador/ | Protocolo P0, calibración |
| INFORME_REVISIONISMO_EXTRACCION_RATIOS.md | Documents/Analizador/ | Síntesis inicial Claude |

---

## Apéndice C: Críticas de GPT5.2Think y Evaluación Pragmática

Este documento incorpora críticas de la revisión final, con evaluación de impacto real:

| Crítica | Solución Aplicada | Impacto Real |
|---------|-------------------|--------------|
| **5. Sin unit test sintético** | `synthetic_ratio_suite.py` obligatorio | 🔴 **CRÍTICO** - Implementar PRIMERO |
| **3. Intra-archivo N variable** | N calculado de datos (mediana), no hardcodeado | 🟡 **ÚTIL** - Mejora comparabilidad |
| **4. Warped bins discontinuo** | Función suave (potencia o log) | 🟢 **MENOR** - Solo si usás warped |
| **2. Título "Validar H3"** | Lenguaje más preciso ("H3 supported under P0") | 🟢 **COSMÉTICO** - Para documentación |
| **1. Valores vs calibración** | Valores como "objetivos aproximados" + calibración como control adicional | ⚪ **AJUSTADO** - No reemplazar, complementar |

### Perspectiva Pragmática

**Lo que realmente importa**:
- El dataset sintético es el cambio más valioso - sin él, podés romper todo sin darte cuenta
- La regla de N fijo es buena práctica, pero el número debe salir de los datos

**Lo que es cosmético**:
- El lenguaje "H3 supported" es más preciso pero no cambia qué código escribís
- Los valores numéricos (85%, 5%, etc.) siguen siendo útiles como sanity check

**Lo que se ajustó**:
- Los valores "esperados" no son solo "referencias históricas" - son objetivos aproximados útiles
- La calibración por baselines es un **control adicional**, no un reemplazo

---

*Documento unificado v2 preparado por Claude Code*
*Integrando aportes de: Claude (auditoría, propuesta doctoral, síntesis) + GPT5.2Think (protocolo P0, calibración, anti-shortcuts, críticas finales)*
*Fecha: 2026-01-30*
*Fase: Revisionismo de Extracción de Ratios*
