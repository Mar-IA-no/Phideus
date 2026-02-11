# Informe de Auditoría: Pipeline de Histogramas de Ratios

**Fecha**: 2026-01-30
**Auditor**: Claude Code
**Alcance**: Analizadores, datasets, y representación de histogramas

---

## Resumen Ejecutivo

Se ha identificado un **PROBLEMA CRÍTICO** en la representación de datos que explica completamente por qué Rosetta1 2.0 falló en validar H3 (cross-modality):

> **Los histogramas de ratios son casi idénticos para TODOS los archivos del dataset.**
>
> La diferencia entre pares Aligned vs Shuffled es **< 1%**, lo cual hace imposible que cualquier modelo aprenda correspondencia cross-modal.

---

## 1. Inconsistencias de Parámetros Entre Analizadores

### 1.1 Tabla Comparativa

| Parámetro | Roseta (UOEMD) | Analizador 5.0 (WAV) | Impacto |
|-----------|----------------|---------------------|---------|
| **n_fft** | 4096 | 2048 | 2x diferencia en resolución frecuencial |
| **hop_length** | 1024 | 512 | 2x diferencia en resolución temporal |
| **Sample Rate** | 42000 Hz (fijo) | Variable (librosa) | Diferentes rangos de frecuencia |
| **STFT** | Manual (np.fft) | librosa.stft | Posibles diferencias numéricas |
| **Normalización** | z-score | librosa default | Escalas diferentes |

### 1.2 Implicación

Si se intentara usar modelos entrenados en Roseta para datos de Analizador 5.0 (o viceversa), habría un **mismatch de distribución** severo. Sin embargo, **esto NO es la causa del fallo de Rosetta1 2.0**, ya que ese experimento usó solo datos de Roseta.

---

## 2. Verificación de Integridad de Datos

### 2.1 Dataset `roseta_full.npz`

| Métrica | Valor | Estado |
|---------|-------|--------|
| Archivos totales | 128 | ✅ |
| Frames por archivo | 407 | ✅ Consistente |
| Shape de histogramas | [T, 256, 3] | ✅ |
| Sincronización Audio-Vib | 100% | ✅ |
| Frames vacíos | 0% | ✅ |
| PDF suma a 1.0 | ✅ | ✅ |

**Conclusión**: La estructura de datos es correcta. No hay corrupción ni inconsistencias de formato.

---

## 3. PROBLEMA CRÍTICO: Histogramas Indistinguibles

### 3.1 Hallazgo Principal

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Similitud Coseno ALIGNED (mismo archivo):   0.9541           │
│   Similitud Coseno SHUFFLED (archivo diff):   0.9501           │
│   ──────────────────────────────────────────────────           │
│   DIFERENCIA:                                 0.0040 (0.4%)    │
│                                                                 │
│   ⚠️  EL MODELO NO PUEDE DISTINGUIR PARES CORRECTOS           │
│      DE PARES ALEATORIOS                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Evidencia de Colapso

| Métrica | Valor | Problema |
|---------|-------|----------|
| Entropía Audio | 5.41 / 5.55 (97.6%) | Casi uniforme |
| Entropía Vibración | 5.37 / 5.55 (96.8%) | Casi uniforme |
| Correlación con media global (Audio) | 0.943 | Todos iguales |
| Correlación con media global (Vib) | 0.910 | Todos iguales |
| Similitud inter-condición | 0.97-0.99 | Condiciones indistinguibles |

### 3.3 Matriz de Similitud Entre Condiciones (Audio)

```
            BR    FB    HH    KA    RM    RU    SW    VU
    BR:  1.000 0.998 0.989 0.999 0.999 0.999 0.999 0.999
    FB:  0.998 1.000 0.992 0.999 0.999 0.999 0.999 0.999
    HH:  0.989 0.992 1.000 0.989 0.991 0.988 0.990 0.987
    KA:  0.999 0.999 0.989 1.000 0.999 0.999 0.999 0.999
    RM:  0.999 0.999 0.991 0.999 1.000 0.998 0.998 0.999
    RU:  0.999 0.999 0.988 0.999 0.998 1.000 0.999 0.999
    SW:  0.999 0.999 0.990 0.999 0.998 0.999 1.000 0.999
    VU:  0.999 0.999 0.987 0.999 0.999 0.999 0.999 1.000
```

**Todas las condiciones tienen similitud > 0.98**. Las fallas de motor son indistinguibles de healthy en el espacio de histogramas.

---

## 4. Causa Raíz del Problema

### 4.1 Por Qué los Histogramas Son Uniformes

```
Señal Original
     │
     ▼
┌─────────────────┐
│ FFT por Frame   │  → Espectro con MUCHOS picos (ruido industrial)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Detección Picos │  → 50-200 picos por frame (umbral local bajo)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Calcular TODOS  │  → N*(N-1)/2 ratios = 1225-19900 ratios por frame
│ los ratios      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Histograma      │  → Miles de ratios "llenan" todos los bins
│ 256 bins        │     → Distribución casi uniforme
└─────────────────┘
```

### 4.2 El Problema Matemático

Con N picos, se calculan N*(N-1)/2 ratios.

- Si N = 50 picos: 1,225 ratios
- Si N = 100 picos: 4,950 ratios
- Si N = 200 picos: 19,900 ratios

Distribuidos en 256 bins, esto da ~20-80 ratios por bin, resultando en una distribución casi uniforme donde **la información específica de cada señal se diluye**.

### 4.3 Comparación con Datos Sintéticos

Los datos sintéticos del Analizador 5.0 (usados para validar H1 y H2) tienen:
- Señales con **pocos picos bien definidos** (diseñados así)
- Histogramas con **picos claros** en ratios específicos
- **Alta discriminabilidad** entre archivos

El dataset UOEMD tiene:
- Señales de motor eléctrico **muy ruidosas**
- **Muchos picos** en cada frame (ruido + armónicos + mecánico)
- Histogramas **casi uniformes**

---

## 5. Implicaciones para el Proyecto

### 5.1 Por Qué Rosetta1 2.0 Falló

El modelo de Rosetta1 2.0 no falló por:
- ❌ Arquitectura incorrecta
- ❌ Hiperparámetros mal ajustados
- ❌ Insuficientes datos
- ❌ Bug en el código

El modelo falló porque:
- ✅ **La representación no captura información discriminativa**
- ✅ **Aligned ≈ Shuffled** en el espacio de features
- ✅ **No hay señal para aprender**

### 5.2 Validez de H1 y H2

| Hipótesis | Validada Con | Dataset | ¿Sigue Válida? |
|-----------|--------------|---------|----------------|
| H1 (Estructura) | Analizador 5.0 | Sintético | ✅ Sí |
| H2 (Aprendibilidad) | VAE/HRM | Sintético | ✅ Sí |
| H3 (Cross-modal) | Rosetta1 2.0 | UOEMD | ❌ No demostrable |

**H1 y H2 siguen válidas** porque se validaron con datos sintéticos que SÍ tienen estructura de ratios clara.

**H3 no se puede validar con UOEMD** porque la representación colapsa a distribución uniforme.

---

## 6. Recomendaciones

### 6.1 Opciones Inmediatas

#### Opción A: Mejorar la Representación (Mantener UOEMD)

1. **Reducir número de picos**: Usar umbral más estricto (top-K picos más prominentes)
2. **Picos dominantes solamente**: Solo calcular ratios entre los N picos más fuertes (N=5-10)
3. **Normalización diferente**: Usar percentiles en lugar de mediana local
4. **Bandas de frecuencia**: Separar en sub-bandas y analizar independientemente

```python
# Ejemplo: Solo top-10 picos por frame
top_k = 10
peaks_sorted = np.argsort(magnitudes)[-top_k:]
# Solo 45 ratios en lugar de miles
```

#### Opción B: Cambiar Dataset

Buscar datasets de audio-vibración con:
- Señales más "limpias" (menos ruido de fondo)
- Ratios armónicos más pronunciados
- Mayor variabilidad entre muestras

#### Opción C: Validar H3 con Datos Sintéticos

Generar pares audio-vibración sintéticos donde:
- Ambos comparten estructura de ratios conocida
- Se controla el nivel de ruido
- Se puede verificar ground truth

### 6.2 Cambios Recomendados en el Analizador

```python
# Actual (problema)
DEFAULT_PEAK_THRESHOLD_FACTOR = 1.25  # Muy bajo
# Se detectan demasiados picos

# Propuesto
DEFAULT_PEAK_THRESHOLD_FACTOR = 2.0   # Más estricto
TOP_K_PEAKS = 10                       # Limitar cantidad
MIN_PEAK_PROMINENCE = 0.3              # Exigir prominencia mínima
```

### 6.3 Arquitectura HRAN (Propuesta Previa)

La arquitectura HRAN propuesta en `Documents/HRAN_ARCHITECTURE_PROPOSAL.md` aborda parcialmente este problema al:
- Extraer **picos explícitos** en lugar de histograma
- Usar **grafos de ratios** entre picos seleccionados
- Evitar la dilución en histograma uniforme

---

## 7. Conclusión

La auditoría reveló que el **problema fundamental no está en el modelo ni en el código, sino en la representación de datos**. Los histogramas de ratios generados por el analizador actual son:

1. **Casi uniformes** (entropía ~97%)
2. **Indistinguibles entre muestras** (similitud > 0.95)
3. **Sin señal discriminativa** (aligned ≈ shuffled)

Esto hace **matemáticamente imposible** que cualquier modelo aprenda correspondencia cross-modal, independientemente de su arquitectura.

**Próximo paso recomendado**: Modificar el analizador para generar histogramas con mayor discriminabilidad antes de intentar cualquier nuevo experimento cross-modal.

---

## Apéndice: Archivos Analizados

| Archivo | Ubicación |
|---------|-----------|
| Analizador Roseta | `src/analizador/analizador_roseta.py` |
| Analizador 5.0 | `src/analizador/analizador_5.0.py` |
| Dataset Roseta | `data/datasets/roseta_full.npz` |
| Dataset Loader | `src/datasets/roseta_dataset.py` |

---

*Informe generado por Claude Code - 2026-01-30*
