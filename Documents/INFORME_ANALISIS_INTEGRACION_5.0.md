# Informe Técnico: Análisis de Integración del Analizador Phideus 5.0

**Fecha**: 13 de enero de 2026
**Contexto**: Evaluación de estrategias para integrar el formato temporal del Analizador 5.0 al pipeline de entrenamiento neuronal
**Autor**: Análisis colaborativo Claude/Usuario

---

## 1. Resumen Ejecutivo

El Analizador 5.0 introduce un cambio paradigmático: de histogramas **estáticos** (un promedio global por archivo) a histogramas **temporales** (una secuencia de frames por archivo). Este cambio incrementa el volumen de datos en **170x**, generando desafíos técnicos pero también oportunidades científicas significativas.

Este informe analiza las opciones disponibles desde perspectivas técnica, científica y práctica, concluyendo con una recomendación fundamentada.

---

## 2. Diagnóstico Técnico

### 2.1 Comparación de Formatos

| Aspecto | Analizador 4.1 | Analizador 5.0 |
|---------|----------------|----------------|
| **Tipo** | Estático (global) | Temporal (frame-by-frame) |
| **Shape por archivo** | (512, 3) | (341, 256, 3)* |
| **Valores por archivo** | 1,536 | 261,888 |
| **Factor de incremento** | 1x | **170x** |
| **Escala de ratios** | log₂ / cents | Lineal (física) |
| **Bins default** | 512 | 256 |

*Para audios de ~4 segundos con hop=512

### 2.2 Proyección de Recursos (848 archivos)

| Métrica | JSON | Binario (float32) |
|---------|------|-------------------|
| **Tamaño en disco** | 10.3 GB | 0.8 GB |
| **RAM para carga completa** | ~12 GB | 0.8 GB |
| **Tiempo de carga** | ~2-5 min | ~5-10 seg |
| **Bytes por valor** | 49.6 | 4.0 |

**Hallazgo clave**: El problema no es la cantidad de datos, sino el formato JSON. En binario, el dataset completo cabe cómodamente en RAM.

---

## 3. Análisis de Opciones

### Opción A: Reducir Resolución Temporal (Mayor Hop)

**Descripción**: Aumentar `hop_length` de 512 a 1024, 2048, o 4096.

| Hop | Frames/archivo | Reducción | Tamaño resultante |
|-----|----------------|-----------|-------------------|
| 512 (actual) | 341 | 1.0x | 10.3 GB |
| 1024 | 170 | 2.0x | 5.2 GB |
| 2048 | 85 | 4.0x | 2.6 GB |
| 4096 | 42 | 8.0x | 1.3 GB |

**Pros**:
- Reducción lineal y predecible del tamaño
- Mantiene resolución espectral (n_fft intacto)
- Implementación trivial (solo cambiar parámetro)

**Contras**:
- Pérdida de resolución temporal
- Con hop=4096, un frame cubre ~93ms - eventos rápidos (<100ms) se pierden
- Viola el principio de diseño del 5.0: capturar dinámica temporal fina

**Impacto científico**: Moderado-Alto. La resolución temporal es precisamente lo que diferencia al 5.0 del 4.1. Reducirla excesivamente anula la ventaja.

---

### Opción B: Reducir Resolución de Ratios (Menos Bins)

**Descripción**: Reducir bins de 256 a 128 o 64.

| Bins | Resolución | Reducción | Tamaño resultante |
|------|------------|-----------|-------------------|
| 256 (actual) | 6.1 cents/bin | 1.0x | 10.3 GB |
| 128 | 12.2 cents/bin | 2.0x | 5.2 GB |
| 64 | 24.4 cents/bin | 4.0x | 2.6 GB |

**Pros**:
- Reducción lineal del tamaño
- Simplifica el espacio latente
- Puede reducir overfitting

**Contras**:
- El 4.1 usó 512 bins - comparación no directa
- 128 bins aún captura semitonos (100 cents), pero pierde microtonos
- 64 bins es muy grueso para análisis armónico fino

**Impacto científico**: Moderado. Los ratios armónicos principales (octava, quinta, cuarta) tienen separaciones de >100 cents, así que 128 bins debería ser suficiente para la mayoría de análisis.

---

### Opción C: Cambiar a Formato Binario

**Descripción**: Guardar datos en numpy (.npz) o HDF5 en lugar de JSON.

**Pros**:
- Reducción de 12x en tamaño (49.6 → 4 bytes/valor)
- Carga 10-50x más rápida
- Estándar en machine learning científico
- Compatible con PyTorch/TensorFlow sin conversión

**Contras**:
- Requiere modificar el analizador 5.0
- Menos legible para inspección humana
- Necesita actualizar scripts de carga

**Impacto científico**: Nulo. El formato de almacenamiento no afecta la información.

**Implementación**: ~30 minutos de trabajo.

---

### Opción D: Usar Subset de Datos

**Descripción**: Entrenar con 100-200 archivos en lugar de 848.

| Subset | Archivos | Tamaño JSON | Validez estadística |
|--------|----------|-------------|---------------------|
| 100 | 100 | 1.2 GB | Baja-Moderada |
| 200 | 200 | 2.4 GB | Moderada |
| 400 | 400 | 4.8 GB | Alta |

**Pros**:
- Solución inmediata, sin modificar código
- Permite validar pipeline rápidamente
- Útil para debugging y prototipado

**Contras**:
- Resultados del 4.1 usaron 848 muestras - comparación no equivalente
- Menor poder estadístico
- No es solución definitiva

**Impacto científico**: Alto si se reduce demasiado. Para conclusiones robustas, se necesitan al menos 400-500 muestras.

---

### Opción E: Estrategia Híbrida Optimizada

**Descripción**: Combinar múltiples optimizaciones de forma balanceada.

**Configuración propuesta**:
- `hop_length`: 1024 (de 512) → 2x reducción
- `bins`: 256 (mantener) → resolución armónica intacta
- `formato`: Binario (numpy) → 12x reducción
- `archivos`: 848 (todos) → validez estadística completa

**Resultado**:
- Tamaño: 10.3 GB → 0.43 GB
- RAM: < 500 MB
- Frames/archivo: 170 (suficiente para dinámica)
- Resolución armónica: Intacta

---

### Opción F: Replantear el Experimento

**Descripción**: Cuestionar la premisa del experimento.

**Pregunta fundamental**: ¿Qué queremos demostrar?

#### Escenario 1: "Replicar experimento 4.1 con datos 5.0"
Si promediamos los frames temporales para el VAE (para mantener compatibilidad), estamos esencialmente volviendo al formato 4.1. El experimento pierde sentido.

#### Escenario 2: "Evaluar si el formato temporal mejora el aprendizaje"
Esto requiere:
- HRM: Procesar secuencias completas ✓ (ya está diseñado para esto)
- VAE: Necesita adaptación temporal (VAE-RNN o VAE-Attention)
- Comparación justa: Ambas arquitecturas deben poder aprovechar la temporalidad

#### Escenario 3: "Comparar capacidad temporal vs estática"
- Entrenar HRM con datos 5.0 (temporal)
- Entrenar HRM con datos 5.0 promediados (estático)
- Comparar: ¿Cuánto mejora tener la dimensión temporal?

---

## 4. Análisis Científico Profundo

### 4.1 ¿Por qué el HRM superó al VAE con datos 4.1?

En los experimentos previos:
- **HRM val_loss**: 2.74
- **VAE val_loss**: 4,212.58
- **Diferencia**: 153,500%

El HRM ganó abrumadoramente a pesar de que los datos 4.1 eran **estáticos**. Esto sugiere que:

1. La arquitectura HRM (GRU + LSTM + Attention) es inherentemente superior para datos armónicos
2. La ventaja NO viene de la temporalidad (los datos eran estáticos)
3. La ventaja viene de la **jerarquía de procesamiento** y los **mecanismos de atención**

### 4.2 ¿Qué aportará el formato temporal 5.0?

**Hipótesis**: El HRM debería beneficiarse aún más del formato temporal porque:
- Puede modelar evolución de patrones armónicos en el tiempo
- La atención multi-cabeza puede encontrar correlaciones entre frames distantes
- El GRU/LSTM fueron diseñados específicamente para secuencias

**Predicción**: La ventaja del HRM sobre VAE debería **aumentar** con datos temporales (asumiendo VAE sin modificar).

### 4.3 Problema de la Comparación Justa

Si usamos datos temporales con HRM pero promediamos para VAE, la comparación es **metodológicamente cuestionable**:

- HRM recibe más información (341 frames)
- VAE recibe menos información (1 frame promediado)
- La diferencia de rendimiento podría deberse a la cantidad de información, no a la arquitectura

**Solución rigurosa**:
1. **Comparación A**: HRM-temporal vs VAE-temporal (ambos con secuencias)
2. **Comparación B**: HRM-estático vs VAE-estático (ambos con promedios)
3. **Comparación C**: HRM-temporal vs HRM-estático (efecto de temporalidad)

---

## 5. Matriz de Decisión

| Criterio | Peso | Opción A | Opción B | Opción C | Opción D | Opción E |
|----------|------|----------|----------|----------|----------|----------|
| Preserva información | 25% | ⚠️ | ⚠️ | ✅ | ✅ | ✅ |
| Viabilidad técnica | 20% | ✅ | ✅ | ✅ | ✅ | ✅ |
| Tiempo implementación | 15% | ✅ | ✅ | ⚠️ | ✅ | ⚠️ |
| Comparabilidad con 4.1 | 20% | ⚠️ | ⚠️ | ✅ | ⚠️ | ⚠️ |
| Validez científica | 20% | ⚠️ | ⚠️ | ✅ | ⚠️ | ✅ |
| **Puntuación** | 100% | 60% | 60% | 90% | 70% | 85% |

---

## 6. Recomendación Profesional

### 6.1 Recomendación Principal: Opción C + Enfoque Científico Riguroso

**Paso 1: Cambiar a formato binario (inmediato)**
- Modificar analizador 5.0 para generar `.npz`
- Reducción de 10.3 GB → 0.8 GB
- Sin pérdida de información
- 30 minutos de implementación

**Paso 2: Generar dataset completo con parámetros originales**
- 848 archivos
- hop=512, bins=256
- Preservar toda la resolución temporal

**Paso 3: Diseñar experimento riguroso**

| Experimento | Arquitectura | Datos | Objetivo |
|-------------|--------------|-------|----------|
| E1 | HRM-temporal | 5.0 (secuencias) | Baseline temporal |
| E2 | VAE-temporal* | 5.0 (secuencias) | Comparación justa |
| E3 | HRM-estático | 5.0 (promediado) | Control temporal |
| E4 | VAE-estático | 5.0 (promediado) | Control VAE |

*Requiere modificar VAE para procesar secuencias (agregar LSTM o Attention)

### 6.2 Justificación

1. **Científica**: Formato binario preserva toda la información. Las comparaciones serán metodológicamente válidas.

2. **Técnica**: 0.8 GB es trivialmente manejable. No hay razón para sacrificar información.

3. **Práctica**: La inversión de 30 minutos para cambiar el formato se amortiza inmediatamente en tiempos de carga y procesamiento.

4. **Estratégica**: El formato binario es estándar en ML y facilitará trabajo futuro.

### 6.3 Alternativa Pragmática (Si hay presión de tiempo)

Si se necesitan resultados rápidos:

1. **Formato binario** (obligatorio - sin esto nada funciona bien)
2. **hop=1024** (reducir frames a 170, aún suficiente para dinámica)
3. **Proceder con comparación HRM vs VAE**
4. **Documentar limitación**: VAE usa promedio temporal, comparación no completamente justa

---

## 7. Conclusión

El "problema" del tamaño del dataset es un **falso problema** causado por el formato JSON. En binario, los 848 archivos ocupan menos de 1 GB - perfectamente manejable.

La pregunta real no es técnica sino científica: **¿Qué queremos demostrar con el Analizador 5.0?**

Si queremos demostrar la superioridad del formato temporal, necesitamos arquitecturas que puedan aprovecharlo (HRM ya puede, VAE necesita modificación).

Si solo queremos "repetir el experimento del 4.1", el esfuerzo del 5.0 pierde sentido.

**Mi recomendación**: Implementar formato binario, generar dataset completo, y diseñar un experimento que realmente evalúe la contribución de la dimensión temporal al aprendizaje de patrones armónicos.

---

## Anexo: Implementación Formato Binario

```python
# Modificación sugerida para analizador_5.0.py
import numpy as np

def save_dataset_binary(data: dict, output_path: str):
    """Guarda dataset en formato numpy comprimido."""
    arrays = {}
    metadata = {}

    for key, entry in data.items():
        safe_key = key.replace('.', '_').replace('/', '_')
        arrays[f"{safe_key}_frames"] = np.array(
            entry['ratio_hist_enriched_frames'],
            dtype=np.float32
        )
        metadata[safe_key] = {
            'sr': entry['sr'],
            'n_fft': entry['n_fft'],
            'hop_length': entry['hop_length'],
            'n_frames': entry['n_frames']
        }

    np.savez_compressed(
        output_path,
        metadata=metadata,
        **arrays
    )
```

Tiempo estimado de implementación: 30 minutos.

---

*Documento generado como parte del proyecto Phideus - Análisis de Estructuras Armónicas Naturales*
