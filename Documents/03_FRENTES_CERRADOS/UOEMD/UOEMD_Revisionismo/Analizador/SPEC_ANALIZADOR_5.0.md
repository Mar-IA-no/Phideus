# Especificación Oficial · Analizador Phideus 5.0

**Versión:** 5.0
**Estado:** Propuesta para estándar interno PHIDEUS
**Autoría:** Equipo PHIDEUS
**Fecha:** 29 nov 2025

---

## 1. Objetivo

El **Analizador Phideus 5.0** define el pipeline estándar para convertir señales 1D (audio o vibración de acelerómetro) en una **secuencia temporal de histogramas de proporciones enriquecidos**, lista para ser usada como entrada del **VAE temporal de PHIDEUS**.

Este analizador:

* Trabaja **siempre en dominio temporal** (ventanas/frames).
* Usa **proporciones físicas adimensionales** (ratios de frecuencia lineales), no escalas antropocéntricas (sin log2, sin cents, sin temperamentos musicales).
* Genera, para cada frame, un histograma enriquecido con **3 canales**.
* Produce un tensor conceptual de salida por archivo: `X ∈ ℝ^{T × B × 3}`, donde:

  * `T` = número de frames temporales,
  * `B` = número de bins de ratio,
  * `3` = canales (proporción, momento, entropía).

---

## 2. Alcance

El Analizador 5.0 se aplica a:

* Archivos **WAV mono** (audio o vibración de acelerómetro) con:

  * frecuencia de muestreo en el orden de 1–50 kHz (configurable),
  * profundidad de bits estándar (16/24 bits).

No se limita al dominio “audio musical”: el diseño está pensado para cualquier fuente de vibración/sonido donde tenga sentido hablar de **picos espectrales** y **relaciones de frecuencia**.

---

## 3. Resumen del pipeline

Dado un archivo WAV de señal 1D `a(t)`:

1. Preprocesado mínimo (eliminación de DC).
2. Ventaneo en el tiempo y cálculo de STFT:

   * para cada frame → espectro de magnitudes.
3. Detección de picos espectrales significativos por frame.
4. Cálculo de **ratios lineales** `r = f_j / f_i` entre pares de picos y sus pesos físicos.
5. Construcción, para cada frame, de un **histograma enriquecido**:

   * Canal 0: proporción (PDF discreta).
   * Canal 1: momento sobre el ratio lineal.
   * Canal 2: entropía local normalizada.
6. Salida en formato JSON con:

   * metadatos (`sr`, `n_fft`, `hop_length`, etc.),
   * tiempos por frame,
   * secuencia de histogramas enriquecidos.

---

## 4. Especificación técnica

### 4.1. Entrada

**Tipo:** archivos `.wav` mono, ubicados en un directorio (recursivo).

* `y`: señal mono en coma flotante.
* `sr`: frecuencia de muestreo (Hz), tomada del WAV sin remuestrear (`sr=None`).

**Preprocesado mínimo:**

* El analizador carga `y` y puede aplicar (en la implementación por defecto):

  * centrado: `y ← y - mean(y)`.
  * filtros adicionales (opcional, fuera de esta spec).

### 4.2. Parámetros por defecto

Valores recomendados (ajustables vía CLI):

* `n_fft = 2048`
* `hop_length = 512`  (≈ 75% de solapamiento)
* `peak_thr_factor = 1.25`
* `local_median_window = 30` (en bins de frecuencia)
* `rel_peak_tol = 0.01` (1% de diferencia relativa en Hz para deduplicar picos)
* `max_band_hz = None` (sin límite superior por defecto)
* `min_ratio = 1.0`
* `max_ratio = 6.0`
* `n_ratio_bins = 256`

### 4.3. STFT y frames temporales

Se calcula la STFT 1D:

* Ventana: Hann.
* Sin centrado en el tiempo (parámetro `center=False`).

Sea:

* `stft ∈ ℂ^{F × T}`:

  * `F` = número de bins de frecuencia (`n_fft/2 + 1` si se usa convención estándar de librosa),
  * `T` = número de frames.

Magnitudes:

* `mag = |stft| ∈ ℝ^{F × T}`
* `freqs = fft_frequencies(sr, n_fft) ∈ ℝ^F`

Cada frame `t` tiene:

* `mag_frame = mag[:, t]`

### 4.4. Detección de picos por frame

Para cada `mag_frame`:

1. Se calcula un **umbral local**:

   * Para cada índice `m`:

     * se toma una ventana local `[m-W, m+W]`,
     * se calcula `median(mag_frame[lo:hi])`,
     * se multiplica por `peak_thr_factor` (`α`).

   * Resultado: vector `thr[m]`.

2. Se usan `find_peaks` (SciPy o equivalente) con:

   * condición de máximo local,
   * `height=thr`.

3. Filtro de banda opcional:

   * si `max_band_hz` no es `None`, se descartan picos cuya frecuencia `freqs[m] > max_band_hz`.

4. Refinamiento sub-bin (interpolación parabólica, opcional pero recomendado):

   * se estima un índice real `b` alrededor de cada pico entero `m`,
   * se interpola sobre `freqs` para obtener `f_peak` más preciso.

5. **Deduplicación de picos** por tolerancia relativa:

   * Se ordenan candidatos por amplitud descendente.
   * Se recorren en ese orden y se acepta un nuevo pico `f` solo si:

     ```
     abs(f - f_prev) / (f_prev + 1e-12) > rel_peak_tol
     ```

     para todos los picos ya aceptados.

Resultado por frame `t`:

* Lista `{(f_i, A_i)}` con:

  * `f_i` en Hz,
  * `A_i` magnitud espectral.

### 4.5. Ratios lineales y pesos físicos

Para cada frame `t` y cada par de picos `(i, j)` con `j > i`:

1. Ratio lineal:

   ```text
   r_ij = f_j / f_i
   ```

2. Filtro de rango:

   ```text
   min_ratio <= r_ij <= max_ratio
   ```

3. Peso físico (energético):

   ```text
   w_ij = sqrt(A_i * A_j)
   ```

Se construyen dos listas:

* `ratios = [r_ij]`
* `weights = [w_ij]`

Si no se obtiene ningún ratio válido en el frame, ese frame será un histograma vacío.

### 4.6. Bins de ratio (escala lineal)

Se define un eje de bins en **escala lineal**:

* `edges_ratio ∈ ℝ^{B+1}` con:

  * `edges_ratio[0] = min_ratio`
  * `edges_ratio[B] = max_ratio`
  * típicamente, `edges_ratio` espaciado linealmente (uniforme).

Cada bin `b` cubre:

```text
[e_b, e_{b+1})   para b = 0..B-1
```

### 4.7. Histograma base por frame

Si `ratios` no está vacío:

* Se calcula un histograma ponderado:

  ```text
  hist_ratio[b] = Σ w_ij   para todos los (r_ij, w_ij) que caen en el bin b
  ```

en código, típicamente:

```python
hist_ratio, _ = np.histogram(ratios, bins=edges_ratio, weights=weights)
```

Si `ratios` está vacío:

* `hist_ratio` se inicializa como un vector de ceros de longitud `B`.

Este `hist_ratio` es la base para construir los 3 canales enriquecidos.

---

## 5. Histograma enriquecido (3 canales por bin y por frame)

Para cada frame `t`, el analizador construye una matriz:

```text
H_t ∈ ℝ^{B × 3}
```

donde cada fila `H_t[b]` contiene:

```text
H_t[b] = (prop_b, mom_b, ent_b)
```

### 5.1. Canal 0: proporción (PDF discreta)

Definiciones:

* `h_b = hist_ratio[b]`
* `T = sum_b h_b`

Si `T > 0`:

```text
prop_b = h_b / (T + ε)
```

Si `T == 0`:

```text
prop_b = 0  para todos los b
```

Este canal describe la **distribución relativa** de los ratios presentes en el frame.

### 5.2. Canal 1: momento sobre ratio lineal

Centros de bin:

```text
c_b = (e_b + e_{b+1}) / 2
```

Momento crudo:

```text
moment_raw_b = h_b * c_b^2
```

Suma total:

```text
M = Σ_b moment_raw_b
```

Si `M > 0`:

```text
mom_b = moment_raw_b / (M + ε)
```

Si `M == 0`:

```text
mom_b = 0  para todos los b
```

Este canal codifica cómo se reparte el peso entre ratios más pequeños y más grandes **en escala lineal**, no logarítmica.

### 5.3. Canal 2: entropía local normalizada

A partir de `prop_b`:

* entropía cruda:

  ```text
  ent_raw_b = - prop_b * log(prop_b + ε)
  ```

* suma total:

  ```text
  E = Σ_b ent_raw_b
  ```

Si `E > 0`:

```text
ent_b = ent_raw_b / (E + ε)
```

Si `E == 0`:

```text
ent_b = 0  para todos los b
```

Este canal representa el grado de **estructura vs dispersión** de la distribución de proporciones dentro del frame.

---

## 6. Forma final y formato de salida

Para un archivo WAV con `T` frames, el analizador produce conceptualmente:

* `H ∈ ℝ^{T × B × 3}`: secuencia de histogramas enriquecidos.

En el JSON se serializa como:

```jsonc
{
  "sr":           int,          // sample rate
  "n_fft":        int,
  "hop_length":   int,
  "frame_times":  [float, ...], // tiempo (segundos) de cada frame
  "n_frames":     int,          // = len(frame_times)
  "n_ratio_bins": int,          // = B

  // Histograma simple (PDF) por frame: opcional pero recomendado
  "ratio_hist_frames": [
    [float, ...], // frame 0, length B
    ...
  ],

  // Histograma enriquecido por frame (canales: prop, mom, ent)
  "ratio_hist_enriched_frames": [
    [               // frame 0
      [p, m, e],    // bin 0
      ...
    ],
    ...
  ]
}
```

A nivel directorio:

```jsonc
{
  "rel/path/file1.wav": { ...objeto anterior... },
  "rel/path/file2.wav": { ... },
  ...
}
```

---

## 7. Integración con el VAE temporal de PHIDEUS

### 7.1. Tensor de entrada esperado

Para el **VAE temporal**, la entrada por muestra se define como:

* Tomar `ratio_hist_enriched_frames` del JSON:

  * shape: `[T, B, 3]`.
* Aplanar el eje de bins y canales por frame:

  * `frame_dim = B * 3`.
  * `X[t] = flatten(H_t) ∈ ℝ^{frame_dim}`.
* Tensor final por ejemplo:

  ```text
  X ∈ ℝ^{T × frame_dim}
  ```

El `TemporalDataset` es responsable de:

* cargar el JSON,
* seleccionar `ratio_hist_enriched_frames`,
* formar `X` con shape `[T, frame_dim]`.

### 7.2. Compatibilidad con versiones anteriores (4.x)

* El Analizador 5.0 **no es retrocompatible a nivel semántico** con los datos generados por 4.x (que usaban escalas log2/cents y, en muchos casos, histogramas globales).
* Se recomienda:

  * etiquetar claramente datasets antiguos como `v4`,
  * entrenar nuevos modelos con datos generados por el **Analizador 5.0**,
  * documentar en el README de entrenamiento qué analizador se usó.

---

## 8. Decisiones de diseño (resumen)

1. **Temporalidad obligatoria**

   * No se admite ya un “modo global por archivo”; todo se hace por frames.

2. **Escala física absoluta**

   * Se abandona el uso de log2/cents/temperamentos.
   * Todos los ratios se miden y se bindean en escala lineal.

3. **Tres canales enriquecidos** por bin:

   * `prop_b` (PDF discreta),
   * `mom_b` (momento normalizado en ratio lineal),
   * `ent_b` (entropía normalizada).

4. **Tolerancia de picos relativa**

   * Se usa `rel_peak_tol` (por defecto 1%) en Hz, no una cantidad fija de cents.

---

## 9. Posibles extensiones futuras

* Añadir canales adicionales (p.ej. **varianza temporal** del bin, a partir de ventanas de múltiples frames).
* Adaptar el analizador a otros dominios (RF, vibración estructural) manteniendo la lógica de:

  * picos espectrales → ratios → histogramas enriquecidos.
* Definir un “modo multimodal” donde distintos sensores compartan la misma especificación de proporciones y bins, para alimentar un latente compartido.

---

*Fin de la especificación del Analizador Phideus 5.0.*
