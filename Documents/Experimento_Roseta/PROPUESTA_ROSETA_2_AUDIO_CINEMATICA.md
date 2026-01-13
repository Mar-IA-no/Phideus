# Propuesta de Experimento Roseta 2: Cross-Modal Audio → Cinemática Visual

**Proyecto**: PHIDEUS v5.0 - Nature's Harmonic Structure Analysis
**Documento**: Propuesta de Investigación Doctoral
**Versión**: 1.0
**Fecha**: Enero 2026
**Estado**: PROPUESTA EN REVISIÓN

---

## Resumen Ejecutivo

Este documento presenta el diseño experimental para **Roseta 2**, la segunda fase del programa de validación cross-modal de PHIDEUS. Mientras que Roseta 1 demostró la correspondencia Audio ↔ Vibración en datos industriales (UOEMD), Roseta 2 extiende la hipótesis a un dominio fundamentalmente diferente: **patrones cinemáticos visuales generados por láser** (figuras de Lissajous).

### Hipótesis Central

> *Las proporciones armónicas (ratios de frecuencia) constituyen un lenguaje universal que se manifiesta tanto en el dominio acústico como en su representación geométrica visual, permitiendo la inferencia bidireccional entre modalidades.*

### Diferenciación de Roseta 1

| Aspecto | Roseta 1 (UOEMD) | Roseta 2 (Lissajous) |
|---------|------------------|----------------------|
| Dominio A | Audio (micrófono) | Audio (micrófono) |
| Dominio B | Vibración (acelerómetro) | Visual (patrones geométricos) |
| Relación física | Indirecta (mismo fenómeno) | **Directa** (ratio = geometría) |
| Dataset | Público (University of Ottawa) | **Generado/Capturado** |
| Novedad científica | Validación industrial | **Extensión teórica fundamental** |

---

## 1. Fundamentación Teórica

### 1.1 Figuras de Lissajous: Definición Matemática

Las figuras de Lissajous son curvas paramétricas generadas por la superposición de dos movimientos armónicos simples perpendiculares:

$$x(t) = A \sin(\omega_1 t + \delta)$$
$$y(t) = B \sin(\omega_2 t)$$

Donde:
- $A, B$: Amplitudes de los componentes
- $\omega_1, \omega_2$: Frecuencias angulares ($\omega = 2\pi f$)
- $\delta$: Diferencia de fase entre los componentes
- $t$: Parámetro temporal

### 1.2 Relación Ratio-Geometría

La **forma** del patrón está determinada únicamente por:

1. **Ratio de frecuencias** $r = \omega_1 / \omega_2 = f_1 / f_2$
2. **Diferencia de fase** $\delta$

Para un ratio $p:q$ (en términos mínimos, donde $\gcd(p,q) = 1$):
- La figura tiene exactamente **p tangencias horizontales**
- La figura tiene exactamente **q tangencias verticales**
- La figura se cierra si y solo si el ratio es **racional**

### 1.3 Tabla de Correspondencias Fundamentales

| Ratio $f_1:f_2$ | Intervalo Musical | Patrón Geométrico | Tangencias H:V |
|-----------------|-------------------|-------------------|----------------|
| 1:1 | Unísono | Elipse/Círculo | 1:1 |
| 2:1 | Octava | Figura-8 (∞) | 2:1 |
| 3:2 | Quinta perfecta | Trébol 3 hojas | 3:2 |
| 4:3 | Cuarta perfecta | 4 loops entrelazados | 4:3 |
| 5:4 | Tercera mayor | Patrón complejo 5×4 | 5:4 |
| 5:3 | Sexta mayor | Patrón 5 loops | 5:3 |
| 6:5 | Tercera menor | Patrón denso | 6:5 |

### 1.4 Influencia de la Fase

La diferencia de fase $\delta$ modifica la **orientación** y **apertura** del patrón:

| Fase $\delta$ | Efecto (para ratio 1:1) |
|---------------|-------------------------|
| 0° | Línea diagonal (45°) |
| 45° | Elipse inclinada |
| 90° | Círculo (si A=B) |
| 135° | Elipse inclinada (opuesta) |
| 180° | Línea diagonal (-45°) |

### 1.5 Invariancia Proporcional

**Propiedad fundamental**: El patrón de Lissajous es **invariante a la escala de frecuencias**:

$$\text{Lissajous}(f_1, f_2) \equiv \text{Lissajous}(k \cdot f_1, k \cdot f_2) \quad \forall k > 0$$

Esto significa que 220 Hz + 330 Hz produce el **mismo patrón** que 440 Hz + 660 Hz (ambos son ratio 2:3). Esta propiedad es crucial para la hipótesis de PHIDEUS: **los ratios son el lenguaje universal, no las frecuencias absolutas**.

---

## 2. Estado del Arte

### 2.1 Aprendizaje Cross-Modal Audio-Visual

#### 2.1.1 Métodos Contrastivos

**CLIP** (Radford et al., 2021): Estableció el paradigma de aprender representaciones alineadas entre modalidades mediante contrastive learning. Arquitectura dual-encoder con InfoNCE loss.

**AudioCLIP** (Guzhov et al., 2022): Extensión de CLIP a tres modalidades (texto, imagen, audio). Utiliza ESResNeXt como encoder de audio. Logra 97.15% en ESC-50 en configuración zero-shot.

**ImageBind** (Girdhar et al., 2023): Alinea seis modalidades diferentes (imagen, texto, audio, profundidad, térmica, IMU) en un espacio de embedding compartido.

#### 2.1.2 Métodos con Attention Cross-Modal

**CrossMAE** (Guo et al., 2024 - CVPR): Masked Autoencoders con attention cross-modal para pre-entrenamiento audio-visual con correspondencias region-aware.

**SCLAV** (2023 - ACM MM): Framework con módulo de attention inter-modal y módulo de integración intra-modal, optimizado con contrastive loss supervisado.

**HiCMAE** (2024): Hierarchical Contrastive Masked Autoencoder para reconocimiento de emociones audio-visual self-supervised.

#### 2.1.3 Sincronización Audio-Visual

**DiVAS** (Fernandez-Labrador et al., 2024 - CVPR): Arquitectura Transformer para sincronización audio-visual de longitud variable, superando la limitación de 5 frames fijos.

**AlignNet** (Wang et al., 2020 - WACV): Feature pyramids aprendibles con capas de warping para estimación de correspondencia temporal.

### 2.2 Diferenciación de PHIDEUS

| Aspecto | Métodos Existentes | PHIDEUS Roseta 2 |
|---------|-------------------|------------------|
| Relación A-V | Semántica (qué hay) | **Física (ratios)** |
| Supervisión | Self-supervised/Weak | **Ground truth matemático** |
| Interpretabilidad | Black-box | **Completamente interpretable** |
| Correspondencia | Aproximada | **Determinística** |

### 2.3 Cimática y Visualización de Sonido

#### 2.3.1 Antecedentes Históricos

| Período | Investigador | Contribución |
|---------|--------------|--------------|
| ~1500 | Leonardo da Vinci | Primera observación de patrones en polvo sobre superficies vibrantes |
| 1680 | Robert Hooke | Demostración sistemática de patrones nodales |
| 1787 | Ernst Chladni | Publicación de "Entdeckungen über die Theorie des Klanges" |
| 1857 | Jules Lissajous | Desarrollo del aparato óptico de espejos vibrantes |
| 1967 | Hans Jenny | Acuñó "Cimática", estudios exhaustivos con osciladores |

#### 2.3.2 Aplicaciones Modernas

- **Diagnóstico estructural**: Patrones de Chladni para detectar defectos
- **Educación musical**: Visualización de intervalos armónicos
- **Arte generativo**: Instalaciones audiovisuales interactivas
- **Investigación bioacústica**: Análisis de vocalizaciones animales

---

## 3. Diseño Experimental

### 3.1 Arquitectura Conceptual

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ROSETA 2: PIPELINE COMPLETO                     │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────────────────┐
│  Generador   │ ──► │  Oscilador   │ ──► │  Proyector Láser XY      │
│  de Tonos    │     │  Audio       │     │  (Galvos o Espejos)      │
│  (Python)    │     │  (DAC)       │     │                          │
└──────────────┘     └──────────────┘     └──────────────────────────┘
       │                    │                          │
       │                    ▼                          ▼
       │             ┌──────────────┐          ┌──────────────┐
       │             │  Micrófono   │          │   Cámara     │
       │             │  (ADC)       │          │   (Frames)   │
       │             └──────────────┘          └──────────────┘
       │                    │                          │
       │                    ▼                          ▼
       │             ┌──────────────┐          ┌──────────────┐
       │             │ Analizador   │          │ Analizador   │
       │             │ Audio 5.0    │          │ Visual       │
       │             └──────────────┘          └──────────────┘
       │                    │                          │
       │                    ▼                          ▼
       │             ┌──────────────┐          ┌──────────────┐
       │             │ Histograma   │          │ Features     │
       │             │ de Ratios    │          │ Geométricos  │
       │             │ [T,B,3]      │          │ o CNN        │
       │             └──────────────┘          └──────────────┘
       │                    │                          │
       │                    └──────────┬───────────────┘
       │                               ▼
       │                    ┌──────────────────────┐
       │                    │     RosetaVAE v2     │
       │                    │                      │
       │                    │  ┌────────────────┐  │
       │                    │  │ Encoder Audio  │  │
       │                    │  └───────┬────────┘  │
       │                    │          │           │
       │                    │          ▼           │
       │                    │  ┌────────────────┐  │
       │                    │  │   z_shared     │◄─┼── InfoNCE Loss
       │                    │  └───────┬────────┘  │
       │                    │          │           │
       │                    │          ▼           │
       │                    │  ┌────────────────┐  │
       │                    │  │ Encoder Visual │  │
       │                    │  └────────────────┘  │
       │                    │                      │
       │                    └──────────────────────┘
       │                               │
       │                               ▼
       ▼                    ┌──────────────────────┐
┌──────────────┐            │   VALIDACIÓN         │
│  Ground      │ ──────────►│                      │
│  Truth       │            │  z_audio ≈ z_visual  │
│  (Ratios     │            │  Cross-Retrieval     │
│  conocidos)  │            │  Pearson > 0.7       │
└──────────────┘            └──────────────────────┘
```

### 3.2 Modalidades de Captura

#### Opción A: Captura Física (Máximo Realismo)

**Equipamiento**:
- Generador de señales dual (función arbitraria)
- Amplificador estéreo
- Sistema láser XY (galvanómetros o espejos en speakers)
- Cámara de alta velocidad (>60 fps)
- Micrófono de medición
- Interface de audio profesional (sincronización)

**Ventajas**:
- Datos completamente reales
- Validación de la teoría en condiciones físicas
- Posibilidad de efectos no lineales interesantes

**Desventajas**:
- Setup complejo
- Ruido y distorsiones
- Sincronización difícil

#### Opción B: Generación Sintética (Máximo Control) - **RECOMENDADA PARA FASE 1**

**Metodología**:
1. Generar audio sintético con ratios controlados
2. Renderizar patrones de Lissajous matemáticamente
3. Sincronización perfecta garantizada
4. Añadir ruido controlado para regularización

**Ventajas**:
- Ground truth perfecto
- Control total de parámetros
- Sin hardware adicional
- Reproducibilidad completa

**Desventajas**:
- No captura efectos físicos reales
- Posible gap dominio sintético→real

#### Opción C: Híbrida (Recomendada para Publicación)

1. **Fase 1**: Validación con datos sintéticos
2. **Fase 2**: Confirmación con captura física
3. **Fase 3**: Transfer learning sintético→real

### 3.3 Protocolo de Generación de Datos Sintéticos

#### 3.3.1 Parámetros del Espacio de Muestreo

| Parámetro | Rango | Distribución | Muestras |
|-----------|-------|--------------|----------|
| Ratio $f_1:f_2$ | [1:1, 8:1] | Discreto (armónicos naturales) | 21 ratios |
| Frecuencia base $f_2$ | [110, 880] Hz | Log-uniforme | 8 valores |
| Fase $\delta$ | [0°, 180°] | Uniforme | 12 valores |
| Amplitud relativa $A/B$ | [0.5, 2.0] | Log-uniforme | 5 valores |
| Duración | 2 segundos | Fijo | - |
| SNR (ruido) | [20, 40] dB | Uniforme | 5 niveles |

**Total de combinaciones**: 21 × 8 × 12 × 5 × 5 = **50,400 muestras**

#### 3.3.2 Ratios Objetivo (Inspirados en Armonía Natural)

```python
RATIOS_OBJETIVO = [
    # Consonancias perfectas
    (1, 1),   # Unísono
    (2, 1),   # Octava
    (3, 2),   # Quinta perfecta
    (4, 3),   # Cuarta perfecta

    # Consonancias imperfectas
    (5, 4),   # Tercera mayor
    (6, 5),   # Tercera menor
    (5, 3),   # Sexta mayor
    (8, 5),   # Sexta menor

    # Armónicos superiores
    (7, 4),   # Séptima menor armónica
    (7, 5),   # Tritono armónico
    (8, 7),   # Segunda mayor armónica

    # Ratios complejos
    (9, 8),   # Tono pitagórico
    (10, 9),  # Tono menor
    (16, 15), # Semitono diatónico

    # Extensiones
    (3, 1),   # Duodécima (octava + quinta)
    (4, 1),   # Doble octava
    (5, 2),   # Décima mayor
    (5, 1),   # Diecisieteava mayor
    (6, 1),   # Octava + quinta + octava
    (7, 3),   # Ratio complejo
    (8, 3),   # Ratio complejo
]
```

#### 3.3.3 Generación de Audio

```python
def generar_audio_bifrequency(f1, f2, fase, amp_ratio, duracion, fs=44100, snr_db=30):
    """
    Genera audio con dos componentes sinusoidales.

    Args:
        f1, f2: Frecuencias de los componentes (Hz)
        fase: Diferencia de fase (radianes)
        amp_ratio: Ratio de amplitudes A1/A2
        duracion: Duración en segundos
        fs: Sample rate
        snr_db: Relación señal-ruido

    Returns:
        audio: Array de audio normalizado
    """
    t = np.linspace(0, duracion, int(fs * duracion), endpoint=False)

    # Componentes sinusoidales
    A1 = amp_ratio / (1 + amp_ratio)
    A2 = 1 / (1 + amp_ratio)

    signal = A1 * np.sin(2 * np.pi * f1 * t + fase) + A2 * np.sin(2 * np.pi * f2 * t)

    # Añadir ruido gaussiano
    noise_power = 10 ** (-snr_db / 10)
    noise = np.sqrt(noise_power) * np.random.randn(len(t))

    audio = signal + noise
    audio = audio / np.max(np.abs(audio))  # Normalizar

    return audio
```

#### 3.3.4 Generación de Patrones Visuales

```python
def generar_lissajous_frame(f1, f2, fase, amp_ratio, t_start, t_end,
                            resolution=256, line_width=2):
    """
    Genera un frame de patrón de Lissajous como imagen.

    Args:
        f1, f2: Frecuencias
        fase: Diferencia de fase (radianes)
        amp_ratio: Ratio de amplitudes
        t_start, t_end: Ventana temporal
        resolution: Tamaño de imagen (cuadrada)
        line_width: Grosor de línea

    Returns:
        image: Array (resolution, resolution) normalizado [0, 1]
    """
    # Generar puntos de la curva
    t = np.linspace(t_start, t_end, 10000)

    A = amp_ratio / (1 + amp_ratio)
    B = 1 / (1 + amp_ratio)

    x = A * np.sin(2 * np.pi * f1 * t + fase)
    y = B * np.sin(2 * np.pi * f2 * t)

    # Mapear a coordenadas de imagen
    margin = 0.1
    x_img = ((x + 1) / 2 * (1 - 2*margin) + margin) * resolution
    y_img = ((y + 1) / 2 * (1 - 2*margin) + margin) * resolution

    # Rasterizar la curva
    image = np.zeros((resolution, resolution), dtype=np.float32)

    for i in range(len(x_img) - 1):
        cv2.line(image,
                 (int(x_img[i]), int(y_img[i])),
                 (int(x_img[i+1]), int(y_img[i+1])),
                 1.0, line_width)

    return image
```

### 3.4 Representación del Dominio Visual

#### Opción 1: CNN Pre-entrenada (Transfer Learning)

Usar backbone como ResNet-18 pre-entrenada en ImageNet, fine-tuned para patrones geométricos.

**Arquitectura**:
```
Imagen [256, 256] → ResNet-18 (sin FC) → Feature [512] → MLP → z_shared
```

**Ventajas**: Capacidad de generalización, features de alto nivel
**Desventajas**: Puede ignorar geometría fina

#### Opción 2: Features Geométricos Handcrafted

Extraer características interpretables:

| Feature | Descripción | Dimensión |
|---------|-------------|-----------|
| Conteo de lóbulos H | Número de tangencias horizontales | 1 |
| Conteo de lóbulos V | Número de tangencias verticales | 1 |
| Circularidad | Área / (Perímetro² / 4π) | 1 |
| Elongación | Ratio ejes del bounding box | 1 |
| Simetría H | Correlación con flip horizontal | 1 |
| Simetría V | Correlación con flip vertical | 1 |
| Compacidad | Área convexa / Área total | 1 |
| Fourier descriptors | Descriptores de forma | 16 |

**Ventajas**: Completamente interpretable
**Desventajas**: Puede perder información sutil

#### Opción 3: Autoencoder Especializado (Recomendada)

CNN autoencoder entrenado específicamente en patrones de Lissajous:

```
Encoder Visual:
    Conv2d(1, 32, 3, stride=2) → ReLU → BatchNorm
    Conv2d(32, 64, 3, stride=2) → ReLU → BatchNorm
    Conv2d(64, 128, 3, stride=2) → ReLU → BatchNorm
    Conv2d(128, 256, 3, stride=2) → ReLU → BatchNorm
    Flatten → Linear(256*16*16, 512) → Linear(512, z_dim)

Decoder Visual:
    Linear(z_dim, 512) → Linear(512, 256*16*16) → Reshape
    ConvTranspose2d(256, 128, 3, stride=2) → ReLU → BatchNorm
    ConvTranspose2d(128, 64, 3, stride=2) → ReLU → BatchNorm
    ConvTranspose2d(64, 32, 3, stride=2) → ReLU → BatchNorm
    ConvTranspose2d(32, 1, 3, stride=2) → Sigmoid
```

**Ventajas**: Aprende representación óptima para el dominio
**Desventajas**: Requiere pre-entrenamiento

### 3.5 Arquitectura RosetaVAE v2

#### 3.5.1 Especificación

```python
class RosetaVAE_v2(nn.Module):
    """
    VAE cross-modal para Audio ↔ Visual (Lissajous)

    Espacio latente factorizado:
        z = [z_shared | z_private_audio | z_private_visual]

    Dimensiones:
        z_shared: 32 (alineado via InfoNCE)
        z_private_audio: 16 (volumen, ruido, etc.)
        z_private_visual: 16 (contraste, grosor línea, etc.)
    """

    def __init__(self,
                 audio_input_dim=256*3,      # Histograma enriquecido
                 visual_input_shape=(1, 256, 256),  # Imagen Lissajous
                 z_shared_dim=32,
                 z_private_dim=16,
                 hidden_dim=256,
                 lstm_layers=2):
        super().__init__()

        # Encoders
        self.audio_encoder = TemporalEncoder(
            input_dim=audio_input_dim,
            hidden_dim=hidden_dim,
            z_dim=z_shared_dim + z_private_dim,
            lstm_layers=lstm_layers
        )

        self.visual_encoder = VisualEncoder(
            input_shape=visual_input_shape,
            hidden_dim=hidden_dim,
            z_dim=z_shared_dim + z_private_dim
        )

        # Decoders
        self.audio_decoder = TemporalDecoder(
            z_dim=z_shared_dim + z_private_dim,
            hidden_dim=hidden_dim,
            output_dim=audio_input_dim,
            lstm_layers=lstm_layers
        )

        self.visual_decoder = VisualDecoder(
            z_dim=z_shared_dim + z_private_dim,
            hidden_dim=hidden_dim,
            output_shape=visual_input_shape
        )

        # Dimensions
        self.z_shared_dim = z_shared_dim
        self.z_private_dim = z_private_dim
```

#### 3.5.2 Función de Pérdida Compuesta

$$\mathcal{L}_{total} = \lambda_{recon} \mathcal{L}_{recon} + \lambda_{kl} \mathcal{L}_{KL} + \lambda_{info} \mathcal{L}_{InfoNCE} + \lambda_{cross} \mathcal{L}_{cross}$$

| Componente | Función | Peso Sugerido |
|------------|---------|---------------|
| $\mathcal{L}_{recon}$ | MSE/BCE reconstrucción (ambos dominios) | 1.0 |
| $\mathcal{L}_{KL}$ | KL divergence (regularización) | 0.1 |
| $\mathcal{L}_{InfoNCE}$ | Alineación z_shared | 2.0 |
| $\mathcal{L}_{cross}$ | Cross-reconstruction loss | 0.5 |

#### 3.5.3 Cross-Reconstruction Loss (Nuevo)

Pérdida adicional que fuerza la generatividad cross-modal:

$$\mathcal{L}_{cross} = \| \text{Dec}_V(z^A_{shared}, z^A_{priv}) - I_{visual} \|^2 + \| \text{Dec}_A(z^V_{shared}, z^V_{priv}) - H_{audio} \|^2$$

Donde usamos el z_shared de una modalidad con el z_private de la **otra** para reconstruir.

---

## 4. Protocolo Experimental

### 4.1 Fase 0: Preparación de Datos

#### 4.1.1 Generación del Dataset Sintético

```bash
python src/generador/generador_roseta_2_sintetico.py \
    --ratios natural \
    --frecuencias-base 110,220,440,880 \
    --fases 12 \
    --amplitudes 5 \
    --snr 20,25,30,35,40 \
    --duracion 2.0 \
    --output data/datasets/roseta_2_sintetico.npz
```

**Salida esperada**:
- ~50,000 pares (audio, imagen)
- Metadatos: ratio, fase, frecuencias, SNR
- Tamaño estimado: 2-3 GB

#### 4.1.2 Procesamiento

1. Audio → Analizador 5.0 → Histogramas temporales [T, 256, 3]
2. Imágenes → Normalización → Secuencias [T, 1, 256, 256]
3. Sincronización temporal (mismo T para ambos)

### 4.2 Fase 1: Pre-entrenamiento de Encoders

#### 4.2.1 Audio Encoder

Usar modelo pre-entrenado de Roseta 1 (UOEMD) como inicialización.

#### 4.2.2 Visual Encoder

Pre-entrenar autoencoder visual en patrones de Lissajous:

```bash
python experiments/pretrain_visual_encoder.py \
    --data data/datasets/roseta_2_sintetico.npz \
    --output models/visual_encoder_pretrained.pt \
    --epochs 100 \
    --batch-size 32
```

**Criterio de éxito**: Reconstrucción loss < 0.01

### 4.3 Fase 2: Entrenamiento Cross-Modal

#### 4.3.1 Split de Datos

| Conjunto | Proporción | Uso |
|----------|------------|-----|
| Train | 70% | Entrenamiento VAE |
| Validation | 15% | Monitoreo, early stopping |
| Test (in-distribution) | 10% | Evaluación final |
| Test (OOD ratios) | 5% | Generalización a ratios no vistos |

**Ratios reservados para OOD**: 7:4, 8:7, 9:8 (no incluidos en entrenamiento)

#### 4.3.2 Hiperparámetros

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| Batch size | 16 | Balance memoria/gradiente |
| Learning rate | 1e-4 | Conservador para estabilidad |
| Optimizer | AdamW | Weight decay para regularización |
| Scheduler | CosineAnnealing | Convergencia suave |
| Epochs | 200 | Suficiente para convergencia |
| Early stopping | 20 epochs | Evitar overfitting |

#### 4.3.3 Comando de Entrenamiento

```bash
python experiments/run_roseta_2_experiment.py \
    --phase train \
    --data data/datasets/roseta_2_sintetico.npz \
    --output data/training_outputs/roseta_2 \
    --epochs 200 \
    --batch-size 16 \
    --lr 1e-4 \
    --lambda-infonce 2.0 \
    --lambda-cross 0.5 \
    --z-shared-dim 32 \
    --z-private-dim 16 \
    --early-stopping 20
```

### 4.4 Fase 3: Evaluación

#### 4.4.1 Métricas de Alineación

| Métrica | Fórmula | Umbral de Éxito |
|---------|---------|-----------------|
| Cosine Similarity | $\cos(z^A_{shared}, z^V_{shared})$ | > 0.75 |
| L2 Distance | $\|z^A_{shared} - z^V_{shared}\|_2$ | < 2.0 |
| Retrieval Accuracy | Top-1 match rate | > 50% |

#### 4.4.2 Métricas de Cross-Retrieval

| Métrica | Descripción | Umbral |
|---------|-------------|--------|
| Pearson A→V | Correlación audio-predicho vs visual-real | > 0.70 |
| Pearson V→A | Correlación visual-predicho vs audio-real | > 0.70 |
| MSE Cross | Error de reconstrucción cruzada | < 0.10 |

#### 4.4.3 Métricas de Generalización (OOD)

| Métrica | Test | Umbral |
|---------|------|--------|
| Ratio Classification | Clasificar ratio desde z_shared | > 80% accuracy |
| Phase Regression | Predecir fase desde z_shared | R² > 0.60 |
| OOD Alignment | Alineación en ratios no vistos | cos_sim > 0.60 |

### 4.5 Fase 4: Ablación y Análisis

#### 4.5.1 Estudios de Ablación

| Experimento | Configuración | Pregunta |
|-------------|---------------|----------|
| Sin InfoNCE | $\lambda_{info} = 0$ | ¿Es necesaria la pérdida contrastiva? |
| Sin z_private | Solo z_shared | ¿Es necesaria la factorización? |
| Sin cross-loss | $\lambda_{cross} = 0$ | ¿Mejora la generatividad cruzada? |
| CNN vs Handcrafted | Comparar encoders visuales | ¿Qué representación es mejor? |
| Temporal vs Estático | Comparar con modelo sin LSTM | ¿Importa la temporalidad? |

#### 4.5.2 Análisis de Sensibilidad

- Variación de $\lambda_{info}$: [0.5, 1.0, 2.0, 4.0]
- Variación de z_shared_dim: [16, 32, 64, 128]
- Variación de SNR: [15, 20, 25, 30, 35, 40] dB

---

## 5. Análisis Estadístico

### 5.1 Diseño Experimental

**Tipo**: Factorial completo con replicación

**Factores principales**:
1. Ratio de frecuencias (21 niveles)
2. Fase (12 niveles)
3. SNR (5 niveles)

**Variable de respuesta**: Cosine similarity entre z_shared

**Replicaciones**: 5 por combinación

### 5.2 Tests Estadísticos

| Pregunta | Test | Justificación |
|----------|------|---------------|
| ¿El ratio afecta la alineación? | ANOVA | Factor categórico, múltiples niveles |
| ¿La fase afecta la alineación? | ANOVA | Factor categórico, múltiples niveles |
| ¿Hay interacción ratio×fase? | Two-way ANOVA | Efectos cruzados |
| ¿Roseta 2 > baseline? | t-test pareado | Comparación con modelo sin InfoNCE |
| ¿OOD generaliza? | Mann-Whitney U | Distribuciones no normales |

### 5.3 Intervalos de Confianza

Para todas las métricas reportadas:
- Nivel de confianza: 95%
- Método: Bootstrap (1000 resamples) para métricas no paramétricas
- Corrección: Bonferroni para comparaciones múltiples

### 5.4 Tamaño del Efecto

| Métrica | Medida | Interpretación |
|---------|--------|----------------|
| Cohen's d | (μ₁ - μ₂) / σ | d > 0.8 = efecto grande |
| η² | Varianza explicada | η² > 0.14 = efecto grande |

---

## 6. Visualizaciones Propuestas

### 6.1 Figura 1: Arquitectura del Sistema

Diagrama de flujo completo del pipeline (ya incluido en sección 3.1).

### 6.2 Figura 2: Espacio Latente Alineado

**Tipo**: Scatter plot 3D con proyección PCA

**Contenido**:
- Puntos azules: z_shared de audio
- Puntos rojos: z_shared de visual
- Líneas grises: Conexiones de pares correspondientes
- Colores por ratio: Gradiente de color según f1/f2

**Éxito visual**: Puntos del mismo par se superponen

### 6.3 Figura 3: Matriz de Cross-Retrieval

**Tipo**: Heatmap 4×4 o similar

**Contenido**:
- Eje X: Patrón visual real (por ratio)
- Eje Y: Patrón visual predicho desde audio
- Color: Correlación de Pearson

**Éxito visual**: Diagonal dominante (alta correlación para pares correctos)

### 6.4 Figura 4: Galería de Reconstrucción Cruzada

**Tipo**: Grid de imágenes

**Contenido por fila**:
1. Audio waveform input
2. Espectrograma/Histograma de ratios
3. Patrón Lissajous predicho
4. Patrón Lissajous real (ground truth)
5. Diferencia (predicho - real)

### 6.5 Figura 5: Análisis de Ablación

**Tipo**: Bar chart con intervalos de confianza

**Contenido**:
- Barra por configuración de ablación
- Altura: Cosine similarity promedio
- Error bars: IC 95%

### 6.6 Figura 6: Generalización OOD

**Tipo**: Line plot

**Contenido**:
- Eje X: Complejidad del ratio (p+q)
- Eje Y: Cosine similarity
- Línea azul: Ratios in-distribution
- Línea roja: Ratios OOD

---

## 7. Riesgos y Mitigaciones

### 7.1 Riesgos Técnicos

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| Colapso de representación | Media | Alto | Regularización varianza, monitoring |
| Overfitting a sintético | Alta | Medio | Data augmentation, captura física |
| Asimetría de dificultad (audio más fácil que visual) | Media | Medio | Balancear loss weights |
| Memoria insuficiente (imágenes grandes) | Baja | Bajo | Reducir resolución, gradient checkpointing |

### 7.2 Riesgos Científicos

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| Gap sintético→real | Alta | Alto | Captura física en Fase 2 |
| Modelo aprende "shortcuts" | Media | Alto | Análisis de atribución, ablación |
| Resultados no reproducibles | Baja | Alto | Seeds fijos, código público |

### 7.3 Riesgos de Validez

| Amenaza | Tipo | Mitigación |
|---------|------|------------|
| Cherry-picking de resultados | Interna | Pre-registro de hipótesis |
| Falta de generalización | Externa | Test OOD, múltiples dominios |
| Sesgo de confirmación | Constructo | Métricas objetivas predefinidas |

---

## 8. Contribuciones Esperadas

### 8.1 Contribuciones Científicas

1. **Validación extendida de la hipótesis PHIDEUS**: Demostración de que los ratios armónicos son un lenguaje cross-modal que abarca dominios acústicos Y visuales.

2. **Primer benchmark Audio↔Lissajous**: Dataset público con ground truth matemático exacto para evaluación de métodos cross-modal.

3. **Análisis de la relación ratio-geometría aprendida**: Interpretabilidad del espacio latente en términos de propiedades físicas/matemáticas.

### 8.2 Contribuciones Técnicas

1. **RosetaVAE v2**: Arquitectura optimizada para cross-modal learning con modalidades de diferente naturaleza (temporal vs espacial).

2. **Analizador Visual para Lissajous**: Pipeline de procesamiento de imágenes de patrones geométricos.

3. **Generador de datos sintéticos**: Herramienta para crear datasets controlados de pares audio-visual.

### 8.3 Aplicaciones Potenciales

| Aplicación | Descripción |
|------------|-------------|
| Educación musical | Visualización en tiempo real de armonías |
| Diagnóstico acústico | Inferir patrones visuales desde audio |
| Arte generativo | Creación de visualizaciones sincronizadas |
| Accesibilidad | Traducción de información sonora a visual |

---

## 9. Estructura de Archivos Propuesta

```
/root/Phideus/
├── src/
│   ├── generador/
│   │   └── generador_roseta_2_sintetico.py    # Generador de datos
│   ├── analizador/
│   │   ├── analizador_5.0.py                  # (existente)
│   │   └── analizador_visual_lissajous.py     # NUEVO: análisis de imágenes
│   ├── datasets/
│   │   ├── roseta_dataset.py                  # (existente)
│   │   └── roseta_2_dataset.py                # NUEVO: loader audio+visual
│   └── RNA/
│       ├── roseta_vae.py                      # (existente)
│       └── roseta_vae_v2.py                   # NUEVO: con encoder visual
├── experiments/
│   ├── run_roseta_experiment.py               # (existente)
│   ├── run_roseta_2_experiment.py             # NUEVO: experimento completo
│   └── pretrain_visual_encoder.py             # NUEVO: pre-entrenamiento
├── data/
│   └── datasets/
│       ├── roseta_full.npz                    # (existente - UOEMD)
│       └── roseta_2_sintetico.npz             # NUEVO: datos sintéticos
└── Documents/
    └── Experimento_Roseta/
        ├── ANALISIS_EXPERIMENTO_ROSETA.md     # (existente)
        └── PROPUESTA_ROSETA_2_AUDIO_CINEMATICA.md  # ESTE DOCUMENTO
```

---

## 10. Conclusiones

### 10.1 Resumen de la Propuesta

El Experimento Roseta 2 representa una **extensión fundamental** de la validación cross-modal de PHIDEUS. Mientras Roseta 1 demostró la correspondencia Audio↔Vibración en un sistema industrial real, Roseta 2 llevará la hipótesis al **dominio visual**, utilizando las figuras de Lissajous como representación geométrica directa de ratios armónicos.

### 10.2 Fortalezas del Diseño

1. **Ground truth matemático**: A diferencia de otros problemas audio-visual (donde la correspondencia es semántica), aquí la relación es **determinística y conocida**.

2. **Control experimental total**: La generación sintética permite aislar variables y estudiar efectos específicos.

3. **Continuidad con Roseta 1**: Reutilización de arquitectura, código y conocimiento adquirido.

4. **Novedad científica alta**: No existe trabajo previo que aplique contrastive learning a la correspondencia audio-Lissajous.

### 10.3 Limitaciones Anticipadas

1. **Gap dominio sintético**: Los resultados en datos generados pueden no transferir perfectamente a captura física.

2. **Simplicidad de las señales**: Audio bi-frecuencial es más simple que señales naturales complejas.

3. **Escalabilidad**: No está claro cómo escala a señales con múltiples componentes frecuenciales.

### 10.4 Próximos Pasos Inmediatos

| Paso | Prioridad | Dependencias |
|------|-----------|--------------|
| Implementar generador sintético | **ALTA** | Ninguna |
| Implementar analizador visual | **ALTA** | Ninguna |
| Crear RosetaVAE v2 | **ALTA** | Encoders |
| Generar dataset | Media | Generador |
| Pre-entrenar visual encoder | Media | Dataset |
| Ejecutar experimento | Media | Todo lo anterior |

---

## Anexo A: Referencias

### A.1 Cross-Modal Learning
- Radford, A. et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision" (CLIP)
- Guzhov, A. et al. (2022). "AudioCLIP: Extending CLIP to Image, Text and Audio"
- Girdhar, R. et al. (2023). "ImageBind: One Embedding Space To Bind Them All"

### A.2 Audio-Visual Synchronization
- Fernandez-Labrador et al. (2024). "DiVAS: Video and Audio Synchronization with Dynamic Frame Rates" (CVPR)
- Wang et al. (2020). "AlignNet: A Unifying Approach to Audio-Visual Alignment" (WACV)

### A.3 Lissajous and Cymatics
- Lissajous, J. (1857). "Mémoire sur l'étude optique des mouvements vibratoires"
- Jenny, H. (1967). "Cymatics: A Study of Wave Phenomena"
- Chladni, E. (1787). "Entdeckungen über die Theorie des Klanges"

### A.4 Recursos Técnicos
- Bourke, P. "Harmonograph Mathematics" - https://paulbourke.net/geometry/harmonograph/
- Analog Devices. "Lissajous Pattern Tutorial" - https://wiki.analog.com/university/courses/alm1k/intro/intro-lissajou-curves

---

## Anexo B: Código de Referencia

### B.1 Generación de Patrón Lissajous

```python
import numpy as np
import cv2

def generate_lissajous_image(f1, f2, phase, amplitude_ratio=1.0,
                             duration=1.0, resolution=256, line_width=2):
    """
    Genera una imagen de patrón de Lissajous.

    Args:
        f1: Frecuencia componente X (Hz)
        f2: Frecuencia componente Y (Hz)
        phase: Diferencia de fase (radianes)
        amplitude_ratio: Ratio A/B de amplitudes
        duration: Duración de la traza (segundos)
        resolution: Tamaño de imagen en píxeles
        line_width: Grosor de la línea

    Returns:
        image: Array numpy (resolution, resolution) con valores [0, 1]
    """
    # Número de períodos de la frecuencia más baja
    min_freq = min(f1, f2)
    n_periods = max(1, int(duration * min_freq))

    # Generar suficientes puntos para suavidad
    n_points = n_periods * 1000
    t = np.linspace(0, n_periods / min_freq, n_points)

    # Calcular coordenadas
    A = amplitude_ratio / (1 + amplitude_ratio)
    B = 1 / (1 + amplitude_ratio)

    x = A * np.sin(2 * np.pi * f1 * t + phase)
    y = B * np.sin(2 * np.pi * f2 * t)

    # Mapear a coordenadas de imagen con margen
    margin = 0.05
    x_px = ((x + 1) / 2 * (1 - 2*margin) + margin) * (resolution - 1)
    y_px = ((y + 1) / 2 * (1 - 2*margin) + margin) * (resolution - 1)

    # Crear imagen
    image = np.zeros((resolution, resolution), dtype=np.float32)

    # Dibujar curva
    points = np.stack([x_px, y_px], axis=1).astype(np.int32)
    for i in range(len(points) - 1):
        cv2.line(image, tuple(points[i]), tuple(points[i+1]), 1.0, line_width)

    return image
```

### B.2 InfoNCE Loss para Audio-Visual

```python
import torch
import torch.nn.functional as F

class AudioVisualInfoNCELoss(nn.Module):
    """
    InfoNCE loss para alinear representaciones de audio y visual.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_audio, z_visual):
        """
        Args:
            z_audio: (batch, z_dim) - embeddings de audio
            z_visual: (batch, z_dim) - embeddings visuales

        Returns:
            loss: Escalar
        """
        # Normalizar
        z_audio = F.normalize(z_audio, dim=-1)
        z_visual = F.normalize(z_visual, dim=-1)

        batch_size = z_audio.shape[0]

        # Matriz de similitud
        logits = torch.mm(z_audio, z_visual.t()) / self.temperature

        # Labels: diagonal (pares correspondientes)
        labels = torch.arange(batch_size, device=z_audio.device)

        # Cross-entropy en ambas direcciones
        loss_a2v = F.cross_entropy(logits, labels)
        loss_v2a = F.cross_entropy(logits.t(), labels)

        return (loss_a2v + loss_v2a) / 2
```

---

*Documento preparado para revisión interna*
*PHIDEUS Project - Enero 2026*
