# ROADMAP — Escalon 2: Speech ↔ EGG Cross-Modal Alignment

> Fecha de creacion: 2026-03-06
> Estado: S2-P0 COMPLETE, S2-P1 COMPLETE, S2-P2-control COMPLETE, S2-P2-main CONCAT COMPLETE, S2-P2.5 INTERPRETED, S2-P2.5b PCA RUNNING

> [!IMPORTANT]
> **Addendum operativo (2026-03-12):** este roadmap ya quedó superado por la ejecución real del frente. `S2-P0` y `S2-P1` están completos; `S2-P2-control` (`D0`) ya cerró con `S=77.8% @ ep25`, `CI=[72.0%, 80.8%]`; `S2-P2-main` por concatenación también ya cerró (`V4-lin=67.8%`, `H-series=59.8%`, `A4-16k=77.8%=D0`); y `S2-P2.5` ya no está corriendo sino **interpretado** en sus `6/6` celdas: `V4-lin-attnbias=70.6%`, `V4-lin-xattn=77.0%`, `H-series-attnbias=78.0%`, `H-series-xattn=73.4%`, `A4-16k-attnbias=77.8%`, `A4-16k-xattn(30ep)=78.0%`. La lectura preregistrada dejó un patrón operacional claro: ningún brazo attention-based superó a `D0` con lift defendible, `V4-lin + attn_bias` fue significativamente peor y la interacción descriptor × mecanismo siguió siendo visible. La fase inmediata ya cambió otra vez: `S2-P2.5b` abre `proj_cond / pca` como contraste mecanístico final, con `V4-lin-pca` corriendo y `H-series-pca` / `A4-16k-pca` en cola. La rama `A10d/A10e` sigue existiendo como posibilidad técnica adyacente, pero no integra el contraste canónico de este corte. Usar [README.md](README.md) como estado canónico del frente, [S2_P2/plan_rectificacion_armonia_natural.md](S2_P2/plan_rectificacion_armonia_natural.md) como marco vivo de rectificación y [S2_P2/PREDICCIONES_EPISTEMOLOGICAS_P25.md](S2_P2/PREDICCIONES_EPISTEMOLOGICAS_P25.md) como preregistro interpretativo falsificable; este documento conserva el desarrollo detallado y los guardrails de apertura.

---

## 1. Que es Escalon 2

### 1.1 Contexto dentro de Phideus

Phideus es un programa de investigacion sobre **Harmonic Information Theory**: la hipotesis de que los ratios de frecuencia constituyen un lenguaje universal que puede ser aprendido y transferido entre modalidades sensoriales distintas.

El programa tiene tres hipotesis centrales:

| Hipotesis | Estado | Evidencia |
|-----------|--------|-----------|
| **H1** — Las senales contienen distribuciones de ratios estructuradas | VALIDADA | Distribuciones no aleatorias en audio y vibracion |
| **H2** — Redes neuronales pueden aprender estas distribuciones | VALIDADA | val_loss < 0.5 en multiples arquitecturas |
| **H3** — La informacion de ratios es transferible entre modalidades | EN INVESTIGACION | H3a (audio↔MIDI): evidencia causal. H3b (speech↔EGG): este escalon |

El **Escalon 1** (brazo neural, aka BIAS_CONTROL) trabajo con **Audio ↔ MIDI** (musica de piano). Los resultados fueron:

- Descriptores de ratios mejoran retrieval cross-modal en **+9.4pp** sobre baseline (Test 02, causal)
- Reorganizan la geometria del embedding (+82% CKA cross-encoder, Test 06)
- Multi-seed record: d4a4 = **84.1% ±2.3pp** (5 seeds × 4 descriptores en supercomputadora)
- Hallazgo central: la ventaja es **geometrica** (reorganizacion de distancias), no de decodificabilidad individual (Test 13G-B: ranking invertido)

**Escalon 2 = primera prueba fuera de musica.** Si la representacion relacional funciona entre Speech y EGG (dos sensores del mismo fenomeno vocal), eso refuerza enormemente la hipotesis H3 de universalidad.

### 1.2 Que es Speech y que es EGG

**Speech** (voz): senal capturada por un microfono. Contiene la vibracion de las cuerdas vocales **filtrada** por el tracto vocal (faringe, boca, nariz). La forma del tracto vocal crea resonancias llamadas **formantes** (F1, F2, F3...) que definen los sonidos del habla (vocales, consonantes). Frecuencias utiles: 50 Hz a 8 kHz.

**EGG** (electroglotograma): senal capturada por dos electrodos colocados en el cuello, a ambos lados de la laringe. Mide la **impedancia electrica** entre los electrodos, que varia segun el area de contacto de los pliegues vocales. Cuando los pliegues se tocan (fase cerrada del ciclo glotal), la impedancia baja; cuando se separan (fase abierta), la impedancia sube. El EGG captura directamente la vibracion laringea **sin** el filtrado del tracto vocal.

**Lo que comparten**: ambos miden el mismo fenomeno fisico — las cuerdas vocales vibrando. La frecuencia fundamental (F0, el "pitch" de la voz) es identica en ambas senales porque la fuente es la misma.

**Lo que difiere**: el speech incluye informacion del tracto vocal (formantes, fricativas, plosivas) que el EGG no tiene. El EGG tiene informacion sobre la dinamica de contacto glotal (cociente de apertura, forma del pulso glotal) que el speech pierde o distorsiona.

### 1.3 Por que Speech ↔ EGG y no otro par

1. **Mismo oscilador, sensores distintos**: a diferencia de Audio↔MIDI (donde MIDI es simbolico), aqui ambas senales son fisicas, continuas, y emergen del mismo sistema mecanico.
2. **F0 continuo**: en musica, el pitch esta cuantizado en semitonos (12 por octava). En voz, F0 varia continuamente. Es la primera oportunidad de trabajar con **ratios reales** (no cuantizados).
3. **Alineacion temporal perfecta**: las grabaciones son simultaneas (mismo reloj de digitalizacion), eliminando ambiguedades de alineacion.
4. **Disponibilidad de datos**: el dataset French Lombard es publico, bien documentado, y tiene 38 hablantes con 4 condiciones de ruido.

### 1.4 Hipotesis formal

**H3b**: La representacion relacional (descriptores basados en ratios de frecuencia) puede transferirse a dos sensores fisicos distintos del mismo fenomeno vocal, superando una baseline lineal de retrieval cross-modal.

En concreto: si inyectamos descriptores de F0 ratios en encoders neurales de Speech y EGG, la metrica de retrieval S debe superar al D0 neural (sin descriptores), de la misma forma que los descriptores mejoraron Audio↔MIDI en el Escalon 1.

---

## 2. El Dataset: French Lombard v1.1

### 2.1 Origen

- **Nombre**: French Lombard Speech and EGG Database
- **Version**: v1.1 (record Zenodo 17340497)
- **Licencia**: CC BY-NC-SA 4.0
- **Tamano**: ~17 GB (descomprimido)
- **Ubicacion local**: `data/lombard/FLombard/`

### 2.2 Contenido

| Propiedad | Valor |
|-----------|-------|
| Hablantes | 38 (20 mujeres, 18 hombres) |
| Edad | 18-57 anos |
| Idioma | Frances |
| Clips totales | 9,120 |
| Duracion media por clip | 7.90 s |
| Duracion total | 20.0 horas |
| Duracion min/max por clip | 3.80 s / 9.37 s |
| Condiciones de ruido | 4 (noise0=silencio, noise1=65dB, noise2=75dB, noise3=85dB) |
| Clips por hablante | 240 (60 por condicion) |
| Sentencias | 60 por hablante, 3 sesiones |
| Sample rate (procesado) | 16 kHz |
| Canales | Mono (speech y EGG en archivos separados) |

### 2.3 Estructura de archivos

```
data/lombard/FLombard/
├── process/
│   ├── wav/                  # Speech a 16 kHz
│   │   ├── 01/               # Speaker 01
│   │   │   ├── 01_s1_l12_sen10_noise0.wav
│   │   │   ├── 01_s1_l12_sen10_noise1.wav
│   │   │   └── ...
│   │   ├── 02/
│   │   └── ...
│   └── egg/                  # EGG a 16 kHz
│       ├── 01/
│       │   ├── 01_s1_l12_sen10_noise0.wav
│       │   └── ...
│       └── ...
├── raw/                      # Originales a 44.1 kHz (no usados)
├── txt/                      # Transcripciones
├── calibration/              # Calibracion de los sensores
└── speakers.txt              # Metadata de hablantes
```

**Convencion de nombres**: `{speaker_id}_s{session}_l{list_id}_sen{sentence_num}_{noise_condition}.wav`

Ejemplo: `14_s2_l4_sen5_noise2.wav` = speaker 14, sesion 2, lista 4, sentencia 5, condicion de ruido 2 (75 dB).

Los archivos de speech y EGG del mismo clip tienen **exactamente el mismo nombre** pero estan en directorios diferentes (`process/wav/` vs `process/egg/`). Ambos son mono, 16 kHz, y tienen exactamente el mismo numero de frames (verificado en los 9,120 pares).

### 2.4 Condiciones de ruido

Los hablantes fueron grabados mientras escuchaban ruido a traves de auriculares a diferentes niveles. Esto provoca el **efecto Lombard**: los hablantes modifican su voz (mas fuerte, pitch mas alto, articulacion mas clara) para compensar el ruido percibido. El ruido no esta en la grabacion — solo afecta al comportamiento del hablante.

| Condicion | Ruido en auriculares | Efecto en la voz |
|-----------|---------------------|-------------------|
| noise0 | Silencio (sin ruido) | Habla normal |
| noise1 | 65 dB SPL | Ligero aumento de volumen |
| noise2 | 75 dB SPL | Aumento notable de F0 y volumen |
| noise3 | 85 dB SPL | Efecto Lombard fuerte |

**Piloto**: Solo usamos condicion noise0 (silencio). Las otras condiciones se agregan en la fase P2.5 con metricas estratificadas.

### 2.5 Efecto Lombard — por que importa

El efecto Lombard cambia la voz de formas sistematicas que afectan tanto al speech como al EGG: F0 sube, amplitud sube, duracion de vocales cambia, espectro se inclina. Si mezclamos condiciones de ruido desde el principio, no sabemos si el modelo esta aprendiendo alineacion Speech↔EGG o simplemente clasificando condicion de ruido (que es un confound fuerte). Por eso el piloto usa solo noise0, y despues se agregan las condiciones con metricas separadas por condicion.

---

## 3. Protocolo Canonico

El protocolo se fija en la fase P0 y es **inmutable** en todas las fases posteriores. Cualquier cambio invalida la comparabilidad entre fases.

### 3.1 Parametros de senal

| Parametro | Valor | Justificacion |
|-----------|-------|---------------|
| Sample rate | 16,000 Hz | Archivos procesados del dataset. Cubre F0 (50-500 Hz) + formantes (hasta 8 kHz) |
| Ventana (segmento) | 2.0 s = 32,000 muestras | Suficiente material voiced en sentencias de 3-9 s |
| Hop | 0.5 s = 8,000 muestras | ~4-15 segmentos por clip, solapamiento del 75% |
| STFT n_fft | 1,024 | Resolucion frecuencial: 15.625 Hz/bin |
| STFT hop_length | 256 | ~62 frames por segundo de senal |

### 3.2 Definicion de positivo

El **positivo canonico** es: la misma ventana temporal del mismo clip, pero de la otra modalidad.

Si el query es `speech[clip_X, 1.0s:3.0s]`, el positivo es `egg[clip_X, 1.0s:3.0s]`.

Esto es posible porque speech y EGG se graban simultaneamente con el mismo reloj de digitalizacion. No hay lag sistematico (verificado en P0: lag_correction_samples = 0).

### 3.3 Segmentacion determinista

La segmentacion se genera con una regla determinista versionada:

```python
def segment_windows(duration_sec, seg_len=2.0, hop=0.5):
    windows = []
    t = 0.0
    while t + seg_len <= duration_sec:
        windows.append((t, t + seg_len))
        t += hop
    return windows
```

Esta funcion genera exactamente las mismas ventanas cada vez que se llama con la misma duracion. Garantiza que P1, P2c, P2m y P2.5 usan exactamente la misma poblacion de segmentos.

### 3.4 Split por hablante

La division train/val/test es por **hablante** (no por clip ni por segmento). Esto evita data leakage: si fragmentos del mismo hablante aparecen en train y test, el modelo puede reconocer al hablante en vez de aprender alineacion cross-modal.

| Split | Hablantes | Genero | Clips | Segmentos (total) | Segmentos (noise0) |
|-------|-----------|--------|-------|-------------------|-------------------|
| Train | 28 | 15F / 13M | 6,720 | 79,548 | 19,910 |
| Validation | 5 | 2F / 3M | 1,200 | 14,493 | 3,624 |
| Test | 5 | 3F / 2M | 1,200 | 14,495 | 3,629 |
| **Total** | **38** | **20F / 18M** | **9,120** | **108,536** | **27,163** |

Hablantes por split:
- **Train**: 01, 03, 05, 07, 09, 10, 11, 12, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38
- **Validation**: 06, 08, 13, 19, 31
- **Test**: 02, 04, 14, 20, 27

Asignacion con seed=42, balanceo por genero.

### 3.5 Metricas

| Metrica | Definicion | Uso |
|---------|-----------|-----|
| **S** | min(S2E@10, E2S@10) | **Metrica primaria**. Toma el peor de los dos sentidos de retrieval |
| S2E@10 | Recall@10 de Speech→EGG | Dado un query de speech, ¿el EGG correcto esta en el top 10? |
| E2S@10 | Recall@10 de EGG→Speech | Dado un query de EGG, ¿el speech correcto esta en el top 10? |
| Pool size | 128 | Cada query compite contra 127 negativos + 1 positivo |
| R@10 random | 10/128 = **7.8%** | Baseline de azar con pool de 128 |
| CI | Grouped bootstrap por speaker, 1000 resamples | Compensa la correlacion intra-hablante en los queries |

**¿Por que min(S2E, E2S)?** El sentido mas facil puede inflar la metrica. Tomar el minimo fuerza al modelo a funcionar bien en ambas direcciones.

**¿Por que grouped bootstrap?** Con solo 5 hablantes en test, si hacemos bootstrap naif por query, los intervalos de confianza son optimistas porque los queries del mismo hablante estan correlacionados (comparten F0, timbre, etc.). El bootstrap agrupado resamplea hablantes completos, dando CIs mas honestos.

### 3.6 Pool estructurado con hard negatives

El pool de candidatos no es aleatorio. Tiene una jerarquia de negativos duros que testea confounds especificos:

| Estrato | N por query | Descripcion | Confound que testea |
|---------|-------------|-------------|---------------------|
| **Positivo** | 1 | Misma ventana del mismo clip, otra modalidad | — |
| **L1** (mas duro) | hasta 16 | Mismo clip, distinta ventana (>=2s de separacion) | ¿Resuelve por identidad de clip/grabacion? |
| **L2** | hasta 16 | Mismo hablante, distinta sentencia | ¿Resuelve por identidad de hablante/timbre? |
| **L3** | hasta 16 | Distinto hablante, misma sentencia | ¿Resuelve por contenido verbal? |
| **L4** (facil) | resto (~80) | Distinto hablante, distinta sentencia | Baseline. El mas facil de distinguir |

**¿Por que importa L1?** Si un modelo obtiene S=80% pero todo viene de superar a L4 (negativos faciles), mientras falla contra L1 (mismo clip, distinta ventana), la senal es espuria: el modelo reconoce la "huella" acustica del clip, no alinea el contenido temporal. Un modelo que realmente alinea speech↔EGG debe superar tambien a L1.

**Restriccion de L1**: las ventanas deben tener separacion temporal >= 2.0s para que no compartan contenido solapado.

**Nota sobre L3**: En el test set, solo 10 sentence_ids son compartidos entre pares de hablantes (02 & 20 comparten 10, 04 & 27 comparten 10). Esto da un promedio de solo 2.0 negativos L3 por query. Es sparse pero aceptable — L1 y L2 son los estratos mas informativos.

### 3.7 Epocas y sizing

Una epoca = un barrido completo del dataset de train. Con noise0 piloto:
- 19,910 segmentos de train
- Con batch_size=64: 311 batches/epoca
- NO se repiten batches artificialmente. Si el dataset es pequeno, una epoca es corta.

---

## 4. Manifest y Segment Index

### 4.1 Manifest clip-level (`data/lombard/manifest.json`)

Un registro por clip original. 9,120 entradas.

Campos de cada clip:

```json
{
  "clip_id": "01_s1_l12_sen10_noise0",
  "speaker_id": "01",
  "gender": "Female",
  "age": 33,
  "noise_condition": "noise0",
  "session": 1,
  "list_id": 12,
  "sentence_num": 10,
  "sentence_id": 119,
  "speech_path": "process/wav/01/01_s1_l12_sen10_noise0.wav",
  "egg_path": "process/egg/01/01_s1_l12_sen10_noise0.wav",
  "duration_sec": 5.3214,
  "split": "train"
}
```

**sentence_id**: Identificador numerico normalizado de la sentencia (no el texto del transcript). Permite identificar cuando dos hablantes dicen "la misma cosa" para el estrato L3 de hard negatives.

### 4.2 Segment index (`data/lombard/segment_index.json`)

Derivado deterministicamente del manifest por la regla de segmentacion. 108,536 entradas.

Campos de cada segmento:

```json
{
  "clip_id": "01_s1_l12_sen10_noise0",
  "segment_idx": 0,
  "start_sec": 0.0,
  "end_sec": 2.0,
  "speaker_id": "01",
  "gender": "Female",
  "noise_condition": "noise0",
  "sentence_id": 119,
  "split": "train"
}
```

Metadata del indice:
- `protocol_version`: "s2-p0-v1"
- `segmentation_rule`: "segment_windows(duration, seg_len=2.0, hop=0.5)"
- `lag_correction_samples`: 0

### 4.3 Alignment audit (`data/lombard/alignment_audit.json`)

Verifica que speech y EGG estan temporalmente alineados:

- **76 clips auditados** (muestra aleatoria)
- **Frame match**: 76/76 (100%). Todos los pares tienen exactamente el mismo numero de frames
- **Clipping**: 0 muestras con |x| > 0.99 en speech ni en EGG
- **Lag**: mediana = 0.0 ms. El metodo de cross-correlacion de envelopes es ruidoso (std=565ms), pero la identidad de frames es prueba definitiva de lag cero
- **lag_correction_samples**: 0 (no se aplica correccion)
- **Voiced fraction speech**: media=0.204, p10=0.149
- **Voiced fraction EGG**: media=0.354, p10=0.264
- **Voiced threshold**: 0.1494 (percentil 10 de voiced_fraction de speech en muestra de auditoria)

---

## 5. Fases del Experimento

### 5.1 Vision general

```
S2-P0  (datos + manifest + audit)                    [COMPLETE]
  |
  v
S2-P1  (baseline lineal: CCA + Ridge + retrieval)    [COMPLETE]
  |
  v
S2-P2-control-mini  (20 batches, VRAM/throughput)     [COMPLETE]
  |
  v
S2-P2-control  (D0 neural 30ep, noise0)              [COMPLETE — S=77.8% @ ep25]
  |
  v
S2-P2-main  (concat descriptors: V4-lin, H-series, A4-16k)  [COMPLETE — resultado negativo sobre mecanismo]
  |
  v
S2-P2.5  (attention-based injection: Factorial 3×2)  [INTERPRETED]
S2-P2.5b (conditioned projection / pca)              [RUNNING]
  |
  v
[DECISION: usuario decide con evidencia + PREDICCIONES pre-registradas]
  |
  v
S2-P3  (opcional: SOTA frozen encoder)               [CONCEPT]
```

Cada fase bloquea la siguiente. No se salta ninguna.

---

### 5.2 S2-P0: Data Ingestion + Manifest + Alignment Audit — COMPLETE

**Pregunta**: "¿Los datos estan bien? ¿Estan alineados? ¿Podemos segmentarlos deterministicamente?"

**Entregables producidos**:
1. Dataset descargado y extraido: `data/lombard/FLombard/` (v1.1, 17 GB)
2. Manifest clip-level: `data/lombard/manifest.json` (9,120 clips)
3. Segment index: `data/lombard/segment_index.json` (108,536 segmentos)
4. Alignment audit: `data/lombard/alignment_audit.json` (76 clips auditados)
5. Protocolo canonico congelado (sr=16kHz, seg=2s, hop=0.5s, pool=128, etc.)

**Script**: `experiments/bias_control/escalon2/s2_p0_manifest.py` (547 lineas)

**Hallazgos clave**:
- v1.1 tiene 38 speakers (vs 4 en v1.0). La v1.0 era una subset preliminar.
- Speech y EGG son archivos WAV separados (no canales stereo).
- 0 clips menores a 2s — todos generan al menos 4 segmentos.
- Alineacion perfecta: 9,120/9,120 pares con frames identicos, lag = 0.

---

### 5.3 S2-P1: Baseline Lineal — COMPLETE

**Pregunta**: "¿Speech y EGG comparten informacion cross-modal utilizable con features simples y metodos lineales?"

**Metodos**:

#### 5.3.1 Features extraidos (20 dimensiones por segmento)

Para cada ventana de 2 segundos (32,000 muestras a 16 kHz), se extraen 20 features tanto de speech como de EGG:

**8 band energies** — Media del log(1+|STFT|) por banda de frecuencia:

| Banda | Bins STFT | Frecuencias | Que captura |
|-------|-----------|-------------|-------------|
| 0 | 3-6 | 47-94 Hz | F0 masculino bajo |
| 1 | 6-12 | 94-188 Hz | F0 femenino, armonicos bajos |
| 2 | 12-24 | 188-375 Hz | Armonicos |
| 3 | 24-48 | 375-750 Hz | Formante F1 |
| 4 | 48-96 | 750-1500 Hz | F1 alto, F2 bajo |
| 5 | 96-192 | 1500-3000 Hz | Formantes F2, F3 |
| 6 | 192-384 | 3000-6000 Hz | Formantes altos |
| 7 | 384-513 | 6000-8000 Hz | Fricativas, ruido |

**8 band energy stds** — Desviacion estandar del log-magnitud por banda dentro de la ventana. Captura variabilidad temporal (transiciones foneticas).

**3 F0 stats** — Media, std y rango del pitch fundamental, estimados por autocorrelacion frame a frame (frames de 1024 muestras, hop 256, rango 50-500 Hz, umbral de voicing 0.3).

**1 voicing fraction** — Fraccion de frames con energia > 1% del maximo.

#### 5.3.2 CCA (Canonical Correlation Analysis)

CCA encuentra transformaciones lineales de ambas modalidades que maximizan la correlacion entre ellas. Es el metodo lineal optimo para alinear dos espacios de features.

1. Se entrena CCA con 10 componentes sobre los 19,910 pares de train (noise0)
2. Cada componente define un "eje" en speech-space y otro en egg-space que estan maximamente correlacionados
3. En test, se proyectan los features a este espacio compartido de 10 dimensiones
4. Se usa similitud coseno en el espacio CCA para retrieval

#### 5.3.3 Ridge Regression

Predice features de EGG desde features de Speech (y viceversa) con regularizacion L2 (alpha=1.0). R² mide cuanta varianza de una modalidad es predecible linealmente desde la otra. Es puramente diagnostico — no se usa para retrieval.

#### 5.3.4 Resultados

| Metodo | S2E@10 | E2S@10 | S | CI (grouped) | vs random |
|--------|--------|--------|---|--------------|-----------|
| Random | — | — | 7.8% | — | 1.0x |
| Raw cosine | 50.4% | 46.8% | **46.8%** | [38.0%, 54.5%] | **6.0x** |
| **CCA** | **68.4%** | **64.4%** | **64.4%** | **[57.8%, 70.2%]** | **8.2x** |

| Metodo auxiliar | Resultado |
|-----------------|-----------|
| Ridge R² Speech→EGG | **0.851** |
| Ridge R² EGG→Speech | 0.694 |

**CCA train correlations** (10 componentes):

| Comp | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|------|---|---|---|---|---|---|---|---|---|---|
| Corr | 0.975 | 0.940 | 0.920 | 0.836 | 0.698 | 0.654 | 0.572 | 0.487 | 0.382 | 0.311 |

**Hard negative strata** (promedio por query):

| Estrato | Promedio | Maximo |
|---------|----------|--------|
| L1 (mismo clip, diff ventana) | 6.1 | 16 |
| L2 (mismo speaker, diff utterance) | 16.0 | 16 |
| L3 (diff speaker, mismo sentence_id) | 2.0 | 16 |
| L4 (random) | 102.9 | — |

**Rank analysis**: mediana rank S2E = 9.0, E2S = 12.0 (en pool de 128).

#### 5.3.5 Interpretacion de los resultados

1. **Senal masiva**: CCA S=64.4% con CI inferior en 57.8%, ambos enormemente por encima de azar (7.8%). Con features de 20 dimensiones y un metodo lineal. Speech y EGG comparten informacion espectral de forma extraordinariamente fuerte.

2. **Ridge R²=0.851 (S→E)**: El 85% de la varianza de los features del EGG es predecible linealmente desde los features del speech. Esto es esperado: la fuente (cuerdas vocales) es la misma, y las bandas de frecuencia baja-media (que dominan el F0 y los primeros armonicos) son casi identicas.

3. **Asimetria S2E > E2S**: Speech contiene mas informacion (formantes, fricativas) que el EGG no tiene. Predecir EGG desde speech es mas facil que al reves.

4. **Top 3 CCA correlations > 0.92**: Existen al menos 3 ejes lineales en los que speech y EGG son casi identicos. Probablemente corresponden a: (1) F0/energia global, (2) distribucion espectral baja, (3) patron de voicing.

5. **Comparacion con Escalon 1**: En Audio↔MIDI, el D0 neural con ~74M parametros y VICReg obtuvo S=75.2%. Aqui, con 20 dims y CCA lineal, ya estamos en 64.4%. La senal cross-modal Speech↔EGG es mucho mas fuerte que Audio↔MIDI.

**Script**: `experiments/bias_control/escalon2/s2_p1_baseline_linear.py` (894 lineas)
**Resultados**: `data/lombard/p1_results/p1_results_noise0.json`
**Feature cache**: `data/lombard/p1_results/features_noise0.npz`

---

### 5.4 S2-P2-control-mini: Throughput Benchmark — COMPLETE

**Pregunta**: "¿El modelo entra en VRAM? ¿Cuanto tarda por epoca?"

**Estado al corte**: el mini-run ya cumplio su funcion operativa como precondicion del entrenamiento largo. El detalle vivo del frente ya no se lee aca sino en el [README canónico de Escalón 2](README.md), porque `S2-P2-control` ya fue lanzado.

Antes de lanzar un entrenamiento de 30 epocas, se corre un mini-run de 1 epoca con max 20 batches para verificar:

- VRAM usage con batch_size=64
- Batches por minuto estables
- Shapes correctas en todo el pipeline
- No OOM (Out Of Memory)

Si batch_size=64 no entra en 24 GB (RTX 3090), se reduce.

**Comando**:
```bash
python experiments/bias_control/escalon2/train_escalon2.py \
    --lombard-dir data/lombard/FLombard \
    --segment-index data/lombard/segment_index.json \
    --output data/lombard/d0_mini \
    --epochs 1 --batch-size 64 --max-batches 20 --seed 42
```

---

### 5.5 S2-P2-control: D0 Neural Baseline — COMPLETE

**Pregunta**: "¿Un modelo neural sin descriptores (D0) supera la baseline lineal CCA (S=64.4%)?"

**Estado al corte**: `S2-P2-control` ya quedó completo sobre `noise0` con `best S=77.8% @ ep25`, `CI=[72.0%, 80.8%]`. El corte temprano de `ep5` ya no debe leerse como estado vivo del frente; el documento canónico sigue siendo [README.md](README.md), y la fase vigente pasó a `S2-P2-main`.

#### 5.5.1 Arquitectura

Dos encoders **identicos** (speech y EGG usan la misma arquitectura pero con pesos independientes):

```
Waveform [B, 32000]  (2s @ 16kHz)
  → Conv1d(1, 256, k=10, s=5) → GroupNorm(16) → GELU   → [B, 256, 6400]
  → Conv1d(256, 256, k=3, s=2) → GroupNorm(16) → GELU  → [B, 256, 3200]
  → Conv1d(256, 256, k=3, s=2) → GroupNorm(16) → GELU  → [B, 256, 1600]
  → Conv1d(256, 512, k=3, s=2) → GroupNorm(16) → GELU  → [B, 512, 800]
  → Positional Embedding (learned, max 1000 positions)
  → TransformerEncoder(4 layers, 8 heads, d=512, ff=2048)
  → Mean pooling → [B, 512]
```

Cada encoder tiene **13.9M parametros**. Dos encoders + dos projection heads = **29.1M parametros** totales, todos entrenables from scratch.

**Projection heads**: MLP 512 → 512 → 256 (patron SimCLR/VICReg de 3 capas con BatchNorm y ReLU).

**Loss**: VICReg (Variance-Invariance-Covariance Regularization) con lambda_inv=10, lambda_var=10, lambda_cov=1. Es la misma loss usada exitosamente en Escalon 1.

VICReg tiene tres componentes:
- **Invariance** (MSE): empuja los embeddings de pares (speech, egg) a ser iguales
- **Variance** (hinge en std): previene el colapso (std de cada dimension > 1.0)
- **Covariance** (off-diagonal → 0): decorrelaciona las dimensiones del embedding

#### 5.5.2 Configuracion de entrenamiento

| Parametro | Valor |
|-----------|-------|
| Epocas | 30 |
| Batch size | 64 (ajustable si no entra en VRAM) |
| Optimizer | AdamW (weight_decay=0.01) |
| LR encoders | 5e-4 |
| LR projections | 1e-3 |
| Warmup | 500 steps lineales |
| Scheduler | Warmup → cosine annealing |
| Grad clip | 1.0 |
| Eval epochs | 5, 10, 15, 20, 25, 28, 29, 30 |
| Checkpoints | Cada epoca |

#### 5.5.3 Anti-ghost

**DriftSentinel**: despues de la primera epoca, verifica que al menos algunos parametros hayan cambiado. Si ningun parametro se movio, el entrenamiento es "fantasma" (frozen por accidente, gradientes a cero, etc.) y se aborta.

#### 5.5.4 Evaluacion durante entrenamiento

En las epocas marcadas (5, 10, 15, ...), se corre la evaluacion completa de retrieval con pool estructurado (misma logica que P1 pero con embeddings neurales). Esto produce S, CI, y detalles por query.

#### 5.5.5 Archivos

- **Encoder**: `src/bias_control/encoders/speech_egg_encoder.py` (80 lineas)
- **Dataset**: `src/bias_control/datasets/lombard_segments.py` (156 lineas)
- **Eval**: `experiments/bias_control/escalon2/eval_escalon2.py` (276 lineas)
- **Training**: `experiments/bias_control/escalon2/train_escalon2.py` (493 lineas)

**Comando**:
```bash
python experiments/bias_control/escalon2/train_escalon2.py \
    --lombard-dir data/lombard/FLombard \
    --segment-index data/lombard/segment_index.json \
    --output data/lombard/d0_seed42 \
    --epochs 30 --batch-size 64 --seed 42
```

#### 5.5.6 Que se espera

El D0 neural deberia superar la CCA baseline (S=64.4%) porque:
- Puede capturar relaciones no-lineales entre speech y EGG
- Tiene capacidad para aprender representaciones mucho mas ricas (256 dims vs 10)
- VICReg previene el colapso y fuerza diversidad

Si D0 neural NO supera CCA, eso indicaria que la relacion Speech↔EGG es fundamentalmente lineal, lo cual seria un hallazgo en si mismo.

---

### 5.6 S2-P2-main: Descriptor Concat — COMPLETE (resultado negativo sobre mecanismo)

> **Resultado (2026-03-09):** Fase cerrada. V4-lin=67.8% (-10pp vs D0), H-series=59.8% (-18pp, colapso ep8), A4-16k=77.8% (=D0). La inferencia válida no es "la armonía natural falló", sino: la concatenación trata al descriptor como feature adicional y no es el mecanismo adecuado para la tesis fuerte. El frente pasó a S2-P2.5 (attention-based injection). Ver `plan_rectificacion_armonia_natural.md` para la lectura completa.

**Solo se ejecuta despues de que P2-control establezca S_control.**

**Pregunta**: "¿Un descriptor basado en ratios de F0 mejora S sobre D0?"

#### 5.6.1 V4: F0 Ratio Descriptor (4 dims)

Primera vez con ratios reales continuos (en Escalon 1, los ratios venian de MIDI cuantizado a semitonos).

Para cada frame voiced en la ventana:
1. Extraer contorno F0 con PYIN (speech) o autocorrelacion (EGG)
2. Calcular:
   - `log2_ratio_prev = log2(F0[t] / F0[t-1])` — ratio con frame anterior
   - `log2_ratio_next = log2(F0[t+1] / F0[t])` — ratio con frame siguiente
   - `voicing_strength` — confianza del estimador de F0
   - `period_regularity` — regularidad del periodo glotal
3. Output: `[B, T', 4]` — 4 dimensiones por frame temporal

**V4 se inyecta en AMBOS encoders** porque ambas modalidades tienen F0 (a diferencia de Escalon 1 donde A4 iba en el audio encoder y D4 en el MIDI encoder).

#### 5.6.2 A4 adaptado a 16 kHz

Variante nueva de A4 (descriptor de banda espectral del Escalon 1) adaptada a 16 kHz, con bandas de frecuencia recalculadas para el rango 0-8 kHz.

#### 5.6.3 Inyeccion

Se usa el patron de concatenacion + linear del Escalon 1 (Gate 4.2):
```
CNN features [B, T', 512] + descriptor [B, T', D]
  → concat → Linear(512+D, 512) → LayerNorm → Transformer
```

#### 5.6.4 Arms de screening (3 epocas cada uno)

| Arm | Descriptor | Dims | Pregunta |
|-----|-----------|------|----------|
| D0 | ninguno | 0 | Control |
| V4 | F0 ratios | 4 | ¿Los ratios de F0 aportan? |
| A4-16k | Band energy deltas | 8 | ¿El patron espectral aporta? |
| V4+A4 | combinado | 12 | ¿Son complementarios? |

El ganador del screening se entrena por 30 epocas junto con D0.

#### 5.6.5 Caching de V4

PYIN/autocorrelacion puede ser lento on-the-fly. Si tarda >10ms/segmento, se precomputa V4 por segmento y se cachea en disco (`data/lombard/v4_cache/`).

---

### 5.7 S2-P2.5: Attention-Based Injection — FACTORIAL 3×2 INTERPRETED

La fase canónica ya leída del frente. Descriptores se inyectan como principios organizacionales de atención, no como features concatenadas.

#### Resultados completos (6 arms)

| Arm | Descriptor | Familia | Mecanismo | Best S | Epoch | Δ vs D0 | Δ vs concat |
|-----|-----------|---------|-----------|--------|-------|---------|-------------|
| V4-lin-attnbias | Ratios lineales F0 | A (dinámica temporal) | Attention bias | **70.6%** | 25 | -7.2pp | +2.8pp |
| V4-lin-xattn | Ratios lineales F0 | A (dinámica temporal) | Cross-attention | **77.0%** | 15 | -0.8pp | +9.2pp |
| H-series-attnbias | Armónicos relativos | B (armónica intra-frame) | Attention bias | **78.0%** | 29 | +0.2pp | **+18.2pp** |
| H-series-xattn | Armónicos relativos | B (armónica intra-frame) | Cross-attention | **73.4%** | 29 | -4.4pp | +13.6pp |
| A4-16k-attnbias | Dinámica espectral | C (control no-ratio) | Attention bias | **77.8%** | 20 | +0.0pp | +0.0pp |
| A4-16k-xattn | Dinámica espectral | C (control no-ratio) | Cross-attention | **78.0%** | 25 | +0.2pp | +0.2pp |

Hallazgos clave del factorial:
- la transición concat → attention quedó validada como hipótesis de mecanismo en las familias A y B;
- pero esa recuperación no alcanzó para producir lift defendible sobre `D0` una vez aplicada la lectura preregistrada;
- `V4-lin-xattn` recupera gran parte de la caída de concat, lo que vuelve interpretable la interacción descriptor × mecanismo;
- `A4-16k` ya cerró sus dos mecanismos comparables y deja un control no-ratio completamente emparejado.

#### Factorial 3×2

| Descriptor | Familia | attn_bias | xattn |
|-----------|---------|-----------|-------|
| V4-lin | A | `70.6%` | `77.0%` |
| H-series | B | `78.0%` | `73.4%` |
| A4-16k | C | `77.8%` | `78.0%` |

El factorial permite separar: efecto descriptor (promediando mecanismos), efecto mecanismo (promediando descriptores), e interacción. La fase canónica activa ahora es su **lectura**, no su corrida.

**Preregistro interpretativo**: `S2_P2/PREDICCIONES_EPISTEMOLOGICAS_P25.md` contiene la regla operativa de comparación (bootstrap pareado sobre Δ, CI_Δ, umbral 2pp) y la matriz de predicciones pre-registrada.

**Referencia técnica**: `S2_P2/Discusion_Inyeccion_descriptores.md` documenta el diseño de los mecanismos de inyección.

#### Interpretación estadística ya completada

| Arm | Best `S` | Δ vs `D0` | CI_Δ (95%) | Declaración |
|-----|----------|-----------|------------|-------------|
| `V4-lin-attnbias` | `70.6%` | `-7.2pp` | `[-10.8, -1.8]` | `D0 > arm` |
| `V4-lin-xattn` | `77.0%` | `-0.8pp` | `[-4.7, +4.1]` | `≈ D0` |
| `H-series-attnbias` | `78.0%` | `+0.2pp` | `[-3.1, +4.5]` | `≈ D0` |
| `H-series-xattn` | `73.4%` | `-4.4pp` | `[-6.5, +0.2]` | `≈ D0` |
| `A4-16k-attnbias` | `77.8%` | `+0.0pp` | `[-2.8, +1.9]` | `≈ D0` |
| `A4-16k-xattn` | `78.0%` | `+0.2pp` | `[-3.4, +4.6]` | `≈ D0` |

Lectura operativa:
- `P4` quedó matched en el sentido prudente: los mecanismos attention-based testeados no mejoraron Speech↔EGG retrieval sobre `D0`;
- no es un null plano: `V4-lin + attn_bias` es significativamente peor y la interacción descriptor × mecanismo sigue siendo informativa;
- por eso el siguiente contraste canónico no es `A10d/A10e` ni `S2-P3`, sino `proj_cond / pca`.

#### S2-P2.5b: Conditioned Projection (PCA / FiLM) — RUNNING

Motivación: `pca` fue el mecanismo audio-side más promisorio de Escalón 1 (`82.6%` vs `79.2%` control) y deja intacto el encoder. En Escalón 2 se usa como chequeo mecanístico liviano: si también queda `≈ D0`, el null mecanístico de este frente queda mucho más cerrado.

| Arm | Descriptor | Estado |
|-----|------------|--------|
| `V4-lin-pca` | Familia A | **Corriendo** |
| `H-series-pca` | Familia B | Pendiente secuencial |
| `A4-16k-pca` | Familia C | Pendiente secuencial |

---

### 5.8 S2-P3: SOTA Frozen Encoder — CONCEPT

Solo despues de cerrar `S2-P2.5b`. Usar WavLM o HuBERT frozen como speech encoder (analogo a MERT-330M en Gate 7.1a del Escalon 1), con encoder pequeno para EGG. No se disena hasta tener lectura final de los cuatro mecanismos (`concat`, `attn_bias`, `xattn`, `pca`).

---

## 6. Archivos del Proyecto

### 6.1 Codigo

| Archivo | Lineas | Fase | Proposito |
|---------|--------|------|-----------|
| `experiments/bias_control/escalon2/__init__.py` | 2 | P0 | Module init |
| `experiments/bias_control/escalon2/s2_p0_manifest.py` | 547 | P0 | Manifest + segment index + alignment audit |
| `experiments/bias_control/escalon2/s2_p1_baseline_linear.py` | 894 | P1 | Features + CCA + Ridge + retrieval eval |
| `experiments/bias_control/escalon2/eval_escalon2.py` | 276 | P2 | Pool builder + retrieval para modelos neurales |
| `experiments/bias_control/escalon2/train_escalon2.py` | 493 | P2 | Training loop D0 neural |
| `experiments/bias_control/escalon2/train_escalon2_pca.py` | nuevo | P2.5b | Conditioned projection / FiLM para Escalón 2 |
| `src/bias_control/datasets/lombard_segments.py` | 156 | P2 | Dataset loader PyTorch |
| `src/bias_control/encoders/speech_egg_encoder.py` | 80 | P2 | Encoder CNN+Transformer 16kHz |
| **Total** | **2,448+** | | |

### 6.2 Datos generados

| Archivo | Tamano | Fase |
|---------|--------|------|
| `data/lombard/FLombard/` | 17 GB | P0 |
| `data/lombard/manifest.json` | 3.8 MB | P0 |
| `data/lombard/segment_index.json` | 27 MB | P0 |
| `data/lombard/alignment_audit.json` | 35 KB | P0 |
| `data/lombard/p1_results/p1_results_noise0.json` | 3 KB | P1 |
| `data/lombard/p1_results/features_noise0.npz` | ~40 MB | P1 |
| `data/lombard/p25_interpretation/p25_full_results.json` | ~15 KB | P2.5 |

### 6.3 Reutilizacion del Escalon 1 (sin modificaciones)

| Componente | Archivo | Que hace |
|-----------|---------|---------|
| VICRegLoss | `src/RNA/vicreg.py` | Loss function para aprendizaje cross-modal |
| ProjectionHead | `src/bias_control/encoders/projection.py` | MLP de proyeccion (encoder output → embedding space) |
| DriftSentinel | `src/bias_control/training/preflight.py` | Detecta parametros congelados por accidente |

### 6.4 Documentos

| Documento | Ubicacion |
|-----------|-----------|
| Copia del plan | `Documents/01_FRENTES_ACTIVOS/ESCALON_2/PLAN_IMPLEMENTACION_ESCALON2.md` |
| Este roadmap | `Documents/01_FRENTES_ACTIVOS/ESCALON_2/ROADMAP_ESCALON_2.md` |
| README canónico del frente | `Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md` |
| Rectificación de armonía natural | `Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/plan_rectificacion_armonia_natural.md` |
| Preregistro interpretativo P2.5 | `Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/PREDICCIONES_EPISTEMOLOGICAS_P25.md` |

---

## 7. Correcciones Metodologicas Incorporadas

El plan de implementacion paso por 4 rondas de revision con Codex. Las correcciones mas importantes:

1. **Baseline lineal PRIMERO** (ronda 1): no saltar directo a modelos neurales. P1 establece el piso antes de P2.

2. **Protocolo canonico congelado en P0** (ronda 2): sr, ventana, hop, definicion de positivo — todo fijo antes de cualquier experimento. Sin esto, los resultados de P1 y P2 no son comparables.

3. **Eval harness es codigo NUEVO** (ronda 2): `evaluate_structured_pool.py` del Escalon 1 esta hardcodeado a piece/composer/audio+midi. No se adapta — se reescribe.

4. **Piloto limpio primero** (ronda 2): no mezclar condiciones de ruido desde el dia 1. Primero noise0, despues agregar las demas con metricas por condicion.

5. **R@10 random = 7.8%, NO 0.78%** (ronda 3): el calculo correcto con pool=128 y k=10 es 10/128 = 7.8%.

6. **L1 es el hard negative mas importante** (ronda 3): sin "mismo clip / distinta ventana", el modelo puede resolver identidad de clip y reportar metricas infladas.

7. **Segment index determinista** (ronda 3): no alcanza con un manifest clip-level. Se necesita un indice window-level que genere exactamente las mismas ventanas en P1, P2c, P2m y P2.5.

8. **CI grouped por speaker** (ronda 4): con 5 test speakers, bootstrap naif es demasiado optimista. Se resamplea por speaker completo.

9. **Epoca = full pass** (ronda 4): no repetir batches artificialmente. Con 19,910 segmentos de train y bs=64, una epoca son ~311 batches.

10. **Lag correction versionado** (ronda 4): si se detecta lag, se incorpora al segment_index ANTES de congelar. En este caso: lag = 0, no se aplica correccion.

---

## 8. Proximo Paso Inmediato

**Fase activa: `S2-P2.5b` — conditioned projection / `pca`**.

**Factorial 3×2 EXECUTED**:
1. V4-lin-attnbias = 70.6%
2. V4-lin-xattn = 77.0%
3. H-series-attnbias = 78.0%
4. H-series-xattn = 73.4%
5. A4-16k-attnbias = 77.8%
6. A4-16k-xattn = 78.0%

**Lectura ya completada**: `S2-P2.5` produjo un patrón `P4` operativo: ningún brazo `attn_bias` / `xattn` superó a `D0` con lift defendible, `V4-lin + attn_bias` fue claramente peor y la interacción descriptor × mecanismo siguió siendo interpretable.

**Paso inmediato**: cerrar `V4-lin-pca`, `H-series-pca` y `A4-16k-pca`, correr `paired_grouped_bootstrap_ci_delta()` para esos 3 brazos contra `D0` y decidir recién entonces si el null mecanístico del frente queda suficientemente cerrado o si justifica rerun puntual / rama secundaria.

---

## 9. Glosario

| Termino | Significado |
|---------|-------------|
| **A4** | Descriptor de 8 dimensiones: temporal deltas del log-magnitud STFT en 8 bandas de octava (47-12000 Hz). Usado en Escalon 1 para inyeccion en audio encoder. |
| **CCA** | Canonical Correlation Analysis. Metodo lineal que encuentra proyecciones de dos conjuntos de variables que maximizan la correlacion entre ellos. |
| **CI** | Confidence Interval (intervalo de confianza). |
| **D0** | Baseline sin descriptores. Solo los encoders + projections + VICReg, sin inyeccion de informacion de ratios. |
| **D4** | Descriptor de 4 dimensiones para MIDI: intervalo con nota anterior, intervalo con nota siguiente, ratio de duracion, diferencia de velocidad. |
| **EGG** | Electroglotograma. Senal que mide la impedancia electrica entre electrodos en el cuello, reflejando el contacto de los pliegues vocales. |
| **F0** | Frecuencia fundamental. La frecuencia de vibracion de las cuerdas vocales. Determina el "pitch" percibido de la voz. Tipicamente 80-300 Hz en habla. |
| **Familia A** | Familia de descriptores: dinámica temporal del oscilador. Mide cambios locales de F0 entre frames. Descriptores: V4-lin, V4-log. Testea si la evolución temporal del oscilador contiene invariantes cross-modales. |
| **Familia B** | Familia de descriptores: estructura armónica natural intra-frame. Mide relaciones entre armónicos (H2/H1..H6/H1). Descriptores: H-series. Testea la tesis fuerte de HIT: si la serie armónica física es un organizador privilegiado. |
| **Familia C** | Familia de descriptores: controles no-ratio (espectrales genéricos). Mide dinámica espectral por bandas sin referencia a F0 ni ratios. Descriptores: A4-16k. Control adversario. |
| **Familia D** | Familia de descriptores: variantes perceptuales/logarítmicas. Versiones log2 de las mismas magnitudes físicas de Familia A. Descriptores: V4-log. Testea sesgo representacional (lineal vs log2), no armonía natural. |
| **FiLM** | Feature-wise Linear Modulation. Tecnica de condicionamiento: gamma * features + beta, donde gamma y beta se generan desde una senal de condicionamiento. |
| **Formantes** | Resonancias del tracto vocal. F1, F2, F3 son los primeros tres formantes. Determinan la identidad de las vocales. |
| **Hard negative** | Negativo dificil de distinguir del positivo. Ejemplo: mismo hablante diciendo otra cosa (L2) — tiene el mismo timbre, F0 similar, pero contenido diferente. |
| **Hop** | Desplazamiento entre ventanas consecutivas. Con hop=0.5s y ventana=2s, las ventanas se solapan en 75%. |
| **L1-L4** | Estratos de dificultad en el pool de negativos. L1=mas duro (mismo clip), L4=mas facil (random). |
| **Lombard effect** | Cambio involuntario en la voz cuando el hablante percibe ruido ambiental: aumenta volumen, pitch, y claridad articulatoria. |
| **PREDICCIONES** | Artefacto de preregistro interpretativo en `S2_P2/PREDICCIONES_EPISTEMOLOGICAS_P25.md`. Contiene la regla operativa de comparación (bootstrap pareado sobre Δ) y la matriz de predicciones pre-registrada para P2.5. |
| **PYIN** | Probabilistic YIN. Algoritmo de estimacion de F0 que modela la incertidumbre del pitch. |
| **R@k** | Recall at k. Fraccion de queries para las cuales el positivo esta entre los top k candidatos del pool. |
| **Ridge** | Regresion Ridge. Regresion lineal con regularizacion L2 (penaliza coeficientes grandes). |
| **S** | Metrica primaria = min(S2E@10, E2S@10). Toma el peor de los dos sentidos de retrieval. |
| **STFT** | Short-Time Fourier Transform. Descompone la senal en tiempo y frecuencia. |
| **V4** | Descriptor de F0 ratios (4 dims): log2_ratio_prev, log2_ratio_next, voicing_strength, period_regularity. Propuesto para Escalon 2. |
| **VICReg** | Variance-Invariance-Covariance Regularization. Loss function para aprendizaje de representaciones que evita el colapso sin negativos explicitos. |
| **Voiced** | Region de la senal donde las cuerdas vocales estan vibrando (vocales, consonantes sonoras). Opuesto: unvoiced (fricativas sordas, silencios). |
