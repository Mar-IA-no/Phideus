Perfecto. Te dejo un **plan de implementación y ejecución del Escalón 1 (MAESTRO: Audio ↔ MIDI)** diseñado para que sea **científicamente robusto**, con **gates GO/NO-GO**, controles negativos obligatorios (para no repetir el autoengaño del coseno), y una ruta clara para llegar a una “prueba Rosetta” convincente.

Contexto clave: MAESTRO tiene \~200 horas de piano con **audio y MIDI alineados \~3 ms**, segmentado por piezas, con metadata (compositor/título/año) y splits train/val/test pensados para que la misma composición no se cuele en varios subconjuntos. ([Magenta](https://magenta.withgoogle.com/datasets/maestro?utm_source=chatgpt.com))

---

# **Objetivo del Escalón 1**

Demostrar (sin ambigüedad) que **es posible** aprender una representación compartida cross-modal entre:

* **Audio real** (grabación de piano con toda su complejidad armónica/ruido/ambiente)  
* **MIDI** (secuencia simbólica de notas \+ pedales \+ velocidad)

…y que esa representación permite **retrieval** (encontrar el par correcto) con **pruebas negativas duras**, no solo “cosine alto”.

---

# **Decisiones de diseño (para que sea “a prueba de trampas”)**

## **Qué “cross-modality” vamos a exigir**

En este escalón conviene **hacer dos niveles**, en orden:

1. **Nivel P (piece-level):** dado audio de una pieza, recuperar el MIDI de esa pieza.  
2. **Nivel S (segment-level):** dado un segmento (p.ej. 2–5 s) de audio, recuperar el segmento MIDI correspondiente (misma pieza y misma ventana temporal).

Si Nivel P no anda, **no tiene sentido** insistir con Nivel S.

---

# **Arquitectura general del trabajo (6 gates)**

## **GATE 0 — Setup “anti-autoengaño” (obligatorio)**

**Entrega:** un “harness” de evaluación y logging que corre igual para todos los modelos.

### **Controles negativos / positivos (se corren SIEMPRE)**

* **NEG-RANDOM:** pares audio↔MIDI completamente random.  
* **NEG-WITHIN-PIECE:** mismo MIDI de la misma pieza pero ventana temporal distinta (hard negative realista).  
* **NEG-SAME-COMPOSER:** MIDI de piezas del mismo compositor (negativo semánticamente difícil).  
* **POS-ORACLE:** construir un positivo artificial “trivial” (por ejemplo, sintetizar audio desde MIDI o construir una representación MIDI-→spectral proxy) para verificar que el pipeline de evaluación detecta señal cuando la hay.

### **Métricas obligatorias**

* **Recall@K** (K=1,5,10)  
* **MRR**  
* **Gap aligned vs shuffled** (con distribuciones, no solo promedio)  
* **Bootstrap CI** (intervalos por queries)

**GO si:** los controles se comportan como corresponde (oracle alto, constantes/negativos \~azar).  
**NO-GO si:** cualquier negativo “parece” positivo → se arregla evaluación antes de entrenar nada.

---

## **GATE 1 — Ingesta y alineación del dataset (MAESTRO)**

**Meta:** que no haya “bugs invisibles” de alineación/tiempo.

### **1.1 Descarga y estructura**

* Usar MAESTRO v3.0.0 (vía fuente oficial o mirror), y **siempre** consumir el JSON/metadata de splits provistos. ([Magenta](https://magenta.withgoogle.com/datasets/maestro?utm_source=chatgpt.com))

### **1.2 Canonicalización**

* **Audio:** resample uniforme (p.ej. 16 kHz o 22.05 kHz), mono (o stereo→mono), normalización RMS.  
* **MIDI:** parse con `pretty_midi` para obtener notas (pitch, onset, offset, velocity) y pedales. ([craffel.github.io](https://craffel.github.io/pretty-midi/?utm_source=chatgpt.com))

### **1.3 Construcción de pares “segment-level”**

Elegí un estándar fijo:

* window\_len \= 2.0 s (o 4.0 s)  
* hop \= 1.0 s  
* generar (audio\_chunk, midi\_chunk) usando el **mismo tiempo absoluto** de la pieza

**Sanity checks:**

* % de ventanas silenciosas (audio) vs ventanas sin notas (MIDI) → deberían correlacionar.  
* un par de piezas al azar: plot de energía audio vs densidad de notas MIDI por ventana.

**GO si:** el pairing temporal es consistente (silencios coinciden, densidad coincide).  
**NO-GO si:** hay drift o mismatch sistemático (arreglar parse/offset/timebase).

---

## **GATE 2 — Baselines “sin deep learning” (deciden si hay señal utilizable)**

La idea es demostrar **señal cross-modal** sin redes, para saber que el problema NO es “imposible”.

### **2.1 Baseline A (muy fuerte y simple): Chroma vs Pitch-Class Profile**

* Audio → chroma (CQT/chroma)  
* MIDI → pitch-class histogram por ventana  
* Similaridad coseno / correlación

Esto ya debería dar una ventaja por encima de random, porque la relación pitch↔frecuencia está “forzada” en piano.

### **2.2 Baseline B: CCA / Ridge cross-modal**

* features densas:  
  * audio: log-mel o CQT promedio/estadísticos por ventana  
  * MIDI: piano-roll downsampleado o histogramas de pitch y duraciones  
* entrenar CCA o ridge para mapear audio↔MIDI y medir retrieval

**GO si:** al menos en **Nivel P** (piece-level) el retrieval supera azar de forma clara.  
**NO-GO si:** ni siquiera estas baselines levantan → el pairing o features están mal, o estás evaluando un objetivo demasiado duro de entrada.

---

## **GATE 3 — Primer modelo cross-modal “estable” (sin depender de batch gigante)**

Acá no recomiendo arrancar con InfoNCE puro por el historial de colapsos y por lo sabido sobre sensibilidad a batch/negativos. (SimCLR mismo avisa que fue tuneado con batch 4096 y puede rendir subóptimo con batches chicos). ([GitHub](https://github.com/google-research/simclr?utm_source=chatgpt.com))

### **3.1 Modelo: Dual Encoder \+ objetivo anti-colapso**

* Encoder\_audio: CNN/Transformer liviano sobre log-mel o CQT  
* Encoder\_midi: CNN/Transformer sobre piano-roll o secuencia de eventos  
* Proyección a z\_shared (dim 128–512)  
* Loss recomendadas (en este orden):  
  1. **VICReg cross-modal** (invariance \+ variance \+ covariance) ([arXiv](https://arxiv.org/abs/2105.04906?utm_source=chatgpt.com))  
  2. **Barlow Twins cross-modal** (cross-correlation ≈ identidad) ([arXiv](https://arxiv.org/abs/2103.03230?utm_source=chatgpt.com))

### **3.2 Evaluación**

* Primero **Nivel P (piece-level)**: embedding promedio de ventanas de una pieza.  
* Después **Nivel S (segment-level)** con candidatos controlados:  
  * set por query: 1 positivo \+ (a) random negatives \+ (b) same composer \+ (c) within piece different time

**GO si:**

* No colapsa (varianza por dimensión no tiende a 0), y  
* Nivel P pasa claro, y  
* Nivel S mejora sobre baselines con NEG-WITHIN-PIECE y NEG-SAME-COMPOSER.

**NO-GO si:** colapso o métricas indistinguibles de azar → revisar input reps y augmentations (siguiente gate).

---

## **GATE 4 — Tu hipótesis: "ratio language / constellations" como extractor principal**

Este gate es donde "Rosetta" se vuelve Phideus: no solo "funciona cross-modal", sino "funciona con relaciones/ratios".

**IMPORTANTE:** Gate 4 se subdivide en **3 sub-gates** con criterios GO/NO-GO independientes:

---

### **GATE 4A — Token Compatibility (validación de extractores)**

**Objetivo:** Verificar que los tokens de audio y MIDI hablan el "mismo lenguaje".

**Métricas requeridas:**

* **Cosine similarity** entre histogramas globales: > 0.9
* **Histogram intersection**: > 0.8
* **KL simétrico**: < 0.2
* **Token ratio** (audio/MIDI): entre 0.5 y 2.0
* **Mean log_ratio** comparable: diferencia < 0.1
* **Balance close/far targets**: 40-60% en ambas modalidades

**Implementación:**

1. Extraer tokens de un par (audio, MIDI) de 60+ segundos
2. Comparar distribuciones marginales
3. Visualizar: histogramas superpuestos, scatter (delta_t vs log_ratio)

**GO si:** todas las métricas dentro de umbrales.
**NO-GO si:** distribuciones incompatibles → iterar extractor (peak picking, harmonics, diversity).

**✅ ESTADO:** PASADO con extractor V2 (cosine=0.965, hist_int=0.853, KL=0.078)

---

### **GATE 4B — Retrieval Baseline Sin Red (prueba Shazam)**

**Objetivo:** Demostrar que hay información de identidad por segmento, no solo estadística global.

**Implementación:**

1. **Segmentación:** 20s con hop 10s
2. **Hashing:** Para cada segmento, crear multiset de hashes:
   ```
   hash = (dt_bin, log_ratio_bin, f_anchor_coarse)
   ```
   * `dt_bin`: 2 frames (~46ms)
   * `log_ratio_bin`: 1/24 octava (~50 cents)
   * `f_anchor_coarse`: banda de frecuencia (4-8 bandas)

3. **Scoring:** `score(query, candidate) = Σ weights_shared_hashes`

4. **Pool de candidatos:** 256 por query, incluyendo:
   * 1 positivo (aligned)
   * N random (otras piezas)
   * K same-piece-diff-time (hard negative temporal)
   * M same-composer (hard negative semántico)

5. **Normalización obligatoria:**
   * L1 normalize histogramas por segmento
   * O usar TF-IDF weighting (hashes raros pesan más)

**Verificación adicional (Self vs Cross):**

```python
# Debe haber separación clara
self_sim = similarity(audio_seg_i, midi_seg_i)      # Aligned
cross_sim = similarity(audio_seg_i, midi_seg_j)    # j≠i, incluyendo misma pieza
assert mean(self_sim) >> mean(cross_sim)
```

**Métricas:**

* **Recall@K** (K=1,5,10,20)
* **MRR**
* **Gap aligned vs hard negatives** (con distribuciones, no solo promedio)

**GO si:** Top-5 significativamente > azar en pools de 256, y aligned separados de hard negatives.
**NO-GO si:** performance = azar → probar variantes:
  * Con/sin f_anchor_coarse
  * IDF weighting
  * Histograma 2D vs hashes discretos

---

### **GATE 4C — Encoder Sobre Tokens (solo si 4B no alcanza)**

**Objetivo:** Aprender matching cross-modal si el directo no es suficiente.

**Implementación:**

* Encoder_audio_ratio: set/sequence encoder (Transformer, DeepSets, o 1D conv)
* Encoder_midi_ratio: mismo tipo de encoder
* Objective: **VICReg** o **Barlow Twins** (mismos del Gate 3)

**Evaluación:** Igual que Gate 4B pero con embeddings aprendidos.

**GO si:** Mejora sobre Gate 4B y/o supera hard negatives.
**NO-GO si:** No mejora → el problema está en la representación ratio, no en el modelo.

---

### **Resumen de Criterios Gate 4**

| Sub-Gate | Criterio GO | Estado |
|----------|-------------|--------|
| **4A** | Token distributions compatibles | ✅ PASADO |
| **4B** | Retrieval sin red > azar significativo | 🔄 PENDIENTE |
| **4C** | Encoder mejora sobre 4B | ⏳ Solo si 4B insuficiente |

**Éxito Rosetta real (definición actualizada):**

* Gate 4A + Gate 4B pasan → **cross-modality con ratio language demostrada sin redes**
* Gate 4C pasa → **robustez adicional con aprendizaje**

---

## **GATE 5 — Volver a contrastivo con negativos “en serio” (solo si hace falta)**

Si querés explorar InfoNCE “clásico”, hacelo cuando ya tenés señal con VICReg/Barlow.

Ahí sí, para robustez:

* **MoCo (queue \+ momentum encoder)** para tener muchos negativos sin batch enorme. ([arXiv](https://arxiv.org/abs/1911.05722?utm_source=chatgpt.com))

**GO si:** mejora retrieval especialmente en negativos duros.  
**NO-GO si:** no mejora o se vuelve inestable → te quedás con VICReg/Barlow.

---

# **Implementación concreta (tareas y entregables)**

## **Paquete A — Repo y herramientas**

**Deliverables:**

* `data/maestro_v3/` \+ script de descarga/verificación  
* `preprocess/` para generar pares ventana a ventana (audio\_chunk, midi\_chunk)  
* `eval/` con harness, métricas, negativos y reportes

**Stack sugerido (Python):**

* PyTorch \+ torchaudio/librosa  
* pretty\_midi para parse MIDI ([craffel.github.io](https://craffel.github.io/pretty-midi/?utm_source=chatgpt.com))  
* mir\_eval (opcional) para métricas MIR si luego lo necesitan ([GitHub](https://github.com/mir-evaluation/mir_eval?utm_source=chatgpt.com))

---

## **Paquete B — DataSpec (contrato de datos)**

Definí un formato único para entrenamiento y para evaluación:

**Ejemplo de registro (por ventana):**

* `piece_id`, `split`, `t0`, `t1`  
* audio:  
  * waveform (opcional) o CQT/mel precomputado  
* midi:  
  * eventos (pitch, onset\_rel, offset\_rel, velocity)  
  * piano-roll downsampleado (opcional)  
* ratio\_tokens (si Gate 4 está activo):  
  * lista de (log\_ratio\_f, delta\_t, amp\_rank, …)

---

## **Paquete C — Baselines**

1. Chroma↔PitchClass retrieval (sin red)  
2. CCA/Ridge retrieval (sin red)  
3. (Opcional) Clasificador de compositor/pieza unimodal (solo para detectar leakage)

**Criterio:** si baselines no ven señal, es pérdida de tiempo entrenar redes.

---

## **Paquete D — Modelos “estables”**

* Dual encoders \+ VICReg cross-modal ([arXiv](https://arxiv.org/abs/2105.04906?utm_source=chatgpt.com))  
* Dual encoders \+ Barlow Twins cross-modal ([arXiv](https://arxiv.org/abs/2103.03230?utm_source=chatgpt.com))  
  Entrenar primero piece-level, luego segment-level.

---

## **Paquete E — Ratio extractor (la parte Phideus)**

Iteración recomendada (en orden, para no frustrarse):

1. MIDI→TF idealizado \+ constelaciones  
2. Audio CQT \+ constelaciones  
3. Matching simple (sin red)  
4. Encoders de tokens \+ VICReg/Barlow  
5. Ablations: sin armónicos, con armónicos; ratio\_f vs delta\_f; ratio\_t vs delta\_t, etc.

---

# **Criterios GO/NO-GO resumidos (para decidir rápido y con seguridad)**

## **Tabla de criterios por Gate**

| Gate | Criterio | Umbral GO |
|------|----------|-----------|
| 0 | Controles funcionan | Oracle > 90%, random ~ 1/N |
| 1 | Alineación temporal | Corr energía-densidad > 0.7 |
| 2 | Baselines | Piece Top-1 > 10× random |
| 3 | Modelo denso | No colapso + Top-1 > baselines |
| **4A** | **Token compatibility** | **cosine > 0.9, KL < 0.2** ✅ |
| **4B** | **Retrieval sin red** | **Top-5 > azar significativo** 🔄 |
| **4C** | **Encoder sobre tokens** | **Mejora sobre 4B** |
| 5 | MoCo negativos duros | Mejora NEG-SAME-COMPOSER |

## **Decisiones**

* **GO (Escalón 1 logrado - Camino Rápido):**
  * Gate 4A pasa (token compatibility) **Y**
  * Gate 4B pasa (retrieval sin red funciona)
  → **Cross-modality con ratio language demostrada SIN entrenar redes.**

* **GO (Escalón 1 logrado - Camino Completo):**
  * Gate 3 pasa (cross-modal con densos funciona y no colapsa), **Y**
  * Gate 4A+4B+4C pasan (ratio tokens muestran señal y encoder mejora).

* **NO-GO parcial (pero valioso):**
  * Gate 4A pasa pero Gate 4B no (retrieval sin red falla).
    → Conclusión: las distribuciones son compatibles pero no hay información de identidad por segmento. Probar Gate 4C.
  * Gate 4B no pasa pero Gate 3 sí (densos funcionan, ratios no).
    → Conclusión: cross-modality es posible, pero el extractor ratio pierde info. Se itera extractor, no arquitectura.

* **NO-GO total:**
  * Ni baselines pasan (Gate 2).
    → Esto casi seguro es bug de pairing/preproc/eval, porque MAESTRO está diseñado justamente para audio↔MIDI alineados. ([Magenta](https://magenta.withgoogle.com/datasets/maestro?utm_source=chatgpt.com))

---

# **Recomendación práctica de arranque (orden de ejecución)**

## **Orden actualizado (post resultados V2)**

Dado que **Gate 4A ya pasó** con el extractor V2, el camino más rápido es:

1. **GATE 4B** (retrieval sin red - Shazam baseline) ← **PRÓXIMO PASO**
2. Si Gate 4B pasa → **ESCALÓN 1 COMPLETADO** (camino rápido)
3. Si Gate 4B no pasa → Gate 4C (encoder sobre tokens)
4. Si Gate 4C no pasa → Gates 2-3 (baselines y modelos densos)

## **Orden completo (si se quiere validación exhaustiva)**

1. **GATE 0** (harness + negativos + oracle)
2. **GATE 1** (pairing temporal + sanity plots)
3. **GATE 2** (chroma/pitchclass + CCA/ridge)
4. **GATE 3** (VICReg o Barlow con densos, primero piece-level)
5. **GATE 4A** (token compatibility) ✅ **PASADO**
6. **GATE 4B** (retrieval sin red - Shazam baseline)
7. **GATE 4C** (encoder sobre tokens, si 4B insuficiente)
8. **GATE 5** (MoCo con negativos duros)

## **Próximo paso inmediato**

Ejecutar **Gate 4B** con 10 pares en `experiments/un_audio_un_midi/Varios_pares/`:

```bash
# Validar extractores V2 con múltiples pares
python experiments/un_audio_un_midi/test_varios_pares.py \
    --input-dir experiments/un_audio_un_midi/Varios_pares/ \
    --output experiments/un_audio_un_midi/Varios_pares/results/ \
    --workers 14
```

Si todos pasan (cosine > 0.9), proceder a retrieval completo en MAESTRO.

---

Si querés, en el próximo mensaje te lo bajo a un **“runbook”** hiper operativo (qué scripts, qué artefactos guardar, y cómo queda una carpeta `/reports/` con gráficos estándar por cada gate), pero con lo de arriba ya tenés una hoja de ruta completa para implementar y ejecutar el Escalón 1 sin volver a quedar atrapados en falsas métricas.

Queda **en el corazón del Escalón 1**, pero con un ajuste clave: en MAESTRO vos tenés que producir **una “constelación compatible” también del lado MIDI**, para que tu extractor (histogramas/constellations estilo Shazam \+ ratios) sea *el puente* entre dominios.

Pensalo así:

## **Dónde entra tu extractor en el plan**

En el plan que te propuse, tu extractor es el **GATE 4** (la prueba Phideus), y cumple 2 roles:

1. **Baseline sin redes (prueba de posibilidad inmediata):**  
   Generás constelaciones/ratios en audio y en MIDI, y hacés matching directo (sin entrenamiento).  
   Si esto ya separa aligned vs shuffled, es un “GO” fuerte: hay señal cross-modal en tu representación.  
2. **Tokenizer para el modelo (si el baseline no alcanza o querés robustez):**  
   Esas constelaciones/ratios pasan a ser “tokens” para un encoder (DeepSets/Transformer), y entrenás alineamiento (VICReg/Barlow/MoCo).

Tu extractor NO desaparece: pasa a ser **la interfaz** entre modalidades.

---

## **Cómo se adapta a MAESTRO sin traicionar el espíritu “Shazam/ratios”**

### **Audio (igual que siempre)**

* Hacés STFT/CQT.  
* Detectás picos tiempo-frecuencia.  
* Construís constelaciones tipo Shazam: pares de picos dentro de una ventana (anchor \+ target) → features como:  
  * (\\Delta t)  
  * (f\_1, f\_2) o mejor **log-ratio**: (\\log(f\_2/f\_1))  
  * opcional: ranking/amplitud de picos, banda

### **MIDI (la parte nueva)**

El MIDI no es un espectrograma, pero **sí es un mapa tiempo-pitch** perfecto. Tenés dos caminos, y ambos sirven:

**Opción A (más pura y directa): “constelación en el plano tiempo-pitch”**

* Cada nota MIDI es un “pico” en (t\_onset, pitch) con “amplitud” \= velocity.  
* Convertís pitch→frecuencia (f) si querés mantener ratios físicos.  
* Armás los pares igual que Shazam:  
  * (\\Delta t) entre onsets  
  * (\\log(f\_2/f\_1))  
  * opcional: velocity, duración

**Opción B (más compatible con el audio real): “MIDI → TF idealizado”**

* Renderizás un pseudo-espectro: para cada nota, ponés energía en su bin de frecuencia (y opcionalmente unos armónicos).  
* Después corrés **el mismo detector de picos** que usás en audio.  
* Armás constelaciones idénticas.

**Qué opción elegir para el “camino más seguro”:**

* Para demostrar posibilidad rápido: **Opción A** (menos pasos, menos fuentes de bug).  
* Para máxima compatibilidad con tu pipeline actual: **Opción B**.

---

## **Y tus “histogramas de ratios”, ¿cómo entran?**

Exactamente igual, solo que el universo de “eventos” cambia:

* Construís un conjunto grande de pares (anchor,target) por ventana/pieza.  
* De cada par sacás tu feature tipo ratio:  
  * (\\log(f\_2/f\_1)) (o ratio directo)  
  * (\\Delta t) (o ratio temporal si querés)  
* Eso lo discretizás en bins y generás:  
  * **histogramas 2D** (ratio\_f × delta\_t)  
  * o “fingerprints” tipo Shazam (hashes)  
  * o ambos

Y listo: ya tenés un “lenguaje de ratios” común.

---

## **Qué cambia respecto a motores (y por qué es bueno)**

En motores, audio↔vib puede estar lleno de transformaciones (estructura, resonancias, ruido aerodinámico, hum). En MAESTRO:

* audio y MIDI están **acoplados por construcción** (la música).  
* hay muchísimos datos y variabilidad.  
* si tu extractor funciona acá, ya probaste “posibilidad” con un margen enorme.

---

## **GO/NO-GO específico para tu extractor (sin redes)**

Esto es lo más importante para ubicarlo:

### **GO (muy fuerte)**

Con **matching directo** entre constelaciones/histogramas (audio query → MIDI candidatos):

* aligned score \>\> shuffled score  
* y además resiste negativos “duros” (misma pieza, distinto tiempo; mismo compositor)

### **NO-GO (pero informativo)**

Si:

* los baselines densos (log-mel/CQT \+ encoder) funcionan,  
* pero tu extractor de ratios no,  
  entonces no es que “cross-modality no existe”: es que **tu extractor está perdiendo info** (por ejemplo, no modela sustain/pedal, o está muy sensible a dinámica, o el peak picking no captura lo relevante).

Eso es buenísimo porque te dice dónde iterar: **extractor**, no “arquitecturas infinitas”.

---

## **Mini-receta de implementación (para aterrizarlo)**

1. Elegí ventana 2–4 s, hop 1 s.  
2. Audio: CQT → picos → pares → tokens/histogramas.  
3. MIDI: (A) eventos → picos (onset,pitch) → pares → tokens/histogramas.  
4. Matching baseline:  
   * histogram intersection / cosine / earth mover (si 2D)  
   * o conteo de hashes coincidentes (Shazam-style)  
5. Si baseline da señal: recién ahí entrenás un encoder sobre esos tokens/histogramas.

---

Si me decís **qué forma exacta tiene hoy tu "ratio language"** (qué bins, qué pares, qué features guarda el token), te lo mapeo 1:1 a MIDI y te propongo dos variantes: una "mínimo cambio" (para reusar tu código) y otra "más limpia" (para maximizar señal en MAESTRO).

---

# **RESULTADOS DE EJECUCIÓN (2026-02-04)**

## Estado Final: ✗ NO-GO (pero científicamente informativo)

### Resumen de Gates Ejecutados

| Gate | Descripción | Estado | Resultado |
|------|-------------|--------|-----------|
| 4A | Token Compatibility | ✓ PASS | cosine = 0.957 |
| 4B-Oracle | Shazam MIDI vs MIDI | ✓ PASS | Piece Acc = 90.9% |
| 4B-CrossModal | Shazam Audio vs MIDI | ✗ FAIL | Piece Acc = 15.5% |

### Resultados Detallados

**Token Compatibility (10 piezas, 110 segmentos):**
- Cosine similarity: **0.957** (umbral > 0.9) ✓
- Token ratio (Audio/MIDI): **1.16x** (umbral 0.5-2.0) ✓
- Mean log_ratio: Audio=0.956, MIDI=0.942 ✓

**Oracle Test (MIDI vs MIDI):**
- Piece Accuracy: **90.9%** ✓
- Offset MAE: **0.14s** ✓
- Conclusión: El algoritmo Shazam funciona correctamente

**Cross-Modal Test (Audio vs MIDI):**
- Piece Accuracy: **15.5%** (random = 10%) ✗
- Offset MAE: **30.87s** ✗
- Recall@5: **50.9%** (random = 50%) ✗
- Conclusión: Los hashes Audio↔MIDI NO coinciden

### Interpretación

El "ratio language" captura **estadística global compatible** (distribuciones similares) pero **NO identidad cross-modal** a nivel de tokens individuales.

**Causa probable**: Los mismos intervalos musicales en Audio y MIDI no producen los mismos hashes debido a:
1. Diferencias en peak picking (Audio real vs pseudo-TF MIDI)
2. Contenido armónico diferente
3. Ruido/dinámica en Audio ausente en MIDI

### Conclusión Científica

| Hipótesis | Estado |
|-----------|--------|
| H1: Distribuciones compatibles | ✓ VALIDADA |
| H2: Shazam voting funciona | ✓ VALIDADA |
| H3: Cross-modal identification | ✗ NO VALIDADA |

**Ver informe completo**: `Documents/01_FRENTES_ACTIVOS/ESCALON_1/RESULTADOS_ESCALON_1.md`

