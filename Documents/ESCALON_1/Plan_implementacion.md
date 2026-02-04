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

## **GATE 4 — Tu hipótesis: “ratio language / constellations” como extractor principal**

Este gate es donde “Rosetta” se vuelve Phideus: no solo “funciona cross-modal”, sino “funciona con relaciones/ratios”.

La clave para que sea justo: **primero construimos un extractor ratio que sea razonable en MIDI**, y luego buscamos su análogo en audio.

### **4.1 Definir el “espacio de ratios” (propuesta práctica)**

Trabajá con **constelaciones en (tiempo, frecuencia)** porque:

* en audio ya existe el paradigma “constellation map” (picos en TF),  
* en MIDI podés construir un TF “ideal” (piano-roll → espectro idealizado).

**Para MIDI (dos variantes):**

* **MIDI→TF idealizado:** construir un “CQT sintético” donde cada nota agrega energía en su bin de frecuencia fundamental (opcional: algunos armónicos).  
* Luego detectar picos y formar constelaciones de pares:  
  * ratio\_f \= f2 / f1  
  * delta\_t \= t2 − t1 (o ratio\_t si querés ratios temporales)

**Para Audio:**

* CQT real del audio  
* peak picking local (en TF)  
* construir las mismas constelaciones (ratio\_f, delta\_t)

Esto te da un extractor de ratios común **sin que el modelo aprenda a inventar ratios**.

### **4.2 GO/NO-GO del extractor (antes del modelo)**

* Si con solo constelaciones \+ matching simple (sin redes) ya hay señal → excelente.  
* Si no hay señal, pero los baselines densos sí → el extractor ratio está perdiendo info; hay que iterarlo.

**GO si:** el extractor ratio ya muestra señal por encima de random en Nivel P o S.  
**NO-GO si:** ratio extractor es ciego → iterar (peak picking, densidad, bins, armónicos, thresholds).

### **4.3 Entrenamiento con ratio tokens**

Una vez que el extractor ratio “ve algo”:

* Encoder\_audio\_ratio: set/sequence encoder (Transformer para sets, DeepSets, o 1D conv sobre tokens)  
* Encoder\_midi\_ratio: igual encoder (mismo tipo)  
* Objective: VICReg/Barlow (mismos del Gate 3\)

**Éxito Rosetta real (definición):**

* que el modelo con **ratio tokens** alcance performance competitiva vs modelo denso (aunque sea un poco menor al principio),  
* y que sea robusto a negativos duros.

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

* **GO (Escalón 1 logrado):**  
  * Gate 3 pasa (cross-modal con densos funciona y no colapsa), **y**  
  * Gate 4 pasa (ratio tokens muestran señal y un modelo aprende con ellos mejor que baselines simples).  
* **NO-GO parcial (pero valioso):**  
  * densos pasan (Gate 3\) pero ratio tokens no (Gate 4).  
    → Conclusión: cross-modality es posible, pero tu extractor ratio todavía no capta lo necesario. Se itera extractor, no arquitectura.  
* **NO-GO total:**  
  * ni baselines pasan (Gate 2).  
    → Esto casi seguro es bug de pairing/preproc/eval, porque MAESTRO está diseñado justamente para audio↔MIDI alineados. ([Magenta](https://magenta.withgoogle.com/datasets/maestro?utm_source=chatgpt.com))

---

# **Recomendación práctica de arranque (orden de ejecución)**

Si mañana arrancan:

1. **GATE 0** (harness \+ negativos \+ oracle)  
2. **GATE 1** (pairing temporal \+ sanity plots)  
3. **GATE 2** (chroma/pitchclass \+ CCA/ridge)  
4. **GATE 3** (VICReg o Barlow con densos, primero piece-level)  
5. **GATE 4** (ratio extractor y luego modelo con tokens)

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

Si me decís **qué forma exacta tiene hoy tu “ratio language”** (qué bins, qué pares, qué features guarda el token), te lo mapeo 1:1 a MIDI y te propongo dos variantes: una “mínimo cambio” (para reusar tu código) y otra “más limpia” (para maximizar señal en MAESTRO).

