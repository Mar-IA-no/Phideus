According to a document from **2026-02-04**, hoy ya tienen un resultado *científicamente concluyente* sobre el enfoque “ratio language \+ hashing exacto”: **funciona intra-modal (Oracle MIDI↔MIDI), pero falla cross-modal (Audio↔MIDI)**. En los Gates ejecutados: *cosine* de compatibilidad global alto (≈0.957), Oracle con **90.9%** y MAE de offset **0.14s**, pero cross-modal con **15.5%** (cerca del azar) y MAE **30.87s**. La interpretación explícita en el plan es que el lenguaje captura **estadística global compatible** pero **no identidad cross-modal a nivel token/hash**.  
Eso está totalmente alineado con el análisis senior: el cuello es estructural (discretización agresiva \+ matching exacto sin “soft matching”), y seguir “tuneando” el extractor los mete en rendimientos decrecientes (óptimo local).

Con eso como base, si tu objetivo ahora es **demostrar cross-modality de forma inequívoca** (aunque haya que tirar el extractor), lo más “seguro” es **pasar al Grupo 2**: *aprendizaje de embeddings con matching suave*, y si quieren sostener el “ratio insight”, que sea como **inductive bias / vista auxiliar**, no como identidad hard-hash.

Abajo te dejo un **plan de implementación (Camino 2: híbrido robusto) centrado 100% en MAESTRO**, con Gates GO/NO-GO y con lugar explícito para el extractor de ratios.

---

## **0\) Premisa de diseño (lo que cambia y lo que se conserva)**

### **Lo que se abandona (como criterio de éxito)**

* “Los hashes individuales Audio↔MIDI deben coincidir” (estilo Shazam). Ya vieron que no pasa; y es esperable: audio real incluye timbre, dinámica, pedales, resonancias, sala, etc., que un MIDI no “emite” igual.

### **Lo que se conserva (como insight)**

* “Las relaciones proporcionales pueden codificar estructura transferible” (tu intuición de ratios). El punto es **cómo** lo incorporás: como *features continuas* o como *objetivo auxiliar*, no como “token identity”.

---

## **1\) Por qué MAESTRO es el banco de pruebas correcto (y qué exige)**

MAESTRO tiene \~200h de **audio+MIDI pareados** y alineados con precisión \~3ms; además el MIDI incluye **velocidad y pedales** (sustain/sostenuto/una corda), que son clave para que el “contenido musical” sea comparable al audio real.  
Esto significa que si un enfoque *embedding-based* no logra señal cross-modal en MAESTRO, es una evidencia fuerte de que el método (o el objetivo) está mal planteado, no de que “el dataset es malo”.

---

## **2\) Objetivo operativo del Camino 2 (definición de “prueba concreta”)**

Una prueba concreta y robusta de cross-modality acá no es “hash-match”, sino:

**Dado un segmento de audio, recuperar el segmento MIDI correspondiente (misma pieza y mismo rango temporal) mejor que azar, de forma consistente y con negativos duros**, usando una métrica suave (distancia en embedding space).

Métricas mínimas:

* **Retrieval** Audio→MIDI y MIDI→Audio: Recall@{1,5,10,20}, MRR  
* **Piece-ID accuracy** (top-1) y **Offset error** (MAE en segundos) si evaluás por ventanas temporales  
* **Gap vs negativos duros**: same piece / different time; same composer; tempo-shifted; etc.

---

## **3\) Dónde queda tu extractor de ratios (respuesta directa)**

En el Camino 2, tu extractor tiene 3 lugares legítimos, *en este orden de prioridad*:

### **(A) Instrumento de diagnóstico (se queda sí o sí)**

Sigue siendo útil para medir “compatibilidad estadística” y detectar sesgos: densidad de tokens, distribución de log\_ratio, sensibilidad a dinámica, etc. Ya les sirvió para aislar que el fallo era *identidad*, no “todo”.

### **(B) Vista auxiliar para entrenamiento (híbrido robusto)**

En vez de convertirlo en hashes exactos, lo convertís en una **representación continua**:

* histogramas 2D suaves (kernel density),  
* bag-of-ratios con embeddings,  
* o “ratio patches” temporales.

Luego lo usás como **tercera vista** junto a (audio, midi) en un esquema multi-view (ej. VICReg multi-view), para empujar al modelo a capturar esas invariantes.

### **(C) Regularizador / control de sesgo**

Podés forzar que el embedding compartido preserve información “ratio-like” con una loss auxiliar (predict ratios) sin obligarlo a calcar tokens.

Conclusión: **no lo tiraría al tacho**, pero **dejaría de medir éxito por matching exacto**.

---

## **4\) Arquitectura recomendada (Camino 2 híbrido robusto)**

### **4.1 Encoders base (primero congelados)**

* **Audio encoder**: usar un foundation model musical tipo **MERT** para extraer embeddings frame-level/segment-level (al inicio congelado).  
* **MIDI encoder**: Transformer de eventos (NOTE\_ON, NOTE\_OFF, TIME\_SHIFT, VELOCITY, PEDAL) o un encoder de piano-roll \+ tiempo.  
* **Projection heads**: MLPs que mapean ambos a un embedding común (p.ej. 256–768 dims).

### **4.2 Objetivo de alineación (no colapsable, sin necesidad de enormes batches)**

* **VICReg cross-modal** (Audio-view vs MIDI-view), porque evita colapso explícitamente y funciona bien sin contraste masivo.  
  * invariance loss (L2 entre embeddings emparejados)  
  * variance loss (evita colapso)  
  * covariance loss (decorrelación)

### **4.3 Control de sesgo / invariancia modal (parte “robusta”)**

Sumás un componente tipo **Domain-Adversarial (DANN)**:

* un “domain classifier” que intenta predecir si el embedding vino de Audio o de MIDI  
* un **gradient reversal layer** para que el embedding compartido se vuelva *modal-agnostic*

Esto ataca el problema típico de cross-modal: el modelo aprende shortcuts del dominio en vez de la estructura.

### **4.4 Vista auxiliar (tu extractor, pero “soft”)**

* Computás ratios (o constelaciones) en ambos dominios, pero NO hasheás para match exacto.  
* Entrenás un **ratio-encoder** pequeño que produce embedding, y agregás pérdidas:  
  * align(audio\_emb, ratio\_emb)  
  * align(midi\_emb, ratio\_emb)  
  * opcional: predict(histograma\_ratio) desde audio\_emb/midi\_emb

---

## **5\) Plan de implementación por Gates (GO/NO-GO), robusto y consistente**

### **Gate 0 — Integridad y alineación del dataset (hard sanity)**

**Objetivo:** garantizar que todo lo que sigue no está contaminado por bugs de slicing/alineación.

* Cargar MAESTRO (ideal v3.0.0; el sitio oficial documenta alineación \~3ms).  
* Definir ventana y hop:  
  * segment\_len \= 4s u 8s (para que haya contenido musical suficiente)  
  * hop \= 1s o 2s  
* Verificar:  
  * para cada par, (audio\_duration ≈ midi\_duration)  
  * slicing consistente (mismo t0/t1 en ambos)  
* Negativo de control: “shuffled pairs” debe destruir performance.

**GO** si:

* los checks pasan en un subset (p.ej. 100 piezas) sin casos raros dominantes.

---

### **Gate 1 — Baselines intra-modales (necesarios para interpretar todo lo demás)**

**Objetivo:** si ni dentro de Audio o dentro de MIDI hay retrieval consistente, lo cross-modal no tiene piso.

* Audio→Audio: usar embeddings del audio encoder \+ cosine  
* MIDI→MIDI: usar embeddings del midi encoder \+ cosine

**GO** si:

* retrieval intra-modal es claramente \> azar (por ejemplo Recall@10 mucho mayor que random), con negativos duros.

---

### **Gate 2 — Cross-modal “Foundation baseline” (sin ratios todavía)**

**Objetivo:** probar cross-modality con el enfoque más “industrial” posible.

* Congelar audio encoder (MERT) \+ entrenar projection head  
* Entrenar MIDI encoder (o congelarlo si arrancás con uno preentrenado) \+ projection head  
* Loss: VICReg(Audio,MIDI)

**GO** si:

* Audio→MIDI y MIDI→Audio superan azar de forma consistente,  
* y resisten negativos duros (misma pieza distinto tiempo).  
  Acá el éxito no tiene que ser altísimo todavía; lo importante es “hay señal” y no se derrumba con duros.

---

### **Gate 3 — Robustez por control de sesgo (DANN)**

**Objetivo:** si el modelo está colapsando en señales de dominio, esto lo fuerza a dejar de hacerlo.

* Agregar domain classifier \+ GRL (DANN).  
* Métrica auxiliar: accuracy del domain classifier debería tender a \~50% si el embedding se volvió modal-agnostic (ojo: sin arruinar retrieval).

**GO** si:

* mejora o estabiliza cross-modal vs Gate 2, especialmente en negativos duros.

---

### **Gate 4 — Híbrido con ratios como vista auxiliar**

**Objetivo:** reinyectar tu “ratio insight” de manera compatible con aprendizaje.

* Activar ratio-encoder y losses multi-view:  
  * VICReg(Audio,Ratio)  
  * VICReg(MIDI,Ratio)  
  * y/o loss de predicción de histograma\_ratio

**GO** si:

* sube el gap vs negativos duros (misma pieza, otro tiempo)  
* baja el error de offset (si evaluás offset)  
* y mejora consistencia por compositor/tempo.

---

### **Gate 5 — Currículum de brecha de dominio (opcional pero muy potente)**

Este Gate es “hacerlo más fácil primero” para demostrar posibilidad.

1. Renderizar MIDI a audio con un piano virtual (brecha chica)  
2. Entrenar alignment ahí  
3. Mezclar progresivamente audio real MAESTRO

**GO** si:

* en el escenario renderizado el cross-modal despega fuerte, y al mezclar real no colapsa totalmente.

---

## **6\) Configuración default recomendada (valores iniciales “sensatos”)**

### **Segmentación**

* segment\_len: 8.0s (si 4s da muy pocos eventos armónicos)  
* hop: 2.0s  
* batch\_size: 64 (o lo que permita GPU)  
* embeddings: 256 o 512 para arrancar

### **Optimización**

* AdamW  
* lr\_heads: 1e-3  
* lr\_midi\_encoder: 1e-4  
* lr\_audio\_encoder: 0 (freeze) → luego 1e-5 si fine-tune parcial  
* weight\_decay: 1e-4  
* warmup: 5% steps

### **VICReg (valores típicos de arranque)**

* invariance\_weight: 25  
* variance\_weight: 25  
* covariance\_weight: 1  
  (la idea es: fuerte en invariance/variance para que alinee sin colapsar; covariance más suave).

### **DANN**

* domain\_loss\_weight: 0.1 (arrancar chico)  
* GRL lambda schedule: de 0 → 1 lineal a lo largo del entrenamiento

### **Ratio-aux**

* ratio\_loss\_weight: 0.05–0.2 (arrancar bajo; subir si mejora duros)

---

## **7\) Evaluación: exactamente qué reportar para que sea “prueba”**

Para cada Gate a partir del 2:

* Audio→MIDI y MIDI→Audio:  
  1. Recall@{1,5,10,20}  
  2. MRR  
* “Hard negatives suite”:  
  1. same piece, different time  
  2. same composer  
  3. tempo-shift (solo MIDI)  
  4. pitch-shift pequeño (audio) *si querés testear invariancia*  
* “Shuffle control”: emparejamientos aleatorios deben caer cerca de azar.  
* Curva “N piezas”: medir performance al crecer N (100 → 500 → 1000+) para descartar overfit de pequeño N.

---

## **8\) Qué espero que pase (mi opinión honesta, como investigador)**

* El resultado que tienen (Oracle ok, cross fail en hashes) **no sugiere que cross-modality sea imposible**; sugiere que **no es un problema de “arquitectura de red” todavía**, sino que el *criterio de identidad* que se le exige al extractor no es compatible con la realidad física de Audio vs MIDI.  
* MAESTRO es un excelente “campo de pruebas” porque la alineación es muy fina (\~3ms) y el MIDI tiene pedales/velocidad, o sea: si hay lugar donde debería aparecer señal cross-modal, es acá.  
* Pasar a embeddings \+ soft matching es el movimiento correcto y está respaldado por el propio análisis senior: conservar el insight, abandonar hashing exacto.

---

## **9\) Próximos 3 pasos concretos (para destrabar ya)**

1. **Implementar Gate 0–1** con subset de 100 piezas (pipeline limpio, slicing consistente, intra-modal con embeddings).  
2. **Gate 2**: baseline cross-modal con MERT (freeze) \+ MIDI encoder \+ VICReg.  
3. Si hay señal, recién ahí: **Gate 3 (DANN)** y **Gate 4 (ratios auxiliares)**.

---

Si querés, en el próximo mensaje puedo bajarte esto a una **estructura de repo \+ lista de scripts exactos** (train.py / eval.py / dataset.py / encoders.py) y un “experiment matrix” (YAMLs) para correrlo como batería reproducible.

