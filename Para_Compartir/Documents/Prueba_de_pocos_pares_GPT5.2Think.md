Prueba de pocos pares.

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

Perfecto. Te paso **defaults concretos** (con números) para que el MIDI “hable el mismo idioma” que tu extractor de audio, y después un set de recomendaciones para evitar trampas típicas (densidad, colapso a ratio≈1, normalización, etc.). Voy a asumir que tu extractor en audio opera sobre un **CQT o STFT en escala log-frecuencia**, detecta **picos TF**, arma **pares (anchor,target)** con una ventana temporal y produce tokens con `delta_t`, `log_ratio`, `weight`.

---

# **Defaults recomendados (MIDI → pseudo-TF → mismo extractor)**

## **1\) Parámetros de timebase (tienen que calzar con audio)**

**Objetivo:** igualar el eje temporal.

* `sr = 22050` (o el que ya uses)  
* `hop_length = 512` → \~23.22 ms por frame  
* `frame_dt = hop_length / sr ≈ 0.02322 s`

**Ventanas para pairing (Shazam-style)**

* `pair_time_window_sec = 2.0`  
* `pair_time_window_frames = round(2.0 / frame_dt) ≈ 86`  
* `delta_t` guardalo en **frames** *o* en **segundos**, pero igual en ambos.  
  Default: **frames** (menos floats, más estable).

**Por qué 2.0 s:** en música te captura patrones rítmicos y transiciones sin explotar combinatoria.

---

## **2\) Eje de frecuencia / bins (log-freq compatible)**

Si tu extractor trabaja en bins de CQT (recomendado), usá:

* `fmin = 27.5 Hz` (A0)  
* `bins_per_octave = 36` (3 bins por semitono)  
* `n_octaves = 7` → cubre hasta \~3520 Hz (A7)  
* `n_bins = bins_per_octave * n_octaves = 252`

Si tu audio ya usa otro set, **no lo cambies**: hacé que el MIDI-TF se construya en esos mismos bins.

---

## **3\) Construcción del pseudo-TF desde MIDI**

### **Notas sostenidas (importantísimo)**

Para cada nota (pitch, onset, offset, velocity):

* La nota “vive” desde `onset` hasta `offset` (o hasta note-off real).  
* Se “pinta” energía en todos los frames cubiertos.

### **Armónicos (para parecerse al audio real)**

Defaults:

* `n_harmonics = 6` (fundamental \+ 5 armónicos)  
* `harmonic_amplitudes = 1 / h` (h=1..6)  
  Alternativa más rápida de decaimiento: `1 / h^1.5` (si te queda muy “brillante”).  
* `harmonic_detune_cents_std = 3` cents (gaussiano por nota, opcional)  
  Esto simula micro-variaciones reales y evita que el TF sintético sea “demasiado perfecto”.

### **Envolvente temporal (para simular piano)**

Defaults (simple ADSR aproximado):

* `attack = 0.01 s`  
* `decay = 0.15 s`  
* `sustain_level = 0.25` (relativo a pico)  
* `release = 0.25 s`

Implementación simple:

* multiplicás la energía por una envolvente por frame desde onset:  
  * subida rápida (attack),  
  * caída (decay),  
  * sustain bajo hasta note-off,  
  * release después de offset (extendés unos frames más).

**Pedal sustain (si tenés CC64):**

* si CC64 \> 64 (ON), extendé offsets de notas hasta que el pedal se suelte.  
* Default: habilitado si está en MIDI (en MAESTRO suele estar).

**Por qué esto importa:** si no agregás sustain y armónicos, el MIDI se vuelve “picos discretos” y tu densidad de tokens nunca va a parecerse al audio.

---

## **4\) Peak picking sobre el MIDI-TF (para que no te explote el \#tokens)**

Tu audio está generando una barbaridad de tokens (2.6M en un solo par). Para MIDI-TF, si pintás armónicos, podrías explotar también. Controlalo con:

* `peaks_per_frame_max = 8` (default)  
* `peak_threshold_db = -40 dB` relativo al máximo global del clip  
  (o un percentil: quedate con picos por encima del P90 de energía del frame)

**Regla:** pico por frame limitado \+ threshold.

---

## **5\) Pairing (anchor → targets)**

Acá es donde en tu audio hoy pareciera estar colapsando a ratio≈1. Defaults que evitan eso:

* `fan_out = 4` (targets por anchor)  
* `anchor_selection`: top peaks del frame (por weight)  
* `target_frames`: desde `t_anchor+1` hasta `t_anchor + pair_time_window_frames`  
* `target_selection`:  
  * elegí targets en **frames distintos**, y  
  * forzá diversidad en frecuencia:  
    * 50% targets “cercanos” (misma banda ± 1 octava)  
    * 50% targets “lejanos” (entre \+1 y \+3 octavas, si existen)

Esto es crítico para que tu hist de ratios no se aplaste en 0\.

---

## **6\) Token definition (híbrido recomendado)**

Para que sea Phideus (ratios) pero no pierdas identificabilidad:

* `delta_t` (en frames)  
* `log_ratio = log2(f_target / f_anchor)` (esto ya lo tenés)  
* `f_anchor_logbin` (índice de bin o log2(f\_anchor))  
* `weight = (w_anchor * w_target)^0.5`

¿Por qué agregar `f_anchor`? Porque si solo usás ratio, muchas estructuras distintas colisionan. Esto no “traiciona” ratios: el ratio sigue siendo la relación; `f_anchor` solo ubica el contexto.

---

## **7\) Normalización de weights (audio vs MIDI)**

Tus plots muestran weights en escalas diferentes (audio hasta \~1, MIDI hasta \~0.7). Normalizá igual:

* en cada clip/segment:  
  * `w = w / (percentile(w, 99) + eps)`  
  * `w = clip(w, 0, 1)`

Esto te hace comparable la escala sin depender de outliers.

---

# **Recomendaciones adicionales (muy pertinentes por lo que mostraron tus gráficos)**

## **A) Hacé que el “MIDI token rate” se acerque al “audio token rate”**

No tiene que ser igual, pero no puede ser 62 vs 0.2 tokens/frame.

Target sano:

* **MIDI-TF tokens/frame** ≈ 10–40  
* **Audio tokens/frame** ≈ 20–80

Si tu audio está siempre al máximo (62.5 fijo), es señal de que el extractor está saturando. Bajá:

* `peaks_per_frame_max` (audio) a 8–12  
* o subí threshold

**Querés variabilidad**, no una tasa constante al techo.

---

## **B) Segmentá: no uses 1000 s para comparar histogramas globales**

Para Rosetta, compará por:

* segmentos de **10–30 s**  
* con hop de 5–10 s

Eso:

* aumenta el número de ejemplos (más pares)  
* evita que “promediar” te destruya estructura temporal

---

## **C) Métrica de matching: no te quedes en “hist global”**

Además del hist global, meté un score “Shazam clásico”:

* discretizá `log_ratio` y `delta_t` en bins gruesos  
* armá un hash (bin\_log\_ratio, bin\_delta\_t, bin\_f\_anchor)  
* score \= cantidad de hashes compartidos (con weights)

Ese score suele “ver” estructura donde el hist global se lava.

---

## **D) Debug rápido para detectar por qué audio colapsa en log\_ratio≈0**

Hacé estos 2 chequeos:

1. Histograma de `f_target - f_anchor` (o ratio) condicionado a delta\_t chico vs grande.  
   Si siempre es \~0 independientemente de delta\_t, tu pairing está “siguiendo líneas” y no capturando relaciones.  
2. Proporción de targets dentro de la misma banda (±1 semitono o ±1 bin).  
   Si es \>80%, estás generando ratios≈1 por diseño.

---

## **E) Defaults de bins para matching (si querés hash estilo Shazam)**

* `delta_t_bin_size = 2 frames` (\~46 ms)  
* `log_ratio_bin_size = 1/24` de octava (\~0.0417 en log2)  
  (equivale a \~50 cents; suficientemente grueso para tolerar variaciones)

---

# **Checklist “mañana a la mañana” (para confirmar que va bien)**

Corré tu script en 20 pares random y mirá:

1. `tokens/frame` audio y midi\_tf: ambos en rango razonable (10–80)  
2. `mean(log_ratio)` no absurdamente separado (no 0.08 vs 1.3)  
3. hist(log\_ratio) de audio **ya no** es un spike gigante en 0  
4. cosine/hist intersection suben de forma consistente  
5. mini-retrieval: en un set de 100 candidatos, el verdadero entra top-5 más veces que azar

---

# **Si querés un “default profile” listo (resumen en una lista)**

* sr=22050, hop=512, fmin=27.5, bins\_per\_oct=36, n\_bins=252  
* midi\_tf: harmonics=6, amp=1/h, detune\_std=3 cents  
* envelope: A=0.01, D=0.15, S=0.25, R=0.25 (+ pedal extend)  
* peak\_pick: peaks\_per\_frame\_max=8, threshold=-40 dB rel  
* pairing: window=2.0s, fan\_out=4, mitad targets cercanos/mitad lejanos  
* token: (delta\_t\_frames, log\_ratio, f\_anchor\_bin, weight\_geommean)  
* weight norm: divide P99 \+ clip\[0,1\]  
* hash bins: dt\_bin=2 frames, log\_ratio\_bin=1/24 oct

---

Si me compartís (aunque sea pegado en texto) el bloque central de tu extractor donde:

* selecciona picos,  
* decide targets,  
* y computa log\_ratio,

te señalo exactamente **dónde** está naciendo el colapso a ratio≈1 y qué línea cambiar para arreglarlo con mínima intervención.

