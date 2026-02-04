Dale. Te dejo dos **especificaciones exactas**, pensadas para que puedas implementarlas tal cual y correr **retrieval \+ GO/NO-GO** sin ambigüedades.

Voy a llamarlas:

* **(A) Ratio-Graph Event-Based** (modalmente agnóstico “de verdad”: tiempo–pitch como espacio común)  
* **(B) TF-Constellations Shazam-Style** (reusa tu extractor: MIDI → pseudo-TF → mismo pipeline que audio)

En ambos casos, la unidad de trabajo es **segmento** (no la pieza entera): 10–30s. Es clave para evitar que los histogramas globales “laven” la estructura y para aumentar N.

---

# **Global: configuración común a ambos enfoques**

## **Segmentación**

* `segment_len_sec = 20.0`  
* `segment_hop_sec = 10.0`  
* Cada pieza → lista de segmentos `(piece_id, t0, t1)`.

## **Evaluación (igual para ambos)**

Para cada query (segmento de audio o MIDI):

* Candidate pool: **N=256** segmentos (incluye el verdadero \+ 255 negativos)  
* Negativos obligatorios:  
  * `NEG_RANDOM`: segmentos random de otras piezas  
  * `NEG_SAME_PIECE_DIFF_TIME`: misma pieza, ventana temporal distinta (hard)  
  * `NEG_SAME_COMPOSER`: mismo compositor, otra pieza (hard semántico)  
* Métricas:  
  * Recall@{1,5,10}  
  * MRR  
  * Distribución score(aligned) vs score(shuffled)

GO mínimo: Rec@1 y MRR **muy por encima** de azar y separación clara aligned vs within-piece negatives.

---

# **(A) Ratio-Graph Event-Based (modalmente agnóstico)**

## **Idea**

Convertís **ambas modalidades** a un set/seq de **eventos musicales** y construís un “lenguaje de ratios” con:

* **ratios de pitch** (intervalos \= log-ratios)  
* **ratios temporales** (proporciones entre IOIs)  
* **constelaciones en tiempo–pitch** (Shazam pero en el plano correcto)

## **A.0 Timebase canónico del Ratio-Graph**

Definimos una grilla temporal canónica para cuantizar tiempos:

* `event_frame_rate = 100 Hz` → `dt_event = 0.01 s`  
* Todo tiempo se cuantiza a frames: `t_frame = round(t_sec / 0.01)`

Pitch en semitonos (MIDI) es ya log-freq:

* `p = pitch_midi` (entero)  
* `log_ratio_f = (p2 - p1) / 12.0` (en octavas, log2)

---

## **A.1 Extracción de eventos desde MIDI (verdad base)**

### **Entrada**

Notas: `(pitch, onset_sec, offset_sec, velocity)` \+ pedal sustain (CC64) si existe.

### **Defaults MIDI→Eventos**

* `sustain_on_threshold = 64` (CC64 \> 64 \= pedal ON)  
* Si pedal ON: extender offsets hasta que pedal OFF (para todas las notas activas)  
* `min_note_dur = 0.05 s` (descartar notas muy cortas)  
* `merge_gap = 0.03 s` (si dos notas del mismo pitch están separadas por \<30ms, merge)

### **Output: lista de eventos**

Cada nota produce un evento:

* `e = (t_on_frame, t_off_frame, pitch_midi, amp)`  
* `amp = velocity / 127.0`

---

## **A.2 Extracción de eventos desde Audio (para piano)**

Acá necesitás *alguna* forma de obtener notas/onsets. No hace falta perfecta; hace falta consistente.

### **Output esperado del transcriptor**

Una lista de notas estimadas:

* `(pitch_midi, onset_sec, offset_sec, confidence, amp_est)`

### **Defaults de post-proceso (para que sea estable)**

* `onset_conf_th = 0.5`  
* `min_note_dur = 0.05 s`  
* `merge_gap = 0.03 s` (mismo pitch)  
* `max_polyphony = 10` (si hay más, quedate con las 10 de mayor amp/conf por frame)

### **Ajuste de offset global (importantísimo)**

Aunque MAESTRO esté muy alineado, tu transcriptor puede tener un delay sistemático. Corrigilo por segmento con un shift escalar:

* Construí dos series binarias a 100 Hz:  
  * `x[t]=1` si hay onset MIDI cerca de t  
  * `y[t]=1` si hay onset AUDIO cerca de t  
* Calculá shift `s` que maximiza `corr(x, shift(y, s))` en rango:  
  * `s ∈ [-20, +20] frames` (±200ms)  
* Corrigí audio events: `t_on += s`, `t_off += s`

Esto hace que el ratio-language no “pierda” por pequeños desfases.

---

## **A.3 Construcción del Ratio-Language (tokens)**

Generás **3 familias** de tokens. En todos, guardás valores y también versión cuantizada para hashing/hist.

### **Quantización común**

* `dt_bin_size = 2 frames` (20ms)  
* `dp_bin_size = 1 semitone`  
* `rt_bin_size = 1/24 octava` (opcional si usás log\_ratio continuo)

#### **Funciones**

* `dt_bin = round((t2 - t1) / 2)`  
* `dp_bin = clamp(p2 - p1, -36, +36)` (±3 octavas)  
* `log_ratio = (p2 - p1)/12`  
* `log_ratio_bin = round(log_ratio / (1/24))` (50 cents)

---

### **Token Tipo 1: Intervalos simultáneos (acorde)**

Agrupá eventos con onsets casi simultáneos:

* `chord_onset_tol = 3 frames` (30ms)  
* Un chord group \= eventos con `|t_on - t_ref| <= 3`

Para cada chord:

* anchor \= nota más grave (min pitch)  
* targets \= otras notas del chord

Token:

* `T_chord = (dt=0, dp, pc_anchor, w)`  
  * `dp = p_target - p_anchor`  
  * `pc_anchor = p_anchor % 12` (pitch class)  
  * `w = sqrt(amp_anchor * amp_target)`

Hash (discreto):

* `(type=1, dp_bin, pc_anchor)`

---

### **Token Tipo 2: Intervalos secuenciales (melódico/rítmico)**

Ordená eventos por onset; definí una “melody proxy” simple:

* Tomá, por cada onset frame, la nota con mayor amp/conf (una por frame).  
  (Esto evita combinatoria en polifonía.)

Tokens entre consecutivos:

* `T_seq = (dt, dp, pc_anchor, w)`  
  * `dt = t_{i+1} - t_i`  
  * `dp = p_{i+1} - p_i`  
  * `w = sqrt(a_i * a_{i+1})`

Hash:

* `(type=2, dt_bin, dp_bin, pc_anchor)`

---

### **Token Tipo 3: Constelación Shazam en tiempo–pitch (la “versión correcta”)**

Esto es lo que más se parece a tu enfoque actual, pero en el plano musical.

Defaults:

* `pair_window = 200 frames` (2.0s)  
* `fan_out = 4`  
* Selección de anchors: top-K eventos por amp en el segmento:  
  * `K_anchors = min(60, #events)` (si hay menos, todos)

Para cada anchor `i`, elegí targets `j` con:

* `0 < t_j - t_i <= pair_window`

Selección de targets con diversidad:

* 2 targets “cercanos”: `|dp| <= 12` semitonos  
* 2 targets “lejanos”: `12 < |dp| <= 36`

Token:

* `T_pair = (dt, dp, pc_anchor, octave_anchor, w)`  
  * `octave_anchor = p_anchor // 12`  
  * `w = sqrt(a_i * a_j)`

Hash:

* `(type=3, dt_bin, dp_bin, pc_anchor, octave_anchor_coarse)`  
  * `octave_anchor_coarse = clamp(octave_anchor, 0..8)`

---

## **A.4 Representación final por segmento**

Elegí una de estas dos (podés hacer ambas):

### **A.4.1 Bag-of-hashes (Shazam-style)**

* Un multiset de hashes con weights.  
* Score(query, cand) \= suma de weights de hashes compartidos (o conteo simple).

### **A.4.2 Histogramas 2D (dt\_bin × dp\_bin) por tipo**

* Para cada tipo (1/2/3), construís hist 2D:  
  * bins dt: `0..(pair_window/2)` → 0..100  
  * bins dp: `-36..+36` → 73 bins  
* Normalización:  
  * `hist = hist / (sum(hist) + eps)`

Para un “lenguaje de ratios” puro, el hist 2D de Tipo 3 ya es un fingerprint excelente.

---

## **A.5 GO/NO-GO interno del enfoque A**

* GO-A1: el retrieval Audio→MIDI por segmentos supera claramente azar con NEG\_SAME\_PIECE\_DIFF\_TIME.  
* GO-A2: el retrieval MIDI→Audio también (simetría).  
* Si falla:  
  * primero revisar **event extraction** (densidad y offset),  
  * recién después tocar tokens.

---

---

# **(B) TF-Constellations Shazam-Style (MIDI → pseudo-TF → mismo extractor)**

## **Idea**

Construís un TF sintético desde MIDI con sustain+armónicos y corrés **exactamente el mismo peak picking \+ pairing** que en audio. Esto convierte el problema en “dos TF maps” comparables.

## **B.0 Time–frequency grid (defaults)**

Usá la misma grilla que tu audio. Si estás libre para elegir:

* `sr = 22050`  
* `hop = 512` → `frame_dt ≈ 0.02322 s`  
* CQT:  
  * `fmin = 27.5 Hz`  
  * `bins_per_octave = 36`  
  * `n_octaves = 7`  
  * `n_bins = 252`

Si hoy ya tenés STFT/CQT definidos, **no cambies**: adaptá el MIDI-TF a eso.

---

## **B.1 MIDI → pseudo-TF (exacto)**

Para cada nota MIDI `(p, onset, offset, velocity)`:

### **Frecuencia fundamental**

* `f0 = 440 * 2^((p-69)/12)`

### **Armónicos**

* `n_harmonics = 6`  
* `harmonic_decay_alpha = 1.3`  
* Amplitud de armónico h:  
  * `A_h = A0 / (h^alpha)`

### **Detune (opcional pero recomendado)**

* `detune_std_cents = 3`  
* `f_h = h*f0 * 2^(detune_cents/1200)`

### **Envolvente temporal piano (simple y efectiva)**

* `attack = 0.01 s`  
* `decay = 0.15 s`  
* `sustain_level = 0.25`  
* `release = 0.25 s`

Pedal sustain:

* CC64 \> 64: extender offset a pedal-off

### **Construcción del mapa S\[t, fbin\]**

Definí `T = #frames`, `F = #bins`. Inicializá `S=0`.

Para cada frame t dentro de `[onset, offset+release]`:

* `env(t)` según ADSR (normalizado a 1 en el onset)  
* `A0 = (velocity/127) * env(t)`  
* para h=1..6:  
  * mapear `f_h` al bin CQT más cercano `b`  
  * sumar energía:  
    * `S[t,b] += A_h`

### **Suavizado en frecuencia (evita “TF demasiado perfecto”)**

* `freq_smooth_sigma_bins = 1.0`  
  Aplicá un gaussian blur 1D por frame sobre el eje freq (barato y efectivo).

### **Compresión dinámica (parecido al espectro real)**

* `S = log1p(gamma * S)` con:  
  * `gamma = 10.0`

---

## **B.2 Peak picking (defaults)**

Querés comparabilidad y evitar explosión de tokens.

* `peaks_per_frame_max = 8`  
* `peak_threshold_rel = 0.2` del máximo del frame **o**  
* `peak_threshold_db = -40 dB` relativo al máximo del segmento

Local max:

* pico si `S[t,b]` es máximo en vecindad `(t±1, b±1)`.

Weight:

* `w = S[t,b]` (después normalizamos)

Normalización de weights por segmento:

* `w = w / (P99(w) + eps)`  
* `w = clip(w, 0, 1)`

---

## **B.3 Pairing tipo Shazam (defaults)**

* `pair_window_sec = 2.0`  
* `pair_window_frames = round(2.0 / frame_dt)` (≈86 si 22050/512)  
* `fan_out = 4`

Selección de targets con diversidad (para evitar ratio≈1):

* 2 targets “cercanos”: `|log_ratio| <= 1.0` octava (±1 octava)  
* 2 targets “lejanos”: `1.0 < |log_ratio| <= 3.0` octavas

Token (recomendado, híbrido para evitar colisiones):

* `dt = t2 - t1` (en frames)  
* `log_ratio = log2(f2/f1)`  
* `f_anchor_bin = b1`  
* `weight = sqrt(w1*w2)`

Quantización para hashing:

* `dt_bin_size = 2 frames`  
* `log_ratio_bin_size = 1/24` octava  
* `f_anchor_coarse = f_anchor_bin // 3` (reduce sensibilidad)

Hash:

* `(dt_bin, log_ratio_bin, f_anchor_coarse)`

---

## **B.4 Representación por segmento (dos opciones)**

### **B.4.1 Shazam hash-bag**

* multiset de hashes  
* score \= suma de weights de hashes compartidos

### **B.4.2 Histogramas**

* hist 2D: (dt\_bin × log\_ratio\_bin), opcional 3D con f\_anchor\_coarse  
* normalización L1

---

## **B.5 GO/NO-GO interno del enfoque B**

* GO-B1: con matching directo (sin red) el par correcto empieza a rankear bien en pools chicos (N=256).  
* GO-B2: si entrenás un encoder sobre tokens/hist, mejora y se separa aligned vs within-piece negatives.  
* Si falla:  
  * casi siempre es porque **peak picking / pairing** está generando tokens que codifican “continuidad” (ratio≈1) y no relaciones informativas.  
  * aplicá la regla de diversidad (targets cercanos+lejanos) y bajá saturación de tokens/frame.

---

# **Recomendaciones prácticas finales (para correr ambos a la vez)**

1. **Mismo protocolo de evaluación** para A y B (mismo pool de negativos).  
2. En cada segmento guardá:  
   * `n_tokens`, `tokens_per_frame`, `mean/std log_ratio` (B) o `mean/std dp` (A)  
3. Objetivo de densidad:  
   * B: 10–80 tokens/frame (no clavado en el techo)  
   * A: \#eventos razonable (piano: decenas–cientos por 20s), no 0

---

# **Qué deberías ver si todo está sano**

* **En A:** las distribuciones de `dp` (intervalos) y `dt` deberían verse comparables entre audio-events y MIDI-events (no idénticas, pero no “universos distintos”).  
* **En B:** la distribución de `log_ratio` en MIDI-TF debería bajar hacia la zona del audio (ya no 1.3 vs 0.08) y los tokens/frame deberían dejar de ser 0.2.

---

Si querés, te armo una “matriz de ablations” mínima pero contundente para estos dos enfoques (qué parámetro tocar primero y qué métrica debería moverse), así iteran sin perderse.

