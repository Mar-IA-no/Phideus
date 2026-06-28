# Qué es un descriptor de Phideus (y cómo proponer uno)

> Documento para un colega ingeniero en sonido que estudia armonía natural.
> Objetivo: que entiendas qué es exactamente un *descriptor* en Phideus, cómo funciona uno real
> (H-series) y, sobre todo, que tengas lo necesario para **proponer la lógica de un descriptor
> nuevo**. Vos ponés el criterio físico-armónico; nosotros lo resolvemos en código.

---

## 1. La idea en una frase

Un **descriptor de Phideus** es una función que toma una señal (un sonido, una vibración, lo que
sea que tenga espectro) y devuelve, **instante a instante**, un puñado chico de números que
describen la **estructura relacional** de su contenido armónico — es decir, los **ratios** entre
sus componentes de frecuencia, no sus valores absolutos.

No es un "feature" cualquiera. Es una manera deliberada de mirar la señal: *qué relación guardan
entre sí sus parciales*, leída como si esas relaciones fueran un lenguaje.

## 2. La apuesta de Phideus (por qué ratios, y por qué naturales)

Tres compromisos que distinguen a un descriptor de Phideus de un feature acústico común:

- **Relacional, no absoluto.** Importa `H2/H1`, no `H2`. Un descriptor de Phideus mira la
  *proporción* entre componentes. Consecuencia práctica: queda invariante a la ganancia, al
  volumen, y en buena medida al registro — lo que cambia es la *forma* de la relación armónica,
  no su escala.

- **Armonía natural, no temperada/perceptual.** Los ratios se miden en **frecuencia lineal**
  (3/2, 5/4, n·f0…), no en semitonos, ni mel, ni log2. Nada de rejillas perceptuales humanas:
  la referencia es la serie armónica física. (Es una directiva dura del proyecto.)

- **Los ratios como lenguaje cross-modal.** La hipótesis de fondo de Phideus es que esa
  estructura de ratios es informacionalmente rica y *compartible entre dominios* (audio,
  vibración, voz, señales fisiológicas). El descriptor es el instrumento que la hace medible.

## 3. El contrato — qué tiene que cumplir una función para ser un descriptor de Phideus

Si vas a proponer uno, apuntá a estas seis propiedades. No son burocracia: son lo que lo hace
**usable por un modelo y comparable entre señales**.

1. **Entra una señal, sale una secuencia.** Input: forma de onda (o espectro) + lo que haga falta
   para anclarlo (p. ej. el F0 estimado). Output: una matriz `[T, D]` — un vector de `D`
   dimensiones por cada *frame* de tiempo. (Hay una variante "global" — un solo vector por señal,
   ej. histogramas — pero hoy preferimos frame-level porque preserva la dinámica temporal.)

2. **Cada dimensión es relacional y tiene nombre físico.** Cada una de las `D` columnas mide una
   relación concreta que vos podés nombrar ("ratio del 3er armónico al fundamental",
   "concentración de energía en la serie", "pendiente de caída"). Nada de cajas negras.

3. **Bajo en dimensión e interpretable.** Típico `D = 4 a 12`. La gracia es que sea legible, no
   un embedding gigante. Si necesitás 200 dimensiones, probablemente no es un descriptor sino otra
   cosa.

4. **Invariancias declaradas.** Tenés que decir a qué debe ser *insensible* (ganancia, fase,
   canal, quizás pitch absoluto) y a qué debe ser *sensible* (la forma de la relación armónica).
   Las invariancias son la mitad del diseño.

5. **Normalizable y comparable entre instancias.** Los números tienen que ser comparables entre
   señales distintas (entre hablantes, entre tomas). En la práctica se z-normaliza con
   estadísticas **congeladas** calculadas una vez sobre un conjunto de referencia. (Por qué
   congeladas: para que el descriptor signifique lo mismo en train y en test.)

6. **Aprendible pero no trivial.** Tiene que cargar señal que un modelo pueda aprovechar — pero
   idealmente *no* ser algo que el modelo ya extrae solo. El valor de un descriptor está en lo que
   aporta *por encima* de lo que la red ya ve. (Esto lo medimos nosotros; te lo cuento para que
   sepas que "que sea informativo" no alcanza: tiene que ser informativo *y* no redundante.)

Si una idea tuya cumple 1–5 y *podría* cumplir 6, es un descriptor de Phideus candidato.

## 4. El arco en Phideus (para que veas que el concepto es general)

El descriptor es el instrumento central del proyecto, y fue mutando de forma sin cambiar de
espíritu:

```
Histogramas de ratios   →   Constelaciones (tokens sparse)   →   Descriptores frame-level
(distribución global de      (pares de picos estilo Shazam:       (un vector [T, D] alineado al
 relaciones de frecuencia     "este parcial vs aquel, a tal        tiempo: H-series, intervalos,
 en toda la señal)            distancia y ratio")                  bandas espectrales…)
```

Todos responden la misma pregunta —*¿cómo se relacionan entre sí las componentes de frecuencia?*—
en audio, vibración, MIDI, voz y señales fisiológicas. Cambia la forma de empaquetar la respuesta,
no la pregunta.

## 5. Ejemplo resuelto: H-series (en lenguaje de ingeniería de sonido)

H-series mide la **estructura de amplitudes de la serie armónica**: cómo se reparte la energía
entre el fundamental y sus armónicos, frame a frame. Es 8-dimensional.

**Cómo se lee la señal:**

```
forma de onda  +  F0 estimado (pyin)  +  máscara de sonoridad (voiced/unvoiced)
      │
      ▼
  STFT  (ventana 2048 muestras @16 kHz  →  resolución ≈ 7.8 Hz por bin)
      │   → magnitud del espectro, frame a frame
      ▼
  Para cada frame SONORO y cada armónico h = 1..6:
      bin_esperado = round(h · F0 / 7.8)
      H_h = pico máximo de magnitud en bin_esperado ± 2 bins   ◄── "peak picking" con tolerancia:
            (toma el máximo local, no el bin exacto)               la voz real es algo inarmónica;
      │                                                            ±2 bins captura el parcial corrido
      ▼
  H1, H2, H3, H4, H5, H6   (amplitud de cada armónico, por frame)
```

**Las 8 dimensiones** (todas se ponen en 0 en frames sin pitch):

| dim | qué mide | cómo | lectura de ing. de sonido |
|-----|----------|------|---------------------------|
| 0–4 | **ratios armónicos** | `log(H₂/H₁), log(H₃/H₁), …, log(H₆/H₁)` | la *forma del espectro armónico* relativa al fundamental — el corazón relacional |
| 5 | concentración armónica | `Σ(H₁..H₆) / energía_total` | cuánta energía está en la serie vs ruido/inarmónico (parecido a un HNR acotado) |
| 6 | desviación armónica | `std(log(Hₙ/H₁))` | cuán *irregular* es la caída de armónicos (envolvente lisa vs dentada) |
| 7 | fuerza de sonoridad | sonoridad suavizada (3 frames) | presencia/estabilidad de la fonación |

Después: se normaliza con estadísticas congeladas (z-score, solo en frames sonoros) y se
**interpola en el tiempo** para que los `T` frames del descriptor coincidan con los frames del
modelo que lo va a consumir. Sale `[T, 8]`.

Fijate que el núcleo (dims 0–4) son **puros ratios de amplitud armónica**: no importa cuán fuerte
suena, importa *cómo decae la serie*. Eso es un descriptor de Phideus de manual.

## 6. Cómo proponer un descriptor nuevo (esto es lo que necesito de vos)

Llená esta plantilla — en tu vocabulario, sin preocuparte por el código:

```
NOMBRE TENTATIVO:
QUÉ FENÓMENO ARMÓNICO CAPTURA (en una frase):
POR QUÉ ES "ARMONÍA NATURAL" (qué relación física/espectral mide):
DIMENSIONES (lista, cada una con su nombre y la relación/ratio que mide):
CÓMO SE LEE DE UN ESPECTRO, FRAME A FRAME (a nivel conceptual):
A QUÉ DEBE SER INVARIANTE (ganancia, pitch, canal…):
A QUÉ DEBE SER SENSIBLE (qué cambio en la señal debería moverlo):
QUÉ ESPERARÍAS QUE DISTINGA (p. ej. "tensión vs relajación", "cuerda pulsada vs frotada"):
```

**Seis semillas** por si ayudan a arrancar (todas son armonía natural, todas encajan en el
contrato):

- **Inarmonicidad (β).** Cuánto se apartan los parciales del múltiplo entero exacto:
  `f_n ≈ n·f₀·√(1+β·n²)`. Física de cuerda rígida / placa. (De hecho ya la estamos modelando en
  otro frente — te va a sonar.)
- **Decaimiento espectral (α).** La pendiente con que caen las amplitudes a lo largo de la serie
  — el "brillo", pero medido como tasa relativa, no como nivel absoluto.
- **HNR / armónico-vs-aperiódico.** Energía en parciales sobre energía total — soplo, ronquera,
  textura.
- **Ratios de formantes.** `F₂/F₁`, `F₃/F₁` — estructura de resonancias como relaciones (timbre,
  vocal, identidad de cuerpo resonante).
- **Rugosidad / batidos.** Disonancia sensorial por parciales en ratios casi-enteros (mistuning
  chico → batido). Muy "armonía natural".
- **Balance par/impar.** `Σ H_par / Σ H_impar` — clarinete vs cuerda, simetría de la fuente.

No necesito que estén perfectas. Con que me des **el fenómeno, las dimensiones relacionales y las
invariancias**, lo demás lo resolvemos.

## 7. Qué resolvemos nosotros (no te preocupes por esto)

Vos das la lógica físico-armónica. Nosotros nos encargamos de:

- Convertirlo a tensores frame-level `[T, D]` y alinearlo temporalmente con el modelo.
- Congelar las estadísticas de normalización (y manejar el caso de re-calibrar por sujeto).
- Decidir cómo se *inyecta* en la red (concatenación, modulación tipo FiLM, atención cruzada…) y
  con qué inicialización para no romper el modelo base.
- Entrenar, evaluar, y —clave— medir si aporta **por encima** de lo que la red ya extrae sola
  (la propiedad 6 del contrato): es la prueba de fuego de que el descriptor sirve.

---

### Resumen para llevar

Un descriptor de Phideus es **una lente relacional sobre el espectro**: un vector chico por frame
que dice *cómo se relacionan entre sí los parciales*, en ratios de frecuencia natural, comparable
entre señales y consumible por un modelo. H-series es un ejemplo (la forma de la serie armónica).
Si podés nombrar un fenómeno armónico, expresarlo como relaciones entre componentes, y decir a qué
debe ser invariante — ya tenés un descriptor candidato. Pasámelo en la plantilla y lo hacemos.
