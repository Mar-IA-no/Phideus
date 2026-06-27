<div align="center">

# EIR-EMR
### Expression-Invariant Ratios / Expression-Modulated Ratios

![Status](https://img.shields.io/badge/Status-Conceptual_Open-F59E0B?style=for-the-badge)
![Focus](https://img.shields.io/badge/Focus-Voice_Expression-1F6FEB?style=for-the-badge)
![Updated](https://img.shields.io/badge/Updated-2026--06--21-F59E0B?style=for-the-badge)

</div>

> [!IMPORTANT]
> **Estado actual**: este frente queda abierto como línea conceptual nueva dentro de Phideus. Todavía no hay dataset canónico, baseline, protocolo ni resultado experimental. La decisión correcta en este punto no es hablar de "autenticidad emocional" ni de "comprensión afectiva" como claim fuerte, sino fijar una pregunta experimental más disciplinada: si existen **ratios invariantes de expresión** y **ratios modulados por la expresión** que puedan medirse, compararse y eventualmente usarse como señal cross-modal o como condicionamiento descriptor-guided.
> **Hipótesis de trabajo**: una parte de la expresión vocal podría descomponerse en dos familias distintas de estructura. `EIR` nombra regularidades ratio-based que permanecen relativamente estables a través de distintas realizaciones expresivas. `EMR` nombra regularidades ratio-based que cambian sistemáticamente con el estado expresivo, la prosodia o la modulación fisiológica.
> **Próximo paso único**: diseñar el problema en términos de fenómeno físico, modalidades medibles, protocolo de dataset y taxonomía de descriptores, antes de abrir cualquier experimento de modelado o síntesis.

## Qué es este frente

EIR-EMR nace de una intuición fuerte pero todavía no verificada: que la voz expresiva no debería leerse primero como una colección difusa de etiquetas afectivas, sino como una organización física y relacional del fenómeno vocal. En ese marco, el interés de Phideus no estaría en "reconocer emociones" como hace una taxonomía estándar de SER, ni en producir una voz "más humana" en un sentido genérico, sino en aislar qué parte de la estructura expresiva se conserva, qué parte se modula y cómo esas dos capas pueden describirse con relaciones trazables.

La distinción propuesta es esta:

- **Expression-Invariant Ratios (`EIR`)**: relaciones relativamente estables que caracterizan una identidad estructural de la voz o del sistema oscilatorio subyacente.
- **Expression-Modulated Ratios (`EMR`)**: relaciones que cambian con la modulación expresiva, el estado fisiológico o la configuración prosódica.

Ese par permite formular preguntas mucho más limpias que "detectar emoción auténtica". Por ejemplo:

- qué relaciones persisten entre una misma persona hablando en estados expresivos distintos;
- qué relaciones cambian sistemáticamente cuando la expresión cambia;
- qué relaciones son específicas de una modalidad y cuáles reaparecen en más de un sensor;
- qué parte del patrón es idiosincrático de un hablante y qué parte generaliza entre hablantes.

## Por qué esto sí puede pertenecer a Phideus

Phideus ya no es solo un programa de retrieval musical. Su pregunta de fondo es si ciertas estructuras armónicas o ratio-based funcionan como organizadores privilegiados de información cross-modal. EIR-EMR puede entrar legítimamente en ese arco porque desplaza la pregunta a otro fenómeno oscilatorio: la voz y, eventualmente, su acoplamiento con otras señales fisiológicas.

La afinidad con el programa ya existe en tres niveles:

1. **Escalón 2** ya abrió el dominio `Speech ↔ EGG` como prueba de invariantes del oscilador glotal.
2. **Escalón 4** ya reservó el espacio de expansión fisiológica fuera de acústica.
3. La tecnología descriptorial de Phideus ya dejó un repertorio reutilizable para separar:
   - dinámica temporal del oscilador,
   - estructura armónica intra-frame,
   - controles no-ratio,
   - variantes perceptuales o paramétricas de comparación.

La condición epistemológica es no exagerar lo que este frente diría. Un resultado positivo en EIR-EMR no probaría por sí solo "comprensión emocional auténtica". A lo sumo mostraría que cierta capa de la expresión vocal y fisiológica puede describirse con invariantes y modulaciones ratio-based útiles para alineación, clasificación o condicionamiento.

## Qué reutiliza de Phideus

Este frente no parte de cero. Reutiliza directamente varias decisiones metodológicas ya refinadas en el programa:

### 1. Taxonomía descriptorial

La lógica de Escalón 2 ya ofrece una plantilla clara:

- **Familia A**: dinámica temporal del oscilador;
- **Familia B**: estructura armónica natural intra-frame;
- **Familia C**: controles no-ratio espectrales o energéticos;
- **Familia D**: variantes perceptuales, logarítmicas o paramétricas.

EIR-EMR puede reaprovechar esa separación casi sin tocarla.

### 2. Regla metodológica

Antes de inferir nada fuerte, Phideus ya aprendió a distinguir:

- observación;
- hipótesis;
- inferencia.

Y también a separar:

- efecto del descriptor;
- efecto del mecanismo de inyección;
- efecto del encoder o del régimen.

Eso es especialmente importante en voz, donde la sobreinterpretación psicológica aparece rápido.

### 3. Mecanismos de inyección

La infraestructura ya explorada en Phideus sigue siendo relevante:

- `concat`;
- `attn_bias`;
- `xattn`;
- projection conditioning / `pca` / `FiLM`.

El frente no debería asumir de entrada que el descriptor debe mejorar síntesis o clasificación por mero contenido. La lección de Escalón 1 y Escalón 2 es más fina: muchas veces el descriptor funciona mejor como principio organizador que como feature agregado.

## Qué no debe afirmarse todavía

Conviene fijarlo desde el inicio para no contaminar el frente:

- No afirmar que EIR-EMR "entiende emociones".
- No afirmar que una voz sintética condicionada por estos descriptores sería automáticamente "auténtica".
- No colapsar ironía, ambigüedad, doble sentido y emoción bajo una sola etiqueta experimental.
- No usar un resultado de clasificación como si fuera prueba de estructura ontológica del afecto.

Lo científicamente defendible, al menos al inicio, es otra cosa:

> investigar si hay estructuras ratio-based relativamente invariantes y relativamente modulables en la expresión vocal y fisiológica, y si esas estructuras mejoran tareas de alineación, identificación de estado o control expresivo bajo protocolos trazables.

## Modalidades plausibles

La versión más fértil de este frente probablemente no sea voz aislada, sino voz más otra señal del mismo fenómeno o de un acoplamiento fisiológico vecino.

Candidatas naturales:

- `Speech ↔ EGG`
- `Speech ↔ Respiración`
- `Speech ↔ ECG/PPG`
- `Speech ↔ Voice-only` como fase más pobre pero más simple para apertura

La lógica preferible para Phideus sería:

1. empezar por pares de modalidades del mismo sistema o de sistemas fuertemente acoplados;
2. recién después abrir síntesis o clonación expresiva.

## Programa mínimo del frente

El frente necesita cuatro capas, en este orden:

1. **Definición del fenómeno**
   Qué parte de la expresión se quiere medir exactamente.

2. **Definición del dataset**
   Qué personas, qué estados, qué protocolo de actuación o evocación, qué modalidades, qué segmentación, qué control de calidad.

3. **Definición de descriptores**
   Qué sería `EIR`, qué sería `EMR`, qué controles no-ratio se usarían y cómo se normalizan.

4. **Definición de tarea**
   Retrieval cross-modal, clasificación de estado, disentanglement identidad/expresión, conditioning de síntesis, o alguna secuencia por etapas.

Sin esas cuatro capas, este frente corre el riesgo de disolverse en una idea sugerente pero metodológicamente suelta.

## Relación con el resto del programa

- **Escalón 1** sigue siendo la validación fuerte de la mecánica descriptor-guided.
- **Escalón 2** sigue siendo la primera prueba disciplinada de armonía natural en voz.
- **EIR-EMR** podría leerse como una bifurcación desde Escalón 2 hacia expresión vocal y organización ratio-based de estados.
- **Escalón 4** sigue siendo el horizonte fisiológico más amplio donde esta línea podría converger.

En otras palabras: EIR-EMR no reemplaza Escalón 2 ni Escalón 4. Puede funcionar como un puente entre ambos.

## Próximo paso recomendado

El siguiente paso correcto no es entrenar un modelo ni abrir una batería de labels emocionales. Lo correcto es producir un diseño inicial del frente con:

1. pregunta experimental;
2. modalidades;
3. dataset mínimo;
4. taxonomía de descriptores;
5. baseline y controles;
6. secuencia de tareas.

Ese trabajo continúa en [ROADMAP_EIR_EMR.md](ROADMAP_EIR_EMR.md).
