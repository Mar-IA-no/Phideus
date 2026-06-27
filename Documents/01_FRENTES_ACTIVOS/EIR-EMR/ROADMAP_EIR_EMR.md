<div align="center">

# Roadmap EIR-EMR
### Diseño inicial del frente Expression-Invariant Ratios / Expression-Modulated Ratios

![Status](https://img.shields.io/badge/Status-Design_Mode-F59E0B?style=for-the-badge)
![Scope](https://img.shields.io/badge/Scope-Voice_Expression-1F6FEB?style=for-the-badge)
![Updated](https://img.shields.io/badge/Updated-2026--06--21-F59E0B?style=for-the-badge)

</div>

> [!IMPORTANT]
> **Propósito**: fijar una arquitectura de problema para EIR-EMR antes de abrir dataset, código o training. Este documento no presupone que la hipótesis sea verdadera. Solo organiza cómo podría ponerse a prueba dentro del marco epistemológico de Phideus.

## 1. Pregunta del frente

La pregunta inicial no debe formularse como "puede una IA sonar emocionalmente auténtica". Eso mezcla demasiados niveles de claim.

La pregunta correcta para apertura es:

> existen relaciones ratio-based relativamente invariantes y relaciones ratio-based relativamente moduladas por la expresión en la voz y en señales fisiológicas acopladas, y esas relaciones sirven como organización útil para tareas cross-modal o de control expresivo?

Esta formulación deja abierto el resultado y evita inflar desde el inicio una tesis psicológica o semántica que el diseño todavía no puede sostener.

## 2. Niveles de claim permitidos

### Nivel 1 — físico/descriptorial

Existen patrones ratio-based medibles en una o más modalidades.

### Nivel 2 — cross-modal

Parte de esos patrones reaparece entre sensores distintos del mismo fenómeno o de fenómenos acoplados.

### Nivel 3 — expresivo

Parte de esos patrones cambia sistemáticamente con estados expresivos o con modulaciones prosódicas controladas.

### Nivel 4 — generativo

Esos patrones pueden usarse para condicionar o regular una síntesis vocal.

Regla: no hablar de Nivel 4 como si Nivel 1-3 ya estuvieran cerrados.

## 3. Hipótesis de trabajo

### H1 — EIR

Hay un subconjunto de relaciones ratio-based relativamente estable a través de distintas realizaciones expresivas de un mismo hablante.

### H2 — EMR

Hay un subconjunto de relaciones ratio-based que se desplaza sistemáticamente con la modulación expresiva o fisiológica.

### H3 — Cross-modalidad

Una parte de EIR y/o EMR reaparece entre dos modalidades del mismo sistema o de sistemas fuertemente acoplados.

### H4 — Utilidad operativa

Inyectar o condicionar con esas relaciones mejora una tarea bien definida frente a baselines y controles no-ratio comparables.

## 4. Modalidades candidatas

Orden recomendado por afinidad con Phideus:

1. **Speech ↔ EGG**
   Es la continuación más natural de Escalón 2.

2. **Speech ↔ Respiración**
   Bueno para modulación temporal y estados de activación.

3. **Speech ↔ ECG/PPG**
   Conecta con Escalón 4 y con acoplamientos autonómicos más lentos.

4. **Speech-only**
   Más fácil para empezar, pero epistemológicamente más débil si la ambición del frente es cross-modal.

Recomendación actual:

- si el objetivo es mantener coherencia estricta con Phideus, abrir por `Speech ↔ EGG` o `Speech ↔ EGG + respiración`;
- evitar empezar por voice cloning o TTS.

## 5. Taxonomía descriptorial inicial

### Familia A — dinámica temporal del oscilador

Hereda de `V4-lin`:

- ratios `F0[t] / F0[t-1]`
- regularidad del período
- estabilidad local del pulso
- fuerza de voicing o periodicidad

Posible rol:
- capturar componentes `EMR` sensibles a activación, tensión, temblor o control prosódico.

### Familia B — estructura armónica intra-frame

Hereda de `H-series`:

- `H2/H1 ... Hn/H1`
- concentración armónica
- desviación armónica
- balance relativo entre subestructuras espectrales vinculadas a la fuente

Posible rol:
- capturar componentes `EIR` y `EMR` ligados a fuente glotal y su configuración.

### Familia C — controles no-ratio

Hereda de `A4-16k`:

- dinámica de energía por bandas
- pendientes locales
- medidas espectrales genéricas

Posible rol:
- control adversario para evitar vender como "ratio" lo que es simplemente una buena codificación espectral.

### Familia D — controles perceptuales / paramétricos

- escalas logarítmicas
- codificaciones mel
- proxies prosódicos no ratio-based

Posible rol:
- separar armonía natural de descriptores más cercanos a codificaciones perceptuales o ingenieriles.

## 6. Primera distinción estructural del frente

Para no usar `EIR` y `EMR` como slogans vagos, conviene fijar una primera lectura operacional:

- **EIR**:
  relaciones que muestran menor varianza intra-hablante entre estados expresivos distintos que entre hablantes distintos.

- **EMR**:
  relaciones que muestran desplazamiento consistente al pasar entre estados o condiciones expresivas definidas.

Eso no cierra la teoría, pero da una regla inicial cuantificable.

## 7. Dataset mínimo defendible

La unidad de registro debería incluir:

- hablante;
- frase o material verbal controlado;
- condición expresiva o prosódica;
- repeticiones;
- modalidad o modalidades sincronizadas;
- metadatos de calidad y protocolo.

### Requisitos mínimos

1. varias personas, no una sola;
2. varias repeticiones por condición;
3. protocolo de elicitación no caricaturesco;
4. sincronización temporal seria;
5. segmentación reproducible.

### Condiciones candidatas

No empezar por una grilla psicológica enorme. Mejor una apertura austera:

- neutral;
- activación alta vs baja;
- tensión vs relajación;
- afirmación directa vs contención;
- ironía solo si se logra operacionalizar bien, no como etiqueta decorativa.

## 8. Tareas candidatas

Orden sugerido:

### T1 — análisis descriptorial puro

Antes de entrenar nada:

- estabilidad intra-hablante;
- separabilidad entre estados;
- correlación entre modalidades;
- sensibilidad a hablante vs condición.

### T2 — retrieval o matching cross-modal

Ejemplos:

- speech ↔ EGG
- speech ↔ respiración
- speech ↔ ECG/PPG

### T3 — clasificación o ranking de estado

No como objetivo final, sino como test operativo de utilidad de descriptor.

### T4 — conditioning generativo

Solo después:

- control de estilo;
- modulación de expresividad;
- síntesis condicionada.

## 9. Mecanismos a probar si el frente avanza

No asumir de entrada que el descriptor entra por concatenación.

Orden lógico:

1. baseline sin descriptor;
2. `concat` como control mecánico simple;
3. `attn_bias` / `xattn` si la tarea lo amerita;
4. conditioned projection / `FiLM` si se busca una intervención más liviana;
5. conditioning generativo solo cuando haya base empírica suficiente.

## 10. Riesgos epistemológicos

### Riesgo 1

Confundir buena clasificación de estados con prueba de estructura expresiva profunda.

### Riesgo 2

Colapsar identidad vocal, prosodia, activación fisiológica y emoción bajo una sola variable.

### Riesgo 3

Usar etiquetas expresivas pobres o teatrales y luego sobreinterpretar descriptores.

### Riesgo 4

Tomar un resultado positivo en voice-only como si ya validara la hipótesis cross-modal.

### Riesgo 5

Llamar "auténtico" a un resultado generativo que solo replica correlatos superficiales.

## 11. Lectura programática recomendada

La formulación más coherente con Phideus hoy es esta:

- EIR-EMR no es todavía una teoría de la emoción;
- es una exploración de invariantes y modulaciones ratio-based de la expresión vocal y fisiológica;
- si eso funciona, después puede abrir implicancias para síntesis, conditioning o incluso para una línea tipo `RPU`;
- pero el frente debe ganarse esa ampliación con datos, no recibirla por anticipado.

## 12. Próximos entregables documentales

Orden sugerido dentro de esta carpeta:

1. `PLAN_DATASET_EIR_EMR.md`
2. `TAXONOMIA_DESCRIPTORES_EIR_EMR.md`
3. `PREGUNTAS_EPISTEMOLOGICAS_EIR_EMR.md`
4. `PROTOCOLO_REGISTRO_EIR_EMR.md`

Antes de cualquiera de esos, este roadmap ya deja fijado el encuadre mínimo del frente.
