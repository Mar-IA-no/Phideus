# Explicación Gate 6

Gate 6 nace de una tensión muy específica que dejó Gate 5B.

Por un lado, `Test02` terminó de confirmar que los descriptores sí importan: cuando se conserva la arquitectura y se destruye la información del descriptor, el rendimiento cae a banda `D0`. Por otro lado, `Test13G-B` mostró algo incómodo: un decoder moderado sobre features pre-pooling no encontró una ventaja descriptor-guided clara. Los descriptores organizan el espacio de retrieval, sí, pero esa organización todavía no apareció como una transcripción mejor.

Gate 6 toma esa incomodidad como punto de partida.

## La pregunta real

La pregunta ya no es si los descriptores ayudan a retrieval. Esa parte quedó bastante bien resuelta.

La pregunta ahora es otra:

**¿esa ventaja geométrica llega a una tarea musical concreta o se queda encerrada en las distancias del embedding?**

Para probarlo, Gate 6 usa AMT: **Automatic Music Transcription**. Es una tarea útil porque obliga al sistema a responder algo musicalmente exigente: convertir audio en notas MIDI con onset, duración y, en algunos casos, velocity.

## Exp 0 — Primero calibrar la herramienta

Antes de modificar nada, hacía falta verificar que el modelo SOTA elegido, `Transkun v2`, funciona razonablemente bien con nuestros propios segmentos.

Ese es el rol de `Exp 0`.

- Se toman `100` segmentos MAESTRO de validación.
- Mitad de `4s`, mitad de `16s`.
- Se transcriben con el modelo pretrained.
- Se comparan contra el MIDI ground truth con la convención fijada para evaluación.

El resultado importante no es “sacar un número lindo”, sino saber si la herramienta está sana.

Y lo está.

- En `4s`, `frame_F1 = 0.784`.
- En `16s`, `frame_F1 = 0.814`.
- Las métricas de nota y offset bajan en segmentos cortos, como era esperable por efectos de borde, pero no aparece una falla de setup que invalide la línea.

Eso deja a `Transkun` validado como banco de pruebas para las fases siguientes.

## Exp A — ¿A4 le aporta algo a un SOTA?

Acá la pregunta es bastante directa.

`Transkun` ya ve el espectrograma completo y ya es muy fuerte en MAESTRO. Entonces, si `A4` mejora algo encima de eso, la lectura sería fuerte: el descriptor estaría aportando información o estructura que el pipeline estándar no está aprovechando del todo.

La clave metodológica es el control.

No alcanza con comparar `Transkun` pretrained contra `Transkun + A4`, porque ahí se mezclan dos efectos:

- más parámetros entrenables,
- información adicional real.

Por eso `Exp A` se diseñó con pares param-matched:

- `A4-event` vs `finetune-noA4`
- `A4-adapter` vs `adapter-noA4`

En ambos casos, los controles tienen el mismo presupuesto entrenable, el mismo schedule y la misma data. La única diferencia relevante es si entra `A4` real o una señal vacía.

Si aparece una mejora consistente, la lectura sería: `A4` aporta algo complementario incluso a un SOTA.

Si no aparece, la lectura también sería útil: el espectrograma ya contenía esa señal de manera suficientemente explotable.

## Exp B — ¿A4 ayuda más cuando el audio empeora?

Este experimento existe porque un resultado nulo en limpio no cerraría necesariamente la historia.

Es perfectamente plausible que `A4` sea redundante cuando el audio está limpio y muy bien representado, pero se vuelva valioso cuando la señal se degrada.

Por eso `Exp B` introduce:

- ruido gaussiano,
- low-pass,
- limitación de datos.

Y fija una regla importante:

**A4 se computa siempre desde el audio degradado.**

Eso evita cualquier contaminación tipo oracle. Si `A4` ayuda, tiene que ayudar bajo las mismas condiciones degradadas que ve el sistema principal.

La hipótesis es simple: a mayor degradación, mayor margen para que un descriptor robusto aporte algo.

## Exp C — Volver a nuestras features, pero con un decoder serio

`Exp C` es la rama más cercana a Gate 5B.

`Test13G-B` ya hizo una primera prueba sobre features pre-pooling y devolvió un resultado flojo y bastante uniforme. Pero ese decoder era todavía un probe relativamente modesto.

La nueva pregunta es:

**si en vez de un probe moderado usamos un decoder AMT serio, aparece una diferencia real entre `D0`, `d4a4`, `a4r` y `d4-a4r`?**

Eso no convertiría automáticamente al modelo en un gran transcriptor. Pero sí diría algo importante sobre el contenido musical que quedó accesible en las representaciones internas.

- Si `d4a4` o `a4r` superan claramente a `D0`, la tesis descriptor-guided gana una validación nueva: no solo retrieval, también mayor utilidad downstream.
- Si todos quedan parecidos, se refuerza una lectura más austera: la ventaja vive sobre todo en la geometría del embedding, no en una representación internamente más “transcribible”.

## Qué lugar ocupa Gate 6 en la estrategia

Gate 6 no reemplaza a Escalón 2.

El programa ya decidió que, después del cierre de Gate 5B, el foco principal pasa a `Speech <-> EGG`. Eso sigue vigente.

Gate 6 ocupa otro lugar:

- es una validación downstream;
- se apoya en el cierre de Gate 5B;
- corre en paralelo sin reabrir el gate anterior.

En otras palabras, Gate 5B dejó la pregunta suficientemente refinada como para que Gate 6 tenga sentido. Y Gate 6 existe para responder justamente si la ventaja descriptor-guided es solo geométrica o también musical en un sentido más concreto.
