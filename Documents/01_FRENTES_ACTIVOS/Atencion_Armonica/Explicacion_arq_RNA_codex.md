# Atención Armónica: cómo funciona la arquitectura

> Explicación conceptual de la arquitectura para lectores con conocimiento básico de audio y redes neuronales. No reemplaza al plan experimental ni al reporte de resultados; funciona como puente entre la intuición sonora, la lógica de atención y la geometría relacional que el frente está probando.

## La idea en una frase

Atención Armónica prueba una red que no mira una mezcla como una lista de picos espectrales sueltos, sino como una **red de relaciones entre picos**. La pregunta no es solamente si dos frecuencias forman un ratio plausible, sino si muchos picos juntos pueden organizarse como una explicación armónica coherente de la mezcla.

En una mezcla polifónica, un pico puede ser ambiguo. Un componente cerca de `300 Hz` puede ser el tercer armónico de una fuente de `100 Hz` o el segundo armónico de otra fuente de `150 Hz`. Mirado en forma aislada, el par puede no alcanzar para decidir. Lo que desambigua es el conjunto: qué otros picos acompañan a cada hipótesis y qué partición global queda más consistente.

La tarea de la red es:

> Dados muchos picos espectrales de una mezcla, estimar qué pares de picos pertenecen a la misma fuente armónica.

La salida principal no es todavía "este instrumento es un violín" ni "esta fuente es el piano". Es una matriz `N x N` donde cada celda responde:

```text
same-source[i,j] = ¿el pico i y el pico j vienen de la misma fuente?
```

Después, esa matriz se convierte en grupos de parciales. Ahí aparece el problema de calibración: la red puede rankear bien los pares, pero formar clusters requiere una decisión adicional.

## Dos planos de cómputo

Un transformer común representa cada entrada como un token. Acá eso sigue existiendo: cada pico tiene una representación `x[i]`. Pero además se construye una segunda representación persistente: una matriz de pares `z[i,j]`.

```text
                    PLANO TOKEN

        pico 1   pico 2   pico 3   pico 4   ... pico N
          x1       x2       x3       x4           xN
           │        │        │        │            │
           └────────┴────────┴────────┴────────────┘
                    atención entre picos
                 sesgada por la matriz z[i,j]


                    PLANO PAR

              matriz N x N de relaciones entre picos

                 j →
             ┌─────────────────────┐
          i  │ z11 z12 z13 z14 ... │
          ↓  │ z21 z22 z23 z24 ... │
             │ z31 z32 z33 z34 ... │
             │ ...                 │
             └─────────────────────┘

        cada celda z[i,j] = estado aprendido de la relación i ↔ j
```

La diferencia central está en esa matriz. En vez de calcular al final si `i` y `j` van juntos, la red mantiene una memoria viva de la relación entre cada par durante todas las capas.

## Un bloque Harmonic Pairformer

Cada bloque de la arquitectura alterna tres operaciones:

```text
┌──────────────────────────────────────────────────────────────┐
│                    BLOQUE HARMONIC PAIRFORMER                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Token attention sesgada por pares                         │
│                                                              │
│     x[i] atiende a x[j], pero la fuerza de esa atención       │
│     depende también del estado z[i,j].                        │
│                                                              │
│     x ───────────────► x'                                    │
│           bias z[i,j]                                        │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  2. Pair update desde tokens                                  │
│                                                              │
│     si cambió lo que sabemos de los picos i y j,              │
│     se actualiza también la relación z[i,j].                  │
│                                                              │
│     x'[i], x'[j] ─────────► z'[i,j]                           │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  3. Triangle update                                           │
│                                                              │
│     para actualizar z[i,j], la red mira todos los caminos     │
│     i → k → j.                                                │
│                                                              │
│     z[i,k] + z[k,j] ───────► z''[i,j]                         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

La primera operación deja que los picos se atiendan entre sí, pero con una atención informada por las relaciones. La segunda permite que los tokens corrijan el estado del par. La tercera es la pieza más específica: actualiza una relación usando evidencia indirecta a través de terceros picos.

## Qué hace el triángulo

El `triangle update` se entiende mejor como una operación sobre tripletes:

```text
              pico k
             /      \
        z[i,k]      z[k,j]
           /          \
       pico i ─────── pico j
              z[i,j]
```

Para decidir si `i` y `j` pertenecen a la misma fuente, la red no mira solo el ratio entre `i` y `j`. Mira si existen otros picos `k` que sostienen una explicación coherente:

```text
si i parece ir con k
y  k parece ir con j
entonces aumenta la evidencia de que i va con j
```

Esto no se impone como regla lógica dura. La red aprende cuándo la evidencia triangular ayuda y cuándo no. Lo importante es que la pertenencia a fuente tiene una estructura transitiva: si varios pares afirman pertenencia común, la partición final debe ser globalmente consistente.

## La geometría subyacente

La geometría que estamos probando no es una geometría euclídea de puntos en 3D. Es una **geometría relacional**.

Cada pico es un nodo. Cada par de picos tiene una relación. La matriz `z[i,j]` funciona como el espacio donde la red piensa:

```text
nodos:     picos espectrales
aristas:   compatibilidad same-source
estructura válida: partición en fuentes armónicas
```

La restricción geométrica no es que tres distancias formen un triángulo físico, como en AlphaFold. La restricción es que las relaciones de pertenencia formen clases de equivalencia coherentes. Cada clase corresponde a una fuente o serie armónica latente.

En términos generativos, cada fuente puede pensarse como una familia discreta de parciales:

```text
f_n = n · f0 · sqrt(1 + beta · n²)
```

Los picos de una misma fuente no son simplemente "cercanos" en frecuencia. Son coherentes con un mismo generador latente: `f0`, `beta`, envolvente, parciales presentes y parciales ausentes. La arquitectura no estima explícitamente esos parámetros como salida principal; aprende una compatibilidad relacional que debería poder convertirse en una partición.

La hipótesis precisa es:

> La armonía puede formularse como una geometría relacional de compatibilidad entre parciales, donde la estructura global válida es una partición en fuentes generativas armónicas.

## En qué se parece a AlphaFold y en qué no

La semejanza está en el patrón computacional:

```text
AlphaFold:
aminoácidos → pares de residuos → consistencia geométrica 3D → estructura plegada

Atención Armónica:
picos espectrales → pares de picos → consistencia de pertenencia armónica → grupos de parciales
```

En ambos casos, el objeto central no es solo el elemento individual, sino la relación entre pares. Y en ambos casos, la actualización triangular deja que terceros elementos corrijan la relación de un par.

La diferencia es la física de fondo. AlphaFold trabaja con distancias y geometría espacial 3D. Atención Armónica trabaja con pertenencia a fuentes armónicas. La restricción no es una desigualdad triangular métrica, sino la consistencia de una partición de parciales.

Por eso no alcanza con decir que los picos viven en `log f`. En `log f`, las diferencias cumplen identidades algebraicas triviales. La no trivialidad aparece cuando preguntamos qué subconjunto de picos puede explicarse por una misma fuente generativa.

## Qué resolvimos y qué quedó abierto

La `Fase 0` dejó logros concretos:

- formuló una pregunta arquitectónica real, distinta de la inyección de descriptores;
- detectó y corrigió dos datasets feature-triviales antes de gastar GPU;
- construyó un benchmark sintético con `ID`, `OOD-poly`, `OOD-regime`, seis modelos y controles;
- mostró que el pair-state explícito es el salto grande;
- mostró que el `triangle` no gana universalmente, pero sí aporta en `OOD-poly` bajo `AUC/AP`;
- confirmó con `B-shuffle` que la estructura relacional importa;
- aisló un problema pendiente de sistema: transformar buen ranking de pares en clustering estable.

La deuda actual es `Fase 0.5`: auditar calibración. La red puede producir una matriz útil de relaciones, pero formar clusters requiere decidir un umbral o un criterio equivalente. `ARI@τ_val` puede fallar aunque el ranking sea bueno. Esa separación entre representación y decisión de partición es ahora parte explícita del frente.

## Caminos derivados

Si la calibración convierte el buen ranking en agrupamiento estable, el frente puede avanzar de forma escalonada:

1. **Fase 0.5**: auditar calibración sobre logits/matrices del setup sintético ya cerrado.
2. **Fase 1a**: pasar de parciales exactos a picos detectados con CQT sobre mezclas renderizadas, manteniendo ground truth exacto.
3. **Fase 1b**: avanzar hacia audio real/stems, donde el ground truth es más difícil.
4. **Fuente como objeto latente**: pasar de matriz `same-source` a slots de fuente con `f0`, `beta`, envolvente y parciales asignados.
5. **Extensión temporal**: unir tracking de parciales y agrupamiento por fuente en el tiempo.
6. **Trunk armónico paralelo**: combinar un encoder foundation de audio con una rama explícita de relaciones armónicas.

La utilidad singular de estas variantes no sería "hacer separación de audio" en abstracto. Sería construir modelos que interpretan una mezcla como un sistema de relaciones armónicas globalmente consistentes, con salidas más explicables y potencialmente más transferibles que una máscara espectrograma opaca.
