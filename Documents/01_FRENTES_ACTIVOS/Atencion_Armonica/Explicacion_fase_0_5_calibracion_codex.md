# Atención Armónica: qué enseñó la Fase 0.5

> Explicación conceptual de la auditoría de calibración posterior a Fase 0. Este texto no reemplaza al reporte numérico (`REPORTE_0.5.md`), sino que reconstruye qué problema se abrió, qué se verificó y qué cambia en la lectura arquitectónica del frente.

## El problema que había quedado abierto

La Fase 0 había mostrado una diferencia importante entre dos niveles del sistema. Por un lado, `Harmonic Pairformer`, la arquitectura con `triangle update`, producía una representación de pares muy fuerte fuera de distribución, especialmente cuando la polifonía aumentaba. En `OOD-poly`, el modelo B ordenaba mejor los pares que B-local: sus métricas `AUC/AP` eran superiores. Eso indicaba que la red estaba aprendiendo una estructura relacional útil.

Por otro lado, cuando esa misma matriz de pares se convertía en clusters con una regla simple de `connected-components` a un umbral `τ`, el agrupamiento de B colapsaba. En la celda más decisiva, `OOD-poly / poly3_hard`, B tenía buen ranking de pares, pero un `ARI@τ_val` muy bajo. La primera lectura fue natural: el umbral elegido en validación no transfería bien al test fuera de distribución.

La Fase 0.5 se diseñó para separar esas dos hipótesis:

```text
Hipótesis A:
la representación es buena, pero el umbral τ está mal calibrado.

Hipótesis B:
la representación es buena, pero connected-components no es el clusterer adecuado
para extraer la partición desde esa geometría de pares.
```

La diferencia no es menor. Si el problema era el umbral, bastaba mejorar calibración: Platt, isotonic, otro criterio de selección de `τ`. Si el problema era el clusterer, la arquitectura estaba produciendo una geometría útil, pero el sistema de lectura era demasiado frágil para convertirla en fuentes.

## Qué auditó Fase 0.5

La auditoría tomó los logits por mezcla y reconstruyó matrices `N x N` para volver a probar reglas de decisión sin reentrenar la arquitectura. La red ya había producido su juicio relacional sobre cada par. Lo que se auditó fue cómo convertir ese juicio en una partición.

El pipeline puede resumirse así:

```text
picos espectrales
      │
      ▼
red neuronal
      │
      ▼
matriz de logits pairwise
      │
      ├── ranking de pares              → AUC/AP
      │
      └── regla de partición             → ARI
             ├── connected-components a τ
             ├── oracle τ global
             ├── oracle τ por mezcla
             └── agglomerative con k verdadero
```

`AUC/AP` responde si la representación ordena bien los pares. `ARI` responde si una regla concreta logra convertir esa matriz en grupos. La Fase 0.5 existe para no confundir esos planos.

## El resultado central

El hallazgo fue más preciso que la hipótesis inicial. En `OOD-poly / poly3_hard`, el modelo B quedó así:

```text
B con connected-components:
baseline τ_val      ARI = 0.134
oracle τ global     ARI = 0.134

B con k verdadero:
agglo_true_k        ARI = 0.605
```

El dato decisivo es que `oracle τ global` no mejora nada. Incluso usando un umbral privilegiado elegido sobre el propio test, B sigue en `0.134`. Eso descarta la explicación simple de calibración: no era que el `τ` de validación no transfería. Para esa representación, no hay un umbral global de connected-components que resuelva el problema.

En cambio, cuando se usa un clusterer que conoce el número verdadero de fuentes, B salta a `0.605`, por encima de B-local (`0.482`), B-minus (`0.427`) y B-shuffle (`0.299`) en la misma celda. La representación de B no estaba rota. Estaba organizada de una manera que connected-components no logra leer.

## Por qué connected-components falla

`Connected-components` es una regla muy simple. Toma todos los pares cuya probabilidad supera un umbral y los convierte en aristas. Después declara que todos los nodos conectados por algún camino pertenecen al mismo grupo.

```text
si A conecta con B
y  B conecta con C
entonces A, B y C quedan en el mismo cluster
```

Esta regla es útil cuando los errores son pocos y no forman puentes entre fuentes. Pero tiene una fragilidad estructural: basta una pequeña cantidad de aristas cruzadas de alta confianza para encadenar grupos que deberían mantenerse separados.

En términos sonoros, puede ocurrir algo así:

```text
fuente 1:  p1 ─ p2 ─ p3
                     │
                     │  arista cruzada espuria
                     │
fuente 2:            p4 ─ p5 ─ p6
```

Aunque la mayoría de los pares estén bien ordenados, una arista equivocada puede fusionar dos fuentes completas. El ranking global puede seguir siendo bueno, porque esa arista es una excepción local; pero la partición final queda dañada, porque connected-components transforma continuidad local en equivalencia global.

Esto es especialmente importante para B. El `triangle update` parece producir una representación más estructurada, más sensible a consistencias globales y más transferible a polifonía nueva. Pero esa misma estructura puede contener aristas fuertes que, bajo una regla transitiva dura como connected-components, operan como puentes indebidos.

La paradoja se vuelve clara:

```text
B ordena mejor los pares.
Pero connected-components lee peor esa matriz.
```

No hay contradicción. Son dos operaciones distintas. Una cosa es producir una geometría relacional rica; otra es extraer de ella una partición estable.

## Qué significa `agglo_true_k`

`agglo_true_k` no es una regla deployable, porque usa el número verdadero de fuentes. No se puede presentar como solución final del sistema. Su valor es diagnóstico: muestra qué pasa si se le da al clusterer una restricción global que connected-components no tiene.

Agglomerative con `k` verdadero no pregunta solamente si cada par supera un umbral. Construye una partición con una cantidad fija de grupos. Eso impide que una arista cruzada fusione indefinidamente todo lo que toca. La decisión ya no depende solo de conectividad local, sino de una organización global de la matriz.

Por eso el salto de B con `agglo_true_k` es tan informativo. Indica que la matriz contiene información suficiente para recuperar mejor las fuentes, siempre que el procedimiento de lectura respete una restricción global de partición.

La conclusión no es que haya que usar un oráculo. La conclusión es que el siguiente sistema debe aprender o estimar una restricción equivalente: número de fuentes, corte espectral, estructura de partición, o una cabeza explícita de clustering.

## Cómo cambia la lectura de Fase 0

La lectura anterior decía:

> B tiene mejor ranking OOD-poly, pero su `τ` no transfiere.

La Fase 0.5 obliga a una formulación más precisa:

> B tiene mejor representación OOD-poly, y esa representación puede producir mejor clustering si se lee con una regla global adecuada. El problema no es principalmente el umbral, sino la fragilidad de connected-components como extractor de particiones.

Esto fortalece una parte de la hipótesis arquitectónica y debilita otra parte del sistema. Fortalece la hipótesis de que el `triangle update` aprende una geometría relacional que generaliza a polifonía nueva. Debilita la idea de que basta con convertir esa geometría en clusters mediante un umbral.

La arquitectura no queda cerrada. Queda mejor localizada.

## La implicancia arquitectónica

Hasta ahora, el sistema tenía dos piezas:

```text
Harmonic Pairformer  →  matriz same-source  →  connected-components
```

La Fase 0.5 muestra que esa última flecha es demasiado pobre. La red produce una matriz que debe leerse como una geometría de compatibilidades, no como una lista de aristas independientes.

El próximo diseño debería tratar la partición como parte del problema arquitectónico:

```text
Harmonic Pairformer
      │
      ▼
matriz relacional de pares
      │
      ▼
clusterer global / estimador de k / cabeza de partición
      │
      ▼
fuentes armónicas
```

Esto cambia el estatuto de la salida. Ya no alcanza con decir que cada par tiene una probabilidad. La matriz completa debe ser interpretada como un objeto geométrico: una superficie de compatibilidades donde los clusters son regiones coherentes, no simplemente componentes conectadas por aristas sobre un umbral.

## Una forma de verlo gráficamente

La diferencia entre las dos lecturas puede representarse así:

```text
Lectura local por umbral

matriz de pares
     │
     ▼
¿z[i,j] > τ?
     │
     ▼
aristas
     │
     ▼
componentes conectadas

Riesgo: una arista cruzada puede fusionar dos fuentes completas.
```

```text
Lectura global de partición

matriz de pares
     │
     ▼
estructura completa de compatibilidades
     │
     ▼
restricción global: k, eigengap, corte, slots o cabeza de fuente
     │
     ▼
partición armónica

Ventaja: la decisión no depende de una arista aislada, sino de la coherencia global.
```

La arquitectura que emerge de Fase 0.5 no es solamente una red que predice pares. `Harmonic Pairformer` empieza a exigir una segunda etapa igualmente estructural: una forma de leer globalmente la geometría que la red produce.

## Qué quedó demostrado y qué no

Quedó demostrado que el caveat de Fase 0 estaba mal localizado. El problema de B en `ARI@τ_val` no se corrige eligiendo mejor `τ`; el `oracle τ global` no ayuda. También quedó demostrado que la representación de B conserva información útil, porque `agglo_true_k` la extrae mejor que en los controles.

No quedó demostrado que ya tengamos un sistema deployable de agrupamiento OOD-poly. Con las reglas deployables actuales, B sigue perdiendo contra B-local en `ARI`. El resultado a favor de B aparece cuando se usa una regla privilegiada con `k` verdadero. Eso no invalida el hallazgo, pero fija su alcance.

La conclusión madura es esta:

> El `triangle update` parece mejorar la geometría relacional de la mezcla cuando aumenta la polifonía, pero esa mejora requiere un mecanismo de partición global para volverse sistema de clustering.

## Caminos que se abren

El siguiente paso no debería ser seguir ajustando calibradores de umbral. Platt e isotonic dan mejoras pequeñas; no resuelven la estructura del problema. La prioridad pasa a ser la lectura global de la matriz.

Hay tres caminos naturales:

1. Un clusterer espectral o aglomerativo con estimación deployable de `k`.
2. Una cabeza de red que prediga el número de fuentes o produzca slots de fuente.
3. Una arquitectura que pase de `same-source` pairwise a objetos latentes de fuente, con parámetros como `f0`, `beta`, envolvente y parciales asignados.

Los tres caminos comparten una misma intuición: la fuente armónica no es una arista, sino una región coherente de la geometría relacional. La red ya produce indicios de esa geometría. Fase 0.5 mostró que el trabajo pendiente es aprender a leerla sin privilegio.

## Implicancia para Fase 1

Antes de pasar a CQT y picos detectados, la lección de Fase 0.5 debe quedar incorporada. La detección real va a introducir picos faltantes, espurios y desplazados. Eso volverá aún más frágil una regla de connected-components por umbral. Si el sistema ya falla con parciales exactos en OOD-poly, con detección real esa fragilidad probablemente aumente.

Por eso Fase 1 no debería ser solamente:

```text
parciales exactos → picos CQT
```

También debería ser:

```text
connected-components → partición global robusta
```

La singularidad del frente queda más definida después de Fase 0.5. Atención Armónica no busca solo agregar features armónicas a una red. Busca construir una arquitectura capaz de pensar una mezcla como una geometría de relaciones y convertir esa geometría en fuentes. El resultado reciente muestra que la primera parte ya empezó a aparecer; la segunda exige una decisión arquitectónica nueva.
