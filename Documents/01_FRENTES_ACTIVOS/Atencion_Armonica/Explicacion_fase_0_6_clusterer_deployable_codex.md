# Atención Armónica: qué enseñó la Fase 0.6

> Explicación conceptual de la etapa posterior a `Fase 0.5`. Este texto no reemplaza a `REPORTE_0.6.md`, sino que reconstruye qué problema seguía abierto después del post-audit de calibración, qué se probó ahora y qué cambia en la lectura arquitectónica del frente.

## El problema que había quedado vivo después de Fase 0.5

La `Fase 0.5` había corregido una lectura importante, pero no había cerrado todavía la pregunta del sistema. Antes de esa auditoría, el colapso de `B` en `OOD-poly` podía leerse de una manera relativamente simple: el modelo ordenaba bien los pares, pero el umbral `τ` elegido en validación no transfería cuando la polifonía del test aumentaba. La explicación era cómoda porque dejaba todo el peso del problema en la calibración.

El post-audit de `Fase 0.5` obligó a una formulación más dura. En `OOD-poly`, ni siquiera un `oracle_tau_global_test` mejoraba a `B` bajo `connected-components`. Eso descartó la hipótesis “el ranking es bueno pero el `τ` está mal elegido”. El problema no estaba en el operating point. Estaba en la regla de partición.

La situación que quedó al cierre de `Fase 0.5` podía resumirse así:

```text
la representación de B mejora
pero connected-components no sabe leerla
```

Eso cambia mucho el estatuto del frente. Si el problema fuera `τ`, la salida natural sería insistir con calibradores, criterios alternativos de selección o thresholds más finos. Si el problema está en el clusterer, entonces el ranking pairwise de `B` ya no basta como argumento ni como sistema. Hace falta mostrar que existe una **lectura deployable** de esa geometría relacional.

La `Fase 0.6` se abre exactamente ahí.

## Qué auditó Fase 0.6

La pregunta ya no era si la arquitectura producía una mejor representación. Esa parte estaba, al menos parcialmente, respondida. La pregunta era más precisa:

> ¿Existe una regla de clustering **deployable**, es decir, sin `k` verdadero ni información privilegiada de test, capaz de extraer la ventaja representacional de `B` en `OOD-poly`?

Eso es distinto de preguntar si existe alguna lectura posible de la matriz. Esa pregunta ya tenía una respuesta diagnóstica positiva desde `agglo_true_k`. Lo que faltaba era saber si esa señal seguía estando disponible cuando se la obliga a pasar por una regla que podría usarse realmente como sistema.

La `Fase 0.6` tomó las matrices ya guardadas por mezcla y volvió a plantear el problema en estos términos:

```text
representación de pares
        │
        ▼
matriz de compatibilidades same-source
        │
        ├── lectura local por conectividad
        │       cc@τ_val
        │       cc_bridge_prune
        │
        └── lectura global de partición
                spectral_eigengap
                agglo_estimated_k
```

La arquitectura no cambió. El dataset no cambió. Los logits no cambiaron. Lo único que cambió fue la forma de convertir la geometría pairwise en fuentes.

Ese detalle importa porque fija bien el alcance del hallazgo. `Fase 0.6` no rediseña `Harmonic Pairformer`. Rediseña la manera de leer su salida.

## El punto de partida: por qué `cc_bridge_prune` era necesario pero insuficiente

La primera familia deployable atacó exactamente el fallo diagnosticado en `Fase 0.5`: los puentes.

La idea de `cc_bridge_prune` es simple. Si una arista `i-j` une dos picos que realmente pertenecen a la misma fuente, lo esperable es que compartan bastante vecindario: ambos deberían conectarse con otros parciales de esa misma familia. Si en cambio `i-j` es un puente espurio entre fuentes distintas, su solapamiento de vecinos debería ser bajo.

La receta opera así:

```text
1. construir aristas con prob >= τ_val
2. medir overlap de vecindarios entre endpoints
3. podar aristas con overlap < θ_prune
4. recién ahí correr connected-components
```

Como diagnóstico, funciona. En `OOD-poly / poly3_hard`, `B` pasa de:

```text
cc@τ_val         0.134
cc_bridge_prune  0.357
```

Es una mejora grande. Confirma que el problema no era una fantasía metodológica: había puentes falsos de alta confianza y podarlos recupera parte de la estructura.

Pero también deja ver el límite de esa familia. Incluso después de la poda, `B` sigue por debajo de `B-local` en la misma celda:

```text
B         0.357
B-local   0.392
```

Y el contraste bootstrap bajo regla común `cc_bridge_prune` queda negativo:

```text
B vs B-local, OOD-poly / poly3_hard = -0.035
CI95 = [-0.042, -0.029]
```

La lectura correcta no es “bridge pruning falló”, sino otra. Bridge pruning confirma el diagnóstico de `Fase 0.5`, pero también muestra que el problema de `B` no se reduce a un pequeño defecto de conectividad local. La representación necesita una lectura más global que connected-components con poda.

## El resultado central: clusterers globales deployables sí recuperan a B

La novedad fuerte de `Fase 0.6` aparece cuando la lectura deja de depender de aristas y pasa a depender de una partición global.

Las dos reglas deployables que hacen ese trabajo son:

- `spectral_eigengap`
- `agglo_estimated_k`

Las dos comparten una intuición común. Ya no preguntan simplemente qué aristas superan un umbral, sino qué organización global de la matriz produce una partición más coherente. La diferencia con `connected-components` es estructural: una arista cruzada deja de tener poder absoluto para fusionar todo lo que toca.

En `OOD-poly / poly3_hard`, el cuadro queda así:

```text
B
  cc@τ_val              0.134
  cc_bridge_prune       0.357
  spectral_eigengap     0.460
  agglo_estimated_k     0.465
  ref_k_known           0.607

B-local
  cc@τ_val              0.252
  cc_bridge_prune       0.392
  spectral_eigengap     0.412
  agglo_estimated_k     0.414
  ref_k_known           0.481
```

La lectura importante no está solo en que `B` mejora mucho. Está en que cambia el orden relativo entre modelos. Bajo `connected-components`, `B` era el peor de la familia fuerte en esa celda. Bajo clusterers globales deployables, pasa a ser el mejor.

Y esta vez el hallazgo ya no depende solo de elegir “la mejor regla para cada modelo”. También aparece bajo **regla común fija**:

```text
OOD-poly / poly3_hard

común spectral_eigengap:
  B vs B-local = +0.048
  CI95 = [+0.042, +0.054]

común agglo_estimated_k:
  B vs B-local = +0.051
  CI95 = [+0.045, +0.057]
```

Eso cambia de manera decisiva la lectura que el frente puede sostener. Ya no se trata solamente de decir:

> “B gana cuando se le busca su mejor regla”.

Ahora puede decir algo más fuerte y más limpio:

> “B gana en `OOD-poly` bajo una familia global deployable de clusterers, incluso cuando la regla es común para ambos modelos”.

Ese es el verdadero núcleo de `Fase 0.6`.

## Por qué esto no es simplemente “mejor tuning”

Hay una tentación natural a leer `Fase 0.6` como un ajuste fino más inteligente. No conviene. Lo que pasó acá no es que un ingeniero eligió mejor los knobs. Lo que apareció es una diferencia de **régimen de lectura**.

`Connected-components` toma decisiones locales y las vuelve transitivas de manera dura:

```text
si i conecta con j
y  j conecta con k
entonces i y k ya quedan fusionados
```

Esa transitividad es ciega a la forma global de la matriz. Le da a una arista fuerte el poder de cerrar una equivalencia completa.

Los clusterers globales hacen otra cosa. Tratan la matriz de compatibilidades como una estructura que debe partirse de forma coherente en grupos. Una arista alta importa, pero ya no reina sola. Tiene que convivir con el resto de la superficie relacional.

La diferencia puede verse así:

```text
lectura local
  aristas -> componentes

lectura global
  matriz -> partición
```

`Fase 0.6` no mejora una regla vieja. Cambia de ontología operativa: deja de tratar el problema como conectividad de edges y pasa a tratarlo como organización global de una partición.

## El caveat nuevo: el problema ya no es `τ`, es `k`

Que `spectral` y `agglo` deployables recuperen a `B` no significa que el frente ya cerró el sistema de agrupamiento. Lo que cambia es la localización exacta de la deuda.

Antes de `Fase 0.5`, la deuda parecía ser:

```text
representación buena
pero τ mal calibrado
```

Después de `Fase 0.5`, pasó a ser:

```text
representación buena
pero connected-components insuficiente
```

Después de `Fase 0.6`, la deuda se vuelve todavía más precisa:

```text
representación buena
clusterer global parcialmente adecuado
pero estimación de k todavía sesgada a la baja
```

La evidencia es visible en la distribución de `k` estimado para `B` en `OOD-poly`:

```text
{1: 792, 2: 9492, 3: 1716}
```

El verdadero `poly3` debería empujar mucho más fuerte hacia `k=3`. En cambio, la masa dominante cae en `k=2`. Eso significa que el sistema ya no se está rompiendo por un puente aislado, pero sí sigue fusionando fuentes porque su lectura global todavía subestima cuántas hay.

Por eso `B` mejora mucho y aun así no llega a la referencia privilegiada:

```text
agglo_estimated_k   0.465
ref_k_known         0.607
```

La deuda no desapareció. Cambió de forma. Y esa nueva forma es mucho más útil, porque ya no es una ambigüedad general sobre “qué le pasa a B”, sino un cuello arquitectónico bien localizado.

## Qué cambia en la lectura de Fase 0

La secuencia completa del frente ahora puede leerse de forma bastante más nítida.

`Fase 0` había dejado tres capas:

1. `B-minus ≫ A-rich`: el pair-state importa.
2. `B ≫ B-shuffle`: la estructura del triángulo importa.
3. `B > B-local` en `AUC/AP` `OOD-poly`: el triángulo parece ayudar a generalizar fuera de distribución.

El problema era que la tercera capa seguía siendo vulnerable a una objeción fuerte:

> quizá B solo rankea mejor, pero no se convierte en mejor sistema de agrupamiento.

`Fase 0.5` desmontó una primera versión de esa objeción: no era culpa de `τ`.

`Fase 0.6` desmonta una segunda versión:

> no, tampoco era solo una ventaja representacional sin salida deployable;
> con la familia correcta de clusterers globales, B sí se vuelve mejor sistema en `OOD-poly`.

Eso no universaliza la victoria. En `IID` y `OOD-regime`, `B-local` sigue ganando. Pero acota mucho mejor qué tipo de positividad tiene el frente:

- no “el triángulo siempre gana”;
- no “el triángulo solo sirve como ranking threshold-free”;
- sino:

> “el triángulo produce una representación cuya ventaja aparece justamente cuando cambia la polifonía, y esa ventaja ya puede extraerse con una lectura deployable global”.

Ese es un resultado más fuerte y más preciso que el que había al cierre de `Fase 0.5`.

## Qué no hay que sobrededucir

La `Fase 0.6` no autoriza a contar la historia como si el problema de agrupamiento estuviera resuelto. Tampoco autoriza a borrar la asimetría por split.

En `IID`, `B-local` sigue arriba. En `OOD-regime`, también. Eso importa porque impide una narrativa demasiado lisa donde el triángulo simplemente “vence” una vez que el clusterer se vuelve más sofisticado. No es eso lo que muestran los datos.

Lo que muestran es algo más singular:

- cuando la distribución cambia por **polifonía**, la estructura relacional de `B` parece más útil;
- cuando la distribución cambia por **régimen** sin ese aumento de cardinalidad, `B-local` conserva ventaja;
- cuando la lectura se queda en conectividad umbralada, la ventaja de `B` se pierde;
- cuando la lectura pasa a una partición global, la ventaja reaparece.

Ese patrón es compatible con la hipótesis arquitectónica del frente. No la clausura. La vuelve más defendible y más concreta.

## Una forma de verlo desde la arquitectura

Después de `Fase 0.6`, el sistema ya no debería imaginarse así:

```text
Harmonic Pairformer
      │
      ▼
probabilidades same-source
      │
      ▼
connected-components
      │
      ▼
fuentes
```

Esa imagen hoy ya es demasiado pobre. La forma más fiel sería:

```text
Harmonic Pairformer
      │
      ▼
geometría relacional de compatibilidades
      │
      ├── lectura local por conectividad        -> insuficiente
      └── lectura global de partición           -> parcialmente adecuada
```

La palabra clave no es “umbral”. Es **lectura**.

La red no produce simplemente scores independientes entre pares. Produce una superficie relacional que necesita una operación de lectura a su altura. `Fase 0.5` había mostrado que una lectura local no bastaba. `Fase 0.6` muestra que una lectura global ya empieza a estar a la altura de ese objeto.

## Qué queda como próximo paso real

La consecuencia de `Fase 0.6` es operativa y bastante nítida. El siguiente paso ya no debería ser más tuning de `τ`, ni otra vuelta general sobre calibradores. Ese frente quedó consumido.

Quedan dos caminos razonables:

1. **Stage B**: agregar una cabeza explícita para `k` o para la partición sobre el Pairformer congelado.
2. **Fase 1a**: pasar a CQT y picos detectados, aceptando que la lectura global todavía carga una deuda sobre `k`.

La primera ruta intenta cerrar el problema de agrupamiento antes de meter ruido de detección. La segunda pone a prueba si el sesgo de generalización sobrevive aun sin resolver del todo esa deuda.

Lo que ya no tiene mucho sentido es hablar del frente como si siguiera parado en la frontera de `Fase 0.5`. Esa frontera ya fue cruzada.

## Qué enseñó realmente Fase 0.6

El valor de `Fase 0.6` no está solo en haber mejorado un número de `ARI`. Su valor es haber cambiado el estatuto del resultado anterior.

Antes, la ventaja de `B` en `OOD-poly` podía narrarse como una promesa:

> hay algo en esa representación que parece mejor, pero todavía no sabemos convertirlo en sistema.

Después de `Fase 0.6`, la formulación más justa es otra:

> la ventaja representacional del triángulo en `OOD-poly` ya no es solo una promesa ni solo una lectura privilegiada; también es extraíble con una familia deployable concreta de clusterers globales. Lo que queda pendiente no es la existencia de esa ventaja, sino cómo cerrar la estimación de `k` para que la partición se acerque al nivel de la representación.

Esa diferencia parece pequeña, pero ordena todo lo que sigue. `Fase 0.6` no cierra Atención Armónica. Le da una base más firme para decidir si el próximo salto debe ser una cabeza de partición o la entrada al mundo de picos detectados y audio menos limpio.
