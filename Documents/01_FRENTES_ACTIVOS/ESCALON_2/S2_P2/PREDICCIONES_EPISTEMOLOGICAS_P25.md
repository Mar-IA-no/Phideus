# Predicciones Epistemologicas — S2-P2.5

> **Preregistro interpretativo**. Creado 2026-03-10, ANTES de que H-series-xattn y A4-16k-xattn produzcan resultados.
> V4-lin-attnbias esta corriendo (ep24/30 al momento de redaccion); H-series-xattn y A4-16k-xattn aun no corrieron.
>
> Este documento **manda sobre como se leen los resultados de S2-P2.5**.
> Para contexto general y taxonomia de familias: `plan_rectificacion_armonia_natural.md`.
> Para estado canonico del frente: `../README.md`.

---

## 1. Regla operativa de comparacion

El protocolo canonico de Escalon 2 usa `S = min(S2E, E2S) @ k=10` con CI grouped bootstrap por hablante.

Para que la matriz de predicciones sea falsificable, se fija la siguiente regla:

> **Delta = S_A - S_B** se computa con **grouped bootstrap pareado** sobre la diferencia: en cada iteracion bootstrap se computa S para ambos modelos sobre el mismo resample de hablantes, y se registra la diferencia. Esto produce un CI directamente sobre Delta.
>
> **A > B** se declara cuando: (1) Delta_point >= 2pp, Y (2) CI_Delta excluye 0. Ambas condiciones deben cumplirse.
>
> **A ≈ B** (indistinguibles bajo esta configuracion) se declara cuando CI_Delta contiene 0, o Delta_point < 2pp.

### Por que bootstrap pareado y no comparacion de CIs individuales

Comparar CIs individuales de A y B es conservador y pierde poder estadistico: dos CIs pueden solaparse y aun asi la diferencia ser significativa. El bootstrap pareado captura la correlacion entre modelos evaluados sobre los mismos datos (mismos hablantes, mismos pools) y produce un CI mas estrecho y mas honesto directamente sobre la magnitud de interes (Delta).

### Implementacion

`eval_escalon2.py` expone `paired_grouped_bootstrap_ci_delta(per_query_A, per_query_B)` que:
1. Agrupa queries por `speaker_id` en ambos conjuntos
2. En cada iteracion (N=1000), resamplea hablantes con reemplazo
3. Computa S_A y S_B sobre el mismo resample
4. Registra Delta = S_A - S_B
5. Retorna Delta_point, CI_Delta_lo, CI_Delta_hi

---

## 2. Taxonomia de familias (resumen operativo)

| Familia | Descriptor P2.5 | Mecanismo | Que testea |
|---------|-----------------|-----------|------------|
| **A** (dinamica temporal) | V4-lin-attnbias | Attention bias en self-attention | Invariantes del oscilador (ratios lineales F0) |
| **B** (armonica intra-frame) | H-series-xattn | Cross-attention post-CNN | **Tesis fuerte de HIT**: serie armonica fisica |
| **C** (control no-ratio) | A4-16k-xattn | Cross-attention post-CNN | Hipotesis nula: informacion espectral generica |
| **ref** | D0 (sin descriptor) | — | Baseline neural sin inyeccion |

**H-series (Familia B) es el test mas directamente alineado con la tesis fuerte de HIT** en Escalon 2. Sus resultados tienen mas peso epistemologico que los de V4-lin para la pregunta central del programa.

**V4-lin (Familia A)** testea una tesis adyacente y valiosa pero distinta: si la dinamica temporal del oscilador contiene invariantes cross-modales privilegiados. Un resultado positivo de V4-lin dice algo sobre dinamica del oscilador, NO sobre la serie armonica.

**A4-16k (Familia C)** es UN control no-ratio especifico, no "todos los posibles controles genericos." Si H-series > A4-16k, la conclusion es "la estructura armonica supera este control espectral particular."

---

## 3. Matriz de predicciones

Los siguientes patrones se pre-registran como anclas interpretativas. Las comparaciones usan la regla operativa de la seccion 1.

| # | Patron de resultados | Interpretacion epistemologica |
|---|---------------------|------------------------------|
| **P1** | H-series-xattn > D0 > A4-16k-xattn | **Evidencia fuerte para HIT**: la estructura armonica es especificamente privilegiada |
| **P2** | H-series-xattn > A4-16k-xattn > D0 | Evidencia para HIT, pero el mecanismo atencional tambien ayuda genericamente |
| **P3** | A4-16k-xattn >= H-series-xattn > D0 | El mecanismo atencional ayuda, pero la estructura armonica NO es privilegiada |
| **P4** | D0 >= todos | Ni el mecanismo ni los descriptores ayudan en esta configuracion |
| **P5** | V4-lin-attnbias > D0, H-series-xattn ≈ D0 | La dinamica del oscilador es util, pero la serie armonica no (bajo esta configuracion) |
| **P6** | H-series-xattn > D0, V4-lin-attnbias ≈ D0 | La serie armonica funciona pero la dinamica temporal no (inesperado pero informativo) |

**Nota sobre "≈ D0"**: Incluye tanto empate como derrota — lo relevante es la ausencia de mejora significativa bajo la regla operativa.

### Alcance de la matriz

Esta es una **matriz minima de patrones ancla**, no exhaustiva. Cubre los patrones epistemologicamente mas informativos. Combinaciones reales pueden no encajar limpiamente en una sola celda (e.g., H-series > D0 en S2E pero no en E2S). En esos casos, se reporta el resultado observado y se indica que celdas son parcialmente compatibles, sin forzar una interpretacion.

La matriz previene racionalizacion post-hoc para los patrones principales; no pretende cubrir toda la superficie de resultados posibles.

---

## 4. Guardrails para interpretar nulls

Un null de H-series (H-series-xattn ≈ D0 o H-series-xattn < D0) **no falsifica automaticamente HIT**. Antes de atribuir un null a la hipotesis sustantiva, se deben verificar las siguientes condiciones:

### 4.1 Training sano

- Convergencia sin colapso: loss estable, no degeneracion de covarianza
- VICReg variance term no colapsado (std > 1.0 en todas las dimensiones)
- Sin divergencia ni NaN

### 4.2 Uso real del mecanismo

- `xattn_scale` no degenerado (no colapsado a 0 ni saturado)
- Contribucion de la rama cross-attention no trivial: medir norma relativa del residuo xattn vs features crudas en eval
- Si la rama xattn contribuye < 1% de la norma total, el mecanismo nunca se engancho y el null no es informativo sobre el descriptor

### 4.3 Sensibilidad al descriptor

Si es factible, ablacion rapida que substituya H-series por ruido gaussiano de misma estadistica:
- Si el resultado no cambia → el mecanismo nunca se engancho al contenido del descriptor
- Si el resultado empeora → el mecanismo si esta usando el descriptor (fortalece el null como informativo)

### 4.4 Regla de lectura

Solo si (4.1), (4.2) y (4.3) se cumplen, un null de H-series es informativo sobre el descriptor y justifica considerar confounds arquitectonicos (e.g., simetria de encoders → Fase 2).

Sin estas verificaciones, un null puede ser simplemente "el mecanismo no se engancho" y no tiene peso epistemologico sobre HIT.

---

## 5. Condiciones sobre A4-16k-xattn

Para que la inferencia comparativa A4-16k vs H-series sea valida:

- A4-16k-xattn debe correr a **30 epochs** (comparable a V4-lin-attnbias y H-series-xattn)
- Un run corto (10ep) solo serviria como filtro de colapso temprano
- Toda inferencia comparativa basada en un arm de < 30ep queda marcada **explicitamente como provisional** y no se usa para declarar A > B ni A ≈ B bajo la regla de la seccion 1

---

## 6. Asunciones explicitas del marco P2.5

Estas asunciones estan vigentes y deben listarse como confounds potenciales si los resultados son ambiguos:

| Asuncion | Status | Impacto si falla |
|---------|--------|-----------------|
| 10ms hop suficiente para dinamica de ratios F0 | Asuncion no validada | V4-lin podria perder dinamica sub-ciclo |
| PYIN (speech) y autocorrelacion (EGG) son estimadores comparables | Confound documentado | V4-lin compara sensores + estimadores |
| F0 per-modalidad previene leakage cross-modal | Documentado, correcto | — |
| Pool=128, k=10 suficientes para resolver diferencias | Decision de protocolo congelada | Fija resolucion estadistica |
| Encoders simetricos adecuados para ambas modalidades | **Asuncion de Fase 1** | Speech y EGG son fisicamente distintos; simetria puede limitar H-series en EGG |
| H-series captura cantidades fisicas distintas en speech vs EGG | Confound reconocido | Si H-series mejora, demuestra que el componente compartido (fuente glotal) es suficiente pese al tracto vocal |
| n_fft=2048 con busqueda ±2 bins suficiente para extraccion armonica | Decision de ingenieria | Podria perder precision en F0 bajos |
| Normalizacion per-modalidad de H-series congela estadisticas correctamente | Implementado | — |

---

## 7. Consecuencias interpretativas de la asimetria Speech/EGG para H-series

H2/H1 en speech = fuente glotal + filtro del tracto vocal.
H2/H1 en EGG = solo fuente glotal.

Son la MISMA computacion pero miden cantidades fisicas DIFERENTES.

**Si H-series mejora la alineacion cross-modal**, esto significaria que el componente compartido (la fuente glotal) es suficiente para organizar la alineacion **a pesar del confound del tracto vocal** en speech. Esto es un resultado MAS fuerte que simple descriptor matching — demuestra que la estructura armonica del oscilador es un invariante cross-modal robusto al filtrado del tracto vocal.

**Si H-series no mejora**, la asimetria speech/EGG es uno de los confounds a considerar antes de interpretar el null como evidencia contra HIT. Encoders asimetricos (Fase 2) podrian manejar mejor esta diferencia.

---

## 8. Fases arquitectonicas condicionales

### Fase 1 (actual): Encoders simples y simetricos

CNN+Transformer, d=512, from scratch, identicos para ambas modalidades. Epistemologicamente correcto: maximiza interpretabilidad, el descriptor es la unica variable.

### Fase 2 (condicional a senal en Fase 1): Asimetria controlada

**Trigger**: Resultados ambiguos (H-series mejora pero dentro del CI de D0) O null limpio de H-series bajo training sano + uso real del mecanismo + sensibilidad al descriptor verificada.

Posibles intervenciones:
- CNNs asimetricas (speech: mas resolucion alta para formantes; EGG: mejor resolucion baja para pulso glotal)
- Puntos de inyeccion diferentes por modalidad
- Ablacion: H-series solo en speech encoder vs solo en EGG encoder vs ambos

### Fase 3 (condicional a Fase 2): Benchmark con encoder fuerte

WavLM/HuBERT frozen como speech encoder, encoder trainable pequeno para EGG. Testea si el efecto descriptor persiste con representaciones de nivel foundation model.

**Principio rector**: Los cambios de arquitectura son **eventos epistemologicos**, no optimizaciones de ingenieria. Cada fase testea preguntas diferentes.

---

## 9. Fecha y estado del preregistro

- **Fecha de creacion**: 2026-03-10
- **H-series-xattn**: No ha corrido aun
- **A4-16k-xattn**: No ha corrido aun. Si entra en inferencia comparativa fuerte, corre a `30ep` comparables. Un corte corto (`10ep`) solo se admite como smoke provisional y no entra en la matriz A>B / A≈B.
- **V4-lin-attnbias**: En ejecucion (ep24/30), resultados no interpretados
- **D0**: Completo (S=77.8% @ ep25)

El preregistro se cierra al momento de creacion de este archivo. Cualquier modificacion posterior a la tabla de predicciones debe documentarse con fecha y justificacion explicita.
