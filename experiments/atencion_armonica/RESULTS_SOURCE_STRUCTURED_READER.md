# Coherencia de fuente en la lectura de particiones

Estado: contraste prospectivo ejecutado el 2026-09-08; auditorías de evidencia y alineación completas.
No hay arquitectura promovida ni decisión GO/NO-GO.

## Resultado

El factor conjunto no muestra una ventaja clara de ARI en el escenario
primario de mayor polifonía. Frente a energía de pares, su diferencia media
es −0,000059, con intervalo nominal ajustado que incluye cero. Aumenta el
error de cardinalidad y la masa de eventos en grupos de menos de tres
miembros. En los secundarios de mayor inarmonicidad y familia deformada,
la diferencia de ARI es negativa. Esto limita esta receta de inferencia;
no refuta toda geometría armónica ni demuestra que falte información temporal.

El sistema sí produce mejores particiones que el lector histórico en
polifonía, pero también lo hace el control de pares sobre el mismo pool.
No corresponde atribuir ese cambio al factor geométrico: pool, prior y
regla de lectura difieren de la referencia histórica.

## Qué se puso a prueba

Se ejecutó el [plan congelado](PLAN_SOURCE_STRUCTURED_READER.md), sin alterar
sus fuentes después de calibrar. Tres checkpoints históricos `pairs_descriptors`
reciben frecuencias observadas y los mismos descriptores; no hubo entrenamiento.
Cada escena comparte un pool de particiones entre cuatro lectores: Pares,
Fuente compartida, Fuente desacoplada y Compatibilidad local. La geometría
actúa al puntuar grupos candidatos, no modifica los logits ni el espacio
neuronal aprendido. El lector histórico conserva sus umbrales originales.

La calibración IID contiene 256 escenas nuevas; eligió γ=1 para Fuente
compartida y Compatibilidad local, y γ=0 para Fuente desacoplada. Esta última
reproduce exactamente Pares: hay un control de selección válido, pero no
un contraste de semántica del factor a intensidad no nula igualada.
El freeze y su auditoría precedieron a los cuatro tests de 256 escenas
cada uno. Las semillas de datos son 2026090780–84 y las de checkpoints
2026090721–23. Las etiquetas sólo intervienen en calibración y evaluación.

## Particiones: los cuatro tests completos

ARI medio: primero tres checkpoints dentro de cada escena, después escenas.
El oracle usa etiquetas para elegir la mejor partición del pool; no es deployable.

| Test | Pares = desacoplada | Compartida | Local | Histórico | Oracle del pool |
|---|---:|---:|---:|---:|---:|
| IID | 0,969729 | 0,968555 | 0,969032 | 0,968431 | 0,982270 |
| Mayor inarmonicidad | 0,723604 | 0,717393 | 0,718756 | 0,787045 | 0,937637 |
| Mayor polifonía — primario | 0,887156 | 0,887097 | 0,887268 | 0,844515 | 0,923224 |
| Familia deformada | 0,725094 | 0,716266 | 0,719578 | 0,738748 | 0,896449 |

Fuente: `data/atencion_armonica/structured_source_reader_v1/<split>_analysis/summary.json`.
Los cuatro nombres de split son `iid`, `ood_beta`, `ood_polyphony` y
`deformed_family`. Cada summary conserva también resultados por checkpoint.

### Contrastes de ARI

Cada diferencia es Compartida menos control. Bootstrap pareado de 2.000
remuestras de escenas; no se tratan las tres predicciones de una escena
como muestras independientes. Los tres contrastes primarios usan cobertura
nominal 98,333333% por Bonferroni; los demás intervalos son descriptivos al
95%, sin corrección conjunta. No son pruebas de equivalencia.

| Test | Control | ΔARI | Intervalo |
|---|---|---:|---|
| Polifonía | Pares / desacoplada | −0,000059 | [−0,001430; +0,001338] |
| Polifonía | Local | −0,000172 | [−0,001224; +0,000800] |
| Polifonía | Histórico | +0,042581 | [+0,035533; +0,049981] |
| IID | Pares / desacoplada | −0,001173 | [−0,002447; −0,000053] |
| IID | Local | −0,000477 | [−0,001421; +0,000329] |
| IID | Histórico | +0,000124 | [−0,002207; +0,002770] |
| Inarmonicidad | Pares / desacoplada | −0,006212 | [−0,009308; −0,003329] |
| Inarmonicidad | Local | −0,001363 | [−0,002596; −0,000320] |
| Inarmonicidad | Histórico | −0,069653 | [−0,079058; −0,060237] |
| Deformada | Pares / desacoplada | −0,008829 | [−0,012619; −0,005507] |
| Deformada | Local | −0,003313 | [−0,006317; −0,001010] |
| Deformada | Histórico | −0,022483 | [−0,029968; −0,015408] |

En polifonía, los ARI por checkpoint de Pares son 0,881163 / 0,883985 /
0,896321 y los de Compartida 0,882736 / 0,883774 / 0,894780: la diferencia
tampoco conserva signo entre las tres redes. El Brier es idéntico entre
lectores: 0,012110 / 0,094324 / 0,050617 / 0,096143 en el orden de la primera
tabla. Mejorar una partición no recalibra retrospectivamente esos logits.

### Cardinalidad y fragmentación

Diferencias Compartida menos Pares; desacoplada es idéntica a Pares.
La masa sub-3 es una fracción de eventos, no un porcentaje de grupos.

| Test | Δpartición exacta | Δerror de pares | Δk | Δerror absoluto de k | Δmasa sub-3 |
|---|---:|---:|---:|---:|---:|
| IID | −0,011719 | +0,000479 | +0,020833 | +0,020833 | +0,001797 |
| Inarmonicidad | −0,009115 | +0,002423 | +0,097656 | +0,097656 | +0,013497 |
| Polifonía | −0,006510 | −0,000221 | +0,087240 | +0,084635 | +0,004745 |
| Deformada | −0,014323 | +0,003444 | +0,152344 | +0,152344 | +0,019469 |

En polifonía, frente a Local también aumenta k (+0,029948), su error
absoluto (+0,029948) y masa sub-3 (+0,000881). Frente al histórico, el
aumento de ARI coexiste con +0,704427 grupos, +0,186198 error absoluto de k
y +0,024890 masa sub-3: activa las banderas descriptivas de ganancia con
más grupos y atribución de fragmentación no resuelta. IID también activa
ambas frente al histórico. Las banderas exigen ΔARI positivo; que no se
activen frente a Pares no elimina el aumento de fragmentación observado.
Los summaries preservan todas estas métricas, intervalos y distribuciones
de costos/RMS por tamaño, con su denominador de escenas elegibles.

Los grupos de uno o dos miembros no reciben restricción geométrica: costo
cero significa ausencia de restricción, no coherencia física. El fit tiene
distinta flexibilidad por tamaño y su witness es aproximado en grilla.
No corresponde comparar costos entre tamaños como probabilidades físicas.

## Soporte del control y límites de atribución

La rotación desacoplada altera el ranking potencial del pool en 768/768
escena×checkpoint y 256/256 escenas con soporte en las tres redes para
IID, inarmonicidad y polifonía. En familia deformada son 767/768 y 255/256:
su soporte es parcial. Se conservaron todas las escenas, sin seleccionar
las que favorecen el sham. Ese soporte potencial no cambia que γ=0 haya
desactivado su factor en el lector elegido.

La separación Pares–oracle es apreciable, especialmente en inarmonicidad
y familia deformada. Hay particiones mejores dentro del pool; eso no prueba
que puedan reconocerse sin etiquetas ni que ampliar búsqueda sea suficiente.
La compatibilidad local también utiliza relaciones físicas y el pool común
incluye un árbol analítico. Este experimento no enfrenta geometría contra
ausencia total de geometría: aísla una operación incremental concreta.

## Ejecución y recuperación

La implementación pasó 40 pruebas CPU antes del acceso a datos. El freeze
de calibración se auditó antes de autorizar tests. Se preservaron observaciones,
sidecars, features, logits completos, todos los candidatos y costos, witnesses,
selección, readouts por escena, índices bootstrap, recursos y manifests.
Calibración reprodujo exactamente 770 artefactos científicos; cada test,
771. Los replays no repitieron forwards ni aportan nuevas muestras. La
auditoría independiente reconstruyó las 1.024 observaciones de test y los
3.072 casos escena×checkpoint, incluidos costos, lectores e intervalos;
no quedaron hallazgos materiales de evidencia o interpretación. Las
inconsistencias de propagación documental se corrigieron y verificaron.

| Test | Forward de tres redes (s) | VRAM reservada máxima (MiB) | Análisis CPU (s) | Replay CPU (s) |
|---|---:|---:|---:|---:|
| IID | 2,777 | 150 | 18,051 | 18,192 |
| Inarmonicidad | 2,811 | 150 | 17,955 | 18,102 |
| Polifonía | 3,295 | 266 | 29,055 | 29,249 |
| Deformada | 2,959 | 148 | 18,742 | 18,588 |

Son tiempos de etapa que incluyen verificaciones, no benchmarks de kernels.
La preparación CPU tardó 2,018–3,855 s por test; el máximo RSS de análisis
y replay fue 312.397.824 bytes. No hubo nuevos trainings ni trabajos remotos.
Fuente de recursos: `<split>_{data,logits,analysis,replay}/resources.json`.

Identidad del conjunto recuperable, bajo
`data/atencion_armonica/structured_source_reader_v1/`:

| Artefacto | SHA-256 |
|---|---|
| `freeze_before_test.json` | `8bd2eabf07d7c2275568927552d3145608988669f55c0507156453742f48dc1e` |
| `test_authorization.json` | `c0308a8778acb12f6a83e39f5a37e3a5c3198b3705888ee7022db745931b81ec` |
| `iid_analysis/manifest.json` | `12326b4eeb6957fc07fd3323e037ee3e850954d593680e96f6be3de73e555349` |
| `ood_beta_analysis/manifest.json` | `ddd88b654b6cb82f93e1f9544daa334225abb5a14c8b642fbe03bf6adea84741` |
| `ood_polyphony_analysis/manifest.json` | `9a7908b2a334e4a46ccad0e2310b1640bb406376d1d49dff59d080829ead4228` |
| `deformed_family_analysis/manifest.json` | `fe6719aa05514cc1d5df1a91ca8176125914ba423df6c7b41657bb99893a2383` |

Los manifests enlazan transitivamente datos, checkpoints y las 36 fuentes
congeladas. El [runner](run_structured_source.py) verifica el SHA del request
y sus dependencias; rechaza sobreescritura y datos sin autorización apropiada.
Los artefactos pesados se preservan localmente, no se incluyen en Git.

## Balance geométrico y continuación

La pregunta local tiene respuesta experimental: bajo esta receta, el factor
compartido no justifica una ventaja de generalización sobre los controles.
La pregunta global sigue abierta. Se aprendió a separar ajuste de familia,
evidencia neuronal, búsqueda y decisión; no apareció una geometría armónica
aprendida ni una identificación física de las fuentes.

No hay razón en estos resultados para escalar directamente el mismo residual
a otra pérdida o entrenar fuentes latentes por inercia. La auditoría de
alineación orienta el [siguiente goal](PLAN_LEARNED_PARTITION_READER.md):
aprender la decisión sobre particiones dentro del pool, con supervisión de
partición completa y controles de pares/estructura, local, compartida y
desacoplada. El gap al oracle motiva el contraste, no promete aprendibilidad
ni ausencia de fragmentación. Su hoja inicial está auditada; requiere
protocolo ejecutable antes de datos. Los tests actuales ya están abiertos
y no pueden volver a llamarse frescos. Tiempo, fase y medición permanecen
alternativas, sin convertir este null en una necesidad informacional.
