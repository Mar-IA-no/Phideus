# Explicacion Test 11 Pre-Proj A/B

**Estado**: CERRADO (`D0 + a4r`)
**Fecha de corte**: 2026-03-01
**Rol en Gate 5B**: diagnosticar si el cuello de botella generativo vive en la proyeccion compartida (`z=256`) o en el encoder mismo.

---

## Pregunta

Cuando el decoder intenta reconstruir eventos MIDI desde los embeddings compartidos de Gate 5B, el `frame F1` queda en torno a `4-5%`. La pregunta de este test es simple:

- ¿el encoder ya capturo informacion musical rica y la proyeccion la destruye?
- ¿o el encoder mismo no preserva suficiente estructura para generar algo reconocible?

---

## Metodo

Se capturan features **pre-projection** con forward hooks:

- **Audio**: `1024d` antes de `1024 -> 512 -> 256`
- **MIDI**: `512d` antes de `512 -> 512 -> 256`

Sobre esas features se entrenan event decoders identicos al baseline perceptual. La comparacion se hace contra los mismos decoders sobre `z=256` post-proj.

---

## Tabla maestra

### MIDI -> Events (intra-domain)

| Arm | z_dim | Best ep | Val CE | Tok acc | Frame F1 | Shuffle gap |
|---|---:|---:|---:|---:|---:|---:|
| D0 | 256 (baseline) | e8 | 3.110 | — | 0.054* | — |
| D0 | 512 (preproj) | e11 | 2.945 | 0.311 | 0.125 | **1.150** |
| a4r | 256 (baseline) | e1 | 3.408 | — | 0.045* | — |
| a4r | 512 (preproj) | e9 | 2.947 | 0.306 | 0.120 | **1.159** |

### Audio -> Events (cross-modal)

| Arm | z_dim | Best ep | Val CE | Tok acc | Frame F1 | Shuffle gap |
|---|---:|---:|---:|---:|---:|---:|
| D0 | 256 (baseline) | e8 | 3.118 | 0.281 | 0.045 | 0.137 |
| D0 | 1024 (preproj) | e10 | 3.070 | 0.290 | 0.050 | 0.186 |
| a4r | 256 (baseline) | e8 | 3.123 | 0.279 | 0.038 | 0.215 |
| a4r | 1024 (preproj) | e10 | 3.070 | 0.290 | 0.046 | **0.304** |

\* Los baselines `midi2events` vienen del inference sweep previo y no tenian la misma bateria de controles durante training.

---

## Hallazgo principal

### Information Retention Ratio

Formula:

`(shuffle_ce_audio - cross_ce) / (shuffle_ce_audio - intra_ce_midi)`

Esta metrica estima cuanta informacion sobre eventos que el encoder MIDI captura **sobrevive al cruce de modalidad**.

| Arm | Information retention |
|---|---:|
| D0 | 0.597 |
| a4r | **0.712** |

Lectura:

- `D0` retiene aproximadamente el `59.7%` de la informacion MIDI al cruzar modalidad.
- `a4r` retiene aproximadamente el `71.2%`.
- Eso implica **+19% relativo** de retencion cross-modal a favor de `a4r`.

Este es el resultado mas fuerte del test porque conecta los descriptores de ratio con algo mas profundo que el `S score`: la capacidad del encoder de audio para preservar estructura musical del tipo que permite regenerar eventos MIDI.

---

## Bottleneck de proyeccion

| Arm | Encoder | Pre-proj gap | Post-proj gap | % destruido |
|---|---|---:|---:|---:|
| D0 | MIDI 512→256 | 1.150 | ~0.137 | ~88% |
| D0 | Audio 1024→256 | 0.186 | 0.137 | ~26% |
| a4r | MIDI 512→256 | 1.159 | ~0.215 | ~81% |
| a4r | Audio 1024→256 | 0.304 | 0.215 | ~29% |

Lectura:

1. La proyeccion MIDI es el cuello de botella dominante.
2. El encoder MIDI ya captura informacion rica; lo que la destruye es la compresion a `256d`.
3. `a4r` pierde menos informacion en el cruce y preserva mejor la estructura musical post-proj.

---

## Asimetria cross-modal

| Decoder | D0 F1 | a4r F1 |
|---|---:|---:|
| midi->events (z=512) | 0.125 | 0.120 |
| audio->events (z=1024) | 0.050 | 0.046 |

Aunque el audio use mas dimensiones (`1024d`), el cruce de modalidad sigue perdiendo mucha estructura. El verdadero problema no es solo la dimensionalidad: es la transferencia de informacion entre ramas.

---

## Implicaciones

1. **Gate 5A C1 (conditioned projections)** ataca exactamente este hallazgo: si la proyeccion puede ser guiada por descriptores, podria preservar parte del `81-88%` que hoy se pierde.
2. **Test 13G** atacó el problema desde el otro lado y dejó una conclusión útil: reentrenar el encoder con loss reconstructiva no rescata la generación mientras la decodificación siga dependiendo de `z=256`.
3. La combinación de ambos resultados sugiere una lectura más precisa para el paper: los descriptores no solo mejoran recuperación, también mejoran la transferencia de informacion musical entre modalidades, pero esa riqueza sigue quedando recortada cuando la representación final colapsa a un vector único de `256d`.

---

## Artefactos clave

- `data/gate5b_results/D0/test11_preproj_ab.json`
- `data/gate5b_results/a4r/test11_preproj_ab.json`
- `data/gate5b_results/test11_preproj_ab_summary.json`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/resultados_compartir/06_gate5b_scientific_validation/test11_perceptual/compilacion/README.md`
- `Documents/NOTAS_CLAUDE-CODEX.md` (secciones `15` y `16`)
