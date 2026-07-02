# Voz Expresiva Phideus

> Frente exploratorio de Phideus que prueba si el patrón **descriptor ratio-based + mecanismo de inyección** (validado en Escalón 1 sobre música) transfiere al territorio de la expresión vocal y los correlatos paralingüísticos/afectivos del habla.

## Estado actual: cierre cross-language EN↔ZH ya consolidado, con positivo acotado a N-adapt y null/negativo en N-strict (2026-07-02)

### Fase 1 — cierre translingüístico actual

> El frente ya no está en el punto “English positivo, Chinese pendiente”. Ese corte quedó atrás. Hoy la lectura útil es más específica: el patrón descriptor-guided **sí** muestra transferencia `EN ↔ ZH` cuando el régimen admite anclaje per-speaker en test (`N-adapt`), pero **no** sostiene una ventaja robusta y simétrica bajo speaker-independence estricto (`N-strict`). La consecuencia importante no es “voz resuelta”, sino una delimitación de alcance: el mecanismo transfiere, pero no bajo cualquier régimen de generalización.

Lectura desagregada del cierre:

| Condición | Resultado |
|---|---|
| **EN, N-strict** | `WavLM-only` ya había levantado el techo del stack clásico, y `concat` había dejado el caso positivo robusto del frente (`+0.039` UAR). |
| **ZH, N-strict** | Esa lectura no replica limpiamente: `concat` queda cerca de nulo y `FiLM` / `xattn` pasan a deltas negativos sobre baseline. |
| **Cross-language, N-strict** | El lift inglés **no** transfiere de forma robusta. La lectura canónica del régimen primario es `null/negativo`, no “replicación parcial”. |
| **EN + ZH, N-adapt** | `concat` y `FiLM` replican limpio entre lenguas; `xattn` mejora en ambos idiomas pero más débil en `ZH`. |
| **Lectura CKA** | Se mantiene la disociación útil: `FiLM` puede mejorar sin reorganizar tanto, mientras `concat` y `xattn` mueven más fuerte la geometría. |

La lectura calibrada del cierre queda así:

- El frente **sí** valida transferencia cross-language del patrón Phideus, pero **solo** en el régimen `N-adapt`.
- **`concat`** y **`FiLM`** replican limpiamente entre `EN` y `ZH` cuando existe calibración per-speaker en test.
- Bajo `N-strict`, la historia es otra: el positive de `EN` no se sostiene como lectura general del frente, y `film/xattn` se vuelven directamente negativos en `ZH`.
- Esto **no** equivale todavía a una validación naturalística ni a una estabilidad speaker-independent amplia.

### Qué cerró realmente el contraste EN↔ZH

La llegada de `EN N-adapt` limpio desde UNC resolvió la única deuda que quedaba para leer el frente sin recipes mezcladas. Eso permitió cerrar el contraste `EN ↔ ZH` con el mismo protocolo y separar dos planos que antes estaban mezclados:

- el plano **cross-language con anclaje per-speaker** (`N-adapt`), donde el patrón descriptor-guided sí replica;
- y el plano **speaker-independent estricto** (`N-strict`), donde esa réplica no aparece.

La formulación honesta del resultado ya no es “ZH corrió bien” ni “la réplica fue ambigua”, sino otra:

- el descriptor armónico **sí** transfiere entre lenguas cuando el régimen deja usar calibración mínima por hablante en test;
- pero ese mismo patrón **no** sostiene una ventaja robusta en el régimen estricto, que es el primario del frente.

El caveat de `0A ZH` sigue importando y no debe esconderse: a nivel univariado pooled, la especificidad ratio se invierte en mandarín (`A/C=0.69` vs `2.88` en `EN`). El punto importante es que ese caveat ya no bloquea el cierre: queda absorbido como parte de la interpretación del régimen `N-strict`, no como deuda metodológica abierta.

### Próximo paso ya decidido

La decisión ya no es “cerrar cross-language”: eso ya ocurrió. La decisión abierta pasa a ser estratégica:

1. **Cerrar Fase 1 con este matiz**: positivo real pero acotado a `N-adapt`, sin sobreleer `N-strict`.
2. **Abrir una Fase 1.2** si se quiere aislar mejor por qué el speaker-independent estricto no replica entre lenguas.
3. **Saltar a `MSP-Podcast` / Fase 3** si se prioriza mover el frente al dominio naturalístico en vez de seguir refinando `ESD`.

### Fase 0B — formulación calibrada del cierre

> Fase 0B **no valida todavía la hipótesis fuerte en speaker-independent estricto**. Sí valida, bajo adaptación mínima por hablante (N-adapt, 25 utts label-agnostic), que la familia A tiene **señal específica no reducible a un control espectral genérico**, y muestra una **mejora pequeña sobre eGeMAPS** que merece ser testeada en Fase 1 con WavLM.

Lectura desagregada:

| Condición | Resultado |
|---|---|
| **N-strict** (sin per-speaker en test) | Nadie supera chance. La hipótesis fuerte NO queda validada en generalización honesta a hablante nuevo con descriptors clásicos. |
| **N-adapt** — especificidad ratio | **Robusta**. A-only > C-only (+0.161-0.173), A+D > C+D (+0.112-0.114), C+D < D-only (-0.103). |
| **N-adapt** — A+D > D-only | Pequeña y clf-dependiente. +0.009 LogReg (marginal), +0.013 SVM RBF (Δ>0 robusto). |
| **N-adapt** — A+B > A-only | +0.047-0.085 robusto. B complementaria. |

Hallazgo metodológico aparte: 25 utts label-agnostic de calibración por hablante sacan a D-only y A+D de chance hasta ~0.6 UAR. Es hallazgo con potencial aplicado, no lectura directa de producto.

### Qué quedó establecido al salir de Fase 0B

- `0B` no cerró la transferencia; sí cerró la pregunta más barata que había que cerrar antes de meter `SSL`.
- El paso a `WavLM` dejó de ser ornamental y pasó a ser el test correcto del frente.
- La utilidad principal de `0B` hoy es haber disciplinado la pregunta que `Fase 1` efectivamente resolvió en `ESD` English.

---

### Fase 0A — cerrada con GO direccional (referencia)

**Resultado Fase 0A**: los descriptores ratio-based muestran señal univariada significativa sobre ESD English (17,500 utterances).

| Familia | Top dim | eta² | Lectura |
|---|---|---|---|
| D — eGeMAPS baseline | F0semitone_percentile80 | **0.589** | el F0 lidera |
| **A — Phideus-ratio** | Hseries_d5_mean (concentración armónica) | **0.385** | señal específica ratio-based, 5× sobre el control |
| B — Voice quality | alpha_ratio | 0.262 | señal moderada |
| **C — control no-ratio** | A416k_d2_max | 0.076 | control cumplió rol (5× debajo de A) |

Lectura operativa:
- El control (Familia C) se quedó muy por debajo de A y B → la señal de A NO se reduce a "información espectral genérica".
- Phideus-ratio (Familia A, `Hseries_d5`) discrimina 5× sobre el control sin entrenar nada.
- eGeMAPS lidera por F0 — la pregunta "¿A aporta sobre D?" queda abierta para Fase 0B (clasificador real).

Decisión del usuario: **GO direccional** para Fase 0B.

### Caveats declarados (por trazabilidad)

- **Silhouette ~ 0 en todas las familias**: las distribuciones de emoción se superponen mucho geométricamente, pero hay señal univariada clara. Es típico de SER actuado, no es problema de descriptores.
- **CPP en escala comprimida**: la implementación manual da ~10× menos que los valores clínicos estándar. Discrimina pero no es directamente comparable a la literatura. Fix posible en 0B.
- **H1-H2 / H1-A3 son proxies**: sin corrección formántica. Capturan parte de la señal del modo de fonación; implementación clínica completa va si Fase 0B la requiere.
- **Normalización transductiva**: z-score por hablante sobre todo el corpus EN. Fase 0B debe redefinir splits y re-normalizar dentro del train.
- **ESD es actuado**: cualquier resultado acá necesita pasar por Fase 3 (MSP-Podcast naturalístico) antes de generalizarse.

### Artefactos de Fase 0A

- `data/esd/descriptors_0A_en.npz` — 15 MB, 17,500 vectores × 4 familias.
- `data/visualizations/voz_expresiva/0A/` — 17 plots + 3 JSONs + REPORTE_0A.md.
- `src/voz_expresiva/` — código del pipeline.
- `experiments/voz_expresiva/` — scripts de extract + analyze.

### Pendiente al cierre de Fase 0A (histórico — ya superado por Fase 0B)

Cumplido. Fase 0B ejecutada y cerrada el mismo 2026-06-22; resultados arriba.

## Estatus de `EIR-EMR/`

`EIR-EMR/` queda preservado como **antecedente exploratorio, no como nombre vigente ni como roadmap activo**. Su contenido fue reformulado en este frente tras la investigación bibliográfica de junio 2026. La nomenclatura "EIR / EMR" como marca maestra del frente fue descartada por ser un lenguaje preconcebido antes de tener evidencia. Para trazabilidad histórica ver `../EIR-EMR/`.

## Documentación clave

- **Roadmap general del frente**: `./ROADMAP_VOZ_EXPRESIVA_PHIDEUS.md`
- **Plan archivado de la réplica ZH**: `./PLAN_FASE_1_ZH.md`
- **Explicación pedagógica del pipeline Fase 1**: `./EXPLICACION_PIPELINE_FASE_1.md`
- **Explicación de mecanismos de inyección**: `./mecanismos_inyeccion_explicacion.md`
- **Reporte cross-language definitivo**: `../../../data/voz_expresiva/REPORTE_CROSS_LANGUAGE_EN_ZH.md`
- **Reporte ZH consolidado**: `../../../data/voz_expresiva/1_zh/REPORTE_1_ZH.md`
- **Spike de compatibilidad previo a Fase 1**: `../../../experiments/voz_expresiva/SPIKE_FASE_1_0.md`
- **Reporte empírico Fase 1**: `../../../data/voz_expresiva/1/REPORTE_1.md`
- **Investigación bibliográfica de junio 2026**: referencia editorial externa al repo Phideus, en `editorial-altermundi/Biblioteca/analisis-carga-emocional-del-habla/`. Cubre 7 reportes de subagentes (productos, OSS, datasets, métodos, entrenamiento/licencias, substrato físico EGG/cross-modal, disentanglement/voice conversion/expressive TTS), un NARRATED_REPORT integrador (~60 KB), bibliografía deduplicada y un CROSS_REPORT denso.
- **Antecedente exploratorio**: `../EIR-EMR/` (preservado, no activo).

## Siguiente corte

La prioridad ya no es “cerrar EN↔ZH”. Eso ya pasó. La prioridad pasa a ser otra:

1. ¿Conviene cerrar `Fase 1` con esta lectura acotada o abrir una `Fase 1.2` que ataque específicamente el cuello de `N-strict`?
2. ¿El siguiente experimento útil es todavía dentro de `ESD`, o el frente ya ganó más pasando a `MSP-Podcast`?
3. ¿La disociación `N-adapt` positivo / `N-strict` null-negativo es un límite del descriptor, del régimen de normalización o del corpus actuado?

### Spike Fase 1.0 (cerrado 2026-06-22)

Documento: `experiments/voz_expresiva/SPIKE_FASE_1_0.md`.

Hallazgo: los mecanismos heredados de E2 (`SpeechEGGEncoderAug`, `SpeechEGGEncoderXAttn`, `ConditionedProjectionHead`) **NO eran drop-in para WavLM**. Asumían una topología distinta y por eso el cierre correcto fue una **reimplementación compatible inspirada en E2**, toda frame-level post-WavLM, pre-pool.

- `FiLM`: reformulado frame-level con la misma lógica de `zero-init`, ya no como módulo utterance-level externo.
- `concat`: reimplementado con `near-identity init`.
- `xattn`: reimplementado con residual casi nulo al inicio.
- Resultado: los tres mecanismos quedaron comparables entre sí dentro de la misma plantilla arquitectural, que es justamente la condición que hizo interpretable el cierre de `Fase 1`.
