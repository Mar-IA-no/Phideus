# Voz Expresiva Phideus

> Frente exploratorio de Phideus que prueba si el patrón **descriptor ratio-based + mecanismo de inyección** (validado en Escalón 1 sobre música) transfiere al territorio de la expresión vocal y los correlatos paralingüísticos/afectivos del habla.

## Estado actual: Fase 1 EN cerrada, ZH full LOSO ya corrido — pendiente cierre analítico cross-language (2026-06-27)

### Fase 1 — cierre direccional actual

> Sobre `ESD` English, `WavLM` frozen ya levantó con claridad el techo que el stack descriptor-only no había podido romper en `N-strict`, y la inyección de la familia `A` ya dejó un primer caso positivo defendible: **`concat` mejora a `WavLM-only` en speaker-independent estricto**. La lectura útil del corte no es “voz resuelta”, sino algo más acotado y más importante para Phideus: la mecánica descriptor-guided ya mostró transferencia funcional a un régimen SSL homogéneo dentro del dominio vocal.

Lectura desagregada:

| Condición | Resultado |
|---|---|
| **WavLM-only, N-strict** | `UAR=0.698 ± 0.099`. El encoder frozen por sí solo ya supera con holgura el piso de chance y reordena la lectura del techo que 0B había dejado abierto. |
| **concat vs WavLM-only, N-strict** | **`+0.039` UAR**, `CI95=[+0.019,+0.060]`, `P(Δ>0)=1.00`. Primer contraste positivo robusto del frente en generalización honesta. |
| **FiLM / xattn vs WavLM-only, N-strict** | Tendencia positiva (`+0.016`, `+0.023`) pero sin cierre robusto todavía. |
| **N-adapt** (secundaria, 1 calib repeat) | Los tres mecanismos pasan con mejora robusta y uniforme: `+0.041` a `+0.044` UAR sobre baseline. |
| **CKA** | `FiLM` mejora funcionalmente con CKA alto (~`0.85`), mientras `concat` y `xattn` mejoran reorganizando fuerte (CKA ~`0.23`). |

La lectura calibrada del cierre queda así:

- `Fase 1` **sí valida transferencia positiva del patrón Phideus a voz expresiva en un régimen SSL homogéneo**, pero no de manera simétrica para todos los mecanismos.
- **`concat`** es, por ahora, el mecanismo más fuerte del frente bajo `N-strict`.
- `FiLM` y `xattn` no quedan descartados: aportan bajo `N-adapt` y dejan una disociación geométrica útil para análisis posterior.
- Esto **no** equivale todavía a estabilidad translingüística ni a generalización naturalística.

### ZH ya ejecutado, pero el frente no está formalmente cerrado

Después de ese cierre en inglés, la réplica `ZH` sobre el mismo corpus `ESD` ya fue corrida completa: `240/240` runs del `LOSO` full terminaron con el manifest corregido (`fix B2`) y quedaron persistidos en `data/voz_expresiva/1_zh/`.

Eso, sin embargo, **todavía no autoriza** un cierre translingüístico del frente. Antes de consolidar la lectura `EN ↔ ZH` faltan dos pasos metodológicamente necesarios:

1. rehacer el brazo `EN N-adapt` con el fix `B2` en `1_en_calibfix/`, preservando `N-strict` heredado de `1/`;
2. correr los reportes intra-EN, intra-ZH y cross-language sobre esos artefactos ya alineados.

Además, el antecedente `0A ZH` dejó un caveat que vuelve todavía más importante esa lectura prudente: la especificidad ratio pooled se invirtió en mandarín (`A/C=0.69`) respecto de inglés (`2.88`). Eso no invalida `Fase 1 ZH`, pero sí impide tratar el training terminado como si ya fuese un claim cross-language cerrado.

### Próximo paso ya decidido

El siguiente corte del frente ya no es “correr ZH”: eso ya pasó. Tampoco es saltar directo a `MSP-Podcast`. La decisión operativa vigente es más disciplinada:

1. **Completar `1_en_calibfix/`** para que el contraste `N-adapt` de inglés use el mismo fix de manifest que ya usó `ZH`.
2. **Consolidar reportes intra-idioma y cross-language** sobre `1_en_calibfix/` y `1_zh/`, no sobre snapshots heterogéneos.
3. Recién después decidir si la lectura translingüística mínima se sostiene y si conviene abrir profundización `1.2`, pasar a `Fase 3` naturalística o priorizar el Carril B.

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
- **Spike de compatibilidad previo a Fase 1**: `../../../experiments/voz_expresiva/SPIKE_FASE_1_0.md`
- **Reporte empírico Fase 1**: `../../../data/voz_expresiva/1/REPORTE_1.md`
- **Investigación bibliográfica de junio 2026**: referencia editorial externa al repo Phideus, en `editorial-altermundi/Biblioteca/analisis-carga-emocional-del-habla/`. Cubre 7 reportes de subagentes (productos, OSS, datasets, métodos, entrenamiento/licencias, substrato físico EGG/cross-modal, disentanglement/voice conversion/expressive TTS), un NARRATED_REPORT integrador (~60 KB), bibliografía deduplicada y un CROSS_REPORT denso.
- **Antecedente exploratorio**: `../EIR-EMR/` (preservado, no activo).

## Siguiente corte

La prioridad ya no es “ver si Fase 1 funciona” en abstracto. Eso ya ocurrió en `ESD` English. La prioridad pasa a ser otra:

1. ¿Qué lectura intra-`ZH` emerge una vez consolidado su reporte con el manifest corregido y el mismo protocolo de conteo/completitud?
2. ¿La comparación `EN ↔ ZH` sigue sosteniendo una lectura mínima de transferencia cuando `EN N-adapt` ya fue recalculado con `fix B2`?
3. ¿`concat` conserva su ventaja relativa y la familia `A` sigue aportando sobre `WavLM-only` cuando el resultado se mira ya sin asimetrías de recipe?

### Spike Fase 1.0 (cerrado 2026-06-22)

Documento: `experiments/voz_expresiva/SPIKE_FASE_1_0.md`.

Hallazgo: los mecanismos heredados de E2 (`SpeechEGGEncoderAug`, `SpeechEGGEncoderXAttn`, `ConditionedProjectionHead`) **NO eran drop-in para WavLM**. Asumían una topología distinta y por eso el cierre correcto fue una **reimplementación compatible inspirada en E2**, toda frame-level post-WavLM, pre-pool.

- `FiLM`: reformulado frame-level con la misma lógica de `zero-init`, ya no como módulo utterance-level externo.
- `concat`: reimplementado con `near-identity init`.
- `xattn`: reimplementado con residual casi nulo al inicio.
- Resultado: los tres mecanismos quedaron comparables entre sí dentro de la misma plantilla arquitectural, que es justamente la condición que hizo interpretable el cierre de `Fase 1`.
