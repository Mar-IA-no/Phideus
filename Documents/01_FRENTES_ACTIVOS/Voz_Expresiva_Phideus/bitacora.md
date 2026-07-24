# Bitácora — Voz Expresiva Phideus

> **Convención de fechas.** El encabezado de cada entrada es **fecha de registro** (el commit que la
> incorporó), no necesariamente la fecha en que ocurrió el trabajo que describe. Cuando difieren, la
> ocurrencia se declara explícitamente dentro de la entrada, anclada a NOTAS/commit. Ésta es la bitácora
> viva y canónica del frente; la copia congelada en ProsodIA
> (`Documentos/phideus_voz_expresiva/SNAPSHOT_fases_voz_expresiva_2026-06-27.md`) no se actualiza.
>
> **Campos por entrada.**
> - `id` — identificador estable. Hay más de una entrada por fecha, así que **la fecha sola no
>   desambigua**: los punteros usan `id`, no fecha.
> - `tipo` — `retrospectiva` | `cierre` | `recurso`. Una `retrospectiva` resume trabajo previo: es
>   material de banco, no novedad.
> - `supersede` / `supersedida_por` — relación **bidireccional** entre entradas, declarada de los dos
>   lados. Puede ser **parcial**, y entonces lleva su alcance (qué sección acota). Un bloque supersedido
>   **no debe ser ancla única** de nada: hay que leer también el que lo acota.
> - `Backfill` — entrada escrita después de su fecha de registro, reconstruida de artefactos con fecha
>   propia (NOTAS, commits, informes). Se declara siempre; no se disfraza de entrada contemporánea.

---

## 2026-06-27 — Retrospectiva del frente: qué salió bien (fases 0A, 0B y 1)

**id:** `2026-06-27-retrospectiva`
**tipo:** `retrospectiva` — resume varias semanas de trabajo, no es una entrada del día.
**supersedida_por:** `2026-07-02-cierre-fase1-crosslang` — **parcial**, alcance `### Fase 1`
(las secciones 0A y 0B siguen vigentes)
**Registro:** commit `ebb82df` (2026-06-27). **Ocurrencia:** 2026-06-21 → 2026-06-23 (por fase, abajo).

> ⚠️ **Parcialmente supersedida.** La lectura de Fase 1 que sigue es el estado de conocimiento al
> 2026-06-27, anterior al cierre cross-language EN↔ZH del 2026-07-02. Ver la **nota de supersesión** al
> final de la sección Fase 1 antes de citar cualquier número o claim de esa sección.

### Fase 0A — viabilidad del descriptor

*Ocurrencia: 2026-06-21 → 2026-06-22 (NOTAS S59).*

**Pregunta:** ¿Familia A (V4-lin + H-series) discrimina emociones más que un control no-armónico?
**Método:** eta² robusto sobre 17 500 utts EN, ANOVA por dimensión, control = A4-16k bandas espectrales.

| Descriptor | eta² | Lectura |
|---|---|---|
| H-series | **0.385** | señal fuerte |
| control C | 0.076 | señal débil |
| **ratio** | **5×** | el descriptor armónico domina sobre el control |

**Conclusión:** el descriptor captura varianza específica del fenómeno emocional; no es ruido común a
cualquier descriptor espectral.

### Fase 0B — descriptor solo con clasificadores clásicos

*Ocurrencia: 2026-06-22 (NOTAS S59 continuación, commit `f8a9c06`).*

**Pregunta:** ¿el descriptor solo (sin SSL) clasifica emoción con LogReg / SVM bajo LOSO?
**Método:** Familia A pooled 48d sobre LOSO 10-fold, dos normas (N-strict / N-adapt 3 repeats).

**Salió bien — bajo N-adapt:**

| Contraste | Lectura |
|---|---|
| A-only > C-only | especificidad ratio robusta |
| A+D > C+D | Familia A aporta sobre eGeMAPSv02 |
| C+D < D-only | el control no agrega |
| A+D > D-only (clf-dep) | mejora pequeña pero consistente |

**Conclusión:** con calibración per-speaker, el descriptor tiene firma propia y no es redundante con
eGeMAPSv02 (88 functionals estándar).

**No salió:** bajo **N-strict** ≈ chance — el descriptor solo no escapa speaker-independent estricto.
Eso motivó la Fase 1, con SSL como techo más alto.

### Fase 1 — descriptor inyectado en WavLM frozen

**supersedida_por:** `2026-07-02-cierre-fase1-crosslang`

*Ocurrencia: 2026-06-22 → 2026-06-23 (NOTAS S60 spike/plan/implementación + S60 continuación, ejecución).*

**Pregunta:** ¿WavLM levanta el techo de 0B, y Phideus aporta sobre WavLM solo?
**Método:** LOSO 10-fold × 4 configs × 2 norm × 3 seeds = 240 runs.

**Baseline WavLM-only escapa chance fuerte:** UAR N-strict = **0.698** (chance = 0.20) → SSL resuelve
buena parte del problema.

**Concat pasa formalmente el contraste primario en N-strict:**
Δ = **+0.039**, CI95 [+0.019, +0.060], P(Δ>0) = 1.00. Encaja con el primer escenario prefigurado del plan.

**N-adapt: los tres mecanismos pasan robustos:**

| Mecanismo | Δ | CI95 | P(Δ>0) |
|---|---|---|---|
| concat | +4.4pp | [+0.022, +0.063] | 1.00 |
| film | +4.1pp | [+0.022, +0.061] | 1.00 |
| xattn | +4.4pp | [+0.028, +0.063] | 1.00 |

→ la calibración per-speaker estabiliza la señal uniformemente.

**Disociación CKA reveladora** (no buscada, emergió de los datos):

| Mecanismo | CKA | Lectura |
|---|---|---|
| concat / xattn | ~0.23 | reorganizan geometría |
| film | ~0.85 | modula sin reorganizar |

→ FiLM logra el efecto funcional manteniendo la representación geométricamente cercana al baseline.
Hallazgo interpretativo de primera línea para el libro.

> #### ⚠️ Nota de supersesión (agregada 2026-07-24; refiere al cierre del 2026-07-02)
>
> Esta sección es el estado al 2026-06-27 y quedó acotada en dos puntos por el cierre posterior del
> frente. **No citar sus claims de Fase 1 sin este contexto.**
>
> 1. **El claim de N-strict quedó acotado.** Lo de arriba ("concat pasa el contraste primario en
>    N-strict", leído entonces como primera evidencia de transferencia) es **solo EN**. El cierre
>    cross-language del 2026-07-02 mostró que **ese lift NO transfiere a ZH**: concat cae a
>    no-significativo, y film/xattn se vuelven **negativos**. El positivo del frente quedó **acotado al
>    régimen N-adapt** — no es GO limpio ni NULL limpio.
> 2. **Los números N-adapt de arriba son pre-fix-B2.** Se calcularon con el bug del `calib_manifest`
>    (mismas 25 utts de calibración para los 10 speakers). Valores limpios: concat **+0.042**,
>    film **+0.036**, xattn **+0.041**. El bug no cambió las conclusiones: el lift era real, no artefacto.
>
> Fuentes: `data/voz_expresiva/REPORTE_CROSS_LANGUAGE_EN_ZH.md` (informe integrado definitivo) y
> `Documents/NOTAS_CLAUDE-CODEX.md` §S67.

### Lo metodológico que salió bien (transversal)

- **Plan mode iterado ~8 rondas con Codex antes de escribir código** → cero ajustes de protocolo durante
  la corrida; 12 decisiones congeladas auditables se respetaron sin desviación.
- **Pre-cache strategy bien dimensionada** → estimación original 2-3 días GPU, corrida real **6.9 h**;
  240 runs sin un solo fallo, sin un solo NaN.
- **Trazabilidad de calibración** (`calib_manifest.json` + SHA256) → bit-exact reproducible quién fue
  calibración y quién fue evaluación.
- **Spike Fase 1.0 pre-implementación** → detectó que los mecanismos E2 no eran drop-in para WavLM antes
  de comprometer el código a esa asunción; reimplementación honesta vs import directo, mecanismos
  paritarios frame-level para comparación limpia.
- **Disciplina contra selección post hoc:** 1 calib repeat N-adapt congelado de entrada (no ampliado al
  ver resultados); contrastes por mecanismo (no "mejor mecanismo vs baseline"); bootstrap sobre
  per-speaker values, no sobre runs; estatuto de N-adapt declarado como secundaria menos estable que 0B.

### Arco completo en una línea

- **0A:** el descriptor tiene firma propia.
- **0B:** solo no alcanza en estricto, pero es específico bajo calibración.
- **1:** inyectado en WavLM frozen, aporta evidencia formal en estricto (concat) y uniforme en adaptativo
  (los tres), con disociación geométrica entre mecanismos que abre la lectura interpretativa.
  *(Ver nota de supersesión: el punto de N-strict quedó acotado por el cierre cross-language.)*

---

## 2026-07-02 — Cierre Fase 1: cross-language EN↔ZH (positivo acotado a N-adapt)

**id:** `2026-07-02-cierre-fase1-crosslang`
**tipo:** `cierre`
**supersede:** `2026-06-27-retrospectiva` — **parcial**, alcance `### Fase 1` (acota su lectura de
N-strict y actualiza sus valores N-adapt; las secciones 0A y 0B no se tocan)
**Registro:** NOTAS §S67 + commits `d4393fa` y `63f1748` (2026-07-02). **Ocurrencia:** 2026-07-02.
**Backfill a esta bitácora:** 2026-07-24 — el cierre estaba registrado en NOTAS y en el REPORTE, pero
no tenía entrada propia acá. Contenido reconstruido de esas fuentes, no de memoria.

**Qué lo desbloqueó.** UNC entregó el **EN N-adapt limpio** (job `1158456`, A30, commit `131b358`,
2026-07-01): 120 records adapt, fix B2 verificado. Era el único bloque que faltaba para cerrar el
contraste translingüístico.

**Método.** WavLM-large frozen + inyección {concat/film/xattn} frame-level, LOSO 10 speakers/lengua,
métrica UAR (chance 0.20; ESD, 5 emociones). Shift cross-language = mean(Δ_ZH) − mean(Δ_EN), bootstrap
independiente 1000 resamples, seed 42 (hablantes distintos por lengua → **no pareado**).

### Hallazgo 1 — el fix B2 no cambió las conclusiones

El bug (mismas 25 utts de calibración para los 10 speakers) afectaba *qué* utts calibraban, no la
*magnitud* del beneficio:

| Mecanismo | Δ buggy | Δ limpio |
|---|---|---|
| concat | +0.044 | **+0.042** |
| film | +0.041 | **+0.036** |
| xattn | +0.044 | **+0.041** |

`none` absoluto 0.6975 → 0.6977. **El lift era real, no artefacto del bug.**

### Hallazgo 2 (central) — disociación N-strict vs N-adapt: lecturas cross-language OPUESTAS

**N-adapt** (hay calibración per-speaker en test):

| Mecanismo | Δ EN | Δ ZH | shift | CI95 shift |
|---|---|---|---|---|
| concat | +0.042 | +0.044 | **+0.001** | [−.027, +.031] |
| film | +0.036 | +0.035 | **−0.001** | [−.036, +.035] |
| xattn | +0.041 | +0.017 | −0.024 | [−.056, +.007] |

concat y film **replican limpio cross-lengua** (ambos Δ con CI que excluye 0, shift centrado en 0).
xattn más débil en ZH; su shift no es significativo.

**N-strict** (sin info per-speaker en test — es la **primaria**): el lift EN **no transfiere**.

| Mecanismo | Δ EN | Δ ZH | shift | CI95 shift |
|---|---|---|---|---|
| concat | +0.039 | +0.009 (cruza 0) | — | — |
| film | +0.016 | **−0.053** | −0.069 | [−.127, −.013] |
| xattn | +0.023 | **−0.032** | −0.055 | [−.099, −.020] |

film y xattn se vuelven **negativos** en ZH.

### Lectura honesta

El descriptor armónico transfiere de forma reproducible entre EN y ZH **cuando existe anclaje
per-speaker en test (N-adapt)**. Bajo speaker-independence estricto (N-strict) WavLM-only ya satura y el
descriptor no agrega de forma robusta ni consistente entre lenguas. **No es GO limpio ni NULL limpio:
positivo acotado al régimen N-adapt.** El juicio GO/NO-GO es del usuario.

### Caveats vigentes

1. **Hardware mixto:** EN N-adapt corrió en A30, el resto en 3090 → caveat sobre el contraste
   *secundario* N-adapt. El **primario N-strict es hardware-limpio**.
2. Piloto n=10 speakers/lengua — señal, no prueba fuerte.
3. ESD es habla **actuada**; el mandarín es tonal (F0 léxico); WavLM está sesgado a EN.
4. El contraste cross-language **no es pareado**.
5. **Resuelto, no es confound:** el régimen de epochs (early-stopping, `epochs_trained` 6–30) es el mismo
   en EN-strict, EN-adapt y ZH.

### Decisión abierta (del usuario)

Cerrar Fase 1 con el matiz / Fase 1.2 atacando N-strict / saltar a Fase 3 naturalística (MSP-Podcast o
EMOVOME — ver entrada siguiente).

**Artefactos:** `data/voz_expresiva/REPORTE_CROSS_LANGUAGE_EN_ZH.md` (informe integrado definitivo),
`Documents/NOTAS_CLAUDE-CODEX.md` §S67.

---

## 2026-07-02 — EMOVOME: solicitud de acceso enviada

**id:** `2026-07-02-emovome`
**tipo:** `recurso`
**Registro:** commit `37795a9` (2026-07-02). **Ocurrencia:** misma fecha.

Se envió la documentación para pedir los audios de **EMOVOME** — voz emocional **espontánea** en español
(Zenodo `10694370`).

- **Estado:** documentación enviada; **acceso pendiente de aprobación**.
- **Por qué importa:** el cierre cross-language EN↔ZH dejó el positivo **acotado a N-adapt** sobre habla
  **actuada** (ESD). EMOVOME sería una vía de habla espontánea para chequear si eso sobrevive fuera de lo
  actuado — candidato de la decisión abierta de Fase 1, junto con MSP-Podcast / Fase 3 naturalística.
- **Antecedente:** señalado por ProsodIA como validación espontánea prioritaria (arousal-first).
- **Estatuto:** sin datos todavía → **no cambia lo empírico del frente**.
