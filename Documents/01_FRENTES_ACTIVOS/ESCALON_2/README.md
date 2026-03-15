<div align="center">

# Escalón 2
### Speech ↔ EGG Cross-Modal Alignment

![Status](https://img.shields.io/badge/Status-S2--P3_Running-0A7E3B?style=for-the-badge)
![Focus](https://img.shields.io/badge/Focus-Speech↔EGG-1F6FEB?style=for-the-badge)
![Updated](https://img.shields.io/badge/Updated-2026--03--15-F59E0B?style=for-the-badge)

</div>

> [!IMPORTANT]
> **Estado actual**: Escalón 2 ya cerró `S2-P0`, `S2-P1`, `S2-P2-control` y `S2-P2-main` por concatenación. Sobre French Lombard `v1.1` (`38` speakers, `9,120` clips, ~`20h`), el baseline lineal dejó `CCA S=64.4%` contra `7.8%` random, y el baseline neural `D0` cerró con `S=77.8% @ ep25`, `CI=[72.0%, 80.8%]`. En `S2-P2.5`, el **factorial `3x2` ya no solo fue ejecutado: ya fue interpretado**. `V4-lin-attnbias=70.6% @ e25`, `V4-lin-xattn=77.0% @ e15`, `H-series-xattn=73.4% @ e29`, `H-series-attnbias=78.0% @ e29`, `A4-16k-attnbias=77.8% @ e20` y `A4-16k-xattn(30ep)=78.0% @ e25` no produjeron lift defendible sobre `D0`; el caso más claro fue `V4-lin + attn_bias`, significativamente peor. `S2-P2.5b` ya también quedó **completo `3/3`**: `H-series-pca=77.4% @ e25`, `A4-16k-pca=77.2% @ e25` y `V4-lin-pca=74.6% @ e29`. Ninguno superó a `D0`, y `V4-lin-pca` volvió a quedar por debajo de forma clara. El frente ya puede leer `concat`, `attn_bias`, `xattn` y `pca` como un primer null mecanístico formalmente cerrado.
> **Corte nuevo**: `S2-P3` ya no está solo decidido. El régimen foundation-encoder quedó **implementado y abierto** con `WavLM-Large` frozen del lado speech, precomputación `noise0` ya consolidada (`27,163` segmentos, `110.5 MB`) y `P3-D0` ya corriendo localmente. Todavía no hay lectura de resultados: el punto documental correcto es ejecución, no interpretación.
> **Próximo paso único**: sostener `P3-D0`, correr después `P3-V4-lin`, `P3-H-series` y `P3-A4-16k`, y recién entonces hacer el diagnóstico comparativo `P2 vs P3` (`CKA`, probes lineales, lectura representacional). La pregunta inmediata ya no es mecanismo, sino régimen de encoder.

## Qué es este frente

Escalón 2 es la primera prueba de generalización fuera de música dentro de Triplescaloneta. La pareja ya no es Audio↔MIDI, sino dos sensores físicos distintos del mismo fenómeno vocal:

- `Speech`: micrófono, fuente glotal más filtro del tracto vocal.
- `EGG`: electroglotógrafo, oscilación glotal medida por electrodos.

La hipótesis ya no se formula en el tono más laxo de “ver si algún descriptor ayuda”. El frente quedó reencuadrado para preguntar algo más exigente: si ciertas estructuras físicamente naturales del fenómeno vocal pueden organizar mejor la alineación cross-modal que un baseline neural sin descriptor y que controles espectrales explícitos.

## Qué ya quedó cerrado

### `S2-P0` — datos, split y población congelada

| Elemento | Estado |
|----------|--------|
| Dataset local | French Lombard `v1.1` |
| Speakers | `38` (`20F/18M`) |
| Clips | `9,120` |
| Duración real | ~`20h` |
| Split speaker | `28/5/5` (`train/validation/test`) |
| Manifest | `data/lombard/manifest.json` |
| Segment index | `data/lombard/segment_index.json` |
| Segmentos totales | `108,536` |
| Piloto limpio `noise0` | `19,910` train, `3,624` val, `3,629` test |
| Alignment audit | `data/lombard/alignment_audit.json` |
| Lag correction | `0` samples |
| Voiced threshold | `0.1494` |
| Clipping auditado | `0` |

### `S2-P1` — baseline lineal

El baseline lineal ya respondió la pregunta de posibilidad básica: Speech↔EGG tiene una señal cross-modal muy fuerte aun antes de entrenar el primer encoder neural.

| Método | Speech2EGG@10 | EGG2Speech@10 | S | CI grouped |
|--------|---------------|---------------|---|------------|
| Raw cosine | `50.4%` | `46.8%` | `46.8%` | `[38.0%, 54.5%]` |
| **CCA** | **`68.4%`** | **`64.4%`** | **`64.4%`** | **`[57.8%, 70.2%]`** |
| Ridge R² | `0.851` Speech→EGG | `0.694` EGG→Speech | — | — |

### `S2-P2-control` — baseline neural `D0`

El primer control neural ya quedó cerrado:

| Epoch | S2E@10 | E2S@10 | S | hard_neg |
|------:|-------:|-------:|--:|---------:|
| 5 | `57.4%` | `61.0%` | `57.4%` | `87.0%` |
| 10 | `75.0%` | `74.8%` | `74.8%` | `92.6%` |
| 15 | `77.4%` | `76.4%` | `76.4%` | `93.4%` |
| 20 | `77.6%` | `76.8%` | `76.8%` | `92.8%` |
| **25** | **`78.4%`** | **`77.8%`** | **`77.8%`** | **`93.8%`** |
| 30 | `79.0%` | `77.8%` | `77.8%` | `93.0%` |

Lectura vigente:
- observación: el frente ya superó claramente `raw cosine` y `CCA`;
- hipótesis compatible: dos encoders simples y simétricos ya capturan una estructura compartida fuerte entre speech y EGG;
- inferencia válida hoy: Escalón 2 ya tiene un baseline neural serio y descriptor-ready.

## Rectificación epistemológica del frente

Escalón 2 ya no sigue la formulación vieja de “`V4` + `A4-16k`”. Ese diseño se volvió insuficiente cuando el programa explicitó mejor qué entiende por **armonía natural**.

La directiva vigente es esta:

> Los descriptores primarios del frente deben derivarse de invariantes físicos del fenómeno medido. Las variantes perceptuales o logarítmicas quedan como controles comparativos, no como default.

Esa rectificación obliga a separar cuatro familias de hipótesis:

1. **Familia A: Dinámica temporal del oscilador**
   Ratios lineales de F0 entre frames sucesivos. Descriptores: `V4-lin`, `V4-log` (control paramétrico).
   Testea: "la dinámica temporal del oscilador contiene invariantes cross-modales privilegiados."
   V4-lin NO es proxy de "armonía natural" en sentido fuerte — no mide la serie armónica.

2. **Familia B: Estructura armónica natural intra-frame**
   Razones de amplitud entre armónicos de F0: H2/H1..H6/H1, concentración armónica, desviación armónica.
   Descriptores: `H-series`.
   **Test más directamente alineado con la tesis fuerte de HIT** en Escalón 2.
   Testea: "la estructura de la serie armónica física es un organizador privilegiado de información cross-modal."

3. **Familia C: Controles no-ratio (espectrales genéricos)**
   Dinámica de energía espectral por bandas, sin referencia a F0 ni ratios. Descriptores: `A4-16k`.
   Control adversario. Si iguala o supera a Familia A/B, la tesis "la armonía natural es especial" se debilita.

4. **Familia D: Variantes perceptuales/logarítmicas**
   Versiones log2 de Familia A. Descriptores: `V4-log`.
   Testea sesgo representacional (lineal vs log2), no armonía natural en sentido fuerte.
   Solo corre si V4-lin muestra señal en P2.5.

## `S2-P2-main` — concatenación ya cerrada como resultado negativo útil

La primera fase descriptor-guided del frente ya no está “corriendo”: quedó cerrada y su lectura es metodológicamente valiosa.

| Arm | Descriptor | Best S | Delta vs `D0` | Lectura mínima |
|-----|------------|--------|---------------|----------------|
| `V4-lin` | ratios lineales de `F0` | `67.8%` | `-10.0pp` | aprende, pero no mejora sobre el baseline |
| `H-series` | armónicos relativos | `59.8%` | `-18.0pp` | el arm colapsó tempranamente y no devolvió una lectura limpia |
| `A4-16k` | dinámica espectral local | `77.8%` | `+0.0pp` | efecto neto cero en concatenación |

La inferencia válida de esta fase no es “la armonía natural falló”, sino otra: **la concatenación trata al descriptor como feature adicional y no parece ser el mecanismo adecuado para la tesis fuerte del frente**.

El resultado negativo es sobre **mecanismo** (concatenación como augmentación de features), no sobre **contenido** (la información que los descriptores portan). La misma evidencia de Escalón 1 (a4r +5.5pp con cross-attention vs concatenación) soporta esta lectura: los descriptores funcionan como principios organizacionales (modulación de atención), no como contenido adicional.

## `S2-P2.5` — Factorial `3x2` ya interpretado

El plan vigente del frente ya no es el diseño base del escalón ni el `S2-P2-main` de concatenación. El estado activo es el rediseño documentado en:

- `Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/plan_rectificacion_armonia_natural.md`

### Resultados completos

| Arm | Descriptor | Familia | Mecanismo | Best S | Delta vs `D0` | Delta vs concat |
|-----|------------|---------|-----------|--------|---------------|-----------------|
| `V4-lin-attnbias` | ratios lineales F0 | A | attention bias | `70.6%` | `-7.2pp` | `+2.8pp` |
| `V4-lin-xattn` | ratios lineales F0 | A | cross-attention | `77.0%` | `-0.8pp` | `+9.2pp` |
| `H-series-attnbias` | armónica intra-frame | **B** | attention bias | `78.0%` | `+0.2pp` | `+18.2pp` |
| `H-series-xattn` | armónica intra-frame | **B** | cross-attention | `73.4%` | `-4.4pp` | `+13.6pp` |
| `A4-16k-attnbias` | control no-ratio | C | attention bias | `77.8%` | `+0.0pp` | `+0.0pp` |
| `A4-16k-xattn` | control no-ratio | C | cross-attention | `78.0% @ ep25` | `+0.2pp` | `+0.2pp` |

Lectura disciplinada del factorial:
- la transición concat → attention ya no es intuición sino dato: las familias A y B recuperan gran parte de la caída de concatenación cuando el descriptor entra como principio atencional;
- pero esa recuperación no alcanzó para producir lift defendible sobre `D0` una vez aplicada la lectura preregistrada;
- `V4-lin-xattn=77.0%` deja a la Familia A mucho más cerca del baseline neural que su brazo `attnbias`, lo que vuelve realmente interpretable la interacción descriptor × mecanismo;
- `A4-16k` cerró sus dos brazos comparables, así que el control no-ratio ya no depende de cortes provisorios a `10ep`.

### Factorial `3x2`

| Descriptor | `attn_bias` | `xattn` |
|------------|-------------|---------|
| `V4-lin` | `70.6%` | `77.0%` |
| `H-series` | `78.0%` | `73.4%` |
| `A4-16k` | `77.8%` | `78.0%` |

El factorial existe para separar tres cosas que la Fase 1 todavía mezclaba:
- efecto descriptor, promediando mecanismos;
- efecto mecanismo, promediando descriptores;
- interacción descriptor × mecanismo.

### Lectura disciplinada de estas familias

- `H-series` es el **test más directamente alineado con la tesis fuerte de HIT** en Escalón 2. Mide la estructura de la serie armónica física — el objeto central de la Harmonic Information Theory. Un resultado negativo de H-series no falsifica automáticamente HIT (puede fallar el descriptor, el mecanismo o la configuración de Fase 1), pero sus resultados tienen más peso epistemológico que los de V4-lin para la pregunta central.
- `V4-lin` testea la **dinámica temporal del oscilador** (Familia A), no la serie armónica. Un resultado positivo de V4-lin dice algo sobre invariantes del oscilador, no sobre la estructura armónica intra-frame.
- `A4-16k` es un **control adversario** (Familia C). Si iguala o supera a H-series, la ventaja no es específica de la armonía natural.
- `V4-log` (Familia D) testea un **sesgo representacional** sobre cómo parametrizar ratios temporales de F0 — secundario respecto a HIT fuerte.
- La frase “armonía natural” debe siempre especificar qué familia. V4-lin y H-series son ambos “naturales” (físicos, no perceptuales), pero testean hipótesis distintas.

### Preregistro interpretativo

La lectura de resultados de P2.5 está gobernada por la **matriz de predicciones pre-registrada** en:

`S2_P2/PREDICCIONES_EPISTEMOLOGICAS_P25.md`

Ese artefacto contiene:
- **Regla operativa**: bootstrap pareado sobre Δ = S_A - S_B, CI_Δ, umbral de 2pp
- **Matriz de predicciones**: 6 patrones de resultados → interpretaciones epistemológicas
- **Guardrails para nulls**: condiciones que un null debe cumplir antes de ser informativo
- **Asunciones explícitas** del marco P2.5

Creado 2026-03-10, antes de que la Fase 1 produjera resultados y antes de cerrar el factorial `3x2`.

### Interpretación estadística ya completada

| Arm | Best `S` | Δ vs `D0` | CI_Δ (95%) | Declaración operativa |
|-----|----------|-----------|------------|------------------------|
| `V4-lin-attnbias` | `70.6%` | `-7.2pp` | `[-10.8, -1.8]` | `D0 > arm` |
| `V4-lin-xattn` | `77.0%` | `-0.8pp` | `[-4.7, +4.1]` | `≈ D0` |
| `H-series-attnbias` | `78.0%` | `+0.2pp` | `[-3.1, +4.5]` | `≈ D0` |
| `H-series-xattn` | `73.4%` | `-4.4pp` | `[-6.5, +0.2]` | `≈ D0` |
| `A4-16k-attnbias` | `77.8%` | `+0.0pp` | `[-2.8, +1.9]` | `≈ D0` |
| `A4-16k-xattn` | `78.0%` | `+0.2pp` | `[-3.4, +4.6]` | `≈ D0` |

La formulación correcta de este corte es más austera que un cierre fuerte de teoría. La observación es que ningún brazo attention-based superó a `D0` con ventaja defendible bajo este protocolo. La hipótesis compatible es que, en Speech↔EGG, `attn_bias` y `xattn` no alcanzan por sí solos para convertir estas familias descriptoriales en lift de retrieval. La inferencia válida hoy es operacional: **los mecanismos attention-based testeados no mejoraron retrieval sobre `D0` en este dominio**, aunque sí dejaron dos señales relevantes: `V4-lin + attn_bias` perjudica claramente, y la interacción descriptor × mecanismo no es trivial.

### `S2-P2.5b` — Conditioned Projection (FiLM / `pca`) — completo

La siguiente pregunta ya no era si hacía falta reabrir concat ni si convenía saltar a un encoder foundation. La pregunta más limpia era otra: si un mecanismo mucho más liviano, que deja intacto el encoder y solo condiciona la projection head, podía rescatar señal donde `attn_bias` y `xattn` no la consolidaron. Ese fue exactamente el rol de `proj_cond / pca`, heredado de Gate 8.

| Arm | Descriptor | Best `S` | Best epoch | Delta vs `D0` | Rol |
|-----|------------|----------|------------|---------------|-----|
| `V4-lin-pca` | Familia A | `74.6%` | `29` | `-3.2pp` | dinámica del oscilador bajo conditioned projection |
| `H-series-pca` | Familia B | `77.4%` | `25` | `-0.4pp` | estructura armónica intra-frame sin tocar el encoder |
| `A4-16k-pca` | Familia C | `77.2%` | `25` | `-0.6pp` | control no-ratio bajo el mismo mecanismo liviano |

Precedente relevante: en Escalón 1, `pca` fue el mecanismo audio-side más promisorio fuera del dual (`82.6%` vs `79.2%` control). En Escalón 2 no produjo lift defendible sobre `D0`, pero sí terminó de completar el contraste mecanístico principal del frente.

## Protocolo canónico congelado

Estos parámetros ya no se tocan mientras corra `S2-P2`:

| Parámetro | Valor |
|-----------|-------|
| Sample rate | `16 kHz` |
| Ventana | `2.0 s` |
| Hop | `0.5 s` |
| Positivo | misma ventana temporal del mismo clip |
| Split | por speaker |
| Piloto inicial | `noise0` |
| Pool size | `128` |
| Métrica primaria | `S = min(Speech2EGG@10, EGG2Speech@10)` |
| CI | grouped bootstrap por speaker |

El `segment_index.json` es parte del protocolo. El frente no puede regenerar población ni redefinir ventanas entre runs.

## Artefactos disponibles

| Artefacto | Ruta | Rol |
|-----------|------|-----|
| Manifest clip-level | `data/lombard/manifest.json` | población base y split |
| Segment index window-level | `data/lombard/segment_index.json` | población canónica para evaluación |
| Alignment audit | `data/lombard/alignment_audit.json` | sincronía, clipping y voiced threshold |
| Resultados `S2-P1` | `data/lombard/p1_results/p1_results_noise0.json` | baseline lineal con CIs grouped |
| Baseline neural `D0` | `data/lombard/d0_control/` | primer control neural cerrado |
| Script `S2-P0` | `experiments/bias_control/escalon2/s2_p0_manifest.py` | ingesta, split y auditoría |
| Script `S2-P1` | `experiments/bias_control/escalon2/s2_p1_baseline_linear.py` | baseline lineal |
| Dataset neural base | `src/bias_control/datasets/lombard_segments.py` | loader canónico para `D0` |
| Encoder base | `src/bias_control/encoders/speech_egg_encoder.py` | encoder simétrico Speech/EGG |
| Eval neural | `experiments/bias_control/escalon2/eval_escalon2.py` | pool builder y retrieval |
| Wrapper WavLM | `src/bias_control/encoders/wavlm_encoder.py` | encoder frozen para `S2-P3` |
| Precomputación WavLM | `experiments/bias_control/escalon2/precompute_wavlm.py` | extracción de features `noise0` para `S2-P3` |
| Dataset precomputado | `src/bias_control/datasets/lombard_precomputed.py` | loader para features `WavLM` + `EGG` raw |
| Training `P3` | `experiments/bias_control/escalon2/train_escalon2_p3.py` | régimen foundation-encoder con `WavLM-Large` frozen |
| Plan rectificado `S2-P2-main` | `S2_P2/plan_rectificacion_armonia_natural.md` | guía viva de la fase descriptor-guided |
| Descriptores nuevos | `src/bias_control/vocal_descriptors.py` | `V4-lin`, `V4-log`, `H-series`, `A4-16k` |
| Encoder augmentado concat | `src/bias_control/encoders/speech_egg_encoder_aug.py` | input augmentation usado en `S2-P2-main` |
| Encoder attn bias | `src/bias_control/encoders/speech_egg_encoder_attn_bias.py` | sesgo bilineal factorizado para `V4-lin` |
| Encoder xattn | `src/bias_control/encoders/speech_egg_encoder_xattn.py` | cross-attention residual para `H-series`/control |
| Dataset augmented | `src/bias_control/datasets/lombard_segments_aug.py` | loader con cache F0 |
| Training concat | `experiments/bias_control/escalon2/train_escalon2_descriptors.py` | fase `S2-P2-main` ya cerrada |
| Training attn | `experiments/bias_control/escalon2/train_escalon2_attn.py` | Fase 1 cerrada y factorial `3x2` ya ejecutado localmente |
| Training `pca` | `experiments/bias_control/escalon2/train_escalon2_pca.py` | `S2-P2.5b`: conditioned projection / FiLM |
| Verificación P2.5 | `experiments/bias_control/escalon2/verify_p25.py` | test suite `9/9 PASS` para attn bias + xattn |
| Preregistro P2.5 | `S2_P2/PREDICCIONES_EPISTEMOLOGICAS_P25.md` | Matriz de predicciones, regla bootstrap pareado, guardrails para nulls |
| Interpretación estadística P2.5 | `data/lombard/p25_interpretation/p25_full_results.json` | Deltas vs `D0`, comparaciones cruzadas y patrón `P4` |
| Features `WavLM` precomputadas | `data/lombard/wavlm_features_noise0.npz` | `27,163` vectores mean-pooled `[1024]` para `S2-P3` |
| Corrida `P3-D0` | `data/lombard/p3_d0_seed42/` | primer arm de `S2-P3` ya abierto, sin lectura final todavía |
| Discusión inyección | `S2_P2/Discusion_Inyeccion_descriptores.md` | Diseño técnico de mecanismos attn bias / xattn |
| Plan A10 continuo (adyacente) | `../BIAS_CONTROL/16_GATE_9_NAT_HARM_DESCRIPTOR/PLAN_GATE9_DESCRIPTOR_REVISION.md` | rama secundaria para descriptores de recurrencia ontology-free; no integra todavía el contraste canónico de `P2.5` |

## Lectura actual

Observación:
- Speech↔EGG ya tiene dataset, protocolo, baseline lineal y baseline neural cerrados.
- La fase concat ya devolvió una primera lectura empírica.
- La fase attention-based ya devolvió las seis celdas del factorial `3x2` y ya fue interpretada bajo preregistro.
- La capa activa ahora ya no es la ejecución de `pca`, sino `S2-P3` ya implementado como contraste de encoder fuerte sobre el mismo problema.

Hipótesis:
- si la armonía natural organiza de verdad parte del fenómeno vocal, todavía podría expresarse mejor bajo un régimen foundation-encoder más rico del lado speech.
- H-series (Familia B) sigue siendo el test primario de esta hipótesis; V4-lin (Familia A) testea una tesis adyacente sobre dinámica del oscilador.

Inferencia válida hoy:
- Escalón 2 ya dejó de ser una promesa de generalización y pasó a ser la primera arena donde la tesis fuerte de Phideus está siendo puesta a prueba de forma disciplinada, con preregistro interpretativo, taxonomía de familias explícita y una secuencia de mecanismos que ya permite leer un null operacional sin sobreactuarlo como cierre fuerte.
- El siguiente contraste ya no debe preguntarse si otro mecanismo pequeño rescata la señal, sino si el régimen foundation-encoder ya abierto cambia el estatuto del null.

## Próximos pasos

1. Dejar terminar `P3-D0` y consolidar su salida canónica en `data/lombard/p3_d0_seed42/`.
2. Correr `P3-V4-lin`, `P3-H-series` y `P3-A4-16k` bajo la misma receta.
3. Ejecutar la comparación posthoc `P2 vs P3` (`paired delta CI`, `CKA`, probes lineales) antes de abrir nuevas variantes.
4. Solo después decidir si `V4-log`, `V4-lin+H` o ramas laterales como `A10d/A10e` merecen recursos adicionales.

## Relación con el resto del programa

- Escalón 1 queda cerrado en su argumento principal; ya no bloquea este frente.
- Gate 6 sigue como validación downstream musical.
- Gate 8 sigue como auditoría de proyecciones.
- Gate 7.1a ya dejó una lección útil: agrandar el encoder congelado no resolvió el problema por sí mismo.

Escalón 2 es, hoy, el lugar donde Phideus deja de apoyarse solo en la mecánica descriptor-guided y pasa a probar si su tesis sobre armonía natural puede sostenerse fuera de música, entre dos sensores físicos del mismo fenómeno vocal.
