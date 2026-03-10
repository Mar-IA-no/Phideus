<div align="center">

# Escalón 2
### Speech ↔ EGG Cross-Modal Alignment

![Status](https://img.shields.io/badge/Status-S2--P2.5_Attention_Running-0A7E3B?style=for-the-badge)
![Focus](https://img.shields.io/badge/Focus-Speech↔EGG-1F6FEB?style=for-the-badge)
![Updated](https://img.shields.io/badge/Updated-2026--03--10-F59E0B?style=for-the-badge)

</div>

> [!IMPORTANT]
> **Estado actual**: Escalón 2 ya cerró `S2-P0`, `S2-P1` y `S2-P2-control`. Sobre French Lombard `v1.1` (`38` speakers, `9,120` clips, ~`20h`), el baseline lineal dejó `CCA S=64.4%` contra `7.8%` random, y el baseline neural `D0` cerró con `S=77.8% @ ep25`, `CI=[72.0%, 80.8%]`. La primera fase descriptor-guided por concatenación también ya cerró: `V4-lin=67.8%`, `H-series=59.8%`, `A4-16k=77.8%=D0`. El frente ya no está corriendo “descriptores por augmentación”, sino `S2-P2.5` bajo una hipótesis más fuerte: **armonía natural como organización atencional**.
> **Próximo paso único**: cerrar `S2-P2.5` (`V4-lin-attnbias`, `H-series-xattn`, `A4-16k-xattn` control) y leerlo contra `D0` y contra la fase concat ya cerrada.

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

## `S2-P2.5` — attention-based injection ya en ejecución

El plan vigente del frente ya no es el diseño base del escalón ni el `S2-P2-main` de concatenación. El estado activo es el rediseño documentado en:

- `Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/plan_rectificacion_armonia_natural.md`

### Familias activas

| Arm activo | Descriptor | Familia | Mecanismo | Rol epistemológico |
|------------|------------|---------|-----------|-------------------|
| `H-series-xattn` | armónica natural intra-frame | **B** | `cross-attention` post-CNN | **test primario de la tesis fuerte de HIT** |
| `V4-lin-attnbias` | temporal natural | A | `attention bias` en self-attention | test de dinámica del oscilador |
| `A4-16k-xattn` | control no-ratio | C | `cross-attention` post-CNN | control adversario |
| `V4-log` | temporal comparativa | D | reservado | control paramétrico (secundario) |

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

Creado 2026-03-10, antes de que H-series-xattn y A4-16k-xattn produzcan resultados.

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
| Plan rectificado `S2-P2-main` | `S2_P2/plan_rectificacion_armonia_natural.md` | guía viva de la fase descriptor-guided |
| Descriptores nuevos | `src/bias_control/vocal_descriptors.py` | `V4-lin`, `V4-log`, `H-series`, `A4-16k` |
| Encoder augmentado concat | `src/bias_control/encoders/speech_egg_encoder_aug.py` | input augmentation usado en `S2-P2-main` |
| Encoder attn bias | `src/bias_control/encoders/speech_egg_encoder_attn_bias.py` | sesgo bilineal factorizado para `V4-lin` |
| Encoder xattn | `src/bias_control/encoders/speech_egg_encoder_xattn.py` | cross-attention residual para `H-series`/control |
| Dataset augmented | `src/bias_control/datasets/lombard_segments_aug.py` | loader con cache F0 |
| Training concat | `experiments/bias_control/escalon2/train_escalon2_descriptors.py` | fase `S2-P2-main` ya cerrada |
| Training attn | `experiments/bias_control/escalon2/train_escalon2_attn.py` | fase activa `S2-P2.5` |
| Verificación P2.5 | `experiments/bias_control/escalon2/verify_p25.py` | test suite `9/9 PASS` para attn bias + xattn |
| Preregistro P2.5 | `S2_P2/PREDICCIONES_EPISTEMOLOGICAS_P25.md` | Matriz de predicciones, regla bootstrap pareado, guardrails para nulls |
| Discusión inyección | `S2_P2/Discusion_Inyeccion_descriptores.md` | Diseño técnico de mecanismos attn bias / xattn |

## Lectura actual

Observación:
- Speech↔EGG ya tiene dataset, protocolo, baseline lineal y baseline neural cerrados.
- La fase concat ya devolvió una primera lectura empírica.
- El frente attention-based ya existe como experimento vivo, no como intención.

Hipótesis:
- si la armonía natural organiza de verdad parte del fenómeno vocal, debería hacerlo de forma más visible cuando entra como principio de atención que cuando entra como feature concatenada.
- H-series (Familia B) es el test primario de esta hipótesis; V4-lin (Familia A) testea una tesis adyacente sobre dinámica del oscilador.

Inferencia válida hoy:
- Escalón 2 ya dejó de ser una promesa de generalización y pasó a ser la primera arena donde la tesis fuerte de Phideus está siendo puesta a prueba de forma disciplinada, con preregistro interpretativo y taxonomía de familias explícita.

## Próximos pasos

1. Cerrar `S2-P2.5`: `H-series-xattn` (test primario, Familia B), `V4-lin-attnbias` (Familia A), y `A4-16k-xattn` (control Familia C, 30ep comparables).
2. Leer esos resultados contra `D0` y contra la **matriz de predicciones pre-registrada** en `PREDICCIONES_EPISTEMOLOGICAS_P25.md`.
3. Leer esos resultados contra los arms concat ya cerrados.
4. Abrir `V4-log` solo si `V4-lin-attnbias` deja señal interpretativa.
5. Abrir `V4-lin+H` o variantes cruzadas solo si hay base para hablar de complementariedad o de mecanismo.
6. Recién después extender el frente a condiciones de ruido y métricas estratificadas.

## Relación con el resto del programa

- Escalón 1 queda cerrado en su argumento principal; ya no bloquea este frente.
- Gate 6 sigue como validación downstream musical.
- Gate 8 sigue como auditoría de proyecciones.
- Gate 7.1a ya dejó una lección útil: agrandar el encoder congelado no resolvió el problema por sí mismo.

Escalón 2 es, hoy, el lugar donde Phideus deja de apoyarse solo en la mecánica descriptor-guided y pasa a probar si su tesis sobre armonía natural puede sostenerse fuera de música, entre dos sensores físicos del mismo fenómeno vocal.
