<div align="center">

# Escalón 2
### Speech ↔ EGG Cross-Modal Alignment

![Status](https://img.shields.io/badge/Status-S2--P2_Main_Running-0A7E3B?style=for-the-badge)
![Focus](https://img.shields.io/badge/Focus-Speech↔EGG-1F6FEB?style=for-the-badge)
![Updated](https://img.shields.io/badge/Updated-2026--03--08-F59E0B?style=for-the-badge)

</div>

> [!IMPORTANT]
> **Estado actual**: Escalón 2 ya cerró `S2-P0`, `S2-P1` y `S2-P2-control`. Sobre French Lombard `v1.1` (`38` speakers, `9,120` clips, ~`20h`), el baseline lineal dejó `CCA S=64.4%` contra `7.8%` random, y el baseline neural `D0` cerró con `S=77.8% @ ep25`, `CI=[72.0%, 80.8%]`. El frente ya no está discutiendo posibilidad básica: está corriendo `S2-P2-main` bajo una rectificación explícita de armonía natural.
> **Próximo paso único**: cerrar la fase primaria descriptor-guided (`V4-lin`, `H-series`, `A4-16k`) y leerla contra `D0`, no contra el azar.

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

Esa rectificación obliga a separar tres familias de hipótesis:

1. **Dinámica temporal del oscilador**  
   Ejemplo: `V4-lin`, ratios lineales de `F0` entre frames sucesivos.
2. **Estructura armónica natural intra-frame**  
   Ejemplo: `H-series`, razones de amplitud armónica alrededor de `k*F0`.
3. **Controles no-ratio**  
   Ejemplo: `A4-16k`, dinámica espectral local por bandas.

## `S2-P2-main` — descriptor-guided ya en ejecución

El plan vigente del frente ya no es el plan base del escalón, sino el rediseño documentado en:

- `Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/plan_rectificacion_armonia_natural.md`

### Familias activas

| Arm | Familia | Qué mide | Rol actual |
|-----|---------|----------|------------|
| `V4-lin` | Temporal natural | dinámica lineal del oscilador glotal | primaria |
| `H-series` | Armónica natural | estructura armónica intra-frame | primaria |
| `A4-16k` | Control espectral | dinámica espectral local no-ratio | primaria |
| `V4-log` | Temporal perceptual | control logarítmico/comparativo | secundaria |
| `V4-lin+H` | Combinado natural | complementariedad temporal + armónica | secundaria |

### Lectura disciplinada de estas familias

- `V4-lin` no es “la armonía natural entera”: es dinámica temporal del oscilador.
- `H-series` es la apuesta más directamente alineada con la tesis fuerte del proyecto.
- `A4-16k` no es un descriptor de armonía natural: es un control de dinámica espectral local.
- `V4-log` no está ahí como default, sino como brazo comparativo para no confundir utilidad relacional con privilegio de coordenadas físicas lineales.

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
| Encoder augmented | `src/bias_control/encoders/speech_egg_encoder_aug.py` | input augmentation descriptor-guided |
| Dataset augmented | `src/bias_control/datasets/lombard_segments_aug.py` | loader con cache F0 |
| Training descriptor-guided | `experiments/bias_control/escalon2/train_escalon2_descriptors.py` | corrida de brazos primarios y secundarios |

## Lectura actual

Observación:
- Speech↔EGG ya tiene dataset, protocolo, baseline lineal y baseline neural cerrados.
- El frente descriptor-guided ya existe como experimento vivo, no como intención.

Hipótesis:
- si la armonía natural organiza de verdad parte del fenómeno vocal, alguna de las familias primarias debería mostrar ventaja sobre `D0` y sobre el control espectral.

Inferencia válida hoy:
- Escalón 2 ya dejó de ser una promesa de generalización y pasó a ser la primera arena donde la tesis fuerte de Phideus está siendo puesta a prueba de forma disciplinada.

## Próximos pasos

1. Cerrar la fase primaria de `S2-P2-main`: `V4-lin`, `H-series`, `A4-16k`.
2. Leer esos resultados contra `D0`, no contra el azar.
3. Abrir `V4-log` solo si `V4-lin` deja señal interpretativa.
4. Abrir `V4-lin+H` solo si hay base para hablar de complementariedad.
5. Recién después extender el frente a condiciones de ruido y métricas estratificadas.

## Relación con el resto del programa

- Escalón 1 queda cerrado en su argumento principal; ya no bloquea este frente.
- Gate 6 sigue como validación downstream musical.
- Gate 8 sigue como auditoría de proyecciones.
- Gate 7.1a ya dejó una lección útil: agrandar el encoder congelado no resolvió el problema por sí mismo.

Escalón 2 es, hoy, el lugar donde Phideus deja de apoyarse solo en la mecánica descriptor-guided y pasa a probar si su tesis sobre armonía natural puede sostenerse fuera de música, entre dos sensores físicos del mismo fenómeno vocal.
