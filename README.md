<div align="center">

# Phideus

### Harmonic Information Theory — Research Program

![Status](https://img.shields.io/badge/Status-Active_Research-0A7E3B?style=for-the-badge)
![Focus](https://img.shields.io/badge/Focus-Escalon_2-1F6FEB?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-111827?style=for-the-badge)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/AlterMundi/Phideus)

*Do frequency ratios constitute a universal informational language?*

</div>

---

## Phideus en una pagina

**Phideus** explora la **Harmonic Information Theory**: la hipotesis de que la armonia natural — razones lineales de frecuencia, serie armonica fisica, regularidades del oscilador — constituye un lenguaje informacional privilegiado para organizar, comprimir y alinear informacion entre modalidades distintas de un mismo fenomeno fisico.

El programa usa arquitecturas aprendidas como instrumentos experimentales. Si las relaciones armonicas naturales mejoran de forma causal, robusta y transferible la alineacion cross-modal entre sensores distintos — y lo hacen por encima de controles espectrales genericos y de codificaciones perceptuales —, eso constituye evidencia de que la armonia natural captura algo real de la organizacion informacional del fenomeno. La posicion epistemologica completa esta en [MARCO_EPISTEMOLOGICO_PHIDEUS.md](MARCO_EPISTEMOLOGICO_PHIDEUS.md).

**Escalon 1** (Audio <-> MIDI) establecio la mecanica: la inyeccion de descriptores reorganiza geometricamente el espacio latente y mejora retrieval de manera causal y robusta. Su resultado ya mas fuerte y metodologicamente homogéneo es `d4a4=84.0% +/-2.7pp` sobre **5 training seeds independientes**, contra `D0=75.2% +/-2.3pp`, con una separacion que sigue sosteniendo la lectura de ventaja geométrica descriptor-guided sin convertir por si sola a Escalon 1 en prueba cerrada de la tesis fuerte. **Escalon 2** (Speech <-> EGG) lleva esa mecanica al test directo de la hipotesis central: descriptores derivados de la **armonia natural** del oscilador glotal (ratios lineales de F0, estructura armonica intra-frame) contra controles espectrales y perceptuales. Al corte actual, ese frente ya cerró su primer null mecanistico: `concat`, `attn_bias`, `xattn` y `pca` dieron `12/12` condiciones `≈ D0` o peores, con `V4-lin + attn_bias` claramente por debajo. Eso no clausura la tesis fuerte, pero sí cierra el contraste sobre encoders from-scratch de este escalón. `S2-P3` ya no es fase futura: su primera pasada con encoder frozen (`WavLM-Large`) ya fue completada, y la tarea viva del frente pasa a ser el diagnostico comparativo **`P2 vs P3`**. En paralelo, Gate 9 / `A10` ya entregaron datos retrospectivos en musica y **Gate 10** ya cerró su barrido causal completo: `concat > FiLM/pca >> attn_bias`, con `a7-concat=76.4%` como mejor arm del gate y una lectura más fuerte de dominio del mecanismo sobre el descriptor. Gate 6 también se volvió más nítido: `Exp A` y `Exp B` ya cerraron negativamente en la rama `Transkun+A4`, mientras `Exp C` sigue como única línea downstream todavía abierta. **Escalon 3** ya dejó atrás la fase de apertura: `P1`, `P2`, `P4`, `P5` y `P6` ya fueron corridos en una primera pasada completa. La lectura vigente del frente es más precisa: `P2-flat` sigue como baseline general de `IID`, `P5-cqtshift` emerge como mejor brazo geométrico/OOD, y `P6` no supera a `P5` bajo la receta actual. En paralelo, el programa ya abrió dos frentes laterales con roles distintos: **Voz Expresiva Phideus**, que ya cerró su contraste `EN ↔ ZH` con una lectura más fina que un simple “replica / no replica”: en `N-adapt`, `concat` y `FiLM` replican limpio cross-language, mientras en `N-strict` el lift inglés no transfiere y `film/xattn` incluso se vuelven negativos en `ZH`; y **Atención Armónica**, que ya cerró `Fase 0`, `0.5` y `0.6`: el pair-state aparece como el salto grande, el `triangle` mejora la generalización `OOD-poly`, `connected-components` quedó falsado como lector suficiente de esa representación, y los clusterers globales deployables (`spectral`/`agglo` con `k` estimado) ya recuperan una ventaja real de `B` sobre `B-local` en `OOD-poly`. El caveat que queda es más preciso: no falta calibrar `τ`, falta resolver mejor la estimación de partición y de `k`.

Ese arco experimental ya tiene tambien una formulacion larga y teoricamente integrada en el repositorio publico del libro HIT, [AlterMundi/harmonic-information-theory](https://github.com/AlterMundi/harmonic-information-theory), con edicion web en [hit.altermundi.net](https://hit.altermundi.net/). Ahi el programa ya quedó articulado como libro de 191 páginas, incluyendo el nuevo problema de `storage / retrieval`, el `activation problem` y la convergencia con `Beacon` como parte del cierre teórico más largo del programa.

El cierre metodológico de Escalon 1 fue publicado como preprint arXiv: **[arXiv:2604.10283](https://arxiv.org/abs/2604.10283)** — *Descriptor-Injected Cross-Modal Learning: A Systematic Exploration of Audio–MIDI Alignment via Spectral and Melodic Features* (CC BY 4.0, `cs.SD` primaria, `cs.LG` cross-list).

---

## Programa actual

| Frente | Dominio | Funcion | Estado |
|---|---|---|---|
| **Escalon 1** | Audio <-> MIDI | Validacion descriptor-guided y geometria cross-modal | **Cerrado** — cierre training-seed `d4a4=84.0% +/- 2.7pp` |
| **Gate 8** | Audio <-> MIDI | Conditioned projections: donde se preserva la informacion descriptorial | **Cerrado (5/5)** — `pcd=84.2%`, `pca=82.6%` |
| **Gate 6 AMT** | Audio -> transcripcion | Validacion downstream de la senal descriptor-guided | **Activo** — `Exp A` y `Exp B` ya cerraron negativamente; `Exp C` queda como línea abierta |
| **Escalon 2** | Speech <-> EGG | Test directo de HIT: armonia natural del oscilador glotal como organizador cross-modal | **Foco principal** — null mecanistico inicial cerrado; `P3` primera pasada completa, sigue `P2 vs P3` |
| **Voz Expresiva Phideus** | Voz expresiva | Test de transferencia descriptor-guided sobre `SSL` vocal y estabilidad translingüística | **Activo** — cierre `EN ↔ ZH` ya consolidado: positivo acotado a `N-adapt`, null/negativo en `N-strict` |
| **Atencion Armonica** | Agrupamiento armónico polifónico | Incubacion arquitectonica para testear `Harmonic Pairformer`: pair-state, transitividad y `triangle` bajo evidencia per-par ambigua | **Fase 0, 0.5 y 0.6 cerradas** — `B` gana `OOD-poly` con clusterers globales deployables; queda subestimacion de `k` y Stage B |
| **Escalon 3** | Audio XY <-> Lissajous | Banco de pruebas sintetico con ground truth determinista para ratios visibles | **Activo** — baseline dual consolidado; primera linea geometrica ya corrida (`P5-cqtshift` mejor brazo OOD actual) |
| **Escalon 4** | ECG <-> PPG | Expansion a dominio fisiologico | **Proyeccion** |

En paralelo, el programa abrió una investigación transversal sobre el déficit de
ground truth para una PPU/Natural Harmonic Geometry. La primera campaña no
encontró una geometría universal de las proporciones: organizó una base
estratificada de oráculos analíticos, simulación causal, cámaras físicas y
evidencia empírica externa, atravesada por falsación adversarial y adjudicación ciega.
Esta base orienta prototipos futuros
sin promover todavía una arquitectura ni declarar GO/NO-GO. La síntesis está en
[Geometría proporcional y bases de verdad](Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/GEOMETRIA_PROPORCIONAL_BASES_DE_VERDAD.md).

Cada frente cumple un papel distinto en la exploracion de HIT. Escalon 1 fija la evidencia de referencia y la mecanica de inyeccion, y hoy ya puede leerse con su resultado flagship cerrado en training multi-seed real. Gate 8 pregunta donde se preserva la informacion armonica en las proyecciones. Gate 6 pregunta si la ventaja sobrevive fuera del retrieval y, por ahora, ya dejó una lectura más dura: la rama `Transkun+A4` no mostró ganancia útil ni en régimen base ni bajo degradación, mientras `Exp C` conserva abierta la pregunta solo desde el decoder serio sobre features congeladas. Escalon 2 es donde la hipotesis central — la armonia natural como organizador informacional privilegiado — se enfrenta directamente con descriptores derivados de la fisica del oscilador, contra controles espectrales y perceptuales. Ese primer contraste mecanistico ya quedó cerrado; `P3` ya fue corrido en una primera pasada, y la tarea que sigue es decidir si la comparación `P2 vs P3` cambia la lectura representacional del frente o confirma que el null descriptorial ya es estable bajo ambos regímenes de encoder. Voz Expresiva cumple otra función: preguntar si ese patrón descriptor-guided sobrevive cuando el backbone pasa a ser un encoder vocal foundation y cuando la comparación deja de ser solo intra-idioma. Ese frente ya no está esperando una réplica, sino leyendo una disociación concreta: el descriptor transfiere de forma reproducible entre `EN` y `ZH` cuando existe anclaje per-speaker en test (`N-adapt`), pero no sostiene una ventaja robusta en el régimen speaker-independent estricto (`N-strict`). Atención Armónica abre todavía otra clase de problema: no reitera la pregunta descriptorial de Escalón 2, sino que ensaya una hipótesis arquitectónica más fuerte sobre cómo incorporar estructura armónica cuando la ambigüedad local ya no puede resolverse con evidencia per-par cerrada. `Fase 0` construyó un problema válido; `Fase 0.5` mostró que el cuello no era `τ` sino `connected-components`; y `Fase 0.6` ya agregó algo más preciso: con clusterers globales deployables, `B` recupera una ventaja real sobre `B-local` en `OOD-poly`, aunque siga quedando lejos de una partición plenamente resuelta por la subestimación de `k`. Gate 9 / `A10` releen retrospectivamente parte de esa deuda dentro de musica, mientras Gate 10 ya dejó de ser un barrido parcial y pasó a ser evidencia cerrada de otra cosa: en esa rama retrospectiva el mecanismo pesa más que el descriptor, con `concat` arriba, `FiLM/pca` en segundo plano y `attn_bias` claramente abajo. Escalon 3, por su parte, ya no vive en `E3-P0`: hoy tiene un baseline dual consolidado, un régimen de probes ya auditado y una primera linea geométrica completa donde `P5-cqtshift` queda como mejor brazo OOD y `P6` no se vuelve el ganador del frente. Escalon 4 conserva la expansion fisiologica fuera de acústica.

---

## Resultados de referencia

### Escalon 1 — Audio <-> MIDI

Referencia canonica sobre MAESTRO. La mejora opera como ventaja geometrica del espacio latente (+82% CKA), no como enriquecimiento de decodificabilidad local.

| Brazo | `S` (canonical reference) | Lectura |
|---|---:|---|
| `D0` | `75.2% +/- 2.3pp` | Baseline sin descriptor |
| `a4r` | `80.7% +/- 1.9pp` | Reverse cross-attention con descriptor audio |
| `d4-a4r` | `81.2% +/- 2.5pp` | Variante mixta |
| `d4a4` | `84.0% +/- 2.7pp` | Mejor referencia del frente. Cierre sobre 5 training seeds independientes |

Los cuatro brazos canónicos de Escalon 1 ya tienen lectura homogénea en training-seed: `D0=75.2% +/- 2.3pp`, `a4r=80.7% +/- 1.9pp`, `d4-a4r=81.2% +/- 2.5pp` y `d4a4=84.0% +/- 2.7pp`.

### Gate 8 — Conditioned Projections

La informacion descriptorial es util incluso inyectada en la projection head (FiLM), no solo en el encoder.

| Brazo | Best `S` | Delta vs ctrl |
|---|---:|---:|
| `ctrl` (sin condicionamiento) | `79.2%` | — |
| `pcm` (MIDI cond) | `80.0%` | `+0.8pp` |
| `pcd-zero` (dual cond, zeros) | `81.8%` | `+2.6pp` |
| `pca` (audio cond) | `82.6%` | `+3.4pp` |
| `pcd` (dual cond A4+D4) | `84.2%` | `+5.0pp` |

`pcd > pca > pcd-zero > pcm > ctrl`: el cierre completo ya deja una lectura mas fuerte. La arquitectura conditioned aporta expresividad (`pcd-zero > ctrl`), el conditioning real aporta senal adicional (`pcd > pcd-zero`), y el lado audio responde mejor que el MIDI-side cuando se lo condiciona de forma aislada (`pca > pcm`).

### Escalon 2 — Speech <-> EGG

| Capa | Resultado | Significado |
|---|---:|---|
| Baseline lineal `CCA` | `S=64.4%` | La senal cross-modal existe antes del primer encoder neural |
| Baseline neural `D0` | `S=77.8%`, `CI=[72.0%, 80.8%]` | Piso solido para comparar descriptores |
| Concatenacion (`S2-P2-main`) | `V4-lin=67.8%`, `H-series=59.8%`, `A4-16k=77.8%` | La concatenacion trata descriptores como features — mecanismo inadecuado |
| Atencion (`S2-P2.5`) | Interpretado | `V4-lin-xattn=77.0%`, `H-series-attnbias=78.0%`, `A4-16k-attnbias=77.8%`, `A4-16k-xattn=78.0%`; ningun brazo mejora a `D0` de forma defendible |
| Proj. condicionada (`S2-P2.5b`) | Completa | `V4-lin-pca=74.6%`, `H-series-pca=77.4%`, `A4-16k-pca=77.2%`; ningun brazo superó a `D0` |
| Regimen foundation (`S2-P3`) | Primera pasada completa | `P3-D0=78.8%`, `P3-A4-16k-pca=78.2%`, `P3-V4-lin-pca=76.8%`, `P3-H-series-pca=75.6%`; siguiente tarea = `P2 vs P3` |

`S2-P2.5` testea la hipotesis central de HIT a nivel de mecanismo: la armonia natural debe guiar la atencion del modelo (organizar la computacion), no aumentar su contenido. `V4-lin` (dinamica del oscilador) entra como Familia A, `H-series` (estructura armonica intra-frame) como Familia B y probe mas directamente alineado con la tesis fuerte, y `A4-16k` queda como control no-ratio de Familia C. Esa fase ya fue leida con el preregistro [PREDICCIONES_EPISTEMOLOGICAS_P25.md](Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/PREDICCIONES_EPISTEMOLOGICAS_P25.md): la conclusion valida hoy es operativa, no grandilocuente. Los mecanismos `concat`, `attn_bias`, `xattn` y `pca` no dieron lift de retrieval sobre `D0` en Speech↔EGG y dejaron un primer null mecanistico formalmente cerrado. `S2-P3` ya cumplió su primera pasada con encoder frozen (`WavLM-Large`) y no desplazó a `P3-D0`; la tarea viva ahora es cerrar `P2 vs P3` con `CKA`, probes lineales y lectura representacional.

---

## Como entrar al repo

| Si queres... | Empezar por... |
|---|---|
| Entender que tipo de conocimiento produce Phideus | [MARCO_EPISTEMOLOGICO_PHIDEUS.md](MARCO_EPISTEMOLOGICO_PHIDEUS.md) |
| Ver el estado canonico del proyecto | [Proyecto_Estado_Actual.md](Documents/00_TRONCAL/Proyecto_Estado_Actual.md) |
| Ver el mapa visual de frentes y dependencias | [MAPA_VISUAL_DEL_PROGRAMA.md](Documents/05_WIKI/MAPA_VISUAL_DEL_PROGRAMA.md) |
| Dar contexto integral del programa a un agente | [LLM_CONTEXT.md](Documents/05_WIKI/LLM_CONTEXT.md) |
| Ver la estructura global de documentacion | [INDICE_DOCUMENTACION.md](Documents/00_TRONCAL/INDICE_DOCUMENTACION.md) |
| Entrar por la formulacion larga del programa | [AlterMundi/harmonic-information-theory](https://github.com/AlterMundi/harmonic-information-theory) |
| Leer la edición web pública del libro HIT | [hit.altermundi.net](https://hit.altermundi.net/) |
| Leer el paper de Escalon 1 | [arXiv:2604.10283](https://arxiv.org/abs/2604.10283) |
| Ir al frente musical consolidado | [ROADMAP_BIAS_CONTROL.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md) |
| Ir al frente vocal actual | [ESCALON_2/README.md](Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md) |
| Ver el frente de voz expresiva | [Voz_Expresiva_Phideus/README.md](Documents/01_FRENTES_ACTIVOS/Voz_Expresiva_Phideus/README.md) |
| Ver la incubación Atención Armónica | [Atencion_Armonica/README.md](Documents/01_FRENTES_ACTIVOS/Atencion_Armonica/README.md) |
| Ver el preregistro interpretativo de Escalón 2 | [PREDICCIONES_EPISTEMOLOGICAS_P25.md](Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/PREDICCIONES_EPISTEMOLOGICAS_P25.md) |
| Ver el nuevo frente Lissajous | [ESCALON_3/README.md](Documents/01_FRENTES_ACTIVOS/ESCALON_3/README.md) |
| Entender la historia de los descriptores | [CATALOGO_NARRATIVO_DESCRIPTORES_RATIOS_PHIDEUS.md](Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/CATALOGO_NARRATIVO_DESCRIPTORES_RATIOS_PHIDEUS.md) |
| Ver la historia larga del proyecto | [INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md](Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md) |
| Ver skills compartidas | [Documents/Skills/README.md](Documents/Skills/README.md) |

---

## Visualizaciones y documentacion viva

### Visualizaciones 3D interactivas

**[altermundi.github.io/Phideus](https://altermundi.github.io/Phideus/)**

Exploraciones de arquitecturas y lineas principales del programa: baseline cross-modal, reverse cross-attention, configuraciones duales de Escalon 1.

### Skills compartidas

**[Documents/Skills/README.md](Documents/Skills/README.md)**

Skills reutilizables concentradas en operacion HPC/SLURM, validacion pre-submit y lecciones operativas.

### Estructura de documentacion

- `Documents/00_TRONCAL/` — estado ejecutivo, indices, documentos troncales
- `Documents/01_FRENTES_ACTIVOS/` — documentacion operativa de cada frente vivo
- `Documents/04_TRANSVERSAL/` — teoria, fundamentos, historia
- `Documents/05_WIKI/` — síntesis viva de frentes, roadmaps, relaciones y alternativas para humanos y agentes

---

## Infraestructura computacional

Parte del programa utiliza recursos de **UNC Supercomputo (CCAD)** de la **Universidad Nacional de Cordoba**, integrados al **Sistema Nacional de Computacion de Alto Desempeno (SNCAD)** de la Republica Argentina.

Para publicaciones derivadas de corridas en esa infraestructura, el proyecto adopta la formulacion institucional recomendada:

**[supercomputo.unc.edu.ar/equipamiento/citar-recursos](https://supercomputo.unc.edu.ar/equipamiento/citar-recursos/)**

---

## Reproduccion minima

### Setup del entorno

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Pipeline base de BIAS_CONTROL

```bash
python experiments/bias_control/run_all_gates.py \
  --maestro-dir data/maestro_v3/maestro-v3.0.0 \
  --output data/bias_control_medium
```

### Ejemplo: Gate 4.3 `d4a4`

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python experiments/bias_control/gate42_training.py \
  --descriptor d4a4 \
  --checkpoint data/bias_control_medium/training_outputs/foundation_locked_e25.pt \
  --output data/bias_control_medium/training_outputs/gate43/d4a4 \
  --maestro-dir data/maestro_v3/maestro-v3.0.0 \
  --epochs 5 --batch-size 16 --num-workers 8 \
  --freeze-policy run-d --seed 42 --device cuda
```

### Evaluacion estructurada

```bash
python experiments/bias_control/evaluate_structured_pool.py \
  --model <checkpoint.pt> \
  --output <output.json> \
  --pool-size 256 --n-queries 500 --seed 42 \
  --maestro-dir data/maestro_v3/maestro-v3.0.0
```

Protocolo canonico: `pool=256`, `queries=500`, `seed=42`.

---

<!-- BELOW THE FOLD -->

<details>
<summary><strong>Roadmap del programa</strong></summary>

### TripleScaloneta

| Escalon | Dominio | Rol | Estado |
|---|---|---|---|
| Escalon 1 | MAESTRO Audio <-> MIDI | Validacion descriptor-guided y cierre cientifico del primer banco de pruebas | **Cerrado** |
| Escalon 2 | Speech <-> EGG | Test directo de HIT: armonia natural del oscilador como organizador cross-modal | **Activo (null mecanistico inicial cerrado; `S2-P3` primera pasada completa)** |
| Escalon 3 | Audio XY <-> Lissajous | Banco sintetico con ratio visible y control total de parametros | **Activo** (`P2/P4/P5/P6` ya corridos en primera pasada) |
| Escalon 4 | ECG <-> PPG | Expansion fisiologica | **Proyeccion** |

### Frentes activos

| Frente | Funcion | Documento |
|---|---|---|
| Gate 6 AMT | Validacion downstream | [12_GATE_6_AMT/README.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/12_GATE_6_AMT/README.md) |
| Gate 8 | Conditioned projections | [15_GATE_8_CONDITIONED_PROJECTIONS/README.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/15_GATE_8_CONDITIONED_PROJECTIONS/README.md) |
| Gate 10 | Mechanism sweep audio-only | [17_GATE_10_MECHANISM_SWEEP/README.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/17_GATE_10_MECHANISM_SWEEP/README.md) |
| Escalon 2 | Frente principal (null mecanistico inicial cerrado; sigue diagnostico `P2 vs P3`) | [ESCALON_2/README.md](Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md) |
| Escalon 3 | Banco Lissajous con baseline dual y primera linea geometrica ya consolidada | [ESCALON_3/README.md](Documents/01_FRENTES_ACTIVOS/ESCALON_3/README.md) |

### Roadmaps canonicos

- [ROADMAP_BIAS_CONTROL.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md)
- [ROADMAP_ESCALON_2.md](Documents/01_FRENTES_ACTIVOS/ESCALON_2/ROADMAP_ESCALON_2.md)
- [ROADMAP_ESCALON_3.md](Documents/01_FRENTES_ACTIVOS/ESCALON_3/ROADMAP_ESCALON_3.md)

</details>

<details>
<summary><strong>Arquitectura y familias descriptoriales</strong></summary>

### Arquitectura general

Phideus trabaja con configuraciones cross-modales contrastivas donde la armonia natural se inyecta como senal organizadora:

```text
modalidad A -> encoder -> projection -> embedding
                  ^
            armonia natural
                  v
modalidad B -> encoder -> projection -> embedding
                    \      VICReg      /
```

La investigacion no se limita a que encoder usar. La pregunta central es como entra la armonia natural (como augmentation, atencion o modulacion), que geometria induce, y si esa geometria es especifica de relaciones armonicas o aparece con cualquier descriptor auxiliar.

### Escalon 1: familias

| Familia | Ejemplos | Rol |
|---|---|---|
| Control | `D0` | Baseline sin descriptor |
| MIDI local | `D4` | Relaciones locales del lado MIDI |
| Audio espectral | `A4`, `A4r` | Dinamica espectral del lado audio |
| Dual | `d4a4`, `d4-a4r` | Combinaciones de mayor rendimiento |

### Escalon 2: taxonomia armonica

| Familia | Descriptor | Rol en la exploracion de HIT |
|---|---|---|
| **Armonia natural temporal** | `V4-lin` | Dinamica lineal del oscilador — testea si ratios naturales de F0 organizan atencion inter-frame |
| **Armonia natural intra-frame** | `H-series` | Estructura armonica (H2/H1..H6/H1) — testea si la serie armonica fisica organiza features |
| Control perceptual | `V4-log` | Misma info que V4-lin en escala logaritmica — testea si la escala importa |
| Control espectral | `A4-16k` | Dinamica espectral generica no-ratio — testea si cualquier descriptor auxiliar ayuda |

Ver: [MARCO_EPISTEMOLOGICO_PHIDEUS.md](MARCO_EPISTEMOLOGICO_PHIDEUS.md) y [plan_rectificacion_armonia_natural.md](Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/plan_rectificacion_armonia_natural.md)

</details>

<details>
<summary><strong>Linea experimental consolidada</strong></summary>

### Escalon 1

| Brazo | `S` (canonical reference) |
|---|---:|
| `D0` | `75.2% +/- 2.3pp` |
| `a4r` | `80.7% +/- 1.9pp` |
| `d4-a4r` | `81.2% +/- 2.5pp` |
| `d4a4` | `84.0% +/- 2.7pp` *(5 training seeds independientes)* |

### Gate 8

| Brazo | Best `S` | Delta vs ctrl |
|---|---:|---:|
| `ctrl` | `79.2%` | — |
| `pcm` | `80.0%` | `+0.8pp` |
| `pcd-zero` | `81.8%` | `+2.6pp` |
| `pcd` | `84.2%` | `+5.0pp` |
| `pca` | `82.6%` | `+3.4pp` |

### Gate 6 / Gate 7.1

| Frente | Corte |
|---|---|
| Gate 6 AMT | `Exp C` local cerro (`F1=0.157`); `Exp A` y `Exp B` ya cerraron negativamente en la rama `Transkun+A4` |
| Gate 7.1a | `D0_mert330m_frozen=75.0%`, sin mejora sobre `D0_lite=75.2%` |

### Escalon 2

| Capa | Resultado |
|---|---:|
| CCA baseline | `S=64.4%` |
| D0 neural | `S=77.8%` |
| Concatenacion | `V4-lin=67.8%`, `H-series=59.8%`, `A4-16k=77.8%` |
| Atencion (`S2-P2.5`) | Interpretado bajo preregistro |

</details>

<details>
<summary><strong>Documentacion clave y estructura del repo</strong></summary>

### Documentos principales

| Documento | Funcion |
|---|---|
| [Proyecto_Estado_Actual.md](Documents/00_TRONCAL/Proyecto_Estado_Actual.md) | Estado ejecutivo |
| [INDICE_DOCUMENTACION.md](Documents/00_TRONCAL/INDICE_DOCUMENTACION.md) | Mapa global |
| [ROADMAP_BIAS_CONTROL.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md) | Roadmap musical |
| [ESCALON_2/README.md](Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md) | Frente vocal |
| [Documents/Skills/README.md](Documents/Skills/README.md) | Skills compartidas |

### Estructura

```text
Phideus/
├── src/                         # Modulos del proyecto
├── experiments/                 # Training, evaluacion y utilidades experimentales
├── Documents/
│   ├── 00_TRONCAL/              # Estado ejecutivo, indices, documentos troncales
│   ├── 01_FRENTES_ACTIVOS/      # Frentes vivos
│   ├── 02_FRENTES_PAUSADOS/     # Frentes pausados
│   ├── 03_FRENTES_CERRADOS/     # Frentes cerrados
│   └── 04_TRANSVERSAL/          # Teoria, fundamentos, historia
├── viz/                         # Visualizaciones interactivas
├── data/                        # Datasets y outputs (no versionados)
└── config/                      # Configuraciones
```

</details>

---

> *"El bosque ya canta. Nuestra tarea es entender su afinacion."*

**Licencia**: MIT — ver [LICENSE.md](LICENSE.md)
