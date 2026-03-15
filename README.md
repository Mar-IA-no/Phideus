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

**Escalon 1** (Audio <-> MIDI) establecio la mecanica: la inyeccion de descriptores reorganiza geometricamente el espacio latente y mejora retrieval de manera causal y robusta (`d4a4=84.1% +/-2.3pp`, +9.4pp sobre baseline, 5 seeds). **Escalon 2** (Speech <-> EGG) lleva esa mecanica al test directo de la hipotesis central: descriptores derivados de la **armonia natural** del oscilador glotal (ratios lineales de F0, estructura armonica intra-frame) contra controles espectrales y perceptuales. Al corte actual, ese frente ya cerró su primer null mecanistico: `concat`, `attn_bias`, `xattn` y `pca` dieron `12/12` condiciones `≈ D0` o peores, con `V4-lin + attn_bias` claramente por debajo. Eso no clausura la tesis fuerte, pero sí cierra el contraste sobre encoders from-scratch de este escalón. El siguiente paso real ya no es más bootstrap ni nuevos barridos ciegos, sino **`S2-P3`**: speech encoder frozen de referencia (`WavLM` o `HuBERT`) seguido por un diagnóstico comparativo `P2 vs P3`. En paralelo, Gate 9 / `A10` ya entregaron datos retrospectivos en musica, **Gate 10** ya está en ejecución parcial en UNC con una lectura provisoria `FiLM/pca > concat > attn_bias`, y **Escalon 3** abre una nueva linea conceptual sobre audio XY <-> figuras de Lissajous.

---

## Programa actual

| Frente | Dominio | Funcion | Estado |
|---|---|---|---|
| **Escalon 1** | Audio <-> MIDI | Validacion descriptor-guided y geometria cross-modal | **Cerrado** — `d4a4=84.1%`, cierre causal robusto |
| **Gate 8** | Audio <-> MIDI | Conditioned projections: donde se preserva la informacion descriptorial | **Cerrado (5/5)** — `pcd=84.2%`, `pca=82.6%` |
| **Gate 6 AMT** | Audio -> transcripcion | Validacion downstream de la senal descriptor-guided | **Activo** — `Exp B` ya leido como cierre negativo; `Exp A` queda en screening minimo pendiente |
| **Escalon 2** | Speech <-> EGG | Test directo de HIT: armonia natural del oscilador glotal como organizador cross-modal | **Foco principal** — null mecanistico inicial cerrado; siguiente fase `S2-P3` |
| **Escalon 3** | Audio XY <-> Lissajous | Banco de pruebas sintetico con ground truth determinista para ratios visibles | **Diseño conceptual** |
| **Escalon 4** | ECG <-> PPG | Expansion a dominio fisiologico | **Proyeccion** |

Cada frente cumple un papel distinto en la exploracion de HIT. Escalon 1 fija la evidencia de referencia y la mecanica de inyeccion. Gate 8 pregunta donde se preserva la informacion armonica en las proyecciones. Gate 6 pregunta si la ventaja sobrevive fuera del retrieval y, por ahora, ya dejo un negativo util en degradacion. Escalon 2 es donde la hipotesis central — la armonia natural como organizador informacional privilegiado — se enfrenta directamente con descriptores derivados de la fisica del oscilador, contra controles espectrales y perceptuales. Ese primer contraste mecanistico ya quedó cerrado; lo que sigue es `S2-P3`, para separar mejor null de descriptor y null de encoder. Gate 9 / `A10` releen retrospectivamente esa deuda dentro de musica, mientras Gate 10 ya paso de ser un barrido “listo” a un barrido parcialmente observado, precisamente para separar contenido descriptorial de mecanismo de inyeccion. Escalon 3, por su parte, abre un banco sintetico donde el ratio pasa a ser visible y generable con ground truth exacto, y Escalon 4 conserva la expansion fisiologica fuera de acústica.

---

## Resultados de referencia

### Escalon 1 — Audio <-> MIDI

Referencia canonica sobre MAESTRO. La mejora opera como ventaja geometrica del espacio latente (+82% CKA), no como enriquecimiento de decodificabilidad local.

| Brazo | `S` (multi-seed) | Lectura |
|---|---:|---|
| `D0` | `75.2% +/- 2.3pp` | Baseline sin descriptor |
| `a4r` | `80.7% +/- 1.9pp` | Reverse cross-attention con descriptor audio |
| `d4-a4r` | `81.2% +/- 2.5pp` | Variante mixta |
| `d4a4` | `84.1% +/- 2.3pp` | Record del frente. Cierre causal: +9.4pp por contenido descriptorial real |

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
| Proximo contraste (`S2-P3`) | Decidido | encoder frozen `WavLM/HuBERT` + diagnostico comparativo `P2 vs P3` |

`S2-P2.5` testea la hipotesis central de HIT a nivel de mecanismo: la armonia natural debe guiar la atencion del modelo (organizar la computacion), no aumentar su contenido. `V4-lin` (dinamica del oscilador) entra como Familia A, `H-series` (estructura armonica intra-frame) como Familia B y probe mas directamente alineado con la tesis fuerte, y `A4-16k` queda como control no-ratio de Familia C. Esa fase ya fue leida con el preregistro [PREDICCIONES_EPISTEMOLOGICAS_P25.md](Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/PREDICCIONES_EPISTEMOLOGICAS_P25.md): la conclusion valida hoy es operativa, no grandilocuente. Los mecanismos `concat`, `attn_bias`, `xattn` y `pca` no dieron lift de retrieval sobre `D0` en Speech↔EGG y dejaron un primer null mecanistico formalmente cerrado. El paso siguiente ya no es seguir abriendo variantes del mismo encoder, sino correr `S2-P3` con un speech encoder frozen de referencia (`WavLM` o `HuBERT`) y comparar despues `P2 vs P3` con `CKA`, probes lineales y lectura representacional.

---

## Como entrar al repo

| Si queres... | Empezar por... |
|---|---|
| Entender que tipo de conocimiento produce Phideus | [MARCO_EPISTEMOLOGICO_PHIDEUS.md](MARCO_EPISTEMOLOGICO_PHIDEUS.md) |
| Ver el estado canonico del proyecto | [Proyecto_Estado_Actual.md](Documents/00_TRONCAL/Proyecto_Estado_Actual.md) |
| Ver la estructura global de documentacion | [INDICE_DOCUMENTACION.md](Documents/00_TRONCAL/INDICE_DOCUMENTACION.md) |
| Ir al frente musical consolidado | [ROADMAP_BIAS_CONTROL.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md) |
| Ir al frente vocal actual | [ESCALON_2/README.md](Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md) |
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
| Escalon 2 | Speech <-> EGG | Test directo de HIT: armonia natural del oscilador como organizador cross-modal | **Activo (null mecanistico inicial cerrado; `S2-P3` decidido)** |
| Escalon 3 | Audio XY <-> Lissajous | Banco sintetico con ratio visible y control total de parametros | **Diseño conceptual** |
| Escalon 4 | ECG <-> PPG | Expansion fisiologica | **Proyeccion** |

### Frentes activos

| Frente | Funcion | Documento |
|---|---|---|
| Gate 6 AMT | Validacion downstream | [12_GATE_6_AMT/README.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/12_GATE_6_AMT/README.md) |
| Gate 8 | Conditioned projections | [15_GATE_8_CONDITIONED_PROJECTIONS/README.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/15_GATE_8_CONDITIONED_PROJECTIONS/README.md) |
| Gate 10 | Mechanism sweep audio-only | [17_GATE_10_MECHANISM_SWEEP/README.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/17_GATE_10_MECHANISM_SWEEP/README.md) |
| Escalon 2 | Frente principal (null mecanistico inicial cerrado; siguiente fase `S2-P3`) | [ESCALON_2/README.md](Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md) |
| Escalon 3 | Frente conceptual Lissajous | [ESCALON_3/README.md](Documents/01_FRENTES_ACTIVOS/ESCALON_3/README.md) |

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

| Brazo | `S` (multi-seed) |
|---|---:|
| `D0` | `75.2% +/- 2.3pp` |
| `a4r` | `80.7% +/- 1.9pp` |
| `d4-a4r` | `81.2% +/- 2.5pp` |
| `d4a4` | `84.1% +/- 2.3pp` |

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
| Gate 6 AMT | `Exp C` local cerro (`F1=0.157`); `Exp B` ya dejo un negativo util y `Exp A` queda reducido a screening pendiente |
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
