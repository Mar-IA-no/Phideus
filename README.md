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

**Phideus** es un programa de investigacion enmarcado en **Harmonic Information Theory**. Su pregunta central es si ciertas relaciones de frecuencia, proporciones y estructuras armonicas pueden funcionar como invariantes privilegiadas para organizar informacion entre modalidades distintas de un mismo fenomeno fisico.

La idea no es tratar a una red neuronal como espejo ontologico del mundo, ni reducir todo a una carrera de benchmarks. Phideus usa arquitecturas aprendidas como instrumentos experimentales: si una estructura relacional mejora de forma causal, robusta y transferible la alineacion cross-modal entre sensores distintos, eso pasa a ser evidencia de que esa estructura captura algo real del fenomeno medido.

En su forma actual, el programa distingue con claridad dos planos. **Escalon 1** establecio que la inyeccion de descriptores puede reorganizar geometricamente el espacio latente y mejorar retrieval de manera fuerte sobre **Audio <-> MIDI**. **Escalon 2** lleva esa intuicion a un frente mas exigente, **Speech <-> EGG**, donde la hipotesis se formula de manera mas estricta desde la **armonia natural**: ratios lineales del oscilador, estructura armonica intra-frame y controles espectrales comparativos.

En otras palabras: Phideus no pregunta solo si "un descriptor ayuda". Pregunta que clase de estructura ayuda, por que ayuda, y si esa ayuda sobrevive cuando cambia el sensor, la modalidad o el dominio.

> [!IMPORTANT]
> **Corte actual (2026-03-08):** Escalon 1 ya tiene un cierre cientifico robusto sobre MAESTRO Audio<->MIDI; Escalon 2 ya cerro su apertura de datos, baseline lineal y baseline neural inicial sobre French Lombard Speech<->EGG. El foco vigente del programa esta en la capa descriptor-guided de Escalon 2, mientras Gate 6 AMT y Gate 8 conditioned projections siguen como lineas activas complementarias.

---

## Programa actual

| Frente | Dominio | Funcion en el programa | Estado actual |
|---|---|---|---|
| **Escalon 1** | Audio <-> MIDI | Banco de validacion descriptor-guided y geometria cross-modal | **Cerrado** como frente principal |
| **Gate 6 AMT** | Audio -> transcripcion | Validacion downstream de la senal descriptor-guided | **Activo** |
| **Gate 8** | Audio <-> MIDI | Auditoria de proyecciones y preservacion de informacion descriptorial | **Activo** |
| **Escalon 2** | Speech <-> EGG | Test de generalizacion y armonia natural sobre dos sensores del mismo fenomeno vocal | **Activo / foco principal** |
| **Escalon 3** | ECG <-> PPG | Expansion prevista a otro dominio fisiologico | **Proyeccion** |

Lo importante de esta estructura es que cada frente cumple un papel distinto. Escalon 1 fija la evidencia de referencia. Gate 6 pregunta si esa ventaja sobrevive fuera del retrieval. Gate 8 pregunta donde se preserva o se pierde la informacion descriptor-guided. Escalon 2, en cambio, es la primera arena donde la tesis fuerte del proyecto puede probarse con una taxonomia descriptorial explicitamente natural.

---

## Resultados de referencia

### Escalon 1 — Audio <-> MIDI

La referencia canonica del programa sigue siendo el bloque descriptor-guided sobre MAESTRO. La lectura vigente es que la mejora principal opera como **ventaja geometrica del espacio latente**, no como simple enriquecimiento de decodificabilidad local.

| Modelo / brazo | Metrica canonica `S` | Lectura |
|---|---:|---|
| `D0` | `75.2% +/- 2.3pp` | Baseline sin descriptor |
| `a4r` | `80.7% +/- 1.9pp` | Descriptor audio con reverse cross-attention |
| `d4-a4r` | `81.2% +/- 2.5pp` | Variante mixta descriptor-guided |
| `d4a4` | `84.1% +/- 2.3pp` | Referencia robusta del frente |

En el cierre causal del bloque:
- el contenido descriptorial real explica la mejora sobre ablaciones parameter-matched;
- `A4/A4r` aparecen como la senal causal dominante del cierre;
- la alineacion representacional cross-modal sube de forma marcada;
- y la linea generativa no lineal no muestra una ventaja descriptor-guided equivalente.

### Escalon 2 — Speech <-> EGG

El nuevo frente ya tiene piso empirico propio:

| Capa | Resultado de referencia | Significado |
|---|---:|---|
| Baseline lineal `CCA` | `S=64.4%` | La senal cross-modal ya existe antes del primer encoder neural |
| Baseline neural `D0` | `S=77.8% @ ep25`, `CI=[72.0%, 80.8%]` | El frente ya esta operativo a nivel descriptor-ready |

Esto cambia el estatuto de Escalon 2: ya no es un dominio "prometedor", sino un frente en el que los descriptores naturales se van a comparar contra un baseline neural serio y contra controles espectrales explicitos.

### Lineas complementarias

| Frente | Corte actual |
|---|---|
| Gate 6 AMT | `Exp 0` completo; `Exp C` local `a4r` cerro con `best_F1=0.1570 @ ep50` |
| Gate 8 | `a4r-ctrl = 79.2%`, `a4r-pcm = 80.0%` |
| Gate 7.1a | `D0_mert330m_frozen = 75.0%`, esencialmente igual a `D0_lite = 75.2%` |

---

## Como entrar al repo

Si es tu primera vez en Phideus, esta es la ruta mas corta y menos ruidosa:

| Si queres... | Empezar por... | Para que sirve |
|---|---|---|
| Entender que es Phideus y que tipo de conocimiento intenta producir | [MARCO_EPISTEMOLOGICO_PHIDEUS.md](MARCO_EPISTEMOLOGICO_PHIDEUS.md) | Posicion epistemologica del programa |
| Ver el estado canonicamente vigente del proyecto | [Proyecto_Estado_Actual.md](Documents/00_TRONCAL/Proyecto_Estado_Actual.md) | Resumen ejecutivo y frentes abiertos |
| Ver la estructura global del programa | [HANDOFF.md](Documents/00_TRONCAL/HANDOFF.md) | Continuidad operativa y mapa sintetico |
| Ir directo al frente musical consolidado | [ROADMAP_BIAS_CONTROL.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md) | Estado de Gates, criterios y decisiones vigentes |
| Ir directo al frente vocal actual | [ESCALON_2/README.md](Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md) | Estado canonicamente vigente de Speech<->EGG |
| Entender la historia de los descriptores | [CATALOGO_NARRATIVO_DESCRIPTORES_RATIOS_PHIDEUS.md](Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/CATALOGO_NARRATIVO_DESCRIPTORES_RATIOS_PHIDEUS.md) | Taxonomia historica y epistemica |
| Ver la historia larga del proyecto | [INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md](Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md) | Genealogia completa del programa |
| Ver skills compartidas del proyecto | [Documents/Skills/README.md](Documents/Skills/README.md) | Skills publicas reutilizables |

---

## Visualizaciones, skills y documentacion viva

Phideus no vive solo en scripts de entrenamiento. El repo mantiene tres capas publicas de lectura que conviene conocer desde el principio.

### Visualizaciones

Las arquitecturas y lineas principales del programa tienen exploraciones 3D interactivas en:

**[altermundi.github.io/Phideus](https://altermundi.github.io/Phideus/)**

Incluyen, entre otras:
- baseline cross-modal;
- variantes `reverse cross-attention`;
- configuraciones duales de Escalon 1;
- lineas historicas como Roseta, JEPA-lite y Constellation Tokens.

### Skills compartidas

El repositorio publica skills reutilizables en:

**[Documents/Skills/README.md](Documents/Skills/README.md)**

Hoy esa capa compartida esta concentrada en operacion HPC/SLURM, validacion pre-submit y lecciones reutilizables de UNC.

### Documentacion viva

La documentacion del repo esta organizada para cumplir funciones distintas:

- `Documents/00_TRONCAL/` reune estado ejecutivo, handoff e indices globales;
- `Documents/01_FRENTES_ACTIVOS/` contiene la documentacion operativa de cada frente vivo;
- `Documents/04_TRANSVERSAL/` concentra teoria, fundamentos, historia y materiales de lectura de nivel programa.

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

Protocolo canonico de Escalon 1:
- `pool=256`
- `queries=500`
- `seed=42`

---

<!-- BELOW THE FOLD -->

<details>
<summary><strong>Roadmap del programa</strong></summary>

### TripleScaloneta

| Escalon | Dominio | Rol | Estado |
|---|---|---|---|
| Escalon 1 | MAESTRO Audio <-> MIDI | Validacion descriptor-guided y cierre cientifico del primer banco de pruebas | **Cerrado** |
| Escalon 2 | Speech <-> EGG | Generalizacion + armonia natural en dos sensores del mismo fenomeno vocal | **Activo** |
| Escalon 3 | ECG <-> PPG | Expansion fisiologica prevista | **Proyeccion** |

### Frentes principales hoy

| Frente | Funcion | Documento |
|---|---|---|
| Gate 6 AMT | Validacion downstream | [12_GATE_6_AMT/README.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/12_GATE_6_AMT/README.md) |
| Gate 8 | Auditoria de proyecciones | [15_GATE_8_CONDITIONED_PROJECTIONS/README.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/15_GATE_8_CONDITIONED_PROJECTIONS/README.md) |
| Escalon 2 | Frente principal activo | [ESCALON_2/README.md](Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md) |

### Roadmaps canonicos

- [ROADMAP_BIAS_CONTROL.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md)
- [ROADMAP_ESCALON_2.md](Documents/01_FRENTES_ACTIVOS/ESCALON_2/ROADMAP_ESCALON_2.md)

</details>

<details>
<summary><strong>Arquitectura y familias descriptoriales</strong></summary>

### Idea arquitectonica general

Phideus trabaja con configuraciones cross-modales contrastivas. La forma concreta cambia por frente, pero la estructura general es estable:

```text
modalidad A -> encoder -> projection -> embedding
modalidad B -> encoder -> projection -> embedding
                    \      VICReg      /
```

La investigacion no se reduce a "que encoder rinde mas". Tambien importa:
- donde entra un descriptor;
- si entra como input augmentation, atencion o modulacion;
- que geometria induce;
- y si esa geometria sobrevive a proyecciones y tareas downstream.

### Escalon 1: familias relevantes

| Familia | Ejemplos | Rol |
|---|---|---|
| Control | `D0` | baseline sin descriptor |
| MIDI local | `D4` | relaciones locales del lado MIDI |
| Audio espectral local | `A4`, `A4r` | dinamica espectral local del lado audio |
| Dual | `d4a4`, `d4-a4r` | combinaciones descriptor-guided de mayor rendimiento |

### Escalon 2: taxonomia actual

| Familia | Descriptor | Rol epistemologico |
|---|---|---|
| Temporal natural | `V4-lin` | dinamica lineal del oscilador |
| Temporal comparativa | `V4-log` | control perceptual/logaritmico |
| Armonica natural | `H-series` | estructura armonica intra-frame |
| Control espectral | `A4-16k` | dinamica espectral local no-ratio |

Para la lectura fuerte de estas familias, ver:
- [MARCO_EPISTEMOLOGICO_PHIDEUS.md](MARCO_EPISTEMOLOGICO_PHIDEUS.md)
- [plan_rectificacion_armonia_natural.md](Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/plan_rectificacion_armonia_natural.md)

</details>

<details>
<summary><strong>Linea experimental consolidada</strong></summary>

### Escalon 1 — referencia actual

| Resultado | Valor |
|---|---:|
| `D0` multi-seed | `75.2% +/- 2.3pp` |
| `a4r` multi-seed | `80.7% +/- 1.9pp` |
| `d4-a4r` multi-seed | `81.2% +/- 2.5pp` |
| `d4a4` multi-seed | `84.1% +/- 2.3pp` |

### Gate 6 / Gate 8 / Gate 7.1

| Frente | Corte vigente |
|---|---|
| Gate 6 AMT | `best_F1=0.1570 @ ep50` para `a4r` local en `Exp C` |
| Gate 8 | `a4r-ctrl = 79.2%`, `a4r-pcm = 80.0%` |
| Gate 7.1a | `D0_mert330m_frozen = 75.0%`, sin mejora clara sobre `D0_lite` |

### Escalon 2 — referencias iniciales

| Capa | Resultado |
|---|---:|
| Baseline lineal `CCA` | `64.4%` |
| Baseline neural `D0` | `77.8% @ ep25`, `CI=[72.0%, 80.8%]` |

</details>

<details>
<summary><strong>Documentacion clave y estructura del repo</strong></summary>

### Documentos principales

| Documento | Funcion |
|---|---|
| [Proyecto_Estado_Actual.md](Documents/00_TRONCAL/Proyecto_Estado_Actual.md) | Estado ejecutivo del programa |
| [HANDOFF.md](Documents/00_TRONCAL/HANDOFF.md) | Continuidad operativa |
| [INDICE_DOCUMENTACION.md](Documents/00_TRONCAL/INDICE_DOCUMENTACION.md) | Mapa global de documentacion |
| [ROADMAP_BIAS_CONTROL.md](Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md) | Roadmap del frente musical |
| [ESCALON_2/README.md](Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md) | Estado del frente vocal |
| [Documents/Skills/README.md](Documents/Skills/README.md) | Skills compartidas |

### Estructura general

```text
Phideus/
├── src/                         # Modulos del proyecto
├── experiments/                 # Training, evaluacion y utilidades experimentales
├── Documents/
│   ├── 00_TRONCAL/              # Estado ejecutivo, handoff, indices
│   ├── 01_FRENTES_ACTIVOS/      # Frentes vivos
│   ├── 02_FRENTES_PAUSADOS/     # Frentes pausados
│   ├── 03_FRENTES_CERRADOS/     # Frentes cerrados
│   └── 04_TRANSVERSAL/          # Teoria, fundamentos, historia, visuales
├── viz/                         # Visualizaciones interactivas
├── data/                        # Datasets y outputs pesados (no versionados)
└── config/                      # Configuraciones
```

</details>

---

> *"El bosque ya canta. Nuestra tarea es entender su afinacion."*

**Licencia**: MIT — ver [LICENSE.md](LICENSE.md)
