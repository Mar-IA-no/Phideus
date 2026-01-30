# Propuesta Arquitectónica: Harmonic Ratio Alignment Network (HRAN)

**Fecha**: 2026-01-28
**Contexto**: Post-mortem Rosetta1 2.0 - Diseño de nueva arquitectura
**Estado**: PROPUESTA - Pendiente implementación

---

## Resumen Ejecutivo

Tras el fracaso de Rosetta1 2.0 (H3 no validada), este documento propone una arquitectura alternativa diseñada específicamente para los objetivos de Phideus: aprendizaje cross-modal basado en estructuras de ratios armónicos.

---

## Lo Que Aprendimos del Fracaso de Rosetta1 2.0

El modelo aprendió a generar el "histograma promedio" del dataset, ignorando completamente la correspondencia real entre pares. Esto revela un problema fundamental:

**El VAE con InfoNCE tiene un conflicto de objetivos:**
- La pérdida de reconstrucción domina (~0.4) vs InfoNCE (~0.01)
- El modelo encuentra un shortcut: ignorar z_shared para reconstrucción
- z_shared colapsa a estadísticas genéricas del dataset

---

## Diagnóstico: Por Qué Falló

```
                    PROBLEMA ACTUAL
    ┌─────────────────────────────────────────┐
    │                                         │
    │   Audio ──► Encoder ──► z_shared ◄──┐   │
    │                            │        │   │
    │                     [InfoNCE loss]  │   │
    │                            │        │   │
    │   Vibr ──► Encoder ──► z_shared ◄───┘   │
    │                                         │
    │   PERO: Decoder ignora z_shared         │
    │         porque z_private es suficiente  │
    │                                         │
    └─────────────────────────────────────────┘
```

El decoder puede reconstruir perfectamente usando solo información de dominio. **No necesita** la información cross-modal.

---

## Propuesta: Harmonic Ratio Alignment Network (HRAN)

### Principio Fundamental

> *"No comprimas señales. Extrae y alinea estructuras de ratios explícitas."*

### Arquitectura Propuesta

```
═══════════════════════════════════════════════════════════════════════════
                         HARMONIC RATIO ALIGNMENT NETWORK (HRAN)
═══════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ┌─────────────┐                                     ┌─────────────┐   │
│  │   AUDIO     │                                     │  VIBRACIÓN  │   │
│  │  (dominio)  │                                     │  (dominio)  │   │
│  └──────┬──────┘                                     └──────┬──────┘   │
│         │                                                   │          │
│         ▼                                                   ▼          │
│  ┌──────────────┐                                   ┌──────────────┐   │
│  │ STAGE 1:     │                                   │ STAGE 1:     │   │
│  │ Peak         │                                   │ Peak         │   │
│  │ Extraction   │                                   │ Extraction   │   │
│  │              │                                   │              │   │
│  │ • Detectar   │                                   │ • Detectar   │   │
│  │   picos      │                                   │   picos      │   │
│  │ • (pos, mag, │                                   │ • (pos, mag, │   │
│  │    width)    │                                   │    width)    │   │
│  └──────┬───────┘                                   └──────┬───────┘   │
│         │                                                   │          │
│         ▼                                                   ▼          │
│  ┌──────────────┐                                   ┌──────────────┐   │
│  │ STAGE 2:     │                                   │ STAGE 2:     │   │
│  │ Ratio Graph  │                                   │ Ratio Graph  │   │
│  │ Construction │                                   │ Construction │   │
│  │              │                                   │              │   │
│  │ Nodos: picos │                                   │ Nodos: picos │   │
│  │ Edges: ratios│                                   │ Edges: ratios│   │
│  │ entre picos  │                                   │ entre picos  │   │
│  └──────┬───────┘                                   └──────┬───────┘   │
│         │                                                   │          │
│         ▼                                                   ▼          │
│  ┌──────────────┐                                   ┌──────────────┐   │
│  │ STAGE 3:     │                                   │ STAGE 3:     │   │
│  │ Graph Neural │     ┌───────────────────┐        │ Graph Neural │   │
│  │ Network      │     │                   │        │ Network      │   │
│  │              │     │   RATIO SPACE     │        │              │   │
│  │ Message      │────►│                   │◄───────│ Message      │   │
│  │ Passing on   │     │  Dimensiones =    │        │ Passing on   │   │
│  │ ratio graph  │     │  ratios canónicos │        │ ratio graph  │   │
│  │              │     │  (1:2, 2:3, 3:4,  │        │              │   │
│  └──────────────┘     │   4:5, 5:6, ...)  │        └──────────────┘   │
│                       │                   │                           │
│                       │  ★ STRUCTURED ★   │                           │
│                       │  ★ INTERPRETABLE ★│                           │
│                       │                   │                           │
│                       └─────────┬─────────┘                           │
│                                 │                                      │
│                    ┌────────────┴────────────┐                        │
│                    │                         │                        │
│                    ▼                         ▼                        │
│            ┌──────────────┐          ┌──────────────┐                 │
│            │  CONTRASTIVE │          │   RATIO      │                 │
│            │  ALIGNMENT   │          │   PREDICTION │                 │
│            │              │          │              │                 │
│            │  Audio ≈ Vib │          │  "¿Qué ratio │                 │
│            │  para pares  │          │   domina?"   │                 │
│            │  correctos   │          │              │                 │
│            └──────────────┘          └──────────────┘                 │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Componentes Detallados

### Stage 1: Peak Extraction (Determinístico)

```python
class PeakExtractor:
    """
    Convierte histograma de ratios → conjunto de picos.

    Input:  histogram [128 bins]
    Output: peaks [(pos₁, mag₁, width₁), (pos₂, mag₂, width₂), ...]

    Esto PRESERVA la información de ratios que el histograma
    promedia y pierde.
    """

    def extract(self, histogram):
        # 1. Encontrar máximos locales
        peaks = find_peaks(histogram, prominence=0.1)

        # 2. Para cada pico, extraer:
        #    - Posición (valor del ratio)
        #    - Magnitud (altura del pico)
        #    - Ancho (spread del pico)

        # 3. Retornar como conjunto ordenado por magnitud
        return peak_set  # Variable size, típicamente 5-15 picos
```

**Por qué esto importa:** El histograma de 128 bins pierde información sobre qué picos específicos están presentes. Dos señales con el mismo histograma promedio pueden tener picos MUY diferentes.

### Stage 2: Ratio Graph Construction

```python
class RatioGraphBuilder:
    """
    Construye grafo donde:
    - Nodos = picos detectados
    - Edges = ratios entre picos

    Ejemplo: Si hay picos en 1.5 y 2.0:
    - Nodo A: (pos=1.5, mag=0.8)
    - Nodo B: (pos=2.0, mag=0.6)
    - Edge A→B: ratio = 2.0/1.5 = 1.333... ≈ 4:3
    """

    def build_graph(self, peaks):
        nodes = peaks
        edges = []

        for i, peak_i in enumerate(peaks):
            for j, peak_j in enumerate(peaks):
                if i != j:
                    ratio = peak_j.pos / peak_i.pos
                    # Clasificar ratio a intervalo musical más cercano
                    canonical_ratio = snap_to_canonical(ratio)
                    edges.append((i, j, canonical_ratio))

        return Graph(nodes, edges)
```

**Insight clave:** Los ratios armónicos son RELACIONES entre frecuencias. Un grafo captura esto naturalmente.

### Stage 3: Graph Neural Network con Ratio-Aware Message Passing

```python
class RatioGNN(nn.Module):
    """
    GNN especializada para grafos de ratios armónicos.

    Innovación: El message passing está condicionado por el
    tipo de ratio del edge (octava, quinta, cuarta, etc.)
    """

    def __init__(self, hidden_dim=64, num_layers=3):
        self.edge_embeddings = nn.Embedding(
            num_canonical_ratios,  # ~12 ratios principales
            hidden_dim
        )

        self.message_nets = nn.ModuleDict({
            '2:1': MLP(hidden_dim * 2, hidden_dim),  # Octava
            '3:2': MLP(hidden_dim * 2, hidden_dim),  # Quinta
            '4:3': MLP(hidden_dim * 2, hidden_dim),  # Cuarta
            '5:4': MLP(hidden_dim * 2, hidden_dim),  # Tercera mayor
            # ... etc
        })

    def forward(self, graph):
        # Message passing consciente del tipo de ratio
        for layer in range(self.num_layers):
            for edge in graph.edges:
                ratio_type = edge.canonical_ratio
                message = self.message_nets[ratio_type](
                    concat(graph.nodes[edge.src], graph.nodes[edge.dst])
                )
                graph.nodes[edge.dst] += message

        # Pooling a representación fija
        return global_attention_pool(graph.nodes)
```

### Stage 4: Structured Ratio Space

```python
class RatioSpace(nn.Module):
    """
    Espacio latente ESTRUCTURADO donde cada dimensión
    corresponde a un ratio canónico específico.

    Dimensión 0: presencia de octava (2:1)
    Dimensión 1: presencia de quinta (3:2)
    Dimensión 2: presencia de cuarta (4:3)
    ...

    Esto hace el espacio INTERPRETABLE y COMPARABLE
    entre dominios.
    """

    CANONICAL_RATIOS = [
        (2, 1),   # Octava
        (3, 2),   # Quinta perfecta
        (4, 3),   # Cuarta perfecta
        (5, 4),   # Tercera mayor
        (6, 5),   # Tercera menor
        (5, 3),   # Sexta mayor
        (8, 5),   # Sexta menor
        (9, 8),   # Tono mayor
        (10, 9),  # Tono menor
        (16, 15), # Semitono
        (7, 4),   # Séptima armónica
        (7, 6),   # Séptima menor
    ]

    def __init__(self):
        self.ratio_dim = len(self.CANONICAL_RATIOS)
        self.projector = nn.Linear(gnn_output_dim, self.ratio_dim)

    def forward(self, gnn_output):
        # Proyectar a espacio de ratios con activación sparse
        ratio_activations = torch.sigmoid(self.projector(gnn_output))
        return ratio_activations  # [batch, 12]
```

**Ventaja crítica:** Si audio y vibración del mismo fenómeno físico activan los mismos ratios (ej: ambos muestran 2:1 y 3:2 prominentes), el alignment es **automático** y **verificable**.

---

## Función de Pérdida: Sin Reconstrucción

```python
class HRANLoss(nn.Module):
    """
    CRÍTICO: No hay pérdida de reconstrucción.

    Esto evita el shortcut que mató a Rosetta1 2.0.
    """

    def forward(self, audio_ratios, vib_ratios, labels):

        # 1. Contrastive Loss (principal)
        #    Pares correctos deben tener ratios similares
        #    Pares incorrectos deben ser diferentes
        contrastive = self.nt_xent_loss(audio_ratios, vib_ratios, labels)

        # 2. Ratio Sparsity Loss
        #    La mayoría de ratios deben estar inactivos
        #    (una señal real tiene ~3-5 ratios dominantes, no 12)
        sparsity = torch.mean(torch.abs(audio_ratios)) + \
                   torch.mean(torch.abs(vib_ratios))

        # 3. Ratio Consistency Loss
        #    Picos detectados deben corresponder a ratios activados
        consistency = self.peak_ratio_consistency(...)

        return contrastive + 0.1 * sparsity + 0.1 * consistency
```

---

## Por Qué Esta Arquitectura Debería Funcionar

### 1. Elimina el Shortcut de Reconstrucción
Sin decoder, no hay forma de "ignorar" el espacio compartido.

### 2. Representación Explícita de Ratios
El histograma pierde información. Los picos y sus relaciones la preservan.

### 3. Espacio Latente Interpretable
Podemos VERIFICAR si audio y vibración activan los mismos ratios. No es una caja negra.

### 4. Invarianza Correcta
- GNN es invariante a permutación de picos ✓
- El ratio 3:2 es el mismo en audio y vibración ✓
- Escala absoluta de frecuencia es ignorada ✓

### 5. Extensible a Multi-Dominio
Agregar un nuevo dominio (ej: imagen Lissajous) solo requiere:
1. Nuevo Peak Extractor para ese dominio
2. El resto de la red es compartido

```
Audio ──► PeakExtract ──► RatioGraph ──► GNN ──┐
                                               │
Vibr ──► PeakExtract ──► RatioGraph ──► GNN ──┼──► RATIO SPACE
                                               │
Image ──► PeakExtract ──► RatioGraph ──► GNN ──┘
          (nuevo)         (compartido)
```

---

## Validación: Cómo Sabríamos Que Funciona

### Test 1: Ratio Activation Agreement
```python
# Para un par (audio, vib) del mismo fenómeno:
audio_ratios = model.encode(audio)  # [0.9, 0.7, 0.1, 0.0, ...]
vib_ratios = model.encode(vib)      # [0.85, 0.65, 0.15, 0.0, ...]

# Deben activar los MISMOS ratios
agreement = cosine_similarity(audio_ratios, vib_ratios)
# Esperado: > 0.8 para pares correctos, < 0.3 para incorrectos
```

### Test 2: Synthetic Ground Truth
```python
# Generar señal sintética con ratios CONOCIDOS
synth_audio = generate_with_ratios([2/1, 3/2])  # Octava + quinta
synth_vib = generate_with_ratios([2/1, 3/2])    # Mismos ratios

audio_ratios = model.encode(synth_audio)
# Dimensiones 0 (octava) y 1 (quinta) deben estar activas
# El resto deben estar ~0
```

### Test 3: Shuffled vs Aligned (el test que falló Rosetta1)
Con HRAN, los pares shuffled tendrán ratios DIFERENTES, porque provienen de condiciones físicas diferentes.

---

## Comparación con Arquitecturas Existentes

| Arquitectura | Problema para Phideus |
|--------------|----------------------|
| **VAE** | Reconstrucción domina, ignora alignment |
| **CLIP** | Genérico, no explota estructura de ratios |
| **CCA** | Linear, no captura relaciones complejas |
| **Standard GNN** | No tiene semántica de ratios musicales |
| **Transformer** | Overkill, no explota invarianzas conocidas |

**HRAN** está diseñado específicamente para el problema de Phideus:
- Estructura de ratios armónicos
- Cross-modal alignment
- Interpretabilidad
- Extensibilidad multi-dominio

---

## Plan de Implementación Sugerido

### Fase 1: Validación del Concepto
1. Implementar PeakExtractor simple
2. Implementar RatioGraph
3. GNN básica (sin las optimizaciones de ratio-aware)
4. Probar en datos sintéticos con ground truth conocido

### Fase 2: Refinamiento
1. Ratio-aware message passing
2. Structured ratio space
3. Función de pérdida completa
4. Probar en UOEMD

### Fase 3: Validación Rigurosa
1. Todos los controles negativos de Rosetta1 2.0
2. Ablations
3. Comparación con baseline VAE

### Fase 4: Multi-Dominio (futuro)
1. Agregar dominio visual (Lissajous)
2. Validar transferencia three-way

---

## Conclusión

El fracaso de Rosetta1 2.0 no significa que la hipótesis H3 sea falsa. Significa que el **enfoque VAE+InfoNCE** no es el correcto para este problema.

HRAN aborda las causas raíz:

| Problema Rosetta1 2.0 | Solución HRAN |
|-----------------------|---------------|
| Reconstrucción como shortcut | Sin reconstrucción |
| Histograma pierde información | Picos explícitos |
| Latente genérico | Espacio de ratios estructurado |
| Caja negra | Interpretable y verificable |

La teoría de Harmonic Information sigue siendo válida. Solo necesitamos una arquitectura que la respete.

---

*Documento generado por Claude Code - 2026-01-28*
*Post-análisis de Rosetta1 2.0 (H3 NO VALIDADA)*
