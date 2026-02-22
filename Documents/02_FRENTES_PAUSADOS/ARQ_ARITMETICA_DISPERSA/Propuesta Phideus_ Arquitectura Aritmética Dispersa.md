Esta es una propuesta formal estructurada desde mi rol como **Investigador Doctoral Senior en Análisis de Datos y Redes Neuronales**, basada en la revisión exhaustiva de la documentación técnica del proyecto Phideus/Rosetta (fases 0, 1, 2, reportes de fallo y literatura académica cargada).

# ---

**Propuesta de Investigación y Desarrollo: Proyecto Phideus v3.0**

**Título:** *Arquitecturas de Inductancia Estructural y Representación Dispersa para la Unificación Cross-Modal de Ratios.*

## **1\. Resumen Ejecutivo y Diagnóstico (El "Post-Mortem" de Rosetta 2.0)**

La revisión de los resultados de la **Fase 2 (Branch feature/extractor-v22)** arroja una conclusión científica ineludible:

* **El síntoma:** Aunque el *Extractor v2.2* logró una discriminabilidad estadística excelente en la entrada (Gap *pre-red* de 0.691, mejorando 172x respecto a v1), el modelo **RosetaVAE** colapsó esta ventaja casi en su totalidad en el espacio latente (Gap *post-red* de 0.007).  
* **La causa raíz:** Existe una **disonancia de representación**. Estamos alimentando histogramas densos (que borran la estructura relacional precisa de "quién se relaciona con quién") a una arquitectura (VAE convolucional/denso) que carece del **sesgo inductivo** (inductive bias) necesario para procesar relaciones multiplicativas (ratios). El modelo "hace trampa": aprende a reconstruir la "forma" estadística del histograma, pero no aprende la *estructura armónica* subyacente.

**Conclusión:** No podemos seguir optimizando el histograma. El cambio debe ser **arquitectónico y representacional**.

## ---

**2\. Hipótesis Refinada (H3-v3)**

*"La correspondencia cross-modal entre dominios oscilatorios (audio $\\leftrightarrow$ vibración) no reside en la distribución estadística de sus ratios (histogramas), sino en la **topología precisa de sus coincidencias proporcionales**. Para capturar esto, se requiere una arquitectura capaz de razonar explícitamente sobre operaciones aritméticas (multiplicación/división) y manejar esparcidad."*

## ---

**3\. La Propuesta: "Phideus-Prism"**

Propongo pivotar el proyecto hacia una nueva arquitectura híbrida que abandone el VAE de reconstrucción densa en favor de un enfoque de **Predicción de Estructura Latente (JEPA)** potenciado por **Unidades Aritméticas Neuronales**.

### **Pilar A: Nueva Representación de Entrada (Ratio Constellations)**

*Según lo sugerido en Fase\_3A.md, pero con una modificación crítica.*

Abandonamos los histogramas (imágenes borrosas de la realidad) y pasamos a **Tokens Dispersos (Tokens)**. Inspirado en algoritmos de *fingerprinting* (como Shazam) y *PointNet*.

* **Formato del Token:** $T\_i \= \[\\log(r), \\Delta t, \\text{weight}, \\text{harmonic\\\_index}\]$  
* **Justificación:** Esto preserva la identidad individual de cada relación. Una señal no es una "suma de ratios", es una "constelación de eventos proporcionales".

### **Pilar B: Arquitectura con Sesgo Inductivo Aritmético (El aporte Doctoral)**

*Basado en la literatura cargada: arXiv:2101.09530 (NALU), Domain Mixed Unit, y Rational Neural Networks.*

Las redes neuronales estándar (MLP con ReLU) fallan en extrapolar relaciones multiplicativas. Para que Phideus "entienda" ratios, sus neuronas deben saber multiplicar.

**Propuesta Técnica:** Integrar capas de **NMU (Neural Multiplication Units)** o **Product Units** dentro del encoder.

* En lugar de un encoder estándar Linear \-\> ReLU, utilizaremos un bloque híbrido:  
  $$h \= \\text{Linear}(x) \+ \\alpha \\cdot \\text{NMU}(x)$$  
* Esto permite a la red aprender tanto relaciones lineales (magnitudes) como multiplicativas (ratios) simultáneamente.  
* *Referencia:* Esto ataca directamente la limitación descrita en el paper *"Deep residual learning with product units"* que subiste.

### **Pilar C: Eliminación del Decoder (Enfoque JEPA)**

*Respondiendo al fallo de la Fase 2 en ROSETTA\_V22\_RESULTS.md.*

El decoder de reconstrucción es un lastre. Obliga al modelo a memorizar ruido para reconstruir la señal original.

* **Nuevo Objetivo:** **PRISM-JEPA (Joint Embedding Predictive Architecture).**  
* No reconstruimos la vibración desde el audio. Predecimos el *embedding* de la vibración desde el *embedding* del audio.  
* **Loss Function:** InfoNCE (Contrastive Loss) pura. Maximizar similitud entre pares alineados $(z\_A, z\_V)$ y minimizarla contra *shuffled* $(z\_A, z\_{V\\\_shuffled})$.

## ---

**4\. Roadmap de Ejecución (Plan de Acción)**

Este plan reemplaza las fases anteriores y se estructura en "Sprints" de investigación de 2 semanas.

### **Fase 3.1: Implementación de Constellations y Encoder Relacional**

* **Objetivo:** Crear el pipeline de datos dispersos.  
* **Acción:** Implementar extractor\_v3.py. Salida: Tensores de tamaño (Batch, N\_Tokens, Features).  
* **Modelo:** Transformer Encoder pequeño (tipo BERT-tiny) en lugar de CNN/MLP. El mecanismo de *Self-Attention* es ideal para relacionar tokens distantes en el tiempo ($\\Delta t$).

### **Fase 3.2: Inyección de Aritmética Neuronal (Validación H3)**

* **Objetivo:** Probar si las unidades aritméticas mejoran la captura de ratios.  
* **Experimento:** Comparar Transformer Standard vs Transformer \+ NMU Layers.  
* **Dataset:** Usar el synthetic\_ratio\_suite.py (que ya creaste en Fase 0\) para validar si el modelo puede aprender la función $f(x) \= 3/2 x$ (ratio perfecto) mejor que una red estándar.

### **Fase 3.3: Entrenamiento Cross-Modal (JEPA)**

* **Objetivo:** Entrenar el modelo final Phideus-Prism.  
* **Configuración:**  
  * Encoder Audio: Transformer \+ NMU.  
  * Encoder Vibración: Transformer \+ NMU.  
  * Loss: InfoNCE (con temperatura aprendible).  
  * Proyector: MLP simple para llevar ambos a un espacio común de 128 dimensiones.

## ---

**5\. Criterios de Éxito (KPIs Actualizados)**

Abandonamos el MSE de reconstrucción. Los nuevos KPIs son:

1. **Alignment Gap (Latent Space):** $\> 0.15$ (Distancia Coseno media entre pares alineados vs. shuffled). *Nota: En Fase 2 fue 0.007.*  
2. **Top-1 Retrieval Accuracy:** $\> 20\\%$ en un batch de 128 muestras (El azar es \~0.8%).  
3. **Ratio Disentanglement:** Capacidad de una *probe* lineal para predecir el ratio dominante (ej. "3:2") a partir del vector latente $z$.

## **6\. Conclusión y Recomendación Inmediata**

Doctor, el proyecto está estancado en un mínimo local porque la herramienta (VAE estándar) no es isomorfa al problema (relaciones de ratios).

**Mi recomendación inmediata:**

1. **Detener** cualquier entrenamiento con histogramas densos.  
2. Aprobar el inicio de la **Fase 3A (Constellations)** inmediatamente.  
3. Autorizar la implementación experimental de una capa **Domain Mixed Unit (DMU)** o **NMU** en PyTorch para ser insertada en la próxima iteración del modelo, alineándonos con el estado del arte que has recopilado en la bibliografía.

¿Procedo a redactar la especificación técnica para el extractor\_v3.py basado en Constellations?