# **INFORME v2 — Revisionismo de Extracción de Ratios (Rosetta / Phideus)**

## **0\) Propósito y alcance**

Este informe v2 reemplaza la versión anterior incorporando ajustes “SOTA / reviewer-proof” en tres frentes:

1. **Protocolo único de evaluación** (definiciones congeladas, sin ambigüedad).  
2. **Criterios GO/NO-GO calibrables** (no umbrales arbitrarios).  
3. **Extractor v2.2 con estabilidad temporal como núcleo \+ controles anti-shortcut obligatorios**.

El objetivo no es “validar H3” en términos absolutos, sino **producir evidencia reproducible de que el pipeline es identificable** y que el aprendizaje depende del pareo correcto (aligned ≫ shuffled) bajo un protocolo fijo. El wording correcto es **“H3 supported under protocol P”**, no “validada”.

---

## **1\) Diagnóstico (causa raíz) resumido**

El sistema falló por **no identificabilidad** inducida por la representación:

* Exceso combinatorio (picos por frame → pares → ratios) produce **distribuciones demasiado densas / cercanas a uniformes**.  
* Eso colapsa la variabilidad inter-archivo / inter-condición y habilita “soluciones triviales” (p. ej. outputs promedio) que no dependen del pairing correcto.  
* Resultado típico del fallo: **aligned ≈ shuffled** y métricas de reconstrucción o correlación “buenas” aun con entradas permutadas o latentes aleatorios.

**Consecuencia:** antes de iterar arquitectura (VAE/JEPA/GNN/etc.), hay que **reconstruir discriminabilidad** del descriptor y blindar evaluación.

---

## **2\) Protocolo único de evaluación (P0) — congelado**

### **2.1 Unidad de ejemplo (definición innegociable)**

* **Ejemplo \= ventana temporal** (no frame aislado), con solapamiento fijo.  
* Para cada ventana se genera un descriptor (histograma o tokens) por modalidad.  
* Cada ejemplo tiene un `pair_id` único (audio/vib del mismo instante y mismo archivo).

### **2.2 Split (anti-leakage)**

* **Split por archivo / ejecución / motor** (GroupSplit).  
* Prohibido que ventanas del mismo archivo aparezcan en train y test.

### **2.3 Tareas y métricas oficiales (solo estas se reportan como “principales”)**

**Tarea A — Cross-modal retrieval (principal)**

* Query: audio → candidatos: vibración (y viceversa).  
* Métricas: Recall@1, Recall@5, Recall@10, MRR.  
* Reporte obligatorio: global \+ intra-condición \+ intra-archivo (harder).

**Tarea B — Dependencia del pairing (test definitorio)**

* Repetir Retrieval con pairing permutado (**shuffled**).  
* Criterio: desempeño debe colapsar hacia random (o degradar fuertemente).

**Tarea C — Regime probing (secundaria)**

* Linear probe (logistic) sobre embedding: Healthy vs Fault \+ multiclass (si aplica).  
* Métricas: AUC (binario), accuracy balanced (multiclass), silhouette (en embedding, no en UMAP).

Nota: “cross-recon” sólo se reporta si existe un módulo predictivo explícito en embedding (JEPA/predictor). Si hay decoders, “recon” pasa a **auxiliar** y **debe** acompañarse por controles anti-shortcut (ver 5.2).

### **2.4 Reporte estadístico (mínimo)**

* 5 seeds (mínimo 3 si la infraestructura es limitada).  
* Promedio ± desvío.  
* Para decisiones GO/NO-GO: incluir al menos 1 intervalo (bootstrap sobre ejemplos o seeds).

---

## **3\) Extractor v2.2 (mínimo viable, pero sólido)**

### **3.1 Objetivo del extractor v2.2**

Generar un descriptor de ratios que sea:

* **escaso** (sparse),  
* **estable temporalmente**,  
* **no dominado por ubiquidad** (no igual a la media del dataset),  
* **sensible al pairing** (aligned ≫ shuffled ya a nivel descriptor o con un encoder mínimo).

### **3.2 Diseño v2.2 (histograma sparse compatible con pipeline actual)**

#### **Paso 1 — Peak picking robusto (por frame interno)**

* Normalización local por banda (o z-score por frame) antes de buscar picos.  
* Score de pico \= prominencia × amplitud (no sólo amplitud).  
* Selección **Top-K** picos (K pequeño: 8–16).

#### **Paso 2 — Estabilidad temporal (CORE, no opcional)**

En una ventana W de frames (p. ej. 0.5–1.5 s):

* Un pico “sobrevive” si aparece en ≥ p% de frames con tolerancia en frecuencia.  
* Sólo sobreviven picos estables → se usan para ratios.  
* Si sobreviven menos de Kmin, relajar umbral de prominencia en esa ventana (fallback controlado).

#### **Paso 3 — Ratios con submuestreo controlado**

En vez de todos-contra-todos:

* calcular ratios sólo entre picos estables.  
* opción A: todos los pares dentro de K (si K\<=12, sigue siendo razonable).  
* opción B: anchor-target (M vecinos por ancla) si K sube o hay demasiado ruido.

#### **Paso 4 — Binning no uniforme (warped bins)**

Mantener ratio lineal, pero bins más densos cerca de 1–2 y más anchos hacia 10:

* reduce “sobre-dispersión” donde hay pocos eventos  
* evita perder resolución donde se concentra la física útil

#### **Paso 5 — Reponderación anti-ubiquidad (TF-IDF de bins) (recomendado)**

* TF: masa del bin en la ventana  
* IDF: (\\log(\\frac{N}{df\_b})) con `df_b` computado sobre train  
* objetivo: que bins “siempre presentes” no dominen

Resultado: sigue saliendo un tensor similar a (\[T,B,C\]) (o (\[B,C\]) por ventana), pero con mucha más señal.

---

## **4\) Calibración de umbrales (evitar arbitrariedad)**

En vez de “entropía \< 85%” o “similitud \< 0.92” fijos, el v2 usa calibración:

### **4.1 Baselines para calibrar (obligatorios)**

Para cada métrica pre-red y post-red, medir distribuciones bajo:

* **Random baseline**: descriptor con bins permutados o ruido con misma energía global.  
* **Shuffled pairing baseline**: audio emparejado con vib aleatorio.  
* **Mean baseline**: predictor que siempre devuelve media (para tareas predictivas).

### **4.2 Umbrales por percentil / margen**

Ejemplos (definir por datos):

* “Descriptor discriminativo” si la similitud inter-archivo del descriptor cae por debajo del **P50 del baseline random** (o por un margen del 20–30%).  
* “Pairing dependency” si Recall@1 aligned supera en ≥ **X sigmas** a Recall@1 shuffled, o si `aligned/shuffled` \> 5×.

Esto vuelve el criterio portable a nuevos datasets/domínios (Phideus real).

---

## **5\) Controles anti-shortcut (obligatorios)**

### **5.1 Para retrieval (siempre)**

* aligned vs shuffled (core)  
* intra-condición retrieval (hard negatives)  
* intra-archivo retrieval (harder)  
* reporte de random chance exacto según N candidatos

### **5.2 Para cualquier tarea “predictiva” (si existe predictor/decoder)**

Obligatorio reportar:

* **variance test**: var(pred) vs var(real)  
* **mean predictor baseline**: pred=mean\_train  
* **random input / random z**: si performance no cae, hay shortcut  
* **shuffled input**: debe caer fuerte

**Regla:** Si el modelo rinde parecido con inputs aleatorios, el experimento es NO-GO aunque el número “sea alto”.

---

## **6\) Plan experimental v2.2 (con barrido, no “un set de defaults”)**

### **6.1 Sweep mínimo del extractor**

Evaluar 12–20 configuraciones (barato) sobre train/val:

* K ∈ {8, 12, 16}  
* prominencia τ ∈ {0.1, 0.2, 0.3} (en escala normalizada)  
* estabilidad p ∈ {0.5, 0.7}  
* tolerancia freq ∈ {Δf pequeño, Δf medio} (en Hz, no bins)  
* bins warped: on/off  
* TF-IDF: on/off

Elegir configuración por **frontera de Pareto**:

* maximiza gap aligned-shuffled (pre-red con encoder mínimo)  
* minimiza similitud con media global  
* mantiene estabilidad temporal aceptable

### **6.2 Encoder mínimo para medir “aprendibilidad”**

Antes de redes grandes, usar un encoder mínimo:

* MLP pequeño o linear probe sobre descriptor → embedding 64D  
* entrenar contrastivo rápido o directamente clasificación de pairing  
* si esto no mejora, no tiene sentido escalar.

---

## **7\) Criterios GO/NO-GO (v2) — realistas y defensibles**

### **GO-1 (descriptor “aprendible”)**

Con extractor elegido por sweep:

* retrieval con encoder mínimo: aligned Recall@1 **≫** shuffled Recall@1 (factor ≥ 5× o diferencia estadísticamente clara)  
* similitud descriptor-media global cae sustancialmente vs baseline anterior  
* estabilidad temporal: la mayoría de ventanas tiene ≥Kmin picos estables (sin “colapsar” a vacío)

### **GO-2 (Rosetta v3 backbone listo)**

Con encoder serio (dual-encoder contrastivo / PRISM-JEPA):

* Recall@1 en test: ≥ 10× random (y estable en seeds)  
* Shuffled cae cerca de random  
* intra-condición Recall@1 no colapsa (demuestra fine-grained)

### **NO-GO**

Si cualquiera de estos ocurre:

* aligned ≈ shuffled (gap pequeño)  
* performance similar con inputs aleatorios / mean predictor  
* leakage detectado (splits incorrectos)

---

## **8\) Interpretación científica correcta (cómo reportar sin inflar claims)**

* Si GO-2 se cumple: **“H3 supported under protocol P0 \+ extractor v2.2”**  
  (no “validada universalmente”)  
* Para escalar a Phideus multi-dominio, el próximo paso es repetir el mismo protocolo con un tercer dominio o con otro dataset mecánico, sin tocar evaluación.

---

## **9\) Entregables (lo que el equipo debe producir sí o sí)**

1. `extractor_v2_2.py` con flags reproducibles y salida versionada (incluye picos estables y bins warped/TF-IDF).  
2. `sweep_extractor.py` que produce `sweep_results.json` \+ ranking Pareto.  
3. `protocol_P0.md` (este documento condensado como spec de evaluación).  
4. `eval_retrieval.py` (aligned/shuffled \+ hard negatives) \+ `eval_regime_probe.py`.  
5. Reporte estándar `report_run.md` con: seeds, splits, métricas, baselines, controles anti-shortcut.

---

## **10\) Nota crítica final**

Este v2 insiste en un principio: **no hay “modelo salvador” si el descriptor no es identificable**. El esfuerzo principal debe concentrarse en:

* representación escasa y estable,  
* calibración por baselines,  
* y dependencia fuerte del pairing (aligned≫shuffled).

Recién después vale implementar la arquitectura Rosetta v3 “SOTA” (dual-encoder \+ JEPA/predictor), porque entonces el entrenamiento deja de ser una lotería.

---

Si querés, puedo devolverte esto también como **SPEC Analizador 6.0** (campos de salida exactos \+ JSON schema \+ tests unitarios \+ checklist de auditoría automática) para que el equipo lo implemente sin ambigüedades.

