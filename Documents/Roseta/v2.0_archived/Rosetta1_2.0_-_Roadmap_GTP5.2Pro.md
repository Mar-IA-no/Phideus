According to a document from **enero de 2026**, el estado real de Rosetta1 hoy es este: **hay alineamiento cross-modal parcial**, la **cross‑reconstruction es moderada** (corr \~0.57–0.70), pero la **factorización z\_shared / z\_private falló por colapso de z\_private** (var(mu) \~0.0002 y audio≈vib en private), y además hay **riesgo metodológico** por split por frames (posible leakage) y métricas de separación de regímenes contradictorias.

Con ese diagnóstico, el roadmap que te propongo (fusión “Opción 1 \+ Opción 3”, con énfasis en la 3\) está diseñado para **forzar una demostración “a prueba de trampas” de cross‑modality real**: que haya **traducción** (audio→vib y vib→audio) y no solo “puntos cerca” en un embedding. Esto está completamente alineado con el objetivo explícito de la fase de mitigación: *“demostrar cross‑modality real (no solo alineamiento)”*.

---

## **0\) Objetivo operativo y definición de “éxito” (para que todo el equipo apunte a lo mismo)**

### **Hipótesis que realmente están testeando (Rosetta1)**

**H3 / HIT**: la estructura armónica (ratios) es preservable cross‑modalmente entre audio y vibración (misma causa física → geometría compatible).

### **Éxito “mínimo viable” (lo que tiene que pasar sí o sí)**

1. **No hay leakage**: split por archivo, no por frame.  
2. **z\_private vive** (no colapsa) y es **realmente modality-specific** (audio ≠ vib).  
3. **Cross‑reconstruction supera baselines triviales** y cae cuando rompes el pareo (controles negativos).  
4. **Retrieval cross‑modal sube de verdad** y también cae con controles negativos.  
5. La historia que cuentan los plots 2D/3D se **reconcilia con métricas en el latente**, no con UMAP “lindo”.

Importante: hoy la cross‑recon te da “moderada”, pero con una señal preocupante: **cross‑recon \= self‑recon** y además **da igual si pasas z\_private=0 o z\_private=target**, lo que es evidencia fuerte de que el modelo no está usando la factorización.

---

## **Roadmap Rosetta1 (fusión Opción 1 \+ Opción 3\)**

Te lo estructuro en **6 paquetes de trabajo (WP)**. Cada WP tiene: *qué se cambia*, *por qué*, *cómo se valida*, *deliverables*.

---

## **WP1 — Congelar baseline \+ trazabilidad (esto va primero, aunque parezca “burocracia”)**

### **Qué**

* Congelar **checkpoint**, **config exacta**, **seeds**, y **artefactos** (latentes extraídos, resultados JSON, plots).  
* Formalizar una carpeta “paper-grade” de evaluación.

### **Por qué**

Porque hoy ya hay contradicciones (p.ej., separación de regímenes “clara” vs métrica 3D 0.03) y riesgo de leakage; si no congelás, no sabés si un cambio mejoró algo o solo cambió la comparación.

### **Cómo (concreto)**

* Adoptar y exigir el esquema propuesto:  
  * Guardar checkpoint del modelo usado para métricas  
  * Guardar NPZ de latentes extraídos  
  * Guardar seeds

### **Deliverables**

* `config/rosetta1_baseline.yaml` (exacto; ver WP3 para versión final)  
* `artifacts/baseline/{checkpoint, latents.npz, results.json}`  
* Un README “cómo reproducir en 1 comando”

---

## **WP2 — Integridad metodológica dura (Opción 1): split correcto \+ controles anti‑leakage**

### **Qué**

1. **Split por archivo (estricto)**.  
2. Añadir **controles negativos** obligatorios en cada corrida de evaluación:  
   * **Shuffle cross-modal pairing** (rompés el pareo audio↔vib).  
   * **Mean predictor baseline** (decodificar al histograma promedio por modalidad).  
   * **Random z\_shared** (test de “decoder ignora input”).

### **Por qué**

* El plan ya identifica el problema: hoy el split por frames permite que frames del mismo archivo caigan en train y test → memorización.  
* Y además, tus métricas actuales permiten un “éxito falso” si los histogramas son parecidos globalmente (decoder que produce algo promedio te da correlaciones decentes).

### **Cómo (concreto)**

* Implementar exactamente el split propuesto (por `file_idx`): train 75%, val 12.5%, test 12.5%, sin mezclar frames del mismo archivo.  
* En `experiments/evaluate_cross_reconstruction.py`, agregar 3 banderas:  
  * `--pairing aligned|shuffled`  
  * `--baseline none|mean_hist`  
  * `--zshared real|random`

### **Validación (Go/No‑Go parcial)**

* Con **pairing=shuffled**, retrieval Top‑1 debe colapsar \~a random (≈1/N), y la cross‑recon debe empeorar *materialmente* respecto al aligned (definí “materialmente” como caída ≥0.15 en corr y empeora MSE/JS).  
  Esto es lo que te garantiza que el modelo no está “haciendo trampa” con señales globales.

### **Deliverables**

* PR en `src/datasets/roseta_dataset.py` con split por archivo (como el plan lista).  
* Evaluación que imprime tabla: aligned vs shuffled vs mean baseline.

---

## **WP3 — Arreglar lo CRÍTICO (Opción 3): matar el colapso de z\_private y hacer que la factorización sea real**

### **Punto de partida (lo que sabemos)**

* z\_private está **colapsado**: var(mu) \~0.0002, diferencia audio‑vib \~0.018 → inútil.  
* Causas probables ya identificadas:  
  * β KL uniforme (β=1) para shared y private  
  * InfoNCE dominó training, no necesitó z\_private  
  * No hay incentivo para modality-specific

### **Qué (cambios mínimos pero contundentes)**

Vas a re‑entrenar con **tres mecanismos simultáneos** (porque con uno solo es común que el modelo encuentre un atajo):

#### **(A) KL selectivo para private (capacidad/annealing)**

* Mantener `beta_shared = 1.0`  
* Bajar agresivo `beta_private = 0.01` (y luego anneal a 0.05–0.1 si hace falta).  
  Esto está explícitamente propuesto como Opción A.

#### **(B) Dropout en z\_shared durante decoding (forzar uso de private)**

Aplicar dropout **solo en train** al z\_shared antes del decoder. Propuesta Opción B: p=0.5.

#### **(C) Loss de diferenciación (pero lo haría “estable”, no explosivo)**

La propuesta actual penaliza si z\_private\_audio ≈ z\_private\_vib maximizando distancia.

**Mi ajuste** (importante): en vez de “maximizar distancia sin cota” (puede explotar si β\_private es chico), usar un **hinge con margen**:

* `diff_loss = relu(margin - ||z_priv_a - z_priv_v||_2).mean()`  
* Elegir `margin` \~ 1.0 (en unidades latentes) y ajustar según varianza observada.

Así imponés “tienen que ser diferentes” pero evitás crecer infinito.

---

### **Además (mi recomendación favorita para evitar que vuelva a colapsar)**

Añadir un **objetivo auxiliar modality‑specific** supervisado o auto‑supervisado sobre `z_private`:

* Si ya extraen features auxiliares (energía/entropía), entrenar una cabecita que prediga esas features desde `z_private` (por modalidad).  
  Esto crea el incentivo que hoy falta (“no hay incentivo para modality-specific”).

---

### **Validación (criterios duros)**

Adoptar (y elevar a “bloqueante”) el criterio de éxito del plan \+ el diagnóstico:

* `var(z_private_mu) > 0.1` (promedio)  
* `|z_private_audio - z_private_vib| > 0.5` (promedio por dimensión o norma)

Si esto no se cumple, **todo lo demás es ruido**: no tiene sentido discutir cross‑recon o separación de regímenes si la arquitectura base no se comporta como dice.

### **Deliverables**

* `config/rosetta1_fix_private.yaml`  
* Notebook/tabla: KL por dim, var(mu), diff audio‑vib, y comparación antes/después.

---

## **WP4 — “Traducción” de verdad: cross‑reconstruction \+ cycle \+ retrieval (pero con métricas que no se puedan “ganar” fácil)**

### **Punto de partida (lo ya hecho)**

Ya corriste el script y reportaste:

* Cross‑reconstruction corr: **0.704 (Audio→Vib)** y **0.574 (Vib→Audio)**  
* Retrieval Top‑1: **7.1%**, Top‑5: 35.7%, Top‑10: 63.9% (batch=1000).

Y el diagnóstico remarca lo raro: cross‑recon \= self‑recon exactamente.

### **Qué (subir el estándar de prueba)**

Mantener las métricas del plan (MSE/MAE/corr/KL y opcional EMD).  
Pero agregar 3 cosas que, en la práctica, separan “traducción” de “output promedio”:

1. **Mejora relativa vs baseline**  
   * Definir baseline: “histograma promedio por modalidad (train)”.  
   * Reportar `ΔMSE = MSE(baseline) - MSE(modelo)` y `Δcorr = corr(modelo) - corr(baseline)`.  
2. **Hard negatives en retrieval**  
   En vez de medir solo retrieval “global”, agregar:  
   * **Retrieval intra‑condición**: candidates solo de la misma condición → obliga a alinear fine‑grained.  
   * **Retrieval intra‑archivo**: candidates del mismo archivo en otros frames → obliga alineación temporal.  
3. **Controles negativos obligatorios** (de WP2) reportados en la misma tabla  
   aligned vs shuffled vs mean predictor.

---

### **Cuándo meter Cycle‑Recon como loss (decisión técnica)**

El plan contempla ciclo y define criterio (\>80% info).

**Mi recomendación**:

* Primero: arreglar z\_private (WP3) y split (WP2).  
* Después: si cross‑recon mejora pero retrieval sigue flojo, **recién ahí** añadir un término de cycle‑consistency como loss (no solo evaluación).  
  Esto evita entrenar un “ciclo” sobre una base colapsada.

### **Deliverables**

* `experiments/evaluate_cross_reconstruction.py` extendido con baselines/controles  
* `experiments/evaluate_retrieval.py` (ya está en el plan como script nuevo)  
* Tabla comparativa “baseline vs fix\_private vs fix\_private+cycle\_loss”

---

## **WP5 — Reconciliar “separación de regímenes” con una métrica única (y dejar de pelear con UMAP)**

### **Por qué**

Hoy está marcado como **débil**: 0.03 en 3D, y hay contradicción con visualizaciones 2D que “parecen” separar.

El propio plan te dice qué hacer: **medir en z\_shared, no en proyección UMAP**, y usar **Silhouette Score** \+ probes lineales.

### **Qué (estándar)**

* Silhouette sobre `z_shared` (healthy vs fault)  
* Linear probe AUC (healthy vs fault)  
* Distancia de centroides normalizada

### **Deliverable**

* Un único reporte “Regime Separation Report” que reemplaza declaraciones ambiguas tipo “se ve claro”.

---

## **WP6 — Ablations y cierre honesto (esto reemplaza “Rosetta2 por ahora”, y conecta con tu punto 6 final)**

### **Ablations mínimas (solo las que realmente responden “¿dónde está la novedad?”)**

Ejecutar la matriz A–D propuesta (descriptor e InfoNCE), y comparar con las mismas métricas: alignment, cross‑recon, retrieval.

Condiciones:

* A: ratio‑hist \+ InfoNCE  
* B: ratio‑hist sin InfoNCE  
* C: raw PSD \+ InfoNCE  
* D: ratio‑hist sin auxiliares \+ InfoNCE

### **Cierre honesto (la recomendación que no se negocia)**

Si la separación de regímenes sigue baja, el plan dice explícitamente: **reportar honestamente** y reformular el claim (“alinea modalidades pero no separa regímenes en z\_shared”).

Esta es la pieza que yo conecto como “tu punto 6 final”:  
**la salida del roadmap no es “ganamos sí o sí”**; es **(a)** demostración robusta de cross‑modality, o **(b)** resultado negativo/mixto pero científicamente valioso, con narrativa correcta y sin autoengaño.

---

# **Go / No‑Go propuesto (ajustado, más realista y más “anti‑trampa”)**

Tomo como base los criterios del diagnóstico (var private, diff private, corr cross‑recon, retrieval, split por archivo), pero los hago más operativos:

## **GO‑A (seguir iterando Rosetta1, ya sin dudas metodológicas)**

Bloqueantes:

* Split por archivo \+ controles negativos implementados.  
* z\_private: `var(mu) > 0.1` y `diff audio-vib > 0.5`.  
* Cross‑recon en **test**: `corr > 0.75` **y** mejora vs mean baseline (`Δcorr > +0.10` o `ΔMSE > 20%`).  
* Retrieval (batch=1000): Top‑1 `> 15%` **y** en shuffled cae cerca de random (\~0.1%).

## **GO‑B (estado “publicable” como claim fuerte de traducción)**

* Todo lo anterior, y además:  
* Cross‑recon `corr > 0.85` sostenido en múltiples seeds.  
* Retrieval Top‑1 `> 30%` y mejora clara en retrieval intra‑condición.  
* Ablation demuestra que ratio‑hist aporta (A \> C) o que al menos InfoNCE es crítico (A \>\> B).

## **NO‑GO (iterar arquitectura antes de cualquier “storytelling”)**

Cualquiera de estos:

* z\_private sigue colapsando tras aplicar (A+B+C) de WP3.  
* Cross‑recon no mejora vs baseline o no cae con shuffled (indicando prueba débil).  
* Evidencia de leakage o resultados inflados por split.

**Salida si NO‑GO persiste 2 iteraciones** (decisión estratégica):

* O bien aceptan que el modelo es “shared‑only” (se retira z\_private del claim y se reporta como *alineamiento*),  
* o cambian el input/representación (p.ej., enriquecer features de magnitud/energía si el ratio-hist está “demasiado normalizado”),  
* o amplían dataset (más variabilidad real), pero sin mover la portería.

---

## **Artefactos actuales (para que el equipo tenga un “mapa” de lo que ya existe)**

* Reportes cross‑recon:  
  * `REPORT_CROSS_RECONSTRUCTION.md`  
  * `REPORT_CROSS_RECONSTRUCTION_with_private.md`  
* Diagnóstico Enero 2026: `DIAGNOSTICO_ROSETTA1_ENERO2026.md`  
* Plan de mitigación: `PLAN_MITIGACION_ROSETTA1.md`  
* Visualizaciones 3D (útiles para exploración, no para métricas):  
  * plot3d\_umap\_trajectories.html  
  * plot3d\_umap\_alignment.html  
  * plot3d\_animation.html

---

## **Recomendación final (dura, pero la que te evita perder meses)**

No avancen “en ancho” (más features, más plots, más narrativa) hasta que WP2+WP3+WP4 estén cerrados con **controles negativos** y **split por archivo**, porque ahora mismo el proyecto está en el punto exacto donde es fácil “creer” que hay traducción cuando todavía puede ser un artefacto o un output promedio. El propio diagnóstico ya te lo está gritando: **la factorización falló** y por eso los dos reportes con/ sin z\_private dan lo mismo.

Si hacen este roadmap en orden, en poco tiempo experimental (no “tiempo calendario”) van a tener una de dos cosas:

1. **Una demostración robusta de traducción cross‑modal** (H3 fuerte), o  
2. **Una conclusión honesta y publicable**: “alineamiento sí, traducción todavía no”, con ablations que explican por qué (descriptor, InfoNCE, auxiliares).

Si querés, en el próximo paso puedo bajarte esto a un **checklist ejecutable por PRs** (orden de commits \+ estructura de carpetas \+ formato estándar de reporte de métricas) usando exactamente los scripts y rutas que ya aparecen en tus documentos.

