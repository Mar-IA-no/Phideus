# Criterios GO / NO-GO de Escalón 3

Fecha de consolidacion: `2026-03-21`

## Proposito

Este documento separa tres capas que en la practica habian quedado mezcladas:

- criterio canónico del frente;
- heuristica operativa local;
- criterio invalido o no bloqueante.

La regla general es simple:

- el **roadmap** manda sobre la direccion cientifica;
- los **planes** y **scripts** pueden fijar thresholds utiles para correr experimentos;
- esos thresholds no deben convertirse automaticamente en criterio de cierre del frente.

## Jerarquia

### 1. Criterio canónico

Son los criterios de cierre del roadmap. Deben guiar decisiones de avance entre fases.

### 2. Heuristica operativa

Son thresholds practicos para monitorear corridas. Sirven para orientacion rapida, no para cerrar una tesis.

### 3. Criterio invalido o no bloqueante

Si una tarea esta parcial o totalmente no identificable por construccion del banco o de la arquitectura, no debe bloquear la fase. Debe pasar a leerse como diagnostico auxiliar.

---

## E3-P0 - Canonical Generator

### GO canónico

- reproducibilidad exacta;
- metadatos completos;
- renders coherentes;
- cero leakage obvio entre ratio y estilo.

### NO-GO / pausa

- scenes inestables;
- metadatos incompletos;
- duplicacion o leakage entre split y estilo.

### Estado de lectura

`GO`. El banco ya existe y hoy es utilizable como objeto canonico.

---

## E3-P1 - Parameter Recovery

### GO canónico

- accuracy alta en IID;
- errores interpretables en OOD;
- baseline fuerte antes de retrieval.

### Core de cierre recomendado

- `ratio_id` recuperable con alta confiabilidad en ambas modalidades;
- lectura OOD interpretable en `ratio-OOD` y `equivalence-OOD`;
- el banco demuestra ser aprendible sin leakage trivial.

### Heuristicas operativas utiles

- `ratio_acc > 0.95` en IID;
- `equiv-OOD image > 0.90` como sanity check;
- `equiv-OOD audio > 0.50` como señal minima de invariancia de ratio.

### No bloquear con esto

- `amp_ratio` en imagen, mientras el renderer siga normalizando eje por eje;
- `phase` en imagen, mientras existan degeneraciones por simetria;
- `phase` en audio, mientras la arquitectura conserve pooling global fuerte.

### NO-GO / pausa

- si `ratio` no se recupera con claridad;
- si OOD queda caotico o arbitrario;
- si se descubre leakage fuerte que trivializa la tarea.

### Estado de lectura actual

`GO acotado para aprendibilidad por ratio`. No leer el `overall` del script como criterio canónico de cierre.

---

## E3-P2 - Flat Cross-Modal Retrieval

### GO canónico

- señal clara de retrieval;
- latente ya organizado por ratio y equivalencia;
- benchmark canónico del frente.

### Core de cierre recomendado

- `IID` con retrieval claro y estable;
- espacio latente estructurado por ratio;
- `render-OOD` no colapsado;
- `scale-OOD` y `equivalence-OOD` evaluados contra atlas OOD con cobertura completa.
- si existen dos referencias `L0`, dejar explícito cuál es baseline canónico y cuál es baseline alternativo.

### Heuristicas operativas utiles

- `silhouette > 0.30` como minima estructura no trivial;
- `S` alrededor de `0.60` como target operativo razonable, no como ley;
- `render-OOD > 0.40` como senal de robustez visual aceptable.

### NO-GO / pausa

- si `IID` no muestra retrieval claro;
- si el latente no queda organizado;
- si OOD se calcula con galerias que dejan queries sin positiva;
- si el banco se vuelve trivial por leakage.

### Estado de lectura actual

`Dual baseline fijado`.

- `P2-flat` = baseline canónico `L0` para retrieval general y gate formal.
- `P2-cqtshift` = baseline alternativo `L0` para invariancia de ratio del lado audio.

No leerlos como si uno invalidara automáticamente al otro. Lo correcto es:

- usar `P2-flat` para la lectura canónica del frente;
- usar `P2-cqtshift` como brazo comparativo serio cuando la pregunta sea audio-side invariance;
- no promediar sus scores ni fusionar sus claims.

---

## E3-P3 - Descriptor x Mechanism Matrix

### GO canónico

- tabla descriptor x mecanismo;
- lectura explicita de que aporta cada familia;
- no repetir la confusion entre descriptor y armonia.

### NO-GO / pausa

- si la matriz no permite separar aporte descriptorial de efecto mecanico;
- si el diseno mezcla demasiadas asimetrias a la vez.

### Nota operativa

Antes de correr `P3`, conviene fijar por adelantado que cuenta como “aporte explicito”.

---

## E3-P4 - Probe Regime on Flat Latent

### GO canónico

- evidencia de que el metodo de lectura cambia la estructura recuperada;
- o evidencia clara de que no la cambia.

### Core de cierre recomendado

- comparar probes sobre el mismo embedding y la misma receta de entrenamiento;
- exigir lectura interpretable frente a controles racionales y aleatorios;
- fijar de antemano las metricas de activation que cuentan como señal.
- si se usan dos baselines `L0`, decidir el gate primero sobre `P2-flat` y luego replicarlo sobre `P2-cqtshift`.

### NO-GO / pausa

- si no aparece ninguna diferencia interpretable entre familias de probe;
- si el efecto solo aparece por artefactos de evaluacion o de galerias mal construidas.

### Estado epistemologico

`P4` ya no debe leerse como veto suficiente sobre `P5/P6`.

Su alcance real es este:

- evaluar lectura post-hoc sobre embeddings entrenados en geometrías planas;
- registrar si aparece o no una ventaja interpretable entre familias de probe;
- dejar una base comparativa antes de entrar en geometrías no planas.

Estado actual:

- `P4` ya fue corrido sobre `P2-flat` y `P2-cqtshift`;
- dejó una mejora marginal de traversal sobre `cosine` en un slice de `flat`, pero sin ventaja diferencial robusta de `phi`;
- `cqtshift` saturó las métricas primarias;
- por decisión explícita del programa, el frente sigue con `P5/P6` completos y no trata el negativo/ambiguo de `P4` como bloqueo definitivo.

---

## E3-P5 - Mixed Geometry Latent

### GO canónico

- training estable;
- mejor estructura o mejor lectura;
- sin colapso raro ni degradacion trivial.

### NO-GO / pausa

- si el modelo mixto no mejora nada defendible;
- si solo agrega complejidad sin lectura nueva;
- si degrada el baseline plano de forma trivial.

### Estado de lectura actual

`Primera pasada completa ya corrida`.

Lectura vigente:

- `P5-flat` no desplazó a `P2-flat` como baseline general;
- pero la ablation mostró que la rama toroidal sí aporta señal causal;
- `P5-cqtshift` quedó como mejor brazo geométrico/OOD del frente en esta receta (`scale_ood S = 0.508`, `equiv_ood S = 0.472`).

La lectura correcta de `P5` es:

- **negativo** como reemplazo universal del baseline plano;
- **positivo parcial** como arquitectura geométrica útil, sobre todo en el brazo `cqtshift`.

---

## E3-P6 - Full T-VICReg

### GO canónico

No pedir solo mejor `R@10`. Pedir al menos una mejora defendible en:

- estructura latente por ratio;
- `activation gain`;
- menor relocking;
- mejor `coverage uniformity`;
- o transferencia al Tier dinamico.

### NO-GO / pausa

- si el toro no mejora ninguna metrica defendible;
- si el efecto solo es cosmeticamente mejor en un score aislado;
- si no supera lo que ya logra `P5`.

### Estado de lectura actual

`Primera pasada completa ya corrida`.

Lectura vigente:

- `P6-flat` salió negativo frente a `P2-flat` y `P5-flat`;
- `P6-cqtshift` dejó una estructura toroidal muy limpia;
- pero no superó a `P5-cqtshift` en las métricas OOD primarias.

La lectura correcta de `P6` es:

- hipótesis geométrica pura interesante;
- pero **no ganadora** bajo la receta actual.

---

## E3-P7 - Dynamic Activation Arena

### GO canónico

- aparece al menos una separacion reproducible entre locking y no-locking;
- la diferencia no se explica por un confound trivial;
- algun mapa de activation deja una lectura defendible frente a controles.

### NO-GO / pausa

- si drift, slip y near-rational regimes no producen ninguna estructura interpretable;
- si el frente vuelve a caer en “figuras lindas” sin lectura de activation.

---

## E3-P8 - Physical Transfer / Beacon Convergence

### GO canónico

- no hace falta cerrar producto;
- hace falta cerrar puente experimental.

### Cierre defendible

- el patron observado en el banco sintetico sobrevive de manera interpretable al pasar a captura fisica;
- hay continuidad experimental con Beacon, no solo analogia narrativa.

### NO-GO / pausa

- si el paso a hardware destruye por completo la lectura del banco;
- si la convergencia con Beacon queda solo como metafora.

---

## Regla final

En Escalón 3 no deberiamos volver a cerrar fases por un solo numero heredado del plan o del script. La lectura correcta es:

- criterio canónico del roadmap;
- heuristicas operativas explicitas;
- y diagnosticos auxiliares que no bloquean cuando el banco no hace identificable ese target por construccion.
