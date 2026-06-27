> **SUPERSEDED (2026-06-26)**: este es el plan Fase 0 v1. La auditoría de validez del pool
> (AUDIT_POOL.md) reveló que con parciales exactos + β=0, las pair features cerradas
> (common_f0_residual, ratio_residual) separan same-source con AUC≈1.0 → tarea feature-trivial
> → B vs A-rich nulo por construcción. Rediseñado en Fase 0 v2 (β>0 + dropout + gate de
> feature-triviality). Ver PLAN_FASE_0.md (v2). Se conserva como registro.

# Plan — Frente nuevo: Atención Armónica (Harmonic Pairformer) — Fase 0 decisiva

> Nombre del frente provisional: **Atención Armónica**. Arquitectura: **Harmonic Pairformer**.
> Renombrable. Corre EN PARALELO a Voz Expresiva (no toca nada de ese frente ni del run ZH).

## Context

Conversación con Codex sobre AlphaFold disparó la pregunta: ¿Phideus puede trascender los "descriptores inyectados en un backbone genérico" hacia una arquitectura cuya atención ya viva dentro de la geometría armónica natural — como AlphaFold razona dentro de la geometría del plegamiento en vez de recibir reglas químicas externas?

**Mi crítica a la propuesta de Codex** (que este frente incorpora): la "magia" de AlphaFold no es el transformer, es la **triangle update sobre una representación de pares que enforca una restricción global NO trivial** (consistencia métrica de distancias 3D). La versión que Codex propuso — triangle update sobre `log fA − log fC = (log fA − log fB) + (log fB − log fC)` — es **algebraicamente trivial**: es una identidad para tres reales cualesquiera, no una restricción. En log-frecuencia las diferencias viven en ℝ bajo la suma (grupo plano, sin restricción de ciclo). Si guardás Δlog f, la consistencia es gratis. La pieza que Codex marcó como "lo más interesante" es justo la que no transfiere tal cual.

**El rescate** (núcleo de este frente): reubicar la no-trivialidad donde sí vive en el dominio armónico — la **transitividad de la pertenencia a fundamental común**. Dado un conjunto de parciales de una mezcla polifónica, agruparlos por serie armónica de origen es un problema de inferencia global. Si `pair(i,j)` = "¿los picos i y j pertenecen al mismo f0?", entonces la **transitividad** (si i~j y j~k entonces i~k; el conjunto debe ser una partición consistente) es una restricción real y no trivial de enforcar desde estimaciones pairwise ruidosas. Esa es la analogía limpia con AlphaFold: pertenencia-a-misma-fuente ↔ contacto/distancia; transitividad de la equivalencia ↔ desigualdad triangular. Y es exactamente donde el pairwise local es ambiguo (un pico a 300 Hz en una mezcla {f0=100, f0=150} es 3×100 y 2×150 a la vez) y la consistencia global desambigua.

**Segunda crítica incorporada**: AlphaFold es inseparable de PDB (supervisión exacta a escala). Por eso el piloto vive donde el ground truth armónico es exacto: **audio sintético polifónico que generamos nosotros**, con etiqueta exacta de qué parcial pertenece a qué fuente.

## Qué es / qué no es este frente

- **Tesis fuerte**: la red aprende atención dentro de una geometría armónica, NO "inyectamos armonía en una red genérica". Los descriptores Phideus pasarían a ser probes / pérdidas auxiliares / controles de interpretabilidad, no payload de entrada.
- **Este primer deliverable (Fase 0) NO es la arquitectura completa**: es un **experimento de falsación barato y decisivo**. Pregunta única: ¿una representación de pares con triangle-update (transitividad) aporta sobre un transformer de posición relativa sobre picos, para agrupamiento armónico en polifonía?
- Si la respuesta es NO incluso en el caso más limpio, la analogía AlphaFold no transfiere a Phideus de forma barata, y nos ahorramos un programa de meses. Si es SÍ, justifica construir el programa grande.

## El experimento decisivo (Fase 0)

### Tarea

**Agrupamiento armónico de fuentes.** Dado un conjunto de N picos (parciales) de una mezcla polifónica, predecir la relación de equivalencia N×N "mismo-fuente" (¿picos i,j pertenecen al mismo f0?). La matriz de pares ES el objeto central — análogo directo al contact/distance map de AlphaFold.

- **Supervisión**: exacta, del generador sintético. Cada pico tiene un source label conocido → la relación mismo-fuente es exacta. Binary cross-entropy sobre los logits de pares. El caso de supervisión perfecta, manufacturado por síntesis (el equivalente local de PDB).
- **Entrada (confirmado)**: **parciales exactos** del ground truth (freq + amplitud exactas del sintetizador) como tokens. Aísla la pregunta arquitectónica pura sin ruido de detección. Detección por CQT = Fase 2.
- **Estructura (confirmado)**: **acordes sostenidos estáticos** (fuentes simultáneas, espectro estático). Reduce el problema a agrupamiento armónico puro en frecuencia — el test más limpio de la idea triangle-sobre-ratios. Sin eje temporal en Fase 0.

### Modelos a comparar (param-matched dentro de ~5%)

**Corrección clave (Codex #1)**: B no puede tener mejores *features* armónicas que su baseline, o B>A confunde "Pairformer/triangle" con "B recibió ratio-a-entero y compatibilidad-f0 y A no". Por eso el baseline decisivo es **A-rich**, que recibe EXACTAMENTE las mismas pair features que B (en su readout/bias) pero sin pair state persistente ni triangle. Así el contraste B vs A-rich aísla la *maquinaria* (pair state mantenido + triangle), con features igualadas.

| Modelo | Descripción | Aísla |
|---|---|---|
| **A-naive** (referencia deflacionaria) | Token self-attention, bias de posición relativa desde Δlog-f. Readout: MLP pairwise sobre (token_i, token_j) → logit. Solo features de token, sin pair features armónicas. | el piso ingenuo (transformer de posición relativa puro) |
| **A-rich** (baseline decisivo, features igualadas a B) | Igual estructura que A-naive, pero el readout pairwise recibe las MISMAS pair features que B (Δlog-f, ratio-a-entero-simple, compatibilidad f0 común) calculadas una sola vez. SIN pair state persistente, SIN pair update, SIN triangle. | el piso con features armónicas completas |
| **B** (Harmonic Pairformer) | Pair rep `z[i,j] ∈ ℝ^c` init desde las pair features; bloques ×L = (1) token attention sesgada por z, (2) pair update desde outer-product de tokens, (3) **triangle multiplicative update** sobre z. Readout simetrizado desde z[i,j]. | la propuesta completa |
| **B-minus** (ablación del triangle) | Igual a B pero SIN la triangle update (pair rep + attention sesgada + pair update, sin triangle). | si la ganancia viene del pair state o específicamente del triangle |
| **B-shuffle** (control negativo PARCIAL) | Igual a B pero las features de init del pair rep se shufflean. | descarta parcialmente confound de capacidad — ver caveat abajo |

**Contrastes decisivos**:
- **B vs A-rich** (PRIMARIO): ¿la maquinaria pair-state+triangle aporta, con features armónicas igualadas? Este es el contraste limpio.
- **B vs B-minus**: ¿el triangle específicamente aporta? (sin esto no se atribuye nada al triangle)
- **A-rich vs A-naive** (secundario): ¿cuánto aportan las pair features solas, sin maquinaria?
- **B-shuffle**: control NEGATIVO PARCIAL, no prueba fuerte. Codex #5: aun con pair init shuffleado, B puede reconstruir relaciones vía outer-product + attention desde los tokens. Lectura: si B-shuffle también gana mucho → confound de capacidad/arquitectura (alarma); si NO gana → apoya la lectura de que la ganancia es la estructura armónica. NO se usa "B-shuffle colapsa a A" como condición decisiva.

### Regímenes (el corazón del test)

- **Fácil**: f0s en ratios disonantes / irracionales → armónicos bien separados, cada pico tiene fuente única clara, pairwise local casi suficiente.
- **Difícil**: f0s en ratios casi-consonantes (cerca de 3:2, 4:3...) → muchos armónicos MUY cercanos en frecuencia, pairwise local máximamente ambiguo, la transitividad global es lo que desambigua. **Este es el régimen crucial.** (Y mapea sobre la tesis Phideus: ratios simples = consonancia.)

**Corrección colisiones (Codex #2)**: NO se usa la política artificial "merge + armónico más bajo gana" — fuerza un target no físico (un pico físicamente compartido por dos fuentes asignado arbitrariamente a una). En su lugar, en el test principal **se evitan las colisiones exactas con jitter en cents**: cada f0 del régimen difícil se desplaza ±(10–30) cents respecto al ratio consonante exacto. Así los armónicos de fuentes distintas quedan MUY cerca pero nunca exactamente coincidentes — cada pico tiene fuente única física (equivalencia limpia y física), y la ambigüedad pairwise sigue máxima (un pico cerca de 300 Hz: ¿3×~100 o 2×~150? la red debe usar estructura global). Si por azar dos parciales caen dentro de la tolerancia ε=10 cents, ese par se **excluye del loss** (se enmascara), no se mergea. Las colisiones EXACTAS (multi-membership real) quedan como subtest posterior, NO como núcleo decisivo.

### Configuraciones de run — ID + 2 OOD separadas (Codex #2)

ID y OOD NO pueden coexistir en un mismo split. El generador produce un **pool** que cubre todas las combinaciones poly(1,2,3)×régimen(fácil,difícil) con muestras suficientes; de ese pool se derivan **tres runs independientes**, cada uno con su propia composición train/val/test. Los 5 modelos se entrenan y evalúan por separado en cada run.

| Run | Train | Val | Test | Pregunta |
|---|---|---|---|---|
| **ID** | poly 1/2/3 × fácil/difícil balanceado | idem (held-out) | idem (held-out) | ¿agrupa bien in-distribution? |
| **OOD-poly** | SOLO poly 1 y 2 (fácil+difícil) | poly 1/2 held-out | SOLO poly 3 | ¿generaliza a más fuentes de las vistas? |
| **OOD-regime** | SOLO régimen fácil (poly 1/2/3) | fácil held-out | SOLO régimen difícil | ¿generaliza a la ambigüedad alta (casi-consonante)? |

Splits por mezcla (no por pico) para evitar leakage. Mismos splits para los 5 modelos en cada run. El contraste B vs A-rich se lee en los tres runs; el inductive bias debería brillar más en OOD-poly y OOD-regime.

### Métrica

**Corrección (Codex #3)**: el modelo predice PARES, no particiones. ARI necesita clusters, y el pair→partición es un knob post-hoc. Por eso:

- **Primaria (congelada)**: **F1 pairwise de mismo-fuente sobre el triángulo superior de la matriz, excluyendo la diagonal**. Se computa directo desde los logits de pares (umbral 0.5), sin clustering. Sin knobs.
- **Secundaria threshold-free (Codex #2)**: **AP / AUPRC pairwise** (y ROC-AUC) sobre los mismos pares del triángulo superior, sin umbral. Distingue "mala calibración" (F1 baja pero ranking bueno) de "mala separación" (ranking malo). NO reemplaza la F1; la complementa.
- **Secundaria (clustering)**: **ARI del clustering inducido**, con post-proceso FIJO declarado de antemano: connected components sobre la matriz de pares binarizada a un umbral τ. **τ se elige UNA sola vez en validación** (barrido sobre val, congelado antes de tocar test). El mismo τ se aplica a los 5 modelos. Se reporta el τ elegido.
- Ambas **reportadas como función de polifonía (1,2,3) × régimen (fácil/difícil)**. Polifonía 1 es degenerada (todo mismo-fuente, F1 trivial) — sanity, no evidencia.
- **Multi-seed** (≥3 seeds), mean±std.
- **Bootstrap CI95** sobre mezclas de test independientes para los contrastes B−A-rich, B−B-minus, A-rich−A-naive.
- Param counts reportados por modelo (verificación de param-matching ~5%).

### Lógica de falsación (prefigurada; GO/NO-GO lo decide el usuario)

Sobre el contraste PRIMARIO **B vs A-rich** (features igualadas) en régimen difícil/OOD:

- **B>A-rich y B>B-minus en difícil/OOD** → el pair state + triangle hacen trabajo real con features igualadas → GO al programa arquitectónico grande. (B-shuffle no gana refuerza; si gana, alarma de capacidad.)
- **B≈A-rich** → la maquinaria pair-state+triangle no aporta sobre tener las features armónicas en un readout simple → la analogía AlphaFold no transfiere barato → NO construir la cosa grande; "features/descriptores armónicos en un modelo simple" basta, que es esencialmente el modo Phideus actual.
- **B>A-rich pero B≈B-minus** → el pair state mantenido ayuda pero el triangle específicamente no → quedarse con pair rep, descartar triangle.
- **A-rich≈A-naive** (lectura lateral) → ni siquiera las pair features armónicas aportan sobre Δlog-f solo → el problema no necesita estructura de ratios explícita en este régimen.

## Generador sintético (NUEVO — necesario)

Los generadores existentes (`src/generador/generador_wavs_ratios_*.py`) NO emiten ground truth per-parcial ni polifonía. El generador Lissajous (`experiments/escalon3/generate_lissajous_dataset.py`) SÍ escribe `meta.json` por escena — ese es el patrón de sidecar a copiar.

Nuevo generador de mezclas armónicas polifónicas. **Grilla CONGELADA antes del primer run (Codex #4)** — sin "a decidir al implementar", para no habilitar ajuste post-hoc:

| Parámetro | Valor congelado |
|---|---|
| K armónicos por fuente | 8 (múltiplos 1..8), amplitud decay 1/n |
| Rango f0 base | 100–500 Hz (log-uniforme) |
| Polifonía | 1, 2, 3 fuentes |
| n mezclas (pool) | pool generado cubre poly(1,2,3)×régimen(fácil,difícil); ~36 000 mezclas total (6 000 por celda poly×régimen). Cada run (ID/OOD-poly/OOD-regime) deriva su train/val/test del pool según la tabla de runs. ID usa ~24k train / 3k val / 6k test |
| Proporción fácil/difícil | 50 / 50 |
| Régimen fácil | ratios entre f0s irracionales/disonantes (√2, φ, etc. + ruido) |
| Régimen difícil | ratios entre f0s casi-consonantes (3:2, 4:3, 5:4, 2:1) + jitter en cents |
| Jitter (difícil) | ±(10–30) cents uniforme sobre cada f0 respecto al ratio exacto |
| Inarmonicidad | β=0 en Fase 0 (armónicos enteros exactos por fuente; stretch queda para fase futura) |
| Tolerancia ε loss-mask | **en cents, no Hz (Codex #3)**: pares de parciales a <10 cents se excluyen del loss (escala-invariante, coherente con el framing log-f). El reporte audita el mask rate por polifonía×régimen×banda |
| Amplitud por fuente | normalizada; mezcla = suma de fuentes |
| Splits por seed | fijos (mismo split de mezclas para los 5 modelos; seeds 42/123/456 solo varían init de pesos) |

- **Sidecar ground truth** (patrón Lissajous meta.json): por mezcla, lista de fuentes (f0 + lista de (harmonic_number, freq, amplitude)), + lista plana de picos con source label, + pares enmascarados por ε.
- Para Fase 0 (parciales exactos): el "audio" puede ni renderizarse — basta el sidecar con los parciales. (Render WAV opcional, útil para Fase 2 CQT.)
- La grilla puede ajustarse SOLO vía nuevo plan-mode si un smoke revela degeneración (ej. tarea trivial al 100% o imposible), NUNCA post-hoc sobre resultados de test.

## Spec de arquitectura

- **Features de token**: `[log-f centrado (invariante a transposición por centrado de la mezcla), log-amplitud]`. Sin Δt en Fase 0 (estático).
- **Model A**: token self-attention (2-4 capas), bias de atención desde Δlog-f relativo entre tokens (reusar idea de bias factorizado); readout = MLP sobre concatenación/producto (token_i, token_j) → logit.
- **Model A-naive**: token self-attention (2-4 capas), bias desde Δlog-f; readout MLP sobre (token_i, token_j).
- **Model A-rich**: igual, pero el readout MLP recibe además las pair features (definidas abajo) concatenadas. Sin pair state, sin triangle.

**Pair features — definición anti-leakage CONGELADA (Codex #3)**. TODAS se computan SOLO desde las frecuencias observadas `(f_i, f_j)` y una grilla racional permitida `p,q ∈ {1..Q}, Q=8`. NUNCA desde los f0 reales del generador ni desde `source_id`. Sea `r = max(f_i,f_j)/min(f_i,f_j) ≥ 1`:
  1. `dlogf = |log f_i − log f_j|` (simétrico).
  2. `ratio_residual = min_{p,q ≤ Q, gcd(p,q)=1} | log r − log(p/q) |` — qué tan cerca está el ratio observado del racional simple más próximo.
  3. `ratio_class_id` = índice (p,q) que minimiza (2) (Codex #4: es un id de clase, NO un one-hot; se pasa por un `nn.Embedding` aprendible — nombrarlo así evita confusión en shapes/param counts).
  4. `common_f0_residual = min_{m,n ≤ K} | log(f_i/m) − log(f_j/n) |` — el mejor residual de fundamental común implícito (si i es armónico m y j es armónico n de un mismo f0, esto es ~0). **Derivado puro de (f_i,f_j)**, sin acceso al f0 verdadero.
  5. `log_amp_diff = |log a_i − log a_j|`.
Estas 5 (con el one-hot embebido) forman el vector de pair features, idéntico para A-rich (en readout) y B (en pair init). Test de anti-leakage: la función NO recibe `source_id` ni los `f0` reales como argumento.

- **Model B**: 
  - pair init `z[i,j]` desde las pair features anti-leakage de arriba (proyectadas a ℝ^c).
  - bloque ×L: token attention sesgada por `z` (bias aditivo por par) → pair update desde outer-product de tokens (`z[i,j] += g(t_i ⊙ t_j)`) → **triangle multiplicative update** propagando transitividad.
  - **Detalles del triangle (Codex #6)**: `z[i,j] ← z[i,j] + scale · (1/|K_ij|) Σ_{k∈válidos} a(z[i,k]) ⊙ b(z[k,j])` donde:
    - la suma EXCLUYE padding y excluye k∈{i,j} (solo k válidos reales);
    - se **normaliza** por `|K_ij|` = número de k válidos (evita que mezclas con más tokens dominen);
    - `scale` es un `nn.Parameter` init pequeño (~0.1) para arranque suave;
    - el pair rep se **mantiene simétrico**: tras cada update `z ← (z + zᵀ)/2` (la relación mismo-fuente es simétrica por definición; sin esto la red aprende artefactos direccionales).
  - readout = MLP sobre `z[i,j]` ya simétrico → logit; además se promedia logit(i,j) y logit(j,i) por las dudas.
- **B-minus**: B sin el paso triangle (conserva simetrización).
- **B-shuffle** (Codex #4): B con las pair features de init permutadas **dentro de los pares válidos, por muestra, preservando la máscara** (el padding NUNCA entra a la permutación → no inyecta info por padding), y **re-simetrizando** tras el shuffle (`z_init ← (z_init + z_initᵀ)/2`) para no romper `z[i,j]=z[j,i]`. El init de tokens queda intacto. Así el control aísla la info armónica del pair init sin introducir artefactos de máscara/simetría.
- **Param-matching**: ajustar hidden dims para igualar params totales entrenables dentro de ~5% entre A-rich y B (el contraste primario). A-naive/A-rich reciben transformer de token más ancho/profundo para compensar la ausencia del track de pares. Reportar counts de los 5 modelos.

## Files to create

```
Documents/01_FRENTES_ACTIVOS/Atencion_Armonica/
├── README.md                       # estado, tesis, resultado Fase 0
├── ROADMAP_ATENCION_ARMONICA.md    # §1 identidad §2 origen (AlphaFold+crítica) §3 estado §4 fases §5 stack §6 GO/NO-GO no congelados
└── PLAN_FASE_0.md                  # copia de este plan (referencia canónica del frente)

src/atencion_armonica/
├── harmonic_synth.py               # generador polifónico + sidecar ground truth (patrón Lissajous meta.json)
├── peak_tokens.py                  # parciales → tokens [N, feat] + targets pair [N, N] + máscaras; política de colisión
├── pairformer.py                   # Model A, B, B-minus (flag), pieza triangle update, B-shuffle (flag init)
└── grouping_dataset.py             # Dataset + collate (padding de N variable + máscara pairwise)

experiments/atencion_armonica/
├── 0_generate.py                   # genera el pool sintético (CPU, 14 workers)
├── 1_train_grouping.py             # train {A-naive, A-rich, B, B-minus, B-shuffle} × 3 seeds × 3 runs (ID/OOD-poly/OOD-regime); flush incremental
├── 1_report.py                     # agrega F1 pairwise (primaria) + ARI (τ de val) por polifonía×régimen×run, bootstrap CIs, plots, REPORTE_0.md
└── harness.py                      # métricas: pairwise F1 (upper-tri), ARI con connected-components, oracle, random baseline (patrón gate0_harness)

data/atencion_armonica/             # gitignored (sidecars, tokens cacheados, resultados)
```

## Reuso explícito (con rutas)

| Componente | Origen | Reuso |
|---|---|---|
| Sidecar ground truth per-archivo | `experiments/escalon3/generate_lissajous_dataset.py` (meta.json) | Patrón de escritura de ground truth |
| Bias de atención pairwise factorizado zero-init | `src/bias_control/encoders/speech_egg_encoder_attn_bias.py` (`AttentionBiasComputer`) | Adaptar para token-attention sesgada por pair state |
| Masked softmax sobre tokens | `src/RNA/constellation_vae.py` (`MLPConstellationEncoder`, masked_fill -inf) | Máscaras en attention |
| Collate con padding + máscaras | `src/datasets/roseta_dataset.py` (`collate_constellation_sequences`), `src/voz_expresiva/esd_dataset.py` (`collate_padded`) | Padding de N variable + máscara pairwise |
| Métricas / harness | `experiments/maestro/gate0_harness.py` (retrieval, oracle, random baseline) | Patrón de harness; escribo F1/ARI nuevos |
| set_seed, flush incremental, tmux, argparse | `experiments/voz_expresiva/1_train.py` | Convenciones de reproducibilidad |
| VICReg | `src/RNA/vicreg.py` | SOLO en etapa self-supervised futura, NO en Fase 0 |

**No reuso de extracción de picos en Fase 0** (parciales exactos del sidecar, sin CQT). `improved_tf_extractor.py` / `analizador_maestro.py` entran en Fase 2.

## Decisiones congeladas en este plan mode

| Eje | Valor |
|---|---|
| Tarea | Agrupamiento armónico: predecir relación de equivalencia mismo-fuente N×N |
| Entrada | Parciales exactos del ground truth (sin detección CQT) |
| Estructura señal | Acordes sostenidos estáticos (sin eje temporal) |
| Supervisión | Exacta (source labels), BCE sobre logits de pares; pares <10 cents excluidos del loss |
| Modelos | A-naive, A-rich (features=B), B (Pairformer+triangle), B-minus (sin triangle), B-shuffle (control parcial) — A-rich y B param-matched ~5% |
| Contraste PRIMARIO | **B vs A-rich** (maquinaria con features igualadas). Secundarios: B−B-minus (triangle), A-rich−A-naive (features) |
| B-shuffle | control negativo PARCIAL, no prueba fuerte (puede reconstruir vía tokens) |
| Regímenes | fácil (f0 disonante) vs difícil (f0 casi-consonante + jitter cents) |
| OOD | train poli≤2→test poli3; train disonante→test consonante |
| Colisiones | EVITADAS con jitter en cents (no merge artificial); pares sub-ε excluidos del loss; colisión exacta = subtest posterior |
| Triangle update | multiplicative mask-aware, normalizado por \|K_ij\|, scale aprendible, z simetrizado tras cada update |
| Token features | [log-f centrado, log-amplitud] |
| Métrica PRIMARIA | F1 pairwise sobre triángulo superior sin diagonal (umbral 0.5, sin clustering) |
| Métrica secundaria | ARI con connected-components a umbral τ elegido SOLO en val, congelado, mismo τ para los 5 modelos |
| Reporte | por polifonía×régimen; multi-seed mean±std; bootstrap CI95 sobre mezclas test |
| Seeds | 42, 123, 456 (varían solo init de pesos; split de mezclas fijo) |
| Generador | grilla congelada (K=8, f0 100-500Hz, pool ~36k, 50/50 fácil/difícil, jitter ±10-30c, β=0, ε=10 cents) |
| GO/NO-GO | lo decide el usuario; no fijo thresholds |
| Generación | CPU ahora, paralelo a ZH |
| Training | GPU, después de que ZH libere la GPU (directiva: NUNCA pilots en CPU) |

## Decisiones que NO se congelan acá

- Profundidad/anchura exactas de los modelos (sujeto a param-matching A-rich vs B ~5%; se reportan).
- **Optimizer/schedule NO es libre** (Codex #1): mismo optimizer, lr, batch, epochs, y misma política de early-stopping (o sin early-stopping) para los **5 modelos**, congelado antes del run decisivo. El smoke solo puede forzar un cambio si detecta NaN/degeneración, y ese cambio se aplica **globalmente a los 5 modelos y se documenta ANTES de mirar test** — nunca por modelo ni según señal de performance. Defaults de arranque: AdamW lr=3e-4, batch 256 mezclas, 50 epochs, sin early-stopping (corrida fija), se reportan.
- Si Fase 0 da GO: forma de Fase 1 (CQT-detected peaks), Fase 2 (mezclas temporales/onsets), Fase 3 (WavLM como single-state + trunk armónico paralelo que construye pair state), self-supervised, real audio, colisiones exactas multi-membership. Difusión sobre grafos de parciales (análogo AF3) = futuro lejano. Nada de eso se diseña ahora.

**NOTA**: la grilla del generador (K, f0, tamaños, jitter, β, ε, splits) SÍ está congelada en la tabla de arriba — NO reaparece como "a decidir". Solo se reabre vía nuevo plan-mode si un smoke revela degeneración.

## Compute & paralelismo

- **Generación sintética**: CPU, 14 workers. Minutos a ~1 h según tamaño. **Corre AHORA en paralelo al run ZH** (ZH es GPU-bound; su DataLoader usa pocos cores).
- **Code + shapes**: escribible y testeable por shapes en CPU ahora (forward/backward de los 4 modelos con N pequeño, sin comprometer GPU).
- **Training del piloto**: modelos chicos (<1M params, N≤40 tokens, pair tensor N×N×c trivial; triangle O(N³c) con N≤40 ~ nada). Por directiva **NO se entrena en CPU** → el training GPU espera a que ZH libere la GPU (~8 h). Sweep multi-seed completo estimado ~1-2 h GPU una vez con GPU libre.
- **Orden**: (1) código + generación ahora, (2) smoke de shapes en CPU, (3) training GPU cuando libre.

## Lo que este plan NO hace

- No toca Voz Expresiva ni el run ZH en curso (ni su GPU hasta que termine).
- No usa detección CQT (Fase 2), no usa estructura temporal/onsets (Fase 2), no usa WavLM (Fase 3), no usa audio real, no usa self-supervised (etapa futura), no difusión (AF3, futuro lejano).
- No entrena en CPU.
- No actualiza `00_TRONCAL/` (propagación de Codex tras primer resultado; frente nuevo = incubación documental).
- No commitea hasta que Fase 0 cierre con resultado.

## Verification

1. `harmonic_synth.py` emite sidecars válidos: la suma de parciales de cada fuente reconstruye la mezcla; source labels exactos; en régimen difícil NO hay colisiones exactas (jitter aplicado); pares sub-ε correctamente marcados para exclusión del loss (test: una mezcla conocida a mano → partición esperada).
2. `peak_tokens.py`: dataset → tokens `[B, N_max, 2]`, máscara `[B, N_max]`, target pair `[B, N_max, N_max]` binario simétrico, máscara pairwise (incluye exclusión de pares sub-ε). Target es relación de equivalencia válida (transitividad verificada en un test).
3. Param-matching: `A-rich` y `B` dentro de ~5% de params entrenables (assert + log). Counts de los 5 modelos reportados.
4. Triangle update: test unitario de que es mask-aware (padding no contribuye), normalizado por |K_ij|, y que `z` queda simétrico tras el bloque.
5. Smoke (CPU, shapes): 1 epoch tiny, los 5 modelos forward+backward sin NaN/Inf, F1 pairwise + ARI computables, flush OK.
6. Run decisivo (GPU): multi-seed; `REPORTE_0.md` con F1 pairwise (primaria) + **AP/AUPRC + ROC-AUC pairwise (threshold-free)** + ARI (τ de val) por polifonía×régimen×run, param counts de los 5 modelos, bootstrap CI95 de B−A-rich / B−B-minus / A-rich−A-naive, τ reportado, mask-rate audit (ε en cents), y lectura contra los escenarios prefigurados.
7. `git status`: solo archivos nuevos bajo `src/atencion_armonica/`, `experiments/atencion_armonica/`, `Documents/01_FRENTES_ACTIVOS/Atencion_Armonica/`. `data/atencion_armonica/` gitignored. Cero cambios en Voz Expresiva.

## Lectura propositiva post-resultado

El valor del corte está en la decisión binaria que habilita: ¿vale la pena el programa arquitectónico AlphaFold-inspired para Phideus, o el modo "descriptor en backbone" sigue siendo el correcto? Fase 0 contesta eso barato y con causalidad atribuible (la batería de ablaciones aísla exactamente al triangle). Cualquiera de los dos resultados es información de primera línea sobre la frontera arquitectónica de Phideus.
