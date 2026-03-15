# Cierre Formal: Escalón 1 — Brazo Shazam

**Fecha de cierre**: 2026-03-05
**Subfase**: Escalón 1-A (Shazam / ratio tokens sin red)
**Estado**: ✅ CERRADO — límite estructural confirmado, pivote justificado
**Documento relacionado**: `INDICE_ESCALON1_COMPLETO.md` (mapa global del Escalón 1)

---

## 1. Posición en la Triplescaloneta

La **Triplescaloneta** es el framework de tres dominios del proyecto Phideus:

| Escalón | Dominio | Dataset | Estado |
|---------|---------|---------|--------|
| **1** | Audio ↔ MIDI | MAESTRO v3 (~200h, 1276 piezas) | CERRADO |
| **2** | Speech ↔ EGG | French Lombard (40 speakers, 9120 clips) | Pendiente |
| **3** | Audio XY ↔ Lissajous | Generado (determinista) | Conceptual |
| **4** | ECG ↔ PPG | BIDMC / MIMIC-III | Futuro |

**Escalón 1** tiene tres subfases. Este documento cierra la primera:

| Subfase | Descripción | Directorio | Estado |
|---------|-------------|------------|--------|
| **1-A** | Brazo Shazam — ratio tokens sin aprendizaje | este directorio | ✅ CERRADO |
| **1-B** | DANN — domain adversarial | `../BIAS_CONTROL/02_GATE_3_DANN/` | ✅ CERRADO (negativo) |
| **1-C** | Brazo Neural — BIAS_CONTROL / VICReg + descriptores | `../BIAS_CONTROL/` | ✅ CERRADO (Gate 5B) |

---

## 2. Objetivo del Brazo Shazam

Demostrar que un **lenguaje de ratios estilo Shazam** (constelaciones de ratios de frecuencia, matching por votación de offsets) puede identificar segmentos audio↔MIDI sin entrenamiento supervisado.

Si el enfoque funcionaba, habría demostrado algo fuerte: la correspondencia cross-modal emerge directamente de la estructura de ratios, sin necesidad de optimizar ningún parámetro.

---

## 3. Cronología

### Fase 1 — Extractor V1: Colapso

El extractor original producía tokens con ratio≈1 (triviales):

| Métrica | V1 |
|---------|----|
| Cosine similarity audio↔MIDI | 0.13 |
| Token ratio (audio/MIDI) | 356× |
| Compatibilidad | ❌ |

**Diagnóstico**: `peaks_per_frame_max` no estaba acotado (~62 por frame), generando tokens que colapsaban al ratio 1.

### Fase 2 — Extractor V2: Token Compatibility

Corrección con recomendaciones de GPT5.2Think:
- `peaks_per_frame_max = 8`
- Diversidad forzada: 50% targets cercanos, 50% lejanos
- `min_ratio = 1.05`
- MIDI pseudo-TF con 6 armónicos + envolvente ADSR

| Métrica | V2 |
|---------|----|
| Cosine similarity | **0.96** ✅ |
| Token ratio (audio/MIDI) | **1.07×** ✅ |
| `mean(log_ratio)` audio | 0.956 |
| `mean(log_ratio)` MIDI | 0.942 |

**Resultado**: Compatibilidad de distribuciones confirmada. **Gate 4A: PASS.**

### Fase 3 — Gate 4B Shazam: NO-GO

Retrieval cross-modal con Shazam (votación de offsets, 10 piezas):

| Métrica | Resultado | Objetivo |
|---------|-----------|---------|
| Oracle (MIDI vs MIDI) | 90.9% | >80% ✅ |
| Cross-modal (Audio vs MIDI) | **15.5%** | >50% ❌ |
| Offset MAE cross-modal | 30.87s | <3s ❌ |
| vs random | 1.5× | — |

**Diagnóstico (`diagnose_hash_collision.py`)**: Colisión genérica.

```
overlap_aligned:    66.23%
overlap_random:     65.13%
Gap (aligned-random): 1.10%  ← casi cero discriminabilidad
```

Los top 10 hashes aparecían en el 100% de las piezas. El extractor producía tokens estadísticamente compatibles (macro) pero sin identidad de instancia (micro).

### Fase 4 — Nuevos Extractores: Route A y Route B

Implementados según recomendaciones de GPT5.2Think para atacar la colisión genérica:

**Route A: Event-Based** (`src/extractors/event_based_extractor.py`)
- Audio → eventos via CQT + onset detection
- MIDI → eventos directos de notas
- Tokens: T_chord (acordes), T_seq (melódicos), T_pair (constelaciones)
- Hash 20 bits: `(type, dt_bin, dp_bin, pc_anchor)`

**Route B: Improved TF-Constellations** (`src/extractors/improved_tf_extractor.py`)
- Onset anchoring: solo frames cerca de onsets como anchors
- Harmonic folding: mapa a pitch class (octave-invariant)
- IDF agresivo: stoplist al 30% (antes 50%)

### Fase 5 — Auditoría (Fase A): Bug Crítico

El piloto original (N=10 pares) mostraba resultados inflados por un bug en `test_retrieval_routes.py`:

```python
# Bug: usaba delta-time en lugar de tiempo absoluto del anchor
t_abs = t.dt   # incorrecto — esto es el delta, no el tiempo del anchor
```

**Impacto**: Solo se generaban 7-10 queries en lugar de ~1177. Con tan pocas queries, la varianza estadística era enorme.

| Métrica | Con bug | Corregido |
|---------|---------|-----------|
| Queries Route A | 7 | 1177 |
| Queries Route B | 10 | 1175 |
| Accuracy Route A | 71.4% | **42.5%** |
| Accuracy Route B | **80.0%** | **32.9%** |

> **Nota importante**: El 80% de Route B frecuentemente recordado como "buen resultado" era artefacto del bug. En todos los tests controlados, Route B quedó por debajo de Route A.

### Fase 6 — Replicación (Fase B): N=20 Piezas Independientes

20 piezas nuevas, distintas a las originales, seed fijo para reproducibilidad:

| Route | Accuracy | vs Random | Queries |
|-------|----------|-----------|---------|
| A | **26.6%** | 5.3× | 2357 |
| B | 21.4% | 4.3× | 2361 |

- ✅ Resultados replicables entre muestras independientes
- ✅ Significativamente mejor que random (4-5×)
- ❌ Accuracy insuficiente para GO (objetivo: >50%)
- ⚠️ Accuracy baja con más piezas (esperado: degradación con pool más grande)

### Fase 7 — Análisis de Errores: Causa Raíz

Ablation por tipo de token y análisis de overlap profundo (`analyze_errors.py`, `analyze_overlap_deep.py`, `ablation_chord_only.py`):

**Overlap cross-modal por tipo de token:**

| Tipo | Descripción | Overlap audio↔MIDI |
|------|-------------|---------------------|
| **Chord** (tipo 1) | Notas simultáneas ±50ms | **62-85%** |
| Sequential (tipo 2) | Notas consecutivas | 4-15% |
| Constellation (tipo 3) | Pares lejanos (ΔT>1s) | 1-4% |

**Causa raíz confirmada: resolución temporal del onset detector**

```
MIDI (ground truth perfecto):
  ΔT=1 frame: 34 tokens
  ΔT=2 frames: 10 tokens

Audio (onset detection, librosa):
  ΔT=1 frame: 0 tokens  ← el detector no captura eventos tan cercanos
  ΔT=2 frames: 0 tokens
  ΔT=5-20 frames: mayoría de tokens
```

El onset detector de audio (librosa HFC) tiene resolución ~50-100ms. MIDI tiene timing exacto. Los hashes resultantes son incompatibles para tokens sequential y constellation — que son el 90%+ del volumen.

**Mejoras intentadas (+8pp overlap → +0.4pp accuracy):**
- DT_BIN_SIZE: 2→10 (más tolerancia temporal): +8pp overlap, **+0.4pp accuracy**
- Boost chord tokens (×2): efecto marginal
- Chord+Sequential only (sin constellation): 27.1% — prácticamente igual

**Conclusión del análisis**: El cuello de botella no es tunable. Las mejoras incrementales tienen rendimientos decrecientes demostrados.

---

## 4. Resultados Consolidados

**Tabla de resultados controlados** (todos los inflados por bug excluidos):

| Experimento | Route | N | Accuracy | vs Random | Estado |
|-------------|-------|---|----------|-----------|--------|
| V2 original (Shazam clásico) | — | 10 | 15.5% | 1.5× | NO-GO |
| Post-auditoría (bug corregido) | A | 10 | 42.5% | 4.2× | — |
| Post-auditoría (bug corregido) | B | 10 | 32.9% | 3.3× | — |
| **Replicación** | **A** | **20** | **26.6%** | **5.3×** | **límite superior confirmado** |
| Replicación | B | 20 | 21.4% | 4.3× | — |
| Post-mejoras (+DT_BIN, +chord boost) | A | 20 | **27.0%** | 5.4× | límite práctico ~27% |

**Gap pre-retrieval por extractor:**

| Extractor | overlap_aligned | overlap_random | Gap |
|-----------|-----------------|----------------|-----|
| V2 original | 66.23% | 65.13% | 1.10% |
| Route A | 21.88% | 12.16% | **9.71%** |
| Route B | 71.46% | 63.29% | 8.17% |

Route A tiene mejor ratio señal/ruido (gap 9.71%) aunque Route B tiene más overlap absoluto. En retrieval, Route A consistentemente superior.

---

## 5. Opciones No Implementadas

Tres opciones identificadas en el análisis de errores que quedaron sin ejecutar, con estimación de impacto y justificación de la decisión:

| Opción | Descripción | Impacto estimado | Por qué no se implementó |
|--------|-------------|------------------|--------------------------|
| **C: Superflux onset detection** | Reemplazar librosa HFC por superflux (más sensible, menos falsos negativos) | +10-15pp accuracy | Requiere tuning extensivo; el límite ~27% con mejoras menores sugiere que el problema es más profundo que el detector específico |
| **D: LSH / Soft matching** | En lugar de hash exacto, usar Locality-Sensitive Hashing (threshold=0.8, 128 permutaciones) | +15-20pp accuracy | Cambio arquitectural mayor; incluso con +20pp llegaríamos a ~47% — todavía bajo el objetivo de >50%, y lejos del ~83% del brazo neural |
| **E: Escalar a N=100+** | Procesar 100-200 piezas para confirmar tendencia | Confirmaría degradación | La tendencia de degradación con N ya es clara (42.5%→26.6%); escalar confirmaría el patrón sin cambiarlo |

**Decisión**: Las opciones C y D podrían haber empujado la performance modestamente, pero el techo proyectado (~47% con la mejora más optimista) quedaba lejos del objetivo, y la evidencia del brazo neural (S=83% con BIAS_CONTROL) hacía redundante continuar esta línea. La Opción E solo habría confirmado la degradación sin aportar información nueva.

---

## 6. Cierre Formal

El brazo Shazam de Escalón 1-A se cierra con las siguientes constataciones:

1. **Compatibilidad macro confirmada**: El lenguaje de ratios produce distribuciones estadísticamente compatibles entre audio y MIDI (cosine=0.96, Gap=9.71%).

2. **Identidad micro no alcanzada**: Los tokens no codifican firma específica de instancia. La causa raíz es estructural: la resolución temporal del onset detector de audio (~50-100ms) es incompatible con el timing exacto del MIDI para tokens sequential y constellation.

3. **Límite práctico cuantificado**: ~27% accuracy con N=20, 5.4× sobre random. Mejoras incrementales confirmadas como no escalables (+8pp overlap → +0.4pp accuracy).

4. **Opciones no implementadas deliberadamente**: Opciones C y D podrían haber aportado +10-20pp, pero incluso el escenario optimista (~47%) quedaba lejos del objetivo y del techo del brazo neural. Tiempo mejor invertido en 1-C.

5. **Route B cerrado sin iteración adicional**: El 80% del piloto era artefacto de bug (solo 10 queries vs 1175 reales). En todos los tests controlados, Route B < Route A. La técnica de harmonic folding (octave-invariance) es conceptualmente válida pero no resuelve el cuello de botella temporal.

**Señal conservada**: El brazo Shazam demostró que el lenguaje de ratios *puede* generar tokens cross-modalmente compatibles a nivel distribucional. Esto valida parcialmente H1 y es evidencia de que "ratio language" no es una idea vacía. El problema es el matching directo sin aprendizaje, no el concepto de ratios en sí.

---

## 7. Lecciones Permanentes

1. **Compatibilidad marginal ≠ identidad de instancia**: Que las distribuciones globales sean similares (cosine=0.96) no garantiza que tokens individuales codifiquen firma del segmento.

2. **Bugs de evaluación inflan métricas catastróficamente**: 10 queries vs 1175 — el 80% no era real. Los protocolos de evaluación necesitan verificar explícitamente el número de queries generadas.

3. **El análisis de errores debe preceder al tuning**: La correlación overlap-accuracy (r≈0.7) mostró que mejorar overlap ayuda, pero con rendimientos decrecientes confirmados. El cuello de botella hay que atacarlo en su causa raíz.

4. **Escalón 1-C demostró que el concepto es correcto con el mecanismo correcto**: Los descriptores A4/D4 sobre representaciones densas (VICReg/MERT) alcanzaron S≈84%, confirmando que la señal cross-modal existe. El Shazam falló en *acceder* a ella, no en que no existiera.

---

## 8. Evidencia Final del Escalón 1

El cierre formal del Escalón 1 completo vive en el brazo neural (Escalón 1-C):

| Evidencia | Resultado | Referencia |
|-----------|-----------|------------|
| Mejor S multi-seed | d4a4: **84.1% ±2.3pp** | `BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/` |
| Causalidad descriptores | Test 02: Δ=+9.4pp sobre control | Gate 5B Test 02 |
| Ventaja geométrica confirmada | Test 11 retention vs Test 13G-B F1 — paradoja de inversión de ranking | Gate 5B Tests 11 + 13G-B |
| Gate 6 AMT (activo) | Exp C `a4r`: **F1=0.1570 @ ep50**; `Exp B` ya cerrado como negativo útil | `BIAS_CONTROL/12_GATE_6_AMT/` |

**Conclusión de Escalón 1**: H3a parcialmente validada. Los descriptores relacionales (A4, D4) capturan estructura cross-modal de manera causal y su ventaja es geométrica (reorganizan la geometría de embeddings) más que de riqueza de features individuales.

---

## Referencias

| Documento | Descripción |
|-----------|-------------|
| `RESULTADOS_ESCALON_1.md` | Cronología completa fases 1-11 |
| `INFORME_FASES_A_B.md` | Auditoría (bug) + replicación N=20 |
| `INFORME_ANALISIS_ERRORES.md` | Análisis causa raíz + ablations |
| `RESULTADOS_NUEVOS_ENFOQUES.md` | Route A/B — piloto original (con caveat bug) |
| `INDICE_ESCALON1_COMPLETO.md` | Mapa global del Escalón 1 completo |
| `../BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md` | Roadmap Escalón 1-C (brazo neural) |
| `../BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/INFORME_COMPLETO_GATE5B.md` | Cierre científico Escalón 1-C |
