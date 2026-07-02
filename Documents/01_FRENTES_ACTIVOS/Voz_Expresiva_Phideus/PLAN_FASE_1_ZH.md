> **Archivo de plan** — copia preservada del plan aprobado de Fase 1 ZH, tal como
> quedó tras las ~6 rondas de revisión con Codex. Se archiva acá porque el plan
> vivía en el plan-mode storage (`~/.claude/plans/`) y ese archivo se reutiliza
> para el siguiente frente. Este markdown es la referencia canónica del plan ZH.
>
> **Estado al archivar (actualizado 2026-07-02)**: el training full LOSO `ZH` ya
> terminó (`240/240` runs, ~`7.1 h` wall-clock) con el manifest corregido (`fix B2`)
> y quedó persistido en `data/voz_expresiva/1_zh/`. El rerun `EN N-adapt` limpio
> también ya llegó y el contraste `EN ↔ ZH` quedó formalmente cerrado. Lectura
> canónica absorbida por el frente: **positivo acotado a `N-adapt` (`concat` y
> `FiLM` replican limpio cross-language) y lectura `null/negativa` en `N-strict`**
> (`concat` cerca de nulo, `film/xattn` negativos en `ZH`). El caveat `0A ZH`
> (`A/C=0.69` vs `2.88` EN) queda preservado como parte de la interpretación, ya
> no como deuda metodológica abierta. Ver `Documents/NOTAS_CLAUDE-CODEX.md` para
> la traza operativa completa y `data/voz_expresiva/REPORTE_CROSS_LANGUAGE_EN_ZH.md`
> para el cierre empírico integrado.

---

# Plan — Fase 1 ZH: replicación cross-language sobre ESD Mandarin (Voz Expresiva Phideus)

## Context

Fase 1 EN cerró con evidencia direccional positiva (commit `bc34c12`, S60 continuación de NOTAS_CLAUDE-CODEX):

- WavLM-only escapa chance fuerte (UAR=0.698 vs 0.20 chance).
- Concat pasa formalmente en N-strict: Δ=+0.039, CI95 [+0.019, +0.060], P(Δ>0)=1.00.
- Los tres mecanismos (concat/film/xattn) pasan robustos en N-adapt con +4 pp uniforme.
- Disociación CKA: FiLM modula sin reorganizar (CKA ~0.85), concat/xattn reorganizan (CKA ~0.23).
- Encaja con el primer escenario prefigurado del plan: Phideus transfiere a SSL.

**Debilidad central**: un solo corpus, un solo idioma, una sola etnografía de actuación. Bootstrap n=10 hablantes es señal, no prueba fuerte. Saltar directo a MSP-Podcast desde un punto único mezcla varios factores (corpus + idioma + naturalístico) y nos deja sin atribución si no transfiere.

**Lo más responsable como siguiente paso**: replicar Fase 1 sobre los 10 hablantes ZH del mismo corpus ESD. Cambia un solo factor — idioma — manteniendo dataset, estructura, emociones, cantidad de frases, metodología LOSO. Si el resultado replica, el claim cross-language sobre habla actuada se sostiene desde dos puntos y el salto a Fase 3 se hace con mayor base. Si no replica, aprendemos algo importante sobre el alcance del descriptor (¿inglés-específico? ¿tonalidad mandarina contamina? ¿la actuación EN y ZH son fenómenos distintos?). Cualquiera de los dos resultados tiene valor.

**Riesgo principal a controlar**: el mandarín es tonal — F0 codifica significado léxico además de afecto. V4-lin y H-series están construidos sobre F0 y armónicos, así que existe la posibilidad de que el descriptor capture varianza tonal en vez de (o además de) varianza emocional. Las 350 frases por hablante están balanceadas sobre las 5 emociones (mismo texto, distinta emoción), así que la distribución tonal léxica debería cancelar al promediar por emoción. Pero hay que verificarlo empíricamente antes de comprometer 7 h GPU.

## Approach

### 0A ZH como caracterización previa, NO como veto

Codex objetó (correctamente) que un eta² débil sobre el descriptor pooled univariado no es equivalente a "descriptor no aporta dentro de WavLM frame-level + injection". Son preguntas distintas y un veto del primero contra el segundo es metodológicamente impropio. Fase 1 puede capitalizar señal frame-level, interacción con WavLM, o información distribuida que el eta² univariado no captura.

**Decisión revisada**: correr 0A ZH como caracterización del descriptor en mandarín, pero **NO como condición de abort** sobre Fase 1 ZH. La pregunta de Fase 1 se ejecuta de modo no condicional. El resultado de 0A ZH alimenta la interpretación del reporte y queda como referencia para futuras lecturas, no como gate.

Cuándo se corre 0A ZH:
- En **paralelo CPU** con la precache WavLM ZH y con el training GPU posterior. No bloquea ninguna etapa.
- El extractor `0A_extract.py` produce A+B+C+D, no solo A. Costo real: ~3-5 h CPU para 17 500 utts (estimado desde 0A EN que tardó similar). Aprovechamos para tener 0A ZH completo (no solo Familia A) — es útil para eventual replicación de 0B en ZH o para la comparación cross-language de especificidad ratio (A > C en ZH también?).
- `0A_analyze.py` produce `ranking_univariate.json`, `silhouette_per_family.json`, `variance_decomposition.json` y `REPORTE_0A.md` en el directorio de salida. Para leer eta² de Familia A en mandarín, abrir `ranking_univariate.json` con la clave `"A"` (no `"family_A_pooled"` — el analyze indexa por la letra corta A/B/C/D, ver `0A_analyze.py:329-334`). Tomar `eta_squared` de cada entry. Comparar mean y top-5 contra los valores reportados en `Documents/01_FRENTES_ACTIVOS/Voz_Expresiva_Phideus/genesis_descriptores_fase1.md` (0A EN: H-series eta² ~0.385, control C key `"C"` ~0.076).

Lectura que produce 0A ZH:
- Eta² Familia A ZH comparable a EN → reaseguro: el descriptor tiene firma armónica robusta cross-language a nivel univariado. Fase 1 ZH se interpreta sin caveat extra sobre el descriptor.
- Eta² Familia A ZH sensiblemente más bajo → caveat declarado en el reporte: el descriptor pooled univariado debilita en mandarín; cualquier mejora que Fase 1 ZH detecte aporta sobre un descriptor que ya empieza más débil que en EN. NO aborta Fase 1.
- Eta² Familia A ZH ~ control C ZH → caveat fuerte declarado. Si además Fase 1 ZH muestra escenario "ningún mecanismo pasa", la combinación es coherente: ni el descriptor solo ni dentro de WavLM tiene firma en ZH. Pero la atribución sigue siendo del experimento Fase 1 ZH, no del gate previo.

> **RESULTADO 0A ZH (post-ejecución, 2026-06-26)**: Familia A mean eta²=0.095 (top 0.419,
> Hseries_d5) vs EN 0.074 (top 0.385). Familia C control mean eta²=0.137 vs EN 0.026.
> **La especificidad ratio se invierte en ZH a nivel univariado pooled**: A/C = 0.69 en ZH
> vs 2.88 en EN. Hipótesis: las bandas espectrales del control en mandarín capturan info
> tonal léxica correlacionada con emoción. Caveat fuerte a declarar en el reporte
> cross-language; NO invalida Fase 1 ZH (pregunta distinta).

Si por presión de cómputo hay que elegir, 0A ZH puede correrse después del training Fase 1 ZH (o no correrse en absoluto en este corte). El plan lo incluye como side-task porque el CPU está disponible y el costo marginal de tener 0A ZH completo es bajo frente al valor diagnóstico que aporta.

### Replicación 1:1 de Fase 1 EN sobre ZH

Idéntica metodología, idéntica arquitectura, idénticos hiperparámetros. Lo único que cambia es el pool de hablantes y el idioma de las grabaciones. Los caches y outputs viven en directorios separados de la corrida EN para preservar independencia.

```
data/voz_expresiva/
├── wavlm_cache/            ← EN, intacto
├── descriptors_cache/      ← EN, intacto
├── 1/                      ← EN results, intactos (REPORTE_1.md cerrado)
├── 1_pre_calibfix/         ← snapshot forense EN original (pre fix B2)
├── 1_en_calibfix/          ← EN corregido (N-strict heredado + N-adapt rerun)
├── wavlm_cache_zh/         ← ~22 GB
├── descriptors_cache_zh/   ← ~250 MB
├── 1_zh/                   ← resultados ZH (manifest corregido fix B2)
└── 1_zh_pre_calibfix_partial/  ← snapshot forense run ZH abortado (87 records)
```

### Backbone

WavLM-large frozen, idéntico a Fase 1 EN. WavLM fue entrenado mayormente sobre habla inglesa, pero su corpus incluye VoxPopuli (multilingüe parlamentario) y GigaSpeech (variado). En la literatura WavLM funciona razonablemente sobre mandarín aunque por debajo de su rendimiento en inglés. El baseline `none` puede salir más bajo en ZH; eso es información válida, no problema metodológico. Lo que el experimento mide es Δ vs baseline, no UAR absoluto.

### Descriptor Familia A — sin cambios

V4-lin (4d) + H-series (8d) = 12d frame-level a 100 Hz, pooled a 50 Hz por mean de 2 frames consecutivos para alinear con WavLM. Mismo extractor (`compute_v4_linear`, `compute_h_series` en `src/bias_control/vocal_descriptors.py`), mismo F0 (librosa.pyin, fmin=75 Hz, fmax=500 Hz). No tocar el rango F0 — cubre el rango adulto mandarín y cambiarlo introduciría confounder en la comparación cross-language.

### Configs, normas, seeds, CV — sin cambios

4 configs (none, concat, film, xattn), 2 normas (N-strict + N-adapt con 1 calib repeat congelado seed=42), 3 seeds (42, 123, 456), LOSO 10-fold sobre los 10 ZH speakers (0001-0010), val_speaker = `speakers[(k+1) % 10]`. Total: 240 runs. Mismo training (AdamW lr=1e-3, batch 64, 30 epochs nominal, cosine decay con warmup, early stop val_UAR patience=5).

### Lectura cross-language (tres niveles de estatuto estadístico)

1. **Resultados ZH stand-alone (estatuto formal interno)**: tabla UAR + CKA + bootstrap de Δ_ZH per-mechanism vs WavLM-only ZH. 1000 resamples sobre 10 per-speaker values ZH, CI95 de Δ. Contraste formal dentro de ZH.
2. **Tabla descriptiva EN ↔ ZH (lectura visual, NO contraste formal)**: mostrar Δ y CI95 por idioma. Declarar explícitamente que solapamiento de CI95 NO es contraste formal de equivalencia — es heurística visual.
3. **Contraste exploratorio del shift mean(Δ_ZH) - mean(Δ_EN) per-mechanism (estatuto secundario)**: bootstrap independiente (no pareado — speakers distintos), 1000 resamples, CI95 del shift. Persistido en `cross_language_shift_bootstrap.json`.
4. **NO comparar UAR absolutos** entre EN y ZH como afirmación cross-language.

### Lecturas prefiguradas (4 escenarios)

- **Concat pasa también en ZH con CI95 robusto, magnitudes similares a EN**: target. Phideus transfiere cross-language sobre habla actuada.
- **Concat pasa en ZH pero con magnitud menor, CI excluye 0**: efecto presente pero atenuado. Hipótesis: tonalidad mandarina absorbe parte del señal armónico.
- **Concat no pasa en N-strict pero los tres pasan en N-adapt como en EN**: el régimen estricto no transfiere pero el adaptativo sí. Despliegue real necesitaría calibración per-speaker.
- **Ningún mecanismo pasa en ZH**: descriptor EN-específico o actuación ZH no capturada. Mirar eta² por dim, distribución F0 ZH vs EN, CKA. Registrar como hallazgo de alcance de la hipótesis.

## Correcciones aplicadas durante la ejecución (post-aprobación del plan)

Estas correcciones surgieron de auditoría Codex DURANTE la ejecución y modifican el plan original:

### Fix B2 — bug de calib_manifest (corrección crítica)

**Bug detectado**: `build_calib_manifest` reinstanciaba `RandomState(CALIB_SEED)` dentro del loop de speakers. Como los 10 speakers comparten inventario en mismo orden, `rng.choice` seleccionaba las MISMAS 25 sentence_ids para todos los speakers. La calibración seguía siendo per-speaker en features, pero la distribución emocional era fija e idéntica (5,4,6,5,5) en todos, no random independiente.

**Fix B2** (aplicado): seed derivada del speaker_id vía `sha256(f"{base_seed}:{spk}")`. Cada speaker recibe RNG independiente, robusto al orden del pool, robusto a agregar/quitar speakers. Se agrega `calib_seed_effective` al manifest y se propaga a `uar_results.json` adapt records.

**Decisión sobre EN** (consensuada con Codex y usuario): el bug afectaba solo N-adapt (N-strict no usa calibración). Para no introducir variable fantasma en la comparación cross-language, se decidió **regenerar EN con el fix** pero de forma B-partial:
- `1/` original preservado intacto + snapshot `1_pre_calibfix/`.
- Nuevo directorio `1_en_calibfix/`: N-strict heredado de `1/` (bit-exact, no afectado por el bug), N-adapt reruneado con fix B2.
- `1_en_calibfix/PROVENANCE.md` documenta la procedencia mixta.
- El cross-language compara ZH contra `1_en_calibfix/` (EN limpio), NO contra `1/`.

### Guardrails agregados a 1_report.py y 1_train.py

- `1_report.py:_validate_completeness`: aborta si los records no cubren exactamente `4 configs × 2 norms × 3 seeds × N speakers`, o si hay duplicados o combos faltantes. Previene reportes sobre experimentos truncados.
- `1_train.py:build_calib_manifest`: valida que un manifest cacheado coincida con el speaker_pool actual y con la política de seeding vigente (calib_seed_effective esperado). Aborta sobre manifest stale.

## Secuencia operativa corregida (post fix B2)

```
1. Kill train_zh viejo + snapshots forenses (1_pre_calibfix, 1_zh_pre_calibfix_partial)
2. Fix B2 + --limit-norms + propagar calib_seed_effective + guardrails
3. Smoke test ZH con manifest corregido (verifica N-strict bit-exact, N-adapt cambia)
4. Full LOSO ZH limpio → data/voz_expresiva/1_zh/                    ~7 h GPU
5. EN N-adapt partial rerun:
   - mkdir 1_en_calibfix/{embeddings,predictions}
   - copy N-strict embeddings+predictions+records de 1/
   - 1_train.py --limit-norms adapt --output-dir 1_en_calibfix      ~3.3 h GPU
   - merge N-strict records (con calib_seed_effective=None)
   - escribir PROVENANCE.md
6. Reportes (orden importa):
   - 1_report.py --results-dir 1_en_calibfix --label-self EN        (consolida EN primero)
   - 1_report.py --results-dir 1_zh --compare-against 1_en_calibfix \
                 --label-self ZH --label-other EN --output-name REPORTE_1_ZH.md
7. Cierre: README frente + NOTAS Codex (S61) + commit + push único
```

## Decisiones congeladas

| Eje | Valor |
|---|---|
| Backbone | WavLM-large frozen, last_hidden_state |
| Descriptor | V4-lin (4d) + H-series (8d), F0 75-500 Hz |
| F0 extractor | librosa.pyin, mismo que EN |
| Configs | none / concat / film / xattn |
| Normas | N-strict + N-adapt con 1 calib repeat |
| Calib seeding | **fix B2**: seed efectiva = sha256(f"{CALIB_SEED}:{spk}"), per-speaker independiente |
| Seeds | 42, 123, 456 |
| CV | LOSO 10-fold sobre 10 ZH speakers (0001-0010) |
| Training | AdamW lr=1e-3, wd=1e-4, batch 64, 30 epochs, cosine + warmup, early stop val_UAR patience=5 |
| Métrica primaria | UAR per-speaker mean sobre 3 seeds → mean ± std sobre 10 speakers |
| Bootstrap | 1000 resamples sobre 10 per-speaker values |
| CKA | Linear CKA utterance-level post-injection pre-head, excluye 25 calib en N-adapt |
| EN reference para cross-language | `1_en_calibfix/` (NO `1/`), procedencia mixta documentada |
| Commit policy | Único commit + push al cierre |

## Caveats que el reporte debe declarar

- Mandarín tonal: F0 codifica significado léxico. El descriptor podría capturar varianza tonal.
- 0A ZH: especificidad ratio univariada invertida (A/C=0.69 vs 2.88 EN). Caveat fuerte.
- WavLM entrenado mayormente sobre inglés; UAR absolutos no comparables directamente EN↔ZH.
- ESD actuado: generalización requiere Fase 3.
- Cross-language EN↔ZH entre poblaciones distintas; atribución a "idioma" no estricta.
- N-adapt con 1 calib repeat: lectura secundaria menos estable que N-adapt de 0B (3 repeats).
- Bootstrap n=10: señal, no prueba fuerte.
- EN reference es `1_en_calibfix` con procedencia mixta (N-strict heredado, N-adapt post-fix).
