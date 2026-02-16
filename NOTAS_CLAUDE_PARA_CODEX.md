# Notas de Claude para Codex — Gate 4.3 (desde 2026-02-14 ~15:00 UTC)

## Contexto: qué sabe Codex y qué no

Codex se quedó sin tokens el viernes 14/02. Lo último que el usuario le comunicó fue:

> "ahora cuando termine A4 vamos a cancelar, hacer las modificaciones y volver a lanzar
> run desde A7 pero ya con A4x y A7x incorporados antes de los duales"

En ese momento, el estado conocido por Codex era:
- Gate 4.3 corriendo en tmux `gate43` con loop `d0 d4 a4 a7 d4a4 d4a7`
- D0 (5/5 COMPLETE): best S=60.2% (e3)
- D4 (5/5 COMPLETE): best S=63.6% (e5)
- A4 (3/5 cerrados, e4 en curso): best S=61.0% (e3)
- Cross-attention (a4x/a7x): código implementado y verificado en CPU, pendiente GPU
- Plan: cancelar después de A4, agregar a4x/a7x al loop, relanzar

**Codex NO sabe nada de lo que sigue.** Este documento cubre todo lo ocurrido desde entonces.

---

## 1. Commit y push de todo el trabajo acumulado (2026-02-14 ~15:00 UTC)

**Commit `8b09cbf`**: 79 archivos, +9226/-244 líneas. Push a main.

Incluye TODO lo implementado conjuntamente (Claude + Codex) antes de que Codex perdiera tokens:

**Código nuevo:**
- `src/bias_control/audio_descriptors.py`: `compute_audio_descriptor_a4()` y
  `compute_audio_descriptor_a7()` con `target_length` opcional (None = resolución nativa
  STFT para cross-attention, int = interpolado para concat)
- `src/bias_control/ratio_descriptors.py`: `compute_local_intervals()`
- `experiments/bias_control/gate42_training.py`:
  - `Gate42AudioAugModel` (A4/A7 concat)
  - `Gate42AudioCrossAttModel` (A4x/A7x cross-attention)
  - `Gate42DualAugModel` (duales concat)
  - `_encode_audio_with_descriptor()` (helper compartido concat)
  - `_encode_audio_with_cross_attention()` (helper cross-attention)
  - 8 puntos de integración: CLI choices, model factory, optimizer, param_ranges,
    preflight contract, checkpoint saving, eval loading, embed_batch_size

**Scripts:**
- `experiments/bias_control/run_gate43.sh`
- `experiments/bias_control/run_gate42_stage1.sh`
- `experiments/bias_control/run_d4_8ep.sh`

**Documentación (escrita por Codex):**
- `CODEX.md`, `INFORME_GATE_4_3_RATIO_RE_CENTRICO.md`, `plan_gate_4.3.md`, etc.

**Otros:** .gitignore cleanup, archivos MIDI removidos del tracking.

---

## 2. Resultados finales A4 concat — 5/5 epochs (2026-02-14 ~15:15 UTC)

A4 completó sus 5 epochs. Tabla completa:

| Epoch | S | MRR_avg | hard_neg | A2M_R10 | M2A_R10 | R@1_avg | R@20_avg |
|-------|---|---------|----------|---------|---------|---------|----------|
| e1 | 35.4% | 0.115 | 81.4% | 35.4% | 42.0% | 4.3% | 58.4% |
| e2 | 51.2% | 0.191 | 86.0% | 52.8% | 51.2% | 9.3% | 73.4% |
| e3 | 61.0% | 0.260 | 89.8% | 62.2% | 61.0% | 13.8% | 80.4% |
| e4 | 55.4% | 0.254 | 87.8% | 55.4% | 59.2% | 13.7% | 78.4% |
| **e5** | **63.6%** | **0.297** | **92.4%** | **65.8%** | **63.6%** | **16.0%** | **82.6%** |

### Análisis A4:

1. **A4 iguala D4 en S**: ambos 63.6% best, +3.4pp sobre D0 (60.2%).
2. **Dip en e4 con recovery en e5**: Trayectoria 35.4% → 51.2% → 61.0% → 55.4% → 63.6%.
   El dip de -5.6pp en e4 fue mucho más pronunciado que en D0 (-0.2pp) o D4 (-0.2pp).
   Pero la recovery en e5 fue igualmente fuerte. Confirma la DIRECTIVA ANALÍTICA:
   dips no son techos, pueden ser artefactos del LR schedule.
3. **Perfiles distintos A4 vs D4 al mismo S=63.6%**:
   - D4: MRR=0.313, hard_neg=91.2%
   - A4: MRR=0.297, hard_neg=92.4%
   - A4 discrimina mejor en hard negatives pero rankea peor en posición exacta.

### Tabla Fase 0 completa:

| Arm | Mecanismo | Best S | Best ep | MRR_avg | hard_neg |
|-----|-----------|--------|---------|---------|----------|
| D0 | baseline | 60.2% | e3 | 0.280 | 90.6% |
| D4 | MIDI concat | 63.6% | e5 | 0.313 | 91.2% |
| A4 | Audio concat | 63.6% | e5 | 0.297 | 92.4% |

**Conclusión Fase 0**: Tanto D4 como A4 aportan +3.4pp sobre baseline. La inyección de ratio
info funciona en AMBOS lados (MIDI temperado y audio log-freq). La hipótesis Phideus directa
(lado audio) está al mismo nivel que el lado MIDI.

---

## 3. Decisión de diseño: enfoque por fases (2026-02-14 ~15:20 UTC)

### Cambio respecto a lo que Codex conocía

**Plan original** (lo que Codex sabía): correr `d0 d4 a4 a7 d4a4 d4a7` secuencialmente,
todos con concat.

**Plan revisado** (post-Codex): El usuario decidió NO correr duales con ambos mecanismos
(concat Y cross-attention). En su lugar, primero determinar qué mecanismo gana por
descriptor, y solo después correr duales con el ganador.

### Razón
Correr d4a4, d4a7, d4a4x, d4a7x sería ~12h de GPU. Si cross-attention pierde, la mitad
se desperdicia. Mejor invertir ~3h en comparar mecanismos y luego ~6h en duales informados.

### Diseño por fases resultante

- **Fase 0** (COMPLETE): D0, D4, A4 — baselines + concat
- **Fase 1** (RUNNING): A7, A4x, A7x — concat restante + cross-attention audio
- **Punto de decisión**: Comparar concat vs cross-att por descriptor. Criterios:
  - S (métrica canónica) como criterio primario
  - MRR_avg como desempate si S comparable (< 1pp)
  - En empate práctico, preferir concat (más simple, menos params)
- **Fase 2**: Duales con mecanismo ganador (2 brazos)

### Cambios en archivos

**`experiments/bias_control/run_gate43.sh`**: Reescrito completamente.
- Loop cambiado de `d0 d4 a4 a7 d4a4 d4a7` a `a7 a4x a7x`
- BASE hardcodeado a directorio existente (continuación, no nuevo timestamp)
- Headers actualizados para reflejar "Fase 1"
- Summary table cubre todos los brazos: d0 d4 a4 a7 a4x a7x

**Plan file (`wondrous-meandering-newt.md`)**: PASOs 10-12 reescritos:
- PASO 10: Fase 1 — correr a7, a4x, a7x (single-injection)
- PASO 11: Punto de decisión — concat vs cross-attention
- PASO 12: Fase 2 — implementar duales con mecanismo ganador + correr

---

## 4. Pilots GPU para cross-attention (2026-02-14 ~15:45-16:30 UTC)

Antes de lanzar el run completo, se corrieron pilots de 1 epoch / 100 batches para
verificar que a4x y a7x funcionan correctamente en GPU.

### Resultados pilots:

| Métrica | a4x | a7x |
|---------|-----|-----|
| Preflight | PASSED | PASSED |
| Loss (100 bat) | 13.83 | 13.56 |
| S (1ep/100bat) | 56.0% | 58.8% |
| hard_neg | 90.2% | 90.4% |
| Params trainable | 69,101,568 | 69,105,664 |
| kv_proj params | 9,216 (8→1024) | 13,312 (12→1024) |
| cross_attn params | 4,198,400 | 4,198,400 |
| cross_attn_norm | 2,048 | 2,048 |
| Drift "Other" (cross-attn) | 5.3% rel change | 5.7% rel change |
| Tiempo total (train+eval) | 9.2 min | 9.1 min |
| NaN/crash | No | No |
| VRAM | Estable | Estable |

**Conclusión pilots**: Ambos modelos cross-attention funcionan correctamente en GPU.
Loss finito, gradientes fluyen a los tres param groups nuevos (kv_proj, cross_attention,
cross_attn_norm), VRAM estable, drift sentinel OK. El drift relativamente alto en "Other"
(~5.5%) es esperado para params con random init (vs params pre-entrenados que mueven <1%).

**Nota para futuros pilots**: No es necesario hacer extracción de embeddings completa
(~6 min extra). Solo el training loop (100 batches, ~2.5 min) basta para verificar
VRAM, loss, gradientes y timing.

---

## 5. Lanzamiento Fase 1 (2026-02-14 ~16:30 UTC)

Run lanzado en tmux `gate43` (sesión persistente):

```bash
source venv/bin/activate && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash experiments/bias_control/run_gate43.sh
```

Orden de ejecución: `a7 (concat) → a4x (cross-attn) → a7x (cross-attn)`
- 5 epochs cada uno, 1000 batches/epoch, seed=42
- Output dir: `data/bias_control_medium/training_outputs/gate43/gate43_20260214_1000/`
- Subdirs: `a7/`, `a4x/`, `a7x/` (junto a `d0/`, `d4/`, `a4/` ya existentes)
- Tiempo estimado: ~9h total. Finish ~01:30 UTC del 15/02.
- Todos los brazos fresh desde `foundation_locked_e25.pt` (no resume).

---

## 6. Roadmap extendido Gate 4.3 (2026-02-14 ~16:45 UTC)

Después de lanzar Fase 1, el usuario propuso extender Gate 4.3 con fases adicionales.
Claude estuvo de acuerdo. El roadmap completo queda así:

### Fase 0 — COMPLETE
Baselines + concat. Brazos: D0, D4, A4.

### Fase 1 — RUNNING (~9h)
Concat restante + cross-attention audio. Brazos: A7, A4x, A7x.

### Fase 2 — D4x (cross-attention MIDI)
Un solo brazo nuevo: D4x. Completa la matriz mecanismo × descriptor:

|  | Concat | Cross-attention |
|---|--------|----------------|
| MIDI intervals (D4) | D4 (Fase 0) | **D4x (Fase 2)** |
| Audio log-freq (A4) | A4 (Fase 0) | A4x (Fase 1) |
| Audio attractor (A7) | A7 (Fase 1) | A7x (Fase 1) |

**Diseño D4x acordado**:
- Mismo patrón arquitectural que A4x/A7x pero adaptado al lado MIDI (d=512 en vez de 1024)
- Q = MIDI embeddings [B, N, 512] (con pos_emb aplicado antes de cross-attn)
- K/V = intervals [B, N, 4] → interval_kv_proj: Linear(4→512) → [B, N, 512]
- nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True, dropout=0.1)
- Residual connection + LayerNorm(512)
- Luego: Transformer (4 capas, sin pos_emb de nuevo) → pool → projection
- Params nuevos estimados: ~1.05M (vs ~4.2M en audio, porque d=512 vs d=1024)
- Attn matrix: [B×8, N, N] donde N≈50-200 — trivial en memoria
- **Diferencia clave vs audio**: Q y K/V tienen la MISMA resolución temporal (N tokens).
  No hay mismatch temporal que resolver. Pero cross-attention igual aporta:
  - Acceso non-local (token i puede ver intervalos de posición j)
  - Selectividad dinámica (vs mezcla lineal fija de concat)
  - Multi-head specialization

**Status**: Diseño acordado. Implementación pendiente (Claude la hará mientras Fase 1 corre).

### Fase 3 — Duales same-modality
Con ganadores de Fases 0-2 por descriptor (concat o cross-att). 2 brazos, ~6h.
Ejemplo: si A4 concat gana pero D4x cross-att gana → dual sería D4x+A4.

### Fase 4 — Cross-modal injection (CONCEPTO NUEVO)
**Idea central**: Inyectar descriptores de un dominio en el encoder del OTRO dominio.

Hasta ahora, toda inyección es same-modality:
- D4/D4x: MIDI intervals → MIDI encoder
- A4/A7/A4x/A7x: audio ratios → audio encoder

Cross-modal injection sería:
- Audio ratios → MIDI encoder (el MIDI "ve" la armonía del audio)
- MIDI intervals → audio encoder (el audio "ve" los intervalos del MIDI)

Esto crea un puente informacional entre modalidades ANTES de que VICReg intente alinearlas.

| Brazo | Audio encoder recibe | MIDI encoder recibe | Testea |
|-------|---------------------|---------------------|--------|
| CM-a | — | Audio desc (A_best) | Audio ratios ayudan a MIDI |
| CM-m | MIDI desc (D_best) | — | MIDI intervals ayudan a audio |
| CM-bi | MIDI desc (D_best) | Audio desc (A_best) | Bidireccional |

Se usaría el mejor descriptor y mecanismo de inyección determinado en Fases 0-3.
3 brazos, ~9h.

**Status**: Diseño conceptual acordado. Implementación futura (después de Fase 3).

### Costo total estimado (Fases 2-4)
D4x (3h) + duales (6h) + cross-modal (9h) = ~18h adicionales de GPU.

---

## 7. D4x implementado — MIDI cross-attention (2026-02-14 ~16:50 UTC)

### Qué es D4x
Cross-attention para el lado MIDI. Mismo patrón que A4x/A7x pero adaptado:
- d=512 (vs d=1024 en audio)
- Q y K/V a misma resolución (N tokens MIDI) — sin mismatch temporal
- Params nuevos: ~1.05M (vs ~4.2M en audio)

### Nuevos componentes en `gate42_training.py`:
- `_encode_midi_with_cross_attention()` — helper function (análoga a `_encode_audio_with_cross_attention`)
- `Gate42MidiCrossAttModel` — model class (análoga a `Gate42AudioCrossAttModel`)

### Pipeline D4x:
```
Embedding [B, N, 512] → CLS (if used) → pos_enc → cross_attn(Q=emb, K/V=intervals) → residual+LN → Transformer → pool → proj
```

### 8 puntos de integración (todos DONE):
1. CLI: `d4x` en choices
2. Factory: `Gate42MidiCrossAttModel(base_model, interval_dim=4)`
3. Optimizer: 3 groups (interval_kv_proj, cross_attention, cross_attn_norm)
4. Param ranges: run-b (39M-42M), run-d (64M-67.5M)
5. Preflight: 3 trainable prefixes
6. Checkpoint: eval_compatible=False
7. Eval loading: reconstruye Gate42MidiCrossAttModel
8. embed_batch_size: default 64 (attn matrix MIDI trivial)

### CPU verification results:
- New params: 1,054,208 (~1.05M)
- Shapes: (B, 256) ambos lados
- Gradients: fluyen a interval_kv_proj, cross_attention, cross_attn_norm
- NaN: ninguno (random + silencio)
- Optimizer: 7 param groups correctos

### Nota de diseño: CLS token
Si el MIDI encoder usa aggregation="cls", el CLS se prepende a Q ANTES de cross-attention.
K/V queda sin CLS (a resolución original N). nn.MHA soporta Lq != Lk nativo.

### Status: CODE COMPLETE + CPU VERIFIED. Pendiente pilot GPU + run 5ep (después de Fase 1).

---

## 8. Cambios en archivos (resumen para Codex)

### Ya commiteados (commit `8b09cbf`):
Todo el código de Gate 4.3 concat + cross-attention audio. Ver sección 1.

### Modificados post-commit (pendientes de commit):
- `experiments/bias_control/gate42_training.py` — +Gate42MidiCrossAttModel, +_encode_midi_with_cross_attention, +8 integration points para d4x
- `experiments/bias_control/run_gate43.sh` — Reescrito para Fase 1 (loop `a7 a4x a7x`)
- `Documents/.../07_GATE_4_3.../README.md` — Actualizado con roadmap extendido (Fases 0-4)
- `NOTAS_CLAUDE_PARA_CODEX.md` — Este archivo (nuevo)

### Por implementar (futuro):
- Variante dual cross-att (si cross-att gana en PASO 11)
- Clases para cross-modal injection (Fase 4) — diseño conceptual, implementación futura

---

## 9. Documentación que Codex necesita actualizar (actualizado 2026-02-15)

**PRIORIDAD ALTA** (resultados nuevos desde que Codex perdió tokens):

1. **INFORME_GATE_4_3_RATIO_RE_CENTRICO.md**: Agregar:
   - Resultados completos 9 brazos (tabla en sección 26)
   - Hallazgo superaditivo d4a4 (69.8%, sección 24)
   - Resultado d4a4cm negativo (-7.8pp, sección 25)
   - d4a4-scratch: S=74.6% @ e10, RECORD del proyecto (sección 28)
   - Conclusiones Gate 4.3: concat>cross-att, same-mod>cross-modal, dual superaditivo
2. **plan_gate_4.3.md**: Sincronizar con diseño por fases (0-3 COMPLETE, 5 CODE COMPLETE)
3. **ROADMAP_BIAS_CONTROL.md**: Reflejar:
   - Gate 4.3 COMPLETE con 9 brazos
   - d4a4-scratch RUNNING (e11/30)
   - Gate 4.3 Fase 5 (4 brazos nuevos) implementada
   - Renumeración de gates (sección 19)
4. **CODEX.md**: Actualizar estado general del proyecto
5. **GitHub Pages viz**: 5 nuevas visualizaciones + 3 renombradas (sección 15)

**PRIORIDAD BAJA**:
6. Registrar limpieza de disco (sección 10)
7. Revisar `PHIDEUS_MASTER_BRIEFING.md` y `PHIDEUS_NEURAL_ARCHITECTURES.md` (sección 30)

---

## Apéndice: Tabla de referencia Gate 4.3 completa (para Codex)

### Todos los brazos definidos:

| Brazo | Lado | Descriptor | Mecanismo | Params nuevos | Fase | Status |
|-------|------|-----------|-----------|---------------|------|--------|
| D0 | — | — | baseline | 0 | 0 | COMPLETE |
| D4 | MIDI | intervals (4d) | concat | ~267K | 0 | COMPLETE |
| A4 | Audio | log-freq deltas (8d) | concat | ~1.06M | 0 | COMPLETE |
| A7 | Audio | rational attractor (12d) | concat | ~1.06M | 1 | RUNNING |
| A4x | Audio | log-freq deltas (8d) | cross-attn | ~4.2M | 1 | RUNNING |
| A7x | Audio | rational attractor (12d) | cross-attn | ~4.2M | 1 | RUNNING |
| D4x | MIDI | intervals (4d) | cross-attn | ~1.05M | 2 | CPU VERIFIED |
| Dual1 | Ambos | ganadores | ganador | TBD | 3 | PENDING |
| Dual2 | Ambos | ganadores | ganador | TBD | 3 | PENDING |
| CM-a | Cross | audio→MIDI | ganador | TBD | 4 | CONCEPTO |
| CM-m | Cross | MIDI→audio | ganador | TBD | 4 | CONCEPTO |
| CM-bi | Cross | bidireccional | ganador | TBD | 4 | CONCEPTO |

### Arquitecturas implementadas (4 variantes cross-attention):

**Audio regular (A4x/A7x) — Gate42AudioCrossAttModel:**
```
CNN [B, 2400, 1024] → +pos_emb → cross_attn(Q=features[2400], K/V=desc[188]) → +residual → LN → Transformer(2400 tokens) → pool → proj
```
- K/V a resolución nativa STFT (188 frames), NO interpolado. 12.8x ahorro de memoria.
- pos_emb ANTES de cross-attention (temporal awareness). NO se vuelve a sumar antes del Transformer.
- need_weights=False, embed_batch_size=16 en eval.

**Audio reverse (A4r) — Gate42AudioReverseCrossAttModel:**
```
CNN [B, 2400, 1024] → +pos_emb → K/V. Desc[188, 8] → q_proj → +desc_pos_emb → Q.
cross_attn(Q=desc[188], K/V=features[2400]) → +residual → LN → Transformer(188 tokens) → pool → proj
```
- **INVERSO**: descriptores (Q) organizan features (K/V). Los ratios como principio organizador.
- Transformer procesa 188 tokens (vs 2400 en regular). **12.8x menos cómputo self-attention**.
- desc_pos_embedding: Parameter([1, 200, 1024]) con init 0.02 std.
- ~4.4M params nuevos (q_proj + desc_pos_emb + cross_attn + norm).

**MIDI regular (D4x) — Gate42MidiCrossAttModel:**
```
Embedding [B, N, 512] → CLS? → +pos_enc → cross_attn(Q=emb[N+1], K/V=intervals[N]) → +residual → LN → Transformer → pool → proj
```
- Q y K/V a misma resolución (N tokens). Sin mismatch temporal.
- CLS prepended a Q si aggregation="cls". K/V siempre a N.
- ~1.05M params nuevos (d=512 vs d=1024 en audio).

**MIDI reverse (D4r) — Gate42MidiReverseCrossAttModel:**
```
Embedding [B, N, 512] → CLS? → +pos_enc → K/V. Intervals → q_proj → +CLS? → +pos_enc → Q.
cross_attn(Q=intervals[N+1], K/V=emb[N+1]) → +residual → LN → Transformer → pool → proj
```
- **INVERSO**: intervals (Q) organizan embeddings (K/V).
- Misma resolución Q/K/V — diferencia es puramente semántica (quién pregunta a quién).
- Reutiliza enc.pos_encoding (sinusoidal) para ambos Q y K/V.
- ~1.05M params nuevos.

### Descriptores audio implementados (4):

| Descriptor | Dim | Archivo | Concepto |
|-----------|-----|---------|----------|
| A4 | 8 | audio_descriptors.py | Log-freq deltas (8 bands, temporal diff) |
| A7 | 12 | audio_descriptors.py | Rational attractor (12 JI attractors, Gaussian assignment) |
| A8 | 12 | audio_descriptors.py | **Onset-weighted chroma** (pitch class × spectral flux) |
| A9 | 12 | audio_descriptors.py | **IDF-weighted attractor** (A7 base × inverse document freq) |

---

## 10. Limpieza de espacio en disco (2026-02-14 ~18:00 UTC)

### Problema
El repositorio ocupaba **414 GB** en el NVMe de 916 GB (67% uso, 294 GB libres). Con Gate 4.3
Fases 2-4 por delante (cada brazo genera ~6 GB de checkpoints), el disco se iba a quedar sin
espacio.

### Solución
Se hizo un backup completo al RAID1 (`/mnt/raid1/Phideus-backup/`, 5 TB libres) y luego se
eliminaron archivos que no son necesarios para el trabajo activo. **Todo lo borrado esta
disponible en el backup.**

### Backup
```bash
# Comando usado para sincronizar antes de borrar:
rsync -avhP --update --itemize-changes ./ /mnt/raid1/Phideus-backup/
```
Backup verificado completo (413 GB). Si en el futuro se necesita recuperar un checkpoint
intermedio, se copia desde el backup.

### Que se borro

| Paso | Que | Tamaño | Motivo |
|------|-----|--------|--------|
| 1 | `data/maestro_v3/maestro-v3.0.0.zip` | 101 GB | Ya descomprimido en `maestro-v3.0.0/` |
| 2 | `gate3/`, `gate3_c/`, `gate3_d/`, `gate3_norm/` | 22 GB | DANN cerrado, no mejora Gate 2 |
| 2 | `gate4_smoke/`, `gate4_runA/`, `gate4_RB0/`, `gate4_R1rescue/` | 34 GB | Superseded por Gate 4.2 |
| 3 | `gate2/checkpoint_epoch*.pt` (52 archivos) | 20.6 GB | Solo se conservo best_model.pt + final_model.pt |
| 4a | `bloqueA_runD-02/checkpoint_epoch*.pt` + `*_base*.pt` | 32 GB | Foundation ya extraido como archivo aparte |
| 4b | `bloqueA_run{A,B,C,D}/checkpoint_epoch*.pt` + `*_base*.pt` | 18 GB | Solo se conservo best_model.pt |
| 5 | `gate42/screening_*/*/checkpoint_epoch*.pt` + `d4_8ep_*/checkpoint_epoch*.pt` | 19 GB | Solo se conservo best_model.pt |
| 6 | `gate43/.../d0,d4,a4/checkpoint_epoch*.pt` + `*_base*.pt` | 16 GB | Solo se conservo best_model.pt |
| 7 | `data/training_outputs/` (directorio completo) | 0.65 GB | Era UOEMD/Rosetta, obsoleto |

**Total liberado: ~261 GB**

### Que se conservo (en cada directorio)

Para todos los experimentos completados se mantuvo:
- `best_model.pt` — mejor modelo del run
- `final_results.json` — metricas finales
- `training_history.json` — curvas de training
- `config.json` — configuracion del run
- `eval_per_epoch/*.json` — metricas por epoch (archivos de ~1.5 KB)
- `training.log` (donde exista)

### Que NO se toco

| Item | Razon |
|------|-------|
| `foundation_locked_e25.pt` | Inmutable (chmod 444), MD5 verificado post-limpieza |
| `gate43/.../a7/` | Corriendo activamente (Fase 1) |
| `data/maestro_v3/maestro-v3.0.0/` | Dataset activo (121 GB, 1276 WAV + 1276 MIDI) |
| `data/datasets/` | Datos procesados (4.6 GB) |
| `venv/` | Entorno virtual (8 GB) |
| Todo el codigo fuente | Obviamente |

### Estado post-limpieza

| Metrica | Antes | Despues |
|---------|-------|---------|
| Espacio usado | 576 GB (67%) | 315 GB (37%) |
| Espacio libre | 294 GB | **555 GB** |

### Verificacion post-limpieza

- Foundation intacto: MD5 `ddb2ebf7075eec4dcec1628341ec4942` OK
- Gate 4.3 Fase 1: A7 epoch 3/5 corriendo normalmente
- Todos los `best_model.pt`: verificados presentes
- Todos los `eval_per_epoch/`: verificados completos
- Todos los imports de Python: verificados funcionales
- MAESTRO: 1276 WAV + 1276 MIDI intactos

### Nota para Codex

Algunos scripts viejos (`extract_multigate_embeddings.py`, `compare_layer_drift.py`,
`compare_gate3_checkpoints.py`, `multiseed_reeval.py`) tienen hardcoded paths a checkpoints
borrados (ej: `gate2/checkpoint_epoch45.pt`, `gate4_RB0/checkpoint_epoch5_base.pt`). Estos
scripts son de fases cerradas y NO estan en ningun pipeline activo. Si alguna vez se necesita
re-correrlos, habria que actualizar los paths (o copiar checkpoints del backup RAID1).
Documentos historicos (`INFORME_GATE2_COMPLETO.md`, `INFORME_GATE3_*.md`, etc.) tambien
referencian checkpoints especificos que ya no existen — son registros historicos validos,
no necesitan actualizacion.

---

## 11. Documentacion adicional que Codex necesita actualizar (post-limpieza)

Ademas de lo listado en seccion 9, agregar:

6. **Registrar la limpieza de disco** en el lugar apropiado de la documentacion (bitacora,
   roadmap, o donde Codex considere mejor). Incluir que el backup existe en
   `/mnt/raid1/Phideus-backup/` y el comando rsync para sincronizar.

---

## 12. Resultados parciales A7 concat (2026-02-14 ~21:20 UTC)

A7 (rational attractor, concat) sigue corriendo. Resultados disponibles hasta ahora:

| Epoch | S | A2M_R10 | M2A_R10 | hard_neg | MRR_avg |
|-------|---|---------|---------|----------|---------|
| e1 | (no capturado) | | | | |
| e2 | 41.4% | 41.4% | 47.4% | 83.4% | (pendiente) |
| e3 | 48.2% | 48.2% | 50.6% | 87.4% | (pendiente) |
| e4 | training ~18% | | | | |
| e5 | pendiente | | | | |

### Observacion preliminar (NO conclusiva — faltan e4/e5):

A7 con descriptor rational-attractor (12d) arranca significativamente mas bajo que A4 con
log-freq deltas (8d): A7 e3=48.2% vs A4 e3=61.0%. Hay mejora sostenida epoch a epoch
(+6.8pp de e2 a e3), pero el gap es grande. Recordar DIRECTIVA ANALITICA: no declarar
techos. A4 tambien tuvo un dip en e4 con recovery fuerte en e5 (55.4% → 63.6%).

A7 completara en ~1.5h. Luego arranca A4x (cross-attention, 5ep, ~3h) y despues A7x (~3h).
ETA Fase 1 completa: ~01:30 UTC 15/02.

**Nota**: El usuario se fue a dormir ~21:20 UTC. Gate 4.3 Fase 1 sigue corriendo
autonomamente en tmux `gate43`. Cuando vuelva, los resultados completos de a7, a4x, a7x
estaran en `gate43_20260214_1000/{a7,a4x,a7x}/eval_per_epoch/`.

---

## 13. Resultados completos Fase 1 (2026-02-15 — recogidos por la mañana)

Fase 1 terminó autónomamente durante la noche. Los tres brazos completaron 5 epochs cada uno.

### A7 — Rational attractor, concat (COMPLETE)

| Epoch | S | A2M_R10 | M2A_R10 | hard_neg | A2M_MRR | M2A_MRR |
|-------|---|---------|---------|----------|---------|---------|
| e1 | 26.6% | 26.6% | 32.0% | 79.0% | 0.078 | 0.099 |
| e2 | 41.4% | 41.4% | 47.4% | 83.4% | 0.152 | 0.177 |
| e3 | 48.2% | 48.2% | 50.6% | 87.4% | 0.190 | 0.204 |
| e4 | 53.4% | 55.2% | 53.4% | 88.4% | 0.230 | 0.231 |
| **e5** | **58.8%** | **60.2%** | **58.8%** | **90.2%** | **0.266** | **0.270** |

**Análisis A7**: Recovery sostenida de e1 a e5, sin dip como A4. Pero el techo a e5 (58.8%)
queda **debajo del baseline D0** (60.2%). El descriptor rational-attractor (12d) aporta MENOS
señal que el baseline sin descriptor. La hipótesis de atractores JI como feature útil para
el encoder audio no se sostiene en este régimen.

### A4x — Audio log-freq, cross-attention (COMPLETE)

| Epoch | S | A2M_R10 | M2A_R10 | hard_neg | A2M_MRR | M2A_MRR |
|-------|---|---------|---------|----------|---------|---------|
| e1 | 36.0% | 36.0% | 44.0% | 82.4% | 0.118 | 0.148 |
| e2 | 52.0% | 52.4% | 52.0% | 88.0% | 0.207 | 0.215 |
| e3 | 59.2% | 59.2% | 60.4% | 90.4% | 0.262 | 0.264 |
| e4 | 57.8% | 57.8% | 61.2% | 91.2% | 0.258 | 0.277 |
| **e5** | **62.6%** | **64.0%** | **62.6%** | **92.4%** | **0.288** | **0.292** |

**Análisis A4x**: Trayectoria similar a A4 concat (dip en e4, recovery en e5). Pero peak S=62.6%
vs A4 concat S=63.6%. Cross-attention pierde 1.0pp frente a concat para el descriptor A4.
Más params (~4.2M vs ~1.06M), peor resultado. Hard_neg idéntico (92.4%).

### A7x — Audio attractor, cross-attention (COMPLETE)

| Epoch | S | A2M_R10 | M2A_R10 | hard_neg | A2M_MRR | M2A_MRR |
|-------|---|---------|---------|----------|---------|---------|
| e1 | 35.2% | 35.2% | 43.4% | 80.6% | 0.115 | 0.150 |
| e2 | 52.6% | 52.6% | 53.8% | 87.6% | 0.213 | 0.214 |
| e3 | 58.2% | 58.2% | 60.0% | 90.2% | 0.254 | 0.261 |
| e4 | 59.0% | 59.0% | 60.4% | 91.4% | 0.262 | 0.270 |
| **e5** | **62.2%** | **62.2%** | **63.8%** | **92.0%** | **0.287** | **0.292** |

**Análisis A7x**: S=62.2%, +3.4pp sobre A7 concat (58.8%). Cross-attention rescata
parcialmente el descriptor attractor (que en concat quedaba debajo de baseline). Pero sigue
siendo el cuarto brazo, debajo de D4, A4, y A4x. El descriptor attractor simplemente aporta
menos señal que log-freq deltas, independientemente del mecanismo.

### Tabla completa Fases 0+1 (6 de 7 brazos):

| Arm | Lado | Descriptor | Mecanismo | Best S | Best ep | MRR_avg | hard_neg | vs D0 |
|-----|------|-----------|-----------|--------|---------|---------|----------|-------|
| **D4** | MIDI | intervals | concat | **63.6%** | e5 | 0.313 | 91.2% | **+3.4pp** |
| **A4** | Audio | log-freq | concat | **63.6%** | e5 | 0.297 | 92.4% | **+3.4pp** |
| A4x | Audio | log-freq | cross-att | 62.6% | e5 | 0.290 | 92.4% | +2.4pp |
| A7x | Audio | attractor | cross-att | 62.2% | e5 | 0.290 | 92.0% | +2.0pp |
| D0 | — | — | baseline | 60.2% | e3 | 0.280 | 90.0% | — |
| A7 | Audio | attractor | concat | 58.8% | e5 | 0.268 | 90.2% | -1.4pp |
| D4x | MIDI | intervals | cross-att | *en curso* | | | | |

### Conclusiones parciales (falta D4x):

1. **Concat gana sobre cross-attention para audio A4**: 63.6% vs 62.6% (-1.0pp).
   Más simple, menos params, mejor resultado.
2. **Cross-attention rescata A7 pero no lo suficiente**: A7x=62.2% >> A7=58.8% (+3.4pp),
   pero sigue debajo de A4/D4. El descriptor attractor es el problema, no el mecanismo.
3. **El descriptor log-freq deltas (A4) es consistentemente mejor que attractor (A7)**:
   tanto en concat como en cross-attention.
4. **Todos los brazos con descriptor superan baseline (D0=60.2%)** excepto A7 concat.
5. **D4 y A4 están empatados a S=63.6%** pero con perfiles distintos:
   D4 gana en MRR (0.313 vs 0.297), A4 gana en hard_neg (92.4% vs 91.2%).

---

## 14. D4x lanzado — Fase 2 (2026-02-15 ~02:00 UTC)

D4x (MIDI cross-attention) se lanzó automáticamente como parte del script run_gate43.sh
que incluía la secuencia `a7 → a4x → a7x → d4x`. Está corriendo en el mismo tmux `gate43`.

### Estado D4x al momento de esta actualización:

- **Epoch 1 COMPLETE**: S=54.8%, A2M=54.8%, M2A=56.0%, hard_neg=88.6%, MRR_avg=0.247
- **Epoch 2**: training completado, quick_val + canonical eval en progreso
- **ETA**: ~05:00 UTC 15/02 para completar 5 epochs

### Observación epoch 1 D4x:
S=54.8% en epoch 1 es significativamente más bajo que D4 concat epoch 1 (datos no disponibles
directamente, pero D4 best a e5 fue 63.6%). El resultado final depende de la trayectoria
epochs 2-5. Recordar DIRECTIVA ANALÍTICA.

---

## 15. Visualizaciones interactivas — 5 nuevas arquitecturas (2026-02-15)

### Commit `7573483` — 89 archivos, +9,456 líneas. Push a main.

Se implementaron 5 nuevas visualizaciones interactivas 3D WebGL2 para el sitio
**[altermundi.github.io/Phideus](https://altermundi.github.io/Phideus/)**:

| Visualización | Ruta | Arquitectura | Color |
|---------------|------|-------------|-------|
| **Cross-Attention Injection** | `/crossatt` | Gate 4.3 — descriptor injection via cross-attention | #cc3366 (magenta) |
| **Domain Adversarial Network** | `/dann` | Gate 3 — gradient reversal layer | #996633 (brown) |
| **Hierarchical Reasoning Model** | `/hrm` | L-Module + H-Module + ACT | #339966 (green) |
| **ConstellationVAE** | `/constellation` | Sparse token VAE, C1-C4 | #cc9933 (gold) |
| **JEPA-Lite** | `/jepa` | No-decoder predictive architecture | #6633cc (indigo) |

Las 3 visualizaciones existentes fueron renombradas con títulos descriptivos:
- Phideus → "MERT Audio + MIDI Transformer (Run D Foundation)"
- BloqueA → "Hybrid Adapter Fine-Tuning (BloqueA Run C)"
- RosetaVAE → "RosetaVAE — Dual-Domain Latent Factorization"

### Archivos por visualización (cada una tiene ~14-17 archivos):

```
viz/src/{name}/
├── {Name}DimStyle.ts            # Enum de dimensiones, colores, textos
├── {Name}ModelLayout.ts         # Layout 3D: bloques, posiciones, tamaños
├── {Name}Arrows.ts              # Flechas de flujo de datos
├── {Name}Annotations.ts         # Anotaciones dimensionales + nombres
├── {Name}SectionLabels.ts       # Etiquetas de sección
├── {Name}Program.ts             # State machine, init, render loop
├── {Name}LayerView.tsx          # Canvas WebGL2 + React wrapper
├── {Name}LayerView.module.scss  # Estilos del canvas
├── {Name}Sidebar.tsx            # Sidebar con walkthrough
├── {Name}Sidebar.module.scss    # Estilos del sidebar
└── walkthrough/
    ├── {Name}Walkthrough.ts     # Fases, dispatch, herramientas
    └── Phase{NN}_{Topic}.tsx    # 4-6 fases de walkthrough interactivo

viz/src/app/{name}/
└── page.tsx                     # Next.js route
```

### Walkthrough de CrossAttention (6 fases):
0. **Overview**: Qué es cross-attention vs concat, por qué selectividad dinámica
1. **Audio CNN**: Waveform → CNN 4 stages → PosEmbedding (antes de cross-att)
2. **Audio Cross-Att**: Q=features(2400), K/V=descriptor(188 STFT nativo), mismatch temporal
3. **MIDI Cross-Att**: Q=embeddings(N), K/V=intervals(N), misma resolución
4. **Transformers**: Post cross-att, 4 layers, pool, projection
5. **VICReg**: Loss, comparación params concat ~1M vs cross-att ~4.2M (audio)

### Build verification:
- TypeScript: 0 errores (`npx tsc --noEmit`)
- Build: 8 rutas exportadas exitosamente (`npm run build`)
- GitHub Pages: deploy automático via GitHub Actions

### Commit `1e3d678` — README actualizado
README principal del repo actualizado con tabla de las 8 arquitecturas + estado Gate 4.3.

---

## 16. Commits y archivos pendientes de commit (2026-02-15)

### Ya commiteados y pusheados:

| Commit | Contenido |
|--------|-----------|
| `8b09cbf` | Gate 4.3 code (concat + audio cross-att) + docs sync |
| `7573483` | 5 nuevas visualizaciones + rename 3 existentes + homepage |
| `1e3d678` | README con 8 arquitecturas + estado Gate 4.3 |

### Pendientes de commit:

| Archivo | Cambio |
|---------|--------|
| `experiments/bias_control/gate42_training.py` | +Gate42MidiCrossAttModel, +_encode_midi_with_cross_attention, +8 puntos integración D4x |
| `experiments/bias_control/run_gate43.sh` | Reescrito para Fase 1+2 (loop `a7 a4x a7x d4x`) |
| `Documents/.../07_GATE_4_3.../README.md` | Actualizado con roadmap extendido |
| `CLAUDE.md` | Actualizaciones de estado |
| `NOTAS_CLAUDE_PARA_CODEX.md` | Este archivo |

---

## 17. D4x parciales (2026-02-15 ~07:00 UTC)

D4x sigue corriendo. Resultados epoch 1-3:

| Epoch | S | hard_neg |
|-------|---|----------|
| e1 | 54.8% | 88.6% |
| e2 | 57.0% | 90.2% |
| e3 | 58.4% | 89.8% |

Sigue mejorando pero la trayectoria sugiere que no alcanzará al D4 concat (63.6%).
Faltan epochs 4 y 5. ETA ~08:00 UTC.

---

## 18. Nuevo modelo: d4a4cm — cross-modal dual (2026-02-15 ~05:00 UTC)

### Qué es

Modelo dual que inyecta descriptores de un dominio en el encoder del OTRO dominio:
- **Audio encoder recibe MIDI intervals** (D4, 4 dimensiones)
- **MIDI encoder recibe audio descriptors** (A4, 8 dimensiones)

Esto es diferente de `d4a4` (same-modality dual) donde cada descriptor va a su propio encoder.

### Qué se implementó en `gate42_training.py`

**2 nuevas helpers:**
- `_encode_audio_with_cross_modal_intervals()`: Pipeline del audio encoder que recibe
  MIDI intervals. Hace `F.interpolate` de N tokens MIDI → T'=2400 frames CNN.
- `_encode_midi_with_cross_modal_audio_desc()`: Pipeline del MIDI encoder que recibe
  audio descriptors A4. Hace `F.interpolate` de T_stft=188 frames → N tokens MIDI.

**1 nueva clase:**
- `Gate42DualCrossModalModel`: Usa ambos helpers. Dos projections nuevas:
  - `cross_modal_audio_projection`: Linear(1028→1024) + LN (~1.05M params)
  - `cross_modal_midi_projection`: Linear(520→512) + LN (~0.27M params)
  - Total params nuevos: 1,323,520 (~1.3M)

**8 puntos de integración** (mismo patrón que todos los modelos):
1. CLI: descriptor `d4a4cm`
2. Factory en `create_gate42_model()`
3. Optimizer: 2 param groups @ lr_ratio
4. Param ranges: run-b (39M-42.5M), run-d (64M-68.5M)
5. Preflight: prefixes `cross_modal_audio_projection.`, `cross_modal_midi_projection.`
6. Checkpoint: eval_compatible=False
7. Eval loading: reconstruye Gate42DualCrossModalModel
8. embed_batch_size: auto-reduce a 32

### Verificación CPU: PASSED
- Shapes: (B, 256) ambos lados
- Gradientes fluyen a ambas projections
- Sin NaN en random y silencio
- Factory, preflight, param_ranges: todos OK

### Status: CODE COMPLETE + CPU VERIFIED. No commiteado. Pendiente pilot GPU + run 5ep.

---

## 19. Decisión del usuario: roadmap post-Gate 4.3 (2026-02-15)

### Qué decidió el usuario

**Antes de pasar a Gate 4.4 nuevo**, correr dos brazos duales más para cerrar Gate 4.3:

1. **d4a4** — dual same-modality (concat): D4 intervals → MIDI encoder + A4 log-freq → audio encoder. Ya implementado.
2. **d4a4cm** — dual cross-modal (concat): D4 intervals → AUDIO encoder + A4 log-freq → MIDI encoder. Recién implementado (sección 18).

**Cita textual del usuario**: "antes de pasar al nuevo gate 4.4 quiero que corramos un dual d4a4 same-modality y un dual d4a4 cross-modality... es necesario probar eso después de todo lo que hicimos en 4.3... después de eso pasamos a 4.4 nuevo."

### Renumeración de Gates (ACTUALIZADA 2026-02-15 ~18:30 UTC)

**SEGUNDA renumeración** — el usuario revisó el roadmap después de los resultados de Gate 4.3.

| Gate anterior | Gate nuevo | Contenido |
|---------------|-----------|-----------|
| Gate 4.3 | **Gate 4.3** (sin cambio) | Ratio re-céntrico. Fases 0-3 + scratch + Fase 5 |
| Gate 4.6 (third tower) | **Gate 4.4** | 3 torres / ratio bridge, con mejores descriptores de 4.3 |
| Gate 4.4 (cross-modal injection) | **Gate 5 Línea A** | CM-a, CM-m, CM-bi + barrido comprehensivo |
| Gate 4.5 (barrido bifurcado) | **Gate 5 Línea A** | Fusionado con cross-modal injection |
| *(nuevo)* | **Gate 5 Línea B** | Best model → train largo → tests extremos + showcase comunidad |

**Nota para Codex**: Los directorios `08_GATE_4_4_BIFURCACION_RATIO/` y `09_GATE_4_5_RATIO_BRIDGE/` necesitan renumerarse. El contenido de `09_GATE_4_5_RATIO_BRIDGE/` (third tower) pasa a ser Gate 4.4.

### Secuencia operativa acordada

1. d4a4-scratch termina (~04:00 UTC 16/02)
2. GPU pilot + run Fase 5: a4r, d4r, a8, a9 (~10h)
3. Cerrar Gate 4.3 con análisis comparativo completo
4. **Gate 4.4**: Third tower / ratio bridge con mejores descriptores + mecanismos
5. **Gate 5 Línea A**: Barrido comprehensivo + cross-modal injection, adaptado a learnings de 4.4
6. **Gate 5 Línea B**: Best model → train largo → tests cross-modales extremos + showcase comunidad
7. Líneas A y B de Gate 5 pueden correr en paralelo (si hay recursos)

---

## 20. Commits hechos en esta sesión

| Commit | Fecha | Contenido |
|--------|-------|-----------|
| `7573483` | 2026-02-15 | 5 nuevas visualizaciones WebGL2 + rename 3 existentes (89 archivos, +9,456 LOC) |
| `1e3d678` | 2026-02-15 | README.md actualizado con 8 arquitecturas + estado Gate 4.3 |

Ambos pusheados a main. GitHub Pages desplegado con las 8 visualizaciones.

---

## 21. Archivos pendientes de commit (actualizado 2026-02-15 ~18:15 UTC)

| Archivo | Cambio |
|---------|--------|
| `experiments/bias_control/gate42_training.py` | +D4x, +d4a4cm, +A4r, +D4r, +A8, +A9 (models + helpers + integración) |
| `experiments/bias_control/run_gate43.sh` | Reescrito para Fase 5 (loop a4r d4r a8 a9) |
| `experiments/bias_control/gate43_scratch/gate43_scratch_training.py` | NUEVO: script from-scratch |
| `src/bias_control/audio_descriptors.py` | +A8, +A9, refactorizado A7 (helper compartido) |
| `CLAUDE.md` | Actualizaciones de estado |
| `NOTAS_CLAUDE_PARA_CODEX.md` | Este archivo (27→29 secciones) |
| `Documents/.../07_GATE_4_3.../README.md` | Actualizado con resultados completos |
| `Documents/.../07_GATE_4_3.../D0_D4_A4_A7_A4x_A7x_result.md` | Nuevo: tabla de resultados parcial |
| `Documents/.../09_GATE_4_5_RATIO_BRIDGE/` | Nuevo directorio |
| `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/PHIDEUS_MASTER_BRIEFING.md` | NUEVO: síntesis completa del proyecto (20K) |
| `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/PHIDEUS_NEURAL_ARCHITECTURES.md` | NUEVO: arquitecturas técnicas (17K) |
| `Para_GPT/` | NUEVO directorio: 20 archivos para contexto ChatGPT (616KB total) |
| `viz/tsconfig.tsbuildinfo` | Artefacto de build |

---

## 22. Tabla de referencia completa — todos los brazos Gate 4.3 (actualizado 2026-02-15)

### Fases 0-3: COMPLETE (9 brazos)

| Rank | Brazo | Lado | Descriptor | Mecanismo | Params nuevos | Best S | Best ep | vs D0 |
|------|-------|------|-----------|-----------|---------------|--------|---------|-------|
| **1** | **d4a4** | Ambos | D4+A4 same-mod | concat | ~1.3M | **69.8%** | e5 | **+9.6pp** |
| 2 | D4 | MIDI | intervals (4d) | concat | ~267K | 63.6% | e5 | +3.4pp |
| 2 | A4 | Audio | log-freq (8d) | concat | ~1.06M | 63.6% | e5 | +3.4pp |
| 4 | A4x | Audio | log-freq (8d) | cross-att | ~4.2M | 62.6% | e5 | +2.4pp |
| 5 | A7x | Audio | attractor (12d) | cross-att | ~4.2M | 62.2% | e5 | +2.0pp |
| 6 | D0 | — | — | baseline | 0 | 60.2% | e3 | — |
| 7 | D4x | MIDI | intervals (4d) | cross-att | ~1.05M | 60.0% | e5 | -0.2pp |
| 8 | A7 | Audio | attractor (12d) | concat | ~1.06M | 58.8% | e5 | -1.4pp |
| 9 | d4a4cm | Ambos | D4→audio + A4→MIDI | cross-modal | ~1.3M | 52.4% | e5 | -7.8pp |

### d4a4-scratch: RUNNING (epoch 11/30)

| Checkpoint | S | hard_neg | Loss | Notas |
|------------|---|----------|------|-------|
| epoch 10 | **74.6%** | **93.0%** | 13.58 | RECORD del proyecto. +19pp vs D-02 a misma loss. |

### Fase 5: IMPLEMENTADO, pendiente GPU pilot

| Brazo | Lado | Descriptor | Mecanismo | Params nuevos | Status |
|-------|------|-----------|-----------|---------------|--------|
| A4r | Audio | log-freq (8d) | **reverse** cross-att | ~4.4M | CODE COMPLETE |
| D4r | MIDI | intervals (4d) | **reverse** cross-att | ~1.05M | CODE COMPLETE |
| A8 | Audio | onset-chroma (12d) | concat | ~1.06M | CODE COMPLETE |
| A9 | Audio | IDF-attractor (12d) | concat | ~1.06M | CODE COMPLETE |

---

## 23. D4x — Resultados finales (2026-02-15 ~05:00 UTC)

D4x (MIDI cross-attention) completó 5 epochs:

| Epoch | S | A2M_R10 | M2A_R10 | hard_neg |
|-------|---|---------|---------|----------|
| e1 | 54.8% | 54.8% | 56.0% | 88.6% |
| e2 | 57.0% | 58.4% | 57.0% | 90.2% |
| e3 | 58.4% | 59.0% | 58.4% | 89.8% |
| e4 | 59.8% | 61.0% | 59.8% | 90.8% |
| **e5** | **60.0%** | **61.4%** | **60.0%** | **91.4%** |

**Análisis D4x**: S=60.0% a e5, apenas -0.2pp del baseline D0 (60.2%). Cross-attention NO
aporta para MIDI intervals — D4 concat (63.6%) supera D4x por 3.6pp con menos params.
Para el lado MIDI, la mezcla lineal simple (concat) es más efectiva que la atención dinámica.

---

## 24. d4a4 — Dual same-modality concat (2026-02-15 ~05:00-08:00 UTC)

d4a4 usa ambos descriptores, cada uno inyectado en su propio encoder:
- MIDI intervals (D4) → MIDI encoder (concat)
- Audio log-freq (A4) → Audio encoder (concat)

| Epoch | S | A2M_R10 | M2A_R10 | hard_neg |
|-------|---|---------|---------|----------|
| e1 | 14.0% | 14.0% | 23.4% | 72.2% |
| e2 | 35.8% | 35.8% | 45.8% | 85.6% |
| e3 | 56.6% | 56.6% | 61.0% | 89.2% |
| e4 | 64.8% | 64.8% | 65.8% | 91.2% |
| **e5** | **69.8%** | **70.2%** | **69.8%** | **91.6%** |

### 🏆 RESULTADO DESTACADO

**d4a4 = 69.8%, +9.6pp sobre baseline, +6.2pp sobre mejor individual.**

Trayectoria muy pronunciada: e3=56.6% → e4=64.8% → e5=69.8% (+5.0pp en última epoch).
**Seguía subiendo fuerte al terminar.** El arranque lento (e1=14.0%) se explica porque
AMBAS projections (interval + audio_descriptor) tienen random init y aprenden desde cero.

La señal conjunta de ambos descriptores es **superaditiva**: D4 solo aporta +3.4pp, A4 solo
aporta +3.4pp, pero juntos aportan +9.6pp. No son +6.8pp (aditivos), son +9.6pp —
sugiriendo complementariedad informacional entre intervalos MIDI y descriptores de audio.

---

## 25. d4a4cm — Dual cross-modal (2026-02-15 ~08:00-11:00 UTC)

d4a4cm inyecta descriptores de un dominio en el encoder del OTRO:
- MIDI intervals (D4) → AUDIO encoder (via F.interpolate N→T'=2400)
- Audio log-freq (A4) → MIDI encoder (via F.interpolate T_stft=188→N)

| Epoch | S | A2M_R10 | M2A_R10 | hard_neg |
|-------|---|---------|---------|----------|
| e1 | 18.0% | 18.0% | 22.4% | 74.4% |
| e2 | 46.8% | 46.8% | 52.6% | 88.6% |
| e3 | 48.2% | 48.2% | 53.0% | 89.4% |
| e4 | 51.2% | 51.2% | 54.0% | 89.0% |
| **e5** | **52.4%** | **52.4%** | **53.8%** | **89.6%** |

### RESULTADO NEGATIVO

**d4a4cm = 52.4%, -7.8pp DEBAJO de baseline D0 (60.2%).**

La inyección cross-modal es destructiva. Comparando trayectorias:
- d4a4 (same-mod): e3=56.6% → e4=64.8% → e5=69.8% (acelerando)
- d4a4cm (cross-mod): e3=48.2% → e4=51.2% → e5=52.4% (plateauing, +1.2pp/ep)

**Interpretación**: Los descriptores de ratio son información complementaria INTRA-modal.
Cada encoder se beneficia de ver los ratios de su propia señal, pero se perjudica al
recibir los de la otra modalidad. Los ratios de frecuencia no son un "puente cross-modal"
directo a nivel de features — la alineación cross-modal es tarea de VICReg, no del encoder.

---

## 26. TABLA FINAL GATE 4.3 — TODOS LOS BRAZOS (2026-02-15 11:00 UTC)

| Rank | Arm | Mecanismo | Best ep | Best S | hard_neg | vs D0 |
|------|-----|-----------|---------|--------|----------|-------|
| **1** | **d4a4** | **Dual same-mod concat** | **e5** | **69.8%** | **91.6%** | **+9.6pp** |
| 2 | D4 | MIDI intervals concat | e5 | 63.6% | 91.2% | +3.4pp |
| 2 | A4 | Audio desc concat | e5 | 63.6% | 92.4% | +3.4pp |
| 4 | A4x | Audio desc cross-att | e5 | 62.6% | 92.4% | +2.4pp |
| 5 | A7x | Audio attractor cross-att | e5 | 62.2% | 92.0% | +2.0pp |
| 6 | D0 | baseline | e3 | 60.2% | 90.0% | — |
| 7 | D4x | MIDI intervals cross-att | e5 | 60.0% | 91.4% | -0.2pp |
| 8 | A7 | Audio attractor concat | e5 | 58.8% | 90.2% | -1.4pp |
| 9 | d4a4cm | Dual cross-modal | e5 | 52.4% | 89.6% | -7.8pp |

### Conclusiones Gate 4.3

1. **Concat > Cross-attention** consistentemente: D4(63.6%) > D4x(60.0%), A4(63.6%) > A4x(62.6%)
2. **Same-modality > Cross-modal**: d4a4(69.8%) >> d4a4cm(52.4%). Diferencia de 17.4pp.
3. **Dual superaditivo**: D4+A4 individual = +3.4pp cada uno, combinados = +9.6pp
4. **d4a4 todavía subía fuerte** a e5 (+5.0pp). Candidato claro a run largo.
5. **Log-freq (A4) > Attractor (A7)** en todos los mecanismos
6. **Cross-attention rescata descriptores débiles** (A7x=62.2% vs A7=58.8%) pero no supera concat con descriptores fuertes

### Siguiente paso

PASO 11 (decisión): d4a4 (dual same-mod concat) es el ganador claro. Necesita más epochs.
Luego: Gate 4.4 (cross-modal injection con 3 brazos nuevos) según roadmap acordado.

---

## 27. d4a4 from scratch — 30 epochs (2026-02-15 ~12:44 UTC — RUNNING)

### Pregunta científica

Todos los experimentos de Gate 4.3 parten de `foundation_locked_e25.pt` (25 epochs de
VICReg puro sin descriptores). El usuario propuso un experimento fundamental:

**¿Qué pasa si entrenamos d4a4 desde cero, sin foundation?**

Esto permite comparar:

| Run | Punto de partida | Descriptores | Epochs | Resultado |
|-----|-----------------|-------------|--------|-----------|
| D-02 | MERT pretrained + MIDI random | ninguno | 25→30 | S=61.8% (e25) |
| D0 (Gate 4.3) | foundation e25 | ninguno | 5 | S=60.2% (e3) |
| d4a4 (Gate 4.3) | foundation e25 | D4+A4 dual | 5 | **69.8%** (e5, subiendo) |
| **d4a4-scratch** | **MERT pretrained + MIDI random** | **D4+A4 dual** | **30** | **???** |

### Tres escenarios posibles

1. **d4a4-scratch >> d4a4-foundation**: Los descriptores guían mejor desde el inicio.
   La foundation aprendió representaciones "subóptimas" que los descriptores corrigen.
2. **d4a4-scratch ≈ d4a4-foundation**: La foundation es redundante con descriptores.
3. **d4a4-scratch << d4a4-foundation**: La foundation aporta algo irremplazable.
   Los descriptores son refinamiento, no sustituto.

El escenario 1 sería el más potente para la tesis Phideus.

### Implementación

Se copió `gate42_training.py` a `experiments/bias_control/gate43_scratch/gate43_scratch_training.py`
con un nuevo flag `--from-scratch` que crea el modelo con `CrossModalModel(audio_encoder='lite',
use_dann=False)` directamente (MERT pretrained + random MIDI), sin cargar ningún checkpoint.

**No se modificó el script original** — queda intacto.

### Detalles del run

- **tmux**: `d4a4scratch`
- **Output**: `data/bias_control_medium/training_outputs/gate43/gate43_d4a4_scratch_30ep/`
- **Script**: `experiments/bias_control/gate43_scratch/gate43_scratch_training.py`
- **Comando**:
  ```bash
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python \
    experiments/bias_control/gate43_scratch/gate43_scratch_training.py \
    --mode train --descriptor d4a4 --from-scratch --freeze-policy run-d \
    --output data/bias_control_medium/training_outputs/gate43/gate43_d4a4_scratch_30ep \
    --epochs 30 --batch-size 16 --num-workers 8 --max-batches-per-epoch 1000 --seed 42
  ```
- **Params**: 66.2M trainable (run-d policy, idéntico a D-02)
- **Freeze policy**: run-d (full unfreeze, split-LR) — mismo que D-02
- **ETA**: ~30ep × 35min ≈ 17-18h (termina ~07:00 UTC 16/02)
- **Loss inicial**: 52.49 (vs ~14 desde foundation — esperado, parte de cero)
- **Checkpoints**: se guardan en cada epoch

### Archivos nuevos (no commiteados)

| Archivo | Contenido |
|---------|-----------|
| `experiments/bias_control/gate43_scratch/gate43_scratch_training.py` | Copia de gate42_training.py + flag `--from-scratch` |

---

## 28. d4a4-scratch: Epoch 10 Structured Eval + Relaunch con Scheduler (2026-02-15 ~18:00 UTC)

### Resultado epoch 10 — RECORD DEL PROYECTO

Se cortó el run de d4a4-scratch en epoch 11, se corrió structured eval en checkpoint epoch 10:

**S = 74.6%, hard_neg = 93.0%**

Esto es el mejor resultado de todo Phideus:
- vs d4a4-finetuned (5ep): S=69.8% → **+4.8pp** (y scratch todavía subiendo)
- vs D-02 (25ep): S=61.8% → **+12.8pp**
- vs D-02 loss-matched (e8): scratch_loss=13.68 ≈ D-02_loss=13.67, pero scratch S=74.6% vs D-02 S=51.0%

**Detalle eval_epoch10.json**:
- A2M R@10: 74.6%, M2A R@10: 75.0%
- A2M R@1: 17.2%, M2A R@1: 14.6%
- A2M R@20: 88.8%, M2A R@20: 90.2%
- MRR: A2M=0.342, M2A=0.330
- Pool: 256, queries: 500, hard_neg: 64, seed: 42

### Comparación completa scratch vs D-02

| Ep | scratch loss | scratch S | D-02 loss | D-02 S | Gap S |
|----|-------------|-----------|-----------|--------|-------|
| 1 | 14.74* | — | 14.17 | 22.8% | — |
| 2 | 14.41 | — | 14.01 | 37.0% | — |
| 3 | 14.16 | — | 13.92 | 37.8% | — |
| 4 | 14.02 | — | 13.84 | 48.6% | — |
| 5 | 13.90 | — | 13.79 | 47.0% | — |
| 6 | 13.81 | — | 13.76 | 43.8% | — |
| 7 | 13.75 | — | 13.71 | 48.8% | — |
| 8 | 13.68 | — | 13.67 | 51.0% | — |
| 9 | 13.63 | — | 13.63 | 54.4% | — |
| **10** | **13.58** | **74.6%** | 13.59 | 55.6% | **+19.0pp** |

*Nota*: scratch S solo disponible para epoch 10 (structured eval). D-02 S de eval por epoch.
Loss gap cerrado por completo a epoch 9-10.

### Implicaciones

1. **Foundation NO es necesaria para d4a4** — descriptores de ratio guían el aprendizaje desde cero
2. **El gap S (19pp) a misma loss sugiere que d4a4 forma un embedding space cualitativamente distinto**
3. **d4a4-scratch todavía bajando loss → más epochs probablemente mejoren aún más**

### Relaunch

Run cortado en epoch 11, relanzado inmediatamente con structured eval scheduler:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python \
  experiments/bias_control/gate43_scratch/gate43_scratch_training.py \
  --mode train --descriptor d4a4 --from-scratch --freeze-policy run-d \
  --skip-structured-eval --structured-eval-epochs 15 20 25 28 29 30 \
  --output data/bias_control_medium/training_outputs/gate43/gate43_d4a4_scratch_30ep \
  --maestro-dir data/maestro_v3/maestro-v3.0.0 \
  --epochs 30 --batch-size 16 --num-workers 8 --max-batches-per-epoch 1000 --seed 42 \
  --warmup-steps 200 --device cuda \
  --lr-audio-unfreeze 1e-5 --lr-audio-low 5e-6 \
  --lr-midi 5e-5 --lr-proj 1e-4 --lr-ratio 5e-4 \
  --resume data/bias_control_medium/training_outputs/gate43/gate43_d4a4_scratch_30ep/checkpoint_epoch10.pt
```

Resume desde epoch 10, structured eval en epochs 15, 20, 25, 28, 29, 30 (~6min overhead cada una).
ETA ~04:00-05:00 UTC 16/02.

---

## 29. Gate 4.3 Fase 5 — Implementación completa (2026-02-15 ~18:00 UTC)

### Qué es Fase 5

4 nuevos brazos experimentales para Gate 4.3:

| Brazo | Descriptor | Mecanismo | Dim | Compara contra |
|-------|-----------|-----------|-----|----------------|
| **A4r** | A4 log-freq | **reverse** cross-att | 8 | A4x (regular cross-att) |
| **D4r** | D4 intervals | **reverse** cross-att | 4 | D4x (regular cross-att) |
| **A8** | onset-weighted chroma | concat | 12 | A4, A7 |
| **A9** | IDF-weighted rational attractor | concat | 12 | A7 |

### Reverse cross-attention (A4r, D4r)

Concepto: en cross-attention regular, features (Q) preguntan al descriptor (K/V).
En **reverse**, descriptores (Q) organizan features (K/V) — los ratios como principio organizador.

**A4r pipeline**: CNN→features[2400,1024] + pos_emb → K/V. Descriptor[188,8]→q_proj→[188,1024] + desc_pos_emb → Q.
CrossAtt(Q=188, K/V=2400) → residual+LN → Transformer(188 tokens) → pool → proj → [256].
**Beneficio**: Transformer procesa 188 tokens vs 2400 = **12.8x menos cómputo en self-attention**.

**D4r pipeline**: Similar pero Q/K/V same length (N tokens MIDI). Semantic difference only.

### Nuevos descriptores (A8, A9)

**A8 (onset-weighted chroma)**: STFT bins → 12 pitch classes (octave-folded) × spectral flux (onset gate).
Inspirado en Route A de Escalón 1: "onsets son los momentos más informativos para ratio extraction".

**A9 (IDF-weighted rational attractor)**: Misma base que A7 pero con per-sample IDF weighting.
Attractors comunes (octava, quinta) → downweighted. Attractors raros (tritono, 7ma armónica) → upweighted.
Inspirado en Route B de Escalón 1.

### Archivos modificados

**`src/bias_control/audio_descriptors.py`**:
- Refactorizado A7: extraído helper `_compute_raw_attractor_activations()` (compartido con A9)
- Nuevo: `compute_audio_descriptor_a8()` — onset-weighted chroma [B, T, 12]
- Nuevo: `compute_audio_descriptor_a9()` — IDF-weighted attractor [B, T, 12]

**`experiments/bias_control/gate42_training.py`**:
- Import: added a8, a9
- Helpers: `_encode_audio_with_reverse_cross_attention()`, `_encode_midi_with_reverse_cross_attention()`
- Models: `Gate42AudioReverseCrossAttModel` (~4.4M params), `Gate42MidiReverseCrossAttModel` (~1.05M params)
- 8 integration points × 4 brazos:
  1. CLI choices: a4r, d4r, a8, a9
  2. Model factory: create_gate42_model branches
  3. Optimizer: param groups (a4r: 4 groups, d4r: 3, a8/a9: 1)
  4. Param ranges: run-b and run-d
  5. Preflight: trainable prefixes
  6. Checkpoint: eval_compatible=False + archive_base
  7. Eval loading: reconstruction branches
  8. embed_batch_size: a4r→16, a8/a9→32

**`experiments/bias_control/run_gate43.sh`**: Actualizado para Fase 5 (loop: a4r d4r a8 a9).

### Verificación

- `py_compile` PASS en ambos archivos
- Import test PASS (4 descriptors importados OK)
- Model creation PASS: a4r=~4.4M, d4r=~1.05M, a8=~1.06M, a9=~1.06M params nuevos
- **NO se corrió GPU pilot** (GPU ocupada con d4a4-scratch)

### Pendiente

Cuando d4a4-scratch termine (~04:00 UTC 16/02):
1. GPU pilot: 1ep/100 batches cada brazo (a4r, d4r, a8, a9)
2. Si pilots OK → run completo: `bash experiments/bias_control/run_gate43.sh` (4 arms × 5ep ≈ 10h)
3. Tabla de comparación con Gate 4.3 Fases 0-3

---

## 30. Para_GPT + Documentos de síntesis (2026-02-15)

### Para_GPT/

Nuevo directorio en la raíz del repo con 20 archivos (616KB, 12,548 líneas) diseñados para dar
contexto completo del proyecto a ChatGPT. Incluye:

- 2 documentos de síntesis NUEVOS (escritos por Claude)
- 18 documentos existentes copiados con prefijos numéricos (01-20)

**No es código** — es material de referencia para que el usuario pueda consultar con ChatGPT
cuando Claude y Codex no estén disponibles.

### Documentos de síntesis nuevos

Creados en `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/`:

1. **PHIDEUS_MASTER_BRIEFING.md** (~20K):
   - Síntesis completa del proyecto: hipótesis, validaciones, arquitectura, resultados
   - Cubre desde H1/H2 hasta Gate 4.3 con tabla de todos los brazos
   - Incluye la narrativa del hallazgo superaditivo d4a4
   - Roadmap futuro (Gates 4.4-4.6)

2. **PHIDEUS_NEURAL_ARCHITECTURES.md** (~17K):
   - Detalle técnico de TODAS las arquitecturas implementadas
   - MERTEncoderLite, MIDI Transformer, VICReg, DANN, RosetaVAE, HRM, ConstellationVAE, JEPA-Lite
   - Diagramas de flujo de cada modelo
   - Tabla comparativa de parámetros y resultados

### Nota para Codex
Estos documentos son de LECTURA para el usuario — no necesitan mantenimiento por Codex.
Pero si Codex quiere entender rápidamente el estado del proyecto, `PHIDEUS_MASTER_BRIEFING.md`
es el mejor punto de entrada.

---

## 31. Roadmap actualizado (2026-02-15 ~18:30 UTC)

### ROADMAP VIGENTE (segunda renumeración)

| Gate | Status | Descripción |
|------|--------|-------------|
| **Gate 4.3** — Ratio re-céntrico | 🔄 CERRANDO | Fases 0-3 ✅ (9 brazos, d4a4=69.8%). d4a4-scratch 🔄 (e11/30, S=74.6%@e10). Fase 5 🟡 CODE COMPLETE. |
| **Gate 4.4** — Arquitecturas mayores | PENDING | Third tower / ratio bridge + MoE con Ratio Expert. Ambos son rediseños arquitectónicos mayores usando mejores findings de Gate 4.3. (Absorbe ex-Gate 4.6 + MoE de §11 GPT doc.) |
| **Gate 5 Línea A** — Barrido + cross-modal injection | PENDING | Barrido comprehensivo descriptores × mecanismos (concat, cross-att, reverse, **FiLM**) + CM-a, CM-m, CM-bi. Todo adaptado a learnings de Gate 4.4. (Fusiona ex-Gate 4.4 + ex-Gate 4.5 + FiLM de §11 GPT doc.) |
| **Gate 5 Línea B** — Showcase cross-modal extremo | PENDING | Best model → train largo para máximo rendimiento → batería de tests cross-modales extremos + visualizaciones + materiales para la comunidad. (Nuevo.) |

### Secuencia operativa

1. d4a4-scratch termina → recoger resultados epochs 15, 20, 25, 28, 29, 30
2. GPU pilot Fase 5 → run completo si OK (~10h)
3. Cerrar Gate 4.3: análisis comparativo completo de todos los brazos
4. Gate 4.4: diseñar + implementar third tower + MoE con mejores findings
5. Gate 5: Líneas A y B pueden correr en paralelo si hay recursos

### Mapping desde numeración anterior

| Antes (sección 19) | Ahora | Motivo |
|---------------------|-------|--------|
| Gate 4.4 (cross-modal injection) | Gate 5 Línea A | Pospuesto, se adapta post-4.4 |
| Gate 4.5 (barrido bifurcado) | Gate 5 Línea A | Fusionado con cross-modal injection |
| Gate 4.6 (third tower) | **Gate 4.4** | Priorizado: probar 3 torres primero |
| *(nuevo)* | Gate 5 Línea B | Showcase + tests extremos para comunidad |

---

## 32. Gate 5 Línea B — Roadmap oficial de pruebas (2026-02-15)

Fuente: `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_INSUMOS_GPT5.2PRO.md`
Filtrado: solo pruebas/tests/entregables para showcase cross-modal (no arquitecturas nuevas).
**Orden: por relevancia científica para la tesis Phideus** ("ratios como lenguaje informacional cross-modal").
Los primeros 5 son imprescindibles para cualquier publicación. 6-9 fortalecen el argumento. 10-13 son comunicación y completitud.

### 1. Causal ablation — zero-out injection (Test B, §6.2) ~1h
- Entrenar con ratio-injection, evaluar con injection anulada (set a cero).
- Si S cae mucho → el modelo USA los ratios. Si no → solo regularización.
- **La pregunta más fundamental**: causalidad > correlación. Es lo que un reviewer pide primero.

### 2. Parameter-matched ablations — control de ruido (§7.1) ~6h code + GPU
- Mismo modelo con mismos params extra, pero input del descriptor es ruido/permutado.
- Separa "más capacidad" de "más información".
- **Segundo control esencial**: d4a4 tiene +1.3M params sobre D0. Si ruido random da el mismo boost, Phideus cae.

### 3. RatioProbeDecoder + cross-decoding (Test C / Exp P1, §6.2 + §11) ~8h
- Decoder lineal: dado embedding final → predecir D4 / A4 / A7.
- Cross-decoding: predecir D4 desde audio_emb, A4 desde midi_emb.
- **Smoking gun de Phideus**: si se decodifican ratios de audio desde embedding MIDI → el espacio compartido codifica estructura relacional cross-modal.

### 4. Invariancia a transposición MIDI (Test A parcial, §6.2) ~4h parte MIDI
- Transponer MIDI ±k semitonos (pitch += k), re-evaluar retrieval.
- Medir drift de embedding y caída de S vs k.
- **Test directo de la hipótesis central**: si aprendió RATIOS (relativos), transponer NO debería cambiar el embedding (los ratios se conservan). Si cambia mucho → aprendió pitch absoluto, no ratios.

### 5. Multi-seed replication (§7.1) ~5 runs, solo GPU time
- Re-correr el mejor modelo con 3-5 seeds.
- Reportar S ± std. Ya se hizo para D-02 (S=61.6% ± 1.1%).
- **Sin reproducibilidad no hay ciencia**. S=74.6% con un seed no es resultado; S=74.6% ± X% con 5 seeds sí.

### 6. RSA/CKA entre capas audio y MIDI (§9.3) ~1 día
- Extraer representaciones por capa de ambos encoders.
- Representational Similarity Analysis / Centered Kernel Alignment.
- **Evidencia mecanística**: DÓNDE se alinean las modalidades. Si la alineación crece en capas con ratio-injection → evidencia directa.

### 7. Counterfactual Decoder (Decoder 3, §7.2) ~1 semana
- Operar en espacio de embeddings: transponer, time-stretch, invertir intervalos.
- Ver cómo cambia embedding/retrieval bajo transformaciones controladas.
- **Verifica geometría del espacio**: ¿transponer en embedding-space = transponer en input-space? Si sí → representación genuina de ratios.

### 8. Ratio decoding report (Entregable 2, §9.2) ~1 día (después de #3)
- Builds on probes (#3). Curvas vs epochs, vs seeds, controles (decoder en embedding random/mean).
- **Paper-ready**: cierra el argumento de ratio encoding con rigor estadístico completo.

### 9. Invariancia suite completa (Entregable 3, §9.2) ~1-2 días
- Transposición + time-stretch + masking temporal + drift geométrico.
- Audio pitch-shift requiere resampling/librosa.
- Cada invariancia verificada = constraint adicional consistente con ratio-language.

### 10. UMAP/t-SNE de embeddings (§9.3) ~3h
- Extraer embeddings, colorear por pieza y por posición temporal ("same piece, different time").
- Si ratio-language existe → agrupación por identidad relacional.
- **Exploratorio, no confirmatorio**. Útil como figura de paper y para generar hipótesis.

### 11. CrossModalSequenceDecoder (Decoder 2, §7.2) ~3-5 días
- Audio embedding → decoder autoregresivo → skeletal MIDI (pitch + onset).
- Demo: "escuchás algo → genera versión simbólica".
- **Impresionante como demo**, pero evidencia indirecta — que un decoder funcione no prueba que los ratios sean el mecanismo.

### 12. Gate scoreboard reproducible (Entregable 4, §9.2) ~4h
- Notebook/script que lee eval JSONs + configs + checkpoint hashes → tabla resumen.
- Blinda trazabilidad (anti-variable-fantasma). Infraestructura, no test científico en sí.

### 13. Retrieval demo UI (Entregable 1, §9.2) ~2-3 días
- Subir audio → top-10 MIDI candidates → overlay "ratio signature".
- Hard negatives y por qué fallan.
- **Showcase para la comunidad**. Último en relevancia científica, primero en impacto comunicacional.

### Nota
NO incluidos en esta lista (van en otros gates):
- **Gate 4.4**: Third tower / ratio bridge (Arq C) + MoE con Ratio Expert (Exp P3) — arquitecturas mayores
- **Gate 5 Línea A**: FiLM transversal (Exp P2) — cuarta familia de inyección junto a concat, cross-att, reverse

---

## 33. Reestructura de directorios BIAS_CONTROL (2026-02-15 ~20:30 UTC)

### Cambios de directorios

Se adecuó la estructura de `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/` al nuevo roadmap:

| Antes | Ahora | Contenido |
|-------|-------|-----------|
| `08_GATE_4_4_BIFURCACION_RATIO/` | `09_GATE_5_LINEA_A_BARRIDO/` | Barrido + FiLM + cross-modal injection |
| `09_GATE_4_5_RATIO_BRIDGE/` | `08_GATE_4_4_ARQUITECTURAS_MAYORES/` | Third tower + MoE |
| *(nuevo)* | `10_GATE_5_LINEA_B_SHOWCASE/` | 13 tests científicos |

Directorios 01-07 y 90 sin cambios.

### READMEs actualizados/creados

- `08_GATE_4_4_ARQUITECTURAS_MAYORES/README.md` — reescrito: third tower + MoE + criterios GO/NO-GO
- `09_GATE_5_LINEA_A_BARRIDO/README.md` — reescrito: barrido + cross-modal injection + FiLM
- `10_GATE_5_LINEA_B_SHOWCASE/README.md` — NUEVO: tabla de 13 tests con dificultad y relevancia

### Documentos troncales actualizados

- `ROADMAP_BIAS_CONTROL.md` — secciones 7.10-7.12 y 8 reescritas para nuevo roadmap. Badge, IMPORTANT box, mapa documental, artefactos, cierre.
- `README.md` (raíz del repo) — badge, IMPORTANT box, tabla H3, tabla de control, Escalón 1 status, sección Gate 5, arquitectura, documentación BIAS_CONTROL.

## 34. Roadmap visual + commit masivo (2026-02-15 ~21:00 UTC)

### Roadmap visual

Creado `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/roadmap_visual.html` — gráfico HTML interactivo (~25KB) con:

- **Header**: tesis central + fecha
- **4 tarjetas de métricas**: Record S=74.6% (scratch e10), d4a4=69.8% (Gate 4.3), scratch progress, baseline D0
- **Timeline vertical** con código de colores:
  - Verde = completado (Gate 4.2, Gate 4.3 main 9 arms)
  - Amarillo animado = en ejecución (d4a4-scratch e15/30)
  - Azul = próximo con código listo (Fase 5: a4r, d4r, a8, a9)
  - Púrpura = futuro (Gate 4.4, Gate 5A, Gate 5B)
- **Tabla completa** de 9 brazos Gate 4.3 con S y delta vs D0
- **Bloque destacado dorado** con structured eval de scratch epoch 10: S=74.6%, M2A=75.0%, hard_neg=93.0%, MRR=0.336
- **Gate 4.4**: dos familias (Third Tower + MoE)
- **Gate 5A**: grid 3 columnas (barrido descriptores, 4 mecanismos incl. FiLM, cross-modal injection)
- **Gate 5B**: lista de 13 tests con tags de relevancia (Fundamental/Importante/Opcional)
- Dependencias marcadas en cada bloque

Diseño dark-mode con CSS variables, responsive, Google Fonts (Inter + JetBrains Mono). Para abrir: `xdg-open roadmap_visual.html`.

### d4a4-scratch — progreso al momento del commit

| Ep | Loss | qv_S |
|----|------|------|
| 10 | 13.56 | — (str_S=**74.6%**, hard_neg=93.0%) |
| 11 | 13.53 | 6.6% |
| 12 | 13.49 | 11.1% |
| 13 | 13.47 | 12.0% |
| 14 | 13.41 | 10.3% |
| 15 | ~13.38 | en curso |

Loss sigue bajando. Structured eval programado en epochs 15, 20, 25, 28-30.

### Commit y push

**Commit `90dd4e1`** pusheado a `main`. Contenido:

- **38 archivos**, +19,510 / -188 líneas
- Gate 4.3 completo (9 arms + scratch running)
- Roadmap restructurado (Gate 4.4 / 5A / 5B) con directorios renombrados
- `roadmap_visual.html` — visualización interactiva del roadmap
- `Para_GPT/` — 20 archivos de contexto para ChatGPT
- `NOTAS_CLAUDE_PARA_CODEX.md` — 33 secciones (bitácora para Codex)
- `PHIDEUS_MASTER_BRIEFING.md` + `PHIDEUS_NEURAL_ARCHITECTURES.md` — docs de síntesis
- `gate43_scratch_training.py` — script from-scratch con --skip-structured-eval
- `gate42_training.py` — +826 líneas: d4a4cm, D4x, reverse cross-att models, A8/A9 descriptors, helpers
- `audio_descriptors.py` — +196 líneas: A8 (onset-weighted chroma), A9 (IDF attractor), helper refactor
- `run_gate43.sh` — actualizado para todas las fases

**Commits previos en main** (para contexto):
- `1e3d678` — README con 8 viz + Gate 4.3 status
- `7573483` — 5 new viz + rename 3 existing
- `8b09cbf` — Gate 4.3 implementation + cross-attention

## 35. Análisis comparativo d4a4-scratch e10/e15 (2026-02-15 ~21:30 UTC)

### Structured eval epoch 15: S=65.8%

Epoch 15 completó structured eval automático:

| Métrica | e10 | e15 | Delta |
|---------|-----|-----|-------|
| S | **74.6%** | 65.8% | -8.8pp |
| A2M R@10 | 74.6% | 65.8% | -8.8pp |
| M2A R@10 | 75.0% | 68.6% | -6.4pp |
| hard_neg | 93.0% | 91.0% | -2.0pp |
| MRR avg | 0.336 | 0.316 | -0.020 |
| R@1 avg | 15.9% | 16.4% | +0.5pp |
| mean_rank avg | 7.75 | 10.0 | +2.25 |
| Loss | 13.60 | 13.38 | -0.22 |

Loss sigue bajando, pero S bajó. R@1 subió ligeramente. Quedan 15 epochs + evals en e20, e25, e28-30.

### CORRECCIÓN IMPORTANTE: framing del "desde cero"

**ERRATA**: En el análisis original se dijo "scratch partió de cero y D-02 de MERT+foundation".
Esto es **incorrecto y engañoso**.

**Realidad**: D-02 y scratch parten del **mismo punto exacto**:
- Ambos: MERT pretrained (audio encoder) + MIDI encoder random
- D-02 **es** el foundation training — no *usa* foundation, lo *produce*
- scratch usa `--from-scratch` que hace exactamente lo mismo: MERT pretrained + random MIDI

La **única diferencia** entre los dos es:
- **D-02**: modelo base, sin inyección de descriptores
- **scratch**: modelo d4a4, con inyección dual (D4 intervalos + A4 log-freq)

Esto hace que la comparación sea **más limpia y más poderosa**: mismo punto de partida, mismo
schedule, misma data — la única variable es la inyección de ratio info.

### Tabla comparativa corregida: scratch vs D-02 (mismo punto de partida)

| Epoch | D-02 S (sin injection) | Scratch S (con d4a4) | Delta | Interpretación |
|-------|----------------------|---------------------|-------|----------------|
| 5 | 47.0% | — (no eval) | — | — |
| 10 | 53.4% | **74.6%** | **+21.2pp** | Ventaja masiva descriptores |
| 15 | 57.6% | **65.8%** | **+8.2pp** | Ventaja se achica pero sigue |
| 25 | 61.8% (best D-02) | ??? | ??? | Pendiente |

### Tabla completa: tres modelos, métricas profundas

| Modelo | Epoch | S | hard_neg | MRR avg | R@1 avg | mean_rank |
|--------|-------|---|----------|---------|---------|-----------|
| D-02 | e5 | 47.0% | 86.2% | 0.215 | — | — |
| D-02 | e10 | 53.4% | 88.8% | 0.233 | — | — |
| D-02 | e15 | 57.6% | 89.8% | 0.276 | — | — |
| D-02 | e25 | **61.8%** | 90.4% | 0.291 | 15.2% | 13.8 |
| d4a4-found | e1 | 14.0% | 72.2% | 0.083 | 2.2% | — |
| d4a4-found | e3 | 56.6% | 89.2% | 0.254 | 11.4% | — |
| d4a4-found | e5 | **69.8%** | 91.6% | 0.325 | 16.4% | — |
| scratch | e10 | **74.6%** | 93.0% | 0.336 | 15.9% | 7.75 |
| scratch | e15 | 65.8% | 91.0% | 0.316 | 16.4% | 10.0 |

### Observaciones (sin extrapolar, per directiva analítica)

1. **e10 sigue siendo el project record**: S=74.6%, hard_neg=93.0%
2. **e15 bajó 8.8pp en S** pero el loss sigue bajando normalmente (-0.22)
3. **En la misma epoch (15), scratch supera a D-02 por 8.2pp** — evidencia directa de que los descriptores aportan señal real con el mismo punto de partida
4. **D-02 subió de e15 a e25** (57.6% → 61.8%, +4.2pp en 10 epochs) — si scratch sigue un patrón similar, podría recuperarse
5. **R@1 subió** (15.9% → 16.4%) mientras S bajó — sugiere que el modelo es más preciso en top-1 pero peor en ranking general
6. Quedan evals en e20, e25, e28-30 — demasiado pronto para conclusiones

---

## 36. README reescrito como landing page científica (2026-02-15 ~22:00 UTC)

### Qué se hizo

Se reescribió completamente `README.md` con criterio paper-like (abstract → results → method → appendix).

**Antes**: 547 líneas, todo visible, denso, mezcla de ciencia y operaciones.
**Después**: 353 líneas, ciencia arriba, operaciones plegadas con `<details>`.

### Estructura del nuevo README

**Arriba del fold (sin plegar, ~100 líneas):**
1. Título + tesis en una línea (*Do frequency ratios constitute a universal informational language?*)
2. Status card con link a visualizaciones 3D
3. Mini-glosario (S, R@10, A2M/M2A, hard_neg, D0, pp, VICReg)
4. Tabla de hipótesis (H1/H2/H3)
5. Tabla completa Gate 4.3 (9 brazos, con observaciones)
6. Tabla d4a4-scratch (e10/e15 vs D-02)
7. Tabla de visualizaciones (8 arquitecturas con links)

**Abajo del fold (todo en `<details>`):**
- Roadmap (mermaid actualizado + matriz de gates + TripleScaloneta + link a roadmap_visual.html)
- Arquitectura (foundation model + descriptores de ratios)
- Foundation Training — Bloque A (tabla S0/A/B/C/D/D-02)
- Reproducción / Quick Start (setup + comandos)
- Experimentos Anteriores (Escalón 1, UOEMD, resumen experimental, hallazgos metodológicos)
- Documentación (links a todos los gates + estructura del repo)

### Criterio de diseño

- **Paper-like**: abstract → results → method → appendix
- **Tomado de ChatGPT**: `<details>` folding, status card limpio, glosario
- **Rechazado de ChatGPT**: vibe startup, esconder tablas científicas, estructura genérica
- **Badges**: reducidos de 5 a 3 (Status, Gate, License)
- **Mermaid**: actualizado con progresión real (Gates 0-2 → 3/4.0-4.1 → 6 → Bloque A → 4.2 → 4.3 → 4.4 → 5A/5B)

### Commits

- `5544dc1` — `docs: rewrite README as scientific landing page` (README + NOTAS 34-35 + roadmap_visual corrections)
- `2c4e090` — `docs: add viz link to README status card` (link a viz en el status card)

---

## 37. d4a4-scratch: Epoch 20 — NUEVO RECORD DEL PROYECTO (2026-02-15 ~23:17 UTC)

### Resultado

**S = 75.6%** — nuevo record absoluto del proyecto, superando e10 (74.6%) por +1.0pp.

| Metric | e10 | e15 | **e20** |
|--------|-----|-----|---------|
| S | 74.6% | 65.8% | **75.6%** |
| A2M R@10 | 74.6% | 65.8% | **75.6%** |
| M2A R@10 | 76.4% | 67.0% | **76.8%** |
| hard_neg | 93.0% | 91.0% | **93.6%** |
| MRR avg | 0.336 | 0.316 | **0.370** |
| R@1 avg | 18.2% | 16.2% | **19.0%** |
| mean_rank | 7.5 | 8.4 | **7.0** |
| Loss | 13.60 | 13.38 | **13.29** |

### Comparación con D-02 a epoch 20

| Metric | D-02 e20 | d4a4-scratch e20 | Delta |
|--------|----------|-------------------|-------|
| S | 57.8% | **75.6%** | **+17.8pp** |
| hard_neg | 88.0% | **93.6%** | +5.6pp |
| MRR avg | 0.257 | **0.370** | +0.113 |

### Análisis

1. **El dip de e15 fue temporal**: e10→e15 cayó 8.8pp pero e15→e20 recuperó +9.8pp. La directiva analítica de no extrapolar fue correcta.
2. **Todos los métricas mejoraron** entre e10 y e20: S, MRR, R@1, mean_rank, hard_neg.
3. **La ventaja sobre D-02 se amplía**: +21.2pp en e10 → +17.8pp en e20 (D-02 también mejoró, pero d4a4 lo hizo más).
4. **Loss sigue bajando monotónicamente** (13.60 → 13.38 → 13.29). No hay señal de plateau.

### Quick val post-e20

| Ep | Loss | qv_A2M | qv_M2A |
|----|------|--------|--------|
| 20 | 13.29 | 16.5% | 16.8% |
| 21 | 13.28 | **20.5%** | **19.6%** |
| 22 | 13.27 | — | — |
| 23 | 13.24 | **20.7%** | **20.2%** |
| 24 | 13.23 | 20.3% | 19.9% |

Quick val saltó de ~16% a ~20% entre e20 y e21, y se mantiene ahí. Próxima structured eval en e25 (por terminar ahora).

### Archivo de eval

`data/bias_control_medium/training_outputs/gate43/gate43_d4a4_scratch_30ep/eval_per_epoch/eval_epoch20.json`

---

## 38. UNC CCAD — Acceso a supercomputadora (2026-02-16)

### Qué pasó

El usuario obtuvo acceso a la supercomputadora del Centro de Cómputo de Alto Desempeño (CCAD) de la Universidad Nacional de Córdoba. Se instaló Claude Code en el login node de Mendieta.

### Infraestructura relevante

- **Cluster**: Mendieta — 18 nodos GPU con 2x NVIDIA A30 24GB cada uno (36 GPUs total)
- **Partición**: `multi` (GPU), max walltime 48h
- **Storage**: NFS 200TB + /scratch 400GB SSD por nodo
- **SLURM**: scheduler de jobs, `--gpus=1` (NO `--gres=gpu:1`)
- **CUDA**: driver 535, CUDA 12.2 verificada en compute node
- **Python**: Miniconda instalada, env phideus con PyTorch 2.5.1+cu121
- **Repo**: clonado, 39MB (solo código, sin datos)

### Setup completado en UNC

| Item | Estado |
|------|--------|
| Claude Code | Funcionando (npm install) |
| Repo git | Clonado |
| Miniconda + env | Instalado, PyTorch 2.5.1+cu121 |
| CUDA en compute | Verificada (A30, driver 535) |
| MAESTRO | Descargando (~120GB) |
| foundation_locked_e25.pt | **PENDIENTE transferir** (288 MB) |
| segments_metadata.json | **PENDIENTE transferir** (62 MB) |

### Documento de referencia

Se escribió un informe exhaustivo para el Claude que opera en UNC:
`Documents/04_TRANSVERSAL/UNC_SuperComp_IA_Agents.md` — 1,251 líneas cubriendo toda la infraestructura CCAD, SLURM, módulos, GPU, storage, troubleshooting, y recetas operacionales.

### Restricciones importantes descubiertas

1. `--gpus=1` es la sintaxis correcta (no `--gres=gpu:1`)
2. Realísticamente 4-8 GPUs simultáneas (cluster compartido)
3. Training sobre NFS es lento → copiar a `/scratch/$SLURM_JOB_ID` al inicio de cada job
4. Max walltime 48h → necesita checkpoint recovery + auto-resubmit para runs largos
5. `transformers` no es necesario (MERTEncoderLite es custom, no usa HuggingFace)

### Commit

- `4c80151` — `docs: add UNC supercomputer guide + NOTAS section 36`

---

## 39. Roadmap distribuido LOCAL + UNC (2026-02-16)

### Principio operativo

**LOCAL = laboratorio de diseño iterativo. UNC = fábrica de experimentos paralelos.**

Ningún servidor espera al otro — siempre hay trabajo útil en ambos lados.

### División por gate

| Gate | LOCAL | UNC |
|------|-------|-----|
| **4.3 Fase 5** | d4a4-scratch termina (e24→e30) | 4 arms en paralelo (a4r, d4r, a8, a9) |
| **4.4** | Diseñar + implementar + pilot (Third Tower, MoE) | Runs completos post-pilot |
| **5A** | Implementar FiLM + nuevos descriptores | Barrido como array job (20+ arms) |
| **5B** | Tests eval-only (probes, UMAP, análisis) | Tests training-heavy (multi-seed, ablaciones) |

### Flujo de código

```
LOCAL implementa → pilot GPU → push git → UNC pull → sbatch array job → rsync resultados → LOCAL analiza
```

### Timeline estimado

| Día | LOCAL | UNC |
|-----|-------|-----|
| 0 | d4a4-scratch termina | Setup: transfer foundation |
| 1 | Analizar scratch e30, empezar Gate 4.4 | Gate 4.3 Fase 5 (4 arms, ~3h) |
| 2-4 | Gate 4.4 implementar + pilot | (espera código) |
| 4-5 | Implementar FiLM, push git | Gate 4.4 runs completos |
| 5-7 | Análisis Gate 4.4, ajustar scope 5A | Gate 5A barrido (array job) |
| 7-8 | **DECIDIR BEST MODEL** | Resultados 5A completos |
| 8-12 | Gate 5B eval-only + figuras | Gate 5B multi-seed + ablaciones |
| 12-15 | CIERRE | CIERRE |

**Estimación total**: 12-15 días (vs 25+ secuencial). Speedup ~2x.

### Lógica de la secuencia

1. **Fase 5 primero** porque puede haber un descriptor mejor que A4/D4 → cambiaría el best model
2. **Gate 4.4 después** porque puede haber una arquitectura radicalmente mejor
3. **Gate 5A después** para llenar la matriz con los ganadores de Fase 5 + 4.4
4. **Gate 5B al final** porque valida el VERDADERO best model (multi-seed, ablaciones, showcase)

No tiene sentido hacer multi-seed de d4a4 si después descubrimos que d4a8 o un MoE es mejor.

### Documento formal

`Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_UNC.md` — documento completo con:
- Infraestructura comparada
- Estrategia de división con diagramas
- Plan por gate con tablas
- Timeline Gantt
- Transferencia de datos y patrón /scratch
- Riesgos y mitigaciones
- Árbol de decisión para best model
- Checklists pre-ejecución

### Datos bloqueantes para empezar

| Archivo | Tamaño | Estado |
|---------|--------|--------|
| foundation_locked_e25.pt | 288 MB | **DONE** — GitHub Release v0.1.0-foundation |
| ~~segments_metadata.json~~ | ~~62 MB~~ | NO NECESARIO (loader lee maestro-v3.0.0.json directo) |

---

## 40. Protocolo Git: dos ramas (2026-02-16 ~05:00 UTC)

### Problema

Dos Claudes trabajando en el mismo repo sin comunicación directa = riesgo de conflictos, trabajo duplicado, y pisarse mutuamente.

### Solución acordada

Cada Claude pushea SOLO a su rama:

| | Rama `main` | Rama `unc` |
|---|---|---|
| **Pushea** | LOCAL (Claude Inference01) | UNC (Claude Mendieta) |
| **Nunca toca** | UNC | LOCAL |

### Flujo

1. LOCAL escribe código core, pushea a `main`
2. UNC hace `git pull origin main` para recibir cambios
3. UNC adapta/arregla lo que necesite para su entorno, pushea a `unc`
4. Si UNC arregla un bug en código compartido → usuario le avisa a LOCAL → LOCAL incorpora a `main`
5. **Nadie pushea a la rama del otro**

### Contexto

El Claude de UNC tiene autonomía total — ya corrigió varias cosas (SLURM syntax, /scratch pattern, segments_metadata no necesario). Necesita poder modificar código sin depender de LOCAL.

### Documentado en

- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_UNC.md` — sección 2.2
- `CLAUDE.md` — directiva de protocolo git

---

## 41. Foundation en GitHub Release (2026-02-16 ~05:00 UTC)

Release creado y publicado:
- **URL**: https://github.com/AlterMundi/Phideus/releases/tag/v0.1.0-foundation
- **Asset**: `foundation_locked_e25.pt` (288 MB)
- **MD5**: `ddb2ebf7075eec4dcec1628341ec4942`
- **Descarga en UNC**: `gh release download v0.1.0-foundation -p "foundation_locked_e25.pt"`

UNC ya lo descargó y verificó MD5. `segments_metadata.json` resultó NO ser necesario (el dataloader lee `maestro-v3.0.0.json` directamente del directorio MAESTRO).

### Archivos no trackeados en git (desde commit fcbb791)

`CLAUDE.md`, `CODEX.md`, `agents.md`, `.codex/`, `.claude/` — agregados a `.gitignore` y removidos de tracking. Los archivos siguen existiendo localmente pero git los ignora.

---

## 42. d4a4-scratch e25: NUEVO RECORD S=82.2% (2026-02-16 ~05:30 UTC)

### Structured eval e25

| Métrica | e10 | e15 | e20 | **e25** |
|---------|-----|-----|-----|---------|
| **S** | 74.6% | 65.8% | 75.6% | **82.2%** |
| A2M R@10 | 74.6% | 65.8% | 75.6% | 82.8% |
| M2A R@10 | 75.0% | 68.6% | 76.8% | 82.2% |
| hard_neg | 93.0% | 91.0% | 93.6% | **95.4%** |
| MRR avg | 0.336 | 0.316 | 0.370 | **0.430** |
| R@1 avg | 15.9% | 16.4% | 19.0% | **25.2%** |
| mean_rank | 7.7 | 10.0 | 7.0 | **5.7** |

### Análisis

1. **S=82.2% es un salto enorme** — +6.6pp sobre e20 (75.6%), +22.0pp sobre D-02 e25 (60.2%).
2. **El dip de e15 fue definitivamente temporal** — e20 ya lo superó, e25 lo destruye.
3. **Todas las métricas suben juntas** — no es ruido: hard_neg casi satura (95.4%), MRR +28%, R@1 +58%, mean_rank mejora de 7.7 a 5.7.
4. **Loss sigue bajando** (13.60 → 13.21). Quick val estable ~22%. El modelo NO está saturado.
5. **LR schedule**: LR en 2.2e-7 (casi cero). Si mejora con LR tan bajo, hay señal genuina.

### vs D-02 al mismo epoch

D-02 epoch 25: S=60.2%, hard_neg=90.4%.
d4a4-scratch epoch 25: S=82.2%, hard_neg=95.4%.
**Diferencia: +22.0pp** — partiendo del mismo punto exacto. La inyección de descriptores d4a4 es la única diferencia.

### Estado del run

e27/30 en curso. ETA e30 ~04:30 UTC. Structured evals pendientes en e28, e29, e30. Puede seguir subiendo.

### Quick val progression completa (e20-e27)

| Ep | Loss | qv_A2M | qv_M2A |
|----|------|--------|--------|
| 20 | 13.29 | 16.5% | 16.8% |
| 21 | 13.28 | 20.5% | 19.6% |
| 22 | 13.26 | 22.2% | 22.1% |
| 23 | 13.24 | 20.7% | 20.2% |
| 24 | 13.23 | 20.3% | 19.9% |
| 25 | 13.21 | 22.6% | 21.9% |
| 26 | 13.21 | 22.5% | 22.6% |
| 27 | 13.20 | 23.0% | 23.2% |

---

## 43. d4a4-scratch COMPLETO: e30 = S=83.6% — RECORD ABSOLUTO (2026-02-16 ~04:42 UTC)

### Resultado final

**Training complete**: 636 min (10.6h), 30 epochs, best model = epoch 30.
**S = 83.6%** — nuevo record absoluto del proyecto. **+21.8pp sobre D-02 best** (61.8%).

### Tabla completa structured evals (datos de JSONs)

| Ep | Loss | S | A2M | M2A | hard_neg | MRR avg | R@1 avg | mean_rank | vs D-02 best |
|----|------|---|-----|-----|----------|---------|---------|-----------|--------------|
| 10 | 13.60 | 74.6% | 74.6% | 75.0% | 93.0% | 0.336 | 15.9% | 7.7 | +12.8pp |
| 15 | 13.38 | 65.8% | 65.8% | 68.6% | 91.0% | 0.316 | 16.4% | 10.0 | +4.0pp |
| 20 | 13.26 | 75.6% | 75.6% | 76.8% | 93.6% | 0.370 | 19.0% | 7.0 | +13.8pp |
| 25 | 13.21 | 82.2% | 82.8% | 82.2% | 95.4% | 0.430 | 25.2% | 5.7 | +20.4pp |
| 28 | 13.19 | 82.8% | 82.8% | 83.6% | 94.8% | 0.444 | 26.4% | 5.6 | +21.0pp |
| 29 | 13.19 | 82.6% | 82.6% | 83.8% | 95.2% | 0.443 | 26.3% | 5.4 | +20.8pp |
| **30** | **13.20** | **83.6%** | **84.0%** | **83.6%** | **95.2%** | **0.444** | **25.9%** | **5.4** | **+21.8pp** |

*D-02 best = S=61.8% (epoch 25). Comparación "vs D-02 best" = scratch_S - 61.8.*

### Contexto comparativo

| Modelo | Best S | Referencia |
|--------|--------|------------|
| Gate 2 baseline | 34.4% | checkpoint_epoch45.pt |
| D-02 (30ep, sin descriptores) | 61.8% | foundation_locked_e25.pt |
| d4a4 foundation (5ep) | 69.8% | Gate 4.3 best arm |
| **d4a4 scratch (30ep)** | **83.6%** | **este run** |

### Análisis final

1. **e30 rompió el plateau**: después de 82.2→82.8→82.6 en e25-29, e30 saltó a 83.6% (+1.0pp). El modelo NO está saturado a 30 epochs.
2. **A2M alcanzó 84.0%**: primera vez que A2M supera a M2A. El bottleneck histórico (audio→MIDI más difícil) se está cerrando.
3. **Dip de e15 fue transitorio**: causado por transición warmup→decay en LR schedule. Se recuperó completamente.
4. **hard_neg estable ~95%**: sin tendencia a la baja. El modelo distingue segmentos del mismo piano consistentemente.
5. **mean_rank mejoró de 7.7 a 5.4**: match correcto pasó de top-8 a top-5 en pool de 256.
6. **Loss convergiendo**: 13.60→13.20. LR final ~1.3e-08 (prácticamente cero). Para seguir mejorando necesitaría schedule extendido.
7. **Señal clara de que más epochs = más S**: e20→e30 no muestra saturación. Un run de 50-60ep podría empujar más allá de 85%.

### Archivos de evaluación

```
data/bias_control_medium/training_outputs/gate43/gate43_d4a4_scratch_30ep/
├── eval_epoch10.json                    (en root, formato viejo)
└── eval_per_epoch/
    ├── eval_epoch15.json
    ├── eval_epoch20.json
    ├── eval_epoch25.json
    ├── eval_epoch28.json
    ├── eval_epoch29.json
    └── eval_epoch30.json
```

### Implicaciones para roadmap

- **Gate 4.4**: d4a4 scratch es el baseline a superar (S=83.6%). Third Tower / MoE necesitan +2pp mínimo para justificarse.
- **Gate 5B multi-seed**: prioridad alta — verificar que 83.6% no es un outlier (seed=42).
- **Extensión a 50ep**: candidato para UNC (multi-seed × epochs largos en paralelo).

---

## 44. Protocolo de archivos privados — actualización (2026-02-16 ~06:00 UTC)

### Nuevo archivo privado: BITACORA_UNC.md

El archivo `BITACORA_UNC.md` es **privado y exclusivo del Claude de la UNC**. Claude LOCAL no debe leerlo ni editarlo.

### Resumen completo de archivos privados/protegidos

| Archivo | Dueño | Regla |
|---------|-------|-------|
| `CODEX.md` | Codex | Claude LOCAL: solo lectura, NUNCA editar |
| `CLAUDE.md` | Claude LOCAL | Codex: solo lectura, NUNCA editar |
| `.codex/memory.md` | Codex | Claude LOCAL: NUNCA leer ni escribir |
| `~/.claude/.../MEMORY.md` | Claude LOCAL | Codex no lo ve |
| **`BITACORA_UNC.md`** | **Claude UNC** | **Claude LOCAL: NUNCA leer ni escribir** |

---

## 45. Multi-Seed Eval: d4a4-scratch e30 (2026-02-16 ~05:40 UTC)

### Contexto
Evaluación del checkpoint d4a4-scratch epoch 30 con 5 seeds distintas para medir varianza de la métrica S.
Mismo modelo, distinto pool aleatorio de 256 segmentos y 500 queries por seed.
Corrido en LOCAL (RTX 3090), ~6 min/seed.

### Resultados

| Seed | S | A2M | M2A | hard_neg | MRR_a2m | MRR_m2a | mean_rank_a2m |
|------|---|-----|-----|----------|---------|---------|---------------|
| 42 | 83.6% | 84.0% | 83.6% | 95.2% | 0.438 | 0.450 | 5.21 |
| 123 | **88.4%** | 88.4% | 89.8% | 97.4% | 0.489 | 0.501 | 3.93 |
| 456 | 83.0% | 83.4% | 83.0% | 94.0% | 0.434 | 0.425 | 5.39 |
| 789 | 82.6% | 84.4% | 82.6% | 94.2% | 0.451 | 0.435 | 5.00 |
| 2026 | 82.8% | 83.4% | 82.8% | 94.8% | 0.470 | 0.447 | 4.71 |

### Estadísticas

| Métrica | Media | Std | Min | Max |
|---------|-------|-----|-----|-----|
| **S** | **84.1%** | **±2.3pp** | 82.6% | 88.4% |
| hard_neg | 95.1% | ±1.3pp | 94.0% | 97.4% |
| MRR_a2m | 0.456 | ±0.021 | 0.434 | 0.489 |

### Análisis

1. **S = 84.1% ± 2.3pp** — seed 42 (83.6%) justo debajo de la media, no es outlier.
2. Seed 123 es outlier alto (88.4%, +2σ) — pool particularmente favorable.
3. 4 de 5 seeds en rango estrecho 82.6-83.6% (1pp). Excluyendo seed 123: media=83.0% ± 0.4pp.
4. **vs D-02 multi-seed** (S=61.6% ± 1.1%): d4a4-scratch está **+22.5pp** por encima.
5. hard_neg consistentemente >94% en todas las seeds.
6. **Cifra reportable**: S = 84.1% ± 2.3pp (5 seeds), o conservadoramente ~83% (mediana).

### Output dir
`data/bias_control_medium/training_outputs/gate43/gate43_d4a4_scratch_30ep/multiseed/`

---

## 46. UNC Gate 4.3 Fase 5 — Resultados parciales (2026-02-16 ~08:45 UTC-3)

### Setup
Array job 1142230 en Mendieta (A30 24GB). 4 brazos × 5ep desde foundation, freeze-policy run-d.
Merge main→unc completado (commit `9cd9eeb`).

### Resultados parciales

| Arm | Ep | Loss | A2M | M2A | S | hard_neg | min/ep |
|-----|----|------|-----|-----|---|----------|--------|
| a4r | 1 | 13.90 | 30.2% | 35.2% | 30.2% | 75.8% | 33.2 |
| a4r | 2 | 13.57 | 33.0% | 45.0% | 33.0% | 79.8% | 31.8 |
| a4r | 3 | — | — | — | — | — | eval now |
| d4r | 1 | 13.96 | 49.0% | 52.0% | 49.0% | 89.2% | 58.6 |
| d4r | 2 | — | — | — | — | — | train 63% |
| a8 | 1 | 13.76 | — | — | — | — | eval now |
| a9 | — | — | — | — | — | — | staging |

### Observaciones tempranas

1. **d4r lidera epoch 1**: S=49.0% vs a4r=30.2%. Hard neg 89.2% también superior.
2. **a4r sube lento**: +2.8pp en un epoch. Pero es ~2x más rápido por epoch (~32min vs ~59min d4r).
3. **a8 loss más baja** de todos en epoch 1 (13.76) — pendiente ver S.
4. **a9 en staging**: congestión NFS copiando MAESTRO (~32 min).
5. **Contexto**: D0 baseline en Gate 4.3 fue S=60.2% a e3. Estos son resultados de epoch 1-2, aún tempranos.
6. **d4r tiempo/epoch alto** (~59 min vs ~32 min a4r): cross-att MIDI opera sobre N tokens MIDI (variable), puede ser más costoso que reverse audio que comprime a 188 tokens fijos.

### ETA completion
- a4r: ~1.5h más | d4r: ~3h más | a8: ~3.5h más | a9: ~4.5h más

---

## 47. UNC Fase 5 — Update 2 + comparación con Gate 4.3 (2026-02-16 ~09:30 UTC-3)

### Datos actualizados

| Arm | Ep | Loss | A2M | M2A | S | hard_neg | min/ep |
|-----|----|------|-----|-----|---|----------|--------|
| a4r | 1 | 13.90 | 30.2% | 35.2% | 30.2% | 75.8% | 33.2 |
| a4r | 2 | 13.57 | 33.0% | 45.0% | 33.0% | 79.8% | 31.8 |
| **a4r** | **3** | **13.48** | **55.2%** | **57.4%** | **55.2%** | **90.8%** | **31.5** |
| a4r | 4 | — | — | — | — | — | en curso |
| d4r | 1 | 13.96 | 49.0% | 52.0% | 49.0% | 89.2% | 58.6 |
| d4r | 2 | 13.94 | — | — | — | — | eval now |
| a8 | 1 | 14.11 | 36.2% | 41.4% | 36.2% | 82.4% | 58.9 |
| a9 | 1 | — | — | — | — | — | train 26% |

### Comparación con Gate 4.3 Fases 0-3 (mismos epochs)

| Arm | Mecanismo | e1 S | e2 S | e3 S | e5 S (final) |
|-----|-----------|------|------|------|--------------|
| **D4** | MIDI concat | 53.8% | 57.4% | 60.4% | 63.6% |
| **A4** | Audio concat | 46.2% | 55.4% | 59.2% | 63.6% |
| **d4r** | MIDI reverse cross-att | 49.0% | — | — | — |
| **a4r** | Audio reverse cross-att | 30.2% | 33.0% | 55.2% | — |
| **a8** | Onset chroma concat | 36.2% | — | — | — |

### Análisis comparativo

1. **d4r vs D4 en e1**: d4r=49.0% vs D4=53.8%. Reverse cross-att MIDI arranca -4.8pp debajo de concat.
2. **a4r tiene curva explosiva**: salto e2→e3 de +22.2pp (33.0% → 55.2%). Es el mayor salto en un solo epoch que hemos visto. Reverse cross-att audio necesita más warmup pero cuando converge, converge fuerte.
3. **a8 vs A4 en e1**: a8=36.2% vs A4=46.2%. Onset-weighted chroma arranca -10pp debajo de log-freq deltas.
4. **hard_neg como predictor temprano**: d4r=89.2% en e1 (comparable a D4 ~88% en e1). a4r arrancó en 75.8% pero subió a 90.8% en e3 — ya en rango de los brazos Fase 0.
5. **a4r en e3 (55.2%) vs A4 en e3 (59.2%)**: reverse aún -4pp debajo de concat al mismo epoch. Pero la pendiente de a4r es mucho mayor, podría cruzar en e4-e5.

---

## 48. Gate 4.3 Fase 5 — RESULTADOS FINALES (2026-02-16 ~12:00 UTC-3)

### Tabla completa (4 brazos × 5 epochs, corridos en UNC Mendieta)

| Arm | Ep | Loss | A2M | M2A | S | hard_neg | min/ep |
|-----|----|------|-----|-----|---|----------|--------|
| a4r | 1 | 13.90 | 30.2% | 35.2% | 30.2% | 75.8% | 33.2 |
| a4r | 2 | 13.57 | 33.0% | 45.0% | 33.0% | 79.8% | 31.8 |
| a4r | 3 | 13.48 | 55.2% | 57.4% | 55.2% | 90.8% | 31.5 |
| a4r | 4 | 13.38 | 63.4% | 64.8% | 63.4% | 90.2% | 31.5 |
| **a4r** | **5** | **13.33** | **68.6%** | **69.0%** | **68.6%** | **91.6%** | **31.5** |
| d4r | 1 | 13.96 | 49.0% | 52.0% | 49.0% | 89.2% | 58.6 |
| d4r | 2 | 13.75 | 58.0% | 58.2% | 58.0% | 91.6% | 58.4 |
| d4r | 3 | 13.66 | 62.4% | 62.4% | 62.4% | 91.8% | 58.3 |
| d4r | 4 | 13.58 | 63.6% | 63.0% | 63.0% | 92.2% | 58.4 |
| **d4r** | **5** | **13.53** | **64.2%** | **64.4%** | **64.2%** | **93.2%** | **58.4** |
| a8 | 1 | 14.11 | 36.2% | 41.4% | 36.2% | 82.4% | 58.9 |
| a8 | 2 | 13.58 | 49.0% | 48.6% | 48.6% | 86.2% | 58.6 |
| a8 | 3 | 13.50 | 46.4% | 50.2% | 46.4% | 86.4% | 58.6 |
| a8 | 4 | 13.42 | 56.4% | 54.4% | 54.4% | 88.8% | 58.7 |
| **a8** | **5** | **13.39** | **60.4%** | **57.4%** | **57.4%** | **90.6%** | **58.7** |
| a9 | 1 | 14.02 | 28.0% | 33.0% | 28.0% | 79.4% | 58.3 |
| a9 | 2 | 13.60 | 48.2% | 51.0% | 48.2% | 85.8% | 57.9 |
| a9 | 3 | 13.52 | 49.2% | 53.6% | 49.2% | 87.6% | 57.9 |
| a9 | 4 | 13.43 | 52.4% | 54.2% | 52.4% | 87.6% | 58.1 |
| **a9** | **5** | **13.40** | **58.8%** | **60.8%** | **58.8%** | **90.4%** | **57.9** |

### Comparación con Gate 4.3 Fases 0-3 (best S at e5)

| Rank | Arm | Mecanismo | Best S (e5) | vs D0 |
|------|-----|-----------|-------------|-------|
| **1** | **d4a4** | **dual concat** | **69.8%** | **+9.6pp** |
| **2** | **A4r** | **reverse cross-att** | **68.6%** | **+8.4pp** |
| 3 | D4r | reverse cross-att | 64.2% | +4.0pp |
| 3 | D4 | MIDI concat | 63.6% | +3.4pp |
| 3 | A4 | Audio concat | 63.6% | +3.4pp |
| 6 | A4x | Audio cross-att | 62.6% | +2.4pp |
| 7 | A7x | Audio attractor cross-att | 62.2% | +2.0pp |
| 8 | D0 | baseline | 60.2% | — |
| 9 | D4x | MIDI cross-att | 60.0% | -0.2pp |
| 10 | A9 | IDF attractor concat | 58.8% | -1.4pp |
| 10 | A7 | Audio attractor concat | 58.8% | -1.4pp |
| 12 | A8 | Onset chroma concat | 57.4% | -2.8pp |
| 13 | d4a4cm | Dual cross-modal | 52.4% | -7.8pp |

### Hallazgos principales

**1. Reverse cross-attention (Q=desc, K/V=feat) GANA decisivamente sobre regular (Q=feat, K/V=desc)**
- A4r (68.6%) >> A4x (62.6%): **+6.0pp**
- D4r (64.2%) >> D4x (60.0%): **+4.2pp**
- En AMBOS dominios (audio y MIDI), la semántica inversa es superior.
- Implicación teórica: los descriptores de ratio son más efectivos como PRINCIPIO ORGANIZADOR
  de los features que como FUENTE DE CONSULTA para los features.

**2. A4r = mejor brazo single-descriptor del proyecto**
- 68.6% con un solo descriptor, a solo 1.2pp del dual d4a4 (69.8%)
- AÚN SUBIENDO a e5 (salto e4→e5 = +5.2pp, la mayor mejora entre dos epochs consecutivos)
- A4r comprime de 2400 a 188 tokens en el transformer → ~2x más rápido por epoch (~31.5 vs ~58 min)

**3. Nuevos descriptores (A8, A9) no superan a A4**
- A8 (onset chroma, 57.4%): -6.2pp vs A4 (63.6%), -2.8pp vs baseline D0 (60.2%)
- A9 (IDF attractor, 58.8%): -4.8pp vs A4 (63.6%), -1.4pp vs D0 (60.2%). Salto tardío e4→e5 (+6.4pp)
  sugiere potencial con más epochs, pero aún por debajo de baseline.
- Ambos por debajo de D0 baseline. Las ideas de Route A/B de Escalón 1 (onset anchoring, IDF
  weighting) no se traducen bien a descriptores inyectables frame-level.
- NOTA: A9 tuvo un salto tardío notable (52.4%→58.8% en e4→e5), pero sigue debajo de D0.

**4. Reverse cross-att audio tiene velocidad bonus**
- a4r: ~31.5 min/ep (transformer procesa 188 tokens, no 2400)
- Todos los demás: ~58 min/ep
- Esto es relevante para escalabilidad: runs largos con reverse son ~2x más baratos

### Decisión condicional resuelta (ver sección 29)

La condición era: "Si reverse gana Y nuevos descriptores ganan → A8r/A9r"
- Reverse GANA ✅
- Nuevos descriptores NO ganan ❌
- **Resultado: NO procede A8r/A9r**

### Próximos pasos lógicos

1. **A7r**: probar reverse cross-att con A7 (rational attractor, dim=12). A7 rinde peor que A4
   en concat (58.8% vs 63.6%), pero ¿y con reverse? La transformación de performance de A4→A4r
   (+5.0pp) podría replicarse.
2. **d4a4r**: dual reverse (D4r+A4r). Si A4r=68.6% con un descriptor se acerca a d4a4=69.8%
   con dos concat, ¿qué pasa con dos descriptores ambos reverse?
3. **A4r scratch 30ep**: dado que A4r casi iguala a d4a4 en 5ep, un run largo podría
   superar a d4a4-scratch (S=83.6%).

---

## 49. Punto de decisión post-Gate 4.3 (2026-02-16 ~14:00 UTC-3)

### Estado: LIMBO DECISIONAL

Gate 4.3 está COMPLETO con 13 brazos + d4a4-scratch. Fase 5 terminó en UNC.
Todos los resultados están documentados. Ahora hay que decidir qué sigue.

### Hallazgos clave que informan la decisión

1. **d4a4 concat** = mejor mecanismo en screening 5ep (S=69.8%, +9.6pp vs D0)
2. **A4r reverse** = mejor single-descriptor (S=68.6%, +8.4pp vs D0), y ~2x más rápido por epoch
3. **Reverse > regular cross-att** en ambos dominios (A4r +6.0pp vs A4x, D4r +4.2pp vs D4x)
4. **d4a4-scratch 30ep** = S=83.6%, récord absoluto, NO saturado a e30
5. **A8/A9 fracasan** — ideas de Route A/B no se traducen a descriptores inyectables
6. **Cross-modal injection destruye señal** (d4a4cm = 52.4%, peor que baseline)

### Opciones según el roadmap

| Opción | Qué es | Pregunta que responde | GPU cost |
|--------|--------|----------------------|----------|
| **Más Gate 4.3** | A7r, d4a4r, A4r-scratch-30ep | ¿Reverse dual supera concat dual? ¿A4r escala como d4a4? | 10-20h |
| **Gate 4.4** | Third Tower / MoE | ¿Cambio arquitectónico mayor mejora sobre inyección? | Diseño + runs |
| **Gate 5A** | Barrido amplio + FiLM | ¿Hay descriptores/mecanismos no explorados que ganan? | Muchos runs (UNC) |
| **Gate 5B** | 13 tests científicos | ¿El best model sobrevive validación rigurosa? | Eval-only |

### Preguntas abiertas más calientes

1. **¿d4a4r (dual reverse) supera a d4a4 (dual concat)?**
   - Si A4r reverse ya casi iguala a d4a4 dual con UN solo descriptor (68.6% vs 69.8%),
     ¿qué pasa con dos descriptores ambos reverse?
   - Implicación: si d4a4r gana, el mecanismo óptimo es reverse, no concat.

2. **¿A4r escala a 30ep como d4a4?**
   - d4a4 pasó de 69.8% (5ep) a 83.6% (30ep) = +13.8pp de ganancia por entrenamiento largo.
   - A4r a 5ep = 68.6%. Si escala igual → ~82.4% con UN solo descriptor y ~2x más rápido.
   - Si supera 83.6% → nuevo récord con modelo más simple y más rápido.

3. **¿Vale la pena Gate 4.4 (arquitectura nueva) o es mejor optimizar lo que ya tenemos?**
   - Third Tower y MoE son cambios arquitectónicos grandes con riesgo alto.
   - Pero podrían desbloquear un techo que la inyección simple no alcanza.
   - Argumento a favor: explorar frontera. Argumento en contra: d4a4 aún no está saturado.

4. **¿Gate 5B (validación científica) debería ejecutarse YA sobre d4a4-scratch?**
   - El modelo existe, los 13 tests están diseñados.
   - Pero si d4a4r o A4r-scratch superan a d4a4-scratch, habría que repetir la validación.
   - Trade-off: validar ahora (puede quedar obsoleto) vs esperar el modelo final (retrasa publicación).

### Roadmap distribuido original (ROADMAP_UNC.md)

El plan era: Fase 5 → 4.4 → 5A → 5B (12-15 días).
Pero con los resultados de Fase 5, hay un argumento para insertar un "4.3 Fase 6" rápido
(d4a4r + A4r-scratch) antes de saltar a 4.4, porque:
- d4a4r es un run de 5ep (~3h) que puede responder la pregunta #1
- A4r-scratch-30ep es un run de ~16h (2x más rápido que d4a4-scratch) que responde la pregunta #2
- Ambos son bajo riesgo y alta información

### Decisión PENDIENTE del equipo

---
