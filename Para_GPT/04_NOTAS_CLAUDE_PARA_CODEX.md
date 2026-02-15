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

## 9. Documentación que Codex necesita actualizar

1. **INFORME_GATE_4_3_RATIO_RE_CENTRICO.md**: Agregar:
   - Resultados finales A4 (5 epochs, tabla completa en sección 2)
   - Resultados pilots a4x/a7x (sección 4)
   - D4x como nuevo brazo (sección 7)
   - Resultados de Fase 1 (a7, a4x, a7x) — disponibles para el lunes
2. **plan_gate_4.3.md**: Sincronizar con diseño por fases (Fases 0-4)
3. **README de Gate 4.3**: YA actualizado por Claude (sección 8) — revisar y refinar
4. **ROADMAP_BIAS_CONTROL.md**: Reflejar las nuevas fases y el roadmap extendido
5. **CODEX.md**: Actualizar estado de Gate 4.3 si corresponde

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

### Arquitecturas cross-attention implementadas:

**Audio (A4x/A7x) — Gate42AudioCrossAttModel:**
```
CNN [B, 2400, 1024] → +pos_emb → cross_attn(Q=2400, K/V=188 STFT nativo) → +residual → LN → Transformer → pool → proj
```
- K/V a resolución nativa STFT (188 frames), NO interpolado. 12.8x ahorro de memoria.
- pos_emb ANTES de cross-attention (temporal awareness). NO se vuelve a sumar antes del Transformer.
- need_weights=False, embed_batch_size=16 en eval.

**MIDI (D4x) — Gate42MidiCrossAttModel (IMPLEMENTADO + CPU VERIFIED):**
```
Embedding [B, N, 512] → CLS? → +pos_enc → cross_attn(Q=N(+1), K/V=N intervals) → +residual → LN → Transformer → pool → proj
```
- Q y K/V a misma resolución (N tokens). Sin mismatch temporal.
- CLS prepended a Q si aggregation="cls". K/V siempre a N.
- Mismos principios: pos_enc pre-attn, residual + LN, need_weights=False.
- ~1.05M params nuevos (d=512 vs d=1024 en audio).

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

### Renumeración de Gates

El usuario decidió renumerar los gates futuros:

| Antes | Ahora | Contenido |
|-------|-------|-----------|
| Gate 4.3 Fase 4 (cross-modal injection) | **Gate 4.4** | 3 brazos CM-a, CM-m, CM-bi |
| Gate 4.4 (barrido bifurcado) | **Gate 4.5** | Barrido amplio de descriptores |
| Gate 4.5 (ratio bridge / third tower) | **Gate 4.6** | Third-tower architecture |

**Nota para Codex**: Los directorios `08_GATE_4_4_BIFURCACION_RATIO/` y `09_GATE_4_5_RATIO_BRIDGE/` necesitan renumerarse para reflejar esto. El contenido de `09_GATE_4_5_RATIO_BRIDGE/` pasa a ser Gate 4.6 conceptualmente.

### Secuencia operativa acordada

1. D4x termina (hoy)
2. Correr d4a4 + d4a4cm (5ep cada uno, ~6h total)
3. Cerrar Gate 4.3 con análisis comparativo completo
4. **Gate 4.4 NUEVO**: Cross-modal injection (3 brazos, ~9h)
5. Gate 4.5: Barrido amplio (post Gate 4.4)
6. Gate 4.6: Third tower (si resultados lo justifican)
7. Sweep final comprehensivo: mejores descriptores × mejores mecanismos

---

## 20. Commits hechos en esta sesión

| Commit | Fecha | Contenido |
|--------|-------|-----------|
| `7573483` | 2026-02-15 | 5 nuevas visualizaciones WebGL2 + rename 3 existentes (89 archivos, +9,456 LOC) |
| `1e3d678` | 2026-02-15 | README.md actualizado con 8 arquitecturas + estado Gate 4.3 |

Ambos pusheados a main. GitHub Pages desplegado con las 8 visualizaciones.

---

## 21. Archivos pendientes de commit (al momento de esta actualización)

| Archivo | Cambio |
|---------|--------|
| `experiments/bias_control/gate42_training.py` | +Gate42MidiCrossAttModel (D4x) + Gate42DualCrossModalModel (d4a4cm) + helpers + integración |
| `experiments/bias_control/run_gate43.sh` | Reescrito para Fases 1+2 |
| `CLAUDE.md` | Actualizaciones de estado |
| `NOTAS_CLAUDE_PARA_CODEX.md` | Este archivo |
| `Documents/.../07_GATE_4_3.../README.md` | Actualizado con resultados |
| `Documents/.../07_GATE_4_3.../D0_D4_A4_A7_A4x_A7x_result.md` | Nuevo: tabla de resultados |
| `Documents/.../09_GATE_4_5_RATIO_BRIDGE/` | Nuevo directorio (creado por Claude antes de la renumeración — ver sección 19) |

---

## 22. Tabla de referencia completa — todos los brazos Gate 4.3

| Brazo | Lado | Descriptor | Mecanismo | Params nuevos | Status | Best S | Best ep |
|-------|------|-----------|-----------|---------------|--------|--------|---------|
| D0 | — | — | baseline | 0 | **COMPLETE** | 60.2% | e3 |
| D4 | MIDI | intervals (4d) | concat | ~267K | **COMPLETE** | **63.6%** | e5 |
| A4 | Audio | log-freq (8d) | concat | ~1.06M | **COMPLETE** | **63.6%** | e5 |
| A7 | Audio | attractor (12d) | concat | ~1.06M | **COMPLETE** | 58.8% | e5 |
| A4x | Audio | log-freq (8d) | cross-att | ~4.2M | **COMPLETE** | 62.6% | e5 |
| A7x | Audio | attractor (12d) | cross-att | ~4.2M | **COMPLETE** | 62.2% | e5 |
| D4x | MIDI | intervals (4d) | cross-att | ~1.05M | **RUNNING** e3/5 | 58.4%* | e3* |
| d4a4 | Ambos | D4+A4 same-modality | concat | ~1.3M | PENDING | — | — |
| d4a4cm | Ambos | D4→audio + A4→MIDI | concat cross-modal | ~1.3M | CPU VERIFIED | — | — |

\* D4x parcial, faltan e4/e5.

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
