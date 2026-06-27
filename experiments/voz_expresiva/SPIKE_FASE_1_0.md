# Spike Fase 1.0 — verificación de compatibilidad de mecanismos heredados de E2

**Fecha**: 2026-06-22
**Objetivo**: antes de comprometer Fase 1 a "importar tal cual" los mecanismos de inyección de Escalón 2 (concat, FiLM, xattn), verificar shapes, puntos de inserción, semántica pre/post-pool y topología asumida.

**Conclusión corta**: los mecanismos heredados **NO son drop-in** para WavLM. Asumen una topología E2 distinta (CNN+Transformer propio, 512d, audio crudo como input). Pero las adaptaciones necesarias son pequeñas (~80-120 líneas totales en un wrapper nuevo `wavlm_injection.py`).

---

## 1. Topología asumida por los mecanismos de E2

### `SpeechEGGEncoder` (base, `speech_egg_encoder.py`)

Encoder standalone CNN+Transformer pensado para reemplazar el backbone:

```
waveform [B, T=32000]
  → Conv1d × 4 (k=10/3/3/3, stride 5/2/2/2)
  → features [B, 512, T'=800]  ← stride total = 40, ~400 Hz
  → transpose → [B, 800, 512]
  → + positional embedding (max 1000 positions)
  → Transformer (4 layers, 8 heads, d=512)
  → mean pool
  → [B, 512]
```

- Input: audio crudo a 16 kHz, ventana fija ~2 s (32000 samples).
- Output: 512-d utterance embedding.
- CNN propio (~15M params), pensado para training end-to-end.

**Mismatch fundamental con WavLM**: WavLM da `[B, T_frames, 1024]` directo desde audio crudo, sin exponer un punto "post-CNN pre-Transformer" donde inyectar. Es una caja negra frozen.

### `SpeechEGGEncoderAug` (concat) — `speech_egg_encoder_aug.py`

Subclase con inyección entre CNN y Transformer:

```python
forward(waveform, descriptor=None) → [B, 512]:
  features = self.feature_extractor(waveform)  # CNN propio → [B, 512, T'=800]
  features = features.transpose(1, 2)            # [B, 800, 512]
  if descriptor is not None:
      # Interpola descriptor a T'=800 si shape no coincide
      features = cat([features, descriptor], dim=-1)   # [B, 800, 512+D]
      features = self.descriptor_proj(features)         # [B, 800, 512]
  + pos_embedding → Transformer → mean
```

- Init near-identity: `W = [I | 0]`, bias = 0 → ep0 ≈ baseline.
- Inyección frame-level antes del Transformer.
- Hardcoded a 512 dims.

### `SpeechEGGEncoderXAttn` (xattn) — `speech_egg_encoder_xattn.py`

Subclase con cross-attention residual:

```python
forward(waveform, descriptor):
  features = self.feature_extractor(waveform)  # [B, 800, 512]
  if descriptor is not None:
      desc_q = self.desc_proj(descriptor)         # [B, 800, 512]
      xattn_out = MultiheadAttention(Q=desc_q, K=features, V=features)
      features += self.xattn_scale * LayerNorm(xattn_out)  # residual, init 0.01
  + pos_embedding → Transformer → mean
```

- Hardcoded a 512d. MultiheadAttention(embed_dim=512, heads=4).
- Init near-zero: `xattn_scale = 0.01` → ~1% perturbación en ep0.
- Cross-attention con Q=descriptor, K/V=features (hipótesis: H-series local, position-independent).

### `ConditionedProjectionHead` (FiLM) — `projection.py`

Cabeza de proyección INDEPENDIENTE del encoder:

```python
forward(x: [B, D_in], cond: [B, cond_dim]) → [B, D_out]:
  for i in range(2):  # 2 hidden blocks
      x = ReLU(BN(Linear(x)))
      gamma, beta = self.film_generators[i](cond).chunk(2)  # zero-init
      x = (1 + gamma) * x + beta
  return self.final_linear(x)
```

- **Es utterance-level**, no frame-level: input `[B, D_in]` ya pooled, cond `[B, cond_dim]` ya pooled.
- Zero-init en FiLM → identidad inicial.
- Drop-in para post-pool: si tomamos WavLM mean-pooled `[B, 1024]` + descriptor mean-pooled `[B, D]`, esto funciona tal cual con `ConditionedProjectionHead(input_dim=1024, cond_dim=D)`.

---

## 2. Adaptación necesaria por mecanismo para WavLM

Los wrappers viven en `src/voz_expresiva/wavlm_injection.py` (a crear en Fase 1 propiamente dicha).

### Concat (frame-level)

- **NO se puede reusar `SpeechEGGEncoderAug`** — su CNN propio reemplaza a WavLM, no lo complementa.
- **Implementación nueva**, ~25 líneas:
  ```python
  features = wavlm(waveform, return_sequence=True)  # [B, T=50Hz, 1024]
  descriptor = interpolate(descriptor, T)           # [B, T, D]
  features = cat([features, descriptor], -1)        # [B, T, 1024+D]
  features = Linear(1024+D, 1024)(features)         # [B, T, 1024]
  embedding = features.mean(dim=1)                  # [B, 1024]
  logits = Linear(1024, 5)(embedding)
  ```
- Aplicar near-identity init de `SpeechEGGEncoderAug` adaptado a 1024+D.

### FiLM (utterance-level)

- **Drop-in directo de `ConditionedProjectionHead`**:
  ```python
  wavlm_emb = wavlm(waveform)                   # [B, 1024], mean-pooled
  descriptor_pool = descriptor.mean(dim=1)      # [B, D], mean over time
  head = ConditionedProjectionHead(
      input_dim=1024, cond_dim=D,
      hidden_dim=512, output_dim=5  # 5 emociones
  )
  logits = head(wavlm_emb, cond=descriptor_pool)
  ```
- Único cambio: instanciar con `input_dim=1024, output_dim=5`. La clase ya lo soporta.
- Pierde resolución frame-level del descriptor, pero es coherente con la idea Phideus de FiLM como modulador global.

### xattn (frame-level)

- **NO se puede reusar `SpeechEGGEncoderXAttn`** — embed_dim=512 hardcoded.
- **Implementación nueva**, ~40 líneas, copiando la lógica:
  ```python
  features = wavlm(waveform, return_sequence=True)  # [B, T, 1024]
  desc = interpolate(descriptor, T)                  # [B, T, D]
  desc_q = Linear(D, 1024)(desc)                     # [B, T, 1024]
  xattn = MultiheadAttention(embed_dim=1024, num_heads=4)
  xattn_out, _ = xattn(query=desc_q, key=features, value=features)
  features = features + xattn_scale * LayerNorm(xattn_out)  # init 0.01
  embedding = features.mean(dim=1)
  logits = Linear(1024, 5)(embedding)
  ```
- Init near-zero idéntico a E2.

---

## 3. Costo real de adaptación

| Mecanismo | Reuso directo | Wrapper nuevo | Líneas |
|---|---|---|---|
| Concat | NO | Sí | ~25 |
| FiLM | **Sí** (`ConditionedProjectionHead` directo, utterance-level) | Wrapper trivial | ~10 |
| xattn | NO | Sí (réplica con embed_dim=1024) | ~40 |
| Boilerplate del módulo (init, base class) | — | Sí | ~30 |
| **Total** | | | **~105 líneas** |

NO es "importar tal cual". Pero tampoco es "reescribir desde cero". Es adaptar dimensiones y wrappear las tres formas en una clase con una bandera `mechanism=concat/film/xattn`.

---

## 4. Consecuencias para el plan de Fase 1

### Cambios al plan ya aprobado

1. **§ "Reuso explícito de código existente"** — corregir entrada de "Mecanismos de inyección" a:
   > NO drop-in. ConditionedProjectionHead (FiLM) se reusa tal cual a utterance-level. Concat y xattn requieren re-implementar la lógica para WavLM (embed_dim=1024 vs 512). Algoritmos heredados (near-identity init, near-zero residual scale) son aplicables.

2. **§ "Files to create"** — `wavlm_injection.py` ahora pesa ~105 líneas (no las "30 líneas de wrapper" que sugerí). Sigue siendo bajo.

3. **§ "Spike de Fase 1.0"** — cumplido. Resultados en este archivo.

### Decisiones nuevas sugeridas (no congeladas, propongo)

- **FiLM a utterance-level**, no frame-level. Es lo natural por la topología de `ConditionedProjectionHead` y matchea el paper E1 (FiLM/pca era post-pool). Si esto pierde demasiada información, Fase 1.2 puede explorar FiLM frame-level.
- **Concat y xattn quedan frame-level**, como en E2. Mantienen la idea de inyección granular.
- **Descriptor pooling para FiLM**: mean over time. Si Fase 1 falla, considerar max o weighted attention pooling.

### Cosas que NO se ven afectadas

- Pre-cache de WavLM frame-level: la estrategia sigue siendo correcta (cachear `[B, T, 1024]` una vez, usarlo en todos los runs).
- LOSO 10-fold, N-strict/N-adapt, agregación per-speaker, bootstrap: idéntico a 0B y a lo planeado.
- Configs núcleo (4): WavLM-only / WavLM+A concat / WavLM+A FiLM / WavLM+A xattn — sigue válido.

---

## 5. Próximo paso

El spike valida que Fase 1 es ejecutable con un esfuerzo modesto de adaptación (~105 líneas). El plan general aprobado se sostiene, pero la sección de reuso debe corregirse para no presentar como "import directo" lo que en realidad es "wrapper de adaptación".

Acción inmediata: actualizar el plan archivado (`~/.claude/plans/velvet-puzzling-rainbow.md`) con estos hallazgos antes de avanzar a implementación de Fase 1, o registrar la corrección en el README del frente.

Una vez registrado, el siguiente paso operativo es:
1. Crear `src/voz_expresiva/wavlm_injection.py` con la clase de wrappers (concat / FiLM via `ConditionedProjectionHead` / xattn).
2. Crear `src/voz_expresiva/esd_dataset.py` (torch Dataset wrapper).
3. `experiments/voz_expresiva/1_precache_wavlm.py` — pre-extract WavLM features.
4. `experiments/voz_expresiva/1_train.py` — LOSO outer loop.
5. `experiments/voz_expresiva/1_report.py` — análisis + REPORTE_1.md.

Tiempo estimado a partir del spike, sin contar el run de training:
- Implementación de los 5 archivos: 1-2 días CPU/coding.
- Pre-cache WavLM: ~1 h GPU.
- Runs de training: 3-5 días GPU según costo recontado (480 runs incluyendo N-adapt repeats).
