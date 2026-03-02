# Gate 6 — AMT with Descriptor Conditioning

**Fecha inicio**: 2026-03-02
**Estado**: Implementación completa, pendiente ejecución

## Motivación

Gate 5B demostró:
- Descriptores reorganizan geometría de embeddings (Test 02: +9.4pp causal)
- NO enriquecen decodificabilidad de features (Test 13G-B: F1~10% todos los arms)
- Duplican CKA cross-encoder (Test 06: d4-a4r +82% vs D0)

**Pregunta abierta**: ¿La ventaja es puramente geométrica (solo retrieval) o se traduce a tareas musicales concretas?

## Diseño experimental

### Exp 0: Verificación baseline
- Transkun v2 pretrained transcribe 100 segmentos MAESTRO (50×4s + 50×16s)
- Establece ceiling de referencia
- Ejecución local, ~2 min

### Exp A: Descriptor-Conditioned Transkun
- 5 configs con controles param-matched
- Fine-tune 50K iter, base congelada
- Métrica primaria: Δ F1(A4-event vs finetune-noA4)

### Exp B: Condiciones degradadas
- 3 degradaciones × 3 niveles × 3 configs = 27 runs
- A4 SIEMPRE del audio degradado (no oracle)
- Hipótesis: Δ F1(A4) crece con severidad

### Exp C: AMT decoder sobre VICReg features
- Decoder 38M (16× Test 13G-B) con focal loss + onset weighting
- 4 arms: D0, d4a4, a4r, d4-a4r (checkpoints Gate 5B congelados)
- Régimen Phideus (24kHz, 4s, 188 frames)

## Hallazgo arquitectónico: Transkun v2

Transkun NO usa "event tracks" como tokens concatenados. El Backbone:
1. CNN frontend: mel spec [B, D, T, F] → downsampled [B, D', T/8, F/4]
2. Concatena posicional embeddings para 90 "output indices" (88 notas + 2 pedales) en la dimensión de frecuencia
3. 6 axial transformer layers (atención separada en F y T)
4. Upsample temporal → per-note features → Semi-CRF scoring

Inyección de A4: como tracks adicionales en la dimensión de frecuencia (event track extension) o via FiLM después de cada BasicBlock.

## Resultados

*Pendiente ejecución*

## Archivos

Ver `experiments/bias_control/gate6/README.md` para la lista completa.
