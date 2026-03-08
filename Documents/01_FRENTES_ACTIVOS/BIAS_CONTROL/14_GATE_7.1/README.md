# Gate 7.1 -- MERT-330M Frozen Cross-Modal Probe

**Estado**: Phase 7.1a implementada y corriendo LOCAL (2026-03-06).

## Proposito

Gate 7.1 responde: si el backbone de audio ya es fuerte (MERT-330M), la ventaja de A4 sobrevive en retrieval cross-modal o desaparece?

La lectura correcta es prudente: sigue siendo un piloto decisional, no un aislamiento causal puro. Cambian simultaneamente backbone, co-adaptacion y regimen de pretraining.

## Contexto

- Gate 5B: A4 mejora retrieval causalmente (+9.4pp, Test 02) via geometria (+82% CKA)
- Gate 7 Exp 7.0: MERT-330M R^2=0.850 para envolvente espectral A4 (vs MERTLite=0.734)

## Diseno: Dos Fases

### Phase 7.1a -- D0 Pilot (infraestructura + baseline)

- MERT-330M frozen + D0 (sin descriptor), 1 seed (42), 30 epochs
- Valida VICReg cross-modal con frozen encoder
- Benchmark de throughput para estimar costo UNC
- Obtiene S(D0_mert330m) como baseline fuerte

### Phase 7.1b -- a4r-MERT (solo si 7.1a GO)

- Variante nueva: K/V de MERT-330M hidden states + lightweight transformer
- Compara delta_A4 = S(a4r_mert) - S(D0_mert) con Gate 5B's +5.5pp
- No es swap trivial de flag: requiere nueva clase de modelo

## Outcome Framework

| Outcome | Signal | Reading |
|---------|--------|---------|
| A | D0_strong ~ D0_lite (75%) | Frozen encoder no escala VICReg |
| B | D0_strong >> 75% AND delta_A4 -> 0 | A4 compensaba encoder debil |
| C | D0_strong >> 75% AND delta_A4 > 0 | Tesis geometrica robusta |
| Inconclusive | D0_strong < D0_lite | Frozen dynamics rompen VICReg |

## Guardrails

- No leer Gate 7 como prueba de que MERT-330M "ya tiene A4" en sentido fuerte; el probe cerro solo sobre la envolvente espectral segment-level.
- No comparar delta_A4 de 7.1b como si fuera identico al +5.5pp multi-seed de Gate 5B.
- 1 seed alcanza para pilot y go/no-go, no para claim estadistico fuerte.

## Archivos

```
experiments/bias_control/gate71/
  __init__.py
  train_gate71.py           # Training script (D0, futuro a4r-mert)

slurm/gate71_d0.sh          # SLURM para Phase 7.1a

src/bias_control/encoders/mert_encoder.py  # Fixed: train() leak, return_sequence
experiments/bias_control/gate5b/checkpoint_loader.py  # Extended: audio_encoder='mert'
```

## Fixes Aplicados

1. **MERTEncoder.train() override**: Mantiene `_model.eval()` cuando frozen
2. **MERTEncoder.forward(return_sequence=True)**: Devuelve [B, T, 1024] pre-pool
3. **Force _load_model()**: Workaround para lazy loading antes de anti-ghost checks
4. **checkpoint_loader.py**: Rutea `audio_encoder='mert'` a Gate71Model

## Anti-Ghost Checks

- Trainable ~14.5M (MIDI + projections), frozen ~315M (MERT)
- Weight snapshot pre/post epoch 1
- model.train() no filtra a MERT eval mode
- Optimizer excluye todos los params del audio encoder

## Throughput (LOCAL, RTX 3090)

- 235 batches/min, 3.0 GB VRAM
- Est. 30 epochs: ~2.1h training + ~1h eval = ~3.5h total

## Resultados Phase 7.1a (COMPLETO, 2026-03-06)

### Structured eval (pool=256 piezas, 500 queries)

| Epoch | A2M R@10 | M2A R@10 | S | hard_neg |
|-------|----------|----------|---|----------|
| 5 | 75.0% | 71.2% | 71.2% | 92.8% |
| 10 | 80.8% | 75.0% | **75.0%** | 94.0% |
| 15 | 81.0% | 74.2% | 74.2% | 94.2% |
| 20 | 78.2% | 70.6% | 70.6% | 93.4% |
| 25 | 79.6% | 74.8% | 74.8% | 94.2% |
| 28 | 79.2% | 72.4% | 72.4% | 93.0% |
| 29 | 79.2% | 71.6% | 71.6% | 93.2% |
| 30 | 81.0% | 74.6% | 74.6% | 93.2% |

Best: epoch 10, S=75.0%. Comparar: D0_lite = 75.2% +/-2.3pp (5 seeds).

### Lectura

Gate 7.1a muestra que fortalecer el audio backbone en modo frozen no mejora el retrieval cross-modal. El limite operativo parece estar en la co-adaptacion y/o en el lado MIDI-projection, no en la accesibilidad lineal de informacion espectral en el encoder de audio.

Precision: esto NO cierra "el cuello no es la capacidad del encoder". Solo muestra que un encoder mas fuerte Y congelado no destraba el sistema. Quedan abiertas: (a) co-adaptacion necesaria entre encoders, (b) cuello en MIDI encoder, (c) cuello en projection heads, (d) cuello en regimen/objetivo.

### Consecuencias

- **Phase 7.1b**: baja de prioridad. Baseline plano hace el test menos informativo.
- **Gate 5A C1** (conditioned projections): sube de prioridad. Dos pistas independientes (Test 11 Pre-Proj + Gate 7.1a) convergen contra el cuello MIDI/projection.

## Documento de trabajo

- `Plan_implementacion.md`: plan v2 completo con fases, riesgos, checks y criterios de lectura.
