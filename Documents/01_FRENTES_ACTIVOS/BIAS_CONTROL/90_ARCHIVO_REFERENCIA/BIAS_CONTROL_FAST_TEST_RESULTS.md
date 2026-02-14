# BIAS_CONTROL Fast Test Results

**Fecha**: 2026-02-04
**Duración**: ~28 minutos
**Estado**: FAILED_GATE_2 (esperado con configuración mínima)

> [!NOTE]
> Addendum de vigencia (2026-02-14): este documento corresponde al fast test histórico.
> El estado operativo actual del frente está en Gate 4.3 (corrida causal 6 brazos en ejecución).
> Referencias vigentes: roadmap, estado troncal e informe Gate 4.3.

---

## Configuración del Fast Test

```bash
python experiments/bias_control/run_all_gates.py \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/bias_control \
    --epochs-gate2 3 --epochs-gate3 2 --epochs-gate4 2 \
    --batch-size 16 --segment-len 4.0 --hop 1.0 \
    --max-batches-per-epoch 50 --num-workers 8 --device cuda
```

| Parámetro | Valor | Nota |
|-----------|-------|------|
| Epochs Gate 2 | 3 | Mínimo para test |
| Batches/Epoch | 50 | De 5994 disponibles (0.8%) |
| Batch Size | 16 | Limitado por VRAM |
| Segment Length | 4.0s | Reducido de 8.0s |
| Hop | 1.0s | - |
| Device | CUDA (RTX 3090) | ~21GB VRAM usado |

---

## Resultados por Gate

### Gate 0: Data Integrity ✅ GO

| Métrica | Resultado | Umbral |
|---------|-----------|--------|
| Alignment Rate | 100% | ≥90% |
| Total Segments | 127,092 | ≥10,000 |
| Pieces | 1,276 | - |
| Shuffling Working | ✓ | - |

**Splits**:
- Train: 95,906 segments (962 pieces)
- Validation: 13,532 segments (137 pieces)
- Test: 17,654 segments (177 pieces)

### Gate 1: Intra-Modal Baselines ⚠️ NO-GO

| Métrica | Audio→Audio | MIDI→MIDI | Umbral |
|---------|-------------|-----------|--------|
| Recall@1 | 2.6% | 0% | - |
| Recall@5 | 95.4% | - | - |
| Recall@10 | 98.4% | - | ≥50% |
| MRR | 0.466 | - | - |

**Nota**: Gate 1 usa embeddings pre-entrenados (MERTLite para audio). El bajo rendimiento de MIDI indica que el encoder MIDI necesita más entrenamiento.

### Gate 2: Cross-Modal Foundation ❌ NO-GO

#### Progresión del Entrenamiento

| Época | Loss | A→M R@10 | M→A R@10 | Gap |
|-------|------|----------|----------|-----|
| 1 | 28.31 | 0.10% | 0.10% | 0.007 |
| 2 | 16.80 | 0.10% | 0.30% | 0.020 |
| 3 | 15.96 | 0.20% | 0.20% | 0.026 |

#### Evaluación Final

| Métrica | Resultado | Umbral | Status |
|---------|-----------|--------|--------|
| Audio→MIDI Recall@10 | 0.17% | 20% | ❌ FAIL |
| MIDI→Audio Recall@10 | 0.19% | 20% | ❌ FAIL |
| vs Random (A→M) | 2.3x | 5.0x | ❌ FAIL |
| vs Random (M→A) | 2.6x | 5.0x | ❌ FAIL |
| Gap | 0.026 | - | - |

### Gates 3-4: Omitidos

Pipeline detenido después de Gate 2 NO-GO.

---

## Observaciones

### Positivas

1. **El modelo está aprendiendo**: Loss bajó de 28.31 → 15.96 (43% reducción)
2. **Gap mejorando**: 0.007 → 0.026 (3.7x mejora)
3. **Pipeline funciona**: Toda la infraestructura ejecutó correctamente
4. **Dataset válido**: 127K segmentos de alta calidad

### Limitaciones del Fast Test

1. **Datos insuficientes**: Solo 50 batches × 3 épocas = 150 batches totales
   - Dataset completo: 5994 batches × 100 épocas = 599,400 batches
   - Fast test vio **0.025%** del entrenamiento completo

2. **Épocas insuficientes**: 3 épocas no permite convergencia
   - VICReg típicamente necesita 50-100 épocas

3. **Sin warmup efectivo**: 50 batches no permite warmup adecuado del learning rate

---

## Conclusión

El fast test **NO es diagnóstico** para H3. Los resultados de NO-GO eran esperados dado que:
- El modelo apenas comenzó a aprender
- La tendencia es positiva (loss ↓, gap ↑)
- Se necesita entrenamiento completo para validar H3

**Recomendación**: Ejecutar Medium Train (30 épocas, 200+ batches/época) para evaluación válida.

---

## Archivos Generados

```
data/bias_control/
├── pipeline_results.json          # Resultados completos
├── pipeline.log                   # Log de ejecución
├── segments/
│   └── segments_metadata.json     # Metadata de segmentos
├── evaluations/
│   └── gate1/                     # Resultados Gate 1
└── training_outputs/
    └── gate2/
        ├── best_model.pt          # Checkpoint del mejor modelo
        ├── final_model.pt         # Modelo final
        └── training_history.json  # Historia de entrenamiento
```

---

## Próximos Pasos

1. **Auditoría del fast test**: Verificar correctitud de la implementación
2. **Medium Train**: 30 épocas, 200 batches/época (~6-8 horas)
3. **Evaluación de H3**: Si Medium Train muestra recall >10%, considerar Full Train
