# Plan de Validación H3: Escalón 1 MAESTRO

**Fecha**: 2026-02-04
**Estado**: 🟡 PENDIENTE - Resultados preliminares prometedores

---

## Contexto

El experimento piloto con 10 pares mostró resultados prometedores:
- Route A: 71.4% accuracy (7 queries)
- Route B: 80.0% accuracy (10 queries)

**Esto NO valida H3.** Para validar necesitamos ejecución rigurosa.

---

## Fases de Validación

### FASE A: Auditoría del Experimento (Inmediato)

**Objetivo**: Verificar que el experimento piloto se hizo correctamente.

#### A.1 Verificar Extractores
- [ ] Revisar `event_based_extractor.py` - ¿genera tokens correctos?
- [ ] Revisar `improved_tf_extractor.py` - ¿onset anchoring funciona?
- [ ] Test unitarios con datos sintéticos conocidos

#### A.2 Verificar Alineación
- [ ] Las 10 piezas tienen alineación correcta audio-MIDI?
- [ ] El offset correction funciona?
- [ ] Visualizar algunos ejemplos

#### A.3 Verificar Protocolo de Evaluación
- [ ] `test_retrieval_routes.py` genera queries correctamente?
- [ ] El Shazam voting está implementado correctamente?
- [ ] Los resultados son reproducibles (seed fijo)?

**Entregable**: Informe de auditoría con checklist completo

---

### FASE B: Replicación con Muestra Independiente

**Objetivo**: Verificar que los resultados no son artefacto de las 10 piezas elegidas.

#### B.1 Selección de Nueva Muestra
- [ ] Elegir 20 piezas **nuevas** del dataset MAESTRO
- [ ] Diversificar: diferentes compositores, duraciones, épocas
- [ ] Copiar a directorio separado

#### B.2 Ejecutar Experimento
```bash
python experiments/un_audio_un_midi/test_retrieval_routes.py \
    --input-dir experiments/un_audio_un_midi/muestra_replicacion \
    --output-dir experiments/un_audio_un_midi/muestra_replicacion/results
```

#### B.3 Comparar Resultados
- [ ] Route A accuracy: ¿similar al piloto?
- [ ] Route B accuracy: ¿similar al piloto?
- [ ] Si hay diferencia >15%, investigar causas

**Criterio GO**: Ambas rutas mantienen >60% accuracy en muestra nueva

---

### FASE C: Validación a Escala (100+ piezas)

**Objetivo**: Validación estadísticamente significativa.

#### C.1 Procesamiento de Dataset
```bash
# Usar muestra grande (100-200 piezas)
python experiments/maestro/scale_test.py \
    --input-dir data/maestro_v3/maestro-v3.0.0 \
    --output-dir data/evaluations/scale_100 \
    --n-pieces 100 \
    --route B \
    --workers 14
```

#### C.2 Protocolo de Evaluación Completo
- **NEG_RANDOM**: Segmentos de otras piezas (fácil)
- **NEG_SAME_PIECE**: Misma pieza, diferente tiempo (medio)
- **NEG_SAME_COMPOSER**: Mismo compositor, otra pieza (difícil)

#### C.3 Métricas con Intervalos de Confianza
- Recall@{1, 5, 10, 20}
- MRR (Mean Reciprocal Rank)
- Bootstrap CI 95%
- Gap aligned vs shuffled

**Criterio GO**:
- Piece Accuracy > 50% (CI 95% no cruza 50%)
- Gap aligned-shuffled > 10%
- NEG_SAME_COMPOSER Recall@10 > 2x random

---

### FASE D: Pipeline Completo Escalón 1

**Objetivo**: Ejecutar el plan original con los nuevos extractores.

#### D.1 Gate 0: Setup Harness
- [ ] Implementar controles negativos formales
- [ ] Oracle test (MIDI vs MIDI sintetizado)

#### D.2 Gate 1: Ingesta Completa
- [ ] Procesar todo MAESTRO (1276 piezas)
- [ ] Generar segmentos (4s window, 2s hop)
- [ ] Verificar alineación con correlación energía-densidad

#### D.3 Gate 2: Baselines Sin DL
- [ ] Chroma vs Pitch-Class
- [ ] CCA/Ridge linear

#### D.4 Gate 3: Modelo Cross-Modal
- [ ] VICReg training
- [ ] Barlow Twins training
- [ ] Detectar colapso

#### D.5 Gate 4: Ratio Tokens con Nuevos Extractores
- [ ] Extraer tokens con Route A o B (mejor de Fase C)
- [ ] Training ConstellationVAE
- [ ] Comparar con baselines

#### D.6 Gate 5: MoCo con Negativos Duros
- [ ] MoCo queue (4096-8192)
- [ ] Hard-mined negatives (mismo compositor)

**Criterio GO Final**:
| Gate | Métrica | Umbral |
|------|---------|--------|
| 0 | Oracle accuracy | > 90% |
| 1 | Correlación energía-densidad | > 0.7 |
| 2 | Baselines Piece Top-1 | > 10x random |
| 3 | No colapso + Top-1 > baselines | - |
| 4 | Ratio tokens > random | Piece Acc > 50% |
| 5 | Mejora NEG_SAME_COMPOSER | > Gate 4 |

---

## Comandos de Ejecución

### Setup
```bash
cd /mnt/m2-1TB/Phideus
source venv/bin/activate
git checkout feature/extractor-v22
```

### Fase A: Auditoría
```bash
# Tests de extractores
python -m pytest src/extractors/test_extractors.py -v

# Reproducibilidad
python experiments/un_audio_un_midi/test_retrieval_routes.py \
    --input-dir experiments/un_audio_un_midi/Varios_pares \
    --output-dir experiments/un_audio_un_midi/Varios_pares/audit_run2
```

### Fase B: Replicación
```bash
# Copiar 20 piezas nuevas
mkdir -p experiments/un_audio_un_midi/muestra_replicacion
# [seleccionar manualmente 20 pares de data/maestro_v3/]

python experiments/un_audio_un_midi/test_retrieval_routes.py \
    --input-dir experiments/un_audio_un_midi/muestra_replicacion
```

### Fase C: Escala
```bash
# Crear script scale_test.py para N=100+
python experiments/maestro/scale_test.py --n-pieces 100 --route B
```

### Fase D: Pipeline Completo
```bash
python experiments/maestro/run_maestro_experiment.py \
    --mode full \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/training_outputs/escalon1_full \
    --epochs 100 --batch-size 64 --num-workers 8
```

---

## Estimación de Tiempo

| Fase | Tiempo Estimado |
|------|-----------------|
| A: Auditoría | 1-2 días |
| B: Replicación | 1 día |
| C: Escala (100 piezas) | 2-3 días |
| D: Pipeline completo | 5-7 días |
| **TOTAL** | **9-13 días** |

---

## Criterios GO/NO-GO Final para H3

| Criterio | Umbral GO | Umbral NO-GO |
|----------|-----------|--------------|
| Piece Accuracy (N=100+) | > 50% | < 30% |
| Gap aligned-shuffled | > 10% | < 5% |
| NEG_SAME_COMPOSER Recall@10 | > 2x random | < 1.5x random |
| Replicación consistente | Δ < 15% | Δ > 25% |

**Si todos los criterios GO se cumplen → H3 VALIDADA**
**Si cualquier criterio NO-GO → H3 NO VALIDADA**
**Zona gris → Más investigación necesaria**

---

## Referencias

- Experimento piloto: `Documents/01_FRENTES_ACTIVOS/ESCALON_1/RESULTADOS_NUEVOS_ENFOQUES.md`
- Plan original: `Documents/01_FRENTES_ACTIVOS/ESCALON_1/Plan_implementacion.md`
- Recomendaciones GPT: `Documents/01_FRENTES_ACTIVOS/ESCALON_1/Extractor_nuevos_enfoques_GPT5.2Think.md`
