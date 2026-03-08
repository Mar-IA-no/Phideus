# Informe Completo de Análisis de Errores - Escalón 1

**Fecha**: 2026-02-04
**Autor**: Claude Code + Usuario
**Objetivo**: Identificar causas de accuracy baja (26.6%) y evaluar mejoras

---

## Resumen Ejecutivo

El análisis de errores reveló que el **problema principal es la resolución temporal del onset detection**, no el algoritmo de hashing ni el IDF. Se implementaron mejoras (Opciones A+B) que aumentaron el overlap cross-modal de 17% a 25%, pero la accuracy solo mejoró marginalmente de 26.6% a 27.0%.

**Conclusión**: El enfoque actual tiene un límite práctico de ~27% accuracy con N=20. Para mejoras significativas se requieren cambios más profundos (mejor onset detection o soft matching).

---

## 1. Metodología

### 1.1 Dataset de Análisis

- **Muestra**: 20 pares audio-MIDI de `muestra_replicacion/`
- **Origen**: MAESTRO v3.0.0 (años 2004-2018)
- **Duración**: Máximo 120s por pieza
- **Total queries**: 2,357 segmentos de 20s

### 1.2 Scripts Creados

```
experiments/un_audio_un_midi/
├── analyze_errors.py           # Análisis principal (Fases 1-2)
├── analyze_overlap_deep.py     # Análisis de componentes de tokens
├── ablation_chord_only.py      # Estudio de ablación por tipo de token
└── error_analysis/             # Reportes JSON
```

### 1.3 Métricas Evaluadas

| Métrica | Descripción |
|---------|-------------|
| Piece Accuracy | % queries donde top-1 = pieza correcta |
| Recall@K | % queries donde pieza correcta está en top-K |
| Cross-Modal Overlap | % hashes de MIDI que coinciden con audio |
| Document Frequency | En cuántas piezas aparece cada hash |

---

## 2. Resultados del Análisis

### 2.1 Distribución de Hashes (Fase 2)

**Configuración original (DT_BIN_SIZE=2)**:

| Métrica | Valor | Evaluación |
|---------|-------|------------|
| Hashes únicos | 10,538 | - |
| DF=1 (único a una pieza) | 72.7% | ✓ Alta discriminabilidad |
| Stopwords (DF > 30%) | 1.5% | ✓ IDF funciona |
| Overlap audio↔MIDI | 13-29% | ✗ Muy bajo |

**Después de mejoras (DT_BIN_SIZE=10)**:

| Métrica | Valor | Cambio |
|---------|-------|--------|
| Hashes únicos | 7,049 | -33% |
| DF=1 | 63.0% | -9.7pp |
| Stopwords | 4.5% | +3pp |
| Overlap audio↔MIDI | 19-37% | **+8pp promedio** |

### 2.2 Overlap por Tipo de Token

| Tipo | Descripción | Overlap (antes) | Overlap (después) |
|------|-------------|-----------------|-------------------|
| Chord (tipo 1) | Notas simultáneas ±30ms | **62-85%** | **~72%** |
| Sequential (tipo 2) | Notas consecutivas | 4-15% | ~15% |
| Constellation (tipo 3) | Pares lejanos (ΔT>1s) | 1-4% | ~5% |

### 2.3 Causa Raíz: Delta-Time Mismatch

**Distribución de ΔT en tokens constellation (tipo 3)**:

```
MIDI (ground truth perfecto):
  DT=1: 34 tokens
  DT=2: 10 tokens
  DT=3: 1 token

Audio (onset detection):
  DT=1: 0 tokens  ← No detecta eventos cercanos
  DT=2: 0 tokens
  DT=5-20: mayoría de tokens
```

**Interpretación**: El onset detector de audio (librosa.onset.onset_detect) tiene resolución limitada (~50-100ms) y pierde notas rápidas, generando intervalos temporales diferentes a los del MIDI.

### 2.4 Alineación Temporal

| Pieza | Eventos Audio | Eventos MIDI | Dentro 50ms | Dentro 100ms |
|-------|---------------|--------------|-------------|--------------|
| 1 | 940 | 855 | 66.4% | 83.9% |
| 2 | 1050 | 1124 | 73.6% | 84.4% |
| 3 | 2301 | 917 | 43.5% | 66.8% |

**Nota**: Pieza 3 detecta 2.5x más eventos en audio → muchos falsos positivos.

### 2.5 Pitch Class Distribution

Las distribuciones de pitch class coinciden bien entre audio y MIDI:

```
PC    Audio   MIDI   Match
C     12.0%   11.7%    ✓
D#    20.7%   20.1%    ✓
F     18.4%   16.1%    ✓
...
```

**Conclusión**: El problema NO es estimación de pitch, sino timing.

### 2.6 Confusiones Sistemáticas

Top confusiones (pieza verdadera → pieza predicha):

| True | Pred | Count | Interpretación |
|------|------|-------|----------------|
| 9 | 2 | 54 | Pieza 2 tiene hashes genéricos |
| 6 | 2 | 33 | " |
| 16 | 17 | 33 | Piezas similares |
| 4 | 17 | 30 | " |

Piezas "confusoras" que roban queries: **2, 17**

### 2.7 Correlación Overlap-Accuracy

| Pieza | Overlap | Accuracy | Correlación |
|-------|---------|----------|-------------|
| 3 | 29.1% → 36.8% | 69.5% | ✓ Alto overlap → alta accuracy |
| 6 | 13.4% → 18.9% | 8.6% | ✓ Bajo overlap → baja accuracy |
| **Promedio** | **17% → 25%** | **26.6% → 27.0%** | r ≈ 0.7 |

---

## 3. Mejoras Implementadas

### 3.1 Opción A: Aumentar Bins de Delta-Time

```python
# Antes
DT_BIN_SIZE = 2   # 20ms bins

# Después
DT_BIN_SIZE = 10  # 100ms bins - más tolerancia temporal
```

**Resultado**: Overlap +8pp, accuracy +0.4pp

### 3.2 Opción B: Boost Chord Tokens

```python
# Antes
CHORD_ONSET_TOL = 3   # 30ms
weight = sqrt(amp1 * amp2)

# Después
CHORD_ONSET_TOL = 5   # 50ms - captura más chords
CHORD_WEIGHT_BOOST = 2.0
weight = sqrt(amp1 * amp2) * CHORD_WEIGHT_BOOST
```

**Resultado**: Marginal

### 3.3 Resultados Post-Mejoras

| Métrica | Antes | Después | Cambio |
|---------|-------|---------|--------|
| Piece Accuracy | 26.6% | 27.0% | +0.4pp |
| Recall@3 | 47.3% | 47.3% | 0 |
| Recall@5 | 64.2% | 61.1% | -3.1pp |
| vs Random | 5.3x | 5.4x | +0.1x |

---

## 4. Estudio de Ablación por Tipo de Token

### 4.1 Configuraciones Probadas

| Config | Chord | Seq | Const | Hashes | Accuracy | Recall@5 |
|--------|-------|-----|-------|--------|----------|----------|
| All | ✓ | ✓ | ✓ | 7,049 | **27.0%** | **61.1%** |
| Chord+Seq | ✓ | ✓ | - | 3,754 | **27.1%** | 60.6% |
| Sin chord | - | ✓ | ✓ | 6,651 | 24.9% | 57.5% |
| Chord+Const | ✓ | - | ✓ | 3,693 | 14.6% | 47.1% |
| **Chord ONLY** | ✓ | - | - | **398** | 13.6% | 45.5% |

### 4.2 Conclusiones del Ablation

1. **Chord solo no funciona** porque genera muy pocos hashes únicos (398)
2. **Sequential tokens son importantes** - aportan volumen de hashes
3. **Constellation tokens son prescindibles** - eliminarlos no afecta
4. **La mejor config es Chord+Seq** (27.1%) pero es igual al baseline

---

## 5. Diagnóstico Final

```
┌─────────────────────────────────────────────────────────────────┐
│ PROBLEMA RAÍZ: El onset detector del audio tiene resolución    │
│ temporal insuficiente para generar tokens compatibles con MIDI │
│ en tipos sequential y constellation.                           │
│                                                                 │
│ LÍMITE ACTUAL: ~27% accuracy con enfoque actual                │
└─────────────────────────────────────────────────────────────────┘
```

### Por qué chord funciona (72% overlap):
- Solo requiere notas en ventana de ±50ms
- Tolerante a pequeñas desalineaciones
- No depende de ΔT exacto

### Por qué sequential/constellation no funcionan (~5-15% overlap):
- Requieren matching exacto de ΔT
- MIDI: ΔT = 1-3 frames
- Audio: ΔT = 5-20 frames
- Los hashes resultantes son diferentes

---

## 6. Opciones No Implementadas

### 6.1 Opción C: Mejorar Onset Detection

```python
# Cambiar de HFC a superflux
onsets = librosa.onset.onset_detect(
    y=audio, sr=sr,
    onset_detect='superflux',  # Más sensible
    threshold=0.5,             # Más bajo
    pre_max=3, post_max=3
)
```

**Estimación**: +10-15% accuracy
**Esfuerzo**: Alto (requiere tuning extensivo)

### 6.2 Opción D: LSH / Soft Matching

```python
# En lugar de hash exacto, usar LSH
from datasketch import MinHash, MinHashLSH

lsh = MinHashLSH(threshold=0.8, num_perm=128)
```

**Estimación**: +15-20% accuracy
**Esfuerzo**: Muy alto (cambio arquitectural)

### 6.3 Opción E: Escalar con 27%

Probar con N=100+ piezas para ver tendencia.

**Proyección**:
- N=100: ~15% accuracy (15x random)
- N=1000: ~5% accuracy (50x random)

---

## 7. Archivos Modificados

### 7.1 Extractor Modificado

**Archivo**: `src/extractors/event_based_extractor.py`

```python
# Cambios aplicados:
DT_BIN_SIZE = 10        # (era 2)
CHORD_ONSET_TOL = 5     # (era 3)
CHORD_WEIGHT_BOOST = 2.0  # (nuevo)

# Nueva firma de extract_all_tokens:
def extract_all_tokens(
    events: List[MusicEvent],
    use_chord: bool = True,
    use_sequential: bool = True,
    use_constellation: bool = True,
) -> List[EventToken]:
```

### 7.2 Scripts de Análisis

| Script | Propósito | Output |
|--------|-----------|--------|
| `analyze_errors.py` | Análisis principal | `error_analysis/error_analysis.json` |
| `analyze_overlap_deep.py` | Componentes de tokens | stdout |
| `ablation_chord_only.py` | Ablation study | stdout |

---

## 8. Conclusiones y Recomendaciones

### 8.1 Estado Final

| Métrica | Valor | vs Objetivo |
|---------|-------|-------------|
| Piece Accuracy | 27.0% | < 40% ✗ |
| Recall@5 | 61.1% | < 80% ✗ |
| vs Random | 5.4x | > 5x ✓ |

### 8.2 Lecciones Aprendidas

1. **El overlap predice accuracy** (r ≈ 0.7)
2. **Mejorar overlap no garantiza mejorar accuracy** (+8pp overlap → +0.4pp acc)
3. **El cuello de botella es onset detection**, no el algoritmo de hashing
4. **Los tokens chord son robustos** pero insuficientes solos
5. **Las mejoras incrementales tienen rendimientos decrecientes**

### 8.3 Recomendación

Dado que:
- Se alcanzó 5.4x random (señal clara sobre baseline)
- Las mejoras incrementales no escalan
- Se requieren cambios mayores para progreso significativo

**Opciones viables**:
1. **Escalar a N=100** para confirmar tendencia (bajo esfuerzo)
2. **Documentar y cerrar** esta línea de investigación
3. **Pivotear** a enfoque diferente (spectrograms, transformers, etc.)

---

## Anexo: Comandos de Reproducción

```bash
# Activar entorno
cd <repo-root>
source venv/bin/activate

# Análisis de errores
python experiments/un_audio_un_midi/analyze_errors.py \
    --input-dir experiments/un_audio_un_midi/muestra_replicacion \
    --route A

# Análisis profundo de overlap
python experiments/un_audio_un_midi/analyze_overlap_deep.py

# Ablation study
python experiments/un_audio_un_midi/ablation_chord_only.py

# Test de retrieval
python experiments/un_audio_un_midi/test_retrieval_routes.py \
    --input-dir experiments/un_audio_un_midi/muestra_replicacion
```
