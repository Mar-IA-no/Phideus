# AUDITORÍA COMPLETA: Implementación MAESTRO

**Fecha**: 2026-02-04
**Estado**: ✅ Auditoría completada, **correcciones aplicadas**

---

## Resumen Ejecutivo

| Categoría | Estado | Issues Críticos |
|-----------|--------|-----------------|
| **Imports y Dependencias** | ⚠️ | Deps externas no instaladas (pretty_midi, mido) |
| **Flujo de Datos** | ✅ **CORREGIDO** | max_tokens ahora se lee del NPZ |
| **GO/NO-GO Criteria** | ✅ | Todos implementados correctamente |
| **Alineación con Plan** | ✅ | 9/10 |

---

## 1. ISSUES CRÍTICOS (Deben corregirse antes de ejecutar)

### 1.1 ❌ MISMATCH DE max_tokens (CRÍTICO)

**Problema**: El analizador genera 64 tokens/frame, pero los modelos esperan 48.

| Componente | Valor | Ubicación |
|------------|-------|-----------|
| `analizador_maestro.py` | **64** (16×4) | Lines 57-58: DEFAULT_MAX_ANCHORS=16, DEFAULT_MAX_TARGETS_PER_ANCHOR=4 |
| `gate4_ratio_tokens.py` | **48** | Line 538: `--max-tokens` default |
| `gate5_moco.py` | **48** | Line 774: `--max-tokens` default |
| `constellation_vae.py` | **48** | Line 535: `max_tokens=48` default |
| `jepa_lite.py` | **48** | Line 123: `max_tokens=48` default |

**Impacto**: Shape mismatch causará error en forward pass o truncará datos silenciosamente.

**Corrección requerida en gate4_ratio_tokens.py** (~línea 650):

```python
# ANTES (incorrecto):
if args.model == 'constellation':
    model = ConstellationVAE(
        ...
        max_tokens=args.max_tokens,  # Usa default 48
        ...
    )

# DESPUÉS (correcto):
# Obtener max_tokens del dataset después de crear dataloaders
train_loader, val_loader, test_loader = create_maestro_dataloaders(...)
max_tokens = train_loader.dataset.max_tokens  # Leer del NPZ (será 64)

if args.model == 'constellation':
    model = ConstellationVAE(
        ...
        max_tokens=max_tokens,  # Usar valor del NPZ
        ...
    )
```

**Corrección requerida en gate5_moco.py** (~línea 825):

```python
# ANTES (incorrecto):
model = MoCoCrossModal(
    ...
    max_tokens=args.max_tokens,  # Usa default 48
    ...
)

# DESPUÉS (correcto):
train_loader, val_loader, test_loader = create_maestro_dataloaders(...)
max_tokens = train_loader.dataset.max_tokens  # Leer del NPZ

model = MoCoCrossModal(
    ...
    max_tokens=max_tokens,  # Usar valor del NPZ
    ...
)
```

### 1.2 ⚠️ Dependencias Externas No Instaladas

```bash
# Requerido instalar antes de ejecutar:
pip install torch librosa pretty_midi scikit-learn mido
```

### 1.3 ⚠️ Issue con src/datasets/__init__.py (MENOR)

El archivo `/mnt/m2-1TB/Phideus/src/datasets/__init__.py` contiene:
```python
from .temporal_dataset_5 import TemporalDataset5, load_dataset_as_static
```

Esto requiere torch al importar el paquete. Los gates ya importan directamente de los módulos, así que no es bloqueante, pero puede causar confusión.

**Corrección opcional**: Comentar o usar lazy import.

---

## 2. VERIFICACIONES POSITIVAS

### 2.1 ✅ GO/NO-GO Criteria Correctamente Implementados

| Gate | Criterio | Implementado | Ubicación |
|------|----------|--------------|-----------|
| **0** | Oracle > 90% | ✅ | gate0_harness.py:287-298, 369, 415 |
| **0** | Random ~ 1/N | ✅ | gate0_harness.py:113-116, 403 |
| **0** | NEG_RANDOM | ✅ | gate0_harness.py:183-188 |
| **0** | NEG_WITHIN_PIECE | ✅ | gate0_harness.py:191-213 |
| **0** | NEG_SAME_COMPOSER | ✅ | gate0_harness.py:216-233 |
| **0** | NEG_CROSS_PIECE | ✅ | gate0_harness.py:236-256 |
| **0** | Bootstrap CI | ✅ | gate0_harness.py:121-168 |
| **1** | Energy-density corr > 0.7 | ✅ | gate1_ingest.py:385-401, 504-515 |
| **2** | Baseline > 10x random | ✅ | gate2_baselines.py:641-652 |
| **3** | No collapse (var > 0.1) | ✅ | gate3_cross_modal.py:335-339, 589 |
| **3** | Better than baselines | ✅ | gate3_cross_modal.py:590, 594 |
| **4** | Baseline matching > random | ✅ | gate4_ratio_tokens.py:120-202 |
| **4** | Model comparable to dense | ✅ | gate4_ratio_tokens.py:352-500+ |
| **5** | Mejora NEG-SAME-COMPOSER | ✅ | gate5_moco.py:331-338, 404-448 |

### 2.2 ✅ Funciones de Pérdida Correctas

| Función | Especificación del Plan | Implementación |
|---------|-------------------------|----------------|
| **VICReg** | λ_inv·invariance + λ_var·variance + λ_cov·covariance | ✅ Exacto en src/RNA/vicreg.py |
| **Barlow** | (diag(C)-1)² + λ·sum(off_diag(C)²) | ✅ Exacto en src/RNA/barlow_twins.py |

Detalles VICReg:
- `invariance_loss()` = `F.mse_loss(z_a, z_b)` ✅
- `variance_loss()` = hinge on `std(z)` ✅
- `covariance_loss()` = off-diagonal covariance ✅
- Defaults: λ_inv=25.0, λ_var=25.0, λ_cov=1.0 ✅

Detalles Barlow:
- Cross-correlation: `C = (z_a.T @ z_b) / N` ✅
- On-diagonal: `(C.diagonal() - 1)²` ✅
- Off-diagonal: `sum(C[i,j]²)` para i≠j ✅
- Auto λ = 1/proj_dim ✅

### 2.3 ✅ Formato de Token Correcto

Especificado: `[log_ratio, delta_t, weight, anchor_band, target_band]`

Verificado en:
- `analizador_maestro.py:210-215`: Construcción correcta
- `midi_utils.py:226+`: `midi_to_constellation_tokens()` existe y usa mismo formato
- `constellation_vae.py`: Docstring confirma formato

### 2.4 ✅ Parámetros de Ventana Correctos

| Parámetro | Plan | Implementación |
|-----------|------|----------------|
| Window | 2.0-4.0s | 4.0s (gate1_ingest.py:71) ✅ |
| Hop | 1.0-2.0s | 2.0s (gate1_ingest.py:72) ✅ |
| Sample Rate | 16k o 22.05k | 22050 Hz (gate1_ingest.py:70) ✅ |

### 2.5 ✅ Función midi_to_constellation_tokens Existe

Verificada en `src/utils/midi_utils.py:226`

### 2.6 ✅ Encoders Temporales Implementados

- VICRegTemporalEncoder: LSTM bidireccional ✅
- BarlowTemporalEncoder: LSTM bidireccional ✅
- MLPConstellationEncoder: LSTM bidireccional con attention pooling ✅
- TransformerConstellationEncoder: Self-attention + temporal transformer ✅

---

## 3. ARCHIVOS Y ESTRUCTURA

### 3.1 Todos los Gates Presentes

```
experiments/maestro/
├── gate0_harness.py          ✅ 20KB - Métricas + controles negativos
├── gate1_ingest.py           ✅ 21KB - Descarga + segmentación MAESTRO
├── gate2_baselines.py        ✅ 25KB - Chroma + CCA/Ridge baselines
├── gate3_cross_modal.py      ✅ 22KB - Training VICReg/Barlow (denso)
├── gate4_ratio_tokens.py     ✅ 28KB - Training constellation tokens
├── gate5_moco.py             ✅ 33KB - MoCo + hard negatives
└── run_maestro_experiment.py ✅ 23KB - Script orquestador
```

### 3.2 Módulos de Soporte

```
src/utils/midi_utils.py           ✅ Parseo MIDI, piano roll, constellation tokens
src/RNA/vicreg.py                 ✅ VICReg loss + encoders temporales
src/RNA/barlow_twins.py           ✅ Barlow Twins loss + encoders temporales
src/RNA/constellation_vae.py      ✅ ConstellationVAE modular (C1-C4)
src/RNA/jepa_lite.py              ✅ JEPA-lite sin decoder (C5-C6)
src/analizador/analizador_maestro.py  ✅ Extractor constellations audio+MIDI
src/datasets/maestro_dataset.py   ✅ DataLoader para tokens MAESTRO
```

---

## 4. CORRECCIONES DETALLADAS

### 4.1 Corregir gate4_ratio_tokens.py

**Archivo**: `/mnt/m2-1TB/Phideus/experiments/maestro/gate4_ratio_tokens.py`

**Cambio 1** - Alrededor de línea 590-600, después de crear dataloaders:

```python
# Buscar esta sección:
train_loader, val_loader, test_loader = create_maestro_dataloaders(
    npz_path=args.data,
    batch_size=args.batch_size,
    max_frames=args.max_frames,
    num_workers=args.num_workers,
    seed=args.seed,
)

# AGREGAR después:
# Leer max_tokens del dataset (NPZ contiene el valor correcto)
max_tokens = train_loader.dataset.max_tokens
print(f"Using max_tokens={max_tokens} from dataset")
```

**Cambio 2** - En la creación del modelo (~línea 650):

```python
# CAMBIAR de:
if args.model == 'constellation':
    model = ConstellationVAE(
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type,
        token_dim=5,
        max_tokens=args.max_tokens,  # ← ESTO USA DEFAULT 48
        ...
    )

# A:
if args.model == 'constellation':
    model = ConstellationVAE(
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type,
        token_dim=5,
        max_tokens=max_tokens,  # ← USAR VALOR DEL NPZ
        ...
    )
```

**Cambio 3** - Similar para JEPA-lite (~línea 663):

```python
# CAMBIAR de:
else:  # jepa-lite
    model = JEPALite(
        encoder_type=args.encoder_type,
        token_dim=5,
        max_tokens=args.max_tokens,
        ...
    )

# A:
else:  # jepa-lite
    model = JEPALite(
        encoder_type=args.encoder_type,
        token_dim=5,
        max_tokens=max_tokens,  # ← USAR VALOR DEL NPZ
        ...
    )
```

### 4.2 Corregir gate5_moco.py

**Archivo**: `/mnt/m2-1TB/Phideus/experiments/maestro/gate5_moco.py`

**Cambio 1** - Después de crear dataloaders (~línea 810):

```python
# Buscar:
train_loader, val_loader, test_loader = create_maestro_dataloaders(
    npz_path=args.data,
    ...
)

# AGREGAR después:
max_tokens = train_loader.dataset.max_tokens
print(f"Using max_tokens={max_tokens} from dataset")
```

**Cambio 2** - En la creación del modelo (~línea 825):

```python
# CAMBIAR de:
model = MoCoCrossModal(
    encoder_type=args.encoder_type,
    token_dim=5,
    max_tokens=args.max_tokens,  # ← ESTO USA DEFAULT 48
    ...
)

# A:
model = MoCoCrossModal(
    encoder_type=args.encoder_type,
    token_dim=5,
    max_tokens=max_tokens,  # ← USAR VALOR DEL NPZ
    ...
)
```

---

## 5. CHECKLIST PRE-EJECUCIÓN

### Correcciones de código:
- [x] ~~Corregir max_tokens en gate4_ratio_tokens.py (3 cambios)~~ ✅ APLICADO
- [x] ~~Corregir max_tokens en gate5_moco.py (2 cambios)~~ ✅ APLICADO

### Dependencias:
- [ ] `pip install torch librosa pretty_midi scikit-learn mido`

### Datos:
- [ ] Descargar MAESTRO v3.0.0 (101GB)
- [ ] Verificar espacio en disco (~180GB necesarios)

### Verificación post-corrección:

```bash
cd /mnt/m2-1TB/Phideus
source venv/bin/activate

# Verificar imports
python -c "
from src.utils.midi_utils import midi_to_constellation_tokens
from src.RNA.constellation_vae import ConstellationVAE
from src.RNA.jepa_lite import JEPALite
from src.datasets.maestro_dataset import MAESTROConstellationDataset
print('✓ Imports OK')
"

# Test rápido de shapes (después de tener datos)
python -c "
import numpy as np
data = np.load('data/maestro_v3/constellations/tokens.npz', allow_pickle=True)
max_tokens = int(data['max_tokens_per_frame'])
print(f'NPZ max_tokens_per_frame = {max_tokens}')
# Debería imprimir 64
"
```

---

## 6. COMANDOS PARA APLICAR CORRECCIONES

Después de la compactación, ejecutar:

```bash
cd /mnt/m2-1TB/Phideus

# Ver este archivo para referencia
cat Documents/ESCALON_1/AUDITORIA_IMPLEMENTACION.md

# Aplicar las correcciones descritas en la sección 4
```

---

## 7. RESUMEN

**Estado de la implementación**: ✅ **100% completa**

**Issues críticos**: ~~1 (mismatch max_tokens)~~ → **CORREGIDO**

**Issues menores**: 2 (dependencias externas, __init__.py - no bloqueantes)

**Próximo paso**: Descargar MAESTRO y ejecutar el experimento.

**Correcciones aplicadas el**: 2026-02-04
