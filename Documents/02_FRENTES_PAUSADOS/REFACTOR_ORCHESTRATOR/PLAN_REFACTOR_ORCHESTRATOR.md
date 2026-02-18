# Refactor: gate43_scratch_training.py → Orchestrator + Self-Describing Arms

**Estado**: PAUSADO — ejecutar post-screening Gate 4.4 y d4-a4r en UNC.
**Fecha plan**: 2026-02-17

## Contexto

El archivo tiene ~3900 líneas, 24 descriptors, 14 model classes y **10 dispatch points** (if/elif chains) que hay que tocar cada vez que se agrega un arm. El último arm (d4-a4r) requirió 14 edits y Codex encontró 2 bugs. Cada gate nuevo hereda y amplifica la deuda.

**Timing**: post-screening de Gate 4.4 y d4-a4r en UNC. Sin presión de producción.

## Diseño: Protocolo Self-Describing

Cada modelo declara todo lo que el orquestador necesita como atributos de clase y métodos:

```python
class GateModel(nn.Module, ABC):
    # Qué descriptors sirve y con qué kwargs del constructor
    DESCRIPTORS: ClassVar[Dict[str, dict]] = {}
    # ¿Es eval-compatible con evaluate_structured_pool.py?
    EVAL_COMPATIBLE: ClassVar[bool] = False
    # Batch size para eval (None = default 64)
    EVAL_BATCH_SIZE: ClassVar[Optional[int]] = None
    # Métricas extra que el training loop debe acumular (e.g. MoE load_balance)
    EXTRA_METRICS: ClassVar[List[str]] = []

    # --- Métodos obligatorios ---
    def forward(self, audio, midi_pitch, midi_velocity, midi_duration, midi_mask=None)
    def compute_total_loss(self, audio, midi_pitch, midi_velocity, midi_duration, ...)

    # --- Métodos que reemplazan dispatch points ---
    def get_descriptor_param_groups(self, lr_ratio) -> List[dict]  # optimizer groups
    def get_trainable_prefixes(self) -> List[str]                  # preflight contract
    def get_param_range(self, freeze_policy) -> Tuple[int, int]    # param count validation
```

**Registry por decorador**:
```python
ARM_REGISTRY: Dict[str, Tuple[type, dict]] = {}

def register_arm(cls):
    for desc_name, kwargs in cls.DESCRIPTORS.items():
        ARM_REGISTRY[desc_name] = (cls, kwargs)
    return cls
```

## Mapeo: 10 Dispatch Points → Protocolo

| Dispatch Point | Antes (if/elif) | Después (protocolo) |
|---|---|---|
| Factory (24 branches) | `create_gate42_model()` | `ARM_REGISTRY[desc]` |
| Optimizer (15 branches) | `create_gate42_optimizer()` | `model.get_descriptor_param_groups()` |
| Param ranges (48 entries) | `GATE42_PARAM_RANGES` dict | `model.get_param_range()` |
| Preflight (15 branches) | `get_gate42_preflight_contract()` | `model.get_trainable_prefixes()` |
| Checkpoint eval_compatible | `descriptor not in (...)` | `model.EVAL_COMPATIBLE` |
| Eval reconstruction (20 branches) | `run_evaluate()` if/elif | Mismo `ARM_REGISTRY[desc]` |
| Eval batch_size (3 branches) | `run_evaluate()` if/elif | `model.EVAL_BATCH_SIZE` |
| CLI choices (hardcoded list) | argparse choices | `list(ARM_REGISTRY.keys())` |
| MoE metrics (startswith) | `descriptor.startswith('moe-')` | `model.EXTRA_METRICS` |
| Gate label | `_is_gate44` / `_is_gate43_ext` | `arch_config['gate']` (ya corregido) |

## File Layout

```
experiments/bias_control/gate43_scratch/
├── gate43_scratch_training.py    # Orquestador slim (~700 líneas)
└── arms/
    ├── __init__.py               # ARM_REGISTRY, register_arm, GateModel ABC, create_model()
    ├── _helpers.py               # 12 helpers compartidos (_encode_audio_*, _encode_midi_*, etc.)
    ├── legacy_auxiliary.py       # d0, d1, d2, d3
    ├── midi_concat.py            # d4
    ├── audio_concat.py           # a4, a7
    ├── dual_concat.py            # d4a4, d4a7
    ├── audio_cross_att.py        # a4x, a7x
    ├── midi_cross_att.py         # d4x
    ├── dual_cross_modal.py       # d4a4cm
    ├── audio_reverse.py          # a4r
    ├── dual_reverse.py           # d4a4r
    ├── dual_mixed.py             # d4-a4r
    ├── third_tower.py            # t3-tri, t3-anc, t3-wt
    ├── film.py                   # film-a4, film-d4, film-dual + FiLMGenerator
    └── moe.py                    # moe-a4, moe-dual + MoEAdapter
```

**Qué queda en el orquestador**: imports, utilities (scheduler, seeding), load_foundation, apply_freeze_policy, create_optimizer (simplificado: base groups + `model.get_descriptor_param_groups()`), save_checkpoint (simplificado: `model.EVAL_COMPATIBLE`), run_structured_eval, quick_val_eval, train_loop (simplificado: `model.EXTRA_METRICS`), run_train, run_evaluate, CLI.

**Qué va a `_helpers.py`**: las 12 funciones helper compartidas entre múltiples modelos (`_encode_audio_with_reverse_cross_attention`, `_encode_midi_with_intervals`, `interpolate_d4_masked`, etc.) + imports comunes (torch, CrossModalModel, audio_descriptors, etc.).

## Estrategia de Migración (4 fases, sin riesgo)

### Fase 0: Andamiaje (puro aditivo, nada cambia)
1. Crear `arms/__init__.py` con `GateModel` ABC, `ARM_REGISTRY`, `register_arm`, `create_model()`
2. Crear `arms/_helpers.py` — copiar las 12 helper functions (no borrar del original todavía)
3. Test: `python -c "from experiments.bias_control.gate43_scratch.arms import ARM_REGISTRY"` → OK, vacío

### Fase 1: Proof of concept — migrar 1 arm (a4, a7)
1. Crear `arms/audio_concat.py` con `Gate42AudioAugModel` adaptado al protocolo
2. En `gate43_scratch_training.py`, agregar fallback dual:
   ```python
   if descriptor in ARM_REGISTRY:
       model = create_model(descriptor, base_model)
   else:
       model = create_gate42_model(descriptor, base_model, ratio_weight)
   ```
3. **Verificación**: pilot 50 batches con a4, comparar loss batch-a-batch vs código viejo (mismo seed). Deben ser idénticos.
4. Checkpoint round-trip: guardar con nuevo, cargar con viejo (y viceversa). `strict=True` pasa.

### Fase 2: Migrar el resto por familias
Orden (de menor a mayor complejidad):
1. **Concat simples**: d4, d4a4, d4a7
2. **Cross-attention**: a4x, a7x, d4x
3. **Dual/mixed**: d4a4cm, a4r, d4a4r, d4-a4r
4. **Gate 4.4**: t3-*, film-*, moe-*
5. **Legacy**: d0, d1, d2, d3 (más diferentes, con auxiliary branch)

Cada grupo = 1 commit. Después de cada commit, pilot de smoke test.

### Fase 3: Eliminar código muerto
1. Borrar las if/elif chains originales (factory, optimizer, preflight, eval reconstruct)
2. Borrar `GATE42_PARAM_RANGES` dict, `_make_descriptor_fn()`
3. Borrar helpers duplicados del monolito (ya viven en `_helpers.py`)
4. Un commit limpio.

## Compat con UNC y checkpoints

- **CLI**: Idéntico. `--descriptor a4 --checkpoint X --output Y` funciona igual.
- **Checkpoints**: Los `nn.Module` no cambian de estructura → `state_dict` keys idénticas → `strict=True` OK.
- **SLURM scripts**: Cero cambios.
- **arch_config**: Sin cambios de formato.

## Verificación end-to-end

Para cada arm migrado:
1. `python gate43_scratch_training.py --mode train --descriptor <arm> --epochs 1 --max-batches-per-epoch 5 --skip-structured-eval --max-val-batches 2` → preflight PASS, loss decrece, checkpoint OK
2. `python gate43_scratch_training.py --mode evaluate --checkpoint <ckpt> --output /tmp/eval.json` → carga OK, eval OK
3. Checkpoint de antes del refactor → cargable con código nuevo

## Resultado esperado

- **Agregar arm nuevo** = 1 archivo en `arms/`, 1 línea de import en `__init__.py`
- **Orquestador**: ~700 líneas (vs 3900 actuales)
- **Total del proyecto**: ~3020 líneas (23% menos por eliminar dispatch duplicado)
- **Cero regresión** en los 24 arms existentes
