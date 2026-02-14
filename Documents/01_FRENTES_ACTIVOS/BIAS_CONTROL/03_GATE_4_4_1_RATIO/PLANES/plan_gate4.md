# Gate 4: Ratio Auxiliary View — Plan de Implementacion

> [!IMPORTANT]
> Este documento queda como **plan histórico de Gate 4 base**.  
> Estado actual del roadmap: **Gate 4 base completado** y continuidad por **Gate 4.1 (DEC-004)**.  
> Referencias vigentes: `COLLAB/DECISIONS.md` (DEC-004) y `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`.

 Contexto

 Gate 3 (DANN) cerrado tras 4 runs — ninguno mejora sobre Gate 2. El mejor checkpoint es Gate 2 epoch 45:
 - R@10 a2m: 34.4%, R@10 m2a: 37.6%, Hard neg: 80.4%, MRR: 0.138

 Gate 4 reinyecta el "ratio insight" (distribucion de intervalos armonicos) como loss auxiliar VICReg junto al alignment principal
 audio-MIDI.

 Problema Critico del Script Actual

 El script gate4_ratio_auxiliary.py congela el modelo base completo (linea 547-548) y solo entrena el ratio encoder. Como la evaluacion
 usa embeddings audio/MIDI (no ratio), las metricas de retrieval serian IDENTICAS a Gate 2 sin importar el training. Esto invalida el
 test.

 Ademas:
 - Linea 537: use_dann=True — Gate 2 checkpoint NO tiene DANN
 - No guarda checkpoints por epoch (solo best + final)
 - No tiene warmup ni LR groups
 - Referencia "gate3" cuando el baseline es Gate 2

 Solucion: Descongelar Base (excepto MERT)

 Descongelar MIDI encoder + projection heads (como en Gate 2/3). La loss auxiliar de ratios actua como regularizador que incentiva que los
  embeddings audio/MIDI tambien se alineen con estructura de intervalos armonicos.

 Diferencia clave vs DANN: DANN fuerza INVARIANCIA (destruye informacion). Ratio auxiliary agrega ESTRUCTURA (senal de alignment
 adicional). Por eso un peso moderado deberia ser mas tolerable.

 ---
 Cambios en experiments/bias_control/gate4_ratio_auxiliary.py

 1. Model Creation (run_gate4, linea ~535)

 - use_dann=False (Gate 2 no tiene DANN)
 - Cargar checkpoint Gate 2
 - Congelar SOLO MERT encoder (param.requires_grad = False)
 - Dejar entrenables: MIDI encoder, audio proj, midi proj, ratio encoder, ratio proj

 2. Optimizer Multi-Group con Warmup (Gate4Trainer.init)

 Group 1: MIDI encoder        — lr=5e-5  (conservador, preservar features aprendidos)
 Group 2: Audio/MIDI proj     — lr=1e-4  (moderado)
 Group 3: Ratio encoder+proj  — lr=5e-4  (modulos nuevos, aprender mas rapido)

 Warmup: 500 steps linear
 Scheduler: CosineAnnealingLR
 Weight decay: 1e-4

 3. Checkpoint Saving (save_checkpoint)

 - Guardar checkpoint en CADA epoch (directiva del proyecto)
 - Incluir: epoch, scheduler_state, optimizer_state, global_step, metrics

 4. Vectorizar compute_ratio_histogram

 - Loop actual por sample (lineas 326-334) es lento
 - Vectorizar: batch-level computation con padding mask
 - O mover computo al DataLoader (prefetch)

 5. Logging y Metricas

 - Log LR por grupo, ratio loss (audio-ratio, midi-ratio) por separado
 - Formato consistente con Gate 2/3 output
 - Renombrar gate3_recall → baseline_recall

 ---
 Configuracion de Training
 ┌─────────────────┬────────────────────┬──────────────────────────────────────────────────────────────┐
 │    Parametro    │       Valor        │                            Razon                             │
 ├─────────────────┼────────────────────┼──────────────────────────────────────────────────────────────┤
 │ Checkpoint base │ Gate 2 ep45        │ Mejor disponible                                             │
 ├─────────────────┼────────────────────┼──────────────────────────────────────────────────────────────┤
 │ Epochs          │ 30                 │ Consistente con Gate 2                                       │
 ├─────────────────┼────────────────────┼──────────────────────────────────────────────────────────────┤
 │ Batch size      │ 64                 │ Consistente con Gate 2                                       │
 ├─────────────────┼────────────────────┼──────────────────────────────────────────────────────────────┤
 │ ratio_weight    │ 0.1                │ Conservador (leccion Gate 3: auxiliares agresivos destruyen) │
 ├─────────────────┼────────────────────┼──────────────────────────────────────────────────────────────┤
 │ Warmup          │ 500 steps          │ Start suave                                                  │
 ├─────────────────┼────────────────────┼──────────────────────────────────────────────────────────────┤
 │ Gradient clip   │ 1.0                │ Consistente con Gate 3                                       │
 ├─────────────────┼────────────────────┼──────────────────────────────────────────────────────────────┤
 │ Val batches     │ Todas (846)        │ Leccion Gate 3: no limitar validacion                        │
 ├─────────────────┼────────────────────┼──────────────────────────────────────────────────────────────┤
 │ num_workers     │ 8, pin_memory=True │ Hardware optimization                                        │
 └─────────────────┴────────────────────┴──────────────────────────────────────────────────────────────┘
 Protocolo de Evaluacion

 Durante Training (cada epoch)

 - R@1/5/10/20 ambas direcciones, MRR, gap
 - Time discrimination: same-piece-same-time vs diff-time gap
 - Ratio loss components (audio-ratio, midi-ratio)

 Final GO/NO-GO

 Structured pool evaluation (evaluate_structured_pool.py):
 - 256 candidatos, 500 queries, seed 42
 - Mismo protocolo exacto que Gate 3

 Criterios
 ┌──────────────┬──────────┬────────────────┬─────────────┐
 │   Metrica    │    GO    │    WEAK-GO     │    NO-GO    │
 ├──────────────┼──────────┼────────────────┼─────────────┤
 │ R@10 a2m     │ >= 34.4% │ >= 32.7% (95%) │ < 32.7%     │
 ├──────────────┼──────────┼────────────────┼─────────────┤
 │ Hard neg     │ >= 80.4% │ >= 76.4% (95%) │ < 76.4%     │
 ├──────────────┼──────────┼────────────────┼─────────────┤
 │ Time discrim │ Mejora   │ Sin cambio     │ Degradacion │
 └──────────────┴──────────┴────────────────┴─────────────┘
 ---
 Ejecucion

 Paso 1: Modificar script

 Aplicar todos los cambios listados arriba.

 Paso 2: Training en tmux

 tmux new -s gate4
 cd /mnt/m2-1TB/Phideus && source venv/bin/activate
 python experiments/bias_control/gate4_ratio_auxiliary.py \
     --maestro-dir data/maestro_v3/maestro-v3.0.0 \
     --checkpoint data/bias_control_medium/training_outputs/gate2/checkpoint_epoch45.pt \
     --output data/bias_control_medium/training_outputs/gate4 \
     --epochs 30 --ratio-weight 0.1 --batch-size 64 --num-workers 8 \
     --baseline-recall 0.36

 Paso 3: Structured pool eval

 python experiments/bias_control/evaluate_structured_pool.py \
     --checkpoint data/bias_control_medium/training_outputs/gate4/best_model.pt \
     --maestro-dir data/maestro_v3/maestro-v3.0.0 \
     --output data/bias_control_medium/evaluations/gate4/ \
     --n-queries 500 --pool-size 256 --n-hard 64 --seed 42

 Paso 4: Decision GO/NO-GO + documentacion

 ---
 Archivos a Modificar

 - experiments/bias_control/gate4_ratio_auxiliary.py — Rewrite significativo del trainer

 Archivos NO Modificar

 - src/bias_control/architectures/cross_modal_model.py
 - src/bias_control/datasets/maestro_segments.py
 - experiments/bias_control/evaluate_structured_pool.py

 Verificacion

 1. Smoke test: 1 epoch, verificar que loss baja y metricas se computan
 2. Training completo: 30 epochs en tmux
 3. Structured pool eval del best checkpoint
 4. Comparar tabla vs Gate 2 baseline
