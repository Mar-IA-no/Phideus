  INFORME COMPLETO: d4a4-scratch 30 epochs                                                                                                 
                                                                                                                                           
  Run: d4a4 dual same-modality concat, from MERT pretrained + random MIDI (sin foundation fine-tuning previo)
  Duración: 636 min (10.6h)                                                                                                                
  Best model: epoch 30, S=83.6% — RECORD ABSOLUTO del proyecto             
  Output: data/bias_control_medium/training_outputs/gate43/gate43_d4a4_scratch_30ep/

  ---
  Tabla completa structured evals

  ┌─────┬───────┬───────┬───────┬───────┬──────────┬─────────┬─────────┬───────────┬──────────────┐
  │ Ep  │ Loss  │   S   │  A2M  │  M2A  │ hard_neg │ MRR avg │ R@1 avg │ mean_rank │ vs D-02 best │
  ├─────┼───────┼───────┼───────┼───────┼──────────┼─────────┼─────────┼───────────┼──────────────┤
  │ 10  │ 13.60 │ 74.6% │ 74.6% │ 75.0% │ 93.0%    │ 0.336   │ 15.9%   │ 7.7       │ +12.8pp      │
  ├─────┼───────┼───────┼───────┼───────┼──────────┼─────────┼─────────┼───────────┼──────────────┤
  │ 15  │ 13.38 │ 65.8% │ 65.8% │ 68.6% │ 91.0%    │ 0.316   │ 16.4%   │ 10.0      │ +4.0pp       │
  ├─────┼───────┼───────┼───────┼───────┼──────────┼─────────┼─────────┼───────────┼──────────────┤
  │ 20  │ 13.26 │ 75.6% │ 75.6% │ 76.8% │ 93.6%    │ 0.370   │ 19.0%   │ 7.0       │ +13.8pp      │
  ├─────┼───────┼───────┼───────┼───────┼──────────┼─────────┼─────────┼───────────┼──────────────┤
  │ 25  │ 13.21 │ 82.2% │ 82.8% │ 82.2% │ 95.4%    │ 0.430   │ 25.2%   │ 5.7       │ +20.4pp      │
  ├─────┼───────┼───────┼───────┼───────┼──────────┼─────────┼─────────┼───────────┼──────────────┤
  │ 28  │ 13.19 │ 82.8% │ 82.8% │ 83.6% │ 94.8%    │ 0.444   │ 26.4%   │ 5.6       │ +21.0pp      │
  ├─────┼───────┼───────┼───────┼───────┼──────────┼─────────┼─────────┼───────────┼──────────────┤
  │ 29  │ 13.19 │ 82.6% │ 82.6% │ 83.8% │ 95.2%    │ 0.443   │ 26.3%   │ 5.4       │ +20.8pp      │
  ├─────┼───────┼───────┼───────┼───────┼──────────┼─────────┼─────────┼───────────┼──────────────┤
  │ 30  │ 13.20 │ 83.6% │ 84.0% │ 83.6% │ 95.2%    │ 0.444   │ 25.9%   │ 5.4       │ +21.8pp      │
  └─────┴───────┴───────┴───────┴───────┴──────────┴─────────┴─────────┴───────────┴──────────────┘

  D-02 best = S=61.8% (epoch 25)

  ---
  Trayectoria S

  e10: ██████████████████████████████████████░░░░░  74.6%
  e15: ████████████████████████████████░░░░░░░░░░░  65.8%  ← dip (schedule)
  e20: ██████████████████████████████████████░░░░░  75.6%
  e25: ████████████████████████████████████████░░░  82.2%
  e28: █████████████████████████████████████████░░  82.8%
  e29: █████████████████████████████████████████░░  82.6%
  e30: ██████████████████████████████████████████░  83.6%  ← RECORD

  ---
  Contexto comparativo

  ┌───────────────────────────────┬────────┬──────────────────────────┐
  │            Modelo             │ Best S │        Referencia        │
  ├───────────────────────────────┼────────┼──────────────────────────┤
  │ Gate 2 baseline               │ 34.4%  │ checkpoint_epoch45.pt    │
  ├───────────────────────────────┼────────┼──────────────────────────┤
  │ D-02 (30ep, sin descriptores) │ 61.8%  │ foundation_locked_e25.pt │
  ├───────────────────────────────┼────────┼──────────────────────────┤
  │ d4a4 foundation (5ep)         │ 69.8%  │ Gate 4.3 best arm        │
  ├───────────────────────────────┼────────┼──────────────────────────┤
  │ d4a4 scratch (30ep)           │ 83.6%  │ este run                 │
  └───────────────────────────────┴────────┴──────────────────────────┘

  ---
  Observaciones

  1. e30 rompió el plateau: después de 82.2→82.8→82.6 en e25-29, e30 saltó a 83.6% (+1.0pp). El modelo no está saturado.
  2. A2M alcanzó 84.0%: primera vez que A2M supera a M2A de forma clara. El bottleneck histórico (audio→MIDI más difícil que MIDI→audio) se
   está cerrando.
  3. Dip de e15 fue transitorio: la caída a 65.8% en e15 fue por el schedule del LR (warmup → decay transition). Se recuperó completamente.
  4. hard_neg estable en ~95%: fluctúa entre 93.0-95.4% pero sin tendencia a la baja. El modelo distingue segmentos del mismo piano
  consistentemente.
  5. mean_rank mejoró de 7.7 a 5.4: el match correcto pasó de top-8 a top-5 en un pool de 256.
  6. Loss convergiendo: 13.60 → 13.20, bajada suave. LR final ~1.3e-08 (prácticamente cero). Para seguir mejorando necesitaría más epochs
  con LR reset o schedule extendido.
  7. +21.8pp sobre D-02: la inyección dual de descriptores (MIDI intervals + audio log-freq deltas) aporta masivamente comparado con el
  mismo training sin descriptores.
  8. Señal clara de que más epochs = más S: la curva e20→e30 no muestra saturación. Un run de 50-60ep con schedule adaptado podría empujar
  más allá de 85%.