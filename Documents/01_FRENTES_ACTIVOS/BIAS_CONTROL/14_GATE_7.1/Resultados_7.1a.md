  Gate 7.1a — RESULTADO FINAL                                                                                                                               
                                                                                                                                                            
  Structured eval (canónica: pool=256 piezas, 500 queries)                                                                                                  
                                                                                                                                                            
  ┌───────┬────────┬────────┬────────┬──────────┐           
  │ Epoch │  A2M   │  M2A   │   S    │ hard_neg │
  ├───────┼────────┼────────┼────────┼──────────┤
  │  5    │ 75.0%  │ 71.2%  │ 71.2%  │  92.8%   │
  │ 10    │ 80.8%  │ 75.0%  │ 75.0% ★│  94.0%   │
  │ 15    │ 81.0%  │ 74.2%  │ 74.2%  │  94.2%   │
  │ 20    │ 78.2%  │ 70.6%  │ 70.6%  │  93.4%   │
  │ 25    │ 79.6%  │ 74.8%  │ 74.8%  │  94.2%   │
  │ 28    │ 79.2%  │ 72.4%  │ 72.4%  │  93.0%   │
  │ 29    │ 79.2%  │ 71.6%  │ 71.6%  │  93.2%   │
  │ 30    │ 81.0%  │ 74.6%  │ 74.6%  │  93.2%   │
  └───────┴────────┴────────┴────────┴──────────┘

  ★ Best model saved: epoch 10, S = 75.0%

  Quick val (pool=400, más ruidoso pero con más resolución temporal)

  ┌───────┬────────┬───────┬────────┬───────┬────────┬───────┬────────┐
  │ Epoch │  val_S │   ep  │  val_S │   ep  │  val_S │   ep  │  val_S │
  ├───────┼────────┼───────┼────────┼───────┼────────┼───────┼────────┤
  │  1    │ 40.7%  │   9   │ 73.5%  │  17   │ 83.2%  │  25   │ 84.2%  │
  │  2    │ 63.5%  │  10   │ 78.3%  │  18   │ 79.5%  │  26   │ 82.0%  │
  │  3    │ 74.0%  │  11   │ 80.5%  │  19   │ 80.5%  │  27   │ 83.2%  │
  │  4    │ 62.5%  │  12   │ 76.2%  │  20   │ 80.2%  │  28   │ 83.2%  │
  │  5    │ 67.8%  │  13   │ 81.2%  │  21   │ 82.0%  │  29   │ 83.5%  │
  │  6    │ 69.7%  │  14   │ 84.0%  │  22   │ 84.0%  │  30   │ 83.7%  │
  │  7    │ 76.7%  │  15   │ 78.5%  │  23   │ 83.2%  │      │        │
  │  8    │ 80.5%  │  16   │ 78.5%  │  24   │ 82.0%  │      │        │
  └───────┴────────┴───────┴────────┴───────┴────────┴───────┴────────┘

  Comparación con D0_lite

  D0_mert330m (frozen):  S = 75.0% (best @ ep10)   ← 1 seed
  D0_lite (MERTLite):    S = 75.2% ±2.3pp           ← 5 seeds

  Δ = -0.2pp  →  prácticamente idénticos

  Observaciones factuales (sin juicio GO/NO-GO)

  1. S plateauea desde epoch 10: oscila 70.6-75.0% en structured eval, sin tendencia ascendente
  2. Quick val más optimista: 80-84% en epochs tardíos, pero structured eval (canónica) no sube
  3. Loss sigue bajando (19.3 → 14.3) mientras S no mejora — el modelo aprende algo que no se traduce a retrieval
  4. M2A es el cuello: A2M llega a 81%, M2A se queda en 70-75%. El MIDI encoder (from scratch) limita
  5. Hard neg estable ~93-94%, comparable a D0_lite
  6. Tiempo total: ~2h50min training + ~45min eval = ~3.5h (exacto a la estimación)

✻ Brewed for 1m 8s

❯ entonces que implicancias tiene esto para nuestro roadmap?                                                                                                
                                                             
● Buena pregunta. Voy a presentar las implicancias factuales y dejar las decisiones para vos.                                                               
                                         
  Lo que Gate 7.1a nos dice                                                                                                                                 
                                                            
  Dato duro: MERT-330M frozen (5x más parámetros, R²=0.850 para A4) produce exactamente el mismo retrieval que MERTLite entrenado con VICReg (75.0% vs      
  75.2%).                                                   

  Lectura directa: el cuello de botella de VICReg cross-modal no es la capacidad del encoder de audio. Un encoder más fuerte con A4 más linealmente
  accesible no escala S.

  Dato secundario: M2A (70-75%) es consistentemente el limitante, mientras A2M llega a 81%. El MIDI encoder (from scratch, ~12.9M params) es el lado débil.

  Implicancias por frente

  Gate 7.1b (a4r-MERT)

  La motivación original era: "si D0_mert >> D0_lite, ¿ΔA4 persiste?" Pero D0_mert ≈ D0_lite, así que el baseline es plano. Testear ΔA4 sobre un baseline
  que no se movió es menos informativo — no distinguiría "A4 no ayuda con encoder fuerte" de "frozen dynamics bloquean todo".

  Gate 5A C1 (Conditioned Projections)

  Se vuelve más relevante, no menos. Dos evidencias independientes convergen:
  - Test 11 Pre-Proj: MIDI projection destruye ~88% de info
  - Gate 7.1a: más capacidad en audio encoder no ayuda

  Ambas apuntan al mismo cuello: projection heads / MIDI encoder, no el audio encoder.

  Gate 6 (AMT)

  No se ve afectado. Gate 6 pregunta algo diferente (¿descriptores ayudan en tareas downstream?). Sigue su curso en UNC.

  Escalón 2

  No estaba bloqueado por Gate 7.1. Sin efecto.

  Hallazgo más amplio

  La co-adaptación encoder↔encoder durante training parece importar. MERTLite entrenado con VICReg (layers 2-3 descongelados) rinde igual que MERT-330M
  frozen. Esto es consistente con Gate 5B: la ventaja es geométrica (reorganización del espacio), y esa reorganización requiere que ambos lados se ajusten
  mutuamente, no que uno sea más poderoso.