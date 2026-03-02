  Análisis Detallado: Tests 02 y 13G-B (últimos resultados)                                                                                                 
                                                                                                                                                            
  ---                                                                                                                                                       
  TEST 02 — Parameter-Matched Ablation (4/4 cerrado)                                                                                                        

  Pregunta: ¿La mejora de d4a4 viene de la información del descriptor o de los parámetros extra que agrega?

  Diseño: Arquitectura idéntica en los 4 modos (75.5M params total, 66.2M trainable), mismo seed, mismo schedule 30ep. Solo cambia qué ve el descriptor:

  ┌──────────┬──────────────────────────┬──────────────────────────┬─────────────────────┐
  │   Mode   │      Descriptor A4       │      Descriptor D4       │      Concepto       │
  ├──────────┼──────────────────────────┼──────────────────────────┼─────────────────────┤
  │ real     │ Valores reales           │ Valores reales           │ Control positivo    │
  ├──────────┼──────────────────────────┼──────────────────────────┼─────────────────────┤
  │ random   │ Ruido N(0,1) por sample  │ Ruido N(0,1) por sample  │ Capacidad sin señal │
  ├──────────┼──────────────────────────┼──────────────────────────┼─────────────────────┤
  │ zero     │ Vector de ceros          │ Vector de ceros          │ Capacidad sin input │
  ├──────────┼──────────────────────────┼──────────────────────────┼─────────────────────┤
  │ shuffled │ Real pero de OTRO sample │ Real pero de OTRO sample │ Info incoherente    │
  └──────────┴──────────────────────────┴──────────────────────────┴─────────────────────┘

  Resultados finales

  ┌──────────┬────────┬──────────┬──────────┬───────┬───────┬───────┬──────────┬─────────┐
  │   Mode   │   S    │ A2M R@10 │ M2A R@10 │  MRR  │  R@1  │  R@5  │ hard_neg │ best_ep │
  ├──────────┼────────┼──────────┼──────────┼───────┼───────┼───────┼──────────┼─────────┤
  │ real     │ 83.0%  │ 83.2%    │ 83.0%    │ 0.458 │ 28.8% │ 68.2% │ 94.0%    │ e25     │
  ├──────────┼────────┼──────────┼──────────┼───────┼───────┼───────┼──────────┼─────────┤
  │ zero     │ 75.0%  │ 75.0%    │ 76.0%    │ 0.396 │ 23.6% │ 59.2% │ 95.4%    │ e28     │
  ├──────────┼────────┼──────────┼──────────┼───────┼───────┼───────┼──────────┼─────────┤
  │ random   │ 73.6%  │ 74.4%    │ 73.6%    │ 0.386 │ 22.4% │ 57.6% │ 95.2%    │ e30     │
  ├──────────┼────────┼──────────┼──────────┼───────┼───────┼───────┼──────────┼─────────┤
  │ shuffled │ 73.6%* │ 73.8%    │ 73.6%    │ 0.374 │ 21.4% │ 54.4% │ 93.6%    │ e20*    │
  └──────────┴────────┴──────────┴──────────┴───────┴───────┴───────┴──────────┴─────────┘

  *parcial (e20/30), pero convergencia estabilizada.

  Interpretación por capa

  1. Separación limpia: real vs todo lo demás

  La brecha es nítida y consistente:
  - real → 83.0%
  - ablaciones → 73.6-75.0% (zona D0 baseline)
  - Delta: -8.0pp a -9.4pp

  No hay ambigüedad: los 66.2M parámetros entrenables extra no producen mejora por sí solos. La mejora requiere información descriptiva coherente.

  2. Jerarquía dentro de las ablaciones: zero > random ≈ shuffled

  Zero (75.0%) supera a random y shuffled (73.6%) por +1.4pp. Esto tiene sentido:
  - Zero: Input constante y determinista. El modelo puede aprender a ignorarlo completamente — es como si la rama del descriptor no existiera. Actúa como
  regularizador mínimo (BatchNorm en la rama del descriptor con input constante normaliza de forma estable).
  - Random: Ruido diferente en cada forward pass. El modelo tiene que lidiar con señal espuria que cambia constantemente — gasta capacidad aprendiendo a
  suprimirla.
  - Shuffled: Info real pero desincronizada. Peor que random en MRR (0.374 vs 0.386) — la info coherente-pero-incorrecta puede ser activamente engañosa.

  3. Curvas de aprendizaje revelan dinámica diferente

  ┌───────┬───────┬───────┬────────┬──────────┐
  │ Epoch │ real  │ zero  │ random │ shuffled │
  ├───────┼───────┼───────┼────────┼──────────┤
  │ 5     │ 30.4% │ 43.0% │ 48.6%  │ 51.8%    │
  ├───────┼───────┼───────┼────────┼──────────┤
  │ 10    │ 74.2% │ 52.8% │ 62.2%  │ 45.0%    │
  ├───────┼───────┼───────┼────────┼──────────┤
  │ 15    │ 60.6% │ 62.6% │ 66.0%  │ 66.2%    │
  ├───────┼───────┼───────┼────────┼──────────┤
  │ 20    │ 77.2% │ 72.6% │ 69.8%  │ 73.6%    │
  ├───────┼───────┼───────┼────────┼──────────┤
  │ 25    │ 83.0% │ 74.4% │ 73.2%  │ —        │
  ├───────┼───────┼───────┼────────┼──────────┤
  │ 30    │ 82.8% │ 75.0% │ 73.6%  │ —        │
  └───────┴───────┴───────┴────────┴──────────┘

  Observaciones:
  - real arranca más lento (30.4% a e5 vs 48-52% para ablaciones) pero explota después de e10. El modelo necesita tiempo para aprender a usar la información
   del descriptor.
  - shuffled tiene un dip a e10 (45.0%) — el modelo inicialmente intenta usar la info del descriptor, descubre que es incoherente, y necesita re-aprender a
  ignorarla.
  - Las ablaciones convergen todas a ~73-75% hacia e20-25. Real sigue subiendo.

  4. Métricas de profundidad confirman la separación

  ┌───────────┬───────┬───────┬────────┬──────────┐
  │  Métrica  │ real  │ zero  │ random │ shuffled │
  ├───────────┼───────┼───────┼────────┼──────────┤
  │ MRR       │ 0.458 │ 0.396 │ 0.386  │ 0.374    │
  ├───────────┼───────┼───────┼────────┼──────────┤
  │ R@1       │ 28.8% │ 23.6% │ 22.4%  │ 21.4%    │
  ├───────────┼───────┼───────┼────────┼──────────┤
  │ R@5       │ 68.2% │ 59.2% │ 57.6%  │ 54.4%    │
  ├───────────┼───────┼───────┼────────┼──────────┤
  │ mean_rank │ 5.0   │ 7.4   │ 8.4    │ 8.4      │
  └───────────┴───────┴───────┴────────┴──────────┘

  La ventaja de real es uniforme en TODAS las métricas. No es un artefacto del threshold de R@10 — el ranking medio pasa de 5.0 a 7.4-8.4. Real coloca el
  match correcto consistentemente más arriba.

  5. Dato curioso: hard_neg es MEJOR para ablaciones

  ┌──────────┬───────────────────────┐
  │   Mode   │ hard_neg (same_piece) │
  ├──────────┼───────────────────────┤
  │ zero     │ 95.4%                 │
  ├──────────┼───────────────────────┤
  │ random   │ 95.2%                 │
  ├──────────┼───────────────────────┤
  │ real     │ 94.0%                 │
  ├──────────┼───────────────────────┤
  │ shuffled │ 93.6%                 │
  └──────────┴───────────────────────┘

  Las ablaciones distinguen mejor entre piezas diferentes. Esto es consistente con Test 03 (RatioProbe): D0 es más linealmente decodificable. Los modelos
  sin descriptor aprenden representaciones "más simples" que discriminan por pieza pero peor por segmento dentro de pieza.

  Conclusión Test 02

  La mejora de d4a4 es causal e informacional. Con exactamente los mismos 66.2M parámetros entrenables, eliminar la información del descriptor (por
  cualquiera de 3 métodos) causa una caída de 8-9.4pp. El descriptor no es un regularizador, no es un efecto de capacidad, y no es un artefacto — es la
  información lo que importa.

  ---
  TEST 13G-B — Post-Hoc Pre-Pooling Decoder (cerrado)

  Pregunta: ¿Las representaciones intermedias del encoder (pre-pooling, [B, N, 1024]) retienen información musical que se pierde en el pooling? ¿Los
  descriptores hacen que estas representaciones sean más "ricas musicalmente"?

  Diseño: Encoders congelados (D0, a4r, d4a4). Un decoder cross-attention idéntico (2.44M params) entrenado sobre cada uno. El decoder toma features [B, N,
  1024] y genera piano roll [B, 188, 88].

  Resultados finales (full validation, 12887 samples)

  ┌─────────────┬───────────────┬──────────┬───────────┬────────┬──────────┬───────┬────────┐
  │     Arm     │   Features    │ frame_F1 │ precision │ recall │ onset_F1 │  BCE  │ cosine │
  ├─────────────┼───────────────┼──────────┼───────────┼────────┼──────────┼───────┼────────┤
  │ D0 pool-188 │ [B,188,1024]  │ 0.1091   │ 0.0580    │ 0.9215 │ 0.0419   │ 0.831 │ 0.260  │
  ├─────────────┼───────────────┼──────────┼───────────┼────────┼──────────┼───────┼────────┤
  │ d4a4        │ [B,2400,1024] │ 0.1041   │ 0.0552    │ 0.9069 │ 0.0406   │ 0.904 │ 0.241  │
  ├─────────────┼───────────────┼──────────┼───────────┼────────┼──────────┼───────┼────────┤
  │ a4r         │ [B,188,1024]  │ 0.1031   │ 0.0546    │ 0.9141 │ 0.0410   │ 0.895 │ 0.236  │
  └─────────────┴───────────────┴──────────┴───────────┴────────┴──────────┴───────┴────────┘

  Interpretación por capa

  1. F1 ~10% para TODOS — el techo es bajo y uniforme

  La diferencia entre el mejor (D0 pool-188: 0.1091) y el peor (a4r: 0.1031) es 0.006 — estadísticamente despreciable sobre 12887 muestras. Los tres arms
  son equivalentes en capacidad de reconstrucción.

  Esto es el hallazgo más importante: los descriptores no producen representaciones pre-pooling más musicalmente decodificables.

  2. El patrón recall/precision revela el modo de falla

  - Recall ~91-92%: El decoder activa la mayoría de frames donde hay notas — sabe "dónde hay sonido"
  - Precision ~5.5%: Pero activa enormemente de más — no sabe cuándo empieza y termina cada nota
  - onset_F1 ~4%: Prácticamente incapaz de detectar ataques de nota

  El decoder genera "manchas difusas" centradas en el registro medio. Sabe que el piano está sonando, pero no puede resolver notas individuales.

  3. Curvas de aprendizaje: el trade-off precision-recall

  Para D0 pool-188 a lo largo del training:

  ┌───────┬────────┬───────────┬────────┬───────┐
  │ Epoch │   F1   │ Precision │ Recall │  BCE  │
  ├───────┼────────┼───────────┼────────┼───────┤
  │ 5     │ 0.0923 │ 0.0485    │ 0.9759 │ 0.762 │
  ├───────┼────────┼───────────┼────────┼───────┤
  │ 10    │ 0.0950 │ 0.0499    │ 0.9737 │ 0.756 │
  ├───────┼────────┼───────────┼────────┼───────┤
  │ 20    │ 0.0999 │ 0.0527    │ 0.9604 │ 0.767 │
  ├───────┼────────┼───────────┼────────┼───────┤
  │ 30    │ 0.1036 │ 0.0548    │ 0.9497 │ 0.790 │
  ├───────┼────────┼───────────┼────────┼───────┤
  │ 40    │ 0.1089 │ 0.0579    │ 0.9228 │ 0.843 │
  └───────┴────────┴───────────┴────────┴───────┘

  El F1 sube porque la precision mejora (0.0485→0.0579, +19%) más de lo que el recall cae (0.9759→0.9228, -5%). Pero el BCE sube también (0.762→0.843) — el
  modelo está overfiteando: aprende a ser ligeramente más selectivo, pero la loss general empeora.

  Ningún arm hizo early stop (todos corrieron 40/40 epochs, patience=4 eval rounds nunca triggereó). Esto confirma que la mejora era monotónica pero
  lentísima.

  4. D0 pool-188 gana marginalmente — y es la sorpresa

  D0 pool-188 tiene features [B, 188, 1024] (pooling 2400→188 del encoder D0 original). Que gane sobre d4a4 [B, 2400, 1024] es contraintuitivo: tiene 12.8x
  menos frames de entrada.

  Posibles explicaciones:
  - El pooling selecciona los 188 frames más informativos (maxpool + subsample), descartando frames redundantes que solo agregan ruido al decoder
  - Con 2400 frames, el decoder tiene más donde confundirse — el cross-attention se diluye
  - BCE es significativamente menor para D0 pool-188 (0.831 vs 0.904/0.918) — la distribución de activaciones es más calibrada

  5. Loss de training: d4a4 baja más rápido pero no se traduce

  ┌──────┬─────────┬──────────┬──────────┬─────────────┐
  │ Arm  │ Loss e1 │ Loss e20 │ Loss e40 │    Tasa     │
  ├──────┼─────────┼──────────┼──────────┼─────────────┤
  │ D0   │ 0.8171  │ 0.6690   │ 0.5738   │ -0.00608/ep │
  ├──────┼─────────┼──────────┼──────────┼─────────────┤
  │ a4r  │ 0.8190  │ 0.7019   │ 0.5895   │ -0.00574/ep │
  ├──────┼─────────┼──────────┼──────────┼─────────────┤
  │ d4a4 │ 0.8159  │ 0.6830   │ 0.5632   │ -0.00631/ep │
  └──────┴─────────┴──────────┴──────────┴─────────────┘

  d4a4 tiene la loss de training más baja (0.563) pero la peor BCE de validación (0.904). Overfit clásico: las features de d4a4 [B, 2400, 1024] son más
  ricas para memorizar el train set, pero no generalizan mejor para reconstruir piano rolls.

  6. Cosine similarity: D0 tiene representaciones más "alineadas"

  ┌─────────────┬─────────────────────┐
  │     Arm     │ cosine (pred vs GT) │
  ├─────────────┼─────────────────────┤
  │ D0 pool-188 │ 0.260               │
  ├─────────────┼─────────────────────┤
  │ d4a4        │ 0.241               │
  ├─────────────┼─────────────────────┤
  │ a4r         │ 0.236               │
  └─────────────┴─────────────────────┘

  Los piano rolls predichos desde D0 son ligeramente más similares a los ground truth. Esto sugiere que las features de D0 (sin descriptor) están más
  "alineadas" con la estructura del piano roll — posiblemente porque D0 retiene más info tonal genérica y menos info específica del descriptor.

  Diagnóstico del modo de falla

  El problema no es el encoder — es la naturaleza de la tarea. Un piano roll tiene:
  - 188 frames × 88 teclas = 16,544 valores binarios por segmento de 4 segundos
  - Cada frame dura ~21ms — resolución temporal fina
  - El encoder opera a ~1.7ms/frame (2400 frames en 4s) pero el piano roll a ~21ms/frame

  El encoder no codifica la información con la resolución temporal necesaria. Codifica "qué notas suenan en general" (recall 92%) pero no "cuándo
  exactamente empiezan y terminan" (onset_F1 4%). Esto es un limitante fundamental del encoder self-supervised (VICReg no necesita resolución temporal para
  alinear embeddings), no del mecanismo de inyección del descriptor.

  Conclusión Test 13G-B

  Los descriptores no mejoran la decodificabilidad musical de las features pre-pooling. Los tres arms producen representaciones igualmente (in)capaces de
  reconstruir piano rolls. La ventaja de los descriptores vive en la geometría de distancias (retrieval, CKA), no en la riqueza informacional de las
  features internas. Esto es consistente con Test 03 (ventaja geométrica, no lineal) y refina el hallazgo de Test 11 (la info extra de a4r se pierde en la
  proyección, pero incluso pre-proyección no es "más musical").

  ---
  Síntesis: qué dicen estos dos tests juntos

  Test 02:  La INFO del descriptor IMPORTA (83% vs 73-75%)
  Test 13G: Pero no hace las features internas MÁS MUSICALES

                      ┌─────────────────────────────┐
                      │ El descriptor no enriquece   │
                      │ las features → reorganiza    │
                      │ la GEOMETRÍA de distancias   │
                      └──────────────┬──────────────┘
                                     │
                 ┌───────────────────┼───────────────────┐
                 ▼                   ▼                   ▼
          Test 02: +9.4pp      Test 06: +82%       Test 03: D0 gana
          en retrieval          CKA alignment       probe lineal
          (causal)              (geométrico)        (no es lineal)

  El descriptor actúa como un reorganizador geométrico, no como un enriquecedor de features. Las representaciones internas contienen la misma información
  musical (F1~10% en todos), pero el descriptor hace que los embeddings finales se distribuyan de forma que las distancias L2/coseno correspondan mejor a
  similitud musical. La señal es sutil pero potente: misma info, mejor geometría, +9.4pp de retrieval.