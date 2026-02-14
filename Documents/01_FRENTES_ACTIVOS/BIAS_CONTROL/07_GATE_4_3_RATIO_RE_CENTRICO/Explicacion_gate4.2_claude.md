● Métricas de monitoreo (durante training)                                                                                                 
                                                                                                                                           
  Estas las vemos en la progress bar de tqdm y en los logs por epoch:                                                                      
   
  loss — VICReg loss total (inv + var + cov). Valor típico ~13-15. Más bajo = mejor alineamiento entre embeddings audio-MIDI. No es        
  directamente comparable entre descriptors distintos porque la inyección cambia la escala.

  std_z1 — Desviación estándar de la primera dimensión del embedding. Si colapsa a ~0 significa que el modelo produce embeddings constantes
   (colapso representacional). Valores sanos: 0.3-0.5. Es un indicador de salud, no de performance.

  lr0 — Learning rate del primer grupo de parámetros (audio layers 0-1). Sirve para saber dónde estamos en el cosine schedule.

  quick[A2M, M2A] — Retrieval rápido sobre un subset del validation set, sin pool estructurado. Números bajos (~8-11%) porque usa batches
  crudos sin control de dificultad. Sirve solo como proxy rápido de tendencia, no como métrica absoluta.

  ---
  Métricas de evaluación canónica (checkpoints)

  Estas se calculan con evaluate_structured_pool.py al final de cada epoch. Usan pool=256 segmentos, 500 queries, seed=42, con 64 hard
  negatives y 32 semi-hard negatives por query.

  Retrieval R@K (Recall at K)

  Para cada query (un segmento de audio o MIDI), buscamos su par correcto entre 256 candidatos. R@K = fracción de queries donde el par
  correcto aparece en el top-K.

  A2M R@10 — Dado un audio, buscar su MIDI correcto entre 256. Porcentaje de veces que aparece en el top-10. Baseline random = 10/256 ≈
  3.9%.

  M2A R@10 — Dado un MIDI, buscar su audio correcto entre 256. Misma lógica.

  S = min(A2M R@10, M2A R@10) — Métrica canónica. Toma el peor de los dos sentidos. Un modelo que solo funciona en una dirección no pasa.

  R@1 — Top-1 accuracy. Mucho más exigente: el par correcto tiene que ser el #1 del ranking. Baseline random = 0.39%.

  R@5 — Top-5. Punto intermedio.

  R@20 — Top-20. Más permisivo, captura si el modelo "anda cerca" aunque no precise.

  MRR (Mean Reciprocal Rank)

  Para cada query, si el par correcto aparece en posición k, su score es 1/k. MRR = promedio sobre las 500 queries.

  MRR_A (A2M) y MRR_M (M2A) — MRR por dirección.

  MRR_avg = (MRR_A + MRR_M) / 2 — Promedio de ambas direcciones.

  Ventaja sobre R@10: es sensible a dónde dentro del ranking cae el resultado. Un modelo que pone el correcto en posición 2 vs posición 9
  tiene el mismo R@10 pero MRR muy distinto (0.5 vs 0.11).

  MRR = 0.313 significa que en promedio el par correcto está en posición ~3.2 (1/0.313).

  Mean Rank (MR)

  MR_A y MR_M — Posición promedio del par correcto en el ranking (de 256). Más bajo = mejor.

  MR=11 significa que el par correcto en promedio está en posición 11 de 256. Baseline random = 128.5.

  Hard Negative Accuracy

  hard_neg — Dado un segmento, distinguir si otro segmento es del mismo momento (par correcto) vs otro momento de la misma pieza. Esto es
  difícil porque dos momentos de la misma pieza comparten timbre, tonalidad, estilo.

  91% = el modelo distingue segmentos temporalmente específicos, no solo "suena como Beethoven".

  ---
  Qué mirar para qué decisión

  ┌─────────────────────────────────────────┬─────────────────────────────┬──────────────────┐
  │                Pregunta                 │      Métrica principal      │      Apoyo       │
  ├─────────────────────────────────────────┼─────────────────────────────┼──────────────────┤
  │ ¿El descriptor aporta?                  │ S (Δ vs D0)                 │ MRR_avg, R@1     │
  ├─────────────────────────────────────────┼─────────────────────────────┼──────────────────┤
  │ ¿Sigue mejorando?                       │ MRR_avg epoch a epoch       │ S, MR            │
  ├─────────────────────────────────────────┼─────────────────────────────┼──────────────────┤
  │ ¿Está colapsando?                       │ std_z1, loss                │ S cayendo        │
  ├─────────────────────────────────────────┼─────────────────────────────┼──────────────────┤
  │ ¿Qué dirección es el cuello de botella? │ A2M vs M2A (R@10, MRR)      │ -                │
  ├─────────────────────────────────────────┼─────────────────────────────┼──────────────────┤
  │ ¿Es temporalmente específico?           │ hard_neg                    │ -                │
  ├─────────────────────────────────────────┼─────────────────────────────┼──────────────────┤
  │ ¿Ranking fino o grueso?                 │ MRR (fino) vs R@10 (grueso) │ R@1 (ultra-fino) │
  └─────────────────────────────────────────┴─────────────────────────────┴──────────────────┘