│ Plan: Gate 7.1 — MERT-330M Frozen Cross-Modal Probe (v2)                                                                                                 │
│                                                                                                                                                          │
│ Contexto                                                                                                                                                 │
│                                                                                                                                                          │
│ Gate 5B cerró con el hallazgo "ventaja geométrica, no de feature richness": A4 mejora retrieval causalmente (+9.4pp, Test 02) reorganizando la geometría │
│  del espacio (+82% CKA), sin enriquecer features individuales (Test 13G-B: ranking invertido). Gate 7 Exp 7.0 mostró que MERT-330M tiene R²=0.850 para   │
│ la envolvente espectral por bandas A4 (vs MERTLite=0.734).                                                                                               │
│                                                                                                                                                          │
│ Pregunta central: ¿La ventaja de A4 persiste cuando el encoder audio ya es fuerte, o estaba compensando la debilidad de MERTLite?                        │
│                                                                                                                                                          │
│ Guardrails metodológicos (de Codex)                                                                                                                      │
│                                                                                                                                                          │
│ 1. No sobredicho: Gate 7 mostró accesibilidad lineal de la envolvente espectral estática por banda — no clausura qué parte del A4 operativo (deltas      │
│ temporales + normalización interna) está internalizado en MERT-330M.                                                                                     │
│ 2. Confundidores explícitos: Cambian simultáneamente backbone, co-adaptación y pretraining. Es un piloto decisional, no aislamiento causal puro.         │
│ 3. Sin umbral mágico: ΔA4 comparado con Gate 5B, no números de corte inventados.                                                                         │
│ 4. Framing: 2 seeds = pilot/go-no-go, no base de claim fuerte.                                                                                           │
│                                                                                                                                                          │
│ Por qué dos fases (corrección principal de Codex)                                                                                                        │
│                                                                                                                                                          │
│ El plan v1 sobredecía la reutilización de infraestructura. Verificación de código confirmó:                                                              │
│                                                                                                                                                          │
│ ┌────────────────────────┬────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│ │         Issue          │ Severidad  │                                                    Detalle                                                    │  │
│ ├────────────────────────┼────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────┤  │
│ │ a4r NO es              │            │ _encode_audio_with_reverse_cross_attention() (L1330-1375) accede a enc.feature_extractor, enc.pos_embedding,  │  │
│ │ plug-compatible con    │ ALTA       │ enc.transformer — atributos de MERTEncoderLite que MERTEncoder NO expone. MERTEncoder encapsula HF model en   │  │
│ │ MERTEncoder            │            │ _model opaco y devuelve embeddings ya agregados.                                                              │  │
│ ├────────────────────────┼────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────┤  │
│ │ Training stack         │ ALTA       │ --from-scratch hardcodea CrossModalModel(audio_encoder='lite') (L3715). apply_freeze_policy() (L2611-2621) y  │  │
│ │ cableado a Lite        │            │ create_gate42_optimizer() (L2654-2678) acceden a .feature_extractor, .transformer.layers[0..3] directamente.  │  │
│ ├────────────────────────┼────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────┤  │
│ │ model.train() leak     │ ALTA       │ MERTEncoder pone _model.eval() al cargar (L84), pero el training loop llama model.train() cada época (L3410), │  │
│ │                        │            │  reactivando dropout en el encoder "congelado".                                                               │  │
│ ├────────────────────────┼────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────┤  │
│ │ Costo no validado      │ MEDIA      │ MERTEncoder.preprocess() procesa sample-by-sample via CPU .numpy() (L117-131). Con 1000 batches/epoch × 30    │  │
│ │                        │            │ epochs, throughput desconocido.                                                                               │  │
│ ├────────────────────────┼────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────┤  │
│ │ Seeds secuenciales     │ MEDIA-BAJA │ Fallo en seed 1 pierde horas y no diagnostica si fue run o job.                                               │  │
│ │ riesgosas              │            │                                                                                                               │  │
│ └────────────────────────┴────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                                                                          │
│ Conclusión: D0 con MERTEncoder sí es viable — Gate42Model.forward() (L348) solo llama base_model(audio, ...) → CrossModalModel.forward() →               │
│ self.audio_encoder(audio), sin tocar internals. Pero a4r requiere trabajo nuevo para adaptar el mecanismo reverse cross-attention a la topología de      │
│ MERTEncoder.                                                                                                                                             │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ Diseño bifásico                                                                                                                                          │
│                                                                                                                                                          │
│ Gate 7.1a — D0 Pilot (infraestructura + baseline fuerte)                                                                                                 │
│                                                                                                                                                          │
│ Pregunta: ¿Sube S(D0) con MERT-330M frozen vs MERTLite (75.2%)?                                                                                          │
│                                                                                                                                                          │
│ Arms: Solo D0 (sin descriptor). 1 seed (42).                                                                                                             │
│                                                                                                                                                          │
│ Objetivos:                                                                                                                                               │
│ 1. Validar que VICReg cross-modal funciona con encoder frozen (anti-ghost + no-leak)                                                                     │
│ 2. Benchmark real de throughput (batches/min) para estimar costo UNC                                                                                     │
│ 3. Obtener S(D0_mert330m) como baseline fuerte                                                                                                           │
│                                                                                                                                                          │
│ Protocolo:                                                                                                                                               │
│ - Audio encoder: CrossModalModel(audio_encoder='mert', audio_encoder_frozen=True)                                                                        │
│ - MIDI encoder + proyecciones: from scratch                                                                                                              │
│ - VICReg (inv=10, var=10, cov=1)                                                                                                                         │
│ - 30 epochs, batch_size=8, max_batches_per_epoch=1000                                                                                                    │
│ - LR: midi=5e-5, proj=1e-4. Audio encoder excluido del optimizer.                                                                                        │
│ - Eval canónica: pool=256, queries=500, seed=42                                                                                                          │
│ - Eval estructurada: epochs [5, 10, 15, 20, 25, 28, 29, 30]                                                                                              │
│                                                                                                                                                          │
│ Go/No-Go para 7.1b (criterio pragmático, no threshold científico):                                                                                       │
│ - GO: curva de S muestra aprendizaje monotónico hasta epoch 10 (S crece epoch-over-epoch) Y throughput viable para UNC (~30ep en < 36h)                  │
│ - NO-GO: loss diverge O S no sube de random (50%) después de 10 epochs O throughput inviable (> 48h para 30ep)                                           │
│ - En zona ambigua: continuar hasta epoch 15 antes de decidir. Un arranque lento con backbone congelado nuevo no es automáticamente no-go.                │
│                                                                                                                                                          │
│ Gate 7.1b — a4r-MERT (variante nueva)                                                                                                                    │
│                                                                                                                                                          │
│ Solo después de que 7.1a valide infraestructura.                                                                                                         │
│                                                                                                                                                          │
│ Pregunta: ΔA4 = S(a4r) − S(D0) comparable a +5.5pp de Gate 5B?                                                                                           │
│                                                                                                                                                          │
│ Trabajo técnico necesario (el que v1 subestimaba):                                                                                                       │
│                                                                                                                                                          │
│ El a4r actual (_encode_audio_with_reverse_cross_attention, L1310-1379) hace:                                                                             │
│ CNN features [B, 2400, 1024]  ← enc.feature_extractor(waveform)                                                                                          │
│ + pos_embedding                ← enc.pos_embedding[:, :T, :]                                                                                             │
│ descriptor Q [B, 188, 1024]   ← q_proj(A4_descriptor) + desc_pos_embedding                                                                               │
│ cross_attn(Q=desc, K/V=features)                                                                                                                         │
│ → Transformer(188 tokens)     ← enc.transformer(desc_proj)                                                                                               │
│ → mean pool → projection                                                                                                                                 │
│                                                                                                                                                          │
│ Para MERT-330M, la adaptación semántica más limpia es:                                                                                                   │
│ HF hidden_states [B, T, 1024] ← self._model(...).hidden_states[-1]  (pre-agregación)                                                                     │
│ descriptor Q [B, 188, 1024]   ← q_proj(A4_descriptor) + desc_pos_embedding                                                                               │
│ cross_attn(Q=desc, K/V=hidden_states)                                                                                                                    │
│ → NEW lightweight transformer (2-4 layers, 188 tokens)                                                                                                   │
│ → mean pool → projection                                                                                                                                 │
│                                                                                                                                                          │
│ Diferencias clave vs a4r-Lite:                                                                                                                           │
│ - K/V = last hidden state de MERT-330M (ya procesado por 24 transformer layers), no CNN features crudas                                                  │
│ - No reutiliza enc.transformer (MERTEncoder no lo expone) → necesita transformer propio NUEVO (2-4 capas, ligero)                                        │
│ - MERT-330M forward devuelve hidden_states si se pide output_hidden_states=True (ya implementado, L166-172)                                              │
│                                                                                                                                                          │
│ Esto es una variante nueva, no un swap de flag. Requiere:                                                                                                │
│ 1. Función _encode_audio_mert330m_reverse_crossatt() nueva                                                                                               │
│ 2. Clase Gate71MERTReverseCrossAttModel nueva (o extensión de Gate42)                                                                                    │
│ 3. MERTEncoder.forward() que retorne hidden_states PRE-agregación (añadir return_sequence=True)                                                          │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ Archivos a crear/modificar                                                                                                                               │
│                                                                                                                                                          │
│ Gate 7.1a (D0 pilot)                                                                                                                                     │
│                                                                                                                                                          │
│ experiments/bias_control/gate71/                                                                                                                         │
│ ├── __init__.py                                                                                                                                          │
│ └── train_gate71.py               # Training script (D0 primero, a4r después)                                                                            │
│                                                                                                                                                          │
│ slurm/gate71_d0.sh                # SLURM: D0, seed 42 (1 job, 1 seed)                                                                                   │
│                                                                                                                                                          │
│ Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/14_GATE_7.1/                                                                                                   │
│ └── README.md                                                                                                                                            │
│                                                                                                                                                          │
│ Modificar:                                                                                                                                               │
│ - src/bias_control/encoders/mert_encoder.py — fix model.train() leak (override train() method)                                                           │
│ - experiments/bias_control/gate5b/checkpoint_loader.py — extender para audio_encoder='mert' + a4r-mert                                                   │
│                                                                                                                                                          │
│ Gate 7.1b (a4r-MERT, solo si 7.1a GO)                                                                                                                    │
│                                                                                                                                                          │
│ Modificar:                                                                                                                                               │
│ - src/bias_control/encoders/mert_encoder.py — añadir return_sequence a forward()                                                                         │
│ - experiments/bias_control/gate71/train_gate71.py — añadir a4r-mert arm                                                                                  │
│                                                                                                                                                          │
│ Nuevo código en train_gate71.py:                                                                                                                         │
│ - Función _encode_audio_mert330m_reverse_crossatt() — K/V = hidden_states pre-pool                                                                       │
│ - Clase Gate71MERTReverseCrossAttModel — wraps CrossModalModel + nuevo transformer ligero                                                                │
│                                                                                                                                                          │
│ slurm/gate71_a4r.sh               # SLURM: a4r-mert, seed 42                                                                                             │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ train_gate71.py — diseño detallado                                                                                                                       │
│                                                                                                                                                          │
│ Gate 7.1a (D0)                                                                                                                                           │
│                                                                                                                                                          │
│ # Modelo D0: usa CrossModalModel directamente, sin tocar internals                                                                                       │
│ base_model = CrossModalModel(audio_encoder='mert', audio_encoder_frozen=True)                                                                            │
│ model = Gate42Model(base_model)  # D0: solo forward + VICReg                                                                                             │
│                                                                                                                                                          │
│ # Optimizer: EXCLUIR audio encoder completamente                                                                                                         │
│ param_groups = [                                                                                                                                         │
│     {'params': list(base_model.midi_encoder.parameters()), 'lr': 5e-5, 'name': 'midi'},                                                                  │
│     {'params': list(base_model.audio_projection.parameters()) +                                                                                          │
│                list(base_model.midi_projection.parameters()), 'lr': 1e-4, 'name': 'proj'},                                                               │
│ ]                                                                                                                                                        │
│ optimizer = AdamW(param_groups)                                                                                                                          │
│                                                                                                                                                          │
│ Fix model.train() leak — en MERTEncoder, override train():                                                                                               │
│ def train(self, mode=True):                                                                                                                              │
│     """Override to keep frozen model in eval mode always."""                                                                                             │
│     super().train(mode)                                                                                                                                  │
│     if self.freeze and self._loaded and self._model is not None:                                                                                         │
│         self._model.eval()                                                                                                                               │
│     return self                                                                                                                                          │
│                                                                                                                                                          │
│ Forzar carga de MERT antes de anti-ghost (lazy load workaround):                                                                                         │
│ # MERTEncoder carga lazy (_model=None hasta primer forward, L52/L61).                                                                                    │
│ # Si no se fuerza carga, .parameters() devuelve 0 y todos los checks pasan vacíos.                                                                       │
│ base_model.audio_encoder._load_model()  # Forzar carga explícita                                                                                         │
│ assert base_model.audio_encoder._loaded, "MERT model failed to load"                                                                                     │
│                                                                                                                                                          │
│ Anti-ghost checks (obligatorios, DESPUÉS de _load_model()):                                                                                              │
│ 1. trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) — debe ser ~15M                                                      │
│ 2. frozen_params = sum(p.numel() for p in base_model.audio_encoder._model.parameters()) — debe ser ~330M                                                 │
│ 3. Snapshot w0 = {n: p.clone() for n, p in model.named_parameters() if 'audio_encoder' in n}                                                             │
│ 4. Post epoch 1: assert all(torch.equal(w0[n], p) for n, p in ... if 'audio_encoder' in n)                                                               │
│ 5. Post model.train(): assert base_model.audio_encoder._model.training == False                                                                          │
│                                                                                                                                                          │
│ Throughput benchmark (batches 5-25, excluyendo warmup):                                                                                                  │
│ # Primeros 4 batches = warmup (lazy load, JIT, cache). Medir desde batch 5.                                                                              │
│ # Log: batches/min (estable), GPU mem peak, time breakdown (preprocess vs forward vs backward)                                                           │
│ # Extrapolar: time_per_epoch = (1000 * time_per_batch) → time_30ep → estimación UNC                                                                      │
│                                                                                                                                                          │
│ Gate 7.1b (a4r-MERT, solo si GO)                                                                                                                         │
│                                                                                                                                                          │
│ # Nuevo: MERTEncoder con return_sequence                                                                                                                 │
│ # En mert_encoder.py, forward() ya tiene hidden_states (L172)                                                                                            │
│ # Añadir parámetro return_sequence=False → si True, devuelve [B, T, 1024] sin pool                                                                       │
│                                                                                                                                                          │
│ # Nueva función de encoding:                                                                                                                             │
│ def _encode_audio_mert330m_reverse_crossatt(                                                                                                             │
│     base_model, audio, descriptor_type,                                                                                                                  │
│     descriptor_q_proj, desc_pos_embedding,                                                                                                               │
│     cross_attention, cross_attn_norm,                                                                                                                    │
│     light_transformer,  # NUEVO: 2-4 layers, propio                                                                                                      │
│ ):                                                                                                                                                       │
│     # 1. MERT-330M hidden states PRE-pool [B, T, 1024]                                                                                                   │
│     hidden_states = base_model.audio_encoder(audio, return_sequence=True)                                                                                │
│                                                                                                                                                          │
│     # 2. Descriptor Q [B, ~188, 1024]                                                                                                                    │
│     with torch.no_grad():                                                                                                                                │
│         desc = compute_audio_descriptor_a4(audio, target_length=None)                                                                                    │
│     desc_proj = descriptor_q_proj(desc.detach())                                                                                                         │
│     desc_proj = desc_proj + desc_pos_embedding[:, :desc_proj.size(1), :]                                                                                 │
│                                                                                                                                                          │
│     # 3. Reverse cross-attention: Q=desc, K/V=hidden_states                                                                                              │
│     attn_out, _ = cross_attention(query=desc_proj, key=hidden_states, value=hidden_states)                                                               │
│     desc_proj = cross_attn_norm(desc_proj + attn_out)                                                                                                    │
│                                                                                                                                                          │
│     # 4. Light transformer (PROPIO, no enc.transformer) → 188 tokens                                                                                     │
│     encoded = light_transformer(desc_proj)                                                                                                               │
│     embeddings = encoded.mean(dim=1)                                                                                                                     │
│                                                                                                                                                          │
│     # 5. Projection                                                                                                                                      │
│     return base_model.audio_projection(embeddings)                                                                                                       │
│                                                                                                                                                          │
│ Parámetros nuevos trainables: ~4M (q_proj + desc_pos + cross_attn + light_transformer 2 layers + LN)                                                     │
│ Total trainable: ~15M (midi) + ~4M (a4r modules) + ~1.3M (projs) ≈ 20M                                                                                   │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ SLURM                                                                                                                                                    │
│                                                                                                                                                          │
│ Gate 7.1a — 1 job, 1 seed                                                                                                                                │
│                                                                                                                                                          │
│ #SBATCH --job-name=g71a_d0                                                                                                                               │
│ #SBATCH --partition=multi                                                                                                                                │
│ #SBATCH --gres=gpu:1                                                                                                                                     │
│ #SBATCH --mem=40G                                                                                                                                        │
│ #SBATCH --time=1-12:00:00                                                                                                                                │
│ #SBATCH --cpus-per-task=8                                                                                                                                │
│                                                                                                                                                          │
│ srun python experiments/bias_control/gate71/train_gate71.py \                                                                                            │
│     --descriptor d0 \                                                                                                                                    │
│     --epochs 30 --batch-size 8 --seed 42 \                                                                                                               │
│     --maestro-dir $SCRATCH/maestro-v3.0.0 \                                                                                                              │
│     --output $OUTDIR/d0_mert330m_seed42 \                                                                                                                │
│     --structured-eval-epochs 5 10 15 20 25 28 29 30                                                                                                      │
│                                                                                                                                                          │
│ Gate 7.1b — 1 job, 1 seed (solo si 7.1a GO)                                                                                                              │
│                                                                                                                                                          │
│ #SBATCH --job-name=g71b_a4r                                                                                                                              │
│ # ... mismos recursos ...                                                                                                                                │
│                                                                                                                                                          │
│ srun python experiments/bias_control/gate71/train_gate71.py \                                                                                            │
│     --descriptor a4r-mert \                                                                                                                              │
│     --epochs 30 --batch-size 8 --seed 42 \                                                                                                               │
│     --maestro-dir $SCRATCH/maestro-v3.0.0 \                                                                                                              │
│     --output $OUTDIR/a4r_mert330m_seed42 \                                                                                                               │
│     --structured-eval-epochs 5 10 15 20 25 28 29 30                                                                                                      │
│                                                                                                                                                          │
│ 1 seed cada uno (no 2 secuenciales). Si los resultados son interesantes, segundo seed en job separado.                                                   │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ Marco de lectura                                                                                                                                         │
│                                                                                                                                                          │
│ ┌───────────────┬────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────┐      │
│ │    Outcome    │           Señal            │                                              Lectura                                               │      │
│ ├───────────────┼────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤      │
│ │ A             │ D0_strong ≈ D0_lite (75%)  │ Encoder frozen no escala VICReg. Congelación es el cuello de botella, no la capacidad.             │      │
│ ├───────────────┼────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤      │
│ │ B             │ D0_strong >> 75% Y ΔA4 → 0 │ A4 compensaba debilidad del encoder. Escalón 2 no necesita descriptores si el encoder es fuerte.   │      │
│ ├───────────────┼────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤      │
│ │ C             │ D0_strong >> 75% Y ΔA4 > 0 │ Tesis geométrica robusta: A4 aporta incluso con encoder fuerte. Diseñar descriptor para Escalón 2. │      │
│ ├───────────────┼────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤      │
│ │ No conclusivo │ D0_strong < D0_lite        │ Congelación total rompe dinámica VICReg. No concluir, documentar.                                  │      │
│ └───────────────┴────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────┘      │
│                                                                                                                                                          │
│ Métrica primaria: ΔA4 = S(a4r_mert) − S(D0_mert) vs +5.5pp de Gate 5B.                                                                                   │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ Orden de ejecución                                                                                                                                       │
│                                                                                                                                                          │
│ Fase 1 — Implementación 7.1a (~3h LOCAL)                                                                                                                 │
│                                                                                                                                                          │
│ 1. Fix MERTEncoder.train() leak en mert_encoder.py                                                                                                       │
│ 2. Crear experiments/bias_control/gate71/train_gate71.py con D0 support                                                                                  │
│ 3. Anti-ghost checks + throughput benchmark integrados                                                                                                   │
│ 4. Test LOCAL: 5 batches D0 con MERT-330M → shapes ok, anti-ghost pass, throughput medido                                                                │
│ 5. Crear slurm/gate71_d0.sh, validar con /validate-sbatch                                                                                                │
│ 6. Crear Documents/.../14_GATE_7.1/README.md                                                                                                             │
│ 7. Push main                                                                                                                                             │
│                                                                                                                                                          │
│ Fase 2 — Ejecución 7.1a (~24-36h UNC)                                                                                                                    │
│                                                                                                                                                          │
│ 1. UNC pull + submit job                                                                                                                                 │
│ 2. Monitoreo                                                                                                                                             │
│                                                                                                                                                          │
│ Fase 3 — Decisión GO/NO-GO 7.1b                                                                                                                          │
│                                                                                                                                                          │
│ - Revisar curva S(D0) epochs 5-10: aprendizaje monotónico + throughput viable → GO                                                                       │
│ - Si ambiguo, esperar hasta epoch 15                                                                                                                     │
│ - Documentar throughput real (batches/min estable) para estimar costo 7.1b                                                                               │
│                                                                                                                                                          │
│ Fase 4 — Implementación 7.1b (solo si GO, ~4h LOCAL)                                                                                                     │
│                                                                                                                                                          │
│ 1. Añadir return_sequence a MERTEncoder.forward()                                                                                                        │
│ 2. Implementar _encode_audio_mert330m_reverse_crossatt() + Gate71MERTReverseCrossAttModel                                                                │
│ 3. Añadir --descriptor a4r-mert al script                                                                                                                │
│ 4. Test LOCAL: 5 batches a4r-mert → shapes ok, anti-ghost pass                                                                                           │
│ 5. Crear slurm/gate71_a4r.sh                                                                                                                             │
│ 6. Push main                                                                                                                                             │
│                                                                                                                                                          │
│ Fase 5 — Ejecución 7.1b + Análisis                                                                                                                       │
│                                                                                                                                                          │
│ 1. UNC pull + submit                                                                                                                                     │
│ 2. Compilar S(D0) vs S(a4r-mert), computar ΔA4                                                                                                           │
│ 3. Documentar en NOTAS + README                                                                                                                          │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ Checkpoint loader + arch_config (trazabilidad)                                                                                                           │
│                                                                                                                                                          │
│ arch_config persistido en checkpoints                                                                                                                    │
│                                                                                                                                                          │
│ arch_config = {                                                                                                                                          │
│     'gate': '7.1',                                                                                                                                       │
│     'descriptor': descriptor,          # 'd0' o 'a4r-mert'                                                                                               │
│     'audio_encoder': 'mert',           # NUEVO: distingue de 'lite'                                                                                      │
│     'audio_encoder_frozen': True,                                                                                                                        │
│     'from_scratch': True,                                                                                                                                │
│     'eval_compatible': True,                                                                                                                             │
│ }                                                                                                                                                        │
│ # Guardar en cada checkpoint junto con model_state_dict, epoch, best_S, etc.                                                                             │
│                                                                                                                                                          │
│ Extensión de checkpoint_loader.py                                                                                                                        │
│                                                                                                                                                          │
│ En load_model_from_checkpoint() (checkpoint_loader.py:113), añadir soporte:                                                                              │
│                                                                                                                                                          │
│ # Antes de reconstruir modelo, leer audio_encoder de arch_config                                                                                         │
│ audio_enc = arch_config.get('audio_encoder', 'lite')  # backward compatible                                                                              │
│                                                                                                                                                          │
│ if audio_enc == 'mert':                                                                                                                                  │
│     base_model = CrossModalModel(audio_encoder='mert', audio_encoder_frozen=True, use_dann=False)                                                        │
│ else:                                                                                                                                                    │
│     base_model = CrossModalModel(audio_encoder='lite', use_dann=False)                                                                                   │
│                                                                                                                                                          │
│ # Para descriptor 'a4r-mert': nueva rama                                                                                                                 │
│ if descriptor == 'a4r-mert':                                                                                                                             │
│     model = Gate71MERTReverseCrossAttModel(base_model, ...)                                                                                              │
│ elif descriptor == 'd0' and audio_enc == 'mert':                                                                                                         │
│     model = Gate42Model(base_model, descriptor_fn=None, ...)                                                                                             │
│ # ... rest of existing branches unchanged                                                                                                                │
│                                                                                                                                                          │
│ También en la rama D0 existente (L159), respetar audio_encoder:                                                                                          │
│ elif descriptor == 'd0':                                                                                                                                 │
│     base_model_cls = CrossModalModel(audio_encoder=audio_enc, use_dann=False)                                                                            │
│     model = Gate42Model(base_model_cls, ...)                                                                                                             │
│                                                                                                                                                          │
│ Structured eval                                                                                                                                          │
│                                                                                                                                                          │
│ Structured eval corre live dentro de train_gate71.py (mismo patrón que gate43_scratch). El script importa y llama extract_all_embeddings() +             │
│ compute_retrieval_metrics() directamente en las epochs marcadas. Post-hoc eval también funciona gracias al loader extendido.                             │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ Comparabilidad ΔA4 (nota metodológica)                                                                                                                   │
│                                                                                                                                                          │
│ ΔA4 en 7.1b no es directamente comparable con el +5.5pp de Gate 5B. Cambian: (a) backbone (Lite → MERT-330M), (b) régimen (fine-tune layers 2-3 → frozen │
│  total), (c) el propio a4r (enc.transformer → light_transformer nuevo). Es evidencia para decisión de programa — ¿vale la pena diseñar descriptores para │
│  Escalón 2? — no continuación del delta histórico.                                                                                                       │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ Verificaciones de sanidad                                                                                                                                │
│                                                                                                                                                          │
│ 1. Anti-ghost (params): trainable ≈ 15M (D0) o 20M (a4r-mert), NO ~345M                                                                                  │
│ 2. Anti-ghost (weights): audio_encoder weights idénticos pre/post epoch 1                                                                                │
│ 3. Anti-ghost (mode): base_model.audio_encoder._model.training == False después de model.train()                                                         │
│ 4. Shape: MERT-330M → [B, 1024] (D0) o [B, T, 1024] (a4r-mert return_sequence), MIDI → [B, 256]                                                          │
│ 5. Throughput: log batches/min en primeros 20 batches, abortar si < 1 batch/min  
