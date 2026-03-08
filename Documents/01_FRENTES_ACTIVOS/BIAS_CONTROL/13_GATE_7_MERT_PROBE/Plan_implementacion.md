│ Plan: Gate 7 — MERT-large Linear Probe                                                                                                                   │
│                                                                                                                                                          │
│ Contexto                                                                                                                                                 │
│                                                                                                                                                          │
│ Gate 5B demostró que A4 mejora retrieval cross-modal causalmente (+9.4pp, causal por Test 02) y que la ventaja es geométrica (Test 06 CKA +82%, Test 11  │
│ retention, paradoja de inversión Test 13G-B). Gate 6 Exp C confirma que el plateau de AMT (~F1=0.157) es compatible con un techo del encoder.            │
│                                                                                                                                                          │
│ Pero queda la ambigüedad central identificada por Codex:                                                                                                 │
│                                                                                                                                                          │
│ "Los experimentos actuales no desambiguan entre límite del encoder, límite del objetivo de entrenamiento, y complementariedad genuina del descriptor     │
│ sobre encoders más fuertes."                                                                                                                             │
│                                                                                                                                                          │
│ Gate 7 estrecha esa ambigüedad con el test más barato y más discriminante disponible: un probe lineal que pregunta si MERT-large (330M params, 160k      │
│ horas música) ya codifica accesiblemente la información que A4 captura. El resultado informa sobre el lado audio de la pregunta; no resuelve por sí solo │
│  la ambigüedad cross-modal completa.                                                                                                                     │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ Qué resuelve y qué no resuelve Gate 7                                                                                                                    │
│                                                                                                                                                          │
│ Lo que reduce: la ambigüedad sobre si A4 es accesible linealmente desde encoders audio más fuertes. Si MERT-large ya codifica A4 accesiblemente → el     │
│ encoder era una limitación relevante en nuestro setup. Si no → A4 retiene estatus de descriptor complementario incluso para encoders más fuertes.        │
│                                                                                                                                                          │
│ Lo que NO resuelve por sí solo:                                                                                                                          │
│ - Un probe alto no prueba que "el cuello era exclusivamente el encoder" — puede reflejar linealidad insuficiente, alineamiento mal planteado, o que el   │
│ objetivo de entrenamiento también fue una limitación                                                                                                     │
│ - Un probe bajo no prueba "complementariedad genuina" — puede reflejar un target mal formulado o arquitectura de probe insuficiente                      │
│ - Gate 7 reduce ambigüedad sobre el lado audio; no resuelve la ambigüedad completa del problema cross-modal                                              │
│                                                                                                                                                          │
│ Lectura correcta: leer los resultados en relación a nulls, en relación a MERTLite, y comparando frame-level vs segment-level dentro de cada encoder — no │
│  con thresholds duros.                                                                                                                                   │
│                                                                                                                                                          │
│ Nota sobre MERTLite en la comparación: MERTLite ya fue entrenado en régimen cross-modal (VICReg sobre MAESTRO). MERT-95M/330M son audio foundation       │
│ models sin régimen cross-modal. No son comparaciones simétricas — la diferencia de R² entre ambos mezcla tamaño, datos de pretraining, y objetivo de     │
│ entrenamiento.                                                                                                                                           │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ Experimentos                                                                                                                                             │
│                                                                                                                                                          │
│ Exp 7.0 — Probe A4 vs encoders (experimento central)                                                                                                     │
│                                                                                                                                                          │
│ Pregunta: ¿Cuánto de A4 está codificado implícitamente en cada encoder?                                                                                  │
│                                                                                                                                                          │
│ Encoders a comparar:                                                                                                                                     │
│                                                                                                                                                          │
│ ┌───────────────────────────────┬────────┬────────────────────────────────┬─────────────────────────────────────────────────────┐                        │
│ │            Encoder            │ Params │             Origen             │                        Nota                         │                        │
│ ├───────────────────────────────┼────────┼────────────────────────────────┼─────────────────────────────────────────────────────┤                        │
│ │ MERTEncoderLite (nuestro, D0) │ ~60M   │ Entrenado con VICReg Gate 5B   │ Baseline — ¿cuánto tiene el encoder que entrenamos? │                        │
│ ├───────────────────────────────┼────────┼────────────────────────────────┼─────────────────────────────────────────────────────┤                        │
│ │ MERT-v1-95M                   │ 95M    │ HuggingFace m-a-p/MERT-v1-95M  │ MERT mediano                                        │                        │
│ ├───────────────────────────────┼────────┼────────────────────────────────┼─────────────────────────────────────────────────────┤                        │
│ │ MERT-v1-330M                  │ 330M   │ HuggingFace m-a-p/MERT-v1-330M │ MERT grande — test principal                        │                        │
│ └───────────────────────────────┴────────┴────────────────────────────────┴─────────────────────────────────────────────────────┘                        │
│                                                                                                                                                          │
│ Dos probes:                                                                                                                                              │
│                                                                                                                                                          │
│ ┌─────────────┬───────────────────────────┬──────────────────────────────────────────────┐                                                               │
│ │    Probe    │           Tipo            │                     Rol                      │                                                               │
│ ├─────────────┼───────────────────────────┼──────────────────────────────────────────────┤                                                               │
│ │ LinearProbe │ 1 capa, sin no-linealidad │ Canónico — claim conservador y limpio        │                                                               │
│ ├─────────────┼───────────────────────────┼──────────────────────────────────────────────┤                                                               │
│ │ MLPProbe    │ 2 capas (como test03)     │ Exploratorio — contraste, no claim principal │                                                               │
│ └─────────────┴───────────────────────────┴──────────────────────────────────────────────┘                                                               │
│                                                                                                                                                          │
│ Jerarquía de endpoints:                                                                                                                                  │
│                                                                                                                                                          │
│ ┌──────────────┬─────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│ │  Prioridad   │          Endpoint           │                                                  Rol                                                   │  │
│ ├──────────────┼─────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤  │
│ │ Primario     │ LinearProbe segment-level   │ Comparación canónica entre encoders. Robusto: mean-pool elimina ruido de alineamiento temporal.        │  │
│ ├──────────────┼─────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤  │
│ │ Secundario   │ LinearProbe frame-level     │ Análisis complementario DENTRO de cada encoder. No usar para comparar entre encoders (stride distinto  │  │
│ │              │                             │ contamina).                                                                                            │  │
│ ├──────────────┼─────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤  │
│ │ Exploratorio │ MLPProbe + Exp 7.0b         │ Solo después del resultado lineal principal.                                                           │  │
│ │              │ per-layer                   │                                                                                                        │  │
│ └──────────────┴─────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                                                                          │
│ Alineamiento temporal (solo para frame-level, endpoint secundario):                                                                                      │
│ MERT-large stride 320 @ 24kHz → T_mert≈300 para 4s. A4 stride 40 → T=2400.                                                                               │
│ Downsample A4 por adaptive average pooling (robusto cuando T_mert no divide exacto a 2400):                                                              │
│ # A4: [B, 2400, 8] → [B, T_mert, 8]                                                                                                                      │
│ a4_ds = F.adaptive_avg_pool1d(                                                                                                                           │
│     a4.permute(0,2,1),   # [B, 8, 2400]                                                                                                                  │
│     output_size=T_mert                                                                                                                                   │
│ ).permute(0,2,1)          # [B, T_mert, 8]                                                                                                               │
│ No usar F.interpolate — inventa valores intermedios, incorrecto para señal agregada por ventana.                                                         │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ Protocolo estadístico (rigor)                                                                                                                            │
│                                                                                                                                                          │
│ Split por pieza — previene data leakage:                                                                                                                 │
│ - MAESTRO validación ~150 piezas                                                                                                                         │
│ - Split 80/20 por pieza → todos los segmentos de una pieza caen en el mismo fold                                                                         │
│ - 5 repeated group splits → CIs sobre varianza de split                                                                                                  │
│                                                                                                                                                          │
│ Dos fuentes de varianza separadas:                                                                                                                       │
│ 1. Varianza por group split: 5 seeds de partición aleatoria por pieza                                                                                    │
│ 2. Varianza por entrenamiento del probe: para LinearProbe canónico, usar solución cerrada (Ridge regression con λ pequeño) → elimina varianza de         │
│ optimización, estabiliza resultados                                                                                                                      │
│                                                                                                                                                          │
│ Null controls:                                                                                                                                           │
│                                                                                                                                                          │
│ ┌─────────────────────────────┬─────────────────────────────────┬──────────────────────────────────────────────────────────────────┐                     │
│ │           Control           │            Qué rompe            │                           R² esperado                            │                     │
│ ├─────────────────────────────┼─────────────────────────────────┼──────────────────────────────────────────────────────────────────┤                     │
│ │ Shuffled between segments   │ Correspondencia segmento↔target │ ≈ 0                                                              │                     │
│ ├─────────────────────────────┼─────────────────────────────────┼──────────────────────────────────────────────────────────────────┤                     │
│ │ Time-shuffle within segment │ Dinámica temporal de A4         │ > 0 si probe captura solo timbre global, ≈ 0 si captura dinámica │                     │
│ ├─────────────────────────────┼─────────────────────────────────┼──────────────────────────────────────────────────────────────────┤                     │
│ │ Dummy (predict train mean)  │ Trivialidad                     │ ≤ 0 en test                                                      │                     │
│ └─────────────────────────────┴─────────────────────────────────┴──────────────────────────────────────────────────────────────────┘                     │
│                                                                                                                                                          │
│ El control time-shuffle es importante para frame-level: si R²_frame_shuffled ≈ R²_frame, el probe solo captura estadística global de timbre/registro, no │
│  sigue la dinámica temporal de A4.                                                                                                                       │
│                                                                                                                                                          │
│ Métricas reportadas:                                                                                                                                     │
│ ─ Primario (segment-level, LinearProbe) ─                                                                                                                │
│ R²_seg ± std  [5 group splits × Ridge cerrado]                                                                                                           │
│   ── por cada banda A4 (8) + promedio global ──                                                                                                          │
│   ── para: MERTLite, MERT-95M, MERT-330M ──                                                                                                              │
│                                                                                                                                                          │
│ ─ Secundario (frame-level, LinearProbe, DENTRO de cada encoder) ─                                                                                        │
│ R²_frame ± std                                                                                                                                           │
│ R²_frame_time_shuffled ± std   (null temporal)                                                                                                           │
│                                                                                                                                                          │
│ ─ Baselines ─                                                                                                                                            │
│ R²_shuffled_between ± std      (null global)                                                                                                             │
│ R²_dummy                       (baseline trivial)                                                                                                        │
│                                                                                                                                                          │
│ ─ Exploratorio (post resultado lineal) ─                                                                                                                 │
│ R²_mlp_frame, R²_mlp_seg                                                                                                                                 │
│                                                                                                                                                          │
│ Exp 7.0b — Análisis por capa de MERT-large                                                                                                               │
│                                                                                                                                                          │
│ Pregunta: ¿En qué capa del transformer emerge información tipo A4?                                                                                       │
│                                                                                                                                                          │
│ MERT-large tiene 24 capas transformer. Probear cada capa por separado revela la trayectoria computacional.                                               │
│                                                                                                                                                          │
│ Diseño: mismos 500 segmentos, probe por cada capa de MERT-large → curva R² vs layer depth.                                                               │
│                                                                                                                                                          │
│ Esto es científicamente valioso: si A4 emerge en capas tempranas (CNN/primeras layers) → es una feature de bajo nivel que cualquier encoder debería      │
│ extraer. Si emerge en capas tardías o no emerge → es información de nivel alto o genuinamente complementaria.                                            │
│                                                                                                                                                          │
│ Costo adicional: solo requiere guardar hidden states de todas las capas → output_hidden_states=True en HF MERT. Mismos 500 segmentos.                    │
│                                                                                                                                                          │
│ Exp 7.1 — Mini Test02 con MERT-large (COMPLETAMENTE DIFERIDA)                                                                                            │
│                                                                                                                                                          │
│ No se diseña ahora. Se define solo después de ver los resultados de Exp 7.0.                                                                             │
│                                                                                                                                                          │
│ La decisión de lanzar Exp 7.1 no depende de un threshold duro de R², sino del patrón:                                                                    │
│ - ¿La señal está claramente por encima de los nulls?                                                                                                     │
│ - ¿Hay diferencia o no entre MERTLite y MERT-large?                                                                                                      │
│ - ¿Queda ambigüedad residual sobre "encoder fuerte vs descriptor complementario"?                                                                        │
│                                                                                                                                                          │
│ Las decisiones operativas (qué MIDI encoder usar, cuántas seeds, local vs UNC) se toman con los datos en mano.                                           │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ Estructura de archivos                                                                                                                                   │
│                                                                                                                                                          │
│ experiments/bias_control/gate7/                                                                                                                          │
│ ├── README.md                                                                                                                                            │
│ ├── mert_large_feature_extractor.py    # Wrapper HF MERT → features por capa                                                                             │
│ └── mert_large_probe.py                # Script principal: carga encoders, extrae features, probe A4                                                     │
│                                                                                                                                                          │
│ experiments/bias_control/slurm/                                                                                                                          │
│ └── gate7_mert_probe.sh                # SLURM (por si se corre en UNC)                                                                                  │
│                                                                                                                                                          │
│ Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/13_GATE_7_MERT_PROBE/                                                                                          │
│ └── README.md                          # Documentación del gate                                                                                          │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ Código a reutilizar                                                                                                                                      │
│                                                                                                                                                          │
│ ┌───────────────────────────────┬────────────────────────────────────────────────────────────┬─────────────────────────────────────────┐                 │
│ │          Componente           │                           Fuente                           │              Uso en Gate 7              │                 │
│ ├───────────────────────────────┼────────────────────────────────────────────────────────────┼─────────────────────────────────────────┤                 │
│ │ compute_audio_descriptor_a4() │ src/bias_control/audio_descriptors.py                      │ Target del probe                        │                 │
│ ├───────────────────────────────┼────────────────────────────────────────────────────────────┼─────────────────────────────────────────┤                 │
│ │ LinearProbe class             │ experiments/bias_control/gate5b/test03_ratio_probe.py:62   │ Probe (simplificar a 1 capa para rigor) │                 │
│ ├───────────────────────────────┼────────────────────────────────────────────────────────────┼─────────────────────────────────────────┤                 │
│ │ train_probe()                 │ experiments/bias_control/gate5b/test03_ratio_probe.py:180  │ Entrenamiento + R²                      │                 │
│ ├───────────────────────────────┼────────────────────────────────────────────────────────────┼─────────────────────────────────────────┤                 │
│ │ MaestroSegmentDataset         │ src/bias_control/datasets/maestro_segments.py              │ 500 segmentos validación                │                 │
│ ├───────────────────────────────┼────────────────────────────────────────────────────────────┼─────────────────────────────────────────┤                 │
│ │ EncoderFeatureExtractor hook  │ experiments/bias_control/gate5b/test13g_posthoc_decoder.py │ Extraer features de MERTLite            │                 │
│ ├───────────────────────────────┼────────────────────────────────────────────────────────────┼─────────────────────────────────────────┤                 │
│ │ SLURM pattern                 │ experiments/bias_control/slurm/gate6_vicreg_decoder.sh     │ Base para gate7 script                  │                 │
│ └───────────────────────────────┴────────────────────────────────────────────────────────────┴─────────────────────────────────────────┘                 │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ mert_large_feature_extractor.py — diseño                                                                                                                 │
│                                                                                                                                                          │
│ class MERTLargeExtractor:                                                                                                                                │
│     """                                                                                                                                                  │
│     Wrapper sobre HuggingFace MERT para extraer features.                                                                                                │
│                                                                                                                                                          │
│     Input:  waveform [B, T] @ 24kHz                                                                                                                      │
│     Output: dict con:                                                                                                                                    │
│         'last_layer': [B, T_mert, hidden_size]   (agnóstico a hidden_size)                                                                               │
│         'pooled':     [B, hidden_size]            (mean pool sobre T_mert)                                                                               │
│         'all_layers': list de [B, T_mert, hidden_size] × n_layers  (solo si pedido)                                                                      │
│     """                                                                                                                                                  │
│     def __init__(self, model_name: str = "m-a-p/MERT-v1-330M", device: str = "cuda"):                                                                    │
│         from transformers import AutoModel, Wav2Vec2FeatureExtractor                                                                                     │
│         self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)                                                    │
│         self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)                                                            │
│         self.model.eval()                                                                                                                                │
│         # hidden_size se lee del modelo, no se hardcodea                                                                                                 │
│         self.hidden_size = self.model.config.hidden_size                                                                                                 │
│                                                                                                                                                          │
│     @torch.no_grad()                                                                                                                                     │
│     def extract(self, waveform: torch.Tensor, return_all_layers: bool = False) -> dict:                                                                  │
│         # waveform: [B, T] already at 24kHz                                                                                                              │
│         outputs = self.model(                                                                                                                            │
│             waveform,                                                                                                                                    │
│             output_hidden_states=return_all_layers                                                                                                       │
│         )                                                                                                                                                │
│         last = outputs.last_hidden_state          # [B, T_mert, hidden_size]                                                                             │
│         result = {                                                                                                                                       │
│             'last_layer': last,                                                                                                                          │
│             'pooled': last.mean(dim=1),           # [B, hidden_size]                                                                                     │
│             'hidden_size': self.hidden_size,                                                                                                             │
│         }                                                                                                                                                │
│         if return_all_layers:                                                                                                                            │
│             result['all_layers'] = outputs.hidden_states  # tuple, cada [B, T_mert, hidden_size]                                                         │
│         return result                                                                                                                                    │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ mert_large_probe.py — diseño del script principal                                                                                                        │
│                                                                                                                                                          │
│ # Flujo principal:                                                                                                                                       │
│ # 1. Split por pieza primero (80/20), luego muestreo balanceado con cap por pieza                                                                        │
│ # 2. Para cada encoder: extraer features → cache .npz (agnóstico a hidden_size)                                                                          │
│ # 3. Computar A4 targets + normalización sin leakage (fit en train, apply a test)                                                                        │
│ # 4. Correr LinearProbe canónico (Ridge cerrado) × 5 group splits                                                                                        │
│ # 5. Correr nulls (shuffled between, dummy)                                                                                                              │
│ # 6. Correr frame-level + time-shuffle within (si segment-level dio señal)                                                                               │
│ # 7. Guardar JSON + plots                                                                                                                                │
│                                                                                                                                                          │
│ Protocolo de muestreo:                                                                                                                                   │
│ # 1. Primero split por pieza                                                                                                                             │
│ pieces_train, pieces_test = group_split_by_piece(maestro_val_pieces, ratio=0.8, seed=seed)                                                               │
│                                                                                                                                                          │
│ # 2. Muestreo con cap por pieza para evitar dominancia de piezas largas                                                                                  │
│ MAX_SEGS_PER_PIECE = 5                                                                                                                                   │
│ segments_train = [seg for p in pieces_train                                                                                                              │
│                   for seg in sample(piece_segments[p], min(MAX_SEGS_PER_PIECE, ...))]                                                                    │
│                                                                                                                                                          │
│ Normalización sin leakage (por cada split):                                                                                                              │
│ # Features: z-score fit en train, apply a test                                                                                                           │
│ feat_mean, feat_std = features_train.mean(0), features_train.std(0) + 1e-8                                                                               │
│ features_train_n = (features_train - feat_mean) / feat_std                                                                                               │
│ features_test_n  = (features_test  - feat_mean) / feat_std                                                                                               │
│                                                                                                                                                          │
│ # A4 targets: z-score por banda fit en train                                                                                                             │
│ a4_mean, a4_std = a4_train.mean(0), a4_train.std(0) + 1e-8                                                                                               │
│ a4_train_n = (a4_train - a4_mean) / a4_std                                                                                                               │
│ a4_test_n  = (a4_test  - a4_mean) / a4_std                                                                                                               │
│                                                                                                                                                          │
│ Output JSON (agnóstico a hidden_size):                                                                                                                   │
│ {                                                                                                                                                        │
│   "encoder": "MERT-v1-330M",                                                                                                                             │
│   "hidden_size": 1024,                                                                                                                                   │
│   "n_segments": 472,                                                                                                                                     │
│   "n_pieces": 150,                                                                                                                                       │
│   "n_splits": 5,                                                                                                                                         │
│   "segment_level_linear": {                                                                                                                              │
│     "r2_per_band_mean": [0.31, 0.28, ..., 0.22],                                                                                                         │
│     "r2_per_band_std":  [0.04, 0.03, ..., 0.05],                                                                                                         │
│     "r2_global_mean": 0.27, "r2_global_std": 0.03                                                                                                        │
│   },                                                                                                                                                     │
│   "nulls": {                                                                                                                                             │
│     "shuffled_between": {"r2_global_mean": 0.01},                                                                                                        │
│     "dummy":            {"r2_global_mean": -0.02}                                                                                                        │
│   }                                                                                                                                                      │
│ }                                                                                                                                                        │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ Alineamiento temporal                                                                                                                                    │
│                                                                                                                                                          │
│ # MERT stride: 320 samples @ 24kHz → ~75 frames/s → T_mert ≈ 300 para 4s                                                                                 │
│ # A4 stride: 40 samples @ 24kHz → ~600 frames/s → T_a4 ≈ 2400 para 4s                                                                                    │
│                                                                                                                                                          │
│ # Estrategia: alinear A4 a resolución de MERT (más baja)                                                                                                 │
│ a4 = compute_audio_descriptor_a4(audio)   # [B, 2400, 8]                                                                                                 │
│ a4_aligned = F.interpolate(                                                                                                                              │
│     a4.permute(0,2,1),     # [B, 8, 2400]                                                                                                                │
│     size=T_mert,                                                                                                                                         │
│     mode='linear',                                                                                                                                       │
│     align_corners=False                                                                                                                                  │
│ ).permute(0,2,1)            # [B, T_mert, 8]                                                                                                             │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ Orden de implementación                                                                                                                                  │
│                                                                                                                                                          │
│ Fase 0: Setup (~2h)                                                                                                                                      │
│                                                                                                                                                          │
│ 1. Instalar transformers en venv local: pip install transformers                                                                                         │
│ 2. Descargar MERT-v1-95M y MERT-v1-330M (HuggingFace cache, ~400MB + ~1.3GB)                                                                             │
│ 3. Verificar inference: 1 batch de 4 segmentos → shape [4, T_mert, 1024] correcta                                                                        │
│ 4. Crear directorios: experiments/bias_control/gate7/ + Documents/.../13_GATE_7_MERT_PROBE/                                                              │
│                                                                                                                                                          │
│ Fase 1: Feature extractor (~3h)                                                                                                                          │
│                                                                                                                                                          │
│ 1. Escribir mert_large_feature_extractor.py con MERTLargeExtractor:                                                                                      │
│   - Wrapper sobre transformers.AutoModel                                                                                                                 │
│   - extract(waveform, return_all_layers=False) → dict de features                                                                                        │
│   - Alineamiento temporal A4 → T_mert con F.interpolate                                                                                                  │
│ 2. Test: shapes correctas, output_hidden_states=True funciona                                                                                            │
│                                                                                                                                                          │
│ Fase 2: Script principal (~5h)                                                                                                                           │
│                                                                                                                                                          │
│ 1. Escribir mert_large_probe.py:                                                                                                                         │
│   - Data loading: MaestroSegmentDataset 500 segmentos validación                                                                                         │
│   - Split por pieza (80/20), 5 seeds de split aleatorio                                                                                                  │
│   - LinearProbe (1 capa) + MLPProbe (2 capas, de test03)                                                                                                 │
│   - Baseline shuffled + baseline dummy                                                                                                                   │
│   - Loop encoders × seeds → R² frame + segment por banda                                                                                                 │
│   - Loop capas MERT-330M (Exp 7.0b)                                                                                                                      │
│   - Output JSON con medias ± std + plots (bar chart por banda, curva R² vs layer)                                                                        │
│ 2. Commit + push                                                                                                                                         │
│                                                                                                                                                          │
│ Fase 3a: Extracción y cache de features core (~2h wall clock)                                                                                            │
│                                                                                                                                                          │
│ 1. Para cada encoder: extraer features sin output_hidden_states y guardar .npz:                                                                          │
│   - MERTLite_features.npz — [N, D] pooled + [N, T', D] pre-pool                                                                                          │
│   - MERT-95M_features.npz — [N, hidden_size_95M] pooled + [N, T_mert, hidden_size_95M]                                                                   │
│   - MERT-330M_features.npz — [N, hidden_size_330M] pooled + [N, T_mert, hidden_size_330M]                                                                │
│   - No cachear hidden_states todavía — se decide después según Fase 3b                                                                                   │
│ 2. Verificar shapes, no NaN, consistencia N entre encoders                                                                                               │
│ 3. .npz en data/gate7_results/features/ (gitignored)                                                                                                     │
│                                                                                                                                                          │
│ Fase 3b: Experimento central (~1h wall clock)                                                                                                            │
│                                                                                                                                                          │
│ 1. LinearProbe segment-level (endpoint primario):                                                                                                        │
│   - 5 group splits × 3 encoders × Ridge cerrado → R² por banda + global                                                                                  │
│ 2. Nulls (shuffled between segments, dummy)                                                                                                              │
│ 3. Punto de decisión: ¿hay señal por encima de nulls? ¿diferencia MERTLite vs MERT-large?                                                                │
│                                                                                                                                                          │
│ Fase 3c: Análisis secundario y exploratorio (condicional a Fase 3b)                                                                                      │
│                                                                                                                                                          │
│ Solo si Fase 3b muestra señal informativa:                                                                                                               │
│ 1. Frame-level + time-shuffle within segment (secondary endpoint)                                                                                        │
│ 2. MLPProbe (exploratorio)                                                                                                                               │
│ 3. Exp 7.0b per-layer: si vale la pena, extraer hidden_states de MERT-330M en float16                                                                    │
│ (o streaming capa por capa para evitar OOM en disco: extraer 1 capa → probe → descartar)                                                                 │
│                                                                                                                                                          │
│ Fase 4 (condicional, post resultados)                                                                                                                    │
│                                                                                                                                                          │
│ Si R² MERT-large < umbral relativo a nulls y MERTLite → diseñar Exp 7.1.                                                                                 │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ Documentación                                                                                                                                            │
│                                                                                                                                                          │
│ 1. Crear Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/13_GATE_7_MERT_PROBE/README.md                                                                        │
│ 2. Actualizar ROADMAP_BIAS_CONTROL.md con sección Gate 7                                                                                                 │
│ 3. Agregar nota en NOTAS_CLAUDE-CODEX.md abriendo Gate 7                                                                                                 │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ Verificación                                                                                                                                             │
│                                                                                                                                                          │
│ 1. Features shape correcta por encoder:                                                                                                                  │
│   - MERTLite: [N, 1024] pooled + [N, T', 1024] pre-pool                                                                                                  │
│   - MERT-HF: [N, 1024] pooled + [N, T_mert, 1024] last layer                                                                                             │
│ 2. A4 downsampled correctamente: shape [N, T_mert, 8], valores en rango z-score                                                                          │
│ 3. Shuffled between segments R² ≤ 0.05 — si falla, hay bug en protocolo                                                                                  │
│ 4. Dummy R² cerca de 0 o levemente negativo — baseline trivial, no criterio de bug                                                                       │
│ 5. Ridge cerrado (LinearProbe): solución estable, sin NaN, R² reproducible entre runs con mismo split                                                    │
│ 6. Output JSON completo (medias ± std, 5 splits, 8 bandas + global, 3 encoders)                                                                          │
│ 7. Plots: bar chart R²_seg por encoder (comparación principal), curva R² vs layer (MERT-330M, post Fase 3b)                                              │
│                                                                                                                                                          │
│ ---                                                                                                                                                      │
│ Preguntas abiertas (diferidas a post Exp 7.0)                                                                                                            │
│                                                                                                                                                          │
│ - ¿Lanzar Exp 7.1? Depende del patrón (señal sobre nulls, diferencia MERTLite vs MERT-large, ambigüedad residual)                                        │
│ - ¿Qué MIDI encoder usar para Exp 7.1?                                                                                                                   │
│ - ¿Corre en UNC o local?                                                                                                                                 │
│ - ¿Vale la pena Exp 7.0b per-layer? Solo si resultado central es informativo.    