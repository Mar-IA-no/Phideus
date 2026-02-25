 Plan: Gate 5B — Universal Loader + Scientific Validation Tests

 Estado de ejecución (2026-02-25, addendum):
 - Test12 Scoreboard: CERRADO.
 - Test01 Causal Ablation: CERRADO.
 - Test04 Transposition: CERRADO.
 - Test03 RatioProbe: CERRADO.
 - Test06 RSA/CKA: CERRADO.
 - Test08 Ratio Decoding: CERRADO.
 - Test10 Visualizations: CERRADO.
 - Test09 Invariance Suite: EN CURSO.
 - Pendientes UNC: Test02 (parameter-matched) y Test05 (multi-seed).
 - Este documento se mantiene como plan base; el estado operativo vivo está en:
   - `README.md` del Gate 5B
   - `INFORME_EJECUCION_TEST01_TEST12_2026-02-25.md`
   - `Documents/NOTAS_CLAUDE-CODEX.md`

 Context

 Gate 5B es la bateria de 13 tests cientificos que valida los mejores modelos del proyecto Phideus. Antes de ejecutar cualquier test,
 necesitamos resolver un problema critico: los checkpoints de modelos aumentados (d4a4, a4r, d4-a4r) tienen eval_compatible=False y el
 script actual de evaluacion (evaluate_structured_pool.py) los carga con strict=False, descartando silenciosamente los componentes de
 augmentacion. Esto produce resultados incorrectos.

 Modelos a evaluar (ya transferidos a models/gate5b/):

 ┌────────┬────────┬────────────────────────────────────┐
 │  Arm   │ Best S │             Checkpoint             │
 ├────────┼────────┼────────────────────────────────────┤
 │ D0     │ 73.4%  │ models/gate5b/D0/best_model.pt     │
 ├────────┼────────┼────────────────────────────────────┤
 │ d4a4   │ 83.8%  │ models/gate5b/d4a4/best_model.pt   │
 ├────────┼────────┼────────────────────────────────────┤
 │ a4r    │ 82.0%  │ models/gate5b/a4r/best_model.pt    │
 ├────────┼────────┼────────────────────────────────────┤
 │ d4-a4r │ 79.8%  │ models/gate5b/d4-a4r/best_model.pt │
 └────────┴────────┴────────────────────────────────────┘

 ---
 Paso 1: Universal Checkpoint Loader

 Problema

 La logica de reconstruccion de modelos vive duplicada en gate43_scratch_training.py::run_evaluate() (lineas 3932-4022) — un bloque de ~90
  lineas de if/elif por descriptor. evaluate_structured_pool.py tiene su propia carga (linea 370-372) que usa strict=False y solo
 instancia CrossModalModel puro, rompiendo modelos aumentados.

 Solucion

 Extraer la logica a un modulo reutilizable.

 Archivo nuevo: experiments/bias_control/gate5b/checkpoint_loader.py

 Nota de acoplamiento: El loader depende de clases definidas en gate43_scratch_training.py (experiments/).
 Ubicarlo en src/ crearia dependencia src/→experiments/ fragil.
 Dejarlo en experiments/bias_control/gate5b/ es mas coherente con la realidad del codigo.

 Funciones publicas:
 def load_model_from_checkpoint(
     checkpoint_path: str,
     device: torch.device = torch.device('cpu'),
 ) -> Tuple[nn.Module, dict]:
     """
     Carga un checkpoint Gate 4.x y reconstruye el modelo completo.

     Returns: (model, metadata)
         model: nn.Module en eval mode, en device
         metadata: dict con arch_config, descriptor, epoch, best_S

     Raises:
         RuntimeError si checkpoint_type == 'archive_base'
         ValueError si descriptor desconocido
     """

 def get_eval_batch_size(descriptor: str) -> int:
     """Batch size optimo para eval segun tipo de descriptor y VRAM."""

 CRITICO — Imports lazy para evitar ciclos:
 - gate43_scratch_training.py importa evaluate_structured_pool.py (lineas 84-90)
 - Si el loader importa clases de gate43_scratch_training.py a top-level, se crea ciclo
 - Solucion: Todos los imports de clases de modelo se hacen dentro de load_model_from_checkpoint(), no a nivel de modulo

 # checkpoint_loader.py — NO top-level imports de gate43
 def load_model_from_checkpoint(checkpoint_path, device=...):
     # Lazy imports dentro de la funcion
     from src.bias_control.architectures.cross_modal_model import CrossModalModel
     from experiments.bias_control.gate43_scratch.gate43_scratch_training import (
         Gate42InputAugModel, Gate42AudioAugModel, ...
         create_gate42_model,
     )
     ...

 Logica interna (reutilizar sin duplicar):
 - Usa dispatch directo para d4, a4, a7, d4a4, a4r, d4-a4r, etc. (misma logica que run_evaluate())
 - Lee arch_config.moe_config para MoE variants
 - Lee arch_config.use_d4a4_injection para t3-anc
 - Siempre strict=True para detectar problemas
 - Leer best_S del checkpoint (NO best_val_S)
 - Nota: checkpoints tienen prefijo base_model.* en state_dict (no solo augmented — D0 tambien)

 Clases de modelo necesarias (importadas lazy):
 - CrossModalModel (de src/bias_control/architectures/cross_modal_model)
 - Las 13 clases wrapper: Gate42InputAugModel, Gate42AudioAugModel, Gate42DualAugModel, Gate42AudioCrossAttModel, Gate42MidiCrossAttModel,
  Gate42DualCrossModalModel, Gate42AudioReverseCrossAttModel, Gate42DualReverseCrossAttModel, Gate42DualMixedModel, Gate44ThirdTowerModel,
  Gate44FiLMModel, Gate44MoEModel, Gate42Model
 - RatioEncoder, create_gate42_model

 Archivos a modificar

 - evaluate_structured_pool.py (lineas 366-374): Reemplazar carga con strict=False por load_model_from_checkpoint(). Respetar
 get_eval_batch_size().
 - CRITICO: En evaluate_structured_pool.py, importar el loader dentro de main() (no top-level) para evitar ciclo con
 gate43_scratch_training.py.

 Verificacion

 # Test 1: Cargar D0 (eval_compatible=True)
 python -c "
 from experiments.bias_control.gate5b.checkpoint_loader import load_model_from_checkpoint
 model, meta = load_model_from_checkpoint('models/gate5b/D0/best_model.pt')
 print(f'D0: {meta[\"descriptor\"]}, epoch={meta[\"epoch\"]}, S={meta[\"best_S\"]}')
 "

 # Test 2: Cargar d4a4 (eval_compatible=False, requiere reconstruccion)
 python -c "
 from experiments.bias_control.gate5b.checkpoint_loader import load_model_from_checkpoint
 model, meta = load_model_from_checkpoint('models/gate5b/d4a4/best_model.pt')
 print(f'd4a4: params={sum(p.numel() for p in model.parameters())/1e6:.1f}M')
 "

 # Test 3: Evaluar d4a4 con evaluate_structured_pool.py
 python experiments/bias_control/evaluate_structured_pool.py \
     --model models/gate5b/d4a4/best_model.pt \
     --output /tmp/test_d4a4_eval.json --seed 42
 # NOTA: evaluate_structured_pool.py guarda a2m/m2a separados, NO gate_metrics.S
 # Calcular S: min(a2m.mean_recall@10, m2a.mean_recall@10) ~ 0.838

 Sobre el calculo de S: evaluate_structured_pool.py serializa a2m.mean_recall@10
 y m2a.mean_recall@10 por separado. S = min(A2M_R@10, M2A_R@10).
 El Test 12 (Scoreboard) define y persiste este calculo explicitamente.

 ---
 Paso 2: Gate 5B Test Harness

 Archivo nuevo: experiments/bias_control/gate5b/harness.py

 Responsabilidades:
 - Carga de modelo via load_model_from_checkpoint()
 - Extraccion de embeddings (reutilizar extract_all_embeddings() de evaluate_structured_pool.py)
 - Carga de dataset MAESTRO validation
 - Formato estandar de resultados JSON (commit hash, seed, fecha, descriptor, test_name)
 - CLI comun: --model, --maestro-dir, --output, --device, --seed

 Funciones:
 def setup_gate5b_test(args) -> Tuple[nn.Module, dict, MaestroSegmentDataset, dict]:
     """Carga modelo + dataset + index. Retorna (model, metadata, dataset, index)."""

 def extract_embeddings(model, dataset, device, descriptor) -> Tuple[Tensor, Tensor]:
     """Wrapper sobre extract_all_embeddings con batch_size correcto."""

 def save_test_result(result: dict, output_path: str, test_name: str):
     """Guarda resultado con metadata estandar."""

 Directorio de resultados: data/gate5b_results/

 data/gate5b_results/
 ├── D0/
 │   ├── test01_causal_ablation.json
 │   ├── test04_transposition.json
 │   └── ...
 ├── d4a4/
 ├── a4r/
 └── d4-a4r/

 ---
 Paso 3: Tests Cientificos (Fase A — LOCAL, inference-only)

 Test 1: Causal Ablation (zero-out injection)

 Archivo: experiments/bias_control/gate5b/test01_causal_ablation.py
 Concepto: Zerear el descriptor en el forward pass y medir cuanto cae S.

 CRITICO — No usar hooks genericos: Los descriptores (A4, D4) se computan dentro del forward pass de cada modelo augmentado (no son inputs
  externos). Por eso, la ablation necesita modos explicitos en el forward:

 Metodo:
 1. Cargar modelo aumentado (d4a4, a4r, d4-a4r)
 2. Evaluar normalmente → S_normal
 3. Modificar el forward pass con modos explicitos:
   - zero_audio: zerear tensor A4 despues de compute_audio_descriptor_a4() y antes de inyeccion
   - zero_midi: zerear tensor D4 despues de compute_local_interval_features() y antes de inyeccion
   - zero_both: zerear ambos
   - noise: reemplazar descriptores por ruido gaussiano con misma mean/std
   - shuffle: permutar descriptores entre samples del batch
 4. Evaluar con cada modo → S_ablated
 5. Delta = S_normal - S_ablated

 Implementacion — Patch target correcto:
 Las funciones de descriptor (compute_audio_descriptor_a4, compute_local_interval_features) se importan
 en el modulo gate43_scratch_training y se usan dentro de los wrappers. Parchear el simbolo en
 src.bias_control.* NO garantiza afectar el forward — hay que parchear el simbolo en el modulo
 gate43_scratch_training (donde las clases wrapper lo resuelven).

 # Correcto:
 import experiments.bias_control.gate43_scratch.gate43_scratch_training as g43
 original_a4 = g43.compute_audio_descriptor_a4
 g43.compute_audio_descriptor_a4 = lambda *args, **kw: torch.zeros_like(original_a4(*args, **kw))

 Sanity check de ablation efectiva: Despues de parchear, verificar que los embeddings cambian
 respecto a la evaluacion normal. Si no cambian, el patch no esta en el path correcto.

 Output: JSON con S_normal, S_ablated por variante, delta
 Para D0: skip (no tiene descriptor), sirve como sanity check (delta debe ser ~0)

 Test 4: Transposition Invariance

 Archivo: experiments/bias_control/gate5b/test04_transposition.py
 Concepto: Si el modelo aprende ratios, transponer MIDI ±N semitonos no deberia cambiar el matching.
 Metodo:
 1. Extraer embeddings de audio normalmente
 2. Para cada transposicion en [-6, -3, -1, +1, +3, +6] semitonos:
 a. Clonar midi_pitch antes de transponer (nunca in-place): pitch_t = midi_pitch.clone()
 b. Transponer solo posiciones validas — midi_mask=True es padding, False es valido:
    valid = ~midi_mask; pitch_t[valid] += shift
 c. Clamp a rango MIDI valido: pitch_t.clamp_(0, 127)
 d. Extraer embeddings MIDI transpuestos
 d. Evaluar retrieval Audio→MIDI_transpuesto
 3. Comparar S por transposicion vs S original
 Output: JSON con S por transposicion, curva de degradacion
 Hipotesis: Si el modelo usa ratios (intervalos relativos), S deberia mantenerse estable. Para D0 deberia degradar mas que para d4a4/a4r.

 Test 10: UMAP/t-SNE

 Archivo nuevo: experiments/bias_control/gate5b/test10_visualizations.py

 NO reutilizable: visualize_embeddings_multigate.py esta hardcodeado para Gate 6
 (paths fijos, labels fijos, multigate_embeddings.npz). extract_multigate_embeddings.py
 usa checkpoints viejos y strict=False. Ambos necesitan reescritura sustancial.

 Metodo:
 1. Extraer embeddings de los 4 checkpoints Gate 5B usando checkpoint_loader
 2. Serializar a NPZ con formato estandar: {arm}_audio_embs, {arm}_midi_embs, piece_idx
 3. UMAP/t-SNE con colores por arm y por modalidad (audio=cyan, midi=magenta)
 4. Figuras comparativas 2x2: D0 vs d4a4 vs a4r vs d4-a4r
 Output: PNGs en data/gate5b_results/visualizations/

 Test 12: Gate Scoreboard (PRIMERO — valida loader)

 Archivo: experiments/bias_control/gate5b/test12_scoreboard.py
 Metodo: Re-evaluar los 4 checkpoints con config canonica fija.

 Config canonica (pinned, serializada en output):
 CANONICAL_EVAL_CONFIG = {
     'pool_size': 256,
     'n_queries': 500,
     'n_hard_negatives': 64,
     'n_semi_hard_negatives': 32,
     'seed': 42,
 }

 Calculo de S explicito:
 S = min(results['a2m']['mean_recall@10'], results['m2a']['mean_recall@10'])

 Output: JSON con tabla completa incluyendo config canonica, S calculado, por arm.
 Funcion critica: Valida que el loader reconstruye correctamente cada modelo (S debe
 coincidir con los valores historicos: D0~73.4%, d4a4~83.8%, a4r~82.0%, d4-a4r~79.8%).

 ---
 Paso 4: Tests Cientificos (Fase B — LOCAL, gradient/activation analysis)

 Test 6: RSA/CKA Between Layers

 Archivo: experiments/bias_control/gate5b/test06_rsa_cka.py
 Concepto: Medir similitud representacional entre capas de audio y MIDI.
 Metodo:
 1. Registrar hooks en las 4 capas Transformer de audio y 4 de MIDI
 2. Forward pass sobre ~500 segmentos
 3. Extraer activaciones por capa: [N, T, D] → mean-pool → [N, D]
 4. Computar RSA: correlacion entre matrices de distancia inter-capa
 5. Computar CKA (linear): similitud de kernel centrado
 Output: Matriz RSA 8x8 (4 audio + 4 MIDI), CKA heatmap, por modelo

 Test 8: Ratio Decoding Report

 Archivo: experiments/bias_control/gate5b/test08_ratio_decoding.py
 Concepto: Que features del descriptor contribuyen mas al embedding final.

 Limitacion critica: Los descriptores se computan bajo torch.no_grad() + .detach()
 (gate43_scratch_training.py lineas 563, 572, 975). Gradient × activation naive sobre
 el input dara cero/no-interpretable.

 Metodo alternativo (no requiere grad a traves del descriptor):
 1. Perturbation-based: Para cada dim del descriptor, perturbar ±epsilon y medir
 cambio en embedding (sensitivity analysis, no requiere grad).
 2. Ablation por feature: Zerear una dim a la vez del descriptor, medir delta S.
 3. Correlation analysis: Correlacion de Pearson entre cada dim del descriptor y
 componentes del embedding final.
 Output: JSON con ranking de importancia por metodo, heatmaps de sensibilidad

 Test 9: Invariance Suite

 Archivo: experiments/bias_control/gate5b/test09_invariance_suite.py
 Concepto: Extender test 4 con mas transformaciones.
 Tests:
 - Temporal shift: mover segmento ±0.5s → embeddings similares?
 - Velocity scaling: midi_velocity × {0.5, 0.8, 1.2, 1.5} → S estable?
 - Octave transposition: ±12 semitonos → S estable?
 - Audio noise: additive Gaussian noise a audio → degradacion gradual?
 Output: JSON con S por transformacion × nivel

 ---
 Paso 5: Tests Cientificos (Fase C — UNC, training required)

 Test 2: Parameter-Matched Ablations

 Archivo: experiments/bias_control/gate5b/test02_param_matched.py + SLURM script
 Concepto: Controlar que la mejora de d4a4 no se debe solo a mas parametros.
 Metodo: Entrenar 3 modelos con ~66.5M params (= d4a4) pero:
   a. Random injection: descriptor reemplazado por ruido gaussiano
   b. Shuffled injection: descriptor real pero shuffled entre samples
   c. Zero injection: descriptor zeroed (parametros existen pero no reciben señal)
 Training: 30 epochs cada uno, seed=42, misma config que d4a4
 Output: S de cada ablation vs d4a4 (83.8%)

 Test 3: RatioProbeDecoder

 Archivo: experiments/bias_control/gate5b/test03_ratio_probe.py + SLURM
 Concepto: "Smoking gun" — entrenar un probe lineal sobre embeddings congelados.
 Metodo:
 1. Congelar mejor modelo
 2. Entrenar MLP pequeno: z_audio [256] → MIDI features (pitch histogram, interval distribution)
 3. Entrenar MLP pequeno: z_midi [256] → Audio features (spectral centroid, chroma)
 4. Cross-decoding: z_audio → MIDI features (el modelo aprendio features del otro dominio?)
 Output: R² de reconstruccion, comparar D0 vs d4a4

 Test 5: Multi-Seed Replication

 Nota: d4a4 30ep ya tiene 5-seed (84.1%±2.3pp). Faltan D0, a4r, d4-a4r.
 Archivo: SLURM array job experiments/bias_control/slurm/gate5b_multiseed.sh
 Metodo: 5 seeds × 3 descriptors × 30 epochs
 Output: Media ± std por descriptor

 Test 7: Counterfactual Decoder (baja prioridad)

 Test 11: CrossModalSequenceDecoder (baja prioridad)

 Estos son los mas costosos y menos criticos. Implementar solo si tests 1-6 dan resultados positivos.

 ---
 Paso 6: Retrieval Demo (Fase D)

 Test 13: Retrieval Demo UI

 Implementar despues de todos los tests cientificos. Streamlit o web basica.

 ---
 Orden de Implementacion

 ┌──────┬──────────────────────────────────────────────────┬──────────────────────────────────────────────────────┬──────────────┐
 │ Paso │                       Que                        │                        Donde                         │ Prerequisito │
 ├──────┼──────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼──────────────┤
 │ 1    │ Checkpoint Loader (lazy imports)                 │ experiments/bias_control/gate5b/checkpoint_loader.py │ —            │
 ├──────┼──────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼──────────────┤
 │ 1b   │ Fix evaluate_structured_pool.py (import en main) │ Lineas 366-374                                       │ Paso 1       │
 ├──────┼──────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼──────────────┤
 │ 2    │ Harness Gate 5B                                  │ experiments/bias_control/gate5b/harness.py           │ Paso 1       │
 ├──────┼──────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼──────────────┤
 │ 2b   │ Test 12: Scoreboard (valida loader!)             │ gate5b/test12_scoreboard.py                          │ Paso 2       │
 ├──────┼──────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼──────────────┤
 │ 3a   │ Test 1: Causal Ablation (modos explicitos)       │ gate5b/test01_causal_ablation.py                     │ Paso 2b      │
 ├──────┼──────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼──────────────┤
 │ 3b   │ Test 4: Transposition (clamp+mask)               │ gate5b/test04_transposition.py                       │ Paso 2b      │
 ├──────┼──────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼──────────────┤
 │ 4a   │ Test 6: RSA/CKA                                  │ gate5b/test06_rsa_cka.py                             │ Paso 2b      │
 ├──────┼──────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼──────────────┤
 │ 4b   │ Test 8: Ratio Decoding                           │ gate5b/test08_ratio_decoding.py                      │ Paso 2b      │
 ├──────┼──────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼──────────────┤
 │ 4c   │ Test 9: Invariance Suite                         │ gate5b/test09_invariance_suite.py                    │ Paso 2b      │
 ├──────┼──────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼──────────────┤
 │ 5a   │ Test 2: Param-matched                            │ gate5b/test02_param_matched.py + SLURM               │ Paso 2b      │
 ├──────┼──────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼──────────────┤
 │ 5b   │ Test 3: RatioProbe                               │ gate5b/test03_ratio_probe.py + SLURM                 │ Paso 2b      │
 ├──────┼──────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼──────────────┤
 │ 5c   │ Test 5: Multi-seed                               │ SLURM array job                                      │ Paso 2b      │
 ├──────┼──────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼──────────────┤
 │ 6    │ Test 10: Viz (script nuevo, no reutilizable)     │ gate5b/test10_visualizations.py                      │ Paso 2b      │
 └──────┴──────────────────────────────────────────────────┴──────────────────────────────────────────────────────┴──────────────┘

 Paralelismo: Pasos 3a/3b/3c se pueden implementar y ejecutar en paralelo. Lo mismo 4a/4b/4c. Los pasos 5a/5b/5c se envian a UNC
 simultaneamente.

 ---
 Archivos Criticos (referencia)

 ┌─────────────────────────────────────────────────────┬─────────────────────────────────────────────────────┐
 │                       Archivo                       │                         Rol                         │
 ├─────────────────────────────────────────────────────┼─────────────────────────────────────────────────────┤
 │ gate43_scratch_training.py:2488-2596                │ create_gate42_model() — factory existente           │
 ├─────────────────────────────────────────────────────┼─────────────────────────────────────────────────────┤
 │ gate43_scratch_training.py:3897-4056                │ run_evaluate() — logica de reconstruccion a extraer │
 ├─────────────────────────────────────────────────────┼─────────────────────────────────────────────────────┤
 │ evaluate_structured_pool.py:91-143                  │ extract_all_embeddings() — reutilizar               │
 ├─────────────────────────────────────────────────────┼─────────────────────────────────────────────────────┤
 │ evaluate_structured_pool.py:196-273                 │ evaluate_with_precomputed_embeddings() — reutilizar │
 ├─────────────────────────────────────────────────────┼─────────────────────────────────────────────────────┤
 │ evaluate_structured_pool.py:366-374                 │ Carga actual (BROKEN para augmented) — FIX          │
 ├─────────────────────────────────────────────────────┼─────────────────────────────────────────────────────┤
 │ src/bias_control/architectures/cross_modal_model.py │ CrossModalModel base                                │
 ├─────────────────────────────────────────────────────┼─────────────────────────────────────────────────────┤
 │ visualize_embeddings_multigate.py                   │ Referencia de estilo (NO reutilizable para Gate 5B) │
 └─────────────────────────────────────────────────────┴─────────────────────────────────────────────────────┘

---
Anexo A: Ejecucion real y lectura causal (2026-02-24 a 2026-02-25)

Objetivo de este anexo

Documentar resultados observados en corrida real de Gate 5B para:
- validar que el loader reconstruye correctamente los modelos complejos;
- cerrar Test 12 (scoreboard) con config canonica fija;
- cerrar Test 01 (causal ablation) con interpretacion de zero/noise/shuffle;
- explicitar impacto sobre la hipotesis D4 vs A4/A4r.

Evidencia principal (artefactos)

- data/gate5b_results/scoreboard.json
- data/gate5b_results/D0/test12_scoreboard.json
- data/gate5b_results/d4a4/test12_scoreboard.json
- data/gate5b_results/a4r/test12_scoreboard.json
- data/gate5b_results/d4-a4r/test12_scoreboard.json
- data/gate5b_results/D0/test01_causal_ablation.json
- data/gate5b_results/d4a4/test01_causal_ablation.json
- data/gate5b_results/a4r/test01_causal_ablation.json
- data/gate5b_results/d4-a4r/test01_causal_ablation.json
- /tmp/gate5b_test01_v2.log

---
A.1 Test 12 Scoreboard (config canonica) — cerrado

Configuracion fija usada:

CANONICAL_EVAL_CONFIG = {
    'pool_size': 256,
    'n_queries': 500,
    'n_hard_negatives': 64,
    'n_semi_hard_negatives': 32,
    'seed': 42,
}

Calculo de score:
S = min(a2m.mean_recall@10, m2a.mean_recall@10)

Resultados observados:

┌────────┬───────┬──────────┬──────────┬────────────┐
│  Arm   │   S   │ A2M R@10 │ M2A R@10 │ Validation │
├────────┼───────┼──────────┼──────────┼────────────┤
│ D0     │ 73.4% │ 74.8%    │ 73.4%    │ PASS       │
├────────┼───────┼──────────┼──────────┼────────────┤
│ d4a4   │ 83.8% │ 84.4%    │ 83.8%    │ PASS       │
├────────┼───────┼──────────┼──────────┼────────────┤
│ a4r    │ 82.0% │ 82.6%    │ 82.0%    │ PASS       │
├────────┼───────┼──────────┼──────────┼────────────┤
│ d4-a4r │ 79.8% │ 81.4%    │ 79.8%    │ PASS       │
└────────┴───────┴──────────┴──────────┴────────────┘

Conclusion operativa:
- El loader universal (reconstruccion de wrappers + strict=True) queda validado para los 4 checkpoints Gate 5B.

---
A.2 Test 01 Causal Ablation — cerrado

Definicion de modos (causalidad del descriptor)

- zero_*: descriptor reemplazado por ceros.
  Interpreta necesidad bruta de la senal.
- noise_*: descriptor reemplazado por ruido N(mean,std) con la misma media/std del descriptor real.
  Interpreta si importa contenido semantico vs estadistica global.
- shuffle_*: descriptor permutado entre samples del batch.
  Interpreta si importa correspondencia muestra-descriptor.

Metrica primaria:
- S = min(A2M R@10, M2A R@10)
- delta = S_normal - S_ablated

---
A.3 Resultados completos por arm

1) D0 (control negativo)

┌────────┬──────────┐
│ Arm    │ S_normal │
├────────┼──────────┤
│ D0     │ 73.4%    │
└────────┴──────────┘

Sin descriptores; no aplica ablation.

2) d4a4 (S_normal=83.8%)

┌──────────────┬────────┬────────────┐
│ Modo         │   S    │   delta    │
├──────────────┼────────┼────────────┤
│ zero_audio   │  7.8%  │ +76.0 pp   │
│ zero_midi    │ 84.4%  │ -0.6 pp    │
│ zero_both    │  7.6%  │ +76.2 pp   │
│ noise_audio  │ 22.0%  │ +61.8 pp   │
│ noise_midi   │ 84.4%  │ -0.6 pp    │
│ noise_both   │ 19.6%  │ +64.2 pp   │
│ shuffle_audio│ 46.6%  │ +37.2 pp   │
│ shuffle_midi │ 83.8%  │ +0.0 pp    │
│ shuffle_both │ 48.4%  │ +35.4 pp   │
└──────────────┴────────┴────────────┘

3) a4r (S_normal=82.0%)

┌──────────────┬────────┬────────────┐
│ Modo         │   S    │   delta    │
├──────────────┼────────┼────────────┤
│ zero_audio   │  4.4%  │ +77.6 pp   │
│ noise_audio  │ 29.0%  │ +53.0 pp   │
│ shuffle_audio│ 49.8%  │ +32.2 pp   │
└──────────────┴────────┴────────────┘

4) d4-a4r (S_normal=79.8%)

┌──────────────┬────────┬────────────┐
│ Modo         │   S    │   delta    │
├──────────────┼────────┼────────────┤
│ zero_audio   │  4.4%  │ +75.4 pp   │
│ zero_midi    │ 79.4%  │ +0.4 pp    │
│ zero_both    │  4.4%  │ +75.4 pp   │
│ noise_audio  │ 26.8%  │ +53.0 pp   │
│ noise_midi   │ 79.8%  │ +0.0 pp    │
│ noise_both   │ 25.6%  │ +54.2 pp   │
│ shuffle_audio│ 47.4%  │ +32.4 pp   │
│ shuffle_midi │ 79.8%  │ +0.0 pp    │
│ shuffle_both │ 47.6%  │ +32.2 pp   │
└──────────────┴────────┴────────────┘

---
A.4 Hallazgos tecnicos (separando observacion / hipotesis / inferencia)

Observacion 1
- En d4a4 y d4-a4r, ablaciones sobre audio (zero/noise/shuffle_audio) degradan fuertemente S.

Observacion 2
- En d4a4 y d4-a4r, ablaciones sobre MIDI descriptor (zero/noise/shuffle_midi) casi no afectan S.

Observacion 3
- Historicamente, d4 solo mejora sobre d0 en Gate 4.2/4.3, pero con ganancia marginal:
  - gate42 screening: 60.4% -> 64.2% (+3.8 pp)
  - gate43 screening: 60.2% -> 63.6% (+3.4 pp)

Hipotesis principal
- D4 contiene senal util pero de baja magnitud; en modelos duales con A4/A4r, su contribucion queda mayormente redundante o dominada por la ruta de audio.

Inferencia operativa (con evidencia actual)
- Para los checkpoints top de Gate 5B, el driver causal principal en inferencia es la rama de audio descriptor.
- D4 no queda descartado globalmente; su utilidad es condicional al modelo/regimen y aparece marginal en esta familia dual.

---
A.5 Incidentes y fix aplicados durante corrida

Incidente observado
- collect_descriptor_stats fallaba al concatenar D4 por longitud temporal variable [B, N, 4] entre batches.

Fix aplicado
- Flatten por batch antes de concatenar para estadistica global:
  flat_midi = torch.cat([v.reshape(-1, v.size(-1)) for v in midi_vals], dim=0)

Impacto
- Test01 completo sin bloqueo; stats de ruido generadas correctamente.

---
A.6 Estado de cierre de fase y siguiente paso recomendado

Estado cerrado:
- Paso 1 (loader) y Paso 1b (fix evaluate_structured_pool): funcionales.
- Paso 2 (harness): funcional.
- Paso 2b (test12 scoreboard): cerrado.
- Paso 3a (test01 causal ablation): cerrado para D0, d4a4, a4r, d4-a4r.

Siguiente paso recomendado:
1. Ejecutar Test 4 (transposition) aprovechando cache de embeddings.
2. Consolidar comparativa de invariancia por arm.
3. Recién despues pasar a Test 6/8/9 (analisis mas costoso).
