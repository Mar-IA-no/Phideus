# Plan: A10 Descriptor Revision — Continuous Variants + Fixes

**Estado**: diseño técnico / no ejecutado  
**Rol**: rama complementaria de Gate 9 y extensión secundaria potencial para Escalón 2  
**Prioridad relativa**: no desplaza el factorial `S2-P2.5`

## Context

A10 recurrence-based descriptors were implemented (3 functions, 7 files). Una auditoría técnica y una revisión epistemológica del usuario identificaron:

 1. A10a/A10b force recurrence onto A7's 12 JI attractor vocabulary — pre-imposes Western music theory ontology that HIT should test, not presuppose. Need
 ontology-free ratio-native variants.
 2. ALTA — Memory explosion: torch.cdist on all BT=12864 frames at once → ~707MB dists tensor. Needs chunking.
 3. MEDIA — A10c impure control: Entropy truncated at max_offset=64 vs A10b's full range (125). Confounds the comparison.
 4. MEDIA — VRAM test incomplete: Only tests dim=8, not dim=12 or dim=32.
 5. BAJA — Default descriptor footgun: --descriptor default is h_series for xattn; launching without explicit --descriptor a10* silently runs wrong arm.

La consecuencia no es “descartar A10”, sino reordenarlo mejor. Esta rama no reemplaza el piloto `A7r/A9r`; lo acompaña con una familia que separa hipótesis dirigidas, controles genéricos y variantes continuas ontology-free.

## Solution: Revised A10 Taxonomy

 ┌──────┬─────────────┬─────┬───────────────────────────────────────────────────────────────┬────────────────────────────────────┐
 │ Arm  │   Family    │ Dim │                           Pipeline                            │                Role                │
 ├──────┼─────────────┼─────┼───────────────────────────────────────────────────────────────┼────────────────────────────────────┤
 │ a10a │ A10-JI      │ 12  │ autocorr → peaks → pairwise ratios → 12 JI Gaussian           │ Hypothesis-directed, A7-comparable │
 ├──────┼─────────────┼─────┼───────────────────────────────────────────────────────────────┼────────────────────────────────────┤
 │ a10b │ A10-JI      │ 12  │ RQA diag profile → peaks → pairwise ratios → 12 JI Gaussian   │ Non-linear, A7-comparable          │
 ├──────┼─────────────┼─────┼───────────────────────────────────────────────────────────────┼────────────────────────────────────┤
 │ a10c │ A10-generic │ 6   │ RQA → 6 generic metrics                                       │ Control (no ratio info)            │
 ├──────┼─────────────┼─────┼───────────────────────────────────────────────────────────────┼────────────────────────────────────┤
 │ a10d │ A10-cont    │ 32  │ autocorr → peaks → pairwise ratios → 32-bin uniform histogram │ Ratio-native, ontology-free        │
 ├──────┼─────────────┼─────┼───────────────────────────────────────────────────────────────┼────────────────────────────────────┤
 │ a10e │ A10-cont    │ 32  │ RQA diag profile → peaks → pairwise ratios → 32-bin histogram │ Non-linear, ontology-free          │
 └──────┴─────────────┴─────┴───────────────────────────────────────────────────────────────┴────────────────────────────────────┘

 Gate 9 suffix convention: a10ar, a10br, a10cr, a10dr, a10er

 A10-cont key design: Same pairwise-ratio machinery as A10-JI (scale-invariant, no pitch/register confound), but final step is a weighted histogram into 32
  uniform bins in log2-folded [0,1) instead of Gaussian assignment to 12 specific JI attractors. The network decides where the signal is.

 32 bins rationale: bin width = 1/32 = 0.03125 in log2 space. Finer than min gap between adjacent JI attractors (0.059 for M3↔m3). Coarse enough to avoid
 sparsity issues with C(8,2)=28 pairs per frame. Sweet spot.

 Scientific Comparisons Enabled

 ┌───────────────────┬─────────────────────────────────────────────────────────────────────────────┐
 │    Comparison     │                                What it tests                                │
 ├───────────────────┼─────────────────────────────────────────────────────────────────────────────┤
 │ A10a vs A10d      │ Do 12 JI attractors help or hurt? (same autocorr, JI vs free histogram)     │
 ├───────────────────┼─────────────────────────────────────────────────────────────────────────────┤
 │ A10b vs A10e      │ Same question, non-linear measurement                                       │
 ├───────────────────┼─────────────────────────────────────────────────────────────────────────────┤
 │ A10d vs A10e      │ Does non-linearity matter? (same 32-bin projection, linear vs RQA)          │
 ├───────────────────┼─────────────────────────────────────────────────────────────────────────────┤
 │ A10d/A10e vs A10c │ Do ratios matter at all? (ratio histogram vs generic RQA metrics)           │
 ├───────────────────┼─────────────────────────────────────────────────────────────────────────────┤
 │ A10d vs A7r       │ Temporal recurrence vs spectral peaks (both ratio-based, different domains) │
 └───────────────────┴─────────────────────────────────────────────────────────────────────────────┘

 ---
 Phase 1: Core Descriptor Functions

 File: src/bias_control/audio_descriptors.py

 1A. New constant + histogram helper (after line 154)

 A10_CONT_NBINS = 32  # ontology-free uniform bins in log2-folded [0,1)

 New helper _weighted_ratio_histogram(log2_r, weight, n_bins=32):
 - Input: log2_r [*, n_pairs] (octave-folded ratios), weight [*, n_pairs]
 - bin_idx = (log2_r * n_bins).long().clamp(0, n_bins - 1)
 - result.scatter_add_(-1, bin_idx, weight) → [*, n_bins]
 - Shared by A10d and A10e

 1B. compute_audio_descriptor_a10d (insert after line 829, after A10a)

 Structurally identical to A10a except:
 - Lines 808-814 (Gaussian → 12 JI) replaced by _weighted_ratio_histogram(log2_r, weight, 32) → [B, T, 32]
 - All shape fallbacks: 32 instead of 12
 - Same energy gate, normalization, interpolation

 1C. Memory chunking for A10b (lines 890-953)

 Replace monolithic torch.cdist(emb_flat, emb_flat) with chunked processing:

 CDIST_CHUNK = 512  # ~28MB per chunk at N_vec=126
 all_results = []
 for start in range(0, BT, CDIST_CHUNK):
     end = min(start + CDIST_CHUNK, BT)
     chunk_emb = emb_flat[start:end]          # [chunk_sz, N_vec, m]
     dists = torch.cdist(chunk_emb, chunk_emb) # [chunk_sz, N_vec, N_vec]
     # ... threshold → RP → diagonal profile → peaks → ratios → JI assignment ...
     all_results.append(chunk_result)
 result = torch.cat(all_results, dim=0)  # [BT, 12]

 The chunk loop encompasses cdist → threshold → RP → diagonal profile → peak detection → pairwise ratios → JI assignment. All operations are per-frame (no
 cross-frame dependencies), so chunking is correct.

 Memory: 512 × 126 × 126 × 4 bytes = ~32MB per chunk vs ~707MB monolithic.

 1D. compute_audio_descriptor_a10e (insert after A10b)

 Same as modified A10b (with chunking) except final step = histogram instead of JI Gaussian. Same chunk infrastructure.

 1E. Fix _batched_rqa_metrics entropy truncation (line 1020)

 # Before:
 max_offset = min(N - 1, 64)  # cap to avoid excessive computation
 # After:
 max_offset = N - 1  # full offset range (consistent with A10b diagonal profile)

 One-line fix. N typically = 126, so range goes from 64 to 125 offsets. ~2× cost in that loop, acceptable.

 Note: This changes A10c output values. Any prior A10c results are invalidated.

 1F. Memory chunking for A10c (lines 1097-1110)

 Same chunk pattern as 1C, but the inner loop calls _batched_rqa_metrics(chunk_rp) per chunk instead of computing diagonal profile + peaks.

 CDIST_CHUNK = 512
 results_list = []
 for start in range(0, BT, CDIST_CHUNK):
     end = min(start + CDIST_CHUNK, BT)
     chunk_emb = emb_flat[start:end]
     chunk_dists = torch.cdist(chunk_emb, chunk_emb)
     # ... threshold → RP ...
     chunk_result = _batched_rqa_metrics(chunk_rp)  # [chunk_sz, 6]
     results_list.append(chunk_result)
 result = torch.cat(results_list, dim=0)  # [BT, 6]

 ---
 Phase 2: Escalón 2 Wrappers

 File: src/bias_control/vocal_descriptors.py

 2A. Add imports (extend existing import block, ~line 512)

 from src.bias_control.audio_descriptors import (
     compute_audio_descriptor_a10d as _a10d_core,
     compute_audio_descriptor_a10e as _a10e_core,
 )

 2B. Add compute_a10d and compute_a10e (after line 581)

 Same wrapper pattern as compute_a10a/compute_a10b:
 - compute_a10d(waveform, target_length, n_fft=1024, hop_length=160, sr=16000) → calls _a10d_core
 - compute_a10e(waveform, target_length, n_fft=1024, hop_length=160, sr=16000, stride=8) → calls _a10e_core

 2C. Update DESCRIPTOR_DIMS (lines 588-596)

 Add: 'a10d': 32, 'a10e': 32

 ---
 Phase 3: Training Script Integration (Escalón 2)

 3A. experiments/bias_control/escalon2/train_escalon2_descriptors.py

 1. Import (add to existing imports): compute_a10d, compute_a10e from vocal_descriptors
 2. DescriptorComputer.call (after line 184): Two new elif branches dispatching to compute_a10d/compute_a10e (waveform-based, not F0-based)
 3. Argparse choices (line 397): Add 'a10d', 'a10e'

 3B. experiments/bias_control/escalon2/train_escalon2_attn.py

 1. Default descriptor warning (line 321-324): Change from logger.info to logger.warning when auto-defaulting:
 if args.descriptor is None:
     args.descriptor = INJECTION_DEFAULT_DESCRIPTOR[args.injection]
     logger.warning(f"⚠ No --descriptor specified! Auto-defaulting to '{args.descriptor}'. "
                    f"For A10 runs, always pass --descriptor explicitly.")
 2. Argparse choices (line 289): Add 'a10d', 'a10e'

 ---
 Phase 4: Gate 9 Integration

 4A. experiments/bias_control/gate43_scratch/gate43_scratch_training.py (8 edits)

 ┌─────┬───────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────┐
 │  #  │                         Location                          │                                    Edit                                    │
 ├─────┼───────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
 │ 1   │ Imports (~line 541)                                       │ Add compute_audio_descriptor_a10d, compute_audio_descriptor_a10e           │
 ├─────┼───────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
 │ 2   │ _encode_audio_with_reverse_cross_attention() (~line 1365) │ Add 2 elif for 'a10d', 'a10e'                                              │
 ├─────┼───────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
 │ 3   │ create_gate42_model() (~line 2532)                        │ a10dr→(a10d,32), a10er→(a10e,32)                                           │
 ├─────┼───────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
 │ 4   │ GATE42_PARAM_RANGES run-b (~line 2976)                    │ a10dr: (39M,46M), a10er: (39M,46M)                                         │
 ├─────┼───────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
 │ 5   │ GATE42_PARAM_RANGES run-d (~line 3013)                    │ a10dr: (64M,72M), a10er: (64M,72M)                                         │
 ├─────┼───────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
 │ 6   │ Trainable prefixes (~line 3092)                           │ Extend tuple to include 'a10dr', 'a10er'                                   │
 ├─────┼───────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
 │ 7   │ Eval reconstruction + eval batch size + argparse          │ Extend existing a10ar/br/cr branches to include a10dr/er with dim dispatch │
 └─────┴───────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────┘

 Eval reconstruction (~line 4005): Extend dict-based dispatch:
 elif descriptor in ('a10ar', 'a10br', 'a10cr', 'a10dr', 'a10er'):
     ad_type = {'a10ar': 'a10a', 'a10br': 'a10b', 'a10cr': 'a10c',
                'a10dr': 'a10d', 'a10er': 'a10e'}[descriptor]
     ad_dim = {'a10ar': 12, 'a10br': 12, 'a10cr': 6,
               'a10dr': 32, 'a10er': 32}[descriptor]

 4B. experiments/bias_control/gate5b/checkpoint_loader.py (2 edits)

 1. Batch sizes (line 33): Add 'a10dr': 16, 'a10er': 16
 2. Reconstruction (~line 175): Extend to include a10dr/a10er with dim dispatch (same pattern as 4A)

 ---
 Phase 5: Verification

 File: experiments/bias_control/escalon2/verify_p25.py

 5A. New imports

 Add compute_a10d, compute_a10e to vocal_descriptors import.

 5B. New tests (5 tests: 15-19)

 ┌──────┬──────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────┐
 │ Test │           Purpose            │                                    Key assertion                                     │
 ├──────┼──────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
 │ 15   │ VRAM stress xattn dim=12     │ B=64, T=800: no OOM, forward+backward completes                                      │
 ├──────┼──────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
 │ 16   │ VRAM stress xattn dim=32     │ B=64, T=800: no OOM, forward+backward completes                                      │
 ├──────┼──────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
 │ 17   │ A10d/A10e shape + finiteness │ Call compute_a10d/a10e on synthetic [2, 32000] 16kHz. Shape = (2, T, 32), all finite │
 ├──────┼──────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
 │ 18   │ Identity bypass xattn dim=32 │ descriptor=None → output unchanged                                                   │
 ├──────┼──────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
 │ 19   │ Near-identity xattn dim=32   │ rel_diff < 0.05                                                                      │
 └──────┴──────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────┘

 5C. Update test list in main()

 Add tests 15-19 to the execution list.

 ---
 Implementation Order

 Phase 1 (audio_descriptors.py)     ← all downstream depends on this
     1A → 1E → 1C → 1F → 1B → 1D
          ↓
 Phase 2 (vocal_descriptors.py)     ← wrappers for E2
          ↓
 Phase 3 (train_escalon2_*.py)      ← dispatch + warning
 Phase 4 (gate43 + checkpoint)      ← Gate 9 wiring (independent of Phase 3)
          ↓
 Phase 5 (verify_p25.py)            ← tests everything

 Verification Strategy

 1. Preflight: python experiments/bias_control/escalon2/verify_p25.py — all 19 tests PASS
 2. Shape smoke (manual): compute_audio_descriptor_a10d(torch.randn(2, 48000).cuda()) → shape (2, T, 32)
 3. Memory regression: A10b/A10c with B=64 on GPU without OOM
 4. Bitwise equivalence: A10a output unchanged (no modifications). A10b output unchanged (chunking = same math)
 5. A10c entropy change: Document that values change intentionally (offset range 64→125)
 6. 1-epoch smoke: a10dr and a10er in Gate 9; a10d-xattn and a10e-xattn in Escalón 2

 Files Modified

 ┌────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────┬───────┐
 │                                File                                │                            Changes                             │ Phase │
 ├────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┼───────┤
 │ src/bias_control/audio_descriptors.py                              │ +constant, +helper, +A10d, +A10e, chunk A10b/A10c, fix entropy │ 1     │
 ├────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┼───────┤
 │ src/bias_control/vocal_descriptors.py                              │ +2 wrappers, update DESCRIPTOR_DIMS                            │ 2     │
 ├────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┼───────┤
 │ experiments/bias_control/escalon2/train_escalon2_descriptors.py    │ +2 dispatch branches, +2 choices                               │ 3     │
 ├────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┼───────┤
 │ experiments/bias_control/escalon2/train_escalon2_attn.py           │ +2 choices, warning on default                                 │ 3     │
 ├────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┼───────┤
 │ experiments/bias_control/gate43_scratch/gate43_scratch_training.py │ +import, +dispatch, +factory, +ranges, +eval, +argparse        │ 4     │
 ├────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┼───────┤
 │ experiments/bias_control/gate5b/checkpoint_loader.py               │ +batch sizes, +reconstruction                                  │ 4     │
 ├────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┼───────┤
 │ experiments/bias_control/escalon2/verify_p25.py                    │ +5 tests (15-19), +imports                                     │ 5     │
 └────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────┴───────┘
