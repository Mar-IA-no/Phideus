# PHIDEUS: A Research Program on Proportional Representations for Cross-Modal Learning

**Research Program Document**
**Version 1.0 - January 2026**

---

## Executive Summary

PHIDEUS is a research program investigating whether **frequency ratio distributions** can serve as a portable representation for learning transferable structure across sensory modalities.

Our central research question is:

> *To what extent can proportional structure—the pattern of ratios among oscillatory components—be learned, transferred, and reasoned upon independently of the physical substrate that generates it?*

This document presents:
1. The theoretical motivation and hypotheses
2. Experimental results to date
3. A roadmap for progressive validation
4. The long-term research vision

We distinguish clearly between **demonstrated results**, **working hypotheses**, and **speculative vision** to maintain scientific rigor while articulating the broader research direction.

---

## 1. Motivation: Why Proportions?

### 1.1 The Observation

Multiple scientific domains have independently converged on a similar insight: **relational structure among oscillatory components carries robust, transferable information**.

| Domain | Observation |
|--------|-------------|
| **Mechanical engineering** | Order tracking normalizes vibration spectra by rotation speed, revealing speed-invariant fault signatures |
| **Music cognition** | Interval recognition is transposition-invariant; a perfect fifth is recognized regardless of absolute pitch |
| **Neuroscience** | Cross-frequency coupling (theta-gamma, alpha-theta) correlates with cognitive states |
| **Ecoacoustics** | Healthy ecosystems show structured frequency partitioning; degraded systems show random overlap |

These observations suggest that **ratios may encode structure that is more fundamental than absolute values** in oscillatory systems.

### 1.2 The Research Question

We ask: Can this insight be operationalized into a general-purpose representation for machine learning?

Specifically:
- Can we build **ratio-based descriptors** that are computable from any oscillatory signal?
- Do these descriptors enable **cross-modal learning** where structure transfers between different sensor types?
- If so, what are the **limits and conditions** for such transfer?

### 1.3 What This Is Not

To be clear about scope:

- This is **not** a claim that ratios are the only important information in signals.
- This is **not** a claim that ratio-based methods will outperform all alternatives on all tasks.
- This is **not** a finished theory, but a **research program** with testable hypotheses.

---

## 2. Theoretical Framework

### 2.1 The Proportional Descriptor

We represent oscillatory signals as **enriched ratio histograms**:

1. **Extract spectral peaks** from the signal (via STFT, wavelet, or domain-appropriate method)
2. **Compute pairwise ratios** between peak frequencies: $r_{ij} = f_j / f_i$
3. **Weight by amplitude**: $w_{ij} = \sqrt{A_i \cdot A_j}$
4. **Bin into histogram**: Fixed-size representation $H \in \mathbb{R}^{B}$
5. **Enrich with channels**: Add energy and entropy per bin → $H \in \mathbb{R}^{B \times 3}$

**Key properties**:
- **Scale-invariant**: Ratios are unchanged by transposition or unit conversion
- **Fixed-size**: Any signal maps to the same tensor shape
- **Domain-agnostic**: Computable from any signal with extractable oscillatory components

### 2.2 The Factorized Latent Space

For cross-modal learning, we employ VAEs with **factorized latent representations**:

$$z = [z_{shared}, z_{private}]$$

- $z_{shared}$: Intended to capture cross-modal structure
- $z_{private}$: Captures modality-specific information (sensor characteristics, noise, etc.)

Training uses:
- **Reconstruction loss**: Each modality should reconstruct its own input
- **Contrastive alignment (InfoNCE)**: Synchronized observations should have similar $z_{shared}$
- **KL regularization**: Standard VAE prior

### 2.3 Core Hypotheses

We organize our research around three hypotheses, stated precisely enough to be testable:

---

**H1: Ratio Structure Exists**

*Claim*: Real-world oscillatory signals contain non-random ratio structure that can be captured in histogram representations.

*Testable prediction*: Ratio histograms from real signals should differ systematically from those computed from noise or randomly permuted spectra.

*Status*: **Supported** by synthetic validation and real-data experiments showing meaningful histogram structure.

---

**H2: Ratio Structure is Learnable**

*Claim*: Neural networks can learn compressed latent representations of ratio histograms that preserve meaningful structure.

*Testable prediction*: VAEs trained on ratio histograms should achieve low reconstruction error and produce latent spaces where similar signals cluster together.

*Status*: **Supported** by VAE experiments achieving val_loss < 0.5 on temporal ratio data (Analizador 5.0 experiments).

---

**H3: Ratio Structure Transfers Across Modalities**

*Claim*: When two modalities observe the same physical phenomenon, their ratio representations share learnable common structure.

*Testable prediction*: A model trained to align modalities via $z_{shared}$ should achieve high similarity (cos_sim > 0.7) between modality embeddings, and enable cross-modal prediction (Pearson > 0.7).

*Status*: **Supported for one modality pair** (audio ↔ vibration) by the Roseta 1 experiment. Generalization to other pairs is **untested**.

---

### 2.4 What Would Falsify These Hypotheses?

| Hypothesis | Falsification criterion |
|------------|------------------------|
| H1 | Ratio histograms from real signals are indistinguishable from noise baselines |
| H2 | VAEs fail to converge or produce latent spaces with no meaningful structure |
| H3 | Cross-modal alignment fails (cos_sim < 0.5) even with synchronized training data |

---

## 3. Experimental Results

### 3.1 Experiment Series: Analizador 5.0 (Single-Modal)

**Objective**: Validate that ratio histograms are a learnable representation for audio signals.

**Setup**:
- 848 synthetic audio files with controlled harmonic content
- Temporal ratio histograms (linear scale, 3 channels)
- Four architecture variants tested (VAE/HRM × Temporal/Static)

**Results**:

| Architecture | Val Loss | Parameters |
|--------------|----------|------------|
| VAE Temporal | **0.4560** | 1.82M |
| HRM Temporal | 0.4607 | 2.27M |
| HRM Static | 0.5906 | 854K |
| VAE Static | 0.5997 | 838K |

**Key finding**: With appropriate data representation (linear scale + temporal), both VAE and HRM architectures achieve comparable performance. The representation matters more than the architecture.

**Supports**: H1 (structure exists), H2 (structure is learnable)

---

### 3.2 Experiment: Roseta 1 (Cross-Modal)

**Objective**: Test whether ratio structure transfers between audio and vibration modalities observing the same physical phenomenon.

**Setup**:
- University of Ottawa Electric Motor Dataset (UOEMD)
- 128 synchronized recordings (audio + vibration)
- 8 operating conditions (1 healthy + 7 fault types)
- RosetaVAE: Dual-encoder VAE with factorized latent space
- InfoNCE contrastive loss for cross-modal alignment

**Results**:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Cross-modal cos_sim | **0.766 ± 0.002** | High alignment consistency |
| ANOVA p-value (across conditions) | 0.548 | No significant condition effect |
| Cross-retrieval Pearson | **0.754 - 0.763** | Successful cross-modal prediction |
| InfoNCE effect size (Cohen's d) | **5.75** | Very large effect |

**Key findings**:
1. Audio and vibration representations align consistently across all 8 conditions
2. The alignment captures physical structure, not condition-specific artifacts
3. Cross-modal prediction achieves Pearson > 0.75

**Supports**: H3 (ratios transfer) **for this specific modality pair**

---

### 3.3 Limitations of Current Evidence

We are explicit about what has **not** been demonstrated:

| Claim | Status |
|-------|--------|
| Ratios transfer between audio and vibration | **Demonstrated** |
| Ratios transfer between any two modalities | **Not tested** |
| Ratio methods outperform spectral baselines | **Not compared** |
| Results generalize to non-laboratory conditions | **Not tested** |
| Results generalize to other physical phenomena | **Not tested** |

These limitations define our near-term research agenda.

---

## 4. Research Roadmap

### 4.1 Immediate Priorities (Validation Phase)

The next experiments are designed to **stress-test H3** by varying the modality pair, physical phenomenon, and data conditions.

---

**Roseta 2: Audio → Visual (Lissajous Patterns)**

*Rationale*: Test transfer between fundamentally different sensor types (microphone → camera). If ratios transfer here, it strengthens the generalization claim significantly.

*Setup*:
- Generate audio tones with controlled frequency ratios
- Capture corresponding Lissajous patterns via laser/mirror cymatics
- Train cross-modal VAE (audio ratio histogram ↔ visual pattern descriptor)

*Success criterion*: cos_sim > 0.6 between audio and visual embeddings

*Failure interpretation*: Ratio transfer may be limited to physically-coupled modalities (audio-vibration share mechanical transmission path; audio-visual do not)

---

**Roseta 3: Physiological Signals**

*Rationale*: Test in a completely different physical domain (biological vs. mechanical).

*Setup*:
- EEG, ECG, respiration recordings from public datasets
- Compute ratio histograms for each modality
- Test cross-modal alignment

*Success criterion*: cos_sim > 0.5 between at least one modality pair

*Failure interpretation*: Ratio transfer may require stronger physical coupling than exists between physiological systems

---

**Baseline Comparisons**

*Rationale*: Determine whether ratio representations offer advantages over alternatives.

*Setup*:
- Same UOEMD dataset
- Compare ratio histograms vs: raw spectrograms, MFCC, wav2vec embeddings
- Evaluate on: cross-modal alignment, downstream classification, robustness to noise

*Success criterion*: Ratio representations competitive or superior on at least one metric

*Failure interpretation*: Ratios may be sufficient but not advantageous; simpler methods may be preferable

---

### 4.2 Medium-Term Goals (Extension Phase)

If validation experiments succeed:

1. **Multi-modal extension**: Scale from 2 to N modalities in shared latent space
2. **Temporal dynamics**: Extend from static alignment to trajectory modeling
3. **Real-world deployment**: Test on field data with noise, sensor drift, missing data
4. **Theoretical formalization**: Develop mathematical framework for ratio-based information measures

### 4.3 Decision Points

We define explicit decision points to maintain research discipline:

| After Roseta 2 | Decision |
|----------------|----------|
| cos_sim > 0.6 | Proceed to Roseta 3 and multi-modal extension |
| cos_sim 0.4-0.6 | Investigate failure modes; consider modality-specific adaptations |
| cos_sim < 0.4 | Revise H3 to specify conditions for transfer; pivot to within-domain applications |

---

## 5. Long-Term Vision

*This section describes our research aspirations. These are not claims but directions we aim to explore, contingent on experimental validation.*

### 5.1 The Broader Question

If ratio-based representations do enable cross-modal learning across diverse domains, this suggests a deeper question:

> *Is proportional structure a general "language" for encoding information about oscillatory systems?*

This is not a claim we can currently support—it is the **horizon we are investigating toward**.

### 5.2 Potential Implications

**If H3 generalizes broadly**, it would suggest:

- **Unified representations**: A single descriptor type could encode structure from audio, vibration, EEG, imagery, electromagnetic signals
- **Transfer learning**: Models trained on one domain could bootstrap learning in others
- **Interpretability**: Ratios have physical meaning (resonance, periodicity), potentially enabling more interpretable models than black-box embeddings

**If H3 has narrow scope**, it would still yield:

- **Domain-specific tools**: Ratio representations for physically-coupled modalities (sensor fusion for machinery, audio-visual correspondence)
- **Theoretical insight**: Understanding *why* transfer works in some cases and not others

Both outcomes advance scientific understanding.

### 5.3 The PHIDEUS Vision

Our long-term aspiration—which we present as vision, not claim—is to build systems that can:

1. **Sense** the world through diverse modalities
2. **Encode** signals into proportional descriptors
3. **Learn** shared structure across domains
4. **Reason** about patterns, anomalies, and relationships
5. **Communicate** findings in interpretable terms

We call this vision "PHIDEUS" (from Greek φειδεύς, "one who reads proportions").

The vision includes:
- Distributed sensor networks producing ratio descriptors
- Multi-domain latent spaces aligning diverse signals
- AI systems that query and reason over proportional structure

**This vision is speculative**. It is not supported by current evidence. It is the direction we are working toward, and we will update it based on experimental results.

---

## 6. Relationship to Existing Work

### 6.1 What We Build On

| Area | Contribution we use |
|------|---------------------|
| Contrastive learning (SimCLR, CLIP, CMC) | InfoNCE loss for cross-modal alignment |
| Multimodal VAEs (JMVAE, MVAE, DMVAE) | Factorized latent spaces (shared + private) |
| Order tracking (mechanical engineering) | Speed-invariant spectral analysis via ratios |
| Cross-frequency coupling (neuroscience) | Ratio-based characterization of oscillatory relationships |

### 6.2 What We Contribute

| Contribution | Status |
|--------------|--------|
| Enriched ratio histogram representation | Implemented, validated |
| Temporal ratio modeling (VAE, HRM) | Implemented, validated |
| Cross-modal alignment via ratio space | Demonstrated for audio ↔ vibration |
| Multi-domain generalization | Hypothesized, untested |

### 6.3 Open Questions

1. **Representation comparison**: How do ratio histograms compare to spectrograms, wavelets, learned embeddings?
2. **Transfer limits**: Under what conditions does cross-modal ratio alignment succeed or fail?
3. **Theoretical foundations**: Can we formalize "ratio-based information" mathematically?
4. **Scalability**: Does alignment quality degrade as we add more modalities?

---

## 7. Summary

### What We Have Demonstrated

1. **Ratio histograms are a learnable representation** for audio signals (Analizador 5.0 experiments)
2. **Cross-modal alignment works for audio ↔ vibration** with cos_sim > 0.76 (Roseta 1)
3. **The alignment generalizes across operating conditions** and enables cross-modal prediction

### What We Hypothesize

1. **H1**: Real-world signals contain meaningful ratio structure (supported)
2. **H2**: This structure is learnable by neural networks (supported)
3. **H3**: This structure transfers across modalities (supported for one pair; generalization untested)

### What We Aspire To

A research program that progressively validates (or refines) the scope of ratio-based cross-modal learning, potentially leading to unified representations for diverse oscillatory signals.

### What We Do Not Claim

- That ratios are the best representation for all tasks
- That our current results generalize beyond the tested conditions
- That the long-term vision is achievable

---

## 8. Technical Appendix

### A.1 Ratio Histogram Specification

```
Input: Signal x(t)
Output: H ∈ ℝ^{B × 3}

1. STFT: X(f,t) = STFT(x, window=4096, hop=1024)
2. Peaks: P = {(f_i, A_i)} = detect_peaks(|X|, threshold=1.25×median)
3. Ratios: R = {(r_ij, w_ij)} where r_ij = f_j/f_i, w_ij = √(A_i·A_j)
4. Histogram: h[b] = Σ w_ij · 𝟙[r_ij ∈ bin_b]
5. Channels:
   - c0 (proportion): p[b] = h[b] / Σh
   - c1 (moment): m[b] = h[b]·b² / Σ(h·b²)
   - c2 (entropy): e[b] = -p[b]·log(p[b]) / Σe
6. Output: H[b,c] = [p, m, e]
```

### A.2 RosetaVAE Architecture

```
Encoder (per modality):
  Input: [B, T, 256, 3]
  → Flatten: [B, T, 768]
  → Linear(768→128) + LayerNorm + GELU + Dropout(0.1)
  → BiLSTM(128→256, layers=2, dropout=0.1)
  → z_shared: Linear(256→32) μ, Linear(256→32) σ
  → z_private: Linear(256→16) μ, Linear(256→16) σ

Decoder (per modality):
  Input: [z_shared, z_private] = [B, T, 48]
  → Linear(48→128) + LayerNorm + GELU
  → BiLSTM(128→256, layers=2)
  → Linear(256→768) → Reshape → Softmax(dim=bins)

Loss:
  L = L_recon_A + L_recon_V + β·L_KL + λ·L_InfoNCE
  where λ=2.0, β=1.0, τ=0.07

Parameters: 3,161,536
```

### A.3 Experimental Results Summary

| Experiment | Metric | Value | Interpretation |
|------------|--------|-------|----------------|
| Analizador 5.0 | VAE val_loss | 0.456 | Good reconstruction |
| Analizador 5.0 | HRM val_loss | 0.461 | Comparable to VAE |
| Roseta 1 | cos_sim | 0.766 | High cross-modal alignment |
| Roseta 1 | cos_sim σ | 0.002 | Consistent across conditions |
| Roseta 1 | Pearson (cross-retrieval) | 0.75+ | Successful cross-modal prediction |
| Roseta 1 | Cohen's d (InfoNCE effect) | 5.75 | Very large effect size |

---

## 9. References

[1] van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation Learning with Contrastive Predictive Coding. arXiv:1807.03748.

[2] Tian, Y., Krishnan, D., & Isola, P. (2020). Contrastive Multiview Coding. ECCV.

[3] Lee, M., & Pavlovic, V. (2021). Private-Shared Disentangled Multimodal VAE for Learning of Latent Representations. CVPR Workshops.

[4] Fyfe, K.R., & Munck, E.D.S. (1997). Analysis of Computed Order Tracking. Mechanical Systems and Signal Processing.

[5] Canolty, R.T., & Knight, R.T. (2010). The Functional Role of Cross-Frequency Coupling. Trends in Cognitive Sciences.

[6] University of Ottawa Electric Motor Dataset. https://data.mendeley.com/datasets/msxs4vj48g/1

---

## 10. Contact and Collaboration

PHIDEUS is an open research program. We welcome:
- Collaborators with relevant datasets (multi-modal synchronized recordings)
- Domain experts in areas where ratio analysis may be applicable
- Critical feedback on methodology and interpretation

---

*Document prepared: January 2026*
*Status: Active research program*

---

**Acknowledgment**

We maintain a clear distinction between what we have demonstrated, what we hypothesize, and what we envision. This document will be updated as experiments progress and evidence accumulates.

*"The measure of a research program is not the boldness of its vision, but the rigor of its validation."*
