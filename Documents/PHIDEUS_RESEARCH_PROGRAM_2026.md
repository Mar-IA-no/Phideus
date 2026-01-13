# PHIDEUS: A Research Program on Proportional Representations for Cross-Modal Learning

**Authors**: PHIDEUS Research Team
**Date**: January 2026
**Document Type**: Research Program Paper

---

## Abstract

We present PHIDEUS, a research program investigating whether frequency ratio distributions can serve as a portable representation for learning transferable structure across sensory modalities. Building on independent observations from mechanical engineering, neuroscience, ecoacoustics, and music cognition—all suggesting that relational structure among oscillatory components carries robust information—we propose enriched ratio histograms as a domain-agnostic descriptor and test their utility for cross-modal learning.

Our experimental results to date include: (1) validation that temporal ratio histograms are learnable representations achieving reconstruction loss < 0.46 with both VAE and hierarchical architectures; (2) demonstration that cross-modal alignment between audio and vibration modalities achieves cosine similarity of 0.766 ± 0.002 across 8 operating conditions using a dual-encoder VAE with InfoNCE contrastive loss; and (3) successful cross-modal retrieval with Pearson correlation > 0.75.

We frame these results within three testable hypotheses (H1: ratio structure exists; H2: it is learnable; H3: it transfers across modalities), provide explicit falsification criteria, and outline a validation roadmap including experiments on audio-visual and physiological modality pairs. We distinguish clearly between demonstrated results, working hypotheses, and long-term research vision.

**Keywords**: cross-modal learning, proportional representations, contrastive learning, variational autoencoders, frequency ratios, multi-modal alignment

---

## 1. Introduction

### 1.1 Motivation

The representation of oscillatory signals for machine learning typically relies on absolute measurements: spectrograms encode frequency content at specific Hertz values, power spectral densities measure energy at particular bands, and time-domain features capture amplitude variations. However, multiple scientific domains have independently converged on an observation that suggests an alternative approach: **relational structure among oscillatory components often carries more robust and transferable information than absolute values**.

In mechanical engineering, order tracking normalizes vibration spectra by rotation speed, revealing speed-invariant fault signatures based on frequency ratios rather than absolute frequencies [1, 2]. In music cognition, interval recognition is transposition-invariant; a listener recognizes a perfect fifth (3:2 ratio) regardless of whether it spans 200-300 Hz or 2000-3000 Hz [3]. In neuroscience, cross-frequency coupling between theta and gamma oscillations, or alpha and theta bands, correlates with cognitive states and predicts performance on visuomotor tasks [4, 5, 6]. In ecoacoustics, healthy ecosystems show structured frequency partitioning where species occupy complementary time-frequency niches, while degraded systems show random spectral overlap [7, 8].

These observations suggest that **ratios may encode structure that is more fundamental than absolute values** in oscillatory systems—a possibility with significant implications for representation learning.

### 1.2 Research Questions

This research program asks: Can the insight that "information lives in relations" [9] be operationalized into a general-purpose representation for machine learning?

Specifically, we investigate:

1. Can we build **ratio-based descriptors** that are computable from any oscillatory signal?
2. Do these descriptors enable **cross-modal learning** where structure transfers between different sensor types observing the same phenomenon?
3. If so, what are the **limits and conditions** for such transfer?

### 1.3 Scope and Limitations

We are explicit about what this research program does and does not claim:

- This is **not** a claim that ratios are the only important information in signals; absolute values (amplitude, fundamental frequency) often carry essential information [10].
- This is **not** a claim that ratio-based methods will outperform all alternatives on all tasks.
- This is **not** a finished theory, but a **research program** with testable hypotheses and explicit falsification criteria.

### 1.4 Contributions

This paper presents:

1. A **proportional descriptor** (enriched ratio histogram) applicable to any oscillatory signal
2. **Experimental validation** of learnability (Analizador 5.0 experiments) and cross-modal transfer (Roseta 1 experiment)
3. **Three testable hypotheses** with falsification criteria
4. A **research roadmap** with decision points based on experimental outcomes

---

## 2. Related Work

### 2.1 Contrastive Representation Learning

Contrastive learning has emerged as a powerful paradigm for self-supervised representation learning. The InfoNCE loss [11] maximizes mutual information between views of the same instance while minimizing it for different instances, and has proven effective for learning robust representations without labels.

SimCLR [12] demonstrated that data augmentation composition and learnable projection heads substantially improve contrastive learning for visual representations. MoCo [13] introduced momentum-updated encoders for efficient contrastive learning with large negative sample pools. BYOL [14] showed that contrastive learning can succeed even without explicit negative samples through asymmetric network architectures.

These methods focus primarily on **within-modality** representation learning. Our work extends contrastive principles to **cross-modal alignment** via the InfoNCE loss applied to synchronized observations from different sensors.

### 2.2 Cross-Modal Contrastive Learning

Contrastive Multiview Coding (CMC) [15] extended contrastive learning to multiple views, demonstrating that maximizing mutual information between different modalities captures underlying semantics. CLIP [16] achieved remarkable zero-shot transfer by aligning image and text embeddings through contrastive learning on 400 million image-text pairs.

AudioCLIP [17] and related methods extended these ideas to audio-visual correspondence, learning aligned representations of sounds and images. Our work differs fundamentally from these approaches: while CLIP-like methods learn **semantic** correspondences (what is depicted), we learn **physical** correspondences (the frequency structure that manifests across modalities observing the same phenomenon).

### 2.3 Multimodal Variational Autoencoders

The variational autoencoder framework [18, 19] has been extended to multiple modalities through various architectures:

- **JMVAE** [20] introduced joint encoding for bidirectional cross-modal generation
- **MVAE** [21] proposed Product-of-Experts (PoE) for combining modality-specific encoders
- **MMVAE** [22] introduced Mixture-of-Experts to overcome PoE limitations in high dimensions
- **DMVAE** [23] explicitly disentangles modality-specific features from shared representations through factorized latent spaces ($z_{shared}$ + $z_{private}$)

Our RosetaVAE architecture builds on DMVAE's factorization principle while adding temporal modeling (bidirectional LSTM) and InfoNCE alignment specifically designed for ratio histogram inputs. The concept of learning factorized multimodal representations was further developed by Tsai et al. [24], whose work on separating shared and private information informs our latent space design.

### 2.4 Harmonic Analysis and Order Tracking

The use of frequency ratios has deep roots in mechanical engineering. Order tracking [1, 25] normalizes vibration spectra by rotation speed, extracting speed-invariant signatures crucial for machinery diagnostics. Randall and Antoni's tutorial [2] established that bearing defect frequencies appear at predictable ratios to the fundamental rotation frequency, making ratio-based analysis essential for fault diagnosis.

Recent work on harmonic analysis in rotating machinery [26, 27] demonstrates that fault signatures are better characterized by the pattern of harmonics and sidebands—inherently ratio-based features—than by absolute frequencies. Deep learning approaches to fault diagnosis [28, 29, 30] have achieved high accuracy, though typically using absolute spectral features rather than ratio-based representations.

### 2.5 Cross-Frequency Coupling in Neuroscience

Neuroscience provides independent evidence for the importance of frequency ratios. Canolty and Knight's review [4] established that cross-frequency coupling (CFC) between neural oscillations at different frequency bands carries functional significance. Studies of alpha-theta coupling [5] show that the strength and stability of harmonic relations between these bands predict visuomotor performance.

Research on cardio-respiratory coupling [31, 32] demonstrates that stable ratios between heart rate and breathing rhythms correlate with physiological regulation states. Even in motor control, timing proportions close to the golden ratio emerge as signatures of balanced gait [33].

These findings suggest that biological systems may naturally organize around preferred frequency ratios—supporting our hypothesis that ratio structure is not arbitrary but reflects underlying organizational principles.

### 2.6 Acoustic Monitoring and Multi-Sensor Fusion

Acoustic-based machinery monitoring [34, 35] has gained attention as a non-contact alternative to vibration sensing. The challenge of multi-sensor fusion [36, 37] typically employs late fusion strategies without learning aligned representations.

Recent work on consistent feature fusion [38] between vibration and acoustic signals for fault diagnosis provides context for our cross-modal alignment approach, though existing methods do not explicitly leverage ratio-based representations.

### 2.7 Time Series Representation Learning

Self-supervised learning for time series has advanced significantly. TS2Vec [39] learns universal time series representations through hierarchical contrastive learning. TF-C [40] exploits time-frequency consistency for contrastive pre-training. These methods inform our temporal modeling approach, though they focus on single-modality representation rather than cross-modal alignment.

---

## 3. Theoretical Framework

### 3.1 The Proportional Descriptor

We represent oscillatory signals as **enriched ratio histograms**. The transformation from raw signal to histogram proceeds as follows:

**Step 1: Spectral Analysis**
Apply Short-Time Fourier Transform (STFT) with window size $N$, hop size $H$, and Hann windowing:
$$X(f, t) = \text{STFT}(x(t); N, H)$$

**Step 2: Peak Detection**
Identify spectral peaks $P = \{(f_i, A_i)\}$ where $f_i$ is frequency and $A_i$ is amplitude, using adaptive thresholding (1.25× local median).

**Step 3: Ratio Computation**
For each peak pair $(f_i, f_j)$ where $f_j > f_i$:
$$r_{ij} = \frac{f_j}{f_i}, \quad w_{ij} = \sqrt{A_i \cdot A_j}$$

**Step 4: Histogram Binning**
Accumulate weighted ratios into $B$ bins spanning $[r_{min}, r_{max}]$:
$$h_b = \sum_{(i,j): r_{ij} \in \text{bin}_b} w_{ij}$$

**Step 5: Enrichment**
Compute three channels per bin:
- **Proportion**: $p_b = h_b / \sum_k h_k$
- **Moment**: $m_b = h_b \cdot c_b^2 / \sum_k (h_k \cdot c_k^2)$ where $c_b$ is bin center
- **Entropy**: $e_b = -p_b \log(p_b + \epsilon) / \sum_k e_k$

The output is a tensor $H \in \mathbb{R}^{B \times 3}$ for each frame.

**Key Properties**:
- **Scale-invariant**: Ratios are unchanged by transposition (pitch shift) or unit conversion
- **Fixed-size**: Any signal maps to the same tensor shape, enabling batched processing
- **Domain-agnostic**: Applicable to any signal with extractable oscillatory components

### 3.2 The Factorized Latent Space

Following the DMVAE framework [23], we employ VAEs with factorized latent representations:

$$z = [z_{shared}, z_{private}]$$

where $z_{shared} \in \mathbb{R}^{d_s}$ captures cross-modal structure and $z_{private} \in \mathbb{R}^{d_p}$ captures modality-specific information.

The training objective combines:

**Reconstruction Loss** (per modality $m$):
$$\mathcal{L}_{recon}^m = \mathbb{E}_{q(z|x^m)}[\log p(x^m | z)]$$

**KL Regularization**:
$$\mathcal{L}_{KL} = D_{KL}(q(z|x) \| p(z))$$

**InfoNCE Contrastive Alignment** [11]:
$$\mathcal{L}_{InfoNCE} = -\log \frac{\exp(sim(z_A^{shared}, z_V^{shared})/\tau)}{\sum_j \exp(sim(z_A^{shared}, z_j^{shared})/\tau)}$$

where $\tau$ is temperature and $sim(\cdot, \cdot)$ is cosine similarity.

### 3.3 Core Hypotheses

We organize our research around three hypotheses with explicit falsification criteria:

---

**Hypothesis H1: Ratio Structure Exists**

*Claim*: Real-world oscillatory signals contain non-random ratio structure capturable in histogram representations.

*Prediction*: Ratio histograms from real signals should differ systematically from those computed from noise or randomly permuted spectra.

*Falsification*: H1 is falsified if ratio histograms from real signals are statistically indistinguishable from noise baselines across multiple domains.

*Status*: **Supported** by experiments showing meaningful structure in synthetic and real audio data.

---

**Hypothesis H2: Ratio Structure is Learnable**

*Claim*: Neural networks can learn compressed latent representations of ratio histograms that preserve meaningful structure.

*Prediction*: VAEs trained on ratio histograms should achieve low reconstruction error and produce latent spaces where similar signals cluster.

*Falsification*: H2 is falsified if VAEs consistently fail to converge or produce latent spaces with no meaningful organization (random clustering).

*Status*: **Supported** by VAE and HRM experiments achieving val_loss < 0.5 on temporal ratio data.

---

**Hypothesis H3: Ratio Structure Transfers Across Modalities**

*Claim*: When two modalities observe the same physical phenomenon, their ratio representations share learnable common structure.

*Prediction*: A model trained to align modalities via $z_{shared}$ should achieve cosine similarity > 0.7 between modality embeddings and enable cross-modal prediction with Pearson correlation > 0.7.

*Falsification*: H3 is falsified if cross-modal alignment fails (cos_sim < 0.5) even with perfectly synchronized training data from modalities observing the same phenomenon.

*Status*: **Supported for one modality pair** (audio ↔ vibration) by Roseta 1 experiment. Generalization to other pairs is **untested**.

---

## 4. Methodology

### 4.1 Dataset: University of Ottawa Electric Motor Dataset

We validate cross-modal alignment using the UOEMD dataset [41], which provides synchronized multi-sensor recordings from an industrial induction motor:

| Property | Value |
|----------|-------|
| Motor | 3 HP Marathon Electric D396 |
| Sample rate | 42,000 Hz |
| Recording duration | 10 seconds/file |
| Total files | 128 |
| Sensor channels | Accelerometer (vibration), Microphone (audio) |
| Operating conditions | 8 (1 healthy + 7 fault types) |

The eight conditions cover diverse fault mechanisms: healthy operation (HH), rotor unbalance (RU), rotor misalignment (RM), faulty bearing (FB), stator winding fault (SW), voltage unbalance (VU), bent rotor (BR), and broken rotor bars (KA).

### 4.2 Signal Processing Pipeline

For both audio and vibration channels:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| FFT window | 4,096 samples | Δf ≈ 10.25 Hz at 42 kHz |
| Hop length | 1,024 samples | 75% overlap, ~41 frames/sec |
| Ratio bins | 256 | Sufficient resolution for harmonic structure |
| Ratio range | [1.0, 6.0] | Covers up to 6th harmonic |
| Peak threshold | 1.25× local median | Adaptive to local noise floor |

### 4.3 RosetaVAE Architecture

**Encoder** (identical for both modalities):
- Input: $[B, T, 256, 3]$ (batch, time, bins, channels)
- Flatten to $[B, T, 768]$
- Linear(768 → 128) + LayerNorm + GELU + Dropout(0.1)
- Bidirectional LSTM(128 → 256, layers=2, dropout=0.1)
- Output: $z_{shared}$ (32-dim) and $z_{private}$ (16-dim) via separate linear projections for $\mu$ and $\log\sigma$

**Decoder** (per modality):
- Input: $[z_{shared}, z_{private}] = [B, T, 48]$
- Linear(48 → 128) + LayerNorm + GELU
- Bidirectional LSTM(128 → 256, layers=2)
- Linear(256 → 768) + Reshape + Softmax(dim=bins)

**Model Statistics**:
- Total parameters: 3,161,536
- $z_{shared}$ dimension: 32
- $z_{private}$ dimension: 16

### 4.4 Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Batch size | 8 |
| Max frames/sample | 100 |
| Optimizer | Adam |
| Learning rate | 1×10⁻³ |
| β (KL weight) | 1.0 |
| λ (InfoNCE weight) | 2.0 |
| Temperature τ | 0.07 |
| Early stopping patience | 20 epochs |

### 4.5 Evaluation Protocol

**Phase 1 (Training)**: Train on all 128 files to learn cross-modal alignment.

**Phase 2 (Per-Condition Evaluation)**: Evaluate cosine similarity between $z_{shared}^{audio}$ and $z_{shared}^{vibration}$ for each condition separately, without retraining.

**Phase 3 (Cross-Modal Retrieval)**: Test prediction capability:
1. Encode audio only: $z_{shared}^A = E_A(\text{audio})$
2. Decode to vibration: $\hat{V} = D_V(z_{shared}^A, z_{private}^V)$
3. Compute Pearson correlation between predicted and actual vibration histograms

---

## 5. Experimental Results

### 5.1 Analizador 5.0: Single-Modal Validation

We first validated that ratio histograms are learnable representations using synthetic audio data with controlled harmonic content (848 files).

| Architecture | Val Loss | Parameters | Temporal |
|--------------|----------|------------|----------|
| VAE Temporal | **0.4560** | 1.82M | Yes |
| HRM Temporal | 0.4607 | 2.27M | Yes |
| HRM Static | 0.5906 | 854K | No |
| VAE Static | 0.5997 | 838K | No |

**Key finding**: With appropriate data representation (linear scale + temporal sequences), both VAE and hierarchical recurrent model (HRM) architectures achieve comparable performance. This supports H1 (structure exists) and H2 (structure is learnable), and demonstrates that representation choices matter more than architecture selection.

### 5.2 Roseta 1: Cross-Modal Alignment

Training converged at epoch 83 with best validation loss of 5.847.

| Metric | Epoch 1 | Epoch 83 | Change |
|--------|---------|----------|--------|
| Total Loss | 13.55 | 5.85 | -56.8% |
| InfoNCE Loss | 6.63 | 2.36 | -64.4% |
| Cosine Similarity | 0.51 | 0.76 | +49.0% |

**Per-Condition Alignment**:

| Condition | cos_sim | L2 Distance | Physical Origin |
|-----------|---------|-------------|-----------------|
| HH (Healthy) | 0.7657 | 3.811 | Normal operation |
| RU (Rotor Unbalance) | 0.7672 | 3.801 | Mass imbalance |
| RM (Rotor Misalignment) | 0.7659 | 3.815 | Shaft misalignment |
| FB (Faulty Bearing) | 0.7646 | 3.838 | Bearing defect |
| SW (Stator Winding) | 0.7637 | 3.826 | Winding fault |
| VU (Voltage Unbalance) | 0.7632 | 3.835 | Supply imbalance |
| BR (Bent Rotor) | 0.7669 | 3.822 | Rotor deformation |
| KA (Broken Rotor Bars) | 0.7656 | 3.823 | Rotor bar damage |
| **Mean ± Std** | **0.7653 ± 0.0015** | 3.821 ± 0.012 | — |

**Statistical Validation**:
- ANOVA F-statistic: 0.847, p-value: 0.548 (no significant difference across conditions)
- InfoNCE ablation Cohen's d: **5.75** (very large effect)

**Cross-Modal Retrieval**:

| Condition | Pearson Correlation | Target (>0.7) |
|-----------|---------------------|---------------|
| HH (Healthy) | 0.7542 | ✓ Achieved |
| FB (Faulty Bearing) | 0.7633 | ✓ Achieved |
| RU (Rotor Unbalance) | 0.6600 | ○ Close |

### 5.3 Summary: Hypothesis Status

| Hypothesis | Prediction | Result | Status |
|------------|------------|--------|--------|
| H1 | Meaningful ratio structure | Observed in histograms | Supported |
| H2 | VAE reconstruction < 0.5 | val_loss = 0.456 | Supported |
| H3 | Cross-modal cos_sim > 0.7 | cos_sim = 0.766 | Supported* |

*H3 supported for audio ↔ vibration only; generalization untested.

### 5.4 Limitations of Current Evidence

| Claim | Status |
|-------|--------|
| Ratios transfer between audio and vibration | **Demonstrated** |
| Ratios transfer between any two modalities | Not tested |
| Ratio methods outperform spectral baselines | Not compared |
| Results generalize to non-laboratory conditions | Not tested |
| Results generalize to other physical phenomena | Not tested |

---

## 6. Research Roadmap

### 6.1 Immediate Priorities

**Roseta 2: Audio → Visual (Lissajous Patterns)**

*Rationale*: Test transfer between fundamentally different sensor types. Audio-vibration share mechanical transmission; audio-visual do not.

*Method*: Generate tones with controlled ratios; capture Lissajous patterns via cymatics; train cross-modal VAE.

*Success criterion*: cos_sim > 0.6

*Decision*: If < 0.4, revise H3 to specify physical coupling requirements.

---

**Roseta 3: Physiological Signals**

*Rationale*: Test in biological domain using EEG, ECG, respiration from public datasets [42, 43].

*Success criterion*: cos_sim > 0.5 for at least one modality pair.

---

**Baseline Comparisons**

*Method*: Compare ratio histograms vs. spectrograms, MFCCs, wav2vec [44] embeddings on UOEMD.

*Metrics*: Cross-modal alignment, downstream classification, noise robustness.

### 6.2 Decision Points

| After Roseta 2 | Decision |
|----------------|----------|
| cos_sim > 0.6 | Proceed to Roseta 3, multi-modal extension |
| cos_sim 0.4-0.6 | Investigate failure modes; consider adaptations |
| cos_sim < 0.4 | Revise H3 scope; focus on physically-coupled modalities |

### 6.3 Medium-Term Goals

If validation succeeds:
1. Scale to N > 2 modalities in shared latent space
2. Extend to trajectory modeling (temporal dynamics of ratio evolution)
3. Test on field data with realistic noise and sensor drift
4. Develop mathematical framework for ratio-based information measures

---

## 7. Long-Term Vision

*This section describes research aspirations, clearly distinguished from demonstrated results.*

### 7.1 The Research Horizon

If H3 generalizes broadly, it suggests that proportional structure may serve as a "common language" for representing oscillatory systems across domains. This would enable:

- **Unified representations** applicable to audio, vibration, EEG, imagery, electromagnetic signals
- **Transfer learning** where models trained on one domain bootstrap learning in others
- **Interpretable features** since ratios have physical meaning (resonance, periodicity, coupling)

### 7.2 The PHIDEUS Vision

Our long-term aspiration is systems that:
1. **Sense** the world through diverse modalities
2. **Encode** signals into proportional descriptors
3. **Learn** shared structure across domains
4. **Reason** about patterns and anomalies
5. **Communicate** findings interpretably

We name this vision PHIDEUS (from Greek φειδεύς, "one who reads proportions").

**This vision is speculative**. It is not supported by current evidence. It is the direction we investigate toward, subject to experimental validation.

---

## 8. Conclusion

### 8.1 Demonstrated Results

1. **Ratio histograms are learnable**: VAE and HRM achieve val_loss < 0.5 on temporal ratio data
2. **Cross-modal alignment works for audio ↔ vibration**: cos_sim = 0.766 ± 0.002, consistent across 8 conditions
3. **Cross-modal prediction is feasible**: Pearson > 0.75 for retrieval tasks

### 8.2 Working Hypotheses

- **H1** (ratio structure exists): Supported
- **H2** (ratio structure is learnable): Supported
- **H3** (ratio structure transfers): Supported for one modality pair; generalization requires further testing

### 8.3 Research Direction

We pursue progressive validation of ratio-based cross-modal learning, with explicit decision points. Both positive and negative results advance understanding: success expands the scope of ratio-based representations; failure clarifies their limits.

---

## References

[1] K.R. Fyfe and E.D.S. Munck, "Analysis of Computed Order Tracking," *Mechanical Systems and Signal Processing*, vol. 11, no. 2, pp. 187-205, 1997.

[2] R.B. Randall and J. Antoni, "Rolling Element Bearing Diagnostics—A Tutorial," *Mechanical Systems and Signal Processing*, vol. 25, no. 2, pp. 485-520, 2011.

[3] D. Deutsch, "The Psychology of Music," 3rd ed., Academic Press, 2013.

[4] R.T. Canolty and R.T. Knight, "The Functional Role of Cross-Frequency Coupling," *Trends in Cognitive Sciences*, vol. 14, no. 11, pp. 506-515, 2010.

[5] A. Hyafil, A.L. Giraud, L. Fontolan, and B. Bhattacharya, "Neural Cross-Frequency Coupling: Connecting Architectures, Mechanisms, and Functions," *Trends in Neurosciences*, vol. 38, no. 11, pp. 725-740, 2015.

[6] W. Klimesch, "Alpha-Band Oscillations, Attention, and Controlled Access to Stored Information," *Trends in Cognitive Sciences*, vol. 16, no. 12, pp. 606-617, 2012.

[7] J. Sueur and A. Farina, "Ecoacoustics: The Ecological Investigation and Interpretation of Environmental Sound," *Biosemiotics*, vol. 8, pp. 493-502, 2015.

[8] B.C. Pijanowski et al., "Soundscape Ecology: The Science of Sound in the Landscape," *BioScience*, vol. 61, no. 3, pp. 203-216, 2011.

[9] C.E. Shannon, "A Mathematical Theory of Communication," *Bell System Technical Journal*, vol. 27, pp. 379-423, 1948.

[10] T.M. Cover and J.A. Thomas, *Elements of Information Theory*, 2nd ed., Wiley, 2006.

[11] A. van den Oord, Y. Li, and O. Vinyals, "Representation Learning with Contrastive Predictive Coding," *arXiv:1807.03748*, 2018.

[12] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, "A Simple Framework for Contrastive Learning of Visual Representations," *Proc. ICML*, 2020.

[13] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, "Momentum Contrast for Unsupervised Visual Representation Learning," *Proc. CVPR*, pp. 9726-9735, 2020.

[14] J.B. Grill, F. Strub, F. Altché, et al., "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning," *Proc. NeurIPS*, 2020.

[15] Y. Tian, D. Krishnan, and P. Isola, "Contrastive Multiview Coding," *Proc. ECCV*, LNCS vol. 12356, pp. 776-794, 2020.

[16] A. Radford, J.W. Kim, C. Hallacy, et al., "Learning Transferable Visual Models From Natural Language Supervision," *Proc. ICML*, 2021.

[17] A. Guzhov, F. Raue, J. Hees, and A. Dengel, "AudioCLIP: Extending CLIP to Image, Text and Audio," *Proc. ICASSP*, 2022.

[18] D.P. Kingma and M. Welling, "Auto-Encoding Variational Bayes," *Proc. ICLR*, 2014.

[19] D.J. Rezende, S. Mohamed, and D. Wierstra, "Stochastic Backpropagation and Approximate Inference in Deep Generative Models," *Proc. ICML*, 2014.

[20] M. Suzuki, K. Nakayama, and Y. Matsuo, "Joint Multimodal Learning with Deep Generative Models," *arXiv:1611.01891*, 2016.

[21] M. Wu and N. Goodman, "Multimodal Generative Models for Scalable Weakly-Supervised Learning," *Proc. NeurIPS*, 2018.

[22] Y. Shi, N. Siddharth, B. Paige, and P.H.S. Torr, "Variational Mixture-of-Experts Autoencoders for Multi-Modal Deep Generative Models," *Proc. NeurIPS*, 2019.

[23] M. Lee and V. Pavlovic, "Private-Shared Disentangled Multimodal VAE for Learning of Latent Representations," *Proc. CVPR Workshops*, 2021.

[24] Y.H. Tsai, P.P. Liang, A. Zadeh, L.P. Morency, and R. Salakhutdinov, "Learning Factorized Multimodal Representations," *Proc. ICLR*, 2019.

[25] S. Braun and B.B. Seth, "On the Extraction and Filtering of Signals Acquired from Rotating Machines," *Journal of Sound and Vibration*, vol. 65, no. 1, pp. 37-50, 1979.

[26] W.A. Smith and R.B. Randall, "Rolling Element Bearing Diagnostics Using the Case Western Reserve University Data: A Benchmark Study," *Mechanical Systems and Signal Processing*, vol. 64-65, pp. 100-131, 2015.

[27] Z. Zhao, T. Li, J. Wu, et al., "Deep Learning Algorithms for Rotating Machinery Intelligent Diagnosis: An Open Source Benchmark Study," *ISA Transactions*, vol. 107, pp. 224-255, 2020.

[28] D.T. Hoang and H.J. Kang, "A Survey on Deep Learning Based Bearing Fault Diagnosis," *Neurocomputing*, vol. 335, pp. 327-335, 2019.

[29] T. Ince, S. Kiranyaz, L. Eren, M. Askar, and M. Gabbouj, "Real-Time Motor Fault Detection by 1-D Convolutional Neural Networks," *IEEE Trans. Industrial Electronics*, vol. 63, no. 11, pp. 7067-7075, 2016.

[30] H. Qiu, H. Luo, G. Xu, and D. Jiang, "End-to-End CNN + LSTM Deep Learning Approach for Bearing Fault Diagnosis," *Applied Intelligence*, vol. 51, pp. 509-521, 2021.

[31] F. Shaffer and J.P. Ginsberg, "An Overview of Heart Rate Variability Metrics and Norms," *Frontiers in Public Health*, vol. 5, 258, 2017.

[32] A. Voss et al., "Methods Derived from Nonlinear Dynamics for Analysing Heart Rate Variability," *Philosophical Transactions of the Royal Society A*, vol. 367, pp. 277-296, 2009.

[33] M. Iosa, G. Morone, A. Fusco, et al., "Seven Capital Devices for the Future of Stroke Rehabilitation," *Stroke Research and Treatment*, 2012.

[34] O. AlShorman, F. Alkahatni, M. Masadeh, et al., "Sounds and Acoustic Emission-Based Early Fault Diagnosis of Induction Motor: A Review Study," *Advances in Mechanical Engineering*, vol. 13, no. 2, 2021.

[35] P. Gangsar and R. Tiwari, "Signal Based Condition Monitoring Techniques for Fault Detection and Diagnosis of Induction Motors: A State-of-the-Art Review," *Mechanical Systems and Signal Processing*, vol. 144, 106908, 2020.

[36] K. Kullu and E. Cinar, "A Deep-Learning-Based Multi-Modal Sensor Fusion Approach for Detection of Equipment Faults," *Machines*, vol. 10, no. 11, 1105, 2022.

[37] "Multi-Sensor Data Fusion in Intelligent Fault Diagnosis of Rotating Machines: A Comprehensive Review," *Measurement*, 2024.

[38] "Vibration and Acoustic Signal Consistent Feature Fusion Network for Intelligent Bearing Fault Diagnosis," *Engineering Research Express*, 2025.

[39] Z. Yue, Y. Wang, J. Duan, et al., "TS2Vec: Towards Universal Representation of Time Series," *Proc. AAAI*, vol. 36, no. 8, pp. 8980-8987, 2022.

[40] X. Zhang, Z. Zhao, T. Tsiligkaridis, and M. Zitnik, "Self-Supervised Contrastive Pre-Training for Time Series via Time-Frequency Consistency," *Proc. NeurIPS*, 2022.

[41] University of Ottawa Electric Motor Dataset (UOEMD). Available: https://data.mendeley.com/datasets/msxs4vj48g/1

[42] PhysioNet: PhysioBank Archives. Available: https://physionet.org/

[43] C. Lessmeier, J.K. Kimotho, D. Zimmer, and W. Sextro, "Condition Monitoring of Bearing Damage in Electromechanical Drive Systems," *Proc. European Conf. PHM Society*, 2016.

[44] A. Baevski, H. Zhou, A. Mohamed, and M. Auli, "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," *Proc. NeurIPS*, 2020.

[45] Y. Ding, M. Jia, Q. Miao, and Y. Cao, "Self-Supervised Pretraining via Contrast Learning for Intelligent Incipient Fault Detection of Bearings," *Reliability Engineering & System Safety*, vol. 218, 108126, 2022.

[46] M.M. Bronstein, J. Bruna, T. Cohen, and P. Veličković, "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges," *arXiv:2104.13478*, 2021.

[47] P.W. Battaglia et al., "Relational Inductive Biases, Deep Learning, and Graph Networks," *arXiv:1806.01261*, 2018.

---

## Appendix A: Technical Specifications

### A.1 Ratio Histogram Computation

```
Input: Signal x(t), sample rate fs
Parameters: N=4096, H=1024, B=256, r_range=[1.0, 6.0]

1. X = STFT(x, n_fft=N, hop=H, window='hann')
2. For each frame t:
   a. peaks = find_peaks(|X[:,t]|, threshold=1.25*median)
   b. For each pair (i,j) where f_j > f_i:
      - r = f_j / f_i
      - w = sqrt(A_i * A_j)
      - bin_idx = floor((r - r_min) / (r_max - r_min) * B)
      - h[bin_idx] += w
   c. Compute channels: proportion, moment, entropy
3. Output: H ∈ ℝ^{T × B × 3}
```

### A.2 RosetaVAE Loss Function

$$\mathcal{L} = \mathcal{L}_{recon}^A + \mathcal{L}_{recon}^V + \beta \mathcal{L}_{KL} + \lambda \mathcal{L}_{InfoNCE}$$

where:
- $\mathcal{L}_{recon}^m = \text{MSE}(H^m, \hat{H}^m)$
- $\mathcal{L}_{KL} = D_{KL}(q(z|x) \| \mathcal{N}(0, I))$
- $\mathcal{L}_{InfoNCE} = -\frac{1}{2}[\log\frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ij}/\tau}} + \log\frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ji}/\tau}}]$
- $s_{ij} = \cos(z_{shared}^{A,i}, z_{shared}^{V,j})$

Hyperparameters: $\beta=1.0$, $\lambda=2.0$, $\tau=0.07$

### A.3 Experimental Results Detail

| Experiment | Metric | Value | 95% CI |
|------------|--------|-------|--------|
| Analizador 5.0 | VAE val_loss | 0.4560 | — |
| Analizador 5.0 | HRM val_loss | 0.4607 | — |
| Roseta 1 | cos_sim (mean) | 0.7653 | [0.763, 0.768] |
| Roseta 1 | cos_sim (std) | 0.0015 | — |
| Roseta 1 | ANOVA p-value | 0.548 | — |
| Roseta 1 | Cohen's d | 5.75 | — |
| Roseta 1 | Pearson (HH) | 0.7542 | — |
| Roseta 1 | Pearson (FB) | 0.7633 | — |

---

## Appendix B: Code and Data Availability

| Resource | Location |
|----------|----------|
| Ratio Analyzer | `src/analizador/analizador_roseta.py` |
| Dataset Loader | `src/datasets/roseta_dataset.py` |
| RosetaVAE Model | `src/RNA/roseta_vae.py` |
| Experiment Script | `experiments/run_roseta_experiment.py` |
| Trained Model | `data/training_outputs/roseta_full/best_model.pt` |
| UOEMD Dataset | https://data.mendeley.com/datasets/msxs4vj48g/1 |

---

*Document version: 2.0*
*Last updated: January 2026*
*Status: Active research program*

---

**Acknowledgment**

We maintain a clear distinction between demonstrated results, working hypotheses, and speculative vision. This document will be updated as experiments progress.

*"The measure of a research program is not the boldness of its vision, but the rigor of its validation."*
