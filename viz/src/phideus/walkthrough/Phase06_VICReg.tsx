import { Vec3 } from "@/src/utils/vector";
import { IPhideusWalkthroughArgs, setPhideusInitialCamera, movePhideusCameraTo } from "./PhideusWalkthrough";
import { PhideusDim } from "../PhideusDimStyle";

export function phideusPhase06_VICReg(args: IPhideusWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept, drawDimLabels } = tools;

    // Dim everything except shared space
    dimExcept([
        layout.audioEmbedding, layout.midiEmbedding,
        layout.vicregInv, layout.vicregVar, layout.vicregCov,
    ]);

    // Start zoomed out to see the full shared space
    let sharedY = layout.audioEmbedding.y;
    setPhideusInitialCamera(state, new Vec3(0, 0, -sharedY - 20), new Vec3(290, 12, 5));

    let invRef = c_blockRef('invariance term', layout.vicregInv);
    let varRef = c_blockRef('variance term', layout.vicregVar);
    let covRef = c_blockRef('covariance term', layout.vicregCov);
    let d256 = c_dimRef('256D', PhideusDim.D_proj);

    commentary()`VICReg — Variance-Invariance-Covariance Regularization — is the heart of how Phideus learns without labels. Three competing forces shape the embedding space: ${invRef} pulls matching pairs together, ${varRef} prevents collapse to a point, and ${covRef} decorrelates dimensions. No negative pairs. No momentum encoders. No asymmetric tricks. Just three elegant loss terms that together create a rich, structured ${d256} representation.`;
    breakAfter();

    // Step 1: Show embeddings arriving with dramatic pulse
    let t0 = afterTime(null, 2.5, 0.5);

    let audioEmb = c_blockRef('z_audio', layout.audioEmbedding);
    let midiEmb = c_blockRef('z_midi', layout.midiEmbedding);

    commentary()`The ${audioEmb} and ${midiEmb} arrive in the shared space — two ${d256} vectors from completely different worlds. An audio recording of a pianist playing a sonata, and the MIDI score of that same sonata. The question Phideus must answer: _how do we pull these representations together without collapsing them into meaningless noise?_`;
    breakAfter();

    if (t0.active) {
        // Embeddings pulse in sync — they're a pair
        let syncPulse = 0.4 + 0.3 * Math.sin(wt.time * 3);
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, syncPulse);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, syncPulse);
        drawDimLabels(layout.audioEmbedding, t0.t * 0.9);
        drawDimLabels(layout.midiEmbedding, t0.t * 0.9);
    }

    breakAfter();

    // Step 2: INVARIANCE — the pull-together force
    let t1 = afterTime(null, 3.0, 0.5);
    let t0c = afterTime(t0, 0.3);
    cleanup(t0c, [t0]);

    let invY = layout.vicregInv.y;
    movePhideusCameraTo(state, t1, new Vec3(0, 0, -invY + 15), new Vec3(290, 8, 2.5));

    let loss = c_dimRef('Loss', PhideusDim.Loss);

    commentary()`_INVARIANCE_ (MSE ${loss}, weight = 25): The ${invRef} computes mean squared error between matching audio-MIDI pairs: ||z_audio - z_midi||^2. This is the gravitational pull — it drags matching representations together across the embedding space. Weight 25 makes this the dominant alignment force.`;
    breakAfter();

    if (t1.active) {
        // Invariance block pulses red — the force of attraction
        let invPulse = 0.5 + 0.35 * Math.sin(wt.time * 3.5);
        layout.vicregInv.highlight = Math.max(layout.vicregInv.highlight, invPulse);
        drawDimLabels(layout.vicregInv, t1.t * 0.8);

        // Embeddings glow softly — they're being pulled
        let embGlow = 0.2 + 0.15 * Math.sin(wt.time * 3.5);
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, embGlow);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, embGlow);
    }

    breakAfter();

    // Step 3: The collapse problem — why invariance alone isn't enough
    let t1b = afterTime(null, 2.0, 0.5);

    commentary()`But there's a trap. If the model only minimizes invariance, it discovers a shortcut: map _everything_ to the same constant vector. z_audio = z_midi = [0, 0, ..., 0]. MSE loss? Zero. Perfect score. Completely useless. This is representational collapse — and the next two terms exist to prevent it.`;
    breakAfter();

    if (t1b.active) {
        // Brief dim of everything to illustrate "collapse"
        let collapseT = Math.min(1, t1b.t * 2);
        let fade = 1 - collapseT * 0.5;
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, 0.1 * fade);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, 0.1 * fade);
        layout.vicregInv.highlight = Math.max(layout.vicregInv.highlight, 0.3 * fade);
    }

    breakAfter();

    // Step 4: VARIANCE — the anti-collapse shield
    let t2 = afterTime(null, 3.0, 0.5);
    let t1bc = afterTime(t1b, 0.3);
    cleanup(t1bc, [t1, t1b]);

    let varY = layout.vicregVar.y;
    movePhideusCameraTo(state, t2, new Vec3(0, 0, -varY + 15), new Vec3(290, 8, 2.5));

    commentary()`_VARIANCE_ (hinge ${loss}, weight = 25): The ${varRef} computes standard deviation for each of the 256 embedding dimensions across the batch, then applies a hinge loss: max(0, 1 - std(z_j)). This means: every dimension must have std >= 1. If any dimension starts collapsing toward a constant, this term fires aggressively, pushing the representations apart.`;
    breakAfter();

    if (t2.active) {
        // Variance block pulses with growing intensity
        let varPulse = 0.5 + 0.35 * Math.sin(wt.time * 4);
        layout.vicregVar.highlight = Math.max(layout.vicregVar.highlight, varPulse);
        drawDimLabels(layout.vicregVar, t2.t * 0.8);

        // Embeddings spread — they're being pushed apart
        let spreadPulse = 0.3 + 0.2 * Math.sin(wt.time * 4 + Math.PI);
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, spreadPulse);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, spreadPulse);
    }

    breakAfter();

    // Step 5: COVARIANCE — the information maximizer
    let t3 = afterTime(null, 3.0, 0.5);
    let t2c = afterTime(t2, 0.3);
    cleanup(t2c, [t2]);

    let covY = layout.vicregCov.y;
    movePhideusCameraTo(state, t3, new Vec3(0, 0, -covY + 15), new Vec3(290, 8, 2.5));

    commentary()`_COVARIANCE_ (decorrelation, weight = 1): The ${covRef} penalizes off-diagonal elements of the covariance matrix C(Z) = (1/N) * (Z - Z_mean)^T * (Z - Z_mean). This forces different dimensions to encode _different_ information. Without it, 200 of your 256 dimensions might redundantly encode pitch, wasting capacity. The low weight (1 vs 25) reflects that it's a soft regularizer, not a hard constraint.`;
    breakAfter();

    if (t3.active) {
        // Covariance block with a different, slower pulse
        let covPulse = 0.5 + 0.3 * Math.sin(wt.time * 2.5);
        layout.vicregCov.highlight = Math.max(layout.vicregCov.highlight, covPulse);
        drawDimLabels(layout.vicregCov, t3.t * 0.8);

        // Gentle glow on embeddings
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, 0.25);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, 0.25);
    }

    breakAfter();

    // Step 6: THE THREE FORCES TOGETHER — dramatic wide view with all pulsing
    let t4 = afterTime(null, 3.5, 0.5);
    let t3c = afterTime(t3, 0.4);
    cleanup(t3c, [t3]);

    movePhideusCameraTo(state, t4, new Vec3(0, 0, -sharedY - 30), new Vec3(290, 15, 6));

    commentary()`Together, the three forces create equilibrium. The total loss: L = 25 * L_inv + 25 * L_var + 1 * L_cov. ${invRef} pulls matching pairs together (alignment). ${varRef} ensures no dimension collapses (information preservation). ${covRef} decorrelates dimensions (information maximization). The result is a 256-dimensional space where every dimension carries unique, meaningful information about the musical content — and where audio and MIDI representations of the same piece land close together.`;
    breakAfter();

    if (t4.active) {
        // All three VICReg terms pulse at different frequencies — alive
        let phase = wt.time * 2;
        layout.vicregInv.highlight = Math.max(layout.vicregInv.highlight, 0.5 + 0.4 * Math.sin(phase));
        layout.vicregVar.highlight = Math.max(layout.vicregVar.highlight, 0.5 + 0.4 * Math.sin(phase + 2.1));
        layout.vicregCov.highlight = Math.max(layout.vicregCov.highlight, 0.5 + 0.4 * Math.sin(phase + 4.2));

        // Embeddings glow steadily
        let embGlow = 0.5 + 0.2 * Math.sin(wt.time * 1.5);
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, embGlow);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, embGlow);

        drawDimLabels(layout.vicregInv, 0.7);
        drawDimLabels(layout.vicregVar, 0.7);
        drawDimLabels(layout.vicregCov, 0.7);
        drawDimLabels(layout.audioEmbedding, 0.8);
        drawDimLabels(layout.midiEmbedding, 0.8);
    }

    breakAfter();
}
