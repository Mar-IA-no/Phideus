import { Vec3 } from "@/src/utils/vector";
import { IT3TowerWalkthroughArgs, setT3TowerInitialCamera, moveT3TowerCameraTo } from "./T3TowerWalkthrough";
import { T3TowerDim } from "../T3TowerDimStyle";

export function t3TowerPhase09_VICReg(args: IT3TowerWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept, drawDimLabels } = tools;

    dimExcept([
        layout.audioEmbedding, layout.midiEmbedding, layout.t3Embedding,
        layout.vicregAM, layout.vicregAT, layout.vicregMT,
    ]);

    let sharedY = layout.audioEmbedding.y;
    setT3TowerInitialCamera(state, new Vec3(0, 0, -sharedY - 20), new Vec3(290, 12, 5));

    let d256 = c_dimRef('256D', T3TowerDim.D_proj);
    let loss = c_dimRef('Loss', T3TowerDim.Loss);

    let vicAMRef = c_blockRef('VICReg(Audio↔MIDI)', layout.vicregAM);
    let vicATRef = c_blockRef('VICReg(Audio↔T3)', layout.vicregAT);
    let vicMTRef = c_blockRef('VICReg(MIDI↔T3)', layout.vicregMT);

    commentary()`3-way VICReg — three independent VICReg losses, one per embedding pair. Each computes invariance (MSE), variance (hinge), and covariance (decorrelation) on its pair. Total ${loss}: L = L_AM + L_AT + L_MT, where each L_pair = 25*inv + 25*var + 1*cov. Nine loss terms in total shape the ${d256} space.`;
    breakAfter();

    // Step 1: Audio ↔ MIDI — the primary alignment
    let t0 = afterTime(null, 3.0, 0.5);

    let audioEmb = c_blockRef('z_audio', layout.audioEmbedding);
    let midiEmb = c_blockRef('z_midi', layout.midiEmbedding);

    commentary()`${vicAMRef}: The primary alignment axis. ${audioEmb} and ${midiEmb} are pulled together (invariance), prevented from collapsing (variance), and decorrelated (covariance). This is the same loss used in the two-tower model — the foundation of the alignment. Weight: equal to the other two pairs.`;
    breakAfter();

    if (t0.active) {
        let amPulse = 0.5 + 0.35 * Math.sin(wt.time * 3.5);
        layout.vicregAM.highlight = Math.max(layout.vicregAM.highlight, amPulse);
        drawDimLabels(layout.vicregAM, t0.t * 0.8);

        let embGlow = 0.2 + 0.15 * Math.sin(wt.time * 3.5);
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, embGlow);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, embGlow);
    }

    breakAfter();

    // Step 2: Audio ↔ T3 — the cross-modal anchor
    let t1 = afterTime(null, 2.5, 0.5);
    let t0c = afterTime(t0, 0.3);
    cleanup(t0c, [t0]);

    let vicATY = layout.vicregAT.y;
    moveT3TowerCameraTo(state, t1, new Vec3(0, 0, -vicATY + 15), new Vec3(290, 8, 3.5));

    let t3Emb = c_blockRef('z_t3', layout.t3Embedding);

    commentary()`${vicATRef}: Aligns the audio representation with the T3 auxiliary signal. This creates a second alignment axis — the audio embedding must now satisfy _two_ partners simultaneously. Information that only served the MIDI alignment may be reinforced or reshaped by the T3 constraint.`;
    breakAfter();

    if (t1.active) {
        let atPulse = 0.5 + 0.35 * Math.sin(wt.time * 3);
        layout.vicregAT.highlight = Math.max(layout.vicregAT.highlight, atPulse);
        drawDimLabels(layout.vicregAT, t1.t * 0.8);

        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, 0.3);
        layout.t3Embedding.highlight = Math.max(layout.t3Embedding.highlight, 0.3);
    }

    breakAfter();

    // Step 3: MIDI ↔ T3 — the transitive bridge
    let t2 = afterTime(null, 2.5, 0.5);
    let t1c = afterTime(t1, 0.3);
    cleanup(t1c, [t1]);

    let vicMTY = layout.vicregMT.y;
    moveT3TowerCameraTo(state, t2, new Vec3(0, 0, -vicMTY + 15), new Vec3(290, 8, 3.5));

    commentary()`${vicMTRef}: The transitive bridge. If audio aligns with MIDI (L_AM), and MIDI aligns with T3 (L_MT), then audio _must_ align with T3 — creating consistency pressure from all three directions. This third loss closes the triangle, making degenerate solutions geometrically impossible.`;
    breakAfter();

    if (t2.active) {
        let mtPulse = 0.5 + 0.35 * Math.sin(wt.time * 3);
        layout.vicregMT.highlight = Math.max(layout.vicregMT.highlight, mtPulse);
        drawDimLabels(layout.vicregMT, t2.t * 0.8);

        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, 0.3);
        layout.t3Embedding.highlight = Math.max(layout.t3Embedding.highlight, 0.3);
    }

    breakAfter();

    // Step 4: All three together — the complete picture
    let t3 = afterTime(null, 3.5, 0.5);
    let t2c = afterTime(t2, 0.4);
    cleanup(t2c, [t2]);

    moveT3TowerCameraTo(state, t3, new Vec3(0, 0, -sharedY - 40), new Vec3(290, 15, 8));

    commentary()`Together: L_total = L_AM + L_AT + L_MT = 3 × (25*inv + 25*var + 1*cov). Nine loss terms. Three alignment planes. The result: a ${d256} space shaped from every angle, where each dimension carries unique cross-modal information. No lazy shortcuts survive the triangular constraint.`;
    breakAfter();

    if (t3.active) {
        // All three VICReg terms pulse at different frequencies
        let phase = wt.time * 2;
        layout.vicregAM.highlight = Math.max(layout.vicregAM.highlight, 0.5 + 0.4 * Math.sin(phase));
        layout.vicregAT.highlight = Math.max(layout.vicregAT.highlight, 0.5 + 0.4 * Math.sin(phase + 2.1));
        layout.vicregMT.highlight = Math.max(layout.vicregMT.highlight, 0.5 + 0.4 * Math.sin(phase + 4.2));

        // All three embeddings glow steadily
        let embGlow = 0.5 + 0.2 * Math.sin(wt.time * 1.5);
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, embGlow);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, embGlow);
        layout.t3Embedding.highlight = Math.max(layout.t3Embedding.highlight, embGlow);

        drawDimLabels(layout.vicregAM, 0.7);
        drawDimLabels(layout.vicregAT, 0.7);
        drawDimLabels(layout.vicregMT, 0.7);
        drawDimLabels(layout.audioEmbedding, 0.8);
        drawDimLabels(layout.midiEmbedding, 0.8);
        drawDimLabels(layout.t3Embedding, 0.8);
    }

    breakAfter();
}
