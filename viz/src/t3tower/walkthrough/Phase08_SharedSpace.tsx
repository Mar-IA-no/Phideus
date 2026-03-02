import { Vec3 } from "@/src/utils/vector";
import { IT3TowerWalkthroughArgs, setT3TowerInitialCamera, moveT3TowerCameraTo } from "./T3TowerWalkthrough";
import { T3TowerDim } from "../T3TowerDimStyle";

export function t3TowerPhase08_SharedSpace(args: IT3TowerWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept, drawDimLabels } = tools;

    dimExcept([
        layout.audioEmbedding, layout.midiEmbedding, layout.t3Embedding,
        layout.vicregAM, layout.vicregAT, layout.vicregMT,
    ]);

    let sharedY = layout.audioEmbedding.y;
    setT3TowerInitialCamera(state, new Vec3(0, 0, -sharedY - 20), new Vec3(290, 12, 6));

    let d256 = c_dimRef('256D', T3TowerDim.D_proj);
    let audioEmb = c_blockRef('z_audio', layout.audioEmbedding);
    let midiEmb = c_blockRef('z_midi', layout.midiEmbedding);
    let t3Emb = c_blockRef('z_t3', layout.t3Embedding);

    commentary()`Three independent encoder towers have compressed their inputs into the same ${d256} space. The ${audioEmb}, ${midiEmb}, and ${t3Emb} now sit side by side — three vectors from completely different worlds, ready to be aligned by the 3-way VICReg loss.`;
    breakAfter();

    // Step 1: All three embeddings arrive
    let t0 = afterTime(null, 2.5, 0.5);

    commentary()`Each embedding carries a different kind of information: ${audioEmb} encodes spectral content from MERT features. ${midiEmb} encodes symbolic musical structure — pitch sequences, velocity dynamics. ${t3Emb} carries the auxiliary signal. The shared space must accommodate all three perspectives simultaneously.`;
    breakAfter();

    if (t0.active) {
        let phase = wt.time * 2;
        // Three embeddings pulse at slightly different rates — they're different modalities
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, 0.4 + 0.3 * Math.sin(phase));
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, 0.4 + 0.3 * Math.sin(phase + 2.1));
        layout.t3Embedding.highlight = Math.max(layout.t3Embedding.highlight, 0.4 + 0.3 * Math.sin(phase + 4.2));

        drawDimLabels(layout.audioEmbedding, t0.t * 0.9);
        drawDimLabels(layout.midiEmbedding, t0.t * 0.9);
        drawDimLabels(layout.t3Embedding, t0.t * 0.9);
    }

    breakAfter();

    // Step 2: The triangular constraint
    let t1 = afterTime(null, 3.0, 0.5);
    let t0c = afterTime(t0, 0.3);
    cleanup(t0c, [t0]);

    moveT3TowerCameraTo(state, t1, new Vec3(0, 0, -sharedY - 40), new Vec3(290, 15, 8));

    let vicAM = c_blockRef('Audio↔MIDI', layout.vicregAM);
    let vicAT = c_blockRef('Audio↔T3', layout.vicregAT);
    let vicMT = c_blockRef('MIDI↔T3', layout.vicregMT);

    commentary()`The key architectural advantage: three pairwise VICReg losses — ${vicAM}, ${vicAT}, ${vicMT} — create a _triangular_ constraint. In a two-tower model, embeddings can drift into degenerate configurations that satisfy one pairwise loss. With three, the geometry is much more constrained: if audio aligns with MIDI and MIDI aligns with T3, then audio must transitively align with T3.`;
    breakAfter();

    if (t1.active) {
        // VICReg blocks pulse sequentially
        let phase = wt.time * 1.5;
        layout.vicregAM.highlight = Math.max(layout.vicregAM.highlight, 0.5 + 0.3 * Math.sin(phase));
        layout.vicregAT.highlight = Math.max(layout.vicregAT.highlight, 0.5 + 0.3 * Math.sin(phase + 2.1));
        layout.vicregMT.highlight = Math.max(layout.vicregMT.highlight, 0.5 + 0.3 * Math.sin(phase + 4.2));

        // Embeddings glow steadily
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, 0.35);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, 0.35);
        layout.t3Embedding.highlight = Math.max(layout.t3Embedding.highlight, 0.35);
    }

    breakAfter();

    // Step 3: Why three towers?
    let t2 = afterTime(null, 2.5, 0.5);
    let t1c = afterTime(t1, 0.3);
    cleanup(t1c, [t1]);

    commentary()`Why add a third tower instead of simply improving the two-tower model? Geometric redundancy. Two-tower VICReg has one alignment axis. Three-tower VICReg has three alignment planes — the embedding space is shaped from multiple angles simultaneously. The T3 tower acts as an _anchor_, preventing the audio-MIDI pair from finding lazy alignment shortcuts.`;
    breakAfter();

    if (t2.active) {
        // All six blocks glow together
        let glow = 0.4 + 0.2 * Math.sin(wt.time * 2);
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, glow);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, glow);
        layout.t3Embedding.highlight = Math.max(layout.t3Embedding.highlight, glow);
        layout.vicregAM.highlight = Math.max(layout.vicregAM.highlight, glow + 0.1);
        layout.vicregAT.highlight = Math.max(layout.vicregAT.highlight, glow + 0.1);
        layout.vicregMT.highlight = Math.max(layout.vicregMT.highlight, glow + 0.1);

        drawDimLabels(layout.vicregAM, 0.7);
        drawDimLabels(layout.vicregAT, 0.7);
        drawDimLabels(layout.vicregMT, 0.7);
    }

    breakAfter();
}
