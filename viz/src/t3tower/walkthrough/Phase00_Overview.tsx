import { Vec3 } from "@/src/utils/vector";
import { IT3TowerWalkthroughArgs, setT3TowerInitialCamera, moveT3TowerCameraTo } from "./T3TowerWalkthrough";
import { getTransformerLayerBlocks } from "../T3TowerModelLayout";
import { T3TowerDim } from "../T3TowerDimStyle";

export function t3TowerPhase00_Overview(args: IT3TowerWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { atTime, afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, drawDimLabels } = tools;

    // Wide view showing all three towers
    setT3TowerInitialCamera(state, new Vec3(30, 0, -600), new Vec3(290, 22, 20));

    let audioTfRef = c_blockRef('Audio Transformer', layout.audioTransformerLayers.flatMap(getTransformerLayerBlocks));
    let midiTfRef = c_blockRef('MIDI Transformer', layout.midiTransformerLayers.flatMap(getTransformerLayerBlocks));
    let t3TfRef = c_blockRef('T3 Transformer', layout.t3TransformerLayers.flatMap(getTransformerLayerBlocks));
    let d256 = c_dimRef('256D', T3TowerDim.D_proj);

    commentary()`The T3 architecture extends the two-tower model with a third, lightweight encoder tower. Three independent towers — the ${audioTfRef}, ${midiTfRef}, and ${t3TfRef} — each process a different modality, mapping all three into a shared ${d256} embedding space via 3-way VICReg self-supervised loss.`;
    breakAfter();

    // Step 1: Full overview — highlight all three towers
    let t0 = afterTime(null, 1.5, 0.5);

    if (t0.active) {
        for (let blk of layout.audioCnn) blk.highlight = Math.max(blk.highlight, t0.t * 0.2);
        for (let layer of layout.audioTransformerLayers) {
            layer.attnMatrix.highlight = Math.max(layer.attnMatrix.highlight, t0.t * 0.25);
        }
        for (let layer of layout.midiTransformerLayers) {
            layer.attnMatrix.highlight = Math.max(layer.attnMatrix.highlight, t0.t * 0.25);
        }
        for (let layer of layout.t3TransformerLayers) {
            layer.attnMatrix.highlight = Math.max(layer.attnMatrix.highlight, t0.t * 0.35);
        }
    }

    breakAfter();

    // Step 2: Zoom to audio tower
    let t1 = afterTime(null, 1.5, 0.5);
    let t0c = afterTime(t0, 0.4);
    cleanup(t0c, [t0]);

    moveT3TowerCameraTo(state, t1, new Vec3(-140, 0, -400), new Vec3(292, 20, 10));

    let cnnRef = c_blockRef('CNN feature extractor', layout.audioCnn);
    let dAudio = c_dimRef('1024D', T3TowerDim.D_audio);

    commentary()`The Audio Tower processes raw 24kHz waveforms through a frozen ${cnnRef} (4 stages), then refines them with 4 transformer layers at ${dAudio}. This is the largest tower — MERT's pre-trained representations provide a strong audio foundation.`;
    breakAfter();

    if (t1.active) {
        for (let layer of layout.audioTransformerLayers) {
            layer.attnMatrix.highlight = Math.max(layer.attnMatrix.highlight, t1.t * 0.5);
        }
        for (let layer of layout.midiTransformerLayers) {
            for (let blk of getTransformerLayerBlocks(layer)) blk.opacity = 0.25;
        }
        for (let layer of layout.t3TransformerLayers) {
            for (let blk of getTransformerLayerBlocks(layer)) blk.opacity = 0.25;
        }
        layout.t3Input.opacity = 0.25;
        layout.t3InputProj.opacity = 0.25;
        layout.t3PosEnc.opacity = 0.25;
    }

    breakAfter();

    // Step 3: Pan to MIDI tower
    let t2 = afterTime(null, 1.5, 0.5);
    let t1c = afterTime(t1, 0.4);
    cleanup(t1c, [t1]);

    moveT3TowerCameraTo(state, t2, new Vec3(40, 0, -400), new Vec3(288, 20, 10));

    let dMidi = c_dimRef('512D', T3TowerDim.D_midi);

    commentary()`The MIDI Tower converts discrete musical events (pitch, velocity, duration) into continuous vectors, then processes them through 4 transformer layers at ${dMidi}. Trained entirely from scratch — there is no pre-trained MIDI model to leverage.`;
    breakAfter();

    if (t2.active) {
        for (let layer of layout.midiTransformerLayers) {
            layer.attnMatrix.highlight = Math.max(layer.attnMatrix.highlight, t2.t * 0.5);
        }
        for (let blk of layout.audioCnn) blk.opacity = 0.25;
        for (let layer of layout.audioTransformerLayers) {
            for (let blk of getTransformerLayerBlocks(layer)) blk.opacity = 0.25;
        }
    }

    breakAfter();

    // Step 4: Pan to T3 tower — the new addition
    let t3 = afterTime(null, 2.0, 0.5);
    let t2c = afterTime(t2, 0.4);
    cleanup(t2c, [t2]);

    moveT3TowerCameraTo(state, t3, new Vec3(200, 0, -300), new Vec3(286, 18, 8));

    let dT3 = c_dimRef('256D', T3TowerDim.D_t3);
    let t3Tower = c_dimRef('T3 Tower', T3TowerDim.T3Tower);

    commentary()`The ${t3Tower} is a lightweight 2-layer transformer at ${dT3} — much smaller than Audio (1024D, 4L) or MIDI (512D, 4L). This compact design is intentional: the third modality requires less capacity, and keeping it small prevents it from dominating the shared space. It processes auxiliary signals that complement the audio-MIDI alignment.`;
    breakAfter();

    if (t3.active) {
        for (let layer of layout.t3TransformerLayers) {
            layer.attnMatrix.highlight = Math.max(layer.attnMatrix.highlight, t3.t * 0.6);
            layer.mlpUp.highlight = Math.max(layer.mlpUp.highlight, t3.t * 0.3);
        }
        layout.t3InputProj.highlight = Math.max(layout.t3InputProj.highlight, t3.t * 0.4);

        let layer0 = layout.t3TransformerLayers[0];
        drawDimLabels(layer0.attnMatrix, t3.t * 0.7);
    }

    breakAfter();

    // Step 5: Zoom out to see shared space — all three converge
    let t4 = afterTime(null, 2.0, 0.5);
    let t3c = afterTime(t3, 0.4);
    cleanup(t3c, [t3]);

    moveT3TowerCameraTo(state, t4, new Vec3(30, 0, -1000), new Vec3(290, 25, 24));

    let audioEmb = c_blockRef('audio embedding', layout.audioEmbedding);
    let midiEmb = c_blockRef('MIDI embedding', layout.midiEmbedding);
    let t3Emb = c_blockRef('T3 embedding', layout.t3Embedding);
    let vicAM = c_blockRef('Audio↔MIDI', layout.vicregAM);
    let vicAT = c_blockRef('Audio↔T3', layout.vicregAT);
    let vicMT = c_blockRef('MIDI↔T3', layout.vicregMT);

    commentary()`All three towers converge into a shared ${d256} space. The ${audioEmb}, ${midiEmb}, and ${t3Emb} are aligned using 3-way VICReg: three pairwise loss terms (${vicAM}, ${vicAT}, ${vicMT}) that enforce alignment without negative pairs. This triangular constraint creates a richer embedding space than any two-tower model can achieve.`;
    breakAfter();

    if (t4.active) {
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, t4.t * 0.6);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, t4.t * 0.6);
        layout.t3Embedding.highlight = Math.max(layout.t3Embedding.highlight, t4.t * 0.6);
        layout.vicregAM.highlight = Math.max(layout.vicregAM.highlight, t4.t * 0.7);
        layout.vicregAT.highlight = Math.max(layout.vicregAT.highlight, t4.t * 0.7);
        layout.vicregMT.highlight = Math.max(layout.vicregMT.highlight, t4.t * 0.7);

        drawDimLabels(layout.audioEmbedding, t4.t * 0.7);
    }

    breakAfter();
}
