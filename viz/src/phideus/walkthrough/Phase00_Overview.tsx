import { Vec3 } from "@/src/utils/vector";
import { IPhideusWalkthroughArgs, setPhideusInitialCamera, movePhideusCameraTo } from "./PhideusWalkthrough";
import { getTransformerLayerBlocks } from "../PhideusModelLayout";
import { PhideusDim } from "../PhideusDimStyle";

export function phideusPhase00_Overview(args: IPhideusWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { atTime, afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, drawDimLabels } = tools;

    // Wide view showing both towers
    setPhideusInitialCamera(state, new Vec3(0, 0, -600), new Vec3(290, 22, 16));

    let audioTfRef = c_blockRef('Audio Transformer', layout.audioTransformerLayers.flatMap(getTransformerLayerBlocks));
    let midiTfRef = c_blockRef('MIDI Transformer', layout.midiTransformerLayers.flatMap(getTransformerLayerBlocks));
    let d256 = c_dimRef('256D', PhideusDim.D_proj);

    commentary()`Phideus is a cross-modal architecture that learns to align audio recordings with their MIDI scores. Two independent encoder towers — the ${audioTfRef} and the ${midiTfRef} — process audio and MIDI, mapping both into a shared ${d256} embedding space via VICReg self-supervised loss.`;
    breakAfter();

    // Step 1: Full overview
    let t0 = afterTime(null, 1.5, 0.5);

    if (t0.active) {
        // Gently highlight both towers
        for (let blk of layout.audioCnn) blk.highlight = Math.max(blk.highlight, t0.t * 0.2);
        for (let layer of layout.audioTransformerLayers) {
            layer.attnMatrix.highlight = Math.max(layer.attnMatrix.highlight, t0.t * 0.25);
        }
        for (let layer of layout.midiTransformerLayers) {
            layer.attnMatrix.highlight = Math.max(layer.attnMatrix.highlight, t0.t * 0.25);
        }
    }

    breakAfter();

    // Step 2: Zoom to audio tower
    let t1 = afterTime(null, 1.5, 0.5);
    let t0c = afterTime(t0, 0.4);
    cleanup(t0c, [t0]);

    movePhideusCameraTo(state, t1, new Vec3(-80, 0, -400), new Vec3(292, 20, 10));

    let cnnRef = c_blockRef('CNN feature extractor', layout.audioCnn);
    let dAudio = c_dimRef('1024D', PhideusDim.D_audio);
    let tAudio = c_dimRef('T_audio', PhideusDim.T_audio);

    commentary()`The Audio Tower processes raw 24kHz waveforms through a frozen ${cnnRef} (4 stages), then refines them with 4 transformer layers. Each layer contains Q/K/V projections, a ${tAudio} x ${tAudio} attention matrix, and an FFN with GELU activation. Output dimension is ${dAudio}.`;
    breakAfter();

    if (t1.active) {
        for (let layer of layout.audioTransformerLayers) {
            layer.attnMatrix.highlight = Math.max(layer.attnMatrix.highlight, t1.t * 0.5);
            layer.mlpUp.highlight = Math.max(layer.mlpUp.highlight, t1.t * 0.3);
            layer.qWeight.highlight = Math.max(layer.qWeight.highlight, t1.t * 0.2);
        }
        for (let layer of layout.midiTransformerLayers) {
            for (let blk of getTransformerLayerBlocks(layer)) blk.opacity = 0.3;
        }
        layout.midiInput.opacity = 0.3;
        layout.midiPitchEmb.opacity = 0.3;
        layout.midiVelEmb.opacity = 0.3;
        layout.midiDurEmb.opacity = 0.3;
        layout.midiCombineLinear.opacity = 0.3;
        layout.midiPosEnc.opacity = 0.3;

        let layer0 = layout.audioTransformerLayers[0];
        drawDimLabels(layer0.attnMatrix, t1.t * 0.7);
    }

    breakAfter();

    // Step 3: Pan to MIDI tower
    let t2 = afterTime(null, 1.5, 0.5);
    let t1c = afterTime(t1, 0.4);
    cleanup(t1c, [t1]);

    movePhideusCameraTo(state, t2, new Vec3(80, 0, -400), new Vec3(288, 20, 10));

    let pitchEmb = c_blockRef('pitch embedding', layout.midiPitchEmb);
    let velEmb = c_blockRef('velocity embedding', layout.midiVelEmb);
    let durEmb = c_blockRef('duration embedding', layout.midiDurEmb);
    let dMidi = c_dimRef('512D', PhideusDim.D_midi);

    commentary()`The MIDI Tower converts discrete musical events (${pitchEmb}, ${velEmb}, ${durEmb}) into continuous vector representations, then processes them through 4 transformer layers. The entire MIDI encoder is trained from scratch at ${dMidi} — there is no pre-trained model to leverage here.`;
    breakAfter();

    if (t2.active) {
        for (let layer of layout.midiTransformerLayers) {
            layer.attnMatrix.highlight = Math.max(layer.attnMatrix.highlight, t2.t * 0.5);
            layer.mlpUp.highlight = Math.max(layer.mlpUp.highlight, t2.t * 0.3);
            layer.qWeight.highlight = Math.max(layer.qWeight.highlight, t2.t * 0.2);
        }
        for (let blk of layout.audioCnn) blk.opacity = 0.3;
        layout.audioPosEmb.opacity = 0.3;
        layout.audioInput.opacity = 0.3;
        for (let layer of layout.audioTransformerLayers) {
            for (let blk of getTransformerLayerBlocks(layer)) blk.opacity = 0.3;
        }

        let layer0 = layout.midiTransformerLayers[0];
        drawDimLabels(layer0.attnMatrix, t2.t * 0.7);
    }

    breakAfter();

    // Step 4: Zoom out to see shared space
    let t3 = afterTime(null, 2.0, 0.5);
    let t2c = afterTime(t2, 0.4);
    cleanup(t2c, [t2]);

    movePhideusCameraTo(state, t3, new Vec3(0, 0, -900), new Vec3(290, 25, 20));

    let audioEmb = c_blockRef('audio embedding', layout.audioEmbedding);
    let midiEmb = c_blockRef('MIDI embedding', layout.midiEmbedding);
    let vicInv = c_blockRef('invariance', layout.vicregInv);
    let vicVar = c_blockRef('variance', layout.vicregVar);
    let vicCov = c_blockRef('covariance', layout.vicregCov);

    commentary()`Both towers converge into a shared ${d256} embedding space through projection heads. The ${audioEmb} and ${midiEmb} are aligned using VICReg loss — three terms (${vicInv}, ${vicVar}, ${vicCov}) that train the model without requiring negative pairs or momentum encoders.`;
    breakAfter();

    if (t3.active) {
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, t3.t * 0.6);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, t3.t * 0.6);
        layout.vicregInv.highlight = Math.max(layout.vicregInv.highlight, t3.t * 0.7);
        layout.vicregVar.highlight = Math.max(layout.vicregVar.highlight, t3.t * 0.7);
        layout.vicregCov.highlight = Math.max(layout.vicregCov.highlight, t3.t * 0.7);

        drawDimLabels(layout.audioEmbedding, t3.t * 0.7);
    }

    breakAfter();
}
