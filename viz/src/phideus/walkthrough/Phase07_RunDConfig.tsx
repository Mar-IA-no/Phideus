import { Vec3 } from "@/src/utils/vector";
import { IPhideusWalkthroughArgs, setPhideusInitialCamera, movePhideusCameraTo } from "./PhideusWalkthrough";
import { getTransformerLayerBlocks } from "../PhideusModelLayout";
import { PhideusDim } from "../PhideusDimStyle";

export function phideusPhase07_RunDConfig(args: IPhideusWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, drawDimLabels } = tools;

    // NO dimming — this phase shows the full model
    setPhideusInitialCamera(state, new Vec3(0, 0, -600), new Vec3(290, 22, 18));

    let frozen = c_dimRef('FROZEN (lr=0)', PhideusDim.Frozen);
    let lrLow = c_dimRef('lr_low (3e-6)', PhideusDim.Trainable);
    let lrHigh = c_dimRef('lr_high (1e-5)', PhideusDim.Trainable);
    let lrMidiRef = c_dimRef('lr_midi (1e-4)', PhideusDim.Trainable);
    let lrProj = c_dimRef('lr_proj (5e-4)', PhideusDim.Projection);

    commentary()`Run D is Phideus's training configuration — the product of careful ablation studies. Five learning rate groups create a gradient of plasticity across the architecture, from completely ${frozen} components to aggressively adapting projection heads at ${lrProj}. The key insight: not all parameters are equal. Pre-trained knowledge must be preserved, while new cross-modal connections must be forged rapidly.`;
    breakAfter();

    // Step 1: Frozen blocks — dramatic close-up on the CNN
    let t0 = afterTime(null, 2.5, 0.5);

    let cnnRef = c_blockRef('CNN feature extractor', layout.audioCnn);
    let audioPosRef = c_blockRef('audio positional embedding', layout.audioPosEmb);
    let midiPosRef = c_blockRef('MIDI positional encoding', layout.midiPosEnc);

    movePhideusCameraTo(state, t0, new Vec3(-100, 0, -150), new Vec3(292, 12, 5));

    commentary()`${frozen}: The ${cnnRef} is a sacred artifact — MERT's convolutional stack, pre-trained on 160K hours of music. These 4 stages of convolutions extract spectral features, onset transients, and harmonic structure from raw audio. The ${audioPosRef} and ${midiPosRef} are also frozen. Zero gradients. Zero risk. These layers are too valuable to modify.`;
    breakAfter();

    if (t0.active) {
        // Frozen blocks glow with a cold, static highlight — no pulse
        for (let blk of layout.audioCnn) {
            blk.opacity = Math.max(blk.opacity, 0.5 + 0.3 * t0.t);
            blk.highlight = Math.max(blk.highlight, t0.t * 0.15);
        }
        layout.audioPosEmb.opacity = Math.max(layout.audioPosEmb.opacity, 0.5 + 0.3 * t0.t);
        layout.audioPosEmb.highlight = Math.max(layout.audioPosEmb.highlight, t0.t * 0.15);
        layout.midiPosEnc.opacity = Math.max(layout.midiPosEnc.opacity, 0.5 + 0.3 * t0.t);
        layout.midiPosEnc.highlight = Math.max(layout.midiPosEnc.highlight, t0.t * 0.15);

        if (t0.t > 0.5) {
            drawDimLabels(layout.audioCnn[0], (t0.t - 0.5) * 2 * 0.6);
            drawDimLabels(layout.audioCnn[3], (t0.t - 0.5) * 2 * 0.6);
        }
    }

    breakAfter();

    // Step 2: lr_low — careful fine-tuning of layers 0-1
    let t1 = afterTime(null, 3.0, 0.5);
    let t0c = afterTime(t0, 0.4);
    cleanup(t0c, [t0]);

    let layer0 = layout.audioTransformerLayers[0];
    movePhideusCameraTo(state, t1, new Vec3(-100, 0, -layer0.ln1.y - 100), new Vec3(291, 14, 5));

    commentary()`${lrLow}: Audio transformer layers 0-1 hold MERT's mid-level representations — the layers that learned pitch tracking, rhythmic pattern recognition, and timbre analysis during pre-training. At 3e-6, we're whispering adjustments: "keep everything you know, but nudge your attention patterns _slightly_ toward cross-modal alignment." Q/K/V weights and FFN parameters shift by micrograms, not kilograms.`;
    breakAfter();

    if (t1.active) {
        // Gentle, slow pulse for conservative layers
        let gentlePulse = 0.15 + 0.1 * Math.sin(wt.time * 2);
        for (let i = 0; i < 2; i++) {
            let layer = layout.audioTransformerLayers[i];
            for (let blk of getTransformerLayerBlocks(layer)) {
                blk.highlight = Math.max(blk.highlight, gentlePulse);
            }
            layer.attnMatrix.highlight = Math.max(layer.attnMatrix.highlight, gentlePulse + 0.15);
            drawDimLabels(layer.attnMatrix, t1.t * 0.5);
        }
        // Dim upper layers
        for (let i = 2; i < layout.audioTransformerLayers.length; i++) {
            for (let blk of getTransformerLayerBlocks(layout.audioTransformerLayers[i])) {
                blk.opacity = Math.min(blk.opacity, 0.25);
            }
        }
    }

    breakAfter();

    // Step 3: lr_high — more aggressive layers 2-3
    let t2 = afterTime(null, 3.0, 0.5);
    let t1c = afterTime(t1, 0.4);
    cleanup(t1c, [t1]);

    let layer2 = layout.audioTransformerLayers[2];
    movePhideusCameraTo(state, t2, new Vec3(-100, 0, -layer2.ln1.y - 100), new Vec3(291, 14, 5));

    commentary()`${lrHigh}: Audio transformer layers 2-3 receive ~3x more gradient pressure. These upper layers need to develop new capabilities — learning which audio features matter for MIDI correspondence, reshaping attention patterns to focus on musically-relevant temporal structures. The Q/K/V projections here learn entirely new query-key relationships optimized for cross-modal matching.`;
    breakAfter();

    if (t2.active) {
        // Stronger pulse for more aggressive training
        let strongPulse = 0.25 + 0.2 * Math.sin(wt.time * 3);
        for (let i = 2; i < 4; i++) {
            let layer = layout.audioTransformerLayers[i];
            for (let blk of getTransformerLayerBlocks(layer)) {
                blk.highlight = Math.max(blk.highlight, strongPulse);
            }
            layer.attnMatrix.highlight = Math.max(layer.attnMatrix.highlight, strongPulse + 0.2);
            layer.mlpUp.highlight = Math.max(layer.mlpUp.highlight, strongPulse + 0.1);
            drawDimLabels(layer.attnMatrix, t2.t * 0.5);
        }
        // Dim lower layers
        for (let i = 0; i < 2; i++) {
            for (let blk of getTransformerLayerBlocks(layout.audioTransformerLayers[i])) {
                blk.opacity = Math.min(blk.opacity, 0.25);
            }
        }
    }

    breakAfter();

    // Step 4: lr_midi — the from-scratch encoder
    let t3 = afterTime(null, 3.0, 0.5);
    let t2c = afterTime(t2, 0.4);
    cleanup(t2c, [t2]);

    movePhideusCameraTo(state, t3, new Vec3(90, 0, -400), new Vec3(288, 18, 10));

    let midiEnc = c_blockRef('MIDI encoder', layout.midiTransformerLayers.flatMap(getTransformerLayerBlocks));

    commentary()`${lrMidiRef}: The entire ${midiEnc} trains from scratch at 33x the base rate. No pre-trained weights exist for this task — MIDI-to-audio alignment is a novel problem. Every Q/K/V projection, every attention pattern, every FFN weight starts from random initialization and must converge to meaningful musical event representations. The embedding tables (pitch, velocity, duration) train at this rate too.`;
    breakAfter();

    if (t3.active) {
        // MIDI tower lights up from bottom to top (cascade)
        let totalLayers = layout.midiTransformerLayers.length;
        for (let i = 0; i < totalLayers; i++) {
            let layerPhase = (t3.t * (totalLayers + 2)) - i;
            if (layerPhase > 0) {
                let intensity = Math.min(1, layerPhase);
                let pulse = intensity * (0.3 + 0.2 * Math.sin(wt.time * 3 + i * 0.8));
                let layer = layout.midiTransformerLayers[i];
                layer.attnMatrix.highlight = Math.max(layer.attnMatrix.highlight, pulse + 0.15);
                layer.mlpUp.highlight = Math.max(layer.mlpUp.highlight, pulse);
                layer.qWeight.highlight = Math.max(layer.qWeight.highlight, pulse * 0.8);
                layer.kWeight.highlight = Math.max(layer.kWeight.highlight, pulse * 0.8);
                layer.vWeight.highlight = Math.max(layer.vWeight.highlight, pulse * 0.8);
            }
        }
        // Embedding tables glow
        let embPulse = 0.3 + 0.15 * Math.sin(wt.time * 2.5);
        layout.midiPitchEmb.highlight = Math.max(layout.midiPitchEmb.highlight, embPulse);
        layout.midiVelEmb.highlight = Math.max(layout.midiVelEmb.highlight, embPulse);
        layout.midiDurEmb.highlight = Math.max(layout.midiDurEmb.highlight, embPulse);
    }

    breakAfter();

    // Step 5: lr_proj — the fastest adapters
    let t4 = afterTime(null, 2.5, 0.5);
    let t3c = afterTime(t3, 0.4);
    cleanup(t3c, [t3]);

    movePhideusCameraTo(state, t4, new Vec3(0, 0, -layout.audioEmbedding.y + 80), new Vec3(290, 16, 7));

    let audioProjRef = c_blockRef('audio projection', layout.audioProjLayers);
    let midiProjRef = c_blockRef('MIDI projection', layout.midiProjLayers);

    commentary()`${lrProj}: The ${audioProjRef} and ${midiProjRef} heads sprint at 167x the base rate. These MLPs must rapidly learn the most critical transformation in the system: mapping 1024D audio features and 512D MIDI features into a unified 256D space. They carry ~2.5M parameters each — the fastest-moving parts of a ~12M trainable parameter system.`;
    breakAfter();

    if (t4.active) {
        // Projection layers pulse intensely
        let fastPulse = 0.5 + 0.35 * Math.sin(wt.time * 4);
        for (let blk of layout.audioProjLayers) blk.highlight = Math.max(blk.highlight, fastPulse);
        for (let blk of layout.midiProjLayers) blk.highlight = Math.max(blk.highlight, fastPulse);

        // Shared space embeddings glow
        let embGlow = 0.4 + 0.2 * Math.sin(wt.time * 3);
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, embGlow);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, embGlow);
    }

    breakAfter();

    // Step 6: GRAND FINALE — cascading illumination of the entire architecture
    let t5 = afterTime(null, 5.0, 0.5);
    let t4c = afterTime(t4, 0.5);
    cleanup(t4c, [t4]);

    movePhideusCameraTo(state, t5, new Vec3(0, 0, -550), new Vec3(290, 25, 20));

    commentary()`The complete Phideus architecture: a pre-trained audio encoder (MERT) carefully fine-tuned with differential learning rates, a MIDI encoder trained from scratch, projection heads that bridge two modalities into a shared 256D space, and VICReg loss that aligns them without negative pairs. ~55M total parameters, ~12M trainable. The learning rate gradient — from ${frozen} to ${lrProj} — is the key design choice: it lets the model leverage MERT's audio expertise while learning entirely new cross-modal relationships.`;
    breakAfter();

    if (t5.active) {
        // CASCADING LIGHT: sweep from top (inputs) to bottom (loss)
        // Total blocks distance: use Y coordinate for ordering
        let minY = 0;
        let maxY = layout.vicregCov.y + layout.vicregCov.dy;
        let sweepSpeed = 1.5; // seconds to sweep through
        let sweepPos = (t5.t * sweepSpeed) * maxY;
        let glowWidth = maxY * 0.25; // how wide the glow band is

        for (let cube of layout.cubes) {
            let cubeY = cube.y + cube.dy / 2;
            let dist = Math.abs(cubeY - sweepPos);

            if (dist < glowWidth) {
                let intensity = 1 - (dist / glowWidth);
                let pulse = intensity * (0.4 + 0.2 * Math.sin(wt.time * 2 + cubeY * 0.01));
                cube.highlight = Math.max(cube.highlight, pulse);
                cube.opacity = Math.max(cube.opacity, 0.7 + intensity * 0.3);
            }

            // After sweep passes, leave a warm afterglow
            if (cubeY < sweepPos - glowWidth * 0.3) {
                let afterglow = 0.15 + 0.08 * Math.sin(wt.time * 1.5 + cubeY * 0.02);
                cube.highlight = Math.max(cube.highlight, afterglow);
                cube.opacity = Math.max(cube.opacity, 0.85);
            }
        }

        // VICReg terms always pulse at the bottom
        if (t5.t > 0.6) {
            let lossPhase = wt.time * 2.5;
            layout.vicregInv.highlight = Math.max(layout.vicregInv.highlight, 0.4 + 0.3 * Math.sin(lossPhase));
            layout.vicregVar.highlight = Math.max(layout.vicregVar.highlight, 0.4 + 0.3 * Math.sin(lossPhase + 2.1));
            layout.vicregCov.highlight = Math.max(layout.vicregCov.highlight, 0.4 + 0.3 * Math.sin(lossPhase + 4.2));
        }
    }

    breakAfter();

    // Step 7: Final slow orbit with everything lit
    let t6 = afterTime(null, 4.0, 0.5);
    let t5c = afterTime(t5, 0.5);
    cleanup(t5c, [t5]);

    // Slow camera orbit — rotates around the Y axis
    movePhideusCameraTo(state, t6, new Vec3(0, 0, -550), new Vec3(310, 22, 20));

    commentary()`From raw waveform to musical understanding. Audio enters as 24kHz samples, MIDI as discrete note events. Through CNNs, transformers, projections, and VICReg alignment, Phideus learns that a C major chord played on a piano and the MIDI notation C4-E4-G4 are the same musical idea — encoded as nearby points in a 256-dimensional space. This is cross-modal representation learning.`;
    breakAfter();

    if (t6.active) {
        // Everything alive with gentle, warm pulsing
        for (let cube of layout.cubes) {
            cube.opacity = Math.max(cube.opacity, 0.8);
            let warmGlow = 0.12 + 0.08 * Math.sin(wt.time * 1.2 + cube.y * 0.008 + cube.x * 0.005);
            cube.highlight = Math.max(cube.highlight, warmGlow);
        }

        // Attention matrices stand out
        for (let layer of layout.audioTransformerLayers) {
            layer.attnMatrix.highlight = Math.max(layer.attnMatrix.highlight, 0.3 + 0.15 * Math.sin(wt.time * 1.8));
        }
        for (let layer of layout.midiTransformerLayers) {
            layer.attnMatrix.highlight = Math.max(layer.attnMatrix.highlight, 0.3 + 0.15 * Math.sin(wt.time * 1.8));
        }

        // Shared space burns bright
        let sharedPulse = 0.6 + 0.3 * Math.sin(wt.time * 2);
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, sharedPulse);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, sharedPulse);
        layout.vicregInv.highlight = Math.max(layout.vicregInv.highlight, sharedPulse * 0.8);
        layout.vicregVar.highlight = Math.max(layout.vicregVar.highlight, sharedPulse * 0.8);
        layout.vicregCov.highlight = Math.max(layout.vicregCov.highlight, sharedPulse * 0.8);
    }

    breakAfter();
}
