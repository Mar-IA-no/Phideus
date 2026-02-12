import { Vec3 } from "@/src/utils/vector";
import { IBloqueaWalkthroughArgs, setBloqueaInitialCamera, moveBloqueaCameraTo } from "./BloqueaWalkthrough";
import { getTransformerLayerBlocks, getAdapterBlocks } from "../BloqueaModelLayout";
import { BloqueaDim } from "../BloqueaDimStyle";

export function bloqueaPhase07_RunCConfig(args: IBloqueaWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, drawDimLabels } = tools;

    setBloqueaInitialCamera(state, new Vec3(0, 0, -600), new Vec3(290, 22, 18));

    let frozen = c_dimRef('FROZEN (lr=0)', BloqueaDim.Frozen);
    let lrAdapter = c_dimRef('lr_adapter (5e-4)', BloqueaDim.Adapter);
    let lrUnfreeze = c_dimRef('lr_unfreeze (1e-5)', BloqueaDim.Unfrozen);
    let lrMidi = c_dimRef('lr_midi (5e-5)', BloqueaDim.Trainable);
    let lrProj = c_dimRef('lr_proj (1e-4)', BloqueaDim.Projection);

    commentary()`Run C's training configuration defines 4 parameter groups plus frozen components. Unlike Run D's 5 groups (which include lr_low and lr_high for audio transformer layers), Run C replaces those with two distinct strategies: ${lrAdapter} for adapter modules in frozen layers 0-1, and ${lrUnfreeze} for directly unfrozen layers 2-3. This hybrid approach aims to match Run D's performance with potentially better parameter efficiency.`;
    breakAfter();

    // Step 1: Frozen components
    let t0 = afterTime(null, 2.5, 0.5);

    moveBloqueaCameraTo(state, t0, new Vec3(-100, 0, -150), new Vec3(292, 12, 5));

    let cnnRef = c_blockRef('CNN', layout.audioCnn);

    commentary()`${frozen}: The ${cnnRef}, positional embedding, and transformer layers 0-1 (excluding adapters) are completely frozen. Zero gradients. MERT's learned audio understanding is preserved intact. This frozen base is _deeper_ than in Run D, where all 4 transformer layers receive some gradient.`;
    breakAfter();

    if (t0.active) {
        for (let blk of layout.audioCnn) {
            blk.opacity = Math.max(blk.opacity, 0.5 + 0.3 * t0.t);
            blk.highlight = Math.max(blk.highlight, t0.t * 0.15);
        }
        layout.audioPosEmb.opacity = Math.max(layout.audioPosEmb.opacity, 0.5 + 0.3 * t0.t);
        layout.audioPosEmb.highlight = Math.max(layout.audioPosEmb.highlight, t0.t * 0.15);
        // Show frozen transformer blocks in layers 0-1
        for (let i = 0; i < 2; i++) {
            let layer = layout.audioTransformerLayers[i];
            for (let blk of getTransformerLayerBlocks(layer)) {
                if (!getAdapterBlocks(layer).includes(blk)) {
                    blk.opacity = Math.max(blk.opacity, 0.45);
                    blk.highlight = Math.max(blk.highlight, t0.t * 0.1);
                }
            }
        }
    }

    breakAfter();

    // Step 2: Adapter modules — the fastest learners
    let t1 = afterTime(null, 3.0, 0.5);
    let t0c = afterTime(t0, 0.4);
    cleanup(t0c, [t0]);

    let layer0 = layout.audioTransformerLayers[0];
    moveBloqueaCameraTo(state, t1, new Vec3(-100, 0, -layer0.ln1.y - 150), new Vec3(291, 16, 7));

    commentary()`${lrAdapter}: The adapter bottlenecks train at the _fastest_ rate — 5e-4. Starting from near-identity initialization (up-projection weights ≈ 0), they must rapidly learn useful residual corrections. Two adapters, ~262K parameters total, represent the only trainable pathway through frozen layers 0-1. Their fast learning rate compensates for their tiny parameter count.`;
    breakAfter();

    if (t1.active) {
        let fastPulse = 0.5 + 0.35 * Math.sin(wt.time * 4);
        for (let i = 0; i < 2; i++) {
            let layer = layout.audioTransformerLayers[i];
            for (let blk of getAdapterBlocks(layer)) {
                blk.highlight = Math.max(blk.highlight, fastPulse);
            }
        }
        // Frozen blocks dim behind
        for (let i = 0; i < 2; i++) {
            for (let blk of getTransformerLayerBlocks(layout.audioTransformerLayers[i])) {
                if (!getAdapterBlocks(layout.audioTransformerLayers[i]).includes(blk)) {
                    blk.opacity = Math.min(blk.opacity, 0.3);
                }
            }
        }
    }

    breakAfter();

    // Step 3: Unfrozen layers 2-3
    let t2 = afterTime(null, 3.0, 0.5);
    let t1c = afterTime(t1, 0.4);
    cleanup(t1c, [t1]);

    let layer2 = layout.audioTransformerLayers[2];
    moveBloqueaCameraTo(state, t2, new Vec3(-100, 0, -layer2.ln1.y - 100), new Vec3(291, 14, 5));

    commentary()`${lrUnfreeze}: Layers 2-3 are directly unfrozen at 1e-5 — 50x slower than the adapters, but with access to _all_ parameters. Every Q/K/V, attention, FFN weight is trainable. This gives the upper layers freedom to fundamentally restructure for cross-modal alignment, unlike the narrow adapter corrections in lower layers.`;
    breakAfter();

    if (t2.active) {
        let pulse = 0.25 + 0.2 * Math.sin(wt.time * 3);
        for (let i = 2; i < 4; i++) {
            let layer = layout.audioTransformerLayers[i];
            for (let blk of getTransformerLayerBlocks(layer)) {
                blk.highlight = Math.max(blk.highlight, pulse);
            }
            layer.attnMatrix.highlight = Math.max(layer.attnMatrix.highlight, pulse + 0.2);
        }
    }

    breakAfter();

    // Step 4: MIDI encoder
    let t3 = afterTime(null, 2.5, 0.5);
    let t2c = afterTime(t2, 0.4);
    cleanup(t2c, [t2]);

    moveBloqueaCameraTo(state, t3, new Vec3(90, 0, -400), new Vec3(288, 18, 10));

    let midiEnc = c_blockRef('MIDI encoder', layout.midiTransformerLayers.flatMap(getTransformerLayerBlocks));

    commentary()`${lrMidi}: The ${midiEnc} trains from scratch at 5e-5. Identical to Run D. No pre-trained weights, no adapters needed — the entire MIDI pathway is fresh and requires its own learning rate tuned for convergence from random initialization.`;
    breakAfter();

    if (t3.active) {
        let totalLayers = layout.midiTransformerLayers.length;
        for (let i = 0; i < totalLayers; i++) {
            let layerPhase = (t3.t * (totalLayers + 2)) - i;
            if (layerPhase > 0) {
                let intensity = Math.min(1, layerPhase);
                let pulse = intensity * (0.3 + 0.2 * Math.sin(wt.time * 3 + i * 0.8));
                let layer = layout.midiTransformerLayers[i];
                layer.attnMatrix.highlight = Math.max(layer.attnMatrix.highlight, pulse + 0.15);
                layer.mlpUp.highlight = Math.max(layer.mlpUp.highlight, pulse);
            }
        }
    }

    breakAfter();

    // Step 5: Projection heads
    let t4 = afterTime(null, 2.0, 0.5);
    let t3c = afterTime(t3, 0.4);
    cleanup(t3c, [t3]);

    moveBloqueaCameraTo(state, t4, new Vec3(0, 0, -layout.audioEmbedding.y + 80), new Vec3(290, 16, 7));

    commentary()`${lrProj}: Projection heads at 1e-4 — fast, but not as fast as the adapters (5e-4). In Run D, projections were the fastest at 5e-4. Run C gives that crown to the adapters, reflecting their need for rapid convergence from near-zero initialization.`;
    breakAfter();

    if (t4.active) {
        let projPulse = 0.5 + 0.35 * Math.sin(wt.time * 4);
        for (let blk of layout.audioProjLayers) blk.highlight = Math.max(blk.highlight, projPulse);
        for (let blk of layout.midiProjLayers) blk.highlight = Math.max(blk.highlight, projPulse);
        let embGlow = 0.4 + 0.2 * Math.sin(wt.time * 3);
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, embGlow);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, embGlow);
    }

    breakAfter();

    // Step 6: GRAND FINALE — cascading illumination
    let t5 = afterTime(null, 5.0, 0.5);
    let t4c = afterTime(t4, 0.5);
    cleanup(t4c, [t4]);

    moveBloqueaCameraTo(state, t5, new Vec3(0, 0, -550), new Vec3(290, 25, 20));

    commentary()`The complete BloqueA Run C architecture: frozen CNN + frozen layers 0-1 with adapter bottlenecks (5e-4) + unfrozen layers 2-3 (1e-5) + MIDI encoder from scratch (5e-5) + projection heads (1e-4) + VICReg alignment. The hybrid adapter-unfreeze strategy aims to combine the parameter efficiency of adapters with the expressive power of direct fine-tuning — a middle ground between Run D's uniform unfreezing and a pure adapter approach.`;
    breakAfter();

    if (t5.active) {
        let maxY = layout.vicregCov.y + layout.vicregCov.dy;
        let sweepSpeed = 1.5;
        let sweepPos = (t5.t * sweepSpeed) * maxY;
        let glowWidth = maxY * 0.25;

        for (let cube of layout.cubes) {
            let cubeY = cube.y + cube.dy / 2;
            let dist = Math.abs(cubeY - sweepPos);

            if (dist < glowWidth) {
                let intensity = 1 - (dist / glowWidth);
                let pulse = intensity * (0.4 + 0.2 * Math.sin(wt.time * 2 + cubeY * 0.01));
                cube.highlight = Math.max(cube.highlight, pulse);
                cube.opacity = Math.max(cube.opacity, 0.7 + intensity * 0.3);
            }

            if (cubeY < sweepPos - glowWidth * 0.3) {
                let afterglow = 0.15 + 0.08 * Math.sin(wt.time * 1.5 + cubeY * 0.02);
                cube.highlight = Math.max(cube.highlight, afterglow);
                cube.opacity = Math.max(cube.opacity, 0.85);
            }
        }

        if (t5.t > 0.6) {
            let lossPhase = wt.time * 2.5;
            layout.vicregInv.highlight = Math.max(layout.vicregInv.highlight, 0.4 + 0.3 * Math.sin(lossPhase));
            layout.vicregVar.highlight = Math.max(layout.vicregVar.highlight, 0.4 + 0.3 * Math.sin(lossPhase + 2.1));
            layout.vicregCov.highlight = Math.max(layout.vicregCov.highlight, 0.4 + 0.3 * Math.sin(lossPhase + 4.2));
        }
    }

    breakAfter();

    // Step 7: Final orbit
    let t6 = afterTime(null, 4.0, 0.5);
    let t5c = afterTime(t5, 0.5);
    cleanup(t5c, [t5]);

    moveBloqueaCameraTo(state, t6, new Vec3(0, 0, -550), new Vec3(310, 22, 20));

    commentary()`From raw waveform to musical understanding. Run C's hybrid approach — adapters for preservation, unfreezing for adaptation — represents one point in the design space between full fine-tuning and pure parameter-efficient methods. The adapters' dramatic 16x bottleneck (1024→64) forces the model to distill its cross-modal corrections into the most compact representation possible, while unfrozen upper layers provide the expressive power for complex alignment patterns.`;
    breakAfter();

    if (t6.active) {
        for (let cube of layout.cubes) {
            cube.opacity = Math.max(cube.opacity, 0.8);
            let warmGlow = 0.12 + 0.08 * Math.sin(wt.time * 1.2 + cube.y * 0.008 + cube.x * 0.005);
            cube.highlight = Math.max(cube.highlight, warmGlow);
        }
        // Adapter blocks extra bright
        for (let i = 0; i < 2; i++) {
            for (let blk of getAdapterBlocks(layout.audioTransformerLayers[i])) {
                blk.highlight = Math.max(blk.highlight, 0.4 + 0.2 * Math.sin(wt.time * 2));
            }
        }
        // Shared space burns
        let sharedPulse = 0.6 + 0.3 * Math.sin(wt.time * 2);
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, sharedPulse);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, sharedPulse);
    }

    breakAfter();
}
