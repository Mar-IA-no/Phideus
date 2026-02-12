import { Vec3 } from "@/src/utils/vector";
import { IBloqueaWalkthroughArgs, setBloqueaInitialCamera, moveBloqueaCameraTo } from "./BloqueaWalkthrough";
import { getTransformerLayerBlocks, getAdapterBlocks } from "../BloqueaModelLayout";
import { BloqueaDim } from "../BloqueaDimStyle";

export function bloqueaPhase00_Overview(args: IBloqueaWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, drawDimLabels } = tools;

    setBloqueaInitialCamera(state, new Vec3(0, 0, -600), new Vec3(290, 22, 16));

    let audioTfRef = c_blockRef('Audio Transformer', layout.audioTransformerLayers.flatMap(getTransformerLayerBlocks));
    let midiTfRef = c_blockRef('MIDI Transformer', layout.midiTransformerLayers.flatMap(getTransformerLayerBlocks));
    let d256 = c_dimRef('256D', BloqueaDim.D_proj);
    let adapterRef = c_dimRef('adapter bottlenecks', BloqueaDim.Adapter);

    commentary()`BloqueA Run C is a _hybrid_ adapter architecture. Unlike Run D's uniform fine-tuning, Run C freezes transformer layers 0-1 and injects ${adapterRef} (1024→64→1024) after each, while directly unfreezing layers 2-3. The ${audioTfRef} and ${midiTfRef} map both modalities into a shared ${d256} space via VICReg loss.`;
    breakAfter();

    // Step 1: Full overview
    let t0 = afterTime(null, 1.5, 0.5);

    if (t0.active) {
        for (let blk of layout.audioCnn) blk.highlight = Math.max(blk.highlight, t0.t * 0.2);
        for (let layer of layout.audioTransformerLayers) {
            layer.attnMatrix.highlight = Math.max(layer.attnMatrix.highlight, t0.t * 0.25);
            // Highlight adapter blocks extra bright
            if (layer.adapter) {
                for (let blk of getAdapterBlocks(layer)) {
                    blk.highlight = Math.max(blk.highlight, t0.t * 0.5);
                }
            }
        }
        for (let layer of layout.midiTransformerLayers) {
            layer.attnMatrix.highlight = Math.max(layer.attnMatrix.highlight, t0.t * 0.25);
        }
    }

    breakAfter();

    // Step 2: Zoom to audio tower — highlight adapters
    let t1 = afterTime(null, 1.5, 0.5);
    let t0c = afterTime(t0, 0.4);
    cleanup(t0c, [t0]);

    moveBloqueaCameraTo(state, t1, new Vec3(-80, 0, -400), new Vec3(292, 20, 10));

    let adapterDown = c_blockRef('adapter down-projection', layout.audioTransformerLayers[0].adapter!.adapterDown);
    let dAdapter = c_dimRef('64D bottleneck', BloqueaDim.D_adapter);

    commentary()`The key innovation: layers 0-1 are completely frozen, but ${adapterDown} modules compress the 1024D output through a ${dAdapter} — a dramatic 16x compression that forces the adapter to learn a compact residual correction. Only ~131K parameters per adapter, trained at lr=5e-4 (the fastest rate).`;
    breakAfter();

    if (t1.active) {
        // Highlight adapter blocks in layers 0-1
        for (let i = 0; i < 2; i++) {
            let layer = layout.audioTransformerLayers[i];
            if (layer.adapter) {
                let pulse = 0.5 + 0.3 * Math.sin(wt.time * 3.5);
                for (let blk of getAdapterBlocks(layer)) {
                    blk.highlight = Math.max(blk.highlight, pulse);
                }
            }
        }
        // Dim MIDI tower
        for (let layer of layout.midiTransformerLayers) {
            for (let blk of getTransformerLayerBlocks(layer)) blk.opacity = 0.3;
        }
        layout.midiInput.opacity = 0.3;
        layout.midiPitchEmb.opacity = 0.3;
        layout.midiVelEmb.opacity = 0.3;
        layout.midiDurEmb.opacity = 0.3;
        layout.midiCombineLinear.opacity = 0.3;
        layout.midiPosEnc.opacity = 0.3;
    }

    breakAfter();

    // Step 3: Pan to show unfrozen layers 2-3
    let t2 = afterTime(null, 1.5, 0.5);
    let t1c = afterTime(t1, 0.4);
    cleanup(t1c, [t1]);

    let layer2 = layout.audioTransformerLayers[2];
    moveBloqueaCameraTo(state, t2, new Vec3(-80, 0, -layer2.ln1.y - 100), new Vec3(290, 18, 8));

    let unfrozenRef = c_dimRef('unfrozen', BloqueaDim.Unfrozen);

    commentary()`Layers 2-3 take the opposite approach: they are directly ${unfrozenRef} at lr=1e-5. No adapters needed — these upper layers can freely restructure their attention patterns and FFN weights for cross-modal alignment. This hybrid strategy gives Run C the best of both worlds: preservation via adapters below, flexibility via unfreezing above.`;
    breakAfter();

    if (t2.active) {
        for (let i = 2; i < 4; i++) {
            let layer = layout.audioTransformerLayers[i];
            let pulse = 0.3 + 0.2 * Math.sin(wt.time * 3);
            for (let blk of getTransformerLayerBlocks(layer)) {
                blk.highlight = Math.max(blk.highlight, pulse);
            }
        }
    }

    breakAfter();

    // Step 4: Zoom out to see shared space
    let t3 = afterTime(null, 2.0, 0.5);
    let t2c = afterTime(t2, 0.4);
    cleanup(t2c, [t2]);

    moveBloqueaCameraTo(state, t3, new Vec3(0, 0, -900), new Vec3(290, 25, 20));

    let audioEmb = c_blockRef('audio embedding', layout.audioEmbedding);
    let midiEmb = c_blockRef('MIDI embedding', layout.midiEmbedding);

    commentary()`Both towers converge into the shared ${d256} embedding space. The ${audioEmb} and ${midiEmb} are aligned using VICReg loss. Run C's 4 optimizer groups — adapter (5e-4), unfreeze (1e-5), MIDI (5e-5), projection (1e-4) — create a carefully tuned gradient of learning rates across the entire architecture.`;
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
