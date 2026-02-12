import { Vec3 } from "@/src/utils/vector";
import { IBloqueaWalkthroughArgs, setBloqueaInitialCamera, moveBloqueaCameraTo } from "./BloqueaWalkthrough";
import { getTransformerLayerBlocks, getAdapterBlocks } from "../BloqueaModelLayout";
import { BloqueaDim } from "../BloqueaDimStyle";

export function bloqueaPhase03_UnfrozenLayers(args: IBloqueaWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept, drawDimLabels } = tools;

    let layers23Blocks = [
        ...getTransformerLayerBlocks(layout.audioTransformerLayers[2]),
        ...getTransformerLayerBlocks(layout.audioTransformerLayers[3]),
    ];
    dimExcept(layers23Blocks);

    let layer2 = layout.audioTransformerLayers[2];
    setBloqueaInitialCamera(state, new Vec3(-100, 0, -layer2.ln1.y - 80), new Vec3(290, 15, 6));

    let unfrozen = c_dimRef('UNFROZEN', BloqueaDim.Unfrozen);
    let lrUnfreeze = c_dimRef('lr=1e-5', BloqueaDim.Unfrozen);

    commentary()`Layers 2-3 take the opposite approach to layers 0-1. Instead of freezing + adapter, they are directly ${unfrozen} at ${lrUnfreeze}. Every parameter — Q/K/V projections, attention patterns, FFN weights — is free to update. No adapters. No bottleneck. Direct gradient flow. This is where Run C diverges most from parameter-efficient fine-tuning.`;
    breakAfter();

    // Step 1: Layer 2 detail
    let t0 = afterTime(null, 2.0, 0.3);

    let attnRef = c_blockRef('Attention', layer2.attnMatrix);
    let ffnRef = c_blockRef('FFN', layer2.mlpUp);

    commentary()`Layer 2: The ${attnRef} matrix can freely restructure its query-key relationships. The ${ffnRef} can reshape its non-linear transformations. At lr=1e-5, the updates are gentle — 5x slower than the adapters — but they affect the _entire_ layer, not just a narrow bottleneck. This gives the model much more expressive capacity for learning cross-modal alignment patterns.`;
    breakAfter();

    if (t0.active) {
        let pulse = 0.25 + 0.2 * Math.sin(wt.time * 3);
        for (let blk of getTransformerLayerBlocks(layer2)) {
            blk.highlight = Math.max(blk.highlight, pulse);
        }
        layer2.attnMatrix.highlight = Math.max(layer2.attnMatrix.highlight, pulse + 0.2);
        drawDimLabels(layer2.attnMatrix, t0.t * 0.7);
        drawDimLabels(layer2.mlpUp, t0.t * 0.5);
    }

    breakAfter();

    // Step 2: Layer 3 detail
    let t1 = afterTime(null, 2.0, 0.3);
    let t0c = afterTime(t0, 0.3);
    cleanup(t0c, [t0]);

    let layer3 = layout.audioTransformerLayers[3];
    moveBloqueaCameraTo(state, t1, new Vec3(-100, 0, -layer3.ln1.y - 80), new Vec3(290, 15, 6));

    commentary()`Layer 3 — the final audio transformer layer — receives the same unfreezing treatment. As the layer closest to the projection head, it has the most direct influence on the final audio representation. The combination of frozen+adapter (layers 0-1) and unfrozen (layers 2-3) creates a gradient of adaptability from stable foundations to flexible upper layers.`;
    breakAfter();

    if (t1.active) {
        let pulse = 0.25 + 0.2 * Math.sin(wt.time * 3);
        for (let blk of getTransformerLayerBlocks(layer3)) {
            blk.highlight = Math.max(blk.highlight, pulse);
        }
        layer3.attnMatrix.highlight = Math.max(layer3.attnMatrix.highlight, pulse + 0.2);
        drawDimLabels(layer3.attnMatrix, t1.t * 0.7);
    }

    breakAfter();

    // Step 3: Compare with adapter layers — show both approaches
    let t2 = afterTime(null, 3.0, 0.5);
    let t1c = afterTime(t1, 0.4);
    cleanup(t1c, [t1]);

    // Wide view showing all 4 layers
    moveBloqueaCameraTo(state, t2, new Vec3(-100, 0, -layout.audioTransformerLayers[0].ln1.y - 250), new Vec3(290, 18, 10));

    let adapterRef = c_dimRef('adapters', BloqueaDim.Adapter);

    commentary()`The contrast is visible in the architecture: layers 0-1 (top) have the narrow ${adapterRef} pinched between frozen blocks, while layers 2-3 (bottom) are uniformly bright — every weight is trainable. Run C hypothesizes that lower layers need only small corrections (adapters), while upper layers need fundamental restructuring (unfreezing). This is computationally cheaper than unfreezing everything, while potentially more expressive than adapters-only.`;
    breakAfter();

    if (t2.active) {
        // Adapter layers: show adapters bright, frozen blocks dim
        for (let i = 0; i < 2; i++) {
            let layer = layout.audioTransformerLayers[i];
            for (let blk of getTransformerLayerBlocks(layer)) {
                if (!getAdapterBlocks(layer).includes(blk)) {
                    blk.opacity = Math.max(blk.opacity, 0.35);
                    blk.highlight = Math.max(blk.highlight, 0.05);
                } else {
                    blk.opacity = 1.0;
                    blk.highlight = Math.max(blk.highlight, 0.4 + 0.2 * Math.sin(wt.time * 3.5));
                }
            }
        }
        // Unfrozen layers: uniformly bright
        for (let i = 2; i < 4; i++) {
            let layer = layout.audioTransformerLayers[i];
            for (let blk of getTransformerLayerBlocks(layer)) {
                blk.opacity = 1.0;
                blk.highlight = Math.max(blk.highlight, 0.2 + 0.1 * Math.sin(wt.time * 2.5));
            }
        }
    }

    breakAfter();
}
