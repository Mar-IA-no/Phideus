import { Vec3 } from "@/src/utils/vector";
import { ID4XA4XWalkthroughArgs, setD4XA4XInitialCamera } from "./D4XA4XWalkthrough";
import { getTransformerLayerBlocks } from "../D4XA4XModelLayout";

export function d4xa4xPhase04_AudioTransformer(args: ID4XA4XWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept, drawDimLabels } = tools;

    let allTfBlocks = layout.audioTransformerLayers.flatMap(getTransformerLayerBlocks);

    setD4XA4XInitialCamera(state, new Vec3(-100, 0, -700), new Vec3(290, 20, 18));

    dimExcept([
        layout.audioForwardNorm,
        ...allTfBlocks,
        layout.audioOutputLN,
        ...layout.audioCnn,
    ], 0.08);
    for (let cnn of layout.audioCnn) cnn.opacity = 0.20;

    let outRef = c_blockRef('2400 tokens', layout.audioForwardNorm);

    commentary()`These 4 Transformer layers process the output of forward cross-attention: all ${outRef}! Unlike reverse cross-attention (which compresses to 188 tokens), forward preserves the full 2400-token sequence. The WIDE transformer blocks match the CNN width above — no compression happened.`;
    breakAfter();

    let t0 = afterTime(null, 2.0, 0.5);
    if (t0.active) {
        for (let i = 0; i < layout.audioTransformerLayers.length; i++) {
            let l = layout.audioTransformerLayers[i];
            l.attnMatrix.highlight = Math.max(l.attnMatrix.highlight, t0.t * (0.3 + i * 0.15));
        }
        if (t0.t > 0.3) drawDimLabels(layout.audioTransformerLayers[0].attnMatrix, (t0.t - 0.3) * 1.4);
    }
    breakAfter();

    commentary()`Each layer: LN1 -> Q/K/V -> Attention (2400x2400) -> +Residual -> LN2 -> FFN Up (1024->4096) -> GELU -> FFN Down (4096->1024) -> +Residual. Self-attention cost per layer: 2400^2 = 5.76M ops — this is the full MERT-scale attention. The descriptor influence is already baked in via the cross-attention residual.`;
    breakAfter();

    let t1 = afterTime(null, 2.0, 0.5);
    if (t1.active) {
        for (let l of layout.audioTransformerLayers) {
            l.mlpUp.highlight = Math.max(l.mlpUp.highlight, t1.t * 0.4);
            l.mlpDown.highlight = Math.max(l.mlpDown.highlight, t1.t * 0.4);
        }
    }
    breakAfter();
}
