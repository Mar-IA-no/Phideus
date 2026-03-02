import { Vec3 } from "@/src/utils/vector";
import { ID4A4WalkthroughArgs, setD4A4InitialCamera } from "./D4A4Walkthrough";
import { getTransformerLayerBlocks } from "../D4A4ModelLayout";

export function d4a4Phase02_AudioTransformer(args: ID4A4WalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept, drawDimLabels } = tools;

    setD4A4InitialCamera(state, new Vec3(-100, 0, -600), new Vec3(290, 18, 18));

    let allLayerBlocks = layout.audioTransformerLayers.flatMap(getTransformerLayerBlocks);
    dimExcept([layout.audioPosEmb, ...allLayerBlocks]);

    commentary()`The audio transformer processes 2400 tokens of 1024D features through 4 layers. Layers 0-1 use lr_low (3e-6) for cautious fine-tuning, layers 2-3 use lr_high (1e-5) for more aggressive learning. Each layer: LayerNorm → Q/K/V → Multi-Head Attention → Residual → LayerNorm → FFN → Residual.`;
    breakAfter();

    for (let i = 0; i < layout.audioTransformerLayers.length; i++) {
        let layer = layout.audioTransformerLayers[i];
        let lrLabel = i < 2 ? 'lr_low (3e-6)' : 'lr_high (1e-5)';
        let layerDesc = i === 0 ? 'learns local spectral patterns'
            : i === 1 ? 'refines frequency relationships'
            : i === 2 ? 'captures higher-level structure'
            : 'integrates global context';

        let lnRef = c_blockRef(`LN1`, layer.ln1);
        let qRef = c_blockRef(`Q`, layer.qWeight);
        let attnRef = c_blockRef(`Attention`, layer.attnMatrix);
        let ffnRef = c_blockRef(`FFN`, layer.mlpUp);

        commentary()`Layer ${i} (${lrLabel}): ${lnRef} normalizes, then ${qRef}/K/V project to 8 heads. The ${attnRef} matrix [2400x2400] captures token interactions. ${ffnRef} expands to 4096D (GELU) then back to 1024D. This layer ${layerDesc}.`;
        breakAfter();

        let t = afterTime(null, 2.0, 0.3);
        if (t.active) {
            for (let blk of getTransformerLayerBlocks(layer)) {
                blk.highlight = Math.max(blk.highlight, t.t * 0.5);
            }
            if (t.t > 0.3) drawDimLabels(layer.attnMatrix, (t.t - 0.3) * 1.4);
        }
        breakAfter();
    }

    commentary()`After 4 layers, the audio encoder outputs [B, 2400, 1024] — a rich representation of the audio signal. Unlike the reverse cross-attention arm (a4r), this sequence is NOT compressed: all 2400 tokens pass through to the concat stage.`;
    breakAfter();
}
