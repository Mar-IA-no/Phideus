import { Vec3 } from "@/src/utils/vector";
import { ID4A4WalkthroughArgs, setD4A4InitialCamera } from "./D4A4Walkthrough";
import { getTransformerLayerBlocks } from "../D4A4ModelLayout";

export function d4a4Phase06_MIDITransformer(args: ID4A4WalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept, drawDimLabels } = tools;

    setD4A4InitialCamera(state, new Vec3(100, 0, -600), new Vec3(290, 18, 18));

    let allLayerBlocks = layout.midiTransformerLayers.flatMap(getTransformerLayerBlocks);
    dimExcept([layout.midiPosEnc, ...allLayerBlocks, layout.midiOutputLN]);

    commentary()`The MIDI encoder processes embedded events through 4 transformer layers, all trained from scratch at lr_midi (1e-4). Each layer follows pre-norm architecture: LayerNorm → Q/K/V → Multi-Head Attention (8 heads) → Residual → LayerNorm → FFN (512->2048->512) → Residual.`;
    breakAfter();

    for (let i = 0; i < layout.midiTransformerLayers.length; i++) {
        let layer = layout.midiTransformerLayers[i];
        let layerDesc = i === 0 ? 'learns local note relationships'
            : i === 1 ? 'discovers melodic patterns'
            : i === 2 ? 'captures harmonic context'
            : 'integrates long-range musical structure';

        let attnRef = c_blockRef(`Attention`, layer.attnMatrix);
        let ffnRef = c_blockRef(`FFN`, layer.mlpUp);

        commentary()`Layer ${i}: the ${attnRef} matrix [NxN] lets each MIDI event attend to all others. The ${ffnRef} block expands 512->2048 (GELU activation) then back to 512. This layer ${layerDesc}.`;
        breakAfter();

        let t = afterTime(null, 1.5, 0.3);
        if (t.active) {
            for (let blk of getTransformerLayerBlocks(layer)) {
                blk.highlight = Math.max(blk.highlight, t.t * 0.5);
            }
            if (t.t > 0.3) drawDimLabels(layer.attnMatrix, (t.t - 0.3) * 1.4);
        }
        breakAfter();
    }

    let lnRef = c_blockRef('Output LayerNorm', layout.midiOutputLN);
    commentary()`After 4 layers, ${lnRef} stabilizes the output [B, N, 512]. The MIDI encoder is smaller (512D vs 1024D for audio) but trained fully from scratch — no frozen pre-trained weights. This output proceeds to the concat stage where D4 descriptor features are appended.`;
    breakAfter();

    let t = afterTime(null, 1.5, 0.5);
    if (t.active) {
        layout.midiOutputLN.highlight = Math.max(layout.midiOutputLN.highlight, t.t * 0.6);
    }
    breakAfter();
}
