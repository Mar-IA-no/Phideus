import { Vec3 } from "@/src/utils/vector";
import { ID4XA4XWalkthroughArgs, setD4XA4XInitialCamera } from "./D4XA4XWalkthrough";
import { getTransformerLayerBlocks } from "../D4XA4XModelLayout";

export function d4xa4xPhase09_MIDITransformer(args: ID4XA4XWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept } = tools;

    let allTfBlocks = layout.midiTransformerLayers.flatMap(getTransformerLayerBlocks);

    setD4XA4XInitialCamera(state, new Vec3(100, 0, -700), new Vec3(290, 20, 18));

    dimExcept([
        layout.midiForwardNorm,
        ...allTfBlocks,
        layout.midiOutputLN,
    ], 0.08);

    commentary()`The MIDI Transformer (4 layers, 8 heads, d=512) processes N tokens from the forward cross-attention output. Like reverse, the MIDI sequence length is unchanged by cross-attention — both directions produce N tokens on the MIDI side. Each layer: LN -> Q/K/V -> Attention (NxN) -> +Residual -> LN -> FFN (512->2048->512) -> +Residual.`;
    breakAfter();

    let t0 = afterTime(null, 2.0, 0.5);
    if (t0.active) {
        for (let i = 0; i < layout.midiTransformerLayers.length; i++) {
            let l = layout.midiTransformerLayers[i];
            l.attnMatrix.highlight = Math.max(l.attnMatrix.highlight, t0.t * (0.3 + i * 0.15));
            l.mlpUp.highlight = Math.max(l.mlpUp.highlight, t0.t * 0.3);
        }
    }
    breakAfter();

    commentary()`The output goes through an Output LayerNorm before pooling. The MIDI transformer has ~13M parameters (vs ~60M for the audio MERT CNN + transformer), reflecting the lighter-weight MIDI representation.`;
    breakAfter();

    let t1 = afterTime(null, 1.5, 0.5);
    if (t1.active) {
        layout.midiOutputLN.highlight = Math.max(layout.midiOutputLN.highlight, t1.t * 0.6);
    }
    breakAfter();
}
