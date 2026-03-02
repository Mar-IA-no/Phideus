import { Vec3 } from "@/src/utils/vector";
import { ID4RA4RWalkthroughArgs, setD4RA4RInitialCamera } from "./D4RA4RWalkthrough";
import { getTransformerLayerBlocks } from "../D4RA4RModelLayout";

export function d4ra4rPhase08_MIDITransformer(args: ID4RA4RWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept } = tools;

    let allTfBlocks = layout.midiTransformerLayers.flatMap(getTransformerLayerBlocks);

    setD4RA4RInitialCamera(state, new Vec3(100, 0, -700), new Vec3(290, 20, 18));

    dimExcept([
        layout.midiReverseOutput,
        ...allTfBlocks,
        layout.midiOutputLN,
    ], 0.08);

    let tfRef = c_blockRef('MIDI Transformer', layout.midiTransformerLayers[0].attnMatrix);

    commentary()`The MIDI Transformer (4 layers, 8 heads, d=512) processes N tokens from the D4 reverse cross-attention output. Unlike audio, the sequence length is unchanged — reverse cross-attention doesn't compress MIDI (both D4 and embeddings have N tokens). Each layer: LN -> Q/K/V -> Attention (NxN) -> +Residual -> LN -> FFN (512->2048->512) -> +Residual.`;
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

    commentary()`The output goes through an Output LayerNorm before pooling. The MIDI transformer has ~13M parameters (vs ~60M for audio MERT CNN + transformer). D4 interval descriptors shape the attention patterns throughout — the transformer processes interval-organized representations rather than raw embeddings.`;
    breakAfter();

    let t1 = afterTime(null, 1.5, 0.5);
    if (t1.active) {
        layout.midiOutputLN.highlight = Math.max(layout.midiOutputLN.highlight, t1.t * 0.6);
    }
    breakAfter();
}
