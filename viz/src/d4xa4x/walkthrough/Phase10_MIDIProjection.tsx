import { Vec3 } from "@/src/utils/vector";
import { ID4XA4XWalkthroughArgs, setD4XA4XInitialCamera } from "./D4XA4XWalkthrough";

export function d4xa4xPhase10_MIDIProjection(args: ID4XA4XWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept } = tools;

    setD4XA4XInitialCamera(state, new Vec3(100, 0, -900), new Vec3(290, 22, 22));

    dimExcept([
        layout.midiOutputLN,
        layout.midiMeanPool,
        layout.midiProjLayer1, layout.midiProjLayer2, layout.midiProjLayer3,
    ], 0.08);

    let poolRef = c_blockRef('Mean Pool', layout.midiMeanPool);
    let proj3Ref = c_blockRef('512->256', layout.midiProjLayer3);

    commentary()`${poolRef} reduces [B, N, 512] -> [B, 512] via mean pooling. The 3-layer MIDI projection MLP maps 512 -> 512 -> 512 -> 256D into the shared embedding space. The projection head architecture is identical across all descriptor arms — only the encoder and cross-attention mechanism differ.`;
    breakAfter();

    let t0 = afterTime(null, 2.0, 0.5);
    if (t0.active) {
        layout.midiOutputLN.highlight = Math.max(layout.midiOutputLN.highlight, t0.t * 0.3);
        layout.midiMeanPool.highlight = Math.max(layout.midiMeanPool.highlight, t0.t * 0.6);
        layout.midiProjLayer1.highlight = Math.max(layout.midiProjLayer1.highlight, t0.t * 0.5);
        layout.midiProjLayer2.highlight = Math.max(layout.midiProjLayer2.highlight, t0.t * 0.4);
        layout.midiProjLayer3.highlight = Math.max(layout.midiProjLayer3.highlight, t0.t * 0.6);
    }
    breakAfter();
}
