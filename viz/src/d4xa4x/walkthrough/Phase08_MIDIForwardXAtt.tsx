import { Vec3 } from "@/src/utils/vector";
import { ID4XA4XWalkthroughArgs, setD4XA4XInitialCamera } from "./D4XA4XWalkthrough";

export function d4xa4xPhase08_MIDIForwardXAtt(args: ID4XA4XWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept, drawDimLabels } = tools;

    setD4XA4XInitialCamera(state, new Vec3(100, 0, -400), new Vec3(290, 18, 13));

    dimExcept([
        layout.midiPosEnc, layout.midiDescOutput,
        layout.midiForwardQProj, layout.midiForwardKProj, layout.midiForwardVProj,
        layout.midiForwardQ, layout.midiForwardK,
        layout.midiForwardMatrix, layout.midiForwardResidual, layout.midiForwardNorm,
    ], 0.08);

    let qProjRef = c_blockRef('Q Proj', layout.midiForwardQProj);
    let kProjRef = c_blockRef('K Proj', layout.midiForwardKProj);
    let qRef = c_blockRef('Q: Embeddings', layout.midiForwardQ);
    let kRef = c_blockRef('K: Intervals', layout.midiForwardK);
    let matRef = c_blockRef('Attn [NxN]', layout.midiForwardMatrix);

    commentary()`MIDI FORWARD cross-attention: note embeddings are projected via ${qProjRef} (Linear 512->512) to serve as Query. The D4 descriptor [B, N, 4] is projected via ${kProjRef} (Linear 4->512) to serve as Key/Value. Each note embedding ASKS the descriptor: "What interval context is relevant to my position?"`;
    breakAfter();

    let t0 = afterTime(null, 2.0, 0.5);
    if (t0.active) {
        layout.midiForwardQProj.highlight = Math.max(layout.midiForwardQProj.highlight, t0.t * 0.6);
        layout.midiForwardKProj.highlight = Math.max(layout.midiForwardKProj.highlight, t0.t * 0.6);
        layout.midiForwardVProj.highlight = Math.max(layout.midiForwardVProj.highlight, t0.t * 0.6);
        layout.midiForwardQ.highlight = Math.max(layout.midiForwardQ.highlight, t0.t * 0.5);
        layout.midiForwardK.highlight = Math.max(layout.midiForwardK.highlight, t0.t * 0.5);
    }
    breakAfter();

    commentary()`Unlike audio, MIDI forward cross-attention produces a SQUARE ${matRef} — both Q (embeddings) and K (intervals) have N tokens. The descriptor doesn't change the sequence length in either forward or reverse on the MIDI side. The advantage is purely in HOW information flows: forward lets each note query for interval guidance.`;
    breakAfter();

    let t1 = afterTime(null, 2.0, 0.5);
    if (t1.active) {
        layout.midiForwardMatrix.highlight = Math.max(layout.midiForwardMatrix.highlight, t1.t * 0.8);
        if (t1.t > 0.3) drawDimLabels(layout.midiForwardMatrix, (t1.t - 0.3) * 1.4);
        layout.midiForwardResidual.highlight = Math.max(layout.midiForwardResidual.highlight, t1.t * 0.4);
        layout.midiForwardNorm.highlight = Math.max(layout.midiForwardNorm.highlight, t1.t * 0.3);
    }
    breakAfter();
}
