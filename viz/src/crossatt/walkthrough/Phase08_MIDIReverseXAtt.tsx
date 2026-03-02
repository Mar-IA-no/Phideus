import { Vec3 } from "@/src/utils/vector";
import { ICrossAttWalkthroughArgs, setCrossAttInitialCamera } from "./CrossAttWalkthrough";

export function crossAttPhase08_MIDIReverseXAtt(args: ICrossAttWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept, drawDimLabels } = tools;

    setCrossAttInitialCamera(state, new Vec3(100, 0, -400), new Vec3(290, 18, 13));

    dimExcept([
        layout.midiPosEnc, layout.midiDescOutput,
        layout.midiReverseIntQProj, layout.midiReversePosEnc,
        layout.midiReverseQ, layout.midiReverseK,
        layout.midiReverseMatrix, layout.midiReverseResidual, layout.midiReverseOutput,
    ], 0.08);

    let qProjRef = c_blockRef('Q Proj', layout.midiReverseIntQProj);
    let posRef = c_blockRef('+PosEnc', layout.midiReversePosEnc);
    let qRef = c_blockRef('Q: Intervals', layout.midiReverseQ);
    let kRef = c_blockRef('K: Embeddings', layout.midiReverseK);
    let matRef = c_blockRef('Attn [NxN]', layout.midiReverseMatrix);

    commentary()`MIDI REVERSE cross-attention flips the Q/K/V roles: intervals are projected via ${qProjRef} (Linear 4->512) to serve as Query, and a shared ${posRef} is added. Embeddings serve as Key/Value. Intervals ORGANIZE embeddings — they DRIVE the representation, not just inform it.`;
    breakAfter();

    let t0 = afterTime(null, 2.0, 0.5);
    if (t0.active) {
        layout.midiReverseIntQProj.highlight = Math.max(layout.midiReverseIntQProj.highlight, t0.t * 0.6);
        layout.midiReversePosEnc.highlight = Math.max(layout.midiReversePosEnc.highlight, t0.t * 0.6);
        layout.midiReverseQ.highlight = Math.max(layout.midiReverseQ.highlight, t0.t * 0.6);
        layout.midiReverseK.highlight = Math.max(layout.midiReverseK.highlight, t0.t * 0.5);
    }
    breakAfter();

    commentary()`Unlike audio, MIDI reverse doesn't change sequence length (both N tokens — the ${matRef} is still SQUARE). The advantage is purely semantic: intervals drive the representation. The combination of audio reverse + MIDI reverse (d4a4r) achieves 83.8% — both domains benefit from descriptors as the organizing principle.`;
    breakAfter();

    let t1 = afterTime(null, 2.0, 0.5);
    if (t1.active) {
        layout.midiReverseMatrix.highlight = Math.max(layout.midiReverseMatrix.highlight, t1.t * 0.8);
        layout.midiReverseOutput.highlight = Math.max(layout.midiReverseOutput.highlight, t1.t * 0.6);
    }
    breakAfter();
}
