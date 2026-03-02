import { Vec3 } from "@/src/utils/vector";
import { ID4RA4RWalkthroughArgs, setD4RA4RInitialCamera } from "./D4RA4RWalkthrough";

export function d4ra4rPhase06_MIDIReverseXAtt(args: ID4RA4RWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept, drawDimLabels } = tools;

    setD4RA4RInitialCamera(state, new Vec3(100, 0, -400), new Vec3(290, 18, 13));

    dimExcept([
        layout.midiPosEnc, layout.midiDescOutput,
        layout.midiReverseIntQProj, layout.midiReversePosEnc,
        layout.midiReverseQ, layout.midiReverseK,
        layout.midiReverseMatrix, layout.midiReverseResidual, layout.midiReverseOutput,
    ], 0.08);

    let qProjRef = c_blockRef('Q Proj', layout.midiReverseIntQProj);
    let posRef = c_blockRef('+PosEnc', layout.midiReversePosEnc);
    let qRef = c_blockRef('Q: D4 Intervals', layout.midiReverseQ);
    let kRef = c_blockRef('K: Embeddings', layout.midiReverseK);
    let matRef = c_blockRef('Attn [NxN]', layout.midiReverseMatrix);

    commentary()`MIDI REVERSE cross-attention: the D4 intervals are projected via ${qProjRef} (Linear 4->512) to serve as Query, and a shared ${posRef} is added. Embeddings serve as Key/Value. The D4 descriptor ORGANIZES the MIDI representation — interval patterns DRIVE how the model processes note sequences.`;
    breakAfter();

    let t0 = afterTime(null, 2.0, 0.5);
    if (t0.active) {
        layout.midiReverseIntQProj.highlight = Math.max(layout.midiReverseIntQProj.highlight, t0.t * 0.6);
        layout.midiReversePosEnc.highlight = Math.max(layout.midiReversePosEnc.highlight, t0.t * 0.6);
        layout.midiReverseQ.highlight = Math.max(layout.midiReverseQ.highlight, t0.t * 0.6);
        layout.midiReverseK.highlight = Math.max(layout.midiReverseK.highlight, t0.t * 0.5);
    }
    breakAfter();

    commentary()`Unlike audio reverse (where Q=188, K/V=2400 → compression), MIDI reverse keeps the same sequence length: ${matRef} is SQUARE [NxN]. The advantage is purely semantic: D4 interval patterns drive the attention weights. Combined with A4 spectral descriptors on the audio side, this is the MIXED DESCRIPTOR strategy — each modality uses its domain-specific descriptor. S=79.8%.`;
    breakAfter();

    let t1 = afterTime(null, 2.0, 0.5);
    if (t1.active) {
        layout.midiReverseMatrix.highlight = Math.max(layout.midiReverseMatrix.highlight, t1.t * 0.8);
        layout.midiReverseOutput.highlight = Math.max(layout.midiReverseOutput.highlight, t1.t * 0.6);
        if (t1.t > 0.3) drawDimLabels(layout.midiReverseMatrix, (t1.t - 0.3) * 1.4);
    }
    breakAfter();
}
