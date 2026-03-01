import { Vec3 } from "@/src/utils/vector";
import { ICrossAttWalkthroughArgs, setCrossAttInitialCamera } from "./CrossAttWalkthrough";

export function crossAttPhase07_MIDIRegularXAtt(args: ICrossAttWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept, drawDimLabels } = tools;

    setCrossAttInitialCamera(state, new Vec3(130, 0, -400), new Vec3(290, 18, 13));

    dimExcept([
        layout.midiPosEnc, layout.midiDescOutput,
        layout.midiRegularIntKVProj, layout.midiRegularQ, layout.midiRegularKV,
        layout.midiRegularMatrix, layout.midiRegularOutput,
    ], 0.08);

    let kvRef = c_blockRef('KV Proj', layout.midiRegularIntKVProj);
    let qRef = c_blockRef('Q: Embeddings', layout.midiRegularQ);
    let kvBlkRef = c_blockRef('K/V: Intervals', layout.midiRegularKV);
    let matRef = c_blockRef('Attn [NxN]', layout.midiRegularMatrix);

    commentary()`In MIDI regular cross-attention (dimmed for comparison), intervals are projected via ${kvRef} (Linear 4->512) to serve as Key/Value. The MIDI embeddings serve as Query. Embeddings ASK: "What interval information is relevant for this note?"`;
    breakAfter();

    let t0 = afterTime(null, 1.5, 0.5);
    if (t0.active) {
        layout.midiRegularIntKVProj.highlight = Math.max(layout.midiRegularIntKVProj.highlight, t0.t * 0.5);
        layout.midiRegularQ.highlight = Math.max(layout.midiRegularQ.highlight, t0.t * 0.4);
        layout.midiRegularKV.highlight = Math.max(layout.midiRegularKV.highlight, t0.t * 0.4);
    }
    breakAfter();

    commentary()`The ${matRef} is SQUARE [N x N] — unlike audio, both Q and K/V have the same length (N MIDI tokens). The attention shape alone doesn't distinguish regular from reverse for MIDI. The difference is purely in which path drives Q (the "asker") and which provides K/V (the "answerer").`;
    breakAfter();

    let t1 = afterTime(null, 1.5, 0.5);
    if (t1.active) {
        layout.midiRegularMatrix.highlight = Math.max(layout.midiRegularMatrix.highlight, t1.t * 0.6);
        layout.midiRegularOutput.highlight = Math.max(layout.midiRegularOutput.highlight, t1.t * 0.4);
    }
    breakAfter();
}
