import { Vec3 } from "@/src/utils/vector";
import { ICrossAttWalkthroughArgs, setCrossAttInitialCamera } from "./CrossAttWalkthrough";

export function crossAttPhase03_MidiCrossAtt(args: ICrossAttWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept } = tools;

    setCrossAttInitialCamera(state, new Vec3(120, 0, -350), new Vec3(290, 18, 11));

    dimExcept([layout.midiInput, layout.midiEmb, layout.midiPosEnc, layout.crossAttMidi, layout.residualNormMidi,
               layout.midiIntervals, layout.intervalKvProj]);

    let embRef = c_blockRef('Embedding', layout.midiEmb);
    let posRef = c_blockRef('PosEncoding', layout.midiPosEnc);
    let intRef = c_blockRef('MIDI Intervals', layout.midiIntervals);
    let kvRef = c_blockRef('interval_kv_proj', layout.intervalKvProj);
    let xaRef = c_blockRef('Cross-Attention', layout.crossAttMidi);

    commentary()`MIDI events are encoded through pitch+velocity+duration ${embRef} to [B, N, 512], then ${posRef} is added. The ${intRef} descriptor computes 4 features per token: pitch difference, IOI (inter-onset interval), velocity delta, and duration ratio. These are computed from the raw MIDI event sequence (no learned parameters).`;
    breakAfter();

    let t0 = afterTime(null, 1.5, 0.5);
    if (t0.active) {
        layout.midiInput.highlight = Math.max(layout.midiInput.highlight, t0.t * 0.4);
        layout.midiEmb.highlight = Math.max(layout.midiEmb.highlight, t0.t * 0.5);
        layout.midiIntervals.highlight = Math.max(layout.midiIntervals.highlight, t0.t * 0.6);
    }
    breakAfter();

    commentary()`Key difference from audio: the MIDI ${xaRef} has Q and K/V at the SAME resolution (N tokens). The attention matrix is [B*8, N, N] — trivial in memory (N is typically 50-200). The ${kvRef} projects from 4D intervals to 512D. Cross-attention here provides dynamic selectivity and non-local access (vs concat which is strictly local per-token).`;
    breakAfter();

    let t1 = afterTime(null, 1.5, 0.5);
    if (t1.active) {
        layout.crossAttMidi.highlight = Math.max(layout.crossAttMidi.highlight, t1.t * 0.8);
        layout.intervalKvProj.highlight = Math.max(layout.intervalKvProj.highlight, t1.t * 0.5);
    }
    breakAfter();

    commentary()`The MIDI cross-attention adds ~1.05M parameters (vs ~4.2M for audio) because d=512 vs d=1024. If a CLS token is used for aggregation, it is prepended to Q before cross-attention but K/V stays at the original N resolution.`;
    breakAfter();
}
