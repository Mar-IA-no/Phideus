import { Vec3 } from "@/src/utils/vector";
import { ID4RA4RWalkthroughArgs, setD4RA4RInitialCamera } from "./D4RA4RWalkthrough";

export function d4ra4rPhase04_MIDIEmbedding(args: ID4RA4RWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept } = tools;

    setD4RA4RInitialCamera(state, new Vec3(100, 0, -300), new Vec3(290, 20, 16));

    dimExcept([
        layout.midiInput, layout.midiPitchEmb, layout.midiVelEmb, layout.midiDurEmb,
        layout.midiCombineLinear, layout.midiPosEnc,
    ], 0.08);

    let pitchRef = c_blockRef('Pitch Emb', layout.midiPitchEmb);
    let velRef = c_blockRef('Velocity Emb', layout.midiVelEmb);
    let durRef = c_blockRef('Duration Emb', layout.midiDurEmb);
    let combRef = c_blockRef('Concat + Linear', layout.midiCombineLinear);

    commentary()`Each MIDI event has three attributes: pitch (128 values), velocity (128), and duration (32 buckets). ${pitchRef} (128->512), ${velRef} (128->512), and ${durRef} (32->512) embed each attribute into 512D space. ${combRef} concatenates all three (1536D) and projects back to 512D.`;
    breakAfter();

    let t0 = afterTime(null, 2.0, 0.5);
    if (t0.active) {
        layout.midiPitchEmb.highlight = Math.max(layout.midiPitchEmb.highlight, t0.t * 0.5);
        layout.midiVelEmb.highlight = Math.max(layout.midiVelEmb.highlight, t0.t * 0.4);
        layout.midiDurEmb.highlight = Math.max(layout.midiDurEmb.highlight, t0.t * 0.4);
        layout.midiCombineLinear.highlight = Math.max(layout.midiCombineLinear.highlight, t0.t * 0.6);
    }
    breakAfter();

    commentary()`Sinusoidal positional encoding is added (frozen, not learned). The result is [B, N, 512] — N MIDI events at 512 dimensions. This embedding stage is identical across all arms and is fully trainable.`;
    breakAfter();

    let t1 = afterTime(null, 1.5, 0.5);
    if (t1.active) {
        layout.midiPosEnc.highlight = Math.max(layout.midiPosEnc.highlight, t1.t * 0.4);
    }
    breakAfter();
}
