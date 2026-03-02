import { Vec3 } from "@/src/utils/vector";
import { ID4XA4XWalkthroughArgs, setD4XA4XInitialCamera } from "./D4XA4XWalkthrough";

export function d4xa4xPhase06_MIDIEmbedding(args: ID4XA4XWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept } = tools;

    setD4XA4XInitialCamera(state, new Vec3(100, 0, -400), new Vec3(290, 20, 14));

    dimExcept([
        layout.midiInput, layout.midiPitchEmb, layout.midiVelEmb, layout.midiDurEmb,
        layout.midiCombineLinear, layout.midiPosEnc,
    ]);

    let inputRef = c_blockRef('MIDI Events', layout.midiInput);
    let pitchRef = c_blockRef('Pitch Emb', layout.midiPitchEmb);
    let velRef = c_blockRef('Velocity Emb', layout.midiVelEmb);
    let durRef = c_blockRef('Duration Emb', layout.midiDurEmb);
    let combRef = c_blockRef('Combine Linear', layout.midiCombineLinear);

    commentary()`${inputRef} (pitch, velocity, duration per note) are embedded through three separate tables: ${pitchRef} (128->512), ${velRef} (128->512), and ${durRef} (32->512). These fan out in 3D (staggered Z) to show the parallel lookup, then converge into ${combRef}: concatenated [1536] -> Linear -> LayerNorm -> [512].`;
    breakAfter();

    let t0 = afterTime(null, 1.5, 0.5);
    if (t0.active) {
        layout.midiInput.highlight = Math.max(layout.midiInput.highlight, t0.t * 0.4);
        layout.midiPitchEmb.highlight = Math.max(layout.midiPitchEmb.highlight, t0.t * 0.6);
        layout.midiVelEmb.highlight = Math.max(layout.midiVelEmb.highlight, t0.t * 0.5);
        layout.midiDurEmb.highlight = Math.max(layout.midiDurEmb.highlight, t0.t * 0.5);
    }
    breakAfter();

    let posRef = c_blockRef('Sinusoidal PosEnc', layout.midiPosEnc);

    commentary()`${posRef} (frozen, not learned) provides temporal ordering. In forward cross-attention, these MIDI embeddings with position info become Q — each note token actively queries the D4 descriptor for relevant interval context.`;
    breakAfter();

    let t1 = afterTime(null, 1.5, 0.5);
    if (t1.active) {
        layout.midiCombineLinear.highlight = Math.max(layout.midiCombineLinear.highlight, t1.t * 0.6);
        layout.midiPosEnc.highlight = Math.max(layout.midiPosEnc.highlight, t1.t * 0.7);
    }
    breakAfter();
}
