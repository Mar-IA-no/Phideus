import { Vec3 } from "@/src/utils/vector";
import { ID4A4WalkthroughArgs, setD4A4InitialCamera } from "./D4A4Walkthrough";

export function d4a4Phase05_MIDIEmbedding(args: ID4A4WalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept, drawDimLabels } = tools;

    setD4A4InitialCamera(state, new Vec3(100, 0, -200), new Vec3(290, 18, 10));

    dimExcept([
        layout.midiInput,
        layout.midiPitchEmb, layout.midiVelEmb, layout.midiDurEmb,
        layout.midiCombineLinear, layout.midiPosEnc,
    ]);

    let inputRef = c_blockRef('MIDI Events', layout.midiInput);
    let pitchRef = c_blockRef('Pitch (128)', layout.midiPitchEmb);
    let velRef = c_blockRef('Velocity (128)', layout.midiVelEmb);
    let durRef = c_blockRef('Duration (32 bins)', layout.midiDurEmb);
    let combRef = c_blockRef('Linear(1536->512)', layout.midiCombineLinear);
    let posRef = c_blockRef('Sinusoidal PosEnc', layout.midiPosEnc);

    commentary()`The MIDI tower converts discrete events into continuous vectors. Each ${inputRef} has three properties: pitch (0-127), velocity (0-127), and duration (quantized into 32 bins). Three independent embedding tables — ${pitchRef}, ${velRef}, ${durRef} — each map to 512D vectors.`;
    breakAfter();

    let t0 = afterTime(null, 2.0, 0.5);
    if (t0.active) {
        layout.midiInput.highlight = Math.max(layout.midiInput.highlight, t0.t * 0.4);
        layout.midiPitchEmb.highlight = Math.max(layout.midiPitchEmb.highlight, t0.t * 0.6);
        layout.midiVelEmb.highlight = Math.max(layout.midiVelEmb.highlight, t0.t * 0.5);
        layout.midiDurEmb.highlight = Math.max(layout.midiDurEmb.highlight, t0.t * 0.5);
        if (t0.t > 0.5) drawDimLabels(layout.midiPitchEmb, (t0.t - 0.5) * 2);
    }
    breakAfter();

    commentary()`The factored design lets each property learn independently. The three 512D embeddings are concatenated (1536D total) then fused via ${combRef} + LayerNorm into a unified 512D representation. ${posRef} (sinusoidal, frozen) adds temporal ordering. All embedding weights train at lr_midi (1e-4).`;
    breakAfter();

    let t1 = afterTime(null, 2.0, 0.5);
    if (t1.active) {
        layout.midiCombineLinear.highlight = Math.max(layout.midiCombineLinear.highlight, t1.t * 0.6);
        layout.midiPosEnc.highlight = Math.max(layout.midiPosEnc.highlight, t1.t * 0.4);
        if (t1.t > 0.4) drawDimLabels(layout.midiCombineLinear, (t1.t - 0.4) * 1.6);
    }
    breakAfter();
}
