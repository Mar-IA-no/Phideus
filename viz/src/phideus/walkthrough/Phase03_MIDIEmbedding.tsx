import { Vec3 } from "@/src/utils/vector";
import { IPhideusWalkthroughArgs, setPhideusInitialCamera, movePhideusCameraTo } from "./PhideusWalkthrough";
import { PhideusDim } from "../PhideusDimStyle";

export function phideusPhase03_MIDIEmbedding(args: IPhideusWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept, drawDimLabels } = tools;

    // Dim everything except MIDI input section
    dimExcept([
        layout.midiInput,
        layout.midiPitchEmb, layout.midiVelEmb, layout.midiDurEmb,
        layout.midiCombineLinear, layout.midiPosEnc,
    ]);

    // Focus on MIDI input + embedding section
    setPhideusInitialCamera(state, new Vec3(100, 0, -20), new Vec3(288, 15, 5));

    let pitchRef = c_blockRef('pitch embedding', layout.midiPitchEmb);
    let velRef = c_blockRef('velocity embedding', layout.midiVelEmb);
    let durRef = c_blockRef('duration embedding', layout.midiDurEmb);
    let dMidi = c_dimRef('512D', PhideusDim.D_midi);

    commentary()`The MIDI Tower converts discrete musical events into continuous vector representations. Each MIDI event has three attributes — ${pitchRef}, ${velRef}, and ${durRef} — that are independently embedded and then combined through a linear projection into a unified ${dMidi} representation.`;
    breakAfter();

    // Step 1: MIDI input
    let tInput = afterTime(null, 1.0, 0.3);

    let inputRef = c_blockRef('MIDI input', layout.midiInput);
    let pitch = c_dimRef('pitch (0-127)', PhideusDim.Pitch);
    let vel = c_dimRef('velocity (0-127)', PhideusDim.Velocity);
    let dur = c_dimRef('duration (32 bins)', PhideusDim.Duration);

    commentary()`MIDI events enter the network as the ${inputRef} — discrete tokens. Each event is a triplet: ${pitch}, ${vel}, and ${dur}. Unlike audio, MIDI is a symbolic representation — there is no waveform to process.`;
    breakAfter();

    if (tInput.active) {
        layout.midiInput.highlight = Math.max(layout.midiInput.highlight, tInput.t * 0.5);
        layout.midiInput.opacity = 0.9;
    }

    // Step 2: Three embedding tables
    let t0 = afterTime(null, 2.0, 0.5);
    let tInputC = afterTime(tInput, 0.3);
    cleanup(tInputC, [tInput]);

    commentary()`Three independent embedding tables convert each attribute to a continuous vector: ${pitchRef} (128 classes to 256D), ${velRef} (128 classes to 128D), ${durRef} (32 classes to 128D). This factored design lets the model learn independent representations for each musical property.`;
    breakAfter();

    if (t0.active) {
        let sub = t0.t * 3;
        if (sub > 0) {
            layout.midiPitchEmb.highlight = Math.max(layout.midiPitchEmb.highlight, Math.min(1, sub) * 0.6);
            if (sub < 1.5) drawDimLabels(layout.midiPitchEmb, Math.min(1, sub) * 0.8);
        }
        if (sub > 1) {
            layout.midiVelEmb.highlight = Math.max(layout.midiVelEmb.highlight, Math.min(1, sub - 1) * 0.6);
            if (sub > 1 && sub < 2.5) drawDimLabels(layout.midiVelEmb, Math.min(1, sub - 1) * 0.8);
        }
        if (sub > 2) {
            layout.midiDurEmb.highlight = Math.max(layout.midiDurEmb.highlight, Math.min(1, sub - 2) * 0.6);
            drawDimLabels(layout.midiDurEmb, Math.min(1, sub - 2) * 0.8);
        }
    }

    breakAfter();

    // Step 3: Combine linear + LN
    let t1 = afterTime(null, 1.5, 0.5);
    let t0c = afterTime(t0, 0.3);
    cleanup(t0c, [t0]);

    movePhideusCameraTo(state, t1, new Vec3(100, 0, -120), new Vec3(288, 15, 4));

    let combineRef = c_blockRef('linear fusion layer', layout.midiCombineLinear);

    commentary()`The three embeddings are concatenated (256+128+128 = 512D) and passed through a ${combineRef} followed by LayerNorm. This fuses the three attribute representations into a single unified event embedding of ${dMidi}, ready for the transformer layers.`;
    breakAfter();

    if (t1.active) {
        layout.midiCombineLinear.highlight = Math.max(layout.midiCombineLinear.highlight, t1.t * 0.5);
        drawDimLabels(layout.midiCombineLinear, t1.t * 0.7);
    }

    breakAfter();

    // Step 4: Sinusoidal pos encoding
    let t2 = afterTime(null, 1.0, 0.5);
    let t1c = afterTime(t1, 0.3);
    cleanup(t1c, [t1]);

    let posRef = c_blockRef('positional encoding', layout.midiPosEnc);
    let frozen = c_dimRef('FROZEN', PhideusDim.Frozen);

    commentary()`Sinusoidal ${posRef} provides temporal position information. Unlike the audio tower's learned embeddings, these are parameter-free mathematical functions — sin/cos at different frequencies. This is ${frozen} (not trainable), providing a fixed positional reference frame.`;
    breakAfter();

    if (t2.active) {
        layout.midiPosEnc.opacity = Math.max(layout.midiPosEnc.opacity, 0.6);
        layout.midiPosEnc.highlight = Math.max(layout.midiPosEnc.highlight, t2.t * 0.5);
        drawDimLabels(layout.midiPosEnc, t2.t * 0.7);
    }

    breakAfter();
}
