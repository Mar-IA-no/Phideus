import { Vec3 } from "@/src/utils/vector";
import { ID4XA4XWalkthroughArgs, setD4XA4XInitialCamera } from "./D4XA4XWalkthrough";

export function d4xa4xPhase07_MIDIDescriptor(args: ID4XA4XWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept, drawDimLabels } = tools;

    setD4XA4XInitialCamera(state, new Vec3(200, 0, -350), new Vec3(290, 18, 11));

    dimExcept([
        layout.midiInput,
        layout.midiDescPitchInput, layout.midiDescForwardDiff,
        layout.midiDescValidityMask, layout.midiDescSemitoneScale,
        layout.midiDescLogRatioScale, layout.midiDescOutput,
    ]);

    let pitchRef = c_blockRef('[B, N] Pitch', layout.midiDescPitchInput);
    let diffRef = c_blockRef('Forward Diff', layout.midiDescForwardDiff);
    let maskRef = c_blockRef('Validity Mask', layout.midiDescValidityMask);
    let semRef = c_blockRef('Semitone Scale', layout.midiDescSemitoneScale);
    let logRef = c_blockRef('Log Ratio Scale', layout.midiDescLogRatioScale);
    let outRef = c_blockRef('[B, N, 4]', layout.midiDescOutput);

    commentary()`The MIDI Descriptor D4 computes 4 local interval features per note from the ${pitchRef} sequence — all with torch.no_grad(). ${diffRef} computes pitch[i+1] - pitch[i], then ${maskRef} validates adjacent pairs (excluding padding boundaries).`;
    breakAfter();

    let t0 = afterTime(null, 2.0, 0.5);
    if (t0.active) {
        layout.midiDescPitchInput.highlight = Math.max(layout.midiDescPitchInput.highlight, t0.t * 0.5);
        layout.midiDescForwardDiff.highlight = Math.max(layout.midiDescForwardDiff.highlight, t0.t * 0.7);
        layout.midiDescValidityMask.highlight = Math.max(layout.midiDescValidityMask.highlight, t0.t * 0.5);
    }
    breakAfter();

    commentary()`Two scaling operations produce 4 features: ${semRef} divides by 24 for semitone_prev and semitone_next. ${logRef} divides by 12, clamps to [-2, 2], and divides by 2 for log_ratio features. The final ${outRef} stacks these into [B, N, 4] interval descriptors — these become K/V for the MIDI forward cross-attention.`;
    breakAfter();

    let t1 = afterTime(null, 2.0, 0.5);
    if (t1.active) {
        layout.midiDescSemitoneScale.highlight = Math.max(layout.midiDescSemitoneScale.highlight, t1.t * 0.6);
        layout.midiDescLogRatioScale.highlight = Math.max(layout.midiDescLogRatioScale.highlight, t1.t * 0.6);
        layout.midiDescOutput.highlight = Math.max(layout.midiDescOutput.highlight, t1.t * 0.8);
        if (t1.t > 0.5) drawDimLabels(layout.midiDescOutput, (t1.t - 0.5) * 2);
    }
    breakAfter();
}
