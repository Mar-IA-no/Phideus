import { Vec3 } from "@/src/utils/vector";
import { ID4RA4RWalkthroughArgs, setD4RA4RInitialCamera } from "./D4RA4RWalkthrough";

export function d4ra4rPhase02_AudioDescriptor(args: ID4RA4RWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept, drawDimLabels } = tools;

    setD4RA4RInitialCamera(state, new Vec3(-200, 0, -400), new Vec3(290, 18, 15));

    dimExcept([
        layout.waveformInput,
        layout.audioDescStftWindow, layout.audioDescStftCompute, layout.audioDescMagnitude,
        layout.audioDescLogMag, layout.audioDescBandGroup, layout.audioDescTemporalDelta,
        layout.audioDescNormalize, layout.audioDescOutput,
    ], 0.08);

    let stftRef = c_blockRef('STFT', layout.audioDescStftCompute);
    let bandRef = c_blockRef('8 Log-Freq Bands', layout.audioDescBandGroup);
    let deltaRef = c_blockRef('Temporal Delta', layout.audioDescTemporalDelta);
    let outRef = c_blockRef('[B, 188, 8]', layout.audioDescOutput);

    commentary()`The A4 descriptor is a pure DSP pipeline — NO learnable parameters. It computes spectral features directly from the raw waveform. The Hann window (n=2048) feeds ${stftRef} with hop=512, producing [B, 1025, 188] complex coefficients. After magnitude and log transform, frequencies are grouped into ${bandRef} (47-12000 Hz, octave spacing).`;
    breakAfter();

    let t0 = afterTime(null, 2.0, 0.5);
    if (t0.active) {
        layout.audioDescStftCompute.highlight = Math.max(layout.audioDescStftCompute.highlight, t0.t * 0.6);
        layout.audioDescBandGroup.highlight = Math.max(layout.audioDescBandGroup.highlight, t0.t * 0.5);
    }
    breakAfter();

    commentary()`${deltaRef} computes frame-to-frame differences — capturing spectral change over time. Z-Score normalization stabilizes the distribution. The final ${outRef} output (transposed to [B, 188, 8]) provides 188 time frames, each with 8 spectral features. This is a SPECTRAL descriptor — it captures "what frequencies are changing" in the audio.`;
    breakAfter();

    let t1 = afterTime(null, 2.0, 0.5);
    if (t1.active) {
        layout.audioDescTemporalDelta.highlight = Math.max(layout.audioDescTemporalDelta.highlight, t1.t * 0.5);
        layout.audioDescOutput.highlight = Math.max(layout.audioDescOutput.highlight, t1.t * 0.7);
        if (t1.t > 0.3) drawDimLabels(layout.audioDescOutput, (t1.t - 0.3) * 1.4);
    }
    breakAfter();
}
