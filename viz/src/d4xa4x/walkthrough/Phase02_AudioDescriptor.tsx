import { Vec3 } from "@/src/utils/vector";
import { ID4XA4XWalkthroughArgs, setD4XA4XInitialCamera } from "./D4XA4XWalkthrough";

export function d4xa4xPhase02_AudioDescriptor(args: ID4XA4XWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept, drawDimLabels } = tools;

    setD4XA4XInitialCamera(state, new Vec3(-200, 0, -350), new Vec3(290, 18, 11));

    dimExcept([
        layout.waveformInput,
        layout.audioDescStftWindow, layout.audioDescStftCompute,
        layout.audioDescMagnitude, layout.audioDescLogMag,
        layout.audioDescBandGroup, layout.audioDescTemporalDelta,
        layout.audioDescNormalize, layout.audioDescOutput,
    ]);

    let stftRef = c_blockRef('STFT', layout.audioDescStftCompute);
    let bandRef = c_blockRef('8 Log-Freq Bands', layout.audioDescBandGroup);
    let deltaRef = c_blockRef('Temporal Delta', layout.audioDescTemporalDelta);
    let outRef = c_blockRef('descriptor output', layout.audioDescOutput);

    commentary()`The Audio Descriptor A4 pipeline computes spectral features from raw audio — entirely with torch.no_grad() (no learned parameters). Raw audio -> Hann Window (n=2048) -> ${stftRef} (hop=512) produces complex [B, 1025, 188] at native STFT resolution. Then |magnitude| -> log(1+x) -> frequency binning.`;
    breakAfter();

    let t0 = afterTime(null, 2.0, 0.5);
    if (t0.active) {
        layout.audioDescStftWindow.highlight = Math.max(layout.audioDescStftWindow.highlight, t0.t * 0.5);
        layout.audioDescStftCompute.highlight = Math.max(layout.audioDescStftCompute.highlight, t0.t * 0.7);
        layout.audioDescMagnitude.highlight = Math.max(layout.audioDescMagnitude.highlight, t0.t * 0.5);
        layout.audioDescLogMag.highlight = Math.max(layout.audioDescLogMag.highlight, t0.t * 0.5);
        if (t0.t > 0.5) drawDimLabels(layout.audioDescStftCompute, (t0.t - 0.5) * 2);
    }
    breakAfter();

    commentary()`The ${bandRef} step compresses 1025 STFT bins -> 8 log-frequency bands (47-94, ..., 6000-12000 Hz). Then ${deltaRef} computes frame[t] - frame[t-1] (temporal difference), and z-score normalization standardizes per-band. The final ${outRef} transposes to [B, 188, 8] — this becomes K/V for forward cross-attention.`;
    breakAfter();

    let t1 = afterTime(null, 2.0, 0.5);
    if (t1.active) {
        layout.audioDescBandGroup.highlight = Math.max(layout.audioDescBandGroup.highlight, t1.t * 0.7);
        layout.audioDescTemporalDelta.highlight = Math.max(layout.audioDescTemporalDelta.highlight, t1.t * 0.6);
        layout.audioDescNormalize.highlight = Math.max(layout.audioDescNormalize.highlight, t1.t * 0.5);
        layout.audioDescOutput.highlight = Math.max(layout.audioDescOutput.highlight, t1.t * 0.8);
        if (t1.t > 0.5) drawDimLabels(layout.audioDescOutput, (t1.t - 0.5) * 2);
    }
    breakAfter();
}
