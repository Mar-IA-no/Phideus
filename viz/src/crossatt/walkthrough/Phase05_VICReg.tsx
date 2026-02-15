import { Vec3 } from "@/src/utils/vector";
import { ICrossAttWalkthroughArgs, setCrossAttInitialCamera } from "./CrossAttWalkthrough";

export function crossAttPhase05_VICReg(args: ICrossAttWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef } = tools;

    setCrossAttInitialCamera(state, new Vec3(0, 0, -700), new Vec3(290, 22, 16));

    let vicRef = c_blockRef('VICReg Loss', layout.vicregLoss);
    let aProjRef = c_blockRef('Audio emb', layout.audioProj);
    let mProjRef = c_blockRef('MIDI emb', layout.midiProj);

    commentary()`The ${vicRef} (Variance-Invariance-Covariance Regularization) brings matched ${aProjRef} and ${mProjRef} pairs together while preventing representation collapse. Three terms: Invariance (MSE between matched pairs, weight=10), Variance (hinge loss to maintain spread, weight=10), and Covariance (decorrelation of embedding dimensions, weight=1).`;
    breakAfter();

    let t0 = afterTime(null, 2.0, 0.5);
    if (t0.active) {
        layout.vicregLoss.highlight = Math.max(layout.vicregLoss.highlight, t0.t * 0.7);
        layout.audioProj.highlight = Math.max(layout.audioProj.highlight, t0.t * 0.4);
        layout.midiProj.highlight = Math.max(layout.midiProj.highlight, t0.t * 0.4);
    }
    breakAfter();

    commentary()`Cross-attention adds ~4.2M parameters on the audio side and ~1.05M on the MIDI side (vs ~1M for concat). The key trade-off: more parameters but dynamic, selective access to ratio information. Concat forces every frame to mix descriptor info at the same ratio; cross-attention lets harmonic frames attend strongly while silence/noise frames can ignore the descriptor entirely via the residual connection.`;
    breakAfter();

    let t1 = afterTime(null, 1.5, 0.5);
    if (t1.active) {
        let pulse = Math.sin(t1.t * Math.PI * 2) * 0.5 + 0.5;
        layout.crossAttAudio.highlight = Math.max(layout.crossAttAudio.highlight, pulse * 0.5);
        layout.crossAttMidi.highlight = Math.max(layout.crossAttMidi.highlight, pulse * 0.5);
        layout.audioDescriptor.highlight = Math.max(layout.audioDescriptor.highlight, pulse * 0.3);
        layout.midiIntervals.highlight = Math.max(layout.midiIntervals.highlight, pulse * 0.3);
    }
    breakAfter();
}
