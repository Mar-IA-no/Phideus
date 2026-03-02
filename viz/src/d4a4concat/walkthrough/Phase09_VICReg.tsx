import { Vec3 } from "@/src/utils/vector";
import { ID4A4WalkthroughArgs, setD4A4InitialCamera } from "./D4A4Walkthrough";

export function d4a4Phase09_VICReg(args: ID4A4WalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept } = tools;

    setD4A4InitialCamera(state, new Vec3(0, 0, -1200), new Vec3(290, 22, 28));

    dimExcept([
        layout.audioMeanPool, layout.audioProjLayer1, layout.audioProjLayer2, layout.audioProjLayer3,
        layout.midiMeanPool, layout.midiProjLayer1, layout.midiProjLayer2, layout.midiProjLayer3,
        layout.audioEmbedding, layout.midiEmbedding,
        layout.vicregInv, layout.vicregVar, layout.vicregCov,
    ], 0.08);

    let aEmbRef = c_blockRef('Audio z', layout.audioEmbedding);
    let mEmbRef = c_blockRef('MIDI z', layout.midiEmbedding);
    let invRef = c_blockRef('Invariance', layout.vicregInv);
    let varRef = c_blockRef('Variance', layout.vicregVar);
    let covRef = c_blockRef('Covariance', layout.vicregCov);

    commentary()`Both towers converge into the shared 256D space. ${aEmbRef} comes from [waveform -> CNN -> Transformer -> Concat(A4) -> Projection]. ${mEmbRef} comes from [MIDI events -> Embedding -> Transformer -> Concat(D4) -> Projection]. VICReg loss aligns matched pairs without labels.`;
    breakAfter();

    let t0 = afterTime(null, 2.0, 0.5);
    if (t0.active) {
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, t0.t * 0.7);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, t0.t * 0.7);
    }
    breakAfter();

    commentary()`${invRef} (weight=10): MSE loss between matched audio-MIDI pairs — pulls them together. ${varRef} (weight=10): hinge loss ensuring std >= 1 per dimension — prevents collapse. ${covRef} (weight=1): decorrelation across dimensions — maximizes information content.`;
    breakAfter();

    let t1 = afterTime(null, 2.5, 0.5);
    if (t1.active) {
        let delay = [0.0, 0.3, 0.6];
        let blocks = [layout.vicregInv, layout.vicregVar, layout.vicregCov];
        for (let i = 0; i < 3; i++) {
            let localT = Math.max(0, (t1.t - delay[i]) / (1 - delay[i]));
            blocks[i].highlight = Math.max(blocks[i].highlight, localT * 0.8);
        }
    }
    breakAfter();

    commentary()`The complete d4a4 architecture: frozen MERT CNN + 4-layer audio transformer + A4 descriptor concat + 4-layer MIDI transformer + D4 descriptor concat + dual projection heads + VICReg. With cosine-tail scheduling (60 epochs), this achieves S=83.8% — the record for Gate 4.3. The simplest descriptor injection (concat) turns out to be the most effective.`;
    breakAfter();

    let t2 = afterTime(null, 2.0, 0.5);
    if (t2.active) {
        // Pulse all three loss components
        let pulse = Math.sin(t2.t * Math.PI * 2) * 0.3 + 0.5;
        layout.vicregInv.highlight = Math.max(layout.vicregInv.highlight, pulse);
        layout.vicregVar.highlight = Math.max(layout.vicregVar.highlight, pulse);
        layout.vicregCov.highlight = Math.max(layout.vicregCov.highlight, pulse);
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, 0.5);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, 0.5);
    }
    breakAfter();
}
