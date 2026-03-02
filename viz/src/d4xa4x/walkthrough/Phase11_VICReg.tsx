import { Vec3 } from "@/src/utils/vector";
import { ID4XA4XWalkthroughArgs, setD4XA4XInitialCamera } from "./D4XA4XWalkthrough";

export function d4xa4xPhase11_VICReg(args: ID4XA4XWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept } = tools;

    setD4XA4XInitialCamera(state, new Vec3(0, 0, -1200), new Vec3(290, 22, 28));

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

    commentary()`Both towers produce 256D embeddings in the shared space: ${aEmbRef} from audio (pooled from 2400 tokens) and ${mEmbRef} from MIDI (pooled from N tokens). VICReg brings matched pairs together while preventing collapse.`;
    breakAfter();

    let t0 = afterTime(null, 2.0, 0.5);
    if (t0.active) {
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, t0.t * 0.6);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, t0.t * 0.6);
    }
    breakAfter();

    commentary()`${invRef}: MSE between matched pairs (weight=10). ${varRef}: hinge loss to maintain spread (weight=10). ${covRef}: decorrelation across dimensions (weight=1). The complete forward cross-attention architecture preserves full temporal resolution at the cost of 163x more compute in the audio transformer — the key experimental question is whether this preservation helps or hurts retrieval quality.`;
    breakAfter();

    let t1 = afterTime(null, 2.0, 0.5);
    if (t1.active) {
        layout.vicregInv.highlight = Math.max(layout.vicregInv.highlight, t1.t * 0.7);
        layout.vicregVar.highlight = Math.max(layout.vicregVar.highlight, t1.t * 0.7);
        layout.vicregCov.highlight = Math.max(layout.vicregCov.highlight, t1.t * 0.7);
    }
    breakAfter();
}
