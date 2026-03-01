import { Vec3 } from "@/src/utils/vector";
import { ICrossAttWalkthroughArgs, setCrossAttInitialCamera } from "./CrossAttWalkthrough";

export function crossAttPhase11_Projections(args: ICrossAttWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept } = tools;

    setCrossAttInitialCamera(state, new Vec3(0, 0, -1200), new Vec3(290, 22, 28));

    dimExcept([
        layout.audioMeanPool, layout.audioProjLayer1, layout.audioProjLayer2, layout.audioProjLayer3,
        layout.midiMeanPool, layout.midiProjLayer1, layout.midiProjLayer2, layout.midiProjLayer3,
        layout.audioEmbedding, layout.midiEmbedding,
        layout.vicregInv, layout.vicregVar, layout.vicregCov,
    ], 0.08);

    let aPoolRef = c_blockRef('Audio Mean Pool', layout.audioMeanPool);
    let mPoolRef = c_blockRef('MIDI Mean Pool', layout.midiMeanPool);
    let aEmbRef = c_blockRef('Audio z', layout.audioEmbedding);
    let mEmbRef = c_blockRef('MIDI z', layout.midiEmbedding);
    let invRef = c_blockRef('Invariance', layout.vicregInv);
    let varRef = c_blockRef('Variance', layout.vicregVar);
    let covRef = c_blockRef('Covariance', layout.vicregCov);

    commentary()`${aPoolRef} reduces [B, 188, 1024] -> [B, 1024] via mean pooling. The audio projection head maps 1024 -> 512 -> 512 -> 256D. Similarly, ${mPoolRef} reduces [B, N, 512] -> [B, 512], and the MIDI projection maps 512 -> 512 -> 512 -> 256D. Both produce embeddings in the shared 256D space.`;
    breakAfter();

    let t0 = afterTime(null, 2.0, 0.5);
    if (t0.active) {
        layout.audioMeanPool.highlight = Math.max(layout.audioMeanPool.highlight, t0.t * 0.5);
        layout.midiMeanPool.highlight = Math.max(layout.midiMeanPool.highlight, t0.t * 0.5);
        layout.audioProjLayer1.highlight = Math.max(layout.audioProjLayer1.highlight, t0.t * 0.4);
        layout.audioProjLayer3.highlight = Math.max(layout.audioProjLayer3.highlight, t0.t * 0.6);
        layout.midiProjLayer1.highlight = Math.max(layout.midiProjLayer1.highlight, t0.t * 0.4);
        layout.midiProjLayer3.highlight = Math.max(layout.midiProjLayer3.highlight, t0.t * 0.6);
    }
    breakAfter();

    commentary()`VICReg (Variance-Invariance-Covariance Regularization) brings matched ${aEmbRef} and ${mEmbRef} pairs together while preventing collapse. ${invRef}: MSE between matched pairs (weight=10). ${varRef}: hinge loss to maintain spread (weight=10). ${covRef}: decorrelation across dimensions (weight=1).`;
    breakAfter();

    let t1 = afterTime(null, 2.0, 0.5);
    if (t1.active) {
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, t1.t * 0.6);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, t1.t * 0.6);
        layout.vicregInv.highlight = Math.max(layout.vicregInv.highlight, t1.t * 0.7);
        layout.vicregVar.highlight = Math.max(layout.vicregVar.highlight, t1.t * 0.7);
        layout.vicregCov.highlight = Math.max(layout.vicregCov.highlight, t1.t * 0.7);
    }
    breakAfter();

    commentary()`The complete architecture: frozen MERT CNN + reverse cross-attention compression (2400->188 tokens) + 4-layer transformer + VICReg. The key insight: descriptors as Q (not K/V) forces the model to organize its representations around harmonic ratio patterns — this is what drives the 82.0% retrieval accuracy.`;
    breakAfter();
}
