import { Vec3 } from "@/src/utils/vector";
import { ICrossAttWalkthroughArgs, setCrossAttInitialCamera } from "./CrossAttWalkthrough";

export function crossAttPhase04_AudioReverseXAtt(args: ICrossAttWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept, drawDimLabels } = tools;

    setCrossAttInitialCamera(state, new Vec3(-100, 0, -400), new Vec3(290, 18, 13));

    dimExcept([
        layout.audioPosEmb, layout.audioDescOutput,
        layout.audioReverseDescQProj, layout.audioReverseDescPosEmb,
        layout.audioReverseQ, layout.audioReverseK,
        layout.audioReverseMatrix, layout.audioReverseAttnOut,
        layout.audioReverseResidual, layout.audioReverseNorm, layout.audioReverseOutput,
    ], 0.08);

    let qProjRef = c_blockRef('Q Proj', layout.audioReverseDescQProj);
    let posEmbRef = c_blockRef('Desc PosEmb', layout.audioReverseDescPosEmb);
    let qRef = c_blockRef('Q: Desc (188)', layout.audioReverseQ);
    let kRef = c_blockRef('K: Features (2400)', layout.audioReverseK);
    let matRef = c_blockRef('Attn [188x2400]', layout.audioReverseMatrix);
    let outRef = c_blockRef('188 tokens', layout.audioReverseOutput);

    commentary()`REVERSE cross-attention flips Q and K/V roles. The descriptor [B, 188, 8] is projected via ${qProjRef} (Linear 8->1024) to serve as Query. A learned ${posEmbRef} is added (only exists in reverse!). The 2400 CNN features serve as Key/Value. Descriptors ASK: "What in these 2400 features matches my ratio patterns?"`;
    breakAfter();

    let t0 = afterTime(null, 2.0, 0.5);
    if (t0.active) {
        layout.audioReverseDescQProj.highlight = Math.max(layout.audioReverseDescQProj.highlight, t0.t * 0.6);
        layout.audioReverseDescPosEmb.highlight = Math.max(layout.audioReverseDescPosEmb.highlight, t0.t * 0.7);
        layout.audioReverseQ.highlight = Math.max(layout.audioReverseQ.highlight, t0.t * 0.6);
        layout.audioReverseK.highlight = Math.max(layout.audioReverseK.highlight, t0.t * 0.5);
    }
    breakAfter();

    commentary()`The ${matRef} is a WIDE rectangle: [8 heads, 188 queries, 2400 keys]. Each of 188 descriptor tokens selects from 2400 features. The output is only ${outRef} — 12.8x sequence compression! The descriptor acts as an information bottleneck, forcing the model to organize features around harmonic patterns.`;
    breakAfter();

    let t1 = afterTime(null, 2.5, 0.5);
    if (t1.active) {
        layout.audioReverseMatrix.highlight = Math.max(layout.audioReverseMatrix.highlight, t1.t * 0.8);
        if (t1.t > 0.3) drawDimLabels(layout.audioReverseMatrix, (t1.t - 0.3) * 1.4);
    }
    breakAfter();

    commentary()`The downstream Transformer now processes only 188 tokens instead of 2400. Self-attention cost: 188^2 = 35K vs 2400^2 = 5.76M — that's 163x cheaper! This compression actually HELPS: the descriptor imposes structure on the representation, forcing the model to organize features around harmonic patterns. Result: S=82.0% (+18.4pp over regular!)`;
    breakAfter();

    let t2 = afterTime(null, 2.0, 0.5);
    if (t2.active) {
        layout.audioReverseOutput.highlight = Math.max(layout.audioReverseOutput.highlight, t2.t * 0.8);
        layout.audioReverseAttnOut.highlight = Math.max(layout.audioReverseAttnOut.highlight, t2.t * 0.4);
        layout.audioReverseResidual.highlight = Math.max(layout.audioReverseResidual.highlight, t2.t * 0.3);
        layout.audioReverseNorm.highlight = Math.max(layout.audioReverseNorm.highlight, t2.t * 0.3);
    }
    breakAfter();
}
