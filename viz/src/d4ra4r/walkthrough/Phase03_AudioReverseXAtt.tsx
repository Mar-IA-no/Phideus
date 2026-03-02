import { Vec3 } from "@/src/utils/vector";
import { ID4RA4RWalkthroughArgs, setD4RA4RInitialCamera } from "./D4RA4RWalkthrough";

export function d4ra4rPhase03_AudioReverseXAtt(args: ID4RA4RWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept, drawDimLabels } = tools;

    setD4RA4RInitialCamera(state, new Vec3(-100, 0, -400), new Vec3(290, 18, 13));

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

    commentary()`REVERSE cross-attention: the A4 descriptor [B, 188, 8] is projected via ${qProjRef} (Linear 8->1024) to serve as Query. A learned ${posEmbRef} is added. The 2400 CNN features serve as Key/Value. The descriptor ASKS the features: "What in these 2400 frames matches my spectral patterns?" — this is identical to the a4r arm.`;
    breakAfter();

    let t0 = afterTime(null, 2.0, 0.5);
    if (t0.active) {
        layout.audioReverseDescQProj.highlight = Math.max(layout.audioReverseDescQProj.highlight, t0.t * 0.6);
        layout.audioReverseDescPosEmb.highlight = Math.max(layout.audioReverseDescPosEmb.highlight, t0.t * 0.7);
        layout.audioReverseQ.highlight = Math.max(layout.audioReverseQ.highlight, t0.t * 0.6);
        layout.audioReverseK.highlight = Math.max(layout.audioReverseK.highlight, t0.t * 0.5);
    }
    breakAfter();

    commentary()`The ${matRef} is [8 heads, 188 queries, 2400 keys]. Each of 188 descriptor tokens selects from 2400 features. Output: only ${outRef} — 12.8x compression. The downstream transformer processes 188 tokens instead of 2400: attention cost 35K vs 5.76M (163x cheaper). The A4 spectral descriptor imposes structure on the representation.`;
    breakAfter();

    let t1 = afterTime(null, 2.5, 0.5);
    if (t1.active) {
        layout.audioReverseMatrix.highlight = Math.max(layout.audioReverseMatrix.highlight, t1.t * 0.8);
        if (t1.t > 0.3) drawDimLabels(layout.audioReverseMatrix, (t1.t - 0.3) * 1.4);
    }
    breakAfter();
}
