import { Vec3 } from "@/src/utils/vector";
import { ID4XA4XWalkthroughArgs, setD4XA4XInitialCamera } from "./D4XA4XWalkthrough";

export function d4xa4xPhase03_AudioForwardXAtt(args: ID4XA4XWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept, drawDimLabels } = tools;

    setD4XA4XInitialCamera(state, new Vec3(-100, 0, -400), new Vec3(290, 18, 13));

    dimExcept([
        layout.audioPosEmb, layout.audioDescOutput,
        layout.audioForwardQProj, layout.audioForwardKProj, layout.audioForwardVProj,
        layout.audioForwardQ, layout.audioForwardK,
        layout.audioForwardMatrix, layout.audioForwardAttnOut,
        layout.audioForwardResidual, layout.audioForwardNorm,
    ], 0.08);

    let qProjRef = c_blockRef('Q Proj', layout.audioForwardQProj);
    let kProjRef = c_blockRef('K Proj', layout.audioForwardKProj);
    let vProjRef = c_blockRef('V Proj', layout.audioForwardVProj);
    let qRef = c_blockRef('Q: Encoder (2400)', layout.audioForwardQ);
    let kRef = c_blockRef('K: Desc (188)', layout.audioForwardK);
    let matRef = c_blockRef('Attn [2400x188]', layout.audioForwardMatrix);

    commentary()`FORWARD cross-attention: the encoder tokens are Q, the descriptor is K/V — the OPPOSITE of reverse. The ${qProjRef} projects each of 2400 CNN features (Linear 1024->1024) to form Query. The descriptor [B, 188, 8] is projected via ${kProjRef} and ${vProjRef} (Linear 8->1024) to form Key and Value. Each encoder token ASKS: "Which descriptor patterns are relevant to me?"`;
    breakAfter();

    let t0 = afterTime(null, 2.0, 0.5);
    if (t0.active) {
        layout.audioForwardQProj.highlight = Math.max(layout.audioForwardQProj.highlight, t0.t * 0.6);
        layout.audioForwardKProj.highlight = Math.max(layout.audioForwardKProj.highlight, t0.t * 0.6);
        layout.audioForwardVProj.highlight = Math.max(layout.audioForwardVProj.highlight, t0.t * 0.6);
        layout.audioForwardQ.highlight = Math.max(layout.audioForwardQ.highlight, t0.t * 0.5);
        layout.audioForwardK.highlight = Math.max(layout.audioForwardK.highlight, t0.t * 0.5);
    }
    breakAfter();

    commentary()`The ${matRef} is a TALL rectangle: [8 heads, 2400 queries, 188 keys]. Each of 2400 encoder tokens attends to 188 descriptor positions. Compare with reverse where it's [188, 2400] — visually transposed! The output preserves all 2400 tokens — NO sequence compression. The descriptor modulates the representation without reducing temporal resolution.`;
    breakAfter();

    let t1 = afterTime(null, 2.5, 0.5);
    if (t1.active) {
        layout.audioForwardMatrix.highlight = Math.max(layout.audioForwardMatrix.highlight, t1.t * 0.8);
        if (t1.t > 0.3) drawDimLabels(layout.audioForwardMatrix, (t1.t - 0.3) * 1.4);
    }
    breakAfter();

    commentary()`The downstream Transformer processes all 2400 tokens — self-attention cost: 2400^2 = 5.76M per layer (vs 188^2 = 35K in reverse). Forward cross-attention is 163x MORE expensive than reverse. The tradeoff: full temporal resolution is preserved, letting the model retain fine-grained temporal information at the cost of compute.`;
    breakAfter();

    let t2 = afterTime(null, 2.0, 0.5);
    if (t2.active) {
        layout.audioForwardAttnOut.highlight = Math.max(layout.audioForwardAttnOut.highlight, t2.t * 0.4);
        layout.audioForwardResidual.highlight = Math.max(layout.audioForwardResidual.highlight, t2.t * 0.3);
        layout.audioForwardNorm.highlight = Math.max(layout.audioForwardNorm.highlight, t2.t * 0.3);
    }
    breakAfter();
}
