import { Vec3 } from "@/src/utils/vector";
import { ICrossAttWalkthroughArgs, setCrossAttInitialCamera } from "./CrossAttWalkthrough";

export function crossAttPhase03_AudioRegularXAtt(args: ICrossAttWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept, drawDimLabels } = tools;

    setCrossAttInitialCamera(state, new Vec3(-130, 0, -400), new Vec3(290, 18, 13));

    dimExcept([
        layout.audioPosEmb, layout.audioDescOutput,
        layout.audioRegularDescKVProj, layout.audioRegularQ, layout.audioRegularK,
        layout.audioRegularMatrix, layout.audioRegularOutput,
    ], 0.08);

    let kvRef = c_blockRef('KV Proj', layout.audioRegularDescKVProj);
    let qRef = c_blockRef('Q: Features (2400)', layout.audioRegularQ);
    let kRef = c_blockRef('K: Desc (188)', layout.audioRegularK);
    let matRef = c_blockRef('Attn [2400x188]', layout.audioRegularMatrix);
    let outRef = c_blockRef('2400 tokens', layout.audioRegularOutput);

    commentary()`In REGULAR cross-attention (shown dimmed for comparison), the descriptor [B, 188, 8] is projected via ${kvRef} (Linear 8->1024) to serve as Key and Value. The 2400 CNN features serve as Query. Features ASK: "What ratio information is relevant here?"`;
    breakAfter();

    let t0 = afterTime(null, 2.0, 0.5);
    if (t0.active) {
        layout.audioRegularDescKVProj.highlight = Math.max(layout.audioRegularDescKVProj.highlight, t0.t * 0.5);
        layout.audioRegularQ.highlight = Math.max(layout.audioRegularQ.highlight, t0.t * 0.5);
        layout.audioRegularK.highlight = Math.max(layout.audioRegularK.highlight, t0.t * 0.5);
    }
    breakAfter();

    commentary()`The ${matRef} is a TALL rectangle: [8 heads, 2400 queries, 188 keys]. Each of 2400 CNN frames independently decides which of the 188 descriptor frames to attend to. The output stays at ${outRef} — the full 2400-length sequence goes to the Transformer. Result: S=63.6%.`;
    breakAfter();

    let t1 = afterTime(null, 2.0, 0.5);
    if (t1.active) {
        layout.audioRegularMatrix.highlight = Math.max(layout.audioRegularMatrix.highlight, t1.t * 0.6);
        layout.audioRegularOutput.highlight = Math.max(layout.audioRegularOutput.highlight, t1.t * 0.4);
        if (t1.t > 0.3) drawDimLabels(layout.audioRegularMatrix, (t1.t - 0.3) * 1.4);
    }
    breakAfter();
}
