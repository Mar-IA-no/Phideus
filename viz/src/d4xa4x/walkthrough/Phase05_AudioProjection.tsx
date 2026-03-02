import { Vec3 } from "@/src/utils/vector";
import { ID4XA4XWalkthroughArgs, setD4XA4XInitialCamera } from "./D4XA4XWalkthrough";

export function d4xa4xPhase05_AudioProjection(args: ID4XA4XWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept } = tools;

    setD4XA4XInitialCamera(state, new Vec3(-100, 0, -900), new Vec3(290, 22, 22));

    dimExcept([
        layout.audioOutputLN,
        layout.audioMeanPool,
        layout.audioProjLayer1, layout.audioProjLayer2, layout.audioProjLayer3,
    ], 0.08);

    let lnRef = c_blockRef('Output LN', layout.audioOutputLN);
    let poolRef = c_blockRef('Mean Pool', layout.audioMeanPool);
    let proj1Ref = c_blockRef('1024->512', layout.audioProjLayer1);
    let proj3Ref = c_blockRef('512->256', layout.audioProjLayer3);

    commentary()`${lnRef} normalizes the transformer output [B, 2400, 1024]. Then ${poolRef} averages across all 2400 time positions into [B, 1024]. The 3-layer projection MLP (${proj1Ref} -> 512->512 -> ${proj3Ref}) maps into the 256D shared embedding space.`;
    breakAfter();

    let t0 = afterTime(null, 2.0, 0.5);
    if (t0.active) {
        layout.audioOutputLN.highlight = Math.max(layout.audioOutputLN.highlight, t0.t * 0.4);
        layout.audioMeanPool.highlight = Math.max(layout.audioMeanPool.highlight, t0.t * 0.6);
        layout.audioProjLayer1.highlight = Math.max(layout.audioProjLayer1.highlight, t0.t * 0.5);
        layout.audioProjLayer2.highlight = Math.max(layout.audioProjLayer2.highlight, t0.t * 0.4);
        layout.audioProjLayer3.highlight = Math.max(layout.audioProjLayer3.highlight, t0.t * 0.6);
    }
    breakAfter();

    commentary()`Note: mean pooling over 2400 tokens (vs 188 in reverse) pools 12.8x more temporal positions. This higher temporal resolution may capture finer-grained patterns, but the pooling bottleneck also discards more positional information. The projection head is identical across all arms.`;
    breakAfter();
}
