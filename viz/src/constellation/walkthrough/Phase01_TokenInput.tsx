import { Vec3 } from "@/src/utils/vector";
import { IConstellationWalkthroughArgs, setConstellationInitialCamera } from "./ConstellationWalkthrough";

export function constellationPhase01_TokenInput(args: IConstellationWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept } = tools;

    setConstellationInitialCamera(state, new Vec3(-40, 0, -400), new Vec3(290, 20, 12));

    dimExcept([layout.audioInput, layout.audioMask, layout.vibInput, layout.vibMask]);

    let audioRef = c_blockRef('Audio Tokens', layout.audioInput);
    let vibRef = c_blockRef('Vib Tokens', layout.vibInput);
    let maskRef = c_blockRef('Mask', layout.audioMask);

    commentary()`Instead of dense histograms [T,256,3], ConstellationVAE uses sparse tokens [B,T,48,5]. Each of the 48 tokens per frame encodes five features: log_ratio, delta_t, weight, anchor_band, and target_band. The ${audioRef} and ${vibRef} paths receive identical formats. A binary ${maskRef} marks which of the 48 token slots contain real data vs padding.`;
    breakAfter();

    let t0 = afterTime(null, 1.5, 0.5);
    if (t0.active) {
        layout.audioInput.highlight = Math.max(layout.audioInput.highlight, t0.t * 0.7);
        layout.vibInput.highlight = Math.max(layout.vibInput.highlight, t0.t * 0.7);
        layout.audioMask.highlight = Math.max(layout.audioMask.highlight, t0.t * 0.4);
        layout.vibMask.highlight = Math.max(layout.vibMask.highlight, t0.t * 0.4);
    }
    breakAfter();

    commentary()`The sparse format captures pairwise frequency ratios between spectral peaks — the core Phideus representation. Average fill is ~11-27 tokens per frame (23-56% of capacity). The mask ensures attention pooling and loss functions only operate on valid tokens, preventing padding from distorting the learned representations.`;
    breakAfter();
}
