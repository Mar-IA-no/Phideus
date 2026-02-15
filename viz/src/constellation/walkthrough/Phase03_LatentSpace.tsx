import { Vec3 } from "@/src/utils/vector";
import { IConstellationWalkthroughArgs, setConstellationInitialCamera } from "./ConstellationWalkthrough";

export function constellationPhase03_LatentSpace(args: IConstellationWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept } = tools;

    setConstellationInitialCamera(state, new Vec3(0, 0, -500), new Vec3(290, 22, 13));

    dimExcept([layout.audioZShared, layout.audioZPrivate, layout.audioZCat,
               layout.vibZShared, layout.vibZPrivate, layout.vibZCat,
               layout.audioBiLstm, layout.vibBiLstm]);

    let sharedRef = c_blockRef('z_shared', layout.audioZShared);
    let privateRef = c_blockRef('z_private', layout.audioZPrivate);
    let catRef = c_blockRef('z_cat', layout.audioZCat);

    commentary()`The latent space is factored into two subspaces: ${sharedRef} (32D) captures cross-modal structure — features that should be similar between audio and vibration for the same physical source. ${privateRef} (16D) captures domain-specific information unique to each modality. The concatenated ${catRef} (48D) feeds the decoder.`;
    breakAfter();

    let t0 = afterTime(null, 1.5, 0.5);
    if (t0.active) {
        let p = t0.t;
        layout.audioZShared.highlight = Math.max(layout.audioZShared.highlight, p * 0.7);
        layout.vibZShared.highlight = Math.max(layout.vibZShared.highlight, p * 0.7);
    }
    breakAfter();

    let t1 = afterTime(null, 1.5, 0.5);
    if (t1.active) {
        let p = t1.t;
        layout.audioZPrivate.highlight = Math.max(layout.audioZPrivate.highlight, p * 0.6);
        layout.vibZPrivate.highlight = Math.max(layout.vibZPrivate.highlight, p * 0.6);
        layout.audioZCat.highlight = Math.max(layout.audioZCat.highlight, p * 0.4);
        layout.vibZCat.highlight = Math.max(layout.vibZCat.highlight, p * 0.4);
    }
    breakAfter();

    commentary()`Both subspaces use the reparameterization trick (mu + sigma * epsilon) for VAE training. The Diff loss enforces orthogonality between shared and private, while InfoNCE aligns the shared representations across domains. In practice, z_private often collapsed to near-zero variance — a known failure mode that the beta_kl_private hyperparameter attempts to prevent.`;
    breakAfter();
}
