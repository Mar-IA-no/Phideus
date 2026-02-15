import { Vec3 } from "@/src/utils/vector";
import { IConstellationWalkthroughArgs, setConstellationInitialCamera } from "./ConstellationWalkthrough";
import { ConstellationDim } from "../ConstellationDimStyle";

export function constellationPhase00_Overview(args: IConstellationWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, c_dimRef } = tools;

    setConstellationInitialCamera(state, new Vec3(0, 0, -600), new Vec3(290, 22, 14));

    let audioRef = c_blockRef('Audio Tokens', layout.audioInput);
    let vibRef = c_blockRef('Vibration Tokens', layout.vibInput);
    let infoNceRef = c_blockRef('InfoNCE loss', layout.infoNceLoss);

    commentary()`ConstellationVAE uses sparse ratio tokens [B,T,48,5] instead of dense histograms. Each token encodes log_ratio, delta_t, weight, anchor_band, target_band. The architecture has dual symmetric encoder paths (${audioRef} and ${vibRef}), a factored latent space (shared + private), and per-domain decoders. Five loss terms balance reconstruction, regularization, alignment, and disentanglement.`;
    breakAfter();

    let t0 = afterTime(null, 1.5, 0.5);

    if (t0.active) {
        layout.audioInput.highlight = Math.max(layout.audioInput.highlight, t0.t * 0.6);
        layout.vibInput.highlight = Math.max(layout.vibInput.highlight, t0.t * 0.6);
        layout.infoNceLoss.highlight = Math.max(layout.infoNceLoss.highlight, t0.t * 0.4);
    }

    breakAfter();

    let reconRef = c_blockRef('Recon loss', layout.reconLoss);
    let klSharedRef = c_blockRef('KL_shared', layout.klSharedLoss);
    let klPrivateRef = c_blockRef('KL_private', layout.klPrivateLoss);
    let diffRef = c_blockRef('Diff loss', layout.diffLoss);

    commentary()`The training objective combines five terms: ${reconRef} ensures faithful decoding, ${klSharedRef} and ${klPrivateRef} regularize the latent distributions, ${infoNceRef} aligns cross-modal shared representations, and ${diffRef} enforces orthogonality between shared and private subspaces.`;
    breakAfter();

    let t1 = afterTime(null, 1.5, 0.5);
    if (t1.active) {
        layout.reconLoss.highlight = Math.max(layout.reconLoss.highlight, t1.t * 0.5);
        layout.klSharedLoss.highlight = Math.max(layout.klSharedLoss.highlight, t1.t * 0.4);
        layout.klPrivateLoss.highlight = Math.max(layout.klPrivateLoss.highlight, t1.t * 0.4);
        layout.infoNceLoss.highlight = Math.max(layout.infoNceLoss.highlight, t1.t * 0.5);
        layout.diffLoss.highlight = Math.max(layout.diffLoss.highlight, t1.t * 0.4);
    }
    breakAfter();
}
