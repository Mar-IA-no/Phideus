import { Vec3 } from "@/src/utils/vector";
import { IConstellationWalkthroughArgs, setConstellationInitialCamera } from "./ConstellationWalkthrough";

export function constellationPhase04_DecodersAndLosses(args: IConstellationWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef } = tools;

    setConstellationInitialCamera(state, new Vec3(0, 0, -700), new Vec3(290, 22, 16));

    let decRef = c_blockRef('Decoder LSTM', layout.audioDecLstm);
    let reconRef = c_blockRef('Recon', layout.reconLoss);
    let klSRef = c_blockRef('KL_shared', layout.klSharedLoss);
    let klPRef = c_blockRef('KL_private', layout.klPrivateLoss);
    let infoRef = c_blockRef('InfoNCE', layout.infoNceLoss);
    let diffRef = c_blockRef('Diff', layout.diffLoss);

    commentary()`Each domain has a ${decRef} that reconstructs the original tokens from z_cat. Two decoder variants exist: histogram decoders (C1/C3) output dense [B,T,256,3] representations, while token decoders (C2/C4) output sparse [B,T,48,5] with validity scores.`;
    breakAfter();

    let t0 = afterTime(null, 1.5, 0.5);
    if (t0.active) {
        let p = t0.t;
        layout.audioDecProj.highlight = Math.max(layout.audioDecProj.highlight, p * 0.5);
        layout.audioDecLstm.highlight = Math.max(layout.audioDecLstm.highlight, p * 0.5);
        layout.audioDecOutput.highlight = Math.max(layout.audioDecOutput.highlight, p * 0.6);
        layout.vibDecProj.highlight = Math.max(layout.vibDecProj.highlight, p * 0.5);
        layout.vibDecLstm.highlight = Math.max(layout.vibDecLstm.highlight, p * 0.5);
        layout.vibDecOutput.highlight = Math.max(layout.vibDecOutput.highlight, p * 0.6);
    }
    breakAfter();

    commentary()`Five loss terms: ${reconRef} ensures faithful decoding. ${klSRef} and ${klPRef} regularize the latent distributions (KL divergence from standard normal). ${infoRef} aligns shared representations across modalities (contrastive). ${diffRef} enforces orthogonality between shared and private subspaces. The JEPA-lite variant (C5/C6) removes the decoder entirely, relying only on predictive losses.`;
    breakAfter();

    let t1 = afterTime(null, 2.0, 0.5);
    if (t1.active) {
        let p = t1.t;
        layout.reconLoss.highlight = Math.max(layout.reconLoss.highlight, p * 0.6);
        layout.klSharedLoss.highlight = Math.max(layout.klSharedLoss.highlight, p * 0.5);
        layout.klPrivateLoss.highlight = Math.max(layout.klPrivateLoss.highlight, p * 0.5);
        layout.infoNceLoss.highlight = Math.max(layout.infoNceLoss.highlight, p * 0.7);
        layout.diffLoss.highlight = Math.max(layout.diffLoss.highlight, p * 0.5);
    }
    breakAfter();
}
