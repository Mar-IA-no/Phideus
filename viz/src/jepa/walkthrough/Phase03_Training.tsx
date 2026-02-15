import { Vec3 } from "@/src/utils/vector";
import { IJepaWalkthroughArgs, setJepaInitialCamera } from "./JepaWalkthrough";
import { JepaDim } from "../JepaDimStyle";
import { getEncoderBlocks } from "../JepaModelLayout";

export function jepaPhase03_Training(args: IJepaWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept } = tools;

    setJepaInitialCamera(state, new Vec3(0, 0, -800), new Vec3(290, 15, 18));

    let lossRef = c_blockRef('InfoNCE loss', layout.infoNCELoss);

    commentary()`The ${lossRef} is a contrastive objective: for each prediction, it must score higher similarity with the true target than with all other samples in the batch. This forces the model to capture discriminative information, not just averages.`;
    breakAfter();

    let t0 = afterTime(null, 1.5, 0.5);
    if (t0.active) {
        dimExcept([layout.predictorAtoV, layout.predictorVtoA, layout.infoNCELoss], 0.08);
        layout.infoNCELoss.highlight = Math.max(layout.infoNCELoss.highlight, t0.t * 0.8);
    }
    breakAfter();

    commentary()`Why no decoder? In a VAE, the decoder reconstructs inputs from latents. This creates a shortcut: the model can encode surface statistics (mean energy, spectral shape) that are easy to reconstruct but uninformative for cross-modal matching. By removing the decoder, JEPA-Lite forces all learning signal to come from cross-modal prediction.`;
    breakAfter();

    let t1 = afterTime(null, 1.5, 0.5);
    if (t1.active) {
        for (let blk of layout.cubes) {
            blk.highlight = Math.max(blk.highlight, t1.t * 0.3);
        }
    }
    breakAfter();

    commentary()`The total loss is L = L_infoNCE(pred_a2v, z_vib) + L_infoNCE(pred_v2a, z_audio). Both directions are weighted equally. Temperature scaling (\u03C4=0.07) sharpens the contrastive distribution, making the model more selective about what constitutes a match.`;
    breakAfter();

    let t2 = afterTime(null, 2.0, 0.5);
    if (t2.active) {
        for (let blk of getEncoderBlocks(layout.audio)) blk.highlight = Math.max(blk.highlight, t2.t * 0.3);
        for (let blk of getEncoderBlocks(layout.vibration)) blk.highlight = Math.max(blk.highlight, t2.t * 0.3);
        layout.predictorAtoV.highlight = Math.max(layout.predictorAtoV.highlight, t2.t * 0.5);
        layout.predictorVtoA.highlight = Math.max(layout.predictorVtoA.highlight, t2.t * 0.5);
        layout.infoNCELoss.highlight = Math.max(layout.infoNCELoss.highlight, t2.t * 0.6);
    }
    breakAfter();
}
