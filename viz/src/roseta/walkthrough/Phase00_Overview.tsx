import { Vec3 } from "@/src/utils/vector";
import { IRosetaWalkthroughArgs, setRosetaInitialCamera, moveRosetaCameraTo } from "./RosetaWalkthrough";
import { getDomainBlocks } from "../RosetaModelLayout";
import { RosetaDim } from "../RosetaDimStyle";

export function rosetaPhase00_Overview(args: IRosetaWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, drawDimLabels } = tools;

    setRosetaInitialCamera(state, new Vec3(0, 0, -400), new Vec3(290, 22, 16));

    let zShared = c_dimRef('z_shared (32D)', RosetaDim.Z_shared);
    let zPrivate = c_dimRef('z_private (16D)', RosetaDim.Z_private);
    let infoNCE = c_dimRef('InfoNCE', RosetaDim.Loss_InfoNCE);

    commentary()`RosetaVAE is a dual-domain variational autoencoder that learns shared and private representations across Audio and Vibration modalities. Each domain has an encoder-decoder pair with a factorized latent space: ${zShared} captures cross-modal structure aligned via ${infoNCE}, while ${zPrivate} encodes modality-specific information pushed apart by a differentiation loss.`;
    breakAfter();

    // Step 1: Wide view
    let t0 = afterTime(null, 1.5, 0.5);

    if (t0.active) {
        // Gentle highlight on both domains
        for (let blk of getDomainBlocks(layout.audio)) {
            blk.highlight = Math.max(blk.highlight, t0.t * 0.15);
        }
        for (let blk of getDomainBlocks(layout.vibration)) {
            blk.highlight = Math.max(blk.highlight, t0.t * 0.15);
        }
        // Latent blocks extra bright
        layout.audio.zShared.highlight = Math.max(layout.audio.zShared.highlight, t0.t * 0.4);
        layout.vibration.zShared.highlight = Math.max(layout.vibration.zShared.highlight, t0.t * 0.4);
    }

    breakAfter();

    // Step 2: Zoom to audio domain
    let t1 = afterTime(null, 1.5, 0.5);
    let t0c = afterTime(t0, 0.4);
    cleanup(t0c, [t0]);

    moveRosetaCameraTo(state, t1, new Vec3(-60, 0, -300), new Vec3(292, 18, 10));

    let lstmEnc = c_blockRef('BiLSTM encoder', layout.audio.lstmEncoder);
    let d128 = c_dimRef('128D', RosetaDim.D_hidden);

    commentary()`The Audio Domain: Input spectral features [T,256,3] are projected to ${d128}, processed by a ${lstmEnc} (2 layers, bidirectional), then split into shared and private latent variables through separate mean/logvar heads. The decoder mirrors this: latent → projection → BiLSTM → output → softmax → reconstruction.`;
    breakAfter();

    if (t1.active) {
        for (let blk of getDomainBlocks(layout.audio)) {
            blk.highlight = Math.max(blk.highlight, t1.t * 0.3);
        }
        // Dim vibration
        for (let blk of getDomainBlocks(layout.vibration)) blk.opacity = 0.3;
        drawDimLabels(layout.audio.lstmEncoder, t1.t * 0.7);
    }

    breakAfter();

    // Step 3: Pan to vibration domain
    let t2 = afterTime(null, 1.5, 0.5);
    let t1c = afterTime(t1, 0.4);
    cleanup(t1c, [t1]);

    moveRosetaCameraTo(state, t2, new Vec3(60, 0, -300), new Vec3(288, 18, 10));

    commentary()`The Vibration Domain is architecturally _identical_ — a symmetric encoder-decoder with the same hidden dimensions. This symmetry is by design: both modalities should be treated equally, with their differences captured in the private latent variables rather than the architecture.`;
    breakAfter();

    if (t2.active) {
        for (let blk of getDomainBlocks(layout.vibration)) {
            blk.highlight = Math.max(blk.highlight, t2.t * 0.3);
        }
        for (let blk of getDomainBlocks(layout.audio)) blk.opacity = 0.3;
        drawDimLabels(layout.vibration.lstmEncoder, t2.t * 0.7);
    }

    breakAfter();

    // Step 4: Zoom out to see losses
    let t3 = afterTime(null, 2.0, 0.5);
    let t2c = afterTime(t2, 0.4);
    cleanup(t2c, [t2]);

    moveRosetaCameraTo(state, t3, new Vec3(0, 0, -600), new Vec3(290, 25, 20));

    let reconLoss = c_blockRef('reconstruction', layout.lossRecon);
    let klLoss = c_blockRef('KL divergence', [layout.lossKLShared, layout.lossKLPrivate]);

    commentary()`Five loss terms train the system: ${reconLoss} (MSE) for both domains, ${klLoss} on shared and private latents (with different weights), ${infoNCE} to align z_shared across modalities, and a differentiation hinge loss to push z_private apart. The balance between these forces shapes how information is factorized between shared and private representations.`;
    breakAfter();

    if (t3.active) {
        layout.lossRecon.highlight = Math.max(layout.lossRecon.highlight, t3.t * 0.6);
        layout.lossKLShared.highlight = Math.max(layout.lossKLShared.highlight, t3.t * 0.6);
        layout.lossKLPrivate.highlight = Math.max(layout.lossKLPrivate.highlight, t3.t * 0.6);
        layout.lossInfoNCE.highlight = Math.max(layout.lossInfoNCE.highlight, t3.t * 0.6);
        layout.lossDiff.highlight = Math.max(layout.lossDiff.highlight, t3.t * 0.6);
    }

    breakAfter();
}
