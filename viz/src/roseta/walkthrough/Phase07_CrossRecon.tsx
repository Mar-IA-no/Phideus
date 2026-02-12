import { Vec3 } from "@/src/utils/vector";
import { IRosetaWalkthroughArgs, setRosetaInitialCamera, moveRosetaCameraTo } from "./RosetaWalkthrough";
import { getDomainBlocks } from "../RosetaModelLayout";
import { RosetaDim } from "../RosetaDimStyle";

export function rosetaPhase07_CrossRecon(args: IRosetaWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept, drawDimLabels } = tools;

    let allBlocks = [
        ...getDomainBlocks(layout.audio), ...getDomainBlocks(layout.vibration),
        layout.lossRecon, layout.lossKLShared, layout.lossKLPrivate,
        layout.lossInfoNCE, layout.lossDiff,
    ];
    dimExcept(allBlocks);

    setRosetaInitialCamera(state, new Vec3(0, 0, -layout.audio.zShared.y - 10), new Vec3(290, 14, 8));

    // Step 1: Cross-reconstruction concept
    let t0 = afterTime(null, 2.5, 0.5);

    let audioZsh = c_blockRef('audio z_shared', layout.audio.zShared);
    let vibZpr = c_blockRef('vibration z_private', layout.vibration.zPrivate);

    commentary()`Cross-reconstruction is the ultimate test of factorization: take ${audioZsh} and combine it with ${vibZpr}, then decode through the vibration decoder. If z_shared truly captures cross-modal structure and z_private captures modality-specific details, this "chimera" should produce a valid vibration signal with the musical content of the audio and the textural characteristics of the vibration.`;
    breakAfter();

    if (t0.active) {
        // Audio z_shared glows gold
        let sharedPulse = 0.6 + 0.3 * Math.sin(wt.time * 2.5);
        layout.audio.zShared.highlight = Math.max(layout.audio.zShared.highlight, sharedPulse);
        // Vibration z_private glows purple
        let privatePulse = 0.5 + 0.3 * Math.sin(wt.time * 2.5 + Math.PI * 0.5);
        layout.vibration.zPrivate.highlight = Math.max(layout.vibration.zPrivate.highlight, privatePulse);
        drawDimLabels(layout.audio.zShared, t0.t * 0.9);
        drawDimLabels(layout.vibration.zPrivate, t0.t * 0.9);
    }

    breakAfter();

    // Step 2: Differentiation loss
    let t1 = afterTime(null, 2.5, 0.5);
    let t0c = afterTime(t0, 0.3);
    cleanup(t0c, [t0]);

    moveRosetaCameraTo(state, t1, new Vec3(0, 0, -layout.lossDiff.y + 15), new Vec3(290, 8, 4));

    let diffLoss = c_blockRef('differentiation loss', layout.lossDiff);
    let lambdaDiff = c_dimRef('λ_diff', RosetaDim.Loss_diff);

    commentary()`The ${diffLoss} (${lambdaDiff}) uses a hinge loss to push z_private representations _apart_ between modalities: max(0, margin - ||z_priv_audio - z_priv_vibration||). If the private latents become too similar, this penalty activates. Combined with InfoNCE pulling z_shared together, this creates opposing forces that enforce the shared/private factorization.`;
    breakAfter();

    if (t1.active) {
        let diffPulse = 0.5 + 0.4 * Math.sin(wt.time * 3.5);
        layout.lossDiff.highlight = Math.max(layout.lossDiff.highlight, diffPulse);
        // Private latents pulse at opposing phase — pushed apart
        let phase1 = 0.4 + 0.25 * Math.sin(wt.time * 3);
        let phase2 = 0.4 + 0.25 * Math.sin(wt.time * 3 + Math.PI);
        layout.audio.zPrivate.highlight = Math.max(layout.audio.zPrivate.highlight, phase1);
        layout.vibration.zPrivate.highlight = Math.max(layout.vibration.zPrivate.highlight, phase2);
        drawDimLabels(layout.lossDiff, t1.t * 0.8);
    }

    breakAfter();

    // Step 3: All losses together
    let t2 = afterTime(null, 2.5, 0.5);
    let t1c = afterTime(t1, 0.3);
    cleanup(t1c, [t1]);

    moveRosetaCameraTo(state, t2, new Vec3(0, 0, -layout.lossRecon.y + 20), new Vec3(290, 10, 5));

    let reconLoss = c_blockRef('reconstruction', layout.lossRecon);
    let infoNCE = c_blockRef('InfoNCE', layout.lossInfoNCE);

    commentary()`The complete RosetaVAE loss: L = L_recon + β_sh*KL_shared + β_pr*KL_private + λ_NCE*InfoNCE + λ_diff*Diff. Five forces in balance — ${reconLoss} ensures fidelity, KL ensures smooth latent geometry, ${infoNCE} aligns shared representations, and differentiation separates private ones. Each weight is a design choice that shapes what the model learns.`;
    breakAfter();

    if (t2.active) {
        // All 5 loss blocks pulse in multi-frequency pattern
        let losses = [layout.lossRecon, layout.lossKLShared, layout.lossKLPrivate, layout.lossInfoNCE, layout.lossDiff];
        for (let i = 0; i < losses.length; i++) {
            let pulse = 0.4 + 0.3 * Math.sin(wt.time * 3 + i * 1.2);
            losses[i].highlight = Math.max(losses[i].highlight, pulse);
        }
    }

    breakAfter();

    // Step 4: Grand sweep — entire architecture
    let t3 = afterTime(null, 3.0, 0.5);
    let t2c = afterTime(t2, 0.4);
    cleanup(t2c, [t2]);

    moveRosetaCameraTo(state, t3, new Vec3(0, 0, -layout.audio.lstmEncoder.y - 20), new Vec3(290, 18, 12));

    commentary()`RosetaVAE: a dual-domain variational autoencoder that learns factorized representations of audio and vibration. The shared latent captures musical structure common to both modalities; the private latent preserves what makes each domain unique. Through contrastive alignment and disentanglement, the model discovers the hidden correspondence between sound and touch.`;
    breakAfter();

    if (t3.active) {
        // Grand sweep: warm glow across all blocks
        let allSweep = [...getDomainBlocks(layout.audio), ...getDomainBlocks(layout.vibration)];
        let maxY = Math.max(...allSweep.map(b => b.y));
        let minY = Math.min(...allSweep.map(b => b.y));
        let sweepPos = minY + (maxY - minY) * t3.t;
        let glowWidth = (maxY - minY) * 0.25;

        for (let b of allSweep) {
            let dist = Math.abs(b.y - sweepPos);
            if (dist < glowWidth) {
                let intensity = 0.6 * (1.0 - dist / glowWidth);
                b.highlight = Math.max(b.highlight, intensity);
            }
            // Afterglow for passed blocks
            if (b.y < sweepPos) {
                let warmGlow = 0.15 + 0.08 * Math.sin(wt.time * 1.2 + b.y * 0.008 + b.x * 0.005);
                b.highlight = Math.max(b.highlight, warmGlow);
            }
        }

        // Loss blocks glow steadily
        let losses = [layout.lossRecon, layout.lossKLShared, layout.lossKLPrivate, layout.lossInfoNCE, layout.lossDiff];
        for (let l of losses) {
            l.highlight = Math.max(l.highlight, 0.3 + 0.15 * Math.sin(wt.time * 2));
        }
    }

    breakAfter();

    // Step 5: Orbit finale
    let t4 = afterTime(null, 4.0, 1.0);
    let t3c = afterTime(t3, 0.5);
    cleanup(t3c, [t3]);

    commentary()`The architecture is complete. Two symmetric towers — audio and vibration — connected through a factorized latent space. InfoNCE pulls shared representations together. KL divergence shapes the latent geometry. Differentiation loss ensures private spaces remain distinct. RosetaVAE doesn't just compress data — it discovers the deep structure that links how we hear and how we feel.`;

    if (t4.active) {
        // Slow orbit around the full model
        let yaw = 290 + 20 * t4.t;
        let pitch = 18 - 6 * Math.sin(t4.t * Math.PI);
        let zoom = 12 + 3 * Math.sin(t4.t * Math.PI * 0.5);
        moveRosetaCameraTo(state, t4, new Vec3(0, 0, -layout.audio.lstmEncoder.y - 20), new Vec3(yaw, pitch, zoom));

        // Everything glows warmly
        let allB = [...getDomainBlocks(layout.audio), ...getDomainBlocks(layout.vibration)];
        for (let b of allB) {
            let warmGlow = 0.2 + 0.1 * Math.sin(wt.time * 1.5 + b.y * 0.006 + b.x * 0.004);
            b.highlight = Math.max(b.highlight, warmGlow);
        }
        let losses = [layout.lossRecon, layout.lossKLShared, layout.lossKLPrivate, layout.lossInfoNCE, layout.lossDiff];
        for (let l of losses) {
            l.highlight = Math.max(l.highlight, 0.35 + 0.15 * Math.sin(wt.time * 2 + 0.5));
        }
    }

    breakAfter();
}
