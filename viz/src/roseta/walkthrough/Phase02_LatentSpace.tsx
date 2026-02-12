import { Vec3 } from "@/src/utils/vector";
import { IRosetaWalkthroughArgs, setRosetaInitialCamera, moveRosetaCameraTo } from "./RosetaWalkthrough";
import { getLatentBlocks } from "../RosetaModelLayout";
import { RosetaDim } from "../RosetaDimStyle";

export function rosetaPhase02_LatentSpace(args: IRosetaWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept, drawDimLabels } = tools;

    let allLatent = [...getLatentBlocks(layout.audio), ...getLatentBlocks(layout.vibration)];
    dimExcept(allLatent);

    let audioLatentY = layout.audio.muShared.y;
    setRosetaInitialCamera(state, new Vec3(0, 0, -audioLatentY - 20), new Vec3(290, 12, 6));

    let zShared = c_dimRef('z_shared (32D)', RosetaDim.Z_shared);
    let zPrivate = c_dimRef('z_private (16D)', RosetaDim.Z_private);

    commentary()`The latent space is the heart of RosetaVAE. Instead of a single latent variable, the representation is _factorized_: ${zShared} captures information common to both modalities (e.g., musical structure, rhythm), while ${zPrivate} captures what's unique to each domain (audio timbre vs vibration texture). This factorization is the key architectural decision.`;
    breakAfter();

    // Step 1: Audio latent - mu/sigma heads
    let t0 = afterTime(null, 2.5, 0.5);

    let muSharedRef = c_blockRef('μ_shared', layout.audio.muShared);
    let sigmaSharedRef = c_blockRef('σ_shared', layout.audio.logvarShared);

    commentary()`From the encoder output, separate linear heads produce ${muSharedRef} and ${sigmaSharedRef} for the shared latent (32D), plus μ_private and σ_private for the private latent (16D). The Z-staggered layout shows the mu/logvar pairs — mean in front, log-variance behind — for each latent type.`;
    breakAfter();

    if (t0.active) {
        let pulse = 0.4 + 0.25 * Math.sin(wt.time * 3);
        layout.audio.muShared.highlight = Math.max(layout.audio.muShared.highlight, pulse);
        layout.audio.logvarShared.highlight = Math.max(layout.audio.logvarShared.highlight, pulse * 0.8);
        layout.audio.muPrivate.highlight = Math.max(layout.audio.muPrivate.highlight, pulse * 0.7);
        layout.audio.logvarPrivate.highlight = Math.max(layout.audio.logvarPrivate.highlight, pulse * 0.6);
        drawDimLabels(layout.audio.muShared, t0.t * 0.8);
        drawDimLabels(layout.audio.muPrivate, t0.t * 0.6);
    }

    breakAfter();

    // Step 2: Reparameterization trick
    let t1 = afterTime(null, 2.5, 0.5);
    let t0c = afterTime(t0, 0.3);
    cleanup(t0c, [t0]);

    moveRosetaCameraTo(state, t1, new Vec3(-80, 0, -layout.audio.zShared.y - 10), new Vec3(290, 10, 3));

    let zSRef = c_blockRef('z_shared', layout.audio.zShared);
    let zPRef = c_blockRef('z_private', layout.audio.zPrivate);

    commentary()`The reparameterization trick: z = μ + σ × ε (where ε ~ N(0,1)). This produces the actual latent samples — ${zSRef} (32D, gold) and ${zPRef} (16D, purple). Notice the size difference: the shared latent is _twice as large_ as the private, reflecting the assumption that cross-modal structure is more complex than modality-specific variations.`;
    breakAfter();

    if (t1.active) {
        let sharedPulse = 0.5 + 0.3 * Math.sin(wt.time * 3.5);
        let privatePulse = 0.4 + 0.25 * Math.sin(wt.time * 3.5 + 1.5);
        layout.audio.zShared.highlight = Math.max(layout.audio.zShared.highlight, sharedPulse);
        layout.audio.zPrivate.highlight = Math.max(layout.audio.zPrivate.highlight, privatePulse);
        drawDimLabels(layout.audio.zShared, t1.t * 0.9);
        drawDimLabels(layout.audio.zPrivate, t1.t * 0.9);
    }

    breakAfter();

    // Step 3: Concatenation and projection
    let t2 = afterTime(null, 2.0, 0.5);
    let t1c = afterTime(t1, 0.3);
    cleanup(t1c, [t1]);

    moveRosetaCameraTo(state, t2, new Vec3(-80, 0, -layout.audio.zConcat.y - 10), new Vec3(290, 10, 3));

    let concatRef = c_blockRef('[z_sh|z_pr]', layout.audio.zConcat);
    let d48 = c_dimRef('48D', RosetaDim.Z_total);
    let d128 = c_dimRef('128D', RosetaDim.D_hidden);

    commentary()`The latent samples are concatenated: ${concatRef} forms a ${d48} vector (32 + 16). Then z_proj maps this back to ${d128} for the decoder. This bottleneck — 768D input compressed through LSTM to 48D latent to 128D decoder input — is the hourglass shape that forces information compression.`;
    breakAfter();

    if (t2.active) {
        let pulse = 0.4 + 0.2 * Math.sin(wt.time * 3);
        layout.audio.zConcat.highlight = Math.max(layout.audio.zConcat.highlight, pulse);
        layout.audio.zProj.highlight = Math.max(layout.audio.zProj.highlight, pulse * 0.8);
        drawDimLabels(layout.audio.zConcat, t2.t * 0.8);
    }

    breakAfter();

    // Step 4: Show both domains' latent spaces side by side
    let t3 = afterTime(null, 2.5, 0.5);
    let t2c = afterTime(t2, 0.4);
    cleanup(t2c, [t2]);

    moveRosetaCameraTo(state, t3, new Vec3(0, 0, -audioLatentY - 30), new Vec3(290, 15, 8));

    commentary()`Both domains produce the same latent factorization. The cross-connection happens here: audio z_shared and vibration z_shared are pulled together by InfoNCE loss (cosine similarity, temperature=0.07), while z_private variables are pushed _apart_ by the differentiation loss. This is how the model learns what's shared vs. what's unique to each modality.`;
    breakAfter();

    if (t3.active) {
        // Audio z_shared and vibration z_shared pulse in sync (InfoNCE alignment)
        let syncPulse = 0.5 + 0.3 * Math.sin(wt.time * 2.5);
        layout.audio.zShared.highlight = Math.max(layout.audio.zShared.highlight, syncPulse);
        layout.vibration.zShared.highlight = Math.max(layout.vibration.zShared.highlight, syncPulse);

        // Private latents pulse at different phase (pushed apart)
        let privatePulse = 0.3 + 0.2 * Math.sin(wt.time * 3 + Math.PI);
        layout.audio.zPrivate.highlight = Math.max(layout.audio.zPrivate.highlight, privatePulse);
        layout.vibration.zPrivate.highlight = Math.max(layout.vibration.zPrivate.highlight, privatePulse);

        drawDimLabels(layout.audio.zShared, 0.8);
        drawDimLabels(layout.vibration.zShared, 0.8);
    }

    breakAfter();
}
