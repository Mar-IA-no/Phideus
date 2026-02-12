import { Vec3 } from "@/src/utils/vector";
import { IRosetaWalkthroughArgs, setRosetaInitialCamera, moveRosetaCameraTo } from "./RosetaWalkthrough";
import { RosetaDim } from "../RosetaDimStyle";

export function rosetaPhase05_InfoNCE(args: IRosetaWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept, drawDimLabels } = tools;

    dimExcept([
        layout.audio.zShared, layout.vibration.zShared,
        layout.lossInfoNCE,
    ]);

    let latentY = layout.audio.zShared.y;
    setRosetaInitialCamera(state, new Vec3(0, 0, -latentY - 10), new Vec3(290, 12, 6));

    let infoNCE = c_blockRef('InfoNCE loss', layout.lossInfoNCE);
    let zShared = c_dimRef('z_shared (32D)', RosetaDim.Z_shared);

    commentary()`InfoNCE is the cross-modal alignment force. It operates _only_ on the shared latent variables ${zShared}, pulling matching audio-vibration pairs together while pushing non-matching pairs apart. Temperature = 0.07 sharpens the cosine similarity distribution, making the model more discriminative about what constitutes a "match."`;
    breakAfter();

    // Step 1: Show both z_shared pulsing in sync
    let t0 = afterTime(null, 2.5, 0.5);

    let audioZ = c_blockRef('audio z_shared', layout.audio.zShared);
    let vibZ = c_blockRef('vibration z_shared', layout.vibration.zShared);

    commentary()`${audioZ} and ${vibZ} are compared via cosine similarity: sim(z_a, z_v) = dot(z_a, z_v) / (||z_a|| * ||z_v||). For matching pairs, the loss is: -log(exp(sim/τ) / Σ exp(sim_neg/τ)). This is contrastive learning — the same principle behind CLIP, but applied to shared latent variables rather than full embeddings.`;
    breakAfter();

    if (t0.active) {
        // Both z_shared pulse in sync — they should be similar
        let syncPulse = 0.5 + 0.35 * Math.sin(wt.time * 3);
        layout.audio.zShared.highlight = Math.max(layout.audio.zShared.highlight, syncPulse);
        layout.vibration.zShared.highlight = Math.max(layout.vibration.zShared.highlight, syncPulse);
        drawDimLabels(layout.audio.zShared, t0.t * 0.9);
        drawDimLabels(layout.vibration.zShared, t0.t * 0.9);
    }

    breakAfter();

    // Step 2: InfoNCE loss block
    let t1 = afterTime(null, 2.5, 0.5);
    let t0c = afterTime(t0, 0.3);
    cleanup(t0c, [t0]);

    moveRosetaCameraTo(state, t1, new Vec3(0, 0, -layout.lossInfoNCE.y + 15), new Vec3(290, 8, 3));

    let lambdaNCE = c_dimRef('λ_InfoNCE', RosetaDim.Loss_InfoNCE);

    commentary()`The ${infoNCE} block (weight = ${lambdaNCE}) provides the primary alignment signal. Unlike VICReg (used in Phideus/BloqueA), InfoNCE uses explicit negative pairs within the batch. Every non-matching audio-vibration pair in the batch serves as a negative example. Larger batch sizes provide more negatives and stronger contrastive signal.`;
    breakAfter();

    if (t1.active) {
        let ncePulse = 0.5 + 0.4 * Math.sin(wt.time * 3.5);
        layout.lossInfoNCE.highlight = Math.max(layout.lossInfoNCE.highlight, ncePulse);
        drawDimLabels(layout.lossInfoNCE, t1.t * 0.8);
        // Z_shared blocks glow softly
        layout.audio.zShared.highlight = Math.max(layout.audio.zShared.highlight, 0.25);
        layout.vibration.zShared.highlight = Math.max(layout.vibration.zShared.highlight, 0.25);
    }

    breakAfter();

    // Step 3: Why only z_shared?
    let t2 = afterTime(null, 2.0, 0.5);
    let t1c = afterTime(t1, 0.3);
    cleanup(t1c, [t1]);

    moveRosetaCameraTo(state, t2, new Vec3(0, 0, -latentY - 10), new Vec3(290, 14, 7));

    commentary()`InfoNCE operates _only_ on z_shared, not z_private. This is deliberate: the shared latent should encode cross-modal information (musical content, timing, dynamics), while the private latent should encode modality-specific details (audio timbre, vibration resonance). By aligning only z_shared, the model is forced to factorize its representation — shared structure flows to z_shared, unique details to z_private.`;
    breakAfter();

    if (t2.active) {
        let sharedPulse = 0.6 + 0.3 * Math.sin(wt.time * 2.5);
        layout.audio.zShared.highlight = Math.max(layout.audio.zShared.highlight, sharedPulse);
        layout.vibration.zShared.highlight = Math.max(layout.vibration.zShared.highlight, sharedPulse);
        // Private latents deliberately dim
        layout.audio.zPrivate.opacity = Math.max(layout.audio.zPrivate.opacity, 0.3);
        layout.vibration.zPrivate.opacity = Math.max(layout.vibration.zPrivate.opacity, 0.3);
    }

    breakAfter();
}
