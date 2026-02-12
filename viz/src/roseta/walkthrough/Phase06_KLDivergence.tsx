import { Vec3 } from "@/src/utils/vector";
import { IRosetaWalkthroughArgs, setRosetaInitialCamera, moveRosetaCameraTo } from "./RosetaWalkthrough";
import { RosetaDim } from "../RosetaDimStyle";

export function rosetaPhase06_KLDivergence(args: IRosetaWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept, drawDimLabels } = tools;

    dimExcept([
        layout.audio.muShared, layout.audio.logvarShared,
        layout.audio.muPrivate, layout.audio.logvarPrivate,
        layout.vibration.muShared, layout.vibration.logvarShared,
        layout.vibration.muPrivate, layout.vibration.logvarPrivate,
        layout.lossKLShared, layout.lossKLPrivate,
    ]);

    let muY = layout.audio.muShared.y;
    setRosetaInitialCamera(state, new Vec3(0, 0, -muY - 15), new Vec3(290, 14, 7));

    let klShared = c_blockRef('KL_shared', layout.lossKLShared);
    let klPrivate = c_blockRef('KL_private', layout.lossKLPrivate);

    commentary()`KL divergence regularizes the latent distributions toward N(0,1). RosetaVAE uses _separate_ KL weights: ${klShared} (β_shared) and ${klPrivate} (β_private). This asymmetry is crucial — the shared latent needs more freedom to encode cross-modal structure, while the private latent can be more tightly regularized.`;
    breakAfter();

    // Step 1: mu/logvar heads — the source of KL
    let t0 = afterTime(null, 2.5, 0.5);

    let muSh = c_blockRef('μ_shared', layout.audio.muShared);
    let sigSh = c_blockRef('σ_shared', layout.audio.logvarShared);

    commentary()`KL divergence is computed from the ${muSh} and ${sigSh} heads: KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²). When μ→0 and σ→1, KL→0 — the latent matches the prior. The β weight controls how strongly we enforce this: too high crushes information, too low allows posterior collapse in some dimensions.`;
    breakAfter();

    if (t0.active) {
        // Pulse mu/logvar pairs for both domains
        let pulse = 0.4 + 0.25 * Math.sin(wt.time * 3);
        layout.audio.muShared.highlight = Math.max(layout.audio.muShared.highlight, pulse);
        layout.audio.logvarShared.highlight = Math.max(layout.audio.logvarShared.highlight, pulse * 0.8);
        layout.vibration.muShared.highlight = Math.max(layout.vibration.muShared.highlight, pulse);
        layout.vibration.logvarShared.highlight = Math.max(layout.vibration.logvarShared.highlight, pulse * 0.8);
        drawDimLabels(layout.audio.muShared, t0.t * 0.8);
        drawDimLabels(layout.vibration.muShared, t0.t * 0.8);
    }

    breakAfter();

    // Step 2: Shared vs private KL weights
    let t1 = afterTime(null, 2.5, 0.5);
    let t0c = afterTime(t0, 0.3);
    cleanup(t0c, [t0]);

    moveRosetaCameraTo(state, t1, new Vec3(0, 0, -muY - 5), new Vec3(290, 12, 5));

    let betaShared = c_dimRef('β_shared', RosetaDim.Loss_KL_shared);
    let betaPrivate = c_dimRef('β_private', RosetaDim.Loss_KL_private);

    commentary()`${betaShared} = 0.1 (loose) vs ${betaPrivate} = 0.5 (tight). The shared latent gets 5x less KL pressure because it must encode rich cross-modal structure — musical content, temporal dynamics, expressive nuances. The private latent is more tightly regularized because modality-specific information should be simpler and more compressible.`;
    breakAfter();

    if (t1.active) {
        // Shared latents glow brightly (more freedom)
        let sharedPulse = 0.6 + 0.3 * Math.sin(wt.time * 2.5);
        layout.audio.muShared.highlight = Math.max(layout.audio.muShared.highlight, sharedPulse);
        layout.audio.logvarShared.highlight = Math.max(layout.audio.logvarShared.highlight, sharedPulse * 0.7);
        layout.vibration.muShared.highlight = Math.max(layout.vibration.muShared.highlight, sharedPulse);
        layout.vibration.logvarShared.highlight = Math.max(layout.vibration.logvarShared.highlight, sharedPulse * 0.7);

        // Private latents dimmer (more constrained)
        let privatePulse = 0.3 + 0.15 * Math.sin(wt.time * 3.5);
        layout.audio.muPrivate.highlight = Math.max(layout.audio.muPrivate.highlight, privatePulse);
        layout.audio.logvarPrivate.highlight = Math.max(layout.audio.logvarPrivate.highlight, privatePulse * 0.7);
        layout.vibration.muPrivate.highlight = Math.max(layout.vibration.muPrivate.highlight, privatePulse);
        layout.vibration.logvarPrivate.highlight = Math.max(layout.vibration.logvarPrivate.highlight, privatePulse * 0.7);
    }

    breakAfter();

    // Step 3: KL loss blocks
    let t2 = afterTime(null, 2.0, 0.5);
    let t1c = afterTime(t1, 0.3);
    cleanup(t1c, [t1]);

    moveRosetaCameraTo(state, t2, new Vec3(0, 0, -layout.lossKLShared.y + 15), new Vec3(290, 8, 3));

    commentary()`The ${klShared} and ${klPrivate} loss blocks sum KL divergence across all latent dimensions, weighted by their respective β values. Together with reconstruction loss and InfoNCE, they form the ELBO — the evidence lower bound that the VAE maximizes. The β-VAE framework gives us explicit control over the information-compression trade-off.`;
    breakAfter();

    if (t2.active) {
        let klsPulse = 0.5 + 0.35 * Math.sin(wt.time * 3);
        let klpPulse = 0.5 + 0.35 * Math.sin(wt.time * 3 + 1.5);
        layout.lossKLShared.highlight = Math.max(layout.lossKLShared.highlight, klsPulse);
        layout.lossKLPrivate.highlight = Math.max(layout.lossKLPrivate.highlight, klpPulse);
        drawDimLabels(layout.lossKLShared, t2.t * 0.8);
        drawDimLabels(layout.lossKLPrivate, t2.t * 0.8);
    }

    breakAfter();
}
