import { Vec3 } from "@/src/utils/vector";
import { IBloqueaWalkthroughArgs, setBloqueaInitialCamera, moveBloqueaCameraTo } from "./BloqueaWalkthrough";
import { BloqueaDim } from "../BloqueaDimStyle";

export function bloqueaPhase06_VICReg(args: IBloqueaWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept, drawDimLabels } = tools;

    dimExcept([
        layout.audioEmbedding, layout.midiEmbedding,
        layout.vicregInv, layout.vicregVar, layout.vicregCov,
    ]);

    let sharedY = layout.audioEmbedding.y;
    setBloqueaInitialCamera(state, new Vec3(0, 0, -sharedY - 20), new Vec3(290, 12, 5));

    let invRef = c_blockRef('invariance', layout.vicregInv);
    let varRef = c_blockRef('variance', layout.vicregVar);
    let covRef = c_blockRef('covariance', layout.vicregCov);
    let d256 = c_dimRef('256D', BloqueaDim.D_proj);

    commentary()`VICReg loss — identical to Run D. Three forces shape the ${d256} space: ${invRef} (MSE, weight=25) pulls matching pairs together, ${varRef} (hinge, weight=25) prevents collapse, and ${covRef} (decorrelation, weight=1) ensures each dimension encodes unique information.`;
    breakAfter();

    // Step 1: Invariance
    let t0 = afterTime(null, 2.5, 0.5);

    let loss = c_dimRef('Loss', BloqueaDim.Loss);

    commentary()`_INVARIANCE_ (${loss}, weight=25): ||z_audio - z_midi||^2 — the gravitational pull that drags matching representations together. This is the primary alignment signal, and it flows back through the projection heads into both towers — including through the adapter bottlenecks in layers 0-1.`;
    breakAfter();

    if (t0.active) {
        let invPulse = 0.5 + 0.35 * Math.sin(wt.time * 3.5);
        layout.vicregInv.highlight = Math.max(layout.vicregInv.highlight, invPulse);
        drawDimLabels(layout.vicregInv, t0.t * 0.8);
        let embGlow = 0.2 + 0.15 * Math.sin(wt.time * 3.5);
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, embGlow);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, embGlow);
    }

    breakAfter();

    // Step 2: Variance
    let t1 = afterTime(null, 2.5, 0.5);
    let t0c = afterTime(t0, 0.3);
    cleanup(t0c, [t0]);

    let varY = layout.vicregVar.y;
    moveBloqueaCameraTo(state, t1, new Vec3(0, 0, -varY + 15), new Vec3(290, 8, 2.5));

    commentary()`_VARIANCE_ (hinge ${loss}, weight=25): max(0, 1 - std(z_j)) for each of the 256 dimensions. Prevents representational collapse — every dimension must maintain std >= 1 across the batch.`;
    breakAfter();

    if (t1.active) {
        let varPulse = 0.5 + 0.35 * Math.sin(wt.time * 4);
        layout.vicregVar.highlight = Math.max(layout.vicregVar.highlight, varPulse);
        drawDimLabels(layout.vicregVar, t1.t * 0.8);
    }

    breakAfter();

    // Step 3: Covariance
    let t2 = afterTime(null, 2.5, 0.5);
    let t1c = afterTime(t1, 0.3);
    cleanup(t1c, [t1]);

    let covY = layout.vicregCov.y;
    moveBloqueaCameraTo(state, t2, new Vec3(0, 0, -covY + 15), new Vec3(290, 8, 2.5));

    commentary()`_COVARIANCE_ (decorrelation, weight=1): Penalizes off-diagonal covariance matrix elements. Forces different dimensions to encode different information — pitch, rhythm, timbre, dynamics each get dedicated capacity.`;
    breakAfter();

    if (t2.active) {
        let covPulse = 0.5 + 0.3 * Math.sin(wt.time * 2.5);
        layout.vicregCov.highlight = Math.max(layout.vicregCov.highlight, covPulse);
        drawDimLabels(layout.vicregCov, t2.t * 0.8);
    }

    breakAfter();

    // Step 4: All three together
    let t3 = afterTime(null, 3.0, 0.5);
    let t2c = afterTime(t2, 0.4);
    cleanup(t2c, [t2]);

    moveBloqueaCameraTo(state, t3, new Vec3(0, 0, -sharedY - 30), new Vec3(290, 15, 6));

    commentary()`L = 25 * L_inv + 25 * L_var + 1 * L_cov — three forces in equilibrium. The gradient from this loss flows back into 4 different parameter groups at 4 different learning rates: adapters learn fastest (5e-4), projections next (1e-4), MIDI encoder (5e-5), and unfrozen audio layers slowest (1e-5). Frozen layers receive zero gradient.`;
    breakAfter();

    if (t3.active) {
        let phase = wt.time * 2;
        layout.vicregInv.highlight = Math.max(layout.vicregInv.highlight, 0.5 + 0.4 * Math.sin(phase));
        layout.vicregVar.highlight = Math.max(layout.vicregVar.highlight, 0.5 + 0.4 * Math.sin(phase + 2.1));
        layout.vicregCov.highlight = Math.max(layout.vicregCov.highlight, 0.5 + 0.4 * Math.sin(phase + 4.2));
        let embGlow = 0.5 + 0.2 * Math.sin(wt.time * 1.5);
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, embGlow);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, embGlow);
        drawDimLabels(layout.vicregInv, 0.7);
        drawDimLabels(layout.vicregVar, 0.7);
        drawDimLabels(layout.vicregCov, 0.7);
    }

    breakAfter();
}
