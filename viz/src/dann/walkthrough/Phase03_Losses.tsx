import { Vec3 } from "@/src/utils/vector";
import { IDannWalkthroughArgs, setDannInitialCamera } from "./DannWalkthrough";
import { DannDim } from "../DannDimStyle";

export function dannPhase03_Losses(args: IDannWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept } = tools;

    setDannInitialCamera(state, new Vec3(0, 0, -900), new Vec3(290, 15, 18));

    let vicRef = c_blockRef('VICReg loss', layout.vicregLoss);
    let dannRef = c_blockRef('DANN CE loss', layout.dannLoss);

    commentary()`DANN training has two loss terms that create a min-max game. The ${vicRef} aligns paired audio-MIDI embeddings — it pushes corresponding representations together while maintaining variance and reducing redundancy (covariance).`;
    breakAfter();

    let t0 = afterTime(null, 1.5, 0.5);
    if (t0.active) {
        dimExcept([layout.audioEmbedding, layout.midiEmbedding, layout.vicregLoss], 0.08);
        layout.vicregLoss.highlight = Math.max(layout.vicregLoss.highlight, t0.t * 0.8);
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, t0.t * 0.5);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, t0.t * 0.5);
    }
    breakAfter();

    commentary()`The ${dannRef} is a standard cross-entropy loss on the domain classifier's predictions. It measures how well the classifier distinguishes audio from MIDI. Lower DANN loss means better discrimination — but through gradient reversal, the encoders are pushed to maximize this loss.`;
    breakAfter();

    let t1 = afterTime(null, 1.5, 0.5);
    if (t1.active) {
        dimExcept([layout.grl, ...layout.classifierLayers, layout.dannLoss], 0.08);
        layout.dannLoss.highlight = Math.max(layout.dannLoss.highlight, t1.t * 0.8);
        for (let blk of layout.classifierLayers) {
            blk.highlight = Math.max(blk.highlight, t1.t * 0.5);
        }
        layout.grl.highlight = Math.max(layout.grl.highlight, t1.t * 0.4);
    }
    breakAfter();

    commentary()`The total loss combines both: L_total = L_vicreg + \u03B1 \u00D7 L_dann. The coefficient \u03B1 balances alignment (VICReg) against domain invariance (DANN). Too much DANN can destroy content information; too little leaves domain-specific artifacts in the embeddings.`;
    breakAfter();

    let t2 = afterTime(null, 2.0, 0.5);
    if (t2.active) {
        layout.vicregLoss.highlight = Math.max(layout.vicregLoss.highlight, t2.t * 0.6);
        layout.dannLoss.highlight = Math.max(layout.dannLoss.highlight, t2.t * 0.6);
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, t2.t * 0.3);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, t2.t * 0.3);
    }
    breakAfter();

    let grlRef = c_blockRef('GRL', layout.grl);

    commentary()`In the Phideus experiments (Gate 3), DANN was tested across 4 configurations. The result: DANN consistently degraded retrieval performance compared to VICReg alone. The ${grlRef} removed domain-specific information that was actually useful for cross-modal alignment — the "domain signal" contained content information needed for matching audio to its corresponding MIDI.`;
    breakAfter();

    let t3 = afterTime(null, 1.5, 0.5);
    if (t3.active) {
        for (let cube of layout.cubes) {
            cube.highlight = Math.max(cube.highlight, t3.t * 0.3);
        }
    }
    breakAfter();
}
