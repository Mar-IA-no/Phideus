import { Vec3 } from "@/src/utils/vector";
import { IDannWalkthroughArgs, setDannInitialCamera } from "./DannWalkthrough";
import { DannDim } from "../DannDimStyle";

export function dannPhase00_Overview(args: IDannWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, c_dimRef } = tools;

    setDannInitialCamera(state, new Vec3(0, 0, -600), new Vec3(290, 22, 14));

    let audioRef = c_blockRef('Audio Tower', [layout.audioCnn, layout.audioTransformer]);
    let midiRef = c_blockRef('MIDI Tower', [layout.midiEmb, layout.midiTransformer]);
    let grlRef = c_blockRef('Gradient Reversal Layer', layout.grl);
    let d256 = c_dimRef('256D', DannDim.D_proj);

    commentary()`Domain Adversarial Neural Network (DANN) extends the cross-modal architecture with adversarial training. The ${audioRef} and ${midiRef} produce ${d256} embeddings, which are aligned via VICReg. Simultaneously, a ${grlRef} feeds a domain classifier that learns to distinguish audio from MIDI — while gradient reversal forces the encoders to make embeddings domain-invariant.`;
    breakAfter();

    let t0 = afterTime(null, 1.5, 0.5);

    if (t0.active) {
        layout.grl.highlight = Math.max(layout.grl.highlight, t0.t * 0.6);
        layout.vicregLoss.highlight = Math.max(layout.vicregLoss.highlight, t0.t * 0.4);
        layout.dannLoss.highlight = Math.max(layout.dannLoss.highlight, t0.t * 0.4);
    }

    breakAfter();

    let vicRef = c_blockRef('VICReg loss', layout.vicregLoss);
    let dannRef = c_blockRef('DANN CE loss', layout.dannLoss);

    commentary()`The training objective has two competing terms: ${vicRef} aligns paired audio-MIDI embeddings, while ${dannRef} (through gradient reversal) pushes the encoder to produce representations where a classifier cannot tell which domain they came from. The idea is to remove domain-specific information while preserving content.`;
    breakAfter();

    let t1 = afterTime(null, 1.5, 0.5);
    if (t1.active) {
        for (let blk of layout.classifierLayers) {
            blk.highlight = Math.max(blk.highlight, t1.t * 0.3);
        }
    }
    breakAfter();
}
