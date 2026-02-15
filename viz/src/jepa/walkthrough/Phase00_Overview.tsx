import { Vec3 } from "@/src/utils/vector";
import { IJepaWalkthroughArgs, setJepaInitialCamera } from "./JepaWalkthrough";
import { JepaDim } from "../JepaDimStyle";
import { getEncoderBlocks } from "../JepaModelLayout";

export function jepaPhase00_Overview(args: IJepaWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, c_dimRef } = tools;

    setJepaInitialCamera(state, new Vec3(0, 0, -600), new Vec3(290, 22, 14));

    let audioRef = c_blockRef('Audio Encoder', [layout.audio.tokenMlp, layout.audio.lstm]);
    let vibRef = c_blockRef('Vibration Encoder', [layout.vibration.tokenMlp, layout.vibration.lstm]);
    let predRef = c_blockRef('Predictors', [layout.predictorAtoV, layout.predictorVtoA]);

    commentary()`JEPA-Lite (Joint Embedding Predictive Architecture) learns cross-modal representations without a decoder. The ${audioRef} and ${vibRef} produce latent representations, and ${predRef} learn to predict one domain's representation from the other.`;
    breakAfter();

    let t0 = afterTime(null, 1.5, 0.5);
    if (t0.active) {
        for (let blk of getEncoderBlocks(layout.audio)) blk.highlight = Math.max(blk.highlight, t0.t * 0.4);
        for (let blk of getEncoderBlocks(layout.vibration)) blk.highlight = Math.max(blk.highlight, t0.t * 0.4);
    }
    breakAfter();

    let sgRef = c_blockRef('Stop-gradient barriers', [layout.audio.stopGrad, layout.vibration.stopGrad]);
    let lossRef = c_blockRef('InfoNCE loss', layout.infoNCELoss);

    commentary()`The key insight: no decoder means no "reconstruction shortcut" — the model cannot simply memorize inputs. Instead, ${sgRef} prevent representation collapse, and ${lossRef} ensures the predictions are contrastive — distinguishing true pairs from random pairings.`;
    breakAfter();

    let t1 = afterTime(null, 1.5, 0.5);
    if (t1.active) {
        layout.audio.stopGrad.highlight = Math.max(layout.audio.stopGrad.highlight, t1.t * 0.7);
        layout.vibration.stopGrad.highlight = Math.max(layout.vibration.stopGrad.highlight, t1.t * 0.7);
        layout.infoNCELoss.highlight = Math.max(layout.infoNCELoss.highlight, t1.t * 0.5);
    }
    breakAfter();
}
