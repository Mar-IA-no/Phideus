import { Vec3 } from "@/src/utils/vector";
import { IDannWalkthroughArgs, setDannInitialCamera } from "./DannWalkthrough";
import { DannDim } from "../DannDimStyle";

export function dannPhase02_GRL(args: IDannWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept, drawDimLabels } = tools;

    setDannInitialCamera(state, new Vec3(0, 0, -800), new Vec3(290, 18, 16));

    let grlRef = c_blockRef('Gradient Reversal Layer', layout.grl);
    let concatRef = c_blockRef('Concatenation', layout.concat);

    commentary()`After the embeddings are formed, the DANN path begins. The ${concatRef} block merges audio and MIDI embeddings into a single batch, followed by L2 normalization. Then comes the key innovation: the ${grlRef}.`;
    breakAfter();

    let t0 = afterTime(null, 1.5, 0.5);
    if (t0.active) {
        dimExcept([layout.audioEmbedding, layout.midiEmbedding, layout.concat, layout.l2norm, layout.grl], 0.08);
        layout.concat.highlight = Math.max(layout.concat.highlight, t0.t * 0.5);
        layout.l2norm.highlight = Math.max(layout.l2norm.highlight, t0.t * 0.4);
        layout.grl.highlight = Math.max(layout.grl.highlight, t0.t * 0.8);
    }
    breakAfter();

    commentary()`The GRL is mathematically elegant: during the forward pass, it acts as an identity function — data passes through unchanged. But during backpropagation, it multiplies all gradients by -\u03BB (negative lambda). This reverses the gradient direction for everything upstream.`;
    breakAfter();

    let t1 = afterTime(null, 1.5, 0.5);
    if (t1.active) {
        dimExcept([layout.grl], 0.06);
        layout.grl.highlight = Math.max(layout.grl.highlight, 0.5 + t1.t * 0.5);
        drawDimLabels(layout.grl, t1.t);
    }
    breakAfter();

    let clfRef = c_blockRef('Domain Classifier', layout.classifierLayers);

    commentary()`Below the GRL sits the ${clfRef}: three linear layers (256\u219264\u219264\u21922) that learn to predict whether an embedding came from audio or MIDI. The classifier gets normal gradients and learns to discriminate domains.`;
    breakAfter();

    let t2 = afterTime(null, 1.5, 0.5);
    if (t2.active) {
        dimExcept([layout.grl, ...layout.classifierLayers, layout.dannLoss], 0.08);
        for (let blk of layout.classifierLayers) {
            blk.highlight = Math.max(blk.highlight, t2.t * 0.6);
        }
    }
    breakAfter();

    commentary()`The adversarial trick: the classifier tries to distinguish audio from MIDI, but the reversed gradients push the encoders to make this impossible. Over training, the encoders learn to produce domain-invariant representations — embeddings where a classifier cannot tell which modality they came from.`;
    breakAfter();

    let t3 = afterTime(null, 2.0, 0.5);
    if (t3.active) {
        layout.grl.highlight = Math.max(layout.grl.highlight, 0.3 + t3.t * 0.5);
        layout.audioTransformer.highlight = Math.max(layout.audioTransformer.highlight, t3.t * 0.3);
        layout.midiTransformer.highlight = Math.max(layout.midiTransformer.highlight, t3.t * 0.3);
        for (let blk of layout.classifierLayers) {
            blk.highlight = Math.max(blk.highlight, t3.t * 0.4);
        }
    }
    breakAfter();

    commentary()`Lambda (\u03BB) controls the reversal strength and is typically scheduled: starting low and increasing during training. This prevents the adversarial signal from destabilizing early learning. In practice, \u03BB ramps from 0 to 1 following a sigmoid schedule over the first epochs.`;
    breakAfter();
}
