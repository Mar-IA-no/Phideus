import { Vec3 } from "@/src/utils/vector";
import { ICrossAttWalkthroughArgs, setCrossAttInitialCamera } from "./CrossAttWalkthrough";

export function crossAttPhase01_AudioCNN(args: ICrossAttWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept } = tools;

    setCrossAttInitialCamera(state, new Vec3(-90, 0, -400), new Vec3(290, 20, 12));

    dimExcept([layout.waveformInput, layout.cnn, layout.posEmb]);

    let wavRef = c_blockRef('Waveform', layout.waveformInput);
    let cnnRef = c_blockRef('CNN', layout.cnn);
    let posRef = c_blockRef('PosEmbedding', layout.posEmb);

    commentary()`The audio path starts with a raw ${wavRef} at 24kHz (96,000 samples for 4 seconds). The ${cnnRef} (4 stages from MERT, frozen weights) downsamples to [B, 2400, 1024] — each of the 2400 frames represents ~1.67ms of audio with a 1024-dimensional feature vector.`;
    breakAfter();

    let t0 = afterTime(null, 1.5, 0.5);
    if (t0.active) {
        layout.waveformInput.highlight = Math.max(layout.waveformInput.highlight, t0.t * 0.5);
        layout.cnn.highlight = Math.max(layout.cnn.highlight, t0.t * 0.6);
    }
    breakAfter();

    commentary()`${posRef} is added BEFORE cross-attention (not after). This gives the cross-attention temporal awareness — without it, the attention pattern would be "temporally blind" and unable to learn position-dependent relationships between audio features and spectral descriptors.`;
    breakAfter();

    let t1 = afterTime(null, 1.5, 0.5);
    if (t1.active) {
        layout.posEmb.highlight = Math.max(layout.posEmb.highlight, t1.t * 0.7);
    }
    breakAfter();
}
