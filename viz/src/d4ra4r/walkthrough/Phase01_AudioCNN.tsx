import { Vec3 } from "@/src/utils/vector";
import { ID4RA4RWalkthroughArgs, setD4RA4RInitialCamera } from "./D4RA4RWalkthrough";

export function d4ra4rPhase01_AudioCNN(args: ID4RA4RWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept } = tools;

    setD4RA4RInitialCamera(state, new Vec3(-100, 0, -400), new Vec3(290, 20, 18));

    dimExcept([
        layout.waveformInput, ...layout.audioCnn, layout.audioPosEmb,
    ], 0.08);

    let waveRef = c_blockRef('Waveform', layout.waveformInput);
    let cnn3Ref = c_blockRef('Conv 3 (1024ch)', layout.audioCnn[3]);
    let posRef = c_blockRef('Learned PosEmb', layout.audioPosEmb);

    commentary()`The audio tower begins with frozen MERT CNN layers. ${waveRef} (24kHz, ~10 seconds) passes through 4 convolutional stages (64, 128, 256, 1024 channels). The final ${cnn3Ref} produces [B, 2400, 1024] — 2400 temporal frames at 1024 dimensions. All CNN weights are frozen (from pre-trained MERT).`;
    breakAfter();

    let t0 = afterTime(null, 2.0, 0.5);
    if (t0.active) {
        for (let cnn of layout.audioCnn) {
            cnn.highlight = Math.max(cnn.highlight, t0.t * 0.3);
        }
    }
    breakAfter();

    commentary()`${posRef} adds learned positional encoding to the CNN features. This is also frozen — it was learned during MERT pre-training. The CNN output + PosEmb gives us 2400 feature vectors that will feed into the reverse cross-attention mechanism (same as the a4r arm).`;
    breakAfter();

    let t1 = afterTime(null, 1.5, 0.5);
    if (t1.active) {
        layout.audioPosEmb.highlight = Math.max(layout.audioPosEmb.highlight, t1.t * 0.5);
    }
    breakAfter();
}
