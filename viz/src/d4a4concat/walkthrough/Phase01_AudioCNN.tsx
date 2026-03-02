import { Vec3 } from "@/src/utils/vector";
import { ID4A4WalkthroughArgs, setD4A4InitialCamera } from "./D4A4Walkthrough";

export function d4a4Phase01_AudioCNN(args: ID4A4WalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept, drawDimLabels } = tools;

    setD4A4InitialCamera(state, new Vec3(-100, 0, -200), new Vec3(290, 18, 10));

    dimExcept([
        layout.waveformInput,
        ...layout.audioCnn,
        layout.audioPosEmb,
    ]);

    let wavRef = c_blockRef('24kHz waveform', layout.waveformInput);
    let conv0Ref = c_blockRef('Conv 0 (64ch)', layout.audioCnn[0]);
    let conv3Ref = c_blockRef('Conv 3 (1024ch)', layout.audioCnn[3]);
    let posRef = c_blockRef('Learned PosEmb', layout.audioPosEmb);

    commentary()`The audio tower begins with MERT's frozen CNN feature extractor. The ${wavRef} input (240,000 samples at 24kHz = 10 seconds) is progressively compressed through 4 convolutional stages: ${conv0Ref} → 128ch → 256ch → ${conv3Ref}. All CNN weights are frozen from pre-training.`;
    breakAfter();

    let t0 = afterTime(null, 2.5, 0.5);
    if (t0.active) {
        layout.waveformInput.highlight = Math.max(layout.waveformInput.highlight, t0.t * 0.4);
        for (let i = 0; i < layout.audioCnn.length; i++) {
            let delay = i * 0.15;
            let localT = Math.max(0, (t0.t - delay) / (1 - delay));
            layout.audioCnn[i].highlight = Math.max(layout.audioCnn[i].highlight, localT * 0.6);
        }
    }
    breakAfter();

    commentary()`Each convolutional stage doubles the channel count while maintaining temporal resolution. The final output is [B, 2400, 1024]: 2400 frames of 1024-dimensional features. The ${posRef} adds learned position information (also frozen). This entire CNN backbone is MERT pre-trained — we only fine-tune the transformer layers above.`;
    breakAfter();

    let t1 = afterTime(null, 2.0, 0.5);
    if (t1.active) {
        layout.audioCnn[3].highlight = Math.max(layout.audioCnn[3].highlight, t1.t * 0.7);
        layout.audioPosEmb.highlight = Math.max(layout.audioPosEmb.highlight, t1.t * 0.5);
        if (t1.t > 0.4) drawDimLabels(layout.audioCnn[3], (t1.t - 0.4) * 1.6);
    }
    breakAfter();
}
