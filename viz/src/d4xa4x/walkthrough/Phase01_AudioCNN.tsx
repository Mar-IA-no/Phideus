import { Vec3 } from "@/src/utils/vector";
import { ID4XA4XWalkthroughArgs, setD4XA4XInitialCamera } from "./D4XA4XWalkthrough";

export function d4xa4xPhase01_AudioCNN(args: ID4XA4XWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept } = tools;

    setD4XA4XInitialCamera(state, new Vec3(-100, 0, -400), new Vec3(290, 20, 14));

    dimExcept([layout.waveformInput, ...layout.audioCnn, layout.audioPosEmb]);

    let wavRef = c_blockRef('Waveform', layout.waveformInput);
    let cnnRef = c_blockRef('CNN', layout.audioCnn);
    let posRef = c_blockRef('PosEmbedding', layout.audioPosEmb);

    commentary()`The audio path starts with a raw ${wavRef} at 24kHz (96,000 samples for 4 seconds). The ${cnnRef} backbone (4 stages from MERT, frozen weights) progressively downsamples: 64ch -> 128ch -> 256ch -> 1024ch, producing [B, 2400, 1024] — each of the 2400 frames represents ~1.67ms of audio with a 1024-dimensional feature vector.`;
    breakAfter();

    let t0 = afterTime(null, 1.5, 0.5);
    if (t0.active) {
        layout.waveformInput.highlight = Math.max(layout.waveformInput.highlight, t0.t * 0.5);
        for (let i = 0; i < layout.audioCnn.length; i++) {
            layout.audioCnn[i].highlight = Math.max(layout.audioCnn[i].highlight, t0.t * (0.3 + i * 0.15));
        }
    }
    breakAfter();

    commentary()`${posRef} (learned, frozen) is added BEFORE cross-attention. In forward cross-attention, the encoder features with position information become Q — they actively query the descriptor. Position encoding ensures each of the 2400 tokens knows its temporal location when formulating queries.`;
    breakAfter();

    let t1 = afterTime(null, 1.5, 0.5);
    if (t1.active) {
        layout.audioPosEmb.highlight = Math.max(layout.audioPosEmb.highlight, t1.t * 0.7);
    }
    breakAfter();
}
