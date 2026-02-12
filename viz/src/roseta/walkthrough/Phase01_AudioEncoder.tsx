import { Vec3 } from "@/src/utils/vector";
import { IRosetaWalkthroughArgs, setRosetaInitialCamera, moveRosetaCameraTo } from "./RosetaWalkthrough";
import { getEncoderBlocks } from "../RosetaModelLayout";
import { RosetaDim } from "../RosetaDimStyle";

export function rosetaPhase01_AudioEncoder(args: IRosetaWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept, drawDimLabels } = tools;

    dimExcept(getEncoderBlocks(layout.audio));

    setRosetaInitialCamera(state, new Vec3(-80, 0, -20), new Vec3(290, 15, 4));

    let inputRef = c_blockRef('audio input', layout.audio.input);
    let d768 = c_dimRef('768D', RosetaDim.D_input);
    let d128 = c_dimRef('128D', RosetaDim.D_hidden);

    commentary()`The Audio Encoder processes spectral features: each frame contains 256 frequency bins × 3 channels = ${d768} input dimensions. The input projection compresses this to ${d128}, creating a sequence ready for the BiLSTM.`;
    breakAfter();

    // Step 1: Input
    let t0 = afterTime(null, 1.5, 0.3);

    if (t0.active) {
        layout.audio.input.highlight = Math.max(layout.audio.input.highlight, t0.t * 0.5);
        layout.audio.input.opacity = 0.9;
        drawDimLabels(layout.audio.input, t0.t * 0.8);
    }

    commentary()`Audio spectral features [T, 256, 3] enter the encoder. Each time step contains a full frequency-domain snapshot of the audio signal — magnitude, phase, and spectral envelope.`;
    breakAfter();

    // Step 2: Input projection
    let t1 = afterTime(null, 2.0, 0.5);
    let t0c = afterTime(t0, 0.3);
    cleanup(t0c, [t0]);

    let projRef = c_blockRef('input projection', layout.audio.inputProj);

    commentary()`The ${projRef} (768→128) compresses each frame from ${d768} to ${d128}. This 6x compression forces the model to learn a compact spectral representation before temporal processing begins.`;
    breakAfter();

    if (t1.active) {
        let pulse = 0.4 + 0.2 * Math.sin(wt.time * 3);
        layout.audio.inputProj.highlight = Math.max(layout.audio.inputProj.highlight, pulse);
        drawDimLabels(layout.audio.inputProj, t1.t * 0.8);
    }

    breakAfter();

    // Step 3: BiLSTM Encoder
    let t2 = afterTime(null, 2.5, 0.5);
    let t1c = afterTime(t1, 0.3);
    cleanup(t1c, [t1]);

    moveRosetaCameraTo(state, t2, new Vec3(-80, 0, -layout.audio.lstmEncoder.y - 10), new Vec3(290, 12, 3.5));

    let lstmRef = c_blockRef('BiLSTM encoder', layout.audio.lstmEncoder);
    let d256 = c_dimRef('256D output', RosetaDim.D_lstm);

    commentary()`The ${lstmRef} — 2 bidirectional layers — processes the compressed sequence. Forward and backward passes capture both past and future context at each time step, producing a ${d256} (2×128) representation. LSTMs are chosen over transformers for their sequential inductive bias, important for temporal signal processing.`;
    breakAfter();

    if (t2.active) {
        let pulse = 0.5 + 0.25 * Math.sin(wt.time * 3);
        layout.audio.lstmEncoder.highlight = Math.max(layout.audio.lstmEncoder.highlight, pulse);
        drawDimLabels(layout.audio.lstmEncoder, t2.t * 0.9);
    }

    breakAfter();
}

