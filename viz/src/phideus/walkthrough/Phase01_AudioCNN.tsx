import { Vec3 } from "@/src/utils/vector";
import { IPhideusWalkthroughArgs, setPhideusInitialCamera, movePhideusCameraTo } from "./PhideusWalkthrough";
import { PhideusDim } from "../PhideusDimStyle";

export function phideusPhase01_AudioCNN(args: IPhideusWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept, drawDimLabels } = tools;

    // Dim everything except audio input section
    dimExcept([
        layout.audioInput,
        ...layout.audioCnn,
        layout.audioPosEmb,
    ]);

    // Focus on input + CNN section
    setPhideusInitialCamera(state, new Vec3(-100, 0, -20), new Vec3(290, 15, 4));

    let inputRef = c_blockRef('audio input', layout.audioInput);
    let cnnRef = c_blockRef('CNN stages', layout.audioCnn);
    let frozen = c_dimRef('FROZEN', PhideusDim.Frozen);

    commentary()`The Audio Tower begins with MERT's CNN feature extractor — a multi-stage convolutional network that processes raw 24kHz audio waveforms into frame-level feature representations. Both the ${cnnRef} and positional embedding are ${frozen} from MERT's pre-training.`;
    breakAfter();

    // Step 1: Waveform input
    let tInput = afterTime(null, 1.0, 0.3);

    if (tInput.active) {
        layout.audioInput.highlight = Math.max(layout.audioInput.highlight, tInput.t * 0.5);
        layout.audioInput.opacity = 0.9;
        drawDimLabels(layout.audioInput, tInput.t * 0.8);
    }

    let tAudio = c_dimRef('T_audio', PhideusDim.T_audio);

    commentary()`Raw audio waveform at 24kHz sample rate enters as the ${inputRef}. A typical 10-second clip contains 240,000 samples. The CNN's job is to compress this into a manageable sequence of ${tAudio} feature frames.`;
    breakAfter();

    // Step 2: Progressive CNN stages
    let t0 = afterTime(null, 2.5, 0.5);
    let tInputC = afterTime(tInput, 0.3);
    cleanup(tInputC, [tInput]);

    let cnn0 = c_blockRef('Conv 0', layout.audioCnn[0]);
    let cnn3 = c_blockRef('Conv 3', layout.audioCnn[layout.audioCnn.length - 1]);
    let channels = c_dimRef('channels', PhideusDim.CNN_channels);

    commentary()`The CNN applies progressive downsampling across 4 stages: ${cnn0} (64ch) through to ${cnn3} (1024ch). Each stage reduces temporal resolution while increasing ${channels} depth. These weights are ${frozen} — they encode MERT's learned audio representations.`;
    breakAfter();

    if (t0.active) {
        let numStages = layout.audioCnn.length;
        let pos = t0.t * numStages;
        for (let i = 0; i < numStages; i++) {
            let blk = layout.audioCnn[i];
            blk.opacity = Math.max(blk.opacity, 0.6);
            if (i <= Math.floor(pos)) {
                let falloff = 1.0 - Math.max(0, pos - i - 1) / 3;
                blk.highlight = Math.max(blk.highlight, falloff * 0.6);
                if (i === Math.floor(pos) || i === numStages - 1) {
                    drawDimLabels(blk, falloff * 0.8);
                }
            }
        }
    }

    breakAfter();

    // Step 3: Positional embedding
    let t1 = afterTime(null, 1.5, 0.5);
    let t0c = afterTime(t0, 0.3);
    cleanup(t0c, [t0]);

    movePhideusCameraTo(state, t1, new Vec3(-100, 0, -180), new Vec3(290, 15, 4));

    let posEmb = c_blockRef('positional embeddings', layout.audioPosEmb);

    commentary()`Learned ${posEmb} are added to the CNN output, providing position information for the downstream transformer layers. These are also ${frozen} from MERT's pre-training — the model already knows how to encode temporal position in audio features.`;
    breakAfter();

    if (t1.active) {
        layout.audioPosEmb.opacity = Math.max(layout.audioPosEmb.opacity, 0.6);
        layout.audioPosEmb.highlight = Math.max(layout.audioPosEmb.highlight, t1.t * 0.5);
        drawDimLabels(layout.audioPosEmb, t1.t * 0.7);
    }

    breakAfter();
}
