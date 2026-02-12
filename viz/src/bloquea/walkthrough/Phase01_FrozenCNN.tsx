import { Vec3 } from "@/src/utils/vector";
import { IBloqueaWalkthroughArgs, setBloqueaInitialCamera, moveBloqueaCameraTo } from "./BloqueaWalkthrough";
import { BloqueaDim } from "../BloqueaDimStyle";

export function bloqueaPhase01_FrozenCNN(args: IBloqueaWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept, drawDimLabels } = tools;

    dimExcept([
        layout.audioInput,
        ...layout.audioCnn,
        layout.audioPosEmb,
    ]);

    setBloqueaInitialCamera(state, new Vec3(-100, 0, -20), new Vec3(290, 15, 4));

    let cnnRef = c_blockRef('CNN stages', layout.audioCnn);
    let frozen = c_dimRef('FROZEN', BloqueaDim.Frozen);

    commentary()`The Audio Tower begins with MERT's CNN feature extractor — identical to Run D. Both the ${cnnRef} and positional embedding are ${frozen} from MERT's pre-training. In Run C, this frozen base extends _deeper_ into the transformer: layers 0-1 are also frozen, with only adapter modules providing trainable capacity.`;
    breakAfter();

    // Step 1: Waveform input
    let tInput = afterTime(null, 1.0, 0.3);

    if (tInput.active) {
        layout.audioInput.highlight = Math.max(layout.audioInput.highlight, tInput.t * 0.5);
        layout.audioInput.opacity = 0.9;
        drawDimLabels(layout.audioInput, tInput.t * 0.8);
    }

    let tAudio = c_dimRef('T_audio', BloqueaDim.T_audio);

    commentary()`Raw audio waveform at 24kHz enters as the input. The CNN compresses this into a sequence of ${tAudio} feature frames through progressive downsampling.`;
    breakAfter();

    // Step 2: Progressive CNN stages
    let t0 = afterTime(null, 2.5, 0.5);
    let tInputC = afterTime(tInput, 0.3);
    cleanup(tInputC, [tInput]);

    let channels = c_dimRef('channels', BloqueaDim.CNN_channels);

    commentary()`Four convolutional stages progressively extract features: 64ch → 128ch → 256ch → 1024ch. Each stage reduces temporal resolution while increasing ${channels} depth. These weights are ${frozen} — MERT's learned audio representations are preserved exactly as pre-trained.`;
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

    moveBloqueaCameraTo(state, t1, new Vec3(-100, 0, -180), new Vec3(290, 15, 4));

    let posEmb = c_blockRef('positional embeddings', layout.audioPosEmb);

    commentary()`Learned ${posEmb} are added to the CNN output. Also ${frozen} from MERT. The frozen foundation in Run C is _deeper_ than in Run D — CNN + PosEmb + Layers 0-1 are all locked. The adapters provide the only trainable pathway through these frozen layers.`;
    breakAfter();

    if (t1.active) {
        layout.audioPosEmb.opacity = Math.max(layout.audioPosEmb.opacity, 0.6);
        layout.audioPosEmb.highlight = Math.max(layout.audioPosEmb.highlight, t1.t * 0.5);
        drawDimLabels(layout.audioPosEmb, t1.t * 0.7);
    }

    breakAfter();
}
