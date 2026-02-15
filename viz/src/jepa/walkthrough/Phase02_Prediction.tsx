import { Vec3 } from "@/src/utils/vector";
import { IJepaWalkthroughArgs, setJepaInitialCamera } from "./JepaWalkthrough";
import { JepaDim } from "../JepaDimStyle";

export function jepaPhase02_Prediction(args: IJepaWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept } = tools;

    setJepaInitialCamera(state, new Vec3(0, 0, -700), new Vec3(290, 18, 16));

    let sgAudioRef = c_blockRef('Audio stop-gradient', layout.audio.stopGrad);
    let sgVibRef = c_blockRef('Vibration stop-gradient', layout.vibration.stopGrad);

    commentary()`After encoding, ${sgAudioRef} and ${sgVibRef} barriers block gradient flow from the prediction targets back to the encoders. This is critical: without stop-gradient, the model could collapse to a trivial solution where all representations are identical.`;
    breakAfter();

    let t0 = afterTime(null, 1.5, 0.5);
    if (t0.active) {
        dimExcept([layout.audio.z, layout.audio.stopGrad, layout.vibration.z, layout.vibration.stopGrad], 0.08);
        layout.audio.stopGrad.highlight = Math.max(layout.audio.stopGrad.highlight, t0.t * 0.8);
        layout.vibration.stopGrad.highlight = Math.max(layout.vibration.stopGrad.highlight, t0.t * 0.8);
    }
    breakAfter();

    let predAVRef = c_blockRef('Predictor a\u2192v', layout.predictorAtoV);
    let predVARef = c_blockRef('Predictor v\u2192a', layout.predictorVtoA);

    commentary()`Two MLP predictors work bidirectionally: ${predAVRef} takes z_audio and predicts z_vibration, while ${predVARef} does the reverse. Each is a 3-layer MLP (32\u219264\u219264\u219232) that learns the mapping between domains.`;
    breakAfter();

    let t1 = afterTime(null, 1.5, 0.5);
    if (t1.active) {
        dimExcept([layout.audio.stopGrad, layout.vibration.stopGrad, layout.predictorAtoV, layout.predictorVtoA], 0.08);
        layout.predictorAtoV.highlight = Math.max(layout.predictorAtoV.highlight, t1.t * 0.7);
        layout.predictorVtoA.highlight = Math.max(layout.predictorVtoA.highlight, t1.t * 0.7);
    }
    breakAfter();

    commentary()`The bidirectional design is important: it forces both encoders to capture shareable information. If only audio predicted vibration, the vibration encoder could produce arbitrary representations. By requiring both directions, both encoders must produce informative, predictable latents.`;
    breakAfter();

    let t2 = afterTime(null, 2.0, 0.5);
    if (t2.active) {
        layout.audio.z.highlight = Math.max(layout.audio.z.highlight, t2.t * 0.5);
        layout.vibration.z.highlight = Math.max(layout.vibration.z.highlight, t2.t * 0.5);
        layout.predictorAtoV.highlight = Math.max(layout.predictorAtoV.highlight, t2.t * 0.4);
        layout.predictorVtoA.highlight = Math.max(layout.predictorVtoA.highlight, t2.t * 0.4);
    }
    breakAfter();
}
