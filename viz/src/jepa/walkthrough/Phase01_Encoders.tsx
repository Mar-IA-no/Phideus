import { Vec3 } from "@/src/utils/vector";
import { IJepaWalkthroughArgs, setJepaInitialCamera } from "./JepaWalkthrough";
import { JepaDim } from "../JepaDimStyle";
import { getEncoderBlocks } from "../JepaModelLayout";

export function jepaPhase01_Encoders(args: IJepaWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept, drawDimLabels } = tools;

    setJepaInitialCamera(state, new Vec3(0, 0, -400), new Vec3(290, 28, 12));

    let tokenRef = c_blockRef('Token MLP', layout.audio.tokenMlp);
    let d5 = c_dimRef('5D', JepaDim.D_token);
    let d128 = c_dimRef('128D', JepaDim.D_hidden);

    commentary()`Each domain starts with sparse ratio tokens [T, 48, 5] — 48 constellation tokens per time frame, each with 5 features (log_ratio, delta_t, weight, anchor_band, target_band). The ${tokenRef} maps each ${d5} token to ${d128}.`;
    breakAfter();

    let t0 = afterTime(null, 1.5, 0.5);
    if (t0.active) {
        dimExcept(getEncoderBlocks(layout.audio), 0.08);
        layout.audio.input.highlight = Math.max(layout.audio.input.highlight, t0.t * 0.5);
        layout.audio.tokenMlp.highlight = Math.max(layout.audio.tokenMlp.highlight, t0.t * 0.7);
        drawDimLabels(layout.audio.tokenMlp, t0.t);
    }
    breakAfter();

    let attnRef = c_blockRef('Attention Pooling', layout.audio.attnPool);

    commentary()`${attnRef} reduces the 48 tokens per frame to a single vector using learned attention weights. This is a permutation-invariant operation — the model learns which tokens are most informative, regardless of their order in the input.`;
    breakAfter();

    let t1 = afterTime(null, 1.5, 0.5);
    if (t1.active) {
        dimExcept([layout.audio.tokenMlp, layout.audio.attnPool, layout.audio.lstm], 0.08);
        layout.audio.attnPool.highlight = Math.max(layout.audio.attnPool.highlight, t1.t * 0.7);
    }
    breakAfter();

    let lstmRef = c_blockRef('BiLSTM', layout.audio.lstm);
    let d32 = c_dimRef('32D', JepaDim.D_z);

    commentary()`The ${lstmRef} (2 layers, bidirectional) models temporal dependencies across frames. It produces ${d32} latent vectors z_audio[T,32] — a compact representation of the audio's ratio structure over time.`;
    breakAfter();

    let t2 = afterTime(null, 1.5, 0.5);
    if (t2.active) {
        dimExcept([layout.audio.lstm, layout.audio.z, layout.vibration.lstm, layout.vibration.z], 0.08);
        layout.audio.lstm.highlight = Math.max(layout.audio.lstm.highlight, t2.t * 0.6);
        layout.audio.z.highlight = Math.max(layout.audio.z.highlight, t2.t * 0.8);
        layout.vibration.lstm.highlight = Math.max(layout.vibration.lstm.highlight, t2.t * 0.6);
        layout.vibration.z.highlight = Math.max(layout.vibration.z.highlight, t2.t * 0.8);
    }
    breakAfter();

    commentary()`Both domains share the same architecture but have independent weights. The vibration encoder processes vibration tokens identically, producing z_vib[T,32]. The symmetry is intentional — it allows fair comparison of the two modalities' information content.`;
    breakAfter();
}
