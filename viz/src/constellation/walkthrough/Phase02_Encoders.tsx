import { Vec3 } from "@/src/utils/vector";
import { IConstellationWalkthroughArgs, setConstellationInitialCamera } from "./ConstellationWalkthrough";

export function constellationPhase02_Encoders(args: IConstellationWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept } = tools;

    setConstellationInitialCamera(state, new Vec3(-80, 0, -400), new Vec3(290, 20, 12));

    dimExcept([layout.audioInput, layout.audioMask, layout.audioTokenMlp, layout.audioAttnPool, layout.audioBiLstm,
               layout.vibInput, layout.vibMask, layout.vibTokenMlp, layout.vibAttnPool, layout.vibBiLstm]);

    let mlpRef = c_blockRef('Token MLP', layout.audioTokenMlp);
    let poolRef = c_blockRef('Attention Pool', layout.audioAttnPool);
    let lstmRef = c_blockRef('BiLSTM', layout.audioBiLstm);

    commentary()`Both encoder paths share the same architecture (but separate weights). The ${mlpRef} projects each 5D token to 128D hidden space. Then ${poolRef} reduces the K=48 token slots to a single vector per timestep — attention weights are learned, so the model focuses on the most informative tokens. Finally, ${lstmRef} captures temporal dependencies across frames.`;
    breakAfter();

    let t0 = afterTime(null, 1.5, 0.5);
    if (t0.active) {
        let p = t0.t;
        layout.audioTokenMlp.highlight = Math.max(layout.audioTokenMlp.highlight, p * 0.5);
        layout.vibTokenMlp.highlight = Math.max(layout.vibTokenMlp.highlight, p * 0.5);
    }
    breakAfter();

    let t1 = afterTime(null, 1.5, 0.5);
    if (t1.active) {
        let p = t1.t;
        layout.audioAttnPool.highlight = Math.max(layout.audioAttnPool.highlight, p * 0.6);
        layout.vibAttnPool.highlight = Math.max(layout.vibAttnPool.highlight, p * 0.6);
        layout.audioBiLstm.highlight = Math.max(layout.audioBiLstm.highlight, p * 0.4);
        layout.vibBiLstm.highlight = Math.max(layout.vibBiLstm.highlight, p * 0.4);
    }
    breakAfter();

    commentary()`An alternative encoder (C3/C4 configs) replaces MLP+AttnPool with a Transformer: token embeddings + CLS token + self-attention, where the CLS output serves as the pooled representation. Both encoder types feed into the same latent factorization downstream.`;
    breakAfter();
}
