import { Vec3 } from "@/src/utils/vector";
import { ID4A4WalkthroughArgs, setD4A4InitialCamera } from "./D4A4Walkthrough";

export function d4a4Phase04_AudioConcatProj(args: ID4A4WalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept, drawDimLabels } = tools;

    setD4A4InitialCamera(state, new Vec3(-100, 0, -700), new Vec3(290, 18, 20));

    dimExcept([
        layout.audioDescOutput,
        layout.audioConcatGate, layout.audioMeanPool,
        layout.audioProjLayer1, layout.audioProjLayer2, layout.audioProjLayer3,
        layout.audioEmbedding,
    ]);

    let descRef = c_blockRef('A4 [188, 8]', layout.audioDescOutput);
    let concatRef = c_blockRef('Concat [1032]', layout.audioConcatGate);
    let poolRef = c_blockRef('Mean Pool', layout.audioMeanPool);
    let proj1Ref = c_blockRef('Proj 1032->512', layout.audioProjLayer1);
    let proj3Ref = c_blockRef('Proj 512->256', layout.audioProjLayer3);
    let embRef = c_blockRef('Audio z (256D)', layout.audioEmbedding);

    commentary()`After the transformer, the encoder output [B, 2400, 1024] is mean-pooled over time. The A4 descriptor ${descRef} is also mean-pooled from [B, 188, 8] to [B, 8]. Then ${concatRef} concatenates: [B, 1024] + [B, 8] = [B, 1032]. The descriptor adds 8 dimensions of harmonic ratio information to each sample.`;
    breakAfter();

    let t0 = afterTime(null, 2.0, 0.5);
    if (t0.active) {
        layout.audioDescOutput.highlight = Math.max(layout.audioDescOutput.highlight, t0.t * 0.6);
        layout.audioConcatGate.highlight = Math.max(layout.audioConcatGate.highlight, t0.t * 0.8);
        if (t0.t > 0.4) drawDimLabels(layout.audioConcatGate, (t0.t - 0.4) * 1.6);
    }
    breakAfter();

    commentary()`The ${poolRef} reduces the temporal dimension. Then the projection head maps 1032D down: ${proj1Ref} (BN+ReLU) -> Proj 512->512 (BN+ReLU) -> ${proj3Ref}. These projections are trained at lr_proj (5e-4), 167x the base learning rate. The final ${embRef} lives in the shared 256D space.`;
    breakAfter();

    let t1 = afterTime(null, 2.5, 0.5);
    if (t1.active) {
        layout.audioMeanPool.highlight = Math.max(layout.audioMeanPool.highlight, t1.t * 0.4);
        let projDelay = [0.2, 0.4, 0.6];
        for (let i = 0; i < 3; i++) {
            let localT = Math.max(0, (t1.t - projDelay[i]) / (1 - projDelay[i]));
            [layout.audioProjLayer1, layout.audioProjLayer2, layout.audioProjLayer3][i].highlight =
                Math.max([layout.audioProjLayer1, layout.audioProjLayer2, layout.audioProjLayer3][i].highlight, localT * 0.6);
        }
        if (t1.t > 0.7) layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, (t1.t - 0.7) * 3 * 0.7);
    }
    breakAfter();

    commentary()`Concat injection is the simplest descriptor injection mechanism: just append the descriptor features. Despite this simplicity, d4a4 achieves S=83.8% — the best single-arm result. The projection head must learn to integrate both the 1024D encoder features and the 8D descriptor signal into a coherent 256D embedding.`;
    breakAfter();
}
