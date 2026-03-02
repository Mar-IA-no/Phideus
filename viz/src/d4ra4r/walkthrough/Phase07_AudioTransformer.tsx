import { Vec3 } from "@/src/utils/vector";
import { ID4RA4RWalkthroughArgs, setD4RA4RInitialCamera } from "./D4RA4RWalkthrough";
import { getTransformerLayerBlocks } from "../D4RA4RModelLayout";

export function d4ra4rPhase07_AudioTransformer(args: ID4RA4RWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept, drawDimLabels } = tools;

    let allTfBlocks = layout.audioTransformerLayers.flatMap(getTransformerLayerBlocks);

    setD4RA4RInitialCamera(state, new Vec3(-100, 0, -700), new Vec3(290, 20, 18));

    dimExcept([
        layout.audioReverseOutput,
        ...allTfBlocks,
        layout.audioOutputLN,
        ...layout.audioCnn,
    ], 0.08);
    for (let cnn of layout.audioCnn) cnn.opacity = 0.20;

    let tfRef = c_blockRef('Audio Transformer', layout.audioTransformerLayers[0].attnMatrix);
    let outRef = c_blockRef('188 tokens input', layout.audioReverseOutput);

    commentary()`These 4 Transformer layers process the output of A4 reverse cross-attention: only ${outRef}! Compare the NARROW transformer blocks (T_stft=188 width) with the WIDE CNN blocks above (T_audio=2400 width) — that's the visual consequence of 12.8x sequence compression. Same mechanism as a4r — the A4 spectral descriptor determines compression.`;
    breakAfter();

    let t0 = afterTime(null, 2.0, 0.5);
    if (t0.active) {
        for (let i = 0; i < layout.audioTransformerLayers.length; i++) {
            let l = layout.audioTransformerLayers[i];
            l.attnMatrix.highlight = Math.max(l.attnMatrix.highlight, t0.t * (0.3 + i * 0.15));
        }
        if (t0.t > 0.3) drawDimLabels(layout.audioTransformerLayers[0].attnMatrix, (t0.t - 0.3) * 1.4);
    }
    breakAfter();

    commentary()`Each layer has the full LN1 -> Q/K/V -> Attention -> Attn Out -> +Residual -> LN2 -> FFN Up (1024->4096) -> GELU -> FFN Down (4096->1024) -> +Residual structure. Self-attention cost per layer: 188^2 = 35K ops vs 2400^2 = 5.76M without compression — 163x cheaper.`;
    breakAfter();

    let t1 = afterTime(null, 2.0, 0.5);
    if (t1.active) {
        for (let l of layout.audioTransformerLayers) {
            l.mlpUp.highlight = Math.max(l.mlpUp.highlight, t1.t * 0.4);
            l.mlpDown.highlight = Math.max(l.mlpDown.highlight, t1.t * 0.4);
        }
    }
    breakAfter();
}
