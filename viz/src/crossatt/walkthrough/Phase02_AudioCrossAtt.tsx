import { Vec3 } from "@/src/utils/vector";
import { ICrossAttWalkthroughArgs, setCrossAttInitialCamera } from "./CrossAttWalkthrough";

export function crossAttPhase02_AudioCrossAtt(args: ICrossAttWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept } = tools;

    setCrossAttInitialCamera(state, new Vec3(-120, 0, -350), new Vec3(290, 18, 11));

    dimExcept([layout.posEmb, layout.crossAttAudio, layout.residualNormAudio,
               layout.audioDescriptor, layout.descKvProj]);

    let descRef = c_blockRef('Audio Descriptor', layout.audioDescriptor);
    let kvRef = c_blockRef('desc_kv_proj', layout.descKvProj);
    let xaRef = c_blockRef('Cross-Attention', layout.crossAttAudio);
    let lnRef = c_blockRef('Residual + LayerNorm', layout.residualNormAudio);

    commentary()`The ${descRef} is computed from the raw audio via STFT (n_fft=2048, hop=512) — no learned parameters, wrapped in torch.no_grad(). For A4 descriptors: 8 log-frequency delta bands. For A7: 12 just-intonation attractor activations. Both produce [B, 188, K] at native STFT resolution (no interpolation).`;
    breakAfter();

    let t0 = afterTime(null, 1.5, 0.5);
    if (t0.active) {
        layout.audioDescriptor.highlight = Math.max(layout.audioDescriptor.highlight, t0.t * 0.7);
        layout.descKvProj.highlight = Math.max(layout.descKvProj.highlight, t0.t * 0.5);
    }
    breakAfter();

    commentary()`The ${kvRef} projects from low-dim (8 or 12) to 1024D for Key/Value. The ${xaRef} (8 heads) has Query=features [B,2400,1024] attending to K/V [B,188,1024]. The attention matrix is [B*8, 2400, 188] — each CNN frame independently attends to all 188 STFT frames. This 12.8x memory savings vs interpolating K/V to 2400 was a key design insight.`;
    breakAfter();

    let t1 = afterTime(null, 2.0, 0.5);
    if (t1.active) {
        layout.crossAttAudio.highlight = Math.max(layout.crossAttAudio.highlight, t1.t * 0.8);
    }
    breakAfter();

    commentary()`The output goes through ${lnRef}: features = LayerNorm(features + attn_output). The residual connection ensures that if the descriptor provides no useful signal, the model can simply ignore it and pass through the original features unchanged. PosEmbedding is NOT applied again after cross-attention — it was already added before.`;
    breakAfter();

    let t2 = afterTime(null, 1.0, 0.5);
    if (t2.active) {
        layout.residualNormAudio.highlight = Math.max(layout.residualNormAudio.highlight, t2.t * 0.6);
    }
    breakAfter();
}
