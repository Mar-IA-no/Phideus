import { Vec3 } from "@/src/utils/vector";
import { ICrossAttWalkthroughArgs, setCrossAttInitialCamera } from "./CrossAttWalkthrough";

export function crossAttPhase04_Transformers(args: ICrossAttWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept } = tools;

    setCrossAttInitialCamera(state, new Vec3(0, 0, -500), new Vec3(290, 22, 14));

    dimExcept([layout.residualNormAudio, layout.audioTransformer, layout.audioPool, layout.audioProj,
               layout.residualNormMidi, layout.midiTransformer, layout.midiPool, layout.midiProj]);

    let aTfRef = c_blockRef('Audio Transformer', layout.audioTransformer);
    let mTfRef = c_blockRef('MIDI Transformer', layout.midiTransformer);
    let aPoolRef = c_blockRef('Audio Mean Pool', layout.audioPool);
    let aProjRef = c_blockRef('Audio Projection', layout.audioProj);

    commentary()`After cross-attention + residual + LayerNorm, both paths enter their respective Transformers. The ${aTfRef} (4 layers, 8 heads, d=1024) and ${mTfRef} (4 layers, 8 heads, d=512) process the enriched features. PosEmbedding was already applied before cross-attention, so it is NOT applied again here.`;
    breakAfter();

    let t0 = afterTime(null, 1.5, 0.5);
    if (t0.active) {
        layout.audioTransformer.highlight = Math.max(layout.audioTransformer.highlight, t0.t * 0.6);
        layout.midiTransformer.highlight = Math.max(layout.midiTransformer.highlight, t0.t * 0.6);
    }
    breakAfter();

    commentary()`${aPoolRef} reduces the temporal dimension via mean pooling: [B, T, D] -> [B, D]. The ${aProjRef} maps from encoder dimension to the shared 256D embedding space (audio: 1024->512->256, MIDI: 512->512->256 via MLP). Both embeddings are now comparable in the shared VICReg space.`;
    breakAfter();

    let t1 = afterTime(null, 1.5, 0.5);
    if (t1.active) {
        layout.audioPool.highlight = Math.max(layout.audioPool.highlight, t1.t * 0.5);
        layout.audioProj.highlight = Math.max(layout.audioProj.highlight, t1.t * 0.6);
        layout.midiPool.highlight = Math.max(layout.midiPool.highlight, t1.t * 0.5);
        layout.midiProj.highlight = Math.max(layout.midiProj.highlight, t1.t * 0.6);
    }
    breakAfter();
}
