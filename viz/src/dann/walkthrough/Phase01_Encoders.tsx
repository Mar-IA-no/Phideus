import { Vec3 } from "@/src/utils/vector";
import { IDannWalkthroughArgs, setDannInitialCamera } from "./DannWalkthrough";
import { DannDim } from "../DannDimStyle";

export function dannPhase01_Encoders(args: IDannWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept, drawDimLabels } = tools;

    setDannInitialCamera(state, new Vec3(0, 0, -400), new Vec3(290, 30, 12));

    let audioTowerBlocks = [layout.audioInput, layout.audioCnn, layout.audioTransformer, layout.audioPool, layout.audioProj];
    let midiTowerBlocks = [layout.midiInput, layout.midiEmb, layout.midiTransformer, layout.midiPool, layout.midiProj];

    let cnnRef = c_blockRef('CNN (4 stages)', layout.audioCnn);
    let audioTfRef = c_blockRef('Audio Transformer', layout.audioTransformer);
    let d1024 = c_dimRef('1024D', DannDim.D_audio);

    commentary()`The audio encoder has two parts. First, ${cnnRef} converts a 4-second waveform (96,000 samples at 24kHz) into ${d1024} feature vectors. These CNN layers are frozen from pre-training — they extract spectral features without further adaptation.`;
    breakAfter();

    let t0 = afterTime(null, 1.5, 0.5);
    if (t0.active) {
        layout.audioCnn.highlight = Math.max(layout.audioCnn.highlight, t0.t * 0.6);
        dimExcept([...audioTowerBlocks, layout.audioEmbedding], 0.08);
        drawDimLabels(layout.audioCnn, t0.t);
    }
    breakAfter();

    commentary()`Then the ${audioTfRef} (4 layers, 8 heads) refines these features with self-attention, allowing each time frame to attend to all others. After mean pooling and projection, the audio side produces a 256D embedding.`;
    breakAfter();

    let t1 = afterTime(null, 1.5, 0.5);
    if (t1.active) {
        layout.audioTransformer.highlight = Math.max(layout.audioTransformer.highlight, t1.t * 0.7);
        layout.audioPool.highlight = Math.max(layout.audioPool.highlight, t1.t * 0.4);
        layout.audioProj.highlight = Math.max(layout.audioProj.highlight, t1.t * 0.5);
    }
    breakAfter();

    let embRef = c_blockRef('Embedding', layout.midiEmb);
    let midiTfRef = c_blockRef('MIDI Transformer', layout.midiTransformer);
    let d512 = c_dimRef('512D', DannDim.D_midi);

    commentary()`The MIDI encoder processes symbolic note events. The ${embRef} layer combines pitch, velocity, and duration into ${d512} vectors. The ${midiTfRef} (4 layers, 8 heads) then models relationships between notes. After pooling and projection, the MIDI side also produces a 256D embedding.`;
    breakAfter();

    let t2 = afterTime(null, 1.5, 0.5);
    if (t2.active) {
        dimExcept([...midiTowerBlocks, layout.midiEmbedding], 0.08);
        for (let blk of midiTowerBlocks) {
            blk.highlight = Math.max(blk.highlight, t2.t * 0.5);
        }
        drawDimLabels(layout.midiEmb, t2.t);
    }
    breakAfter();

    let d256 = c_dimRef('256D', DannDim.D_proj);
    let audioZRef = c_blockRef('Audio z', layout.audioEmbedding);
    let midiZRef = c_blockRef('MIDI z', layout.midiEmbedding);

    commentary()`Both towers converge to ${d256} embeddings in a shared space. The ${audioZRef} and ${midiZRef} vectors are where alignment happens — VICReg pushes paired embeddings together, while DANN (via gradient reversal) pushes them to be domain-indistinguishable.`;
    breakAfter();

    let t3 = afterTime(null, 1.5, 0.5);
    if (t3.active) {
        dimExcept([layout.audioEmbedding, layout.midiEmbedding, layout.vicregLoss], 0.12);
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, t3.t * 0.8);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, t3.t * 0.8);
    }
    breakAfter();
}
