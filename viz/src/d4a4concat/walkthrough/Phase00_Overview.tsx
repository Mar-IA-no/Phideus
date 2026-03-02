import { Vec3 } from "@/src/utils/vector";
import { ID4A4WalkthroughArgs, setD4A4InitialCamera } from "./D4A4Walkthrough";
import { getTransformerLayerBlocks } from "../D4A4ModelLayout";

export function d4a4Phase00_Overview(args: ID4A4WalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept } = tools;

    // Wide view showing both towers
    setD4A4InitialCamera(state, new Vec3(0, 0, -800), new Vec3(290, 22, 30));

    let allBlocks = layout.cubes;
    dimExcept(allBlocks, 0.6);

    let audioTowerRef = c_blockRef('Audio Tower', layout.audioCnn[0]);
    let midiTowerRef = c_blockRef('MIDI Tower', layout.midiInput);
    let a4Ref = c_blockRef('A4 descriptor', layout.audioDescOutput);
    let d4Ref = c_blockRef('D4 descriptor', layout.midiDescOutput);
    let concatAudioRef = c_blockRef('audio concat', layout.audioConcatGate);
    let concatMidiRef = c_blockRef('MIDI concat', layout.midiConcatGate);

    commentary()`The d4a4 architecture uses CONCAT DESCRIPTOR INJECTION: each encoder tower computes a descriptor (NO_GRAD), then concatenates it with encoder output before projection. The ${audioTowerRef} (left) processes 24kHz waveforms through frozen MERT CNN + 4-layer transformer. The ${midiTowerRef} (right) processes discrete MIDI events through embeddings + 4-layer transformer.`;
    breakAfter();

    let t0 = afterTime(null, 2.5, 0.5);
    if (t0.active) {
        for (let blk of layout.audioCnn) blk.highlight = Math.max(blk.highlight, t0.t * 0.3);
        for (let layer of layout.audioTransformerLayers) {
            for (let blk of getTransformerLayerBlocks(layer)) blk.highlight = Math.max(blk.highlight, t0.t * 0.2);
        }
        for (let layer of layout.midiTransformerLayers) {
            for (let blk of getTransformerLayerBlocks(layer)) blk.highlight = Math.max(blk.highlight, t0.t * 0.2);
        }
    }
    breakAfter();

    commentary()`The key innovation: ${a4Ref} computes 8-band octave spectral features (NO_GRAD DSP) and ${d4Ref} computes 4-dim local interval features. These descriptors are CONCATENATED with encoder output — ${concatAudioRef} produces [1024+8=1032]-dim vectors and ${concatMidiRef} produces [512+4=516]-dim vectors. This injects harmonic ratio information directly into the representation before projection.`;
    breakAfter();

    let t1 = afterTime(null, 2.0, 0.5);
    if (t1.active) {
        layout.audioDescOutput.highlight = Math.max(layout.audioDescOutput.highlight, t1.t * 0.7);
        layout.midiDescOutput.highlight = Math.max(layout.midiDescOutput.highlight, t1.t * 0.7);
        layout.audioConcatGate.highlight = Math.max(layout.audioConcatGate.highlight, t1.t * 0.8);
        layout.midiConcatGate.highlight = Math.max(layout.midiConcatGate.highlight, t1.t * 0.8);
    }
    breakAfter();

    let aEmbRef = c_blockRef('Audio z', layout.audioEmbedding);
    let mEmbRef = c_blockRef('MIDI z', layout.midiEmbedding);

    commentary()`Both towers converge into a shared 256D space: ${aEmbRef} and ${mEmbRef} are trained with VICReg loss (invariance + variance + covariance). Result: S=83.8% retrieval accuracy — the BEST performing arm in Gate 4.3.`;
    breakAfter();

    let t2 = afterTime(null, 2.0, 0.5);
    if (t2.active) {
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, t2.t * 0.7);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, t2.t * 0.7);
        layout.vicregInv.highlight = Math.max(layout.vicregInv.highlight, t2.t * 0.5);
        layout.vicregVar.highlight = Math.max(layout.vicregVar.highlight, t2.t * 0.5);
        layout.vicregCov.highlight = Math.max(layout.vicregCov.highlight, t2.t * 0.5);
    }
    breakAfter();
}
