import { Vec3 } from "@/src/utils/vector";
import { ICrossAttWalkthroughArgs, setCrossAttInitialCamera } from "./CrossAttWalkthrough";
import { getTransformerLayerBlocks } from "../CrossAttModelLayout";

export function crossAttPhase00_Overview(args: ICrossAttWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, c_str } = tools;

    setCrossAttInitialCamera(state, new Vec3(0, 0, -1200), new Vec3(290, 22, 28));

    let revAudioRef = c_blockRef('Audio REVERSE Cross-Attention', layout.audioReverseMatrix);
    let revMidiRef = c_blockRef('MIDI REVERSE Cross-Attention', layout.midiReverseMatrix);
    let descRef = c_blockRef('Audio Descriptor A4', layout.audioDescOutput);
    let intRef = c_blockRef('MIDI Descriptor D4', layout.midiDescOutput);

    commentary()`This visualization shows the complete cross-attention architecture for Phideus (Gate 4.3). Two towers (Audio MERT + MIDI Transformer) are connected by descriptor-driven cross-attention. The key innovation: REVERSE cross-attention, where descriptors (Q) organize features (K/V) instead of the other way around. This achieves 82.0% retrieval accuracy vs 63.6% for regular cross-attention.`;
    breakAfter();

    let t0 = afterTime(null, 2.0, 0.5);
    if (t0.active) {
        layout.audioReverseMatrix.highlight = Math.max(layout.audioReverseMatrix.highlight, t0.t * 0.7);
        layout.midiReverseMatrix.highlight = Math.max(layout.midiReverseMatrix.highlight, t0.t * 0.7);
        layout.audioDescOutput.highlight = Math.max(layout.audioDescOutput.highlight, t0.t * 0.4);
        layout.midiDescOutput.highlight = Math.max(layout.midiDescOutput.highlight, t0.t * 0.4);
    }
    breakAfter();

    commentary()`The ${descRef} computes spectral features from raw audio via STFT (no learned parameters, NO_GRAD). The ${intRef} computes pitch interval features from MIDI events. Both descriptor pipelines are shown as amber-tinted blocks on the sides, while the bright cyan blocks in the center are the learnable REVERSE cross-attention mechanism. Dimmed gray blocks show the regular cross-attention for comparison.`;
    breakAfter();
}
