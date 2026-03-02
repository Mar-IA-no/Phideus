import { Vec3 } from "@/src/utils/vector";
import { ID4XA4XWalkthroughArgs, setD4XA4XInitialCamera } from "./D4XA4XWalkthrough";

export function d4xa4xPhase00_Overview(args: ID4XA4XWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef } = tools;

    setD4XA4XInitialCamera(state, new Vec3(0, 0, -1200), new Vec3(290, 22, 28));

    let fwdAudioRef = c_blockRef('Audio FORWARD Cross-Attention', layout.audioForwardMatrix);
    let fwdMidiRef = c_blockRef('MIDI FORWARD Cross-Attention', layout.midiForwardMatrix);
    let descRef = c_blockRef('Audio Descriptor A4', layout.audioDescOutput);
    let intRef = c_blockRef('MIDI Descriptor D4', layout.midiDescOutput);

    commentary()`This visualization shows the FORWARD cross-attention architecture (d4x-a4x). Two towers (Audio MERT + MIDI Transformer) are connected by descriptor-driven cross-attention. The key difference from reverse: here encoder tokens are Q (LARGE), and descriptors provide K/V (SMALL). The encoder "asks" the descriptor for guidance, preserving full temporal resolution.`;
    breakAfter();

    let t0 = afterTime(null, 2.0, 0.5);
    if (t0.active) {
        layout.audioForwardMatrix.highlight = Math.max(layout.audioForwardMatrix.highlight, t0.t * 0.7);
        layout.midiForwardMatrix.highlight = Math.max(layout.midiForwardMatrix.highlight, t0.t * 0.7);
        layout.audioDescOutput.highlight = Math.max(layout.audioDescOutput.highlight, t0.t * 0.4);
        layout.midiDescOutput.highlight = Math.max(layout.midiDescOutput.highlight, t0.t * 0.4);
    }
    breakAfter();

    commentary()`The ${descRef} computes spectral features from raw audio via STFT (NO_GRAD). The ${intRef} computes pitch interval features from MIDI events. Both feed into FORWARD cross-attention as Key/Value — the magenta-tinted blocks. Unlike reverse cross-attention (where descriptors are Q and compress the sequence), forward cross-attention keeps all 2400 encoder tokens, using descriptors to modulate each token's representation.`;
    breakAfter();
}
