import { Vec3 } from "@/src/utils/vector";
import { ID4RA4RWalkthroughArgs, setD4RA4RInitialCamera } from "./D4RA4RWalkthrough";
import { getTransformerLayerBlocks } from "../D4RA4RModelLayout";

export function d4ra4rPhase00_Overview(args: ID4RA4RWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, c_str } = tools;

    setD4RA4RInitialCamera(state, new Vec3(0, 0, -1200), new Vec3(290, 22, 28));

    let revAudioRef = c_blockRef('Audio REVERSE Cross-Attention', layout.audioReverseMatrix);
    let revMidiRef = c_blockRef('MIDI REVERSE Cross-Attention', layout.midiReverseMatrix);
    let descRef = c_blockRef('Audio Descriptor A4', layout.audioDescOutput);
    let intRef = c_blockRef('MIDI Descriptor D4', layout.midiDescOutput);

    commentary()`This is the d4r-a4r arm — the MIXED DESCRIPTOR strategy. Both towers use REVERSE cross-attention, but each tower uses a DIFFERENT descriptor type. Audio uses A4 (8-band spectral), MIDI uses D4 (4-dim local intervals). The key question: does using domain-specific descriptors on each modality outperform using the same descriptor type everywhere? S=79.8%.`;
    breakAfter();

    let t0 = afterTime(null, 2.0, 0.5);
    if (t0.active) {
        layout.audioReverseMatrix.highlight = Math.max(layout.audioReverseMatrix.highlight, t0.t * 0.7);
        layout.midiReverseMatrix.highlight = Math.max(layout.midiReverseMatrix.highlight, t0.t * 0.7);
        layout.audioDescOutput.highlight = Math.max(layout.audioDescOutput.highlight, t0.t * 0.5);
        layout.midiDescOutput.highlight = Math.max(layout.midiDescOutput.highlight, t0.t * 0.5);
    }
    breakAfter();

    commentary()`The ${descRef} (amber blocks, left) computes spectral features from raw audio via STFT — 8 octave bands, temporal deltas, z-score normalized. The ${intRef} (green blocks, right) computes pitch interval features from MIDI events — forward differences, semitone scaling. Both pipelines run under NO_GRAD. The dual-color scheme reflects the mixed descriptor strategy: amber for spectral (A4), green for structural (D4).`;
    breakAfter();
}
