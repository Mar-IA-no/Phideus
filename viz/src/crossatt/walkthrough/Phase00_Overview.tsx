import { Vec3 } from "@/src/utils/vector";
import { ICrossAttWalkthroughArgs, setCrossAttInitialCamera } from "./CrossAttWalkthrough";

export function crossAttPhase00_Overview(args: ICrossAttWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef } = tools;

    setCrossAttInitialCamera(state, new Vec3(0, 0, -700), new Vec3(290, 22, 16));

    let audioXA = c_blockRef('Audio Cross-Attention', layout.crossAttAudio);
    let midiXA = c_blockRef('MIDI Cross-Attention', layout.crossAttMidi);
    let descRef = c_blockRef('Audio Descriptor', layout.audioDescriptor);
    let intRef = c_blockRef('MIDI Intervals', layout.midiIntervals);

    commentary()`Cross-attention descriptor injection (Gate 4.3) allows each encoder to selectively query ratio information. The ${audioXA} block lets audio features attend to spectral descriptors, while the ${midiXA} block lets MIDI embeddings attend to interval features. Unlike concatenation (which forces a fixed linear mix), cross-attention lets the model decide dynamically which ratio information is relevant for each frame.`;
    breakAfter();

    let t0 = afterTime(null, 2.0, 0.5);
    if (t0.active) {
        layout.crossAttAudio.highlight = Math.max(layout.crossAttAudio.highlight, t0.t * 0.7);
        layout.crossAttMidi.highlight = Math.max(layout.crossAttMidi.highlight, t0.t * 0.7);
        layout.audioDescriptor.highlight = Math.max(layout.audioDescriptor.highlight, t0.t * 0.4);
        layout.midiIntervals.highlight = Math.max(layout.midiIntervals.highlight, t0.t * 0.4);
    }
    breakAfter();

    commentary()`The ${descRef} computes spectral features from audio via STFT (no learned parameters, NO_GRAD). The ${intRef} computes pitch/velocity/duration deltas from MIDI events. Both are projected to their encoder's dimension via learned kv_proj layers before serving as Key/Value in the cross-attention. The audio side has a temporal mismatch (Q=2400 CNN frames vs K/V=188 STFT frames), which cross-attention resolves naturally.`;
    breakAfter();
}
