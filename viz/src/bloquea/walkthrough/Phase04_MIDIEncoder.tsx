import { Vec3 } from "@/src/utils/vector";
import { IBloqueaWalkthroughArgs, setBloqueaInitialCamera, moveBloqueaCameraTo } from "./BloqueaWalkthrough";
import { getTransformerLayerBlocks } from "../BloqueaModelLayout";
import { BloqueaDim } from "../BloqueaDimStyle";

export function bloqueaPhase04_MIDIEncoder(args: IBloqueaWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept, drawDimLabels } = tools;

    let midiBlocks = [
        layout.midiInput,
        layout.midiPitchEmb, layout.midiVelEmb, layout.midiDurEmb,
        layout.midiCombineLinear, layout.midiPosEnc,
        ...layout.midiTransformerLayers.flatMap(getTransformerLayerBlocks),
        layout.midiOutputLN,
    ];
    dimExcept(midiBlocks);

    setBloqueaInitialCamera(state, new Vec3(100, 0, -20), new Vec3(288, 18, 8));

    let dMidi = c_dimRef('512D', BloqueaDim.D_midi);
    let lrMidi = c_dimRef('lr_midi (5e-5)', BloqueaDim.Trainable);

    commentary()`The MIDI Tower is identical across Run C and Run D — trained entirely from scratch at ${lrMidi}. No pre-trained weights exist for MIDI-to-audio alignment. The encoder converts discrete musical events into ${dMidi} continuous representations through 4 transformer layers.`;
    breakAfter();

    // Step 1: Embedding tables
    let t0 = afterTime(null, 2.0, 0.5);

    let pitchEmb = c_blockRef('pitch', layout.midiPitchEmb);
    let velEmb = c_blockRef('velocity', layout.midiVelEmb);
    let durEmb = c_blockRef('duration', layout.midiDurEmb);

    commentary()`Three embedding tables — ${pitchEmb} (128 pitches → 256D), ${velEmb} (128 levels → 128D), and ${durEmb} (32 bins → 128D) — convert discrete MIDI attributes into continuous vectors. These are concatenated and projected to ${dMidi} through a linear layer with LayerNorm.`;
    breakAfter();

    if (t0.active) {
        let pulse = 0.4 + 0.2 * Math.sin(wt.time * 3);
        layout.midiPitchEmb.highlight = Math.max(layout.midiPitchEmb.highlight, pulse);
        layout.midiVelEmb.highlight = Math.max(layout.midiVelEmb.highlight, pulse * 0.9);
        layout.midiDurEmb.highlight = Math.max(layout.midiDurEmb.highlight, pulse * 0.8);
        layout.midiCombineLinear.highlight = Math.max(layout.midiCombineLinear.highlight, t0.t * 0.3);
        drawDimLabels(layout.midiPitchEmb, t0.t * 0.7);
    }

    breakAfter();

    // Step 2: Transformer layers cascade
    let t1 = afterTime(null, 3.0, 0.5);
    let t0c = afterTime(t0, 0.3);
    cleanup(t0c, [t0]);

    moveBloqueaCameraTo(state, t1, new Vec3(100, 0, -400), new Vec3(288, 18, 10));

    commentary()`Four transformer layers process the MIDI representations. All trained from scratch — every Q/K/V projection, attention pattern, and FFN weight must learn musical structure from random initialization. The sinusoidal positional encoding (frozen) provides temporal ordering information.`;
    breakAfter();

    if (t1.active) {
        let totalLayers = layout.midiTransformerLayers.length;
        for (let i = 0; i < totalLayers; i++) {
            let layerPhase = (t1.t * (totalLayers + 2)) - i;
            if (layerPhase > 0) {
                let intensity = Math.min(1, layerPhase);
                let pulse = intensity * (0.3 + 0.2 * Math.sin(wt.time * 3 + i * 0.8));
                let layer = layout.midiTransformerLayers[i];
                layer.attnMatrix.highlight = Math.max(layer.attnMatrix.highlight, pulse + 0.15);
                layer.mlpUp.highlight = Math.max(layer.mlpUp.highlight, pulse);
                layer.qWeight.highlight = Math.max(layer.qWeight.highlight, pulse * 0.8);
            }
        }
    }

    breakAfter();
}
