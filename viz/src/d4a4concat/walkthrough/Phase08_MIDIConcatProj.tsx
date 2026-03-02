import { Vec3 } from "@/src/utils/vector";
import { ID4A4WalkthroughArgs, setD4A4InitialCamera } from "./D4A4Walkthrough";

export function d4a4Phase08_MIDIConcatProj(args: ID4A4WalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept, drawDimLabels } = tools;

    setD4A4InitialCamera(state, new Vec3(100, 0, -700), new Vec3(290, 18, 20));

    dimExcept([
        layout.midiDescOutput,
        layout.midiConcatGate, layout.midiOutputLN, layout.midiMeanPool,
        layout.midiProjLayer1, layout.midiProjLayer2, layout.midiProjLayer3,
        layout.midiEmbedding,
    ]);

    let descRef = c_blockRef('D4 [N, 4]', layout.midiDescOutput);
    let concatRef = c_blockRef('Concat [516]', layout.midiConcatGate);
    let lnRef = c_blockRef('Output LN', layout.midiOutputLN);
    let poolRef = c_blockRef('Mean Pool', layout.midiMeanPool);
    let proj1Ref = c_blockRef('Proj 516->512', layout.midiProjLayer1);
    let proj3Ref = c_blockRef('Proj 512->256', layout.midiProjLayer3);
    let embRef = c_blockRef('MIDI z (256D)', layout.midiEmbedding);

    commentary()`Mirror of the audio side: the MIDI encoder output [B, N, 512] is concatenated with the D4 descriptor ${descRef}. The ${concatRef} block appends 4 interval dimensions: [B, N, 512] + [B, N, 4] = [B, N, 516]. Then ${lnRef} normalizes before ${poolRef} reduces over time.`;
    breakAfter();

    let t0 = afterTime(null, 2.0, 0.5);
    if (t0.active) {
        layout.midiDescOutput.highlight = Math.max(layout.midiDescOutput.highlight, t0.t * 0.6);
        layout.midiConcatGate.highlight = Math.max(layout.midiConcatGate.highlight, t0.t * 0.8);
        layout.midiOutputLN.highlight = Math.max(layout.midiOutputLN.highlight, t0.t * 0.4);
        layout.midiMeanPool.highlight = Math.max(layout.midiMeanPool.highlight, t0.t * 0.4);
        if (t0.t > 0.4) drawDimLabels(layout.midiConcatGate, (t0.t - 0.4) * 1.6);
    }
    breakAfter();

    commentary()`The MIDI projection head maps 516D down to 256D: ${proj1Ref} (BN+ReLU) -> 512->512 (BN+ReLU) -> ${proj3Ref}. All projection layers train at lr_proj (5e-4). The resulting ${embRef} meets Audio z in the shared 256D space — aligned by VICReg loss.`;
    breakAfter();

    let t1 = afterTime(null, 2.5, 0.5);
    if (t1.active) {
        let projBlocks = [layout.midiProjLayer1, layout.midiProjLayer2, layout.midiProjLayer3];
        let projDelay = [0.1, 0.3, 0.5];
        for (let i = 0; i < 3; i++) {
            let localT = Math.max(0, (t1.t - projDelay[i]) / (1 - projDelay[i]));
            projBlocks[i].highlight = Math.max(projBlocks[i].highlight, localT * 0.6);
        }
        if (t1.t > 0.7) layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, (t1.t - 0.7) * 3 * 0.7);
    }
    breakAfter();

    commentary()`Note the asymmetry: audio projection starts from 1032D (1024+8), MIDI from 516D (512+4). The descriptor contributes a small fraction of the input dimensions (0.8% for audio, 0.8% for MIDI) — but Test 01 shows A4 is completely causal for the +10pp improvement. Small signal, large impact.`;
    breakAfter();
}
