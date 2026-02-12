import { Vec3 } from "@/src/utils/vector";
import { IPhideusWalkthroughArgs, setPhideusInitialCamera, movePhideusCameraTo } from "./PhideusWalkthrough";
import { getTransformerLayerBlocks } from "../PhideusModelLayout";
import { PhideusDim } from "../PhideusDimStyle";

export function phideusPhase04_MIDITransformer(args: IPhideusWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept, drawDimLabels } = tools;

    // Dim everything except MIDI transformer layers
    let allTfBlocks = layout.midiTransformerLayers.flatMap(getTransformerLayerBlocks);
    dimExcept(allTfBlocks);

    let firstLayer = layout.midiTransformerLayers[0];
    let startY = firstLayer ? firstLayer.ln1.y : 0;
    setPhideusInitialCamera(state, new Vec3(100, 0, -startY - 80), new Vec3(288, 15, 6));

    let dMidi = c_dimRef('512D', PhideusDim.D_midi);
    let tMidi = c_dimRef('T_midi', PhideusDim.T_midi);
    let dFfn = c_dimRef('2048D', PhideusDim.D_ffn);
    let nHeads = c_dimRef('8 heads', PhideusDim.N_heads);
    let lrMidi = c_dimRef('lr_midi (1e-4)', PhideusDim.Trainable);

    commentary()`Four transformer layers process the embedded MIDI events. Each layer has the same structure: LN, Q/K/V (${nHeads}), Attention (${tMidi} x ${tMidi}), FFN (${dMidi} to ${dFfn} to ${dMidi}). The entire MIDI encoder is trained from scratch at ${lrMidi}.`;
    breakAfter();

    let prevStepTimers: any[] = [];

    for (let i = 0; i < layout.midiTransformerLayers.length; i++) {
        let layer = layout.midiTransformerLayers[i];

        // Dim other layers, keep current layer bright
        if (i > 0) {
            for (let j = 0; j < layout.midiTransformerLayers.length; j++) {
                if (j !== i) {
                    for (let blk of getTransformerLayerBlocks(layout.midiTransformerLayers[j])) {
                        blk.opacity = Math.min(blk.opacity, 0.25);
                    }
                }
            }
        }

        // Step 1: LN1
        let t0 = afterTime(null, 1.0, 0.3);

        // Cleanup previous layer
        if (prevStepTimers.length > 0) {
            let cleanT = afterTime(prevStepTimers[prevStepTimers.length - 1], 0.3);
            cleanup(cleanT, prevStepTimers);
            prevStepTimers = [];
        }

        if (i > 0) {
            movePhideusCameraTo(state, t0, new Vec3(100, 0, -layer.ln1.y - 60), new Vec3(288, 15, 6));
        }

        let ln1Ref = c_blockRef('LayerNorm', layer.ln1);
        let layerDescs = [
            'begins learning MIDI event relationships from scratch.',
            'builds hierarchical representations — combining local note patterns with broader phrases.',
            'captures longer time spans — repeating motifs, call-and-response patterns, phrase-level structure.',
            'produces the richest MIDI representation for downstream projection.',
        ];
        commentary()`Layer ${i}: ${ln1Ref} stabilizes activations. This layer ${layerDescs[i]}`;

        if (t0.active) {
            layer.ln1.highlight = Math.max(layer.ln1.highlight, t0.t * 0.5);
            drawDimLabels(layer.ln1, t0.t * 0.6);
        }
        prevStepTimers.push(t0);

        breakAfter();

        // Step 2: Q/K/V
        let t1 = afterTime(null, 1.2, 0.3);

        let qRef = c_blockRef('Q', layer.qWeight);
        let kRef = c_blockRef('K', layer.kWeight);
        let vRef = c_blockRef('V', layer.vWeight);
        let qkvDescs = [
            'each head can specialize in different musical relationships — pitch intervals, rhythmic groupings.',
            'combines local note-to-note relationships with broader phrases.',
            'can now capture long-range musical dependencies and repeating patterns.',
            'produce the most abstract musical representations at the final layer.',
        ];
        commentary()`${qRef}, ${kRef}, ${vRef} projections — ${nHeads}, each on a 64D subspace of ${dMidi}. Trained from scratch, ${qkvDescs[i]}`;

        if (t1.active) {
            layer.qWeight.highlight = Math.max(layer.qWeight.highlight, t1.t * 0.6);
            layer.kWeight.highlight = Math.max(layer.kWeight.highlight, t1.t * 0.6);
            layer.vWeight.highlight = Math.max(layer.vWeight.highlight, t1.t * 0.6);
            drawDimLabels(layer.qWeight, t1.t * 0.7);
        }
        prevStepTimers.push(t1);

        breakAfter();

        // Step 3: Attention Matrix
        let t2 = afterTime(null, 1.2, 0.3);
        movePhideusCameraTo(state, t2, new Vec3(100, 0, -layer.attnMatrix.y - 25), new Vec3(288, 12, 4));

        let attnRef = c_blockRef('Attention Matrix', layer.attnMatrix);
        commentary()`The ${attnRef} (${tMidi} x ${tMidi}) captures pairwise relationships between MIDI events. ${i === 0 ? 'Early layers learn local patterns: adjacent notes, chord membership.' : i === 3 ? 'The final attention matrix captures global musical relationships.' : 'Increasingly abstract patterns emerge — pitch intervals, rhythmic groupings, harmonic progressions.'}`;

        if (t2.active) {
            layer.attnMatrix.highlight = Math.max(layer.attnMatrix.highlight, t2.t * 0.7);
            layer.attnOut.highlight = Math.max(layer.attnOut.highlight, t2.t * 0.3);
            drawDimLabels(layer.attnMatrix, t2.t * 0.8);
        }
        prevStepTimers.push(t2);

        breakAfter();

        // Step 4: FFN
        let t3 = afterTime(null, 1.2, 0.3);
        movePhideusCameraTo(state, t3, new Vec3(100, 0, -layer.mlpUp.y - 15), new Vec3(288, 12, 4));

        let mlpUpRef = c_blockRef('FFN expand', layer.mlpUp);
        let mlpDownRef = c_blockRef('FFN contract', layer.mlpDown);
        commentary()`${mlpUpRef} expands ${dMidi} to ${dFfn} (GELU), then ${mlpDownRef} contracts back. ${i === 3 ? 'Output FFN produces 512D features ready for mean pooling and projection to 256D.' : 'The 4x expansion provides capacity for complex transformations of the attended features.'}`;

        if (t3.active) {
            layer.mlpUp.highlight = Math.max(layer.mlpUp.highlight, t3.t * 0.6);
            layer.mlpAct.highlight = Math.max(layer.mlpAct.highlight, t3.t * 0.5);
            layer.mlpDown.highlight = Math.max(layer.mlpDown.highlight, t3.t * 0.6);
            drawDimLabels(layer.mlpUp, t3.t * 0.7);
            drawDimLabels(layer.mlpDown, t3.t * 0.7);
        }
        prevStepTimers.push(t3);

        breakAfter();
    }

    // Output LN + Mean Pool
    if (prevStepTimers.length > 0) {
        let cleanT = afterTime(prevStepTimers[prevStepTimers.length - 1], 0.3);
        cleanup(cleanT, prevStepTimers);
    }

    let tOutput = afterTime(null, 1.5, 0.5);
    let lnY = layout.midiOutputLN.y;
    movePhideusCameraTo(state, tOutput, new Vec3(100, 0, -lnY - 30), new Vec3(288, 15, 4));

    // Restore output blocks from dimming
    if (tOutput.active) {
        layout.midiOutputLN.opacity = 1.0;
        layout.midiMeanPool.opacity = 1.0;
    }

    let outputLN = c_blockRef('final LayerNorm', layout.midiOutputLN);
    let meanPool = c_blockRef('mean pooling', layout.midiMeanPool);

    commentary()`After the transformer stack, ${outputLN} normalizes the output features. ${meanPool} aggregates the variable-length sequence (${tMidi} events) into a single ${dMidi} vector — the MIDI tower's summary representation before projection.`;
    breakAfter();

    if (tOutput.active) {
        layout.midiOutputLN.highlight = Math.max(layout.midiOutputLN.highlight, tOutput.t * 0.5);
        layout.midiMeanPool.highlight = Math.max(layout.midiMeanPool.highlight, tOutput.t * 0.5);
        drawDimLabels(layout.midiMeanPool, tOutput.t * 0.6);
    }

    breakAfter();
}
