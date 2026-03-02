import { Vec3 } from "@/src/utils/vector";
import { IT3TowerWalkthroughArgs, setT3TowerInitialCamera, moveT3TowerCameraTo } from "./T3TowerWalkthrough";
import { getTransformerLayerBlocks } from "../T3TowerModelLayout";
import { T3TowerDim } from "../T3TowerDimStyle";

export function t3TowerPhase05_MIDITransformer(args: IT3TowerWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept, drawDimLabels } = tools;

    let allTfBlocks = layout.midiTransformerLayers.flatMap(getTransformerLayerBlocks);
    dimExcept(allTfBlocks);

    let firstLayer = layout.midiTransformerLayers[0];
    let startY = firstLayer ? firstLayer.ln1.y : 0;
    setT3TowerInitialCamera(state, new Vec3(40, 0, -startY - 80), new Vec3(288, 15, 6));

    let dMidi = c_dimRef('512D', T3TowerDim.D_midi);
    let tMidi = c_dimRef('T_midi', T3TowerDim.T_midi);
    let dFfn = c_dimRef('2048D', T3TowerDim.D_ffn);
    let nHeads = c_dimRef('8 heads', T3TowerDim.N_heads);

    commentary()`Four transformer layers process the MIDI embeddings. Each layer: LayerNorm, Q/K/V (${nHeads}), Attention (${tMidi} x ${tMidi}), Output + Residual, FFN (${dMidi}→${dFfn}→${dMidi} with GELU), Residual. All layers trained from scratch — no pre-trained MIDI model exists.`;
    breakAfter();

    let prevStepTimers: any[] = [];

    for (let i = 0; i < layout.midiTransformerLayers.length; i++) {
        let layer = layout.midiTransformerLayers[i];

        if (i > 0) {
            for (let j = 0; j < layout.midiTransformerLayers.length; j++) {
                if (j !== i) {
                    for (let blk of getTransformerLayerBlocks(layout.midiTransformerLayers[j])) {
                        blk.opacity = Math.min(blk.opacity, 0.25);
                    }
                }
            }
        }

        let t0 = afterTime(null, 1.0, 0.3);

        if (prevStepTimers.length > 0) {
            let cleanT = afterTime(prevStepTimers[prevStepTimers.length - 1], 0.3);
            cleanup(cleanT, prevStepTimers);
            prevStepTimers = [];
        }

        if (i > 0) {
            moveT3TowerCameraTo(state, t0, new Vec3(40, 0, -layer.ln1.y - 60), new Vec3(288, 15, 6));
        }

        let ln1Ref = c_blockRef('LayerNorm', layer.ln1);
        commentary()`MIDI Layer ${i}: Pre-attention ${ln1Ref} normalizes input activations.`;

        if (t0.active) {
            layer.ln1.highlight = Math.max(layer.ln1.highlight, t0.t * 0.5);
        }
        prevStepTimers.push(t0);

        breakAfter();

        let t1 = afterTime(null, 1.0, 0.3);

        let attnRef = c_blockRef('Attention', layer.attnMatrix);
        commentary()`${attnRef} (${tMidi} x ${tMidi}) learns pairwise event relationships — which notes attend to each other.`;

        if (t1.active) {
            layer.qWeight.highlight = Math.max(layer.qWeight.highlight, t1.t * 0.5);
            layer.kWeight.highlight = Math.max(layer.kWeight.highlight, t1.t * 0.5);
            layer.vWeight.highlight = Math.max(layer.vWeight.highlight, t1.t * 0.5);
            layer.attnMatrix.highlight = Math.max(layer.attnMatrix.highlight, t1.t * 0.7);
            drawDimLabels(layer.attnMatrix, t1.t * 0.8);
        }
        prevStepTimers.push(t1);

        breakAfter();

        let t2 = afterTime(null, 1.0, 0.3);
        moveT3TowerCameraTo(state, t2, new Vec3(40, 0, -layer.mlpUp.y - 15), new Vec3(288, 12, 4));

        let mlpRef = c_blockRef('FFN', layer.mlpUp);
        commentary()`${mlpRef} expands ${dMidi}→${dFfn} with GELU, then contracts back. 4x expansion provides capacity for non-linear feature refinement.`;

        if (t2.active) {
            layer.mlpUp.highlight = Math.max(layer.mlpUp.highlight, t2.t * 0.6);
            layer.mlpAct.highlight = Math.max(layer.mlpAct.highlight, t2.t * 0.5);
            layer.mlpDown.highlight = Math.max(layer.mlpDown.highlight, t2.t * 0.6);
            layer.ffnResidual.highlight = Math.max(layer.ffnResidual.highlight, t2.t * 0.2);
            drawDimLabels(layer.mlpUp, t2.t * 0.7);
        }
        prevStepTimers.push(t2);

        breakAfter();
    }
}
