import { Vec3 } from "@/src/utils/vector";
import { IT3TowerWalkthroughArgs, setT3TowerInitialCamera, moveT3TowerCameraTo } from "./T3TowerWalkthrough";
import { getTransformerLayerBlocks } from "../T3TowerModelLayout";
import { T3TowerDim } from "../T3TowerDimStyle";

export function t3TowerPhase02_AudioTransformer(args: IT3TowerWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept, drawDimLabels } = tools;

    let allTfBlocks = layout.audioTransformerLayers.flatMap(getTransformerLayerBlocks);
    dimExcept(allTfBlocks);

    let firstLayer = layout.audioTransformerLayers[0];
    let startY = firstLayer ? firstLayer.ln1.y : 0;
    setT3TowerInitialCamera(state, new Vec3(-140, 0, -startY - 80), new Vec3(290, 15, 6));

    let dAudio = c_dimRef('1024D', T3TowerDim.D_audio);
    let tAudio = c_dimRef('T_audio', T3TowerDim.T_audio);
    let dFfn = c_dimRef('4096D', T3TowerDim.D_ffn);
    let nHeads = c_dimRef('8 heads', T3TowerDim.N_heads);

    commentary()`Four transformer layers process the CNN features, each containing: LayerNorm, Q/K/V projections (${nHeads}), Attention Matrix (${tAudio} x ${tAudio}), Output Projection, Residual, FFN (${dAudio} to ${dFfn} to ${dAudio} with GELU), and Residual. Layers 0-1 at lr_low (3e-6), layers 2-3 at lr_high (1e-5).`;
    breakAfter();

    let prevStepTimers: any[] = [];

    for (let i = 0; i < layout.audioTransformerLayers.length; i++) {
        let layer = layout.audioTransformerLayers[i];
        let lrVal = i < 2 ? '3e-6' : '1e-5';

        if (i > 0) {
            for (let j = 0; j < layout.audioTransformerLayers.length; j++) {
                if (j !== i) {
                    for (let blk of getTransformerLayerBlocks(layout.audioTransformerLayers[j])) {
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
            moveT3TowerCameraTo(state, t0, new Vec3(-140, 0, -layer.ln1.y - 60), new Vec3(290, 15, 6));
        }

        let ln1Ref = c_blockRef('LayerNorm', layer.ln1);
        let trainable = c_dimRef(i < 2 ? 'lr_low' : 'lr_high', T3TowerDim.Trainable);
        commentary()`Layer ${i} (${trainable}, ${lrVal}): Pre-attention ${ln1Ref} stabilizes activations.`;

        if (t0.active) {
            layer.ln1.highlight = Math.max(layer.ln1.highlight, t0.t * 0.5);
            drawDimLabels(layer.ln1, t0.t * 0.6);
        }
        prevStepTimers.push(t0);

        breakAfter();

        let t1 = afterTime(null, 1.2, 0.3);

        let qRef = c_blockRef('Q', layer.qWeight);
        let kRef = c_blockRef('K', layer.kWeight);
        let vRef = c_blockRef('V', layer.vWeight);
        commentary()`${qRef}, ${kRef}, ${vRef} projections — ${nHeads}, each on a 128D subspace.`;

        if (t1.active) {
            layer.qWeight.highlight = Math.max(layer.qWeight.highlight, t1.t * 0.6);
            layer.kWeight.highlight = Math.max(layer.kWeight.highlight, t1.t * 0.6);
            layer.vWeight.highlight = Math.max(layer.vWeight.highlight, t1.t * 0.6);
            drawDimLabels(layer.qWeight, t1.t * 0.7);
        }
        prevStepTimers.push(t1);

        breakAfter();

        let t2 = afterTime(null, 1.2, 0.3);
        moveT3TowerCameraTo(state, t2, new Vec3(-140, 0, -layer.attnMatrix.y - 25), new Vec3(290, 12, 4));

        let attnRef = c_blockRef('Attention Matrix', layer.attnMatrix);
        commentary()`The ${attnRef} (${tAudio} x ${tAudio}) computes pairwise relationships between all audio frames.`;

        if (t2.active) {
            layer.attnMatrix.highlight = Math.max(layer.attnMatrix.highlight, t2.t * 0.7);
            layer.attnOut.highlight = Math.max(layer.attnOut.highlight, t2.t * 0.3);
            layer.attnResidual.highlight = Math.max(layer.attnResidual.highlight, t2.t * 0.2);
            drawDimLabels(layer.attnMatrix, t2.t * 0.8);
        }
        prevStepTimers.push(t2);

        breakAfter();

        let t3 = afterTime(null, 1.2, 0.3);
        moveT3TowerCameraTo(state, t3, new Vec3(-140, 0, -layer.mlpUp.y - 15), new Vec3(290, 12, 4));

        let mlpUpRef = c_blockRef('FFN expand', layer.mlpUp);
        let mlpDownRef = c_blockRef('FFN contract', layer.mlpDown);
        commentary()`${mlpUpRef} expands ${dAudio} to ${dFfn} (GELU), then ${mlpDownRef} contracts back. The 4x expansion gives capacity for complex non-linear transformations.`;

        if (t3.active) {
            layer.ln2.highlight = Math.max(layer.ln2.highlight, t3.t * 0.3);
            layer.mlpUp.highlight = Math.max(layer.mlpUp.highlight, t3.t * 0.6);
            layer.mlpAct.highlight = Math.max(layer.mlpAct.highlight, t3.t * 0.5);
            layer.mlpDown.highlight = Math.max(layer.mlpDown.highlight, t3.t * 0.6);
            layer.ffnResidual.highlight = Math.max(layer.ffnResidual.highlight, t3.t * 0.2);
            drawDimLabels(layer.mlpUp, t3.t * 0.7);
        }
        prevStepTimers.push(t3);

        breakAfter();
    }
}
