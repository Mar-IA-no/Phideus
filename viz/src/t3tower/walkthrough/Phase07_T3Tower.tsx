import { Vec3 } from "@/src/utils/vector";
import { IT3TowerWalkthroughArgs, setT3TowerInitialCamera, moveT3TowerCameraTo } from "./T3TowerWalkthrough";
import { getTransformerLayerBlocks } from "../T3TowerModelLayout";
import { T3TowerDim } from "../T3TowerDimStyle";

export function t3TowerPhase07_T3Tower(args: IT3TowerWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept, drawDimLabels } = tools;

    let t3Blocks = [
        layout.t3Input, layout.t3InputProj, layout.t3PosEnc,
        ...layout.t3TransformerLayers.flatMap(getTransformerLayerBlocks),
        layout.t3OutputLN, layout.t3MeanPool,
        ...layout.t3ProjLayers,
        layout.t3Embedding,
    ];
    dimExcept(t3Blocks);

    setT3TowerInitialCamera(state, new Vec3(200, 0, -20), new Vec3(286, 18, 6));

    let dT3 = c_dimRef('256D', T3TowerDim.D_t3);
    let tT3 = c_dimRef('T_t3', T3TowerDim.T_t3);
    let t3Label = c_dimRef('T3 Tower', T3TowerDim.T3Tower);
    let d256 = c_dimRef('256D', T3TowerDim.D_proj);

    commentary()`The ${t3Label} is the architectural novelty — a lightweight 2-layer transformer at ${dT3}, much smaller than Audio (1024D, 4L) or MIDI (512D, 4L). It processes an auxiliary third modality, adding a triangular constraint to the shared ${d256} space.`;
    breakAfter();

    // Step 1: Input + projection
    let t0 = afterTime(null, 2.0, 0.5);

    let inputRef = c_blockRef('T3 input', layout.t3Input);
    let projRef = c_blockRef('input projection', layout.t3InputProj);

    commentary()`The ${inputRef} enters the tower and is immediately mapped through an ${projRef} (input_dim→256) to match the tower's model dimension ${dT3}. This linear projection adapts any input dimensionality to the T3 format.`;
    breakAfter();

    if (t0.active) {
        layout.t3Input.highlight = Math.max(layout.t3Input.highlight, t0.t * 0.5);
        layout.t3Input.opacity = 0.9;
        layout.t3InputProj.highlight = Math.max(layout.t3InputProj.highlight, t0.t * 0.6);
        drawDimLabels(layout.t3InputProj, t0.t * 0.8);
    }

    breakAfter();

    // Step 2: Position encoding
    let t1 = afterTime(null, 1.5, 0.5);
    let t0c = afterTime(t0, 0.3);
    cleanup(t0c, [t0]);

    let posRef = c_blockRef('sinusoidal encoding', layout.t3PosEnc);
    let frozen = c_dimRef('FROZEN', T3TowerDim.Frozen);

    commentary()`${posRef} adds temporal position — parameter-free sin/cos functions that are ${frozen} by design. The T3 tower uses the same sinusoidal strategy as MIDI, providing a fixed positional reference without adding trainable parameters.`;
    breakAfter();

    if (t1.active) {
        layout.t3PosEnc.opacity = Math.max(layout.t3PosEnc.opacity, 0.6);
        layout.t3PosEnc.highlight = Math.max(layout.t3PosEnc.highlight, t1.t * 0.5);
        drawDimLabels(layout.t3PosEnc, t1.t * 0.7);
    }

    breakAfter();

    // Step 3: Transformer layers (2 layers, compact)
    let prevTimers: any[] = [];

    for (let i = 0; i < layout.t3TransformerLayers.length; i++) {
        let layer = layout.t3TransformerLayers[i];

        if (i > 0) {
            for (let j = 0; j < layout.t3TransformerLayers.length; j++) {
                if (j !== i) {
                    for (let blk of getTransformerLayerBlocks(layout.t3TransformerLayers[j])) {
                        blk.opacity = Math.min(blk.opacity, 0.25);
                    }
                }
            }
        }

        let tLN = afterTime(null, 1.0, 0.3);

        if (prevTimers.length > 0) {
            let cleanT = afterTime(prevTimers[prevTimers.length - 1], 0.3);
            cleanup(cleanT, prevTimers);
            prevTimers = [];
        }

        moveT3TowerCameraTo(state, tLN, new Vec3(200, 0, -layer.ln1.y - 50), new Vec3(286, 15, 5));

        let nHeads = c_dimRef('4 heads', T3TowerDim.N_heads);
        let dFfnT3 = c_dimRef('1024D', T3TowerDim.D_ffn_t3);

        commentary()`T3 Layer ${i}: ${nHeads} attention at ${dT3} (64D per head). The ${tT3} x ${tT3} attention matrix is compact — ${i === 0 ? 'the first layer captures local patterns in the auxiliary signal.' : 'the second layer integrates global context.'}`;
        breakAfter();

        if (tLN.active) {
            layer.ln1.highlight = Math.max(layer.ln1.highlight, tLN.t * 0.5);
            layer.qWeight.highlight = Math.max(layer.qWeight.highlight, tLN.t * 0.5);
            layer.kWeight.highlight = Math.max(layer.kWeight.highlight, tLN.t * 0.5);
            layer.vWeight.highlight = Math.max(layer.vWeight.highlight, tLN.t * 0.5);
            layer.attnMatrix.highlight = Math.max(layer.attnMatrix.highlight, tLN.t * 0.7);
            drawDimLabels(layer.attnMatrix, tLN.t * 0.8);
        }
        prevTimers.push(tLN);

        breakAfter();

        let tFFN = afterTime(null, 1.0, 0.3);
        moveT3TowerCameraTo(state, tFFN, new Vec3(200, 0, -layer.mlpUp.y - 15), new Vec3(286, 12, 4));

        commentary()`FFN expands ${dT3}→${dFfnT3} with GELU, contracts back. Same 4x expansion ratio as the larger towers.`;

        if (tFFN.active) {
            layer.mlpUp.highlight = Math.max(layer.mlpUp.highlight, tFFN.t * 0.6);
            layer.mlpAct.highlight = Math.max(layer.mlpAct.highlight, tFFN.t * 0.5);
            layer.mlpDown.highlight = Math.max(layer.mlpDown.highlight, tFFN.t * 0.6);
            layer.ffnResidual.highlight = Math.max(layer.ffnResidual.highlight, tFFN.t * 0.2);
            drawDimLabels(layer.mlpUp, tFFN.t * 0.7);
        }
        prevTimers.push(tFFN);

        breakAfter();
    }

    // Step 4: Output LN + Pool + Projection → embedding
    let tOut = afterTime(null, 2.5, 0.5);

    if (prevTimers.length > 0) {
        let cleanT = afterTime(prevTimers[prevTimers.length - 1], 0.3);
        cleanup(cleanT, prevTimers);
    }

    moveT3TowerCameraTo(state, tOut, new Vec3(200, 0, -layout.t3OutputLN.y - 60), new Vec3(286, 12, 4));

    let outputLN = c_blockRef('output LayerNorm', layout.t3OutputLN);
    let meanPool = c_blockRef('mean pooling', layout.t3MeanPool);
    let t3Proj = c_blockRef('T3 projection', layout.t3ProjLayers);

    commentary()`${outputLN} normalizes the final output, ${meanPool} collapses the sequence to a single vector, then ${t3Proj} maps 256→256→256→${d256} through a 3-layer MLP with BatchNorm + ReLU. The projection maps to the same shared space as the other two towers.`;
    breakAfter();

    if (tOut.active) {
        layout.t3OutputLN.highlight = Math.max(layout.t3OutputLN.highlight, tOut.t * 0.4);
        layout.t3MeanPool.highlight = Math.max(layout.t3MeanPool.highlight, tOut.t * 0.5);

        for (let i = 0; i < layout.t3ProjLayers.length; i++) {
            let layerPhase = (tOut.t * (layout.t3ProjLayers.length + 1)) - i;
            if (layerPhase > 0) {
                let intensity = Math.min(1, layerPhase);
                layout.t3ProjLayers[i].highlight = Math.max(layout.t3ProjLayers[i].highlight, intensity * 0.5);
            }
        }
    }

    breakAfter();

    // Step 5: T3 embedding arrives
    let tEmb = afterTime(null, 2.0, 0.5);
    let tOutC = afterTime(tOut, 0.3);
    cleanup(tOutC, [tOut]);

    moveT3TowerCameraTo(state, tEmb, new Vec3(200, 0, -layout.t3Embedding.y + 20), new Vec3(286, 10, 3));

    let t3Emb = c_blockRef('z_t3', layout.t3Embedding);

    commentary()`The result is ${t3Emb}: a ${d256} vector from the lightweight tower. Despite having only 2 transformer layers at 256D (vs Audio's 4L/1024D), the T3 embedding creates a triangular anchor — three pairwise VICReg constraints instead of one. This geometric redundancy strengthens the shared space.`;
    breakAfter();

    if (tEmb.active) {
        let embPulse = 0.5 + 0.3 * Math.sin(wt.time * 2.5);
        layout.t3Embedding.highlight = Math.max(layout.t3Embedding.highlight, embPulse);
        drawDimLabels(layout.t3Embedding, tEmb.t * 0.9);
    }

    breakAfter();
}
