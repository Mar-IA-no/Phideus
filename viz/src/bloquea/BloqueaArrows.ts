import { addLine } from "@/src/llm/render/lineRender";
import { IRenderState } from "@/src/llm/render/modelRender";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { IBloqueaModelLayout, IBloqueaTransformerLayer } from "./BloqueaModelLayout";
import { IBlkDef } from "@/src/llm/GptModelLayout";

let flowColor = new Vec4(0.45, 0.45, 0.45, 0.6);
let convergenceColor = Vec4.fromHexColor('#8844bb');
let lossColor = Vec4.fromHexColor('#cc3333');
let attnColor = new Vec4(0.3, 0.4, 0.7, 0.5);       // blue-ish for Q/K/V -> Attn
let residualColor = new Vec4(0.6, 0.3, 0.1, 0.25);   // brown for residual bypass
let adapterResidualColor = new Vec4(0.7, 0.4, 0.0, 0.3); // orange-ish for adapter bypass

export function drawBloqueaArrows(render: IRenderState, layout: IBloqueaModelLayout) {

    // Audio tower: input -> CNN -> posEmb -> transformers -> pool -> proj
    drawVertArrow(render, layout.audioInput, layout.audioCnn[0]);
    for (let i = 0; i < layout.audioCnn.length - 1; i++) {
        drawVertArrow(render, layout.audioCnn[i], layout.audioCnn[i + 1]);
    }
    drawVertArrow(render, layout.audioCnn[layout.audioCnn.length - 1], layout.audioPosEmb);
    drawVertArrow(render, layout.audioPosEmb, layout.audioTransformerLayers[0].ln1);

    drawTransformerChain(render, layout.audioTransformerLayers);

    let lastAudioTf = layout.audioTransformerLayers[layout.audioTransformerLayers.length - 1];
    // If the last audio transformer layer has an adapter, connect from adapterUp; otherwise ffnResidual
    let lastAudioOut = lastAudioTf.adapter ? lastAudioTf.adapter.adapterUp : lastAudioTf.ffnResidual;
    drawVertArrow(render, lastAudioOut, layout.audioMeanPool);
    drawVertArrow(render, layout.audioMeanPool, layout.audioProjLayers[0]);
    for (let i = 0; i < layout.audioProjLayers.length - 1; i++) {
        drawVertArrow(render, layout.audioProjLayers[i], layout.audioProjLayers[i + 1]);
    }

    // MIDI tower: input -> embs -> combine -> posEnc -> transformers -> LN -> pool -> proj
    drawVertArrow(render, layout.midiInput, layout.midiPitchEmb);
    drawVertArrow(render, layout.midiInput, layout.midiVelEmb);
    drawVertArrow(render, layout.midiInput, layout.midiDurEmb);
    drawVertArrow(render, layout.midiPitchEmb, layout.midiCombineLinear);
    drawVertArrow(render, layout.midiVelEmb, layout.midiCombineLinear);
    drawVertArrow(render, layout.midiDurEmb, layout.midiCombineLinear);
    drawVertArrow(render, layout.midiCombineLinear, layout.midiPosEnc);
    drawVertArrow(render, layout.midiPosEnc, layout.midiTransformerLayers[0].ln1);

    drawTransformerChain(render, layout.midiTransformerLayers);

    let lastMidiTf = layout.midiTransformerLayers[layout.midiTransformerLayers.length - 1];
    drawVertArrow(render, lastMidiTf.ffnResidual, layout.midiOutputLN);
    drawVertArrow(render, layout.midiOutputLN, layout.midiMeanPool);
    drawVertArrow(render, layout.midiMeanPool, layout.midiProjLayers[0]);
    for (let i = 0; i < layout.midiProjLayers.length - 1; i++) {
        drawVertArrow(render, layout.midiProjLayers[i], layout.midiProjLayers[i + 1]);
    }

    // Convergence: proj -> shared space embeddings
    let lastAP = layout.audioProjLayers[layout.audioProjLayers.length - 1];
    let lastMP = layout.midiProjLayers[layout.midiProjLayers.length - 1];
    drawDiagonalArrow(render, lastAP, layout.audioEmbedding, convergenceColor, 2.0);
    drawDiagonalArrow(render, lastMP, layout.midiEmbedding, convergenceColor, 2.0);

    // Shared -> VICReg loss
    drawDiagonalArrow(render, layout.audioEmbedding, layout.vicregInv, lossColor, 1.5);
    drawDiagonalArrow(render, layout.midiEmbedding, layout.vicregInv, lossColor, 1.5);
    drawDiagonalArrow(render, layout.audioEmbedding, layout.vicregVar, lossColor, 1.5);
    drawDiagonalArrow(render, layout.midiEmbedding, layout.vicregVar, lossColor, 1.5);
    drawDiagonalArrow(render, layout.audioEmbedding, layout.vicregCov, lossColor, 1.5);
    drawDiagonalArrow(render, layout.midiEmbedding, layout.vicregCov, lossColor, 1.5);

    // Residual connections (two per layer -- one for attention, one for FFN)
    for (let layer of layout.audioTransformerLayers) {
        drawAttentionResidualBypass(render, layer);
        drawFFNResidualBypass(render, layer);
    }
    for (let layer of layout.midiTransformerLayers) {
        drawAttentionResidualBypass(render, layer);
        drawFFNResidualBypass(render, layer);
    }
}

function blkTopCenter(blk: IBlkDef): Vec3 {
    return new Vec3(blk.x + blk.dx / 2, blk.y, blk.z + blk.dz / 2);
}

function blkBotCenter(blk: IBlkDef): Vec3 {
    return new Vec3(blk.x + blk.dx / 2, blk.y + blk.dy, blk.z + blk.dz / 2);
}

function drawVertArrow(render: IRenderState, from: IBlkDef, to: IBlkDef) {
    let opacity = Math.min(from.opacity, to.opacity);
    let color = flowColor.mul(opacity);
    if (color.w < 0.03) return;

    let p0 = blkBotCenter(from);
    let p1 = blkTopCenter(to);
    addLine(render.lineRender, 1.0, color, p0, p1, undefined);

    let headLen = 2.0;
    let headW = 1.0;
    let dir = p1.sub(p0).normalize();
    let base = p1.sub(dir.mul(headLen));
    let perp = new Vec3(headW, 0, 0);
    addLine(render.lineRender, 1.0, color, base.add(perp), p1, undefined);
    addLine(render.lineRender, 1.0, color, base.sub(perp), p1, undefined);
}

function drawDiagonalArrow(render: IRenderState, from: IBlkDef, to: IBlkDef, color: Vec4, thickness: number) {
    let p0 = blkBotCenter(from);
    let p1 = blkTopCenter(to);
    addLine(render.lineRender, thickness, color, p0, p1, undefined);

    let headLen = 3.0;
    let headW = 1.5;
    let dir = p1.sub(p0).normalize();
    let base = p1.sub(dir.mul(headLen));
    let perp = new Vec3(dir.z, 0, -dir.x).mul(headW);
    addLine(render.lineRender, thickness, color, base.add(perp), p1, undefined);
    addLine(render.lineRender, thickness, color, base.sub(perp), p1, undefined);
}

function drawTransformerChain(render: IRenderState, layers: IBloqueaTransformerLayer[]) {
    for (let i = 0; i < layers.length; i++) {
        let l = layers[i];

        // ---- Self-Attention sub-block ----
        // LN1 -> Q, K, V (fan out from LN1 to each weight block)
        drawDiagonalArrow(render, l.ln1, l.qWeight, attnColor, 0.8);
        drawDiagonalArrow(render, l.ln1, l.kWeight, attnColor, 0.8);
        drawDiagonalArrow(render, l.ln1, l.vWeight, attnColor, 0.8);

        // Q, K -> Attention Matrix (scores = Q @ K^T)
        drawDiagonalArrow(render, l.qWeight, l.attnMatrix, attnColor, 0.8);
        drawDiagonalArrow(render, l.kWeight, l.attnMatrix, attnColor, 0.8);

        // V -> Attn Out (output = softmax(scores) @ V)
        // Draw V as a side arrow into attnOut, bypassing the attention matrix
        drawVToAttnOut(render, l.vWeight, l.attnOut, l.attnMatrix);

        // Attn Matrix -> Attn Out
        drawVertArrow(render, l.attnMatrix, l.attnOut);

        // Attn Out -> Residual Add
        drawVertArrow(render, l.attnOut, l.attnResidual);

        // ---- FFN sub-block ----
        // Attn Residual -> LN2
        drawVertArrow(render, l.attnResidual, l.ln2);

        // LN2 -> MLP Up (expand)
        drawVertArrow(render, l.ln2, l.mlpUp);

        // MLP Up -> GELU
        drawVertArrow(render, l.mlpUp, l.mlpAct);

        // GELU -> MLP Down (contract)
        drawVertArrow(render, l.mlpDown, l.ffnResidual);

        // MLP Act -> MLP Down
        drawVertArrow(render, l.mlpAct, l.mlpDown);

        // ---- Adapter bottleneck (layers 0-1 only) ----
        if (l.adapter) {
            let a = l.adapter;
            // Main flow: ffnResidual -> adapter down -> adapter act -> adapter up
            drawVertArrow(render, l.ffnResidual, a.adapterDown);
            drawVertArrow(render, a.adapterDown, a.adapterAct);
            drawVertArrow(render, a.adapterAct, a.adapterUp);

            // Adapter residual bypass: from ffnResidual to adapterUp (skips the bottleneck)
            drawAdapterResidualBypass(render, l.ffnResidual, a.adapterUp);

            // Between layers: adapterUp -> next layer's ln1
            if (i < layers.length - 1) {
                drawVertArrow(render, a.adapterUp, layers[i + 1].ln1);
            }
        } else {
            // Between layers: ffnResidual -> next layer's ln1
            if (i < layers.length - 1) {
                drawVertArrow(render, l.ffnResidual, layers[i + 1].ln1);
            }
        }
    }
}

// V bypasses the attention matrix and feeds directly into attnOut
// Draw as a side arrow: V bottom -> right/left elbow -> attnOut side
function drawVToAttnOut(render: IRenderState, vWeight: IBlkDef, attnOut: IBlkDef, attnMatrix: IBlkDef) {
    let opacity = Math.min(vWeight.opacity, attnOut.opacity);
    let color = attnColor.mul(opacity);
    if (color.w < 0.02) return;

    // V is on the right side of the QKV group
    let vBotX = vWeight.x + vWeight.dx;
    let vBotY = vWeight.y + vWeight.dy;
    let vZ = vWeight.z + vWeight.dz / 2;

    // Route to the right of the attention matrix, then into attnOut
    let xOff = Math.max(attnMatrix.dx / 2, vWeight.dx) + 5;
    let routeX = attnMatrix.x + attnMatrix.dx / 2 + xOff;

    let p0 = new Vec3(vBotX, vBotY, vZ);
    let p1 = new Vec3(routeX, vBotY, vZ);
    let p2 = new Vec3(routeX, attnOut.y + attnOut.dy / 2, vZ);
    let p3 = new Vec3(attnOut.x + attnOut.dx, attnOut.y + attnOut.dy / 2, vZ);

    addLine(render.lineRender, 0.8, color, p0, p1, undefined);
    addLine(render.lineRender, 0.8, color, p1, p2, undefined);
    addLine(render.lineRender, 0.8, color, p2, p3, undefined);

    // Small arrowhead pointing left into attnOut
    let headLen = 2.0;
    addLine(render.lineRender, 0.8, color, new Vec3(p3.x + headLen, p3.y - 1, p3.z), p3, undefined);
    addLine(render.lineRender, 0.8, color, new Vec3(p3.x + headLen, p3.y + 1, p3.z), p3, undefined);
}

// Attention residual bypass: from LN1 top (input) to attnResidual
function drawAttentionResidualBypass(render: IRenderState, layer: IBloqueaTransformerLayer) {
    let opacity = Math.min(layer.ln1.opacity, layer.attnResidual.opacity);
    let color = residualColor.mul(opacity);
    if (color.w < 0.02) return;

    let xOff = layer.ln1.dx / 2 + 5;
    let p0 = new Vec3(
        layer.ln1.x + layer.ln1.dx / 2 + xOff,
        layer.ln1.y,
        layer.ln1.z + layer.ln1.dz / 2
    );
    let p1 = new Vec3(
        layer.attnResidual.x + layer.attnResidual.dx / 2 + xOff,
        layer.attnResidual.y + layer.attnResidual.dy,
        layer.attnResidual.z + layer.attnResidual.dz / 2
    );

    addLine(render.lineRender, 0.8, color, p0, p1, undefined);

    // Horizontal ticks
    let tickLen = 3;
    addLine(render.lineRender, 0.8, color, new Vec3(p0.x - tickLen, p0.y, p0.z), p0, undefined);
    addLine(render.lineRender, 0.8, color, new Vec3(p1.x - tickLen, p1.y, p1.z), p1, undefined);
}

// FFN residual bypass: from attnResidual to ffnResidual
function drawFFNResidualBypass(render: IRenderState, layer: IBloqueaTransformerLayer) {
    let opacity = Math.min(layer.attnResidual.opacity, layer.ffnResidual.opacity);
    let color = residualColor.mul(opacity);
    if (color.w < 0.02) return;

    let xOff = layer.ln2.dx / 2 + 5;
    let p0 = new Vec3(
        layer.attnResidual.x + layer.attnResidual.dx / 2 + xOff,
        layer.attnResidual.y,
        layer.attnResidual.z + layer.attnResidual.dz / 2
    );
    let p1 = new Vec3(
        layer.ffnResidual.x + layer.ffnResidual.dx / 2 + xOff,
        layer.ffnResidual.y + layer.ffnResidual.dy,
        layer.ffnResidual.z + layer.ffnResidual.dz / 2
    );

    addLine(render.lineRender, 0.8, color, p0, p1, undefined);

    let tickLen = 3;
    addLine(render.lineRender, 0.8, color, new Vec3(p0.x - tickLen, p0.y, p0.z), p0, undefined);
    addLine(render.lineRender, 0.8, color, new Vec3(p1.x - tickLen, p1.y, p1.z), p1, undefined);
}

// Adapter residual bypass: from ffnResidual to adapterUp (skips the bottleneck)
// Same pattern as drawAttentionResidualBypass but with an orange-ish color
function drawAdapterResidualBypass(render: IRenderState, ffnResidual: IBlkDef, adapterUp: IBlkDef) {
    let opacity = Math.min(ffnResidual.opacity, adapterUp.opacity);
    let color = adapterResidualColor.mul(opacity);
    if (color.w < 0.02) return;

    let xOff = ffnResidual.dx / 2 + 8;
    let p0 = new Vec3(
        ffnResidual.x + ffnResidual.dx / 2 + xOff,
        ffnResidual.y,
        ffnResidual.z + ffnResidual.dz / 2
    );
    let p1 = new Vec3(
        adapterUp.x + adapterUp.dx / 2 + xOff,
        adapterUp.y + adapterUp.dy,
        adapterUp.z + adapterUp.dz / 2
    );

    addLine(render.lineRender, 0.8, color, p0, p1, undefined);

    // Horizontal ticks
    let tickLen = 3;
    addLine(render.lineRender, 0.8, color, new Vec3(p0.x - tickLen, p0.y, p0.z), p0, undefined);
    addLine(render.lineRender, 0.8, color, new Vec3(p1.x - tickLen, p1.y, p1.z), p1, undefined);
}
