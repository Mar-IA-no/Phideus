import { addLine } from "@/src/llm/render/lineRender";
import { IRenderState } from "@/src/llm/render/modelRender";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { IT3TowerModelLayout, IT3TowerTransformerLayer } from "./T3TowerModelLayout";
import { IBlkDef } from "@/src/llm/GptModelLayout";

let flowColor = new Vec4(0.55, 0.55, 0.55, 0.75);
let convergenceColor = Vec4.fromHexColor('#8844bb');
let lossColor = Vec4.fromHexColor('#cc3333');
let attnColor = new Vec4(0.3, 0.4, 0.7, 0.5);
let residualColor = new Vec4(0.6, 0.3, 0.1, 0.35);
let t3Color = Vec4.fromHexColor('#2299aa');

export function drawT3TowerArrows(render: IRenderState, layout: IT3TowerModelLayout) {

    // Audio tower: input → CNN → posEmb → transformers → pool → proj
    drawVertArrow(render, layout.audioInput, layout.audioCnn[0]);
    for (let i = 0; i < layout.audioCnn.length - 1; i++) {
        drawVertArrow(render, layout.audioCnn[i], layout.audioCnn[i + 1]);
    }
    drawVertArrow(render, layout.audioCnn[layout.audioCnn.length - 1], layout.audioPosEmb);
    drawVertArrow(render, layout.audioPosEmb, layout.audioTransformerLayers[0].ln1);

    drawTransformerChain(render, layout.audioTransformerLayers);

    let lastAudioTf = layout.audioTransformerLayers[layout.audioTransformerLayers.length - 1];
    drawVertArrow(render, lastAudioTf.ffnResidual, layout.audioMeanPool);
    drawVertArrow(render, layout.audioMeanPool, layout.audioProjLayers[0]);
    for (let i = 0; i < layout.audioProjLayers.length - 1; i++) {
        drawVertArrow(render, layout.audioProjLayers[i], layout.audioProjLayers[i + 1]);
    }

    // MIDI tower: input → embs → combine → posEnc → transformers → LN → pool → proj
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

    // T3 tower: input → proj → posEnc → transformers → LN → pool → proj
    drawVertArrow(render, layout.t3Input, layout.t3InputProj);
    drawVertArrow(render, layout.t3InputProj, layout.t3PosEnc);
    drawVertArrow(render, layout.t3PosEnc, layout.t3TransformerLayers[0].ln1);

    drawTransformerChain(render, layout.t3TransformerLayers);

    let lastT3Tf = layout.t3TransformerLayers[layout.t3TransformerLayers.length - 1];
    drawVertArrow(render, lastT3Tf.ffnResidual, layout.t3OutputLN);
    drawVertArrow(render, layout.t3OutputLN, layout.t3MeanPool);
    drawVertArrow(render, layout.t3MeanPool, layout.t3ProjLayers[0]);
    for (let i = 0; i < layout.t3ProjLayers.length - 1; i++) {
        drawVertArrow(render, layout.t3ProjLayers[i], layout.t3ProjLayers[i + 1]);
    }

    // Convergence: proj → shared space embeddings
    let lastAP = layout.audioProjLayers[layout.audioProjLayers.length - 1];
    let lastMP = layout.midiProjLayers[layout.midiProjLayers.length - 1];
    let lastTP = layout.t3ProjLayers[layout.t3ProjLayers.length - 1];
    drawDiagonalArrow(render, lastAP, layout.audioEmbedding, convergenceColor, 2.0);
    drawDiagonalArrow(render, lastMP, layout.midiEmbedding, convergenceColor, 2.0);
    drawDiagonalArrow(render, lastTP, layout.t3Embedding, t3Color.mul(0.9), 2.0);

    // Shared → 3-way VICReg loss
    drawDiagonalArrow(render, layout.audioEmbedding, layout.vicregAM, lossColor, 1.5);
    drawDiagonalArrow(render, layout.midiEmbedding, layout.vicregAM, lossColor, 1.5);
    drawDiagonalArrow(render, layout.audioEmbedding, layout.vicregAT, lossColor, 1.5);
    drawDiagonalArrow(render, layout.t3Embedding, layout.vicregAT, lossColor, 1.5);
    drawDiagonalArrow(render, layout.midiEmbedding, layout.vicregMT, lossColor, 1.5);
    drawDiagonalArrow(render, layout.t3Embedding, layout.vicregMT, lossColor, 1.5);

    // Residual connections
    for (let layer of layout.audioTransformerLayers) {
        drawAttentionResidualBypass(render, layer);
        drawFFNResidualBypass(render, layer);
    }
    for (let layer of layout.midiTransformerLayers) {
        drawAttentionResidualBypass(render, layer);
        drawFFNResidualBypass(render, layer);
    }
    for (let layer of layout.t3TransformerLayers) {
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
    addLine(render.lineRender, 1.2, color, p0, p1, undefined);

    let headLen = 2.5;
    let headW = 1.3;
    let dir = p1.sub(p0).normalize();
    let base = p1.sub(dir.mul(headLen));
    let perp = new Vec3(headW, 0, 0);
    addLine(render.lineRender, 1.2, color, base.add(perp), p1, undefined);
    addLine(render.lineRender, 1.2, color, base.sub(perp), p1, undefined);
}

function drawDiagonalArrow(render: IRenderState, from: IBlkDef, to: IBlkDef, color: Vec4, thickness: number) {
    let opacity = Math.min(from.opacity, to.opacity);
    let c = color.mul(opacity);
    if (c.w < 0.02) return;
    let p0 = blkBotCenter(from);
    let p1 = blkTopCenter(to);
    addLine(render.lineRender, thickness, c, p0, p1, undefined);

    let headLen = 3.0;
    let headW = 1.5;
    let dir = p1.sub(p0).normalize();
    let base = p1.sub(dir.mul(headLen));
    let perp = new Vec3(dir.z, 0, -dir.x).mul(headW);
    addLine(render.lineRender, thickness, c, base.add(perp), p1, undefined);
    addLine(render.lineRender, thickness, c, base.sub(perp), p1, undefined);
}

function drawTransformerChain(render: IRenderState, layers: IT3TowerTransformerLayer[]) {
    for (let i = 0; i < layers.length; i++) {
        let l = layers[i];

        drawDiagonalArrow(render, l.ln1, l.qWeight, attnColor, 0.8);
        drawDiagonalArrow(render, l.ln1, l.kWeight, attnColor, 0.8);
        drawDiagonalArrow(render, l.ln1, l.vWeight, attnColor, 0.8);

        drawDiagonalArrow(render, l.qWeight, l.attnMatrix, attnColor, 0.8);
        drawDiagonalArrow(render, l.kWeight, l.attnMatrix, attnColor, 0.8);

        drawVToAttnOut(render, l.vWeight, l.attnOut, l.attnMatrix);

        drawVertArrow(render, l.attnMatrix, l.attnOut);
        drawVertArrow(render, l.attnOut, l.attnResidual);
        drawVertArrow(render, l.attnResidual, l.ln2);
        drawVertArrow(render, l.ln2, l.mlpUp);
        drawVertArrow(render, l.mlpUp, l.mlpAct);
        drawVertArrow(render, l.mlpAct, l.mlpDown);
        drawVertArrow(render, l.mlpDown, l.ffnResidual);

        if (i < layers.length - 1) {
            drawVertArrow(render, l.ffnResidual, layers[i + 1].ln1);
        }
    }
}

function drawVToAttnOut(render: IRenderState, vWeight: IBlkDef, attnOut: IBlkDef, attnMatrix: IBlkDef) {
    let opacity = Math.min(vWeight.opacity, attnOut.opacity);
    let color = attnColor.mul(opacity);
    if (color.w < 0.02) return;

    let vBotX = vWeight.x + vWeight.dx;
    let vBotY = vWeight.y + vWeight.dy;
    let vZ = vWeight.z + vWeight.dz / 2;

    let xOff = Math.max(attnMatrix.dx / 2, vWeight.dx) + 5;
    let routeX = attnMatrix.x + attnMatrix.dx / 2 + xOff;

    let p0 = new Vec3(vBotX, vBotY, vZ);
    let p1 = new Vec3(routeX, vBotY, vZ);
    let p2 = new Vec3(routeX, attnOut.y + attnOut.dy / 2, vZ);
    let p3 = new Vec3(attnOut.x + attnOut.dx, attnOut.y + attnOut.dy / 2, vZ);

    addLine(render.lineRender, 0.8, color, p0, p1, undefined);
    addLine(render.lineRender, 0.8, color, p1, p2, undefined);
    addLine(render.lineRender, 0.8, color, p2, p3, undefined);

    let headLen = 2.0;
    addLine(render.lineRender, 0.8, color, new Vec3(p3.x + headLen, p3.y - 1, p3.z), p3, undefined);
    addLine(render.lineRender, 0.8, color, new Vec3(p3.x + headLen, p3.y + 1, p3.z), p3, undefined);
}

function drawAttentionResidualBypass(render: IRenderState, layer: IT3TowerTransformerLayer) {
    let opacity = Math.min(layer.ln1.opacity, layer.attnResidual.opacity);
    let color = residualColor.mul(opacity);
    if (color.w < 0.02) return;

    let xOff = layer.ln1.dx / 2 + 5;
    let p0 = new Vec3(layer.ln1.x + layer.ln1.dx / 2 + xOff, layer.ln1.y, layer.ln1.z + layer.ln1.dz / 2);
    let p1 = new Vec3(layer.attnResidual.x + layer.attnResidual.dx / 2 + xOff, layer.attnResidual.y + layer.attnResidual.dy, layer.attnResidual.z + layer.attnResidual.dz / 2);

    addLine(render.lineRender, 0.8, color, p0, p1, undefined);
    let tickLen = 3;
    addLine(render.lineRender, 0.8, color, new Vec3(p0.x - tickLen, p0.y, p0.z), p0, undefined);
    addLine(render.lineRender, 0.8, color, new Vec3(p1.x - tickLen, p1.y, p1.z), p1, undefined);
}

function drawFFNResidualBypass(render: IRenderState, layer: IT3TowerTransformerLayer) {
    let opacity = Math.min(layer.attnResidual.opacity, layer.ffnResidual.opacity);
    let color = residualColor.mul(opacity);
    if (color.w < 0.02) return;

    let xOff = layer.ln2.dx / 2 + 5;
    let p0 = new Vec3(layer.attnResidual.x + layer.attnResidual.dx / 2 + xOff, layer.attnResidual.y, layer.attnResidual.z + layer.attnResidual.dz / 2);
    let p1 = new Vec3(layer.ffnResidual.x + layer.ffnResidual.dx / 2 + xOff, layer.ffnResidual.y + layer.ffnResidual.dy, layer.ffnResidual.z + layer.ffnResidual.dz / 2);

    addLine(render.lineRender, 0.8, color, p0, p1, undefined);
    let tickLen = 3;
    addLine(render.lineRender, 0.8, color, new Vec3(p0.x - tickLen, p0.y, p0.z), p0, undefined);
    addLine(render.lineRender, 0.8, color, new Vec3(p1.x - tickLen, p1.y, p1.z), p1, undefined);
}
