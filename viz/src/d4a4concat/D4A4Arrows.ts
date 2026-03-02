import { addLine } from "@/src/llm/render/lineRender";
import { IRenderState } from "@/src/llm/render/modelRender";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { ID4A4ModelLayout, ID4A4TransformerLayer } from "./D4A4ModelLayout";
import { IBlkDef } from "@/src/llm/GptModelLayout";

let flowColor = new Vec4(0.55, 0.55, 0.55, 0.75);
let convergenceColor = Vec4.fromHexColor('#8844bb');
let lossColor = Vec4.fromHexColor('#cc3333');
let attnColor = new Vec4(0.3, 0.4, 0.7, 0.5);
let residualColor = new Vec4(0.6, 0.3, 0.1, 0.35);
let descriptorAmberColor = Vec4.fromHexColor('#e09530');
let descriptorGreenColor = Vec4.fromHexColor('#7ab445');
let concatColor = Vec4.fromHexColor('#bb4488');

export function drawD4A4Arrows(render: IRenderState, layout: ID4A4ModelLayout) {

    // Audio tower: input -> CNN -> posEmb -> transformers -> concat -> pool -> proj
    drawVertArrow(render, layout.waveformInput, layout.audioCnn[0]);
    for (let i = 0; i < layout.audioCnn.length - 1; i++) {
        drawVertArrow(render, layout.audioCnn[i], layout.audioCnn[i + 1]);
    }
    drawVertArrow(render, layout.audioCnn[layout.audioCnn.length - 1], layout.audioPosEmb);
    drawVertArrow(render, layout.audioPosEmb, layout.audioTransformerLayers[0].ln1);

    drawTransformerChain(render, layout.audioTransformerLayers);

    let lastAudioTf = layout.audioTransformerLayers[layout.audioTransformerLayers.length - 1];
    drawVertArrow(render, lastAudioTf.ffnResidual, layout.audioConcatGate);

    // Audio A4 descriptor flow (amber arrows)
    drawVertArrow(render, layout.audioDescStftWindow, layout.audioDescStftCompute, descriptorAmberColor, 1.5);
    drawVertArrow(render, layout.audioDescStftCompute, layout.audioDescMagnitude, descriptorAmberColor, 1.5);
    drawVertArrow(render, layout.audioDescMagnitude, layout.audioDescLogMag, descriptorAmberColor, 1.5);
    drawVertArrow(render, layout.audioDescLogMag, layout.audioDescBandGroup, descriptorAmberColor, 1.5);
    drawVertArrow(render, layout.audioDescBandGroup, layout.audioDescTemporalDelta, descriptorAmberColor, 1.5);
    drawVertArrow(render, layout.audioDescTemporalDelta, layout.audioDescNormalize, descriptorAmberColor, 1.5);
    drawVertArrow(render, layout.audioDescNormalize, layout.audioDescOutput, descriptorAmberColor, 1.5);

    // Audio descriptor -> concat (diagonal merge arrow)
    drawDiagonalArrow(render, layout.audioDescOutput, layout.audioConcatGate, concatColor, 2.0);

    // Audio concat -> pool -> proj
    drawVertArrow(render, layout.audioConcatGate, layout.audioMeanPool);
    drawVertArrow(render, layout.audioMeanPool, layout.audioProjLayer1);
    drawVertArrow(render, layout.audioProjLayer1, layout.audioProjLayer2);
    drawVertArrow(render, layout.audioProjLayer2, layout.audioProjLayer3);

    // MIDI tower: input -> embs -> combine -> posEnc -> transformers -> concat -> LN -> pool -> proj
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
    drawVertArrow(render, lastMidiTf.ffnResidual, layout.midiConcatGate);

    // MIDI D4 descriptor flow (green arrows)
    drawVertArrow(render, layout.midiDescPitchInput, layout.midiDescForwardDiff, descriptorGreenColor, 1.5);
    drawVertArrow(render, layout.midiDescForwardDiff, layout.midiDescValidityMask, descriptorGreenColor, 1.5);
    drawVertArrow(render, layout.midiDescValidityMask, layout.midiDescSemitoneScale, descriptorGreenColor, 1.5);
    drawVertArrow(render, layout.midiDescSemitoneScale, layout.midiDescLogRatioScale, descriptorGreenColor, 1.5);
    drawVertArrow(render, layout.midiDescLogRatioScale, layout.midiDescOutput, descriptorGreenColor, 1.5);

    // MIDI descriptor -> concat (diagonal merge arrow)
    drawDiagonalArrow(render, layout.midiDescOutput, layout.midiConcatGate, concatColor, 2.0);

    // MIDI concat -> LN -> pool -> proj
    drawVertArrow(render, layout.midiConcatGate, layout.midiOutputLN);
    drawVertArrow(render, layout.midiOutputLN, layout.midiMeanPool);
    drawVertArrow(render, layout.midiMeanPool, layout.midiProjLayer1);
    drawVertArrow(render, layout.midiProjLayer1, layout.midiProjLayer2);
    drawVertArrow(render, layout.midiProjLayer2, layout.midiProjLayer3);

    // Convergence: proj -> shared space
    drawDiagonalArrow(render, layout.audioProjLayer3, layout.audioEmbedding, convergenceColor, 2.0);
    drawDiagonalArrow(render, layout.midiProjLayer3, layout.midiEmbedding, convergenceColor, 2.0);

    // Shared -> VICReg loss
    drawDiagonalArrow(render, layout.audioEmbedding, layout.vicregInv, lossColor, 1.5);
    drawDiagonalArrow(render, layout.midiEmbedding, layout.vicregInv, lossColor, 1.5);
    drawDiagonalArrow(render, layout.audioEmbedding, layout.vicregVar, lossColor, 1.5);
    drawDiagonalArrow(render, layout.midiEmbedding, layout.vicregVar, lossColor, 1.5);
    drawDiagonalArrow(render, layout.audioEmbedding, layout.vicregCov, lossColor, 1.5);
    drawDiagonalArrow(render, layout.midiEmbedding, layout.vicregCov, lossColor, 1.5);

    // Residual connections
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

function drawVertArrow(render: IRenderState, from: IBlkDef, to: IBlkDef, color?: Vec4, thickness?: number) {
    let opacity = Math.min(from.opacity, to.opacity);
    let c = (color ?? flowColor).mul(opacity);
    if (c.w < 0.03) return;
    let t = thickness ?? 1.2;

    let p0 = blkBotCenter(from);
    let p1 = blkTopCenter(to);
    addLine(render.lineRender, t, c, p0, p1, undefined);

    let headLen = 2.5;
    let headW = 1.3;
    let dir = p1.sub(p0).normalize();
    let base = p1.sub(dir.mul(headLen));
    let perp = new Vec3(headW, 0, 0);
    addLine(render.lineRender, t, c, base.add(perp), p1, undefined);
    addLine(render.lineRender, t, c, base.sub(perp), p1, undefined);
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

function drawTransformerChain(render: IRenderState, layers: ID4A4TransformerLayer[]) {
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

function drawAttentionResidualBypass(render: IRenderState, layer: ID4A4TransformerLayer) {
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

function drawFFNResidualBypass(render: IRenderState, layer: ID4A4TransformerLayer) {
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
