import { addLine } from "@/src/llm/render/lineRender";
import { IRenderState } from "@/src/llm/render/modelRender";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { IDannModelLayout } from "./DannModelLayout";
import { IBlkDef } from "@/src/llm/GptModelLayout";

let flowColor = new Vec4(0.45, 0.45, 0.45, 0.6);
let convergenceColor = Vec4.fromHexColor('#8844bb');
let lossColor = Vec4.fromHexColor('#cc3333');
let grlColor = Vec4.fromHexColor('#996633');

export function drawDannArrows(render: IRenderState, layout: IDannModelLayout) {
    // Audio tower flow
    drawVertArrow(render, layout.audioInput, layout.audioCnn);
    drawVertArrow(render, layout.audioCnn, layout.audioTransformer);
    drawVertArrow(render, layout.audioTransformer, layout.audioPool);
    drawVertArrow(render, layout.audioPool, layout.audioProj);

    // MIDI tower flow
    drawVertArrow(render, layout.midiInput, layout.midiEmb);
    drawVertArrow(render, layout.midiEmb, layout.midiTransformer);
    drawVertArrow(render, layout.midiTransformer, layout.midiPool);
    drawVertArrow(render, layout.midiPool, layout.midiProj);

    // Convergence to shared space
    drawDiagonalArrow(render, layout.audioProj, layout.audioEmbedding, convergenceColor, 2.0);
    drawDiagonalArrow(render, layout.midiProj, layout.midiEmbedding, convergenceColor, 2.0);

    // Shared to VICReg
    drawDiagonalArrow(render, layout.audioEmbedding, layout.vicregLoss, lossColor, 1.5);
    drawDiagonalArrow(render, layout.midiEmbedding, layout.vicregLoss, lossColor, 1.5);

    // Shared to concat
    drawDiagonalArrow(render, layout.audioEmbedding, layout.concat, flowColor, 1.0);
    drawDiagonalArrow(render, layout.midiEmbedding, layout.concat, flowColor, 1.0);

    // DANN path
    drawVertArrow(render, layout.concat, layout.l2norm);
    drawVertArrowColored(render, layout.l2norm, layout.grl, grlColor, 1.5);

    // GRL to classifier
    drawVertArrowColored(render, layout.grl, layout.classifierLayers[0], grlColor, 1.5);
    for (let i = 0; i < layout.classifierLayers.length - 1; i++) {
        drawVertArrow(render, layout.classifierLayers[i], layout.classifierLayers[i + 1]);
    }

    // Classifier to DANN loss
    let lastClf = layout.classifierLayers[layout.classifierLayers.length - 1];
    drawVertArrowColored(render, lastClf, layout.dannLoss, lossColor, 1.5);
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

function drawVertArrowColored(render: IRenderState, from: IBlkDef, to: IBlkDef, color: Vec4, thickness: number) {
    let opacity = Math.min(from.opacity, to.opacity);
    let c = color.mul(opacity);
    if (c.w < 0.03) return;

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
