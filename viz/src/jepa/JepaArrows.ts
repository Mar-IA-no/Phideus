import { addLine } from "@/src/llm/render/lineRender";
import { IRenderState } from "@/src/llm/render/modelRender";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { IJepaModelLayout, IJepaEncoderPath } from "./JepaModelLayout";
import { IBlkDef } from "@/src/llm/GptModelLayout";

let flowColor = new Vec4(0.45, 0.45, 0.45, 0.6);
let predColor = Vec4.fromHexColor('#6633cc');
let lossColor = Vec4.fromHexColor('#cc3333');

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

function drawDomainArrows(render: IRenderState, path: IJepaEncoderPath) {
    drawVertArrow(render, path.input, path.tokenMlp);
    drawVertArrow(render, path.tokenMlp, path.attnPool);
    drawVertArrow(render, path.attnPool, path.lstm);
    drawVertArrow(render, path.lstm, path.z);
    drawVertArrow(render, path.z, path.stopGrad);
}

export function drawJepaArrows(render: IRenderState, layout: IJepaModelLayout) {
    drawDomainArrows(render, layout.audio);
    drawDomainArrows(render, layout.vibration);

    // Predictor arrows
    drawDiagonalArrow(render, layout.audio.stopGrad, layout.predictorAtoV, predColor, 1.5);
    drawDiagonalArrow(render, layout.vibration.stopGrad, layout.predictorVtoA, predColor, 1.5);

    // Loss connections
    drawDiagonalArrow(render, layout.predictorAtoV, layout.infoNCELoss, lossColor, 1.5);
    drawDiagonalArrow(render, layout.predictorVtoA, layout.infoNCELoss, lossColor, 1.5);
}
