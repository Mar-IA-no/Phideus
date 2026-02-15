import { addLine } from "@/src/llm/render/lineRender";
import { IRenderState } from "@/src/llm/render/modelRender";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { IConstellationModelLayout } from "./ConstellationModelLayout";
import { IBlkDef } from "@/src/llm/GptModelLayout";

let flowColor = new Vec4(0.45, 0.45, 0.45, 0.6);
let convergenceColor = Vec4.fromHexColor('#cc9933');
let lossColor = Vec4.fromHexColor('#cc3333');
let latentColor = Vec4.fromHexColor('#9933cc');

export function drawConstellationArrows(render: IRenderState, layout: IConstellationModelLayout) {
    // Audio encoder flow
    drawVertArrow(render, layout.audioInput, layout.audioMask);
    drawVertArrow(render, layout.audioMask, layout.audioTokenMlp);
    drawVertArrow(render, layout.audioTokenMlp, layout.audioAttnPool);
    drawVertArrow(render, layout.audioAttnPool, layout.audioBiLstm);

    // Vibration encoder flow
    drawVertArrow(render, layout.vibInput, layout.vibMask);
    drawVertArrow(render, layout.vibMask, layout.vibTokenMlp);
    drawVertArrow(render, layout.vibTokenMlp, layout.vibAttnPool);
    drawVertArrow(render, layout.vibAttnPool, layout.vibBiLstm);

    // Encoder to latent space
    drawDiagonalArrow(render, layout.audioBiLstm, layout.audioZShared, convergenceColor, 2.0);
    drawDiagonalArrow(render, layout.audioBiLstm, layout.audioZPrivate, convergenceColor, 1.5);
    drawDiagonalArrow(render, layout.vibBiLstm, layout.vibZShared, convergenceColor, 2.0);
    drawDiagonalArrow(render, layout.vibBiLstm, layout.vibZPrivate, convergenceColor, 1.5);

    // Latent concatenation
    drawVertArrow(render, layout.audioZShared, layout.audioZCat);
    drawVertArrow(render, layout.audioZPrivate, layout.audioZCat);
    drawVertArrow(render, layout.vibZShared, layout.vibZCat);
    drawVertArrow(render, layout.vibZPrivate, layout.vibZCat);

    // Latent to decoder
    drawVertArrow(render, layout.audioZCat, layout.audioDecProj);
    drawVertArrow(render, layout.vibZCat, layout.vibDecProj);

    // Decoder flow
    drawVertArrow(render, layout.audioDecProj, layout.audioDecLstm);
    drawVertArrow(render, layout.audioDecLstm, layout.audioDecOutput);
    drawVertArrow(render, layout.vibDecProj, layout.vibDecLstm);
    drawVertArrow(render, layout.vibDecLstm, layout.vibDecOutput);

    // Losses - from decoder outputs
    drawDiagonalArrow(render, layout.audioDecOutput, layout.reconLoss, lossColor, 1.5);
    drawDiagonalArrow(render, layout.vibDecOutput, layout.reconLoss, lossColor, 1.5);

    // Losses - from latent space
    drawDiagonalArrow(render, layout.audioZShared, layout.klSharedLoss, lossColor, 1.0);
    drawDiagonalArrow(render, layout.vibZShared, layout.klSharedLoss, lossColor, 1.0);
    drawDiagonalArrow(render, layout.audioZShared, layout.infoNceLoss, latentColor, 1.5);
    drawDiagonalArrow(render, layout.vibZShared, layout.infoNceLoss, latentColor, 1.5);
    drawDiagonalArrow(render, layout.audioZPrivate, layout.klPrivateLoss, lossColor, 1.0);
    drawDiagonalArrow(render, layout.vibZPrivate, layout.klPrivateLoss, lossColor, 1.0);
    drawDiagonalArrow(render, layout.audioZShared, layout.diffLoss, lossColor, 1.0);
    drawDiagonalArrow(render, layout.audioZPrivate, layout.diffLoss, lossColor, 1.0);
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
