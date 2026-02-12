import { addLine } from "@/src/llm/render/lineRender";
import { IRenderState } from "@/src/llm/render/modelRender";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { IRosetaModelLayout, IRosetaEncoderPath } from "./RosetaModelLayout";
import { IBlkDef } from "@/src/llm/GptModelLayout";

let flowColor = new Vec4(0.45, 0.45, 0.45, 0.6);
let infoNCEColor = Vec4.fromHexColor('#22aacc');
let lossReconColor = Vec4.fromHexColor('#44aa44');
let lossKLSharedColor = Vec4.fromHexColor('#dd3333');
let lossKLPrivateColor = Vec4.fromHexColor('#bb4444');
let lossInfoNCEColor = Vec4.fromHexColor('#22aacc');
let lossDiffColor = Vec4.fromHexColor('#cc44cc');

export function drawRosetaArrows(render: IRenderState, layout: IRosetaModelLayout) {

    // ==========================================
    //  Audio domain data flow
    // ==========================================
    drawDomainArrows(render, layout.audio);

    // ==========================================
    //  Vibration domain data flow
    // ==========================================
    drawDomainArrows(render, layout.vibration);

    // ==========================================
    //  Cross-connections: audio.zShared <-> vibration.zShared (InfoNCE)
    // ==========================================
    drawHorizontalCrossArrow(render, layout.audio.zShared, layout.vibration.zShared, infoNCEColor, 1.5);

    // ==========================================
    //  Loss connections
    // ==========================================

    // Both recons -> lossRecon (diagonal, green)
    drawDiagonalArrow(render, layout.audio.recon, layout.lossRecon, lossReconColor, 1.5);
    drawDiagonalArrow(render, layout.vibration.recon, layout.lossRecon, lossReconColor, 1.5);

    // Both zShared -> lossKLShared (diagonal, red)
    drawDiagonalArrow(render, layout.audio.zShared, layout.lossKLShared, lossKLSharedColor, 1.5);
    drawDiagonalArrow(render, layout.vibration.zShared, layout.lossKLShared, lossKLSharedColor, 1.5);

    // Both zPrivate -> lossKLPrivate (diagonal, dark red)
    drawDiagonalArrow(render, layout.audio.zPrivate, layout.lossKLPrivate, lossKLPrivateColor, 1.5);
    drawDiagonalArrow(render, layout.vibration.zPrivate, layout.lossKLPrivate, lossKLPrivateColor, 1.5);

    // Both zShared -> lossInfoNCE (diagonal, cyan)
    drawDiagonalArrow(render, layout.audio.zShared, layout.lossInfoNCE, lossInfoNCEColor, 1.5);
    drawDiagonalArrow(render, layout.vibration.zShared, layout.lossInfoNCE, lossInfoNCEColor, 1.5);

    // Both zPrivate -> lossDiff (diagonal, magenta)
    drawDiagonalArrow(render, layout.audio.zPrivate, layout.lossDiff, lossDiffColor, 1.5);
    drawDiagonalArrow(render, layout.vibration.zPrivate, layout.lossDiff, lossDiffColor, 1.5);
}

/** Draw data-flow arrows for one domain's encoder-decoder pipeline. */
function drawDomainArrows(render: IRenderState, path: IRosetaEncoderPath) {
    // Encoder: input -> inputProj -> lstmEncoder
    drawVertArrow(render, path.input, path.inputProj);
    drawVertArrow(render, path.inputProj, path.lstmEncoder);

    // Fan-out: lstmEncoder -> muShared, logvarShared
    drawDiagonalArrow(render, path.lstmEncoder, path.muShared, flowColor, 0.8);
    drawDiagonalArrow(render, path.lstmEncoder, path.logvarShared, flowColor, 0.8);

    // Fan-out: lstmEncoder -> muPrivate, logvarPrivate
    drawDiagonalArrow(render, path.lstmEncoder, path.muPrivate, flowColor, 0.8);
    drawDiagonalArrow(render, path.lstmEncoder, path.logvarPrivate, flowColor, 0.8);

    // muShared/logvarShared -> zShared (diagonal reparameterization arrows)
    drawDiagonalArrow(render, path.muShared, path.zShared, flowColor, 0.8);
    drawDiagonalArrow(render, path.logvarShared, path.zShared, flowColor, 0.8);

    // muPrivate/logvarPrivate -> zPrivate (diagonal reparameterization arrows)
    drawDiagonalArrow(render, path.muPrivate, path.zPrivate, flowColor, 0.8);
    drawDiagonalArrow(render, path.logvarPrivate, path.zPrivate, flowColor, 0.8);

    // zShared -> zConcat, zPrivate -> zConcat
    drawDiagonalArrow(render, path.zShared, path.zConcat, flowColor, 0.8);
    drawDiagonalArrow(render, path.zPrivate, path.zConcat, flowColor, 0.8);

    // Decoder: zConcat -> zProj -> lstmDecoder -> outputProj -> softmax -> recon
    drawVertArrow(render, path.zConcat, path.zProj);
    drawVertArrow(render, path.zProj, path.lstmDecoder);
    drawVertArrow(render, path.lstmDecoder, path.outputProj);
    drawVertArrow(render, path.outputProj, path.softmax);
    drawVertArrow(render, path.softmax, path.recon);
}

function blkTopCenter(blk: IBlkDef): Vec3 {
    return new Vec3(blk.x + blk.dx / 2, blk.y, blk.z + blk.dz / 2);
}

function blkBotCenter(blk: IBlkDef): Vec3 {
    return new Vec3(blk.x + blk.dx / 2, blk.y + blk.dy, blk.z + blk.dz / 2);
}

function blkMidRight(blk: IBlkDef): Vec3 {
    return new Vec3(blk.x + blk.dx, blk.y + blk.dy / 2, blk.z + blk.dz / 2);
}

function blkMidLeft(blk: IBlkDef): Vec3 {
    return new Vec3(blk.x, blk.y + blk.dy / 2, blk.z + blk.dz / 2);
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

/** Horizontal cross-arrow connecting two blocks at the same Y level (e.g., InfoNCE link). */
function drawHorizontalCrossArrow(render: IRenderState, from: IBlkDef, to: IBlkDef, color: Vec4, thickness: number) {
    let opacity = Math.min(from.opacity, to.opacity);
    let c = color.mul(opacity);
    if (c.w < 0.03) return;

    let p0 = blkMidRight(from);
    let p1 = blkMidLeft(to);
    addLine(render.lineRender, thickness, c, p0, p1, undefined);

    // Arrow head pointing right (toward vibration)
    let headLen = 3.0;
    let headW = 1.5;
    let dir = p1.sub(p0).normalize();
    let base = p1.sub(dir.mul(headLen));
    let perp = new Vec3(0, headW, 0);
    addLine(render.lineRender, thickness, c, base.add(perp), p1, undefined);
    addLine(render.lineRender, thickness, c, base.sub(perp), p1, undefined);

    // Arrow head pointing left (toward audio) — bidirectional
    let base2 = p0.add(dir.mul(headLen));
    addLine(render.lineRender, thickness, c, base2.add(perp), p0, undefined);
    addLine(render.lineRender, thickness, c, base2.sub(perp), p0, undefined);

    // "InfoNCE" label at midpoint
    // (label rendering is handled by section labels / annotations, not here)
}
