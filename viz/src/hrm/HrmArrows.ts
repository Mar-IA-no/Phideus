import { addLine } from "@/src/llm/render/lineRender";
import { IRenderState } from "@/src/llm/render/modelRender";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { IHrmModelLayout } from "./HrmModelLayout";
import { IBlkDef } from "@/src/llm/GptModelLayout";

let flowColor = new Vec4(0.45, 0.45, 0.45, 0.6);
let feedbackColor = Vec4.fromHexColor('#cc7733');
let actColor = Vec4.fromHexColor('#cc4444');
let hModuleColor = Vec4.fromHexColor('#6644aa');

export function drawHrmArrows(render: IRenderState, layout: IHrmModelLayout) {
    // L-Module vertical flow
    drawVertArrow(render, layout.inputHistogram, layout.encoderMlp);
    drawVertArrow(render, layout.encoderMlp, layout.contextIntegration);
    drawVertArrow(render, layout.contextIntegration, layout.gruStack);
    drawVertArrow(render, layout.gruStack, layout.spectralAttention);
    drawVertArrow(render, layout.spectralAttention, layout.lOutput);

    // L-Output splits to H-Module and ACT
    drawDiagonalArrow(render, layout.lOutput, layout.lAggregator, hModuleColor, 2.0);
    drawDiagonalArrow(render, layout.lOutput, layout.qNet1, actColor, 2.0);

    // H-Module vertical flow
    drawVertArrowColored(render, layout.lAggregator, layout.harmonicAttention, hModuleColor, 1.0);
    drawVertArrowColored(render, layout.harmonicAttention, layout.memoryLstm, hModuleColor, 1.0);
    drawVertArrowColored(render, layout.memoryLstm, layout.contextGenerator, hModuleColor, 1.0);
    drawVertArrowColored(render, layout.contextGenerator, layout.hContext, hModuleColor, 1.5);

    // ACT vertical flow
    drawVertArrowColored(render, layout.qNet1, layout.qNet2, actColor, 1.0);
    drawVertArrowColored(render, layout.qNet2, layout.qNet3, actColor, 1.0);
    drawVertArrowColored(render, layout.qNet3, layout.haltDecision, actColor, 1.5);

    // Feedback loop: h_context -> context integration (curved path)
    drawFeedbackArrow(render, layout.hContext, layout.contextIntegration, feedbackColor, 2.0);
}

function blkTopCenter(blk: IBlkDef): Vec3 {
    return new Vec3(blk.x + blk.dx / 2, blk.y, blk.z + blk.dz / 2);
}

function blkBotCenter(blk: IBlkDef): Vec3 {
    return new Vec3(blk.x + blk.dx / 2, blk.y + blk.dy, blk.z + blk.dz / 2);
}

function blkLeftCenter(blk: IBlkDef): Vec3 {
    return new Vec3(blk.x, blk.y + blk.dy / 2, blk.z + blk.dz / 2);
}

function blkRightCenter(blk: IBlkDef): Vec3 {
    return new Vec3(blk.x + blk.dx, blk.y + blk.dy / 2, blk.z + blk.dz / 2);
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

function drawFeedbackArrow(render: IRenderState, from: IBlkDef, to: IBlkDef, color: Vec4, thickness: number) {
    let opacity = Math.min(from.opacity, to.opacity);
    let c = color.mul(opacity);
    if (c.w < 0.03) return;

    // Draw a multi-segment path that goes: hContext left -> up along left side -> right to contextIntegration left
    let p0 = blkLeftCenter(from);
    let p3 = blkLeftCenter(to);

    let farLeft = Math.min(from.x, to.x) - 40;

    let p1 = new Vec3(farLeft, p0.y, p0.z);
    let p2 = new Vec3(farLeft, p3.y, p3.z);

    addLine(render.lineRender, thickness, c, p0, p1, undefined);
    addLine(render.lineRender, thickness, c, p1, p2, undefined);
    addLine(render.lineRender, thickness, c, p2, p3, undefined);

    // Arrowhead at p3
    let headLen = 3.0;
    let headW = 1.5;
    let dir = p3.sub(p2).normalize();
    let base = p3.sub(dir.mul(headLen));
    let perp = new Vec3(0, headW, 0);
    addLine(render.lineRender, thickness, c, base.add(perp), p3, undefined);
    addLine(render.lineRender, thickness, c, base.sub(perp), p3, undefined);
}
