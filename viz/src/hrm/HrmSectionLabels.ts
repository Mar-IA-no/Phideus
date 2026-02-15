import { measureTextWidth, writeTextToBuffer } from "@/src/llm/render/fontRender";
import { addLine } from "@/src/llm/render/lineRender";
import { IRenderState } from "@/src/llm/render/modelRender";
import { Mat4f } from "@/src/utils/matrix";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { IHrmModelLayout } from "./HrmModelLayout";

let lModuleColor = Vec4.fromHexColor('#339966');
let hModuleColor = Vec4.fromHexColor('#6644aa');
let actColor = Vec4.fromHexColor('#cc4444');
let feedbackColor = Vec4.fromHexColor('#cc7733');

export function drawHrmSectionLabels(render: IRenderState, layout: IHrmModelLayout) {
    let { margin } = layout;

    // L-Module (center, main processing)
    {
        let rightX = layout.encoderMlp.x + layout.encoderMlp.dx + margin * 2.5;
        let tl = new Vec3(rightX, layout.encoderMlp.y, 0);
        let br = new Vec3(rightX, layout.lOutput.y + layout.lOutput.dy, 0);
        drawSectionLabelRight(render, 'L-Module (Fast)', tl, br, lModuleColor, 12);
    }

    // H-Module (below-left)
    {
        let tl = new Vec3(layout.lAggregator.x - margin * 2.5, layout.lAggregator.y, 0);
        let br = new Vec3(layout.lAggregator.x - margin * 2.5, layout.hContext.y + layout.hContext.dy, 0);
        drawSectionLabel(render, 'H-Module (Slow)', tl, br, hModuleColor, 12);
    }

    // ACT (below-right)
    {
        let rightX = layout.qNet1.x + layout.qNet1.dx + margin * 2;
        let tl = new Vec3(rightX, layout.qNet1.y, 0);
        let br = new Vec3(rightX, layout.haltDecision.y + layout.haltDecision.dy, 0);
        drawSectionLabelRight(render, 'ACT (Halting)', tl, br, actColor, 12);
    }
}

function drawSectionLabel(render: IRenderState, text: string, tl: Vec3, br: Vec3, color: Vec4, fontSize: number) {
    let mtx = new Mat4f();
    mtx[14] = (tl.z + br.z) / 2;
    let pad = 10;
    let lineColor = color.mul(0.4);
    let tw = measureTextWidth(render.modelFontBuf, text, fontSize);
    writeTextToBuffer(render.modelFontBuf, text, color, tl.x - tw - 2 * pad, (tl.y + br.y) / 2 - fontSize / 2, fontSize, mtx);
    let p0 = new Vec3(tl.x, tl.y, (tl.z + br.z) / 2);
    let p1 = new Vec3(br.x, br.y, (tl.z + br.z) / 2);
    let inward = new Vec3(1, 0, 0);
    addLine(render.lineRender, 1.0, lineColor, p0.mulAdd(inward, -pad), p1.mulAdd(inward, -pad), undefined);
    addLine(render.lineRender, 1.0, lineColor, p0.mulAdd(inward, -pad), p0, undefined);
    addLine(render.lineRender, 1.0, lineColor, p1.mulAdd(inward, -pad), p1, undefined);
}

function drawSectionLabelRight(render: IRenderState, text: string, tl: Vec3, br: Vec3, color: Vec4, fontSize: number) {
    let mtx = new Mat4f();
    mtx[14] = (tl.z + br.z) / 2;
    let pad = 10;
    let lineColor = color.mul(0.4);
    writeTextToBuffer(render.modelFontBuf, text, color, tl.x + 2 * pad, (tl.y + br.y) / 2 - fontSize / 2, fontSize, mtx);
    let p0 = new Vec3(tl.x, tl.y, (tl.z + br.z) / 2);
    let p1 = new Vec3(br.x, br.y, (tl.z + br.z) / 2);
    let inward = new Vec3(-1, 0, 0);
    addLine(render.lineRender, 1.0, lineColor, p0.mulAdd(inward, -pad), p1.mulAdd(inward, -pad), undefined);
    addLine(render.lineRender, 1.0, lineColor, p0.mulAdd(inward, -pad), p0, undefined);
    addLine(render.lineRender, 1.0, lineColor, p1.mulAdd(inward, -pad), p1, undefined);
}
