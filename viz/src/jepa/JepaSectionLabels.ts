import { measureTextWidth, writeTextToBuffer } from "@/src/llm/render/fontRender";
import { addLine } from "@/src/llm/render/lineRender";
import { IRenderState } from "@/src/llm/render/modelRender";
import { Mat4f } from "@/src/utils/matrix";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { IJepaModelLayout } from "./JepaModelLayout";

let audioColor = new Vec4(0.2, 0.4, 0.8, 1.0);
let vibColor = new Vec4(0.8, 0.4, 0.2, 1.0);
let predColor = Vec4.fromHexColor('#6633cc');
let lossColor = Vec4.fromHexColor('#cc3333');

export function drawJepaSectionLabels(render: IRenderState, layout: IJepaModelLayout) {
    let { margin } = layout;

    // Audio Domain
    {
        let tl = new Vec3(layout.audio.input.x - margin * 2.5, layout.audio.input.y, 0);
        let br = new Vec3(layout.audio.input.x - margin * 2.5, layout.audio.stopGrad.y + layout.audio.stopGrad.dy, 0);
        drawSectionLabel(render, 'Audio Domain', tl, br, audioColor, 14);
    }

    // Vibration Domain
    {
        let rightX = layout.vibration.input.x + layout.vibration.input.dx + margin * 2.5;
        let tl = new Vec3(rightX, layout.vibration.input.y, 0);
        let br = new Vec3(rightX, layout.vibration.stopGrad.y + layout.vibration.stopGrad.dy, 0);
        drawSectionLabelRight(render, 'Vibration Domain', tl, br, vibColor, 14);
    }

    // Predictors
    {
        let tl = new Vec3(layout.predictorAtoV.x - margin * 2, layout.predictorAtoV.y, 0);
        let br = new Vec3(layout.predictorAtoV.x - margin * 2, layout.predictorVtoA.y + layout.predictorVtoA.dy, 0);
        drawSectionLabel(render, 'Predictors', tl, br, predColor, 12);
    }

    // Loss
    {
        let rightX = layout.infoNCELoss.x + layout.infoNCELoss.dx + margin * 2;
        let tl = new Vec3(rightX, layout.infoNCELoss.y, 0);
        let br = new Vec3(rightX, layout.infoNCELoss.y + layout.infoNCELoss.dy, 0);
        drawSectionLabelRight(render, 'InfoNCE', tl, br, lossColor, 12);
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
