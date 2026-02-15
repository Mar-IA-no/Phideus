import { measureTextWidth, writeTextToBuffer } from "@/src/llm/render/fontRender";
import { addLine } from "@/src/llm/render/lineRender";
import { IRenderState } from "@/src/llm/render/modelRender";
import { Mat4f } from "@/src/utils/matrix";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { IConstellationModelLayout } from "./ConstellationModelLayout";

let audioColor = new Vec4(0.2, 0.4, 0.8, 1.0);
let vibColor = new Vec4(0.8, 0.4, 0.2, 1.0);
let latentColor = Vec4.fromHexColor('#cc9933');
let lossColor = Vec4.fromHexColor('#cc3333');

export function drawConstellationSectionLabels(render: IRenderState, layout: IConstellationModelLayout) {
    let { margin } = layout;

    // Audio Domain
    {
        let tl = new Vec3(layout.audioInput.x - margin * 2.5, layout.audioInput.y, 0);
        let br = new Vec3(layout.audioInput.x - margin * 2.5, layout.audioBiLstm.y + layout.audioBiLstm.dy, 0);
        drawSectionLabel(render, 'Audio Encoder', tl, br, audioColor, 12);
    }

    // Vibration Domain
    {
        let rightX = layout.vibInput.x + layout.vibInput.dx + margin * 2.5;
        let tl = new Vec3(rightX, layout.vibInput.y, 0);
        let br = new Vec3(rightX, layout.vibBiLstm.y + layout.vibBiLstm.dy, 0);
        drawSectionLabelRight(render, 'Vibration Encoder', tl, br, vibColor, 12);
    }

    // Latent Space
    {
        let tl = new Vec3(layout.audioZShared.x - margin * 2, layout.audioZShared.y, 0);
        let br = new Vec3(layout.audioZShared.x - margin * 2, layout.audioZCat.y + layout.audioZCat.dy, 0);
        drawSectionLabel(render, 'Latent Space', tl, br, latentColor, 12);
    }

    // Losses
    {
        let rightX = layout.diffLoss.x + layout.diffLoss.dx + margin * 2;
        let tl = new Vec3(rightX, layout.reconLoss.y, 0);
        let br = new Vec3(rightX, layout.reconLoss.y + layout.reconLoss.dy, 0);
        drawSectionLabelRight(render, 'Losses (5)', tl, br, lossColor, 12);
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
