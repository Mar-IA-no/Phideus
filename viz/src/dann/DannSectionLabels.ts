import { measureTextWidth, writeTextToBuffer } from "@/src/llm/render/fontRender";
import { addLine } from "@/src/llm/render/lineRender";
import { IRenderState } from "@/src/llm/render/modelRender";
import { Mat4f } from "@/src/utils/matrix";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { IDannModelLayout } from "./DannModelLayout";

let baseColor = new Vec4(0.4, 0.4, 0.4, 1.0);
let sharedColor = Vec4.fromHexColor('#9933cc');
let grlColor = Vec4.fromHexColor('#996633');
let lossColor = Vec4.fromHexColor('#cc3333');

export function drawDannSectionLabels(render: IRenderState, layout: IDannModelLayout) {
    let { margin } = layout;

    // Audio Tower
    {
        let tl = new Vec3(layout.audioInput.x - margin * 2.5, layout.audioInput.y, 0);
        let br = new Vec3(layout.audioInput.x - margin * 2.5, layout.audioProj.y + layout.audioProj.dy, 0);
        drawSectionLabel(render, 'Audio Tower', tl, br, baseColor, 14);
    }

    // MIDI Tower
    {
        let rightX = layout.midiEmb.x + layout.midiEmb.dx + margin * 2.5;
        let tl = new Vec3(rightX, layout.midiInput.y, 0);
        let br = new Vec3(rightX, layout.midiProj.y + layout.midiProj.dy, 0);
        drawSectionLabelRight(render, 'MIDI Tower', tl, br, baseColor, 14);
    }

    // Shared Space
    {
        let tl = new Vec3(layout.audioEmbedding.x - margin * 2.5, layout.audioEmbedding.y, 0);
        let br = new Vec3(layout.audioEmbedding.x - margin * 2.5, layout.vicregLoss.y + layout.vicregLoss.dy, 0);
        drawSectionLabel(render, 'Shared Space', tl, br, sharedColor, 12);
    }

    // DANN Path
    {
        let lastClf = layout.classifierLayers[layout.classifierLayers.length - 1];
        let rightX = layout.concat.x + layout.concat.dx + margin * 2;
        let tl = new Vec3(rightX, layout.concat.y, 0);
        let br = new Vec3(rightX, layout.dannLoss.y + layout.dannLoss.dy, 0);
        drawSectionLabelRight(render, 'DANN Path', tl, br, grlColor, 12);
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
