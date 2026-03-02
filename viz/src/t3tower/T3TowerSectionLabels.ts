import { measureTextWidth, writeTextToBuffer } from "@/src/llm/render/fontRender";
import { addLine } from "@/src/llm/render/lineRender";
import { IRenderState } from "@/src/llm/render/modelRender";
import { Mat4f } from "@/src/utils/matrix";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { IT3TowerModelLayout } from "./T3TowerModelLayout";

let baseColor = new Vec4(0.4, 0.4, 0.4, 1.0);
let frozenColor = new Vec4(0.5, 0.5, 0.5, 0.6);
let sharedColor = Vec4.fromHexColor('#9933cc');
let t3Color = Vec4.fromHexColor('#2299aa');

export function drawT3TowerSectionLabels(render: IRenderState, layout: IT3TowerModelLayout) {
    let { margin } = layout;

    // Audio Tower main label
    {
        let topBlk = layout.audioInput;
        let bottomBlk = layout.audioProjLayers[layout.audioProjLayers.length - 1];
        if (topBlk && bottomBlk) {
            let tl = new Vec3(topBlk.x - margin * 3, topBlk.y, 0);
            let br = new Vec3(topBlk.x - margin * 3, bottomBlk.y + bottomBlk.dy, 0);
            drawSectionLabel(render, 'Audio Tower (MERT 330M)', tl, br, baseColor, 16);
        }
    }

    // Audio CNN subsection
    {
        let topBlk = layout.audioCnn[0];
        let bottomBlk = layout.audioCnn[layout.audioCnn.length - 1];
        if (topBlk && bottomBlk) {
            let tl = new Vec3(topBlk.x - margin * 1.5, topBlk.y, 0);
            let br = new Vec3(topBlk.x - margin * 1.5, bottomBlk.y + bottomBlk.dy, 0);
            let color = frozenColor.mul(Math.min(topBlk.opacity, 0.8));
            drawSectionLabel(render, 'CNN (frozen)', tl, br, color, 8);
        }
    }

    // Audio Transformer layers
    for (let i = 0; i < layout.audioTransformerLayers.length; i++) {
        let layer = layout.audioTransformerLayers[i];
        let topBlk = layer.ln1;
        let bottomBlk = layer.ffnResidual;
        let lr = i < 2 ? 'lr_low' : 'lr_high';
        let tl = new Vec3(topBlk.x - margin * 1.5, topBlk.y, 0);
        let br = new Vec3(topBlk.x - margin * 1.5, bottomBlk.y + bottomBlk.dy, 0);
        drawSectionLabel(render, `Layer ${i} (${lr})`, tl, br, baseColor, 8);
    }

    // Audio Projection
    {
        let topBlk = layout.audioMeanPool;
        let bottomBlk = layout.audioProjLayers[layout.audioProjLayers.length - 1];
        if (bottomBlk) {
            let tl = new Vec3(topBlk.x - margin * 1.5, topBlk.y, 0);
            let br = new Vec3(topBlk.x - margin * 1.5, bottomBlk.y + bottomBlk.dy, 0);
            drawSectionLabel(render, 'Projection', tl, br, baseColor, 8);
        }
    }

    // MIDI Tower main label
    {
        let topBlk = layout.midiInput;
        let bottomBlk = layout.midiProjLayers[layout.midiProjLayers.length - 1];
        if (bottomBlk) {
            let rightX = layout.midiDurEmb.x + layout.midiDurEmb.dx + margin * 1.5;
            let tl = new Vec3(rightX, topBlk.y, 0);
            let br = new Vec3(rightX, bottomBlk.y + bottomBlk.dy, 0);
            drawSectionLabelRight(render, 'MIDI Tower', tl, br, baseColor, 14);
        }
    }

    // MIDI Embedding subsection
    {
        let topBlk = layout.midiPitchEmb;
        let bottomBlk = layout.midiPosEnc;
        let rightX = layout.midiDurEmb.x + layout.midiDurEmb.dx + margin * 1.5;
        let tl = new Vec3(rightX, topBlk.y, 0);
        let br = new Vec3(rightX, bottomBlk.y + bottomBlk.dy, 0);
        drawSectionLabelRight(render, 'Embedding', tl, br, baseColor, 8);
    }

    // MIDI Transformer layers
    for (let i = 0; i < layout.midiTransformerLayers.length; i++) {
        let layer = layout.midiTransformerLayers[i];
        let topBlk = layer.ln1;
        let bottomBlk = layer.ffnResidual;
        let rightX = topBlk.x + topBlk.dx + margin * 1.5;
        let tl = new Vec3(rightX, topBlk.y, 0);
        let br = new Vec3(rightX, bottomBlk.y + bottomBlk.dy, 0);
        drawSectionLabelRight(render, `Layer ${i} (lr_midi)`, tl, br, baseColor, 8);
    }

    // T3 Tower main label
    {
        let topBlk = layout.t3Input;
        let bottomBlk = layout.t3ProjLayers[layout.t3ProjLayers.length - 1];
        if (bottomBlk) {
            let rightX = topBlk.x + topBlk.dx + margin * 2;
            let tl = new Vec3(rightX, topBlk.y, 0);
            let br = new Vec3(rightX, bottomBlk.y + bottomBlk.dy, 0);
            drawSectionLabelRight(render, 'T3 Tower (d=256, 2L)', tl, br, t3Color, 14);
        }
    }

    // T3 Transformer layers
    for (let i = 0; i < layout.t3TransformerLayers.length; i++) {
        let layer = layout.t3TransformerLayers[i];
        let topBlk = layer.ln1;
        let bottomBlk = layer.ffnResidual;
        let rightX = topBlk.x + topBlk.dx + margin * 1.5;
        let tl = new Vec3(rightX, topBlk.y, 0);
        let br = new Vec3(rightX, bottomBlk.y + bottomBlk.dy, 0);
        drawSectionLabelRight(render, `Layer ${i} (lr_t3)`, tl, br, t3Color.mul(0.7), 8);
    }

    // Shared Space label
    {
        let topBlk = layout.audioEmbedding;
        let bottomBlk = layout.vicregMT;
        let tl = new Vec3(topBlk.x - margin * 3, topBlk.y, 0);
        let br = new Vec3(topBlk.x - margin * 3, bottomBlk.y + bottomBlk.dy, 0);
        drawSectionLabel(render, 'Shared Space (256D)', tl, br, sharedColor, 14);
    }

    // 3-way VICReg label
    {
        let rightX = layout.vicregMT.x + layout.vicregMT.dx + margin * 1.5;
        let tl = new Vec3(rightX, layout.vicregAM.y, 0);
        let br = new Vec3(rightX, layout.vicregMT.y + layout.vicregMT.dy, 0);
        drawSectionLabelRight(render, '3-way VICReg', tl, br, Vec4.fromHexColor('#cc3333'), 10);
    }
}

function drawSectionLabel(render: IRenderState, text: string, tl: Vec3, br: Vec3, color: Vec4, fontSize: number) {
    let mtx = new Mat4f();
    mtx[14] = (tl.z + br.z) / 2;

    let pad = 10;
    let textColor = color;
    let lineColor = color.mul(0.4);

    let tw = measureTextWidth(render.modelFontBuf, text, fontSize);
    writeTextToBuffer(render.modelFontBuf, text, textColor, tl.x - tw - 2 * pad, (tl.y + br.y) / 2 - fontSize / 2, fontSize, mtx);

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
    let textColor = color;
    let lineColor = color.mul(0.4);

    writeTextToBuffer(render.modelFontBuf, text, textColor, tl.x + 2 * pad, (tl.y + br.y) / 2 - fontSize / 2, fontSize, mtx);

    let p0 = new Vec3(tl.x, tl.y, (tl.z + br.z) / 2);
    let p1 = new Vec3(br.x, br.y, (tl.z + br.z) / 2);
    let inward = new Vec3(-1, 0, 0);

    addLine(render.lineRender, 1.0, lineColor, p0.mulAdd(inward, -pad), p1.mulAdd(inward, -pad), undefined);
    addLine(render.lineRender, 1.0, lineColor, p0.mulAdd(inward, -pad), p0, undefined);
    addLine(render.lineRender, 1.0, lineColor, p1.mulAdd(inward, -pad), p1, undefined);
}
