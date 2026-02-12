import { measureTextWidth, writeTextToBuffer } from "@/src/llm/render/fontRender";
import { addLine } from "@/src/llm/render/lineRender";
import { IRenderState } from "@/src/llm/render/modelRender";
import { Mat4f } from "@/src/utils/matrix";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { IRosetaModelLayout } from "./RosetaModelLayout";

let audioColor = Vec4.fromHexColor('#3366cc');
let vibrationColor = Vec4.fromHexColor('#cc6633');
let latentColor = Vec4.fromHexColor('#d4930a');
let lossColor = Vec4.fromHexColor('#dd3333');
let subLabelColor = new Vec4(0.4, 0.4, 0.4, 1.0);

export function drawRosetaSectionLabels(render: IRenderState, layout: IRosetaModelLayout) {
    let { margin } = layout;

    // ==========================================
    //  AUDIO DOMAIN (left side)
    // ==========================================

    // Main "Audio Domain" label
    {
        let topBlk = layout.audio.input;
        let bottomBlk = layout.audio.recon;
        if (topBlk && bottomBlk) {
            let tl = new Vec3(topBlk.x - margin * 3, topBlk.y, 0);
            let br = new Vec3(topBlk.x - margin * 3, bottomBlk.y + bottomBlk.dy, 0);
            drawSectionLabel(render, 'Audio Domain', tl, br, audioColor, 16);
        }
    }

    // Audio "Encoder" sub-label
    {
        let topBlk = layout.audio.input;
        let bottomBlk = layout.audio.lstmEncoder;
        if (topBlk && bottomBlk) {
            let opacity = Math.min(topBlk.opacity, bottomBlk.opacity, 1.0);
            if (opacity > 0.05) {
                let tl = new Vec3(topBlk.x - margin * 1.5, topBlk.y, 0);
                let br = new Vec3(topBlk.x - margin * 1.5, bottomBlk.y + bottomBlk.dy, 0);
                drawSectionLabel(render, 'Encoder', tl, br, subLabelColor.mul(opacity), 8);
            }
        }
    }

    // Audio "Latent Space" sub-label
    {
        let topBlk = layout.audio.muShared;
        let bottomBlk = layout.audio.zProj;
        if (topBlk && bottomBlk) {
            let opacity = Math.min(topBlk.opacity, bottomBlk.opacity, 1.0);
            if (opacity > 0.05) {
                let tl = new Vec3(topBlk.x - margin * 1.5, topBlk.y, 0);
                let br = new Vec3(topBlk.x - margin * 1.5, bottomBlk.y + bottomBlk.dy, 0);
                drawSectionLabel(render, 'Latent Space', tl, br, latentColor.mul(opacity), 8);
            }
        }
    }

    // Audio "Decoder" sub-label
    {
        let topBlk = layout.audio.lstmDecoder;
        let bottomBlk = layout.audio.recon;
        if (topBlk && bottomBlk) {
            let opacity = Math.min(topBlk.opacity, bottomBlk.opacity, 1.0);
            if (opacity > 0.05) {
                let tl = new Vec3(topBlk.x - margin * 1.5, topBlk.y, 0);
                let br = new Vec3(topBlk.x - margin * 1.5, bottomBlk.y + bottomBlk.dy, 0);
                drawSectionLabel(render, 'Decoder', tl, br, subLabelColor.mul(opacity), 8);
            }
        }
    }

    // ==========================================
    //  VIBRATION DOMAIN (right side)
    // ==========================================

    // Main "Vibration Domain" label
    {
        let topBlk = layout.vibration.input;
        let bottomBlk = layout.vibration.recon;
        if (topBlk && bottomBlk) {
            let rightX = topBlk.x + topBlk.dx + margin * 3;
            let tl = new Vec3(rightX, topBlk.y, 0);
            let br = new Vec3(rightX, bottomBlk.y + bottomBlk.dy, 0);
            drawSectionLabelRight(render, 'Vibration Domain', tl, br, vibrationColor, 16);
        }
    }

    // Vibration "Encoder" sub-label
    {
        let topBlk = layout.vibration.input;
        let bottomBlk = layout.vibration.lstmEncoder;
        if (topBlk && bottomBlk) {
            let opacity = Math.min(topBlk.opacity, bottomBlk.opacity, 1.0);
            if (opacity > 0.05) {
                let rightX = topBlk.x + topBlk.dx + margin * 1.5;
                let tl = new Vec3(rightX, topBlk.y, 0);
                let br = new Vec3(rightX, bottomBlk.y + bottomBlk.dy, 0);
                drawSectionLabelRight(render, 'Encoder', tl, br, subLabelColor.mul(opacity), 8);
            }
        }
    }

    // Vibration "Latent Space" sub-label
    {
        let topBlk = layout.vibration.muShared;
        let bottomBlk = layout.vibration.zProj;
        if (topBlk && bottomBlk) {
            let opacity = Math.min(topBlk.opacity, bottomBlk.opacity, 1.0);
            if (opacity > 0.05) {
                let rightX = bottomBlk.x + bottomBlk.dx + margin * 1.5;
                let tl = new Vec3(rightX, topBlk.y, 0);
                let br = new Vec3(rightX, bottomBlk.y + bottomBlk.dy, 0);
                drawSectionLabelRight(render, 'Latent Space', tl, br, latentColor.mul(opacity), 8);
            }
        }
    }

    // Vibration "Decoder" sub-label
    {
        let topBlk = layout.vibration.lstmDecoder;
        let bottomBlk = layout.vibration.recon;
        if (topBlk && bottomBlk) {
            let opacity = Math.min(topBlk.opacity, bottomBlk.opacity, 1.0);
            if (opacity > 0.05) {
                let rightX = topBlk.x + topBlk.dx + margin * 1.5;
                let tl = new Vec3(rightX, topBlk.y, 0);
                let br = new Vec3(rightX, bottomBlk.y + bottomBlk.dy, 0);
                drawSectionLabelRight(render, 'Decoder', tl, br, subLabelColor.mul(opacity), 8);
            }
        }
    }

    // ==========================================
    //  LOSS SPACE (center bottom)
    // ==========================================
    {
        let leftBlk = layout.lossRecon;
        let rightBlk = layout.lossDiff;
        if (leftBlk && rightBlk) {
            // Center below all loss blocks
            let midX = (leftBlk.x + rightBlk.x + rightBlk.dx) / 2;
            let tl = new Vec3(midX, leftBlk.y + leftBlk.dy + margin * 0.5, 0);
            let br = new Vec3(midX, leftBlk.y + leftBlk.dy + margin * 0.5, 0);
            drawSectionLabelBelow(render, 'Loss Space', midX, leftBlk.y + leftBlk.dy + margin * 1.0, lossColor, 14);
        }
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

function drawSectionLabelBelow(render: IRenderState, text: string, centerX: number, y: number, color: Vec4, fontSize: number) {
    let mtx = new Mat4f();
    mtx[14] = 0;

    let tw = measureTextWidth(render.modelFontBuf, text, fontSize);
    writeTextToBuffer(render.modelFontBuf, text, color, centerX - tw / 2, y, fontSize, mtx);
}
