import { measureTextWidth, writeTextToBuffer } from "@/src/llm/render/fontRender";
import { addLine } from "@/src/llm/render/lineRender";
import { IRenderState } from "@/src/llm/render/modelRender";
import { Mat4f } from "@/src/utils/matrix";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { IBloqueaModelLayout } from "./BloqueaModelLayout";

let baseColor = new Vec4(0.4, 0.4, 0.4, 1.0);
let frozenColor = new Vec4(0.5, 0.5, 0.5, 0.6);
let sharedColor = Vec4.fromHexColor('#9933cc');
let adapterColor = Vec4.fromHexColor('#ff6600');

export function drawBloqueaSectionLabels(render: IRenderState, layout: IBloqueaModelLayout) {
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

    // Audio Input label
    {
        let blk = layout.audioInput;
        let opacity = Math.min(blk.opacity, 1.0);
        if (opacity > 0.05) {
            let tl = new Vec3(blk.x - margin * 1.5, blk.y, 0);
            let br = new Vec3(blk.x - margin * 1.5, blk.y + blk.dy, 0);
            drawSectionLabel(render, 'Input', tl, br, baseColor.mul(opacity), 7);
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

    // Audio PosEmb label
    {
        let blk = layout.audioPosEmb;
        let tl = new Vec3(blk.x - margin * 1.5, blk.y, 0);
        let br = new Vec3(blk.x - margin * 1.5, blk.y + blk.dy, 0);
        drawSectionLabel(render, 'PosEmb (frozen)', tl, br, frozenColor, 7);
    }

    // Audio Transformer layers
    for (let i = 0; i < layout.audioTransformerLayers.length; i++) {
        let layer = layout.audioTransformerLayers[i];
        let topBlk = layer.ln1;
        // For layers with adapter, use the adapter's last block as bottom
        let bottomBlk = layer.adapter ? layer.adapter.adapterUp : layer.ffnResidual;
        let label: string;
        if (i < 2) {
            label = `Layer ${i} (frozen + adapter)`;
        } else {
            label = `Layer ${i} (unfrozen)`;
        }
        let tl = new Vec3(topBlk.x - margin * 1.5, topBlk.y, 0);
        let br = new Vec3(topBlk.x - margin * 1.5, bottomBlk.y + bottomBlk.dy, 0);
        drawSectionLabel(render, label, tl, br, baseColor, 8);

        // Add "Adapter" sub-label for layers 0-1
        if (layer.adapter) {
            let aTopBlk = layer.adapter.adapterDown;
            let aBotBlk = layer.adapter.adapterUp;
            let aTl = new Vec3(aTopBlk.x - margin * 0.8, aTopBlk.y, 0);
            let aBr = new Vec3(aTopBlk.x - margin * 0.8, aBotBlk.y + aBotBlk.dy, 0);
            drawSectionLabel(render, 'Adapter', aTl, aBr, adapterColor, 6);
        }
    }

    // Audio Projection subsection
    {
        let topBlk = layout.audioMeanPool;
        let bottomBlk = layout.audioProjLayers[layout.audioProjLayers.length - 1];
        if (bottomBlk) {
            let tl = new Vec3(topBlk.x - margin * 1.5, topBlk.y, 0);
            let br = new Vec3(topBlk.x - margin * 1.5, bottomBlk.y + bottomBlk.dy, 0);
            drawSectionLabel(render, 'Projection (lr_proj)', tl, br, baseColor, 8);
        }
    }

    // MIDI Tower main label
    {
        let topBlk = layout.midiInput;
        let bottomBlk = layout.midiProjLayers[layout.midiProjLayers.length - 1];
        if (bottomBlk) {
            let rightX = layout.midiDurEmb.x + layout.midiDurEmb.dx + margin * 3;
            let tl = new Vec3(rightX, topBlk.y, 0);
            let br = new Vec3(rightX, bottomBlk.y + bottomBlk.dy, 0);
            drawSectionLabelRight(render, 'MIDI Tower', tl, br, baseColor, 16);
        }
    }

    // MIDI Input label
    {
        let blk = layout.midiInput;
        let rightX = blk.x + blk.dx + margin * 1.5;
        let tl = new Vec3(rightX, blk.y, 0);
        let br = new Vec3(rightX, blk.y + blk.dy, 0);
        drawSectionLabelRight(render, 'Input', tl, br, baseColor.mul(blk.opacity), 7);
    }

    // MIDI Embedding subsection
    {
        let topBlk = layout.midiPitchEmb;
        let bottomBlk = layout.midiPosEnc;
        let rightX = layout.midiDurEmb.x + layout.midiDurEmb.dx + margin * 1.5;
        let tl = new Vec3(rightX, topBlk.y, 0);
        let br = new Vec3(rightX, bottomBlk.y + bottomBlk.dy, 0);
        drawSectionLabelRight(render, 'Embedding (lr_midi)', tl, br, baseColor, 8);
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

    // MIDI Output LN + Projection
    {
        let topBlk = layout.midiOutputLN;
        let bottomBlk = layout.midiProjLayers[layout.midiProjLayers.length - 1];
        if (bottomBlk) {
            let lastMidiTf = layout.midiTransformerLayers[layout.midiTransformerLayers.length - 1];
            let rightX = lastMidiTf.ln1.x + lastMidiTf.ln1.dx + margin * 1.5;
            let tl = new Vec3(rightX, topBlk.y, 0);
            let br = new Vec3(rightX, bottomBlk.y + bottomBlk.dy, 0);
            drawSectionLabelRight(render, 'Projection (lr_proj)', tl, br, baseColor, 8);
        }
    }

    // Shared Space label
    {
        let topBlk = layout.audioEmbedding;
        let bottomBlk = layout.vicregCov; // Use the last VICReg block as bottom
        let tl = new Vec3(topBlk.x - margin * 3, topBlk.y, 0);
        let br = new Vec3(topBlk.x - margin * 3, bottomBlk.y + bottomBlk.dy, 0);
        drawSectionLabel(render, 'Shared Space (256D)', tl, br, sharedColor, 14);
    }

    // VICReg sub-labels
    {
        let invBlk = layout.vicregInv;
        let covBlk = layout.vicregCov;
        let rightX = covBlk.x + covBlk.dx + margin * 1.5;
        let tl = new Vec3(rightX, invBlk.y, 0);
        let br = new Vec3(rightX, covBlk.y + covBlk.dy, 0);
        drawSectionLabelRight(render, 'VICReg Loss', tl, br, Vec4.fromHexColor('#cc3333'), 10);
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
