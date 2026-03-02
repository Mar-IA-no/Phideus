import { measureTextWidth, writeTextToBuffer } from "@/src/llm/render/fontRender";
import { addLine } from "@/src/llm/render/lineRender";
import { IRenderState } from "@/src/llm/render/modelRender";
import { Mat4f } from "@/src/utils/matrix";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { ID4RA4RModelLayout } from "./D4RA4RModelLayout";

let baseColor = new Vec4(0.4, 0.4, 0.4, 1.0);
let descriptorColor = Vec4.fromHexColor('#e09530');   // amber — A4
let intervalColor = Vec4.fromHexColor('#6b9e3a');      // green — D4
let reverseXAttColor = Vec4.fromHexColor('#00bbcc');
let lossColor = Vec4.fromHexColor('#cc3333');
let sharedColor = Vec4.fromHexColor('#9933cc');

export function drawD4RA4RSectionLabels(render: IRenderState, layout: ID4RA4RModelLayout) {
    let { margin } = layout;

    // ===== AUDIO TOWER (left) =====

    {
        let tl = new Vec3(layout.waveformInput.x - margin * 2.5, layout.waveformInput.y, 0);
        let br = new Vec3(layout.waveformInput.x - margin * 2.5, layout.audioProjLayer3.y + layout.audioProjLayer3.dy, 0);
        drawSectionLabel(render, 'Audio Tower (MERT)', tl, br, baseColor, 14);
    }

    {
        let tl = new Vec3(layout.waveformInput.x - margin * 0.5, layout.waveformInput.y, 0);
        let br = new Vec3(layout.waveformInput.x - margin * 0.5, layout.audioPosEmb.y + layout.audioPosEmb.dy, 0);
        drawSectionLabel(render, 'CNN (frozen)', tl, br, baseColor, 9);
    }

    // Audio Descriptor A4 — AMBER
    {
        let tl = new Vec3(layout.audioDescStftWindow.x - margin * 1.5, layout.audioDescStftWindow.y, 0);
        let br = new Vec3(layout.audioDescStftWindow.x - margin * 1.5, layout.audioDescOutput.y + layout.audioDescOutput.dy, 0);
        drawSectionLabel(render, 'Audio Descriptor A4', tl, br, descriptorColor, 12);
        drawSectionLabel(render, 'NO_GRAD', new Vec3(tl.x, tl.y + 18, 0), new Vec3(br.x, tl.y + 18, 0), descriptorColor.mul(0.6), 8);
    }

    // REVERSE Cross-Attention (audio)
    {
        let x = layout.audioReverseDescQProj.x - margin * 1.2;
        let tl = new Vec3(x, layout.audioReverseDescQProj.y, 0);
        let br = new Vec3(x, layout.audioReverseOutput.y + layout.audioReverseOutput.dy, 0);
        drawSectionLabel(render, 'REVERSE XAtt', tl, br, reverseXAttColor, 10);
    }

    for (let i = 0; i < layout.audioTransformerLayers.length; i++) {
        let l = layout.audioTransformerLayers[i];
        let x = l.ln1.x - margin * 1.2;
        let tl = new Vec3(x, l.ln1.y, 0);
        let br = new Vec3(x, l.ffnResidual.y + l.ffnResidual.dy, 0);
        let lr = i < 2 ? 'lr_low' : 'lr_high';
        drawSectionLabel(render, `Audio L${i} (${lr})`, tl, br, baseColor, 7);
    }

    {
        let tl = new Vec3(layout.audioProjLayer1.x - margin * 1.2, layout.audioOutputLN.y, 0);
        let br = new Vec3(layout.audioProjLayer1.x - margin * 1.2, layout.audioProjLayer3.y + layout.audioProjLayer3.dy, 0);
        drawSectionLabel(render, 'Audio Projection', tl, br, baseColor, 8);
    }

    // ===== MIDI TOWER (right) =====

    {
        let rightX = layout.midiCombineLinear.x + layout.midiCombineLinear.dx + margin * 2.5;
        let tl = new Vec3(rightX, layout.midiInput.y, 0);
        let br = new Vec3(rightX, layout.midiProjLayer3.y + layout.midiProjLayer3.dy, 0);
        drawSectionLabelRight(render, 'MIDI Tower', tl, br, baseColor, 14);
    }

    {
        let rightX = layout.midiCombineLinear.x + layout.midiCombineLinear.dx + margin * 0.5;
        let tl = new Vec3(rightX, layout.midiInput.y, 0);
        let br = new Vec3(rightX, layout.midiPosEnc.y + layout.midiPosEnc.dy, 0);
        drawSectionLabelRight(render, 'Event Embedding', tl, br, baseColor, 9);
    }

    // MIDI Descriptor D4 — GREEN
    {
        let rightX = layout.midiDescPitchInput.x + layout.midiDescPitchInput.dx + margin * 1.5;
        let tl = new Vec3(rightX, layout.midiDescPitchInput.y, 0);
        let br = new Vec3(rightX, layout.midiDescOutput.y + layout.midiDescOutput.dy, 0);
        drawSectionLabelRight(render, 'MIDI Descriptor D4', tl, br, intervalColor, 12);
        drawSectionLabelRight(render, 'NO_GRAD', new Vec3(rightX, tl.y + 18, 0), new Vec3(rightX, tl.y + 18, 0), intervalColor.mul(0.6), 8);
    }

    // MIDI REVERSE Cross-Attention
    {
        let rightX = layout.midiReverseIntQProj.x + layout.midiReverseIntQProj.dx + margin * 1.2;
        let tl = new Vec3(rightX, layout.midiReverseIntQProj.y, 0);
        let br = new Vec3(rightX, layout.midiReverseOutput.y + layout.midiReverseOutput.dy, 0);
        drawSectionLabelRight(render, 'REVERSE XAtt', tl, br, reverseXAttColor, 10);
    }

    for (let i = 0; i < layout.midiTransformerLayers.length; i++) {
        let l = layout.midiTransformerLayers[i];
        let rightX = l.ln1.x + l.ln1.dx + margin * 1.2;
        let tl = new Vec3(rightX, l.ln1.y, 0);
        let br = new Vec3(rightX, l.ffnResidual.y + l.ffnResidual.dy, 0);
        drawSectionLabelRight(render, `MIDI L${i}`, tl, br, baseColor, 7);
    }

    {
        let rightX = layout.midiProjLayer1.x + layout.midiProjLayer1.dx + margin * 1.2;
        let tl = new Vec3(rightX, layout.midiOutputLN.y, 0);
        let br = new Vec3(rightX, layout.midiProjLayer3.y + layout.midiProjLayer3.dy, 0);
        drawSectionLabelRight(render, 'MIDI Projection', tl, br, baseColor, 8);
    }

    // ===== CENTER =====

    {
        let tl = new Vec3(layout.audioEmbedding.x - margin * 2, layout.audioEmbedding.y, 0);
        let br = new Vec3(layout.audioEmbedding.x - margin * 2, layout.midiEmbedding.y + layout.midiEmbedding.dy, 0);
        drawSectionLabel(render, 'Shared Space (256D)', tl, br, sharedColor, 12);
    }

    {
        let tl = new Vec3(layout.vicregInv.x - margin * 2, layout.vicregInv.y, 0);
        let br = new Vec3(layout.vicregInv.x - margin * 2, layout.vicregCov.y + layout.vicregCov.dy, 0);
        drawSectionLabel(render, 'VICReg Loss', tl, br, lossColor, 12);
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
