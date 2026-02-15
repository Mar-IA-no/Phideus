import { addLine } from "@/src/llm/render/lineRender";
import { IRenderState } from "@/src/llm/render/modelRender";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { ICrossAttModelLayout } from "./CrossAttModelLayout";
import { IBlkDef } from "@/src/llm/GptModelLayout";

let flowColor = new Vec4(0.45, 0.45, 0.45, 0.6);
let crossAttColor = Vec4.fromHexColor('#cc3366');
let lossColor = Vec4.fromHexColor('#cc3333');
let descriptorColor = Vec4.fromHexColor('#8855cc');
let intervalColor = Vec4.fromHexColor('#6b9e3a');

export function drawCrossAttArrows(render: IRenderState, layout: ICrossAttModelLayout) {
    // Audio tower flow
    drawVertArrow(render, layout.waveformInput, layout.cnn);
    drawVertArrow(render, layout.cnn, layout.posEmb);
    drawVertArrow(render, layout.posEmb, layout.crossAttAudio);
    drawVertArrow(render, layout.crossAttAudio, layout.residualNormAudio);
    drawVertArrow(render, layout.residualNormAudio, layout.audioTransformer);
    drawVertArrow(render, layout.audioTransformer, layout.audioPool);
    drawVertArrow(render, layout.audioPool, layout.audioProj);

    // MIDI tower flow
    drawVertArrow(render, layout.midiInput, layout.midiEmb);
    drawVertArrow(render, layout.midiEmb, layout.midiPosEnc);
    drawVertArrow(render, layout.midiPosEnc, layout.crossAttMidi);
    drawVertArrow(render, layout.crossAttMidi, layout.residualNormMidi);
    drawVertArrow(render, layout.residualNormMidi, layout.midiTransformer);
    drawVertArrow(render, layout.midiTransformer, layout.midiPool);
    drawVertArrow(render, layout.midiPool, layout.midiProj);

    // Audio descriptor side-channel to cross-attention (diagonal, colored, thicker)
    drawDiagonalArrow(render, layout.audioDescriptor, layout.descKvProj, descriptorColor, 1.5);
    drawDiagonalArrow(render, layout.descKvProj, layout.crossAttAudio, crossAttColor, 2.5);

    // MIDI interval side-channel to cross-attention (diagonal, colored, thicker)
    drawDiagonalArrow(render, layout.midiIntervals, layout.intervalKvProj, intervalColor, 1.5);
    drawDiagonalArrow(render, layout.intervalKvProj, layout.crossAttMidi, crossAttColor, 2.5);

    // Projections to VICReg loss (diagonal)
    drawDiagonalArrow(render, layout.audioProj, layout.vicregLoss, lossColor, 1.5);
    drawDiagonalArrow(render, layout.midiProj, layout.vicregLoss, lossColor, 1.5);
}

function blkTopCenter(blk: IBlkDef): Vec3 {
    return new Vec3(blk.x + blk.dx / 2, blk.y, blk.z + blk.dz / 2);
}

function blkBotCenter(blk: IBlkDef): Vec3 {
    return new Vec3(blk.x + blk.dx / 2, blk.y + blk.dy, blk.z + blk.dz / 2);
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
