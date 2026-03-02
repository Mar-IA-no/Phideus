import { cellPosition, IBlkDef, IModelLayout } from "@/src/llm/GptModelLayout";
import { addLine } from "@/src/llm/render/lineRender";
import { IRenderState } from "@/src/llm/render/modelRender";
import { measureTextWidth, writeTextToBuffer } from "@/src/llm/render/fontRender";
import { Mat4f } from "@/src/utils/matrix";
import { Dim, Vec3, Vec4 } from "@/src/utils/vector";
import { D4XA4XDim, d4xa4xDimColor, d4xa4xDimText } from "./D4XA4XDimStyle";
import { ID4XA4XModelLayout } from "./D4XA4XModelLayout";

export function d4xa4xBlockDimension(
    render: IRenderState,
    layout: IModelLayout,
    blk: IBlkDef,
    dim: Dim,
    phDim: D4XA4XDim,
    t: number,
) {
    if (t < 0.01 || phDim === D4XA4XDim.None) return;

    let text = d4xa4xDimText(phDim);
    if (!text) return;

    let color = d4xa4xDimColor(phDim).mul(t);
    let cx = dim === Dim.X ? blk.cx : blk.cy;
    if (cx <= 1) return;

    let start = cellPosition(layout, blk, dim, 0);
    let end = cellPosition(layout, blk, dim, cx - 1) + layout.cell;
    let mid = (start + end) / 2;
    let extent = end - start;

    let fontSize = Math.min(3.5, extent * 0.1);
    fontSize = Math.max(1.6, fontSize);

    let tw = measureTextWidth(render.modelFontBuf, text, fontSize);
    let textPad = tw / 2 + fontSize * 0.4;
    let edgeH2 = fontSize * 0.3;
    let botPad = fontSize * 0.8;
    let thickness = fontSize * 0.04;
    let zOff = blk.z + blk.dz + 2.0;
    let n = new Vec3(0, 0, 1);
    let mtx = new Mat4f();
    mtx[14] = zOff;

    if (dim === Dim.X) {
        let lineY = blk.y + blk.dy + botPad + fontSize * 0.5;
        writeTextToBuffer(render.modelFontBuf, text, color, mid - tw / 2, lineY - fontSize * 0.35, fontSize, mtx);
        let tooWide = tw >= extent - fontSize;
        if (!tooWide) {
            addLine(render.lineRender, thickness, color, new Vec3(start, lineY, zOff), new Vec3(mid - textPad, lineY, zOff), n);
            addLine(render.lineRender, thickness, color, new Vec3(mid + textPad, lineY, zOff), new Vec3(end, lineY, zOff), n);
        }
        addLine(render.lineRender, thickness, color, new Vec3(start, lineY - edgeH2, zOff), new Vec3(start, lineY + edgeH2, zOff), n);
        addLine(render.lineRender, thickness, color, new Vec3(end, lineY - edgeH2, zOff), new Vec3(end, lineY + edgeH2, zOff), n);
    } else {
        let lineX = blk.x - botPad;
        writeTextToBuffer(render.modelFontBuf, text, color, lineX - tw - fontSize * 0.3, mid - fontSize * 0.35, fontSize, mtx);
        let tooTall = tw >= extent - fontSize;
        if (!tooTall) {
            addLine(render.lineRender, thickness, color, new Vec3(lineX, start, zOff), new Vec3(lineX, mid - textPad, zOff), n);
            addLine(render.lineRender, thickness, color, new Vec3(lineX, mid + textPad, zOff), new Vec3(lineX, end, zOff), n);
        }
        addLine(render.lineRender, thickness, color, new Vec3(lineX - edgeH2, start, zOff), new Vec3(lineX + edgeH2, start, zOff), n);
        addLine(render.lineRender, thickness, color, new Vec3(lineX - edgeH2, end, zOff), new Vec3(lineX + edgeH2, end, zOff), n);
    }
}

function drawBlockNameLabel(render: IRenderState, blk: IBlkDef, opacity: number) {
    if (opacity < 0.02 || !blk.name) return;

    let fontSize = Math.min(3.5, blk.dx * 0.12, blk.dy * 0.3);
    fontSize = Math.max(1.6, fontSize);

    let tw = measureTextWidth(render.modelFontBuf, blk.name, fontSize);
    if (tw > blk.dx * 1.4) {
        fontSize *= (blk.dx * 1.3) / tw;
        fontSize = Math.max(1.2, fontSize);
        tw = measureTextWidth(render.modelFontBuf, blk.name, fontSize);
    }

    let color: Vec4;
    let dimX = blk.dimX as unknown as D4XA4XDim;

    // Forward cross-att blocks (special=1) → magenta
    if (blk.special === 1) {
        color = new Vec4(0.8, 0.2, 0.7, opacity);
    }
    // NoGrad DSP blocks → amber
    else if (dimX === D4XA4XDim.NoGrad || dimX === D4XA4XDim.D_stft_bins || dimX === D4XA4XDim.D_bands || dimX === D4XA4XDim.D_semitone || dimX === D4XA4XDim.D_log_ratio) {
        color = new Vec4(0.85, 0.6, 0.25, opacity);
    }
    // TrainableXAtt blocks → magenta
    else if (dimX === D4XA4XDim.TrainableXAtt) {
        color = new Vec4(0.8, 0.2, 0.7, opacity);
    }
    else {
        switch (blk.t) {
            case 'w': color = new Vec4(0.2, 0.2, 0.55, opacity); break;
            case 'i': color = new Vec4(0.1, 0.35, 0.1, opacity); break;
            case 'a': color = new Vec4(0.45, 0.3, 0.0, opacity); break;
            default:  color = new Vec4(0.3, 0.3, 0.3, opacity);
        }
    }

    let cx = blk.x + blk.dx / 2;
    let cy = blk.y - fontSize * 1.0 - 1.5;
    let z = blk.z + blk.dz + 2.0;
    let mtx = new Mat4f();
    mtx[14] = z;

    writeTextToBuffer(render.modelFontBuf, blk.name, color, cx - tw / 2, cy, fontSize, mtx);
}

export function drawD4XA4XBlockNames(render: IRenderState, layout: ID4XA4XModelLayout) {
    for (let blk of layout.cubes) {
        if (blk.opacity < 0.08) continue;
        let labelOpacity = blk.opacity * 0.6;
        if (blk.highlight > 0.05) {
            labelOpacity = Math.max(labelOpacity, Math.min(blk.highlight * 2.5, blk.opacity * 0.9));
        }
        drawBlockNameLabel(render, blk, labelOpacity);
    }
}
