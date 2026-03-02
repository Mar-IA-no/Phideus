import { cellPosition, IBlkDef, IModelLayout } from "@/src/llm/GptModelLayout";
import { ICamera } from "@/src/llm/Camera";
import { addLine } from "@/src/llm/render/lineRender";
import { IRenderState } from "@/src/llm/render/modelRender";
import { measureTextWidth, writeTextToBuffer } from "@/src/llm/render/fontRender";
import { Mat4f } from "@/src/utils/matrix";
import { Dim, Vec3, Vec4 } from "@/src/utils/vector";
import { T3TowerDim, t3TowerDimColor, t3TowerDimText } from "./T3TowerDimStyle";
import { IT3TowerModelLayout } from "./T3TowerModelLayout";

function camScaleToScreenLocal(camera: ICamera, renderSize: Vec3, modelPt: Vec3) {
    let camDist = camera.camPosModel.dist(modelPt);
    return camDist / renderSize.y * 5.0;
}

export function t3TowerBlockDimension(
    render: IRenderState,
    layout: IModelLayout,
    blk: IBlkDef,
    dim: Dim,
    phDim: T3TowerDim,
    t: number,
    camera?: ICamera,
) {
    if (t < 0.01 || phDim === T3TowerDim.None) return;

    let text = t3TowerDimText(phDim);
    if (!text) return;

    let color = t3TowerDimColor(phDim).mul(t);

    let cx = dim === Dim.X ? blk.cx : blk.cy;
    if (cx <= 1) return;

    let start = cellPosition(layout, blk, dim, 0);
    let end = cellPosition(layout, blk, dim, cx - 1) + layout.cell;
    let mid = (start + end) / 2;
    let extent = end - start;

    let fontSize = Math.min(3.5, extent * 0.1);
    fontSize = Math.max(1.6, fontSize);

    if (camera) {
        let midPos = new Vec3(mid, blk.y + blk.dy, blk.z + blk.dz);
        let scale = Math.min(camScaleToScreenLocal(camera, render.size, midPos), 1);
        fontSize = 3.5 * scale;
        fontSize = Math.max(1.2, fontSize);
    }

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
    switch (blk.t) {
        case 'w': color = new Vec4(0.2, 0.2, 0.55, opacity); break;
        case 'i': color = new Vec4(0.1, 0.35, 0.1, opacity); break;
        case 'a': color = new Vec4(0.45, 0.3, 0.0, opacity); break;
        default:  color = new Vec4(0.3, 0.3, 0.3, opacity);
    }

    let cx = blk.x + blk.dx / 2;
    let cy = blk.y - fontSize * 1.0 - 1.5;
    let z = blk.z + blk.dz + 2.0;
    let mtx = new Mat4f();
    mtx[14] = z;

    writeTextToBuffer(render.modelFontBuf, blk.name, color, cx - tw / 2, cy, fontSize, mtx);
}

export function drawT3TowerBlockNames(render: IRenderState, layout: IT3TowerModelLayout) {
    for (let blk of layout.cubes) {
        if (blk.opacity < 0.08) continue;

        let labelOpacity = blk.opacity * 0.4;
        if (blk.highlight > 0.05) {
            labelOpacity = Math.max(labelOpacity, Math.min(blk.highlight * 2.5, blk.opacity * 0.9));
        }

        drawBlockNameLabel(render, blk, labelOpacity);
    }
}
