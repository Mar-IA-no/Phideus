import { genModelViewMatrices, ICamera, updateCamera } from "@/src/llm/Camera";
import { IBlkDef } from "@/src/llm/GptModelLayout";
import { IDisplayState, IMouseState } from "@/src/llm/Program";
import { IFontAtlasData, IFontOpts, drawText, measureText } from "@/src/llm/render/fontRender";
import { IRenderState, IRenderView, initRender, renderModel, resetRenderBuffers } from "@/src/llm/render/modelRender";
import { RenderPhase } from "@/src/llm/render/sharedRender";
import { beginQueryAndGetPrevMs, endQuery } from "@/src/llm/render/queryManager";
import { Mat4f } from "@/src/utils/matrix";
import { Subscriptions } from "@/src/utils/hooks";
import { isNotNil } from "@/src/utils/data";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { ILayout } from "@/src/utils/layout";
import { defaultConstellationShape, genConstellationModelLayout, IConstellationModelLayout, IConstellationModelShape } from "./ConstellationModelLayout";
import { drawConstellationArrows } from "./ConstellationArrows";
import { drawConstellationSectionLabels } from "./ConstellationSectionLabels";
import { drawConstellationBlockNames } from "./ConstellationAnnotations";
import { initConstellationWalkthrough, IConstellationWalkthrough, runConstellationWalkthrough } from "./walkthrough/ConstellationWalkthrough";

export interface IConstellationProgramState {
    render: IRenderState;
    camera: ICamera;
    layout: IConstellationModelLayout;
    shape: IConstellationModelShape;
    mouse: IMouseState;
    display: IDisplayState;
    inWalkthrough: boolean;
    walkthrough: IConstellationWalkthrough;
    htmlSubs: Subscriptions;
    pageLayout: ILayout;
    markDirty: () => void;
    examples: never[];
}

export function initConstellationProgramState(canvasEl: HTMLCanvasElement, fontAtlasData: IFontAtlasData): IConstellationProgramState | null {
    let render = initRender(canvasEl, fontAtlasData);
    if (!render) return null;

    let walkthrough = initConstellationWalkthrough();
    let shape = defaultConstellationShape;

    let camera: ICamera = {
        angle: new Vec3(290, 22, 14.0),
        center: new Vec3(0, 0, -500),
        transition: {},
        modelMtx: new Mat4f(),
        viewMtx: new Mat4f(),
        lookAtMtx: new Mat4f(),
        camPos: new Vec3(),
        camPosModel: new Vec3(),
    };

    return {
        render, camera, shape,
        layout: genConstellationModelLayout(shape),
        mouse: { mousePos: new Vec3() },
        display: {
            tokenColors: null, tokenIdxColors: null, tokenOutputColors: null,
            lines: [], hoverTarget: null, dimHover: null, blkIdxHover: null,
        },
        inWalkthrough: true,
        walkthrough,
        htmlSubs: new Subscriptions(),
        pageLayout: { height: 0, width: 0, isDesktop: true, isPhone: false },
        markDirty: () => {},
        examples: [],
    };
}

export function runConstellationProgram(view: IRenderView, state: IConstellationProgramState) {
    let timer0 = performance.now();
    if (!state.render) return;

    resetRenderBuffers(state.render);
    state.render.sharedRender.activePhase = RenderPhase.Opaque;
    state.display.lines = [];
    state.display.hoverTarget = null;

    state.layout = genConstellationModelLayout(state.shape);
    genModelViewMatrices(state as any, state.layout);

    let queryRes = beginQueryAndGetPrevMs(state.render.queryManager, 'render');
    if (isNotNil(queryRes)) state.render.lastGpuMs = queryRes;
    state.render.renderTiming = false;

    if (state.inWalkthrough) runConstellationWalkthrough(state, view);
    updateCamera(state as any, view);

    let bestDist = Infinity;
    let bestCube: IBlkDef | null = null;
    for (let cube of state.layout.cubes) {
        if (cube.opacity < 0.1) continue;
        let center = new Vec3(cube.x + cube.dx / 2, cube.y + cube.dy / 2, cube.z + cube.dz / 2);
        let ndc = state.camera.viewMtx.mulVec3Proj(state.camera.modelMtx.mulVec3Affine(center));
        let screenPos = new Vec3((ndc.x + 1) * 0.5 * state.render.size.x, (1 - ndc.y) * 0.5 * state.render.size.y, 0);
        let dist = screenPos.sub(state.mouse.mousePos).len();
        if (dist < 40 && dist < bestDist) { bestDist = dist; bestCube = cube; }
    }
    if (bestCube) {
        bestCube.highlight = Math.max(bestCube.highlight, 0.3);
        state.display.hoverTarget = { mainCube: bestCube, subCube: bestCube, mainIdx: new Vec3() };
    }

    if (state.display.blkIdxHover) {
        for (let cube of state.layout.cubes) {
            if (state.display.blkIdxHover.includes(cube.idx)) {
                cube.highlight = Math.max(cube.highlight, 0.5);
            }
        }
    }

    drawConstellationArrows(state.render, state.layout);
    drawConstellationSectionLabels(state.render, state.layout);
    drawConstellationBlockNames(state.render, state.layout);

    let lineNo = 1;
    let tw = state.render.size.x;
    state.render.sharedRender.activePhase = RenderPhase.Overlay2D;
    for (let line of state.display.lines) {
        let opts: IFontOpts = { color: new Vec4(), size: 14 };
        let w = measureText(state.render.modelFontBuf, line, opts);
        drawText(state.render.modelFontBuf, line, tw - w - 4, lineNo * opts.size * 1.3 + 4, opts);
        lineNo++;
    }

    renderModel(state as any);
    endQuery(state.render.queryManager, 'render');
    state.render.gl.flush();
    state.render.lastJsMs = performance.now() - timer0;
}
