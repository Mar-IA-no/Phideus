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
import { DimStyle } from "@/src/llm/walkthrough/WalkthroughTools";
import { defaultBloqueaShape, genBloqueaModelLayout, IBloqueaModelLayout, IBloqueaModelShape } from "./BloqueaModelLayout";
import { drawBloqueaArrows } from "./BloqueaArrows";
import { drawBloqueaSectionLabels } from "./BloqueaSectionLabels";
import { drawBloqueaBlockNames } from "./BloqueaAnnotations";
import { initBloqueaWalkthrough, IBloqueaWalkthrough, runBloqueaWalkthrough } from "./walkthrough/BloqueaWalkthrough";

// Adapter interface that makes our state compatible with renderModel()
// renderModel() accesses: state.layout, state.render, state.camera, state.examples
export interface IBloqueaProgramState {
    render: IRenderState;
    camera: ICamera;
    layout: IBloqueaModelLayout;
    shape: IBloqueaModelShape;
    mouse: IMouseState;
    display: IDisplayState;
    inWalkthrough: boolean;
    walkthrough: IBloqueaWalkthrough;
    htmlSubs: Subscriptions;
    pageLayout: ILayout;
    markDirty: () => void;

    // renderModel() also accesses state.examples -- we provide an empty array
    examples: never[];
}

export function initBloqueaProgramState(canvasEl: HTMLCanvasElement, fontAtlasData: IFontAtlasData): IBloqueaProgramState | null {
    let render = initRender(canvasEl, fontAtlasData);
    if (!render) return null;

    let walkthrough = initBloqueaWalkthrough();
    let shape = defaultBloqueaShape;

    let camera: ICamera = {
        angle: new Vec3(290, 20, 8.0),
        center: new Vec3(0, 0, -400),
        transition: {},
        modelMtx: new Mat4f(),
        viewMtx: new Mat4f(),
        lookAtMtx: new Mat4f(),
        camPos: new Vec3(),
        camPosModel: new Vec3(),
    };

    return {
        render,
        camera,
        shape,
        layout: genBloqueaModelLayout(shape),
        mouse: { mousePos: new Vec3() },
        display: {
            tokenColors: null,
            tokenIdxColors: null,
            tokenOutputColors: null,
            lines: [],
            hoverTarget: null,
            dimHover: null,
            blkIdxHover: null,
        },
        inWalkthrough: true,
        walkthrough,
        htmlSubs: new Subscriptions(),
        pageLayout: { height: 0, width: 0, isDesktop: true, isPhone: false },
        markDirty: () => {},
        examples: [],
    };
}

export function runBloqueaProgram(view: IRenderView, state: IBloqueaProgramState) {
    let timer0 = performance.now();

    if (!state.render) return;

    resetRenderBuffers(state.render);
    state.render.sharedRender.activePhase = RenderPhase.Opaque;
    state.display.lines = [];
    state.display.hoverTarget = null;

    // Regenerate layout each frame (cheap, no GPU model)
    state.layout = genBloqueaModelLayout(state.shape);

    // Generate camera matrices
    genModelViewMatrices(state as any, state.layout);

    let queryRes = beginQueryAndGetPrevMs(state.render.queryManager, 'render');
    if (isNotNil(queryRes)) {
        state.render.lastGpuMs = queryRes;
    }

    state.render.renderTiming = false;

    // Run walkthrough (modifies layout, camera, etc)
    if (state.inWalkthrough) {
        runBloqueaWalkthrough(state, view);
    }

    updateCamera(state as any, view);

    // Hit testing: highlight block closest to mouse cursor
    let bestDist = Infinity;
    let bestCube: IBlkDef | null = null;
    for (let cube of state.layout.cubes) {
        if (cube.opacity < 0.1) continue;
        let center = new Vec3(cube.x + cube.dx / 2, cube.y + cube.dy / 2, cube.z + cube.dz / 2);
        let ndc = state.camera.viewMtx.mulVec3Proj(state.camera.modelMtx.mulVec3Affine(center));
        let screenPos = new Vec3(
            (ndc.x + 1) * 0.5 * state.render.size.x,
            (1 - ndc.y) * 0.5 * state.render.size.y,
            0);
        let dist = screenPos.sub(state.mouse.mousePos).len();
        if (dist < 40 && dist < bestDist) {
            bestDist = dist;
            bestCube = cube;
        }
    }
    if (bestCube) {
        bestCube.highlight = Math.max(bestCube.highlight, 0.3);
        state.display.hoverTarget = { mainCube: bestCube, subCube: bestCube, mainIdx: new Vec3() };
    }

    // Handle blkIdxHover from sidebar BlockText component
    if (state.display.blkIdxHover) {
        for (let cube of state.layout.cubes) {
            if (state.display.blkIdxHover.includes(cube.idx)) {
                cube.highlight = Math.max(cube.highlight, 0.5);
            }
        }
    }

    // Draw arrows, labels, and block names
    drawBloqueaArrows(state.render, state.layout);
    drawBloqueaSectionLabels(state.render, state.layout);
    drawBloqueaBlockNames(state.render, state.layout);

    // Overlay text
    let lineNo = 1;
    let tw = state.render.size.x;
    state.render.sharedRender.activePhase = RenderPhase.Overlay2D;
    for (let line of state.display.lines) {
        let opts: IFontOpts = { color: new Vec4(), size: 14 };
        let w = measureText(state.render.modelFontBuf, line, opts);
        drawText(state.render.modelFontBuf, line, tw - w - 4, lineNo * opts.size * 1.3 + 4, opts);
        lineNo++;
    }

    // Execute all GL draw calls
    renderModel(state as any);

    endQuery(state.render.queryManager, 'render');
    state.render.gl.flush();

    state.render.lastJsMs = performance.now() - timer0;
}
