import { ICameraPos } from "@/src/llm/Camera";
import { IBlkDef } from "@/src/llm/GptModelLayout";
import { IRenderView } from "@/src/llm/render/modelRender";
import { commentary as llmCommentary, ICommentary, ICommentaryRes, ITimeInfo } from "@/src/llm/walkthrough/WalkthroughTools";
import { IConstellationProgramState } from "../ConstellationProgram";
import { IConstellationModelLayout } from "../ConstellationModelLayout";
import { constellationPhase00_Overview } from "./Phase00_Overview";
import { constellationPhase01_TokenInput } from "./Phase01_TokenInput";
import { constellationPhase02_Encoders } from "./Phase02_Encoders";
import { constellationPhase03_LatentSpace } from "./Phase03_LatentSpace";
import { constellationPhase04_DecodersAndLosses } from "./Phase04_DecodersAndLosses";

export enum ConstellationPhase {
    None,
    Overview,
    TokenInput,
    Encoders,
    LatentSpace,
    DecodersAndLosses,
}

export enum ConstellationPhaseGroup {
    Introduction,
    Architecture,
    Training,
}

export interface IConstellationPhaseGroup {
    groupId: ConstellationPhaseGroup;
    title: string;
    phases: IConstellationPhaseDef[];
}

export interface IConstellationPhaseDef {
    id: ConstellationPhase;
    title: string;
}

export interface IConstellationWalkthrough {
    phase: ConstellationPhase;
    time: number;
    viewDt: number;
    dt: number;
    prevTime: number;
    prevPhase: ConstellationPhase;
    running: boolean;
    speed: number;
    cameraInitial: ICameraPos | null;
    commentary: ICommentaryRes | null;
    times: (ITimeInfo | ICommentary)[];
    phaseLength: number;
    dimHighlightBlocks: IBlkDef[] | null;
    markDirty: () => void;
    phaseData: Map<ConstellationPhase, any>;
    phaseTransitiveData: any;
    phaseList: IConstellationPhaseGroup[];
}

export interface IConstellationWalkthroughArgs {
    state: IConstellationProgramState;
    layout: IConstellationModelLayout;
    walkthrough: IConstellationWalkthrough;
    tools: ReturnType<typeof constellationPhaseTools>;
}

export function initConstellationWalkthrough(): IConstellationWalkthrough {
    return {
        phase: ConstellationPhase.Overview,
        time: 0, viewDt: 0, dt: 0, prevTime: 0,
        prevPhase: ConstellationPhase.None,
        running: false, speed: 1,
        cameraInitial: null, commentary: null,
        times: [], phaseLength: 0,
        dimHighlightBlocks: null,
        markDirty: () => {},
        phaseData: new Map(),
        phaseTransitiveData: null,
        phaseList: [{
            groupId: ConstellationPhaseGroup.Introduction,
            title: 'Introduction',
            phases: [{ id: ConstellationPhase.Overview, title: 'Architecture Overview' }],
        }, {
            groupId: ConstellationPhaseGroup.Architecture,
            title: 'Architecture',
            phases: [
                { id: ConstellationPhase.TokenInput, title: 'Sparse Token Input' },
                { id: ConstellationPhase.Encoders, title: 'Dual Encoders' },
                { id: ConstellationPhase.LatentSpace, title: 'Factored Latent Space' },
            ],
        }, {
            groupId: ConstellationPhaseGroup.Training,
            title: 'Training',
            phases: [{ id: ConstellationPhase.DecodersAndLosses, title: 'Decoders & Loss Terms' }],
        }],
    };
}

export function phaseToConstellationGroup(wt: IConstellationWalkthrough) {
    return wt.phaseList.find(g => g.phases.find(p => p.id === wt.phase))!;
}

export function jumpConstellationPhase(wt: IConstellationWalkthrough, delta: number) {
    let group = phaseToConstellationGroup(wt);
    let groupIdx = wt.phaseList.indexOf(group);
    let phaseGroupIdx = group.phases.findIndex((p: IConstellationPhaseDef) => p.id === wt.phase);
    let newIdx = phaseGroupIdx + delta;

    if (newIdx < 0) {
        if (groupIdx > 0) {
            let newGroup = wt.phaseList[groupIdx - 1];
            wt.phase = newGroup.phases[newGroup.phases.length - 1].id;
        }
    } else if (newIdx >= group.phases.length) {
        if (groupIdx < wt.phaseList.length - 1) {
            let newGroup = wt.phaseList[groupIdx + 1];
            wt.phase = newGroup.phases[0].id;
        }
    } else {
        wt.phase = group.phases[newIdx].id;
    }

    wt.time = 0;
    wt.running = false;
}

import { clamp } from "@/src/utils/data";
import { Dim, Vec3, Vec4 } from "@/src/utils/vector";
import { DimStyle, dimStyleColor } from "@/src/llm/walkthrough/WalkthroughTools";
import { ConstellationDim, constellationDimColor } from "../ConstellationDimStyle";
import { constellationBlockDimension } from "../ConstellationAnnotations";

function createAtTime(wt: IConstellationWalkthrough, start: number, duration?: number, wait?: number): ITimeInfo {
    duration = duration ?? 0;
    wait = wait ?? 0;
    let info: ITimeInfo = {
        name: '', start, duration, wait,
        t: duration === 0 ? (wt.time > start ? 1 : 0) : clamp((wt.time - start) / duration, 0, 1),
        active: wt.time > start,
    };
    wt.times.push(info);
    wt.phaseLength = Math.max(wt.phaseLength, start + duration + wait);
    return info;
}

export function constellationPhaseTools(state: IConstellationProgramState) {
    let wt = state.walkthrough;

    function c_str(str: string, duration: number = 0.3, dim?: ConstellationDim) {
        let color = dim ? constellationDimColor(dim) : new Vec4(0.1, 0.5, 0.8, 1);
        return { str, duration, start: 0, t: 0.0, color };
    }

    function c_blockRef(str: string, blk: IBlkDef | IBlkDef[], style?: DimStyle) {
        let firstBlk = Array.isArray(blk) ? blk[0] : blk;
        style ??= firstBlk.t === 'i' ? DimStyle.Intermediates : firstBlk.t === 'w' ? DimStyle.Weights : DimStyle.Aggregates;
        return { str, duration: 0, start: 0, t: 0.0, color: dimStyleColor(style), blk };
    }

    function c_dimRef(str: string, dim: ConstellationDim) {
        return { str, duration: 0, start: 0, t: 0.0, color: constellationDimColor(dim), dim: dim as unknown as DimStyle };
    }

    function atTime(start: number, duration?: number, wait?: number): ITimeInfo {
        return createAtTime(wt, start, duration, wait);
    }

    function afterTime(prev: ITimeInfo | null, duration: number, wait?: number): ITimeInfo {
        prev = prev ?? wt.times[wt.times.length - 1];
        return atTime(prev.start + prev.duration + prev.wait, duration, wait);
    }

    function cleanup(t: ITimeInfo, times: ITimeInfo[] = wt.times as ITimeInfo[]) {
        if (t.t > 0.0) {
            for (let prevTime of times) {
                prevTime.t = 1.0 - t.t;
                if (t.t >= 1.0) prevTime.active = false;
            }
        }
    }

    function breakAfter(evt?: ITimeInfo) {
        evt = evt ?? wt.times[wt.times.length - 1] as ITimeInfo;
        if (!evt) return;
        let breakEvt = afterTime(evt, 0.001);
        if (wt.running && wt.time - wt.dt < breakEvt.start && wt.time >= breakEvt.start) {
            wt.running = false;
            wt.speed = 1.0;
            wt.time = breakEvt.start + breakEvt.duration;
        }
        breakEvt.isBreak = true;
    }

    function commentary(prev?: ITimeInfo | null, duration?: number) {
        return llmCommentary(wt as any, prev, duration);
    }

    function dimExcept(keepBlocks: IBlkDef[], dimOpacity: number = 0.12) {
        let keepSet = new Set(keepBlocks);
        for (let cube of state.layout.cubes) {
            if (!keepSet.has(cube)) cube.opacity = dimOpacity;
        }
    }

    function drawDimLabels(blk: IBlkDef, t: number) {
        let dimX = blk.dimX as unknown as ConstellationDim;
        let dimY = blk.dimY as unknown as ConstellationDim;
        if (dimX && (dimX as number) !== 0 && blk.cx > 2)
            constellationBlockDimension(state.render, state.layout, blk, Dim.X, dimX, t);
        if (dimY && (dimY as number) !== 0 && blk.cy > 2)
            constellationBlockDimension(state.render, state.layout, blk, Dim.Y, dimY, t);
    }

    return { atTime, afterTime, cleanup, breakAfter, c_str, c_blockRef, c_dimRef, commentary, dimExcept, drawDimLabels };
}

export function setConstellationInitialCamera(state: IConstellationProgramState, target: Vec3, rot: Vec3) {
    let wt = state.walkthrough;
    wt.cameraInitial = { angle: rot, center: target };
    wt.phaseTransitiveData ??= {};
    let data = wt.phaseTransitiveData;
    if (wt.time === 0) {
        data.cameraSrc ??= { angle: state.camera.angle, center: state.camera.center };
        data.cameraT ??= 0;
        if (data.cameraT < 1) {
            let src = data.cameraSrc;
            let dest = wt.cameraInitial;
            let t = data.cameraT;
            state.camera.angle = src.angle.lerp(dest.angle, t);
            state.camera.center = src.center.lerp(dest.center, t);
            data.cameraT = t + wt.viewDt / 1000 * 1.5;
            wt.markDirty();
        }
    }
}

export function runConstellationWalkthrough(state: IConstellationProgramState, view: IRenderView) {
    let wt = state.walkthrough;
    wt.viewDt = view.dt;

    if (wt.running) {
        let dtSeconds = view.dt / 1000 * wt.speed;
        wt.time += dtSeconds;
        wt.dt = dtSeconds;
        if (wt.time > wt.phaseLength) { wt.running = false; wt.time = wt.phaseLength; }
        view.markDirty();
    }

    if (wt.prevPhase !== wt.phase) wt.phaseTransitiveData = null;

    wt.cameraInitial = null;
    wt.times = [];
    wt.phaseLength = 0;
    wt.dimHighlightBlocks = null;
    wt.commentary = null;

    let args: IConstellationWalkthroughArgs = {
        state, layout: state.layout, walkthrough: wt,
        tools: constellationPhaseTools(state),
    };

    switch (wt.phase) {
        case ConstellationPhase.Overview: constellationPhase00_Overview(args); break;
        case ConstellationPhase.TokenInput: constellationPhase01_TokenInput(args); break;
        case ConstellationPhase.Encoders: constellationPhase02_Encoders(args); break;
        case ConstellationPhase.LatentSpace: constellationPhase03_LatentSpace(args); break;
        case ConstellationPhase.DecodersAndLosses: constellationPhase04_DecodersAndLosses(args); break;
    }

    wt.prevPhase = wt.phase;
    wt.prevTime = wt.time;
}
