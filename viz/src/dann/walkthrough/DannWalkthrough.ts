import { ICameraPos } from "@/src/llm/Camera";
import { IBlkDef } from "@/src/llm/GptModelLayout";
import { IRenderView } from "@/src/llm/render/modelRender";
import { commentary as llmCommentary, ICommentary, ICommentaryRes, ITimeInfo } from "@/src/llm/walkthrough/WalkthroughTools";
import { IDannProgramState } from "../DannProgram";
import { IDannModelLayout } from "../DannModelLayout";
import { dannPhase00_Overview } from "./Phase00_Overview";
import { dannPhase01_Encoders } from "./Phase01_Encoders";
import { dannPhase02_GRL } from "./Phase02_GRL";
import { dannPhase03_Losses } from "./Phase03_Losses";

export enum DannPhase {
    None,
    Overview,
    Encoders,
    GRL,
    Losses,
}

export enum DannPhaseGroup {
    Overview,
    Architecture,
    Training,
}

export interface IDannPhaseGroup {
    groupId: DannPhaseGroup;
    title: string;
    phases: IDannPhaseDef[];
}

export interface IDannPhaseDef {
    id: DannPhase;
    title: string;
}

export interface IDannWalkthrough {
    phase: DannPhase;
    time: number;
    viewDt: number;
    dt: number;
    prevTime: number;
    prevPhase: DannPhase;
    running: boolean;
    speed: number;
    cameraInitial: ICameraPos | null;
    commentary: ICommentaryRes | null;
    times: (ITimeInfo | ICommentary)[];
    phaseLength: number;
    dimHighlightBlocks: IBlkDef[] | null;
    markDirty: () => void;
    phaseData: Map<DannPhase, any>;
    phaseTransitiveData: any;
    phaseList: IDannPhaseGroup[];
}

export interface IDannWalkthroughArgs {
    state: IDannProgramState;
    layout: IDannModelLayout;
    walkthrough: IDannWalkthrough;
    tools: ReturnType<typeof dannPhaseTools>;
}

export function initDannWalkthrough(): IDannWalkthrough {
    return {
        phase: DannPhase.Overview,
        time: 0, viewDt: 0, dt: 0, prevTime: 0,
        prevPhase: DannPhase.None,
        running: false, speed: 1,
        cameraInitial: null, commentary: null,
        times: [], phaseLength: 0,
        dimHighlightBlocks: null,
        markDirty: () => {},
        phaseData: new Map(),
        phaseTransitiveData: null,
        phaseList: [{
            groupId: DannPhaseGroup.Overview,
            title: 'Introduction',
            phases: [{ id: DannPhase.Overview, title: 'Architecture Overview' }],
        }, {
            groupId: DannPhaseGroup.Architecture,
            title: 'Architecture',
            phases: [
                { id: DannPhase.Encoders, title: 'Encoder Towers' },
                { id: DannPhase.GRL, title: 'Gradient Reversal Layer' },
            ],
        }, {
            groupId: DannPhaseGroup.Training,
            title: 'Training',
            phases: [{ id: DannPhase.Losses, title: 'VICReg + DANN Loss' }],
        }],
    };
}

export function phaseToDannGroup(wt: IDannWalkthrough) {
    return wt.phaseList.find(g => g.phases.find(p => p.id === wt.phase))!;
}

export function jumpDannPhase(wt: IDannWalkthrough, delta: number) {
    let group = phaseToDannGroup(wt);
    let groupIdx = wt.phaseList.indexOf(group);
    let phaseGroupIdx = group.phases.findIndex((p: IDannPhaseDef) => p.id === wt.phase);
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
import { DannDim, dannDimColor } from "../DannDimStyle";
import { dannBlockDimension } from "../DannAnnotations";

function createAtTime(wt: IDannWalkthrough, start: number, duration?: number, wait?: number): ITimeInfo {
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

export function dannPhaseTools(state: IDannProgramState) {
    let wt = state.walkthrough;

    function c_str(str: string, duration: number = 0.3, dim?: DannDim) {
        let color = dim ? dannDimColor(dim) : new Vec4(0.1, 0.5, 0.8, 1);
        return { str, duration, start: 0, t: 0.0, color };
    }

    function c_blockRef(str: string, blk: IBlkDef | IBlkDef[], style?: DimStyle) {
        let firstBlk = Array.isArray(blk) ? blk[0] : blk;
        style ??= firstBlk.t === 'i' ? DimStyle.Intermediates : firstBlk.t === 'w' ? DimStyle.Weights : DimStyle.Aggregates;
        return { str, duration: 0, start: 0, t: 0.0, color: dimStyleColor(style), blk };
    }

    function c_dimRef(str: string, dim: DannDim) {
        return { str, duration: 0, start: 0, t: 0.0, color: dannDimColor(dim), dim: dim as unknown as DimStyle };
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
        let dimX = blk.dimX as unknown as DannDim;
        let dimY = blk.dimY as unknown as DannDim;
        if (dimX && (dimX as number) !== 0 && blk.cx > 2)
            dannBlockDimension(state.render, state.layout, blk, Dim.X, dimX, t);
        if (dimY && (dimY as number) !== 0 && blk.cy > 2)
            dannBlockDimension(state.render, state.layout, blk, Dim.Y, dimY, t);
    }

    return { atTime, afterTime, cleanup, breakAfter, c_str, c_blockRef, c_dimRef, commentary, dimExcept, drawDimLabels };
}

export function setDannInitialCamera(state: IDannProgramState, target: Vec3, rot: Vec3) {
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

export function runDannWalkthrough(state: IDannProgramState, view: IRenderView) {
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

    let args: IDannWalkthroughArgs = {
        state, layout: state.layout, walkthrough: wt,
        tools: dannPhaseTools(state),
    };

    switch (wt.phase) {
        case DannPhase.Overview: dannPhase00_Overview(args); break;
        case DannPhase.Encoders: dannPhase01_Encoders(args); break;
        case DannPhase.GRL: dannPhase02_GRL(args); break;
        case DannPhase.Losses: dannPhase03_Losses(args); break;
    }

    wt.prevPhase = wt.phase;
    wt.prevTime = wt.time;
}
