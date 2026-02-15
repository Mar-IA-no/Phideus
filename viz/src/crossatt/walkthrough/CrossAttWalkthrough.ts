import { ICameraPos } from "@/src/llm/Camera";
import { IBlkDef } from "@/src/llm/GptModelLayout";
import { IRenderView } from "@/src/llm/render/modelRender";
import { commentary as llmCommentary, ICommentary, ICommentaryRes, ITimeInfo } from "@/src/llm/walkthrough/WalkthroughTools";
import { ICrossAttProgramState } from "../CrossAttProgram";
import { ICrossAttModelLayout } from "../CrossAttModelLayout";
import { crossAttPhase00_Overview } from "./Phase00_Overview";
import { crossAttPhase01_AudioCNN } from "./Phase01_AudioCNN";
import { crossAttPhase02_AudioCrossAtt } from "./Phase02_AudioCrossAtt";
import { crossAttPhase03_MidiCrossAtt } from "./Phase03_MidiCrossAtt";
import { crossAttPhase04_Transformers } from "./Phase04_Transformers";
import { crossAttPhase05_VICReg } from "./Phase05_VICReg";

export enum CrossAttPhase {
    None,
    Overview,
    AudioCNN,
    AudioCrossAtt,
    MidiCrossAtt,
    Transformers,
    VICReg,
}

export enum CrossAttPhaseGroup {
    Introduction,
    AudioPath,
    MidiPath,
    Training,
}

export interface ICrossAttPhaseGroup {
    groupId: CrossAttPhaseGroup;
    title: string;
    phases: ICrossAttPhaseDef[];
}

export interface ICrossAttPhaseDef {
    id: CrossAttPhase;
    title: string;
}

export interface ICrossAttWalkthrough {
    phase: CrossAttPhase;
    time: number;
    viewDt: number;
    dt: number;
    prevTime: number;
    prevPhase: CrossAttPhase;
    running: boolean;
    speed: number;
    cameraInitial: ICameraPos | null;
    commentary: ICommentaryRes | null;
    times: (ITimeInfo | ICommentary)[];
    phaseLength: number;
    dimHighlightBlocks: IBlkDef[] | null;
    markDirty: () => void;
    phaseData: Map<CrossAttPhase, any>;
    phaseTransitiveData: any;
    phaseList: ICrossAttPhaseGroup[];
}

export interface ICrossAttWalkthroughArgs {
    state: ICrossAttProgramState;
    layout: ICrossAttModelLayout;
    walkthrough: ICrossAttWalkthrough;
    tools: ReturnType<typeof crossAttPhaseTools>;
}

export function initCrossAttWalkthrough(): ICrossAttWalkthrough {
    return {
        phase: CrossAttPhase.Overview,
        time: 0, viewDt: 0, dt: 0, prevTime: 0,
        prevPhase: CrossAttPhase.None,
        running: false, speed: 1,
        cameraInitial: null, commentary: null,
        times: [], phaseLength: 0,
        dimHighlightBlocks: null,
        markDirty: () => {},
        phaseData: new Map(),
        phaseTransitiveData: null,
        phaseList: [{
            groupId: CrossAttPhaseGroup.Introduction,
            title: 'Introduction',
            phases: [{ id: CrossAttPhase.Overview, title: 'Architecture Overview' }],
        }, {
            groupId: CrossAttPhaseGroup.AudioPath,
            title: 'Audio Path',
            phases: [
                { id: CrossAttPhase.AudioCNN, title: 'Audio CNN + PosEmb' },
                { id: CrossAttPhase.AudioCrossAtt, title: 'Audio Cross-Attention' },
            ],
        }, {
            groupId: CrossAttPhaseGroup.MidiPath,
            title: 'MIDI Path',
            phases: [{ id: CrossAttPhase.MidiCrossAtt, title: 'MIDI Cross-Attention' }],
        }, {
            groupId: CrossAttPhaseGroup.Training,
            title: 'Training',
            phases: [
                { id: CrossAttPhase.Transformers, title: 'Transformers + Pooling' },
                { id: CrossAttPhase.VICReg, title: 'VICReg Loss' },
            ],
        }],
    };
}

export function phaseToCrossAttGroup(wt: ICrossAttWalkthrough) {
    return wt.phaseList.find(g => g.phases.find(p => p.id === wt.phase))!;
}

export function jumpCrossAttPhase(wt: ICrossAttWalkthrough, delta: number) {
    let group = phaseToCrossAttGroup(wt);
    let groupIdx = wt.phaseList.indexOf(group);
    let phaseGroupIdx = group.phases.findIndex((p: ICrossAttPhaseDef) => p.id === wt.phase);
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
import { CrossAttDim, crossAttDimColor } from "../CrossAttDimStyle";
import { crossAttBlockDimension } from "../CrossAttAnnotations";

function createAtTime(wt: ICrossAttWalkthrough, start: number, duration?: number, wait?: number): ITimeInfo {
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

export function crossAttPhaseTools(state: ICrossAttProgramState) {
    let wt = state.walkthrough;

    function c_str(str: string, duration: number = 0.3, dim?: CrossAttDim) {
        let color = dim ? crossAttDimColor(dim) : new Vec4(0.1, 0.5, 0.8, 1);
        return { str, duration, start: 0, t: 0.0, color };
    }

    function c_blockRef(str: string, blk: IBlkDef | IBlkDef[], style?: DimStyle) {
        let firstBlk = Array.isArray(blk) ? blk[0] : blk;
        style ??= firstBlk.t === 'i' ? DimStyle.Intermediates : firstBlk.t === 'w' ? DimStyle.Weights : DimStyle.Aggregates;
        return { str, duration: 0, start: 0, t: 0.0, color: dimStyleColor(style), blk };
    }

    function c_dimRef(str: string, dim: CrossAttDim) {
        return { str, duration: 0, start: 0, t: 0.0, color: crossAttDimColor(dim), dim: dim as unknown as DimStyle };
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
        let dimX = blk.dimX as unknown as CrossAttDim;
        let dimY = blk.dimY as unknown as CrossAttDim;
        if (dimX && (dimX as number) !== 0 && blk.cx > 2)
            crossAttBlockDimension(state.render, state.layout, blk, Dim.X, dimX, t);
        if (dimY && (dimY as number) !== 0 && blk.cy > 2)
            crossAttBlockDimension(state.render, state.layout, blk, Dim.Y, dimY, t);
    }

    return { atTime, afterTime, cleanup, breakAfter, c_str, c_blockRef, c_dimRef, commentary, dimExcept, drawDimLabels };
}

export function setCrossAttInitialCamera(state: ICrossAttProgramState, target: Vec3, rot: Vec3) {
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

export function runCrossAttWalkthrough(state: ICrossAttProgramState, view: IRenderView) {
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

    let args: ICrossAttWalkthroughArgs = {
        state, layout: state.layout, walkthrough: wt,
        tools: crossAttPhaseTools(state),
    };

    switch (wt.phase) {
        case CrossAttPhase.Overview: crossAttPhase00_Overview(args); break;
        case CrossAttPhase.AudioCNN: crossAttPhase01_AudioCNN(args); break;
        case CrossAttPhase.AudioCrossAtt: crossAttPhase02_AudioCrossAtt(args); break;
        case CrossAttPhase.MidiCrossAtt: crossAttPhase03_MidiCrossAtt(args); break;
        case CrossAttPhase.Transformers: crossAttPhase04_Transformers(args); break;
        case CrossAttPhase.VICReg: crossAttPhase05_VICReg(args); break;
    }

    wt.prevPhase = wt.phase;
    wt.prevTime = wt.time;
}
