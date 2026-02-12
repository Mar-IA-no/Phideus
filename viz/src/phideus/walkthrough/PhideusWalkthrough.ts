import { ICameraPos } from "@/src/llm/Camera";
import { IBlkDef } from "@/src/llm/GptModelLayout";
import { IRenderView } from "@/src/llm/render/modelRender";
import { commentary as llmCommentary, ICommentary, ICommentaryRes, ITimeInfo } from "@/src/llm/walkthrough/WalkthroughTools";
import { IPhideusProgramState } from "../PhideusProgram";
import { IPhideusModelLayout } from "../PhideusModelLayout";
import { phideusPhase00_Overview } from "./Phase00_Overview";
import { phideusPhase01_AudioCNN } from "./Phase01_AudioCNN";
import { phideusPhase02_AudioTransformer } from "./Phase02_AudioTransformer";
import { phideusPhase03_MIDIEmbedding } from "./Phase03_MIDIEmbedding";
import { phideusPhase04_MIDITransformer } from "./Phase04_MIDITransformer";
import { phideusPhase05_Projections } from "./Phase05_Projections";
import { phideusPhase06_VICReg } from "./Phase06_VICReg";
import { phideusPhase07_RunDConfig } from "./Phase07_RunDConfig";

export enum PhideusPhase {
    None,
    Overview,
    AudioCNN,
    AudioTransformer,
    MIDIEmbedding,
    MIDITransformer,
    Projections,
    VICReg,
    RunDConfig,
}

export enum PhideusPhaseGroup {
    Overview,
    AudioTower,
    MIDITower,
    SharedSpace,
}

export interface IPhideusPhaseGroup {
    groupId: PhideusPhaseGroup;
    title: string;
    phases: IPhideusPhaseDef[];
}

export interface IPhideusPhaseDef {
    id: PhideusPhase;
    title: string;
}

export interface IPhideusWalkthrough {
    phase: PhideusPhase;
    time: number;
    viewDt: number;
    dt: number;
    prevTime: number;
    prevPhase: PhideusPhase;
    running: boolean;
    speed: number;
    cameraInitial: ICameraPos | null;
    commentary: ICommentaryRes | null;
    times: (ITimeInfo | ICommentary)[];
    phaseLength: number;
    dimHighlightBlocks: IBlkDef[] | null;
    markDirty: () => void;
    phaseData: Map<PhideusPhase, any>;
    phaseTransitiveData: any;
    phaseList: IPhideusPhaseGroup[];
}

export interface IPhideusWalkthroughArgs {
    state: IPhideusProgramState;
    layout: IPhideusModelLayout;
    walkthrough: IPhideusWalkthrough;
    tools: ReturnType<typeof phideusPhaseTools>;
}

export function initPhideusWalkthrough(): IPhideusWalkthrough {
    return {
        phase: PhideusPhase.Overview,
        time: 0,
        viewDt: 0,
        dt: 0,
        prevTime: 0,
        prevPhase: PhideusPhase.None,
        running: false,
        speed: 1,
        cameraInitial: null,
        commentary: null,
        times: [],
        phaseLength: 0,
        dimHighlightBlocks: null,
        markDirty: () => {},
        phaseData: new Map(),
        phaseTransitiveData: null,
        phaseList: [{
            groupId: PhideusPhaseGroup.Overview,
            title: 'Introduction',
            phases: [
                { id: PhideusPhase.Overview, title: 'Architecture Overview' },
            ],
        }, {
            groupId: PhideusPhaseGroup.AudioTower,
            title: 'Audio Tower',
            phases: [
                { id: PhideusPhase.AudioCNN, title: 'CNN Feature Extractor' },
                { id: PhideusPhase.AudioTransformer, title: 'Audio Transformer' },
            ],
        }, {
            groupId: PhideusPhaseGroup.MIDITower,
            title: 'MIDI Tower',
            phases: [
                { id: PhideusPhase.MIDIEmbedding, title: 'Event Embedding' },
                { id: PhideusPhase.MIDITransformer, title: 'MIDI Transformer' },
            ],
        }, {
            groupId: PhideusPhaseGroup.SharedSpace,
            title: 'Shared Space',
            phases: [
                { id: PhideusPhase.Projections, title: 'Projection Heads' },
                { id: PhideusPhase.VICReg, title: 'VICReg Loss' },
                { id: PhideusPhase.RunDConfig, title: 'Run D Configuration' },
            ],
        }],
    };
}

export function phaseToPhideusGroup(wt: IPhideusWalkthrough) {
    return wt.phaseList.find(g => g.phases.find(p => p.id === wt.phase))!;
}

export function jumpPhideusPhase(wt: IPhideusWalkthrough, delta: number) {
    let group = phaseToPhideusGroup(wt);
    let groupIdx = wt.phaseList.indexOf(group);
    let phaseGroupIdx = group.phases.findIndex((p: IPhideusPhaseDef) => p.id === wt.phase);
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
import { Dim, Vec4 } from "@/src/utils/vector";
import { DimStyle, dimStyleColor } from "@/src/llm/walkthrough/WalkthroughTools";
import { PhideusDim, phideusDimColor } from "../PhideusDimStyle";
import { phideusBlockDimension } from "../PhideusAnnotations";

function createAtTime(wt: IPhideusWalkthrough, start: number, duration?: number, wait?: number): ITimeInfo {
    duration = duration ?? 0;
    wait = wait ?? 0;
    let info: ITimeInfo = {
        name: '',
        start,
        duration,
        wait,
        t: duration === 0 ? (wt.time > start ? 1 : 0) : clamp((wt.time - start) / duration, 0, 1),
        active: wt.time > start,
    };
    wt.times.push(info);
    wt.phaseLength = Math.max(wt.phaseLength, start + duration + wait);
    return info;
}

export function phideusPhaseTools(state: IPhideusProgramState) {
    let wt = state.walkthrough;

    function c_str(str: string, duration: number = 0.3, dim?: PhideusDim) {
        let color = dim ? phideusDimColor(dim) : new Vec4(0.1, 0.5, 0.8, 1);
        return { str, duration, start: 0, t: 0.0, color };
    }

    function c_blockRef(str: string, blk: IBlkDef | IBlkDef[], style?: DimStyle) {
        let firstBlk = Array.isArray(blk) ? blk[0] : blk;
        style ??= firstBlk.t === 'i' ? DimStyle.Intermediates : firstBlk.t === 'w' ? DimStyle.Weights : DimStyle.Aggregates;
        return { str, duration: 0, start: 0, t: 0.0, color: dimStyleColor(style), blk };
    }

    function c_dimRef(str: string, dim: PhideusDim) {
        return { str, duration: 0, start: 0, t: 0.0, color: phideusDimColor(dim), dim: dim as unknown as DimStyle };
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
                if (t.t >= 1.0) {
                    prevTime.active = false;
                }
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

    /** Dim all blocks except the keepBlocks list. Call at top of phase for focus. */
    function dimExcept(keepBlocks: IBlkDef[], dimOpacity: number = 0.12) {
        let keepSet = new Set(keepBlocks);
        for (let cube of state.layout.cubes) {
            if (!keepSet.has(cube)) {
                cube.opacity = dimOpacity;
            }
        }
    }

    /** Draw dimension labels (|--- label ---|) on a block's X and Y edges. */
    function drawDimLabels(blk: IBlkDef, t: number) {
        let dimX = blk.dimX as unknown as PhideusDim;
        let dimY = blk.dimY as unknown as PhideusDim;
        if (dimX && (dimX as number) !== 0 && blk.cx > 2) {
            phideusBlockDimension(state.render, state.layout, blk, Dim.X, dimX, t, state.camera);
        }
        if (dimY && (dimY as number) !== 0 && blk.cy > 2) {
            phideusBlockDimension(state.render, state.layout, blk, Dim.Y, dimY, t, state.camera);
        }
    }

    return { atTime, afterTime, cleanup, breakAfter, c_str, c_blockRef, c_dimRef, commentary, dimExcept, drawDimLabels };
}

export function setPhideusInitialCamera(state: IPhideusProgramState, target: import("@/src/utils/vector").Vec3, rot: import("@/src/utils/vector").Vec3) {
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

export function movePhideusCameraTo(state: IPhideusProgramState, time: ITimeInfo, target: import("@/src/utils/vector").Vec3, rot: import("@/src/utils/vector").Vec3) {
    let wt = state.walkthrough;
    let phaseData = wt.phaseData.get(wt.phase);
    if (!phaseData) {
        wt.phaseData.set(wt.phase, phaseData = { cameraData: null });
    }
    if (!phaseData.cameraData) {
        phaseData.cameraData = new Map<number, any>();
    }

    let prevTime = [...phaseData.cameraData.entries()].filter(([t]: [number, any]) => t < time.start).pop()?.[1];

    let camData = phaseData.cameraData.get(time.start);
    if (!camData) {
        phaseData.cameraData.set(time.start, camData = {
            initialCaptured: prevTime ? undefined : wt.cameraInitial ?? {
                angle: state.camera.angle,
                center: state.camera.center,
            },
            target: { angle: rot, center: target },
        });
    }

    let src = prevTime?.target ?? wt.cameraInitial ?? camData.initialCaptured;
    let dest: ICameraPos = { center: target, angle: rot };

    let isMoving = wt.running || wt.time !== wt.prevTime;
    let prevWasActive = wt.prevTime >= time.start && wt.prevTime <= time.start + time.duration;

    if (src && isMoving && (time.active || prevWasActive)) {
        let t = time.t;
        state.camera.angle = src.angle.lerp(dest.angle, t);
        state.camera.center = src.center.lerp(dest.center, t);
    }
}

export function runPhideusWalkthrough(state: IPhideusProgramState, view: IRenderView) {
    let wt = state.walkthrough;
    wt.viewDt = view.dt;

    if (wt.running) {
        let dtSeconds = view.dt / 1000 * wt.speed;
        wt.time += dtSeconds;
        wt.dt = dtSeconds;

        if (wt.time > wt.phaseLength) {
            wt.running = false;
            wt.time = wt.phaseLength;
        }

        view.markDirty();
    }

    if (wt.prevPhase !== wt.phase) {
        wt.phaseTransitiveData = null;
    }

    wt.cameraInitial = null;
    wt.times = [];
    wt.phaseLength = 0;
    wt.dimHighlightBlocks = null;
    wt.commentary = null;

    let args: IPhideusWalkthroughArgs = {
        state,
        layout: state.layout,
        walkthrough: wt,
        tools: phideusPhaseTools(state),
    };

    // Dispatch to active phase
    switch (wt.phase) {
        case PhideusPhase.Overview:
            phideusPhase00_Overview(args);
            break;
        case PhideusPhase.AudioCNN:
            phideusPhase01_AudioCNN(args);
            break;
        case PhideusPhase.AudioTransformer:
            phideusPhase02_AudioTransformer(args);
            break;
        case PhideusPhase.MIDIEmbedding:
            phideusPhase03_MIDIEmbedding(args);
            break;
        case PhideusPhase.MIDITransformer:
            phideusPhase04_MIDITransformer(args);
            break;
        case PhideusPhase.Projections:
            phideusPhase05_Projections(args);
            break;
        case PhideusPhase.VICReg:
            phideusPhase06_VICReg(args);
            break;
        case PhideusPhase.RunDConfig:
            phideusPhase07_RunDConfig(args);
            break;
    }

    wt.prevPhase = wt.phase;
    wt.prevTime = wt.time;
}
