import { ICameraPos } from "@/src/llm/Camera";
import { IBlkDef } from "@/src/llm/GptModelLayout";
import { IRenderView } from "@/src/llm/render/modelRender";
import { commentary as llmCommentary, ICommentary, ICommentaryRes, ITimeInfo } from "@/src/llm/walkthrough/WalkthroughTools";
import { IT3TowerProgramState } from "../T3TowerProgram";
import { IT3TowerModelLayout } from "../T3TowerModelLayout";
import { t3TowerPhase00_Overview } from "./Phase00_Overview";
import { t3TowerPhase01_AudioCNN } from "./Phase01_AudioCNN";
import { t3TowerPhase02_AudioTransformer } from "./Phase02_AudioTransformer";
import { t3TowerPhase03_AudioProjection } from "./Phase03_AudioProjection";
import { t3TowerPhase04_MIDIEmbedding } from "./Phase04_MIDIEmbedding";
import { t3TowerPhase05_MIDITransformer } from "./Phase05_MIDITransformer";
import { t3TowerPhase06_MIDIProjection } from "./Phase06_MIDIProjection";
import { t3TowerPhase07_T3Tower } from "./Phase07_T3Tower";
import { t3TowerPhase08_SharedSpace } from "./Phase08_SharedSpace";
import { t3TowerPhase09_VICReg } from "./Phase09_VICReg";

export enum T3TowerPhase {
    None,
    Overview,
    AudioCNN,
    AudioTransformer,
    AudioProjection,
    MIDIEmbedding,
    MIDITransformer,
    MIDIProjection,
    T3Tower,
    SharedSpace,
    VICReg,
}

export enum T3TowerPhaseGroup {
    Introduction,
    AudioTower,
    MIDITower,
    T3Tower,
    SharedSpace,
}

export interface IT3TowerPhaseGroup {
    groupId: T3TowerPhaseGroup;
    title: string;
    phases: IT3TowerPhaseDef[];
}

export interface IT3TowerPhaseDef {
    id: T3TowerPhase;
    title: string;
}

export interface IT3TowerWalkthrough {
    phase: T3TowerPhase;
    time: number;
    viewDt: number;
    dt: number;
    prevTime: number;
    prevPhase: T3TowerPhase;
    running: boolean;
    speed: number;
    cameraInitial: ICameraPos | null;
    commentary: ICommentaryRes | null;
    times: (ITimeInfo | ICommentary)[];
    phaseLength: number;
    dimHighlightBlocks: IBlkDef[] | null;
    markDirty: () => void;
    phaseData: Map<T3TowerPhase, any>;
    phaseTransitiveData: any;
    phaseList: IT3TowerPhaseGroup[];
}

export interface IT3TowerWalkthroughArgs {
    state: IT3TowerProgramState;
    layout: IT3TowerModelLayout;
    walkthrough: IT3TowerWalkthrough;
    tools: ReturnType<typeof t3TowerPhaseTools>;
}

export function initT3TowerWalkthrough(): IT3TowerWalkthrough {
    return {
        phase: T3TowerPhase.Overview,
        time: 0,
        viewDt: 0,
        dt: 0,
        prevTime: 0,
        prevPhase: T3TowerPhase.None,
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
            groupId: T3TowerPhaseGroup.Introduction,
            title: 'Introduction',
            phases: [
                { id: T3TowerPhase.Overview, title: 'Architecture Overview' },
            ],
        }, {
            groupId: T3TowerPhaseGroup.AudioTower,
            title: 'Audio Tower',
            phases: [
                { id: T3TowerPhase.AudioCNN, title: 'CNN Feature Extractor' },
                { id: T3TowerPhase.AudioTransformer, title: 'Audio Transformer' },
                { id: T3TowerPhase.AudioProjection, title: 'Audio Projection' },
            ],
        }, {
            groupId: T3TowerPhaseGroup.MIDITower,
            title: 'MIDI Tower',
            phases: [
                { id: T3TowerPhase.MIDIEmbedding, title: 'Event Embedding' },
                { id: T3TowerPhase.MIDITransformer, title: 'MIDI Transformer' },
                { id: T3TowerPhase.MIDIProjection, title: 'MIDI Projection' },
            ],
        }, {
            groupId: T3TowerPhaseGroup.T3Tower,
            title: 'T3 Tower',
            phases: [
                { id: T3TowerPhase.T3Tower, title: 'T3 Lightweight Transformer' },
            ],
        }, {
            groupId: T3TowerPhaseGroup.SharedSpace,
            title: 'Shared Space',
            phases: [
                { id: T3TowerPhase.SharedSpace, title: '3-way Shared Space' },
                { id: T3TowerPhase.VICReg, title: '3-way VICReg Loss' },
            ],
        }],
    };
}

export function phaseToT3TowerGroup(wt: IT3TowerWalkthrough) {
    return wt.phaseList.find(g => g.phases.find(p => p.id === wt.phase))!;
}

export function jumpT3TowerPhase(wt: IT3TowerWalkthrough, delta: number) {
    let group = phaseToT3TowerGroup(wt);
    let groupIdx = wt.phaseList.indexOf(group);
    let phaseGroupIdx = group.phases.findIndex((p: IT3TowerPhaseDef) => p.id === wt.phase);
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
import { T3TowerDim, t3TowerDimColor } from "../T3TowerDimStyle";
import { t3TowerBlockDimension } from "../T3TowerAnnotations";

function createAtTime(wt: IT3TowerWalkthrough, start: number, duration?: number, wait?: number): ITimeInfo {
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

export function t3TowerPhaseTools(state: IT3TowerProgramState) {
    let wt = state.walkthrough;

    function c_str(str: string, duration: number = 0.3, dim?: T3TowerDim) {
        let color = dim ? t3TowerDimColor(dim) : new Vec4(0.1, 0.5, 0.8, 1);
        return { str, duration, start: 0, t: 0.0, color };
    }

    function c_blockRef(str: string, blk: IBlkDef | IBlkDef[], style?: DimStyle) {
        let firstBlk = Array.isArray(blk) ? blk[0] : blk;
        style ??= firstBlk.t === 'i' ? DimStyle.Intermediates : firstBlk.t === 'w' ? DimStyle.Weights : DimStyle.Aggregates;
        return { str, duration: 0, start: 0, t: 0.0, color: dimStyleColor(style), blk };
    }

    function c_dimRef(str: string, dim: T3TowerDim) {
        return { str, duration: 0, start: 0, t: 0.0, color: t3TowerDimColor(dim), dim: dim as unknown as DimStyle };
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

    function dimExcept(keepBlocks: IBlkDef[], dimOpacity: number = 0.12) {
        let keepSet = new Set(keepBlocks);
        for (let cube of state.layout.cubes) {
            if (!keepSet.has(cube)) {
                cube.opacity = dimOpacity;
            }
        }
    }

    function drawDimLabels(blk: IBlkDef, t: number) {
        let dimX = blk.dimX as unknown as T3TowerDim;
        let dimY = blk.dimY as unknown as T3TowerDim;
        if (dimX && (dimX as number) !== 0 && blk.cx > 2) {
            t3TowerBlockDimension(state.render, state.layout, blk, Dim.X, dimX, t, state.camera);
        }
        if (dimY && (dimY as number) !== 0 && blk.cy > 2) {
            t3TowerBlockDimension(state.render, state.layout, blk, Dim.Y, dimY, t, state.camera);
        }
    }

    return { atTime, afterTime, cleanup, breakAfter, c_str, c_blockRef, c_dimRef, commentary, dimExcept, drawDimLabels };
}

export function setT3TowerInitialCamera(state: IT3TowerProgramState, target: import("@/src/utils/vector").Vec3, rot: import("@/src/utils/vector").Vec3) {
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

export function moveT3TowerCameraTo(state: IT3TowerProgramState, time: ITimeInfo, target: import("@/src/utils/vector").Vec3, rot: import("@/src/utils/vector").Vec3) {
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

export function runT3TowerWalkthrough(state: IT3TowerProgramState, view: IRenderView) {
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

    let args: IT3TowerWalkthroughArgs = {
        state,
        layout: state.layout,
        walkthrough: wt,
        tools: t3TowerPhaseTools(state),
    };

    switch (wt.phase) {
        case T3TowerPhase.Overview:
            t3TowerPhase00_Overview(args);
            break;
        case T3TowerPhase.AudioCNN:
            t3TowerPhase01_AudioCNN(args);
            break;
        case T3TowerPhase.AudioTransformer:
            t3TowerPhase02_AudioTransformer(args);
            break;
        case T3TowerPhase.AudioProjection:
            t3TowerPhase03_AudioProjection(args);
            break;
        case T3TowerPhase.MIDIEmbedding:
            t3TowerPhase04_MIDIEmbedding(args);
            break;
        case T3TowerPhase.MIDITransformer:
            t3TowerPhase05_MIDITransformer(args);
            break;
        case T3TowerPhase.MIDIProjection:
            t3TowerPhase06_MIDIProjection(args);
            break;
        case T3TowerPhase.T3Tower:
            t3TowerPhase07_T3Tower(args);
            break;
        case T3TowerPhase.SharedSpace:
            t3TowerPhase08_SharedSpace(args);
            break;
        case T3TowerPhase.VICReg:
            t3TowerPhase09_VICReg(args);
            break;
    }

    wt.prevPhase = wt.phase;
    wt.prevTime = wt.time;
}
