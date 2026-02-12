import { ICameraPos } from "@/src/llm/Camera";
import { IBlkDef } from "@/src/llm/GptModelLayout";
import { IRenderView } from "@/src/llm/render/modelRender";
import { commentary as llmCommentary, ICommentary, ICommentaryRes, ITimeInfo } from "@/src/llm/walkthrough/WalkthroughTools";
import { IBloqueaProgramState } from "../BloqueaProgram";
import { IBloqueaModelLayout } from "../BloqueaModelLayout";
import { bloqueaPhase00_Overview } from "./Phase00_Overview";
import { bloqueaPhase01_FrozenCNN } from "./Phase01_FrozenCNN";
import { bloqueaPhase02_AdapterLayers } from "./Phase02_AdapterLayers";
import { bloqueaPhase03_UnfrozenLayers } from "./Phase03_UnfrozenLayers";
import { bloqueaPhase04_MIDIEncoder } from "./Phase04_MIDIEncoder";
import { bloqueaPhase05_Projections } from "./Phase05_Projections";
import { bloqueaPhase06_VICReg } from "./Phase06_VICReg";
import { bloqueaPhase07_RunCConfig } from "./Phase07_RunCConfig";

export enum BloqueaPhase {
    None,
    Overview,
    FrozenCNN,
    AdapterLayers,
    UnfrozenLayers,
    MIDIEncoder,
    Projections,
    VICReg,
    RunCConfig,
}

export enum BloqueaPhaseGroup {
    Overview,
    AudioTower,
    MIDITower,
    SharedSpace,
}

export interface IBloqueaPhaseGroup {
    groupId: BloqueaPhaseGroup;
    title: string;
    phases: IBloqueaPhaseDef[];
}

export interface IBloqueaPhaseDef {
    id: BloqueaPhase;
    title: string;
}

export interface IBloqueaWalkthrough {
    phase: BloqueaPhase;
    time: number;
    viewDt: number;
    dt: number;
    prevTime: number;
    prevPhase: BloqueaPhase;
    running: boolean;
    speed: number;
    cameraInitial: ICameraPos | null;
    commentary: ICommentaryRes | null;
    times: (ITimeInfo | ICommentary)[];
    phaseLength: number;
    dimHighlightBlocks: IBlkDef[] | null;
    markDirty: () => void;
    phaseData: Map<BloqueaPhase, any>;
    phaseTransitiveData: any;
    phaseList: IBloqueaPhaseGroup[];
}

export interface IBloqueaWalkthroughArgs {
    state: IBloqueaProgramState;
    layout: IBloqueaModelLayout;
    walkthrough: IBloqueaWalkthrough;
    tools: ReturnType<typeof bloqueaPhaseTools>;
}

export function initBloqueaWalkthrough(): IBloqueaWalkthrough {
    return {
        phase: BloqueaPhase.Overview,
        time: 0,
        viewDt: 0,
        dt: 0,
        prevTime: 0,
        prevPhase: BloqueaPhase.None,
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
            groupId: BloqueaPhaseGroup.Overview,
            title: 'Introduction',
            phases: [
                { id: BloqueaPhase.Overview, title: 'Architecture Overview' },
            ],
        }, {
            groupId: BloqueaPhaseGroup.AudioTower,
            title: 'Audio Tower',
            phases: [
                { id: BloqueaPhase.FrozenCNN, title: 'Frozen CNN' },
                { id: BloqueaPhase.AdapterLayers, title: 'Adapter Layers 0-1' },
                { id: BloqueaPhase.UnfrozenLayers, title: 'Unfrozen Layers 2-3' },
            ],
        }, {
            groupId: BloqueaPhaseGroup.MIDITower,
            title: 'MIDI Tower',
            phases: [
                { id: BloqueaPhase.MIDIEncoder, title: 'MIDI Encoder' },
            ],
        }, {
            groupId: BloqueaPhaseGroup.SharedSpace,
            title: 'Shared Space',
            phases: [
                { id: BloqueaPhase.Projections, title: 'Projection Heads' },
                { id: BloqueaPhase.VICReg, title: 'VICReg Loss' },
                { id: BloqueaPhase.RunCConfig, title: 'Run C Configuration' },
            ],
        }],
    };
}

export function phaseToBloqueaGroup(wt: IBloqueaWalkthrough) {
    return wt.phaseList.find(g => g.phases.find(p => p.id === wt.phase))!;
}

export function jumpBloqueaPhase(wt: IBloqueaWalkthrough, delta: number) {
    let group = phaseToBloqueaGroup(wt);
    let groupIdx = wt.phaseList.indexOf(group);
    let phaseGroupIdx = group.phases.findIndex((p: IBloqueaPhaseDef) => p.id === wt.phase);
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
import { BloqueaDim, bloqueaDimColor } from "../BloqueaDimStyle";
import { bloqueaBlockDimension } from "../BloqueaAnnotations";

function createAtTime(wt: IBloqueaWalkthrough, start: number, duration?: number, wait?: number): ITimeInfo {
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

export function bloqueaPhaseTools(state: IBloqueaProgramState) {
    let wt = state.walkthrough;

    function c_str(str: string, duration: number = 0.3, dim?: BloqueaDim) {
        let color = dim ? bloqueaDimColor(dim) : new Vec4(0.1, 0.5, 0.8, 1);
        return { str, duration, start: 0, t: 0.0, color };
    }

    function c_blockRef(str: string, blk: IBlkDef | IBlkDef[], style?: DimStyle) {
        let firstBlk = Array.isArray(blk) ? blk[0] : blk;
        style ??= firstBlk.t === 'i' ? DimStyle.Intermediates : firstBlk.t === 'w' ? DimStyle.Weights : DimStyle.Aggregates;
        return { str, duration: 0, start: 0, t: 0.0, color: dimStyleColor(style), blk };
    }

    function c_dimRef(str: string, dim: BloqueaDim) {
        return { str, duration: 0, start: 0, t: 0.0, color: bloqueaDimColor(dim), dim: dim as unknown as DimStyle };
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
        let dimX = blk.dimX as unknown as BloqueaDim;
        let dimY = blk.dimY as unknown as BloqueaDim;
        if (dimX && (dimX as number) !== 0 && blk.cx > 2) {
            bloqueaBlockDimension(state.render, state.layout, blk, Dim.X, dimX, t, state.camera);
        }
        if (dimY && (dimY as number) !== 0 && blk.cy > 2) {
            bloqueaBlockDimension(state.render, state.layout, blk, Dim.Y, dimY, t, state.camera);
        }
    }

    return { atTime, afterTime, cleanup, breakAfter, c_str, c_blockRef, c_dimRef, commentary, dimExcept, drawDimLabels };
}

export function setBloqueaInitialCamera(state: IBloqueaProgramState, target: import("@/src/utils/vector").Vec3, rot: import("@/src/utils/vector").Vec3) {
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

export function moveBloqueaCameraTo(state: IBloqueaProgramState, time: ITimeInfo, target: import("@/src/utils/vector").Vec3, rot: import("@/src/utils/vector").Vec3) {
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

export function runBloqueaWalkthrough(state: IBloqueaProgramState, view: IRenderView) {
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

    let args: IBloqueaWalkthroughArgs = {
        state,
        layout: state.layout,
        walkthrough: wt,
        tools: bloqueaPhaseTools(state),
    };

    switch (wt.phase) {
        case BloqueaPhase.Overview:
            bloqueaPhase00_Overview(args);
            break;
        case BloqueaPhase.FrozenCNN:
            bloqueaPhase01_FrozenCNN(args);
            break;
        case BloqueaPhase.AdapterLayers:
            bloqueaPhase02_AdapterLayers(args);
            break;
        case BloqueaPhase.UnfrozenLayers:
            bloqueaPhase03_UnfrozenLayers(args);
            break;
        case BloqueaPhase.MIDIEncoder:
            bloqueaPhase04_MIDIEncoder(args);
            break;
        case BloqueaPhase.Projections:
            bloqueaPhase05_Projections(args);
            break;
        case BloqueaPhase.VICReg:
            bloqueaPhase06_VICReg(args);
            break;
        case BloqueaPhase.RunCConfig:
            bloqueaPhase07_RunCConfig(args);
            break;
    }

    wt.prevPhase = wt.phase;
    wt.prevTime = wt.time;
}
