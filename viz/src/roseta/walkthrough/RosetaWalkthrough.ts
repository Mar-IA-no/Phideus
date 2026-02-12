import { ICameraPos } from "@/src/llm/Camera";
import { IBlkDef } from "@/src/llm/GptModelLayout";
import { IRenderView } from "@/src/llm/render/modelRender";
import { commentary as llmCommentary, ICommentary, ICommentaryRes, ITimeInfo } from "@/src/llm/walkthrough/WalkthroughTools";
import { IRosetaProgramState } from "../RosetaProgram";
import { IRosetaModelLayout } from "../RosetaModelLayout";
import { rosetaPhase00_Overview } from "./Phase00_Overview";
import { rosetaPhase01_AudioEncoder } from "./Phase01_AudioEncoder";
import { rosetaPhase02_LatentSpace } from "./Phase02_LatentSpace";
import { rosetaPhase03_VibrationEncoder } from "./Phase03_VibrationEncoder";
import { rosetaPhase04_Decoders } from "./Phase04_Decoders";
import { rosetaPhase05_InfoNCE } from "./Phase05_InfoNCE";
import { rosetaPhase06_KLDivergence } from "./Phase06_KLDivergence";
import { rosetaPhase07_CrossRecon } from "./Phase07_CrossRecon";

export enum RosetaPhase {
    None,
    Overview,
    AudioEncoder,
    LatentSpace,
    VibrationEncoder,
    Decoders,
    InfoNCE,
    KLDivergence,
    CrossRecon,
}

export enum RosetaPhaseGroup {
    Overview,
    Encoding,
    LatentAlignment,
    Losses,
}

export interface IRosetaPhaseGroup {
    groupId: RosetaPhaseGroup;
    title: string;
    phases: IRosetaPhaseDef[];
}

export interface IRosetaPhaseDef {
    id: RosetaPhase;
    title: string;
}

export interface IRosetaWalkthrough {
    phase: RosetaPhase;
    time: number;
    viewDt: number;
    dt: number;
    prevTime: number;
    prevPhase: RosetaPhase;
    running: boolean;
    speed: number;
    cameraInitial: ICameraPos | null;
    commentary: ICommentaryRes | null;
    times: (ITimeInfo | ICommentary)[];
    phaseLength: number;
    dimHighlightBlocks: IBlkDef[] | null;
    markDirty: () => void;
    phaseData: Map<RosetaPhase, any>;
    phaseTransitiveData: any;
    phaseList: IRosetaPhaseGroup[];
}

export interface IRosetaWalkthroughArgs {
    state: IRosetaProgramState;
    layout: IRosetaModelLayout;
    walkthrough: IRosetaWalkthrough;
    tools: ReturnType<typeof rosetaPhaseTools>;
}

export function initRosetaWalkthrough(): IRosetaWalkthrough {
    return {
        phase: RosetaPhase.Overview,
        time: 0,
        viewDt: 0,
        dt: 0,
        prevTime: 0,
        prevPhase: RosetaPhase.None,
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
            groupId: RosetaPhaseGroup.Overview,
            title: 'Introduction',
            phases: [
                { id: RosetaPhase.Overview, title: 'Architecture Overview' },
            ],
        }, {
            groupId: RosetaPhaseGroup.Encoding,
            title: 'Encoding',
            phases: [
                { id: RosetaPhase.AudioEncoder, title: 'Audio Encoder' },
                { id: RosetaPhase.LatentSpace, title: 'Latent Space' },
                { id: RosetaPhase.VibrationEncoder, title: 'Vibration Encoder' },
                { id: RosetaPhase.Decoders, title: 'Decoders' },
            ],
        }, {
            groupId: RosetaPhaseGroup.LatentAlignment,
            title: 'Latent Alignment',
            phases: [
                { id: RosetaPhase.InfoNCE, title: 'InfoNCE Alignment' },
                { id: RosetaPhase.KLDivergence, title: 'KL Divergence' },
            ],
        }, {
            groupId: RosetaPhaseGroup.Losses,
            title: 'Training & Losses',
            phases: [
                { id: RosetaPhase.CrossRecon, title: 'Cross-Recon & Finale' },
            ],
        }],
    };
}

export function phaseToRosetaGroup(wt: IRosetaWalkthrough) {
    return wt.phaseList.find(g => g.phases.find(p => p.id === wt.phase))!;
}

export function jumpRosetaPhase(wt: IRosetaWalkthrough, delta: number) {
    let group = phaseToRosetaGroup(wt);
    let groupIdx = wt.phaseList.indexOf(group);
    let phaseGroupIdx = group.phases.findIndex((p: IRosetaPhaseDef) => p.id === wt.phase);
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
import { RosetaDim, rosetaDimColor } from "../RosetaDimStyle";
import { rosetaBlockDimension } from "../RosetaAnnotations";

function createAtTime(wt: IRosetaWalkthrough, start: number, duration?: number, wait?: number): ITimeInfo {
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

export function rosetaPhaseTools(state: IRosetaProgramState) {
    let wt = state.walkthrough;

    function c_str(str: string, duration: number = 0.3, dim?: RosetaDim) {
        let color = dim ? rosetaDimColor(dim) : new Vec4(0.1, 0.5, 0.8, 1);
        return { str, duration, start: 0, t: 0.0, color };
    }

    function c_blockRef(str: string, blk: IBlkDef | IBlkDef[], style?: DimStyle) {
        let firstBlk = Array.isArray(blk) ? blk[0] : blk;
        style ??= firstBlk.t === 'i' ? DimStyle.Intermediates : firstBlk.t === 'w' ? DimStyle.Weights : DimStyle.Aggregates;
        return { str, duration: 0, start: 0, t: 0.0, color: dimStyleColor(style), blk };
    }

    function c_dimRef(str: string, dim: RosetaDim) {
        return { str, duration: 0, start: 0, t: 0.0, color: rosetaDimColor(dim), dim: dim as unknown as DimStyle };
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
        let dimX = blk.dimX as unknown as RosetaDim;
        let dimY = blk.dimY as unknown as RosetaDim;
        if (dimX && (dimX as number) !== 0 && blk.cx > 2) {
            rosetaBlockDimension(state.render, state.layout, blk, Dim.X, dimX, t, state.camera);
        }
        if (dimY && (dimY as number) !== 0 && blk.cy > 2) {
            rosetaBlockDimension(state.render, state.layout, blk, Dim.Y, dimY, t, state.camera);
        }
    }

    return { atTime, afterTime, cleanup, breakAfter, c_str, c_blockRef, c_dimRef, commentary, dimExcept, drawDimLabels };
}

export function setRosetaInitialCamera(state: IRosetaProgramState, target: import("@/src/utils/vector").Vec3, rot: import("@/src/utils/vector").Vec3) {
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

export function moveRosetaCameraTo(state: IRosetaProgramState, time: ITimeInfo, target: import("@/src/utils/vector").Vec3, rot: import("@/src/utils/vector").Vec3) {
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

export function runRosetaWalkthrough(state: IRosetaProgramState, view: IRenderView) {
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

    let args: IRosetaWalkthroughArgs = {
        state,
        layout: state.layout,
        walkthrough: wt,
        tools: rosetaPhaseTools(state),
    };

    switch (wt.phase) {
        case RosetaPhase.Overview:
            rosetaPhase00_Overview(args);
            break;
        case RosetaPhase.AudioEncoder:
            rosetaPhase01_AudioEncoder(args);
            break;
        case RosetaPhase.LatentSpace:
            rosetaPhase02_LatentSpace(args);
            break;
        case RosetaPhase.VibrationEncoder:
            rosetaPhase03_VibrationEncoder(args);
            break;
        case RosetaPhase.Decoders:
            rosetaPhase04_Decoders(args);
            break;
        case RosetaPhase.InfoNCE:
            rosetaPhase05_InfoNCE(args);
            break;
        case RosetaPhase.KLDivergence:
            rosetaPhase06_KLDivergence(args);
            break;
        case RosetaPhase.CrossRecon:
            rosetaPhase07_CrossRecon(args);
            break;
    }

    wt.prevPhase = wt.phase;
    wt.prevTime = wt.time;
}
