import { ICameraPos } from "@/src/llm/Camera";
import { IBlkDef } from "@/src/llm/GptModelLayout";
import { IRenderView } from "@/src/llm/render/modelRender";
import { commentary as llmCommentary, ICommentary, ICommentaryRes, ITimeInfo } from "@/src/llm/walkthrough/WalkthroughTools";
import { ID4XA4XProgramState } from "../D4XA4XProgram";
import { ID4XA4XModelLayout } from "../D4XA4XModelLayout";
import { d4xa4xPhase00_Overview } from "./Phase00_Overview";
import { d4xa4xPhase01_AudioCNN } from "./Phase01_AudioCNN";
import { d4xa4xPhase02_AudioDescriptor } from "./Phase02_AudioDescriptor";
import { d4xa4xPhase03_AudioForwardXAtt } from "./Phase03_AudioForwardXAtt";
import { d4xa4xPhase04_AudioTransformer } from "./Phase04_AudioTransformer";
import { d4xa4xPhase05_AudioProjection } from "./Phase05_AudioProjection";
import { d4xa4xPhase06_MIDIEmbedding } from "./Phase06_MIDIEmbedding";
import { d4xa4xPhase07_MIDIDescriptor } from "./Phase07_MIDIDescriptor";
import { d4xa4xPhase08_MIDIForwardXAtt } from "./Phase08_MIDIForwardXAtt";
import { d4xa4xPhase09_MIDITransformer } from "./Phase09_MIDITransformer";
import { d4xa4xPhase10_MIDIProjection } from "./Phase10_MIDIProjection";
import { d4xa4xPhase11_VICReg } from "./Phase11_VICReg";

export enum D4XA4XPhase {
    None = 0,
    Overview = 1,
    AudioCNN = 2,
    AudioDescriptor = 3,
    AudioForwardXAtt = 4,
    AudioTransformer = 5,
    AudioProjection = 6,
    MIDIEmbedding = 7,
    MIDIDescriptor = 8,
    MIDIForwardXAtt = 9,
    MIDITransformer = 10,
    MIDIProjection = 11,
    VICReg = 12,
}

export enum D4XA4XPhaseGroup {
    Introduction = 0,
    AudioPath = 1,
    MidiPath = 2,
    Training = 3,
}

export interface ID4XA4XPhaseGroup {
    groupId: D4XA4XPhaseGroup;
    title: string;
    phases: ID4XA4XPhaseDef[];
}

export interface ID4XA4XPhaseDef {
    id: D4XA4XPhase;
    title: string;
}

export interface ID4XA4XWalkthrough {
    phase: D4XA4XPhase;
    time: number;
    viewDt: number;
    dt: number;
    prevTime: number;
    prevPhase: D4XA4XPhase;
    running: boolean;
    speed: number;
    cameraInitial: ICameraPos | null;
    commentary: ICommentaryRes | null;
    times: (ITimeInfo | ICommentary)[];
    phaseLength: number;
    dimHighlightBlocks: IBlkDef[] | null;
    markDirty: () => void;
    phaseData: Map<D4XA4XPhase, any>;
    phaseTransitiveData: any;
    phaseList: ID4XA4XPhaseGroup[];
}

export interface ID4XA4XWalkthroughArgs {
    state: ID4XA4XProgramState;
    layout: ID4XA4XModelLayout;
    walkthrough: ID4XA4XWalkthrough;
    tools: ReturnType<typeof d4xa4xPhaseTools>;
}

export function initD4XA4XWalkthrough(): ID4XA4XWalkthrough {
    return {
        phase: D4XA4XPhase.Overview,
        time: 0, viewDt: 0, dt: 0, prevTime: 0,
        prevPhase: D4XA4XPhase.None,
        running: false, speed: 1,
        cameraInitial: null, commentary: null,
        times: [], phaseLength: 0,
        dimHighlightBlocks: null,
        markDirty: () => {},
        phaseData: new Map(),
        phaseTransitiveData: null,
        phaseList: [{
            groupId: D4XA4XPhaseGroup.Introduction,
            title: 'Introduction',
            phases: [{ id: D4XA4XPhase.Overview, title: 'Architecture Overview' }],
        }, {
            groupId: D4XA4XPhaseGroup.AudioPath,
            title: 'Audio Path',
            phases: [
                { id: D4XA4XPhase.AudioCNN, title: 'Audio CNN + PosEmb' },
                { id: D4XA4XPhase.AudioDescriptor, title: 'Audio Descriptor A4' },
                { id: D4XA4XPhase.AudioForwardXAtt, title: 'FORWARD Cross-Attention' },
                { id: D4XA4XPhase.AudioTransformer, title: 'Audio Transformer (2400 tokens!)' },
                { id: D4XA4XPhase.AudioProjection, title: 'Audio Projection' },
            ],
        }, {
            groupId: D4XA4XPhaseGroup.MidiPath,
            title: 'MIDI Path',
            phases: [
                { id: D4XA4XPhase.MIDIEmbedding, title: 'MIDI Event Embedding' },
                { id: D4XA4XPhase.MIDIDescriptor, title: 'MIDI Descriptor D4' },
                { id: D4XA4XPhase.MIDIForwardXAtt, title: 'MIDI FORWARD XAtt' },
                { id: D4XA4XPhase.MIDITransformer, title: 'MIDI Transformer' },
                { id: D4XA4XPhase.MIDIProjection, title: 'MIDI Projection' },
            ],
        }, {
            groupId: D4XA4XPhaseGroup.Training,
            title: 'Training',
            phases: [
                { id: D4XA4XPhase.VICReg, title: 'Shared Space + VICReg' },
            ],
        }],
    };
}

export function phaseToD4XA4XGroup(wt: ID4XA4XWalkthrough) {
    return wt.phaseList.find(g => g.phases.find(p => p.id === wt.phase))!;
}

export function jumpD4XA4XPhase(wt: ID4XA4XWalkthrough, delta: number) {
    let group = phaseToD4XA4XGroup(wt);
    let groupIdx = wt.phaseList.indexOf(group);
    let phaseGroupIdx = group.phases.findIndex((p: ID4XA4XPhaseDef) => p.id === wt.phase);
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
import { D4XA4XDim, d4xa4xDimColor } from "../D4XA4XDimStyle";
import { d4xa4xBlockDimension } from "../D4XA4XAnnotations";

function createAtTime(wt: ID4XA4XWalkthrough, start: number, duration?: number, wait?: number): ITimeInfo {
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

export function d4xa4xPhaseTools(state: ID4XA4XProgramState) {
    let wt = state.walkthrough;

    function c_str(str: string, duration: number = 0.3, dim?: D4XA4XDim) {
        let color = dim ? d4xa4xDimColor(dim) : new Vec4(0.1, 0.5, 0.8, 1);
        return { str, duration, start: 0, t: 0.0, color };
    }

    function c_blockRef(str: string, blk: IBlkDef | IBlkDef[], style?: DimStyle) {
        let firstBlk = Array.isArray(blk) ? blk[0] : blk;
        style ??= firstBlk.t === 'i' ? DimStyle.Intermediates : firstBlk.t === 'w' ? DimStyle.Weights : DimStyle.Aggregates;
        return { str, duration: 0, start: 0, t: 0.0, color: dimStyleColor(style), blk };
    }

    function c_dimRef(str: string, dim: D4XA4XDim) {
        return { str, duration: 0, start: 0, t: 0.0, color: d4xa4xDimColor(dim), dim: dim as unknown as DimStyle };
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
        let dimX = blk.dimX as unknown as D4XA4XDim;
        let dimY = blk.dimY as unknown as D4XA4XDim;
        if (dimX && (dimX as number) !== 0 && blk.cx > 2)
            d4xa4xBlockDimension(state.render, state.layout, blk, Dim.X, dimX, t);
        if (dimY && (dimY as number) !== 0 && blk.cy > 2)
            d4xa4xBlockDimension(state.render, state.layout, blk, Dim.Y, dimY, t);
    }

    return { atTime, afterTime, cleanup, breakAfter, c_str, c_blockRef, c_dimRef, commentary, dimExcept, drawDimLabels };
}

export function setD4XA4XInitialCamera(state: ID4XA4XProgramState, target: Vec3, rot: Vec3) {
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

export function runD4XA4XWalkthrough(state: ID4XA4XProgramState, view: IRenderView) {
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

    let args: ID4XA4XWalkthroughArgs = {
        state, layout: state.layout, walkthrough: wt,
        tools: d4xa4xPhaseTools(state),
    };

    switch (wt.phase) {
        case D4XA4XPhase.Overview: d4xa4xPhase00_Overview(args); break;
        case D4XA4XPhase.AudioCNN: d4xa4xPhase01_AudioCNN(args); break;
        case D4XA4XPhase.AudioDescriptor: d4xa4xPhase02_AudioDescriptor(args); break;
        case D4XA4XPhase.AudioForwardXAtt: d4xa4xPhase03_AudioForwardXAtt(args); break;
        case D4XA4XPhase.AudioTransformer: d4xa4xPhase04_AudioTransformer(args); break;
        case D4XA4XPhase.AudioProjection: d4xa4xPhase05_AudioProjection(args); break;
        case D4XA4XPhase.MIDIEmbedding: d4xa4xPhase06_MIDIEmbedding(args); break;
        case D4XA4XPhase.MIDIDescriptor: d4xa4xPhase07_MIDIDescriptor(args); break;
        case D4XA4XPhase.MIDIForwardXAtt: d4xa4xPhase08_MIDIForwardXAtt(args); break;
        case D4XA4XPhase.MIDITransformer: d4xa4xPhase09_MIDITransformer(args); break;
        case D4XA4XPhase.MIDIProjection: d4xa4xPhase10_MIDIProjection(args); break;
        case D4XA4XPhase.VICReg: d4xa4xPhase11_VICReg(args); break;
    }

    wt.prevPhase = wt.phase;
    wt.prevTime = wt.time;
}
