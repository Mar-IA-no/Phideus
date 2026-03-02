import { ICameraPos } from "@/src/llm/Camera";
import { IBlkDef } from "@/src/llm/GptModelLayout";
import { IRenderView } from "@/src/llm/render/modelRender";
import { commentary as llmCommentary, ICommentary, ICommentaryRes, ITimeInfo } from "@/src/llm/walkthrough/WalkthroughTools";
import { ID4RA4RProgramState } from "../D4RA4RProgram";
import { ID4RA4RModelLayout } from "../D4RA4RModelLayout";
import { d4ra4rPhase00_Overview } from "./Phase00_Overview";
import { d4ra4rPhase01_AudioCNN } from "./Phase01_AudioCNN";
import { d4ra4rPhase02_AudioDescriptor } from "./Phase02_AudioDescriptor";
import { d4ra4rPhase03_AudioReverseXAtt } from "./Phase03_AudioReverseXAtt";
import { d4ra4rPhase04_MIDIEmbedding } from "./Phase04_MIDIEmbedding";
import { d4ra4rPhase05_MIDIDescriptor } from "./Phase05_MIDIDescriptor";
import { d4ra4rPhase06_MIDIReverseXAtt } from "./Phase06_MIDIReverseXAtt";
import { d4ra4rPhase07_AudioTransformer } from "./Phase07_AudioTransformer";
import { d4ra4rPhase08_MIDITransformer } from "./Phase08_MIDITransformer";
import { d4ra4rPhase09_Projections } from "./Phase09_Projections";

export enum D4RA4RPhase {
    None = 0,
    Overview = 1,
    AudioCNN = 2,
    AudioDescriptor = 3,
    AudioReverseXAtt = 4,
    MIDIEmbedding = 5,
    MIDIDescriptor = 6,
    MIDIReverseXAtt = 7,
    AudioTransformer = 8,
    MIDITransformer = 9,
    Projections = 10,
}

export enum D4RA4RPhaseGroup {
    Introduction = 0,
    AudioPath = 1,
    MidiPath = 2,
    Training = 3,
}

export interface ID4RA4RPhaseGroup {
    groupId: D4RA4RPhaseGroup;
    title: string;
    phases: ID4RA4RPhaseDef[];
}

export interface ID4RA4RPhaseDef {
    id: D4RA4RPhase;
    title: string;
}

export interface ID4RA4RWalkthrough {
    phase: D4RA4RPhase;
    time: number;
    viewDt: number;
    dt: number;
    prevTime: number;
    prevPhase: D4RA4RPhase;
    running: boolean;
    speed: number;
    cameraInitial: ICameraPos | null;
    commentary: ICommentaryRes | null;
    times: (ITimeInfo | ICommentary)[];
    phaseLength: number;
    dimHighlightBlocks: IBlkDef[] | null;
    markDirty: () => void;
    phaseData: Map<D4RA4RPhase, any>;
    phaseTransitiveData: any;
    phaseList: ID4RA4RPhaseGroup[];
}

export interface ID4RA4RWalkthroughArgs {
    state: ID4RA4RProgramState;
    layout: ID4RA4RModelLayout;
    walkthrough: ID4RA4RWalkthrough;
    tools: ReturnType<typeof d4ra4rPhaseTools>;
}

export function initD4RA4RWalkthrough(): ID4RA4RWalkthrough {
    return {
        phase: D4RA4RPhase.Overview,
        time: 0, viewDt: 0, dt: 0, prevTime: 0,
        prevPhase: D4RA4RPhase.None,
        running: false, speed: 1,
        cameraInitial: null, commentary: null,
        times: [], phaseLength: 0,
        dimHighlightBlocks: null,
        markDirty: () => {},
        phaseData: new Map(),
        phaseTransitiveData: null,
        phaseList: [{
            groupId: D4RA4RPhaseGroup.Introduction,
            title: 'Introduction',
            phases: [{ id: D4RA4RPhase.Overview, title: 'Mixed Descriptor Overview' }],
        }, {
            groupId: D4RA4RPhaseGroup.AudioPath,
            title: 'Audio Path',
            phases: [
                { id: D4RA4RPhase.AudioCNN, title: 'Audio CNN + PosEmb' },
                { id: D4RA4RPhase.AudioDescriptor, title: 'Audio Descriptor A4 (spectral)' },
                { id: D4RA4RPhase.AudioReverseXAtt, title: 'Audio REVERSE Cross-Att' },
            ],
        }, {
            groupId: D4RA4RPhaseGroup.MidiPath,
            title: 'MIDI Path',
            phases: [
                { id: D4RA4RPhase.MIDIEmbedding, title: 'MIDI Event Embedding' },
                { id: D4RA4RPhase.MIDIDescriptor, title: 'MIDI Descriptor D4 (intervals)' },
                { id: D4RA4RPhase.MIDIReverseXAtt, title: 'MIDI REVERSE Cross-Att' },
            ],
        }, {
            groupId: D4RA4RPhaseGroup.Training,
            title: 'Training',
            phases: [
                { id: D4RA4RPhase.AudioTransformer, title: 'Audio Transformer (188 tokens!)' },
                { id: D4RA4RPhase.MIDITransformer, title: 'MIDI Transformer' },
                { id: D4RA4RPhase.Projections, title: 'Projections + VICReg' },
            ],
        }],
    };
}

export function phaseToD4RA4RGroup(wt: ID4RA4RWalkthrough) {
    return wt.phaseList.find(g => g.phases.find(p => p.id === wt.phase))!;
}

export function jumpD4RA4RPhase(wt: ID4RA4RWalkthrough, delta: number) {
    let group = phaseToD4RA4RGroup(wt);
    let groupIdx = wt.phaseList.indexOf(group);
    let phaseGroupIdx = group.phases.findIndex((p: ID4RA4RPhaseDef) => p.id === wt.phase);
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
import { D4RA4RDim, d4ra4rDimColor } from "../D4RA4RDimStyle";
import { d4ra4rBlockDimension } from "../D4RA4RAnnotations";

function createAtTime(wt: ID4RA4RWalkthrough, start: number, duration?: number, wait?: number): ITimeInfo {
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

export function d4ra4rPhaseTools(state: ID4RA4RProgramState) {
    let wt = state.walkthrough;

    function c_str(str: string, duration: number = 0.3, dim?: D4RA4RDim) {
        let color = dim ? d4ra4rDimColor(dim) : new Vec4(0.1, 0.5, 0.8, 1);
        return { str, duration, start: 0, t: 0.0, color };
    }

    function c_blockRef(str: string, blk: IBlkDef | IBlkDef[], style?: DimStyle) {
        let firstBlk = Array.isArray(blk) ? blk[0] : blk;
        style ??= firstBlk.t === 'i' ? DimStyle.Intermediates : firstBlk.t === 'w' ? DimStyle.Weights : DimStyle.Aggregates;
        return { str, duration: 0, start: 0, t: 0.0, color: dimStyleColor(style), blk };
    }

    function c_dimRef(str: string, dim: D4RA4RDim) {
        return { str, duration: 0, start: 0, t: 0.0, color: d4ra4rDimColor(dim), dim: dim as unknown as DimStyle };
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
        let dimX = blk.dimX as unknown as D4RA4RDim;
        let dimY = blk.dimY as unknown as D4RA4RDim;
        if (dimX && (dimX as number) !== 0 && blk.cx > 2)
            d4ra4rBlockDimension(state.render, state.layout, blk, Dim.X, dimX, t);
        if (dimY && (dimY as number) !== 0 && blk.cy > 2)
            d4ra4rBlockDimension(state.render, state.layout, blk, Dim.Y, dimY, t);
    }

    return { atTime, afterTime, cleanup, breakAfter, c_str, c_blockRef, c_dimRef, commentary, dimExcept, drawDimLabels };
}

export function setD4RA4RInitialCamera(state: ID4RA4RProgramState, target: Vec3, rot: Vec3) {
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

export function runD4RA4RWalkthrough(state: ID4RA4RProgramState, view: IRenderView) {
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

    let args: ID4RA4RWalkthroughArgs = {
        state, layout: state.layout, walkthrough: wt,
        tools: d4ra4rPhaseTools(state),
    };

    switch (wt.phase) {
        case D4RA4RPhase.Overview: d4ra4rPhase00_Overview(args); break;
        case D4RA4RPhase.AudioCNN: d4ra4rPhase01_AudioCNN(args); break;
        case D4RA4RPhase.AudioDescriptor: d4ra4rPhase02_AudioDescriptor(args); break;
        case D4RA4RPhase.AudioReverseXAtt: d4ra4rPhase03_AudioReverseXAtt(args); break;
        case D4RA4RPhase.MIDIEmbedding: d4ra4rPhase04_MIDIEmbedding(args); break;
        case D4RA4RPhase.MIDIDescriptor: d4ra4rPhase05_MIDIDescriptor(args); break;
        case D4RA4RPhase.MIDIReverseXAtt: d4ra4rPhase06_MIDIReverseXAtt(args); break;
        case D4RA4RPhase.AudioTransformer: d4ra4rPhase07_AudioTransformer(args); break;
        case D4RA4RPhase.MIDITransformer: d4ra4rPhase08_MIDITransformer(args); break;
        case D4RA4RPhase.Projections: d4ra4rPhase09_Projections(args); break;
    }

    wt.prevPhase = wt.phase;
    wt.prevTime = wt.time;
}
