import { ICameraPos } from "@/src/llm/Camera";
import { IBlkDef } from "@/src/llm/GptModelLayout";
import { IRenderView } from "@/src/llm/render/modelRender";
import { commentary as llmCommentary, ICommentary, ICommentaryRes, ITimeInfo } from "@/src/llm/walkthrough/WalkthroughTools";
import { ID4A4ProgramState } from "../D4A4Program";
import { ID4A4ModelLayout } from "../D4A4ModelLayout";
import { d4a4Phase00_Overview } from "./Phase00_Overview";
import { d4a4Phase01_AudioCNN } from "./Phase01_AudioCNN";
import { d4a4Phase02_AudioTransformer } from "./Phase02_AudioTransformer";
import { d4a4Phase03_AudioDescA4 } from "./Phase03_AudioDescA4";
import { d4a4Phase04_AudioConcatProj } from "./Phase04_AudioConcatProj";
import { d4a4Phase05_MIDIEmbedding } from "./Phase05_MIDIEmbedding";
import { d4a4Phase06_MIDITransformer } from "./Phase06_MIDITransformer";
import { d4a4Phase07_MIDIDescD4 } from "./Phase07_MIDIDescD4";
import { d4a4Phase08_MIDIConcatProj } from "./Phase08_MIDIConcatProj";
import { d4a4Phase09_VICReg } from "./Phase09_VICReg";

export enum D4A4Phase {
    None,
    Overview,
    AudioCNN,
    AudioTransformer,
    AudioDescA4,
    AudioConcatProj,
    MIDIEmbedding,
    MIDITransformer,
    MIDIDescD4,
    MIDIConcatProj,
    VICReg,
}

export enum D4A4PhaseGroup {
    Overview,
    AudioTower,
    MIDITower,
    SharedSpace,
}

export interface ID4A4PhaseGroup {
    groupId: D4A4PhaseGroup;
    title: string;
    phases: ID4A4PhaseDef[];
}

export interface ID4A4PhaseDef {
    id: D4A4Phase;
    title: string;
}

export interface ID4A4Walkthrough {
    phase: D4A4Phase;
    time: number;
    viewDt: number;
    dt: number;
    prevTime: number;
    prevPhase: D4A4Phase;
    running: boolean;
    speed: number;
    cameraInitial: ICameraPos | null;
    commentary: ICommentaryRes | null;
    times: (ITimeInfo | ICommentary)[];
    phaseLength: number;
    dimHighlightBlocks: IBlkDef[] | null;
    markDirty: () => void;
    phaseData: Map<D4A4Phase, any>;
    phaseTransitiveData: any;
    phaseList: ID4A4PhaseGroup[];
}

export interface ID4A4WalkthroughArgs {
    state: ID4A4ProgramState;
    layout: ID4A4ModelLayout;
    walkthrough: ID4A4Walkthrough;
    tools: ReturnType<typeof d4a4PhaseTools>;
}

export function initD4A4Walkthrough(): ID4A4Walkthrough {
    return {
        phase: D4A4Phase.Overview,
        time: 0,
        viewDt: 0,
        dt: 0,
        prevTime: 0,
        prevPhase: D4A4Phase.None,
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
            groupId: D4A4PhaseGroup.Overview,
            title: 'Introduction',
            phases: [
                { id: D4A4Phase.Overview, title: 'Architecture Overview' },
            ],
        }, {
            groupId: D4A4PhaseGroup.AudioTower,
            title: 'Audio Tower',
            phases: [
                { id: D4A4Phase.AudioCNN, title: 'CNN Feature Extractor' },
                { id: D4A4Phase.AudioTransformer, title: 'Audio Transformer' },
                { id: D4A4Phase.AudioDescA4, title: 'Audio Descriptor A4' },
                { id: D4A4Phase.AudioConcatProj, title: 'Audio Concat + Projection' },
            ],
        }, {
            groupId: D4A4PhaseGroup.MIDITower,
            title: 'MIDI Tower',
            phases: [
                { id: D4A4Phase.MIDIEmbedding, title: 'Event Embedding' },
                { id: D4A4Phase.MIDITransformer, title: 'MIDI Transformer' },
                { id: D4A4Phase.MIDIDescD4, title: 'MIDI Descriptor D4' },
                { id: D4A4Phase.MIDIConcatProj, title: 'MIDI Concat + Projection' },
            ],
        }, {
            groupId: D4A4PhaseGroup.SharedSpace,
            title: 'Shared Space',
            phases: [
                { id: D4A4Phase.VICReg, title: 'VICReg Loss + Results' },
            ],
        }],
    };
}

export function phaseToD4A4Group(wt: ID4A4Walkthrough) {
    return wt.phaseList.find(g => g.phases.find(p => p.id === wt.phase))!;
}

export function jumpD4A4Phase(wt: ID4A4Walkthrough, delta: number) {
    let group = phaseToD4A4Group(wt);
    let groupIdx = wt.phaseList.indexOf(group);
    let phaseGroupIdx = group.phases.findIndex((p: ID4A4PhaseDef) => p.id === wt.phase);
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
import { D4A4Dim, d4a4DimColor } from "../D4A4DimStyle";
import { d4a4BlockDimension } from "../D4A4Annotations";

function createAtTime(wt: ID4A4Walkthrough, start: number, duration?: number, wait?: number): ITimeInfo {
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

export function d4a4PhaseTools(state: ID4A4ProgramState) {
    let wt = state.walkthrough;

    function c_str(str: string, duration: number = 0.3, dim?: D4A4Dim) {
        let color = dim ? d4a4DimColor(dim) : new Vec4(0.1, 0.5, 0.8, 1);
        return { str, duration, start: 0, t: 0.0, color };
    }

    function c_blockRef(str: string, blk: IBlkDef | IBlkDef[], style?: DimStyle) {
        let firstBlk = Array.isArray(blk) ? blk[0] : blk;
        style ??= firstBlk.t === 'i' ? DimStyle.Intermediates : firstBlk.t === 'w' ? DimStyle.Weights : DimStyle.Aggregates;
        return { str, duration: 0, start: 0, t: 0.0, color: dimStyleColor(style), blk };
    }

    function c_dimRef(str: string, dim: D4A4Dim) {
        return { str, duration: 0, start: 0, t: 0.0, color: d4a4DimColor(dim), dim: dim as unknown as DimStyle };
    }

    function atTime(start: number, duration?: number, wait?: number): ITimeInfo {
        return createAtTime(wt, start, duration, wait);
    }

    function afterTime(prev: ITimeInfo | null, duration: number, wait?: number): ITimeInfo {
        prev = prev ?? wt.times[wt.times.length - 1];
        return atTime(prev.start + prev.duration + prev.wait, duration, wait);
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
        let dimX = blk.dimX as unknown as D4A4Dim;
        let dimY = blk.dimY as unknown as D4A4Dim;
        if (dimX && (dimX as number) !== 0 && blk.cx > 2) {
            d4a4BlockDimension(state.render, state.layout, blk, Dim.X, dimX, t, state.camera);
        }
        if (dimY && (dimY as number) !== 0 && blk.cy > 2) {
            d4a4BlockDimension(state.render, state.layout, blk, Dim.Y, dimY, t, state.camera);
        }
    }

    return { atTime, afterTime, breakAfter, c_str, c_blockRef, c_dimRef, commentary, dimExcept, drawDimLabels };
}

export function setD4A4InitialCamera(state: ID4A4ProgramState, target: import("@/src/utils/vector").Vec3, rot: import("@/src/utils/vector").Vec3) {
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

export function runD4A4Walkthrough(state: ID4A4ProgramState, view: IRenderView) {
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

    let args: ID4A4WalkthroughArgs = {
        state,
        layout: state.layout,
        walkthrough: wt,
        tools: d4a4PhaseTools(state),
    };

    switch (wt.phase) {
        case D4A4Phase.Overview: d4a4Phase00_Overview(args); break;
        case D4A4Phase.AudioCNN: d4a4Phase01_AudioCNN(args); break;
        case D4A4Phase.AudioTransformer: d4a4Phase02_AudioTransformer(args); break;
        case D4A4Phase.AudioDescA4: d4a4Phase03_AudioDescA4(args); break;
        case D4A4Phase.AudioConcatProj: d4a4Phase04_AudioConcatProj(args); break;
        case D4A4Phase.MIDIEmbedding: d4a4Phase05_MIDIEmbedding(args); break;
        case D4A4Phase.MIDITransformer: d4a4Phase06_MIDITransformer(args); break;
        case D4A4Phase.MIDIDescD4: d4a4Phase07_MIDIDescD4(args); break;
        case D4A4Phase.MIDIConcatProj: d4a4Phase08_MIDIConcatProj(args); break;
        case D4A4Phase.VICReg: d4a4Phase09_VICReg(args); break;
    }

    wt.prevPhase = wt.phase;
    wt.prevTime = wt.time;
}
