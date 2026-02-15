import { IBlkDef, IBlkLabel, IModelLayout } from "@/src/llm/GptModelLayout";
import { DimStyle } from "@/src/llm/walkthrough/WalkthroughTools";
import { CrossAttDim } from "./CrossAttDimStyle";

export interface ICrossAttModelShape {
    B: number;
    T_audio: number;
    T_midi: number;
    T_stft: number;
    D_audio: number;
    D_midi: number;
    D_proj: number;
    D_desc: number;
    D_interval: number;
}

export const defaultCrossAttShape: ICrossAttModelShape = {
    B: 1,
    T_audio: 14,
    T_midi: 10,
    T_stft: 6,
    D_audio: 14,
    D_midi: 10,
    D_proj: 8,
    D_desc: 4,
    D_interval: 3,
};

interface IBlkDefArgs {
    t: 'w' | 'i' | 'a';
    xL?: number;
    xR?: number;
    xM?: number;
    zF?: number;
    zB?: number;
    zM?: number;
    name?: string;
    y: number;
    cx: number;
    cz: number;
    cy: number;
    dimX: DimStyle | CrossAttDim;
    dimY: DimStyle | CrossAttDim;
    opacity?: number;
    highlight?: number;
    special?: number;
}

export interface ICrossAttModelLayout extends IModelLayout {
    cubes: IBlkDef[];
    cell: number;
    height: number;
    margin: number;
    labels: IBlkLabel[];
    shape: ICrossAttModelShape;

    // Audio tower
    waveformInput: IBlkDef;
    cnn: IBlkDef;
    posEmb: IBlkDef;
    crossAttAudio: IBlkDef;
    residualNormAudio: IBlkDef;
    audioTransformer: IBlkDef;
    audioPool: IBlkDef;
    audioProj: IBlkDef;

    // MIDI tower
    midiInput: IBlkDef;
    midiEmb: IBlkDef;
    midiPosEnc: IBlkDef;
    crossAttMidi: IBlkDef;
    residualNormMidi: IBlkDef;
    midiTransformer: IBlkDef;
    midiPool: IBlkDef;
    midiProj: IBlkDef;

    // Audio side-channel
    audioDescriptor: IBlkDef;
    descKvProj: IBlkDef;

    // MIDI side-channel
    midiIntervals: IBlkDef;
    intervalKvProj: IBlkDef;

    // Center
    vicregLoss: IBlkDef;
}

export function genCrossAttModelLayout(shape: ICrossAttModelShape): ICrossAttModelLayout {
    let { T_audio, T_midi, T_stft, D_audio, D_midi, D_proj, D_desc, D_interval } = shape;

    let cell = 2.0;
    let margin = 20;
    let layerGap = 6;
    let sectionGap = 20;

    let depthThin = 2;
    let depthMedium = 4;
    let depthDeep = 6;

    let cubes: IBlkDef[] = [];
    let labels: IBlkLabel[] = [];

    let audioCenter = -90;
    let midiCenter = 90;

    let D = (d: CrossAttDim) => d as unknown as DimStyle;

    function mk(args: IBlkDefArgs): IBlkDef {
        let dx = args.cx * cell;
        let dz = args.cz * cell;
        let x: number;
        let z: number;

        if (args.xL !== undefined) x = args.xL;
        else if (args.xR !== undefined) x = args.xR - dx;
        else x = args.xM! - dx / 2;

        if (args.zB !== undefined) z = args.zB;
        else if (args.zF !== undefined) z = args.zF - dz;
        else z = args.zM! - dz / 2;

        return {
            dx, dy: args.cy * cell, dz,
            t: args.t,
            x, y: args.y, z,
            cx: args.cx, cy: args.cy, cz: args.cz,
            dimX: args.dimX as DimStyle,
            dimY: args.dimY as DimStyle,
            name: args.name ?? '',
            opacity: args.opacity ?? 1.0,
            highlight: args.highlight ?? 0.0,
            small: false,
            special: args.special ?? 0,
            idx: -1,
        };
    }

    function add(blk: IBlkDef): IBlkDef {
        blk.idx = cubes.length;
        cubes.push(blk);
        return blk;
    }

    // ==========================================
    //  AUDIO TOWER (left)
    // ==========================================
    let y = 0;

    let waveformInput = add(mk({
        t: 'i', xM: audioCenter, zM: 0, y,
        cx: T_audio * 2, cz: depthThin, cy: 1,
        dimX: D(CrossAttDim.T_audio), dimY: D(CrossAttDim.D_audio),
        name: 'Waveform (24kHz)', opacity: 0.7,
    }));
    y += 1 * cell + sectionGap;

    let cnn = add(mk({
        t: 'w', xM: audioCenter, zM: 0, y,
        cx: T_audio, cz: depthDeep, cy: D_audio,
        dimX: D(CrossAttDim.T_audio), dimY: D(CrossAttDim.CNN_channels),
        name: 'CNN (4 stages, frozen)', opacity: 0.4,
    }));
    y += D_audio * cell + layerGap;

    let posEmb = add(mk({
        t: 'a', xM: audioCenter, zM: 0, y,
        cx: T_audio, cz: depthThin, cy: 3,
        dimX: D(CrossAttDim.T_audio), dimY: D(CrossAttDim.D_audio),
        name: 'PosEmbedding',
    }));
    y += 3 * cell + sectionGap;

    let crossAttAudio = add(mk({
        t: 'w', xM: audioCenter, zM: 0, y,
        cx: T_audio, cz: depthDeep, cy: D_audio,
        dimX: D(CrossAttDim.T_audio), dimY: D(CrossAttDim.D_audio),
        name: 'Cross-Attention (8h)', special: 1,
    }));
    y += D_audio * cell + layerGap;

    let residualNormAudio = add(mk({
        t: 'a', xM: audioCenter, zM: 0, y,
        cx: T_audio, cz: depthThin, cy: 3,
        dimX: D(CrossAttDim.T_audio), dimY: D(CrossAttDim.Residual),
        name: 'Residual + LayerNorm',
    }));
    y += 3 * cell + sectionGap;

    let audioTransformer = add(mk({
        t: 'w', xM: audioCenter, zM: 0, y,
        cx: T_audio, cz: depthDeep + 2, cy: D_audio + 4,
        dimX: D(CrossAttDim.T_audio), dimY: D(CrossAttDim.D_audio),
        name: 'Transformer (4L, 8h)',
    }));
    y += (D_audio + 4) * cell + sectionGap;

    let audioPool = add(mk({
        t: 'a', xM: audioCenter, zM: 0, y,
        cx: 4, cz: depthThin, cy: D_audio,
        dimX: D(CrossAttDim.B), dimY: D(CrossAttDim.D_audio),
        name: 'Mean Pool',
    }));
    y += D_audio * cell + layerGap;

    let audioProj = add(mk({
        t: 'i', xM: audioCenter, zM: 0, y,
        cx: 4, cz: depthDeep, cy: D_proj,
        dimX: D(CrossAttDim.B), dimY: D(CrossAttDim.D_proj),
        name: 'Projection (256D)',
    }));
    y += D_proj * cell + sectionGap;
    let audioBottomY = y;

    // ==========================================
    //  AUDIO SIDE-CHANNEL (left-far)
    // ==========================================
    let audioDescCenter = audioCenter - 70;
    let descY = crossAttAudio.y - 40;

    let audioDescriptor = add(mk({
        t: 'i', xM: audioDescCenter, zM: 0, y: descY,
        cx: T_stft, cz: depthMedium, cy: D_desc,
        dimX: D(CrossAttDim.T_stft), dimY: D(CrossAttDim.D_desc_a4),
        name: 'Audio Descriptor', opacity: 0.5,
    }));

    let descKvProj = add(mk({
        t: 'w', xM: audioDescCenter, zM: 0, y: descY + D_desc * cell + layerGap,
        cx: T_stft, cz: depthMedium, cy: D_desc + 2,
        dimX: D(CrossAttDim.T_stft), dimY: D(CrossAttDim.D_audio),
        name: 'desc_kv_proj', opacity: 0.7,
    }));

    // ==========================================
    //  MIDI TOWER (right)
    // ==========================================
    y = 0;

    let midiInput = add(mk({
        t: 'i', xM: midiCenter, zM: 0, y,
        cx: T_midi, cz: depthThin, cy: 1,
        dimX: D(CrossAttDim.T_midi), dimY: D(CrossAttDim.D_midi),
        name: 'MIDI Events', opacity: 0.7,
    }));
    y += 1 * cell + sectionGap;

    let midiEmb = add(mk({
        t: 'w', xM: midiCenter, zM: 0, y,
        cx: T_midi + 4, cz: depthMedium, cy: D_midi,
        dimX: D(CrossAttDim.T_midi), dimY: D(CrossAttDim.D_midi),
        name: 'Embedding (P+V+D)',
    }));
    y += D_midi * cell + layerGap;

    let midiPosEnc = add(mk({
        t: 'a', xM: midiCenter, zM: 0, y,
        cx: T_midi, cz: depthThin, cy: 3,
        dimX: D(CrossAttDim.T_midi), dimY: D(CrossAttDim.D_midi),
        name: 'PosEncoding',
    }));
    y += 3 * cell + sectionGap;

    let crossAttMidi = add(mk({
        t: 'w', xM: midiCenter, zM: 0, y,
        cx: T_midi, cz: depthDeep, cy: D_midi,
        dimX: D(CrossAttDim.T_midi), dimY: D(CrossAttDim.D_midi),
        name: 'Cross-Attention (8h)', special: 1,
    }));
    y += D_midi * cell + layerGap;

    let residualNormMidi = add(mk({
        t: 'a', xM: midiCenter, zM: 0, y,
        cx: T_midi, cz: depthThin, cy: 3,
        dimX: D(CrossAttDim.T_midi), dimY: D(CrossAttDim.Residual),
        name: 'Residual + LayerNorm',
    }));
    y += 3 * cell + sectionGap;

    let midiTransformer = add(mk({
        t: 'w', xM: midiCenter, zM: 0, y,
        cx: T_midi, cz: depthDeep + 2, cy: D_midi + 4,
        dimX: D(CrossAttDim.T_midi), dimY: D(CrossAttDim.D_midi),
        name: 'Transformer (4L, 8h)',
    }));
    y += (D_midi + 4) * cell + sectionGap;

    let midiPool = add(mk({
        t: 'a', xM: midiCenter, zM: 0, y,
        cx: 4, cz: depthThin, cy: D_midi,
        dimX: D(CrossAttDim.B), dimY: D(CrossAttDim.D_midi),
        name: 'Mean Pool',
    }));
    y += D_midi * cell + layerGap;

    let midiProj = add(mk({
        t: 'i', xM: midiCenter, zM: 0, y,
        cx: 4, cz: depthDeep, cy: D_proj,
        dimX: D(CrossAttDim.B), dimY: D(CrossAttDim.D_proj),
        name: 'Projection (256D)',
    }));
    y += D_proj * cell + sectionGap;
    let midiBottomY = y;

    // ==========================================
    //  MIDI SIDE-CHANNEL (right-far)
    // ==========================================
    let midiIntCenter = midiCenter + 70;
    let intY = crossAttMidi.y - 40;

    let midiIntervals = add(mk({
        t: 'i', xM: midiIntCenter, zM: 0, y: intY,
        cx: T_midi, cz: depthMedium, cy: D_interval,
        dimX: D(CrossAttDim.T_midi), dimY: D(CrossAttDim.D_interval),
        name: 'MIDI Intervals', opacity: 0.5,
    }));

    let intervalKvProj = add(mk({
        t: 'w', xM: midiIntCenter, zM: 0, y: intY + D_interval * cell + layerGap,
        cx: T_midi, cz: depthMedium, cy: D_interval + 2,
        dimX: D(CrossAttDim.T_midi), dimY: D(CrossAttDim.D_midi),
        name: 'interval_kv_proj', opacity: 0.7,
    }));

    // ==========================================
    //  CENTER: VICReg Loss
    // ==========================================
    y = Math.max(audioBottomY, midiBottomY) + sectionGap;

    let vicregLoss = add(mk({
        t: 'a', xM: 0, zM: 0, y,
        cx: 8, cz: depthDeep, cy: 4,
        dimX: D(CrossAttDim.Loss), dimY: D(CrossAttDim.D_proj),
        name: 'VICReg Loss', highlight: 0.4,
    }));

    y += 4 * cell + margin;
    let height = y;

    return {
        cell, height, margin, cubes, labels, shape,

        waveformInput, cnn, posEmb, crossAttAudio, residualNormAudio,
        audioTransformer, audioPool, audioProj,

        midiInput, midiEmb, midiPosEnc, crossAttMidi, residualNormMidi,
        midiTransformer, midiPool, midiProj,

        audioDescriptor, descKvProj,
        midiIntervals, intervalKvProj,

        vicregLoss,
    };
}
