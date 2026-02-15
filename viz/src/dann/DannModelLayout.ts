import { IBlkDef, IBlkLabel, IModelLayout } from "@/src/llm/GptModelLayout";
import { DimStyle } from "@/src/llm/walkthrough/WalkthroughTools";
import { DannDim } from "./DannDimStyle";

export interface IDannModelShape {
    B: number;
    T_audio: number;
    T_midi: number;
    D_audio: number;
    D_midi: number;
    D_proj: number;
    D_hidden: number;
}

export const defaultDannShape: IDannModelShape = {
    B: 1,
    T_audio: 14,
    T_midi: 10,
    D_audio: 14,
    D_midi: 10,
    D_proj: 8,
    D_hidden: 6,
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
    dimX: DimStyle | DannDim;
    dimY: DimStyle | DannDim;
    opacity?: number;
    highlight?: number;
    special?: number;
}

export interface IDannModelLayout extends IModelLayout {
    cubes: IBlkDef[];
    cell: number;
    height: number;
    margin: number;
    labels: IBlkLabel[];
    shape: IDannModelShape;

    audioInput: IBlkDef;
    audioCnn: IBlkDef;
    audioTransformer: IBlkDef;
    audioPool: IBlkDef;
    audioProj: IBlkDef;

    midiInput: IBlkDef;
    midiEmb: IBlkDef;
    midiTransformer: IBlkDef;
    midiPool: IBlkDef;
    midiProj: IBlkDef;

    audioEmbedding: IBlkDef;
    midiEmbedding: IBlkDef;

    concat: IBlkDef;
    l2norm: IBlkDef;
    grl: IBlkDef;
    classifierLayers: IBlkDef[];

    vicregLoss: IBlkDef;
    dannLoss: IBlkDef;
}

export function genDannModelLayout(shape: IDannModelShape): IDannModelLayout {
    let { T_audio, T_midi, D_audio, D_midi, D_proj, D_hidden } = shape;

    let cell = 2.0;
    let margin = 20;
    let layerGap = 6;
    let sectionGap = 20;

    let depthThin = 2;
    let depthMedium = 4;
    let depthDeep = 6;

    let cubes: IBlkDef[] = [];
    let labels: IBlkLabel[] = [];

    let audioCenter = -80;
    let midiCenter = 80;

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

    let D = (d: DannDim) => d as unknown as DimStyle;

    // ==========================================
    //  AUDIO TOWER (left)
    // ==========================================
    let y = 0;

    let audioInput = add(mk({
        t: 'i', xM: audioCenter, zM: 0, y,
        cx: T_audio * 2, cz: depthThin, cy: 1,
        dimX: D(DannDim.T_audio), dimY: D(DannDim.D_audio),
        name: 'Waveform (24kHz)', opacity: 0.7,
    }));
    y += 1 * cell + sectionGap;

    let audioCnn = add(mk({
        t: 'w', xM: audioCenter, zM: 0, y,
        cx: T_audio, cz: depthDeep, cy: D_audio,
        dimX: D(DannDim.T_audio), dimY: D(DannDim.CNN_channels),
        name: 'CNN (4 stages, frozen)', opacity: 0.4,
    }));
    y += D_audio * cell + sectionGap;

    let audioTransformer = add(mk({
        t: 'w', xM: audioCenter, zM: 0, y,
        cx: T_audio, cz: depthDeep + 2, cy: D_audio + 4,
        dimX: D(DannDim.T_audio), dimY: D(DannDim.D_audio),
        name: 'Transformer (4L, 8h)',
    }));
    y += (D_audio + 4) * cell + sectionGap;

    let audioPool = add(mk({
        t: 'a', xM: audioCenter, zM: 0, y,
        cx: 4, cz: depthThin, cy: D_audio,
        dimX: D(DannDim.B), dimY: D(DannDim.D_audio),
        name: 'Mean Pool',
    }));
    y += D_audio * cell + layerGap;

    let audioProj = add(mk({
        t: 'i', xM: audioCenter, zM: 0, y,
        cx: 4, cz: depthDeep, cy: D_proj,
        dimX: D(DannDim.B), dimY: D(DannDim.D_proj),
        name: 'Projection (256D)',
    }));
    y += D_proj * cell + sectionGap;
    let audioBottomY = y;

    // ==========================================
    //  MIDI TOWER (right)
    // ==========================================
    y = 0;

    let midiInput = add(mk({
        t: 'i', xM: midiCenter, zM: 0, y,
        cx: T_midi, cz: depthThin, cy: 1,
        dimX: D(DannDim.T_midi), dimY: D(DannDim.D_midi),
        name: 'MIDI Events', opacity: 0.7,
    }));
    y += 1 * cell + sectionGap;

    let midiEmb = add(mk({
        t: 'w', xM: midiCenter, zM: 0, y,
        cx: T_midi + 4, cz: depthMedium, cy: D_midi,
        dimX: D(DannDim.T_midi), dimY: D(DannDim.D_midi),
        name: 'Embedding (P+V+D)',
    }));
    y += D_midi * cell + sectionGap;

    let midiTransformer = add(mk({
        t: 'w', xM: midiCenter, zM: 0, y,
        cx: T_midi, cz: depthDeep + 2, cy: D_midi + 4,
        dimX: D(DannDim.T_midi), dimY: D(DannDim.D_midi),
        name: 'Transformer (4L, 8h)',
    }));
    y += (D_midi + 4) * cell + sectionGap;

    let midiPool = add(mk({
        t: 'a', xM: midiCenter, zM: 0, y,
        cx: 4, cz: depthThin, cy: D_midi,
        dimX: D(DannDim.B), dimY: D(DannDim.D_midi),
        name: 'Mean Pool',
    }));
    y += D_midi * cell + layerGap;

    let midiProj = add(mk({
        t: 'i', xM: midiCenter, zM: 0, y,
        cx: 4, cz: depthDeep, cy: D_proj,
        dimX: D(DannDim.B), dimY: D(DannDim.D_proj),
        name: 'Projection (256D)',
    }));
    y += D_proj * cell + sectionGap;
    let midiBottomY = y;

    // ==========================================
    //  SHARED SPACE (center)
    // ==========================================
    y = Math.max(audioBottomY, midiBottomY) + sectionGap;

    let sharedSpacing = 30;

    let audioEmbedding = add(mk({
        t: 'i', xM: -sharedSpacing / 2, zM: 0, y,
        cx: D_proj, cz: depthMedium, cy: 4,
        dimX: D(DannDim.D_proj), dimY: D(DannDim.B),
        name: 'Audio z (256D)', highlight: 0.15,
    }));

    let midiEmbedding = add(mk({
        t: 'i', xM: sharedSpacing / 2, zM: 0, y,
        cx: D_proj, cz: depthMedium, cy: 4,
        dimX: D(DannDim.D_proj), dimY: D(DannDim.B),
        name: 'MIDI z (256D)', highlight: 0.15,
    }));

    y += 4 * cell + sectionGap;

    // VICReg Loss (left-center)
    let vicregLoss = add(mk({
        t: 'a', xM: -40, zM: 0, y,
        cx: 8, cz: depthDeep, cy: 4,
        dimX: D(DannDim.Loss), dimY: D(DannDim.D_proj),
        name: 'VICReg Loss', highlight: 0.4,
    }));

    // ==========================================
    //  DANN PATH (center-right, below shared space)
    // ==========================================

    // Concatenate embeddings
    let concat = add(mk({
        t: 'i', xM: 0, zM: 0, y,
        cx: D_proj * 2, cz: depthMedium, cy: 3,
        dimX: D(DannDim.D_proj), dimY: D(DannDim.B),
        name: 'Concat [2B, 256]',
    }));
    y += 3 * cell + layerGap;

    // L2 Normalize
    let l2norm = add(mk({
        t: 'a', xM: 0, zM: 0, y,
        cx: D_proj * 2, cz: depthThin, cy: 2,
        dimX: D(DannDim.D_proj), dimY: D(DannDim.B),
        name: 'L2 Normalize',
    }));
    y += 2 * cell + sectionGap;

    // Gradient Reversal Layer (distinctive visual)
    let grl = add(mk({
        t: 'a', xM: 0, zM: 0, y,
        cx: D_proj * 2, cz: depthMedium + 2, cy: 4,
        dimX: D(DannDim.GRL), dimY: D(DannDim.D_proj),
        name: 'Gradient Reversal Layer', highlight: 0.5, special: 1,
    }));
    y += 4 * cell + sectionGap;

    // Domain Classifier: 3 linear layers
    let classifierLayers: IBlkDef[] = [];
    let clfSpecs = [
        { cx: D_proj, cy: D_hidden, name: '256 -> 64 + ReLU + Drop' },
        { cx: D_hidden, cy: D_hidden, name: '64 -> 64 + ReLU + Drop' },
        { cx: D_hidden, cy: 2, name: '64 -> 2 (logits)' },
    ];

    for (let spec of clfSpecs) {
        let blk = add(mk({
            t: 'w', xM: 0, zM: 0, y,
            cx: spec.cx, cz: depthDeep, cy: spec.cy,
            dimX: D(DannDim.Classifier), dimY: D(DannDim.D_hidden),
            name: spec.name,
        }));
        classifierLayers.push(blk);
        y += spec.cy * cell + layerGap;
    }

    y += sectionGap;

    // DANN CE Loss
    let dannLoss = add(mk({
        t: 'a', xM: 0, zM: 0, y,
        cx: 8, cz: depthDeep, cy: 4,
        dimX: D(DannDim.Loss), dimY: D(DannDim.Classifier),
        name: 'DANN CE Loss', highlight: 0.4,
    }));

    y += 4 * cell + margin;
    let height = y;

    return {
        cell, height, margin, cubes, labels, shape,

        audioInput, audioCnn, audioTransformer, audioPool, audioProj,
        midiInput, midiEmb, midiTransformer, midiPool, midiProj,

        audioEmbedding, midiEmbedding,
        concat, l2norm, grl, classifierLayers,

        vicregLoss, dannLoss,
    };
}
