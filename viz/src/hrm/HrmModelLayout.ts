import { IBlkDef, IBlkLabel, IModelLayout } from "@/src/llm/GptModelLayout";
import { DimStyle } from "@/src/llm/walkthrough/WalkthroughTools";
import { HrmDim } from "./HrmDimStyle";

export interface IHrmModelShape {
    B: number;
    T: number;
    D_input: number;
    D_enc: number;
    D_gru: number;
    D_context: number;
    D_act: number;
}

export const defaultHrmShape: IHrmModelShape = {
    B: 1,
    T: 12,
    D_input: 14,
    D_enc: 10,
    D_gru: 8,
    D_context: 6,
    D_act: 2,
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
    dimX: DimStyle | HrmDim;
    dimY: DimStyle | HrmDim;
    opacity?: number;
    highlight?: number;
    special?: number;
}

export interface IHrmModelLayout extends IModelLayout {
    cubes: IBlkDef[];
    cell: number;
    height: number;
    margin: number;
    labels: IBlkLabel[];
    shape: IHrmModelShape;

    // Input
    inputHistogram: IBlkDef;

    // L-Module
    encoderMlp: IBlkDef;
    contextIntegration: IBlkDef;
    gruStack: IBlkDef;
    spectralAttention: IBlkDef;
    lOutput: IBlkDef;

    // H-Module
    lAggregator: IBlkDef;
    harmonicAttention: IBlkDef;
    memoryLstm: IBlkDef;
    contextGenerator: IBlkDef;
    hContext: IBlkDef;

    // ACT
    qNet1: IBlkDef;
    qNet2: IBlkDef;
    qNet3: IBlkDef;
    haltDecision: IBlkDef;
}

export function genHrmModelLayout(shape: IHrmModelShape): IHrmModelLayout {
    let { T, D_input, D_enc, D_gru, D_context, D_act } = shape;

    let cell = 2.0;
    let margin = 20;
    let layerGap = 6;
    let sectionGap = 20;

    let depthThin = 2;
    let depthMedium = 4;
    let depthDeep = 6;

    let cubes: IBlkDef[] = [];
    let labels: IBlkLabel[] = [];

    let centerX = 0;
    let hModuleCenter = -70;
    let actCenter = 70;

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

    let D = (d: HrmDim) => d as unknown as DimStyle;

    // ==========================================
    //  INPUT
    // ==========================================
    let y = 0;

    let inputHistogram = add(mk({
        t: 'i', xM: centerX, zM: 0, y,
        cx: T * 2, cz: depthThin, cy: 1,
        dimX: D(HrmDim.T), dimY: D(HrmDim.D_input),
        name: 'Input Histogram [B,512,3]', opacity: 0.7,
    }));
    y += 1 * cell + sectionGap;

    // ==========================================
    //  L-MODULE (center, large)
    // ==========================================

    let encoderMlp = add(mk({
        t: 'w', xM: centerX, zM: 0, y,
        cx: D_input, cz: depthDeep, cy: D_enc,
        dimX: D(HrmDim.D_input), dimY: D(HrmDim.D_enc),
        name: 'Encoder MLP (1536->512->256)',
    }));
    y += D_enc * cell + layerGap;

    let contextIntegration = add(mk({
        t: 'a', xM: centerX, zM: 0, y,
        cx: D_enc + D_context, cz: depthMedium, cy: D_gru,
        dimX: D(HrmDim.D_enc), dimY: D(HrmDim.D_gru),
        name: 'Context Integration (+h_context)', highlight: 0.15,
    }));
    y += D_gru * cell + layerGap;

    let gruStack = add(mk({
        t: 'w', xM: centerX, zM: 0, y,
        cx: D_gru, cz: depthDeep + 2, cy: D_gru + 4,
        dimX: D(HrmDim.D_gru), dimY: D(HrmDim.D_gru),
        name: 'GRU Stack (2L, d=256)',
    }));
    y += (D_gru + 4) * cell + layerGap;

    let spectralAttention = add(mk({
        t: 'w', xM: centerX, zM: 0, y,
        cx: D_gru, cz: depthDeep, cy: D_gru,
        dimX: D(HrmDim.D_attn), dimY: D(HrmDim.D_gru),
        name: 'Spectral Attention (8h)',
    }));
    y += D_gru * cell + layerGap;

    let lOutput = add(mk({
        t: 'i', xM: centerX, zM: 0, y,
        cx: 4, cz: depthMedium, cy: D_gru,
        dimX: D(HrmDim.B), dimY: D(HrmDim.D_gru),
        name: 'L-Output [B,256]', highlight: 0.2,
    }));
    y += D_gru * cell + sectionGap;
    let lModuleBottomY = y;

    // ==========================================
    //  H-MODULE (below-left)
    // ==========================================
    y = lModuleBottomY + sectionGap;

    let lAggregator = add(mk({
        t: 'a', xM: hModuleCenter, zM: 0, y,
        cx: D_gru, cz: depthMedium, cy: D_gru,
        dimX: D(HrmDim.D_lAgg), dimY: D(HrmDim.D_gru),
        name: 'L-Aggregator',
    }));
    y += D_gru * cell + layerGap;

    let harmonicAttention = add(mk({
        t: 'w', xM: hModuleCenter, zM: 0, y,
        cx: D_gru, cz: depthDeep, cy: D_gru,
        dimX: D(HrmDim.D_hAttn), dimY: D(HrmDim.D_gru),
        name: 'Harmonic Attention',
    }));
    y += D_gru * cell + layerGap;

    let memoryLstm = add(mk({
        t: 'w', xM: hModuleCenter, zM: 0, y,
        cx: D_gru, cz: depthDeep + 2, cy: D_gru + 2,
        dimX: D(HrmDim.D_lstm), dimY: D(HrmDim.D_gru),
        name: 'Memory LSTM',
    }));
    y += (D_gru + 2) * cell + layerGap;

    let contextGenerator = add(mk({
        t: 'w', xM: hModuleCenter, zM: 0, y,
        cx: D_gru, cz: depthMedium, cy: D_context,
        dimX: D(HrmDim.D_gru), dimY: D(HrmDim.D_context),
        name: 'Context Generator',
    }));
    y += D_context * cell + layerGap;

    let hContext = add(mk({
        t: 'i', xM: hModuleCenter, zM: 0, y,
        cx: 4, cz: depthMedium, cy: D_context,
        dimX: D(HrmDim.B), dimY: D(HrmDim.D_context),
        name: 'h_context [128]', highlight: 0.3, special: 1,
    }));

    // ==========================================
    //  ACT (below-right)
    // ==========================================
    y = lModuleBottomY + sectionGap;

    let qNet1 = add(mk({
        t: 'w', xM: actCenter, zM: 0, y,
        cx: D_gru, cz: depthDeep, cy: D_context,
        dimX: D(HrmDim.D_gru), dimY: D(HrmDim.D_act),
        name: 'Q-Net: 256 -> 64',
    }));
    y += D_context * cell + layerGap;

    let qNet2 = add(mk({
        t: 'w', xM: actCenter, zM: 0, y,
        cx: D_context, cz: depthDeep, cy: D_context,
        dimX: D(HrmDim.D_act), dimY: D(HrmDim.D_act),
        name: 'Q-Net: 64 -> 64 + ReLU',
    }));
    y += D_context * cell + layerGap;

    let qNet3 = add(mk({
        t: 'w', xM: actCenter, zM: 0, y,
        cx: D_context, cz: depthMedium, cy: D_act,
        dimX: D(HrmDim.D_act), dimY: D(HrmDim.D_act),
        name: 'Q-Net: 64 -> 2',
    }));
    y += D_act * cell + sectionGap;

    let haltDecision = add(mk({
        t: 'a', xM: actCenter, zM: 0, y,
        cx: 6, cz: depthMedium, cy: 3,
        dimX: D(HrmDim.D_act), dimY: D(HrmDim.ACT),
        name: '[halt / continue]', highlight: 0.4, special: 2,
    }));

    y += 3 * cell + margin;
    let height = y;

    return {
        cell, height, margin, cubes, labels, shape,

        inputHistogram,

        encoderMlp, contextIntegration, gruStack, spectralAttention, lOutput,

        lAggregator, harmonicAttention, memoryLstm, contextGenerator, hContext,

        qNet1, qNet2, qNet3, haltDecision,
    };
}
