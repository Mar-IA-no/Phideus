import { IBlkDef, IBlkLabel, IModelLayout } from "@/src/llm/GptModelLayout";
import { DimStyle } from "@/src/llm/walkthrough/WalkthroughTools";
import { ConstellationDim } from "./ConstellationDimStyle";

export interface IConstellationModelShape {
    B: number;
    T: number;
    K_tokens: number;
    D_token: number;
    D_hidden: number;
    D_lstm: number;
    Z_shared: number;
    Z_private: number;
    Z_total: number;
    D_decoder: number;
}

export const defaultConstellationShape: IConstellationModelShape = {
    B: 1,
    T: 10,
    K_tokens: 8,
    D_token: 5,
    D_hidden: 10,
    D_lstm: 10,
    Z_shared: 6,
    Z_private: 4,
    Z_total: 10,
    D_decoder: 10,
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
    dimX: DimStyle | ConstellationDim;
    dimY: DimStyle | ConstellationDim;
    opacity?: number;
    highlight?: number;
    special?: number;
}

export interface IConstellationModelLayout extends IModelLayout {
    cubes: IBlkDef[];
    cell: number;
    height: number;
    margin: number;
    labels: IBlkLabel[];
    shape: IConstellationModelShape;

    // Audio encoder path (left)
    audioInput: IBlkDef;
    audioMask: IBlkDef;
    audioTokenMlp: IBlkDef;
    audioAttnPool: IBlkDef;
    audioBiLstm: IBlkDef;

    // Vibration encoder path (right)
    vibInput: IBlkDef;
    vibMask: IBlkDef;
    vibTokenMlp: IBlkDef;
    vibAttnPool: IBlkDef;
    vibBiLstm: IBlkDef;

    // Latent space (center)
    audioZShared: IBlkDef;
    audioZPrivate: IBlkDef;
    audioZCat: IBlkDef;
    vibZShared: IBlkDef;
    vibZPrivate: IBlkDef;
    vibZCat: IBlkDef;

    // Decoders
    audioDecProj: IBlkDef;
    audioDecLstm: IBlkDef;
    audioDecOutput: IBlkDef;
    vibDecProj: IBlkDef;
    vibDecLstm: IBlkDef;
    vibDecOutput: IBlkDef;

    // Losses
    reconLoss: IBlkDef;
    klSharedLoss: IBlkDef;
    klPrivateLoss: IBlkDef;
    infoNceLoss: IBlkDef;
    diffLoss: IBlkDef;
}

export function genConstellationModelLayout(shape: IConstellationModelShape): IConstellationModelLayout {
    let { T, K_tokens, D_token, D_hidden, D_lstm, Z_shared, Z_private, Z_total, D_decoder } = shape;

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
    let vibCenter = 80;

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

    let D = (d: ConstellationDim) => d as unknown as DimStyle;

    // ==========================================
    //  AUDIO ENCODER (left)
    // ==========================================
    let y = 0;

    let audioInput = add(mk({
        t: 'i', xM: audioCenter, zM: 0, y,
        cx: K_tokens, cz: D_token, cy: T,
        dimX: D(ConstellationDim.K_tokens), dimY: D(ConstellationDim.T),
        name: 'Audio Tokens [B,T,48,5]', opacity: 0.7,
    }));
    y += T * cell + layerGap;

    let audioMask = add(mk({
        t: 'a', xM: audioCenter, zM: 0, y,
        cx: K_tokens, cz: depthThin, cy: T,
        dimX: D(ConstellationDim.K_tokens), dimY: D(ConstellationDim.T),
        name: 'Mask [B,T,48]', opacity: 0.4,
    }));
    y += T * cell + sectionGap;

    let audioTokenMlp = add(mk({
        t: 'w', xM: audioCenter, zM: 0, y,
        cx: D_hidden, cz: depthDeep, cy: T,
        dimX: D(ConstellationDim.D_hidden), dimY: D(ConstellationDim.T),
        name: 'Token MLP (5 -> 128)',
    }));
    y += T * cell + layerGap;

    let audioAttnPool = add(mk({
        t: 'a', xM: audioCenter, zM: 0, y,
        cx: D_hidden, cz: depthMedium, cy: T,
        dimX: D(ConstellationDim.D_hidden), dimY: D(ConstellationDim.T),
        name: 'Attention Pool (K -> 1)', highlight: 0.15,
    }));
    y += T * cell + layerGap;

    let audioBiLstm = add(mk({
        t: 'w', xM: audioCenter, zM: 0, y,
        cx: D_lstm, cz: depthDeep + 2, cy: T + 2,
        dimX: D(ConstellationDim.D_lstm), dimY: D(ConstellationDim.T),
        name: 'BiLSTM (2 layers)',
    }));
    y += (T + 2) * cell + sectionGap;
    let audioEncoderBottomY = y;

    // ==========================================
    //  VIBRATION ENCODER (right)
    // ==========================================
    y = 0;

    let vibInput = add(mk({
        t: 'i', xM: vibCenter, zM: 0, y,
        cx: K_tokens, cz: D_token, cy: T,
        dimX: D(ConstellationDim.K_tokens), dimY: D(ConstellationDim.T),
        name: 'Vib Tokens [B,T,48,5]', opacity: 0.7,
    }));
    y += T * cell + layerGap;

    let vibMask = add(mk({
        t: 'a', xM: vibCenter, zM: 0, y,
        cx: K_tokens, cz: depthThin, cy: T,
        dimX: D(ConstellationDim.K_tokens), dimY: D(ConstellationDim.T),
        name: 'Mask [B,T,48]', opacity: 0.4,
    }));
    y += T * cell + sectionGap;

    let vibTokenMlp = add(mk({
        t: 'w', xM: vibCenter, zM: 0, y,
        cx: D_hidden, cz: depthDeep, cy: T,
        dimX: D(ConstellationDim.D_hidden), dimY: D(ConstellationDim.T),
        name: 'Token MLP (5 -> 128)',
    }));
    y += T * cell + layerGap;

    let vibAttnPool = add(mk({
        t: 'a', xM: vibCenter, zM: 0, y,
        cx: D_hidden, cz: depthMedium, cy: T,
        dimX: D(ConstellationDim.D_hidden), dimY: D(ConstellationDim.T),
        name: 'Attention Pool (K -> 1)', highlight: 0.15,
    }));
    y += T * cell + layerGap;

    let vibBiLstm = add(mk({
        t: 'w', xM: vibCenter, zM: 0, y,
        cx: D_lstm, cz: depthDeep + 2, cy: T + 2,
        dimX: D(ConstellationDim.D_lstm), dimY: D(ConstellationDim.T),
        name: 'BiLSTM (2 layers)',
    }));
    y += (T + 2) * cell + sectionGap;
    let vibEncoderBottomY = y;

    // ==========================================
    //  LATENT SPACE (center)
    // ==========================================
    y = Math.max(audioEncoderBottomY, vibEncoderBottomY) + sectionGap;

    let latentSpacing = 50;

    // Audio latent (left-center)
    let audioZShared = add(mk({
        t: 'i', xM: -latentSpacing, zM: 0, y,
        cx: Z_shared, cz: depthMedium, cy: T,
        dimX: D(ConstellationDim.Z_shared), dimY: D(ConstellationDim.T),
        name: 'z_shared [T,32]', highlight: 0.2,
    }));

    // Vibration latent (right-center)
    let vibZShared = add(mk({
        t: 'i', xM: latentSpacing, zM: 0, y,
        cx: Z_shared, cz: depthMedium, cy: T,
        dimX: D(ConstellationDim.Z_shared), dimY: D(ConstellationDim.T),
        name: 'z_shared [T,32]', highlight: 0.2,
    }));

    y += T * cell + layerGap;

    let audioZPrivate = add(mk({
        t: 'i', xM: -latentSpacing, zM: 0, y,
        cx: Z_private, cz: depthMedium, cy: T,
        dimX: D(ConstellationDim.Z_private), dimY: D(ConstellationDim.T),
        name: 'z_private [T,16]', highlight: 0.15,
    }));

    let vibZPrivate = add(mk({
        t: 'i', xM: latentSpacing, zM: 0, y,
        cx: Z_private, cz: depthMedium, cy: T,
        dimX: D(ConstellationDim.Z_private), dimY: D(ConstellationDim.T),
        name: 'z_private [T,16]', highlight: 0.15,
    }));

    y += T * cell + layerGap;

    let audioZCat = add(mk({
        t: 'i', xM: -latentSpacing, zM: 0, y,
        cx: Z_total, cz: depthMedium, cy: T,
        dimX: D(ConstellationDim.Z_total), dimY: D(ConstellationDim.T),
        name: 'z = cat [T,48]', highlight: 0.1,
    }));

    let vibZCat = add(mk({
        t: 'i', xM: latentSpacing, zM: 0, y,
        cx: Z_total, cz: depthMedium, cy: T,
        dimX: D(ConstellationDim.Z_total), dimY: D(ConstellationDim.T),
        name: 'z = cat [T,48]', highlight: 0.1,
    }));

    y += T * cell + sectionGap;

    // ==========================================
    //  DECODERS (below latent)
    // ==========================================

    // Audio decoder (left)
    let audioDecProj = add(mk({
        t: 'w', xM: -latentSpacing, zM: 0, y,
        cx: D_decoder, cz: depthDeep, cy: T,
        dimX: D(ConstellationDim.D_decoder), dimY: D(ConstellationDim.T),
        name: 'Proj + LSTM',
    }));

    let vibDecProj = add(mk({
        t: 'w', xM: latentSpacing, zM: 0, y,
        cx: D_decoder, cz: depthDeep, cy: T,
        dimX: D(ConstellationDim.D_decoder), dimY: D(ConstellationDim.T),
        name: 'Proj + LSTM',
    }));

    y += T * cell + layerGap;

    let audioDecLstm = add(mk({
        t: 'w', xM: -latentSpacing, zM: 0, y,
        cx: D_decoder, cz: depthDeep + 2, cy: T + 2,
        dimX: D(ConstellationDim.D_decoder), dimY: D(ConstellationDim.T),
        name: 'Decoder LSTM',
    }));

    let vibDecLstm = add(mk({
        t: 'w', xM: latentSpacing, zM: 0, y,
        cx: D_decoder, cz: depthDeep + 2, cy: T + 2,
        dimX: D(ConstellationDim.D_decoder), dimY: D(ConstellationDim.T),
        name: 'Decoder LSTM',
    }));

    y += (T + 2) * cell + layerGap;

    let audioDecOutput = add(mk({
        t: 'i', xM: -latentSpacing, zM: 0, y,
        cx: K_tokens, cz: D_token, cy: T,
        dimX: D(ConstellationDim.K_tokens), dimY: D(ConstellationDim.T),
        name: 'Output [B,T,48,5]', opacity: 0.7,
    }));

    let vibDecOutput = add(mk({
        t: 'i', xM: latentSpacing, zM: 0, y,
        cx: K_tokens, cz: D_token, cy: T,
        dimX: D(ConstellationDim.K_tokens), dimY: D(ConstellationDim.T),
        name: 'Output [B,T,48,5]', opacity: 0.7,
    }));

    y += T * cell + sectionGap * 2;

    // ==========================================
    //  LOSSES (bottom center)
    // ==========================================
    let lossSpacing = 28;

    let reconLoss = add(mk({
        t: 'a', xM: -2 * lossSpacing, zM: 0, y,
        cx: 6, cz: depthDeep, cy: 4,
        dimX: D(ConstellationDim.Loss), dimY: D(ConstellationDim.D_decoder),
        name: 'Recon', highlight: 0.4,
    }));

    let klSharedLoss = add(mk({
        t: 'a', xM: -lossSpacing, zM: 0, y,
        cx: 6, cz: depthDeep, cy: 4,
        dimX: D(ConstellationDim.Loss), dimY: D(ConstellationDim.Latent),
        name: 'KL_shared', highlight: 0.4,
    }));

    let infoNceLoss = add(mk({
        t: 'a', xM: 0, zM: 0, y,
        cx: 6, cz: depthDeep, cy: 4,
        dimX: D(ConstellationDim.Loss), dimY: D(ConstellationDim.Latent),
        name: 'InfoNCE', highlight: 0.5,
    }));

    let klPrivateLoss = add(mk({
        t: 'a', xM: lossSpacing, zM: 0, y,
        cx: 6, cz: depthDeep, cy: 4,
        dimX: D(ConstellationDim.Loss), dimY: D(ConstellationDim.Latent),
        name: 'KL_private', highlight: 0.4,
    }));

    let diffLoss = add(mk({
        t: 'a', xM: 2 * lossSpacing, zM: 0, y,
        cx: 6, cz: depthDeep, cy: 4,
        dimX: D(ConstellationDim.Loss), dimY: D(ConstellationDim.Latent),
        name: 'Diff', highlight: 0.4,
    }));

    y += 4 * cell + margin;
    let height = y;

    return {
        cell, height, margin, cubes, labels, shape,

        audioInput, audioMask, audioTokenMlp, audioAttnPool, audioBiLstm,
        vibInput, vibMask, vibTokenMlp, vibAttnPool, vibBiLstm,

        audioZShared, audioZPrivate, audioZCat,
        vibZShared, vibZPrivate, vibZCat,

        audioDecProj, audioDecLstm, audioDecOutput,
        vibDecProj, vibDecLstm, vibDecOutput,

        reconLoss, klSharedLoss, klPrivateLoss, infoNceLoss, diffLoss,
    };
}
