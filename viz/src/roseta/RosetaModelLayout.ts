import { IBlkDef, IBlkLabel, IModelLayout } from "@/src/llm/GptModelLayout";
import { DimStyle } from "@/src/llm/walkthrough/WalkthroughTools";
import { Vec3 } from "@/src/utils/vector";
import { RosetaDim } from "./RosetaDimStyle";

export interface IRosetaModelShape {
    B: number;
    T: number;           // sequence length
    D_input: number;     // 768 (256*3 input features) -> cells
    D_hidden: number;    // 128 (LSTM hidden) -> cells
    D_lstm: number;      // 256 (BiLSTM output = 2*hidden) -> cells
    Z_shared: number;    // 32 -> cells
    Z_private: number;   // 16 -> cells
    Z_total: number;     // 48 -> cells
    nLstmLayers: number; // 2
}

export const defaultRosetaShape: IRosetaModelShape = {
    B: 1,
    T: 10,
    D_input: 12,     // 768 -> 12 cells
    D_hidden: 6,     // 128 -> 6 cells
    D_lstm: 8,       // 256 -> 8 cells
    Z_shared: 6,     // 32 -> 6 cells
    Z_private: 3,    // 16 -> 3 cells
    Z_total: 9,      // 48 -> 9 cells
    nLstmLayers: 2,
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
    dimX: DimStyle | RosetaDim;
    dimY: DimStyle | RosetaDim;
    small?: boolean;
    hidden?: boolean;
}

// Encoder path for one domain
export interface IRosetaEncoderPath {
    input: IBlkDef;
    inputProj: IBlkDef;
    lstmEncoder: IBlkDef;
    muShared: IBlkDef;
    logvarShared: IBlkDef;
    muPrivate: IBlkDef;
    logvarPrivate: IBlkDef;
    zShared: IBlkDef;
    zPrivate: IBlkDef;
    zConcat: IBlkDef;
    zProj: IBlkDef;
    lstmDecoder: IBlkDef;
    outputProj: IBlkDef;
    softmax: IBlkDef;
    recon: IBlkDef;
}

export interface IRosetaModelLayout extends IModelLayout {
    cubes: IBlkDef[];
    cell: number;
    height: number;
    margin: number;
    labels: IBlkLabel[];
    shape: IRosetaModelShape;

    // Audio domain (left)
    audio: IRosetaEncoderPath;

    // Vibration domain (right)
    vibration: IRosetaEncoderPath;

    // Loss blocks (center bottom)
    lossRecon: IBlkDef;
    lossKLShared: IBlkDef;
    lossKLPrivate: IBlkDef;
    lossInfoNCE: IBlkDef;
    lossDiff: IBlkDef;
}

export function genRosetaModelLayout(shape: IRosetaModelShape): IRosetaModelLayout {
    let { B, T, D_input, D_hidden, D_lstm, Z_shared, Z_private, Z_total } = shape;

    let cell = 2.0;
    let margin = 20;
    let layerGap = 6;
    let sectionGap = 20;
    let latentGap = 4;

    let depthThin = 2;
    let depthMedium = 4;
    let depthDeep = 6;
    let depthLstm = 5;
    let depthLatent = 4;

    let cubes: IBlkDef[] = [];
    let labels: IBlkLabel[] = [];

    // Tower separation (closer than Phideus since no wide transformers)
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

        let blk: IBlkDef = {
            dx, dy: args.cy * cell, dz,
            t: args.t,
            x, y: args.y, z,
            cx: args.cx, cy: args.cy, cz: args.cz,
            dimX: args.dimX as DimStyle,
            dimY: args.dimY as DimStyle,
            name: args.name ?? '',
            opacity: args.hidden ? 0.0 : 1.0,
            highlight: 0.0,
            small: args.small ?? false,
            special: 0,
            idx: -1,
        };
        return blk;
    }

    function mkLabel(blks?: IBlkDef[]): IBlkLabel {
        return { visible: 0, cubes: blks ?? [] };
    }

    function add(blk: IBlkDef): IBlkDef {
        blk.idx = cubes.length;
        cubes.push(blk);
        return blk;
    }

    let D = (d: RosetaDim) => d as unknown as DimStyle;

    // ==========================================
    //  Helper: generate one encoder-decoder path (Audio or Vibration)
    // ==========================================
    function genDomainPath(
        centerX: number,
        domainName: string,
    ): { path: IRosetaEncoderPath, bottomY: number } {
        let y = 0;

        // --- Input [B, T, 256, 3] -> conceptually [B, T, D_input] ---
        let input = add(mk({
            t: 'i', xM: centerX, zM: 0, y,
            cx: T, cz: depthThin, cy: D_input,
            dimX: D(RosetaDim.T_seq), dimY: D(RosetaDim.D_input),
            name: `${domainName} Input [T,256,3]`,
        }));
        input.opacity = 0.7;
        y += D_input * cell + sectionGap;

        // --- Input Projection (768 -> 128) ---
        let inputProj = add(mk({
            t: 'w', xM: centerX, zM: 0, y,
            cx: T, cz: depthMedium, cy: D_hidden,
            dimX: D(RosetaDim.T_seq), dimY: D(RosetaDim.D_hidden),
            name: `InputProj (768→128)`,
        }));
        y += D_hidden * cell + layerGap;

        // --- BiLSTM Encoder (2 layers, 128 -> 256 output) ---
        let lstmEncoder = add(mk({
            t: 'w', xM: centerX, zM: 0, y,
            cx: T, cz: depthLstm, cy: D_lstm,
            dimX: D(RosetaDim.T_seq), dimY: D(RosetaDim.D_lstm),
            name: `BiLSTM Encoder (2L, 128→256)`,
        }));
        y += D_lstm * cell + sectionGap;

        // --- Latent Space: mu/logvar heads for shared and private ---
        let latentSpacing = 3;
        let sharedW = Z_shared;
        let privateW = Z_private;
        let totalLatentW = (sharedW + privateW) * cell + latentSpacing;
        let latentStartX = centerX - totalLatentW / 2;

        // Z_shared: mu and logvar (stacked Z)
        let muShared = add(mk({
            t: 'i', xL: latentStartX, zM: -depthLatent * cell * 0.4, y,
            cx: sharedW, cz: depthLatent, cy: 3,
            dimX: D(RosetaDim.Z_shared), dimY: D(RosetaDim.Mu),
            name: `μ_shared (32D)`,
        }));

        let logvarShared = add(mk({
            t: 'i', xL: latentStartX, zM: depthLatent * cell * 0.4, y,
            cx: sharedW, cz: depthLatent, cy: 3,
            dimX: D(RosetaDim.Z_shared), dimY: D(RosetaDim.Sigma),
            name: `σ_shared (32D)`,
        }));

        // Z_private: mu and logvar
        let muPrivate = add(mk({
            t: 'i', xL: latentStartX + sharedW * cell + latentSpacing, zM: -depthLatent * cell * 0.4, y,
            cx: privateW, cz: depthLatent, cy: 3,
            dimX: D(RosetaDim.Z_private), dimY: D(RosetaDim.Mu),
            name: `μ_private (16D)`,
        }));

        let logvarPrivate = add(mk({
            t: 'i', xL: latentStartX + sharedW * cell + latentSpacing, zM: depthLatent * cell * 0.4, y,
            cx: privateW, cz: depthLatent, cy: 3,
            dimX: D(RosetaDim.Z_private), dimY: D(RosetaDim.Sigma),
            name: `σ_private (16D)`,
        }));

        y += 3 * cell + layerGap;

        // --- Reparameterized samples ---
        let zShared = add(mk({
            t: 'i', xL: latentStartX, zM: 0, y,
            cx: sharedW, cz: depthLatent, cy: 3,
            dimX: D(RosetaDim.Z_shared), dimY: D(RosetaDim.B),
            name: `z_shared (32D)`,
        }));
        zShared.highlight = 0.2;

        let zPrivate = add(mk({
            t: 'i', xL: latentStartX + sharedW * cell + latentSpacing, zM: 0, y,
            cx: privateW, cz: depthLatent, cy: 3,
            dimX: D(RosetaDim.Z_private), dimY: D(RosetaDim.B),
            name: `z_private (16D)`,
        }));
        zPrivate.highlight = 0.15;

        y += 3 * cell + layerGap;

        // --- Concatenated latent [z_shared | z_private] ---
        let zConcat = add(mk({
            t: 'i', xM: centerX, zM: 0, y,
            cx: Z_total, cz: depthLatent, cy: 3,
            dimX: D(RosetaDim.Z_total), dimY: D(RosetaDim.B),
            name: `[z_sh|z_pr] (48D)`,
        }));
        zConcat.highlight = 0.15;
        y += 3 * cell + layerGap;

        // --- z_proj (48 -> 128) ---
        let zProj = add(mk({
            t: 'w', xM: centerX, zM: 0, y,
            cx: D_hidden, cz: depthMedium, cy: 4,
            dimX: D(RosetaDim.D_hidden), dimY: D(RosetaDim.Z_total),
            name: `z_proj (48→128)`,
        }));
        y += 4 * cell + sectionGap;

        // --- BiLSTM Decoder (2 layers) ---
        let lstmDecoder = add(mk({
            t: 'w', xM: centerX, zM: 0, y,
            cx: T, cz: depthLstm, cy: D_lstm,
            dimX: D(RosetaDim.T_seq), dimY: D(RosetaDim.D_lstm),
            name: `BiLSTM Decoder (2L)`,
        }));
        y += D_lstm * cell + layerGap;

        // --- Output Projection (256 -> 768) ---
        let outputProj = add(mk({
            t: 'w', xM: centerX, zM: 0, y,
            cx: T, cz: depthMedium, cy: D_input,
            dimX: D(RosetaDim.T_seq), dimY: D(RosetaDim.D_input),
            name: `OutputProj (256→768)`,
        }));
        y += D_input * cell + layerGap;

        // --- Softmax ---
        let softmax = add(mk({
            t: 'a', xM: centerX, zM: 0, y,
            cx: T, cz: depthThin, cy: D_input,
            dimX: D(RosetaDim.T_seq), dimY: D(RosetaDim.D_input),
            name: `Softmax`,
        }));
        y += D_input * cell + layerGap;

        // --- Reconstruction output ---
        let recon = add(mk({
            t: 'i', xM: centerX, zM: 0, y,
            cx: T, cz: depthThin, cy: D_input,
            dimX: D(RosetaDim.T_seq), dimY: D(RosetaDim.Recon),
            name: `${domainName} Recon [T,256,3]`,
        }));
        recon.highlight = 0.1;
        y += D_input * cell + margin;

        let allBlks = [input, inputProj, lstmEncoder, muShared, logvarShared, muPrivate, logvarPrivate, zShared, zPrivate, zConcat, zProj, lstmDecoder, outputProj, softmax, recon];
        let label = mkLabel(allBlks);
        labels.push(label);

        return {
            path: {
                input, inputProj, lstmEncoder,
                muShared, logvarShared, muPrivate, logvarPrivate,
                zShared, zPrivate, zConcat, zProj,
                lstmDecoder, outputProj, softmax, recon,
            },
            bottomY: y,
        };
    }

    // ==========================================
    //  Generate both domains
    // ==========================================
    let audioResult = genDomainPath(audioCenter, 'Audio');
    let vibResult = genDomainPath(vibCenter, 'Vibration');

    let audio = audioResult.path;
    let vibration = vibResult.path;

    // ==========================================
    //  LOSS SPACE (center bottom)
    // ==========================================
    let y = Math.max(audioResult.bottomY, vibResult.bottomY) + sectionGap;

    let lossW = 6;
    let lossSpacing = 6;
    let numLosses = 5;
    let lossTotalW = lossW * numLosses * cell + lossSpacing * (numLosses - 1);
    let lossStartX = -lossTotalW / 2;

    function addLoss(idx: number, name: string, dimX: RosetaDim): IBlkDef {
        let blk = add(mk({
            t: 'a',
            xL: lossStartX + idx * (lossW * cell + lossSpacing), zM: 0, y,
            cx: lossW, cz: depthDeep, cy: 4,
            dimX: D(dimX), dimY: D(RosetaDim.B),
            name,
        }));
        blk.highlight = 0.4;
        return blk;
    }

    let lossRecon = addLoss(0, 'Recon (MSE)', RosetaDim.Loss_recon);
    let lossKLShared = addLoss(1, 'KL_shared', RosetaDim.Loss_KL_shared);
    let lossKLPrivate = addLoss(2, 'KL_private', RosetaDim.Loss_KL_private);
    let lossInfoNCE = addLoss(3, 'InfoNCE', RosetaDim.Loss_InfoNCE);
    let lossDiff = addLoss(4, 'Diff Loss', RosetaDim.Loss_diff);

    y += 4 * cell + margin;

    let height = y;

    return {
        cell, height, margin, cubes, labels, shape,
        audio, vibration,
        lossRecon, lossKLShared, lossKLPrivate, lossInfoNCE, lossDiff,
    };
}

/** Get all blocks for a domain encoder-decoder path. */
export function getDomainBlocks(path: IRosetaEncoderPath): IBlkDef[] {
    return [
        path.input, path.inputProj, path.lstmEncoder,
        path.muShared, path.logvarShared, path.muPrivate, path.logvarPrivate,
        path.zShared, path.zPrivate, path.zConcat, path.zProj,
        path.lstmDecoder, path.outputProj, path.softmax, path.recon,
    ];
}

/** Get encoder-only blocks. */
export function getEncoderBlocks(path: IRosetaEncoderPath): IBlkDef[] {
    return [path.input, path.inputProj, path.lstmEncoder];
}

/** Get latent blocks. */
export function getLatentBlocks(path: IRosetaEncoderPath): IBlkDef[] {
    return [path.muShared, path.logvarShared, path.muPrivate, path.logvarPrivate,
            path.zShared, path.zPrivate, path.zConcat, path.zProj];
}

/** Get decoder blocks. */
export function getDecoderBlocks(path: IRosetaEncoderPath): IBlkDef[] {
    return [path.lstmDecoder, path.outputProj, path.softmax, path.recon];
}
