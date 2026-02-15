import { IBlkDef, IBlkLabel, IModelLayout } from "@/src/llm/GptModelLayout";
import { DimStyle } from "@/src/llm/walkthrough/WalkthroughTools";
import { JepaDim } from "./JepaDimStyle";

export interface IJepaModelShape {
    B: number;
    T: number;
    K: number;
    D_token: number;
    D_hidden: number;
    D_z: number;
    D_pred: number;
}

export const defaultJepaShape: IJepaModelShape = {
    B: 1,
    T: 8,
    K: 10,
    D_token: 3,
    D_hidden: 8,
    D_z: 6,
    D_pred: 6,
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
    dimX: DimStyle | JepaDim;
    dimY: DimStyle | JepaDim;
    opacity?: number;
    highlight?: number;
    special?: number;
}

export interface IJepaEncoderPath {
    input: IBlkDef;
    tokenMlp: IBlkDef;
    attnPool: IBlkDef;
    lstm: IBlkDef;
    z: IBlkDef;
    stopGrad: IBlkDef;
}

export interface IJepaModelLayout extends IModelLayout {
    cubes: IBlkDef[];
    cell: number;
    height: number;
    margin: number;
    labels: IBlkLabel[];
    shape: IJepaModelShape;

    audio: IJepaEncoderPath;
    vibration: IJepaEncoderPath;

    predictorAtoV: IBlkDef;
    predictorVtoA: IBlkDef;
    infoNCELoss: IBlkDef;
}

export function genJepaModelLayout(shape: IJepaModelShape): IJepaModelLayout {
    let { T, K, D_token, D_hidden, D_z, D_pred } = shape;

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

    let D = (d: JepaDim) => d as unknown as DimStyle;

    function genEncoderPath(centerX: number, domainDim: JepaDim, domainName: string): IJepaEncoderPath {
        let y = 0;

        let input = add(mk({
            t: 'i', xM: centerX, zM: 0, y,
            cx: K, cz: depthThin, cy: D_token,
            dimX: D(JepaDim.K_tokens), dimY: D(JepaDim.D_token),
            name: `${domainName} Tokens [T,48,5]`, opacity: 0.7,
        }));
        y += D_token * cell + sectionGap;

        let tokenMlp = add(mk({
            t: 'w', xM: centerX, zM: 0, y,
            cx: K, cz: depthDeep, cy: D_hidden,
            dimX: D(JepaDim.K_tokens), dimY: D(JepaDim.D_hidden),
            name: 'Token MLP (5->128)',
        }));
        y += D_hidden * cell + layerGap;

        let attnPool = add(mk({
            t: 'a', xM: centerX, zM: 0, y,
            cx: 4, cz: depthMedium, cy: D_hidden,
            dimX: D(JepaDim.B), dimY: D(JepaDim.D_hidden),
            name: 'Attention Pool (K->1)',
        }));
        y += D_hidden * cell + sectionGap;

        let lstm = add(mk({
            t: 'w', xM: centerX, zM: 0, y,
            cx: T, cz: depthDeep + 2, cy: D_z + 2,
            dimX: D(JepaDim.T), dimY: D(JepaDim.D_lstm),
            name: 'BiLSTM (2 layers)',
        }));
        y += (D_z + 2) * cell + sectionGap;

        let z = add(mk({
            t: 'i', xM: centerX, zM: 0, y,
            cx: T, cz: depthMedium, cy: D_z,
            dimX: D(JepaDim.T), dimY: D(JepaDim.D_z),
            name: `z_${domainName.toLowerCase()} [T,32]`, highlight: 0.2,
        }));
        y += D_z * cell + layerGap;

        let stopGrad = add(mk({
            t: 'a', xM: centerX, zM: 0, y,
            cx: T, cz: depthThin + 2, cy: 3,
            dimX: D(JepaDim.StopGrad), dimY: D(JepaDim.D_z),
            name: 'STOP GRADIENT', special: 1, highlight: 0.3,
        }));
        y += 3 * cell + sectionGap;

        return { input, tokenMlp, attnPool, lstm, z, stopGrad };
    }

    let audio = genEncoderPath(audioCenter, D(JepaDim.Audio) as unknown as JepaDim, 'Audio');
    let vibration = genEncoderPath(vibCenter, D(JepaDim.Vibration) as unknown as JepaDim, 'Vibration');

    // Predictor section (center, below stop gradients)
    let predY = Math.max(
        audio.stopGrad.y + audio.stopGrad.dy,
        vibration.stopGrad.y + vibration.stopGrad.dy,
    ) + sectionGap;

    let predictorAtoV = add(mk({
        t: 'w', xM: 0, zM: -10, y: predY,
        cx: D_pred + 2, cz: depthDeep, cy: D_z,
        dimX: D(JepaDim.D_pred), dimY: D(JepaDim.D_z),
        name: 'Predictor a->v (MLP)',
    }));

    let predictorVtoA = add(mk({
        t: 'w', xM: 0, zM: 10, y: predY,
        cx: D_pred + 2, cz: depthDeep, cy: D_z,
        dimX: D(JepaDim.D_pred), dimY: D(JepaDim.D_z),
        name: 'Predictor v->a (MLP)',
    }));

    let lossY = predY + D_z * cell + sectionGap * 2;

    let infoNCELoss = add(mk({
        t: 'a', xM: 0, zM: 0, y: lossY,
        cx: 8, cz: depthDeep, cy: 4,
        dimX: D(JepaDim.Loss), dimY: D(JepaDim.D_z),
        name: 'InfoNCE Loss', highlight: 0.4,
    }));

    let height = lossY + 4 * cell + margin;

    return {
        cell, height, margin, cubes, labels, shape,
        audio, vibration,
        predictorAtoV, predictorVtoA, infoNCELoss,
    };
}

export function getEncoderBlocks(path: IJepaEncoderPath): IBlkDef[] {
    return [path.input, path.tokenMlp, path.attnPool, path.lstm, path.z, path.stopGrad];
}
