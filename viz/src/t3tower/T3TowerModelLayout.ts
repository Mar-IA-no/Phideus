import { IBlkDef, IBlkLabel, IModelLayout } from "@/src/llm/GptModelLayout";
import { DimStyle } from "@/src/llm/walkthrough/WalkthroughTools";
import { Vec3 } from "@/src/utils/vector";
import { T3TowerDim } from "./T3TowerDimStyle";

export interface IT3TowerModelShape {
    B: number;
    T_audio: number;
    T_midi: number;
    T_t3: number;
    D_audio: number;
    D_midi: number;
    D_t3: number;
    D_proj: number;
    D_ffn_audio: number;
    D_ffn_midi: number;
    D_ffn_t3: number;
    nHeadsAudio: number;
    nHeadsMidi: number;
    nHeadsT3: number;
    nLayersAudio: number;
    nLayersMidi: number;
    nLayersT3: number;
    cnnStages: number;
}

export const defaultT3TowerShape: IT3TowerModelShape = {
    B: 1,
    T_audio: 16,
    T_midi: 12,
    T_t3: 8,
    D_audio: 16,
    D_midi: 12,
    D_t3: 8,
    D_proj: 8,
    D_ffn_audio: 24,
    D_ffn_midi: 18,
    D_ffn_t3: 12,
    nHeadsAudio: 8,
    nHeadsMidi: 8,
    nHeadsT3: 4,
    nLayersAudio: 4,
    nLayersMidi: 4,
    nLayersT3: 2,
    cnnStages: 4,
};

export enum TrainState {
    Frozen,
    TrainableLow,
    TrainableHigh,
    TrainableMidi,
    TrainableT3,
    TrainableProj,
}

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
    dimX: DimStyle | T3TowerDim;
    dimY: DimStyle | T3TowerDim;
    small?: boolean;
    hidden?: boolean;
    trainState?: TrainState;
}

export interface IT3TowerModelLayout extends IModelLayout {
    cubes: IBlkDef[];
    cell: number;
    height: number;
    margin: number;
    labels: IBlkLabel[];
    shape: IT3TowerModelShape;

    // Input representations
    audioInput: IBlkDef;
    midiInput: IBlkDef;

    // Audio tower
    audioCnn: IBlkDef[];
    audioPosEmb: IBlkDef;
    audioTransformerLayers: IT3TowerTransformerLayer[];
    audioMeanPool: IBlkDef;
    audioProjLayers: IBlkDef[];

    // MIDI tower
    midiPitchEmb: IBlkDef;
    midiVelEmb: IBlkDef;
    midiDurEmb: IBlkDef;
    midiCombineLinear: IBlkDef;
    midiPosEnc: IBlkDef;
    midiTransformerLayers: IT3TowerTransformerLayer[];
    midiOutputLN: IBlkDef;
    midiMeanPool: IBlkDef;
    midiProjLayers: IBlkDef[];

    // T3 tower
    t3Input: IBlkDef;
    t3InputProj: IBlkDef;
    t3PosEnc: IBlkDef;
    t3TransformerLayers: IT3TowerTransformerLayer[];
    t3OutputLN: IBlkDef;
    t3MeanPool: IBlkDef;
    t3ProjLayers: IBlkDef[];

    // Shared space
    audioEmbedding: IBlkDef;
    midiEmbedding: IBlkDef;
    t3Embedding: IBlkDef;
    vicregAM: IBlkDef;  // audio-midi
    vicregAT: IBlkDef;  // audio-t3
    vicregMT: IBlkDef;  // midi-t3
}

export interface IT3TowerTransformerLayer {
    ln1: IBlkDef;
    qWeight: IBlkDef;
    kWeight: IBlkDef;
    vWeight: IBlkDef;
    attnMatrix: IBlkDef;
    attnOut: IBlkDef;
    attnResidual: IBlkDef;
    ln2: IBlkDef;
    mlpUp: IBlkDef;
    mlpAct: IBlkDef;
    mlpDown: IBlkDef;
    ffnResidual: IBlkDef;
    label: IBlkLabel;
}

function applyTrainState(blk: IBlkDef, state?: TrainState) {
    if (state === undefined) return;
    switch (state) {
        case TrainState.Frozen:
            blk.opacity = 0.4;
            blk.highlight = 0;
            break;
        case TrainState.TrainableLow:
            blk.opacity = 1.0;
            blk.highlight = 0.08;
            break;
        case TrainState.TrainableHigh:
            blk.opacity = 1.0;
            blk.highlight = 0.18;
            break;
        case TrainState.TrainableMidi:
            blk.opacity = 1.0;
            blk.highlight = 0.12;
            break;
        case TrainState.TrainableT3:
            blk.opacity = 1.0;
            blk.highlight = 0.15;
            break;
        case TrainState.TrainableProj:
            blk.opacity = 1.0;
            blk.highlight = 0.28;
            break;
    }
}

export function genT3TowerModelLayout(shape: IT3TowerModelShape): IT3TowerModelLayout {
    let { B, T_audio, T_midi, T_t3, D_audio, D_midi, D_t3, D_proj,
          D_ffn_audio, D_ffn_midi, D_ffn_t3, nLayersAudio, nLayersMidi, nLayersT3, cnnStages } = shape;

    let cell = 2.0;
    let margin = 20;
    let layerGap = 6;
    let sectionGap = 20;
    let subBlockGap = 8;

    let depthThin = 2;
    let depthMedium = 4;
    let depthDeep = 6;
    let depthAttn = 8;
    let depthQKV = 6;
    let depth = depthMedium;

    let cubes: IBlkDef[] = [];
    let labels: IBlkLabel[] = [];

    // Three towers: Audio left, MIDI center-left, T3 right
    let audioCenter = -140;
    let midiCenter = 40;
    let t3Center = 200;

    let D = (d: T3TowerDim) => d as unknown as DimStyle;

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

        applyTrainState(blk, args.trainState);
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

    function genTransformerLayer(
        centerX: number,
        yStart: number,
        T: number,
        Dim: number,
        D_ffn: number,
        nHeads: number,
        layerIdx: number,
        dimT: T3TowerDim,
        dimD: T3TowerDim,
        dimFfn: T3TowerDim,
        trainState: TrainState,
        towerName: string,
        ffnUpLabel: string,
        ffnDownLabel: string,
    ): { layer: IT3TowerTransformerLayer, yEnd: number } {
        let y = yStart;

        let ln1 = add(mk({
            t: 'a', xM: centerX, zM: 0, y,
            cx: T, cz: depthThin, cy: 2,
            dimX: dimT, dimY: dimD,
            name: `LN1 (${towerName} L${layerIdx})`,
            trainState,
        }));
        y += 2 * cell + layerGap;

        let qkvCx = Math.max(4, Math.floor(Dim / 2));
        let qkvCy = Math.max(4, Math.floor(Dim / 2));
        let qkvSpacing = 3;
        let qkvTotalW = 3 * qkvCx * cell + 2 * qkvSpacing;
        let qkvStartX = centerX - qkvTotalW / 2;
        let qkvZStagger = depthQKV * cell * 0.6;

        let qWeight = add(mk({
            t: 'w', xL: qkvStartX, zM: -qkvZStagger, y,
            cx: qkvCx, cz: depthQKV, cy: qkvCy,
            dimX: dimD, dimY: dimD,
            name: `W_Q (${nHeads}h)`,
            trainState,
        }));

        let kWeight = add(mk({
            t: 'w', xL: qkvStartX + qkvCx * cell + qkvSpacing, zM: 0, y,
            cx: qkvCx, cz: depthQKV, cy: qkvCy,
            dimX: dimD, dimY: dimD,
            name: `W_K (${nHeads}h)`,
            trainState,
        }));

        let vWeight = add(mk({
            t: 'w', xL: qkvStartX + (qkvCx * cell + qkvSpacing) * 2, zM: qkvZStagger, y,
            cx: qkvCx, cz: depthQKV, cy: qkvCy,
            dimX: dimD, dimY: dimD,
            name: `W_V (${nHeads}h)`,
            trainState,
        }));
        y += qkvCy * cell + layerGap;

        let attnMatrix = add(mk({
            t: 'i', xM: centerX, zM: 0, y,
            cx: T, cz: depthAttn, cy: T,
            dimX: dimT, dimY: dimT,
            name: `Attention (${T}×${T}×${nHeads}h)`,
            trainState,
        }));
        y += T * cell + layerGap;

        let attnOut = add(mk({
            t: 'i', xM: centerX, zM: 0, y,
            cx: T, cz: depthMedium, cy: 4,
            dimX: dimT, dimY: dimD,
            name: `Attn Out + W_O`,
            trainState,
        }));
        y += 4 * cell + layerGap;

        let attnResidual = add(mk({
            t: 'i', xM: centerX, zM: 0, y,
            cx: T, cz: depthThin, cy: 2,
            dimX: dimT, dimY: dimD,
            name: `+ Residual`,
            trainState,
        }));
        y += 2 * cell + subBlockGap;

        let ln2 = add(mk({
            t: 'a', xM: centerX, zM: 0, y,
            cx: T, cz: depthThin, cy: 2,
            dimX: dimT, dimY: dimD,
            name: `LN2 (${towerName} L${layerIdx})`,
            trainState,
        }));
        y += 2 * cell + layerGap;

        let mlpUp = add(mk({
            t: 'w', xM: centerX, zM: 0, y,
            cx: D_ffn, cz: depthDeep, cy: 4,
            dimX: D(dimFfn), dimY: dimD,
            name: `FFN Up (${ffnUpLabel})`,
            trainState,
        }));
        y += 4 * cell + layerGap;

        let mlpAct = add(mk({
            t: 'i', xM: centerX, zM: 0, y,
            cx: D_ffn, cz: depthDeep, cy: 3,
            dimX: D(dimFfn), dimY: dimT,
            name: `GELU`,
            trainState,
        }));
        y += 3 * cell + layerGap;

        let mlpDown = add(mk({
            t: 'w', xM: centerX, zM: 0, y,
            cx: Dim, cz: depthDeep, cy: 4,
            dimX: dimD, dimY: D(dimFfn),
            name: `FFN Down (${ffnDownLabel})`,
            trainState,
        }));
        y += 4 * cell + layerGap;

        let ffnResidual = add(mk({
            t: 'i', xM: centerX, zM: 0, y,
            cx: T, cz: depthThin, cy: 2,
            dimX: dimT, dimY: dimD,
            name: `+ Residual`,
            trainState,
        }));
        y += 2 * cell + sectionGap;

        let allBlks = [ln1, qWeight, kWeight, vWeight, attnMatrix, attnOut, attnResidual, ln2, mlpUp, mlpAct, mlpDown, ffnResidual];
        let label = mkLabel(allBlks);
        labels.push(label);

        let layer: IT3TowerTransformerLayer = {
            ln1, qWeight, kWeight, vWeight,
            attnMatrix, attnOut, attnResidual,
            ln2, mlpUp, mlpAct, mlpDown, ffnResidual,
            label,
        };

        return { layer, yEnd: y };
    }

    // ==========================================
    //  AUDIO TOWER (left, centered at -140)
    // ==========================================
    let y = 0;

    let audioInput = add(mk({
        t: 'i', xM: audioCenter, zM: 0, y,
        cx: T_audio * 2, cz: depthThin, cy: 1,
        dimX: D(T3TowerDim.T_audio), dimY: D(T3TowerDim.D_audio),
        name: 'Waveform (24kHz)',
    }));
    audioInput.opacity = 0.7;
    y += 1 * cell + sectionGap;

    let audioCnn: IBlkDef[] = [];
    let cnnWidths  = [T_audio * 2, T_audio + 4, T_audio, T_audio];
    let cnnHeights = [4, 6, 8, D_audio];
    let cnnDepths  = [3, 4, 5, 7];

    for (let i = 0; i < cnnStages; i++) {
        let blk = add(mk({
            t: 'w', xM: audioCenter, zM: 0, y,
            cx: cnnWidths[i], cz: cnnDepths[i], cy: cnnHeights[i],
            dimX: D(T3TowerDim.T_audio), dimY: D(T3TowerDim.CNN_channels),
            name: `Conv ${i} (${[64, 128, 256, 1024][i]}ch)`,
            trainState: TrainState.Frozen,
        }));
        audioCnn.push(blk);
        y += cnnHeights[i] * cell + layerGap;
    }

    y += sectionGap - layerGap;

    let audioPosEmb = add(mk({
        t: 'w', xM: audioCenter, zM: 0, y,
        cx: T_audio, cz: depthMedium, cy: D_audio,
        dimX: D(T3TowerDim.T_audio), dimY: D(T3TowerDim.D_audio),
        name: 'Learned PosEmb',
        trainState: TrainState.Frozen,
    }));
    y += D_audio * cell + sectionGap;

    let audioTransformerLayers: IT3TowerTransformerLayer[] = [];
    for (let i = 0; i < nLayersAudio; i++) {
        let ts = i < 2 ? TrainState.TrainableLow : TrainState.TrainableHigh;
        let result = genTransformerLayer(
            audioCenter, y, T_audio, D_audio, D_ffn_audio,
            shape.nHeadsAudio, i, T3TowerDim.T_audio, T3TowerDim.D_audio,
            T3TowerDim.D_ffn, ts, 'Audio', '1024→4096', '4096→1024',
        );
        audioTransformerLayers.push(result.layer);
        y = result.yEnd;
    }

    let audioMeanPool = add(mk({
        t: 'a', xM: audioCenter, zM: 0, y,
        cx: 4, cz: depthThin, cy: D_audio,
        dimX: D(T3TowerDim.B), dimY: D(T3TowerDim.D_audio),
        name: 'Mean Pool (T→1)',
    }));
    y += D_audio * cell + sectionGap;

    let audioProjLayers: IBlkDef[] = [];
    let aProjCy = [D_audio, D_proj + 2, D_proj];
    let aProjNames = ['Linear + BN + ReLU', 'Linear + BN + ReLU', 'Linear → 256D'];
    for (let i = 0; i < 3; i++) {
        let blk = add(mk({
            t: 'i', xM: audioCenter, zM: 0, y,
            cx: 4, cz: depthDeep, cy: aProjCy[i],
            dimX: D(T3TowerDim.B), dimY: i < 2 ? D(T3TowerDim.D_audio) : D(T3TowerDim.D_proj),
            name: `Proj ${aProjNames[i]}`,
            trainState: TrainState.TrainableProj,
        }));
        audioProjLayers.push(blk);
        y += aProjCy[i] * cell + layerGap;
    }
    let audioBottomY = y;

    // ==========================================
    //  MIDI TOWER (center-left, centered at +40)
    // ==========================================
    y = 0;

    let midiInput = add(mk({
        t: 'i', xM: midiCenter, zM: 0, y,
        cx: T_midi, cz: depthThin, cy: 1,
        dimX: D(T3TowerDim.T_midi), dimY: D(T3TowerDim.D_midi),
        name: 'MIDI Events',
    }));
    midiInput.opacity = 0.7;
    y += 1 * cell + sectionGap;

    let embSpacing = 4;
    let pitchW = 16;
    let velW = 10;
    let durW = 6;
    let totalEmbW = (pitchW + velW + durW) * cell + embSpacing * 2;
    let embStartX = midiCenter - totalEmbW / 2;
    let embZStagger = depthMedium * cell * 0.5;

    let midiPitchEmb = add(mk({
        t: 'w', xL: embStartX, zM: -embZStagger, y,
        cx: pitchW, cz: depthMedium + 1, cy: D_midi / 2,
        dimX: D(T3TowerDim.Pitch), dimY: D(T3TowerDim.D_midi),
        name: 'Pitch Emb (128→256)',
        trainState: TrainState.TrainableMidi,
    }));

    let midiVelEmb = add(mk({
        t: 'w', xL: midiPitchEmb.x + midiPitchEmb.dx + embSpacing, zM: 0, y,
        cx: velW, cz: depthMedium, cy: D_midi / 4,
        dimX: D(T3TowerDim.Velocity), dimY: D(T3TowerDim.D_midi),
        name: 'Velocity Emb (128→128)',
        trainState: TrainState.TrainableMidi,
    }));

    let midiDurEmb = add(mk({
        t: 'w', xL: midiVelEmb.x + midiVelEmb.dx + embSpacing, zM: embZStagger, y,
        cx: durW, cz: depthMedium - 1, cy: D_midi / 4,
        dimX: D(T3TowerDim.Duration), dimY: D(T3TowerDim.D_midi),
        name: 'Duration Emb (32→128)',
        trainState: TrainState.TrainableMidi,
    }));
    y += (D_midi / 2) * cell + sectionGap;

    let midiCombineLinear = add(mk({
        t: 'i', xM: midiCenter, zM: 0, y,
        cx: T_midi, cz: depthMedium, cy: D_midi,
        dimX: D(T3TowerDim.T_midi), dimY: D(T3TowerDim.D_midi),
        name: 'Concat → Linear → LN',
        trainState: TrainState.TrainableMidi,
    }));
    y += D_midi * cell + layerGap;

    let midiPosEnc = add(mk({
        t: 'w', xM: midiCenter, zM: 0, y,
        cx: T_midi, cz: depthMedium, cy: D_midi,
        dimX: D(T3TowerDim.T_midi), dimY: D(T3TowerDim.D_midi),
        name: 'Sinusoidal PosEnc',
        trainState: TrainState.Frozen,
    }));
    y += D_midi * cell + sectionGap;

    let midiTransformerLayers: IT3TowerTransformerLayer[] = [];
    for (let i = 0; i < nLayersMidi; i++) {
        let result = genTransformerLayer(
            midiCenter, y, T_midi, D_midi, D_ffn_midi,
            shape.nHeadsMidi, i, T3TowerDim.T_midi, T3TowerDim.D_midi,
            T3TowerDim.D_ffn, TrainState.TrainableMidi, 'MIDI', '512→2048', '2048→512',
        );
        midiTransformerLayers.push(result.layer);
        y = result.yEnd;
    }

    let midiOutputLN = add(mk({
        t: 'a', xM: midiCenter, zM: 0, y,
        cx: T_midi, cz: depthThin, cy: 2,
        dimX: D(T3TowerDim.T_midi), dimY: D(T3TowerDim.D_midi),
        name: 'Output LayerNorm',
        trainState: TrainState.TrainableMidi,
    }));
    y += 2 * cell + sectionGap;

    let midiMeanPool = add(mk({
        t: 'a', xM: midiCenter, zM: 0, y,
        cx: 4, cz: depthThin, cy: D_midi,
        dimX: D(T3TowerDim.B), dimY: D(T3TowerDim.D_midi),
        name: 'Mean Pool (T→1)',
    }));
    y += D_midi * cell + sectionGap;

    let midiProjLayers: IBlkDef[] = [];
    let mProjCy = [D_midi, D_proj + 2, D_proj];
    let mProjNames = ['Linear + BN + ReLU', 'Linear + BN + ReLU', 'Linear → 256D'];
    for (let i = 0; i < 3; i++) {
        let blk = add(mk({
            t: 'i', xM: midiCenter, zM: 0, y,
            cx: 4, cz: depthDeep, cy: mProjCy[i],
            dimX: D(T3TowerDim.B), dimY: i < 2 ? D(T3TowerDim.D_midi) : D(T3TowerDim.D_proj),
            name: `Proj ${mProjNames[i]}`,
            trainState: TrainState.TrainableProj,
        }));
        midiProjLayers.push(blk);
        y += mProjCy[i] * cell + layerGap;
    }
    let midiBottomY = y;

    // ==========================================
    //  T3 TOWER (right, centered at +200)
    // ==========================================
    y = 0;

    let t3Input = add(mk({
        t: 'i', xM: t3Center, zM: 0, y,
        cx: T_t3, cz: depthThin, cy: 1,
        dimX: D(T3TowerDim.T_t3), dimY: D(T3TowerDim.D_t3),
        name: 'T3 Input',
    }));
    t3Input.opacity = 0.7;
    y += 1 * cell + sectionGap;

    let t3InputProj = add(mk({
        t: 'w', xM: t3Center, zM: 0, y,
        cx: T_t3, cz: depthMedium, cy: D_t3,
        dimX: D(T3TowerDim.T_t3), dimY: D(T3TowerDim.D_t3),
        name: 'Input Proj (dim→256)',
        trainState: TrainState.TrainableT3,
    }));
    y += D_t3 * cell + layerGap;

    let t3PosEnc = add(mk({
        t: 'w', xM: t3Center, zM: 0, y,
        cx: T_t3, cz: depthMedium, cy: D_t3,
        dimX: D(T3TowerDim.T_t3), dimY: D(T3TowerDim.D_t3),
        name: 'Sinusoidal PosEnc',
        trainState: TrainState.Frozen,
    }));
    y += D_t3 * cell + sectionGap;

    let t3TransformerLayers: IT3TowerTransformerLayer[] = [];
    for (let i = 0; i < nLayersT3; i++) {
        let result = genTransformerLayer(
            t3Center, y, T_t3, D_t3, D_ffn_t3,
            shape.nHeadsT3, i, T3TowerDim.T_t3, T3TowerDim.D_t3,
            T3TowerDim.D_ffn_t3, TrainState.TrainableT3, 'T3', '256→1024', '1024→256',
        );
        t3TransformerLayers.push(result.layer);
        y = result.yEnd;
    }

    let t3OutputLN = add(mk({
        t: 'a', xM: t3Center, zM: 0, y,
        cx: T_t3, cz: depthThin, cy: 2,
        dimX: D(T3TowerDim.T_t3), dimY: D(T3TowerDim.D_t3),
        name: 'Output LayerNorm',
        trainState: TrainState.TrainableT3,
    }));
    y += 2 * cell + sectionGap;

    let t3MeanPool = add(mk({
        t: 'a', xM: t3Center, zM: 0, y,
        cx: 4, cz: depthThin, cy: D_t3,
        dimX: D(T3TowerDim.B), dimY: D(T3TowerDim.D_t3),
        name: 'Mean Pool (T→1)',
    }));
    y += D_t3 * cell + sectionGap;

    let t3ProjLayers: IBlkDef[] = [];
    let tProjCy = [D_t3, D_proj + 2, D_proj];
    let tProjNames = ['Linear + BN + ReLU', 'Linear + BN + ReLU', 'Linear → 256D'];
    for (let i = 0; i < 3; i++) {
        let blk = add(mk({
            t: 'i', xM: t3Center, zM: 0, y,
            cx: 4, cz: depthDeep, cy: tProjCy[i],
            dimX: D(T3TowerDim.B), dimY: D(T3TowerDim.D_proj),
            name: `Proj ${tProjNames[i]}`,
            trainState: TrainState.TrainableProj,
        }));
        t3ProjLayers.push(blk);
        y += tProjCy[i] * cell + layerGap;
    }
    let t3BottomY = y;

    // ==========================================
    //  SHARED SPACE (center, below all three towers)
    // ==========================================
    y = Math.max(audioBottomY, midiBottomY, t3BottomY) + sectionGap * 2;

    let sharedSpacing = 50;

    let audioEmbedding = add(mk({
        t: 'i', xM: -sharedSpacing, zM: 0, y,
        cx: D_proj, cz: depthMedium, cy: 4,
        dimX: D(T3TowerDim.D_proj), dimY: D(T3TowerDim.B),
        name: 'Audio z (256D)',
    }));
    audioEmbedding.highlight = 0.15;

    let midiEmbedding = add(mk({
        t: 'i', xM: 0, zM: 0, y,
        cx: D_proj, cz: depthMedium, cy: 4,
        dimX: D(T3TowerDim.D_proj), dimY: D(T3TowerDim.B),
        name: 'MIDI z (256D)',
    }));
    midiEmbedding.highlight = 0.15;

    let t3Embedding = add(mk({
        t: 'i', xM: sharedSpacing, zM: 0, y,
        cx: D_proj, cz: depthMedium, cy: 4,
        dimX: D(T3TowerDim.D_proj), dimY: D(T3TowerDim.B),
        name: 'T3 z (256D)',
    }));
    t3Embedding.highlight = 0.15;

    y += 4 * cell + sectionGap * 1.5;

    // 3-way VICReg Loss: one block per pair
    let lossW = 6;
    let lossSpacing = 8;
    let lossTotalW = lossW * 3 * cell + lossSpacing * 2;
    let lossStartX = -lossTotalW / 2;

    let vicregAM = add(mk({
        t: 'a', xL: lossStartX, zM: 0, y,
        cx: lossW, cz: depthDeep, cy: 4,
        dimX: D(T3TowerDim.Loss), dimY: D(T3TowerDim.D_proj),
        name: 'VICReg (Audio↔MIDI)',
    }));
    vicregAM.highlight = 0.4;

    let vicregAT = add(mk({
        t: 'a', xL: lossStartX + lossW * cell + lossSpacing, zM: 0, y,
        cx: lossW, cz: depthDeep, cy: 4,
        dimX: D(T3TowerDim.Loss), dimY: D(T3TowerDim.D_proj),
        name: 'VICReg (Audio↔T3)',
    }));
    vicregAT.highlight = 0.4;

    let vicregMT = add(mk({
        t: 'a', xL: lossStartX + (lossW * cell + lossSpacing) * 2, zM: 0, y,
        cx: lossW, cz: depthDeep, cy: 4,
        dimX: D(T3TowerDim.Loss), dimY: D(T3TowerDim.D_proj),
        name: 'VICReg (MIDI↔T3)',
    }));
    vicregMT.highlight = 0.4;

    y += 4 * cell + margin;

    let height = y;

    return {
        cell, height, margin, cubes, labels, shape,

        audioInput, midiInput,

        audioCnn, audioPosEmb, audioTransformerLayers,
        audioMeanPool, audioProjLayers,

        midiPitchEmb, midiVelEmb, midiDurEmb,
        midiCombineLinear, midiPosEnc,
        midiTransformerLayers, midiOutputLN,
        midiMeanPool, midiProjLayers,

        t3Input, t3InputProj, t3PosEnc,
        t3TransformerLayers, t3OutputLN,
        t3MeanPool, t3ProjLayers,

        audioEmbedding, midiEmbedding, t3Embedding,
        vicregAM, vicregAT, vicregMT,
    };
}

export function getTransformerLayerBlocks(layer: IT3TowerTransformerLayer): IBlkDef[] {
    return [
        layer.ln1, layer.qWeight, layer.kWeight, layer.vWeight,
        layer.attnMatrix, layer.attnOut, layer.attnResidual,
        layer.ln2, layer.mlpUp, layer.mlpAct, layer.mlpDown, layer.ffnResidual,
    ];
}
