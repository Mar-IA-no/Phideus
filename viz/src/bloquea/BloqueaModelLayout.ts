import { IBlkDef, IBlkLabel, IModelLayout } from "@/src/llm/GptModelLayout";
import { DimStyle } from "@/src/llm/walkthrough/WalkthroughTools";
import { Vec3 } from "@/src/utils/vector";
import { BloqueaDim } from "./BloqueaDimStyle";

export interface IBloqueaModelShape {
    B: number;
    T_audio: number;
    T_midi: number;
    D_audio: number;
    D_midi: number;
    D_proj: number;
    D_ffn_audio: number;
    D_ffn_midi: number;
    D_adapter: number;  // adapter bottleneck dim (64 -> 1 cell)
    nHeadsAudio: number;
    nHeadsMidi: number;
    nLayers: number;
    cnnStages: number;
}

export const defaultBloqueaShape: IBloqueaModelShape = {
    B: 1,
    T_audio: 16,
    T_midi: 12,
    D_audio: 16,    // 1024 -> 16 cells
    D_midi: 12,     // 512 -> 12 cells
    D_proj: 8,      // 256 -> 8 cells
    D_ffn_audio: 24, // 4096 -> 24 cells
    D_ffn_midi: 18,  // 2048 -> 18 cells
    D_adapter: 2,    // 64 -> 2 cells (dramatic pinch!)
    nHeadsAudio: 8,
    nHeadsMidi: 8,
    nLayers: 4,
    cnnStages: 4,
};

export enum BloqueaTrainState {
    Frozen,
    AdapterTrain,   // lr_adapter = 5e-4 (fastest)
    Unfrozen,       // lr_audio_unfreeze = 1e-5
    TrainableMidi,  // lr_midi = 5e-5
    TrainableProj,  // lr_proj = 1e-4
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
    dimX: DimStyle | BloqueaDim;
    dimY: DimStyle | BloqueaDim;
    small?: boolean;
    hidden?: boolean;
    trainState?: BloqueaTrainState;
}

// Adapter bottleneck blocks: Down + GELU + Up
export interface IBloqueaAdapterBottleneck {
    adapterDown: IBlkDef;   // Linear(1024->64)
    adapterAct: IBlkDef;    // GELU activation
    adapterUp: IBlkDef;     // Linear(64->1024)
}

export interface IBloqueaTransformerLayer {
    // Self-Attention sub-block
    ln1: IBlkDef;
    qWeight: IBlkDef;
    kWeight: IBlkDef;
    vWeight: IBlkDef;
    attnMatrix: IBlkDef;
    attnOut: IBlkDef;
    attnResidual: IBlkDef;

    // FFN sub-block
    ln2: IBlkDef;
    mlpUp: IBlkDef;
    mlpAct: IBlkDef;
    mlpDown: IBlkDef;
    ffnResidual: IBlkDef;

    // Adapter (only for layers 0-1)
    adapter: IBloqueaAdapterBottleneck | null;

    label: IBlkLabel;
}

export interface IBloqueaModelLayout extends IModelLayout {
    cubes: IBlkDef[];
    cell: number;
    height: number;
    margin: number;
    labels: IBlkLabel[];
    shape: IBloqueaModelShape;

    // Input representations
    audioInput: IBlkDef;
    midiInput: IBlkDef;

    // Audio tower
    audioCnn: IBlkDef[];
    audioPosEmb: IBlkDef;
    audioTransformerLayers: IBloqueaTransformerLayer[];
    audioMeanPool: IBlkDef;
    audioProjLayers: IBlkDef[];

    // MIDI tower
    midiPitchEmb: IBlkDef;
    midiVelEmb: IBlkDef;
    midiDurEmb: IBlkDef;
    midiCombineLinear: IBlkDef;
    midiPosEnc: IBlkDef;
    midiTransformerLayers: IBloqueaTransformerLayer[];
    midiOutputLN: IBlkDef;
    midiMeanPool: IBlkDef;
    midiProjLayers: IBlkDef[];

    // Shared space
    audioEmbedding: IBlkDef;
    midiEmbedding: IBlkDef;
    vicregInv: IBlkDef;
    vicregVar: IBlkDef;
    vicregCov: IBlkDef;
}

function applyTrainState(blk: IBlkDef, state?: BloqueaTrainState) {
    if (state === undefined) return;
    switch (state) {
        case BloqueaTrainState.Frozen:
            blk.opacity = 0.4;
            blk.highlight = 0;
            break;
        case BloqueaTrainState.AdapterTrain:
            blk.opacity = 1.0;
            blk.highlight = 0.25;
            break;
        case BloqueaTrainState.Unfrozen:
            blk.opacity = 1.0;
            blk.highlight = 0.12;
            break;
        case BloqueaTrainState.TrainableMidi:
            blk.opacity = 1.0;
            blk.highlight = 0.12;
            break;
        case BloqueaTrainState.TrainableProj:
            blk.opacity = 1.0;
            blk.highlight = 0.28;
            break;
    }
}

export function genBloqueaModelLayout(shape: IBloqueaModelShape): IBloqueaModelLayout {
    let { B, T_audio, T_midi, D_audio, D_midi, D_proj, D_ffn_audio, D_ffn_midi, D_adapter, nLayers, cnnStages } = shape;

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
    let depthAdapter = 4;

    let cubes: IBlkDef[] = [];
    let labels: IBlkLabel[] = [];

    let audioCenter = -100;
    let midiCenter = 100;

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
            dx,
            dy: args.cy * cell,
            dz,
            t: args.t,
            x, y: args.y, z,
            cx: args.cx,
            cy: args.cy,
            cz: args.cz,
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

    let D = (d: BloqueaDim) => d as unknown as DimStyle;

    // ==========================================
    //  Helper: generate transformer layer with optional adapter
    // ==========================================
    function genTransformerLayer(
        centerX: number,
        yStart: number,
        T: number,
        Dim: number,
        D_ffn: number,
        nHeads: number,
        layerIdx: number,
        dimT: BloqueaDim,
        dimD: BloqueaDim,
        trainState: BloqueaTrainState,
        towerName: string,
        hasAdapter: boolean,
    ): { layer: IBloqueaTransformerLayer, yEnd: number } {
        let y = yStart;

        // --- LN1 ---
        let ln1 = add(mk({
            t: 'a', xM: centerX, zM: 0, y,
            cx: T, cz: depthThin, cy: 2,
            dimX: dimT, dimY: dimD,
            name: `LN1 (${towerName} L${layerIdx})`,
            trainState,
        }));
        y += 2 * cell + layerGap;

        // --- Q/K/V ---
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

        // --- Attention Matrix ---
        let attnMatrix = add(mk({
            t: 'i', xM: centerX, zM: 0, y,
            cx: T, cz: depthAttn, cy: T,
            dimX: dimT, dimY: dimT,
            name: `Attention (${T}×${T}×${nHeads}h)`,
            trainState,
        }));
        y += T * cell + layerGap;

        // --- Attention Output ---
        let attnOut = add(mk({
            t: 'i', xM: centerX, zM: 0, y,
            cx: T, cz: depthMedium, cy: 4,
            dimX: dimT, dimY: dimD,
            name: `Attn Out + W_O`,
            trainState,
        }));
        y += 4 * cell + layerGap;

        // --- Attention Residual ---
        let attnResidual = add(mk({
            t: 'i', xM: centerX, zM: 0, y,
            cx: T, cz: depthThin, cy: 2,
            dimX: dimT, dimY: dimD,
            name: `+ Residual`,
            trainState,
        }));
        y += 2 * cell + subBlockGap;

        // --- LN2 ---
        let ln2 = add(mk({
            t: 'a', xM: centerX, zM: 0, y,
            cx: T, cz: depthThin, cy: 2,
            dimX: dimT, dimY: dimD,
            name: `LN2 (${towerName} L${layerIdx})`,
            trainState,
        }));
        y += 2 * cell + layerGap;

        // --- MLP Up ---
        let mlpUp = add(mk({
            t: 'w', xM: centerX, zM: 0, y,
            cx: D_ffn, cz: depthDeep, cy: 4,
            dimX: D(BloqueaDim.D_ffn), dimY: dimD,
            name: `FFN Up (${Dim === D_audio ? '1024→4096' : '512→2048'})`,
            trainState,
        }));
        y += 4 * cell + layerGap;

        // --- GELU ---
        let mlpAct = add(mk({
            t: 'i', xM: centerX, zM: 0, y,
            cx: D_ffn, cz: depthDeep, cy: 3,
            dimX: D(BloqueaDim.D_ffn), dimY: dimT,
            name: `GELU`,
            trainState,
        }));
        y += 3 * cell + layerGap;

        // --- MLP Down ---
        let mlpDown = add(mk({
            t: 'w', xM: centerX, zM: 0, y,
            cx: Dim, cz: depthDeep, cy: 4,
            dimX: dimD, dimY: D(BloqueaDim.D_ffn),
            name: `FFN Down (${Dim === D_audio ? '4096→1024' : '2048→512'})`,
            trainState,
        }));
        y += 4 * cell + layerGap;

        // --- FFN Residual ---
        let ffnResidual = add(mk({
            t: 'i', xM: centerX, zM: 0, y,
            cx: T, cz: depthThin, cy: 2,
            dimX: dimT, dimY: dimD,
            name: `+ Residual`,
            trainState,
        }));
        y += 2 * cell;

        // --- ADAPTER BOTTLENECK (only for frozen layers 0-1) ---
        let adapter: IBloqueaAdapterBottleneck | null = null;
        if (hasAdapter) {
            y += layerGap;

            // Down: Linear(1024 -> 64) — dramatic narrowing
            let adapterDown = add(mk({
                t: 'w', xM: centerX, zM: 0, y,
                cx: D_adapter, cz: depthAdapter, cy: 3,
                dimX: D(BloqueaDim.D_adapter), dimY: dimD,
                name: `Adapter Down (1024→64)`,
                trainState: BloqueaTrainState.AdapterTrain,
            }));
            y += 3 * cell + layerGap;

            // GELU activation
            let adapterAct = add(mk({
                t: 'i', xM: centerX, zM: 0, y,
                cx: D_adapter, cz: depthAdapter, cy: 2,
                dimX: D(BloqueaDim.D_adapter), dimY: D(BloqueaDim.D_adapter),
                name: `GELU`,
                trainState: BloqueaTrainState.AdapterTrain,
            }));
            y += 2 * cell + layerGap;

            // Up: Linear(64 -> 1024) — expansion back
            let adapterUp = add(mk({
                t: 'w', xM: centerX, zM: 0, y,
                cx: Dim, cz: depthAdapter, cy: 3,
                dimX: dimD, dimY: D(BloqueaDim.D_adapter),
                name: `Adapter Up (64→1024)`,
                trainState: BloqueaTrainState.AdapterTrain,
            }));
            y += 3 * cell;

            adapter = { adapterDown, adapterAct, adapterUp };
        }

        y += sectionGap;

        let allBlks = [ln1, qWeight, kWeight, vWeight, attnMatrix, attnOut, attnResidual, ln2, mlpUp, mlpAct, mlpDown, ffnResidual];
        if (adapter) {
            allBlks.push(adapter.adapterDown, adapter.adapterAct, adapter.adapterUp);
        }
        let label = mkLabel(allBlks);
        labels.push(label);

        let layer: IBloqueaTransformerLayer = {
            ln1, qWeight, kWeight, vWeight,
            attnMatrix, attnOut, attnResidual,
            ln2, mlpUp, mlpAct, mlpDown, ffnResidual,
            adapter,
            label,
        };

        return { layer, yEnd: y };
    }

    // ==========================================
    //  AUDIO TOWER (left, centered at -100)
    // ==========================================
    let y = 0;

    let audioInput = add(mk({
        t: 'i',
        xM: audioCenter, zM: 0, y,
        cx: T_audio * 2, cz: depthThin, cy: 1,
        dimX: D(BloqueaDim.T_audio), dimY: D(BloqueaDim.D_audio),
        name: 'Waveform (24kHz)',
    }));
    audioInput.opacity = 0.7;
    y += 1 * cell + sectionGap;

    // --- CNN Feature Extractor (4 stages, frozen) ---
    let audioCnn: IBlkDef[] = [];
    let cnnWidths  = [T_audio * 2, T_audio + 4, T_audio, T_audio];
    let cnnHeights = [4, 6, 8, D_audio];
    let cnnDepths  = [3, 4, 5, 7];

    for (let i = 0; i < cnnStages; i++) {
        let blk = add(mk({
            t: 'w',
            xM: audioCenter, zM: 0, y,
            cx: cnnWidths[i], cz: cnnDepths[i], cy: cnnHeights[i],
            dimX: D(BloqueaDim.T_audio), dimY: D(BloqueaDim.CNN_channels),
            name: `Conv ${i} (${[64, 128, 256, 1024][i]}ch)`,
            trainState: BloqueaTrainState.Frozen,
        }));
        audioCnn.push(blk);
        y += cnnHeights[i] * cell + layerGap;
    }

    y += sectionGap - layerGap;

    // --- Positional Embedding (frozen) ---
    let audioPosEmb = add(mk({
        t: 'w',
        xM: audioCenter, zM: 0, y,
        cx: T_audio, cz: depthMedium, cy: D_audio,
        dimX: D(BloqueaDim.T_audio), dimY: D(BloqueaDim.D_audio),
        name: 'Learned PosEmb',
        trainState: BloqueaTrainState.Frozen,
    }));
    y += D_audio * cell + sectionGap;

    // --- Transformer Layers x4 ---
    // Layers 0-1: FROZEN + Adapter (lr_adapter=5e-4)
    // Layers 2-3: UNFROZEN (lr_audio_unfreeze=1e-5)
    let audioTransformerLayers: IBloqueaTransformerLayer[] = [];

    for (let i = 0; i < nLayers; i++) {
        let isFrozenWithAdapter = i < 2;
        let ts = isFrozenWithAdapter ? BloqueaTrainState.Frozen : BloqueaTrainState.Unfrozen;
        let result = genTransformerLayer(
            audioCenter, y, T_audio, D_audio, D_ffn_audio,
            shape.nHeadsAudio, i, BloqueaDim.T_audio, BloqueaDim.D_audio,
            ts, 'Audio',
            isFrozenWithAdapter, // hasAdapter
        );
        audioTransformerLayers.push(result.layer);
        y = result.yEnd;
    }

    // --- Mean Pooling ---
    let audioMeanPool = add(mk({
        t: 'a',
        xM: audioCenter, zM: 0, y,
        cx: 4, cz: depthThin, cy: D_audio,
        dimX: D(BloqueaDim.B), dimY: D(BloqueaDim.D_audio),
        name: 'Mean Pool (T→1)',
    }));
    y += D_audio * cell + sectionGap;

    // --- Projection Head: 3 layers ---
    let audioProjLayers: IBlkDef[] = [];
    let aProjCy = [D_audio, D_proj + 2, D_proj];
    let aProjNames = ['Linear + BN + ReLU', 'Linear + BN + ReLU', 'Linear → 256D'];

    for (let i = 0; i < 3; i++) {
        let blk = add(mk({
            t: 'i',
            xM: audioCenter, zM: 0, y,
            cx: 4, cz: depthDeep, cy: aProjCy[i],
            dimX: D(BloqueaDim.B), dimY: i < 2 ? D(BloqueaDim.D_audio) : D(BloqueaDim.D_proj),
            name: `Proj ${aProjNames[i]}`,
            trainState: BloqueaTrainState.TrainableProj,
        }));
        audioProjLayers.push(blk);
        y += aProjCy[i] * cell + layerGap;
    }

    let audioBottomY = y;

    // ==========================================
    //  MIDI TOWER (right, centered at +100)
    // ==========================================
    y = 0;

    let midiInput = add(mk({
        t: 'i',
        xM: midiCenter, zM: 0, y,
        cx: T_midi, cz: depthThin, cy: 1,
        dimX: D(BloqueaDim.T_midi), dimY: D(BloqueaDim.D_midi),
        name: 'MIDI Events',
    }));
    midiInput.opacity = 0.7;
    y += 1 * cell + sectionGap;

    // --- Embedding Tables ---
    let embSpacing = 4;
    let pitchW = 16;
    let velW = 10;
    let durW = 6;
    let totalEmbW = (pitchW + velW + durW) * cell + embSpacing * 2;
    let embStartX = midiCenter - totalEmbW / 2;
    let embZStagger = depthMedium * cell * 0.5;

    let midiPitchEmb = add(mk({
        t: 'w',
        xL: embStartX, zM: -embZStagger, y,
        cx: pitchW, cz: depthMedium + 1, cy: D_midi / 2,
        dimX: D(BloqueaDim.Pitch), dimY: D(BloqueaDim.D_midi),
        name: 'Pitch Emb (128→256)',
        trainState: BloqueaTrainState.TrainableMidi,
    }));

    let midiVelEmb = add(mk({
        t: 'w',
        xL: midiPitchEmb.x + midiPitchEmb.dx + embSpacing, zM: 0, y,
        cx: velW, cz: depthMedium, cy: D_midi / 4,
        dimX: D(BloqueaDim.Velocity), dimY: D(BloqueaDim.D_midi),
        name: 'Velocity Emb (128→128)',
        trainState: BloqueaTrainState.TrainableMidi,
    }));

    let midiDurEmb = add(mk({
        t: 'w',
        xL: midiVelEmb.x + midiVelEmb.dx + embSpacing, zM: embZStagger, y,
        cx: durW, cz: depthMedium - 1, cy: D_midi / 4,
        dimX: D(BloqueaDim.Duration), dimY: D(BloqueaDim.D_midi),
        name: 'Duration Emb (32→128)',
        trainState: BloqueaTrainState.TrainableMidi,
    }));

    y += (D_midi / 2) * cell + sectionGap;

    // --- Concat + Linear + LayerNorm ---
    let midiCombineLinear = add(mk({
        t: 'i',
        xM: midiCenter, zM: 0, y,
        cx: T_midi, cz: depthMedium, cy: D_midi,
        dimX: D(BloqueaDim.T_midi), dimY: D(BloqueaDim.D_midi),
        name: 'Concat → Linear → LN',
        trainState: BloqueaTrainState.TrainableMidi,
    }));
    y += D_midi * cell + layerGap;

    // --- Sinusoidal Positional Encoding ---
    let midiPosEnc = add(mk({
        t: 'w',
        xM: midiCenter, zM: 0, y,
        cx: T_midi, cz: depthMedium, cy: D_midi,
        dimX: D(BloqueaDim.T_midi), dimY: D(BloqueaDim.D_midi),
        name: 'Sinusoidal PosEnc',
        trainState: BloqueaTrainState.Frozen,
    }));
    y += D_midi * cell + sectionGap;

    // --- MIDI Transformer Layers x4 (no adapters) ---
    let midiTransformerLayers: IBloqueaTransformerLayer[] = [];

    for (let i = 0; i < nLayers; i++) {
        let result = genTransformerLayer(
            midiCenter, y, T_midi, D_midi, D_ffn_midi,
            shape.nHeadsMidi, i, BloqueaDim.T_midi, BloqueaDim.D_midi,
            BloqueaTrainState.TrainableMidi, 'MIDI',
            false, // no adapters in MIDI tower
        );
        midiTransformerLayers.push(result.layer);
        y = result.yEnd;
    }

    // --- Output LayerNorm ---
    let midiOutputLN = add(mk({
        t: 'a',
        xM: midiCenter, zM: 0, y,
        cx: T_midi, cz: depthThin, cy: 2,
        dimX: D(BloqueaDim.T_midi), dimY: D(BloqueaDim.D_midi),
        name: 'Output LayerNorm',
        trainState: BloqueaTrainState.TrainableMidi,
    }));
    y += 2 * cell + sectionGap;

    // --- Mean Pooling ---
    let midiMeanPool = add(mk({
        t: 'a',
        xM: midiCenter, zM: 0, y,
        cx: 4, cz: depthThin, cy: D_midi,
        dimX: D(BloqueaDim.B), dimY: D(BloqueaDim.D_midi),
        name: 'Mean Pool (T→1)',
    }));
    y += D_midi * cell + sectionGap;

    // --- MIDI Projection Head ---
    let midiProjLayers: IBlkDef[] = [];
    let mProjCy = [D_midi, D_proj + 2, D_proj];
    let mProjNames = ['Linear + BN + ReLU', 'Linear + BN + ReLU', 'Linear → 256D'];

    for (let i = 0; i < 3; i++) {
        let blk = add(mk({
            t: 'i',
            xM: midiCenter, zM: 0, y,
            cx: 4, cz: depthDeep, cy: mProjCy[i],
            dimX: D(BloqueaDim.B), dimY: i < 2 ? D(BloqueaDim.D_midi) : D(BloqueaDim.D_proj),
            name: `Proj ${mProjNames[i]}`,
            trainState: BloqueaTrainState.TrainableProj,
        }));
        midiProjLayers.push(blk);
        y += mProjCy[i] * cell + layerGap;
    }

    let midiBottomY = y;

    // ==========================================
    //  SHARED SPACE
    // ==========================================
    y = Math.max(audioBottomY, midiBottomY) + sectionGap * 2;

    let sharedSpacing = 40;

    let audioEmbedding = add(mk({
        t: 'i',
        xM: -sharedSpacing / 2, zM: 0, y,
        cx: D_proj, cz: depthMedium, cy: 4,
        dimX: D(BloqueaDim.D_proj), dimY: D(BloqueaDim.B),
        name: 'Audio z (256D)',
    }));
    audioEmbedding.highlight = 0.15;

    let midiEmbedding = add(mk({
        t: 'i',
        xM: sharedSpacing / 2, zM: 0, y,
        cx: D_proj, cz: depthMedium, cy: 4,
        dimX: D(BloqueaDim.D_proj), dimY: D(BloqueaDim.B),
        name: 'MIDI z (256D)',
    }));
    midiEmbedding.highlight = 0.15;

    y += 4 * cell + sectionGap * 1.5;

    // --- VICReg Loss ---
    let lossW = 6;
    let lossSpacing = 8;
    let lossTotalW = lossW * 3 * cell + lossSpacing * 2;
    let lossStartX = -lossTotalW / 2;

    let vicregInv = add(mk({
        t: 'a',
        xL: lossStartX, zM: 0, y,
        cx: lossW, cz: depthDeep, cy: 4,
        dimX: D(BloqueaDim.Loss), dimY: D(BloqueaDim.D_proj),
        name: 'Invariance (MSE)',
    }));
    vicregInv.highlight = 0.4;

    let vicregVar = add(mk({
        t: 'a',
        xL: lossStartX + lossW * cell + lossSpacing, zM: 0, y,
        cx: lossW, cz: depthDeep, cy: 4,
        dimX: D(BloqueaDim.Loss), dimY: D(BloqueaDim.D_proj),
        name: 'Variance (hinge)',
    }));
    vicregVar.highlight = 0.4;

    let vicregCov = add(mk({
        t: 'a',
        xL: lossStartX + (lossW * cell + lossSpacing) * 2, zM: 0, y,
        cx: lossW, cz: depthDeep, cy: 4,
        dimX: D(BloqueaDim.Loss), dimY: D(BloqueaDim.D_proj),
        name: 'Covariance (decorr)',
    }));
    vicregCov.highlight = 0.4;

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

        audioEmbedding, midiEmbedding,
        vicregInv, vicregVar, vicregCov,
    };
}

/** Get all IBlkDef from a transformer layer (for use with dimExcept). */
export function getTransformerLayerBlocks(layer: IBloqueaTransformerLayer): IBlkDef[] {
    let blks = [
        layer.ln1, layer.qWeight, layer.kWeight, layer.vWeight,
        layer.attnMatrix, layer.attnOut, layer.attnResidual,
        layer.ln2, layer.mlpUp, layer.mlpAct, layer.mlpDown, layer.ffnResidual,
    ];
    if (layer.adapter) {
        blks.push(layer.adapter.adapterDown, layer.adapter.adapterAct, layer.adapter.adapterUp);
    }
    return blks;
}

/** Get only adapter blocks from a layer. */
export function getAdapterBlocks(layer: IBloqueaTransformerLayer): IBlkDef[] {
    if (!layer.adapter) return [];
    return [layer.adapter.adapterDown, layer.adapter.adapterAct, layer.adapter.adapterUp];
}
