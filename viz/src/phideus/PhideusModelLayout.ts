import { IBlkDef, IBlkLabel, IModelLayout } from "@/src/llm/GptModelLayout";
import { DimStyle } from "@/src/llm/walkthrough/WalkthroughTools";
import { Vec3 } from "@/src/utils/vector";
import { PhideusDim } from "./PhideusDimStyle";

export interface IPhideusModelShape {
    B: number;
    T_audio: number;
    T_midi: number;
    D_audio: number;
    D_midi: number;
    D_proj: number;
    D_ffn_audio: number;
    D_ffn_midi: number;
    nHeadsAudio: number;
    nHeadsMidi: number;
    nLayers: number;
    cnnStages: number;
}

export const defaultPhideusShape: IPhideusModelShape = {
    B: 1,
    T_audio: 16,
    T_midi: 12,
    D_audio: 16,    // 1024 → 16 cells
    D_midi: 12,     // 512 → 12 cells
    D_proj: 8,      // 256 → 8 cells
    D_ffn_audio: 24, // 4096 → 24 cells (shows expansion from D=16)
    D_ffn_midi: 18,  // 2048 → 18 cells (shows expansion from D=12)
    nHeadsAudio: 8,
    nHeadsMidi: 8,
    nLayers: 4,
    cnnStages: 4,
};

export enum TrainState {
    Frozen,
    TrainableLow,
    TrainableHigh,
    TrainableMidi,
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
    dimX: DimStyle | PhideusDim;
    dimY: DimStyle | PhideusDim;
    small?: boolean;
    hidden?: boolean;
    trainState?: TrainState;
}

export interface IPhideusModelLayout extends IModelLayout {
    cubes: IBlkDef[];
    cell: number;
    height: number;
    margin: number;
    labels: IBlkLabel[];
    shape: IPhideusModelShape;

    // Input representations
    audioInput: IBlkDef;
    midiInput: IBlkDef;

    // Audio tower
    audioCnn: IBlkDef[];
    audioPosEmb: IBlkDef;
    audioTransformerLayers: IPhideusTransformerLayer[];
    audioMeanPool: IBlkDef;
    audioProjLayers: IBlkDef[];

    // MIDI tower
    midiPitchEmb: IBlkDef;
    midiVelEmb: IBlkDef;
    midiDurEmb: IBlkDef;
    midiCombineLinear: IBlkDef;
    midiPosEnc: IBlkDef;
    midiTransformerLayers: IPhideusTransformerLayer[];
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

export interface IPhideusTransformerLayer {
    // Self-Attention sub-block
    ln1: IBlkDef;           // Pre-norm LayerNorm
    qWeight: IBlkDef;       // W_Q projection weight
    kWeight: IBlkDef;       // W_K projection weight
    vWeight: IBlkDef;       // W_V projection weight
    attnMatrix: IBlkDef;    // Attention scores (T × T) — THE iconic block
    attnOut: IBlkDef;       // Attention output (after V weighting + out projection)
    attnResidual: IBlkDef;  // + residual connection

    // FFN sub-block
    ln2: IBlkDef;           // Pre-norm LayerNorm for FFN
    mlpUp: IBlkDef;         // FFN expand weight (D → 4D) — WIDER block
    mlpAct: IBlkDef;        // GELU activation intermediate — WIDER
    mlpDown: IBlkDef;       // FFN contract weight (4D → D) — back to normal
    ffnResidual: IBlkDef;   // + residual connection

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
        case TrainState.TrainableProj:
            blk.opacity = 1.0;
            blk.highlight = 0.28;
            break;
    }
}

export function genPhideusModelLayout(shape: IPhideusModelShape): IPhideusModelLayout {
    let { B, T_audio, T_midi, D_audio, D_midi, D_proj, D_ffn_audio, D_ffn_midi, nLayers, cnnStages } = shape;

    let cell = 2.0;
    let margin = 20;
    let layerGap = 6;
    let sectionGap = 20;
    let subBlockGap = 8;  // gap between attention and FFN sub-blocks

    // Z-depth values for different block types — explore 3D
    let depthThin = 2;       // LN, residuals — flat normalization ops
    let depthMedium = 4;     // intermediates, embeddings
    let depthDeep = 6;       // FFN, projection, VICReg — computational weight
    let depthAttn = 8;       // attention matrix — stacked heads make a cube
    let depthQKV = 6;        // Q/K/V weight projections — show head structure
    let depth = depthMedium; // default for non-transformer blocks

    let cubes: IBlkDef[] = [];
    let labels: IBlkLabel[] = [];

    // Tower separation
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

    // helper for PhideusDim → DimStyle cast
    let D = (d: PhideusDim) => d as unknown as DimStyle;

    // ==========================================
    //  Helper: generate detailed transformer layer
    // ==========================================
    function genTransformerLayer(
        centerX: number,
        yStart: number,
        T: number,
        Dim: number,
        D_ffn: number,
        nHeads: number,
        layerIdx: number,
        dimT: PhideusDim,
        dimD: PhideusDim,
        trainState: TrainState,
        towerName: string,
    ): { layer: IPhideusTransformerLayer, yEnd: number } {
        let y = yStart;

        // --- LN1: Pre-attention LayerNorm (thin — simple operation) ---
        let ln1 = add(mk({
            t: 'a', xM: centerX, zM: 0, y,
            cx: T, cz: depthThin, cy: 2,
            dimX: dimT, dimY: dimD,
            name: `LN1 (${towerName} L${layerIdx})`,
            trainState,
        }));
        y += 2 * cell + layerGap;

        // --- Q/K/V weight projections (3 blocks, staggered in Z for 3D fan-out) ---
        let qkvCx = Math.max(4, Math.floor(Dim / 2));
        let qkvCy = Math.max(4, Math.floor(Dim / 2));
        let qkvSpacing = 3;
        let qkvTotalW = 3 * qkvCx * cell + 2 * qkvSpacing;
        let qkvStartX = centerX - qkvTotalW / 2;
        let qkvZStagger = depthQKV * cell * 0.6; // Z offset between Q, K, V

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

        // --- Attention Matrix (T × T × nHeads) — THE iconic cube ---
        let attnMatrix = add(mk({
            t: 'i', xM: centerX, zM: 0, y,
            cx: T, cz: depthAttn, cy: T,
            dimX: dimT, dimY: dimT,
            name: `Attention (${T}×${T}×${nHeads}h)`,
            trainState,
        }));
        y += T * cell + layerGap;

        // --- Attention Output (after V weighting + output projection) ---
        let attnOut = add(mk({
            t: 'i', xM: centerX, zM: 0, y,
            cx: T, cz: depthMedium, cy: 4,
            dimX: dimT, dimY: dimD,
            name: `Attn Out + W_O`,
            trainState,
        }));
        y += 4 * cell + layerGap;

        // --- Attention Residual Add (thin — just addition) ---
        let attnResidual = add(mk({
            t: 'i', xM: centerX, zM: 0, y,
            cx: T, cz: depthThin, cy: 2,
            dimX: dimT, dimY: dimD,
            name: `+ Residual`,
            trainState,
        }));
        y += 2 * cell + subBlockGap;

        // --- LN2: Pre-FFN LayerNorm (thin) ---
        let ln2 = add(mk({
            t: 'a', xM: centerX, zM: 0, y,
            cx: T, cz: depthThin, cy: 2,
            dimX: dimT, dimY: dimD,
            name: `LN2 (${towerName} L${layerIdx})`,
            trainState,
        }));
        y += 2 * cell + layerGap;

        // --- MLP Up: D → 4D (WIDER and DEEP — shows expansion in X and Z) ---
        let mlpUp = add(mk({
            t: 'w', xM: centerX, zM: 0, y,
            cx: D_ffn, cz: depthDeep, cy: 4,
            dimX: D(PhideusDim.D_ffn), dimY: dimD,
            name: `FFN Up (${Dim === D_audio ? '1024→4096' : '512→2048'})`,
            trainState,
        }));
        y += 4 * cell + layerGap;

        // --- GELU activation (WIDER and DEEP — matches expanded dimension) ---
        let mlpAct = add(mk({
            t: 'i', xM: centerX, zM: 0, y,
            cx: D_ffn, cz: depthDeep, cy: 3,
            dimX: D(PhideusDim.D_ffn), dimY: dimT,
            name: `GELU`,
            trainState,
        }));
        y += 3 * cell + layerGap;

        // --- MLP Down: 4D → D (contracts back — still deep) ---
        let mlpDown = add(mk({
            t: 'w', xM: centerX, zM: 0, y,
            cx: Dim, cz: depthDeep, cy: 4,
            dimX: dimD, dimY: D(PhideusDim.D_ffn),
            name: `FFN Down (${Dim === D_audio ? '4096→1024' : '2048→512'})`,
            trainState,
        }));
        y += 4 * cell + layerGap;

        // --- FFN Residual Add (thin) ---
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

        let layer: IPhideusTransformerLayer = {
            ln1, qWeight, kWeight, vWeight,
            attnMatrix, attnOut, attnResidual,
            ln2, mlpUp, mlpAct, mlpDown, ffnResidual,
            label,
        };

        return { layer, yEnd: y };
    }

    // ==========================================
    //  AUDIO TOWER (left, centered at -100)
    // ==========================================
    let y = 0;

    // --- Waveform Input (thin — raw 1D signal) ---
    let audioInput = add(mk({
        t: 'i',
        xM: audioCenter, zM: 0, y,
        cx: T_audio * 2, cz: depthThin, cy: 1,
        dimX: D(PhideusDim.T_audio), dimY: D(PhideusDim.D_audio),
        name: 'Waveform (24kHz)',
    }));
    audioInput.opacity = 0.7;
    y += 1 * cell + sectionGap;

    // --- CNN Feature Extractor (4 stages, frozen, progressive depth) ---
    let audioCnn: IBlkDef[] = [];
    let cnnWidths  = [T_audio * 2, T_audio + 4, T_audio, T_audio];
    let cnnHeights = [4, 6, 8, D_audio];
    let cnnDepths  = [3, 4, 5, 7]; // progressive depth — channels grow

    for (let i = 0; i < cnnStages; i++) {
        let blk = add(mk({
            t: 'w',
            xM: audioCenter, zM: 0, y,
            cx: cnnWidths[i], cz: cnnDepths[i], cy: cnnHeights[i],
            dimX: D(PhideusDim.T_audio), dimY: D(PhideusDim.CNN_channels),
            name: `Conv ${i} (${[64, 128, 256, 1024][i]}ch)`,
            trainState: TrainState.Frozen,
        }));
        audioCnn.push(blk);
        y += cnnHeights[i] * cell + layerGap;
    }

    y += sectionGap - layerGap;

    // --- Positional Embedding (frozen, learnable, medium depth) ---
    let audioPosEmb = add(mk({
        t: 'w',
        xM: audioCenter, zM: 0, y,
        cx: T_audio, cz: depthMedium, cy: D_audio,
        dimX: D(PhideusDim.T_audio), dimY: D(PhideusDim.D_audio),
        name: 'Learned PosEmb',
        trainState: TrainState.Frozen,
    }));
    y += D_audio * cell + sectionGap;

    // --- Transformer Layers x4 (DETAILED) ---
    let audioTransformerLayers: IPhideusTransformerLayer[] = [];

    for (let i = 0; i < nLayers; i++) {
        let ts = i < 2 ? TrainState.TrainableLow : TrainState.TrainableHigh;
        let result = genTransformerLayer(
            audioCenter, y, T_audio, D_audio, D_ffn_audio,
            shape.nHeadsAudio, i, PhideusDim.T_audio, PhideusDim.D_audio,
            ts, 'Audio',
        );
        audioTransformerLayers.push(result.layer);
        y = result.yEnd;
    }

    // --- Mean Pooling (T→1, thin — aggregation operation) ---
    let audioMeanPool = add(mk({
        t: 'a',
        xM: audioCenter, zM: 0, y,
        cx: 4, cz: depthThin, cy: D_audio,
        dimX: D(PhideusDim.B), dimY: D(PhideusDim.D_audio),
        name: 'Mean Pool (T→1)',
    }));
    y += D_audio * cell + sectionGap;

    // --- Projection Head: 3 layers (deep — heavy computation) ---
    let audioProjLayers: IBlkDef[] = [];
    let aProjCy = [D_audio, D_proj + 2, D_proj];
    let aProjNames = ['Linear + BN + ReLU', 'Linear + BN + ReLU', 'Linear → 256D'];

    for (let i = 0; i < 3; i++) {
        let blk = add(mk({
            t: 'i',
            xM: audioCenter, zM: 0, y,
            cx: 4, cz: depthDeep, cy: aProjCy[i],
            dimX: D(PhideusDim.B), dimY: i < 2 ? D(PhideusDim.D_audio) : D(PhideusDim.D_proj),
            name: `Proj ${aProjNames[i]}`,
            trainState: TrainState.TrainableProj,
        }));
        audioProjLayers.push(blk);
        y += aProjCy[i] * cell + layerGap;
    }

    let audioBottomY = y;

    // ==========================================
    //  MIDI TOWER (right, centered at +100)
    // ==========================================
    y = 0;

    // --- MIDI Event Input (thin — discrete events) ---
    let midiInput = add(mk({
        t: 'i',
        xM: midiCenter, zM: 0, y,
        cx: T_midi, cz: depthThin, cy: 1,
        dimX: D(PhideusDim.T_midi), dimY: D(PhideusDim.D_midi),
        name: 'MIDI Events',
    }));
    midiInput.opacity = 0.7;
    y += 1 * cell + sectionGap;

    // --- Embedding Tables (side by side, staggered Z for 3D fan) ---
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
        dimX: D(PhideusDim.Pitch), dimY: D(PhideusDim.D_midi),
        name: 'Pitch Emb (128→256)',
        trainState: TrainState.TrainableMidi,
    }));

    let midiVelEmb = add(mk({
        t: 'w',
        xL: midiPitchEmb.x + midiPitchEmb.dx + embSpacing, zM: 0, y,
        cx: velW, cz: depthMedium, cy: D_midi / 4,
        dimX: D(PhideusDim.Velocity), dimY: D(PhideusDim.D_midi),
        name: 'Velocity Emb (128→128)',
        trainState: TrainState.TrainableMidi,
    }));

    let midiDurEmb = add(mk({
        t: 'w',
        xL: midiVelEmb.x + midiVelEmb.dx + embSpacing, zM: embZStagger, y,
        cx: durW, cz: depthMedium - 1, cy: D_midi / 4,
        dimX: D(PhideusDim.Duration), dimY: D(PhideusDim.D_midi),
        name: 'Duration Emb (32→128)',
        trainState: TrainState.TrainableMidi,
    }));

    y += (D_midi / 2) * cell + sectionGap;

    // --- Concat + Linear + LayerNorm ---
    let midiCombineLinear = add(mk({
        t: 'i',
        xM: midiCenter, zM: 0, y,
        cx: T_midi, cz: depthMedium, cy: D_midi,
        dimX: D(PhideusDim.T_midi), dimY: D(PhideusDim.D_midi),
        name: 'Concat → Linear → LN',
        trainState: TrainState.TrainableMidi,
    }));
    y += D_midi * cell + layerGap;

    // --- Sinusoidal Positional Encoding ---
    let midiPosEnc = add(mk({
        t: 'w',
        xM: midiCenter, zM: 0, y,
        cx: T_midi, cz: depthMedium, cy: D_midi,
        dimX: D(PhideusDim.T_midi), dimY: D(PhideusDim.D_midi),
        name: 'Sinusoidal PosEnc',
        trainState: TrainState.Frozen,
    }));
    y += D_midi * cell + sectionGap;

    // --- Transformer Layers x4 (DETAILED) ---
    let midiTransformerLayers: IPhideusTransformerLayer[] = [];

    for (let i = 0; i < nLayers; i++) {
        let result = genTransformerLayer(
            midiCenter, y, T_midi, D_midi, D_ffn_midi,
            shape.nHeadsMidi, i, PhideusDim.T_midi, PhideusDim.D_midi,
            TrainState.TrainableMidi, 'MIDI',
        );
        midiTransformerLayers.push(result.layer);
        y = result.yEnd;
    }

    // --- Output LayerNorm (thin) ---
    let midiOutputLN = add(mk({
        t: 'a',
        xM: midiCenter, zM: 0, y,
        cx: T_midi, cz: depthThin, cy: 2,
        dimX: D(PhideusDim.T_midi), dimY: D(PhideusDim.D_midi),
        name: 'Output LayerNorm',
        trainState: TrainState.TrainableMidi,
    }));
    y += 2 * cell + sectionGap;

    // --- Mean Pooling (thin — aggregation) ---
    let midiMeanPool = add(mk({
        t: 'a',
        xM: midiCenter, zM: 0, y,
        cx: 4, cz: depthThin, cy: D_midi,
        dimX: D(PhideusDim.B), dimY: D(PhideusDim.D_midi),
        name: 'Mean Pool (T→1)',
    }));
    y += D_midi * cell + sectionGap;

    // --- Projection Head (deep — heavy projection) ---
    let midiProjLayers: IBlkDef[] = [];
    let mProjCy = [D_midi, D_proj + 2, D_proj];
    let mProjNames = ['Linear + BN + ReLU', 'Linear + BN + ReLU', 'Linear → 256D'];

    for (let i = 0; i < 3; i++) {
        let blk = add(mk({
            t: 'i',
            xM: midiCenter, zM: 0, y,
            cx: 4, cz: depthDeep, cy: mProjCy[i],
            dimX: D(PhideusDim.B), dimY: i < 2 ? D(PhideusDim.D_midi) : D(PhideusDim.D_proj),
            name: `Proj ${mProjNames[i]}`,
            trainState: TrainState.TrainableProj,
        }));
        midiProjLayers.push(blk);
        y += mProjCy[i] * cell + layerGap;
    }

    let midiBottomY = y;

    // ==========================================
    //  SHARED SPACE (center, below both towers)
    // ==========================================
    y = Math.max(audioBottomY, midiBottomY) + sectionGap * 2;

    let sharedSpacing = 40;

    let audioEmbedding = add(mk({
        t: 'i',
        xM: -sharedSpacing / 2, zM: 0, y,
        cx: D_proj, cz: depthMedium, cy: 4,
        dimX: D(PhideusDim.D_proj), dimY: D(PhideusDim.B),
        name: 'Audio z (256D)',
    }));
    audioEmbedding.highlight = 0.15;

    let midiEmbedding = add(mk({
        t: 'i',
        xM: sharedSpacing / 2, zM: 0, y,
        cx: D_proj, cz: depthMedium, cy: 4,
        dimX: D(PhideusDim.D_proj), dimY: D(PhideusDim.B),
        name: 'MIDI z (256D)',
    }));
    midiEmbedding.highlight = 0.15;

    y += 4 * cell + sectionGap * 1.5;

    // --- VICReg Loss: 3 separate blocks (deep — substantial computations) ---
    let lossW = 6;
    let lossSpacing = 8;
    let lossTotalW = lossW * 3 * cell + lossSpacing * 2;
    let lossStartX = -lossTotalW / 2;

    let vicregInv = add(mk({
        t: 'a',
        xL: lossStartX, zM: 0, y,
        cx: lossW, cz: depthDeep, cy: 4,
        dimX: D(PhideusDim.Loss), dimY: D(PhideusDim.D_proj),
        name: 'Invariance (MSE)',
    }));
    vicregInv.highlight = 0.4;

    let vicregVar = add(mk({
        t: 'a',
        xL: lossStartX + lossW * cell + lossSpacing, zM: 0, y,
        cx: lossW, cz: depthDeep, cy: 4,
        dimX: D(PhideusDim.Loss), dimY: D(PhideusDim.D_proj),
        name: 'Variance (hinge)',
    }));
    vicregVar.highlight = 0.4;

    let vicregCov = add(mk({
        t: 'a',
        xL: lossStartX + (lossW * cell + lossSpacing) * 2, zM: 0, y,
        cx: lossW, cz: depthDeep, cy: 4,
        dimX: D(PhideusDim.Loss), dimY: D(PhideusDim.D_proj),
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
export function getTransformerLayerBlocks(layer: IPhideusTransformerLayer): IBlkDef[] {
    return [
        layer.ln1, layer.qWeight, layer.kWeight, layer.vWeight,
        layer.attnMatrix, layer.attnOut, layer.attnResidual,
        layer.ln2, layer.mlpUp, layer.mlpAct, layer.mlpDown, layer.ffnResidual,
    ];
}
