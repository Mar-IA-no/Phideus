import { IBlkDef, IBlkLabel, IModelLayout } from "@/src/llm/GptModelLayout";
import { DimStyle } from "@/src/llm/walkthrough/WalkthroughTools";
import { CrossAttDim } from "./CrossAttDimStyle";

export interface ICrossAttModelShape {
    B: number;
    T_audio: number;     // CNN frames (14 cells → 2400)
    T_midi: number;      // MIDI events (10 cells → N)
    T_stft: number;      // STFT frames (6 cells → 188)
    D_audio: number;     // 1024D → 14 cells
    D_midi: number;      // 512D → 10 cells
    D_proj: number;      // 256D → 8 cells
    D_desc: number;      // A4 descriptor dims (4 cells → 8)
    D_interval: number;  // interval dims (3 cells → 4)
    D_ffn_audio: number; // 4096 → 18 cells
    D_ffn_midi: number;  // 2048 → 14 cells
    D_stft_full: number; // 1025 STFT bins → 8 cells
    D_bands: number;     // 8 log-freq bands → 4 cells
    nHeadsAudio: number;
    nHeadsMidi: number;
    nLayers: number;
    cnnStages: number;
    depthAttn: number;   // nHeads for attention matrix depth
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
    D_ffn_audio: 18,
    D_ffn_midi: 14,
    D_stft_full: 8,
    D_bands: 4,
    nHeadsAudio: 8,
    nHeadsMidi: 8,
    nLayers: 4,
    cnnStages: 4,
    depthAttn: 8,
};

// ==========================================
//  TrainState system (from Phideus, extended)
// ==========================================

export enum TrainState {
    Frozen,          // opacity=0.40, highlight=0.00 (CNN, pos encodings)
    TrainableLow,    // opacity=0.80, highlight=0.10 (lower transformer layers)
    TrainableHigh,   // opacity=1.00, highlight=0.18 (upper transformer layers)
    TrainableMidi,   // opacity=1.00, highlight=0.15 (MIDI encoder)
    TrainableProj,   // opacity=1.00, highlight=0.20 (projection heads)
    TrainableXAtt,   // opacity=1.00, highlight=0.25 (cross-att learnable, cyan tint)
    NoGrad,          // opacity=0.55, highlight=0.00 (descriptor computation — torch.no_grad())
    Ghost,           // opacity=0.25, highlight=0.00 (regular cross-att comparison blocks)
}

function applyTrainState(blk: IBlkDef, state?: TrainState) {
    if (state === undefined) return;
    switch (state) {
        case TrainState.Frozen:
            blk.opacity = 0.40; blk.highlight = 0; break;
        case TrainState.TrainableLow:
            blk.opacity = 0.80; blk.highlight = 0.10; break;
        case TrainState.TrainableHigh:
            blk.opacity = 1.00; blk.highlight = 0.18; break;
        case TrainState.TrainableMidi:
            blk.opacity = 1.00; blk.highlight = 0.15; break;
        case TrainState.TrainableProj:
            blk.opacity = 1.00; blk.highlight = 0.20; break;
        case TrainState.TrainableXAtt:
            blk.opacity = 1.00; blk.highlight = 0.25; break;
        case TrainState.NoGrad:
            blk.opacity = 0.55; blk.highlight = 0.00; break;
        case TrainState.Ghost:
            blk.opacity = 0.25; blk.highlight = 0.00; break;
    }
}

// ==========================================
//  Block definition args
// ==========================================

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
    small?: boolean;
    trainState?: TrainState;
}

// ==========================================
//  Transformer layer interface (from Phideus)
// ==========================================

export interface ICrossAttTransformerLayer {
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

// ==========================================
//  Main layout interface
// ==========================================

export interface ICrossAttModelLayout extends IModelLayout {
    cubes: IBlkDef[];
    cell: number;
    height: number;
    margin: number;
    labels: IBlkLabel[];
    shape: ICrossAttModelShape;

    // ===== AUDIO TOWER =====

    // Input + CNN (6 blocks)
    waveformInput: IBlkDef;
    audioCnn: IBlkDef[];          // 4 CNN stages
    audioPosEmb: IBlkDef;

    // Audio Descriptor A4 — DSP Pipeline (8 blocks, all NoGrad)
    audioDescStftWindow: IBlkDef;
    audioDescStftCompute: IBlkDef;
    audioDescMagnitude: IBlkDef;
    audioDescLogMag: IBlkDef;
    audioDescBandGroup: IBlkDef;
    audioDescTemporalDelta: IBlkDef;
    audioDescNormalize: IBlkDef;
    audioDescOutput: IBlkDef;

    // Audio REVERSE Cross-Attention (main, bright, 10 blocks)
    audioReverseDescQProj: IBlkDef;
    audioReverseDescPosEmb: IBlkDef;
    audioReverseQ: IBlkDef;
    audioReverseK: IBlkDef;
    audioReverseMatrix: IBlkDef;
    audioReverseAttnOut: IBlkDef;
    audioReverseResidual: IBlkDef;
    audioReverseNorm: IBlkDef;
    audioReverseOutput: IBlkDef;

    // Audio Transformer (4 layers x 12 blocks = 48 blocks)
    audioTransformerLayers: ICrossAttTransformerLayer[];

    // Audio post-processing (5 blocks)
    audioOutputLN: IBlkDef;
    audioMeanPool: IBlkDef;
    audioProjLayer1: IBlkDef;
    audioProjLayer2: IBlkDef;
    audioProjLayer3: IBlkDef;

    // ===== MIDI TOWER =====

    // Input + Embedding (6 blocks)
    midiInput: IBlkDef;
    midiPitchEmb: IBlkDef;
    midiVelEmb: IBlkDef;
    midiDurEmb: IBlkDef;
    midiCombineLinear: IBlkDef;
    midiPosEnc: IBlkDef;

    // MIDI Descriptor D4 — Interval Pipeline (6 blocks, all NoGrad)
    midiDescPitchInput: IBlkDef;
    midiDescForwardDiff: IBlkDef;
    midiDescValidityMask: IBlkDef;
    midiDescSemitoneScale: IBlkDef;
    midiDescLogRatioScale: IBlkDef;
    midiDescOutput: IBlkDef;

    // MIDI REVERSE Cross-Attention (main, bright, 8 blocks)
    midiReverseIntQProj: IBlkDef;
    midiReversePosEnc: IBlkDef;
    midiReverseQ: IBlkDef;
    midiReverseK: IBlkDef;
    midiReverseMatrix: IBlkDef;
    midiReverseResidual: IBlkDef;
    midiReverseOutput: IBlkDef;

    // MIDI Transformer (4 layers x 12 blocks = 48 blocks)
    midiTransformerLayers: ICrossAttTransformerLayer[];

    // MIDI post-processing (5 blocks)
    midiOutputLN: IBlkDef;
    midiMeanPool: IBlkDef;
    midiProjLayer1: IBlkDef;
    midiProjLayer2: IBlkDef;
    midiProjLayer3: IBlkDef;

    // ===== SHARED SPACE =====
    audioEmbedding: IBlkDef;
    midiEmbedding: IBlkDef;
    vicregInv: IBlkDef;
    vicregVar: IBlkDef;
    vicregCov: IBlkDef;
}

// ==========================================
//  Layout generation
// ==========================================

export function genCrossAttModelLayout(shape: ICrossAttModelShape): ICrossAttModelLayout {
    let { T_audio, T_midi, T_stft, D_audio, D_midi, D_proj, D_desc, D_interval,
          D_ffn_audio, D_ffn_midi, D_stft_full, D_bands, nLayers, cnnStages } = shape;

    let cell = 2.0;
    let margin = 20;
    let layerGap = 6;
    let sectionGap = 20;
    let subBlockGap = 8;

    let depthThin = 2;
    let depthMedium = 4;
    let depthDeep = 6;
    let depthAttn = shape.depthAttn;
    let depthQKV = 6;

    let cubes: IBlkDef[] = [];
    let labels: IBlkLabel[] = [];

    let audioCenter = -100;
    let midiCenter = 100;

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

        let blk: IBlkDef = {
            dx, dy: args.cy * cell, dz,
            t: args.t,
            x, y: args.y, z,
            cx: args.cx, cy: args.cy, cz: args.cz,
            dimX: args.dimX as DimStyle,
            dimY: args.dimY as DimStyle,
            name: args.name ?? '',
            opacity: args.opacity ?? 1.0,
            highlight: args.highlight ?? 0.0,
            small: args.small ?? false,
            special: args.special ?? 0,
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
        dimT: CrossAttDim,
        dimD: CrossAttDim,
        trainState: TrainState,
        towerName: string,
    ): { layer: ICrossAttTransformerLayer, yEnd: number } {
        let y = yStart;

        let ln1 = add(mk({
            t: 'a', xM: centerX, zM: 0, y,
            cx: T, cz: depthThin, cy: 2,
            dimX: D(dimT), dimY: D(dimD),
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
            dimX: D(dimD), dimY: D(dimD),
            name: `W_Q (${nHeads}h)`,
            trainState,
        }));

        let kWeight = add(mk({
            t: 'w', xL: qkvStartX + qkvCx * cell + qkvSpacing, zM: 0, y,
            cx: qkvCx, cz: depthQKV, cy: qkvCy,
            dimX: D(dimD), dimY: D(dimD),
            name: `W_K (${nHeads}h)`,
            trainState,
        }));

        let vWeight = add(mk({
            t: 'w', xL: qkvStartX + (qkvCx * cell + qkvSpacing) * 2, zM: qkvZStagger, y,
            cx: qkvCx, cz: depthQKV, cy: qkvCy,
            dimX: D(dimD), dimY: D(dimD),
            name: `W_V (${nHeads}h)`,
            trainState,
        }));
        y += qkvCy * cell + layerGap;

        let attnMatrix = add(mk({
            t: 'i', xM: centerX, zM: 0, y,
            cx: T, cz: depthAttn, cy: T,
            dimX: D(dimT), dimY: D(dimT),
            name: `Attention (${T}x${T}x${nHeads}h)`,
            trainState,
        }));
        y += T * cell + layerGap;

        let attnOut = add(mk({
            t: 'i', xM: centerX, zM: 0, y,
            cx: T, cz: depthMedium, cy: 4,
            dimX: D(dimT), dimY: D(dimD),
            name: `Attn Out + W_O`,
            trainState,
        }));
        y += 4 * cell + layerGap;

        let attnResidual = add(mk({
            t: 'i', xM: centerX, zM: 0, y,
            cx: T, cz: depthThin, cy: 2,
            dimX: D(dimT), dimY: D(dimD),
            name: `+ Residual`,
            trainState,
        }));
        y += 2 * cell + subBlockGap;

        let ln2 = add(mk({
            t: 'a', xM: centerX, zM: 0, y,
            cx: T, cz: depthThin, cy: 2,
            dimX: D(dimT), dimY: D(dimD),
            name: `LN2 (${towerName} L${layerIdx})`,
            trainState,
        }));
        y += 2 * cell + layerGap;

        let ffnDimT = Dim === D_audio ? CrossAttDim.D_ffn_audio : CrossAttDim.D_ffn_midi;
        let ffnLabel = Dim === D_audio ? '1024->4096' : '512->2048';
        let ffnLabelDown = Dim === D_audio ? '4096->1024' : '2048->512';

        let mlpUp = add(mk({
            t: 'w', xM: centerX, zM: 0, y,
            cx: D_ffn, cz: depthDeep, cy: 4,
            dimX: D(ffnDimT), dimY: D(dimD),
            name: `FFN Up (${ffnLabel})`,
            trainState,
        }));
        y += 4 * cell + layerGap;

        let mlpAct = add(mk({
            t: 'i', xM: centerX, zM: 0, y,
            cx: D_ffn, cz: depthDeep, cy: 3,
            dimX: D(ffnDimT), dimY: D(dimT),
            name: `GELU`,
            trainState,
        }));
        y += 3 * cell + layerGap;

        let mlpDown = add(mk({
            t: 'w', xM: centerX, zM: 0, y,
            cx: Dim, cz: depthDeep, cy: 4,
            dimX: D(dimD), dimY: D(ffnDimT),
            name: `FFN Down (${ffnLabelDown})`,
            trainState,
        }));
        y += 4 * cell + layerGap;

        let ffnResidual = add(mk({
            t: 'i', xM: centerX, zM: 0, y,
            cx: T, cz: depthThin, cy: 2,
            dimX: D(dimT), dimY: D(dimD),
            name: `+ Residual`,
            trainState,
        }));
        y += 2 * cell + sectionGap;

        let allBlks = [ln1, qWeight, kWeight, vWeight, attnMatrix, attnOut, attnResidual, ln2, mlpUp, mlpAct, mlpDown, ffnResidual];
        let label = mkLabel(allBlks);
        labels.push(label);

        let layer: ICrossAttTransformerLayer = {
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

    // --- Waveform Input ---
    let waveformInput = add(mk({
        t: 'i', xM: audioCenter, zM: 0, y,
        cx: T_audio * 2, cz: depthThin, cy: 1,
        dimX: D(CrossAttDim.T_audio), dimY: D(CrossAttDim.D_audio),
        name: 'Waveform (24kHz)', opacity: 0.7,
    }));
    y += 1 * cell + sectionGap;

    // --- CNN Feature Extractor (4 stages, frozen, progressive) ---
    let audioCnn: IBlkDef[] = [];
    let cnnWidths  = [T_audio * 2, T_audio + 4, T_audio, T_audio];
    let cnnHeights = [4, 6, 8, D_audio];
    let cnnDepths  = [3, 4, 5, 7];

    for (let i = 0; i < cnnStages; i++) {
        let blk = add(mk({
            t: 'w', xM: audioCenter, zM: 0, y,
            cx: cnnWidths[i], cz: cnnDepths[i], cy: cnnHeights[i],
            dimX: D(CrossAttDim.T_audio), dimY: D(CrossAttDim.CNN_channels),
            name: `Conv ${i} (${[64, 128, 256, 1024][i]}ch)`,
            trainState: TrainState.Frozen,
        }));
        audioCnn.push(blk);
        y += cnnHeights[i] * cell + layerGap;
    }
    y += sectionGap - layerGap;

    // --- Positional Embedding (frozen, learnable) ---
    let audioPosEmb = add(mk({
        t: 'w', xM: audioCenter, zM: 0, y,
        cx: T_audio, cz: depthMedium, cy: D_audio,
        dimX: D(CrossAttDim.T_audio), dimY: D(CrossAttDim.D_audio),
        name: 'Learned PosEmb',
        trainState: TrainState.Frozen,
    }));
    y += D_audio * cell + sectionGap;

    // Save Y position after CNN for descriptor alignment
    let audioCnnBottomY = y;

    // ==========================================
    //  AUDIO DESCRIPTOR A4 — DSP Pipeline (far left)
    // ==========================================
    let audioDescCenter = audioCenter - 100;
    let descY = waveformInput.y; // start from top, aligned with waveform

    let audioDescStftWindow = add(mk({
        t: 'i', xM: audioDescCenter, zM: 0, y: descY,
        cx: D_desc, cz: depthThin, cy: 2,
        dimX: D(CrossAttDim.D_desc_a4), dimY: D(CrossAttDim.NoGrad),
        name: 'Hann Window (n=2048)',
        trainState: TrainState.NoGrad,
    }));
    descY += 2 * cell + layerGap;

    let audioDescStftCompute = add(mk({
        t: 'i', xM: audioDescCenter, zM: 0, y: descY,
        cx: D_stft_full, cz: depthMedium, cy: T_stft,
        dimX: D(CrossAttDim.D_stft_bins), dimY: D(CrossAttDim.T_stft),
        name: 'STFT (hop=512)',
        trainState: TrainState.NoGrad,
    }));
    descY += T_stft * cell + layerGap;

    let audioDescMagnitude = add(mk({
        t: 'i', xM: audioDescCenter, zM: 0, y: descY,
        cx: D_stft_full, cz: 3, cy: T_stft,
        dimX: D(CrossAttDim.D_stft_bins), dimY: D(CrossAttDim.T_stft),
        name: '|magnitude|',
        trainState: TrainState.NoGrad,
    }));
    descY += T_stft * cell + layerGap;

    let audioDescLogMag = add(mk({
        t: 'i', xM: audioDescCenter, zM: 0, y: descY,
        cx: D_stft_full, cz: 3, cy: T_stft,
        dimX: D(CrossAttDim.D_stft_bins), dimY: D(CrossAttDim.T_stft),
        name: 'log(1+x)',
        trainState: TrainState.NoGrad,
    }));
    descY += T_stft * cell + layerGap;

    // Band grouping: SHRINKS width from D_stft_full to D_bands (1025→8)
    let audioDescBandGroup = add(mk({
        t: 'i', xM: audioDescCenter, zM: 0, y: descY,
        cx: D_bands, cz: depthMedium, cy: T_stft,
        dimX: D(CrossAttDim.D_bands), dimY: D(CrossAttDim.T_stft),
        name: '8 Log-Freq Bands',
        trainState: TrainState.NoGrad,
    }));
    descY += T_stft * cell + layerGap;

    let audioDescTemporalDelta = add(mk({
        t: 'i', xM: audioDescCenter, zM: 0, y: descY,
        cx: D_bands, cz: 3, cy: T_stft,
        dimX: D(CrossAttDim.D_bands), dimY: D(CrossAttDim.T_stft),
        name: 'Temporal Delta (diff)',
        trainState: TrainState.NoGrad,
    }));
    descY += T_stft * cell + layerGap;

    let audioDescNormalize = add(mk({
        t: 'i', xM: audioDescCenter, zM: 0, y: descY,
        cx: D_bands, cz: depthThin, cy: T_stft,
        dimX: D(CrossAttDim.D_bands), dimY: D(CrossAttDim.T_stft),
        name: 'Z-Score Norm',
        trainState: TrainState.NoGrad,
    }));
    descY += T_stft * cell + layerGap;

    // Transpose: cx↔cy swap — [B, 8, 188] → [B, 188, 8]
    let audioDescOutput = add(mk({
        t: 'i', xM: audioDescCenter, zM: 0, y: descY,
        cx: T_stft, cz: depthThin, cy: D_bands,
        dimX: D(CrossAttDim.T_stft), dimY: D(CrossAttDim.D_desc_a4),
        name: '[B, 188, 8] (transposed)',
        trainState: TrainState.NoGrad,
    }));
    descY += D_bands * cell + sectionGap;

    let qkSpacing = 4;

    // ==========================================
    //  AUDIO REVERSE CROSS-ATTENTION (main, bright)
    // ==========================================
    let audioReverseY = audioCnnBottomY;

    let audioReverseDescQProj = add(mk({
        t: 'w', xM: audioCenter, zM: 0, y: audioReverseY,
        cx: T_stft, cz: 3, cy: D_audio,
        dimX: D(CrossAttDim.T_stft), dimY: D(CrossAttDim.D_audio),
        name: 'Q Proj (8->1024)',
        trainState: TrainState.TrainableXAtt,
    }));
    audioReverseY += D_audio * cell + layerGap;

    // Learned positional embedding for descriptor (ONLY in reverse!)
    let audioReverseDescPosEmb = add(mk({
        t: 'w', xM: audioCenter, zM: 0, y: audioReverseY,
        cx: T_stft, cz: depthThin, cy: 3,
        dimX: D(CrossAttDim.T_stft), dimY: D(CrossAttDim.D_audio),
        name: '+Desc PosEmb (learned)',
        trainState: TrainState.TrainableXAtt,
    }));
    audioReverseY += 3 * cell + layerGap;

    // Q = descriptor (NARROW), K = features (WIDE)
    let audioReverseQ = add(mk({
        t: 'i', xL: audioCenter - (T_stft * cell + qkSpacing) / 2, zM: 0, y: audioReverseY,
        cx: T_stft, cz: 3, cy: 4,
        dimX: D(CrossAttDim.T_stft), dimY: D(CrossAttDim.D_audio),
        name: 'Q: Desc (188)',
        trainState: TrainState.TrainableXAtt,
    }));

    let audioReverseK = add(mk({
        t: 'i', xL: audioCenter + qkSpacing / 2, zM: 0, y: audioReverseY,
        cx: T_audio, cz: 3, cy: 4,
        dimX: D(CrossAttDim.T_audio), dimY: D(CrossAttDim.D_audio),
        name: 'K: Features (2400)',
        trainState: TrainState.TrainableXAtt,
    }));
    audioReverseY += 4 * cell + layerGap;

    // Attention matrix: WIDE rectangle [188 x 2400] — TRANSPOSED vs regular!
    let audioReverseMatrix = add(mk({
        t: 'i', xM: audioCenter, zM: 0, y: audioReverseY,
        cx: T_stft, cz: depthAttn, cy: T_audio,
        dimX: D(CrossAttDim.T_stft), dimY: D(CrossAttDim.T_audio),
        name: 'Attn [188x2400]',
        trainState: TrainState.TrainableXAtt, special: 1,
    }));
    audioReverseY += T_audio * cell + layerGap;

    let audioReverseAttnOut = add(mk({
        t: 'i', xM: audioCenter, zM: 0, y: audioReverseY,
        cx: T_stft, cz: depthMedium, cy: 4,
        dimX: D(CrossAttDim.T_stft), dimY: D(CrossAttDim.D_audio),
        name: 'Attn Out + W_O',
        trainState: TrainState.TrainableXAtt,
    }));
    audioReverseY += 4 * cell + layerGap;

    let audioReverseResidual = add(mk({
        t: 'i', xM: audioCenter, zM: 0, y: audioReverseY,
        cx: T_stft, cz: depthThin, cy: 2,
        dimX: D(CrossAttDim.T_stft), dimY: D(CrossAttDim.D_audio),
        name: '+ Residual',
        trainState: TrainState.TrainableXAtt,
    }));
    audioReverseY += 2 * cell + layerGap;

    let audioReverseNorm = add(mk({
        t: 'a', xM: audioCenter, zM: 0, y: audioReverseY,
        cx: T_stft, cz: depthThin, cy: 2,
        dimX: D(CrossAttDim.T_stft), dimY: D(CrossAttDim.D_audio),
        name: 'LayerNorm',
        trainState: TrainState.TrainableXAtt,
    }));
    audioReverseY += 2 * cell + layerGap;

    let audioReverseOutput = add(mk({
        t: 'i', xM: audioCenter, zM: 0, y: audioReverseY,
        cx: T_stft, cz: depthThin, cy: 2,
        dimX: D(CrossAttDim.T_stft), dimY: D(CrossAttDim.D_audio),
        name: '-> 188 tokens (12.8x)',
        trainState: TrainState.TrainableXAtt,
    }));
    audioReverseY += 2 * cell + sectionGap;

    // ==========================================
    //  AUDIO TRANSFORMER (4 layers, T_stft width!)
    // ==========================================
    // IMPORTANT: uses T_stft=6 width (188 tokens), NOT T_audio=14!
    // This is the consequence of reverse cross-attention compression.
    y = audioReverseY;
    let audioTransformerLayers: ICrossAttTransformerLayer[] = [];

    for (let i = 0; i < nLayers; i++) {
        let ts = i < 2 ? TrainState.TrainableLow : TrainState.TrainableHigh;
        let result = genTransformerLayer(
            audioCenter, y, T_stft, D_audio, D_ffn_audio,
            shape.nHeadsAudio, i, CrossAttDim.T_stft, CrossAttDim.D_audio,
            ts, 'Audio',
        );
        audioTransformerLayers.push(result.layer);
        y = result.yEnd;
    }

    // --- Audio post-processing ---
    let audioOutputLN = add(mk({
        t: 'a', xM: audioCenter, zM: 0, y,
        cx: T_stft, cz: depthThin, cy: 2,
        dimX: D(CrossAttDim.T_stft), dimY: D(CrossAttDim.D_audio),
        name: 'Output LayerNorm',
    }));
    y += 2 * cell + sectionGap;

    let audioMeanPool = add(mk({
        t: 'a', xM: audioCenter, zM: 0, y,
        cx: 4, cz: depthThin, cy: D_audio,
        dimX: D(CrossAttDim.B), dimY: D(CrossAttDim.D_audio),
        name: 'Mean Pool (T->1)',
    }));
    y += D_audio * cell + sectionGap;

    let audioProjLayer1 = add(mk({
        t: 'i', xM: audioCenter, zM: 0, y,
        cx: 4, cz: depthDeep, cy: D_audio,
        dimX: D(CrossAttDim.B), dimY: D(CrossAttDim.D_audio),
        name: 'Proj 1024->512',
        trainState: TrainState.TrainableProj,
    }));
    y += D_audio * cell + layerGap;

    let audioProjLayer2 = add(mk({
        t: 'i', xM: audioCenter, zM: 0, y,
        cx: 4, cz: depthDeep, cy: D_proj + 2,
        dimX: D(CrossAttDim.B), dimY: D(CrossAttDim.D_audio),
        name: 'Proj 512->512',
        trainState: TrainState.TrainableProj,
    }));
    y += (D_proj + 2) * cell + layerGap;

    let audioProjLayer3 = add(mk({
        t: 'i', xM: audioCenter, zM: 0, y,
        cx: 4, cz: depthDeep, cy: D_proj,
        dimX: D(CrossAttDim.B), dimY: D(CrossAttDim.D_proj),
        name: 'Proj 512->256',
        trainState: TrainState.TrainableProj,
    }));
    y += D_proj * cell + sectionGap;
    let audioBottomY = y;

    // ==========================================
    //  MIDI TOWER (right, centered at +100)
    // ==========================================
    y = 0;

    // --- MIDI Event Input ---
    let midiInput = add(mk({
        t: 'i', xM: midiCenter, zM: 0, y,
        cx: T_midi, cz: depthThin, cy: 1,
        dimX: D(CrossAttDim.T_midi), dimY: D(CrossAttDim.D_midi),
        name: 'MIDI Events', opacity: 0.7,
    }));
    y += 1 * cell + sectionGap;

    // --- Embedding Tables (3 blocks, Z-staggered like Phideus) ---
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
        dimX: D(CrossAttDim.Pitch), dimY: D(CrossAttDim.D_midi),
        name: 'Pitch Emb (128->512)',
        trainState: TrainState.TrainableMidi,
    }));

    let midiVelEmb = add(mk({
        t: 'w', xL: midiPitchEmb.x + midiPitchEmb.dx + embSpacing, zM: 0, y,
        cx: velW, cz: depthMedium, cy: D_midi / 4,
        dimX: D(CrossAttDim.Velocity), dimY: D(CrossAttDim.D_midi),
        name: 'Velocity Emb (128->512)',
        trainState: TrainState.TrainableMidi,
    }));

    let midiDurEmb = add(mk({
        t: 'w', xL: midiVelEmb.x + midiVelEmb.dx + embSpacing, zM: embZStagger, y,
        cx: durW, cz: depthMedium - 1, cy: D_midi / 4,
        dimX: D(CrossAttDim.Duration), dimY: D(CrossAttDim.D_midi),
        name: 'Duration Emb (32->512)',
        trainState: TrainState.TrainableMidi,
    }));
    y += (D_midi / 2) * cell + sectionGap;

    // --- Concat + Linear + LN ---
    let midiCombineLinear = add(mk({
        t: 'i', xM: midiCenter, zM: 0, y,
        cx: T_midi, cz: depthMedium, cy: D_midi,
        dimX: D(CrossAttDim.T_midi), dimY: D(CrossAttDim.D_midi),
        name: 'Concat -> Linear(1536->512) + LN',
        trainState: TrainState.TrainableMidi,
    }));
    y += D_midi * cell + layerGap;

    // --- Sinusoidal PosEnc (frozen) ---
    let midiPosEnc = add(mk({
        t: 'w', xM: midiCenter, zM: 0, y,
        cx: T_midi, cz: depthMedium, cy: D_midi,
        dimX: D(CrossAttDim.T_midi), dimY: D(CrossAttDim.D_midi),
        name: 'Sinusoidal PosEnc',
        trainState: TrainState.Frozen,
    }));
    y += D_midi * cell + sectionGap;

    let midiCnnBottomY = y;

    // ==========================================
    //  MIDI DESCRIPTOR D4 — Interval Pipeline (far right)
    // ==========================================
    let midiIntCenter = midiCenter + 100;
    let intY = midiInput.y; // start from top

    let midiDescPitchInput = add(mk({
        t: 'i', xM: midiIntCenter, zM: 0, y: intY,
        cx: T_midi, cz: depthThin, cy: 2,
        dimX: D(CrossAttDim.T_midi), dimY: D(CrossAttDim.D_interval),
        name: '[B, N] Pitch',
        trainState: TrainState.NoGrad,
    }));
    intY += 2 * cell + layerGap;

    let midiDescForwardDiff = add(mk({
        t: 'i', xM: midiIntCenter, zM: 0, y: intY,
        cx: T_midi, cz: 3, cy: 2,
        dimX: D(CrossAttDim.T_midi), dimY: D(CrossAttDim.D_interval),
        name: 'Fwd Diff (pitch[i+1]-pitch[i])',
        trainState: TrainState.NoGrad,
    }));
    intY += 2 * cell + layerGap;

    let midiDescValidityMask = add(mk({
        t: 'i', xM: midiIntCenter, zM: 0, y: intY,
        cx: T_midi, cz: depthThin, cy: 2,
        dimX: D(CrossAttDim.T_midi), dimY: D(CrossAttDim.D_interval),
        name: 'Validity Mask',
        trainState: TrainState.NoGrad,
    }));
    intY += 2 * cell + layerGap;

    let midiDescSemitoneScale = add(mk({
        t: 'i', xM: midiIntCenter, zM: 0, y: intY,
        cx: T_midi, cz: 3, cy: 2,
        dimX: D(CrossAttDim.T_midi), dimY: D(CrossAttDim.D_semitone),
        name: '/24 (semitone prev,next)',
        trainState: TrainState.NoGrad,
    }));
    intY += 2 * cell + layerGap;

    let midiDescLogRatioScale = add(mk({
        t: 'i', xM: midiIntCenter, zM: 0, y: intY,
        cx: T_midi, cz: 3, cy: 2,
        dimX: D(CrossAttDim.T_midi), dimY: D(CrossAttDim.D_log_ratio),
        name: '/12, clamp[-2,2], /2',
        trainState: TrainState.NoGrad,
    }));
    intY += 2 * cell + layerGap;

    let midiDescOutput = add(mk({
        t: 'i', xM: midiIntCenter, zM: 0, y: intY,
        cx: T_midi, cz: depthThin, cy: D_interval,
        dimX: D(CrossAttDim.T_midi), dimY: D(CrossAttDim.D_interval),
        name: '[B, N, 4] intervals',
        trainState: TrainState.NoGrad,
    }));
    intY += D_interval * cell + sectionGap;

    // ==========================================
    //  MIDI REVERSE CROSS-ATTENTION (main, bright)
    // ==========================================
    let midiReverseY = midiCnnBottomY;

    let midiReverseIntQProj = add(mk({
        t: 'w', xM: midiCenter, zM: 0, y: midiReverseY,
        cx: T_midi, cz: 3, cy: D_midi,
        dimX: D(CrossAttDim.T_midi), dimY: D(CrossAttDim.D_midi),
        name: 'Q Proj (4->512)',
        trainState: TrainState.TrainableXAtt,
    }));
    midiReverseY += D_midi * cell + layerGap;

    // Shared sinusoidal pos encoding
    let midiReversePosEnc = add(mk({
        t: 'w', xM: midiCenter, zM: 0, y: midiReverseY,
        cx: T_midi, cz: depthThin, cy: 3,
        dimX: D(CrossAttDim.T_midi), dimY: D(CrossAttDim.D_midi),
        name: '+PosEnc (shared)',
        trainState: TrainState.TrainableXAtt,
    }));
    midiReverseY += 3 * cell + layerGap;

    // Q = intervals, K = embeddings — SWAPPED vs regular
    let midiReverseQ = add(mk({
        t: 'i', xL: midiCenter - (T_midi * cell + qkSpacing) / 2, zM: 0, y: midiReverseY,
        cx: T_midi, cz: 3, cy: 4,
        dimX: D(CrossAttDim.T_midi), dimY: D(CrossAttDim.D_midi),
        name: 'Q: Intervals',
        trainState: TrainState.TrainableXAtt,
    }));

    let midiReverseK = add(mk({
        t: 'i', xL: midiCenter + qkSpacing / 2, zM: 0, y: midiReverseY,
        cx: T_midi, cz: 3, cy: 4,
        dimX: D(CrossAttDim.T_midi), dimY: D(CrossAttDim.D_midi),
        name: 'K: Embeddings',
        trainState: TrainState.TrainableXAtt,
    }));
    midiReverseY += 4 * cell + layerGap;

    // Attention matrix: SQUARE [N x N] — same shape as regular, but Q/K/V roles swapped!
    let midiReverseMatrix = add(mk({
        t: 'i', xM: midiCenter, zM: 0, y: midiReverseY,
        cx: T_midi, cz: depthAttn, cy: T_midi,
        dimX: D(CrossAttDim.T_midi), dimY: D(CrossAttDim.T_midi),
        name: 'Attn [NxN]',
        trainState: TrainState.TrainableXAtt, special: 1,
    }));
    midiReverseY += T_midi * cell + layerGap;

    let midiReverseResidual = add(mk({
        t: 'i', xM: midiCenter, zM: 0, y: midiReverseY,
        cx: T_midi, cz: depthThin, cy: 2,
        dimX: D(CrossAttDim.T_midi), dimY: D(CrossAttDim.D_midi),
        name: '+Residual + LN',
        trainState: TrainState.TrainableXAtt,
    }));
    midiReverseY += 2 * cell + layerGap;

    let midiReverseOutput = add(mk({
        t: 'i', xM: midiCenter, zM: 0, y: midiReverseY,
        cx: T_midi, cz: depthThin, cy: 2,
        dimX: D(CrossAttDim.T_midi), dimY: D(CrossAttDim.D_midi),
        name: '-> N tokens',
        trainState: TrainState.TrainableXAtt,
    }));
    midiReverseY += 2 * cell + sectionGap;

    // ==========================================
    //  MIDI TRANSFORMER (4 layers, T_midi width)
    // ==========================================
    y = midiReverseY;
    let midiTransformerLayers: ICrossAttTransformerLayer[] = [];

    for (let i = 0; i < nLayers; i++) {
        let result = genTransformerLayer(
            midiCenter, y, T_midi, D_midi, D_ffn_midi,
            shape.nHeadsMidi, i, CrossAttDim.T_midi, CrossAttDim.D_midi,
            TrainState.TrainableMidi, 'MIDI',
        );
        midiTransformerLayers.push(result.layer);
        y = result.yEnd;
    }

    // --- MIDI post-processing ---
    let midiOutputLN = add(mk({
        t: 'a', xM: midiCenter, zM: 0, y,
        cx: T_midi, cz: depthThin, cy: 2,
        dimX: D(CrossAttDim.T_midi), dimY: D(CrossAttDim.D_midi),
        name: 'Output LayerNorm',
        trainState: TrainState.TrainableMidi,
    }));
    y += 2 * cell + sectionGap;

    let midiMeanPool = add(mk({
        t: 'a', xM: midiCenter, zM: 0, y,
        cx: 4, cz: depthThin, cy: D_midi,
        dimX: D(CrossAttDim.B), dimY: D(CrossAttDim.D_midi),
        name: 'Mean Pool (T->1)',
    }));
    y += D_midi * cell + sectionGap;

    let midiProjLayer1 = add(mk({
        t: 'i', xM: midiCenter, zM: 0, y,
        cx: 4, cz: depthDeep, cy: D_midi,
        dimX: D(CrossAttDim.B), dimY: D(CrossAttDim.D_midi),
        name: 'Proj 512->512',
        trainState: TrainState.TrainableProj,
    }));
    y += D_midi * cell + layerGap;

    let midiProjLayer2 = add(mk({
        t: 'i', xM: midiCenter, zM: 0, y,
        cx: 4, cz: depthDeep, cy: D_proj + 2,
        dimX: D(CrossAttDim.B), dimY: D(CrossAttDim.D_midi),
        name: 'Proj 512->512',
        trainState: TrainState.TrainableProj,
    }));
    y += (D_proj + 2) * cell + layerGap;

    let midiProjLayer3 = add(mk({
        t: 'i', xM: midiCenter, zM: 0, y,
        cx: 4, cz: depthDeep, cy: D_proj,
        dimX: D(CrossAttDim.B), dimY: D(CrossAttDim.D_proj),
        name: 'Proj 512->256',
        trainState: TrainState.TrainableProj,
    }));
    y += D_proj * cell + sectionGap;
    let midiBottomY = y;

    // ==========================================
    //  SHARED SPACE (center, below both towers)
    // ==========================================
    y = Math.max(audioBottomY, midiBottomY) + sectionGap * 2;
    let sharedSpacing = 40;

    let audioEmbedding = add(mk({
        t: 'i', xM: -sharedSpacing / 2, zM: 0, y,
        cx: D_proj, cz: depthMedium, cy: 4,
        dimX: D(CrossAttDim.D_proj), dimY: D(CrossAttDim.B),
        name: 'Audio z (256D)',
    }));
    audioEmbedding.highlight = 0.15;

    let midiEmbedding = add(mk({
        t: 'i', xM: sharedSpacing / 2, zM: 0, y,
        cx: D_proj, cz: depthMedium, cy: 4,
        dimX: D(CrossAttDim.D_proj), dimY: D(CrossAttDim.B),
        name: 'MIDI z (256D)',
    }));
    midiEmbedding.highlight = 0.15;

    y += 4 * cell + sectionGap * 1.5;

    // --- VICReg Loss: 3 separate blocks ---
    let lossW = 6;
    let lossSpacing = 8;
    let lossTotalW = lossW * 3 * cell + lossSpacing * 2;
    let lossStartX = -lossTotalW / 2;

    let vicregInv = add(mk({
        t: 'a', xL: lossStartX, zM: 0, y,
        cx: lossW, cz: depthDeep, cy: 4,
        dimX: D(CrossAttDim.Loss), dimY: D(CrossAttDim.D_proj),
        name: 'Invariance MSE (l=10)',
    }));
    vicregInv.highlight = 0.4;

    let vicregVar = add(mk({
        t: 'a', xL: lossStartX + lossW * cell + lossSpacing, zM: 0, y,
        cx: lossW, cz: depthDeep, cy: 4,
        dimX: D(CrossAttDim.Loss), dimY: D(CrossAttDim.D_proj),
        name: 'Variance Hinge (l=10)',
    }));
    vicregVar.highlight = 0.4;

    let vicregCov = add(mk({
        t: 'a', xL: lossStartX + (lossW * cell + lossSpacing) * 2, zM: 0, y,
        cx: lossW, cz: depthDeep, cy: 4,
        dimX: D(CrossAttDim.Loss), dimY: D(CrossAttDim.D_proj),
        name: 'Covariance (l=1)',
    }));
    vicregCov.highlight = 0.4;

    y += 4 * cell + margin;
    let height = y;

    return {
        cell, height, margin, cubes, labels, shape,

        // Audio tower
        waveformInput, audioCnn, audioPosEmb,

        // Audio descriptor
        audioDescStftWindow, audioDescStftCompute, audioDescMagnitude,
        audioDescLogMag, audioDescBandGroup, audioDescTemporalDelta,
        audioDescNormalize, audioDescOutput,

        // Audio reverse cross-att (main)
        audioReverseDescQProj, audioReverseDescPosEmb,
        audioReverseQ, audioReverseK, audioReverseMatrix,
        audioReverseAttnOut, audioReverseResidual, audioReverseNorm,
        audioReverseOutput,

        // Audio transformer + post
        audioTransformerLayers, audioOutputLN, audioMeanPool,
        audioProjLayer1, audioProjLayer2, audioProjLayer3,

        // MIDI tower
        midiInput, midiPitchEmb, midiVelEmb, midiDurEmb,
        midiCombineLinear, midiPosEnc,

        // MIDI descriptor
        midiDescPitchInput, midiDescForwardDiff, midiDescValidityMask,
        midiDescSemitoneScale, midiDescLogRatioScale, midiDescOutput,

        // MIDI reverse cross-att (main)
        midiReverseIntQProj, midiReversePosEnc,
        midiReverseQ, midiReverseK, midiReverseMatrix,
        midiReverseResidual, midiReverseOutput,

        // MIDI transformer + post
        midiTransformerLayers, midiOutputLN, midiMeanPool,
        midiProjLayer1, midiProjLayer2, midiProjLayer3,

        // Shared
        audioEmbedding, midiEmbedding,
        vicregInv, vicregVar, vicregCov,
    };
}

/** Get all IBlkDef from a transformer layer (for use with dimExcept). */
export function getTransformerLayerBlocks(layer: ICrossAttTransformerLayer): IBlkDef[] {
    return [
        layer.ln1, layer.qWeight, layer.kWeight, layer.vWeight,
        layer.attnMatrix, layer.attnOut, layer.attnResidual,
        layer.ln2, layer.mlpUp, layer.mlpAct, layer.mlpDown, layer.ffnResidual,
    ];
}
