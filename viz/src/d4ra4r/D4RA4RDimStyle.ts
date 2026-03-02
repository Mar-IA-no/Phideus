import { Vec4 } from "@/src/utils/vector";

/**
 * Dimension enum for d4r-a4r (Mixed Descriptors + Reverse Cross-Attention).
 * Starts at 400 to avoid collision with PhideusDim(100), D4A4Dim(200), D4XA4XDim(300), CrossAttDim(500).
 */
export enum D4RA4RDim {
    _None = 0,

    B = 400,
    T_audio = 401,     // 2400 CNN frames
    T_midi = 402,      // N MIDI events
    T_stft = 403,      // 188 STFT frames
    D_audio = 404,     // 1024D
    D_midi = 405,      // 512D
    D_proj = 406,      // 256D (shared)
    D_desc_a4 = 407,   // A4 descriptor (8D)
    D_interval = 409,  // MIDI intervals (4D)

    CNN_channels = 412,
    CrossAtt = 413,
    Residual = 414,
    Loss = 418,
    NoGrad = 419,

    D_ffn_audio = 420, // 4096 = 4x1024
    D_ffn_midi = 421,  // 2048 = 4x512
    N_heads = 422,
    Pitch = 424,
    Velocity = 425,
    Duration = 426,

    TrainableXAtt = 430,
    D_stft_bins = 434,
    D_bands = 435,
    D_semitone = 438,
    D_log_ratio = 439,
}

export function d4ra4rDimColor(dim: D4RA4RDim): Vec4 {
    switch (dim) {
        case D4RA4RDim.T_audio: return Vec4.fromHexColor('#359da8');
        case D4RA4RDim.T_midi: return Vec4.fromHexColor('#2d8a94');
        case D4RA4RDim.T_stft: return Vec4.fromHexColor('#5bb0b8');
        case D4RA4RDim.D_audio: return Vec4.fromHexColor('#ce2983');
        case D4RA4RDim.D_midi: return Vec4.fromHexColor('#d94e9c');
        case D4RA4RDim.D_proj: return Vec4.fromHexColor('#e06840');
        case D4RA4RDim.D_desc_a4: return Vec4.fromHexColor('#e09530');   // amber for A4 descriptor
        case D4RA4RDim.D_interval: return Vec4.fromHexColor('#6b9e3a');  // green for D4 descriptor
        case D4RA4RDim.CNN_channels: return Vec4.fromHexColor('#7c3c8d');
        case D4RA4RDim.CrossAtt: return Vec4.fromHexColor('#cc3366');
        case D4RA4RDim.Residual: return Vec4.fromHexColor('#aa5577');
        case D4RA4RDim.Loss: return Vec4.fromHexColor('#dd3333');
        case D4RA4RDim.NoGrad: return Vec4.fromHexColor('#e09530');
        case D4RA4RDim.D_ffn_audio: return Vec4.fromHexColor('#c75530');
        case D4RA4RDim.D_ffn_midi: return Vec4.fromHexColor('#c75530');
        case D4RA4RDim.N_heads: return Vec4.fromHexColor('#d368a4');
        case D4RA4RDim.Pitch: return Vec4.fromHexColor('#6b4ca0');
        case D4RA4RDim.Velocity: return Vec4.fromHexColor('#4a90d9');
        case D4RA4RDim.Duration: return Vec4.fromHexColor('#50b050');
        case D4RA4RDim.TrainableXAtt: return Vec4.fromHexColor('#00bbcc');
        case D4RA4RDim.D_stft_bins: return Vec4.fromHexColor('#aa7744');
        case D4RA4RDim.D_bands: return Vec4.fromHexColor('#cc8833');
        case D4RA4RDim.D_semitone: return Vec4.fromHexColor('#7799aa');
        case D4RA4RDim.D_log_ratio: return Vec4.fromHexColor('#88aa77');
        case D4RA4RDim.B: return Vec4.fromHexColor('#666666');
        default: return new Vec4(0.5, 0.5, 0.5, 1);
    }
}

export function d4ra4rDimText(dim: D4RA4RDim): string {
    switch (dim) {
        case D4RA4RDim.T_audio: return 'T_audio (2400)';
        case D4RA4RDim.T_midi: return 'T_midi (N)';
        case D4RA4RDim.T_stft: return 'T_stft (188)';
        case D4RA4RDim.D_audio: return 'D_audio (1024)';
        case D4RA4RDim.D_midi: return 'D_midi (512)';
        case D4RA4RDim.D_proj: return 'D_proj (256)';
        case D4RA4RDim.D_desc_a4: return 'A4 desc (8)';
        case D4RA4RDim.D_interval: return 'Interval (4)';
        case D4RA4RDim.CNN_channels: return 'Channels';
        case D4RA4RDim.CrossAtt: return 'Cross-Attention';
        case D4RA4RDim.Residual: return 'Residual + LN';
        case D4RA4RDim.Loss: return 'Loss';
        case D4RA4RDim.NoGrad: return 'NO_GRAD';
        case D4RA4RDim.D_ffn_audio: return 'D_ffn (4096)';
        case D4RA4RDim.D_ffn_midi: return 'D_ffn (2048)';
        case D4RA4RDim.N_heads: return 'N_heads (8)';
        case D4RA4RDim.Pitch: return 'Pitch (128)';
        case D4RA4RDim.Velocity: return 'Velocity (128)';
        case D4RA4RDim.Duration: return 'Duration (32)';
        case D4RA4RDim.TrainableXAtt: return 'Cross-Att (learnable)';
        case D4RA4RDim.D_stft_bins: return 'STFT bins (1025)';
        case D4RA4RDim.D_bands: return 'Bands (8)';
        case D4RA4RDim.D_semitone: return 'Semitone';
        case D4RA4RDim.D_log_ratio: return 'Log Ratio';
        case D4RA4RDim.B: return 'B';
        default: return '';
    }
}
