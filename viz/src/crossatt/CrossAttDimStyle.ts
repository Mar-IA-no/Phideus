import { Vec4 } from "@/src/utils/vector";

export enum CrossAttDim {
    _None = 0,

    B = 500,
    T_audio = 501,     // 2400 CNN frames
    T_midi = 502,      // N MIDI events
    T_stft = 503,      // 188 STFT frames
    D_audio = 504,     // 1024D
    D_midi = 505,      // 512D
    D_proj = 506,      // 256D (shared)
    D_desc_a4 = 507,   // A4 descriptor (8D)
    D_desc_a7 = 508,   // A7 descriptor (12D)
    D_interval = 509,  // MIDI intervals (4D)

    CNN_channels = 512,
    CrossAtt = 513,
    Residual = 514,
    Loss = 518,
    NoGrad = 519,

    // From Phideus (shared concepts)
    D_ffn_audio = 520, // 4096 = 4x1024
    D_ffn_midi = 521,  // 2048 = 4x512
    N_heads = 522,
    N_layers = 523,
    Pitch = 524,       // 128 MIDI pitches
    Velocity = 525,    // 128 velocities
    Duration = 526,    // 32 durations
    Frozen = 527,
    Trainable = 528,
    Projection = 529,

    // New for CrossAtt (descriptor-specific)
    TrainableXAtt = 530,  // cross-att learnable components (cyan)
    Ghost = 531,          // dimmed comparison blocks
    D_stft_bins = 534,    // 1025 STFT frequency bins
    D_bands = 535,        // 8 log-freq bands
    D_semitone = 538,     // semitone scale
    D_log_ratio = 539,    // log ratio scale
}

export function crossAttDimColor(dim: CrossAttDim): Vec4 {
    switch (dim) {
        case CrossAttDim.T_audio: return Vec4.fromHexColor('#359da8');
        case CrossAttDim.T_midi: return Vec4.fromHexColor('#2d8a94');
        case CrossAttDim.T_stft: return Vec4.fromHexColor('#5bb0b8');
        case CrossAttDim.D_audio: return Vec4.fromHexColor('#ce2983');
        case CrossAttDim.D_midi: return Vec4.fromHexColor('#d94e9c');
        case CrossAttDim.D_proj: return Vec4.fromHexColor('#e06840');
        case CrossAttDim.D_desc_a4: return Vec4.fromHexColor('#8855cc');
        case CrossAttDim.D_desc_a7: return Vec4.fromHexColor('#7744bb');
        case CrossAttDim.D_interval: return Vec4.fromHexColor('#6b9e3a');
        case CrossAttDim.CNN_channels: return Vec4.fromHexColor('#7c3c8d');
        case CrossAttDim.CrossAtt: return Vec4.fromHexColor('#cc3366');
        case CrossAttDim.Residual: return Vec4.fromHexColor('#aa5577');
        case CrossAttDim.Loss: return Vec4.fromHexColor('#dd3333');
        case CrossAttDim.NoGrad: return Vec4.fromHexColor('#e09530');       // brighter amber for DSP
        case CrossAttDim.D_ffn_audio: return Vec4.fromHexColor('#c75530');
        case CrossAttDim.D_ffn_midi: return Vec4.fromHexColor('#c75530');
        case CrossAttDim.N_heads: return Vec4.fromHexColor('#d368a4');
        case CrossAttDim.N_layers: return Vec4.fromHexColor('#8888cc');
        case CrossAttDim.Pitch: return Vec4.fromHexColor('#6b4ca0');
        case CrossAttDim.Velocity: return Vec4.fromHexColor('#4a90d9');
        case CrossAttDim.Duration: return Vec4.fromHexColor('#50b050');
        case CrossAttDim.Frozen: return Vec4.fromHexColor('#888888');
        case CrossAttDim.Trainable: return Vec4.fromHexColor('#44aa44');
        case CrossAttDim.Projection: return Vec4.fromHexColor('#e06840');
        case CrossAttDim.TrainableXAtt: return Vec4.fromHexColor('#00bbcc');  // bright cyan
        case CrossAttDim.Ghost: return Vec4.fromHexColor('#666666');          // faded gray
        case CrossAttDim.D_stft_bins: return Vec4.fromHexColor('#aa7744');    // warm brown
        case CrossAttDim.D_bands: return Vec4.fromHexColor('#cc8833');        // amber
        case CrossAttDim.D_semitone: return Vec4.fromHexColor('#7799aa');
        case CrossAttDim.D_log_ratio: return Vec4.fromHexColor('#88aa77');
        case CrossAttDim.B: return Vec4.fromHexColor('#666666');
        default: return new Vec4(0.5, 0.5, 0.5, 1);
    }
}

export function crossAttDimText(dim: CrossAttDim): string {
    switch (dim) {
        case CrossAttDim.T_audio: return 'T_audio (2400)';
        case CrossAttDim.T_midi: return 'T_midi (N)';
        case CrossAttDim.T_stft: return 'T_stft (188)';
        case CrossAttDim.D_audio: return 'D_audio (1024)';
        case CrossAttDim.D_midi: return 'D_midi (512)';
        case CrossAttDim.D_proj: return 'D_proj (256)';
        case CrossAttDim.D_desc_a4: return 'A4 desc (8)';
        case CrossAttDim.D_desc_a7: return 'A7 desc (12)';
        case CrossAttDim.D_interval: return 'Interval (4)';
        case CrossAttDim.CNN_channels: return 'Channels';
        case CrossAttDim.CrossAtt: return 'Cross-Attention';
        case CrossAttDim.Residual: return 'Residual + LN';
        case CrossAttDim.Loss: return 'Loss';
        case CrossAttDim.NoGrad: return 'NO_GRAD';
        case CrossAttDim.D_ffn_audio: return 'D_ffn (4096)';
        case CrossAttDim.D_ffn_midi: return 'D_ffn (2048)';
        case CrossAttDim.N_heads: return 'N_heads (8)';
        case CrossAttDim.N_layers: return 'N_layers (4)';
        case CrossAttDim.Pitch: return 'Pitch (128)';
        case CrossAttDim.Velocity: return 'Velocity (128)';
        case CrossAttDim.Duration: return 'Duration (32)';
        case CrossAttDim.Frozen: return 'Frozen';
        case CrossAttDim.Trainable: return 'Trainable';
        case CrossAttDim.Projection: return 'Projection';
        case CrossAttDim.TrainableXAtt: return 'Cross-Att (learnable)';
        case CrossAttDim.Ghost: return 'Comparison (dimmed)';
        case CrossAttDim.D_stft_bins: return 'STFT bins (1025)';
        case CrossAttDim.D_bands: return 'Bands (8)';
        case CrossAttDim.D_semitone: return 'Semitone';
        case CrossAttDim.D_log_ratio: return 'Log Ratio';
        case CrossAttDim.B: return 'B';
        default: return '';
    }
}
