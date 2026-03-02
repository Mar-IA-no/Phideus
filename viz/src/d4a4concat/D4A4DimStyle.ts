import { Vec4 } from "@/src/utils/vector";

// D4A4 concat dimension styles, starting at 200 to avoid collision
export enum D4A4Dim {
    None = 0,

    // Time/sequence dimensions
    T_audio = 200,
    T_midi = 201,
    T_stft = 202,       // 188 STFT frames (for A4 descriptor)

    // Feature dimensions
    D_audio = 203,       // 1024 (MERT output)
    D_midi = 204,        // 512 (MIDI encoder output)
    D_proj = 205,        // 256 (shared space)

    // Descriptor dimensions
    D_desc_a4 = 206,     // 8 (A4 octave band features)
    D_interval = 207,    // 4 (D4 local intervals)

    // Concat dimensions (descriptors appended)
    D_audio_concat = 208,  // 1024 + 8 = 1032
    D_midi_concat = 209,   // 512 + 4 = 516

    // MIDI attributes
    Pitch = 210,
    Velocity = 211,
    Duration = 212,

    // Structural
    N_heads = 213,
    CNN_channels = 214,

    // Block types
    Frozen = 215,
    NoGrad = 216,
    Loss = 217,

    // FFN intermediate
    D_ffn = 218,

    // Batch
    B = 219,

    // DSP intermediates
    D_stft_bins = 220,   // 1025 STFT bins
    D_bands = 221,       // 8 log-freq bands
    D_semitone = 222,
    D_log_ratio = 223,
}

export function d4a4DimColor(dim: D4A4Dim): Vec4 {
    switch (dim) {
        case D4A4Dim.T_audio:
            return Vec4.fromHexColor('#359da8');
        case D4A4Dim.T_midi:
            return Vec4.fromHexColor('#2d8a94');
        case D4A4Dim.T_stft:
            return Vec4.fromHexColor('#44aaaa');
        case D4A4Dim.D_audio:
            return Vec4.fromHexColor('#ce2983');
        case D4A4Dim.D_midi:
            return Vec4.fromHexColor('#d94e9c');
        case D4A4Dim.D_proj:
            return Vec4.fromHexColor('#e06840');
        case D4A4Dim.D_desc_a4:
            return Vec4.fromHexColor('#e09530');   // amber for A4
        case D4A4Dim.D_interval:
            return Vec4.fromHexColor('#7ab445');    // green for D4
        case D4A4Dim.D_audio_concat:
            return Vec4.fromHexColor('#bb4488');    // mixed
        case D4A4Dim.D_midi_concat:
            return Vec4.fromHexColor('#99aa33');    // mixed
        case D4A4Dim.Pitch:
            return Vec4.fromHexColor('#6b4ca0');
        case D4A4Dim.Velocity:
            return Vec4.fromHexColor('#4a90d9');
        case D4A4Dim.Duration:
            return Vec4.fromHexColor('#50b050');
        case D4A4Dim.N_heads:
            return Vec4.fromHexColor('#d368a4');
        case D4A4Dim.CNN_channels:
            return Vec4.fromHexColor('#7c3c8d');
        case D4A4Dim.D_ffn:
            return Vec4.fromHexColor('#c75530');
        case D4A4Dim.Frozen:
            return Vec4.fromHexColor('#888888');
        case D4A4Dim.NoGrad:
            return Vec4.fromHexColor('#e09530');
        case D4A4Dim.Loss:
            return Vec4.fromHexColor('#dd3333');
        case D4A4Dim.B:
            return Vec4.fromHexColor('#666666');
        case D4A4Dim.D_stft_bins:
            return Vec4.fromHexColor('#aa7733');
        case D4A4Dim.D_bands:
            return Vec4.fromHexColor('#cc8833');
        case D4A4Dim.D_semitone:
            return Vec4.fromHexColor('#55aa55');
        case D4A4Dim.D_log_ratio:
            return Vec4.fromHexColor('#66bb66');
        default:
            return new Vec4(0.5, 0.5, 0.5, 1);
    }
}

export function d4a4DimText(dim: D4A4Dim): string {
    switch (dim) {
        case D4A4Dim.T_audio: return 'T_audio (2400)';
        case D4A4Dim.T_midi: return 'T_midi (N)';
        case D4A4Dim.T_stft: return 'T_stft (188)';
        case D4A4Dim.D_audio: return 'D_audio (1024)';
        case D4A4Dim.D_midi: return 'D_midi (512)';
        case D4A4Dim.D_proj: return 'D_proj (256)';
        case D4A4Dim.D_desc_a4: return 'A4 (8)';
        case D4A4Dim.D_interval: return 'D4 (4)';
        case D4A4Dim.D_audio_concat: return '1024+8=1032';
        case D4A4Dim.D_midi_concat: return '512+4=516';
        case D4A4Dim.Pitch: return 'Pitch';
        case D4A4Dim.Velocity: return 'Velocity';
        case D4A4Dim.Duration: return 'Duration';
        case D4A4Dim.N_heads: return 'N_heads';
        case D4A4Dim.CNN_channels: return 'Channels';
        case D4A4Dim.D_ffn: return 'D_ffn (4×D)';
        case D4A4Dim.Frozen: return 'Frozen';
        case D4A4Dim.NoGrad: return 'NO_GRAD';
        case D4A4Dim.Loss: return 'Loss';
        case D4A4Dim.B: return 'B';
        case D4A4Dim.D_stft_bins: return 'STFT bins (1025)';
        case D4A4Dim.D_bands: return '8 bands';
        case D4A4Dim.D_semitone: return 'semitone';
        case D4A4Dim.D_log_ratio: return 'log ratio';
        default: return '';
    }
}
