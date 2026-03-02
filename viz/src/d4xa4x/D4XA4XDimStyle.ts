import { Vec4 } from "@/src/utils/vector";

/**
 * Dimension enum for d4xa4x (Forward Cross-Attention).
 * Starts at 300 to avoid collision with PhideusDim(100), D4A4Dim(200), CrossAttDim(500).
 */
export enum D4XA4XDim {
    None = 0,

    // Temporal
    T_audio = 301,     // 2400 CNN frames
    T_midi = 302,      // N MIDI events
    T_stft = 303,      // 188 STFT frames (descriptor resolution)

    // Feature
    D_audio = 304,     // 1024D audio features
    D_midi = 305,      // 512D MIDI features
    D_proj = 306,      // 256D projection
    D_desc_a4 = 307,   // 8D A4 descriptor
    D_interval = 308,  // 4D D4 interval descriptor
    D_audio_xatt = 309, // forward cross-att output dim
    D_midi_xatt = 310,  // forward cross-att output dim

    // Derived
    B = 311,           // batch
    CNN_channels = 312,
    Pitch = 313,
    Velocity = 314,
    Duration = 315,
    D_ffn = 316,
    Loss = 317,

    // DSP intermediates
    D_stft_bins = 320,
    D_bands = 321,
    D_semitone = 322,
    D_log_ratio = 323,

    // Training state markers
    NoGrad = 330,
    TrainableXAtt = 331,
}

export function d4xa4xDimColor(dim: D4XA4XDim): Vec4 {
    switch (dim) {
        case D4XA4XDim.T_audio:     return new Vec4(0.21, 0.61, 0.66, 1);  // teal
        case D4XA4XDim.T_midi:      return new Vec4(0.18, 0.54, 0.58, 1);  // teal
        case D4XA4XDim.T_stft:      return new Vec4(0.36, 0.69, 0.72, 1);  // lighter teal
        case D4XA4XDim.D_audio:     return new Vec4(0.22, 0.47, 0.75, 1);  // blue
        case D4XA4XDim.D_midi:      return new Vec4(0.30, 0.55, 0.25, 1);  // green
        case D4XA4XDim.D_proj:      return new Vec4(0.50, 0.30, 0.70, 1);  // purple
        case D4XA4XDim.D_desc_a4:   return new Vec4(0.88, 0.58, 0.19, 1);  // amber #e09530
        case D4XA4XDim.D_interval:  return new Vec4(0.42, 0.62, 0.23, 1);  // green #6b9e3a
        case D4XA4XDim.D_audio_xatt: return new Vec4(0.87, 0.20, 0.80, 1); // magenta (forward xatt)
        case D4XA4XDim.D_midi_xatt:  return new Vec4(0.87, 0.20, 0.80, 1); // magenta
        case D4XA4XDim.B:           return new Vec4(0.50, 0.50, 0.50, 1);  // gray
        case D4XA4XDim.CNN_channels: return new Vec4(0.35, 0.35, 0.55, 1); // muted blue
        case D4XA4XDim.Pitch:       return new Vec4(0.60, 0.35, 0.15, 1);  // warm brown
        case D4XA4XDim.Velocity:    return new Vec4(0.50, 0.30, 0.55, 1);  // purple
        case D4XA4XDim.Duration:    return new Vec4(0.30, 0.50, 0.30, 1);  // green
        case D4XA4XDim.D_ffn:       return new Vec4(0.50, 0.45, 0.25, 1);  // olive
        case D4XA4XDim.Loss:        return new Vec4(0.75, 0.15, 0.15, 1);  // red
        case D4XA4XDim.D_stft_bins: return new Vec4(0.55, 0.45, 0.25, 1);
        case D4XA4XDim.D_bands:     return new Vec4(0.65, 0.50, 0.20, 1);
        case D4XA4XDim.D_semitone:  return new Vec4(0.40, 0.55, 0.25, 1);
        case D4XA4XDim.D_log_ratio: return new Vec4(0.35, 0.50, 0.30, 1);
        case D4XA4XDim.NoGrad:      return new Vec4(0.88, 0.58, 0.19, 1);  // amber
        case D4XA4XDim.TrainableXAtt: return new Vec4(0.87, 0.20, 0.80, 1); // magenta
        default:                    return new Vec4(0.5, 0.5, 0.5, 1);
    }
}

export function d4xa4xDimText(dim: D4XA4XDim): string {
    switch (dim) {
        case D4XA4XDim.T_audio:     return '2400 frames';
        case D4XA4XDim.T_midi:      return 'N events';
        case D4XA4XDim.T_stft:      return '188 frames';
        case D4XA4XDim.D_audio:     return '1024D';
        case D4XA4XDim.D_midi:      return '512D';
        case D4XA4XDim.D_proj:      return '256D';
        case D4XA4XDim.D_desc_a4:   return '8D (A4)';
        case D4XA4XDim.D_interval:  return '4D (D4)';
        case D4XA4XDim.D_audio_xatt: return 'FwdXAtt';
        case D4XA4XDim.D_midi_xatt:  return 'FwdXAtt';
        case D4XA4XDim.B:           return 'B';
        case D4XA4XDim.CNN_channels: return 'channels';
        case D4XA4XDim.Pitch:       return 'pitch (128)';
        case D4XA4XDim.Velocity:    return 'velocity';
        case D4XA4XDim.Duration:    return 'duration';
        case D4XA4XDim.D_ffn:       return 'FFN';
        case D4XA4XDim.Loss:        return 'loss';
        case D4XA4XDim.D_stft_bins: return '1025 bins';
        case D4XA4XDim.D_bands:     return '8 bands';
        case D4XA4XDim.D_semitone:  return 'semitone';
        case D4XA4XDim.D_log_ratio: return 'log ratio';
        case D4XA4XDim.NoGrad:      return 'NO_GRAD';
        case D4XA4XDim.TrainableXAtt: return 'FwdXAtt';
        default:                    return '';
    }
}
