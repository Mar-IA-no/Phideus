import { Vec4 } from "@/src/utils/vector";

// BloqueA-specific dimension styles, starting at 200 to avoid collision
export enum BloqueaDim {
    None = 0,

    // Time/sequence dimensions
    T_audio = 200,
    T_midi = 201,

    // Feature dimensions
    D_audio = 202,   // 1024 (MERT output)
    D_midi = 203,    // 512 (MIDI encoder output)
    D_proj = 204,    // 256 (shared space)

    // MIDI attributes
    Pitch = 205,
    Velocity = 206,
    Duration = 207,

    // Structural
    N_heads = 208,
    N_layers = 209,
    CNN_channels = 210,

    // Block types
    Frozen = 211,
    Trainable = 212,
    Projection = 213,
    Loss = 214,

    // Tower labels
    AudioTower = 215,
    MIDITower = 216,
    SharedSpace = 217,

    // FFN intermediate
    D_ffn = 219,

    // Batch
    B = 220,

    // Adapter-specific
    Adapter = 221,      // adapter bottleneck dimension
    D_adapter = 222,    // adapter inner dim (64)
    Unfrozen = 223,     // unfrozen layers
}

export function bloqueaDimColor(dim: BloqueaDim): Vec4 {
    switch (dim) {
        case BloqueaDim.T_audio:
            return Vec4.fromHexColor('#359da8');
        case BloqueaDim.T_midi:
            return Vec4.fromHexColor('#2d8a94');
        case BloqueaDim.D_audio:
            return Vec4.fromHexColor('#ce2983');
        case BloqueaDim.D_midi:
            return Vec4.fromHexColor('#d94e9c');
        case BloqueaDim.D_proj:
            return Vec4.fromHexColor('#e06840');
        case BloqueaDim.Pitch:
            return Vec4.fromHexColor('#6b4ca0');
        case BloqueaDim.Velocity:
            return Vec4.fromHexColor('#4a90d9');
        case BloqueaDim.Duration:
            return Vec4.fromHexColor('#50b050');
        case BloqueaDim.N_heads:
            return Vec4.fromHexColor('#d368a4');
        case BloqueaDim.N_layers:
            return Vec4.fromHexColor('#8888cc');
        case BloqueaDim.CNN_channels:
            return Vec4.fromHexColor('#7c3c8d');
        case BloqueaDim.D_ffn:
            return Vec4.fromHexColor('#c75530');
        case BloqueaDim.Frozen:
            return Vec4.fromHexColor('#888888');
        case BloqueaDim.Trainable:
            return Vec4.fromHexColor('#44aa44');
        case BloqueaDim.Projection:
            return Vec4.fromHexColor('#e06840');
        case BloqueaDim.Loss:
            return Vec4.fromHexColor('#dd3333');
        case BloqueaDim.AudioTower:
            return Vec4.fromHexColor('#3366cc');
        case BloqueaDim.MIDITower:
            return Vec4.fromHexColor('#cc6633');
        case BloqueaDim.SharedSpace:
            return Vec4.fromHexColor('#9933cc');
        case BloqueaDim.B:
            return Vec4.fromHexColor('#666666');
        case BloqueaDim.Adapter:
            return Vec4.fromHexColor('#ff6600');
        case BloqueaDim.D_adapter:
            return Vec4.fromHexColor('#ff9933');
        case BloqueaDim.Unfrozen:
            return Vec4.fromHexColor('#22bb55');
        default:
            return new Vec4(0.5, 0.5, 0.5, 1);
    }
}

export function bloqueaDimText(dim: BloqueaDim): string {
    switch (dim) {
        case BloqueaDim.T_audio: return 'T_audio';
        case BloqueaDim.T_midi: return 'T_midi';
        case BloqueaDim.D_audio: return 'D_audio (1024)';
        case BloqueaDim.D_midi: return 'D_midi (512)';
        case BloqueaDim.D_proj: return 'D_proj (256)';
        case BloqueaDim.Pitch: return 'Pitch';
        case BloqueaDim.Velocity: return 'Velocity';
        case BloqueaDim.Duration: return 'Duration';
        case BloqueaDim.N_heads: return 'N_heads';
        case BloqueaDim.N_layers: return 'N_layers';
        case BloqueaDim.CNN_channels: return 'Channels';
        case BloqueaDim.D_ffn: return 'D_ffn (4×D)';
        case BloqueaDim.Frozen: return 'Frozen';
        case BloqueaDim.Trainable: return 'Trainable';
        case BloqueaDim.Projection: return 'Projection';
        case BloqueaDim.Loss: return 'Loss';
        case BloqueaDim.AudioTower: return 'Audio Tower';
        case BloqueaDim.MIDITower: return 'MIDI Tower';
        case BloqueaDim.SharedSpace: return 'Shared Space';
        case BloqueaDim.B: return 'B';
        case BloqueaDim.Adapter: return 'Adapter';
        case BloqueaDim.D_adapter: return 'D_adapter (64)';
        case BloqueaDim.Unfrozen: return 'Unfrozen';
        default: return '';
    }
}
