import { Vec4 } from "@/src/utils/vector";

// Phideus-specific dimension styles, starting at 100 to avoid collision with DimStyle enum
export enum PhideusDim {
    None = 0,

    // Time/sequence dimensions
    T_audio = 100,
    T_midi = 101,

    // Feature dimensions
    D_audio = 102,   // 1024 (MERT output)
    D_midi = 103,    // 512 (MIDI encoder output)
    D_proj = 104,    // 256 (shared space)

    // MIDI attributes
    Pitch = 105,
    Velocity = 106,
    Duration = 107,

    // Structural
    N_heads = 108,
    N_layers = 109,
    CNN_channels = 110,

    // Block types
    Frozen = 111,
    Trainable = 112,
    Projection = 113,
    Loss = 114,

    // Tower labels
    AudioTower = 115,
    MIDITower = 116,
    SharedSpace = 117,

    // FFN intermediate
    D_ffn = 119,

    // Batch
    B = 120,
}

export function phideusDimColor(dim: PhideusDim): Vec4 {
    switch (dim) {
        case PhideusDim.T_audio:
            return Vec4.fromHexColor('#359da8');
        case PhideusDim.T_midi:
            return Vec4.fromHexColor('#2d8a94');
        case PhideusDim.D_audio:
            return Vec4.fromHexColor('#ce2983');
        case PhideusDim.D_midi:
            return Vec4.fromHexColor('#d94e9c');
        case PhideusDim.D_proj:
            return Vec4.fromHexColor('#e06840');
        case PhideusDim.Pitch:
            return Vec4.fromHexColor('#6b4ca0');
        case PhideusDim.Velocity:
            return Vec4.fromHexColor('#4a90d9');
        case PhideusDim.Duration:
            return Vec4.fromHexColor('#50b050');
        case PhideusDim.N_heads:
            return Vec4.fromHexColor('#d368a4');
        case PhideusDim.N_layers:
            return Vec4.fromHexColor('#8888cc');
        case PhideusDim.CNN_channels:
            return Vec4.fromHexColor('#7c3c8d');
        case PhideusDim.D_ffn:
            return Vec4.fromHexColor('#c75530');
        case PhideusDim.Frozen:
            return Vec4.fromHexColor('#888888');
        case PhideusDim.Trainable:
            return Vec4.fromHexColor('#44aa44');
        case PhideusDim.Projection:
            return Vec4.fromHexColor('#e06840');
        case PhideusDim.Loss:
            return Vec4.fromHexColor('#dd3333');
        case PhideusDim.AudioTower:
            return Vec4.fromHexColor('#3366cc');
        case PhideusDim.MIDITower:
            return Vec4.fromHexColor('#cc6633');
        case PhideusDim.SharedSpace:
            return Vec4.fromHexColor('#9933cc');
        case PhideusDim.B:
            return Vec4.fromHexColor('#666666');
        default:
            return new Vec4(0.5, 0.5, 0.5, 1);
    }
}

export function phideusDimText(dim: PhideusDim): string {
    switch (dim) {
        case PhideusDim.T_audio: return 'T_audio';
        case PhideusDim.T_midi: return 'T_midi';
        case PhideusDim.D_audio: return 'D_audio (1024)';
        case PhideusDim.D_midi: return 'D_midi (512)';
        case PhideusDim.D_proj: return 'D_proj (256)';
        case PhideusDim.Pitch: return 'Pitch';
        case PhideusDim.Velocity: return 'Velocity';
        case PhideusDim.Duration: return 'Duration';
        case PhideusDim.N_heads: return 'N_heads';
        case PhideusDim.N_layers: return 'N_layers';
        case PhideusDim.CNN_channels: return 'Channels';
        case PhideusDim.D_ffn: return 'D_ffn (4×D)';
        case PhideusDim.Frozen: return 'Frozen';
        case PhideusDim.Trainable: return 'Trainable';
        case PhideusDim.Projection: return 'Projection';
        case PhideusDim.Loss: return 'Loss';
        case PhideusDim.AudioTower: return 'Audio Tower';
        case PhideusDim.MIDITower: return 'MIDI Tower';
        case PhideusDim.SharedSpace: return 'Shared Space';
        case PhideusDim.B: return 'B';
        default: return '';
    }
}
