import { Vec4 } from "@/src/utils/vector";

// T3Tower-specific dimension styles, starting at 600 to avoid collision
export enum T3TowerDim {
    None = 0,

    // Time/sequence dimensions
    T_audio = 600,
    T_midi = 601,
    T_t3 = 602,

    // Feature dimensions
    D_audio = 603,   // 1024 (MERT output)
    D_midi = 604,    // 512 (MIDI encoder output)
    D_t3 = 605,      // 256 (T3 model dimension)
    D_proj = 606,    // 256 (shared space)

    // MIDI attributes
    Pitch = 607,
    Velocity = 608,
    Duration = 609,

    // Structural
    N_heads = 610,
    N_layers = 611,
    CNN_channels = 612,

    // Block types
    Frozen = 613,
    Trainable = 614,
    Projection = 615,
    Loss = 616,

    // Tower labels
    AudioTower = 617,
    MIDITower = 618,
    T3Tower = 619,
    SharedSpace = 620,

    // FFN intermediate
    D_ffn = 621,
    D_ffn_t3 = 622,

    // Batch
    B = 623,
}

export function t3TowerDimColor(dim: T3TowerDim): Vec4 {
    switch (dim) {
        case T3TowerDim.T_audio:
            return Vec4.fromHexColor('#359da8');
        case T3TowerDim.T_midi:
            return Vec4.fromHexColor('#2d8a94');
        case T3TowerDim.T_t3:
            return Vec4.fromHexColor('#2299aa');
        case T3TowerDim.D_audio:
            return Vec4.fromHexColor('#ce2983');
        case T3TowerDim.D_midi:
            return Vec4.fromHexColor('#d94e9c');
        case T3TowerDim.D_t3:
            return Vec4.fromHexColor('#2299aa');
        case T3TowerDim.D_proj:
            return Vec4.fromHexColor('#e06840');
        case T3TowerDim.Pitch:
            return Vec4.fromHexColor('#6b4ca0');
        case T3TowerDim.Velocity:
            return Vec4.fromHexColor('#4a90d9');
        case T3TowerDim.Duration:
            return Vec4.fromHexColor('#50b050');
        case T3TowerDim.N_heads:
            return Vec4.fromHexColor('#d368a4');
        case T3TowerDim.N_layers:
            return Vec4.fromHexColor('#8888cc');
        case T3TowerDim.CNN_channels:
            return Vec4.fromHexColor('#7c3c8d');
        case T3TowerDim.D_ffn:
            return Vec4.fromHexColor('#c75530');
        case T3TowerDim.D_ffn_t3:
            return Vec4.fromHexColor('#1d8899');
        case T3TowerDim.Frozen:
            return Vec4.fromHexColor('#888888');
        case T3TowerDim.Trainable:
            return Vec4.fromHexColor('#44aa44');
        case T3TowerDim.Projection:
            return Vec4.fromHexColor('#e06840');
        case T3TowerDim.Loss:
            return Vec4.fromHexColor('#dd3333');
        case T3TowerDim.AudioTower:
            return Vec4.fromHexColor('#3366cc');
        case T3TowerDim.MIDITower:
            return Vec4.fromHexColor('#cc6633');
        case T3TowerDim.T3Tower:
            return Vec4.fromHexColor('#2299aa');
        case T3TowerDim.SharedSpace:
            return Vec4.fromHexColor('#9933cc');
        case T3TowerDim.B:
            return Vec4.fromHexColor('#666666');
        default:
            return new Vec4(0.5, 0.5, 0.5, 1);
    }
}

export function t3TowerDimText(dim: T3TowerDim): string {
    switch (dim) {
        case T3TowerDim.T_audio: return 'T_audio';
        case T3TowerDim.T_midi: return 'T_midi';
        case T3TowerDim.T_t3: return 'T_t3';
        case T3TowerDim.D_audio: return 'D_audio (1024)';
        case T3TowerDim.D_midi: return 'D_midi (512)';
        case T3TowerDim.D_t3: return 'D_t3 (256)';
        case T3TowerDim.D_proj: return 'D_proj (256)';
        case T3TowerDim.Pitch: return 'Pitch';
        case T3TowerDim.Velocity: return 'Velocity';
        case T3TowerDim.Duration: return 'Duration';
        case T3TowerDim.N_heads: return 'N_heads';
        case T3TowerDim.N_layers: return 'N_layers';
        case T3TowerDim.CNN_channels: return 'Channels';
        case T3TowerDim.D_ffn: return 'D_ffn (4×D)';
        case T3TowerDim.D_ffn_t3: return 'D_ffn_t3 (4×D)';
        case T3TowerDim.Frozen: return 'Frozen';
        case T3TowerDim.Trainable: return 'Trainable';
        case T3TowerDim.Projection: return 'Projection';
        case T3TowerDim.Loss: return 'Loss';
        case T3TowerDim.AudioTower: return 'Audio Tower';
        case T3TowerDim.MIDITower: return 'MIDI Tower';
        case T3TowerDim.T3Tower: return 'T3 Tower';
        case T3TowerDim.SharedSpace: return 'Shared Space';
        case T3TowerDim.B: return 'B';
        default: return '';
    }
}
