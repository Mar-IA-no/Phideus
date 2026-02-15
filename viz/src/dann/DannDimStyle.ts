import { Vec4 } from "@/src/utils/vector";

export enum DannDim {
    None = 0,

    T_audio = 600,
    T_midi = 601,
    D_audio = 602,
    D_midi = 603,
    D_proj = 604,

    Pitch = 605,
    Velocity = 606,
    Duration = 607,

    N_heads = 608,
    CNN_channels = 610,
    D_ffn = 611,

    Frozen = 612,
    Trainable = 613,
    Loss = 614,

    AudioTower = 615,
    MIDITower = 616,
    SharedSpace = 617,

    // DANN-specific
    GRL = 618,
    Classifier = 619,
    D_hidden = 620,

    B = 621,
}

export function dannDimColor(dim: DannDim): Vec4 {
    switch (dim) {
        case DannDim.T_audio: return Vec4.fromHexColor('#359da8');
        case DannDim.T_midi: return Vec4.fromHexColor('#2d8a94');
        case DannDim.D_audio: return Vec4.fromHexColor('#ce2983');
        case DannDim.D_midi: return Vec4.fromHexColor('#d94e9c');
        case DannDim.D_proj: return Vec4.fromHexColor('#e06840');
        case DannDim.Pitch: return Vec4.fromHexColor('#6b4ca0');
        case DannDim.Velocity: return Vec4.fromHexColor('#4a90d9');
        case DannDim.Duration: return Vec4.fromHexColor('#50b050');
        case DannDim.N_heads: return Vec4.fromHexColor('#d368a4');
        case DannDim.CNN_channels: return Vec4.fromHexColor('#7c3c8d');
        case DannDim.D_ffn: return Vec4.fromHexColor('#c75530');
        case DannDim.Frozen: return Vec4.fromHexColor('#888888');
        case DannDim.Trainable: return Vec4.fromHexColor('#44aa44');
        case DannDim.Loss: return Vec4.fromHexColor('#dd3333');
        case DannDim.AudioTower: return Vec4.fromHexColor('#3366cc');
        case DannDim.MIDITower: return Vec4.fromHexColor('#cc6633');
        case DannDim.SharedSpace: return Vec4.fromHexColor('#9933cc');
        case DannDim.GRL: return Vec4.fromHexColor('#996633');
        case DannDim.Classifier: return Vec4.fromHexColor('#885522');
        case DannDim.D_hidden: return Vec4.fromHexColor('#aa7744');
        case DannDim.B: return Vec4.fromHexColor('#666666');
        default: return new Vec4(0.5, 0.5, 0.5, 1);
    }
}

export function dannDimText(dim: DannDim): string {
    switch (dim) {
        case DannDim.T_audio: return 'T_audio';
        case DannDim.T_midi: return 'T_midi';
        case DannDim.D_audio: return 'D_audio (1024)';
        case DannDim.D_midi: return 'D_midi (512)';
        case DannDim.D_proj: return 'D_proj (256)';
        case DannDim.Pitch: return 'Pitch';
        case DannDim.Velocity: return 'Velocity';
        case DannDim.Duration: return 'Duration';
        case DannDim.N_heads: return 'N_heads';
        case DannDim.CNN_channels: return 'Channels';
        case DannDim.D_ffn: return 'D_ffn';
        case DannDim.Frozen: return 'Frozen';
        case DannDim.Trainable: return 'Trainable';
        case DannDim.Loss: return 'Loss';
        case DannDim.AudioTower: return 'Audio Tower';
        case DannDim.MIDITower: return 'MIDI Tower';
        case DannDim.SharedSpace: return 'Shared Space';
        case DannDim.GRL: return 'Gradient Reversal';
        case DannDim.Classifier: return 'Domain Classifier';
        case DannDim.D_hidden: return 'D_hidden (64)';
        case DannDim.B: return 'B';
        default: return '';
    }
}
