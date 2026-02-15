import { Vec4 } from "@/src/utils/vector";

export enum CrossAttDim {
    _None = 0,

    B = 500,
    T_audio = 501,
    T_midi = 502,
    T_stft = 503,
    D_audio = 504,
    D_midi = 505,
    D_proj = 506,
    D_desc_a4 = 507,
    D_desc_a7 = 508,
    D_interval = 509,

    CNN_channels = 512,
    CrossAtt = 513,
    Residual = 514,
    Loss = 518,
    NoGrad = 519,
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
        case CrossAttDim.NoGrad: return Vec4.fromHexColor('#888888');
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
        case CrossAttDim.B: return 'B';
        default: return '';
    }
}
