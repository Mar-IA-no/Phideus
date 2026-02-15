import { Vec4 } from "@/src/utils/vector";

export enum ConstellationDim {
    None = 0,

    B = 700,
    T = 701,
    K_tokens = 702,
    D_token = 703,
    D_hidden = 704,
    D_lstm = 705,

    Z_shared = 706,
    Z_private = 707,
    Z_total = 708,
    D_decoder = 709,

    Audio = 712,
    Vibration = 713,
    Loss = 718,
    Latent = 719,
}

export function constellationDimColor(dim: ConstellationDim): Vec4 {
    switch (dim) {
        case ConstellationDim.B: return Vec4.fromHexColor('#666666');
        case ConstellationDim.T: return Vec4.fromHexColor('#359da8');
        case ConstellationDim.K_tokens: return Vec4.fromHexColor('#4a90d9');
        case ConstellationDim.D_token: return Vec4.fromHexColor('#ce2983');
        case ConstellationDim.D_hidden: return Vec4.fromHexColor('#aa7744');
        case ConstellationDim.D_lstm: return Vec4.fromHexColor('#7c3c8d');
        case ConstellationDim.Z_shared: return Vec4.fromHexColor('#cc9933');
        case ConstellationDim.Z_private: return Vec4.fromHexColor('#996633');
        case ConstellationDim.Z_total: return Vec4.fromHexColor('#bb8822');
        case ConstellationDim.D_decoder: return Vec4.fromHexColor('#c75530');
        case ConstellationDim.Audio: return Vec4.fromHexColor('#3366cc');
        case ConstellationDim.Vibration: return Vec4.fromHexColor('#cc6633');
        case ConstellationDim.Loss: return Vec4.fromHexColor('#dd3333');
        case ConstellationDim.Latent: return Vec4.fromHexColor('#9933cc');
        default: return new Vec4(0.5, 0.5, 0.5, 1);
    }
}

export function constellationDimText(dim: ConstellationDim): string {
    switch (dim) {
        case ConstellationDim.B: return 'B';
        case ConstellationDim.T: return 'T (frames)';
        case ConstellationDim.K_tokens: return 'K (48 tokens)';
        case ConstellationDim.D_token: return 'D_token (5)';
        case ConstellationDim.D_hidden: return 'D_hidden (128)';
        case ConstellationDim.D_lstm: return 'D_lstm (128)';
        case ConstellationDim.Z_shared: return 'Z_shared (32)';
        case ConstellationDim.Z_private: return 'Z_private (16)';
        case ConstellationDim.Z_total: return 'Z_total (48)';
        case ConstellationDim.D_decoder: return 'D_decoder';
        case ConstellationDim.Audio: return 'Audio';
        case ConstellationDim.Vibration: return 'Vibration';
        case ConstellationDim.Loss: return 'Loss';
        case ConstellationDim.Latent: return 'Latent Space';
        default: return '';
    }
}
