import { Vec4 } from "@/src/utils/vector";

export enum HrmDim {
    None = 0,

    B = 400,
    T = 401,
    D_input = 402,
    D_enc = 403,
    D_gru = 404,
    D_attn = 405,

    D_lAgg = 406,
    D_hAttn = 407,
    D_lstm = 408,
    D_context = 409,
    D_act = 410,

    LModule = 414,
    HModule = 415,
    ACT = 416,
    Loop = 417,
}

export function hrmDimColor(dim: HrmDim): Vec4 {
    switch (dim) {
        case HrmDim.B: return Vec4.fromHexColor('#666666');
        case HrmDim.T: return Vec4.fromHexColor('#359da8');
        case HrmDim.D_input: return Vec4.fromHexColor('#339966');
        case HrmDim.D_enc: return Vec4.fromHexColor('#2d8a54');
        case HrmDim.D_gru: return Vec4.fromHexColor('#44aa66');
        case HrmDim.D_attn: return Vec4.fromHexColor('#55bb77');

        case HrmDim.D_lAgg: return Vec4.fromHexColor('#6b4ca0');
        case HrmDim.D_hAttn: return Vec4.fromHexColor('#8855aa');
        case HrmDim.D_lstm: return Vec4.fromHexColor('#7744aa');
        case HrmDim.D_context: return Vec4.fromHexColor('#cc7733');
        case HrmDim.D_act: return Vec4.fromHexColor('#dd3333');

        case HrmDim.LModule: return Vec4.fromHexColor('#339966');
        case HrmDim.HModule: return Vec4.fromHexColor('#6644aa');
        case HrmDim.ACT: return Vec4.fromHexColor('#cc4444');
        case HrmDim.Loop: return Vec4.fromHexColor('#cc7733');
        default: return new Vec4(0.5, 0.5, 0.5, 1);
    }
}

export function hrmDimText(dim: HrmDim): string {
    switch (dim) {
        case HrmDim.B: return 'B';
        case HrmDim.T: return 'T (512)';
        case HrmDim.D_input: return 'D_input (1536)';
        case HrmDim.D_enc: return 'D_enc (512)';
        case HrmDim.D_gru: return 'D_gru (256)';
        case HrmDim.D_attn: return 'D_attn (256)';
        case HrmDim.D_lAgg: return 'D_lAgg';
        case HrmDim.D_hAttn: return 'D_hAttn';
        case HrmDim.D_lstm: return 'D_lstm (256)';
        case HrmDim.D_context: return 'D_context (128)';
        case HrmDim.D_act: return 'D_act (2)';
        case HrmDim.LModule: return 'L-Module';
        case HrmDim.HModule: return 'H-Module';
        case HrmDim.ACT: return 'ACT';
        case HrmDim.Loop: return 'Feedback Loop';
        default: return '';
    }
}
