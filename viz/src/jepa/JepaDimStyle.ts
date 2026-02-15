import { Vec4 } from "@/src/utils/vector";

export enum JepaDim {
    _None = 0,

    // Shared dimensions (offset 800)
    B = 800,
    T = 801,
    K_tokens = 802,
    D_token = 803,
    D_hidden = 804,
    D_lstm = 805,
    D_z = 806,
    D_pred = 807,

    // Domain markers
    Audio = 810,
    Vibration = 811,

    // Structural
    Loss = 818,
    StopGrad = 819,
    Predictor = 820,
}

export function jepaDimColor(dim: JepaDim): Vec4 {
    switch (dim) {
        case JepaDim.B: return new Vec4(0.5, 0.5, 0.5, 1);
        case JepaDim.T: return new Vec4(0.4, 0.2, 0.8, 1);
        case JepaDim.K_tokens: return new Vec4(0.5, 0.3, 0.7, 1);
        case JepaDim.D_token: return new Vec4(0.3, 0.2, 0.7, 1);
        case JepaDim.D_hidden: return new Vec4(0.4, 0.2, 0.8, 1);
        case JepaDim.D_lstm: return new Vec4(0.35, 0.15, 0.75, 1);
        case JepaDim.D_z: return new Vec4(0.3, 0.1, 0.9, 1);
        case JepaDim.D_pred: return new Vec4(0.5, 0.2, 0.7, 1);
        case JepaDim.Audio: return new Vec4(0.2, 0.4, 0.8, 1);
        case JepaDim.Vibration: return new Vec4(0.8, 0.4, 0.2, 1);
        case JepaDim.Loss: return new Vec4(0.8, 0.2, 0.2, 1);
        case JepaDim.StopGrad: return new Vec4(0.6, 0.1, 0.6, 1);
        case JepaDim.Predictor: return new Vec4(0.5, 0.2, 0.7, 1);
        default: return new Vec4(0.5, 0.5, 0.5, 1);
    }
}

export function jepaDimText(dim: JepaDim): string {
    switch (dim) {
        case JepaDim.B: return 'Batch';
        case JepaDim.T: return 'Time steps';
        case JepaDim.K_tokens: return 'Tokens (K=48)';
        case JepaDim.D_token: return 'Token dim (5)';
        case JepaDim.D_hidden: return 'Hidden (128)';
        case JepaDim.D_lstm: return 'LSTM (32)';
        case JepaDim.D_z: return 'Latent z (32)';
        case JepaDim.D_pred: return 'Predictor (64)';
        case JepaDim.Audio: return 'Audio domain';
        case JepaDim.Vibration: return 'Vibration domain';
        case JepaDim.Loss: return 'Loss';
        case JepaDim.StopGrad: return 'Stop gradient';
        case JepaDim.Predictor: return 'Predictor MLP';
        default: return '';
    }
}
