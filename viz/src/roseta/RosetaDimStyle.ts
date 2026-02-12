import { Vec4 } from "@/src/utils/vector";

// Roseta-specific dimension styles, starting at 300 to avoid collision
export enum RosetaDim {
    None = 0,

    // Temporal
    T_seq = 300,        // sequence length T

    // Input/output dimensions
    D_input = 301,      // 768 (input features per frame, 256*3)
    D_hidden = 302,     // 128 (LSTM hidden dim)

    // Latent space
    Z_shared = 303,     // 32D shared latent
    Z_private = 304,    // 16D private latent
    Z_total = 305,      // 48D total latent (shared + private)

    // LSTM
    D_lstm = 306,       // 256 (BiLSTM output = 2*128)

    // Reconstruction
    Recon = 307,        // reconstruction output

    // Loss terms
    Loss_recon = 308,   // reconstruction loss
    Loss_KL_shared = 309,  // KL divergence on z_shared
    Loss_KL_private = 310, // KL divergence on z_private
    Loss_InfoNCE = 311,    // InfoNCE alignment loss
    Loss_diff = 312,       // differentiation loss

    // Domain labels
    AudioDomain = 313,
    VibrationDomain = 314,
    LatentSpace = 315,
    LossSpace = 316,

    // Mu/sigma
    Mu = 317,
    Sigma = 318,

    // Batch
    B = 320,
}

export function rosetaDimColor(dim: RosetaDim): Vec4 {
    switch (dim) {
        case RosetaDim.T_seq:
            return Vec4.fromHexColor('#2d9494');
        case RosetaDim.D_input:
            return Vec4.fromHexColor('#ce2983');
        case RosetaDim.D_hidden:
            return Vec4.fromHexColor('#4477cc');
        case RosetaDim.Z_shared:
            return Vec4.fromHexColor('#d4930a');
        case RosetaDim.Z_private:
            return Vec4.fromHexColor('#8844bb');
        case RosetaDim.Z_total:
            return Vec4.fromHexColor('#cc7722');
        case RosetaDim.D_lstm:
            return Vec4.fromHexColor('#3388bb');
        case RosetaDim.Recon:
            return Vec4.fromHexColor('#44aa44');
        case RosetaDim.Loss_recon:
            return Vec4.fromHexColor('#44aa44');
        case RosetaDim.Loss_KL_shared:
            return Vec4.fromHexColor('#dd3333');
        case RosetaDim.Loss_KL_private:
            return Vec4.fromHexColor('#bb4444');
        case RosetaDim.Loss_InfoNCE:
            return Vec4.fromHexColor('#22aacc');
        case RosetaDim.Loss_diff:
            return Vec4.fromHexColor('#cc44cc');
        case RosetaDim.AudioDomain:
            return Vec4.fromHexColor('#3366cc');
        case RosetaDim.VibrationDomain:
            return Vec4.fromHexColor('#cc6633');
        case RosetaDim.LatentSpace:
            return Vec4.fromHexColor('#d4930a');
        case RosetaDim.LossSpace:
            return Vec4.fromHexColor('#dd3333');
        case RosetaDim.Mu:
            return Vec4.fromHexColor('#cc8800');
        case RosetaDim.Sigma:
            return Vec4.fromHexColor('#9966cc');
        case RosetaDim.B:
            return Vec4.fromHexColor('#666666');
        default:
            return new Vec4(0.5, 0.5, 0.5, 1);
    }
}

export function rosetaDimText(dim: RosetaDim): string {
    switch (dim) {
        case RosetaDim.T_seq: return 'T';
        case RosetaDim.D_input: return 'D_input (768)';
        case RosetaDim.D_hidden: return 'D_hidden (128)';
        case RosetaDim.Z_shared: return 'z_shared (32D)';
        case RosetaDim.Z_private: return 'z_private (16D)';
        case RosetaDim.Z_total: return 'z_total (48D)';
        case RosetaDim.D_lstm: return 'D_lstm (256)';
        case RosetaDim.Recon: return 'Recon';
        case RosetaDim.Loss_recon: return 'Recon Loss';
        case RosetaDim.Loss_KL_shared: return 'KL_shared';
        case RosetaDim.Loss_KL_private: return 'KL_private';
        case RosetaDim.Loss_InfoNCE: return 'InfoNCE';
        case RosetaDim.Loss_diff: return 'Diff Loss';
        case RosetaDim.AudioDomain: return 'Audio';
        case RosetaDim.VibrationDomain: return 'Vibration';
        case RosetaDim.LatentSpace: return 'Latent Space';
        case RosetaDim.LossSpace: return 'Loss Space';
        case RosetaDim.Mu: return 'μ (mean)';
        case RosetaDim.Sigma: return 'σ (logvar)';
        case RosetaDim.B: return 'B';
        default: return '';
    }
}
