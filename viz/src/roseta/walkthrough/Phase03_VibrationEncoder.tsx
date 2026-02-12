import { Vec3 } from "@/src/utils/vector";
import { IRosetaWalkthroughArgs, setRosetaInitialCamera, moveRosetaCameraTo } from "./RosetaWalkthrough";
import { getEncoderBlocks, getLatentBlocks } from "../RosetaModelLayout";
import { RosetaDim } from "../RosetaDimStyle";

export function rosetaPhase03_VibrationEncoder(args: IRosetaWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept, drawDimLabels } = tools;

    let vibBlocks = [...getEncoderBlocks(layout.vibration), ...getLatentBlocks(layout.vibration)];
    dimExcept(vibBlocks);

    setRosetaInitialCamera(state, new Vec3(80, 0, -20), new Vec3(288, 15, 4));

    let d128 = c_dimRef('128D', RosetaDim.D_hidden);

    commentary()`The Vibration Encoder mirrors the Audio Encoder exactly: input projection (768→128), BiLSTM (2 layers, bidirectional), then split into shared/private latent heads. The architecture is intentionally symmetric — differences between audio and vibration should emerge from the _data_ and be captured in the private latent, not from architectural asymmetry.`;
    breakAfter();

    // Step 1: Input + projection
    let t0 = afterTime(null, 2.0, 0.3);

    let vibInput = c_blockRef('vibration input', layout.vibration.input);
    let vibProj = c_blockRef('input projection', layout.vibration.inputProj);

    commentary()`${vibInput} features [T,256,3] — accelerometer spectral data — are compressed by the ${vibProj} to ${d128}. The same 6x compression as audio, but the vibration signal has fundamentally different characteristics: lower frequency content, different noise profiles, mechanical resonance patterns.`;
    breakAfter();

    if (t0.active) {
        layout.vibration.input.highlight = Math.max(layout.vibration.input.highlight, t0.t * 0.4);
        layout.vibration.input.opacity = 0.9;
        layout.vibration.inputProj.highlight = Math.max(layout.vibration.inputProj.highlight, t0.t * 0.5);
        drawDimLabels(layout.vibration.inputProj, t0.t * 0.7);
    }

    breakAfter();

    // Step 2: BiLSTM + latent split
    let t1 = afterTime(null, 2.5, 0.5);
    let t0c = afterTime(t0, 0.3);
    cleanup(t0c, [t0]);

    moveRosetaCameraTo(state, t1, new Vec3(80, 0, -layout.vibration.lstmEncoder.y - 40), new Vec3(288, 14, 5));

    let lstmRef = c_blockRef('BiLSTM', layout.vibration.lstmEncoder);
    let zShared = c_dimRef('z_shared', RosetaDim.Z_shared);
    let zPrivate = c_dimRef('z_private', RosetaDim.Z_private);

    commentary()`The ${lstmRef} processes the vibration sequence, then the output splits into ${zShared} (32D) and ${zPrivate} (16D). The shared latent will be aligned with audio's shared latent via InfoNCE — this is how RosetaVAE learns that a piano note and its vibration signature share underlying musical structure.`;
    breakAfter();

    if (t1.active) {
        layout.vibration.lstmEncoder.highlight = Math.max(layout.vibration.lstmEncoder.highlight, t1.t * 0.5);
        let pulse = 0.4 + 0.2 * Math.sin(wt.time * 3);
        layout.vibration.zShared.highlight = Math.max(layout.vibration.zShared.highlight, pulse);
        layout.vibration.zPrivate.highlight = Math.max(layout.vibration.zPrivate.highlight, pulse * 0.7);
        drawDimLabels(layout.vibration.lstmEncoder, t1.t * 0.7);
        drawDimLabels(layout.vibration.zShared, t1.t * 0.8);
    }

    breakAfter();
}
