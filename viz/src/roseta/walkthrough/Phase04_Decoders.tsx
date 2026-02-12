import { Vec3 } from "@/src/utils/vector";
import { IRosetaWalkthroughArgs, setRosetaInitialCamera, moveRosetaCameraTo } from "./RosetaWalkthrough";
import { getDecoderBlocks } from "../RosetaModelLayout";
import { RosetaDim } from "../RosetaDimStyle";

export function rosetaPhase04_Decoders(args: IRosetaWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept, drawDimLabels } = tools;

    let decoderBlocks = [
        layout.audio.zConcat, layout.audio.zProj, ...getDecoderBlocks(layout.audio),
        layout.vibration.zConcat, layout.vibration.zProj, ...getDecoderBlocks(layout.vibration),
    ];
    dimExcept(decoderBlocks);

    let decoderY = layout.audio.zProj.y;
    setRosetaInitialCamera(state, new Vec3(-80, 0, -decoderY - 20), new Vec3(290, 12, 4));

    let d48 = c_dimRef('48D', RosetaDim.Z_total);
    let d128 = c_dimRef('128D', RosetaDim.D_hidden);

    commentary()`The decoders reverse the encoding process. From the concatenated ${d48} latent, z_proj expands to ${d128}, then a BiLSTM decoder reconstructs the temporal sequence. Output projection maps back to 768D, and softmax produces the final reconstruction. The decoder must recreate the original signal from just 48 latent dimensions — a 16x information bottleneck.`;
    breakAfter();

    // Step 1: Audio decoder cascade
    let t0 = afterTime(null, 3.0, 0.5);

    let zProjRef = c_blockRef('z_proj', layout.audio.zProj);
    let lstmDec = c_blockRef('BiLSTM decoder', layout.audio.lstmDecoder);

    commentary()`Audio decoder: ${zProjRef} maps 48D → 128D, then the ${lstmDec} (2 layers, bidirectional) reconstructs the temporal structure. The output projection expands 256D → 768D, and softmax normalizes across the 256 frequency bins for each of the 3 channels.`;
    breakAfter();

    if (t0.active) {
        let blocks = [layout.audio.zProj, layout.audio.lstmDecoder, layout.audio.outputProj, layout.audio.softmax, layout.audio.recon];
        for (let i = 0; i < blocks.length; i++) {
            let layerPhase = (t0.t * (blocks.length + 1)) - i;
            if (layerPhase > 0) {
                let intensity = Math.min(1, layerPhase);
                let pulse = intensity * (0.4 + 0.2 * Math.sin(wt.time * 3 + i * 1.0));
                blocks[i].highlight = Math.max(blocks[i].highlight, pulse);
                if (i < 3) drawDimLabels(blocks[i], intensity * 0.7);
            }
        }
    }

    breakAfter();

    // Step 2: Vibration decoder
    let t1 = afterTime(null, 2.5, 0.5);
    let t0c = afterTime(t0, 0.4);
    cleanup(t0c, [t0]);

    let vibDecoderY = layout.vibration.zProj.y;
    moveRosetaCameraTo(state, t1, new Vec3(80, 0, -vibDecoderY - 20), new Vec3(288, 12, 4));

    commentary()`The vibration decoder follows the identical architecture. Symmetric design ensures that any performance difference between domains reflects the inherent difficulty of the reconstruction task, not architectural bias. Vibration signals may be easier to reconstruct than audio due to their narrower spectral content.`;
    breakAfter();

    if (t1.active) {
        let blocks = [layout.vibration.zProj, layout.vibration.lstmDecoder, layout.vibration.outputProj, layout.vibration.softmax, layout.vibration.recon];
        for (let i = 0; i < blocks.length; i++) {
            let layerPhase = (t1.t * (blocks.length + 1)) - i;
            if (layerPhase > 0) {
                let intensity = Math.min(1, layerPhase);
                let pulse = intensity * (0.4 + 0.2 * Math.sin(wt.time * 3 + i * 1.0));
                blocks[i].highlight = Math.max(blocks[i].highlight, pulse);
            }
        }
    }

    breakAfter();

    // Step 3: Both reconstructions
    let t2 = afterTime(null, 2.0, 0.5);
    let t1c = afterTime(t1, 0.4);
    cleanup(t1c, [t1]);

    moveRosetaCameraTo(state, t2, new Vec3(0, 0, -layout.audio.recon.y + 40), new Vec3(290, 16, 8));

    let audioRecon = c_blockRef('audio reconstruction', layout.audio.recon);
    let vibRecon = c_blockRef('vibration reconstruction', layout.vibration.recon);

    commentary()`Both ${audioRecon} and ${vibRecon} outputs are compared against their original inputs via MSE loss. The reconstruction quality depends on how much information the 48D latent preserves — and how well the shared/private factorization captures the essential structure of each domain.`;
    breakAfter();

    if (t2.active) {
        let pulse = 0.5 + 0.3 * Math.sin(wt.time * 2.5);
        layout.audio.recon.highlight = Math.max(layout.audio.recon.highlight, pulse);
        layout.vibration.recon.highlight = Math.max(layout.vibration.recon.highlight, pulse);
        drawDimLabels(layout.audio.recon, t2.t * 0.7);
        drawDimLabels(layout.vibration.recon, t2.t * 0.7);
    }

    breakAfter();
}
