import { Vec3 } from "@/src/utils/vector";
import { IBloqueaWalkthroughArgs, setBloqueaInitialCamera, moveBloqueaCameraTo } from "./BloqueaWalkthrough";
import { BloqueaDim } from "../BloqueaDimStyle";

export function bloqueaPhase05_Projections(args: IBloqueaWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept, drawDimLabels } = tools;

    dimExcept([
        layout.audioMeanPool, ...layout.audioProjLayers,
        layout.midiOutputLN, layout.midiMeanPool, ...layout.midiProjLayers,
        layout.audioEmbedding, layout.midiEmbedding,
    ]);

    let audioPoolY = layout.audioMeanPool.y;
    setBloqueaInitialCamera(state, new Vec3(-100, 0, -audioPoolY - 30), new Vec3(292, 10, 3));

    let d256 = c_dimRef('256D', BloqueaDim.D_proj);
    let lrProj = c_dimRef('lr_proj (1e-4)', BloqueaDim.Projection);

    commentary()`The projection heads compress encoder outputs into ${d256} embeddings. In Run C, these train at ${lrProj} — the second-fastest rate after the adapters. Three-layer MLPs with BatchNorm and ReLU provide the bridge between modality-specific representations and the shared alignment space.`;
    breakAfter();

    // Step 1: Audio projection cascade
    let t0 = afterTime(null, 2.5, 0.5);

    let audioPool = c_blockRef('audio mean pooling', layout.audioMeanPool);

    commentary()`${audioPool} collapses the temporal dimension into a single 1024D summary vector. Then three projection layers progressively transform: 1024→1024→1024→256. The final layer outputs the ${d256} embedding with no activation — allowing full use of the real-number line.`;
    breakAfter();

    if (t0.active) {
        let pulse = 0.4 + 0.15 * Math.sin(wt.time * 4);
        layout.audioMeanPool.highlight = Math.max(layout.audioMeanPool.highlight, pulse);
        for (let i = 0; i < layout.audioProjLayers.length; i++) {
            let layerPhase = (t0.t * (layout.audioProjLayers.length + 1)) - i;
            if (layerPhase > 0) {
                let intensity = Math.min(1, layerPhase);
                let p = intensity * (0.5 + 0.2 * Math.sin(wt.time * 3 + i * 1.2));
                layout.audioProjLayers[i].highlight = Math.max(layout.audioProjLayers[i].highlight, p);
                drawDimLabels(layout.audioProjLayers[i], intensity * 0.7);
            }
        }
    }

    breakAfter();

    // Step 2: MIDI projection
    let t1 = afterTime(null, 2.5, 0.5);
    let t0c = afterTime(t0, 0.4);
    cleanup(t0c, [t0]);

    let midiPoolY = layout.midiMeanPool.y;
    moveBloqueaCameraTo(state, t1, new Vec3(100, 0, -midiPoolY - 80), new Vec3(288, 12, 3.5));

    let midiProj = c_blockRef('MIDI projection', layout.midiProjLayers);

    commentary()`The ${midiProj} follows the same pattern: 512→512→512→256. Symmetric architecture by design — both towers emit the same ${d256} dimensionality for alignment.`;
    breakAfter();

    if (t1.active) {
        let poolPulse = 0.4 + 0.15 * Math.sin(wt.time * 4);
        layout.midiMeanPool.highlight = Math.max(layout.midiMeanPool.highlight, poolPulse);
        for (let i = 0; i < layout.midiProjLayers.length; i++) {
            let layerPhase = (t1.t * (layout.midiProjLayers.length + 1)) - i;
            if (layerPhase > 0) {
                let intensity = Math.min(1, layerPhase);
                let p = intensity * (0.5 + 0.2 * Math.sin(wt.time * 3 + i * 1.2));
                layout.midiProjLayers[i].highlight = Math.max(layout.midiProjLayers[i].highlight, p);
                drawDimLabels(layout.midiProjLayers[i], intensity * 0.7);
            }
        }
    }

    breakAfter();

    // Step 3: Convergence
    let t2 = afterTime(null, 2.5, 0.5);
    let t1c = afterTime(t1, 0.4);
    cleanup(t1c, [t1]);

    let sharedY = layout.audioEmbedding.y;
    moveBloqueaCameraTo(state, t2, new Vec3(0, 0, -sharedY + 50), new Vec3(290, 18, 8));

    let audioEmb = c_blockRef('z_audio', layout.audioEmbedding);
    let midiEmb = c_blockRef('z_midi', layout.midiEmbedding);

    commentary()`Convergence: ${audioEmb} and ${midiEmb} — two ${d256} vectors from completely different modalities — arrive side by side in the shared space. Matching audio-MIDI pairs should land _close_ together. This is where VICReg alignment takes over.`;
    breakAfter();

    if (t2.active) {
        let embPulse = 0.5 + 0.3 * Math.sin(wt.time * 2.5);
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, embPulse);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, embPulse);
        drawDimLabels(layout.audioEmbedding, t2.t * 0.9);
        drawDimLabels(layout.midiEmbedding, t2.t * 0.9);
    }

    breakAfter();
}
