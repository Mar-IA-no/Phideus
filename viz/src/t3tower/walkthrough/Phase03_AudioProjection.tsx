import { Vec3 } from "@/src/utils/vector";
import { IT3TowerWalkthroughArgs, setT3TowerInitialCamera, moveT3TowerCameraTo } from "./T3TowerWalkthrough";
import { T3TowerDim } from "../T3TowerDimStyle";

export function t3TowerPhase03_AudioProjection(args: IT3TowerWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept, drawDimLabels } = tools;

    dimExcept([
        layout.audioMeanPool, ...layout.audioProjLayers,
        layout.audioEmbedding,
    ]);

    let audioPoolY = layout.audioMeanPool.y;
    setT3TowerInitialCamera(state, new Vec3(-140, 0, -audioPoolY - 30), new Vec3(292, 10, 3));

    let dAudio = c_dimRef('1024D', T3TowerDim.D_audio);
    let d256 = c_dimRef('256D', T3TowerDim.D_proj);
    let lrProj = c_dimRef('lr_proj (5e-4)', T3TowerDim.Projection);

    commentary()`The audio projection head compresses the rich ${dAudio} representations into compact ${d256} vectors. Trained at ${lrProj} — the fastest learning rate in the system — this 3-layer MLP maps audio features into the shared embedding space.`;
    breakAfter();

    // Mean pooling
    let t0 = afterTime(null, 2.0, 0.5);

    let audioPool = c_blockRef('audio mean pooling', layout.audioMeanPool);

    commentary()`First, ${audioPool} collapses the temporal dimension — averaging all T_audio frame vectors into a single ${dAudio} summary vector that captures the audio excerpt's essence.`;
    breakAfter();

    if (t0.active) {
        let pulse = 0.4 + 0.15 * Math.sin(wt.time * 4);
        layout.audioMeanPool.highlight = Math.max(layout.audioMeanPool.highlight, pulse);
        drawDimLabels(layout.audioMeanPool, t0.t * 0.8);
    }

    breakAfter();

    // Projection cascade
    let t1 = afterTime(null, 2.5, 0.5);
    let t0c = afterTime(t0, 0.3);
    cleanup(t0c, [t0]);

    moveT3TowerCameraTo(state, t1, new Vec3(-140, 0, -audioPoolY - 80), new Vec3(291, 12, 3.5));

    let audioProj = c_blockRef('audio projection MLP', layout.audioProjLayers);

    commentary()`The ${audioProj} fires in sequence: Layer 1 maps 1024→1024 with BatchNorm + ReLU. Layer 2 repeats. Layer 3 is a linear projection to ${d256} — no activation, allowing the full real-number line for the final representation.`;
    breakAfter();

    if (t1.active) {
        for (let i = 0; i < layout.audioProjLayers.length; i++) {
            let layerPhase = (t1.t * (layout.audioProjLayers.length + 1)) - i;
            if (layerPhase > 0) {
                let intensity = Math.min(1, layerPhase);
                let pulse = intensity * (0.5 + 0.2 * Math.sin(wt.time * 3 + i * 1.2));
                layout.audioProjLayers[i].highlight = Math.max(layout.audioProjLayers[i].highlight, pulse);
                drawDimLabels(layout.audioProjLayers[i], intensity * 0.7);
            }
        }
        layout.audioMeanPool.highlight = Math.max(layout.audioMeanPool.highlight, 0.2);
    }

    breakAfter();

    // Arrival in shared space
    let t2 = afterTime(null, 2.0, 0.5);
    let t1c = afterTime(t1, 0.3);
    cleanup(t1c, [t1]);

    moveT3TowerCameraTo(state, t2, new Vec3(-140, 0, -layout.audioEmbedding.y + 20), new Vec3(290, 10, 3));

    let audioEmb = c_blockRef('z_audio', layout.audioEmbedding);

    commentary()`The result is ${audioEmb}: a ${d256} vector ready for the shared space. In the T3 architecture, this vector must align with _both_ MIDI and T3 embeddings — a more demanding task than the original two-tower model.`;
    breakAfter();

    if (t2.active) {
        let embPulse = 0.5 + 0.3 * Math.sin(wt.time * 2.5);
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, embPulse);
        drawDimLabels(layout.audioEmbedding, t2.t * 0.9);
    }

    breakAfter();
}
