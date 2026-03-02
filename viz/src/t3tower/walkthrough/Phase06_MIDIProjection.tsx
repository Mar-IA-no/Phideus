import { Vec3 } from "@/src/utils/vector";
import { IT3TowerWalkthroughArgs, setT3TowerInitialCamera, moveT3TowerCameraTo } from "./T3TowerWalkthrough";
import { T3TowerDim } from "../T3TowerDimStyle";

export function t3TowerPhase06_MIDIProjection(args: IT3TowerWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept, drawDimLabels } = tools;

    dimExcept([
        layout.midiOutputLN, layout.midiMeanPool, ...layout.midiProjLayers,
        layout.midiEmbedding,
    ]);

    let midiLNY = layout.midiOutputLN.y;
    setT3TowerInitialCamera(state, new Vec3(40, 0, -midiLNY - 20), new Vec3(288, 10, 3));

    let dMidi = c_dimRef('512D', T3TowerDim.D_midi);
    let d256 = c_dimRef('256D', T3TowerDim.D_proj);
    let lrProj = c_dimRef('lr_proj (5e-4)', T3TowerDim.Projection);

    commentary()`The MIDI projection head compresses ${dMidi} representations into ${d256} vectors. Like the audio projection, trained at ${lrProj} — the fastest learning rate — these layers carry the heaviest burden of cross-modal alignment.`;
    breakAfter();

    // Output LayerNorm
    let t0 = afterTime(null, 1.5, 0.5);

    let outputLN = c_blockRef('output LayerNorm', layout.midiOutputLN);

    commentary()`First, an ${outputLN} normalizes the final transformer output. This stabilizes the distribution before mean pooling, ensuring consistent scale across different input lengths.`;
    breakAfter();

    if (t0.active) {
        layout.midiOutputLN.highlight = Math.max(layout.midiOutputLN.highlight, t0.t * 0.5);
        drawDimLabels(layout.midiOutputLN, t0.t * 0.7);
    }

    breakAfter();

    // Mean pooling + projection
    let t1 = afterTime(null, 2.5, 0.5);
    let t0c = afterTime(t0, 0.3);
    cleanup(t0c, [t0]);

    moveT3TowerCameraTo(state, t1, new Vec3(40, 0, -layout.midiMeanPool.y - 60), new Vec3(288, 12, 3.5));

    let midiPool = c_blockRef('MIDI mean pooling', layout.midiMeanPool);
    let midiProj = c_blockRef('MIDI projection', layout.midiProjLayers);

    commentary()`${midiPool} collapses T_midi event tokens into a single ${dMidi} vector. Then the ${midiProj} cascades through three layers: 512→512 (BN+ReLU), 512→512 (BN+ReLU), 512→${d256} (linear). The architectures are symmetric — both audio and MIDI projections emit the same dimensionality.`;
    breakAfter();

    if (t1.active) {
        let poolPulse = 0.4 + 0.15 * Math.sin(wt.time * 4);
        layout.midiMeanPool.highlight = Math.max(layout.midiMeanPool.highlight, poolPulse);

        for (let i = 0; i < layout.midiProjLayers.length; i++) {
            let layerPhase = (t1.t * (layout.midiProjLayers.length + 1)) - i;
            if (layerPhase > 0) {
                let intensity = Math.min(1, layerPhase);
                let pulse = intensity * (0.5 + 0.2 * Math.sin(wt.time * 3 + i * 1.2));
                layout.midiProjLayers[i].highlight = Math.max(layout.midiProjLayers[i].highlight, pulse);
                drawDimLabels(layout.midiProjLayers[i], intensity * 0.7);
            }
        }
    }

    breakAfter();

    // Arrival in shared space
    let t2 = afterTime(null, 2.0, 0.5);
    let t1c = afterTime(t1, 0.3);
    cleanup(t1c, [t1]);

    moveT3TowerCameraTo(state, t2, new Vec3(40, 0, -layout.midiEmbedding.y + 20), new Vec3(288, 10, 3));

    let midiEmb = c_blockRef('z_midi', layout.midiEmbedding);

    commentary()`The result is ${midiEmb}: a ${d256} vector ready for the shared space. In the T3 architecture, this must align with both audio and T3 embeddings — creating a richer constraint than any two-tower model.`;
    breakAfter();

    if (t2.active) {
        let embPulse = 0.5 + 0.3 * Math.sin(wt.time * 2.5);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, embPulse);
        drawDimLabels(layout.midiEmbedding, t2.t * 0.9);
    }

    breakAfter();
}
