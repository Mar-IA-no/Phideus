import { Vec3 } from "@/src/utils/vector";
import { IPhideusWalkthroughArgs, setPhideusInitialCamera, movePhideusCameraTo } from "./PhideusWalkthrough";
import { PhideusDim } from "../PhideusDimStyle";

export function phideusPhase05_Projections(args: IPhideusWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept, drawDimLabels } = tools;

    // Dim everything except pool + projection blocks + shared space
    dimExcept([
        layout.audioMeanPool, ...layout.audioProjLayers,
        layout.midiOutputLN, layout.midiMeanPool, ...layout.midiProjLayers,
        layout.audioEmbedding, layout.midiEmbedding,
    ]);

    // Start tight on the audio pooling output — dramatic close-up
    let audioPoolY = layout.audioMeanPool.y;
    setPhideusInitialCamera(state, new Vec3(-100, 0, -audioPoolY - 30), new Vec3(292, 10, 3));

    let dAudio = c_dimRef('1024D', PhideusDim.D_audio);
    let dMidi = c_dimRef('512D', PhideusDim.D_midi);
    let d256 = c_dimRef('256D', PhideusDim.D_proj);
    let lrProj = c_dimRef('lr_proj (5e-4)', PhideusDim.Projection);

    commentary()`The projection heads are the bridge between worlds. They compress the rich, high-dimensional representations from each encoder into compact ${d256} vectors in a shared embedding space. Trained at ${lrProj} — the fastest learning rate in the entire system (167x the base rate) — these 3-layer MLPs must learn the most difficult mapping: making audio and MIDI _comparable_.`;
    breakAfter();

    // Step 1: Audio mean pool close-up
    let t0 = afterTime(null, 2.5, 0.5);

    let audioPool = c_blockRef('audio mean pooling', layout.audioMeanPool);

    commentary()`First, ${audioPool} collapses the temporal dimension. The transformer produced a sequence of ${dAudio} vectors — one per audio frame. Mean pooling averages them into a single summary vector that captures the essence of the entire audio excerpt.`;
    breakAfter();

    if (t0.active) {
        let pulse = 0.4 + 0.15 * Math.sin(wt.time * 4);
        layout.audioMeanPool.highlight = Math.max(layout.audioMeanPool.highlight, pulse);
        drawDimLabels(layout.audioMeanPool, t0.t * 0.8);
    }

    breakAfter();

    // Step 2: Audio projection cascade — layer by layer
    let t1 = afterTime(null, 3.0, 0.5);
    let t0c = afterTime(t0, 0.3);
    cleanup(t0c, [t0]);

    movePhideusCameraTo(state, t1, new Vec3(-100, 0, -audioPoolY - 80), new Vec3(291, 12, 3.5));

    let audioProj = c_blockRef('audio projection MLP', layout.audioProjLayers);

    commentary()`The ${audioProj} fires in sequence: Layer 1 maps 1024 to 1024 with BatchNorm + ReLU, preserving information while normalizing the distribution. Layer 2 repeats this transformation, further refining. Layer 3 is a linear projection that compresses to ${d256} — no activation function, allowing the network to use the full real-number line for its final representation.`;
    breakAfter();

    if (t1.active) {
        // Cascade effect: each layer lights up sequentially with staggered timing
        for (let i = 0; i < layout.audioProjLayers.length; i++) {
            let layerPhase = (t1.t * (layout.audioProjLayers.length + 1)) - i;
            if (layerPhase > 0) {
                let intensity = Math.min(1, layerPhase);
                let pulse = intensity * (0.5 + 0.2 * Math.sin(wt.time * 3 + i * 1.2));
                layout.audioProjLayers[i].highlight = Math.max(layout.audioProjLayers[i].highlight, pulse);
                drawDimLabels(layout.audioProjLayers[i], intensity * 0.7);
            }
        }
        // Keep pool glowing softly
        layout.audioMeanPool.highlight = Math.max(layout.audioMeanPool.highlight, 0.2);
    }

    breakAfter();

    // Step 3: Dramatic pan to MIDI tower
    let t2 = afterTime(null, 3.0, 0.5);
    let t1c = afterTime(t1, 0.4);
    cleanup(t1c, [t1]);

    let midiPoolY = layout.midiMeanPool.y;
    movePhideusCameraTo(state, t2, new Vec3(100, 0, -midiPoolY - 80), new Vec3(288, 12, 3.5));

    let midiPool = c_blockRef('MIDI mean pooling', layout.midiMeanPool);
    let midiProj = c_blockRef('MIDI projection MLP', layout.midiProjLayers);

    commentary()`On the MIDI side, the same principle applies. ${midiPool} collapses T_midi event tokens into a single ${dMidi} vector. Then the ${midiProj} projects through three layers: 512 to 512, 512 to 512, and finally 512 to ${d256}. The architectures are symmetric by design — both emit the same dimensionality.`;
    breakAfter();

    if (t2.active) {
        let poolPulse = 0.4 + 0.15 * Math.sin(wt.time * 4);
        layout.midiMeanPool.highlight = Math.max(layout.midiMeanPool.highlight, poolPulse);
        drawDimLabels(layout.midiMeanPool, t2.t * 0.8);

        for (let i = 0; i < layout.midiProjLayers.length; i++) {
            let layerPhase = (t2.t * (layout.midiProjLayers.length + 1)) - i;
            if (layerPhase > 0) {
                let intensity = Math.min(1, layerPhase);
                let pulse = intensity * (0.5 + 0.2 * Math.sin(wt.time * 3 + i * 1.2));
                layout.midiProjLayers[i].highlight = Math.max(layout.midiProjLayers[i].highlight, pulse);
                drawDimLabels(layout.midiProjLayers[i], intensity * 0.7);
            }
        }
    }

    breakAfter();

    // Step 4: CONVERGENCE — dramatic wide zoom showing both projections flowing into shared space
    let t3 = afterTime(null, 3.0, 0.5);
    let t2c = afterTime(t2, 0.4);
    cleanup(t2c, [t2]);

    let sharedY = layout.audioEmbedding.y;
    movePhideusCameraTo(state, t3, new Vec3(0, 0, -sharedY + 50), new Vec3(290, 18, 8));

    let audioEmb = c_blockRef('z_audio', layout.audioEmbedding);
    let midiEmb = c_blockRef('z_midi', layout.midiEmbedding);

    commentary()`And here they converge. Two completely different modalities — raw audio waveforms and discrete MIDI events — compressed through independent towers, projected through separate MLPs, arriving as ${audioEmb} and ${midiEmb}: two ${d256} vectors sitting side by side in the same mathematical space. For a matching audio-MIDI pair, these vectors should be _close_. This is where the magic happens.`;
    breakAfter();

    if (t3.active) {
        // Both projection chains glow together
        for (let blk of layout.audioProjLayers) {
            blk.highlight = Math.max(blk.highlight, 0.3 + 0.1 * Math.sin(wt.time * 2));
        }
        for (let blk of layout.midiProjLayers) {
            blk.highlight = Math.max(blk.highlight, 0.3 + 0.1 * Math.sin(wt.time * 2));
        }

        // Shared embeddings pulse dramatically — the arrival
        let embPulse = 0.5 + 0.3 * Math.sin(wt.time * 2.5);
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, embPulse);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, embPulse);

        drawDimLabels(layout.audioEmbedding, t3.t * 0.9);
        drawDimLabels(layout.midiEmbedding, t3.t * 0.9);
    }

    breakAfter();

    // Step 5: Final reflection — tight on the two embeddings
    let t4 = afterTime(null, 2.0, 0.5);
    let t3c = afterTime(t3, 0.3);
    cleanup(t3c, [t3]);

    movePhideusCameraTo(state, t4, new Vec3(0, 0, -sharedY - 10), new Vec3(290, 8, 2.5));

    commentary()`At ${lrProj}, the projection heads adapt 167 times faster than the frozen audio layers. They carry the heaviest burden of the alignment task: learning to distill modality-specific features into a universal language where cosine similarity between ${d256} vectors measures musical correspondence.`;
    breakAfter();

    if (t4.active) {
        let embPulse = 0.6 + 0.25 * Math.sin(wt.time * 3);
        layout.audioEmbedding.highlight = Math.max(layout.audioEmbedding.highlight, embPulse);
        layout.midiEmbedding.highlight = Math.max(layout.midiEmbedding.highlight, embPulse);
        drawDimLabels(layout.audioEmbedding, 0.9);
        drawDimLabels(layout.midiEmbedding, 0.9);
    }

    breakAfter();
}
