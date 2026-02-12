import { Vec3 } from "@/src/utils/vector";
import { IBloqueaWalkthroughArgs, setBloqueaInitialCamera, moveBloqueaCameraTo } from "./BloqueaWalkthrough";
import { getTransformerLayerBlocks, getAdapterBlocks } from "../BloqueaModelLayout";
import { BloqueaDim } from "../BloqueaDimStyle";

export function bloqueaPhase02_AdapterLayers(args: IBloqueaWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools } = args;
    let { afterTime, cleanup, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept, drawDimLabels } = tools;

    // Focus on layers 0-1 with adapters
    let layers01Blocks = [
        ...getTransformerLayerBlocks(layout.audioTransformerLayers[0]),
        ...getTransformerLayerBlocks(layout.audioTransformerLayers[1]),
    ];
    dimExcept(layers01Blocks);

    let layer0 = layout.audioTransformerLayers[0];
    setBloqueaInitialCamera(state, new Vec3(-100, 0, -layer0.ln1.y - 80), new Vec3(290, 15, 6));

    let frozen = c_dimRef('FROZEN', BloqueaDim.Frozen);
    let adapterDim = c_dimRef('adapter', BloqueaDim.Adapter);
    let d64 = c_dimRef('64D bottleneck', BloqueaDim.D_adapter);

    commentary()`This is the heart of Run C's innovation. Transformer layers 0-1 are completely ${frozen} — their Q/K/V projections, attention matrices, and FFN weights are locked. But after each layer's output, an ${adapterDim} module is inserted: a ${d64} that compresses, transforms, and re-expands the representation. The adapter adds a _learned residual correction_ to the frozen output.`;
    breakAfter();

    // Step 1: Show frozen layer 0 components
    let t0 = afterTime(null, 2.0, 0.3);

    let ln1Ref = c_blockRef('LayerNorm', layer0.ln1);
    let attnRef = c_blockRef('Attention', layer0.attnMatrix);

    commentary()`Layer 0: The entire transformer block — ${ln1Ref}, Q/K/V, ${attnRef}, FFN — runs at zero learning rate. These weights encode MERT's fundamental audio understanding: onset detection, pitch tracking, harmonic analysis. Run C says: "don't touch these — they're too valuable to risk."`;
    breakAfter();

    if (t0.active) {
        // Show frozen blocks with cold static glow
        for (let blk of getTransformerLayerBlocks(layer0)) {
            if (!layer0.adapter || !getAdapterBlocks(layer0).includes(blk)) {
                blk.opacity = Math.max(blk.opacity, 0.5);
                blk.highlight = Math.max(blk.highlight, t0.t * 0.15);
            }
        }
        drawDimLabels(layer0.attnMatrix, t0.t * 0.6);
    }

    breakAfter();

    // Step 2: THE ADAPTER — dramatic zoom into the bottleneck
    let t1 = afterTime(null, 3.0, 0.5);
    let t0c = afterTime(t0, 0.3);
    cleanup(t0c, [t0]);

    let adapter0 = layer0.adapter!;
    moveBloqueaCameraTo(state, t1, new Vec3(-100, 0, -adapter0.adapterDown.y - 10), new Vec3(290, 10, 2.5));

    let downRef = c_blockRef('down-projection', adapter0.adapterDown);
    let upRef = c_blockRef('up-projection', adapter0.adapterUp);

    commentary()`The ${adapterDim}: After the frozen FFN output, the ${downRef} compresses 1024D → 64D — a 16x squeeze through a linear layer. Then GELU activation introduces non-linearity. Then the ${upRef} expands 64D → 1024D. The key: the up-projection is initialized with _near-zero weights_, so the adapter starts as an identity function (via its residual connection). It learns to add small, targeted corrections. ~131K parameters per adapter, trained at lr=5e-4.`;
    breakAfter();

    if (t1.active) {
        // Cascading light through adapter: down → GELU → up
        let adapterBlks = [adapter0.adapterDown, adapter0.adapterAct, adapter0.adapterUp];
        for (let i = 0; i < adapterBlks.length; i++) {
            let layerPhase = (t1.t * (adapterBlks.length + 1)) - i;
            if (layerPhase > 0) {
                let intensity = Math.min(1, layerPhase);
                let pulse = intensity * (0.5 + 0.3 * Math.sin(wt.time * 3.5 + i * 1.5));
                adapterBlks[i].highlight = Math.max(adapterBlks[i].highlight, pulse);
                drawDimLabels(adapterBlks[i], intensity * 0.8);
            }
        }

        // Frozen blocks remain dim
        for (let blk of getTransformerLayerBlocks(layer0)) {
            if (!getAdapterBlocks(layer0).includes(blk)) {
                blk.opacity = Math.max(blk.opacity, 0.35);
            }
        }
    }

    breakAfter();

    // Step 3: The dramatic "pinch" — visual comparison
    let t2 = afterTime(null, 2.5, 0.5);
    let t1c = afterTime(t1, 0.3);
    cleanup(t1c, [t1]);

    moveBloqueaCameraTo(state, t2, new Vec3(-100, 0, -adapter0.adapterDown.y + 5), new Vec3(292, 8, 2));

    commentary()`Look at the shape: the adapter down-projection is _dramatically narrow_ compared to the frozen blocks above and below it. This visual pinch represents the information bottleneck — 1024 dimensions compressed to just 64. The adapter must learn which 64 dimensions of correction matter most. This is parameter efficiency in action: 131K adapter params vs 7M+ frozen layer params.`;
    breakAfter();

    if (t2.active) {
        let pulse = 0.6 + 0.3 * Math.sin(wt.time * 4);
        adapter0.adapterDown.highlight = Math.max(adapter0.adapterDown.highlight, pulse);
        adapter0.adapterAct.highlight = Math.max(adapter0.adapterAct.highlight, pulse * 0.8);
        adapter0.adapterUp.highlight = Math.max(adapter0.adapterUp.highlight, pulse);
        drawDimLabels(adapter0.adapterDown, 0.9);
        drawDimLabels(adapter0.adapterUp, 0.9);
    }

    breakAfter();

    // Step 4: Move to Layer 1's adapter
    let t3 = afterTime(null, 2.5, 0.5);
    let t2c = afterTime(t2, 0.4);
    cleanup(t2c, [t2]);

    let layer1 = layout.audioTransformerLayers[1];
    let adapter1 = layer1.adapter!;
    moveBloqueaCameraTo(state, t3, new Vec3(-100, 0, -adapter1.adapterDown.y - 10), new Vec3(290, 10, 2.5));

    commentary()`Layer 1 follows the exact same pattern: frozen transformer + adapter bottleneck. The two adapters together add ~262K trainable parameters to layers that contain ~14M frozen weights. That's less than 2% additional capacity — but it's the _right_ 2%, placed precisely where the model needs cross-modal adaptation. The near-identity initialization means training starts stable and gradually discovers useful corrections.`;
    breakAfter();

    if (t3.active) {
        let adapterBlks = [adapter1.adapterDown, adapter1.adapterAct, adapter1.adapterUp];
        for (let i = 0; i < adapterBlks.length; i++) {
            let layerPhase = (t3.t * (adapterBlks.length + 1)) - i;
            if (layerPhase > 0) {
                let intensity = Math.min(1, layerPhase);
                let pulse = intensity * (0.5 + 0.3 * Math.sin(wt.time * 3.5 + i * 1.5));
                adapterBlks[i].highlight = Math.max(adapterBlks[i].highlight, pulse);
                drawDimLabels(adapterBlks[i], intensity * 0.8);
            }
        }

        // Show layer 0 adapter still glowing softly
        for (let blk of getAdapterBlocks(layer0)) {
            blk.opacity = Math.max(blk.opacity, 0.7);
            blk.highlight = Math.max(blk.highlight, 0.2);
        }
    }

    breakAfter();

    // Step 5: Both adapters pulsing together
    let t4 = afterTime(null, 2.0, 0.5);
    let t3c = afterTime(t3, 0.3);
    cleanup(t3c, [t3]);

    moveBloqueaCameraTo(state, t4, new Vec3(-100, 0, -layer0.ln1.y - 150), new Vec3(290, 16, 7));

    commentary()`Together, the two adapter modules form a lightweight bypass network that threads through the frozen transformer. They are trained at lr=5e-4 — the fastest learning rate in the system — because they start from near-zero and must rapidly learn useful corrections. The frozen layers provide the stable foundation; the adapters provide the adaptive capacity.`;
    breakAfter();

    if (t4.active) {
        // Both adapters pulse in sync
        let syncPulse = 0.5 + 0.35 * Math.sin(wt.time * 3);
        for (let i = 0; i < 2; i++) {
            let layer = layout.audioTransformerLayers[i];
            for (let blk of getAdapterBlocks(layer)) {
                blk.highlight = Math.max(blk.highlight, syncPulse);
            }
            // Frozen blocks dim but visible
            for (let blk of getTransformerLayerBlocks(layer)) {
                if (!getAdapterBlocks(layer).includes(blk)) {
                    blk.opacity = Math.max(blk.opacity, 0.4);
                    blk.highlight = Math.max(blk.highlight, 0.08);
                }
            }
        }
    }

    breakAfter();
}
