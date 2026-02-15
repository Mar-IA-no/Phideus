import { Vec3 } from "@/src/utils/vector";
import { IHrmWalkthroughArgs, setHrmInitialCamera } from "./HrmWalkthrough";

export function hrmPhase03_ACT(args: IHrmWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept } = tools;

    setHrmInitialCamera(state, new Vec3(30, 0, -400), new Vec3(290, 20, 12));

    dimExcept([layout.qNet1, layout.qNet2, layout.qNet3, layout.haltDecision, layout.lOutput]);

    let qnetRef = c_blockRef('Q-Network', [layout.qNet1, layout.qNet2, layout.qNet3]);
    let haltRef = c_blockRef('halt/continue', layout.haltDecision);

    commentary()`Adaptive Computation Time (ACT) decides when to stop the hierarchical loop. The ${qnetRef} (256->64->64->2) takes the L-Module output and produces a ${haltRef} decision. This is trained via Q-learning: the agent learns the optimal number of iterations for each input.`;
    breakAfter();

    let t0 = afterTime(null, 1.5, 0.5);
    if (t0.active) {
        layout.qNet1.highlight = Math.max(layout.qNet1.highlight, t0.t * 0.3);
        layout.qNet2.highlight = Math.max(layout.qNet2.highlight, t0.t * 0.3);
        layout.qNet3.highlight = Math.max(layout.qNet3.highlight, t0.t * 0.3);
        layout.haltDecision.highlight = Math.max(layout.haltDecision.highlight, t0.t * 0.7);
    }
    breakAfter();

    commentary()`The ACT mechanism uses epsilon-greedy exploration during training (min=2, max=20 steps). Simpler inputs (e.g., pure tones) require fewer iterations, while complex harmonic structures may need many passes. This makes computation proportional to input complexity.`;
    breakAfter();
}
