import { Vec3 } from "@/src/utils/vector";
import { IHrmWalkthroughArgs, setHrmInitialCamera } from "./HrmWalkthrough";

export function hrmPhase04_Integration(args: IHrmWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef } = tools;

    setHrmInitialCamera(state, new Vec3(0, 0, -600), new Vec3(290, 22, 16));

    let lRef = c_blockRef('L-Module', [layout.encoderMlp, layout.gruStack]);
    let hRef = c_blockRef('H-Module', [layout.lAggregator, layout.memoryLstm]);
    let actRef = c_blockRef('ACT', layout.haltDecision);

    commentary()`The full processing loop: Input enters the ${lRef}, which processes it with context from the H-Module. The L-Module output splits: one path goes to the ${hRef} for aggregation and context generation, the other to ${actRef} for the halt decision. If the ACT says "continue", the H-Module generates new context and the loop repeats.`;
    breakAfter();

    let t0 = afterTime(null, 2.0, 0.5);
    if (t0.active) {
        let pulse = Math.sin(t0.t * Math.PI * 2) * 0.5 + 0.5;
        layout.contextIntegration.highlight = Math.max(layout.contextIntegration.highlight, pulse * 0.5);
        layout.hContext.highlight = Math.max(layout.hContext.highlight, pulse * 0.6);
        layout.lOutput.highlight = Math.max(layout.lOutput.highlight, pulse * 0.4);
    }
    breakAfter();

    commentary()`This hierarchical design was inspired by the hypothesis that harmonic structure in signals has both local (spectral peaks) and global (tonal centers, modulation) properties. The L-Module captures local features while the H-Module builds global understanding. In practice, the HRM achieved val_loss comparable to simpler architectures, suggesting the hierarchy helps efficiency more than accuracy.`;
    breakAfter();
}
