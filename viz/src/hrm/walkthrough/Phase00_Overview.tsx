import { Vec3 } from "@/src/utils/vector";
import { IHrmWalkthroughArgs, setHrmInitialCamera } from "./HrmWalkthrough";
import { HrmDim } from "../HrmDimStyle";

export function hrmPhase00_Overview(args: IHrmWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, c_dimRef } = tools;

    setHrmInitialCamera(state, new Vec3(0, 0, -600), new Vec3(290, 22, 14));

    let lModRef = c_blockRef('L-Module', [layout.encoderMlp, layout.gruStack, layout.spectralAttention]);
    let hModRef = c_blockRef('H-Module', [layout.lAggregator, layout.harmonicAttention, layout.memoryLstm]);
    let actRef = c_blockRef('ACT', layout.haltDecision);

    commentary()`The Hierarchical Reasoning Model (HRM) processes input through two nested modules. The ${lModRef} runs fast, processing each input step. The ${hModRef} runs slowly, aggregating L-Module outputs and generating context. An ${actRef} mechanism decides when to stop iterating.`;
    breakAfter();

    let t0 = afterTime(null, 1.5, 0.5);
    if (t0.active) {
        layout.hContext.highlight = Math.max(layout.hContext.highlight, t0.t * 0.6);
        layout.haltDecision.highlight = Math.max(layout.haltDecision.highlight, t0.t * 0.4);
    }
    breakAfter();

    let ctxRef = c_blockRef('h_context', layout.hContext);
    commentary()`The key innovation is the feedback loop: the H-Module generates ${ctxRef} which feeds back into the L-Module's context integration layer. This creates a hierarchical processing loop where fast local processing is guided by slow global reasoning. The number of iterations is learned adaptively via Q-learning.`;
    breakAfter();
}
