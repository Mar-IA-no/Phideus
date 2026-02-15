import { Vec3 } from "@/src/utils/vector";
import { IHrmWalkthroughArgs, setHrmInitialCamera } from "./HrmWalkthrough";

export function hrmPhase02_HModule(args: IHrmWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, dimExcept } = tools;

    setHrmInitialCamera(state, new Vec3(-30, 0, -400), new Vec3(290, 20, 12));

    dimExcept([layout.lAggregator, layout.harmonicAttention, layout.memoryLstm, layout.contextGenerator, layout.hContext, layout.lOutput]);

    let aggRef = c_blockRef('L-Aggregator', layout.lAggregator);
    let hattnRef = c_blockRef('Harmonic Attention', layout.harmonicAttention);
    let lstmRef = c_blockRef('Memory LSTM', layout.memoryLstm);
    let genRef = c_blockRef('Context Generator', layout.contextGenerator);

    commentary()`The H-Module (slow path) aggregates information across multiple L-Module iterations. The ${aggRef} collects the last 10 L-Module outputs. ${hattnRef} (4 heads, d=128) identifies harmonically significant patterns across the aggregated outputs.`;
    breakAfter();

    let t0 = afterTime(null, 1.5, 0.5);
    if (t0.active) {
        layout.lAggregator.highlight = Math.max(layout.lAggregator.highlight, t0.t * 0.5);
        layout.harmonicAttention.highlight = Math.max(layout.harmonicAttention.highlight, t0.t * 0.5);
    }
    breakAfter();

    let ctxRef = c_blockRef('h_context', layout.hContext);
    commentary()`The ${lstmRef} maintains persistent state across iterations, enabling the H-Module to build a long-term representation. The ${genRef} (Linear + Tanh) produces ${ctxRef}, a 128-dimensional context vector that feeds back to the L-Module. This creates the hierarchical loop: fast local processing informed by slow global reasoning.`;
    breakAfter();

    let t1 = afterTime(null, 1.5, 0.5);
    if (t1.active) {
        layout.memoryLstm.highlight = Math.max(layout.memoryLstm.highlight, t1.t * 0.4);
        layout.hContext.highlight = Math.max(layout.hContext.highlight, t1.t * 0.7);
    }
    breakAfter();
}
