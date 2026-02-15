import { Vec3 } from "@/src/utils/vector";
import { IHrmWalkthroughArgs, setHrmInitialCamera } from "./HrmWalkthrough";
import { HrmDim } from "../HrmDimStyle";

export function hrmPhase01_LModule(args: IHrmWalkthroughArgs) {
    let { state, layout, tools } = args;
    let { afterTime, breakAfter, commentary, c_blockRef, c_dimRef, dimExcept, drawDimLabels } = tools;

    setHrmInitialCamera(state, new Vec3(0, 0, -350), new Vec3(290, 20, 11));

    dimExcept([layout.inputHistogram, layout.encoderMlp, layout.contextIntegration, layout.gruStack, layout.spectralAttention, layout.lOutput]);

    let encRef = c_blockRef('Encoder MLP', layout.encoderMlp);
    let ctxRef = c_blockRef('Context Integration', layout.contextIntegration);
    let gruRef = c_blockRef('GRU Stack', layout.gruStack);
    let attnRef = c_blockRef('Spectral Attention', layout.spectralAttention);

    commentary()`The L-Module (fast path) processes input through four stages. The ${encRef} compresses input histograms from 1536 to 256 dimensions. Then ${ctxRef} concatenates with h_context from the H-Module, allowing global context to guide local processing.`;
    breakAfter();

    let t0 = afterTime(null, 1.0, 0.5);
    if (t0.active) {
        drawDimLabels(layout.encoderMlp, t0.t);
        drawDimLabels(layout.contextIntegration, t0.t);
    }
    breakAfter();

    commentary()`The ${gruRef} (2 layers, d=256) processes the contextualized features recurrently. ${attnRef} with 8 heads allows the model to focus on harmonically relevant frequency regions. The L-Module output is a 256-dimensional vector per timestep.`;
    breakAfter();

    let t1 = afterTime(null, 1.0, 0.5);
    if (t1.active) {
        layout.gruStack.highlight = Math.max(layout.gruStack.highlight, t1.t * 0.4);
        layout.spectralAttention.highlight = Math.max(layout.spectralAttention.highlight, t1.t * 0.4);
        layout.lOutput.highlight = Math.max(layout.lOutput.highlight, t1.t * 0.6);
    }
    breakAfter();
}
