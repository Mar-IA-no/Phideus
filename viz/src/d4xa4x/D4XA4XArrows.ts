import { addLine } from "@/src/llm/render/lineRender";
import { IRenderState } from "@/src/llm/render/modelRender";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { ID4XA4XModelLayout, ID4XA4XTransformerLayer } from "./D4XA4XModelLayout";
import { IBlkDef } from "@/src/llm/GptModelLayout";

// ==========================================
//  Color palette
// ==========================================

let flowColor = new Vec4(0.55, 0.55, 0.55, 0.75);
let convergenceColor = Vec4.fromHexColor('#8844bb');
let lossColor = Vec4.fromHexColor('#cc3333');
let attnColor = new Vec4(0.3, 0.4, 0.7, 0.5);
let residualColor = new Vec4(0.6, 0.3, 0.1, 0.35);
let forwardXAttColor = Vec4.fromHexColor('#dd33cc');       // magenta — forward cross-att
let descriptorColor = Vec4.fromHexColor('#e09530');         // amber — A4 descriptor
let intervalColor = Vec4.fromHexColor('#7ab445');           // green — D4 interval

// ==========================================
//  Helpers
// ==========================================

function blkTopCenter(blk: IBlkDef): Vec3 {
    return new Vec3(blk.x + blk.dx / 2, blk.y, blk.z + blk.dz / 2);
}

function blkBotCenter(blk: IBlkDef): Vec3 {
    return new Vec3(blk.x + blk.dx / 2, blk.y + blk.dy, blk.z + blk.dz / 2);
}

function drawVertArrow(render: IRenderState, from: IBlkDef, to: IBlkDef, colorOverride?: Vec4, thicknessOverride?: number) {
    let opacity = Math.min(from.opacity, to.opacity);
    let color = (colorOverride ?? flowColor).mul(opacity);
    if (color.w < 0.03) return;

    let p0 = blkBotCenter(from);
    let p1 = blkTopCenter(to);
    let thickness = thicknessOverride ?? 1.2;
    addLine(render.lineRender, thickness, color, p0, p1, undefined);

    let headLen = 2.5;
    let headW = 1.3;
    let dir = p1.sub(p0).normalize();
    let base = p1.sub(dir.mul(headLen));
    let perp = new Vec3(headW, 0, 0);
    addLine(render.lineRender, thickness, color, base.add(perp), p1, undefined);
    addLine(render.lineRender, thickness, color, base.sub(perp), p1, undefined);
}

function drawDiagonalArrow(render: IRenderState, from: IBlkDef, to: IBlkDef, color: Vec4, thickness: number) {
    let opacity = Math.min(from.opacity, to.opacity);
    let c = color.mul(opacity);
    if (c.w < 0.02) return;

    let p0 = blkBotCenter(from);
    let p1 = blkTopCenter(to);
    addLine(render.lineRender, thickness, c, p0, p1, undefined);

    let headLen = 3.0;
    let headW = 1.5;
    let dir = p1.sub(p0).normalize();
    let base = p1.sub(dir.mul(headLen));
    let perp = new Vec3(dir.z, 0, -dir.x).mul(headW);
    addLine(render.lineRender, thickness, c, base.add(perp), p1, undefined);
    addLine(render.lineRender, thickness, c, base.sub(perp), p1, undefined);
}

// ==========================================
//  Transformer chain arrows
// ==========================================

function drawTransformerChain(render: IRenderState, layers: ID4XA4XTransformerLayer[]) {
    for (let i = 0; i < layers.length; i++) {
        let l = layers[i];

        drawDiagonalArrow(render, l.ln1, l.qWeight, attnColor, 0.8);
        drawDiagonalArrow(render, l.ln1, l.kWeight, attnColor, 0.8);
        drawDiagonalArrow(render, l.ln1, l.vWeight, attnColor, 0.8);

        drawDiagonalArrow(render, l.qWeight, l.attnMatrix, attnColor, 0.8);
        drawDiagonalArrow(render, l.kWeight, l.attnMatrix, attnColor, 0.8);

        drawVToAttnOut(render, l.vWeight, l.attnOut, l.attnMatrix);
        drawVertArrow(render, l.attnMatrix, l.attnOut);
        drawVertArrow(render, l.attnOut, l.attnResidual);

        drawVertArrow(render, l.attnResidual, l.ln2);
        drawVertArrow(render, l.ln2, l.mlpUp);
        drawVertArrow(render, l.mlpUp, l.mlpAct);
        drawVertArrow(render, l.mlpAct, l.mlpDown);
        drawVertArrow(render, l.mlpDown, l.ffnResidual);

        if (i < layers.length - 1) {
            drawVertArrow(render, l.ffnResidual, layers[i + 1].ln1);
        }
    }
}

function drawVToAttnOut(render: IRenderState, vWeight: IBlkDef, attnOut: IBlkDef, attnMatrix: IBlkDef) {
    let opacity = Math.min(vWeight.opacity, attnOut.opacity);
    let color = attnColor.mul(opacity);
    if (color.w < 0.02) return;

    let vBotX = vWeight.x + vWeight.dx;
    let vBotY = vWeight.y + vWeight.dy;
    let vZ = vWeight.z + vWeight.dz / 2;

    let xOff = Math.max(attnMatrix.dx / 2, vWeight.dx) + 5;
    let routeX = attnMatrix.x + attnMatrix.dx / 2 + xOff;

    let p0 = new Vec3(vBotX, vBotY, vZ);
    let p1 = new Vec3(routeX, vBotY, vZ);
    let p2 = new Vec3(routeX, attnOut.y + attnOut.dy / 2, vZ);
    let p3 = new Vec3(attnOut.x + attnOut.dx, attnOut.y + attnOut.dy / 2, vZ);

    addLine(render.lineRender, 0.8, color, p0, p1, undefined);
    addLine(render.lineRender, 0.8, color, p1, p2, undefined);
    addLine(render.lineRender, 0.8, color, p2, p3, undefined);

    let headLen = 2.0;
    addLine(render.lineRender, 0.8, color, new Vec3(p3.x + headLen, p3.y - 1, p3.z), p3, undefined);
    addLine(render.lineRender, 0.8, color, new Vec3(p3.x + headLen, p3.y + 1, p3.z), p3, undefined);
}

function drawAttentionResidualBypass(render: IRenderState, layer: ID4XA4XTransformerLayer) {
    let opacity = Math.min(layer.ln1.opacity, layer.attnResidual.opacity);
    let color = residualColor.mul(opacity);
    if (color.w < 0.02) return;

    let xOff = layer.ln1.dx / 2 + 5;
    let p0 = new Vec3(layer.ln1.x + layer.ln1.dx / 2 + xOff, layer.ln1.y, layer.ln1.z + layer.ln1.dz / 2);
    let p1 = new Vec3(layer.attnResidual.x + layer.attnResidual.dx / 2 + xOff, layer.attnResidual.y + layer.attnResidual.dy, layer.attnResidual.z + layer.attnResidual.dz / 2);

    addLine(render.lineRender, 0.8, color, p0, p1, undefined);
    let tickLen = 3;
    addLine(render.lineRender, 0.8, color, new Vec3(p0.x - tickLen, p0.y, p0.z), p0, undefined);
    addLine(render.lineRender, 0.8, color, new Vec3(p1.x - tickLen, p1.y, p1.z), p1, undefined);
}

function drawFFNResidualBypass(render: IRenderState, layer: ID4XA4XTransformerLayer) {
    let opacity = Math.min(layer.attnResidual.opacity, layer.ffnResidual.opacity);
    let color = residualColor.mul(opacity);
    if (color.w < 0.02) return;

    let xOff = layer.ln2.dx / 2 + 5;
    let p0 = new Vec3(layer.attnResidual.x + layer.attnResidual.dx / 2 + xOff, layer.attnResidual.y, layer.attnResidual.z + layer.attnResidual.dz / 2);
    let p1 = new Vec3(layer.ffnResidual.x + layer.ffnResidual.dx / 2 + xOff, layer.ffnResidual.y + layer.ffnResidual.dy, layer.ffnResidual.z + layer.ffnResidual.dz / 2);

    addLine(render.lineRender, 0.8, color, p0, p1, undefined);
    let tickLen = 3;
    addLine(render.lineRender, 0.8, color, new Vec3(p0.x - tickLen, p0.y, p0.z), p0, undefined);
    addLine(render.lineRender, 0.8, color, new Vec3(p1.x - tickLen, p1.y, p1.z), p1, undefined);
}

// ==========================================
//  Audio descriptor pipeline arrows
// ==========================================

function drawAudioDescriptorPipeline(render: IRenderState, layout: ID4XA4XModelLayout) {
    let c = descriptorColor;
    drawVertArrow(render, layout.audioDescStftWindow, layout.audioDescStftCompute, c, 1.2);
    drawVertArrow(render, layout.audioDescStftCompute, layout.audioDescMagnitude, c, 1.2);
    drawVertArrow(render, layout.audioDescMagnitude, layout.audioDescLogMag, c, 1.2);
    drawVertArrow(render, layout.audioDescLogMag, layout.audioDescBandGroup, c, 1.2);
    drawVertArrow(render, layout.audioDescBandGroup, layout.audioDescTemporalDelta, c, 1.2);
    drawVertArrow(render, layout.audioDescTemporalDelta, layout.audioDescNormalize, c, 1.2);
    drawVertArrow(render, layout.audioDescNormalize, layout.audioDescOutput, c, 1.2);
}

function drawMidiDescriptorPipeline(render: IRenderState, layout: ID4XA4XModelLayout) {
    let c = intervalColor;
    drawVertArrow(render, layout.midiDescPitchInput, layout.midiDescForwardDiff, c, 1.2);
    drawVertArrow(render, layout.midiDescForwardDiff, layout.midiDescValidityMask, c, 1.2);
    drawVertArrow(render, layout.midiDescValidityMask, layout.midiDescSemitoneScale, c, 1.2);
    drawVertArrow(render, layout.midiDescSemitoneScale, layout.midiDescLogRatioScale, c, 1.2);
    drawVertArrow(render, layout.midiDescLogRatioScale, layout.midiDescOutput, c, 1.2);
}

// ==========================================
//  Audio FORWARD Cross-Attention arrows
// ==========================================

function drawAudioForwardXAttArrows(render: IRenderState, layout: ID4XA4XModelLayout) {
    let c = forwardXAttColor;
    // Encoder features → Q projection
    drawVertArrow(render, layout.audioPosEmb, layout.audioForwardQProj, flowColor, 1.0);
    // Descriptor → K/V projections (amber→magenta transition)
    drawDiagonalArrow(render, layout.audioDescOutput, layout.audioForwardKProj, descriptorColor, 2.0);
    drawDiagonalArrow(render, layout.audioDescOutput, layout.audioForwardVProj, descriptorColor, 2.0);
    // Q proj → Q
    drawVertArrow(render, layout.audioForwardQProj, layout.audioForwardQ, c, 1.5);
    // K proj → K
    drawDiagonalArrow(render, layout.audioForwardKProj, layout.audioForwardK, c, 1.5);
    // V proj → (bypasses to attn out)
    // Q, K → matrix
    drawDiagonalArrow(render, layout.audioForwardQ, layout.audioForwardMatrix, c, 1.5);
    drawDiagonalArrow(render, layout.audioForwardK, layout.audioForwardMatrix, c, 1.5);
    // Matrix → attn out
    drawVertArrow(render, layout.audioForwardMatrix, layout.audioForwardAttnOut, c, 1.2);
    // Attn out → residual → norm
    drawVertArrow(render, layout.audioForwardAttnOut, layout.audioForwardResidual, flowColor, 1.0);
    drawVertArrow(render, layout.audioForwardResidual, layout.audioForwardNorm, flowColor, 1.0);
}

// ==========================================
//  MIDI FORWARD Cross-Attention arrows
// ==========================================

function drawMidiForwardXAttArrows(render: IRenderState, layout: ID4XA4XModelLayout) {
    let c = forwardXAttColor;
    drawVertArrow(render, layout.midiPosEnc, layout.midiForwardQProj, flowColor, 1.0);
    drawDiagonalArrow(render, layout.midiDescOutput, layout.midiForwardKProj, intervalColor, 2.0);
    drawDiagonalArrow(render, layout.midiDescOutput, layout.midiForwardVProj, intervalColor, 2.0);
    drawVertArrow(render, layout.midiForwardQProj, layout.midiForwardQ, c, 1.5);
    drawDiagonalArrow(render, layout.midiForwardKProj, layout.midiForwardK, c, 1.5);
    drawDiagonalArrow(render, layout.midiForwardQ, layout.midiForwardMatrix, c, 1.5);
    drawDiagonalArrow(render, layout.midiForwardK, layout.midiForwardMatrix, c, 1.5);
    drawVertArrow(render, layout.midiForwardMatrix, layout.midiForwardResidual, flowColor, 1.0);
    drawVertArrow(render, layout.midiForwardResidual, layout.midiForwardNorm, flowColor, 1.0);
}

// ==========================================
//  Main orchestrator
// ==========================================

export function drawD4XA4XArrows(render: IRenderState, layout: ID4XA4XModelLayout) {

    // 1. Audio CNN chain
    drawVertArrow(render, layout.waveformInput, layout.audioCnn[0]);
    for (let i = 0; i < layout.audioCnn.length - 1; i++) {
        drawVertArrow(render, layout.audioCnn[i], layout.audioCnn[i + 1]);
    }
    drawVertArrow(render, layout.audioCnn[layout.audioCnn.length - 1], layout.audioPosEmb);

    // Waveform → descriptor pipeline
    drawDiagonalArrow(render, layout.waveformInput, layout.audioDescStftWindow, descriptorColor, 1.5);

    // 2. Audio descriptor DSP chain
    drawAudioDescriptorPipeline(render, layout);

    // 3. Audio forward cross-att arrows
    drawAudioForwardXAttArrows(render, layout);

    // 4. Forward xatt output → first transformer layer
    drawVertArrow(render, layout.audioForwardNorm, layout.audioTransformerLayers[0].ln1);

    // 5. Audio transformer chains with residual bypasses
    drawTransformerChain(render, layout.audioTransformerLayers);
    for (let layer of layout.audioTransformerLayers) {
        drawAttentionResidualBypass(render, layer);
        drawFFNResidualBypass(render, layer);
    }

    // 6. Audio post
    let lastAudioTf = layout.audioTransformerLayers[layout.audioTransformerLayers.length - 1];
    drawVertArrow(render, lastAudioTf.ffnResidual, layout.audioOutputLN);
    drawVertArrow(render, layout.audioOutputLN, layout.audioMeanPool);
    drawVertArrow(render, layout.audioMeanPool, layout.audioProjLayer1);
    drawVertArrow(render, layout.audioProjLayer1, layout.audioProjLayer2);
    drawVertArrow(render, layout.audioProjLayer2, layout.audioProjLayer3);

    // 7. MIDI embedding chain
    drawVertArrow(render, layout.midiInput, layout.midiPitchEmb);
    drawVertArrow(render, layout.midiInput, layout.midiVelEmb);
    drawVertArrow(render, layout.midiInput, layout.midiDurEmb);
    drawVertArrow(render, layout.midiPitchEmb, layout.midiCombineLinear);
    drawVertArrow(render, layout.midiVelEmb, layout.midiCombineLinear);
    drawVertArrow(render, layout.midiDurEmb, layout.midiCombineLinear);
    drawVertArrow(render, layout.midiCombineLinear, layout.midiPosEnc);

    // MIDI input → descriptor
    drawDiagonalArrow(render, layout.midiInput, layout.midiDescPitchInput, intervalColor, 1.5);

    // 8. MIDI descriptor chain
    drawMidiDescriptorPipeline(render, layout);

    // 9. MIDI forward cross-att arrows
    drawMidiForwardXAttArrows(render, layout);

    // 10. Forward xatt output → first MIDI transformer
    drawVertArrow(render, layout.midiForwardNorm, layout.midiTransformerLayers[0].ln1);

    // 11. MIDI transformer chains
    drawTransformerChain(render, layout.midiTransformerLayers);
    for (let layer of layout.midiTransformerLayers) {
        drawAttentionResidualBypass(render, layer);
        drawFFNResidualBypass(render, layer);
    }

    // 12. MIDI post
    let lastMidiTf = layout.midiTransformerLayers[layout.midiTransformerLayers.length - 1];
    drawVertArrow(render, lastMidiTf.ffnResidual, layout.midiOutputLN);
    drawVertArrow(render, layout.midiOutputLN, layout.midiMeanPool);
    drawVertArrow(render, layout.midiMeanPool, layout.midiProjLayer1);
    drawVertArrow(render, layout.midiProjLayer1, layout.midiProjLayer2);
    drawVertArrow(render, layout.midiProjLayer2, layout.midiProjLayer3);

    // 13. Convergence: proj → shared embeddings
    drawDiagonalArrow(render, layout.audioProjLayer3, layout.audioEmbedding, convergenceColor, 2.0);
    drawDiagonalArrow(render, layout.midiProjLayer3, layout.midiEmbedding, convergenceColor, 2.0);

    // 14. Loss: embeddings → VICReg
    drawDiagonalArrow(render, layout.audioEmbedding, layout.vicregInv, lossColor, 1.5);
    drawDiagonalArrow(render, layout.midiEmbedding, layout.vicregInv, lossColor, 1.5);
    drawDiagonalArrow(render, layout.audioEmbedding, layout.vicregVar, lossColor, 1.5);
    drawDiagonalArrow(render, layout.midiEmbedding, layout.vicregVar, lossColor, 1.5);
    drawDiagonalArrow(render, layout.audioEmbedding, layout.vicregCov, lossColor, 1.5);
    drawDiagonalArrow(render, layout.midiEmbedding, layout.vicregCov, lossColor, 1.5);
}
