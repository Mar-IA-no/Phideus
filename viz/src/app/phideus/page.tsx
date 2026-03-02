import React from 'react';
import { PhideusLayerView } from '@/src/phideus/PhideusLayerView';
import { Header } from '@/src/homepage/Header';

export const metadata = {
    title: 'Gate 4.3 — D0 Baseline (MERT + MIDI)',
    description: 'D0 baseline architecture: MERT audio encoder + MIDI transformer with split learning rates. Foundation model for all Gate 4.x experiments.',
};

export default function Page() {
    return <>
        <Header title="Gate 4.3 — D0 Baseline (MERT + MIDI)" />
        <PhideusLayerView />
        <div id="portal-container"></div>
    </>;
}
