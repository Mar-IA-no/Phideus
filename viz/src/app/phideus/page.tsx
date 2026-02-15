import React from 'react';
import { PhideusLayerView } from '@/src/phideus/PhideusLayerView';
import { Header } from '@/src/homepage/Header';

export const metadata = {
    title: 'MERT + MIDI Transformer — Phideus Architecture',
    description: 'A 3D visualization of the MERT Audio + MIDI Transformer cross-modal architecture (Run D Foundation).',
};

export default function Page() {
    return <>
        <Header title="MERT + MIDI Transformer" />
        <PhideusLayerView />
        <div id="portal-container"></div>
    </>;
}
