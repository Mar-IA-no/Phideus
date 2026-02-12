import React from 'react';
import { PhideusLayerView } from '@/src/phideus/PhideusLayerView';
import { Header } from '@/src/homepage/Header';

export const metadata = {
    title: 'Phideus Architecture Visualization',
    description: 'A 3D visualization of the Phideus cross-modal audio-MIDI architecture.',
};

export default function Page() {
    return <>
        <Header title="Phideus Architecture" />
        <PhideusLayerView />
        <div id="portal-container"></div>
    </>;
}
