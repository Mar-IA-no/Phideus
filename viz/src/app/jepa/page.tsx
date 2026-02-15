import React from 'react';
import { JepaLayerView } from '@/src/jepa/JepaLayerView';
import { Header } from '@/src/homepage/Header';

export const metadata = {
    title: 'JEPA-Lite — No-Decoder Predictive Architecture',
    description: 'A 3D visualization of JEPA-Lite, a no-decoder predictive architecture for cross-modal learning.',
};

export default function JepaPage() {
    return <>
        <Header title="JEPA-Lite — No-Decoder Predictive" />
        <JepaLayerView />
        <div id="portal-container"></div>
    </>;
}
