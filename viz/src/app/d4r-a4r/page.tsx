import React from 'react';
import { D4RA4RLayerView } from '@/src/d4ra4r/D4RA4RLayerView';
import { Header } from '@/src/homepage/Header';

export const metadata = {
    title: 'Gate 4.3 — d4r-a4r Mixed Descriptors + Reverse',
    description: 'Mixed descriptor strategy: A4 (spectral) on audio + D4 (intervals) on MIDI, both with reverse cross-attention. S=79.8%.',
};

export default function D4RA4RPage() {
    return <>
        <Header title="Gate 4.3 — d4r-a4r Mixed Descriptors + Reverse" />
        <D4RA4RLayerView />
        <div id="portal-container"></div>
    </>;
}
