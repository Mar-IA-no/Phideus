import React from 'react';
import { CrossAttLayerView } from '@/src/crossatt/CrossAttLayerView';
import { Header } from '@/src/homepage/Header';

export const metadata = {
    title: 'Gate 4.3 — a4r Reverse Cross-Attention',
    description: 'Reverse cross-attention architecture: descriptors (Q) organize encoder features (K/V) for audio-MIDI retrieval.',
};

export default function CrossAttPage() {
    return <>
        <Header title="Gate 4.3 — a4r Reverse Cross-Attention" />
        <CrossAttLayerView />
        <div id="portal-container"></div>
    </>;
}
