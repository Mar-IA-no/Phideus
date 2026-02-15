import React from 'react';
import { CrossAttLayerView } from '@/src/crossatt/CrossAttLayerView';
import { Header } from '@/src/homepage/Header';

export const metadata = {
    title: 'Cross-Attention Descriptor Injection (Gate 4.3)',
    description: 'A 3D visualization of the cross-attention descriptor injection architecture for audio and MIDI encoding with ratio-based features.',
};

export default function CrossAttPage() {
    return <>
        <Header title="Cross-Attention Descriptor Injection" />
        <CrossAttLayerView />
        <div id="portal-container"></div>
    </>;
}
