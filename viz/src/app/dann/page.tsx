import React from 'react';
import { DannLayerView } from '@/src/dann/DannLayerView';
import { Header } from '@/src/homepage/Header';

export const metadata = {
    title: 'DANN — Domain Adversarial Neural Network',
    description: 'A 3D visualization of the Domain Adversarial Neural Network (DANN) architecture with gradient reversal.',
};

export default function DannPage() {
    return <>
        <Header title="DANN — Domain Adversarial Network" />
        <DannLayerView />
        <div id="portal-container"></div>
    </>;
}
