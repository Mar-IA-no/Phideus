import React from 'react';
import { BloqueaLayerView } from '@/src/bloquea/BloqueaLayerView';
import { Header } from '@/src/homepage/Header';

export const metadata = {
    title: 'Hybrid Adapter Fine-Tuning — BloqueA Architecture',
    description: 'A 3D visualization of the BloqueA Run C hybrid adapter fine-tuning architecture.',
};

export default function Page() {
    return <>
        <Header title="Hybrid Adapter Fine-Tuning" />
        <BloqueaLayerView />
        <div id="portal-container"></div>
    </>;
}
