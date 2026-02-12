import React from 'react';
import { BloqueaLayerView } from '@/src/bloquea/BloqueaLayerView';
import { Header } from '@/src/homepage/Header';

export const metadata = {
    title: 'BloqueA Run C Architecture Visualization',
    description: 'A 3D visualization of the BloqueA Run C hybrid adapter architecture.',
};

export default function Page() {
    return <>
        <Header title="BloqueA Architecture" />
        <BloqueaLayerView />
        <div id="portal-container"></div>
    </>;
}
