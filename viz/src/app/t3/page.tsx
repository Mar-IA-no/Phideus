import React from 'react';
import { T3TowerLayerView } from '@/src/t3tower/T3TowerLayerView';
import { Header } from '@/src/homepage/Header';

export const metadata = {
    title: 'Gate 5A — T3 Three-Tower Architecture',
    description: 'Three-tower architecture: Audio, MIDI, and T3 lightweight transformer converge via 3-way VICReg in shared 256D space.',
};

export default function T3TowerPage() {
    return <>
        <Header title="Gate 5A — T3 Three-Tower Architecture" />
        <T3TowerLayerView />
        <div id="portal-container"></div>
    </>;
}
