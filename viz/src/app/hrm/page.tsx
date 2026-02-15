import React from 'react';
import { HrmLayerView } from '@/src/hrm/HrmLayerView';
import { Header } from '@/src/homepage/Header';

export const metadata = {
    title: 'HRM — Hierarchical Reasoning Model',
    description: 'A 3D visualization of the Hierarchical Reasoning Model with L-Module, H-Module, and Adaptive Computation Time.',
};

export default function HrmPage() {
    return <>
        <Header title="HRM — Hierarchical Reasoning Model" />
        <HrmLayerView />
        <div id="portal-container"></div>
    </>;
}
