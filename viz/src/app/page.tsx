import React from 'react';
import { PhideusHomePage } from '@/src/homepage/PhideusHomePage';

export const metadata = {
    title: 'Phideus — Interactive Neural Architecture Visualizations',
    description: 'Explore the Phideus research program neural network architectures in interactive 3D.',
};

export default function Page() {
    return <>
        <PhideusHomePage />
        <div id="portal-container"></div>
    </>;
}
