import React from 'react';
import { RosetaLayerView } from '@/src/roseta/RosetaLayerView';
import { Header } from '@/src/homepage/Header';

export const metadata = {
    title: 'RosetaVAE — Dual-Domain Latent Factorization',
    description: 'A 3D visualization of the RosetaVAE dual-domain variational autoencoder with shared/private latent factorization.',
};

export default function Page() {
    return <>
        <Header title="RosetaVAE Architecture" />
        <RosetaLayerView />
        <div id="portal-container"></div>
    </>;
}
