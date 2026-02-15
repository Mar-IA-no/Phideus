import React from 'react';
import { ConstellationLayerView } from '@/src/constellation/ConstellationLayerView';
import { Header } from '@/src/homepage/Header';

export const metadata = {
    title: 'ConstellationVAE — Sparse Token VAE (C1-C4)',
    description: 'A 3D visualization of the ConstellationVAE architecture with dual symmetric encoders, factored latent space, and modular decoders.',
};

export default function ConstellationPage() {
    return <>
        <Header title="ConstellationVAE Architecture" />
        <ConstellationLayerView />
        <div id="portal-container"></div>
    </>;
}
