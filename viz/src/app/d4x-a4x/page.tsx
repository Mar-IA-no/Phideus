import React from 'react';
import { D4XA4XLayerView } from '@/src/d4xa4x/D4XA4XLayerView';
import { Header } from '@/src/homepage/Header';

export const metadata = {
    title: 'Gate 4.3 — d4x-a4x Forward Cross-Attention',
    description: 'Forward cross-attention arm: encoder tokens (Q) query descriptor features (K/V), preserving full temporal resolution. Magenta blocks show the learnable cross-attention mechanism.',
};

export default function Page() {
    return <>
        <Header title="Gate 4.3 — d4x-a4x Forward Cross-Attention" />
        <D4XA4XLayerView />
        <div id="portal-container"></div>
    </>;
}
