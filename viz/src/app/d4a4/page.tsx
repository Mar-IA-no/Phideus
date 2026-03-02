import React from 'react';
import { D4A4LayerView } from '@/src/d4a4concat/D4A4LayerView';
import { Header } from '@/src/homepage/Header';

export const metadata = {
    title: 'Gate 4.3 — d4a4 Concat Descriptor Injection',
    description: 'Concat injection arm: A4 audio descriptor + D4 MIDI descriptor concatenated with encoder outputs before projection. Best arm: S=83.8%.',
};

export default function Page() {
    return <>
        <Header title="Gate 4.3 — d4a4 Concat Descriptor Injection" />
        <D4A4LayerView />
        <div id="portal-container"></div>
    </>;
}
