'use client';
import Link from 'next/link';
import React from 'react';
import s from './HomePage.module.scss';

const visualizations = [
    {
        title: 'Gate 4.3 — D0 Baseline (MERT + MIDI)',
        route: '/phideus',
        description: 'Full cross-modal architecture. D0 baseline: MERT audio encoder + MIDI transformer with split learning rates. Foundation model (S=73.4%) for all Gate 4.x experiments.',
        color: '#3366cc',
    },
    {
        title: 'Gate 4.3 — a4r Reverse Cross-Attention',
        route: '/crossatt',
        description: 'Reverse cross-attention: descriptors (Q) organize encoder features (K/V). A4 spectral descriptor on audio, D4 interval descriptor on MIDI. S=82.0%.',
        color: '#cc3366',
    },
    {
        title: 'Gate 4.3 — d4a4 Concat Injection',
        route: '/d4a4',
        description: 'Descriptors concatenated to encoder output before projection. A4 (8-band octave DSP) on audio, D4 (local intervals) on MIDI. S=83.8% (record).',
        color: '#e09530',
    },
    {
        title: 'Gate 4.3 — d4x-a4x Forward Cross-Attention',
        route: '/d4x-a4x',
        description: 'Forward cross-attention: encoder tokens (Q) query descriptor (K/V). Encoder "asks" the descriptor for guidance. Audio attn [2400, 188] — full temporal resolution preserved.',
        color: '#dd33cc',
    },
    {
        title: 'Gate 4.3 — d4r-a4r Mixed Descriptors + Reverse',
        route: '/d4r-a4r',
        description: 'Mixed descriptor strategy: A4 (spectral) on audio + D4 (intervals) on MIDI, both with reverse cross-attention. S=79.8%.',
        color: '#6b9e3a',
    },
    {
        title: 'Gate 5A — T3 Third Tower',
        route: '/t3',
        description: 'Three independent encoder towers converge into shared 256D space. T3 is a lightweight 2-layer transformer (d=256) with 3-way VICReg loss.',
        color: '#2299aa',
    },
    {
        title: 'Gate 4.3 — Bloque A Training Results',
        route: '/bloquea',
        description: 'Hybrid adapter architecture. Adapters on frozen layers 0-1, direct unfreeze on layers 2-3. S=49.4%, hard negatives 88.4%.',
        color: '#cc6633',
    },
    {
        title: 'Gate 3 — DANN Adversarial Analysis',
        route: '/dann',
        description: 'Gradient reversal layer for domain-invariant embeddings. Audio and MIDI towers with domain classifier and adversarial training.',
        color: '#996633',
    },
    {
        title: 'HRM Architecture (Research)',
        route: '/hrm',
        description: 'L-Module (fast local) + H-Module (slow global) with Adaptive Computation Time. Q-learning decides when to halt the hierarchical processing loop.',
        color: '#339966',
    },
    {
        title: 'Constellation Tokens (UOEMD)',
        route: '/constellation',
        description: 'Dual symmetric encoders with sparse ratio tokens [B,T,48,5]. Factored latent space (shared+private) with modular encoder/decoder configurations.',
        color: '#cc9933',
    },
    {
        title: 'JEPA-Lite (UOEMD)',
        route: '/jepa',
        description: 'Symmetric encoder paths without decoder. Bidirectional predictors with stop-gradient and InfoNCE contrastive alignment.',
        color: '#6633cc',
    },
    {
        title: 'Roseta VAE (UOEMD)',
        route: '/roseta',
        description: 'Dual-domain variational autoencoder. Audio and vibration encoders with shared/private latent space factorization and InfoNCE contrastive loss.',
        color: '#9933cc',
    },
];

export const PhideusHomePage: React.FC = () => {
    return <div className={s.homePage}>
        <div className={s.headerSection} style={{ flexDirection: 'column', padding: '2em 1em' }}>
            <div className={s.name}>Phideus</div>
            <div className={s.subhead}>Interactive Neural Architecture Visualizations</div>
        </div>
        <div className={s.projectsSection}>
            <div className={s.sectionTitle}>Visualizations</div>
            {visualizations.map(viz => (
                <Link key={viz.route} href={viz.route} style={{ textDecoration: 'none', color: 'inherit' }}>
                    <div className={s.projectCard}>
                        <div className={s.cardContent}>
                            <div className={s.cardTitle} style={{ borderLeft: `4px solid ${viz.color}`, paddingLeft: '0.5em' }}>
                                {viz.title}
                            </div>
                            <div className={s.cardText}>
                                {viz.description}
                            </div>
                        </div>
                    </div>
                </Link>
            ))}
        </div>
        <div style={{ marginTop: '2em', textAlign: 'center', color: '#888', fontSize: '0.85em', padding: '1em' }}>
            Built with the rendering engine from{' '}
            <a href="https://github.com/bbycroft/llm-viz" target="_blank" rel="noopener noreferrer" style={{ color: '#666' }}>
                bbycroft/llm-viz
            </a>
            {' '}by Brendan Bycroft.
            <br />
            Part of the{' '}
            <a href="https://github.com/AlterMundi/Phideus" target="_blank" rel="noopener noreferrer" style={{ color: '#666' }}>
                Phideus
            </a>
            {' '}research program.
        </div>
    </div>;
};
