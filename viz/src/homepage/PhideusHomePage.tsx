'use client';
import Link from 'next/link';
import React from 'react';
import s from './HomePage.module.scss';

const visualizations = [
    {
        title: 'Cross-Attention Descriptor Injection (Gate 4.3)',
        route: '/crossatt',
        description: 'Audio and MIDI encoders with cross-attention to ratio descriptors. Features selectively query spectral and interval information via learned attention patterns.',
        color: '#cc3366',
    },
    {
        title: 'MERT Audio + MIDI Transformer (Run D Foundation)',
        route: '/phideus',
        description: 'Full cross-modal architecture. Run D: full transformer unfreeze with split learning rates. Foundation model for all Gate 4.x experiments.',
        color: '#3366cc',
    },
    {
        title: 'Hybrid Adapter Fine-Tuning (BloqueA Run C)',
        route: '/bloquea',
        description: 'Hybrid adapter architecture. Adapters on frozen layers 0-1, direct unfreeze on layers 2-3. S=49.4%, hard negatives 88.4%.',
        color: '#cc6633',
    },
    {
        title: 'Domain Adversarial Network (Gate 3 DANN)',
        route: '/dann',
        description: 'Gradient reversal layer for domain-invariant embeddings. Audio and MIDI towers with domain classifier and adversarial training.',
        color: '#996633',
    },
    {
        title: 'Hierarchical Reasoning Model (HRM)',
        route: '/hrm',
        description: 'L-Module (fast local) + H-Module (slow global) with Adaptive Computation Time. Q-learning decides when to halt the hierarchical processing loop.',
        color: '#339966',
    },
    {
        title: 'ConstellationVAE — Sparse Token VAE (C1-C4)',
        route: '/constellation',
        description: 'Dual symmetric encoders with sparse ratio tokens [B,T,48,5]. Factored latent space (shared+private) with modular encoder/decoder configurations.',
        color: '#cc9933',
    },
    {
        title: 'JEPA-Lite — No-Decoder Predictive Architecture',
        route: '/jepa',
        description: 'Symmetric encoder paths without decoder. Bidirectional predictors with stop-gradient and InfoNCE contrastive alignment.',
        color: '#6633cc',
    },
    {
        title: 'RosetaVAE — Dual-Domain Latent Factorization',
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
