'use client';
import Link from 'next/link';
import React from 'react';
import s from './HomePage.module.scss';

const visualizations = [
    {
        title: 'Phideus / Run D',
        route: '/phideus',
        description: 'Full cross-modal architecture (Audio + MIDI). Run D: full transformer unfreeze with split learning rates. Foundation model — S=51.0%, hard negatives 89.2%.',
        color: '#3366cc',
    },
    {
        title: 'BloqueA / Run C',
        route: '/bloquea',
        description: 'Hybrid adapter architecture. Adapters on frozen layers 0-1, direct unfreeze on layers 2-3. S=49.4%, hard negatives 88.4%.',
        color: '#cc6633',
    },
    {
        title: 'RosetaVAE',
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
