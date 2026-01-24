#!/usr/bin/env python3
"""
3D Visualization of RosetaVAE Latent Space
==========================================

Interactive 3D visualizations with trajectories, audio-vib alignment lines,
and temporal animations using Plotly.

Outputs:
- plot3d_umap_trajectories.html: Interactive UMAP 3D with trajectories
- plot3d_umap_alignment.html: Audio↔Vib connection lines per frame
- plot3d_animation.html: Temporal animation showing evolution
- plot3d_pca_static.png: Static PCA 3D for paper

Usage:
------
python experiments/visualize_roseta_latents_3d.py \
    --latents data/visualizations/roseta_latents.npz \
    --output data/visualizations/roseta_plots_3d
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Dimensionality reduction
from sklearn.decomposition import PCA
import umap


def load_latents(npz_path: str) -> Dict[str, np.ndarray]:
    """Load extracted latents from NPZ file."""
    data = np.load(npz_path, allow_pickle=True)

    def _convert_object_array(arr):
        """Convert object arrays to proper numpy arrays."""
        if arr.dtype == object:
            if arr.ndim == 1 and len(arr) > 0 and hasattr(arr[0], '__iter__'):
                converted = []
                for item in arr:
                    if hasattr(item, 'dtype') and item.dtype == object:
                        converted.append(_convert_object_array(item))
                    elif hasattr(item, '__iter__') and not isinstance(item, str):
                        converted.append(np.array(item, dtype=np.float32))
                    else:
                        converted.append(item)
                return converted
            else:
                try:
                    return arr.astype(np.float32)
                except (ValueError, TypeError):
                    return arr
        return arr

    latents = {}
    for key in data.files:
        arr = data[key]
        if key in ['mu_shared_audio', 'mu_shared_vib', 'mu_private_audio',
                   'mu_private_vib', 'logvar_shared_audio', 'logvar_shared_vib']:
            latents[key] = _convert_object_array(arr)
        else:
            latents[key] = arr

    return latents


def prepare_dataframe(latents: Dict) -> pd.DataFrame:
    """
    Prepare DataFrame with flattened latents and metadata.

    Returns DataFrame with columns: file_idx, frame_idx, condition, is_healthy,
    plus mu_shared_audio and mu_shared_vib as separate columns.
    """
    rows = []

    n_files = len(latents['condition'])

    for i in range(n_files):
        mu_audio = latents['mu_shared_audio'][i]
        mu_vib = latents['mu_shared_vib'][i]

        if isinstance(mu_audio, np.ndarray):
            T = mu_audio.shape[0]
        else:
            T = len(mu_audio)

        condition = str(latents['condition'][i])
        is_healthy = bool(latents['is_healthy'][i])
        filename = str(latents['filename'][i]) if 'filename' in latents else f'file_{i}'

        for t in range(T):
            rows.append({
                'file_idx': i,
                'frame_idx': t,
                'condition': condition,
                'is_healthy': is_healthy,
                'filename': filename,
                'mu_audio': np.array(mu_audio[t], dtype=np.float32),
                'mu_vib': np.array(mu_vib[t], dtype=np.float32),
            })

    return pd.DataFrame(rows)


def project_to_3d(df: pd.DataFrame, method: str = 'umap') -> Tuple[np.ndarray, np.ndarray, object]:
    """
    Project audio and vib latents to shared 3D space.

    Returns:
        xyz_audio: [N, 3] array
        xyz_vib: [N, 3] array
        reducer: fitted reducer object
    """
    # Stack all latents for joint projection
    mu_audio = np.stack(df['mu_audio'].values)
    mu_vib = np.stack(df['mu_vib'].values)

    # Concatenate for joint fitting
    all_mu = np.vstack([mu_audio, mu_vib])

    if method == 'umap':
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=30,
            min_dist=0.05,
            metric='euclidean',
            random_state=42
        )
    else:  # PCA
        reducer = PCA(n_components=3, random_state=42)

    print(f"Fitting {method.upper()} on {all_mu.shape[0]} points...")
    all_xyz = reducer.fit_transform(all_mu)

    # Split back
    n = len(df)
    xyz_audio = all_xyz[:n]
    xyz_vib = all_xyz[n:]

    return xyz_audio, xyz_vib, reducer


def plot_3d_trajectories(df: pd.DataFrame, xyz_audio: np.ndarray, xyz_vib: np.ndarray,
                         output_path: Path, method: str = 'UMAP'):
    """
    Plot 1: Interactive 3D scatter with trajectories per file.
    """
    # Add coordinates to dataframe
    df_plot = df.copy()
    df_plot['x_audio'] = xyz_audio[:, 0]
    df_plot['y_audio'] = xyz_audio[:, 1]
    df_plot['z_audio'] = xyz_audio[:, 2]
    df_plot['x_vib'] = xyz_vib[:, 0]
    df_plot['y_vib'] = xyz_vib[:, 1]
    df_plot['z_vib'] = xyz_vib[:, 2]

    # Color map for conditions
    conditions = df_plot['condition'].unique()
    colors = px.colors.qualitative.Set1[:len(conditions)]
    color_map = dict(zip(conditions, colors))

    fig = go.Figure()

    # Add scatter points for audio
    for cond in conditions:
        sub = df_plot[df_plot['condition'] == cond]
        fig.add_trace(go.Scatter3d(
            x=sub['x_audio'], y=sub['y_audio'], z=sub['z_audio'],
            mode='markers',
            marker=dict(size=3, color=color_map[cond], symbol='circle'),
            name=f'Audio-{cond}',
            text=[f"File: {f}<br>Frame: {t}" for f, t in zip(sub['filename'], sub['frame_idx'])],
            hoverinfo='text+name',
            legendgroup=cond,
        ))

    # Add scatter points for vibration
    for cond in conditions:
        sub = df_plot[df_plot['condition'] == cond]
        fig.add_trace(go.Scatter3d(
            x=sub['x_vib'], y=sub['y_vib'], z=sub['z_vib'],
            mode='markers',
            marker=dict(size=3, color=color_map[cond], symbol='x'),
            name=f'Vib-{cond}',
            text=[f"File: {f}<br>Frame: {t}" for f, t in zip(sub['filename'], sub['frame_idx'])],
            hoverinfo='text+name',
            legendgroup=cond,
            showlegend=False,
        ))

    # Add trajectories per file (audio)
    for file_idx in df_plot['file_idx'].unique():
        sub = df_plot[df_plot['file_idx'] == file_idx].sort_values('frame_idx')
        cond = sub['condition'].iloc[0]
        fig.add_trace(go.Scatter3d(
            x=sub['x_audio'], y=sub['y_audio'], z=sub['z_audio'],
            mode='lines',
            line=dict(width=2, color=color_map[cond]),
            showlegend=False,
            opacity=0.4,
            hoverinfo='skip',
        ))

    # Add trajectories per file (vibration)
    for file_idx in df_plot['file_idx'].unique():
        sub = df_plot[df_plot['file_idx'] == file_idx].sort_values('frame_idx')
        cond = sub['condition'].iloc[0]
        fig.add_trace(go.Scatter3d(
            x=sub['x_vib'], y=sub['y_vib'], z=sub['z_vib'],
            mode='lines',
            line=dict(width=2, color=color_map[cond], dash='dash'),
            showlegend=False,
            opacity=0.4,
            hoverinfo='skip',
        ))

    fig.update_layout(
        title=f'RosetaVAE: z_shared in 3D ({method}) with Trajectories',
        scene=dict(
            xaxis_title='Dim 1',
            yaxis_title='Dim 2',
            zaxis_title='Dim 3',
        ),
        legend=dict(itemsizing='constant'),
        width=1200,
        height=800,
    )

    fig.write_html(output_path / 'plot3d_umap_trajectories.html')
    print(f"Saved: {output_path / 'plot3d_umap_trajectories.html'}")


def plot_3d_alignment_lines(df: pd.DataFrame, xyz_audio: np.ndarray, xyz_vib: np.ndarray,
                            output_path: Path, sample_every: int = 5):
    """
    Plot 2: Audio↔Vib connection lines showing instantaneous alignment.
    """
    df_plot = df.copy()
    df_plot['x_audio'] = xyz_audio[:, 0]
    df_plot['y_audio'] = xyz_audio[:, 1]
    df_plot['z_audio'] = xyz_audio[:, 2]
    df_plot['x_vib'] = xyz_vib[:, 0]
    df_plot['y_vib'] = xyz_vib[:, 1]
    df_plot['z_vib'] = xyz_vib[:, 2]

    fig = go.Figure()

    # Scatter for audio (circles)
    healthy = df_plot['is_healthy']
    fig.add_trace(go.Scatter3d(
        x=df_plot.loc[healthy, 'x_audio'],
        y=df_plot.loc[healthy, 'y_audio'],
        z=df_plot.loc[healthy, 'z_audio'],
        mode='markers',
        marker=dict(size=3, color='green', symbol='circle'),
        name='Audio-Healthy',
    ))
    fig.add_trace(go.Scatter3d(
        x=df_plot.loc[~healthy, 'x_audio'],
        y=df_plot.loc[~healthy, 'y_audio'],
        z=df_plot.loc[~healthy, 'z_audio'],
        mode='markers',
        marker=dict(size=3, color='red', symbol='circle'),
        name='Audio-Fault',
    ))

    # Scatter for vibration (x markers)
    fig.add_trace(go.Scatter3d(
        x=df_plot.loc[healthy, 'x_vib'],
        y=df_plot.loc[healthy, 'y_vib'],
        z=df_plot.loc[healthy, 'z_vib'],
        mode='markers',
        marker=dict(size=3, color='green', symbol='x'),
        name='Vib-Healthy',
    ))
    fig.add_trace(go.Scatter3d(
        x=df_plot.loc[~healthy, 'x_vib'],
        y=df_plot.loc[~healthy, 'y_vib'],
        z=df_plot.loc[~healthy, 'z_vib'],
        mode='markers',
        marker=dict(size=3, color='red', symbol='x'),
        name='Vib-Fault',
    ))

    # Connection lines (sampled to avoid clutter)
    sampled = df_plot.iloc[::sample_every]
    for idx, row in sampled.iterrows():
        color = 'green' if row['is_healthy'] else 'red'
        fig.add_trace(go.Scatter3d(
            x=[row['x_audio'], row['x_vib']],
            y=[row['y_audio'], row['y_vib']],
            z=[row['z_audio'], row['z_vib']],
            mode='lines',
            line=dict(width=1, color=color),
            showlegend=False,
            opacity=0.3,
            hoverinfo='skip',
        ))

    fig.update_layout(
        title='RosetaVAE: Audio↔Vib Alignment (connection lines)',
        scene=dict(
            xaxis_title='Dim 1',
            yaxis_title='Dim 2',
            zaxis_title='Dim 3',
        ),
        width=1200,
        height=800,
    )

    fig.write_html(output_path / 'plot3d_umap_alignment.html')
    print(f"Saved: {output_path / 'plot3d_umap_alignment.html'}")


def plot_3d_animation(df: pd.DataFrame, xyz_audio: np.ndarray, xyz_vib: np.ndarray,
                      output_path: Path, max_frames: int = 50):
    """
    Plot 3: Temporal animation showing latent evolution.
    """
    df_plot = df.copy()
    df_plot['x_audio'] = xyz_audio[:, 0]
    df_plot['y_audio'] = xyz_audio[:, 1]
    df_plot['z_audio'] = xyz_audio[:, 2]
    df_plot['x_vib'] = xyz_vib[:, 0]
    df_plot['y_vib'] = xyz_vib[:, 1]
    df_plot['z_vib'] = xyz_vib[:, 2]

    # Limit frames for animation performance
    df_anim = df_plot[df_plot['frame_idx'] < max_frames].copy()

    # Prepare data for animation (combine audio and vib)
    df_audio = df_anim[['file_idx', 'frame_idx', 'condition', 'is_healthy',
                        'x_audio', 'y_audio', 'z_audio']].copy()
    df_audio.columns = ['file_idx', 'frame_idx', 'condition', 'is_healthy', 'x', 'y', 'z']
    df_audio['modality'] = 'Audio'

    df_vib = df_anim[['file_idx', 'frame_idx', 'condition', 'is_healthy',
                      'x_vib', 'y_vib', 'z_vib']].copy()
    df_vib.columns = ['file_idx', 'frame_idx', 'condition', 'is_healthy', 'x', 'y', 'z']
    df_vib['modality'] = 'Vibration'

    df_combined = pd.concat([df_audio, df_vib], ignore_index=True)
    df_combined['label'] = df_combined['condition'] + '-' + df_combined['modality']

    # Sort by frame for animation
    df_combined = df_combined.sort_values(['frame_idx', 'file_idx'])

    fig = px.scatter_3d(
        df_combined,
        x='x', y='y', z='z',
        color='condition',
        symbol='modality',
        animation_frame='frame_idx',
        hover_data=['file_idx', 'condition'],
        opacity=0.8,
        title='RosetaVAE: Latent Space Evolution Over Time'
    )

    fig.update_layout(
        width=1200,
        height=800,
        scene=dict(
            xaxis_title='Dim 1',
            yaxis_title='Dim 2',
            zaxis_title='Dim 3',
        ),
    )

    fig.write_html(output_path / 'plot3d_animation.html')
    print(f"Saved: {output_path / 'plot3d_animation.html'}")


def plot_3d_pca_static(df: pd.DataFrame, output_path: Path):
    """
    Plot 4: Static PCA 3D for paper (matplotlib).
    """
    # Get PCA projection
    mu_audio = np.stack(df['mu_audio'].values)
    mu_vib = np.stack(df['mu_vib'].values)
    all_mu = np.vstack([mu_audio, mu_vib])

    pca = PCA(n_components=3, random_state=42)
    all_xyz = pca.fit_transform(all_mu)

    n = len(df)
    xyz_audio = all_xyz[:n]
    xyz_vib = all_xyz[n:]

    # Create figure
    fig = plt.figure(figsize=(14, 6))

    # Plot 1: By condition
    ax1 = fig.add_subplot(121, projection='3d')

    conditions = df['condition'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(conditions)))
    color_map = dict(zip(conditions, colors))

    for cond in conditions:
        mask = df['condition'] == cond
        c = color_map[cond]
        ax1.scatter(xyz_audio[mask, 0], xyz_audio[mask, 1], xyz_audio[mask, 2],
                   c=[c], marker='o', s=10, alpha=0.5, label=f'A-{cond}')
        ax1.scatter(xyz_vib[mask, 0], xyz_vib[mask, 1], xyz_vib[mask, 2],
                   c=[c], marker='x', s=10, alpha=0.5)

    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    ax1.set_title('PCA 3D: By Condition')
    ax1.legend(loc='upper left', fontsize=6, ncol=2)

    # Plot 2: Healthy vs Fault
    ax2 = fig.add_subplot(122, projection='3d')

    healthy = df['is_healthy'].values

    ax2.scatter(xyz_audio[healthy, 0], xyz_audio[healthy, 1], xyz_audio[healthy, 2],
               c='green', marker='o', s=10, alpha=0.5, label='Audio-Healthy')
    ax2.scatter(xyz_audio[~healthy, 0], xyz_audio[~healthy, 1], xyz_audio[~healthy, 2],
               c='red', marker='o', s=10, alpha=0.5, label='Audio-Fault')
    ax2.scatter(xyz_vib[healthy, 0], xyz_vib[healthy, 1], xyz_vib[healthy, 2],
               c='green', marker='x', s=10, alpha=0.3, label='Vib-Healthy')
    ax2.scatter(xyz_vib[~healthy, 0], xyz_vib[~healthy, 1], xyz_vib[~healthy, 2],
               c='red', marker='x', s=10, alpha=0.3, label='Vib-Fault')

    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_zlabel('PC3')
    ax2.set_title('PCA 3D: Healthy vs Fault')
    ax2.legend(loc='upper left', fontsize=8)

    # Variance explained
    var_exp = pca.explained_variance_ratio_
    fig.suptitle(f'RosetaVAE z_shared (PCA 3D) - Variance: {var_exp[0]:.1%}, {var_exp[1]:.1%}, {var_exp[2]:.1%}',
                 fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path / 'plot3d_pca_static.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'plot3d_pca_static.png'}")

    return var_exp


def main():
    parser = argparse.ArgumentParser(description='3D Visualization of RosetaVAE latents')
    parser.add_argument('--latents', type=str,
                        default='data/visualizations/roseta_latents.npz',
                        help='Path to extracted latents NPZ')
    parser.add_argument('--output', type=str,
                        default='data/visualizations/roseta_plots_3d',
                        help='Output directory for plots')
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load latents
    print(f"Loading latents from {args.latents}")
    latents = load_latents(args.latents)

    print(f"Files: {len(latents['condition'])}")
    print(f"Conditions: {np.unique(latents['condition'])}")

    # Prepare dataframe
    print("\nPreparing data...")
    df = prepare_dataframe(latents)
    print(f"Total frames: {len(df)}")
    print(f"z_shared dim: {len(df['mu_audio'].iloc[0])}")

    # Project to 3D with UMAP
    print("\n" + "="*50)
    print("Projecting to 3D with UMAP...")
    xyz_audio_umap, xyz_vib_umap, _ = project_to_3d(df, method='umap')

    # Generate plots
    print("\n" + "="*50)
    print("Generating 3D visualizations...")

    print("\n[1/4] Trajectories plot...")
    plot_3d_trajectories(df, xyz_audio_umap, xyz_vib_umap, output_path)

    print("\n[2/4] Alignment lines plot...")
    plot_3d_alignment_lines(df, xyz_audio_umap, xyz_vib_umap, output_path, sample_every=10)

    print("\n[3/4] Animation plot...")
    plot_3d_animation(df, xyz_audio_umap, xyz_vib_umap, output_path, max_frames=50)

    print("\n[4/4] Static PCA plot...")
    var_exp = plot_3d_pca_static(df, output_path)

    # Summary
    print("\n" + "="*50)
    print("3D Visualization Summary")
    print("="*50)
    print(f"Output directory: {output_path}")
    print(f"\nGenerated files:")
    print(f"  - plot3d_umap_trajectories.html (interactive)")
    print(f"  - plot3d_umap_alignment.html (interactive)")
    print(f"  - plot3d_animation.html (animated)")
    print(f"  - plot3d_pca_static.png (for paper)")
    print(f"\nPCA variance explained: {var_exp[0]:.1%} + {var_exp[1]:.1%} + {var_exp[2]:.1%} = {sum(var_exp):.1%}")


if __name__ == '__main__':
    main()
