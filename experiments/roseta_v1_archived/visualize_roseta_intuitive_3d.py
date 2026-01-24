#!/usr/bin/env python3
"""
Intuitive 3D Visualizations for RosetaVAE
=========================================

Clear, intuitive visualizations showing:
- Separate clouds for audio vs vibration
- Convergence between modalities (alignment)
- Distinct healthy vs fault regions

Outputs:
- plot3d_clouds_convergence.html: Convex hulls + centroids showing alignment
- plot3d_density_isosurface.html: 3D density surfaces
- plot3d_regime_separation.html: Clear healthy vs fault separation
- plot3d_alignment_centroids.png: Static centroid diagram for paper

Usage:
------
python experiments/visualize_roseta_intuitive_3d.py \
    --latents data/visualizations/roseta_latents.npz \
    --output data/visualizations/roseta_plots_3d
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import ConvexHull

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import umap


def load_latents(npz_path: str) -> Dict[str, np.ndarray]:
    """Load extracted latents from NPZ file."""
    data = np.load(npz_path, allow_pickle=True)

    def _convert_object_array(arr):
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


def prepare_data(latents: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare flattened arrays for visualization.

    Returns:
        mu_audio: [N_frames, D]
        mu_vib: [N_frames, D]
        is_healthy: [N_frames] boolean
    """
    mu_audio_list = []
    mu_vib_list = []
    is_healthy_list = []

    n_files = len(latents['condition'])

    for i in range(n_files):
        mu_a = latents['mu_shared_audio'][i]
        mu_v = latents['mu_shared_vib'][i]
        healthy = latents['condition'][i] == 'HH'

        T = len(mu_a) if hasattr(mu_a, '__len__') else mu_a.shape[0]

        for t in range(T):
            mu_audio_list.append(np.array(mu_a[t], dtype=np.float32))
            mu_vib_list.append(np.array(mu_v[t], dtype=np.float32))
            is_healthy_list.append(healthy)

    return (np.array(mu_audio_list),
            np.array(mu_vib_list),
            np.array(is_healthy_list))


def project_to_3d(mu_audio: np.ndarray, mu_vib: np.ndarray,
                  method: str = 'pca') -> Tuple[np.ndarray, np.ndarray]:
    """Joint projection to 3D space."""
    all_mu = np.vstack([mu_audio, mu_vib])

    if method == 'umap':
        reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.1,
                           metric='euclidean', random_state=42)
    else:
        reducer = PCA(n_components=3, random_state=42)

    all_xyz = reducer.fit_transform(all_mu)
    n = len(mu_audio)

    return all_xyz[:n], all_xyz[n:]


def create_ellipsoid(center: np.ndarray, cov: np.ndarray,
                     n_points: int = 20, scale: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create ellipsoid surface points from center and covariance.
    Scale=2 gives ~95% confidence region.
    """
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Ensure positive eigenvalues
    eigenvalues = np.maximum(eigenvalues, 1e-6)

    # Generate sphere
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # Scale by eigenvalues and rotate
    for i in range(n_points):
        for j in range(n_points):
            point = np.array([x[i,j], y[i,j], z[i,j]])
            # Scale by sqrt of eigenvalues
            point = point * np.sqrt(eigenvalues) * scale
            # Rotate by eigenvectors
            point = eigenvectors @ point
            # Translate
            point = point + center
            x[i,j], y[i,j], z[i,j] = point

    return x, y, z


def plot_clouds_convergence(xyz_audio: np.ndarray, xyz_vib: np.ndarray,
                            is_healthy: np.ndarray, output_path: Path):
    """
    Plot 1: Intuitive visualization with ellipsoid clouds and convergence lines.

    Shows 4 clouds: Audio-Healthy, Audio-Fault, Vib-Healthy, Vib-Fault
    With centroid connections showing alignment.
    """
    fig = go.Figure()

    # Define groups
    groups = {
        'Audio-Healthy': (xyz_audio[is_healthy], 'rgba(65, 105, 225, 0.15)', 'blue', 'circle'),
        'Audio-Fault': (xyz_audio[~is_healthy], 'rgba(220, 20, 60, 0.15)', 'darkred', 'circle'),
        'Vib-Healthy': (xyz_vib[is_healthy], 'rgba(50, 205, 50, 0.15)', 'green', 'x'),
        'Vib-Fault': (xyz_vib[~is_healthy], 'rgba(255, 140, 0, 0.15)', 'darkorange', 'x'),
    }

    centroids = {}

    for name, (points, hull_color, point_color, marker) in groups.items():
        # Compute centroid and covariance
        centroid = points.mean(axis=0)
        cov = np.cov(points.T)
        centroids[name] = centroid

        # Sample points for scatter (avoid clutter)
        n_sample = min(500, len(points))
        idx = np.random.choice(len(points), n_sample, replace=False)
        sampled = points[idx]

        # Add scatter points
        fig.add_trace(go.Scatter3d(
            x=sampled[:, 0], y=sampled[:, 1], z=sampled[:, 2],
            mode='markers',
            marker=dict(size=2, color=point_color, symbol=marker, opacity=0.4),
            name=name,
            legendgroup=name,
        ))

        # Add ellipsoid surface (95% confidence)
        try:
            ex, ey, ez = create_ellipsoid(centroid, cov, n_points=15, scale=1.5)
            fig.add_trace(go.Surface(
                x=ex, y=ey, z=ez,
                colorscale=[[0, hull_color], [1, hull_color]],
                showscale=False,
                opacity=0.3,
                name=f'{name} cloud',
                legendgroup=name,
                showlegend=False,
            ))
        except:
            pass  # Skip if covariance is singular

        # Add centroid marker
        fig.add_trace(go.Scatter3d(
            x=[centroid[0]], y=[centroid[1]], z=[centroid[2]],
            mode='markers',
            marker=dict(size=12, color=point_color, symbol='diamond',
                       line=dict(width=2, color='white')),
            name=f'{name} centroid',
            legendgroup=name,
            showlegend=False,
        ))

    # Add convergence lines (Audio ↔ Vib for same regime)
    # Healthy convergence
    fig.add_trace(go.Scatter3d(
        x=[centroids['Audio-Healthy'][0], centroids['Vib-Healthy'][0]],
        y=[centroids['Audio-Healthy'][1], centroids['Vib-Healthy'][1]],
        z=[centroids['Audio-Healthy'][2], centroids['Vib-Healthy'][2]],
        mode='lines+markers',
        line=dict(width=8, color='green'),
        marker=dict(size=15, symbol='diamond'),
        name='HEALTHY ALIGNMENT',
    ))

    # Fault convergence
    fig.add_trace(go.Scatter3d(
        x=[centroids['Audio-Fault'][0], centroids['Vib-Fault'][0]],
        y=[centroids['Audio-Fault'][1], centroids['Vib-Fault'][1]],
        z=[centroids['Audio-Fault'][2], centroids['Vib-Fault'][2]],
        mode='lines+markers',
        line=dict(width=8, color='red'),
        marker=dict(size=15, symbol='diamond'),
        name='FAULT ALIGNMENT',
    ))

    # Calculate alignment distances
    dist_healthy = np.linalg.norm(centroids['Audio-Healthy'] - centroids['Vib-Healthy'])
    dist_fault = np.linalg.norm(centroids['Audio-Fault'] - centroids['Vib-Fault'])

    # Add annotations
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(size=0),
        name=f'Healthy dist: {dist_healthy:.2f}',
        showlegend=True,
    ))
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(size=0),
        name=f'Fault dist: {dist_fault:.2f}',
        showlegend=True,
    ))

    fig.update_layout(
        title=dict(
            text='<b>Cross-Modal Convergence: Audio ↔ Vibration</b><br>' +
                 '<sup>Circles=Audio, X=Vibration | Blue/Green=Healthy, Red/Orange=Fault</sup>',
            font=dict(size=16)
        ),
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        width=1200,
        height=900,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
    )

    fig.write_html(output_path / 'plot3d_clouds_convergence.html')
    print(f"Saved: {output_path / 'plot3d_clouds_convergence.html'}")

    return centroids, dist_healthy, dist_fault


def plot_regime_separation(xyz_audio: np.ndarray, xyz_vib: np.ndarray,
                           is_healthy: np.ndarray, output_path: Path):
    """
    Plot 2: Side-by-side comparison showing regime separation.

    Left: Healthy regime (audio + vib should overlap)
    Right: Fault regime (audio + vib should overlap, but in different location)
    """
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=('HEALTHY Regime: Audio ↔ Vib Alignment',
                       'FAULT Regime: Audio ↔ Vib Alignment'),
        horizontal_spacing=0.05,
    )

    # Sample for performance
    n_sample = 1000

    # --- HEALTHY PANEL ---
    healthy_audio = xyz_audio[is_healthy]
    healthy_vib = xyz_vib[is_healthy]

    idx_h = np.random.choice(len(healthy_audio), min(n_sample, len(healthy_audio)), replace=False)

    fig.add_trace(go.Scatter3d(
        x=healthy_audio[idx_h, 0], y=healthy_audio[idx_h, 1], z=healthy_audio[idx_h, 2],
        mode='markers',
        marker=dict(size=4, color='blue', opacity=0.6),
        name='Audio (Healthy)',
    ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=healthy_vib[idx_h, 0], y=healthy_vib[idx_h, 1], z=healthy_vib[idx_h, 2],
        mode='markers',
        marker=dict(size=4, color='cyan', symbol='x', opacity=0.6),
        name='Vibration (Healthy)',
    ), row=1, col=1)

    # Connection lines for healthy (sampled)
    for i in idx_h[::20]:  # Every 20th point
        fig.add_trace(go.Scatter3d(
            x=[healthy_audio[i, 0], healthy_vib[i, 0]],
            y=[healthy_audio[i, 1], healthy_vib[i, 1]],
            z=[healthy_audio[i, 2], healthy_vib[i, 2]],
            mode='lines',
            line=dict(width=1, color='green'),
            showlegend=False,
            opacity=0.3,
        ), row=1, col=1)

    # --- FAULT PANEL ---
    fault_audio = xyz_audio[~is_healthy]
    fault_vib = xyz_vib[~is_healthy]

    idx_f = np.random.choice(len(fault_audio), min(n_sample, len(fault_audio)), replace=False)

    fig.add_trace(go.Scatter3d(
        x=fault_audio[idx_f, 0], y=fault_audio[idx_f, 1], z=fault_audio[idx_f, 2],
        mode='markers',
        marker=dict(size=4, color='red', opacity=0.6),
        name='Audio (Fault)',
    ), row=1, col=2)

    fig.add_trace(go.Scatter3d(
        x=fault_vib[idx_f, 0], y=fault_vib[idx_f, 1], z=fault_vib[idx_f, 2],
        mode='markers',
        marker=dict(size=4, color='orange', symbol='x', opacity=0.6),
        name='Vibration (Fault)',
    ), row=1, col=2)

    # Connection lines for fault (sampled)
    for i in idx_f[::20]:
        fig.add_trace(go.Scatter3d(
            x=[fault_audio[i, 0], fault_vib[i, 0]],
            y=[fault_audio[i, 1], fault_vib[i, 1]],
            z=[fault_audio[i, 2], fault_vib[i, 2]],
            mode='lines',
            line=dict(width=1, color='darkred'),
            showlegend=False,
            opacity=0.3,
        ), row=1, col=2)

    fig.update_layout(
        title=dict(
            text='<b>Regime Separation: Same Alignment, Different Regions</b>',
            font=dict(size=16)
        ),
        width=1400,
        height=700,
        showlegend=True,
    )

    fig.write_html(output_path / 'plot3d_regime_separation.html')
    print(f"Saved: {output_path / 'plot3d_regime_separation.html'}")


def plot_combined_view(xyz_audio: np.ndarray, xyz_vib: np.ndarray,
                       is_healthy: np.ndarray, output_path: Path):
    """
    Plot 3: Combined view showing the key insight.

    Audio and Vibration converge in BOTH regimes, but the regimes themselves
    are in different regions of the latent space.
    """
    fig = go.Figure()

    # Compute centroids
    c_audio_h = xyz_audio[is_healthy].mean(axis=0)
    c_audio_f = xyz_audio[~is_healthy].mean(axis=0)
    c_vib_h = xyz_vib[is_healthy].mean(axis=0)
    c_vib_f = xyz_vib[~is_healthy].mean(axis=0)

    # Sample points
    n_sample = 800

    # Healthy points (semi-transparent)
    idx_h = np.random.choice(is_healthy.sum(), min(n_sample, is_healthy.sum()), replace=False)
    fig.add_trace(go.Scatter3d(
        x=xyz_audio[is_healthy][idx_h, 0],
        y=xyz_audio[is_healthy][idx_h, 1],
        z=xyz_audio[is_healthy][idx_h, 2],
        mode='markers',
        marker=dict(size=3, color='rgba(0, 100, 255, 0.3)'),
        name='Audio-Healthy',
        legendgroup='healthy',
    ))
    fig.add_trace(go.Scatter3d(
        x=xyz_vib[is_healthy][idx_h, 0],
        y=xyz_vib[is_healthy][idx_h, 1],
        z=xyz_vib[is_healthy][idx_h, 2],
        mode='markers',
        marker=dict(size=3, color='rgba(0, 200, 100, 0.3)', symbol='x'),
        name='Vib-Healthy',
        legendgroup='healthy',
    ))

    # Fault points
    idx_f = np.random.choice((~is_healthy).sum(), min(n_sample, (~is_healthy).sum()), replace=False)
    fig.add_trace(go.Scatter3d(
        x=xyz_audio[~is_healthy][idx_f, 0],
        y=xyz_audio[~is_healthy][idx_f, 1],
        z=xyz_audio[~is_healthy][idx_f, 2],
        mode='markers',
        marker=dict(size=3, color='rgba(255, 50, 50, 0.3)'),
        name='Audio-Fault',
        legendgroup='fault',
    ))
    fig.add_trace(go.Scatter3d(
        x=xyz_vib[~is_healthy][idx_f, 0],
        y=xyz_vib[~is_healthy][idx_f, 1],
        z=xyz_vib[~is_healthy][idx_f, 2],
        mode='markers',
        marker=dict(size=3, color='rgba(255, 165, 0, 0.3)', symbol='x'),
        name='Vib-Fault',
        legendgroup='fault',
    ))

    # Large centroid markers
    fig.add_trace(go.Scatter3d(
        x=[c_audio_h[0]], y=[c_audio_h[1]], z=[c_audio_h[2]],
        mode='markers+text',
        marker=dict(size=20, color='blue', symbol='circle',
                   line=dict(width=3, color='white')),
        text=['Audio<br>Healthy'],
        textposition='top center',
        name='Centroid: Audio-Healthy',
        showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=[c_vib_h[0]], y=[c_vib_h[1]], z=[c_vib_h[2]],
        mode='markers+text',
        marker=dict(size=20, color='green', symbol='diamond',
                   line=dict(width=3, color='white')),
        text=['Vib<br>Healthy'],
        textposition='top center',
        name='Centroid: Vib-Healthy',
        showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=[c_audio_f[0]], y=[c_audio_f[1]], z=[c_audio_f[2]],
        mode='markers+text',
        marker=dict(size=20, color='red', symbol='circle',
                   line=dict(width=3, color='white')),
        text=['Audio<br>Fault'],
        textposition='top center',
        name='Centroid: Audio-Fault',
        showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=[c_vib_f[0]], y=[c_vib_f[1]], z=[c_vib_f[2]],
        mode='markers+text',
        marker=dict(size=20, color='orange', symbol='diamond',
                   line=dict(width=3, color='white')),
        text=['Vib<br>Fault'],
        textposition='top center',
        name='Centroid: Vib-Fault',
        showlegend=False,
    ))

    # CONVERGENCE LINES (thick, prominent)
    # Healthy alignment
    fig.add_trace(go.Scatter3d(
        x=[c_audio_h[0], c_vib_h[0]],
        y=[c_audio_h[1], c_vib_h[1]],
        z=[c_audio_h[2], c_vib_h[2]],
        mode='lines',
        line=dict(width=12, color='lime'),
        name='CONVERGENCE: Healthy',
    ))
    # Fault alignment
    fig.add_trace(go.Scatter3d(
        x=[c_audio_f[0], c_vib_f[0]],
        y=[c_audio_f[1], c_vib_f[1]],
        z=[c_audio_f[2], c_vib_f[2]],
        mode='lines',
        line=dict(width=12, color='crimson'),
        name='CONVERGENCE: Fault',
    ))

    # REGIME SEPARATION LINE (dashed, connecting healthy to fault)
    midpoint_h = (c_audio_h + c_vib_h) / 2
    midpoint_f = (c_audio_f + c_vib_f) / 2
    fig.add_trace(go.Scatter3d(
        x=[midpoint_h[0], midpoint_f[0]],
        y=[midpoint_h[1], midpoint_f[1]],
        z=[midpoint_h[2], midpoint_f[2]],
        mode='lines',
        line=dict(width=6, color='purple', dash='dash'),
        name='REGIME SHIFT',
    ))

    # Calculate metrics
    conv_healthy = np.linalg.norm(c_audio_h - c_vib_h)
    conv_fault = np.linalg.norm(c_audio_f - c_vib_f)
    regime_dist = np.linalg.norm(midpoint_h - midpoint_f)

    fig.update_layout(
        title=dict(
            text=f'<b>Key Insight: Cross-Modal Convergence + Regime Separation</b><br>' +
                 f'<sup>Healthy convergence: {conv_healthy:.2f} | ' +
                 f'Fault convergence: {conv_fault:.2f} | ' +
                 f'Regime separation: {regime_dist:.2f}</sup>',
            font=dict(size=16)
        ),
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3',
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2))
        ),
        width=1200,
        height=900,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor='rgba(255,255,255,0.9)',
            font=dict(size=12)
        ),
    )

    fig.write_html(output_path / 'plot3d_key_insight.html')
    print(f"Saved: {output_path / 'plot3d_key_insight.html'}")

    return conv_healthy, conv_fault, regime_dist


def plot_static_diagram(xyz_audio: np.ndarray, xyz_vib: np.ndarray,
                        is_healthy: np.ndarray, output_path: Path):
    """
    Plot 4: Static matplotlib diagram for paper.
    """
    fig = plt.figure(figsize=(14, 6))

    # Compute centroids
    c_audio_h = xyz_audio[is_healthy].mean(axis=0)
    c_audio_f = xyz_audio[~is_healthy].mean(axis=0)
    c_vib_h = xyz_vib[is_healthy].mean(axis=0)
    c_vib_f = xyz_vib[~is_healthy].mean(axis=0)

    # --- Left panel: Full view ---
    ax1 = fig.add_subplot(121, projection='3d')

    # Sample points
    n_sample = 500
    idx_h = np.random.choice(is_healthy.sum(), min(n_sample, is_healthy.sum()), replace=False)
    idx_f = np.random.choice((~is_healthy).sum(), min(n_sample, (~is_healthy).sum()), replace=False)

    ax1.scatter(xyz_audio[is_healthy][idx_h, 0], xyz_audio[is_healthy][idx_h, 1],
               xyz_audio[is_healthy][idx_h, 2], c='blue', s=5, alpha=0.2, label='Audio-H')
    ax1.scatter(xyz_vib[is_healthy][idx_h, 0], xyz_vib[is_healthy][idx_h, 1],
               xyz_vib[is_healthy][idx_h, 2], c='cyan', s=5, alpha=0.2, marker='x', label='Vib-H')
    ax1.scatter(xyz_audio[~is_healthy][idx_f, 0], xyz_audio[~is_healthy][idx_f, 1],
               xyz_audio[~is_healthy][idx_f, 2], c='red', s=5, alpha=0.2, label='Audio-F')
    ax1.scatter(xyz_vib[~is_healthy][idx_f, 0], xyz_vib[~is_healthy][idx_f, 1],
               xyz_vib[~is_healthy][idx_f, 2], c='orange', s=5, alpha=0.2, marker='x', label='Vib-F')

    # Centroids
    ax1.scatter(*c_audio_h, c='blue', s=200, marker='o', edgecolors='white', linewidths=2)
    ax1.scatter(*c_vib_h, c='green', s=200, marker='D', edgecolors='white', linewidths=2)
    ax1.scatter(*c_audio_f, c='red', s=200, marker='o', edgecolors='white', linewidths=2)
    ax1.scatter(*c_vib_f, c='orange', s=200, marker='D', edgecolors='white', linewidths=2)

    # Convergence lines
    ax1.plot([c_audio_h[0], c_vib_h[0]], [c_audio_h[1], c_vib_h[1]], [c_audio_h[2], c_vib_h[2]],
            'g-', linewidth=4, label='Healthy align')
    ax1.plot([c_audio_f[0], c_vib_f[0]], [c_audio_f[1], c_vib_f[1]], [c_audio_f[2], c_vib_f[2]],
            'r-', linewidth=4, label='Fault align')

    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    ax1.set_title('Latent Space: Full View')
    ax1.legend(loc='upper left', fontsize=7)

    # --- Right panel: Centroid diagram ---
    ax2 = fig.add_subplot(122, projection='3d')

    # Only centroids and lines
    ax2.scatter(*c_audio_h, c='blue', s=400, marker='o', edgecolors='black', linewidths=2, label='Audio-Healthy')
    ax2.scatter(*c_vib_h, c='green', s=400, marker='D', edgecolors='black', linewidths=2, label='Vib-Healthy')
    ax2.scatter(*c_audio_f, c='red', s=400, marker='o', edgecolors='black', linewidths=2, label='Audio-Fault')
    ax2.scatter(*c_vib_f, c='orange', s=400, marker='D', edgecolors='black', linewidths=2, label='Vib-Fault')

    # Convergence lines
    ax2.plot([c_audio_h[0], c_vib_h[0]], [c_audio_h[1], c_vib_h[1]], [c_audio_h[2], c_vib_h[2]],
            'g-', linewidth=5)
    ax2.plot([c_audio_f[0], c_vib_f[0]], [c_audio_f[1], c_vib_f[1]], [c_audio_f[2], c_vib_f[2]],
            'r-', linewidth=5)

    # Regime shift line
    midpoint_h = (c_audio_h + c_vib_h) / 2
    midpoint_f = (c_audio_f + c_vib_f) / 2
    ax2.plot([midpoint_h[0], midpoint_f[0]], [midpoint_h[1], midpoint_f[1]],
            [midpoint_h[2], midpoint_f[2]], 'purple', linewidth=3, linestyle='--', label='Regime shift')

    # Metrics
    conv_h = np.linalg.norm(c_audio_h - c_vib_h)
    conv_f = np.linalg.norm(c_audio_f - c_vib_f)
    regime = np.linalg.norm(midpoint_h - midpoint_f)

    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_zlabel('PC3')
    ax2.set_title(f'Centroids Only\nConvergence H:{conv_h:.2f} F:{conv_f:.2f} | Regime:{regime:.2f}')
    ax2.legend(loc='upper left', fontsize=8)

    plt.suptitle('RosetaVAE: Cross-Modal Convergence with Regime Separation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'plot3d_intuitive_static.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'plot3d_intuitive_static.png'}")


def main():
    parser = argparse.ArgumentParser(description='Intuitive 3D visualizations')
    parser.add_argument('--latents', type=str,
                        default='data/visualizations/roseta_latents.npz')
    parser.add_argument('--output', type=str,
                        default='data/visualizations/roseta_plots_3d')
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading latents...")
    latents = load_latents(args.latents)

    print("Preparing data...")
    mu_audio, mu_vib, is_healthy = prepare_data(latents)
    print(f"  Total frames: {len(mu_audio)}")
    print(f"  Healthy: {is_healthy.sum()} ({100*is_healthy.mean():.1f}%)")
    print(f"  Fault: {(~is_healthy).sum()} ({100*(~is_healthy).mean():.1f}%)")

    # Try both methods
    print("\nProjecting to 3D with PCA...")
    xyz_audio_pca, xyz_vib_pca = project_to_3d(mu_audio, mu_vib, method='pca')

    print("Projecting to 3D with UMAP...")
    xyz_audio, xyz_vib = project_to_3d(mu_audio, mu_vib, method='umap')

    print("\n" + "="*60)
    print("Generating intuitive visualizations...")
    print("="*60)

    print("\n[1/4] Clouds + Convergence...")
    centroids, dist_h, dist_f = plot_clouds_convergence(xyz_audio, xyz_vib, is_healthy, output_path)

    print("\n[2/4] Regime separation (side-by-side)...")
    plot_regime_separation(xyz_audio, xyz_vib, is_healthy, output_path)

    print("\n[3/4] Key insight visualization...")
    conv_h, conv_f, regime = plot_combined_view(xyz_audio, xyz_vib, is_healthy, output_path)

    print("\n[4/4] Static diagram for paper (using PCA)...")
    plot_static_diagram(xyz_audio_pca, xyz_vib_pca, is_healthy, output_path)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Cross-Modal Alignment Results")
    print("="*60)
    print(f"\nConvergence (smaller = better alignment):")
    print(f"  Healthy regime: {conv_h:.3f}")
    print(f"  Fault regime:   {conv_f:.3f}")
    print(f"\nRegime separation (larger = better discrimination):")
    print(f"  Distance:       {regime:.3f}")
    print(f"\nRatio (regime_sep / avg_convergence):")
    ratio = regime / ((conv_h + conv_f) / 2)
    print(f"  {ratio:.2f}x (>1 means regimes are more separated than modalities)")

    print(f"\nOutput files in: {output_path}/")
    print("  - plot3d_clouds_convergence.html")
    print("  - plot3d_regime_separation.html")
    print("  - plot3d_key_insight.html")
    print("  - plot3d_intuitive_static.png")


if __name__ == '__main__':
    main()
